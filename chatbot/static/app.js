/* Chatty web UI.
 *
 * Single-file, no build step: vanilla JS on top of two vendored helpers
 * (marked for markdown, DOMPurify to sanitize the model's rendered HTML --
 * answers are LLM output that can quote arbitrary document text).
 *
 * Transport: fetch() + a tiny SSE parser (EventSource is GET-only, and the
 * chat POSTs its body). One deterministic scroll rule (ported from the old
 * Gradio JS): follow the newest content only while the user is already
 * near the bottom; scroll up and the stream stops fighting them; sending a
 * new message resumes following.
 */
"use strict";

(function () {
  // ------------------------------------------------------------------ utils

  const $ = (sel) => document.querySelector(sel);

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function toast(message, isError) {
    const node = $("#toast");
    node.textContent = message;
    node.classList.toggle("is-error", !!isError);
    node.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { node.hidden = true; }, isError ? 6000 : 3500);
  }

  /** "2026-10-02 09:30" / ISO string -> "14:32" (today) or "Oct 2 · 14:32". */
  function fmtTime(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d)) return "";
    const now = new Date();
    const hm = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    if (d.toDateString() === now.toDateString()) return hm;
    return d.toLocaleDateString([], { month: "short", day: "numeric" }) + " · " + hm;
  }

  /** epoch seconds (server) or null -> "just now" / "2 h ago" / ... */
  function fmtAge(seconds) {
    if (!seconds) return "never";
    const age = Math.max(0, Date.now() / 1000 - seconds);
    if (age < 90) return "just now";
    if (age < 3600) return Math.floor(age / 60) + " min ago";
    if (age < 86400) return Math.floor(age / 3600) + " h ago";
    return Math.floor(age / 86400) + " d ago";
  }

  // --------------------------------------------------------------- markdown

  marked.setOptions({ gfm: true, breaks: false });

  /** Render model markdown as safe HTML. The raw text is parsed by marked,
   * then DOMPurify strips anything structural (scripts, handlers, iframes);
   * outbound links get target=_blank so a cited doc URL doesn't navigate
   * away from the app. */
  function renderMarkdown(text) {
    const html = DOMPurify.sanitize(marked.parse(text || ""));
    const tmp = document.createElement("div");
    tmp.innerHTML = html;
    tmp.querySelectorAll("a").forEach((a) => {
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    });
    return tmp.innerHTML;
  }

  // ------------------------------------------------------------------- SSE

  /** Parse an SSE byte stream (from a POST's response body) into
   * {event, data} objects. Resolves when the stream ends. */
  async function readSse(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        let event = "message";
        let data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        let parsed = data;
        try { parsed = JSON.parse(data); } catch (_e) { /* non-JSON: pass raw */ }
        onEvent(event, parsed);
      }
    }
  }

  async function api(path, options) {
    const resp = await fetch(path, options);
    if (resp.status === 401) { location.href = "/login"; throw new Error("unauthorized"); }
    if (!resp.ok) {
      let detail = "HTTP " + resp.status;
      try { detail = (await resp.json()).detail || detail; } catch (_e) { /* keep */ }
      throw new Error(detail);
    }
    return resp;
  }

  // ------------------------------------------------------------------ chat

  const messagesEl = $("#messages");
  const inputEl = $("#composer-input");
  const sendBtn = $("#btn-send");
  const stopBtn = $("#btn-stop");
  const clearBtn = $("#btn-clear");

  const inner = el("div", "messages-inner");
  messagesEl.appendChild(inner);

  let streaming = false;
  let abortCtrl = null;
  let followScroll = true;

  messagesEl.addEventListener("scroll", () => {
    followScroll = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 80;
  }, { passive: true });

  function scrollDown(force) {
    if (force || followScroll) messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  /** Live "thinking" bubble: bouncing dots + a seconds counter that starts
   * as soon as the stream is pending and is removed once real text lands. */
  function makeTyping() {
    const wrap = el("div", "typing-indicator");
    for (let i = 0; i < 3; i++) wrap.appendChild(el("span"));
    const timer = el("span", "typing-timer", "0s");
    wrap.appendChild(timer);
    let secs = 0;
    const iv = setInterval(() => {
      if (!wrap.isConnected) { clearInterval(iv); return; }
      secs += 1;
      timer.textContent = secs + "s";
    }, 1000);
    return {
      node: wrap,
      stop() { clearTimeout(iv); clearInterval(iv); },
    };
  }

  /** Create an assistant message row. Returns mutators plus the raw-text
   * holder used for copy + the feedback (question/answer) pairing. */
  function addAssistant(opts) {
    const row = el("div", "msg msg-assistant");
    const bubble = el("div", "msg-bubble");
    row.appendChild(bubble);

    const meta = el("div", "msg-meta");
    row.appendChild(meta);

    const tools = el("div", "msg-tools");
    const copyBtn = el("button", "tool-btn", "⧉ Copy");
    const likeBtn = el("button", "tool-btn", "👍");
    const dislikeBtn = el("button", "tool-btn", "👎");
    tools.append(copyBtn, likeBtn, dislikeBtn);
    row.appendChild(tools);

    const stamp = () => {
      const t = fmtTime(new Date().toISOString());
      if (t) meta.textContent = t;
    };
    stamp();

    inner.appendChild(row);
    scrollDown();
    return { row, bubble, tools, copyBtn, likeBtn, dislikeBtn, stamp };
  }

  function addUser(text) {
    const row = el("div", "msg msg-user");
    const bubble = el("div", "msg-bubble", text);
    row.appendChild(bubble);
    const meta = el("div", "msg-meta", fmtTime(new Date().toISOString()));
    row.appendChild(meta);
    inner.appendChild(row);
    scrollDown(true); // sending always resumes following
  }

  function renderWelcome(suggestions) {
    inner.innerHTML = "";
    const w = el("div", "welcome");
    w.appendChild(el("div", "welcome-icon", "🐦"));
    w.appendChild(el("h2", null, "Chatty here."));
    w.appendChild(el("p", null,
      "Ask about anything in your documents — everything from Google Drive " +
      "and your uploads is searchable."));
    if (suggestions && suggestions.length) {
      const row = el("div", "suggestions");
      for (const s of suggestions) {
        const b = el("button", "chip-btn", s);
        b.addEventListener("click", () => {
          inputEl.value = s;
          autoGrow();
          inputEl.focus();
        });
        row.appendChild(b);
      }
      w.appendChild(row);
    }
    inner.appendChild(w);
  }

  // Sources payload from the server -> chip row under the answer.
  function renderSources(bubble, sources, stopped) {
    if (stopped) {
      bubble.appendChild(el("div", "msg-stopped", "⏹ Stopped"));
      return;
    }
    if (!sources) return;
    const box = el("div", "sources");
    if (sources.kind === "catalog") {
      box.appendChild(el("span", "sources-label",
        "Source: index metadata (" + sources.documents + " document" +
        (sources.documents === 1 ? "" : "s") + ", not a content search)"));
    } else {
      box.appendChild(el("span", "sources-label", sources.total > 1 ? "Sources" : "Source"));
    }
    const labels = sources.labels || [];
    labels.slice(0, 8).forEach((label) => box.appendChild(el("span", "src-chip", label)));
    if (labels.length > 8) box.appendChild(el("span", null, "…" + (labels.length - 8) + " more"));
    bubble.appendChild(box);
  }

  function renderError(bubble, message) {
    const box = el("div", "msg-error");
    // safe_error_message embeds "(error id: abc12345)" — show it in mono,
    // and point at the server log (which has the full traceback).
    const idMatch = /error id: ([a-f0-9]+)/.exec(message || "");
    const body = el("div", null, "⚠ " + (message || "Something went wrong."));
    box.appendChild(body);
    if (idMatch) box.appendChild(el("div", "error-id", "error id: " + idMatch[1]));
    box.appendChild(el("span", "error-hint", "Full details are in the server log."));
    bubble.appendChild(box);
  }

  // Wire the per-answer toolbar. question is the user turn this answer
  // responds to (sent with the feedback record so the server doesn't have
  // to reconstruct the thread).
  function wireTools(a, getRawText, question) {
    const { tools, copyBtn, likeBtn, dislikeBtn } = a;
    copyBtn.addEventListener("click", () => {
      const text = getRawText();
      if (!text) return;
      (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject())
        .then(() => toast("Copied answer"))
        .catch(() => toast("Couldn't copy to clipboard", true));
    });
    const rate = (rating, btn) => btn.addEventListener("click", () => {
      const answer = getRawText();
      if (!answer) return;
      const active = btn.classList.contains("is-rated");
      btn.classList.remove("is-rated", "dislike");
      likeBtn.classList.remove("is-rated");
      dislikeBtn.classList.remove("is-rated", "dislike");
      if (!active) {
        btn.classList.add("is-rated");
        if (rating === "dislike") btn.classList.add("dislike");
        fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: question || "", answer: answer, rating: rating }),
        }).catch(() => toast("Couldn't save feedback", true));
        if (rating === "dislike") toast("Noted — teach it the right answer in Library → Recent mis-answers");
        else toast("Thanks — helps me calibrate the index");
      }
    });
    rate("like", likeBtn);
    rate("dislike", dislikeBtn);
  }

  // Restore a persisted message (history load). The feedback question is
  // the nearest preceding user turn, found by walking the rendered rows.
  function renderHistoryMessage(m) {
    if (m.role === "user") {
      const row = el("div", "msg msg-user");
      row.appendChild(el("div", "msg-bubble", m.content));
      row.appendChild(el("div", "msg-meta", fmtTime(m.created_at)));
      inner.appendChild(row);
      return;
    }
    let question = null;
    for (const child of Array.from(inner.children).reverse()) {
      if (child.classList.contains("msg-user")) {
        question = child.querySelector(".msg-bubble").textContent;
        break;
      }
    }
    const a = addAssistant();
    a.bubble.innerHTML = renderMarkdown(m.content);
    a.row.querySelector(".msg-meta").textContent = fmtTime(m.created_at);
    wireTools(a, () => m.content, question);
    scrollDown();
  }

  // The one send path: Enter, the Send button, and (later) suggestion chips
  // all funnel through here.
  async function send() {
    const text = inputEl.value.trim();
    if (!text || streaming) return;

    followScroll = true;
    addUser(text);
    inputEl.value = "";
    autoGrow();
    setStreaming(true);

    const asst = addAssistant();
    const typing = makeTyping();
    asst.bubble.appendChild(typing.node);
    let raw = "";

    const state = { asst, typing, raw: () => raw };
    const question = text;

    try {
      abortCtrl = new AbortController();
      const resp = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
        signal: abortCtrl.signal,
      });
      await readSse(resp, (event, data) => {
        if (event === "status") return; // dots already up
        if (event === "token") {
          if (!raw) typing.stop(); // first real text: retire the dots
          if (typing.node.parentNode) typing.node.remove();
          raw = data.text;
          asst.bubble.innerHTML = renderMarkdown(raw);
          scrollDown();
        } else if (event === "done") {
          if (typing.node.parentNode) { typing.stop(); typing.node.remove(); }
          renderSources(asst.bubble, data.sources, data.stopped);
        } else if (event === "error") {
          if (typing.node.parentNode) { typing.stop(); typing.node.remove(); }
          if (raw) {
            // Mid-stream failure: keep the partial text, append the error.
            renderError(asst.bubble, data.message);
          } else {
            asst.bubble.innerHTML = "";
            renderError(asst.bubble, data.message);
          }
        }
      });
    } catch (err) {
      if (typing.node.parentNode) typing.stop();
      if (typing.node.parentNode) typing.node.remove();
      if (err && err.name === "AbortError") {
        // Client aborted: the server got the disconnect + /api/chat/stop;
        // show what we have as a stopped answer (or a bare stopped note).
        if (!raw) { asst.bubble.innerHTML = ""; renderSources(asst.bubble, null, true); }
        else renderSources(asst.bubble, null, true);
      } else if (!raw) {
        asst.bubble.innerHTML = "";
        renderError(asst.bubble, "Couldn't reach the server. " + (err && err.message ? err.message : ""));
      } else {
        renderError(asst.bubble, "Connection dropped mid-answer. " + (err && err.message ? err.message : ""));
      }
    } finally {
      wireTools(asst, () => raw, question);
      setStreaming(false);
      abortCtrl = null;
    }
  }

  function stopStreaming() {
    if (!streaming) return;
    if (abortCtrl) abortCtrl.abort();            // stop rendering immediately
    fetch("/api/chat/stop", { method: "POST" }).catch(() => {}); // release the model stream now
  }

  function setStreaming(on) {
    streaming = on;
    sendBtn.hidden = on;
    stopBtn.hidden = !on;
    if (on) inputEl.focus();
  }

  // Autosize: grow with content (up to max-height), reset after send.
  function autoGrow() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + "px";
  }
  inputEl.addEventListener("input", autoGrow);
  inputEl.addEventListener("keydown", (e) => {
    // Enter sends; Shift+Enter makes a newline. Ignore IME composition
    // (composing CJK/emoji with Enter must not fire a send).
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  });
  sendBtn.addEventListener("click", send);
  stopBtn.addEventListener("click", stopStreaming);

  clearBtn.addEventListener("click", () => {
    if (!window.confirm("Delete your saved chat history? This can't be undone.")) return;
    fetch("/api/history", { method: "DELETE" })
      .then(() => {
        renderWelcome(null);
        toast("Chat history cleared");
      })
      .catch(() => toast("Couldn't clear history", true));
  });

  // ------------------------------------------------------------ first load

  async function loadHistory() {
    try {
      const data = await (await api("/api/history")).json();
      const msgs = data.messages || [];
      if (!msgs.length) {
        // A fresh user gets the welcome state (with real-document chips if
        // the index already has content) instead of a blank window.
        let sugg = null;
        try { sugg = (await (await api("/api/summary")).json()).suggestions; } catch (_e) {}
        renderWelcome(sugg);
      } else {
        for (const m of msgs) renderHistoryMessage(m);
        scrollDown(true);
      }
    } catch (_e) {
      renderWelcome(null);
    }
  }

  // -------------------------------------------------------------- top bar

  const chipEl = $("#index-chip");

  async function refreshChip() {
    try {
      const s = await (await api("/api/summary")).json();
      chipEl.textContent =
        s.documents + " documents · " + s.chunks + " chunks · Drive synced " + fmtAge(s.last_sync);
    } catch (_e) { /* session expired mid-poll: the next action redirects */ }
  }

  // -------------------------------------------------------- view switching

  const navBtns = Array.from(document.querySelectorAll(".nav-btn"));
  const views = { chat: $("#view-chat"), library: $("#view-library") };
  let libraryLoaded = false;

  for (const b of navBtns) {
    b.addEventListener("click", () => {
      const view = b.dataset.view;
      navBtns.forEach((x) => {
        const active = x === b;
        x.classList.toggle("is-active", active);
        x.setAttribute("aria-selected", active ? "true" : "false");
      });
      for (const k of Object.keys(views)) views[k].classList.toggle("is-active", k === view);
      if (view === "library") {
        refreshStats();
        if (!libraryLoaded) { libraryLoaded = true; refreshFeedback(); }
      } else {
        inputEl.focus();
      }
    });
  }

  // --------------------------------------------------------------- library

  const statsEl = $("#index-stats");

  // force=true bypasses the server's brief source-metadata cache (the manual
  // refresh button); the 60s auto-refresh stays cached, which is cheap.
  async function refreshStats(force) {
    try {
      const s = await (await api(force ? "/api/library/summary?force=1" : "/api/library/summary")).json();
      statsEl.innerHTML = "";
      const line = (key, val) => {
        const d = el("div", "stat-line");
        d.appendChild(el("span", "stat-key", key));
        d.appendChild(el("span", "stat-val", val));
        statsEl.appendChild(d);
      };
      line("Documents", String(s.documents));
      line("Indexed chunks", String(s.chunks));
      line("Drive synced", fmtAge(s.last_sync));
      if (s.top_sources && s.top_sources.length) {
        const d = el("div", "stat-line");
        d.appendChild(el("span", "stat-key", "Top sources"));
        const v = el("span", "stat-val");
        s.top_sources.slice(0, 5).forEach((t, i) => {
          const span = el("span", "top-source", t.source + " (" + t.chunks + ")");
          v.appendChild(span);
          if (i < Math.min(s.top_sources.length, 5) - 1) v.appendChild(document.createElement("br"));
        });
        d.appendChild(v);
        statsEl.appendChild(d);
      }
    } catch (_e) { /* ignore */ }
  }
  $("#btn-stats-refresh").addEventListener("click", async () => {
    await refreshStats(true);
    toast("Index stats refreshed");
  });

  // Streaming line-based task (upload/sync/teach share the same SSE shape).
  function runSseTask(path, init, statusEl, body) {
    const setBusy = (msg) => {
      statusEl.className = "status is-busy";
      statusEl.innerHTML = "";
      statusEl.appendChild(el("span", "spinner"));
      statusEl.appendChild(document.createTextNode(msg || "Working…"));
    };
    setBusy(init || "Starting…");
    return api(path, { method: "POST", body: body })
      .then((resp) => readSse(resp, (event, data) => {
        if (event === "progress") {
          statusEl.className = "status is-busy";
          statusEl.innerHTML = "";
          statusEl.appendChild(el("span", "spinner"));
          statusEl.appendChild(document.createTextNode(data.message));
        } else if (event === "done") {
          statusEl.className = "status is-done";
          statusEl.textContent = data.message;
        } else if (event === "error") {
          statusEl.className = "status is-error";
          statusEl.textContent = data.message;
        }
      }))
      .catch((err) => {
        statusEl.className = "status is-error";
        statusEl.textContent = (err && err.message) ? err.message : "Task failed.";
      });
  }

  // Uploads: single files send their base name; folder files send the
  // relative path (webkitRelativePath) so citations keep structure.
  async function uploadFiles(list, statusEl, usePaths) {
    if (!list || !list.length) return;
    const fd = new FormData();
    for (const f of list) {
      const name = usePaths ? (f.webkitRelativePath || f.name) : f.name;
      fd.append("files", f, name);
    }
    await runSseTask("/api/library/upload", "Uploading " + list.length + " file(s)…", statusEl, fd);
  }

  function wireDropzone(zoneId, inputId, statusId, usePaths) {
    const zone = $(zoneId);
    const input = $(inputId);
    const status = $(statusId);
    input.addEventListener("change", () => {
      uploadFiles(Array.from(input.files), status, usePaths);
      input.value = ""; // re-selecting the same file should work
    });
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("is-dragover"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("is-dragover"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("is-dragover");
      const files = Array.from(e.dataTransfer.files || []);
      if (files.length) uploadFiles(files, status, usePaths);
    });
  }
  wireDropzone("#dz-files", "#file-input", "#status-files", false);
  wireDropzone("#dz-folder", "#folder-input", "#status-folder", true);

  $("#btn-sync").addEventListener("click", function () {
    this.disabled = true;
    runSseTask("/api/library/sync", "Starting…", $("#status-sync"))
      .finally(() => { this.disabled = false; });
  });

  $("#btn-teach").addEventListener("click", () => {
    const body = JSON.stringify({
      question: $("#teach-question").value,
      answer: $("#teach-answer").value,
    });
    runSseTask("/api/library/teach", "Indexing the correction…", $("#status-teach"),
      new Blob([body], { type: "application/json" }));
  });

  // Mis-answer review: pick a dislike -> prefill the correction form.
  const fbSelect = $("#fb-select");
  const fbPreview = $("#fb-preview");
  let fbRows = [];

  async function refreshFeedback() {
    try {
      fbRows = await (await api("/api/library/feedback")).json();
    } catch (_e) { fbRows = []; }
    fbSelect.innerHTML = "";
    if (!fbRows.length) {
      fbSelect.appendChild(new Option("— no dislikes recorded yet —", ""));
      fbPreview.innerHTML = "";
      return;
    }
    for (const row of fbRows) {
      fbSelect.appendChild(new Option(row.created.slice(0, 16) + "  " + String(row.question).slice(0, 80), String(row.id)));
    }
    showFbPreview(fbRows[0]);
  }

  function showFbPreview(row) {
    if (!row) { fbPreview.innerHTML = ""; return; }
    fbPreview.innerHTML = "";
    fbPreview.appendChild(el("span", "fb-q", (row.question || "").slice(0, 200)));
    fbPreview.appendChild(el("span", "fb-a", "Chatty answered: " + (row.answer || "").slice(0, 400)));
  }

  fbSelect.addEventListener("change", () => {
    const row = fbRows.find((r) => String(r.id) === fbSelect.value);
    if (row) $("#teach-question").value = row.question || "";
    showFbPreview(row);
  });
  $("#btn-fb-refresh").addEventListener("click", refreshFeedback);

  // ----------------------------------------------------------------- init

  refreshChip();
  setInterval(refreshChip, 60000);
  refreshStats();
  setInterval(refreshStats, 60000);
  loadHistory();
  inputEl.focus();
})();
