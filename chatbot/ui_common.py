"""Shared Chatty UI: theme, CSS/JS, the Chat tab, and the index/feedback
widget formatters.

Both build_app() variants (ChromaDB and pgvector) render the entire UI from
here, so the two backends can never drift into looking like different
products again -- that's what the old per-variant build_app() copies did
(one had a theme, the other didn't; "Stop Chat" vs "Stop Response",
"Ask about the README" vs "Your Message").
"""
import html
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import gradio as gr

from gradio.themes.utils.colors import Color

from rag_common import CHAT_UI_JS, TYPING_INDICATOR_CSS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

# Brand blue ramp centered on #2563a7. Gradio 6 themes take a full 50->950
# Color ramp (a bare hex string is not accepted), so the brand color is
# expressed as its ramp here.
CHATTY_PRIMARY = Color(
    "#f2f6fc",  # c50
    "#e3ecf8",  # c100
    "#c6d9f0",  # c200
    "#9cbbe3",  # c300
    "#6f9ad4",  # c400
    "#4a7cc6",  # c500
    "#2563a7",  # c600 (brand)
    "#1e528f",  # c700
    "#1a4474",  # c800
    "#16385d",  # c900
    "#0f2440",  # c950
    name="chatty",
)

# One theme for both backends (the old state: pgv used a named community
# theme, chroma used the bare default).
CHATTY_THEME = gr.themes.Default(
    primary_hue=CHATTY_PRIMARY,
    secondary_hue="teal",
    neutral_hue="slate",
)

CHATTY_CSS = TYPING_INDICATOR_CSS + """
/* Top app bar: brand + live index chip + sign-out, spread across the row. */
#chatty-topbar { align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; }
#chatty-topbar h1 { font-size: 1.35rem; line-height: 1.15; margin: 0; }
#chatty-chip { font-size: 0.78rem; opacity: 0.7; margin: 0; line-height: 1.4; }
#signout-link { text-align: right; }
@media (max-width: 740px) {
    #chatty-topbar { flex-direction: column; align-items: flex-start; }
    #signout-link { text-align: left; }
}

/* Chat decorations. `respond()` appends these divs to the final message
   content -- the same HTML channel the typing indicator already uses. */
.chatty-sources {
    font-size: 0.78rem; opacity: 0.65; margin-top: 0.5rem; padding-top: 0.4rem;
    border-top: 1px solid rgba(127, 127, 127, 0.25); line-height: 1.5;
}
.chatty-error {
    display: block; border-left: 3px solid var(--color-error-500, #b91c1c);
    background: var(--color-error-50, rgba(185, 28, 28, 0.07));
    padding: 0.5rem 0.75rem; border-radius: 6px;
}
.chatty-stopped { display: block; font-size: 0.8rem; opacity: 0.6; margin-top: 0.4rem; }

/* Composer: Stop reads as a secondary action and is only clickable while
   a response is streaming (COMPOSER_JS mirrors the composer's disabled
   state onto it). */
#chatty-stop button:disabled { opacity: 0.45; }
"""

# Client behavior installed on page load (alongside CHAT_UI_JS which handles
# smart scrolling and the typing timer). Keeps the Stop button enabled only
# while this session's submit event is running: Gradio disables the
# composer's inputs for the duration of an in-flight event, so the
# textbox's disabled state is exactly the "a response is streaming" signal.
COMPOSER_JS = """(function () {
    var wrap = document.getElementById('chatty-composer');
    var stopWrap = document.getElementById('chatty-stop');
    if (!wrap || !stopWrap) { return; }
    var box = wrap.querySelector('textarea');
    var stopBtn = stopWrap.querySelector('button');
    if (!box || !stopBtn) { return; }
    function sync() { stopBtn.disabled = box.disabled; }
    sync();
    new MutationObserver(sync).observe(box, { attributes: true, attributeFilter: ['disabled'] });
    setInterval(sync, 500);
})();"""


def load_js() -> str:
    """All client-side chat behavior, installed once per page load."""
    return CHAT_UI_JS + COMPOSER_JS


# ---------------------------------------------------------------------------
# Index stats formatters (pure -- unit-testable without a backend)
# ---------------------------------------------------------------------------

def humanize_age(timestamp: Optional[float]) -> str:
    """Relative time for the 'Drive synced 2 h ago' line."""
    if not timestamp:
        return "never"
    age = max(0.0, time.time() - timestamp)
    if age < 90:
        return "just now"
    if age < 3600:
        return f"{int(age // 60)} min ago"
    if age < 86400:
        return f"{int(age // 3600)} h ago"
    return f"{int(age // 86400)} d ago"


def format_index_chip(summary: Dict[str, Any]) -> str:
    """One-line index summary for the top bar. Never raises -- a stats
    failure must not break the app around it."""
    try:
        return (
            f"📄 {summary.get('documents', 0)} document(s) · "
            f"{summary.get('chunks', 0)} chunks · "
            f"Drive synced {humanize_age(summary.get('last_sync'))}"
        )
    except Exception:
        logger.warning("Index chip formatting failed", exc_info=True)
        return ""


def format_index_summary(summary: Dict[str, Any]) -> str:
    """Full index card for the Library tab."""
    try:
        lines = [
            f"**Documents:** {summary.get('documents', 0)}",
            f"**Indexed chunks:** {summary.get('chunks', 0)}",
            f"**Drive:** last sync {humanize_age(summary.get('last_sync'))}",
        ]
        top_sources = summary.get("top_sources") or []
        if top_sources:
            lines.append("")
            lines.append("**Top sources (by chunks):**")
            for source, count in top_sources:
                lines.append(f"- {html.escape(str(source))} ({count})")
        return "\n".join(lines)
    except Exception:
        logger.warning("Index summary formatting failed", exc_info=True)
        return ""


INDEX_CHIP_INTERVAL_SECONDS = 60


# ---------------------------------------------------------------------------
# Feedback (dislike review) helpers -- pure, fn-injected so they're
# unit-testable without a backend (same pattern as admin_ui handlers).
# ---------------------------------------------------------------------------

def refresh_feedback_rows(
    feedback_rows_fn: Optional[Callable[[str], List[Dict[str, Any]]]],
    request: "gr.Request" = None,
):
    """Load the signed-in user's recent disliked answers (newest first).

    feedback_rows_fn(email) -> [{"id", "question", "answer", "created"}].
    Returns (dropdown choices, preview markdown, rows-as-state) so the
    dropdown selection can be resolved client-side without a round trip.
    """
    if not feedback_rows_fn:
        return [], "", []
    from app_session import get_email_from_request
    from gdrive_config import AUTH_CONFIG

    email = get_email_from_request(request, AUTH_CONFIG["session_secret"])
    try:
        rows = feedback_rows_fn(email) if email else []
    except Exception:
        logger.warning("Failed to load feedback rows", exc_info=True)
        rows = []
    rows = rows or []
    if not rows:
        return (
            [],
            "No dislikes recorded yet — tap the 👎 under a wrong answer "
            "and it will show up here, ready to be turned into a correction.",
            rows,
        )
    choices = [
        (
            str(row.get("id")),
            f"{str(row.get('created', ''))[:16]}  {str(row.get('question', ''))[:90]}",
        )
        for row in rows
    ]
    newest = rows[0]
    answer_preview = html.escape(str(newest.get("answer", ""))[:400]).replace("\n", " ")
    preview = (
        f"**Most recent:** {html.escape(str(newest.get('question', '')))}\n\n"
        f"Chatty answered:\n\n> {answer_preview}"
    )
    return choices, preview, rows


def feedback_prefill(selected: Optional[str], rows: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Prefill the 'question it answered wrong' box when the user picks a
    dislike from the dropdown. Returns None (gradio leaves the target
    untouched) when the selection can't be resolved, e.g. the dropdown
    was cleared."""
    for row in rows or []:
        if str(row.get("id")) == str(selected or ""):
            return str(row.get("question", ""))
    return None


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

def build_chat_tab(chat: Any, blocks: "gr.Blocks") -> None:
    """Build the Chat tab with the shared layout/labels. `chat` is a bound
    ChromaChat or PGVectorChat instance; both backends expose the same
    handler names (respond / stop_chat / clear_chat_ui / load_history_ui /
    record_feedback), so this stays backend-agnostic.

    Layout: message list, then one grouped composer row --
    [Message …] [Send][Stop][Clear chat].
    """
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(label="Chat", autoscroll=False, height=520)
        state = gr.State(value={})
        with gr.Row(equal_height=True):
            msg = gr.Textbox(
                label="Message",
                placeholder="Ask about your documents…",
                elem_id="chatty-composer",
                scale=5,
            )
            with gr.Column(scale=2, min_width=210):
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary", scale=3)
                    # Gradio 6 has no constructor-level `disabled`; the
                    # button's disabled state is owned by COMPOSER_JS, which
                    # on load mirrors the composer's (at rest) disabled
                    # state onto it. A stray click before the JS runs is
                    # harmless: respond() clears the stop Event next time.
                    stop_btn = gr.Button("Stop", scale=2, elem_id="chatty-stop")
                    clear_btn = gr.Button("Clear chat", size="sm", scale=3, variant="hugging")

        submit_inputs = [msg, chatbot, state]
        submit_outputs = [chatbot, msg, state]
        # One timer: 'full' scoped to the chatbot only (the runtime timer
        # would otherwise render on EVERY output component -- two of them
        # here).
        msg.submit(chat.respond, submit_inputs, submit_outputs,
                   queue=True, show_progress="full", show_progress_on=[chatbot])
        send_btn.click(chat.respond, submit_inputs, submit_outputs,
                       queue=True, show_progress="full", show_progress_on=[chatbot])
        stop_btn.click(chat.stop_chat, [chatbot, state], submit_outputs,
                       show_progress="hidden")
        # Clearing wipes *persisted* history, so it must not be one
        # misclick away: the js runs first and returning false cancels the
        # Python handler.
        clear_btn.click(chat.clear_chat_ui, None, submit_outputs,
                        show_progress="hidden",
                        js="() => window.confirm('Delete your saved chat history?')")
        # Like/Dislike on answers -> persisted for review; a dislike shows
        # up in the Library tab's 'Teach Chatty a correction' prefill.
        chatbot.like(chat.record_feedback, chatbot, show_progress="hidden")
        blocks.load(chat.load_history_ui, None, chatbot, js=load_js(),
                    show_progress="hidden")
