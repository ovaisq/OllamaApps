# Chatty — UI/UX Redesign Spec (v1, Gradio — superseded)

> **Status: Superseded.** This spec documents the v1 Gradio redesign and was
> implemented 2026-09-29. On 2026-10-02 the Gradio front-end was replaced
> wholesale: the chat UI is now a hand-built single page in `static/`
> (vanilla JS + vendored marked/DOMPurify, no build step), served by
> `api_routes.py` (JSON + SSE over FastAPI) with `library.py` handlers.
> `admin_ui.py`/`ui_common.py` were deleted, `gradio` dropped from
> `requirements.txt`, and the Google-OAuth gate (`auth_routes.py`) plus both
> RAG backends are unchanged. The event protocol and behaviors specified
> below were carried over 1:1 to the SSE/JSON surface.

Status: **Implemented** (P0 + P1 + P2, 2026-09-29 — 162 tests passing, incl. new
coverage for every redesigned behavior)
Scope (as built): `chatbot/` — `chromadb_chatty.py`, `pgv_chatty.py`, `admin_ui.py`, `auth_routes.py`, `rag_common.py`, new `ui_common.py`
Companion to: `AGENTS.md` (repo conventions)

As-implemented deltas (all forced by the Gradio 6.28 API or better behavior):
- Clear-chat confirm uses `window.confirm` via the event's `js=` arg (Gradio 6
  has no `gr.Modal`).
- Composer stays **single-line** (Enter submits in 6.28; verified in the
  frontend bundle that multi-line uses Shift+Enter-to-submit), so no
  Shift+Enter-for-newline JS was added.
- Auth flow gained a `/oauth-start` route (the `/login` landing page's
  button), as §7 sketches.
- Feedback review is a `Dropdown` (with a preview + Refresh), not a
  clickable Dataframe — Gradio 6 has no `Dataframe.click`.
- Theme is `gr.themes.Default(primary_hue=Color(<brand ramp>))` (the
  `Builder`-style hex constructor from the spec doesn't exist in 6.28);
  periodic refresh uses `gr.Timer(60).tick(...)` (component `fn=` removed).

---

## 1. Current State & Problems

The app works, but ships as two visually different "products" and lacks the
feedback states people expect from a chat UI.

### 1.1 The two variants disagree with each other

| Element | `chromadb_chatty.py` | `pgv_chatty.py` |
|---|---|---|
| Theme | *none* (Gradio default) | `JohnSmith9982/small_and_pretty` |
| Chatbot label | *(unlabeled)* | `"Chat History"` |
| Textbox label | `"Ask about the README"` (stale — pre-dates docs/Drive) | `"Your Message"` |
| Stop button | `"Stop Chat"` | `"Stop Response"` |

Same product, different names, fonts, and colors depending on which backend is
running. Root cause: `build_app()` is duplicated in both files with drift.

### 1.2 Chat tab

- **No Send button** — Enter-to-submit only; awkward on mobile keyboards.
- **Stop / Clear History buttons float** between the textbox and nothing else,
  ungrouped from the composer they belong to.
- **No source citations** — answers come from retrieved chunks
  (`retrieve_context` already returns source metadata, `rag_common.py:215`)
  but the user never sees *which* documents answered; `format_source_label`
  (`rag_common.py:344`) is prompt-side only.
- **No empty state** — a fresh user gets a blank window with a textbox labeled
  "Ask about the README".
- **Errors render as plain assistant text** — `safe_error_message`
  (`rag_common.py:665`) is visually indistinguishable from an answer, as are
  "(stopped)" and "(no response content)" (`chromadb_chatty.py:419`).

### 1.3 Admin tab

- One long **single-column scroll**: Upload files → Upload folder → Google
  Drive → Teach correction → Index status. On a laptop you scroll past uploads
  to reach stats (`admin_ui.py:146-207`).
- **Index stats are stale at rest** — only update on manual "Refresh stats"
  click; no counts shown on page load.
- Two near-identical `gr.File` upload boxes stacked with two separate status
  lines — reads as duplicated, not as "two ways to add the same content".
- No record of dislikeds anywhere — feedback is persisted
  (`record_feedback`) but the UI never shows it back.

### 1.4 Auth

- OAuth failure paths return **raw JSON 400/403 bodies**
  (`auth_routes.py:75,82,94,108`) — a user who hits a bad Google consent flow
  or uses a non-allowlisted account sees:
  `{"status":"error","detail":"not authorized"}`.
- No branded login page: `/login` is a bare redirect to Google; on the way
  back from a *denied* consent there is no "Chatty" screen at all.

### 1.5 Portability

- Hard-coded LAN Ollama IP and port 7860 (fine today; noted for the
  monorepo-level issue where all three Gradio apps default to 7860).

---

## 2. Design Goals

1. **One identity for Chatty** — both backends render byte-identical chrome:
   same theme, same labels, same layout. One shared UI builder, one theme.
2. **A real chat surface** — composer with Send, sources under answers,
   distinct error/stopped states, a friendly empty state.
3. **A scannable Admin tab** — content, sync, and learning at a glance on one
   laptop screen; live index stats.
4. **Failures that read as failures** — styled auth error pages, styled
   in-chat error bubbles. Never a JSON body, never an error dressed as an
   answer.
5. **Keep the streaming UX we already own** — the typing indicator + elapsed
   timer + smart-scroll JS (`rag_common.py:71-138`) is good; the redesign
   keeps it and builds around it.

Non-goals: multi-user UX, dark mode (P2 consideration, §8), PWA (the app is
behind Google OAuth + LAN use), reworking RAG internals.

---

## 3. Information Architecture

```
Chatty                              ← full-height, no page scroll on desktop
├── Top bar (sticky)
│   ├── Brand: "Chatty" + subtitle "Document & Drive Assistant"
│   ├── Index chip:  "412 chunks · Drive synced 2 h ago"   (auto, 60 s)
│   └── [Sign out]
├── Tabs
│   ├── Chat
│   │   ├── Message list (existing chatbot, type="messages")
│   │   │   ├── welcome (empty state) for new users
│   │   │   ├── assistant messages: answer + "Sources: a.md, b.pdf" footer
│   │   │   ├── error messages:  ⚠ styled bubble + error id
│   │   │   └── stopped messages: "⏹ Stopped" + partial text
│   │   └── Composer (grouped row, sticky at bottom of tab)
│   │       ├── [Send]  [⏹ Stop]  [🗑 Clear]
│   │       └── Textbox: "Ask about your documents…"  (Enter sends)
│   └── Library
│       ├── Row of 3 cards
│       │   ├── Add files      (multi-file upload + status)
│       │   ├── Add a folder   (directory upload + status)
│       │   └── Google Drive   (status + "Sync now")
│       ├── Row of 2 cards
│       │   ├── Teach a correction  (Q/A + submit + status)
│       │   └── Index         (auto-refreshing stats: total chunks,
│       │                      documents count, last sync, top sources)
│       └── (P2) Recent feedback — dislikeds table → "Teach correction"
```

Notes:

- **Admin → "Library"**: the tab manages the user's content, not the
  server. (If "Admin" must survive in muscle memory, label it
  "Library (Admin)").
- Top bar replaces the current `gr.Row()` of `# Chatty: Document & Drive
  Assistant` + sign-out markdown (`chromadb_chatty.py:486-488`) with a proper
  app bar: brand + live index chip + sign-out button.

---

## 4. Design System

Define **once** as `gr.themes.Builder`, used by both `build_app()` variants.
No more theme name in one variant and none in the other.

### 4.1 Tokens

| Token | Value | Rationale |
|---|---|---|
| Primary | deep teal/blue `#2563a7` (or `Builder(primary_hue=...)` equivalent) | Calm, document-ish; distinct from CodeReviewAssistant's blue default |
| Background | Gradio light (keep the theme family light) | Chatty is read-mostly; dark mode deferred |
| Font (body) | theme default sans | consistency |
| Font (code/mono) | theme default mono | source names, error ids |
| Radii | medium (theme default) | match Gradio 6 chrome so custom CSS blends |
| Error | theme error hue + ⚠ glyph | see §5.4 |
| Success/status | theme success hue, for "Synced…", "Indexed N" | status lines in Library tab |

Implementation: new module `chatbot/ui_common.py` (sibling of
`admin_ui.py`/`rag_common.py`) holding:

- `CHATTY_THEME` — the `gr.themes.Builder(...)` instance
- `CHATTY_CSS` — consolidated custom CSS (current `#signout-link` hack,
  `TYPING_INDICATOR_CSS`, plus new composer/error/source-footer styles below)
- `build_chat_tab(...)` — the Chat-tab builder (see §6.1)
- `index_chip(...)` — the auto-refreshing stats markdown for the top bar

Both `build_app()`s then become: top bar → `build_chat_tab(...)` →
`build_admin_tab(...)` (renamed, see §6.2) → `mount_gradio_app(theme=CHATTY_THEME, css=CHATTY_CSS)`.
The per-variant `build_app()` keeps only backend-specific wiring (which
`index_text`/`count_chunks`/`sync_drive` to inject) — layout and labels live
in exactly one place, so drift becomes impossible.

### 4.2 Component vocabulary (labels must match across variants)

| Component | Label |
|---|---|
| Chatbot | `Chat` (no "Chat History" — history is implicit) |
| Textbox | placeholder `Ask about your documents…`; label `Message` |
| Send | `Send` (primary variant) |
| Stop | `Stop` — enabled only while a response is streaming (toggled in `respond`'s first/last yield) |
| Clear | `Clear chat` — with a confirm: `gr.Modal` or `confirm.js`; it deletes **persisted** history, so it must not be one misclick away |
| Index chip | `412 chunks · 37 documents · Drive synced 2 h ago` (or `never synced`) |

---

## 5. Chat Tab — Flows & States

### 5.1 Normal ask (existing, kept)

Submit → instant typing indicator + elapsed timer (unchanged,
`rag_common.py:24-48,71-138`) → stream replaces dots from first token
(unchanged) → **Sources footer** appears (new, §5.2).

### 5.2 Sources footer (new)

`respond()` currently throws away the metadata it retrieves. Change
`get_answer_stream` to also return the source labels used (unique, in
retrieval order, via existing `format_source_label`). On stream completion
append, in the same answer bubble:

```
—
Sources: Kona-April-2026.pdf | owner: Jane | ·  2019_resume.pdf
```

as a muted, smaller footer (CSS class `.chatty-sources`).
**Spike first:** check whether Gradio 6's `type="messages"` per-message
`metadata/` data field renders a cleaner footer; if so, prefer it over text.
Catalog-style answers ("list all PDFs") show instead:
`Sources: index metadata (metadata query, not a content search)`.

### 5.3 Empty state (new)

On page load with no persisted history, seed one assistant message:

> **Chatty here.** Ask about anything in your documents — Drive and uploads
> are all searchable. Try: *"What's in my Kona trip doc?" ·
> "List all PDFs" · "Who shared the Q3 report with me?"*

(Example questions sourced from real indexed filenames when available, via
the existing `list_sources()` cached lookup.)

### 5.4 Error & stopped states (new)

`respond()`'s three degraded outcomes are currently plain text. Restyle:

| Outcome | Render |
|---|---|
| Exception | assistant bubble with ⚠ prefix + error bubble styling (`.chatty-error`: theme error-hue border/background) + `error id: abc12345` in mono, and a hint line "Check the server log" |
| Stopped before first token | `⏹ Stopped` (muted, not an error style) |
| Stopped mid-stream | partial text preserved + `⏹ Stopped` footer line |
| No response content | `⚠ The model returned no content (error id: …)` |

Mechanism: wrap these in a markdown sentinel the CSS hooks (e.g. a
`<div class="chatty-error">` in the message content — content is already
HTML-rendered for the typing indicator, so this is the same channel).

### 5.5 Composer (new grouping)

Current order: `chatbot, textbox, stop_btn, [state], clear_btn` — four
siblings. New order, all in one `gr.Row` under the chatbot:

```
[ [Send] [⏹ Stop] [Clear chat] ]  [ Message: Ask about your documents…        ]
```

- Single `gr.Row`: a compact control `gr.Column` (wrap-enabled) + the
  textbox with matching height.
- **Send** button wired to the same `chat.respond` as `msg.submit`; Enter
  still submits (shared handler, so both paths are identical).
- **Stop** disabled at rest: `respond`'s first yield returns a flag output
  (add one `gr.State` or use the existing `state` dict) that enables it;
  final yield disables it. (One extra yield output or a
  `gr.update(disabled=...)` via a small helper — decide in spike.)
- **Clear chat** opens a `gr.Modal` ("Delete your saved chat history?") →
  `chat.clear_chat_ui`. Current behavior deletes **persisted** history
  (`clear_history`, `chromadb_chatty.py:138`) — a confirm is mandatory for
  that.
- Keyboard: Enter = send (existing), Shift+Enter = newline (small JS add-on
  on the composer textarea; if brittle, document Enter-only as the rule).

### 5.6 Like/Dislike (kept, minor)

`chatbot.like(...)` stays. Dislike is the seed for "Teach a correction";
add a one-line hint next to the feedback affordance only on first use
(a dismissible admin-banner is out of scope — text in the tab header is
enough).

---

## 6. Tab Structure

### 6.1 `build_chat_tab(chat, state)` — in `ui_common.py`

Wraps §5. Both variants call it with their `respond`/`stop_chat`/
`clear_chat_ui`/`load_history_ui`/`record_feedback` bound. No labels,
layout, or CSS live in `chromadb_chatty.py` / `pgv_chatty.py` anymore.

### 6.2 Library tab (redesigned `build_admin_tab`)

Replace the five stacked sections with two `gr.Row`s of cards:

**Card grid — Row 1:**

| Add files | Add a folder | Google Drive |
|---|---|---|
| `gr.File(file_count="multiple", file_types=[.md,.txt,.pdf,.xlsx])` | `gr.File(file_count="directory")` | Status line (auto: last sync + outcome) + `Sync Google Drive now` |
| Status: live "Indexing i/N…" (unchanged generator) | Same handler, separate status | Unchanged `sync_drive_now` |

- One shared `upload_and_index` handler (already shared internally), two
  distinct status lines preserved (parallel uploads otherwise interleave).
  Visually the two share a caption: "Pick files, or a whole folder —
  .md .txt .pdf .xlsx are indexed, anything else is skipped and named."

**Card grid — Row 2:**

| Teach a correction | Index |
|---|---|
| Q / A textboxes (unchanged copy) + `Teach this correction` + status | **Auto-refreshing** (`every=60`): total chunks, distinct documents, top 5 sources by chunk count, last Drive sync. Replaces manual "Refresh stats" (keep a manual refresh icon next to it). |

Index card needs one new backend read (distinct source count; pg variant
already has it cheaply via SQL `DISTINCT source`; Chroma variant via the
existing cached `list_sources()`). Last-sync time: persist a timestamp file
next to the token store on sync completion (write in the `finally` of
`sync_drive` in both variants).

Responsive: rows collapse to single column below ~900px (theme default
behavior of `gr.Row` with enough children handled by a media-query in
`CHATTY_CSS` if needed).

### 6.3 Top bar

- `gr.Row` with brand markdown (`# Chatty` + subtitle, smaller),
  `index_chip` markdown (shared `every=60` source), and a real
  `gr.Button("Sign out", size="sm")` linked-styled (use
  `gr.HTML('<a href="/logout">…')` in the button, or keep
  `gr.Markdown("[Sign out](/logout)")` styled — spike which survives
  theming).
- The current `#signout-link` CSS hack is replaced by the builder layout
  (flex space-between in `CHATTY_CSS`).

---

## 7. Auth Screen (new)

Replace raw JSON failures with branded HTML (no new deps — inline
`HTMLResponse`, same tokens as `CHATTY_THEME`):

| Route | Today | Redesign |
|---|---|---|
| `/login` (GET, no session) | redirect to Google | **Login landing page**: Chatty brand, one button "Sign in with Google →" that `location.href`s to the OAuth flow. (Middleware currently redirects all unauthenticated GETs to `/login` *before* it can redirect again — so `/login` must render a page if hit directly after consent denial; keep the auto-redirect for the *app* path `/`, and serve the page at `/login` when `?page=1`, or more simply: middleware redirects app paths to `/login`, `/login` shows the landing page, landing-page button POSTs/GETs `/oauth-start` which does the Google redirect. This also fixes the current double-redirect loop risk where consent-denied → `/login` would infinite-redirect.) |
| OAuth `error` / missing code | JSON 400 | `/auth-error?detail=…` style page: "Sign-in didn't complete" + the detail (sanitized: **never** the full `detail` for stack-ish messages; map known errors to user copy) + "Try again" button |
| State mismatch / replay | JSON 403 | Same page, "Session expired — sign in again" |
| Non-allowlisted email | JSON 403 `not authorized` | **Same** page, generic copy only ("That account isn't authorized for this Chatty instance") — do **not** echo the email back; keep server-side log as-is |
| Code-exchange / userinfo failure | JSON 400 | Same page, generic copy + error id (reuse `safe_error_message` pattern, `rag_common.py:665`) |

Security notes (must stay true after redesign):

- No new cookies/headers; errors stay `httponly`-free HTML, no
  `refresh_token` ever written on rejected accounts (existing behavior,
  `auth_routes.py:104-108`).
- `detail` values from Google (`exchange_code_for_tokens` RuntimeError) may
  contain URLs/tokens in query strings — **strip query strings** before
  display, or map to fixed copy per error family.

---

## 8. Out of Scope / Later

- **P2 — Dark mode:** ship CSS variables for `CHATTY_THEME` tokens so a dark
  `Builder` can be added without layout changes; do not build it now.
- **P2 — Feedback review:** read the `feedback` table, list last 20 dislikeds
  in the Library tab with a "Teach correction →" button that pre-fills the
  correction card (wire: `gr.Dataframe`/`Markdown` + a `gr.State` carrying
  the prefill).
- **P2 — Mobile polish:** composer sticky-on-scroll, tap targets ≥ 44px.
- **Monorepo (separate ticket):** all Gradio apps collide on port 7860;
  hard-coded LAN IP in several projects.

---

## 9. Phased Implementation Plan

**P0 — One identity (do first, small diff):**
1. New `ui_common.py`: `CHATTY_THEME`, `CHATTY_CSS`, `build_chat_tab(...)`.
2. Refactor both `build_app()`s to use it; delete duplicated layout.
3. Label unification per §4.2.
4. Auth: branded `/login` page + `/auth-error` page; remove JSON 400/403s.

Verification: screenshot/inspect both variants side by side; labels, theme,
layout identical; `GET /login`, broken-callback, and wrong-email paths all
show the branded page (no JSON anywhere).

**P1 — Chat surface:**
5. Composer row: Send / Stop (stream-gated) / Clear with confirm modal;
   Shift+Enter newline JS.
6. Sources footer (spike Gradio metadata field first, §5.2).
7. Empty-state welcome message.
8. Error/stopped styling (§5.4).

Verification: fresh-browser chat (welcome → ask → see sources), mid-stream
Stop, Stop at 0 tokens, oversized-message rejection, cleared history
persists across reloads, confirm dialog fires.

**P2 — Library tab + extras:**
9. Card grid layout, single shared caption, index card with `every=60`
   stats + last-sync timestamp file.
10. Feedback review list → prefill correction card.
11. Responsive tweaks; (optional) dark-mode token groundwork.

---

## 10. Acceptance Criteria (whole redesign)

- [ ] Both backends render the same theme, labels, and layout.
- [ ] A first-time user sees a welcome state, asks one question, and sees
      which documents answered (source footer).
- [ ] Errors are never rendered as plain answers; no route serves JSON to a
      browser.
- [ ] "Clear chat" always asks for confirmation.
- [ ] Stop is only clickable while a response is streaming.
- [ ] Index chip and Index card update without manual clicks.
- [ ] Existing features intact: typing timer, smart scroll, like/dislike,
      teach-correction, Drive sync gate, per-session stop events.
