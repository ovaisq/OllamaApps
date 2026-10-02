# Markdown RAG Chat Assistant — pgvector & ChromaDB backends, custom web UI

This project provides a pipeline for indexing markdown content and chatting
with it locally via Ollama, with two interchangeable storage backends:

* **pgvector variant** (`pgv_chatty.py`, `pgv_indexer.py`): chunks + embeddings
  in PostgreSQL with HNSW indexing (`pgv_config.py.template`, `pgv_schema.sql`).
* **ChromaDB variant** (`chromadb_chatty.py`, `chromadb_indexer.py`): the same
  RAG flow over a local Chroma persistent store (`chroma_config.py.template`).

Both front-ends are the same hand-built single-page web UI (vanilla JS, no
build step, in `static/`) served by a thin FastAPI app (`api_routes.py`) that
delivers chat as JSON + SSE. Gradio is gone.

---

## Features

* **Markdown Chunking**: Splits markdown documents into manageable chunks.
* **Vector Embeddings**: Embeds chunks via an Ollama embedding model.
* **Semantic Retrieval**: Real pgvector/Chroma nearest-neighbor search, with
  name-routing for queries that explicitly mention a document.
* **Streaming chat UI**: Token streaming with typing indicator, elapsed
  timer, smart scroll, per-session Stop, and a sources footer citing the
  chunks the answer drew on.
* **Session history**: SQLite/Postgres-persisted per-user chat history that
  reloads after a sign-in.
* **Library tab** (see below): upload files/folders, Google Drive sync,
  teach-from-misanswer, index stats.

## Architecture

* `api_routes.py` — the only delivery-aware module: web UI + JSON/SSE routes,
  the single chat-turn engine (`chat_stream_events`), and the
  sync-generator→SSE bridge (`sse_response`). `build_app(chat)` is shared by
  both backends, so the delivered UI can't drift between them.
* `rag_common.py` — backend-agnostic RAG plumbing: Ollama client helpers,
  retry/backoff, prompt building, source-payload builders, per-user stop
  events (`StopEvents`).
* `library.py` — pure handlers for upload / Drive sync / teach, streamed as
  SSE `progress`/`done` events.
* `auth_routes.py` — Google-OAuth session gate (middleware); everything but
  `/health` and `/static/*` requires an allowlisted account.
* `static/` — the UI: `index.html`, `app.js`, `styles.css`, vendored
  `marked.min.js` + DOMPurify (`vendor/`) for markdown rendering.

Chat SSE event protocol: `status` (thinking), `token` (cumulative text),
`done` (sources + stopped flag) or `error` (safe message).

## Configuration

* **pgvector variant**: copy `pgv_config.py.template` to `pgv_config.py` and set env vars (`DB_NAME`, `DB_USER`, `DB_HOST`, `OLLAMA_HOST`, etc.) or export them directly.
* **ChromaDB variant**: copy `chroma_config.py.template` to `chroma_config.py` and set env vars similarly (e.g. `CHROMA_DB_PATH`, `CHROMA_COLLECTION`).

Both `*_config.py` files are gitignored — never commit real credentials.

```bash
cd chatbot
python3 chromadb_chatty.py    # ChromaDB backend
python3 pgv_chatty.py         # PostgreSQL/pgvector backend
```

## Docker Compose

```bash
cp .env.example .env   # fill in real DB/Ollama/Google Drive values
docker compose up -d --build
```

This builds the chatbot image (pgv_chatty.py by default) plus a pgvector-enabled
Postgres. `.env` is required — `docker compose` looks for it in the same
directory as `docker-compose.yml` and refuses to start without the vars it
references (`DB_PASSWORD` in particular).

## Logging in / Library tab

The whole app is gated behind Google sign-in — nothing is reachable except
`GET /health` and static assets until you log in with an allowlisted account:

1. Set `SESSION_SECRET` (e.g. `openssl rand -hex 32`) and `ALLOWED_EMAILS`
   (comma-separated Google account emails) in `.env`. Empty `ALLOWED_EMAILS`
   means nobody can log in — fails closed, not open.
2. Visit `https://<your-host>:7860/` — you'll be redirected to `/login`, then
   to Google's consent screen (identity + Drive readonly, one login covers
   both). API calls without a valid session get a 401 JSON body.
3. On success you land back on the chat UI with a session cookie (7-day
   default, `SESSION_MAX_AGE_SECONDS`). `/logout` clears it.

Once logged in, the **Library** tab lets you:
* Upload a `.md`/`.txt`/`.pdf`/`.doc`/`.docx`/`.xlsx` file (or a folder) to
  index immediately (no CLI/SSH needed).
* Click "Sync Google Drive now" to pull new/changed Drive content on demand
  (Google Docs/Sheets, Word files, .md/.txt/.xlsx, PDFs).
* Teach Chatty a correction from a mis-answer you flagged with 👎.
* See live index stats (document/chunk counts, top sources, last Drive sync),
  with a manual refresh that busts the brief server-side metadata cache.

`gdrive_indexer.py` (the CLI/cron path) uses the same stored refresh token,
so logging in once as an allowlisted account also unlocks scheduled syncs.

## Production notes

* Retrieval uses real pgvector/Chroma nearest-neighbor search over query embeddings (not chunk length).
* `pgv_chatty.py` uses a pooled, health-checked DB connection (`psycopg2.pool.ThreadedConnectionPool`).
* Both chat apps validate message length, retry transient Ollama failures with backoff, and never leak internal exception details to end users (a correlation `error id` is logged server-side instead).
* Both apps expose `GET /health` (200/503) for k8s liveness/readiness probes; every other route requires a valid session (API clients get 401 JSON, browsers redirect to `/login`).
* The Stop button only stops the requesting browser session's stream, not every concurrent user's; the SSE response also releases when the tab closes.
* OAuth `state` is verified one-shot on `/oauth2callback` (no replay), and a login from a non-allowlisted Google account is rejected without ever touching the stored Drive refresh token.

## Tests

```bash
cd chatbot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest pgvector
pytest tests/
```
