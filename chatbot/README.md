# Markdown Embedding Indexer with PostgreSQL & pgvector and ChromaDB + Gradio Front-end

This project provides a pipeline for indexing markdown content into a PostgreSQL database with [pgvector](https://github.com/pgvector/pgvector) for efficient semantic search. It includes:

* A PostgreSQL schema for storing text chunks and their vector embeddings.
* A Python script for parsing markdown files, generating embeddings, and inserting them into the database.
* A deployment script for setting up and running the system.

---

## Features

* **Markdown Chunking**: Splits markdown documents into manageable chunks.
* **Vector Embeddings**: Generates vector representations of chunks for semantic similarity.
* **pgvector Integration**: Stores embeddings in PostgreSQL with HNSW indexing for fast nearest-neighbor search.
* **Metadata Storage**: Keeps track of additional context via JSONB fields.
* **Deployment Script**: Simplifies database setup and indexing process.

## Configuration

* **pgvector variant** (`pgv_chatty.py`, `pgv_indexer.py`): copy `pgv_config.py.template` to `pgv_config.py` and set env vars (`DB_NAME`, `DB_USER`, `DB_HOST`, `OLLAMA_HOST`, etc.) or export them directly.
* **ChromaDB variant** (`chromadb_chatty.py`, `chromadb_indexer.py`): copy `chroma_config.py.template` to `chroma_config.py` and set env vars similarly.

Both `*_config.py` files are gitignored — never commit real credentials.

```bash
export OLLAMA_HOST="http://localhost:11434"
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

## Logging in / Admin tab

The whole app (chat + admin) is gated behind Google sign-in — nothing is
reachable except `GET /health` until you log in with an allowlisted account:

1. Set `SESSION_SECRET` (e.g. `openssl rand -hex 32`) and `ALLOWED_EMAILS`
   (comma-separated Google account emails) in `.env`. Empty `ALLOWED_EMAILS`
   means nobody can log in — fails closed, not open.
2. Visit `https://<your-host>:7860/` — you'll be redirected to `/login`, then
   to Google's consent screen (identity + Drive readonly, one login covers
   both).
3. On success you land back on the chat UI with a session cookie (7-day
   default, `SESSION_MAX_AGE_SECONDS`). `/logout` clears it.

Once logged in, the **Admin** tab lets you:
* Upload a `.md`/`.txt`/`.pdf` file to index immediately (no CLI/SSH needed).
* Click "Sync Google Drive now" to pull new/changed Drive content on demand.
* See total indexed chunk count.

`gdrive_indexer.py` (the CLI/cron path) uses the same stored refresh token,
so logging in once as an allowlisted account also unlocks scheduled syncs.

## Production notes

* Retrieval uses real pgvector/Chroma nearest-neighbor search over query embeddings (not chunk length).
* `pgv_chatty.py` uses a pooled, health-checked DB connection (`psycopg2.pool.ThreadedConnectionPool`).
* Both chat apps validate message length, retry transient Ollama failures with backoff, and never leak internal exception details to end users (a correlation `error id` is logged server-side instead).
* Both apps expose `GET /health` (200/503) for k8s liveness/readiness probes, alongside the Gradio UI on the same port -- every other route requires a valid session.
* The "Stop Response" button only stops the requesting browser session's stream, not every concurrent user's.
* OAuth `state` is verified one-shot on `/oauth2callback` (no replay), and a login from a non-allowlisted Google account is rejected without ever touching the stored Drive refresh token.

## Tests

```bash
cd chatbot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest pgvector
pytest tests/ -v
```
