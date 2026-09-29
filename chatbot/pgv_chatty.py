#!/usr/bin/env python3
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import fastapi
import gradio as gr
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from pgvector import Vector
from pgvector.psycopg2 import register_vector

from admin_ui import build_admin_tab
from app_session import get_email_from_request
from auth_routes import register_routes as register_auth_routes
from gdrive_config import AUTH_CONFIG
from pgv_config import CHAT_CONFIG, DB_CONFIG, DB_POOL_CONFIG, OLLAMA_CONFIG
from pgv_schema import ensure_markdown_chunks_schema
from rag_common import (
    CATEGORY_MIME_TYPES,
    TYPING_INDICATOR_HTML,
    build_catalog_chat_messages,
    build_catalog_sources_footer,
    build_chat_messages,
    build_ollama_client,
    build_sources_footer,
    detect_catalog_intent,
    detect_mentioned_sources,
    DriveSyncGate,
    embed_text,
    ensure_model_loaded,
    error_bubble,
    read_drive_sync_timestamp,
    record_drive_sync_timestamp,
    safe_error_message,
    stopped_html,
    WELCOME_MESSAGE,
    validate_message,
    with_retries,
)
from ui_common import (
    CHATTY_CSS,
    CHATTY_THEME,
    INDEX_CHIP_INTERVAL_SECONDS,
    build_chat_tab,
    format_index_chip,
)

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

logger = logging.getLogger(__name__)


class PGVectorChat:
    def __init__(self):
        self._validate_config()
        self.pool = ThreadedConnectionPool(
            DB_POOL_CONFIG["minconn"], DB_POOL_CONFIG["maxconn"], **DB_CONFIG
        )
        # Register the pgvector adapter once, process-wide, so every pooled
        # connection can bind python lists to the `vector` column type.
        bootstrap_conn = self.pool.getconn()
        try:
            register_vector(bootstrap_conn, globally=True)
        finally:
            self.pool.putconn(bootstrap_conn)
        self.ollama_client = build_ollama_client(OLLAMA_CONFIG["host"], OLLAMA_CONFIG["timeout"])
        # Prewarm both models (keep_alive=-1 keeps them resident forever)
        # so no user request pays a cold load. /api/ps is checked first,
        # so an already-running model is never touched or reloaded.
        ensure_model_loaded(
            self.ollama_client,
            OLLAMA_CONFIG["chat_model"],
            keep_alive=OLLAMA_CONFIG["keep_alive"],
            num_ctx=OLLAMA_CONFIG["num_ctx"],
        )
        ensure_model_loaded(
            self.ollama_client,
            OLLAMA_CONFIG["embedding_model"],
            keep_alive=OLLAMA_CONFIG["keep_alive"],
        )
        self._ensure_chat_history_schema()
        self._ensure_feedback_schema()
        # Serializes Drive syncs: one click already spawns a worker pool;
        # a second concurrent click would re-embed the whole corpus again.
        self._sync_gate = DriveSyncGate()
        with self._connection() as conn:
            ensure_markdown_chunks_schema(
                conn, OLLAMA_CONFIG["embedding_dim"], OLLAMA_CONFIG["embedding_model"]
            )

    def _ensure_chat_history_schema(self):
        """Idempotent so this self-migrates on an already-initialized DB
        volume too -- pgvector.sql only runs on a brand-new empty volume,
        it won't retroactively add this table to an existing deployment.
        """
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id BIGSERIAL PRIMARY KEY,
                        user_email TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS chat_history_user_email_idx "
                    "ON chat_history (user_email, created_at)"
                )
            conn.commit()

    def save_message(self, user_email: str, role: str, content: str) -> None:
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO chat_history (user_email, role, content) VALUES (%s, %s, %s)",
                    (user_email, role, content),
                )
            conn.commit()

    def load_history(self, user_email: str, limit: int = 200) -> List[Dict]:
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT role, content FROM ("
                    "  SELECT role, content, created_at FROM chat_history"
                    "  WHERE user_email = %s ORDER BY created_at DESC, id DESC LIMIT %s"
                    ") sub ORDER BY created_at ASC",
                    (user_email, limit),
                )
                return [{"role": role, "content": content} for role, content in cursor.fetchall()]

    def clear_history(self, user_email: str) -> None:
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM chat_history WHERE user_email = %s", (user_email,))
            conn.commit()

    def _ensure_feedback_schema(self):
        """Like/Dislike feedback on answers (created idempotently at startup
        so an existing deployment self-migrates, same as chat_history)."""
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedback (
                        id BIGSERIAL PRIMARY KEY,
                        user_email TEXT NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        rating TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
            conn.commit()

    def record_feedback(self, history: List[Dict], like_data: "gr.LikeData",
                        request: "gr.Request" = None) -> None:
        """Persist a Like/Dislike on an assistant message together with the
        question it answered -- the reviewable record of what the bot got
        wrong (and the seed for teaching a correction in the Admin tab).
        Never raises: failing to record feedback must not break the chat.
        """
        try:
            idx = like_data.index
            if isinstance(idx, tuple):
                idx = idx[0]
            answer = history[idx].get("content", "") if isinstance(idx, int) and 0 <= idx < len(history) else ""
            question = ""
            for message in reversed(history[:idx]):
                if message.get("role") == "user":
                    question = message.get("content", "")
                    break
            rating = (
                "like" if like_data.liked is True
                else "dislike" if like_data.liked is False
                else str(like_data.liked)
            )
            user_email = get_email_from_request(request, AUTH_CONFIG["session_secret"]) or "anonymous"
            with self._connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO feedback (user_email, question, answer, rating) "
                        "VALUES (%s, %s, %s, %s)",
                        (user_email, question, answer, rating),
                    )
                conn.commit()
            logger.info(
                "Recorded %s feedback from %s on the answer to %r",
                rating, user_email, question[:80],
            )
        except Exception:
            logger.warning("Failed to record chat feedback", exc_info=True)

    @staticmethod
    def _validate_config():
        missing = [
            key for key in ("dbname", "user", "host")
            if not DB_CONFIG.get(key)
        ]
        if missing:
            raise RuntimeError(
                f"Missing required DB configuration: {', '.join(missing)}. "
                "Set DB_NAME/DB_USER/DB_HOST (see pgv_config.py.template)."
            )

    @contextmanager
    def _connection(self):
        """Borrow a pooled connection, verifying it is alive first."""
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
        except psycopg2.Error:
            logger.warning("Stale pooled connection detected, reconnecting")
            self.pool.putconn(conn, close=True)
            conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    def health_check(self) -> bool:
        try:
            with self._connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            return True
        except Exception:
            logger.exception("Health check failed")
            return False

    def list_sources(self) -> List[str]:
        """Distinct indexed document names, used for name-routing."""
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT metadata->>'source' FROM markdown_chunks")
                return [row[0] for row in cursor.fetchall() if row[0]]

    def get_context_chunks(self, query: str) -> List[Tuple[str, Optional[Dict]]]:
        """Retrieve the most relevant (chunk, metadata) pairs via pgvector
        nearest-neighbor search.

        A document the query explicitly names is routed ahead of pure
        distance order: with a corpus dominated by dense numeric chunks,
        embedding luck can keep a named document out of the top-k
        entirely (the "Kona April 2026" failure).
        """
        query_embedding = with_retries(
            lambda: embed_text(
                self.ollama_client,
                query,
                OLLAMA_CONFIG["embedding_model"],
                keep_alive=OLLAMA_CONFIG["keep_alive"],
            ),
            attempts=OLLAMA_CONFIG["retry_attempts"],
        )
        mentioned_sources = detect_mentioned_sources(query, self.list_sources())
        with self._connection() as conn:
            with conn.cursor() as cursor:
                if mentioned_sources:
                    cursor.execute(
                        "SELECT chunk, metadata, embedding <-> %s AS distance "
                        "FROM markdown_chunks "
                        "ORDER BY CASE WHEN metadata->>'source' = ANY(%s) "
                        "THEN 0 ELSE 1 END, distance LIMIT %s",
                        (Vector(query_embedding), mentioned_sources, CHAT_CONFIG["top_k"]),
                    )
                else:
                    cursor.execute(
                        "SELECT chunk, metadata, embedding <-> %s AS distance "
                        "FROM markdown_chunks ORDER BY distance LIMIT %s",
                        (Vector(query_embedding), CHAT_CONFIG["top_k"]),
                    )
                rows = cursor.fetchall()

        if rows:
            logger.info("Closest retrieved chunk distance for query: %.4f", rows[0][2])

        max_distance = CHAT_CONFIG.get("max_context_distance")
        if max_distance is not None:
            rows = [r for r in rows if r[2] <= max_distance]

        return [(r[0], r[1]) for r in rows]

    def list_documents(
        self, person: Optional[str] = None, category: Optional[str] = None,
        shared_with_me: bool = False,
    ) -> List[Dict]:
        """Answer "show me all documents shared by X" / "list all PDFs"
        style questions directly from stored metadata -- a filter/
        enumeration question, not something embedding similarity search
        can answer completely or reliably.
        """
        where_clauses = ["TRUE"]
        params: List = []

        if person:
            where_clauses.append(
                "(metadata->>'owner' ILIKE %s OR metadata->>'shared_by' ILIKE %s)"
            )
            pattern = f"%{person}%"
            params.extend([pattern, pattern])

        if category:
            mime_types = list(CATEGORY_MIME_TYPES.get(category, set()))
            if mime_types:
                where_clauses.append("metadata->>'mime_type' = ANY(%s)")
                params.append(mime_types)

        if shared_with_me:
            where_clauses.append("metadata->>'shared' = 'true'")

        query = (
            "SELECT DISTINCT metadata->>'source' AS source, metadata->>'owner' AS owner, "
            "metadata->>'shared_by' AS shared_by, metadata->>'mime_type' AS mime_type "
            f"FROM markdown_chunks WHERE {' AND '.join(where_clauses)}"
        )
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                cols = ("source", "owner", "shared_by", "mime_type")
                return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def get_last_conversation(self, history: List[Dict], count: int = 2) -> List[tuple]:
        """Get last conversation turns."""
        result = []
        i = len(history) - 1
        while i >= 0 and len(result) < count:
            if history[i]["role"] == "assistant" and i > 0 and history[i - 1]["role"] == "user":
                result.append((history[i - 1]["content"], history[i]["content"]))
                i -= 2
            else:
                i -= 1
        return result[::-1]

    def _prepare_messages(self, query: str, history: List[Dict]):
        """Build the RAG prompt AND the 'Sources' footer that gets shown
        under the answer (the documents actually fed to the model). Split
        from get_answer_stream so respond() can attach the footer without
        retrieving the context a second time."""
        conversation_context = "\n".join(
            f"User: {u}\nAssistant: {a}" for u, a in self.get_last_conversation(history)
        )

        catalog_intent = detect_catalog_intent(query)
        if catalog_intent:
            docs = self.list_documents(**catalog_intent)
            messages = build_catalog_chat_messages(docs, conversation_context, query)
            sources_md = build_catalog_sources_footer(docs)
        else:
            context_chunks = self.get_context_chunks(query)
            messages = build_chat_messages(context_chunks, conversation_context, query)
            sources_md = build_sources_footer(context_chunks)
        return messages, sources_md

    def _stream_answer(self, messages: List[Dict], stop_event: threading.Event):
        """Stream the model's answer, yielding the growing text."""
        response_stream = self.ollama_client.chat(
            model=OLLAMA_CONFIG["chat_model"],
            messages=messages,
            stream=True,
            options={"num_ctx": OLLAMA_CONFIG["num_ctx"]},
            keep_alive=OLLAMA_CONFIG["keep_alive"],
        )

        full_response = ""
        for chunk in response_stream:
            if stop_event.is_set():
                break
            content = chunk.get("message", {}).get("content", "")
            if not content:
                # Ollama's first chunk is role-only (empty content).
                # Yielding it replaced the typing indicator with an empty
                # bubble that sat there for the whole prefill window --
                # keep the dots up until real text arrives instead.
                continue
            full_response += content
            yield full_response

    def get_answer_stream(self, query: str, history: List[Dict], stop_event: threading.Event):
        """Generate streaming response."""
        messages, _sources = self._prepare_messages(query, history)
        yield from self._stream_answer(messages, stop_event)

    def respond(self, message: str, history: List[Dict], state: Dict, request: gr.Request = None):
        """Handle chat response. `state` is a per-session dict holding this
        session's own stop Event, so one user's Stop button can't affect
        another user's in-flight stream. Persists both sides of the
        exchange to chat_history for the logged-in user (from the session
        cookie), if the request can be identified.

        Display vs persistence: the user sees the raw model text plus
        decorations (sources footer, error/stopped styling); what's saved
        to history is the raw text, so reloads and feedback records stay
        clean.
        """
        if state is None:
            state = {}
        stop_event = state.setdefault("stop_event", threading.Event())
        stop_event.clear()

        user_email = get_email_from_request(request, AUTH_CONFIG["session_secret"])
        base = (history or []) + [{"role": "user", "content": message}]

        try:
            validate_message(message, CHAT_CONFIG["max_message_length"])
        except ValueError as e:
            yield (base + [{"role": "assistant", "content": error_bubble(str(e))}], "", state)
            return

        yield (base + [{"role": "assistant", "content": TYPING_INDICATOR_HTML}], "", state)

        raw_answer = None      # model text as-is (what gets persisted)
        sources_md = None      # footer for completed content-search answers
        error_text = None      # set when preparation/streaming failed
        persist_text = None
        try:
            messages, sources_md = self._prepare_messages(message, history)
            for partial_response in self._stream_answer(messages, stop_event):
                if stop_event.is_set():
                    break
                raw_answer = partial_response
                yield (base + [{"role": "assistant", "content": partial_response}], "", state)
        except Exception as e:
            error_text = safe_error_message(e, logger)
            yield (base + [{"role": "assistant", "content": error_bubble(error_text)}], "", state)

        if error_text is not None:
            # Keep the old behavior: the (safe, generic) error text is
            # history that should be reviewable, so it gets persisted.
            persist_text = error_text
        elif raw_answer is None:
            # The stream produced no text (stopped before the first token,
            # or the model returned nothing at all): don't leave the typing
            # dots frozen on screen as if it were still thinking.
            if stop_event.is_set():
                yield (base + [{"role": "assistant", "content": stopped_html()}], "", state)
            else:
                yield (
                    base + [{"role": "assistant",
                             "content": error_bubble("The model returned no response content.")}],
                    "",
                    state,
                )
        else:
            display = raw_answer
            if stop_event.is_set():
                display += stopped_html()
            elif sources_md:
                display += sources_md
            yield (base + [{"role": "assistant", "content": display}], "", state)
            persist_text = raw_answer

        if user_email and persist_text:
            self.save_message(user_email, "user", message)
            self.save_message(user_email, "assistant", persist_text)

    def stop_chat(self, history: List[Dict], state: Dict):
        """Stop current chat response for this session only."""
        if state is None:
            state = {}
        stop_event = state.setdefault("stop_event", threading.Event())
        stop_event.set()
        return (history, "", state)

    def load_history_ui(self, request: gr.Request = None) -> List[Dict]:
        """Populate the Chat tab with this user's persisted history on page
        load (a fresh gr.State is per-browser-session, so without this a
        reload looks like history was lost even though it's saved
        server-side). A user with no history yet gets the welcome message
        so the window doesn't open blank (only displayed, never persisted).
        """
        user_email = get_email_from_request(request, AUTH_CONFIG["session_secret"])
        if not user_email:
            return []
        return self.load_history(user_email) or [WELCOME_MESSAGE]

    def clear_chat_ui(self, request: gr.Request = None):
        """Clear both the visible chat and this user's persisted history."""
        user_email = get_email_from_request(request, AUTH_CONFIG["session_secret"])
        if user_email:
            self.clear_history(user_email)
        return ([], "", {})

    def count_chunks(self) -> int:
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
                return cursor.fetchone()[0]

    def index_summary(self, force: bool = False) -> Dict[str, Any]:
        """Index stats for the Library tab / top-bar chip: total chunks,
        distinct documents, top sources by chunk count, and the last
        successful Drive sync time. Postgres does the counting; the
        `force` flag is a no-op here (no client-side cache to bust)."""
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
                chunks = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT metadata->>'source' AS source, COUNT(*) AS n "
                    "FROM markdown_chunks "
                    "WHERE metadata->>'source' IS NOT NULL "
                    "GROUP BY source ORDER BY n DESC, source ASC"
                )
                top_sources = cursor.fetchall()
        return {
            "chunks": chunks,
            "documents": len(top_sources),
            "top_sources": [(row[0], row[1]) for row in top_sources[:5]],
            "last_sync": read_drive_sync_timestamp(),
        }

    def load_feedback(self, user_email: str, limit: int = 20) -> List[Dict[str, Any]]:
        """This user's recent disliked answers, newest first -- the review
        queue behind the Library tab's 'teach a correction' prefill."""
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, question, answer, to_char(created_at, 'YYYY-MM-DD HH24:MI') "
                    "FROM feedback WHERE user_email = %s AND rating = 'dislike' "
                    "ORDER BY id DESC LIMIT %s",
                    (user_email, limit),
                )
                rows = cursor.fetchall()
        return [
            {"id": row[0], "question": row[1], "answer": row[2], "created": row[3] or ""}
            for row in rows
        ]

    def index_text(self, text: str, source: str, extra_metadata: dict = None) -> int:
        from pgv_indexer import index_text

        with self._connection() as conn:
            return index_text(text, source, conn, extra_metadata=extra_metadata)

    def sync_drive(self) -> int:
        """Sync Drive content into the index. Returns the number of new
        chunks indexed, or -1 when a sync is already in progress or one
        finished within the cooldown window (blocked to prevent a sync
        storm from re-embedding the whole corpus).
        """
        from gdrive_indexer import get_access_token, run_pgvector_backend

        if not self._sync_gate.try_begin():
            logger.warning("Ignoring Drive sync trigger: a sync is already running or just finished")
            return -1
        try:
            added = run_pgvector_backend(get_access_token())
        finally:
            self._sync_gate.finish()
        if added >= 0:
            # 'Drive synced 2 h ago' in the UI; failures never stamp it.
            record_drive_sync_timestamp()
        return added


def build_app(chat: "PGVectorChat") -> fastapi.FastAPI:
    """Build the Gradio UI + FastAPI app around an existing PGVectorChat
    instance. Split out from main() so the UI construction (which uses a real
    Gradio/FastAPI API surface) is exercised by tests, not just chat's methods.
    """
    with gr.Blocks(title="Chatty") as blocks:
        with gr.Row(elem_id="chatty-topbar"):
            gr.Markdown("# Chatty\nDocument & Drive Assistant", elem_id="chatty-brand")
            index_chip = gr.Markdown("", elem_id="chatty-chip")
            gr.Markdown("[Sign out](/logout)", elem_id="signout-link")

        # All chrome (theme, labels, layout, chat wiring) lives in the
        # shared builders so the ChromaDB variant renders identically.
        build_chat_tab(chat, blocks)
        build_admin_tab(
            chat.index_text, chat.index_summary, chat.sync_drive,
            feedback_rows_fn=chat.load_feedback, blocks=blocks,
        )

        # Top-bar chip: first paint on load, then live via the timer.
        def _chip() -> str:
            return format_index_chip(chat.index_summary())

        blocks.load(_chip, None, index_chip, show_progress="hidden")
        gr.Timer(INDEX_CHIP_INTERVAL_SECONDS).tick(_chip, None, index_chip)

    app = fastapi.FastAPI()

    @app.get("/health")
    def health():
        ok = chat.health_check()
        status_code = 200 if ok else 503
        return fastapi.responses.JSONResponse(
            {"status": "ok" if ok else "unavailable"}, status_code=status_code
        )

    # Gates every other route (including the Gradio UI mounted below) behind
    # Google sign-in restricted to AUTH_CONFIG['allowed_emails'].
    register_auth_routes(app)

    gr.mount_gradio_app(
        app,
        blocks.queue(),
        path="/",
        footer_links=[],
        theme=CHATTY_THEME,
        css=CHATTY_CSS,
    )
    return app


def main():
    """Main function for chat interface."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    chat = PGVectorChat()
    app = build_app(chat)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    main()
