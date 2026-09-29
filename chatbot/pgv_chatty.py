#!/usr/bin/env python3
import logging
import os
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

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
    TYPING_INDICATOR_CSS,
    TYPING_INDICATOR_HTML,
    build_catalog_chat_messages,
    build_chat_messages,
    build_ollama_client,
    detect_catalog_intent,
    embed_text,
    ensure_model_loaded,
    safe_error_message,
    validate_message,
    with_retries,
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

    def get_context_chunks(self, query: str) -> List[Tuple[str, Optional[Dict]]]:
        """Retrieve the most relevant (chunk, metadata) pairs via pgvector
        nearest-neighbor search.
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
        with self._connection() as conn:
            with conn.cursor() as cursor:
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

    def get_answer_stream(self, query: str, history: List[Dict], stop_event: threading.Event):
        """Generate streaming response."""
        conversation_context = "\n".join(
            f"User: {u}\nAssistant: {a}" for u, a in self.get_last_conversation(history)
        )

        catalog_intent = detect_catalog_intent(query)
        if catalog_intent:
            docs = self.list_documents(**catalog_intent)
            messages = build_catalog_chat_messages(docs, conversation_context, query)
        else:
            context_chunks = self.get_context_chunks(query)
            messages = build_chat_messages(context_chunks, conversation_context, query)

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
            full_response += content
            yield full_response

    def respond(self, message: str, history: List[Dict], state: Dict, request: gr.Request = None):
        """Handle chat response. `state` is a per-session dict holding this
        session's own stop Event, so one user's Stop button can't affect
        another user's in-flight stream. Persists both sides of the
        exchange to chat_history for the logged-in user (from the session
        cookie), if the request can be identified.
        """
        if state is None:
            state = {}
        stop_event = state.setdefault("stop_event", threading.Event())
        stop_event.clear()

        user_email = get_email_from_request(request, AUTH_CONFIG["session_secret"])
        new_history = history + [{"role": "user", "content": message}]

        try:
            validate_message(message, CHAT_CONFIG["max_message_length"])
        except ValueError as e:
            yield (new_history + [{"role": "assistant", "content": str(e)}], "", state)
            return

        yield (new_history + [{"role": "assistant", "content": TYPING_INDICATOR_HTML}], "", state)

        final_response = None
        try:
            for partial_response in self.get_answer_stream(message, history, stop_event):
                if stop_event.is_set():
                    break
                final_response = partial_response
                yield (
                    new_history + [{"role": "assistant", "content": partial_response}],
                    "",
                    state,
                )
        except Exception as e:
            final_response = safe_error_message(e, logger)
            yield (new_history + [{"role": "assistant", "content": final_response}], "", state)

        if user_email and final_response:
            self.save_message(user_email, "user", message)
            self.save_message(user_email, "assistant", final_response)

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
        reload looks like history was lost even though it's saved server-side).
        """
        user_email = get_email_from_request(request, AUTH_CONFIG["session_secret"])
        if not user_email:
            return []
        return self.load_history(user_email)

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

    def index_text(self, text: str, source: str, extra_metadata: dict = None) -> int:
        from pgv_indexer import index_text

        with self._connection() as conn:
            return index_text(text, source, conn, extra_metadata=extra_metadata)

    def sync_drive(self) -> int:
        from gdrive_indexer import get_access_token, run_pgvector_backend

        return run_pgvector_backend(get_access_token())


def build_app(chat: "PGVectorChat") -> fastapi.FastAPI:
    """Build the Gradio UI + FastAPI app around an existing PGVectorChat
    instance. Split out from main() so the UI construction (which uses a real
    Gradio/FastAPI API surface) is exercised by tests, not just chat's methods.
    """
    with gr.Blocks(title="Chatty") as chatty:
        with gr.Row():
            gr.Markdown("# Chatty — Document & Drive Assistant")
            gr.Markdown("[Sign out](/logout)", elem_id="signout-link")

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Chat History")
            msg = gr.Textbox(label="Your Message")
            stop_btn = gr.Button("Stop Response")
            state = gr.State(value={})

            msg.submit(chat.respond, [msg, chatbot, state], [chatbot, msg, state], queue=True)
            stop_btn.click(chat.stop_chat, [chatbot, state], [chatbot, msg, state])

            clear_btn = gr.Button("Clear History")
            clear_btn.click(chat.clear_chat_ui, None, [chatbot, msg, state])

            chatty.load(chat.load_history_ui, None, chatbot)

        build_admin_tab(chat.index_text, chat.count_chunks, chat.sync_drive)

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
        chatty.queue(),
        path="/",
        footer_links=[],
        theme="JohnSmith9982/small_and_pretty",
        css=f"#signout-link {{text-align: right;}}\n{TYPING_INDICATOR_CSS}",
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
