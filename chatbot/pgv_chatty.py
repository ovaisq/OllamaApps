#!/usr/bin/env python3
"""Markdown Chatbot using Ollama, Postgres/pgvector, and a custom web UI.

Same role as chromadb_chatty.py, backed by pgvector instead of ChromaDB.
The UI is the shared custom web app (api_routes.py serving static/), not
Gradio: this module provides the backend (retrieval, prompting, persistence,
indexing) through the small protocol the web API drives (prepare_turn /
stream_answer / save_message / ...).
"""
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import fastapi
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from pgvector import Vector
from pgvector.psycopg2 import register_vector

from api_routes import build_app as build_web_app
from pgv_config import CHAT_CONFIG, DB_CONFIG, DB_POOL_CONFIG, OLLAMA_CONFIG
from pgv_schema import ensure_markdown_chunks_schema
from rag_common import (
    CATEGORY_MIME_TYPES,
    build_catalog_chat_messages,
    build_ollama_client,
    build_chat_messages,
    catalog_sources_payload,
    content_sources_payload,
    detect_catalog_intent,
    detect_mentioned_sources,
    DriveSyncGate,
    embed_text,
    ensure_model_loaded,
    read_drive_sync_timestamp,
    record_drive_sync_timestamp,
    StopEvents,
    with_retries,
)

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
        # Per-user stop Events (one per signed-in operator); the web API
        # keys streams by session email, so Stop can't cross sessions.
        self._stop_events = StopEvents()
        self.max_message_length = CHAT_CONFIG["max_message_length"]
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
        """This user's persisted messages, oldest first. created_at comes
        back as ISO-8601 (psycopg2 returns aware datetimes for timestamptz)
        so the web UI can render it in local time."""
        from datetime import timezone

        def _iso(value) -> Optional[str]:
            if not hasattr(value, "astimezone"):
                return None
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()

        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT role, content, created_at FROM ("
                    "  SELECT role, content, created_at FROM chat_history"
                    "  WHERE user_email = %s ORDER BY created_at DESC, id DESC LIMIT %s"
                    ") sub ORDER BY created_at ASC",
                    (user_email, limit),
                )
                return [
                    {"role": role, "content": content, "created_at": _iso(created)}
                    for role, content, created in cursor.fetchall()
                ]

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

    def record_feedback(self, email: str, question: str, answer: str, rating: str) -> None:
        """Persist a 👍/👎 rating on an answer together with the question it
        answered -- the reviewable record of what the bot got wrong (and the
        seed for teaching a correction in the Library view; the web UI sends
        the thread's question + answer text, so no index lookups here).
        Never raises: failing to record feedback must not break the chat.
        """
        try:
            with self._connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO feedback (user_email, question, answer, rating) "
                        "VALUES (%s, %s, %s, %s)",
                        (email, question, answer, rating),
                    )
                conn.commit()
            logger.info(
                "Recorded %s feedback from %s on the answer to %r",
                rating, email, (question or "")[:80],
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

    def prepare_turn(self, query: str, history: List[Dict]):
        """Build the RAG prompt AND the structured 'Sources' payload that
        the web UI renders under the answer (the documents actually fed to
        the model). Split from the streaming step so the turn's driver
        (api_routes.chat_stream_events) can attach it without retrieving the
        context a second time."""
        conversation_context = "\n".join(
            f"User: {u}\nAssistant: {a}" for u, a in self.get_last_conversation(history)
        )

        catalog_intent = detect_catalog_intent(query)
        if catalog_intent:
            docs = self.list_documents(**catalog_intent)
            messages = build_catalog_chat_messages(docs, conversation_context, query)
            sources = catalog_sources_payload(docs)
        else:
            context_chunks = self.get_context_chunks(query)
            messages = build_chat_messages(context_chunks, conversation_context, query)
            sources = content_sources_payload(context_chunks)
        return messages, sources

    def stream_answer(self, messages: List[Dict], stop_event: threading.Event):
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

    # The chat turn's orchestration (validate -> prepare -> stream ->
    # persist, plus error/stopped classification) is backend-agnostic and
    # lives in api_routes.chat_stream_events, driven through this class's
    # prepare_turn/stream_answer/save_message methods. Stop is signaled via
    # self._stop_events (per signed-in user), not a per-request state dict.


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
    """Build the FastAPI app (health + auth gate + web UI/API) around
    an existing PGVectorChat instance. The assembly lives in
    api_routes so both backends deliver byte-identical chrome; kept here
    so tests/main() keep importing build_app from the variant module."""
    return build_web_app(chat)


def main():
    """Main function for chat interface."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    chat = PGVectorChat()
    app = build_app(chat)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    main()
