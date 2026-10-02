#!/usr/bin/env python3
"""Markdown Chatbot using Ollama, ChromaDB, and a custom web UI.

This script implements a chatbot that uses local LLMs (via Ollama) to answer
questions based on a knowledge base stored in ChromaDB. It supports
conversation history, context retrieval, and real-time streaming of responses.

The UI is the shared custom web app (api_routes.py serving static/), not
Gradio: this module's job is the ChromaDB backend -- retrieval, prompting,
persistence, indexing -- exposed through the small protocol the web API
drives (prepare_turn / stream_answer / save_message / ...).
"""
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import fastapi

from api_routes import build_app as build_web_app
from chroma_config import CHAT_CONFIG, CHROMA_CONFIG, OLLAMA_CONFIG
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

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ChromaChat:
    def __init__(self):
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
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_CONFIG["db_path"])
        self._collection_lock = threading.Lock()
        self._collection = self.chroma_client.get_collection(name=CHROMA_CONFIG["collection"])
        self._stop_reload = threading.Event()
        self._reload_thread = threading.Thread(target=self._background_reloader, daemon=True)
        self._reload_thread.start()

        # No relational DB in this backend, so chat history lives in its own
        # small SQLite file (stdlib, no new dependency) alongside the Chroma
        # persistent store rather than inside the vector collection.
        self._history_db_path = os.path.join(
            os.path.dirname(os.path.abspath(CHROMA_CONFIG["db_path"])), "chat_history.sqlite3"
        )
        self._ensure_chat_history_schema()
        # Serializes Drive syncs: one click already spawns a worker pool;
        # a second concurrent click would re-embed the whole corpus again.
        self._sync_gate = DriveSyncGate()
        # Per-user stop Events (one per signed-in operator); the web API
        # keys streams by session email, so Stop can't cross sessions.
        self._stop_events = StopEvents()
        self.max_message_length = CHAT_CONFIG["max_message_length"]
        # Distinct-source cache for name-routing AND the Library tab's
        # index card (populated on demand): (source names, per-source chunk
        # counts) so two viewers share one metadata scan.
        self._sources_cache: Optional[Tuple[List[str], Dict[str, int]]] = None
        self._sources_cached_at: float = 0.0

    def _history_connection(self):
        return sqlite3.connect(self._history_db_path)

    def _ensure_chat_history_schema(self):
        with self._history_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chat_history_user_email_idx "
                "ON chat_history (user_email, created_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save_message(self, user_email: str, role: str, content: str) -> None:
        with self._history_connection() as conn:
            conn.execute(
                "INSERT INTO chat_history (user_email, role, content) VALUES (?, ?, ?)",
                (user_email, role, content),
            )

    def load_history(self, user_email: str, limit: int = 200) -> List[Dict]:
        """This user's persisted messages, oldest first. SQLite's
        CURRENT_TIMESTAMP is UTC without a suffix, so created_at comes back
        normalized to ISO-8601 UTC -- the web UI renders it in local time."""
        from datetime import datetime, timezone

        def _iso(value) -> Optional[str]:
            try:
                return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc).isoformat()
            except (TypeError, ValueError):
                return None

        with self._history_connection() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM ("
                "  SELECT role, content, created_at, id FROM chat_history"
                "  WHERE user_email = ? ORDER BY created_at DESC, id DESC LIMIT ?"
                ") sub ORDER BY created_at ASC, id ASC",
                (user_email, limit),
            ).fetchall()
        return [
            {"role": role, "content": content, "created_at": _iso(created)}
            for role, content, created in rows
        ]

    def clear_history(self, user_email: str) -> None:
        with self._history_connection() as conn:
            conn.execute("DELETE FROM chat_history WHERE user_email = ?", (user_email,))

    def record_feedback(self, email: str, question: str, answer: str, rating: str) -> None:
        """Persist a 👍/👎 rating on an answer together with the question it
        answered -- the reviewable record of what the bot got wrong (and the
        seed for teaching a correction in the Library view; the web UI sends
        the thread's question + answer text, so no index lookups here).
        Never raises: failing to record feedback must not break the chat.
        """
        try:
            with self._history_connection() as conn:
                conn.execute(
                    "INSERT INTO feedback (user_email, question, answer, rating) "
                    "VALUES (?, ?, ?, ?)",
                    (email, question, answer, rating),
                )
            logger.info(
                "Recorded %s feedback from %s on the answer to %r",
                rating, email, (question or "")[:80],
            )
        except Exception:
            logger.warning("Failed to record chat feedback", exc_info=True)

    @property
    def collection(self):
        with self._collection_lock:
            return self._collection

    def _background_reloader(self):
        while not self._stop_reload.wait(CHAT_CONFIG["reload_interval"]):
            try:
                new_collection = self.chroma_client.get_collection(
                    name=CHROMA_CONFIG["collection"]
                )
                with self._collection_lock:
                    self._collection = new_collection
                logger.info("Index reloaded in background")
            except Exception:
                logger.exception("Failed to reload index")

    def health_check(self) -> bool:
        try:
            self.collection.count()
            return True
        except Exception:
            logger.exception("Health check failed")
            return False

    def list_sources(self, cache_seconds: float = 300.0) -> List[str]:
        """Distinct indexed document names (cached briefly -- chroma has no
        DISTINCT, and a full metadata scan per query is wasteful)."""
        sources, _counts = self._load_source_metadata(cache_seconds)
        return sources

    def _load_source_metadata(self, cache_seconds: float = 300.0) -> Tuple[List[str], Dict[str, int]]:
        """(distinct sources, per-source chunk counts), cached briefly.
        Shared by name-routing (list_sources) and the Library tab's index
        card (index_summary) so one metadata scan serves both."""
        now = time.time()
        if (
            self._sources_cache is not None
            and now - self._sources_cached_at < cache_seconds
        ):
            return self._sources_cache
        metas = (self.collection.get(include=["metadatas"]) or {}).get("metadatas") or []
        counts: Dict[str, int] = {}
        for meta in metas:
            meta = meta or {}
            source = meta.get("source")
            if source:
                counts[source] = counts.get(source, 0) + 1
        sources = sorted(counts)
        self._sources_cache = (sources, counts)
        self._sources_cached_at = now
        return sources, counts

    def index_summary(self, force: bool = False) -> Dict[str, Any]:
        """Index stats for the Library tab / top-bar chip: total chunks,
        distinct documents, top sources by chunk count, and the last
        successful Drive sync time. Uses the same brief metadata cache as
        name-routing; force=True (the manual 'Refresh stats' button) busts
        it."""
        sources, counts = self._load_source_metadata(0.0 if force else 300.0)
        top_sources = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        return {
            "chunks": self.count_chunks(),
            "documents": len(sources),
            "top_sources": top_sources,
            "last_sync": read_drive_sync_timestamp(),
        }

    def load_feedback(self, user_email: str, limit: int = 20) -> List[Dict[str, Any]]:
        """This user's recent disliked answers, newest first -- the review
        queue behind the Library tab's 'teach a correction' prefill."""
        with self._history_connection() as conn:
            rows = conn.execute(
                "SELECT id, question, answer, created_at FROM feedback "
                "WHERE user_email = ? AND rating = 'dislike' "
                "ORDER BY id DESC LIMIT ?",
                (user_email, limit),
            ).fetchall()
        return [
            {"id": row_id, "question": question, "answer": answer,
             "created": created or ""}
            for row_id, question, answer, created in rows
        ]

    def retrieve_context(self, query: str) -> List[Tuple[str, Optional[Dict]]]:
        """Retrieve the most relevant (chunk, metadata) pairs from ChromaDB.

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
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=CHAT_CONFIG["top_k"]
        )
        documents = results["documents"][0]
        metadatas = results.get("metadatas") or [[]]
        metadatas = metadatas[0] if metadatas[0] else [{}] * len(documents)
        distances = (results.get("distances") or [[]])[0] or [None] * len(documents)

        mentioned_sources = detect_mentioned_sources(query, self.list_sources())
        if mentioned_sources:
            where = (
                {"source": {"$eq": mentioned_sources[0]}} if len(mentioned_sources) == 1
                else {"$or": [{"source": {"$eq": m}} for m in mentioned_sources]}
            )
            focused = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=CHAT_CONFIG["top_k"],
                where=where,
            )
            f_docs = focused["documents"][0]
            f_metas = (focused.get("metadatas") or [[]])[0] or [{}] * len(f_docs)
            f_dists = (focused.get("distances") or [[]])[0] or [None] * len(f_docs)
            seen = set(f_docs)
            keep = [
                i for i, doc in enumerate(documents) if doc not in seen
            ]
            documents = (f_docs + [documents[i] for i in keep])[:CHAT_CONFIG["top_k"]]
            metadatas = (f_metas + [metadatas[i] for i in keep])[:CHAT_CONFIG["top_k"]]
            distances = (f_dists + [distances[i] for i in keep])[:CHAT_CONFIG["top_k"]]

        if distances:
            logger.info("Closest retrieved chunk distance for query: %.4f", distances[0])

        max_distance = CHAT_CONFIG.get("max_context_distance")
        rows = list(zip(documents, metadatas, distances))
        if max_distance is not None:
            rows = [r for r in rows if r[2] is not None and r[2] <= max_distance]

        return [(text, meta) for text, meta, _ in rows]

    def list_documents(
        self, person: Optional[str] = None, category: Optional[str] = None,
        shared_with_me: bool = False,
    ) -> List[Dict]:
        """Answer "show me all documents shared by X" / "list all PDFs"
        style questions directly from stored metadata. Chroma has no
        substring/ILIKE `where` filter, so this fetches all metadatas (fine
        at personal/small-collection scale) and filters in Python.
        """
        all_metadatas = self.collection.get(include=["metadatas"]).get("metadatas") or []
        mime_types = CATEGORY_MIME_TYPES.get(category, set()) if category else None

        seen_sources = set()
        docs = []
        for meta in all_metadatas:
            meta = meta or {}
            source = meta.get("source")
            if not source or source in seen_sources:
                continue
            if person:
                owner, shared_by = (meta.get("owner") or ""), (meta.get("shared_by") or "")
                if person.lower() not in owner.lower() and person.lower() not in shared_by.lower():
                    continue
            if mime_types and meta.get("mime_type") not in mime_types:
                continue
            if shared_with_me and not meta.get("shared"):
                continue
            seen_sources.add(source)
            docs.append({
                "source": source, "owner": meta.get("owner"),
                "shared_by": meta.get("shared_by"), "mime_type": meta.get("mime_type"),
            })
        return docs

    def get_last_conversation(self, history: List[Dict], pairs: int = 3) -> List[tuple]:
        conv = []
        i = len(history) - 1
        while i > 0 and len(conv) < pairs:
            if history[i]["role"] == "assistant" and history[i - 1]["role"] == "user":
                conv.append((history[i - 1]["content"], history[i]["content"]))
                i -= 2
            else:
                i -= 1
        conv.reverse()
        return conv

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
            context_chunks = self.retrieve_context(query)
            messages = build_chat_messages(context_chunks, conversation_context, query)
            sources = content_sources_payload(context_chunks)
        return messages, sources

    def stream_answer(self, messages: List[Dict], stop_event: threading.Event):
        """Stream the model's answer, yielding the growing text."""
        stream = self.ollama_client.chat(
            model=OLLAMA_CONFIG["chat_model"],
            options={"num_ctx": OLLAMA_CONFIG["num_ctx"]},
            keep_alive=OLLAMA_CONFIG["keep_alive"],
            messages=messages,
            stream=True,
        )
        answer = ""
        for chunk in stream:
            if stop_event.is_set():
                break
            content = chunk.get("message", {}).get("content", "")
            if not content:
                # Ollama's first chunk is role-only (empty content).
                # Yielding it replaced the typing indicator with an empty
                # bubble that sat there for the whole prefill window --
                # keep the dots up until real text arrives instead.
                continue
            answer += content
            yield answer

    # The chat turn's orchestration (validate -> prepare -> stream ->
    # persist, plus error/stopped classification) is backend-agnostic and
    # lives in api_routes.chat_stream_events, driven through this class's
    # prepare_turn/stream_answer/save_message methods. Stop is signaled via
    # self._stop_events (per signed-in user), not a per-request state dict.


    def count_chunks(self) -> int:
        return self.collection.count()

    def index_text(self, text: str, source: str, extra_metadata: dict = None) -> int:
        from chromadb_indexer import index_text

        return index_text(text, source, self.collection, self.ollama_client, extra_metadata=extra_metadata)

    def sync_drive(self) -> int:
        """Sync Drive content into the index. Returns the number of new
        chunks indexed, or -1 when a sync is already in progress or one
        finished within the cooldown window (blocked to prevent a sync
        storm from re-embedding the whole corpus).
        """
        from gdrive_indexer import get_access_token, run_chromadb_backend

        if not self._sync_gate.try_begin():
            logger.warning("Ignoring Drive sync trigger: a sync is already running or just finished")
            return -1
        try:
            added = run_chromadb_backend(get_access_token())
        finally:
            self._sync_gate.finish()
        if added >= 0:
            # 'Drive synced 2 h ago' in the UI; failures never stamp it.
            record_drive_sync_timestamp()
        return added


def build_app(chat: "ChromaChat") -> fastapi.FastAPI:
    """Build the FastAPI app (health + auth gate + web UI/API) around an
    existing ChromaChat instance. The assembly lives in api_routes so both
    backends deliver byte-identical chrome; kept here so tests/main() keep
    importing build_app from the variant module."""
    return build_web_app(chat)


def main():
    chat = ChromaChat()
    app = build_app(chat)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    main()
