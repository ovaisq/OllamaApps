#!/usr/bin/env python3
"""Markdown Chatbot using Ollama, ChromaDB, and Gradio.

This script implements a chatbot that uses local LLMs (via Ollama) to answer
questions based on a knowledge base stored in ChromaDB. It supports
conversation history, context retrieval, and real-time streaming of responses.
"""
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import fastapi
import gradio as gr

from admin_ui import build_admin_tab
from app_session import get_email_from_request
from auth_routes import register_routes as register_auth_routes
from chroma_config import CHAT_CONFIG, CHROMA_CONFIG, OLLAMA_CONFIG
from gdrive_config import AUTH_CONFIG
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
    validate_message,
    WELCOME_MESSAGE,
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
        with self._history_connection() as conn:
            rows = conn.execute(
                "SELECT role, content FROM ("
                "  SELECT role, content, created_at, id FROM chat_history"
                "  WHERE user_email = ? ORDER BY created_at DESC, id DESC LIMIT ?"
                ") sub ORDER BY created_at ASC, id ASC",
                (user_email, limit),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def clear_history(self, user_email: str) -> None:
        with self._history_connection() as conn:
            conn.execute("DELETE FROM chat_history WHERE user_email = ?", (user_email,))

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
            with self._history_connection() as conn:
                conn.execute(
                    "INSERT INTO feedback (user_email, question, answer, rating) "
                    "VALUES (?, ?, ?, ?)",
                    (user_email, question, answer, rating),
                )
            logger.info(
                "Recorded %s feedback from %s on the answer to %r",
                rating, user_email, question[:80],
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
            context_chunks = self.retrieve_context(query)
            messages = build_chat_messages(context_chunks, conversation_context, query)
            sources_md = build_sources_footer(context_chunks)
        return messages, sources_md

    def _stream_answer(self, messages: List[Dict], stop_event: threading.Event):
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

    def get_answer_stream(self, query: str, history: List[Dict], stop_event: threading.Event):
        messages, _sources = self._prepare_messages(query, history)
        yield from self._stream_answer(messages, stop_event)

    def respond(self, message: str, chat_history: List[Dict], state: Dict, request: gr.Request = None):
        """Handle a new message. `state` is a per-session dict holding this
        session's own stop Event so one user's Stop button can't affect
        another user's in-flight stream. Persists both sides of the
        exchange for the logged-in user (from the session cookie), if the
        request can be identified.

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
        chat_history = chat_history or []
        base = chat_history + [{"role": "user", "content": message}]

        try:
            validate_message(message, CHAT_CONFIG["max_message_length"])
        except ValueError as e:
            yield (base + [{"role": "assistant", "content": error_bubble(str(e))}],
                   "", state)
            return

        yield (base + [{"role": "assistant", "content": TYPING_INDICATOR_HTML}], "", state)

        raw_answer = None      # model text as-is (what gets persisted)
        sources_md = None      # footer for completed content-search answers
        error_text = None      # set when preparation/streaming failed
        persist_text = None
        try:
            messages, sources_md = self._prepare_messages(message, chat_history)
            for partial in self._stream_answer(messages, stop_event):
                if stop_event.is_set():
                    break
                raw_answer = partial
                yield (base + [{"role": "assistant", "content": partial}], "", state)
        except Exception as e:
            error_text = safe_error_message(e, logger)
            yield (base + [{"role": "assistant", "content": error_bubble(error_text)}],
                   "", state)

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
                    "", state,
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

    def stop_chat(self, chat_history: List[Dict], state: Dict):
        """Stop current chat response for this session only."""
        if state is None:
            state = {}
        stop_event = state.setdefault("stop_event", threading.Event())
        stop_event.set()
        return (chat_history, "", state)

    def load_history_ui(self, request: gr.Request = None) -> List[Dict]:
        """Populate the Chat tab with this user's persisted history on page
        load. A user with no history yet gets the welcome message so the
        window doesn't open blank (it's only displayed, never persisted)."""
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
    """Build the Gradio UI + FastAPI app around an existing ChromaChat
    instance. Split out from main() so the UI construction (which uses a real
    Gradio/FastAPI API surface) is exercised by tests, not just chat's methods.
    """
    with gr.Blocks(title="Chatty") as blocks:
        with gr.Row(elem_id="chatty-topbar"):
            gr.Markdown("# Chatty\nDocument & Drive Assistant", elem_id="chatty-brand")
            index_chip = gr.Markdown("", elem_id="chatty-chip")
            gr.Markdown("[Sign out](/logout)", elem_id="signout-link")

        # All chrome (theme, labels, layout, chat wiring) lives in the
        # shared builders so the pgvector variant renders identically.
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
    chat = ChromaChat()
    app = build_app(chat)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    main()
