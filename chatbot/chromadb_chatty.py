#!/usr/bin/env python3
"""Markdown Chatbot using Ollama, ChromaDB, and Gradio.

This script implements a chatbot that uses local LLMs (via Ollama) to answer
questions based on a knowledge base stored in ChromaDB. It supports
conversation history, context retrieval, and real-time streaming of responses.
"""
import logging
import os
import threading
from typing import Dict, List

import chromadb
import fastapi
import gradio as gr

from admin_ui import build_admin_tab
from auth_routes import register_routes as register_auth_routes
from chroma_config import CHAT_CONFIG, CHROMA_CONFIG, OLLAMA_CONFIG
from rag_common import (
    build_ollama_client,
    embed_text,
    safe_error_message,
    validate_message,
    with_retries,
)

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ChromaChat:
    def __init__(self):
        self.ollama_client = build_ollama_client(OLLAMA_CONFIG["host"], OLLAMA_CONFIG["timeout"])
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_CONFIG["db_path"])
        self._collection_lock = threading.Lock()
        self._collection = self.chroma_client.get_collection(name=CHROMA_CONFIG["collection"])
        self._stop_reload = threading.Event()
        self._reload_thread = threading.Thread(target=self._background_reloader, daemon=True)
        self._reload_thread.start()

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

    def retrieve_context(self, query: str) -> List[str]:
        """Retrieve the most relevant documents from ChromaDB for a query."""
        query_embedding = with_retries(
            lambda: embed_text(self.ollama_client, query, OLLAMA_CONFIG["embedding_model"]),
            attempts=OLLAMA_CONFIG["retry_attempts"],
        )
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=CHAT_CONFIG["top_k"]
        )
        return results["documents"][0]

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

    def get_answer_stream(self, query: str, history: List[Dict], stop_event: threading.Event):
        context_chunks = self.retrieve_context(query)
        context_text = "\n".join(context_chunks)
        conversation_context = "\n".join(
            f"User: {u}\nAssistant: {a}" for u, a in self.get_last_conversation(history)
        )

        prompt = f"""
You are a helpful assistant. Use the following context to answer the question accurately.
Context:
{context_text}

Previous conversation:
{conversation_context}

Question: {query}
Answer:
"""

        stream = self.ollama_client.chat(
            model=OLLAMA_CONFIG["chat_model"],
            options={"num_ctx": CHAT_CONFIG["max_context_length"]},
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        answer = ""
        for chunk in stream:
            if stop_event.is_set():
                break
            answer += chunk.get("message", {}).get("content", "")
            yield answer

    def respond(self, message: str, chat_history: List[Dict], state: Dict):
        """Handle a new message. `state` is a per-session dict holding this
        session's own stop Event so one user's Stop button can't affect
        another user's in-flight stream.
        """
        if state is None:
            state = {}
        stop_event = state.setdefault("stop_event", threading.Event())
        stop_event.clear()

        chat_history = chat_history or []

        try:
            validate_message(message, CHAT_CONFIG["max_message_length"])
        except ValueError as e:
            yield (
                chat_history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": str(e)},
                ],
                "",
                state,
            )
            return

        try:
            partial = ""
            for partial in self.get_answer_stream(message, chat_history, stop_event):
                if stop_event.is_set():
                    break
                yield (
                    chat_history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": partial},
                    ],
                    "",
                    state,
                )
        except Exception as e:
            error_msg = safe_error_message(e, logger)
            yield (
                chat_history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": error_msg},
                ],
                "",
                state,
            )

    def stop_chat(self, chat_history: List[Dict], state: Dict):
        """Stop current chat response for this session only."""
        if state is None:
            state = {}
        stop_event = state.setdefault("stop_event", threading.Event())
        stop_event.set()
        return (chat_history, "", state)

    def count_chunks(self) -> int:
        return self.collection.count()

    def index_text(self, text: str, source: str) -> int:
        from chromadb_indexer import index_text

        return index_text(text, source, self.collection, self.ollama_client)

    def sync_drive(self) -> int:
        from gdrive_indexer import get_access_token, run_chromadb_backend

        return run_chromadb_backend(get_access_token())


def build_app(chat: "ChromaChat") -> fastapi.FastAPI:
    """Build the Gradio UI + FastAPI app around an existing ChromaChat
    instance. Split out from main() so the UI construction (which uses a real
    Gradio/FastAPI API surface) is exercised by tests, not just chat's methods.
    """
    with gr.Blocks(
        title="Markdown Chatbot",
        css="footer {display: none !important;} #signout-link {text-align: right;}",
    ) as blocks:
        with gr.Row():
            gr.Markdown("# ChromaDB: Markdown Chatbot")
            gr.Markdown("[Sign out](/logout)", elem_id="signout-link")

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Ask about the README")
            stop_btn = gr.Button("Stop Chat")
            state = gr.State(value={})

            msg.submit(chat.respond, [msg, chatbot, state], [chatbot, msg, state], queue=True)
            stop_btn.click(chat.stop_chat, [chatbot, state], [chatbot, msg, state])

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

    gr.mount_gradio_app(app, blocks.queue(), path="/")
    return app


def main():
    chat = ChromaChat()
    app = build_app(chat)

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    main()
