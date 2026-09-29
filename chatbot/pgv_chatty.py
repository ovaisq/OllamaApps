#!/usr/bin/env python3
import logging
import os
import threading
from contextlib import contextmanager
from typing import Dict, List

import fastapi
import gradio as gr
import ollama
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from pgvector.psycopg2 import register_vector

from pgv_config import CHAT_CONFIG, DB_CONFIG, DB_POOL_CONFIG, OLLAMA_CONFIG
from rag_common import embed_text, safe_error_message, validate_message, with_retries

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
        self.ollama_client = ollama.Client(
            host=OLLAMA_CONFIG["host"], timeout=OLLAMA_CONFIG["timeout"]
        )

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

    def get_context_chunks(self, query: str) -> List[str]:
        """Retrieve the most relevant chunks via pgvector nearest-neighbor search."""
        query_embedding = with_retries(
            lambda: embed_text(self.ollama_client, query, OLLAMA_CONFIG["embedding_model"]),
            attempts=OLLAMA_CONFIG["retry_attempts"],
        )
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT chunk FROM markdown_chunks "
                    "ORDER BY embedding <-> %s LIMIT %s",
                    (query_embedding, CHAT_CONFIG["top_k"]),
                )
                return [row[0] for row in cursor.fetchall()]

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
        context_chunks = self.get_context_chunks(query)
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

        response_stream = self.ollama_client.chat(
            model=OLLAMA_CONFIG["chat_model"],
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"num_ctx": CHAT_CONFIG["max_context_length"]},
        )

        full_response = ""
        for chunk in response_stream:
            if stop_event.is_set():
                break
            content = chunk.get("message", {}).get("content", "")
            full_response += content
            yield full_response

    def respond(self, message: str, history: List[Dict], state: Dict):
        """Handle chat response. `state` is a per-session dict holding this
        session's own stop Event, so one user's Stop button can't affect
        another user's in-flight stream.
        """
        if state is None:
            state = {}
        stop_event = state.setdefault("stop_event", threading.Event())
        stop_event.clear()

        new_history = history + [{"role": "user", "content": message}]

        try:
            validate_message(message, CHAT_CONFIG["max_message_length"])
        except ValueError as e:
            yield (new_history + [{"role": "assistant", "content": str(e)}], "", state)
            return

        try:
            for partial_response in self.get_answer_stream(message, history, stop_event):
                if stop_event.is_set():
                    break
                yield (
                    new_history + [{"role": "assistant", "content": partial_response}],
                    "",
                    state,
                )
        except Exception as e:
            error_msg = safe_error_message(e, logger)
            yield (new_history + [{"role": "assistant", "content": error_msg}], "", state)

    def stop_chat(self, history: List[Dict], state: Dict):
        """Stop current chat response for this session only."""
        if state is None:
            state = {}
        stop_event = state.setdefault("stop_event", threading.Event())
        stop_event.set()
        return (history, "", state)


def main():
    """Main function for chat interface."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    chat = PGVectorChat()

    with gr.Blocks(
        title="Markdown Document Chatbot",
        css="footer {display: none !important;}",
        theme="JohnSmith9982/small_and_pretty",
    ) as chatty:
        gr.Markdown("# Markdown Document Chatbot")

        chatbot = gr.Chatbot(label="Chat History")
        msg = gr.Textbox(label="Your Message")
        stop_btn = gr.Button("Stop Response")
        state = gr.State(value={})

        msg.submit(chat.respond, [msg, chatbot, state], [chatbot, msg, state], queue=True)
        stop_btn.click(chat.stop_chat, [chatbot, state], [chatbot, msg, state])

        clear_btn = gr.Button("Clear History")
        clear_btn.click(lambda: ([], "", {}), None, [chatbot, msg, state])

    app = fastapi.FastAPI()

    @app.get("/health")
    def health():
        ok = chat.health_check()
        status_code = 200 if ok else 503
        return fastapi.responses.JSONResponse(
            {"status": "ok" if ok else "unavailable"}, status_code=status_code
        )

    try:
        from gdrive_oauth_routes import register_routes as register_gdrive_routes
        register_gdrive_routes(app)
    except ImportError:
        logger.info("gdrive_config.py not present; Google Drive OAuth routes disabled")

    gr.mount_gradio_app(app, chatty.queue(), path="/")

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    main()
