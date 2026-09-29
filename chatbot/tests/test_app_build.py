"""Smoke tests that actually build the Gradio UI + FastAPI app for both chat
backends (real gr.Blocks/gr.Chatbot/gr.mount_gradio_app calls, not mocked) so
a Gradio API break (like the type= kwarg removal) or a bad UI wiring change
surfaces here instead of only at deploy time.
"""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def _fake_chat():
    chat = MagicMock()
    chat.health_check.return_value = True
    chat.count_chunks.return_value = 0
    return chat


def test_pgv_chatty_build_app_mounts_health_and_gates_ui():
    import pgv_chatty

    app = pgv_chatty.build_app(_fake_chat())
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/", follow_redirects=False).status_code in (302, 307)


def test_chromadb_chatty_build_app_mounts_health_and_gates_ui():
    import chromadb_chatty

    app = chromadb_chatty.build_app(_fake_chat())
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/", follow_redirects=False).status_code in (302, 307)
