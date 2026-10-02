"""Tests for the web API (api_routes.py) and the shared chat-turn engine
(chat_stream_events) that replaced the per-variant Gradio respond()
generators.

Two layers:
- engine tests drive chat_stream_events directly (no HTTP) with a fake
  backend, asserting the event protocol and persistence rules;
- HTTP tests build a real FastAPI app (auth middleware + static mount +
  routes) and speak the actual wire format (JSON in, SSE out).
"""
import json
from unittest.mock import MagicMock

import fastapi
import pytest
from fastapi.testclient import TestClient

from api_routes import chat_stream_events, register_api_routes, sse_response
from app_session import SESSION_COOKIE, create_session_token
from auth_routes import register_routes as register_auth_routes
from rag_common import StopEvents

# Matches conftest's env (SESSION_SECRET / ALLOWED_EMAILS), which the real
# gdrive_config picks up at import time.
SECRET = "test-session-secret"
ALLOWED = "allowed@example.com"
COOKIE = {SESSION_COOKIE: create_session_token(ALLOWED, SECRET, 3600)}


# ------------------------------------------------------------------ fakes


class FakeChat:
    """Implements the backend protocol (api_routes module docstring) with
    in-memory state and scriptable stream/prepare behavior."""

    def __init__(self):
        self._stop_events = StopEvents()
        self.max_message_length = 4000
        self.saved = []
        self.feedback_rows = []
        self.indexed = []
        self.dislikes = []
        self.prepare_turn_calls = []
        self.stream_pieces = ["Hello", "Hello world"]
        self.stopped_before_first_token = False
        self.sources = {"kind": "content", "labels": ["doc.md"], "total": 1}
        self.summary = {
            "chunks": 10, "documents": 2,
            "top_sources": [("a.md", 6), ("b.pdf", 4)],
            "last_sync": None,
        }

    def prepare_turn(self, query, history):
        self.prepare_turn_calls.append(query)
        if query == "boom":
            raise RuntimeError("db-password=hunter2 must not leak")
        return [{"role": "system", "content": "s"}], self.sources

    def stream_answer(self, messages, stop_event):
        if self.stopped_before_first_token:
            stop_event.set()
            return
        for piece in self.stream_pieces:
            if stop_event.is_set():
                return
            yield piece

    def load_history(self, email, limit=200):
        return list(self.saved)

    def save_message(self, email, role, content):
        from datetime import datetime, timezone
        self.saved.append(
            {"role": role, "content": content,
             "created_at": datetime.now(timezone.utc).isoformat()}
        )

    def clear_history(self, email):
        self.saved.clear()

    def record_feedback(self, email, question, answer, rating):
        self.feedback_rows.append(
            {"id": len(self.feedback_rows) + 1, "question": question,
             "answer": answer, "rating": rating, "user": email}
        )

    def load_feedback(self, email, limit=20):
        return [
            {"id": r["id"], "question": r["question"], "answer": r["answer"], "created": ""}
            for r in self.feedback_rows if r["rating"] == "dislike"
        ]

    def index_summary(self, force=False):
        return self.summary

    def index_text(self, text, source, extra_metadata=None):
        self.indexed.append((text, source, extra_metadata))
        return 1

    def sync_drive(self):
        return 5

    def health_check(self):
        return True


def make_app(chat) -> TestClient:
    """Real assembly (auth middleware + static + API routes), not a mock of
    it -- the same shape api_routes.build_app produces for the backends."""
    app = fastapi.FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    register_auth_routes(app)
    register_api_routes(app, chat)
    return TestClient(app)


def parse_sse(body: str):
    """SSE body -> [(event, parsed_data), ...] in frame order."""
    events = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        event, data = "message", ""
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if data:
            events.append((event, json.loads(data)))
    return events


# ------------------------------------------------- engine (no HTTP) tests


def test_engine_streams_tokens_and_done_with_sources():
    chat = FakeChat()
    events = list(chat_stream_events(chat, "hi", [], "u@example.com", StopEvents().event_for(None), 4000))

    kinds = [e for e, _ in events]
    assert kinds == ["status", "token", "token", "done"]
    assert events[2][1]["text"] == "Hello world"
    assert events[3][1] == {"sources": chat.sources, "stopped": False}
    # Raw text persisted (no decorations), both roles.
    assert [(m["role"], m["content"]) for m in chat.saved] == [
        ("user", "hi"), ("assistant", "Hello world"),
    ]


def test_engine_validation_error_short_circuits_before_prepare():
    chat = FakeChat()
    events = list(chat_stream_events(chat, "x" * 5000, [], "u@example.com", StopEvents().event_for(None), 4000))

    assert events == [("error", {"message": "Message exceeds the 4000 character limit."})]
    assert chat.prepare_turn_calls == []
    assert chat.saved == []


def test_engine_error_is_generic_and_persisted():
    chat = FakeChat()
    events = list(chat_stream_events(chat, "boom", [], "u@example.com", StopEvents().event_for(None), 4000))

    assert events[-1][0] == "error"
    assert "hunter2" not in json.dumps(events)
    assert "error id" in events[-1][1]["message"]
    # The safe text is reviewable history, so it persists (old behavior).
    assert chat.saved[-1]["role"] == "assistant"
    assert chat.saved[-1]["content"] == events[-1][1]["message"]


def test_engine_empty_model_response_is_an_error_not_done():
    chat = FakeChat()
    chat.stream_pieces = []
    events = list(chat_stream_events(chat, "hi", [], None, StopEvents().event_for(None), 4000))

    assert events[-1][0] == "error"
    assert "no response content" in events[-1][1]["message"]
    assert chat.saved == []  # nothing to review


def test_engine_stopped_before_first_token_is_done_not_error():
    chat = FakeChat()
    chat.stopped_before_first_token = True
    events = list(chat_stream_events(chat, "hi", [], None, StopEvents().event_for(None), 4000))

    assert events[-1] == ("done", {"sources": None, "stopped": True})
    assert chat.saved == []


def test_engine_stale_stop_event_is_cleared_on_begin():
    """Stop left over from a previous answer must not kill the new stream
    (the registry reuses one Event per user)."""
    chat = FakeChat()
    stop_event = StopEvents().event_for("u@example.com")
    stop_event.set()  # stale stop from an earlier session

    events = list(chat_stream_events(chat, "hi", [], "u@example.com", stop_event, 4000))

    assert events[-1][0] == "done"
    assert events[-1][1]["stopped"] is False
    assert events[-1][1]["sources"] == chat.sources


def test_engine_anonymous_user_never_persisted():
    chat = FakeChat()
    list(chat_stream_events(chat, "hi", [], None, StopEvents().event_for(None), 4000))
    assert chat.saved == []


# ----------------------------------------------------------- HTTP layer


def test_index_page_gated_but_static_assets_are_public():
    client = make_app(FakeChat())

    # The page requires the session (redirects browsers to the login page)...
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/login"

    # ...but its assets don't (no secrets in there, and the login flow
    # should not 404 while the page is loading).
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200

    resp = client.get("/", cookies=COOKIE)
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Chatty" in resp.text


def test_api_calls_get_401_json_not_a_redirect():
    client = make_app(FakeChat())
    resp = client.get("/api/summary", follow_redirects=False)
    assert resp.status_code == 401
    assert resp.json() == {"detail": "unauthorized"}
    # A browser page still gets the branded redirect (see test above).
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)


def test_chat_streams_sse_and_persists_history():
    chat = FakeChat()
    client = make_app(chat)

    resp = client.post("/api/chat", json={"message": "hi"}, cookies=COOKIE)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(resp.text)
    assert [e for e, _ in events] == ["status", "token", "token", "done"]
    assert events[3][1] == {"sources": chat.sources, "stopped": False}
    assert chat.prepare_turn_calls == ["hi"]
    assert [(m["role"], m["content"]) for m in chat.saved] == [
        ("user", "hi"), ("assistant", "Hello world"),
    ]

    hist = client.get("/api/history", cookies=COOKIE).json()
    assert [m["content"] for m in hist["messages"]] == ["hi", "Hello world"]
    assert all(m["created_at"] for m in hist["messages"])


def test_chat_rejects_oversized_message_without_calling_backend():
    chat = FakeChat()
    client = make_app(chat)

    resp = client.post("/api/chat", json={"message": "x" * 5000}, cookies=COOKIE)

    events = parse_sse(resp.text)
    assert events == [("error", {"message": "Message exceeds the 4000 character limit."})]
    assert chat.prepare_turn_calls == []


def test_chat_error_response_hides_internals():
    chat = FakeChat()
    client = make_app(chat)

    resp = client.post("/api/chat", json={"message": "boom"}, cookies=COOKIE)

    assert "hunter2" not in resp.text
    events = parse_sse(resp.text)
    assert events[-1][0] == "error"
    assert "error id" in events[-1][1]["message"]


def test_chat_stop_endpoint_sets_the_users_event():
    chat = FakeChat()
    client = make_app(chat)

    resp = client.post("/api/chat/stop", cookies=COOKIE)

    assert resp.json() == {"status": "stopping"}
    assert chat._stop_events.event_for(ALLOWED).is_set()
    # Another user's event is untouched.
    assert not chat._stop_events.event_for("other@example.com").is_set()


def test_history_clear_removes_persisted_messages():
    chat = FakeChat()
    chat.saved.append({"role": "user", "content": "old", "created_at": "t"})
    client = make_app(chat)

    resp = client.delete("/api/history", cookies=COOKIE)

    assert resp.json() == {"status": "cleared"}
    assert client.get("/api/history", cookies=COOKIE).json() == {"messages": []}


def test_feedback_records_and_normalizes_rating():
    chat = FakeChat()
    client = make_app(chat)

    client.post("/api/feedback", json={
        "question": "q", "answer": "a", "rating": "like"}, cookies=COOKIE)
    client.post("/api/feedback", json={
        "question": "q2", "answer": "a2", "rating": "banana"}, cookies=COOKIE)

    assert [r["rating"] for r in chat.feedback_rows] == ["like", "dislike"]
    assert chat.feedback_rows[0]["user"] == ALLOWED


def test_summary_exposes_suggestions_from_real_sources():
    client = make_app(FakeChat())

    data = client.get("/api/summary", cookies=COOKIE).json()

    assert data["chunks"] == 10
    assert data["documents"] == 2
    assert data["top_sources"][0] == {"source": "a.md", "chunks": 6}
    # Real document names lead the welcome chips, generic questions fill in.
    assert data["suggestions"][0] == "What’s in \"a.md\"?"
    assert "List all PDFs" in data["suggestions"]


def test_library_upload_streams_progress_and_indexes():
    chat = FakeChat()
    client = make_app(chat)

    resp = client.post(
        "/api/library/upload",
        files=[("files", ("notes.md", b"hello world, indexed content", "text/markdown"))],
        cookies=COOKIE,
    )

    events = parse_sse(resp.text)
    assert [e for e, _ in events][:-1] == ["progress"] * (len(events) - 1)
    assert events[-1][0] == "done"
    assert "Indexed 1 file(s), 1 new chunk(s)." in events[-1][1]["message"]
    assert chat.indexed[0][1] == "notes.md"
    assert chat.indexed[0][2] == {"mime_type": "text/markdown"}
    assert "hello world" in chat.indexed[0][0]


def test_library_upload_folder_keeps_relative_paths_as_sources():
    """Folder uploads arrive with relative-path file names (the client sets
    them from webkitRelativePath); those become the stored source so
    citations keep their structure and same-named files don't collide."""
    chat = FakeChat()
    client = make_app(chat)

    resp = client.post(
        "/api/library/upload",
        files=[("files", ("reports/2025/notes.md", b"folder file content",
                          "text/markdown"))],
        cookies=COOKIE,
    )

    assert resp.status_code == 200
    assert chat.indexed[0][1] == "reports/2025/notes.md"


def test_library_upload_rejects_empty_upload():
    chat = FakeChat()
    client = make_app(chat)

    resp = client.post("/api/library/upload", cookies=COOKIE)

    # FastAPI requires the `files` field; a no-file request is a 422.
    assert resp.status_code == 422


def test_library_sync_streams_started_then_done():
    chat = FakeChat()
    client = make_app(chat)

    resp = client.post("/api/library/sync", cookies=COOKIE)

    events = parse_sse(resp.text)
    assert events[0] == ("progress", {"message": "Starting Google Drive sync... this can take a few minutes."})
    assert events[-1] == ("done", {"message": "Synced Google Drive: 5 new chunk(s) indexed."})


def test_library_teach_requires_both_fields():
    chat = FakeChat()
    client = make_app(chat)

    resp = client.post("/api/library/teach", json={"question": "q", "answer": ""},
                       cookies=COOKIE)

    events = parse_sse(resp.text)
    assert events[-1][1]["message"] == "Fill in both the question it got wrong and the correct answer."
    assert chat.indexed == []


def test_library_teach_indexes_with_correction_metadata():
    chat = FakeChat()
    client = make_app(chat)

    resp = client.post("/api/library/teach",
                       json={"question": "what host?", "answer": "deploy-5"}, cookies=COOKIE)

    events = parse_sse(resp.text)
    assert "Correction saved" in events[-1][1]["message"]
    text, source, meta = chat.indexed[0]
    assert source == "user-corrections"
    assert meta == {"type": "correction", "user": ALLOWED}


def test_library_feedback_rows_comes_from_backend():
    chat = FakeChat()
    chat.feedback_rows = [
        {"id": 3, "question": "q?", "answer": "a", "rating": "dislike", "user": ALLOWED},
    ]
    client = make_app(chat)

    rows = client.get("/api/library/feedback", cookies=COOKIE).json()

    assert rows[0]["question"] == "q?"


def test_sse_worker_failure_becomes_an_error_event():
    """A crash inside a sync worker must not hang/kill the response: it
    surfaces as an error SSE event (logged server-side with the traceback)."""
    from fastapi import Request

    app = fastapi.FastAPI()

    @app.post("/crash")
    async def crash(request: Request):
        def produce():
            yield ("progress", {"message": "ok so far"})
            raise RuntimeError("worker boom")

        return sse_response(produce, request)

    client = TestClient(app)
    resp = client.post("/crash")

    events = parse_sse(resp.text)
    assert events[0] == ("progress", {"message": "ok so far"})
    assert events[-1][0] == "error"


# ---------------------------------------------- real-assembly smoke tests


@pytest.mark.parametrize("variant", ["chromadb", "pgv"])
def test_variant_build_app_wires_health_auth_gate_and_web_ui(variant):
    """The old Gradio mount-smoke test, rebased: both backends assemble the
    SAME app shape (health, auth gate, static UI) from api_routes.build_app."""
    if variant == "chromadb":
        import chromadb_chatty
        app = chromadb_chatty.build_app(MagicMock(health_check=lambda: True))
    else:
        import pgv_chatty
        app = pgv_chatty.build_app(MagicMock(health_check=lambda: True))

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    # UI + API are session-gated; static assets are public.
    assert client.get("/", follow_redirects=False).status_code in (302, 307)
    assert client.get("/api/summary", follow_redirects=False).status_code == 401
    assert client.get("/static/styles.css").status_code == 200
