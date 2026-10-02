"""The Chatty web API: a small JSON + SSE surface serving the custom web UI
in static/. Replaces the old Gradio app (gr.mount_gradio_app) that both
backends mounted at / -- the new UI is hand-built (no build step, no
Gradio), so this module is the only place that knows how a chat turn is
delivered.

Routes (all behind the Google-OAuth session middleware from auth_routes):

    GET    /                        the single-page UI (static/index.html)
    POST   /api/chat                SSE: status / token / done|error events
    POST   /api/chat/stop           set this user's stop Event
    GET    /api/history             persisted messages (oldest first)
    DELETE /api/history             clear this user's persisted history
    POST   /api/feedback            record a like/dislike on an answer
    GET    /api/summary             index stats + welcome suggestion chips

    POST   /api/library/upload      SSE: index multipart file(s)/folder
    POST   /api/library/sync        SSE: sync Google Drive
    POST   /api/library/teach       SSE: index a user correction
    GET    /api/library/summary     index stats (Library card)
    GET    /api/library/feedback    this user's recent dislikes (review)

The backend is duck-typed (`chat`): both ChromaChat and PGVectorChat expose
the same protocol, so this module stays backend-agnostic:

    chat.prepare_turn(query, history)      -> (messages, sources_payload|None)
    chat.stream_answer(messages, stop_event) -> iterator of cumulative text
    chat.load_history(email) / chat.save_message(email, role, content)
    chat.clear_history(email) / chat.load_feedback(email)
    chat.record_feedback(email, question, answer, rating)
    chat.index_summary() / chat.index_text(text, source, extra_metadata=None)
    chat.sync_drive() / chat.health_check() / chat._stop_events (rag_common.StopEvents)
    chat.max_message_length -> int

The SSE transport runs each blocking backend generator in a worker thread
and bridges its items onto the event loop with call_soon_threadsafe: the
model stream (Ollama over HTTP) is synchronous, and an HTTP request must not
block the loop. Client disconnects are polled between items so a closed tab
stops the stream, and on_finish fires (setting the stop Event for chat)
whenever the response ends for any reason.
"""
import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import fastapi
from fastapi import File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from app_session import get_email_from_request
from auth_routes import register_routes as register_auth_routes
from gdrive_config import AUTH_CONFIG
from library import sync_drive_now, teach_correction, upload_and_index
from rag_common import build_suggestions, safe_error_message, validate_message

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Sent with every SSE response: no intermediate cache/proxy buffering
# (X-Accel-Buffering is nginx's), or a token stream would arrive in clumps.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

_STREAM_END = object()  # sentinel: worker thread finished, stream may close


def encode_sse(event: str, data: Any) -> str:
    """One SSE frame. data is JSON-serialized, so it is always single-line
    (control characters escaped) and needs no multi-line `data:` splitting."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_response(
    produce: Callable[[], Iterator[Tuple[str, Any]]],
    request: Request,
    on_finish: Optional[Callable[[], None]] = None,
) -> StreamingResponse:
    """Serve a *synchronous* generator of (event, payload) tuples as SSE.

    Items are relayed through an asyncio.Queue from a worker thread (the
    backend does blocking HTTP/DB work). Between items the server polls
    request.is_disconnected(), so a dropped client ends the response, which
    in turn triggers on_finish -- for chat that sets the per-user stop
    Event, releasing the Ollama stream without waiting for the next token.
    """
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue" = asyncio.Queue()

    def _push(item: Any) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            pass  # the event loop is shutting down; nothing to serve

    def worker() -> None:
        try:
            for item in produce():
                _push(item)
        except Exception:
            logger.exception("SSE worker for %s failed", request.url.path)
            _push(("error", {"message": "Internal error; check the server log."}))
        finally:
            _push(_STREAM_END)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    async def stream():
        try:
            while True:
                item = await queue.get()
                if item is _STREAM_END:
                    break
                if await request.is_disconnected():
                    break
                yield encode_sse(item[0], item[1])
        finally:
            if on_finish is not None:
                on_finish()

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


def line_events(gen: Iterator[str]) -> Iterator[Tuple[str, Any]]:
    """Wrap a plain string-line generator (upload_and_index, sync_drive_now,
    teach_correction) as SSE events: every intermediate line is `progress`,
    the final line is additionally `done` so the client knows the run ended
    and can clear its spinner."""
    last = ""
    for line in gen:
        last = line
        yield ("progress", {"message": line})
    yield ("done", {"message": last})


def chat_stream_events(
    chat: Any,
    message: str,
    history: List[Dict],
    email: Optional[str],
    stop_event: threading.Event,
    max_length: int,
) -> Iterator[Tuple[str, Any]]:
    """The single implementation of a chat turn -- what the old per-backend
    Gradio `respond()` generators did, deduplicated: validate, prepare the
    RAG prompt + sources, stream the model, persist, and classify the
    outcome. Yields (event, payload) tuples for SSE:

        status  {"phase": "thinking"}          once, until first real text
        token   {"text": <cumulative answer>}  every real content chunk
        done    {"sources": payload|None,
                 "stopped": bool}              normal completion (incl. a
                                               user-stopped stream)
        error   {"message": str}               failure, or an empty model
                                               response

    Display vs persistence: the client renders the raw model text plus its
    own decorations (sources footer, stopped line, error styling); what's
    saved is the raw text (or the safe generic error text), so reloads and
    feedback records stay clean -- the same rule the Gradio version had.
    """
    try:
        validate_message(message, max_length)
    except ValueError as e:
        yield ("error", {"message": str(e)})
        return

    stop_event.clear()
    yield ("status", {"phase": "thinking"})

    persist: Optional[str] = None
    try:
        messages, sources = chat.prepare_turn(message, history or [])
        raw_answer: Optional[str] = None
        for partial in chat.stream_answer(messages, stop_event):
            raw_answer = partial
            yield ("token", {"text": partial})

        if raw_answer is None:
            if stop_event.is_set():
                # Stopped before the first token: not an error, don't leave
                # the client staring at a frozen "thinking" state.
                yield ("done", {"sources": None, "stopped": True})
            else:
                yield ("error", {"message": "The model returned no response content."})
        else:
            stopped = stop_event.is_set()
            yield ("done", {"sources": None if stopped else sources, "stopped": stopped})
            # A user-stop keeps whatever text arrived, so it persists too.
            persist = raw_answer
    except Exception as e:
        persist = safe_error_message(e, logger)
        yield ("error", {"message": persist})

    if email and persist:
        chat.save_message(email, "user", message)
        chat.save_message(email, "assistant", persist)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ChatBody(BaseModel):
    message: str


class FeedbackBody(BaseModel):
    question: str = ""
    answer: str = ""
    rating: str  # "like" | "dislike"


class TeachBody(BaseModel):
    question: str
    answer: str


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def register_api_routes(app: fastapi.FastAPI, chat: Any) -> None:
    """Mount the web UI + JSON/SSE API on `app` (auth middleware already
    added). Shared by both backend variants, so neither keeps a copy of the
    delivery plumbing."""
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    def _email(request: Request) -> Optional[str]:
        return get_email_from_request(request, AUTH_CONFIG["session_secret"])

    # -- Chat ---------------------------------------------------------------

    @app.post("/api/chat")
    async def api_chat(body: ChatBody, request: Request):
        email = _email(request)
        stop_event = chat._stop_events.event_for(email)
        history = chat.load_history(email) if email else []
        return sse_response(
            lambda: chat_stream_events(
                chat, body.message, history, email, stop_event, chat.max_message_length
            ),
            request,
            on_finish=stop_event.set,
        )

    @app.post("/api/chat/stop")
    def api_chat_stop(request: Request) -> Dict[str, str]:
        """Set this user's stop Event. The client ALSO aborts its fetch (the
        disconnect poll releases the stream between tokens); this endpoint
        makes the release immediate even while the model is mid-prefill and
        no token is arriving."""
        stop_event = chat._stop_events.event_for(_email(request))
        stop_event.set()
        return {"status": "stopping"}

    @app.get("/api/history")
    def api_history(request: Request) -> Dict[str, List[Dict]]:
        email = _email(request)
        return {"messages": chat.load_history(email) if email else []}

    @app.delete("/api/history")
    def api_history_clear(request: Request) -> Dict[str, str]:
        email = _email(request)
        if email:
            chat.clear_history(email)
        return {"status": "cleared"}

    @app.post("/api/feedback")
    def api_feedback(body: FeedbackBody, request: Request) -> Dict[str, str]:
        # The client sends the question + answer it's rating (it renders
        # the thread), so no index lookups server-side; rating is normalized
        # so a tampered client can't write other values into the table.
        rating = body.rating if body.rating in ("like", "dislike") else "dislike"
        chat.record_feedback(_email(request) or "anonymous", body.question, body.answer, rating)
        return {"status": "recorded"}

    @app.get("/api/summary")
    def api_summary(force: bool = False) -> Dict[str, Any]:
        """Top-bar chip AND the welcome-state suggestion chips in one call
        (the page needs both on load; they share the cached metadata scan).
        force=True busts the backend's brief source-metadata cache (the
        Library tab's manual 'Refresh stats' button uses the forced variant)."""
        summary = chat.index_summary(force=force)
        top = [
            {"source": source, "chunks": count}
            for source, count in (summary.get("top_sources") or [])
        ]
        return {
            "chunks": summary.get("chunks", 0),
            "documents": summary.get("documents", 0),
            "last_sync": summary.get("last_sync"),
            "top_sources": top,
            "suggestions": build_suggestions([t["source"] for t in top]),
        }

    # -- Library ------------------------------------------------------------

    @app.post("/api/library/upload")
    async def api_library_upload(request: Request, files: List[UploadFile] = File(...)):
        if not files:
            return JSONResponse({"detail": "No file received."}, status_code=400)
        # Copy to server temp files first: extractors read from disk paths,
        # and the originals are only in memory (multipart buffers).
        tmpdir = tempfile.mkdtemp(prefix="chatty-upload-")
        stubs = []

        class _Stub:
            """upload_and_index's expected file shape (path + display name),
            plus `source`: the name stored as the document's source."""

            def __init__(self, path: str, orig_name: str, source: str):
                self.path = path
                self.orig_name = orig_name
                self.source = source

        for f in files:
            # The client controls this string (single upload: the base name;
            # folder upload: the relative path, so citations keep structure
            # and same-named files in different subfolders stay distinct).
            rel = (f.filename or "").replace("\\", "/").lstrip("/") or "upload"
            name = Path(rel).name or "upload"
            tmp_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}-{name}")
            with open(tmp_path, "wb") as out:
                shutil.copyfileobj(f.file, out)
            stubs.append(_Stub(tmp_path, name, source=rel))

        def produce():
            try:
                yield from line_events(upload_and_index(stubs, chat.index_text))
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        return sse_response(produce, request)

    @app.post("/api/library/sync")
    async def api_library_sync(request: Request):
        return sse_response(lambda: line_events(sync_drive_now(chat.sync_drive)), request)

    @app.post("/api/library/teach")
    async def api_library_teach(body: TeachBody, request: Request):
        return sse_response(
            lambda: line_events(teach_correction(body.question, body.answer, chat.index_text, request)),
            request,
        )

    @app.get("/api/library/summary")
    def api_library_summary(force: bool = False) -> Dict[str, Any]:
        summary = chat.index_summary(force=force)
        return {
            "chunks": summary.get("chunks", 0),
            "documents": summary.get("documents", 0),
            "last_sync": summary.get("last_sync"),
            "top_sources": [
                {"source": source, "chunks": count}
                for source, count in (summary.get("top_sources") or [])
            ],
        }

    @app.get("/api/library/feedback")
    def api_library_feedback(request: Request) -> List[Dict[str, Any]]:
        email = _email(request)
        if not email:
            return []
        try:
            return chat.load_feedback(email)
        except Exception:
            logger.warning("Failed to load feedback rows", exc_info=True)
            return []


def build_app(chat: Any) -> fastapi.FastAPI:
    """Assemble the full HTTP app around an existing chat backend instance:
    health endpoint, the Google-OAuth session gate, and the web UI + JSON/SSE
    API. Both backend variants call this, so the wiring (and therefore the
    delivered UI) can't drift between them.

    `chat` must satisfy the protocol documented at the top of this module."""
    app = fastapi.FastAPI()

    @app.get("/health")
    def health():
        ok = chat.health_check()
        status_code = 200 if ok else 503
        return JSONResponse({"status": "ok" if ok else "unavailable"}, status_code=status_code)

    # Gates every other route (including the web UI) behind Google sign-in
    # restricted to AUTH_CONFIG['allowed_emails'].
    register_auth_routes(app)
    register_api_routes(app, chat)
    return app
