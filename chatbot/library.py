"""Pure 'Library' handlers: upload-to-index, Google Drive sync, and
teach-a-correction. Used by the web API (api_routes.py) as SSE progress
generators. Built once and used by both pgv_chatty.py and chromadb_chatty.py
so the two backends don't duplicate this logic.

The handler functions are plain, dependency-injected callables -- no Gradio,
no closures over app internals -- so they can be unit tested without any
backend. Each is a generator whose yields are rendered live: the API layers
stream them one line at a time, so a big folder upload shows
"Indexing 7/23: report.pdf..." while it embeds instead of silence until the
batch ends.
"""
import logging
import os

from rag_common import extract_text_from_upload, safe_error_message

logger = logging.getLogger(__name__)

# So uploaded files get the same "mime_type" metadata Drive-sourced content
# does, letting "list all PDFs"/"list all spreadsheets" catch uploads too.
_EXTENSION_MIME_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def upload_and_index(file_paths, index_text_fn):
    """Index uploaded file(s) or a folder's files, yielding progress as it
    works.

    file_paths may be None, a single file (a string path or an object with
    .path/.orig_name), or a list of them. An entry may also carry a `source`
    attribute: the name to store as the document's source (the web API sets
    a folder upload's relative path so citations keep their structure and
    same-named files in different subfolders don't collide).

    index_text_fn(text, source, extra_metadata=None) -> int chunks_added.

    The final yield is the summary; unsupported types and files without
    extractable text are skipped and named.
    """
    if file_paths is None:
        yield "No file selected."
        return
    if not isinstance(file_paths, (list, tuple)):
        file_paths = [file_paths]

    indexed, skipped, total_chunks = 0, [], 0
    total = len(file_paths)
    for i, f in enumerate(file_paths, start=1):
        path = getattr(f, "path", None) or getattr(f, "name", None) or f
        name = os.path.basename(getattr(f, "orig_name", None) or str(path))
        source = getattr(f, "source", None) or name
        yield (f"Indexing {i}/{total}: {name}..." if total > 1 else f"Indexing {name}...")
        ext = os.path.splitext(name)[1].lower()
        if ext not in _EXTENSION_MIME_TYPES:
            skipped.append(f"{name} (unsupported: {ext or 'no extension'})")
            continue
        text = extract_text_from_upload(path)
        if not text or not text.strip():
            skipped.append(f"{name} (no extractable text)")
            continue
        mime_type = _EXTENSION_MIME_TYPES.get(ext)
        try:
            added = index_text_fn(
                text, source,
                extra_metadata={"mime_type": mime_type} if mime_type else None,
            )
        except Exception as e:
            logger.warning("Indexing uploaded file %s failed: %s", name, e)
            skipped.append(f"{name} (indexing failed)")
            continue
        indexed += 1
        total_chunks += added

    if not indexed and not skipped:
        yield "No files selected."
        return
    summary = f"Indexed {indexed} file(s), {total_chunks} new chunk(s)."
    if skipped:
        shown = skipped[:5]
        more = len(skipped) - len(shown)
        summary += " Skipped: " + ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")
    yield summary


def sync_drive_now(drive_sync_fn):
    """drive_sync_fn() -> int chunks_added, or -1 when a sync is already
    running / in cooldown. Yields an immediate "started" line (a full
    sync takes minutes) then the outcome. Raises RuntimeError if Drive
    isn't connected yet (no refresh token stored)."""
    yield "Starting Google Drive sync... this can take a few minutes."
    try:
        added = drive_sync_fn()
    except RuntimeError as e:
        yield f"{e} Visit /login to connect Google Drive first."
        return
    except Exception as e:
        yield safe_error_message(e, logger)
        return
    if added < 0:
        yield "A Drive sync is already running (or just finished) -- not starting another."
        return
    yield f"Synced Google Drive: {added} new chunk(s) indexed."


def teach_correction(question: str, answer: str, index_text_fn, request=None):
    """Index a user-reported correction as first-class knowledge so future
    similar questions retrieve it as context (this is how the app "learns"
    from mistakes users point out).

    Yields progress lines (rendered live) ending with the outcome.
    request: an object with a .cookies mapping (FastAPI Request or a test
    stand-in) used to attribute the correction to the signed-in user.
    """
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question or not answer:
        yield "Fill in both the question it got wrong and the correct answer."
        return
    from app_session import get_email_from_request
    from gdrive_config import AUTH_CONFIG

    user_email = get_email_from_request(request, AUTH_CONFIG["session_secret"]) or "unknown"
    yield "Indexing the correction..."
    added = index_text_fn(
        f"Question: {question}\nCorrect answer: {answer}",
        source="user-corrections",
        extra_metadata={"type": "correction", "user": user_email},
    )
    logger.info("User %s taught a correction for %r (%d chunks)", user_email, question[:60], added)
    yield (
        "Thanks! Correction saved and indexed -- future similar questions "
        f"will answer from it ({added} chunk(s) added)."
    )
