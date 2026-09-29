"""Shared 'Library' Gradio tab (formerly 'Admin'): upload-to-index, Google
Drive sync, teach-a-correction (with recent-dislike prefill), and live
index stats. Built once here and used by both pgv_chatty.py and
chromadb_chatty.py so the two backends don't duplicate this UI wiring.

The handler functions below are plain, dependency-injected callables (not
closures over app internals) so they can be unit tested without spinning up
Gradio or a real backend.
"""
import logging
import os

import gradio as gr

from rag_common import extract_text_from_upload, safe_error_message
from ui_common import (
    INDEX_CHIP_INTERVAL_SECONDS,
    feedback_prefill,
    format_index_summary,
    refresh_feedback_rows,
)

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
    """Index uploaded file(s) or a whole folder, yielding progress as it
    works.

    file_paths may be None, a single gradio FileData (or a path string),
    or a list of them (gradio's file_count="multiple"/"directory" modes).
    index_text_fn(text, source, extra_metadata=None) -> int chunks_added.

    As a generator, every yield is rendered live in the upload status
    line (gradio streams generator handlers), so a big folder shows
    "Indexing 7/23: report.pdf..." while it embeds instead of silence
    until the batch ends. The final yield is the summary; unsupported
    types and files without extractable text are skipped and named.
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
                text, name,
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


def get_index_stats(count_fn) -> str:
    try:
        total = count_fn()
    except Exception as e:
        return safe_error_message(e, logger)
    return f"Total indexed chunks: {total}"


def teach_correction(question: str, answer: str, index_text_fn, request=None):
    """Index a user-reported correction as first-class knowledge so future
    similar questions retrieve it as context (this is how the app "learns"
    from mistakes users point out).

    Yields progress lines (rendered live) ending with the outcome.
    index_text_fn(text, source, extra_metadata=None) -> int chunks_added.
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


def build_admin_tab(index_text_fn, summary_fn, drive_sync_fn,
                    feedback_rows_fn=None, blocks: "gr.Blocks" = None) -> None:
    """Adds the 'Library' tab to the enclosing gr.Blocks context.

    summary_fn(force=False) -> {"chunks", "documents", "top_sources",
    "last_sync"} dict (see the chat classes' index_summary).
    feedback_rows_fn(email) -> recent disliked answers for the
    correction prefill (both backends implement it; None disables the bit).

    The old tab was one long single-column scroll (uploads, Drive, teach,
    stats); now it's a two-row card grid so a laptop screen shows all of
    it, and the index card refreshes itself instead of sitting stale until
    someone notices the 'Refresh stats' button.
    """
    with gr.Tab("Library"):
        gr.Markdown(
            "Add files or a folder — **.md, .txt, .pdf, .doc, .docx, .xlsx** "
            "get indexed; anything else is skipped and named in the status "
            "below."
        )
        # Row 1: content in, Drive synced, all visible at a glance.
        with gr.Row(equal_height=False):
            with gr.Column():
                gr.Markdown("### Add files")
                upload = gr.File(
                    label="Upload files (.md, .txt, .pdf, .doc, .docx, .xlsx)",
                    file_count="multiple",
                    file_types=[".md", ".txt", ".pdf", ".doc", ".docx", ".xlsx"],
                )
                upload_status = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Add a folder")
                folder_upload = gr.File(
                    label="Upload a folder",
                    file_count="directory",
                )
                folder_upload_status = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Google Drive")
                drive_status = gr.Markdown()
                sync_btn = gr.Button("Sync Google Drive now", variant="primary")

        with gr.Row(equal_height=False):
            with gr.Column():
                gr.Markdown(
                    "### Teach Chatty a correction\n"
                    "Did it get something wrong? Tell it the right answer here and it "
                    "will use the correction for similar questions from now on."
                )
                wrong_question = gr.Textbox(label="A question it answered wrong")
                correct_answer = gr.Textbox(label="The correct answer", lines=3)
                teach_status = gr.Markdown()
                teach_btn = gr.Button("Teach this correction")

            with gr.Column():
                gr.Markdown("### Index")
                # No backend call at build time: the load event below paints
                # it on page load, the timer keeps it live, the button forces
                # a fresh read.
                stats = gr.Markdown("_Index stats load on page open._")
                refresh_btn = gr.Button("Refresh stats", size="sm")

        def _index_uploads(files):
            # Generator handlers stream every yield to the status line.
            yield from upload_and_index(files, index_text_fn)

        upload.upload(
            _index_uploads, upload, upload_status,
            queue=True, show_progress="full", show_progress_on=[upload_status],
        )
        folder_upload.upload(
            _index_uploads, folder_upload, folder_upload_status,
            queue=True, show_progress="full", show_progress_on=[folder_upload_status],
        )

        def _sync_drive():
            yield from sync_drive_now(drive_sync_fn)

        sync_btn.click(_sync_drive, None, drive_status,
                       queue=True, show_progress="full", show_progress_on=[drive_status])

        def _teach(question, answer, request: gr.Request = None):
            yield from teach_correction(question, answer, index_text_fn, request)

        teach_btn.click(_teach, [wrong_question, correct_answer], teach_status,
                        queue=True, show_progress="full", show_progress_on=[teach_status])

        # Index card: live (timer + first paint on load) and manually
        # refreshable (force=True busts the sources cache).
        def _stats(force: bool = False) -> str:
            return format_index_summary(summary_fn(force=force))

        refresh_btn.click(lambda: _stats(force=True), None, stats,
                          show_progress="hidden")
        if blocks is not None:
            blocks.load(lambda: _stats(), None, stats, show_progress="hidden")
            gr.Timer(INDEX_CHIP_INTERVAL_SECONDS).tick(lambda: _stats(), None, stats)

        # Recent-dislike review: pick a mis-answer, its question prefills
        # the correction form above, the user supplies the right answer.
        if feedback_rows_fn is not None:
            gr.Markdown(
                "### Recent mis-answers\n"
                "From your 👎 feedback — pick one to start teaching a correction."
            )
            feedback_dd = gr.Dropdown(
                label="A mis-answer to fix", choices=[], interactive=True
            )
            feedback_preview = gr.Markdown()
            fb_state = gr.State([])

            def _refresh_feedback(request: gr.Request = None):
                return refresh_feedback_rows(feedback_rows_fn, request)

            fb_outputs = [feedback_dd, feedback_preview, fb_state]
            if blocks is not None:
                blocks.load(_refresh_feedback, None, fb_outputs, show_progress="hidden")
            feedback_refresh_btn = gr.Button("Refresh list", size="sm")
            feedback_refresh_btn.click(_refresh_feedback, None, fb_outputs,
                                       show_progress="hidden")
            feedback_dd.select(feedback_prefill, [feedback_dd, fb_state],
                               [wrong_question], show_progress="hidden")
