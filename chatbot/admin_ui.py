"""Shared 'Admin' Gradio tab: upload-to-index, Google Drive sync, and index
stats. Built once here and used by both pgv_chatty.py and chromadb_chatty.py
so the two backends don't duplicate this UI wiring.

The handler functions below are plain, dependency-injected callables (not
closures over app internals) so they can be unit tested without spinning up
Gradio or a real backend.
"""
import logging
import os

import gradio as gr

from rag_common import extract_text_from_upload, safe_error_message

logger = logging.getLogger(__name__)

# So uploaded files get the same "mime_type" metadata Drive-sourced content
# does, letting "list all PDFs"/"list all spreadsheets" catch uploads too.
_EXTENSION_MIME_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def upload_and_index(file_path: str, index_text_fn) -> str:
    """index_text_fn(text: str, source: str, extra_metadata: dict = None) -> int chunks_added."""
    if not file_path:
        return "No file selected."
    text = extract_text_from_upload(file_path)
    if not text or not text.strip():
        return "No extractable text found in that file (supported: .md, .txt, .pdf, .xlsx)."
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = _EXTENSION_MIME_TYPES.get(ext)
    try:
        added = index_text_fn(
            text, os.path.basename(file_path),
            extra_metadata={"mime_type": mime_type} if mime_type else None,
        )
    except Exception as e:
        return safe_error_message(e, logger)
    return f"Indexed {added} new chunk(s) from {os.path.basename(file_path)}."


def sync_drive_now(drive_sync_fn) -> str:
    """drive_sync_fn() -> int chunks_added. Raises RuntimeError if Drive isn't
    connected yet (no refresh token stored)."""
    try:
        added = drive_sync_fn()
    except RuntimeError as e:
        return f"{e} Visit /login to connect Google Drive first."
    except Exception as e:
        return safe_error_message(e, logger)
    return f"Synced Google Drive: {added} new chunk(s) indexed."


def get_index_stats(count_fn) -> str:
    try:
        total = count_fn()
    except Exception as e:
        return safe_error_message(e, logger)
    return f"Total indexed chunks: {total}"


def build_admin_tab(index_text_fn, count_fn, drive_sync_fn) -> None:
    """Adds an 'Admin' tab to the enclosing gr.Blocks context."""
    with gr.Tab("Admin"):
        gr.Markdown("## Add content")
        upload = gr.File(
            label="Upload a .md / .txt / .pdf / .xlsx file",
            file_types=[".md", ".txt", ".pdf", ".xlsx"],
        )
        upload_status = gr.Markdown()
        upload.upload(
            lambda f: upload_and_index(f.name if f else None, index_text_fn), upload, upload_status
        )

        gr.Markdown("## Google Drive")
        drive_status = gr.Markdown()
        sync_btn = gr.Button("Sync Google Drive now")
        sync_btn.click(lambda: sync_drive_now(drive_sync_fn), None, drive_status)

        gr.Markdown("## Index status")
        stats = gr.Markdown()
        refresh_btn = gr.Button("Refresh stats")
        refresh_btn.click(lambda: get_index_stats(count_fn), None, stats)
