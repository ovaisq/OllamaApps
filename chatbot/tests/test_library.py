"""Tests for the shared Library handlers (library.py): upload-to-index,
Drive sync, and teach-a-correction. These are plain dependency-injected
generators (no Gradio, no backend), so they're tested directly."""
from unittest.mock import MagicMock

import library


def _consume(gen):
    return list(gen)


class _UploadedFile:
    """Stand-in for the web API's file stub (path + display name, plus an
    optional `source` override for folder relative paths)."""

    def __init__(self, path, orig_name, source=None):
        self.path = path
        self.orig_name = orig_name
        if source is not None:
            self.source = source


class _StrPath:
    """A file given as a bare string path (no attributes)."""

    def __init__(self, path):
        self.path = path


def test_sync_drive_now_reports_blocked_sync():
    """The UI must say why nothing happened when a second sync is rejected,
    not silently claim success."""
    outputs = _consume(library.sync_drive_now(lambda: -1))
    assert outputs[0] == "Starting Google Drive sync... this can take a few minutes."
    assert "already running" in outputs[-1]
    assert "Synced" not in outputs[-1]


def test_sync_drive_now_reports_success():
    outputs = _consume(library.sync_drive_now(lambda: 5))
    assert outputs[0].startswith("Starting Google Drive sync")
    assert outputs[-1] == "Synced Google Drive: 5 new chunk(s) indexed."


def test_sync_drive_now_wraps_missing_drive_auth():
    def no_token():
        raise RuntimeError("No refresh token stored")

    outputs = _consume(library.sync_drive_now(no_token))
    assert "Visit /login" in outputs[-1]
    assert "No refresh token stored" in outputs[-1]


def test_sync_drive_now_hides_internal_errors():
    def boom():
        raise ConnectionError("db-password=hunter2 leaked")

    outputs = _consume(library.sync_drive_now(boom))
    assert "hunter2" not in outputs[-1]
    assert "error id" in outputs[-1]


def test_upload_and_index_rejects_unselected_file():
    assert _consume(library.upload_and_index(None, MagicMock())) == ["No file selected."]


def test_upload_and_index_still_accepts_a_single_file(tmp_path):
    f = tmp_path / "single.md"
    f.write_text("single file content")
    index_text_fn = MagicMock(return_value=1)

    outputs = _consume(library.upload_and_index(str(f), index_text_fn))

    assert outputs[0] == "Indexing single.md..."
    assert "Indexed 1 file(s), 1 new chunk(s)." in outputs[-1]
    assert index_text_fn.call_args.args[1] == "single.md"


def test_upload_and_index_streams_per_file_progress(tmp_path):
    """A multi-file batch must show per-file progress while it works, not
    silence until the end."""
    first = tmp_path / "notes.md"
    first.write_text("Some markdown notes about Kona the dog.")
    second = tmp_path / "numbers.txt"
    second.write_text("plain text file with some content")
    index_text_fn = MagicMock(return_value=2)

    outputs = _consume(library.upload_and_index([str(first), str(second)], index_text_fn))

    assert outputs[0] == "Indexing 1/2: notes.md..."
    assert outputs[1] == "Indexing 2/2: numbers.txt..."
    assert "Indexed 2 file(s), 4 new chunk(s)." in outputs[-1]
    assert [c.args[1] for c in index_text_fn.call_args_list] == ["notes.md", "numbers.txt"]


def test_upload_and_index_handles_folder_uploads(tmp_path):
    """Folder uploads arrive as a list including files of every type;
    unsupported ones are skipped by name, supported ones indexed."""
    doc = tmp_path / "folder" / "notes.md"
    doc.parent.mkdir()
    doc.write_text("markdown inside an uploaded folder")
    image = tmp_path / "folder" / "logo.png"
    image.write_bytes(b"\x89PNG")
    index_text_fn = MagicMock(return_value=1)

    outputs = _consume(library.upload_and_index(
        [_UploadedFile(str(doc), "folder/notes.md"),
         _UploadedFile(str(image), "folder/logo.png")],
        index_text_fn,
    ))

    assert outputs[0] == "Indexing 1/2: notes.md..."
    assert index_text_fn.call_count == 1
    assert index_text_fn.call_args.args[1] == "notes.md"
    assert "Indexed 1 file(s), 1 new chunk(s)." in outputs[-1]
    assert "logo.png (unsupported: .png)" in outputs[-1]


def test_upload_and_index_uses_explicit_source_override(tmp_path):
    """The web API passes a folder upload's relative path as `source` so
    citations keep their structure and same-named files in different
    subfolders don't collide. When `source` is absent the base name is used."""
    doc = tmp_path / "sub" / "memo.md"
    doc.parent.mkdir()
    doc.write_text("relative path source")
    index_text_fn = MagicMock(return_value=1)

    # With an explicit source attribute, it wins over the base name.
    _consume(library.upload_and_index(
        [_UploadedFile(str(doc), "sub/memo.md", source="reports/2025/memo.md")],
        index_text_fn,
    ))
    assert index_text_fn.call_args.args[1] == "reports/2025/memo.md"

    # Without one, the display base name is stored (single-file uploads).
    index_text_fn.reset_mock()
    _consume(library.upload_and_index(
        [_UploadedFile(str(doc), "memo.md")],
        index_text_fn,
    ))
    assert index_text_fn.call_args.args[1] == "memo.md"


def test_upload_and_index_reports_indexing_failures_per_file(tmp_path):
    good = tmp_path / "good.md"
    good.write_text("good content")
    bad = tmp_path / "bad.md"
    bad.write_text("bad content")
    index_text_fn = MagicMock(side_effect=[3, RuntimeError("boom")])

    outputs = _consume(library.upload_and_index([str(good), str(bad)], index_text_fn))

    assert "Indexed 1 file(s), 3 new chunk(s)." in outputs[-1]
    assert "bad.md (indexing failed)" in outputs[-1]


def test_teach_correction_indexes_the_correction_as_authoritative_knowledge():
    """Teaching a correction is how the app learns from user-reported
    mistakes: the correction must land in the vector index (so future
    similar questions retrieve it) tagged as a correction."""
    index_text_fn = MagicMock(return_value=1)

    outputs = _consume(library.teach_correction("what is the deploy host?", "deploy-5", index_text_fn))

    index_text_fn.assert_called_once()
    call = index_text_fn.call_args
    text = call.args[0]
    assert call.kwargs["source"] == "user-corrections"
    assert "what is the deploy host?" in text
    assert "deploy-5" in text
    assert call.kwargs["extra_metadata"]["type"] == "correction"
    assert outputs[0] == "Indexing the correction..."
    assert "Correction saved" in outputs[-1]


def test_teach_correction_requires_both_fields():
    index_text_fn = MagicMock()
    assert _consume(library.teach_correction("only a question", "", index_text_fn)) == [
        "Fill in both the question it got wrong and the correct answer."
    ]
    assert _consume(library.teach_correction("", "only an answer", index_text_fn)) == [
        "Fill in both the question it got wrong and the correct answer."
    ]
    index_text_fn.assert_not_called()


def _docx_bytes(paragraphs):
    """A real minimal .docx for upload-path tests (OOXML is a zip of XML)."""
    import io
    import zipfile

    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>' for p in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<w:document {ns}><w:body>{body}</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", document)
    return buf.getvalue()


def test_word_extensions_are_supported_with_drive_matching_mimes():
    assert ".docx" in library._EXTENSION_MIME_TYPES
    assert ".doc" in library._EXTENSION_MIME_TYPES
    # The stored mime_type must match Drive-sourced Word files, so
    # "list all docs" style catalog queries catch uploads and Drive files alike.
    assert library._EXTENSION_MIME_TYPES[".docx"] == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert library._EXTENSION_MIME_TYPES[".doc"] == "application/msword"


def test_upload_and_index_handles_word_document_uploads(tmp_path, monkeypatch):
    """.docx (stdlib extractor) and legacy .doc (antiword/catdoc) are
    indexed like any other supported type -- previously they were rejected
    as unsupported on upload."""
    docx = tmp_path / "memo.docx"
    docx.write_bytes(_docx_bytes(["word doc content"]))
    doc = tmp_path / "legacy.doc"
    doc.write_bytes(b"legacy body")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    import shutil as _sh
    fake = bin_dir / "antiword"
    fake.write_text('#!/bin/sh\n' + _sh.which("cat") + ' "$1"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))

    index_text_fn = MagicMock(return_value=2)
    outputs = _consume(library.upload_and_index([str(docx), str(doc)], index_text_fn))

    assert index_text_fn.call_count == 2
    assert [c.args[1] for c in index_text_fn.call_args_list] == ["memo.docx", "legacy.doc"]
    assert "Indexed 2 file(s), 4 new chunk(s)." in outputs[-1]
    mimes = [c.kwargs["extra_metadata"]["mime_type"] for c in index_text_fn.call_args_list]
    assert mimes == [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ]


def test_line_events_marks_the_final_line_done():
    """The API wraps each handler's string-line generator as SSE events:
    intermediate lines are `progress`, and the final line is additionally
    `done` so the client clears its spinner exactly once."""
    from api_routes import line_events

    events = list(line_events(iter(["a", "b", "c"])))
    assert events == [
        ("progress", {"message": "a"}),
        ("progress", {"message": "b"}),
        ("progress", {"message": "c"}),
        ("done", {"message": "c"}),
    ]
