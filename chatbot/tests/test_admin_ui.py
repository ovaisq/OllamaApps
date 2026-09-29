"""Tests for the shared Admin tab handlers."""
from unittest.mock import MagicMock

import admin_ui


def _consume(gen):
    return list(gen)


class _UploadedFile:
    """Stand-in for gradio's FileData as passed to upload event handlers."""

    def __init__(self, path, orig_name):
        self.path = path
        self.orig_name = orig_name


def test_sync_drive_now_reports_blocked_sync():
    """The UI must say why nothing happened when a second sync is rejected,
    not silently claim success."""
    outputs = _consume(admin_ui.sync_drive_now(lambda: -1))

    assert outputs[0] == "Starting Google Drive sync... this can take a few minutes."
    assert "already running" in outputs[-1]
    assert "Synced" not in outputs[-1]


def test_sync_drive_now_reports_success():
    outputs = _consume(admin_ui.sync_drive_now(lambda: 5))

    assert outputs[0].startswith("Starting Google Drive sync")
    assert outputs[-1] == "Synced Google Drive: 5 new chunk(s) indexed."


def test_sync_drive_now_wraps_missing_drive_auth():
    def no_token():
        raise RuntimeError("No refresh token stored")

    outputs = _consume(admin_ui.sync_drive_now(no_token))

    assert "Visit /login" in outputs[-1]
    assert "No refresh token stored" in outputs[-1]


def test_sync_drive_now_hides_internal_errors():
    def boom():
        raise ConnectionError("db-password=hunter2 leaked")

    outputs = _consume(admin_ui.sync_drive_now(boom))

    assert "hunter2" not in outputs[-1]
    assert "error id" in outputs[-1]


def test_upload_and_index_rejects_unselected_file():
    assert _consume(admin_ui.upload_and_index(None, MagicMock())) == ["No file selected."]


def test_upload_and_index_still_accepts_a_single_file(tmp_path):
    f = tmp_path / "single.md"
    f.write_text("single file content")
    index_text_fn = MagicMock(return_value=1)

    outputs = _consume(admin_ui.upload_and_index(str(f), index_text_fn))

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

    outputs = _consume(admin_ui.upload_and_index([str(first), str(second)], index_text_fn))

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

    outputs = _consume(admin_ui.upload_and_index(
        [_UploadedFile(str(doc), "folder/notes.md"),
         _UploadedFile(str(image), "folder/logo.png")],
        index_text_fn,
    ))

    assert outputs[0] == "Indexing 1/2: notes.md..."
    assert index_text_fn.call_count == 1
    assert index_text_fn.call_args.args[1] == "notes.md"
    assert "Indexed 1 file(s), 1 new chunk(s)." in outputs[-1]
    assert "logo.png (unsupported: .png)" in outputs[-1]


def test_upload_and_index_reports_indexing_failures_per_file(tmp_path):
    good = tmp_path / "good.md"
    good.write_text("good content")
    bad = tmp_path / "bad.md"
    bad.write_text("bad content")
    index_text_fn = MagicMock(side_effect=[3, RuntimeError("boom")])

    outputs = _consume(admin_ui.upload_and_index([str(good), str(bad)], index_text_fn))

    assert "Indexed 1 file(s), 3 new chunk(s)." in outputs[-1]
    assert "bad.md (indexing failed)" in outputs[-1]


def test_teach_correction_indexes_the_correction_as_authoritative_knowledge():
    """Teaching a correction is how the app learns from user-reported
    mistakes: the correction must land in the vector index (so future
    similar questions retrieve it) tagged as a correction."""
    index_text_fn = MagicMock(return_value=1)

    outputs = _consume(
        admin_ui.teach_correction("what is the deploy host?", "deploy-5", index_text_fn)
    )

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

    assert _consume(admin_ui.teach_correction("only a question", "", index_text_fn)) == [
        "Fill in both the question it got wrong and the correct answer."
    ]
    assert _consume(admin_ui.teach_correction("", "only an answer", index_text_fn)) == [
        "Fill in both the question it got wrong and the correct answer."
    ]

    index_text_fn.assert_not_called()
