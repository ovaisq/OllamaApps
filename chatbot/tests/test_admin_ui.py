"""Tests for the shared Admin tab handlers."""
from unittest.mock import MagicMock

import admin_ui


def test_sync_drive_now_reports_blocked_sync():
    """The UI must say why nothing happened when a second sync is rejected,
    not silently claim success."""
    msg = admin_ui.sync_drive_now(lambda: -1)

    assert "already running" in msg
    assert "Synced" not in msg


def test_sync_drive_now_reports_success():
    msg = admin_ui.sync_drive_now(lambda: 5)

    assert "Synced Google Drive: 5 new chunk(s) indexed." == msg


def test_sync_drive_now_wraps_missing_drive_auth():
    def no_token():
        raise RuntimeError("No refresh token stored")

    msg = admin_ui.sync_drive_now(no_token)

    assert "Visit /login" in msg
    assert "No refresh token stored" in msg


def test_sync_drive_now_hides_internal_errors():
    def boom():
        raise ConnectionError("db-password=hunter2 leaked")

    msg = admin_ui.sync_drive_now(boom)

    assert "hunter2" not in msg
    assert "error id" in msg


def test_upload_and_index_rejects_unselected_file():
    assert admin_ui.upload_and_index(None, MagicMock()) == "No file selected."


def test_teach_correction_indexes_the_correction_as_authoritative_knowledge():
    """Teaching a correction is how the app learns from user-reported
    mistakes: the correction must land in the vector index (so future
    similar questions retrieve it) tagged as a correction."""
    index_text_fn = MagicMock(return_value=1)

    msg = admin_ui.teach_correction("what is the deploy host?", "deploy-5", index_text_fn)

    index_text_fn.assert_called_once()
    call = index_text_fn.call_args
    text = call.args[0]
    assert call.kwargs["source"] == "user-corrections"
    assert "what is the deploy host?" in text
    assert "deploy-5" in text
    assert call.kwargs["extra_metadata"]["type"] == "correction"
    assert "Correction saved" in msg


def test_teach_correction_requires_both_fields():
    index_text_fn = MagicMock()

    assert "Fill in both" in admin_ui.teach_correction("only a question", "", index_text_fn)
    assert "Fill in both" in admin_ui.teach_correction("", "only an answer", index_text_fn)

    index_text_fn.assert_not_called()
