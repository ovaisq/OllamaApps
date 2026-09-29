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
