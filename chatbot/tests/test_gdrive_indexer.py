import threading
import time
from unittest.mock import MagicMock, patch

import gdrive_indexer


def test_get_access_token_raises_when_never_authorized():
    with patch("gdrive_indexer.load_refresh_token", return_value=None):
        try:
            gdrive_indexer.get_access_token()
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "gdrive_authorize" in str(e) or "connect-drive" in str(e) or "refresh token" in str(e)


def test_run_pgvector_backend_skips_files_with_no_extractable_text():
    files = [
        {"id": "1", "name": "a.md", "mimeType": "text/markdown"},
        {"id": "2", "name": "b.png", "mimeType": "image/png"},
    ]

    # Files are processed concurrently (GDRIVE_MAX_WORKERS threads), so keyed
    # by file_id rather than a fixed call-order list.
    def fake_fetch(access_token, file_id, mime_type):
        return "hello world" if file_id == "1" else None

    with patch("gdrive_indexer.list_files", return_value=iter(files)), \
         patch("gdrive_indexer.fetch_file_text", side_effect=fake_fetch), \
         patch("pgv_indexer.index_text", return_value=3) as mock_index_text, \
         patch("psycopg2.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        total = gdrive_indexer.run_pgvector_backend("access-token")

    assert total == 3
    mock_index_text.assert_called_once_with(
        "hello world", "a.md", mock_conn, extra_metadata={"mime_type": "text/markdown"}
    )
    mock_conn.close.assert_called_once()


def test_files_are_processed_with_up_to_max_workers_concurrency():
    files = [{"id": str(i), "name": f"f{i}.md", "mimeType": "text/markdown"} for i in range(4)]
    concurrent_calls = {"current": 0, "max_seen": 0}
    lock = threading.Lock()

    def fake_fetch(access_token, file_id, mime_type):
        with lock:
            concurrent_calls["current"] += 1
            concurrent_calls["max_seen"] = max(concurrent_calls["max_seen"], concurrent_calls["current"])
        time.sleep(0.05)  # hold the "slot" long enough for overlap to show up
        with lock:
            concurrent_calls["current"] -= 1
        return "text"

    with patch("gdrive_indexer.list_files", return_value=iter(files)), \
         patch("gdrive_indexer.fetch_file_text", side_effect=fake_fetch), \
         patch("pgv_indexer.index_text", return_value=1), \
         patch("psycopg2.connect", return_value=MagicMock()), \
         patch("gdrive_indexer.DRIVE_CONFIG", {"folder_id": "", "max_workers": 2}):
        total = gdrive_indexer.run_pgvector_backend("access-token")

    assert total == 4
    assert concurrent_calls["max_seen"] == 2


def test_one_failing_file_does_not_abort_the_rest():
    files = [
        {"id": "1", "name": "good.md", "mimeType": "text/markdown"},
        {"id": "2", "name": "bad.md", "mimeType": "text/markdown"},
    ]

    def fake_fetch(access_token, file_id, mime_type):
        if file_id == "2":
            raise RuntimeError("Drive API blew up")
        return "hello world"

    with patch("gdrive_indexer.list_files", return_value=iter(files)), \
         patch("gdrive_indexer.fetch_file_text", side_effect=fake_fetch), \
         patch("pgv_indexer.index_text", return_value=5), \
         patch("psycopg2.connect", return_value=MagicMock()):
        total = gdrive_indexer.run_pgvector_backend("access-token")

    assert total == 5
