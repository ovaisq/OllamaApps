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

    with patch("gdrive_indexer.list_files", return_value=iter(files)), \
         patch("gdrive_indexer.fetch_file_text", side_effect=["hello world", None]), \
         patch("pgv_indexer.index_text", return_value=3) as mock_index_text, \
         patch("psycopg2.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        total = gdrive_indexer.run_pgvector_backend("access-token")

    assert total == 3
    mock_index_text.assert_called_once_with("hello world", "a.md", mock_conn)
    mock_conn.close.assert_called_once()
