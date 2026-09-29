from unittest.mock import MagicMock, patch

import httpx
import pytest

import gdrive_auth
import gdrive_client
import gdrive_token_store


def test_get_google_auth_url_requests_readonly_scope_and_offline_access():
    url = gdrive_auth.get_google_auth_url("cid", "http://host/oauth2callback", state="s")

    assert "client_id=cid" in url
    assert "drive.readonly" in url
    assert "access_type=offline" in url
    assert "prompt=consent" in url


def test_refresh_access_token_returns_access_token():
    fake_response = httpx.Response(200, json={"access_token": "abc123"})
    with patch("httpx.post", return_value=fake_response):
        token = gdrive_auth.refresh_access_token("cid", "secret", "refresh-tok")
    assert token == "abc123"


def test_refresh_access_token_raises_on_failure():
    fake_response = httpx.Response(400, text="invalid_grant")
    with patch("httpx.post", return_value=fake_response):
        with pytest.raises(RuntimeError):
            gdrive_auth.refresh_access_token("cid", "secret", "bad-tok")


def test_list_files_filters_to_supported_mime_types_and_paginates():
    page1 = httpx.Response(
        200,
        json={
            "nextPageToken": "p2",
            "files": [
                {"id": "1", "name": "doc.gdoc", "mimeType": "application/vnd.google-apps.document"},
                {"id": "2", "name": "image.png", "mimeType": "image/png"},
            ],
        },
    )
    page2 = httpx.Response(
        200,
        json={"files": [{"id": "3", "name": "notes.md", "mimeType": "text/markdown"}]},
    )

    mock_client = MagicMock()
    mock_client.get.side_effect = [page1, page2]
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False

    with patch("gdrive_client.httpx.Client", return_value=mock_client):
        files = list(gdrive_client.list_files("token", folder_id="f1"))

    ids = [f["id"] for f in files]
    assert ids == ["1", "3"]  # image.png (unsupported mimeType) is filtered out


def test_fetch_file_text_exports_google_docs_as_plain_text():
    with patch("httpx.get", return_value=httpx.Response(200, text="doc contents")):
        text = gdrive_client.fetch_file_text(
            "token", "fid", "application/vnd.google-apps.document"
        )
    assert text == "doc contents"


def test_fetch_file_text_returns_none_for_unsupported_mime_type():
    assert gdrive_client.fetch_file_text("token", "fid", "video/mp4") is None


def test_token_store_round_trips_refresh_token(tmp_path):
    path = str(tmp_path / "token.json")
    gdrive_token_store.save_refresh_token(path, "my-refresh-token")

    loaded = gdrive_token_store.load_refresh_token(path)
    assert loaded == "my-refresh-token"


def test_token_store_returns_none_when_missing(tmp_path):
    path = str(tmp_path / "does-not-exist.json")
    assert gdrive_token_store.load_refresh_token(path) is None


def test_token_store_returns_none_on_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("not json")
    assert gdrive_token_store.load_refresh_token(str(path)) is None
