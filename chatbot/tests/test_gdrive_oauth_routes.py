from unittest.mock import patch

import fastapi
import pytest
from fastapi.testclient import TestClient

import gdrive_oauth_routes
from gdrive_oauth_routes import register_routes


@pytest.fixture(autouse=True)
def reset_pending_state():
    gdrive_oauth_routes._pending_state["value"] = None
    yield
    gdrive_oauth_routes._pending_state["value"] = None


def make_client():
    app = fastapi.FastAPI()
    register_routes(app)
    return TestClient(app)


def test_connect_drive_rejects_requests_without_a_valid_admin_token():
    client = make_client()
    with patch("gdrive_oauth_routes.GOOGLE_CONFIG", {"admin_token": "secret"}):
        resp = client.get("/connect-drive", follow_redirects=False)
    assert resp.status_code == 403

    with patch("gdrive_oauth_routes.GOOGLE_CONFIG", {"admin_token": "secret"}):
        resp = client.get("/connect-drive?token=wrong-token", follow_redirects=False)
    assert resp.status_code == 403


def test_connect_drive_redirects_to_google_consent_screen_with_valid_token():
    client = make_client()
    config = {"admin_token": "secret", "client_id": "cid", "redirect_uri": "http://h/cb"}
    with patch("gdrive_oauth_routes.GOOGLE_CONFIG", config):
        resp = client.get("/connect-drive?token=secret", follow_redirects=False)

    assert resp.status_code in (302, 307)
    assert "accounts.google.com" in resp.headers["location"]
    assert "drive.readonly" in resp.headers["location"]


def test_oauth2callback_rejects_state_that_was_never_issued():
    client = make_client()
    resp = client.get("/oauth2callback", params={"code": "abc", "state": "unknown-state"})
    assert resp.status_code == 403


def test_oauth2callback_rejects_replay_of_an_already_consumed_state(tmp_path):
    gdrive_oauth_routes._pending_state["value"] = "s1"
    client = make_client()

    with patch("gdrive_oauth_routes.DRIVE_CONFIG", {"token_store_path": str(tmp_path / "t.json")}), \
         patch(
             "gdrive_oauth_routes.exchange_code_for_tokens",
             return_value={"access_token": "a", "refresh_token": "r"},
         ):
        first = client.get("/oauth2callback", params={"code": "abc", "state": "s1"})
        second = client.get("/oauth2callback", params={"code": "abc", "state": "s1"})

    assert first.status_code == 200
    assert second.status_code == 403


def test_oauth2callback_saves_refresh_token_on_success(tmp_path):
    gdrive_oauth_routes._pending_state["value"] = "s2"
    client = make_client()
    token_path = tmp_path / "token.json"

    with patch("gdrive_oauth_routes.DRIVE_CONFIG", {"token_store_path": str(token_path)}), \
         patch(
             "gdrive_oauth_routes.exchange_code_for_tokens",
             return_value={"access_token": "a", "refresh_token": "r"},
         ):
        resp = client.get("/oauth2callback", params={"code": "abc", "state": "s2"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert token_path.exists()


def test_oauth2callback_reports_error_without_leaking_when_google_denies_consent():
    client = make_client()
    resp = client.get("/oauth2callback", params={"error": "access_denied"})

    assert resp.status_code == 400
    assert resp.json()["detail"] == "access_denied"


def test_oauth2callback_requires_code_or_error():
    client = make_client()
    resp = client.get("/oauth2callback")
    assert resp.status_code == 400
