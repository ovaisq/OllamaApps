from unittest.mock import patch

import fastapi
import pytest
from fastapi.testclient import TestClient

import auth_routes
from app_session import SESSION_COOKIE, create_session_token
from auth_routes import register_routes

TEST_AUTH_CONFIG = {
    "session_secret": "test-secret-that-is-long-enough-1234567890",
    "session_max_age_seconds": 3600,
    "allowed_emails": {"allowed@example.com"},
}


@pytest.fixture(autouse=True)
def reset_pending_state():
    auth_routes._pending_state["value"] = None
    yield
    auth_routes._pending_state["value"] = None


@pytest.fixture(autouse=True)
def patched_auth_config():
    with patch("auth_routes.AUTH_CONFIG", TEST_AUTH_CONFIG):
        yield


def make_client():
    app = fastapi.FastAPI()

    @app.get("/protected")
    def protected():
        return {"status": "ok"}

    register_routes(app)
    return TestClient(app)


def test_login_and_oauth_start_are_reachable_without_a_session():
    client = make_client()
    assert client.get("/login", follow_redirects=False).status_code == 200
    assert client.get("/oauth-start", follow_redirects=False).status_code in (302, 307)


def test_login_renders_a_branded_page_not_json():
    """/login must be a styled Chatty screen (with a retry affordance via
    /oauth-start), not a bare redirect whose failure paths leave the user
    on a JSON body."""
    client = make_client()
    resp = client.get("/login")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Chatty" in resp.text
    assert 'href="/oauth-start"' in resp.text
    assert '"status"' not in resp.text  # never a raw JSON body


def test_protected_route_redirects_to_login_without_a_session_cookie():
    client = make_client()
    resp = client.get("/protected", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/login"


def test_protected_route_redirects_when_session_email_is_not_allowlisted():
    client = make_client()
    token = create_session_token("attacker@example.com", "test-secret-that-is-long-enough-1234567890", 3600)
    client.cookies.set(SESSION_COOKIE, token)
    resp = client.get("/protected", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/login"


def test_protected_route_accessible_with_a_valid_allowlisted_session():
    client = make_client()
    token = create_session_token("allowed@example.com", "test-secret-that-is-long-enough-1234567890", 3600)
    client.cookies.set(SESSION_COOKIE, token)
    resp = client.get("/protected")
    assert resp.status_code == 200


def test_oauth_start_redirects_to_google_with_combined_scope():
    """The /login landing page's button hits /oauth-start, which must do
    the Google redirect (combining app login + Drive read in one consent)."""
    client = make_client()
    resp = client.get("/oauth-start", follow_redirects=False)
    assert "accounts.google.com" in resp.headers["location"]
    assert "drive.readonly" in resp.headers["location"]
    # The redirect must have armed a one-shot state for the callback.
    assert auth_routes._pending_state["value"]


def test_oauth2callback_error_renders_branded_page_not_json():
    """A cancelled consent used to be `{"status":"error","detail":...}` --
    it must now be a styled error page (still a 4xx status for monitoring)."""
    client = make_client()
    resp = client.get("/oauth2callback", params={"error": "access_denied"})
    assert resp.status_code == 400
    assert "text/html" in resp.headers["content-type"]
    assert "Chatty" in resp.text


def test_oauth2callback_state_mismatch_renders_branded_page():
    client = make_client()
    resp = client.get("/oauth2callback", params={"code": "abc", "state": "unknown"})
    assert resp.status_code == 403
    assert "text/html" in resp.headers["content-type"]
    assert '"status"' not in resp.text


def test_oauth2callback_rejects_unknown_state():
    client = make_client()
    resp = client.get("/oauth2callback", params={"code": "abc", "state": "unknown"})
    assert resp.status_code == 403


def test_oauth2callback_rejects_non_allowlisted_email_and_does_not_save_token(tmp_path):
    auth_routes._pending_state["value"] = "s1"
    client = make_client()

    with patch(
        "auth_routes.exchange_code_for_tokens",
        return_value={"access_token": "a", "refresh_token": "r"},
    ), patch("auth_routes.fetch_userinfo", return_value={"email": "attacker@example.com"}), \
       patch("auth_routes.DRIVE_CONFIG", {"token_store_path": str(tmp_path / "t.json")}), \
       patch("auth_routes.save_refresh_token") as mock_save:
        resp = client.get("/oauth2callback", params={"code": "abc", "state": "s1"})

    assert resp.status_code == 403
    mock_save.assert_not_called()


def test_oauth2callback_accepts_allowlisted_email_and_sets_session_cookie(tmp_path):
    auth_routes._pending_state["value"] = "s2"
    client = make_client()

    with patch(
        "auth_routes.exchange_code_for_tokens",
        return_value={"access_token": "a", "refresh_token": "r"},
    ), patch("auth_routes.fetch_userinfo", return_value={"email": "Allowed@Example.com"}), \
       patch("auth_routes.DRIVE_CONFIG", {"token_store_path": str(tmp_path / "t.json")}), \
       patch("auth_routes.save_refresh_token") as mock_save:
        resp = client.get("/oauth2callback", params={"code": "abc", "state": "s2"}, follow_redirects=False)

    assert resp.status_code in (302, 307)
    assert SESSION_COOKIE in resp.cookies
    mock_save.assert_called_once_with(str(tmp_path / "t.json"), "r")


def test_logout_clears_the_session_cookie():
    client = make_client()
    token = create_session_token("allowed@example.com", "test-secret-that-is-long-enough-1234567890", 3600)
    client.cookies.set(SESSION_COOKIE, token)

    resp = client.get("/logout", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/login"

    set_cookie = resp.headers["set-cookie"]
    assert f'{SESSION_COOKIE}=""' in set_cookie
    assert "Max-Age=0" in set_cookie
