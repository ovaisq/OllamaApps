import time
from unittest.mock import MagicMock

import jwt

from app_session import SESSION_COOKIE, create_session_token, get_email_from_request, verify_session_token

SECRET = "secret-that-is-long-enough-1234567890"


def test_round_trips_email():
    token = create_session_token("user@example.com", "secret-that-is-long-enough-1234567890", max_age_seconds=3600)
    assert verify_session_token(token, "secret-that-is-long-enough-1234567890") == "user@example.com"


def test_rejects_token_signed_with_a_different_secret():
    token = create_session_token("user@example.com", "secret-that-is-long-enough-1234567890", max_age_seconds=3600)
    assert verify_session_token(token, "wrong-secret") is None


def test_rejects_expired_token():
    expired_payload = {
        "email": "user@example.com",
        "iat": int(time.time()) - 100,
        "exp": int(time.time()) - 1,
    }
    expired_token = jwt.encode(expired_payload, "secret-that-is-long-enough-1234567890", algorithm="HS256")
    assert verify_session_token(expired_token, "secret-that-is-long-enough-1234567890") is None


def test_rejects_none_token():
    assert verify_session_token(None, "secret-that-is-long-enough-1234567890") is None


def test_rejects_garbage_token():
    assert verify_session_token("not-a-jwt", "secret-that-is-long-enough-1234567890") is None


def test_get_email_from_request_reads_dict_like_cookies():
    token = create_session_token("user@example.com", SECRET, 3600)
    request = MagicMock(cookies={SESSION_COOKIE: token})

    assert get_email_from_request(request, SECRET) == "user@example.com"


def test_get_email_from_request_reads_attribute_style_cookies():
    """gr.Request wraps cookies in an Obj (attribute access), not always a
    plain dict, depending on environment -- must support both.
    """
    token = create_session_token("user@example.com", SECRET, 3600)

    class CookieObj:
        pass

    cookies = CookieObj()
    setattr(cookies, SESSION_COOKIE, token)
    request = MagicMock(cookies=cookies)

    assert get_email_from_request(request, SECRET) == "user@example.com"


def test_get_email_from_request_returns_none_for_no_request():
    assert get_email_from_request(None, SECRET) is None


def test_get_email_from_request_returns_none_when_cookie_missing():
    request = MagicMock(cookies={})
    assert get_email_from_request(request, SECRET) is None
