import time

import jwt

from app_session import create_session_token, verify_session_token


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
