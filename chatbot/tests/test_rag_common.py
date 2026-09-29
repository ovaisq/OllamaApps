from unittest.mock import MagicMock

import pytest

from rag_common import (
    chunk_hash,
    normalize_text,
    safe_error_message,
    validate_message,
    with_retries,
)


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  a\n\nb\t c  ") == "a b c"


def test_chunk_hash_is_stable_and_content_sensitive():
    assert chunk_hash("abc") == chunk_hash("abc")
    assert chunk_hash("abc") != chunk_hash("abd")


def test_validate_message_rejects_empty():
    with pytest.raises(ValueError):
        validate_message("   ", max_length=100)


def test_validate_message_rejects_oversized():
    with pytest.raises(ValueError, match="exceeds"):
        validate_message("x" * 10, max_length=5)


def test_validate_message_accepts_valid_message():
    assert validate_message("hello", max_length=100) == "hello"


def test_with_retries_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert with_retries(flaky, attempts=3, backoff_seconds=0) == "ok"
    assert calls["n"] == 3


def test_with_retries_raises_after_exhausting_attempts():
    def always_fails():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        with_retries(always_fails, attempts=2, backoff_seconds=0)


def test_safe_error_message_never_leaks_exception_text():
    logger = MagicMock()
    msg = safe_error_message(Exception("db-password=hunter2"), logger)

    assert "hunter2" not in msg
    assert "error id" in msg
    logger.error.assert_called_once()
