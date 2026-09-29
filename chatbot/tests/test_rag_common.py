from unittest.mock import MagicMock

import pytest

from rag_common import (
    build_ollama_client,
    chunk_hash,
    create_chunks,
    extract_text_from_upload,
    normalize_text,
    safe_error_message,
    validate_message,
    with_retries,
)


def test_build_ollama_client_uses_short_connect_and_long_read_timeout():
    """Regression test: a single shared float timeout caused chat streaming
    on a large/cold-loaded model to time out (30s wasn't enough), so the
    connect and read timeouts must be split.
    """
    client = build_ollama_client("http://example.com:11434", 300.0)
    timeout = client._client.timeout
    assert timeout.connect == 10.0
    assert timeout.read == 300.0


def test_create_chunks_actually_splits_text():
    """Regression test: this hits the real langchain_text_splitters import
    (no mocking) so a broken/renamed import surfaces here instead of only in
    production, where every other test patches create_chunks out entirely.
    """
    text = "hello world. " * 200
    chunks = create_chunks(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and c for c in chunks)


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


def test_extract_text_from_upload_reads_markdown(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# hello")
    assert extract_text_from_upload(str(f)) == "# hello"


def test_extract_text_from_upload_reads_txt(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("plain text")
    assert extract_text_from_upload(str(f)) == "plain text"


def test_extract_text_from_upload_returns_none_for_unsupported_extension(tmp_path):
    f = tmp_path / "notes.docx"
    f.write_text("hi")
    assert extract_text_from_upload(str(f)) is None
