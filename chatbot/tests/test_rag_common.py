from unittest.mock import MagicMock

import pytest

from rag_common import (
    build_chat_messages,
    build_ollama_client,
    chunk_hash,
    create_chunks,
    extract_text_from_upload,
    extract_xlsx_text,
    format_context_chunks,
    normalize_text,
    safe_error_message,
    validate_message,
    with_retries,
)


def test_format_context_chunks_tags_each_chunk_with_its_source():
    text = format_context_chunks([("chunk one", "doc_a.md"), ("chunk two", "doc_b.pdf")])
    assert "[Source: doc_a.md]\nchunk one" in text
    assert "[Source: doc_b.pdf]\nchunk two" in text


def test_format_context_chunks_labels_missing_source():
    text = format_context_chunks([("orphan chunk", None)])
    assert "[Source: unknown source]" in text


def test_format_context_chunks_empty_says_nothing_found():
    assert format_context_chunks([]) == "(no relevant documents found)"


def test_build_chat_messages_splits_system_and_user_roles():
    """Regression test: instructions bundled into a single user-role
    message get much weaker instruction-following from chat-tuned models
    than a proper system message -- this is why the bot would answer
    general-knowledge questions (e.g. "python") from its own training data
    instead of grounding in (or admitting it found nothing in) the context.
    """
    messages = build_chat_messages([("some chunk", "doc.md")], "", "what is X?")

    assert messages[0]["role"] == "system"
    assert "Context (grouped" not in messages[0]["content"]  # data lives in the user message
    assert messages[1]["role"] == "user"
    assert "doc.md" in messages[1]["content"]
    assert "what is X?" in messages[1]["content"]


def test_build_chat_messages_tells_model_not_to_substitute_own_knowledge():
    messages = build_chat_messages([], "", "python?")
    system_content = messages[0]["content"]
    assert "own (possibly wrong) general knowledge" in system_content
    assert "(no relevant documents found)" in messages[1]["content"]


def _build_test_xlsx_bytes() -> bytes:
    import io

    from openpyxl import Workbook

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Reports"
    ws1.append(["Quarter", "Status"])
    ws1.append(["Q1", "On track"])
    ws2 = wb.create_sheet("Notes")
    ws2.append(["Engineering summary here"])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


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


def test_extract_xlsx_text_reads_all_sheets():
    """Regression test for the "Engineering Reports" incident: a Google
    Sheet/.xlsx full of real content was silently skipped everywhere because
    nothing extracted text from spreadsheets at all.
    """
    text = extract_xlsx_text(_build_test_xlsx_bytes())
    assert "Sheet: Reports" in text
    assert "Q1,On track" in text
    assert "Sheet: Notes" in text
    assert "Engineering summary here" in text


def test_extract_xlsx_text_returns_none_for_empty_bytes():
    assert extract_xlsx_text(b"") is None


def test_extract_xlsx_text_returns_none_for_garbage_bytes():
    assert extract_xlsx_text(b"not a real xlsx file") is None


def test_extract_text_from_upload_reads_xlsx(tmp_path):
    f = tmp_path / "report.xlsx"
    f.write_bytes(_build_test_xlsx_bytes())
    text = extract_text_from_upload(str(f))
    assert "Engineering summary here" in text
