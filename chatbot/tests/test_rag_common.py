"""Unit tests for rag_common's Ollama prewarm helpers."""
from unittest.mock import MagicMock

import rag_common

CHAT = "qwen3.8:27b-mtp-bf16"
EMBED = "qwen3-embedding:0.6b"


def running(model, context_length=None):
    """Stand-in for an ollama ProcessResponse.Model /api/ps entry."""
    return MagicMock(model=model, context_length=context_length)


def test_skips_load_when_model_already_running_with_full_context():
    """Checking /api/ps before loading saves the load round trip and, more
    importantly, avoids touching an already-running model: a load request
    with different options would force a full reload of it."""
    client = MagicMock()
    client.ps.return_value.models = [running(CHAT, 262144)]

    assert rag_common.ensure_model_loaded(client, CHAT, keep_alive=-1, num_ctx=262144) is False
    client.generate.assert_not_called()


def test_loads_model_that_is_not_running():
    client = MagicMock()
    client.ps.return_value.models = []

    assert rag_common.ensure_model_loaded(client, CHAT, keep_alive=-1, num_ctx=262144) is True
    client.generate.assert_called_once_with(
        model=CHAT, keep_alive=-1, options={"num_ctx": 262144}
    )


def test_reloads_when_running_context_is_smaller_than_requested():
    """A model running with a 32K window can't serve a 256K request -- it
    must be reloaded, not reused."""
    client = MagicMock()
    client.ps.return_value.models = [running(CHAT, 32768)]

    assert rag_common.ensure_model_loaded(client, CHAT, keep_alive=-1, num_ctx=262144) is True
    client.generate.assert_called_once()


def test_treats_running_model_with_unreported_context_as_loaded():
    """If the server doesn't report the loaded window size, assume the
    running instance is fine -- blindly reloading could evict a
    keep-forever instance we can't verify."""
    client = MagicMock()
    client.ps.return_value.models = [running(CHAT, None)]

    assert rag_common.ensure_model_loaded(client, CHAT, keep_alive=-1, num_ctx=262144) is False
    client.generate.assert_not_called()


def test_skips_load_when_running_models_cannot_be_listed():
    """/api/ps failing means we can't know what's running -- don't risk
    disturbing it with a load request."""
    client = MagicMock()
    client.ps.side_effect = ConnectionError("server unreachable")

    assert rag_common.ensure_model_loaded(client, CHAT, keep_alive=-1) is False
    client.generate.assert_not_called()


def test_load_failure_is_swallowed_not_raised():
    """Prewarm must never take the app down -- a failed load is logged and
    the first request pays the cold-load cost instead."""
    client = MagicMock()
    client.ps.return_value.models = []
    client.generate.side_effect = ConnectionError("boom")

    assert rag_common.ensure_model_loaded(client, CHAT, keep_alive=-1) is False


def test_embedding_model_is_preloaded_without_an_explicit_context_window():
    """Embedding models must be preloaded with their own default window --
    inflating a 0.6b embedding model to the chat model's 256K context
    would allocate a huge KV cache for nothing."""
    client = MagicMock()
    client.ps.return_value.models = []

    rag_common.ensure_model_loaded(client, EMBED, keep_alive=-1)

    client.generate.assert_called_once_with(model=EMBED, keep_alive=-1, options=None)


def test_embed_text_passes_keep_alive_through():
    client = MagicMock()
    client.embeddings.return_value = {"embedding": [0.5]}

    rag_common.embed_text(client, "hello", EMBED, keep_alive=-1)

    client.embeddings.assert_called_once_with(model=EMBED, prompt="hello", keep_alive=-1)


def test_embed_text_keep_alive_defaults_to_server_default():
    """Omitting keep_alive must keep working for callers that don't care
    (None is dropped by the SDK, leaving the server default in place)."""
    client = MagicMock()
    client.embeddings.return_value = {"embedding": [0.5]}

    rag_common.embed_text(client, "hello", EMBED)

    client.embeddings.assert_called_once_with(model=EMBED, prompt="hello", keep_alive=None)


def test_sync_gate_allows_one_sync_at_a_time():
    """A trigger arriving while a sync is running must be rejected outright,
    not queued -- a pile of waiting clicks would still run back-to-back
    full-corpus syncs when the lock frees up."""
    gate = rag_common.DriveSyncGate(cooldown_seconds=0.0)

    assert gate.try_begin() is True
    assert gate.try_begin() is False
    gate.finish()
    assert gate.try_begin() is True


def test_sync_gate_cooldown_blocks_immediate_retrigger():
    """Rapid re-clicks right after a sync finishes are a sync storm too --
    the cooldown swallows them."""
    gate = rag_common.DriveSyncGate(cooldown_seconds=60.0)

    gate.try_begin()
    gate.finish()

    assert gate.try_begin() is False


def test_sync_gate_releases_after_failed_sync():
    """sync_drive calls finish() in a finally; a crashed sync must not wedge
    the gate forever."""
    gate = rag_common.DriveSyncGate(cooldown_seconds=0.0)

    gate.try_begin()
    gate.finish()

    assert gate.try_begin() is True


def test_system_prompt_treats_user_corrections_as_authoritative():
    """Corrections taught via the Admin tab only change answers if the model
    is told to treat [user-corrections] chunks as overriding documents."""
    assert "user-corrections" in rag_common.CHAT_SYSTEM_PROMPT
    assert "override" in rag_common.CHAT_SYSTEM_PROMPT


def test_chat_ui_js_renders_a_visible_elapsed_timer():
    """Gradio's status tracker cannot be the chat timer -- it hides itself
    as soon as an event starts streaming, and the typing-indicator yield
    makes the chat stream immediately (verified against the 6.28
    statustracker bundle; show_progress settings can't produce a visible
    chat timer). The UI JS must render its own: an elapsed counter next to
    the typing dots while the answer is pending."""
    assert "typing-timer" in rag_common.CHAT_UI_JS
    assert "typing-timer" in rag_common.TYPING_INDICATOR_CSS


def test_detect_mentioned_sources_routes_named_documents():
    sources = ["Kona April 2026", "Movie Data Starter Project", "R"]
    assert rag_common.detect_mentioned_sources(
        "What is the Kona April 2026 document about?", sources
    ) == ["Kona April 2026"]
    # Matching is case- and whitespace-insensitive.
    assert rag_common.detect_mentioned_sources("kona  APRIL 2026?", sources) == [
        "Kona April 2026"
    ]


def test_detect_mentioned_sources_matches_tokens_in_any_order():
    """Substring matching missed real questions: 'Show Ovais resume from
    2019' never contains '2019_RESUME.pdf' verbatim (different word
    order, no extension). Token matching must catch it -- while NOT
    routing to the other resume files it doesn't fully name."""
    sources = [
        "05_2023_OQ_Exec_Resume.pdf",
        "2019_RESUME.pdf",
        "Kona April 2026",
        "Teal Health Defect Report November 19, 2024.pdf",
    ]
    assert rag_common.detect_mentioned_sources(
        "Show Ovais resume from 2019", sources
    ) == ["2019_RESUME.pdf"]


def test_detect_mentioned_sources_requires_all_significant_tokens():
    """A query using only some of a document's name tokens must not
    hijack retrieval toward that document."""
    sources = ["Teal Health Defect Report November 19, 2024.pdf"]
    assert rag_common.detect_mentioned_sources(
        "what do the defect reports say about screening", sources
    ) == []


def test_detect_mentioned_sources_ignores_tiny_source_names():
    """A 1-char source like 'R' has no significant tokens and can never
    hijack retrieval."""
    assert rag_common.detect_mentioned_sources("what is R about?", ["R"]) == []


def test_is_low_information_skips_numeric_dumps_but_keeps_prose():
    """Spreadsheet cell dumps bury real documents in vector search and
    carry nothing for RAG answers; prose must never be flagged."""
    assert rag_common.is_low_information("2020,4,29,4.39,202 2020,4,30,3.85,113")
    assert not rag_common.is_low_information(
        "Kona does not like peeing in new environments, so may initially hold it in."
    )


# ---------------------------------------------------------------------------
# Word document extraction (.docx via stdlib zip+XML; .doc via antiword/catdoc)
# ---------------------------------------------------------------------------

def _make_docx_bytes(paragraphs):
    """Build a minimal but valid .docx (a zip with word/document.xml) in
    memory -- .docx is OOXML, so this is a real fixture, not a mock."""
    import io
    import zipfile

    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = "".join(
        f"<w:p><w:r><w:t xml:space=\"preserve\">{p}</w:t></w:r></w:p>"
        for p in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<w:document {ns}><w:body>{body}</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
        )
        zf.writestr("word/document.xml", document)
    return buf.getvalue()


def _make_docx_with_runs():
    """A .docx whose text is split across multiple <w:t> runs in one
    paragraph -- extractors must join runs, not just take the first."""
    import io
    import zipfile

    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f"<w:document {ns}><w:body>"
        "<w:p><w:r><w:t>hello </w:t></w:r><w:r><w:t>bolded</w:t></w:r>"
        "<w:r><w:t> world</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>second para</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", document)
    return buf.getvalue()


def test_extract_docx_text_reads_paragraphs():
    text = rag_common.extract_docx_text(_make_docx_bytes(["para one", "para two"]))
    assert text == "para one\npara two"


def test_extract_docx_text_joins_runs_within_a_paragraph():
    text = rag_common.extract_docx_text(_make_docx_with_runs())
    assert text == "hello bolded world\nsecond para"


def test_extract_docx_text_rejects_garbage_and_empty():
    assert rag_common.extract_docx_text(b"not a zip at all") is None
    assert rag_common.extract_docx_text(b"") is None


def test_extract_docx_text_empty_document_is_none():
    assert rag_common.extract_docx_text(_make_docx_bytes([])) is None


def _install_fake_tool(tmp_path, name, script):
    """Write an executable `name` into a temp bin dir and return it. The
    script must use absolute paths for any command it invokes, because the
    tests run with a stripped PATH (no /usr/bin)."""
    import shutil as _sh
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    cat = _sh.which("cat")  # resolved against the real PATH before we strip it
    fake = bin_dir / name
    fake.write_text("#!/bin/sh\n" + script.replace("{CAT}", cat))
    fake.chmod(0o755)
    return bin_dir


def test_extract_doc_text_shells_out_to_installed_tool(tmp_path, monkeypatch):
    """The .doc path hands the bytes to antiword/catdoc and returns their
    stdout (verified with a fake tool that just cats the file, so the test
    also proves the content reaches the tool intact)."""
    bin_dir = _install_fake_tool(tmp_path, "antiword", '{CAT} "$1"')
    monkeypatch.setenv("PATH", str(bin_dir))

    assert rag_common.extract_doc_text(b"legacy doc body") == "legacy doc body"


def test_extract_doc_text_prefers_antiword_over_catdoc(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "antiword").write_text('#!/bin/sh\nprintf "FROM-ANTIWORD\\n"\n')
    (bin_dir / "catdoc").write_text('#!/bin/sh\nprintf "FROM-CATDOC\\n"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))

    assert rag_common.extract_doc_text(b"whatever") == "FROM-ANTIWORD"


def test_extract_doc_text_returns_none_when_no_tool_installed(tmp_path, monkeypatch):
    """Headless envs without antiword/catdoc: files are skipped (None),
    never crash the upload/sync."""
    empty_bin = tmp_path / "empty"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    assert rag_common.extract_doc_text(b"whatever") is None


def test_extract_doc_text_returns_none_when_tool_fails(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "catdoc"
    fake.write_text('#!/bin/sh\nexit 1\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))

    assert rag_common.extract_doc_text(b"whatever") is None


def test_extract_text_from_upload_dispatches_word_files(tmp_path, monkeypatch):
    docx = tmp_path / "memo.docx"
    docx.write_bytes(_make_docx_bytes(["uploaded docx text"]))
    assert rag_common.extract_text_from_upload(str(docx)) == "uploaded docx text"

    bin_dir = _install_fake_tool(tmp_path, "antiword", '{CAT} "$1"')
    monkeypatch.setenv("PATH", str(bin_dir))
    doc = tmp_path / "legacy.doc"
    doc.write_bytes(b"legacy body")
    assert rag_common.extract_text_from_upload(str(doc)) == "legacy body"
