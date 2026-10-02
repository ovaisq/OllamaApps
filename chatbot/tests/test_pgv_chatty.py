"""Tests for the pgvector backend class (PGVectorChat).

The chat-turn *orchestration* (validate -> prepare -> stream -> persist, plus
error/stopped classification) is backend-agnostic and lives in
api_routes.chat_stream_events -- it is tested once in test_api_routes.py,
not duplicated here. What remains class-specific and is tested here: the
SQL retrieval logic, prepare_turn (RAG prompt + structured sources payload),
stream_answer (Ollama call params + empty-first-chunk handling), and the
Postgres persistence layer.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from pgvector import Vector

import pgv_chatty


def make_fake_pool(fetchall_return=None):
    """Build a fake connection pool whose cursor().execute/fetchall are inspectable."""
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return or []
    # PGVectorChat.__init__ runs pgv_schema's dimension check, which reads
    # atttypmod; report a matching dimension so the fixture exercises the
    # no-migration path. (pgvector stores the raw dim in atttypmod.)
    cursor.fetchone.return_value = (pgv_chatty.OLLAMA_CONFIG["embedding_dim"],)
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False

    conn = MagicMock()
    conn.cursor.return_value = cursor

    pool = MagicMock()
    pool.getconn.return_value = conn
    return pool, conn, cursor


@pytest.fixture
def chat():
    pool, conn, cursor = make_fake_pool()
    # build_ollama_client is patched (like the chroma fixture) so __init__'s
    # model prewarm hits a mock instead of the real Ollama server.
    with patch("pgv_chatty.ThreadedConnectionPool", return_value=pool), \
         patch("pgv_chatty.register_vector"), \
         patch("pgv_chatty.build_ollama_client", return_value=MagicMock()):
        instance = pgv_chatty.PGVectorChat()
    instance.ollama_client = MagicMock()
    return instance, cursor


# ----------------------------------------------------------------- retrieval


def test_get_context_chunks_uses_vector_similarity_search(chat):
    """Regression test: retrieval must rank by embedding distance, not chunk length."""
    instance, cursor = chat
    cursor.fetchall.return_value = [("relevant chunk", "some_doc.md", 0.12)]
    instance.ollama_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

    result = instance.get_context_chunks("what is pgvector?")

    assert result == [("relevant chunk", "some_doc.md")]
    instance.ollama_client.embeddings.assert_called_once()
    sql, params = cursor.execute.call_args.args
    assert "<->" in sql
    assert "ORDER BY" in sql
    # Must be a pgvector.Vector, not a plain list: psycopg2 has no adapter
    # for plain lists, so a bare list gets sent as a numeric[] literal and
    # `vector <-> numeric[]` fails in Postgres with UndefinedFunction.
    assert isinstance(params[0], Vector)
    assert params[0].to_list() == pytest.approx([0.1, 0.2, 0.3])


def test_get_context_chunks_filters_out_irrelevant_matches_when_threshold_set(chat):
    """A far-away nearest neighbor (e.g. asking "python" and getting a job
    posting chunk back) should be dropped when max_context_distance is set,
    rather than fed to the model as if it were relevant.
    """
    instance, cursor = chat
    cursor.fetchall.return_value = [
        ("close match", "doc_a.md", 0.1),
        ("far, barely-related match", "doc_b.md", 5.0),
    ]
    instance.ollama_client.embeddings.return_value = {"embedding": [0.1]}

    with patch("pgv_chatty.CHAT_CONFIG", {**pgv_chatty.CHAT_CONFIG, "max_context_distance": 1.0}):
        result = instance.get_context_chunks("query")

    assert result == [("close match", "doc_a.md")]


def test_get_context_chunks_does_not_order_by_length(chat):
    """The old (broken) implementation ordered by LENGTH(chunk); make sure it's gone."""
    instance, cursor = chat
    instance.ollama_client.embeddings.return_value = {"embedding": [0.1]}

    instance.get_context_chunks("query")

    sql = cursor.execute.call_args.args[0]
    assert "LENGTH(chunk)" not in sql


def test_embedding_call_keeps_model_loaded(chat):
    """Embedding calls must pass keep_alive=-1 too -- without it each call
    resets the embedding model's unload timer to the server's 5-minute
    default, evicting a keep-forever instance after 5 idle minutes."""
    instance, cursor = chat
    instance.ollama_client.embeddings.return_value = {"embedding": [0.1]}

    instance.get_context_chunks("query")

    assert instance.ollama_client.embeddings.call_args.kwargs["keep_alive"] == -1


def test_named_documents_are_routed_to_the_front_of_retrieval(chat):
    """A document the user names explicitly must reach the model's context
    even when dense numeric chunks would otherwise own the top-k (the
    'Kona April 2026' failure: real content chunks existed in the index
    but never got retrieved)."""
    instance, cursor = chat
    instance.ollama_client.embeddings.return_value = {"embedding": [0.1]}
    cursor.fetchall.side_effect = [
        [("Kona April 2026",), ("bq-results-20231216",)],
        [("Kona content chunk", {"source": "Kona April 2026"}, 0.5)],
    ]

    result = instance.get_context_chunks("What is the Kona April 2026 document about?")

    assert result == [("Kona content chunk", {"source": "Kona April 2026"})]
    sql, params = cursor.execute.call_args.args
    assert "ANY(" in sql
    assert params[1] == ["Kona April 2026"]


# -------------------------------------------------------------- prepare_turn


def test_prepare_turn_builds_content_sources_payload(chat):
    """A content question retrieves chunks and returns a structured Sources
    payload (the documents that actually fed the model) alongside the prompt
    messages -- the web UI renders those as the under-answer footer."""
    instance, cursor = chat
    instance.ollama_client.embeddings.return_value = {"embedding": [0.1]}
    cursor.fetchall.return_value = [("chunk", {"source": "doc.md"}, 0.1)]

    _messages, sources = instance.prepare_turn("hello", [])

    assert sources["kind"] == "content"
    assert "doc.md" in sources["labels"]
    assert sources["total"] == 1


def test_prepare_turn_routes_catalog_queries_away_from_embedding_search(chat):
    """"Show me all documents shared by Jen" must not trigger an embedding
    call / vector search -- it's a metadata enumeration, not a content
    question, and top-k similarity search can't answer it completely. The
    catalog payload carries the matching documents."""
    instance, cursor = chat
    cursor.fetchall.return_value = [("Engineering Reports", "Jane", "Jen", "application/pdf")]

    messages, sources = instance.prepare_turn("Show me all documents shared by Jen", [])

    instance.ollama_client.embeddings.assert_not_called()
    assert sources["kind"] == "catalog"
    assert any("Engineering Reports" in label for label in sources["labels"])
    assert sources["documents"] == 1
    assert "Engineering Reports" in messages[1]["content"]


# ------------------------------------------------------------- stream_answer


def test_chat_uses_configured_context_window_and_keep_alive(chat):
    """Chat must send the configured num_ctx -- a smaller window (the old
    hardcoded 8192) forces a full reload of a model running with 256K --
    and keep_alive=-1 so no request resets the unload timer."""
    instance, _cursor = chat
    instance.ollama_client.chat.return_value = iter([{"message": {"content": "hi"}}])

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    list(instance.stream_answer(messages, MagicMock(is_set=lambda: False)))

    kwargs = instance.ollama_client.chat.call_args.kwargs
    assert kwargs["options"]["num_ctx"] == pgv_chatty.OLLAMA_CONFIG["num_ctx"]
    assert kwargs["options"]["num_ctx"] == 262144
    assert kwargs["keep_alive"] == -1


def test_stream_answer_skips_empty_first_chunk(chat):
    """Ollama's first chunk is role-only (empty content). The stream must
    yield only real text, so the client's typing indicator stays up until
    the first real characters arrive (not an empty bubble)."""
    instance, _cursor = chat
    instance.ollama_client.chat.return_value = iter([
        {"message": {"content": ""}},
        {"message": {"content": "Hello"}},
        {"message": {"content": " world"}},
    ])

    messages = [{"role": "user", "content": "hi"}]
    parts = list(instance.stream_answer(messages, MagicMock(is_set=lambda: False)))

    assert parts == ["Hello", "Hello world"]


# --------------------------------------------------------------- construction


def test_construction_prewarms_only_models_not_already_running():
    """Startup prewarm must check /api/ps first: an already-running model
    with a sufficient context window is left untouched (a redundant load
    request with different options would force a reload), while a
    not-running one is preloaded with keep_alive so it stays resident."""
    pool, conn, cursor = make_fake_pool()
    mock_ollama = MagicMock()
    chat_running = MagicMock(
        model=pgv_chatty.OLLAMA_CONFIG["chat_model"],
        context_length=pgv_chatty.OLLAMA_CONFIG["num_ctx"],
    )
    mock_ollama.ps.return_value.models = [chat_running]

    with patch("pgv_chatty.ThreadedConnectionPool", return_value=pool), \
         patch("pgv_chatty.register_vector"), \
         patch("pgv_chatty.build_ollama_client", return_value=mock_ollama):
        pgv_chatty.PGVectorChat()

    loaded_models = [c.kwargs["model"] for c in mock_ollama.generate.call_args_list]
    assert loaded_models == [pgv_chatty.OLLAMA_CONFIG["embedding_model"]]
    load_call = mock_ollama.generate.call_args.kwargs
    assert load_call["keep_alive"] == pgv_chatty.OLLAMA_CONFIG["keep_alive"]
    # Embedding model gets its own default window, not the chat model's.
    assert load_call["options"] is None


# ---------------------------------------------------------------- list_docs


def test_list_documents_filters_by_person_across_owner_and_shared_by(chat):
    instance, cursor = chat
    cursor.fetchall.return_value = [("Engineering Reports", "Jane", None, "application/pdf")]

    instance.list_documents(person="Jen")

    sql, params = cursor.execute.call_args.args
    assert "ILIKE" in sql
    assert params == ["%Jen%", "%Jen%"]


def test_list_documents_filters_by_category_mime_types(chat):
    instance, cursor = chat
    cursor.fetchall.return_value = []

    instance.list_documents(category="pdf")

    sql, params = cursor.execute.call_args.args
    assert "mime_type" in sql
    assert params == [["application/pdf"]]


def test_list_documents_returns_structured_dicts(chat):
    instance, cursor = chat
    cursor.fetchall.return_value = [("Engineering Reports", "Jane", "John", "application/pdf")]

    docs = instance.list_documents()

    assert docs == [{
        "source": "Engineering Reports", "owner": "Jane",
        "shared_by": "John", "mime_type": "application/pdf",
    }]


# ------------------------------------------------------------------- history


def test_save_message_inserts_row(chat):
    instance, cursor = chat
    instance.save_message("user@example.com", "user", "hello")

    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO chat_history" in sql
    assert params == ("user@example.com", "user", "hello")


def test_load_history_returns_role_content_and_iso_created_at(chat):
    """The web UI stamps a time on every message, so load_history must pass
    created_at through, normalized to ISO-8601 UTC (psycopg2 hands back
    aware datetimes for timestamptz)."""
    instance, cursor = chat
    ts = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    cursor.fetchall.return_value = [("user", "hi", ts), ("assistant", "hello there", ts)]

    history = instance.load_history("user@example.com")

    assert history == [
        {"role": "user", "content": "hi", "created_at": "2026-01-02T03:04:05+00:00"},
        {"role": "assistant", "content": "hello there", "created_at": "2026-01-02T03:04:05+00:00"},
    ]


def test_load_history_treats_naive_timestamps_as_utc(chat):
    instance, cursor = chat
    cursor.fetchall.return_value = [("user", "hi", datetime(2026, 1, 2, 3, 4, 5))]

    history = instance.load_history("user@example.com")

    assert history[0]["created_at"] == "2026-01-02T03:04:05+00:00"


def test_clear_history_deletes_rows_for_user(chat):
    instance, cursor = chat
    instance.clear_history("user@example.com")

    sql, params = cursor.execute.call_args.args
    assert "DELETE FROM chat_history" in sql
    assert params == ("user@example.com",)


# ----------------------------------------------------------------- feedback


def test_record_feedback_inserts_question_answer_and_rating(chat):
    """The web UI sends the thread's question + answer + rating directly, so
    record_feedback must persist exactly that (no index lookups)."""
    instance, cursor = chat
    instance.record_feedback("user@example.com", "the question", "the wrong answer", "dislike")

    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO feedback" in sql
    assert params == ("user@example.com", "the question", "the wrong answer", "dislike")


def test_record_feedback_never_raises(chat):
    """A feedback click must never break the chat, even if the DB is down."""
    instance, _cursor = chat
    with patch.object(instance, "_connection", side_effect=RuntimeError("db down")):
        instance.record_feedback("u@x.com", "q", "a", "dislike")


def test_load_feedback_returns_recent_dislikes_for_user(chat):
    """The Library view's correction prefill lists this user's dislikes only."""
    instance, cursor = chat
    cursor.fetchall.return_value = [(7, "what host?", "wrong", "2026-01-02 03:04")]

    rows = instance.load_feedback("user@example.com")

    sql, params = cursor.execute.call_args.args
    assert "rating = 'dislike'" in sql
    assert params == ("user@example.com", 20)
    assert rows == [{"id": 7, "question": "what host?", "answer": "wrong",
                     "created": "2026-01-02 03:04"}]


# --------------------------------------------------------------------- index


def test_index_summary_reports_counts_and_top_sources(chat):
    instance, cursor = chat
    cursor.fetchone.return_value = (42,)
    cursor.fetchall.return_value = [("Engineering Reports", 3), ("notes.md", 1)]

    summary = instance.index_summary()

    assert summary["chunks"] == 42
    assert summary["documents"] == 2
    assert summary["top_sources"] == [("Engineering Reports", 3), ("notes.md", 1)]
    assert summary["last_sync"] is None


# --------------------------------------------------------------------- drive


def test_sync_drive_rejects_concurrent_triggers(chat):
    """A second Drive sync trigger while one is running must be rejected
    (-1) -- it would spawn a second concurrent embed worker pool re-embedding
    the same corpus (a sync storm)."""
    instance, cursor = chat
    assert instance._sync_gate.try_begin() is True  # simulate a running sync

    assert instance.sync_drive() == -1

    instance._sync_gate.finish()


def test_sync_drive_runs_when_idle_and_records_the_sync_time(chat):
    instance, cursor = chat
    with patch("gdrive_indexer.get_access_token", return_value="token"), \
         patch("gdrive_indexer.run_pgvector_backend", return_value=42) as run, \
         patch("pgv_chatty.record_drive_sync_timestamp") as record:
        assert instance.sync_drive() == 42

    run.assert_called_once_with("token")
    # A successful sync stamps the 'Drive synced 2 h ago' timestamp;
    # blocked syncs (-1 path, other test) and failures never do.
    record.assert_called_once()
