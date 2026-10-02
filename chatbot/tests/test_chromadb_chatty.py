"""Tests for the ChromaDB backend class (ChromaChat).

The chat-turn *orchestration* (validate -> prepare -> stream -> persist, plus
error/stopped classification) is backend-agnostic and lives in
api_routes.chat_stream_events -- it is tested once in test_api_routes.py,
not duplicated here. What remains class-specific and is tested here: the
retrieval logic, prepare_turn (RAG prompt + structured sources payload),
stream_answer (Ollama call params + empty-first-chunk handling), and the
Chroma/SQLite persistence layer.
"""
from unittest.mock import MagicMock, patch

import pytest

import chromadb_chatty


@pytest.fixture
def chat(tmp_path):
    mock_ollama_client = MagicMock()
    # Real (but tmp-dir, hermetic) sqlite chat-history file: exercising the
    # real file I/O here catches schema/query bugs that a mocked connection
    # would hide, without leaving stray chat_history.sqlite3 files around.
    with patch("chromadb_chatty.build_ollama_client", return_value=mock_ollama_client), \
         patch("chromadb_chatty.chromadb.PersistentClient") as mock_chroma_cls, \
         patch(
              "chromadb_chatty.CHROMA_CONFIG",
              {**chromadb_chatty.CHROMA_CONFIG, "db_path": str(tmp_path / "chroma_db")},
          ):
        mock_collection = MagicMock()
        mock_chroma_cls.return_value.get_collection.return_value = mock_collection
        instance = chromadb_chatty.ChromaChat()
    instance._stop_reload.set()  # stop the background reloader thread for the test
    return instance, mock_collection, mock_ollama_client


# ----------------------------------------------------------------- retrieval


def test_retrieve_context_embeds_query_before_similarity_search(chat):
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1, 0.2]}
    collection.query.return_value = {
        "documents": [["relevant chunk"]],
        "metadatas": [[{"source": "some_doc.md"}]],
        "distances": [[0.1]],
    }

    result = instance.retrieve_context("what is chromadb?")

    assert result == [("relevant chunk", {"source": "some_doc.md"})]
    ollama_client.embeddings.assert_called_once()
    _, kwargs = collection.query.call_args
    assert kwargs["query_embeddings"] == [[0.1, 0.2]]


def test_retrieve_context_filters_out_irrelevant_matches_when_threshold_set(chat):
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {
        "documents": [["close match", "far, barely-related match"]],
        "metadatas": [[{"source": "doc_a.md"}, {"source": "doc_b.md"}]],
        "distances": [[0.1, 5.0]],
    }

    with patch("chromadb_chatty.CHAT_CONFIG", {**chromadb_chatty.CHAT_CONFIG, "max_context_distance": 1.0}):
        result = instance.retrieve_context("query")

    assert result == [("close match", {"source": "doc_a.md"})]


def test_embedding_call_keeps_model_loaded(chat):
    """Embedding calls must pass keep_alive=-1 too -- without it each call
    resets the embedding model's unload timer to the server's 5-minute
    default, evicting a keep-forever instance after 5 idle minutes."""
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {"documents": [[]], "metadatas": [[]]}

    instance.retrieve_context("query")

    assert ollama_client.embeddings.call_args.kwargs["keep_alive"] == -1


def test_named_documents_are_routed_to_the_front_of_retrieval(chat):
    """A document the user names explicitly must lead the context even
    when global vector ranking would miss it."""
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.get.return_value = {
        "metadatas": [{"source": "Kona April 2026"}, {"source": "bq-results"}]
    }
    collection.query.side_effect = [
        {"documents": [["numeric row chunk"]],
         "metadatas": [[{"source": "bq-results"}]],
         "distances": [[0.9]]},
        {"documents": [["kona care chunk"]],
         "metadatas": [[{"source": "Kona April 2026"}]],
         "distances": [[0.5]]},
    ]

    result = instance.retrieve_context("What is the Kona April 2026 document about?")

    assert result[0] == ("kona care chunk", {"source": "Kona April 2026"})
    assert len(result) == 2
    routed_call = collection.query.call_args_list[1]
    assert routed_call.kwargs["where"] == {"source": {"$eq": "Kona April 2026"}}


# -------------------------------------------------------------- prepare_turn


def test_prepare_turn_builds_content_sources_payload(chat):
    """A content question retrieves chunks and returns a structured Sources
    payload (the documents that actually fed the model) alongside the prompt
    messages -- the web UI renders those as the under-answer footer."""
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {
        "documents": [["chunk"]], "metadatas": [[{"source": "doc.md"}]], "distances": [[0.1]],
    }

    _messages, sources = instance.prepare_turn("hello", [])

    assert sources["kind"] == "content"
    assert "doc.md" in sources["labels"]
    assert sources["total"] == 1


def test_prepare_turn_routes_catalog_queries_away_from_embedding_search(chat):
    """"Show me all documents shared by Jen" must not trigger an embedding
    call / vector search -- it's a metadata enumeration, not a content
    question, and top-k similarity search can't answer it completely. The
    catalog payload carries the matching documents."""
    instance, collection, ollama_client = chat
    collection.get.return_value = {
        "metadatas": [{"source": "Engineering Reports", "owner": "Jane", "shared_by": "Jen"}]
    }

    messages, sources = instance.prepare_turn("Show me all documents shared by Jen", [])

    ollama_client.embeddings.assert_not_called()
    assert sources["kind"] == "catalog"
    assert any("Engineering Reports" in label for label in sources["labels"])
    assert sources["documents"] == 1
    # The catalog text is what the model turns into a natural-language list.
    assert "Engineering Reports" in messages[1]["content"]


# ------------------------------------------------------------- stream_answer


def test_chat_uses_configured_context_window_and_keep_alive(chat):
    """Chat must send the configured num_ctx -- a smaller window (the old
    hardcoded 8192) forces a full reload of a model running with 256K --
    and keep_alive=-1 so no request resets the unload timer."""
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    ollama_client.chat.return_value = iter([{"message": {"content": "hi"}}])

    messages, _sources = instance.prepare_turn("hello", [])
    list(instance.stream_answer(messages, MagicMock(is_set=lambda: False)))

    kwargs = ollama_client.chat.call_args.kwargs
    assert kwargs["options"]["num_ctx"] == chromadb_chatty.OLLAMA_CONFIG["num_ctx"]
    assert kwargs["options"]["num_ctx"] == 262144
    assert kwargs["keep_alive"] == -1


def test_stream_answer_skips_empty_first_chunk(chat):
    """Ollama's first chunk is role-only (empty content). Yielding it
    replaced the typing dots with an empty bubble that sat on screen for the
    whole prefill window -- the confusing dots, then empty, then answer
    sequence users reported. The stream must yield only real text."""
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    ollama_client.chat.return_value = iter([
        {"message": {"content": ""}},
        {"message": {"content": "Hello"}},
        {"message": {"content": " world"}},
    ])

    messages, _sources = instance.prepare_turn("hi", [])
    parts = list(instance.stream_answer(messages, MagicMock(is_set=lambda: False)))

    assert parts == ["Hello", "Hello world"]


# --------------------------------------------------------------- construction


def test_construction_prewarms_only_models_not_already_running(tmp_path):
    """Startup prewarm must check /api/ps first: an already-running model
    with a sufficient context window is left untouched (a redundant load
    request with different options would force a reload), while a
    not-running one is preloaded with keep_alive so it stays resident."""
    mock_ollama = MagicMock()
    chat_running = MagicMock(
        model=chromadb_chatty.OLLAMA_CONFIG["chat_model"],
        context_length=chromadb_chatty.OLLAMA_CONFIG["num_ctx"],
    )
    mock_ollama.ps.return_value.models = [chat_running]

    with patch("chromadb_chatty.build_ollama_client", return_value=mock_ollama), \
         patch("chromadb_chatty.chromadb.PersistentClient") as mock_chroma_cls, \
         patch(
             "chromadb_chatty.CHROMA_CONFIG",
             {**chromadb_chatty.CHROMA_CONFIG, "db_path": str(tmp_path / "chroma_db")},
          ):
        mock_chroma_cls.return_value.get_collection.return_value = MagicMock()
        instance = chromadb_chatty.ChromaChat()
    instance._stop_reload.set()

    loaded_models = [c.kwargs["model"] for c in mock_ollama.generate.call_args_list]
    assert loaded_models == [chromadb_chatty.OLLAMA_CONFIG["embedding_model"]]
    load_call = mock_ollama.generate.call_args.kwargs
    assert load_call["keep_alive"] == chromadb_chatty.OLLAMA_CONFIG["keep_alive"]
    # Embedding model gets its own default window, not the chat model's.
    assert load_call["options"] is None


# ---------------------------------------------------------------- list_docs


def test_list_documents_filters_by_person_across_owner_and_shared_by(chat):
    instance, collection, _ollama_client = chat
    collection.get.return_value = {
        "metadatas": [
            {"source": "Engineering Reports", "owner": "Jane", "shared_by": "Jennifer"},
            {"source": "Unrelated Doc", "owner": "Bob"},
        ]
    }

    docs = instance.list_documents(person="Jen")

    assert docs == [{
        "source": "Engineering Reports", "owner": "Jane",
        "shared_by": "Jennifer", "mime_type": None,
    }]


def test_list_documents_filters_by_category(chat):
    instance, collection, _ollama_client = chat
    collection.get.return_value = {
        "metadatas": [
            {"source": "report.pdf", "mime_type": "application/pdf"},
            {"source": "notes.md", "mime_type": "text/markdown"},
        ]
    }

    docs = instance.list_documents(category="pdf")
    assert [d["source"] for d in docs] == ["report.pdf"]


def test_list_documents_dedupes_by_source(chat):
    instance, collection, _ollama_client = chat
    collection.get.return_value = {
        "metadatas": [
            {"source": "report.pdf", "mime_type": "application/pdf"},
            {"source": "report.pdf", "mime_type": "application/pdf"},
        ]
    }
    assert len(instance.list_documents()) == 1


# ------------------------------------------------------------------- history


def test_save_and_load_history_round_trips_including_created_at(chat):
    """The web UI shows a timestamp per message, so load_history must pass
    created_at through (normalized to ISO-8601 UTC) in addition to
    role/content."""
    instance, _collection, _ollama_client = chat
    instance.save_message("user@example.com", "user", "hi")
    instance.save_message("user@example.com", "assistant", "hello there")
    instance.save_message("someone-else@example.com", "user", "not mine")

    history = instance.load_history("user@example.com")

    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "hi"), ("assistant", "hello there"),
    ]
    assert all(m["created_at"] for m in history)


def test_clear_history_only_clears_that_user(chat):
    instance, _collection, _ollama_client = chat
    instance.save_message("user@example.com", "user", "hi")
    instance.save_message("someone-else@example.com", "user", "keep me")

    instance.clear_history("user@example.com")

    assert instance.load_history("user@example.com") == []
    assert [(m["role"], m["content"]) for m in instance.load_history("someone-else@example.com")] == [
        ("user", "keep me"),
    ]


# ----------------------------------------------------------------- feedback


def test_record_feedback_inserts_question_answer_and_rating(chat):
    """Like/Dislike must persist the answer AND the question it answered
    (the web UI sends both, so no index lookups) into the real (tmp-dir)
    sqlite store."""
    instance, _collection, _ollama_client = chat
    instance.record_feedback("user@example.com", "the question", "the wrong answer", "dislike")

    import sqlite3
    with sqlite3.connect(instance._history_db_path) as conn:
        rows = conn.execute(
            "SELECT user_email, question, answer, rating FROM feedback"
        ).fetchall()
    assert rows == [("user@example.com", "the question", "the wrong answer", "dislike")]


def test_record_feedback_never_raises(chat):
    """A feedback click must never break the chat, even if the DB is down."""
    instance, _collection, _ollama_client = chat
    with patch.object(instance, "_history_connection", side_effect=RuntimeError("db down")):
        instance.record_feedback("u@x.com", "q", "a", "dislike")


def test_load_feedback_returns_only_this_users_dislikes(chat):
    """The Library view's correction prefill lists dislikes only (likes are
    fine, they're not things to correct)."""
    instance, _collection, _ollama_client = chat
    instance.record_feedback("anonymous", "the question", "the wrong answer", "dislike")
    instance.record_feedback("anonymous", "the good question", "right answer", "like")

    rows = instance.load_feedback("anonymous")

    assert [r["question"] for r in rows] == ["the question"]
    assert rows[0]["answer"] == "the wrong answer"


# --------------------------------------------------------------------- index


def test_index_summary_counts_documents_and_top_sources(chat):
    instance, collection, _ollama_client = chat
    collection.get.return_value = {
        "metadatas": [{"source": "a.md"}, {"source": "a.md"}, {"source": "b.pdf"}],
    }
    collection.count.return_value = 12

    summary = instance.index_summary()

    assert summary["chunks"] == 12
    assert summary["documents"] == 2
    assert summary["top_sources"] == [("a.md", 2), ("b.pdf", 1)]
    assert summary["last_sync"] is None


# --------------------------------------------------------------------- drive


def test_sync_drive_rejects_concurrent_triggers(chat):
    """A second Drive sync trigger while one is running must be rejected
    (-1) -- it would spawn a second concurrent embed worker pool re-embedding
    the same corpus (a sync storm)."""
    instance, _collection, _ollama_client = chat
    assert instance._sync_gate.try_begin() is True  # simulate a running sync

    assert instance.sync_drive() == -1

    instance._sync_gate.finish()


def test_sync_drive_runs_when_idle_and_records_the_sync_time(chat):
    instance, _collection, _ollama_client = chat
    with patch("gdrive_indexer.get_access_token", return_value="token"), \
         patch("gdrive_indexer.run_chromadb_backend", return_value=42) as run, \
         patch("chromadb_chatty.record_drive_sync_timestamp") as record:
        assert instance.sync_drive() == 42

    run.assert_called_once_with("token")
    # A successful sync stamps the 'Drive synced 2 h ago' timestamp;
    # blocked syncs (-1 path, other test) and failures never do.
    record.assert_called_once()
