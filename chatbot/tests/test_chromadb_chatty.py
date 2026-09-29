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


def test_respond_rejects_oversized_message_without_calling_llm(chat):
    instance, _collection, ollama_client = chat
    outputs = list(instance.respond("x" * 100000, [], {}))

    ollama_client.chat.assert_not_called()
    assert "exceeds" in outputs[-1][0][-1]["content"]


def test_respond_hides_internal_error_details_from_user(chat):
    instance, _collection, ollama_client = chat
    ollama_client.embeddings.side_effect = ConnectionError("db-password=hunter2 leaked")

    outputs = list(instance.respond("hello", [], {}))

    final_message = outputs[-1][0][-1]["content"]
    assert "hunter2" not in final_message
    assert "error id" in final_message


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

    docs = instance.list_documents()
    assert len(docs) == 1


def test_get_answer_stream_routes_catalog_queries_away_from_embedding_search(chat):
    instance, collection, ollama_client = chat
    collection.get.return_value = {
        "metadatas": [{"source": "Engineering Reports", "owner": "Jane", "shared_by": "Jen"}]
    }
    ollama_client.chat.return_value = iter([{"message": {"content": "Found it"}}])

    list(instance.get_answer_stream("Show me all documents shared by Jen", [], MagicMock(is_set=lambda: False)))

    ollama_client.embeddings.assert_not_called()
    messages = ollama_client.chat.call_args.kwargs["messages"]
    assert "Engineering Reports" in messages[1]["content"]


def test_save_and_load_history_round_trips(chat):
    instance, _collection, _ollama_client = chat
    instance.save_message("user@example.com", "user", "hi")
    instance.save_message("user@example.com", "assistant", "hello there")
    instance.save_message("someone-else@example.com", "user", "not mine")

    history = instance.load_history("user@example.com")

    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]


def test_clear_history_only_clears_that_user(chat):
    instance, _collection, _ollama_client = chat
    instance.save_message("user@example.com", "user", "hi")
    instance.save_message("someone-else@example.com", "user", "keep me")

    instance.clear_history("user@example.com")

    assert instance.load_history("user@example.com") == []
    assert instance.load_history("someone-else@example.com") == [{"role": "user", "content": "keep me"}]


def test_respond_persists_messages_for_identified_user(chat):
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {
        "documents": [["chunk"]], "metadatas": [[{"source": "doc.md"}]], "distances": [[0.1]],
    }
    ollama_client.chat.return_value = iter([{"message": {"content": "hi there"}}])

    with patch("chromadb_chatty.get_email_from_request", return_value="user@example.com"):
        list(instance.respond("hello", [], {}, request=MagicMock()))

    assert instance.load_history("user@example.com") == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_respond_skips_persistence_when_user_not_identified(chat):
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {
        "documents": [["chunk"]], "metadatas": [[{"source": "doc.md"}]], "distances": [[0.1]],
    }
    ollama_client.chat.return_value = iter([{"message": {"content": "hi"}}])

    with patch("chromadb_chatty.get_email_from_request", return_value=None):
        list(instance.respond("hello", [], {}))

    assert instance.load_history("anyone@example.com") == []


def test_load_history_ui_returns_empty_for_unauthenticated_request(chat):
    instance, _collection, _ollama_client = chat
    with patch("chromadb_chatty.get_email_from_request", return_value=None):
        assert instance.load_history_ui(MagicMock()) == []


def test_clear_chat_ui_clears_persisted_history_for_identified_user(chat):
    instance, _collection, _ollama_client = chat
    instance.save_message("user@example.com", "user", "hi")

    with patch("chromadb_chatty.get_email_from_request", return_value="user@example.com"):
        result = instance.clear_chat_ui(MagicMock())

    assert result == ([], "", {})
    assert instance.load_history("user@example.com") == []


def test_stop_chat_only_stops_its_own_session(chat):
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {
        "documents": [["chunk"]],
        "metadatas": [[{"source": "doc.md"}]],
        "distances": [[0.1]],
    }
    ollama_client.chat.return_value = iter(
        [{"message": {"content": "hi"}}, {"message": {"content": " there"}}]
    )

    _, _, state_a = instance.stop_chat([], {})
    assert state_a["stop_event"].is_set()

    outputs = list(instance.respond("hello", [], {}))
    assert outputs[-1][0][-1]["content"] == "hi there"
