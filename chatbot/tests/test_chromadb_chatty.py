from unittest.mock import MagicMock, patch

import pytest

import chromadb_chatty


@pytest.fixture
def chat():
    mock_ollama_client = MagicMock()
    with patch("chromadb_chatty.build_ollama_client", return_value=mock_ollama_client), \
         patch("chromadb_chatty.chromadb.PersistentClient") as mock_chroma_cls:
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

    assert result == [("relevant chunk", "some_doc.md")]
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

    assert result == [("close match", "doc_a.md")]


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
