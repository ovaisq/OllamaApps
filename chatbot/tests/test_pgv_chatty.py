from unittest.mock import MagicMock, patch

import pytest
from pgvector import Vector

import pgv_chatty


def make_fake_pool(fetchall_return=None):
    """Build a fake connection pool whose cursor().execute/fetchall are inspectable."""
    cursor = MagicMock()
    cursor.fetchall.return_value = fetchall_return or []
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
    with patch("pgv_chatty.ThreadedConnectionPool", return_value=pool), \
         patch("pgv_chatty.register_vector"):
        instance = pgv_chatty.PGVectorChat()
    instance.ollama_client = MagicMock()
    return instance, cursor


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


def test_respond_rejects_oversized_message_without_calling_llm(chat):
    instance, _cursor = chat
    long_message = "x" * 100000

    outputs = list(instance.respond(long_message, [], {}))

    instance.ollama_client.chat.assert_not_called()
    final_history = outputs[-1][0]
    assert "exceeds" in final_history[-1]["content"]


def test_respond_hides_internal_error_details_from_user(chat):
    instance, _cursor = chat
    instance.ollama_client.embeddings.side_effect = ConnectionError("db-password=hunter2 leaked")

    outputs = list(instance.respond("hello", [], {}))

    final_message = outputs[-1][0][-1]["content"]
    assert "hunter2" not in final_message
    assert "error id" in final_message


def test_stop_chat_only_stops_its_own_session(chat):
    instance, cursor = chat
    cursor.fetchall.return_value = [("chunk", "doc.md", 0.1)]
    instance.ollama_client.embeddings.return_value = {"embedding": [0.1]}
    instance.ollama_client.chat.return_value = iter(
        [{"message": {"content": "hi"}}, {"message": {"content": " there"}}]
    )

    _, _, state_a = instance.stop_chat([], {})

    # session A's stop event must not leak into a fresh session B
    assert state_a["stop_event"].is_set()

    outputs = list(instance.respond("hello", [], {}))
    assert outputs[-1][0][-1]["content"] == "hi there"
