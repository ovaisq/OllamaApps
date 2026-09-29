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
    # no-migration path.
    cursor.fetchone.return_value = (pgv_chatty.OLLAMA_CONFIG["embedding_dim"] + 4,)
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


def test_get_answer_stream_routes_catalog_queries_away_from_embedding_search(chat):
    """"Show me all documents shared by Jen" must not trigger an embedding
    call / vector search -- it's a metadata enumeration, not a content
    question, and top-k similarity search can't answer it completely.
    """
    instance, cursor = chat
    cursor.fetchall.return_value = [("Engineering Reports", "Jane", "Jen", "application/pdf")]
    instance.ollama_client.chat.return_value = iter([{"message": {"content": "Found it"}}])

    list(instance.get_answer_stream("Show me all documents shared by Jen", [], MagicMock(is_set=lambda: False)))

    instance.ollama_client.embeddings.assert_not_called()
    messages = instance.ollama_client.chat.call_args.kwargs["messages"]
    assert "Engineering Reports" in messages[1]["content"]


def test_save_message_inserts_row(chat):
    instance, cursor = chat
    instance.save_message("user@example.com", "user", "hello")

    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO chat_history" in sql
    assert params == ("user@example.com", "user", "hello")


def test_load_history_returns_role_content_dicts(chat):
    instance, cursor = chat
    cursor.fetchall.return_value = [("user", "hi"), ("assistant", "hello there")]

    history = instance.load_history("user@example.com")

    assert history == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]


def test_clear_history_deletes_rows_for_user(chat):
    instance, cursor = chat
    instance.clear_history("user@example.com")

    sql, params = cursor.execute.call_args.args
    assert "DELETE FROM chat_history" in sql
    assert params == ("user@example.com",)


def test_respond_persists_messages_for_identified_user(chat):
    instance, _cursor = chat
    instance.ollama_client.embeddings.return_value = {"embedding": [0.1]}
    instance.ollama_client.chat.return_value = iter([{"message": {"content": "hi there"}}])

    with patch("pgv_chatty.get_email_from_request", return_value="user@example.com"), \
         patch.object(instance, "save_message") as mock_save:
        list(instance.respond("hello", [], {}, request=MagicMock()))

    mock_save.assert_any_call("user@example.com", "user", "hello")
    mock_save.assert_any_call("user@example.com", "assistant", "hi there")


def test_respond_skips_persistence_when_user_not_identified(chat):
    instance, _cursor = chat
    instance.ollama_client.embeddings.return_value = {"embedding": [0.1]}
    instance.ollama_client.chat.return_value = iter([{"message": {"content": "hi"}}])

    with patch("pgv_chatty.get_email_from_request", return_value=None), \
         patch.object(instance, "save_message") as mock_save:
        list(instance.respond("hello", [], {}))

    mock_save.assert_not_called()


def test_load_history_ui_returns_empty_for_unauthenticated_request(chat):
    instance, _cursor = chat
    with patch("pgv_chatty.get_email_from_request", return_value=None):
        assert instance.load_history_ui(MagicMock()) == []


def test_clear_chat_ui_clears_persisted_history_for_identified_user(chat):
    instance, _cursor = chat
    with patch("pgv_chatty.get_email_from_request", return_value="user@example.com"), \
         patch.object(instance, "clear_history") as mock_clear:
        result = instance.clear_chat_ui(MagicMock())

    mock_clear.assert_called_once_with("user@example.com")
    assert result == ([], "", {})


def test_stop_chat_only_stops_its_own_session(chat):
    instance, cursor = chat
    cursor.fetchall.return_value = [("chunk", {"source": "doc.md"}, 0.1)]
    instance.ollama_client.embeddings.return_value = {"embedding": [0.1]}
    instance.ollama_client.chat.return_value = iter(
        [{"message": {"content": "hi"}}, {"message": {"content": " there"}}]
    )

    _, _, state_a = instance.stop_chat([], {})

    # session A's stop event must not leak into a fresh session B
    assert state_a["stop_event"].is_set()

    outputs = list(instance.respond("hello", [], {}))
    assert outputs[-1][0][-1]["content"] == "hi there"
