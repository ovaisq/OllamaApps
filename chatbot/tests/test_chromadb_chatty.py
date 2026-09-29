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


def test_embedding_call_keeps_model_loaded(chat):
    """Embedding calls must pass keep_alive=-1 too -- without it each call
    resets the embedding model's unload timer to the server's 5-minute
    default, evicting a keep-forever instance after 5 idle minutes."""
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {"documents": [[]], "metadatas": [[]]}

    instance.retrieve_context("query")

    assert ollama_client.embeddings.call_args.kwargs["keep_alive"] == -1


def test_chat_uses_configured_context_window_and_keep_alive(chat):
    """Chat must send the configured num_ctx -- a smaller window (the old
    hardcoded 8192) forces a full reload of a model running with 256K --
    and keep_alive=-1 so no request resets the unload timer."""
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    ollama_client.chat.return_value = iter([{"message": {"content": "hi"}}])

    list(instance.get_answer_stream("hello", [], MagicMock(is_set=lambda: False)))

    kwargs = ollama_client.chat.call_args.kwargs
    assert kwargs["options"]["num_ctx"] == chromadb_chatty.OLLAMA_CONFIG["num_ctx"]
    assert kwargs["options"]["num_ctx"] == 262144
    assert kwargs["keep_alive"] == -1


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


def test_sync_drive_rejects_concurrent_triggers(chat):
    """A second Drive sync trigger while one is running must be rejected
    (-1) -- it would spawn a second concurrent embed worker pool re-embedding
    the same corpus (a sync storm)."""
    instance, _collection, _ollama_client = chat
    assert instance._sync_gate.try_begin() is True  # simulate a running sync

    assert instance.sync_drive() == -1

    instance._sync_gate.finish()


def test_sync_drive_runs_when_idle(chat):
    instance, _collection, _ollama_client = chat
    with patch("gdrive_indexer.get_access_token", return_value="token"), \
         patch("gdrive_indexer.run_chromadb_backend", return_value=42) as run:
        assert instance.sync_drive() == 42

    run.assert_called_once_with("token")


def test_scroll_follows_output_only_while_user_is_at_bottom(chat):
    """Gradio's built-in autoscroll yanks the view to the bottom on every
    streamed token inside its threshold gap -- users read it as "a pending
    answer blocks the scroll". The app must disable it and wire
    SMART_SCROLL_JS instead, which follows only while the user is already
    at the bottom."""
    instance, _collection, _ollama_client = chat
    app = chromadb_chatty.build_app(instance)

    mount = next(r for r in app.routes if type(r).__name__ == "Mount")
    config = mount.app.blocks.config
    chatbots = [c for c in config["components"] if c.get("type") == "chatbot"]
    assert chatbots, "no chatbot component in the built app"
    assert all(c["props"].get("autoscroll") is False for c in chatbots)
    assert any(
        "bubble-wrap" in (d.get("js") or "") for d in config["dependencies"]
    ), "SMART_SCROLL_JS is not wired to the load event"


def test_like_dislike_is_recorded_with_its_question(chat):
    """Like/Dislike on an answer must persist the answer AND the question it
    answered, into the real (tmp-dir) sqlite store."""
    instance, collection, _ollama_client = chat
    like_data = MagicMock(index=1, liked=False)
    history = [
        {"role": "user", "content": "the question"},
        {"role": "assistant", "content": "the wrong answer"},
    ]

    instance.record_feedback(history, like_data)

    import sqlite3
    with sqlite3.connect(instance._history_db_path) as conn:
        rows = conn.execute(
            "SELECT user_email, question, answer, rating FROM feedback"
        ).fetchall()
    assert rows == [("anonymous", "the question", "the wrong answer", "dislike")]


def test_record_feedback_never_raises(chat):
    """A Like/Dislike click must never break the chat, even with junk
    event data."""
    instance, _collection, _ollama_client = chat
    like_data = MagicMock()
    like_data.index = MagicMock(side_effect=TypeError("bad event data"))

    instance.record_feedback([{"role": "assistant", "content": "x"}], like_data)


def test_like_event_is_wired_to_feedback_recording(chat):
    instance, _collection, _ollama_client = chat
    app = chromadb_chatty.build_app(instance)

    mount = next(r for r in app.routes if type(r).__name__ == "Mount")
    config = mount.app.blocks.config
    assert any(
        any(isinstance(t, (list, tuple)) and t[1] == "like" for t in d.get("targets", []))
        for d in config["dependencies"]
    ), "the chatbot's like event is not wired to feedback recording"


def test_stream_keeps_typing_indicator_until_real_text(chat):
    """Ollama's first chunk is role-only (empty content). Yielding it
    replaced the typing dots with an empty bubble that sat on screen for
    the whole prefill window -- the confusing dots, then empty, then
    answer sequence users reported."""
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    ollama_client.chat.return_value = iter([
        {"message": {"content": ""}},
        {"message": {"content": "Hello"}},
        {"message": {"content": " world"}},
    ])

    parts = list(instance.get_answer_stream("hi", [], MagicMock(is_set=lambda: False)))

    assert parts == ["Hello", "Hello world"]


def test_stream_with_no_text_yields_placeholder_not_frozen_dots(chat):
    """If the stream never produces text (stopped early, or the model
    returned nothing), the typing dots must not be left frozen on
    screen."""
    instance, collection, ollama_client = chat
    ollama_client.embeddings.return_value = {"embedding": [0.1]}
    collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
    ollama_client.chat.return_value = iter([{"message": {"content": ""}}])

    outputs = list(instance.respond("hello", [], {}))

    assert "typing-indicator" in outputs[0][0][-1]["content"]
    assert "no response content" in outputs[-1][0][-1]["content"]


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


def test_progress_timer_shows_in_one_place(chat):
    """Gradio's default ('full' with no show_progress_on) renders the
    runtime timer on EVERY output component -- chat submit has two outputs
    (chatbot + textbox), so users saw two timers. 'minimal' hides the
    trackers entirely (zero timers, verified against the 6.28 statustracker
    bundle). One timer = 'full' scoped to a single component via
    show_progress_on; quick UI actions must be 'hidden'."""
    instance, collection, ollama_client = chat
    app = chromadb_chatty.build_app(instance)

    mount = next(r for r in app.routes if type(r).__name__ == "Mount")
    config = mount.app.blocks.config
    deps = config["dependencies"]
    chatbot_ids = {c["id"] for c in config["components"] if c.get("type") == "chatbot"}

    submit_dep = next(
        d for d in deps
        if any(isinstance(t, (list, tuple)) and t[1] == "submit" for t in d["targets"])
    )
    assert submit_dep["show_progress"] == "full"
    assert set(submit_dep["show_progress_on"]) == chatbot_ids, (
        "the chat timer must be scoped to the chatbot only, not every output"
    )
    like_dep = next(
        d for d in deps
        if any(isinstance(t, (list, tuple)) and t[1] == "like" for t in d["targets"])
    )
    assert like_dep["show_progress"] == "hidden"
    for d in deps:
        if d["show_progress"] == "full":
            assert len(d["show_progress_on"]) == 1, (
                "'full' without a single show_progress_on target renders "
                "one timer per output component (duplicates)"
            )


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
