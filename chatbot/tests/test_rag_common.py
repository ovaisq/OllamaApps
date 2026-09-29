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
