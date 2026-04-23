"""Tests for middleware-persisted assistant-node metadata.

Exercises the opt-in ``_persist`` namespace merge in
``ConversationManager._finalize_completion`` (Change A) and the
``PromoteToPersistMiddleware`` adapter (Change B).

The fixture pattern mirrors ``test_scoped_middleware.py`` — real
``ConversationMiddleware`` subclasses, ``EchoProvider``,
``FileSystemPromptLibrary``, and ``DataknobsConversationStorage`` over
``AsyncMemoryDatabase``. No mocks.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, List

import pytest
import yaml

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_llm.conversations import (
    ConversationManager,
    ConversationMiddleware,
    DataknobsConversationStorage,
    PromoteToPersistMiddleware,
)
from dataknobs_llm.conversations.storage import ConversationState
from dataknobs_llm.llm import EchoProvider, LLMConfig, LLMMessage, LLMResponse
from dataknobs_llm.prompts import AsyncPromptBuilder, FileSystemPromptLibrary


# ---------------------------------------------------------------------------
# Test middleware helpers
# ---------------------------------------------------------------------------


class PersistingMiddleware(ConversationMiddleware):
    """Writes a payload under ``response.metadata["_persist"]``."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def process_request(
        self, messages: List[LLMMessage], state: ConversationState
    ) -> List[LLMMessage]:
        return messages

    async def process_response(
        self, response: LLMResponse, state: ConversationState
    ) -> LLMResponse:
        if response.metadata is None:
            response.metadata = {}
        persist = response.metadata.setdefault("_persist", {})
        persist.update(self._payload)
        return response


class EphemeralMiddleware(ConversationMiddleware):
    """Writes flat ``response.metadata`` keys (not under ``_persist``)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def process_request(
        self, messages: List[LLMMessage], state: ConversationState
    ) -> List[LLMMessage]:
        return messages

    async def process_response(
        self, response: LLMResponse, state: ConversationState
    ) -> LLMResponse:
        if response.metadata is None:
            response.metadata = {}
        response.metadata.update(self._payload)
        return response


class NullMetadataMiddleware(ConversationMiddleware):
    """Explicitly sets ``response.metadata = None``."""

    async def process_request(
        self, messages: List[LLMMessage], state: ConversationState
    ) -> List[LLMMessage]:
        return messages

    async def process_response(
        self, response: LLMResponse, state: ConversationState
    ) -> LLMResponse:
        response.metadata = None
        return response


class NonDictPersistMiddleware(ConversationMiddleware):
    """Writes a non-dict value to ``response.metadata['_persist']``."""

    async def process_request(
        self, messages: List[LLMMessage], state: ConversationState
    ) -> List[LLMMessage]:
        return messages

    async def process_response(
        self, response: LLMResponse, state: ConversationState
    ) -> LLMResponse:
        if response.metadata is None:
            response.metadata = {}
        response.metadata["_persist"] = "not a dict"
        return response


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _create_prompts(prompt_dir: Path) -> None:
    system_dir = prompt_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "assistant.yaml").write_text(
        yaml.dump({"template": "You are a helpful assistant"})
    )


@pytest.fixture
async def manager_factory():
    """Yield a factory that builds a fresh manager per test.

    The EchoProvider is shared across ``_make`` calls within one test —
    and so is its response queue. Tests that need a pre-scripted
    ``LLMResponse`` call ``manager.llm.set_responses(...)`` after creating
    the manager. If a test creates two managers and calls ``complete()``
    on both, both calls pop from the same queue; extend by instantiating a
    per-call provider if that becomes a problem.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        prompt_dir = Path(tmpdir) / "prompts"
        _create_prompts(prompt_dir)

        config = LLMConfig(
            provider="echo", model="echo-model", options={"echo_prefix": ""}
        )
        llm = EchoProvider(config)
        builder = AsyncPromptBuilder(library=FileSystemPromptLibrary(prompt_dir))
        storage = DataknobsConversationStorage(AsyncMemoryDatabase())

        async def _make(
            middleware: list[ConversationMiddleware] | None = None,
        ) -> ConversationManager:
            return await ConversationManager.create(
                llm=llm,
                prompt_builder=builder,
                storage=storage,
                system_prompt_name="assistant",
                middleware=middleware or [],
            )

        yield _make

        await llm.close()


def _assistant_metadata(manager: ConversationManager) -> dict[str, Any]:
    """Return the current assistant node's metadata dict.

    Assumes the most recent ``complete()`` or ``stream_complete()`` advanced
    the manager's position to the assistant node.
    """
    current_node = manager.state.get_current_node()
    assert current_node is not None
    return current_node.data.metadata


# ---------------------------------------------------------------------------
# Change A: _persist namespace merge in _finalize_completion
# ---------------------------------------------------------------------------


class TestPersistNamespaceMerge:
    """The ``_persist`` sub-dict on ``response.metadata`` flows to the node."""

    @pytest.mark.asyncio
    async def test_persist_namespace_flows_to_node(self, manager_factory):
        """Keys inside ``_persist`` land on the assistant node metadata."""
        mw = PersistingMiddleware({"audit": {"kept": 5, "dropped": 2}})
        manager = await manager_factory(middleware=[mw])
        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        metadata = _assistant_metadata(manager)
        assert metadata["audit"] == {"kept": 5, "dropped": 2}
        # The `_persist` marker itself is not propagated.
        assert "_persist" not in metadata

    @pytest.mark.asyncio
    async def test_ephemeral_metadata_not_persisted(self, manager_factory):
        """Flat ``response.metadata`` writes stay ephemeral (not on node)."""
        mw = EphemeralMiddleware({"telemetry_count": 5})
        manager = await manager_factory(middleware=[mw])
        await manager.add_message(role="user", content="Hello")
        response = await manager.complete()

        metadata = _assistant_metadata(manager)
        # The flat key is NOT on the persisted node.
        assert "telemetry_count" not in metadata
        # But it IS still on the in-memory response object.
        assert response.metadata["telemetry_count"] == 5

    @pytest.mark.asyncio
    async def test_persist_does_not_clobber_canonical_fields(
        self, manager_factory
    ):
        """Canonical framework fields win over ``_persist`` values."""
        mw = PersistingMiddleware(
            {"usage": "bogus", "model": "bogus", "custom": "ok"}
        )
        manager = await manager_factory(middleware=[mw])
        await manager.add_message(role="user", content="Hello")
        response = await manager.complete()

        metadata = _assistant_metadata(manager)
        # Canonical fields set by _finalize_completion win.
        assert metadata["usage"] == response.usage
        assert metadata["model"] == response.model
        # Non-colliding persist key still lands.
        assert metadata["custom"] == "ok"

    @pytest.mark.asyncio
    async def test_caller_metadata_wins_over_persist(self, manager_factory):
        """Caller-passed ``metadata=`` wins over middleware-provided values."""
        mw = PersistingMiddleware({"shared": "middleware"})
        manager = await manager_factory(middleware=[mw])
        await manager.add_message(role="user", content="Hello")
        await manager.complete(metadata={"shared": "caller"})

        metadata = _assistant_metadata(manager)
        assert metadata["shared"] == "caller"

    @pytest.mark.asyncio
    async def test_missing_response_metadata_handled(self, manager_factory):
        """No error when a middleware sets ``response.metadata = None``."""
        mw = NullMetadataMiddleware()
        manager = await manager_factory(middleware=[mw])
        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        metadata = _assistant_metadata(manager)
        # Canonical fields still present.
        assert "model" in metadata
        assert "provider" in metadata

    @pytest.mark.asyncio
    async def test_non_dict_persist_value_skipped_and_warned(
        self, manager_factory, caplog
    ):
        """Non-dict ``_persist`` values are skipped with a WARNING log."""
        mw = NonDictPersistMiddleware()
        manager = await manager_factory(middleware=[mw])
        await manager.add_message(role="user", content="Hello")

        with caplog.at_level(
            logging.WARNING, logger="dataknobs_llm.conversations.manager"
        ):
            await manager.complete()

        metadata = _assistant_metadata(manager)
        assert "_persist" not in metadata
        # No crash, and the warning was emitted.
        assert any(
            "expected dict, got str" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_multiple_middleware_merge_disjoint_keys(
        self, manager_factory
    ):
        """Disjoint ``_persist`` writes from multiple middleware all land."""
        mw_a = PersistingMiddleware({"a": 1})
        mw_b = PersistingMiddleware({"b": 2})
        manager = await manager_factory(middleware=[mw_a, mw_b])
        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        metadata = _assistant_metadata(manager)
        assert metadata["a"] == 1
        assert metadata["b"] == 2

    @pytest.mark.asyncio
    async def test_outer_middleware_wins_on_shared_persist_key(
        self, manager_factory
    ):
        """With onion ordering, the last ``process_response`` to run wins.

        ``middleware = [mw_outer, mw_inner]`` — onion convention:
        ``middleware[0]`` is outer, ``middleware[-1]`` is inner.
        ``process_response`` iterates ``reversed(middleware)``, so mw_inner
        runs first and mw_outer runs last. Both call ``persist.update(...)``
        with the same key; mw_outer's write wins because it runs last. At
        merge time, ``setdefault`` sees mw_outer's value on ``_persist``.
        """
        mw_outer = PersistingMiddleware({"shared": "outer"})
        mw_inner = PersistingMiddleware({"shared": "inner"})
        manager = await manager_factory(middleware=[mw_outer, mw_inner])
        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        metadata = _assistant_metadata(manager)
        assert metadata["shared"] == "outer"

    @pytest.mark.asyncio
    async def test_persist_on_stream_complete(self, manager_factory):
        """``stream_complete()`` path also merges ``_persist`` into the node."""
        mw = PersistingMiddleware({"audit": {"kept": 7}})
        manager = await manager_factory(middleware=[mw])
        await manager.add_message(role="user", content="Hello")
        async for _chunk in manager.stream_complete():
            pass

        metadata = _assistant_metadata(manager)
        assert metadata["audit"] == {"kept": 7}
        assert "_persist" not in metadata

    @pytest.mark.asyncio
    async def test_provider_write_to_persist_is_also_merged(
        self, manager_factory
    ):
        """Provider-sourced ``_persist`` writes are merged (writer-agnostic)."""
        manager = await manager_factory()
        # Pre-script the provider to return an LLMResponse with _persist
        # already populated — simulating a provider that writes directly.
        scripted = LLMResponse(
            content="scripted content",
            model="echo-model",
            finish_reason="stop",
            metadata={"_persist": {"provider_audit": "ok"}},
        )
        manager.llm.set_responses([scripted])

        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        metadata = _assistant_metadata(manager)
        assert metadata["provider_audit"] == "ok"
        assert "_persist" not in metadata


# ---------------------------------------------------------------------------
# Change B: PromoteToPersistMiddleware
# ---------------------------------------------------------------------------


class TestPromoteToPersistMiddleware:
    """The adapter middleware promotes flat keys into ``_persist``."""

    @pytest.mark.asyncio
    async def test_promote_middleware_moves_flat_keys_to_persist(
        self, manager_factory
    ):
        """Promoter at position [0] captures flat keys written by later
        middleware (earlier on response due to onion reversal)."""
        promoter = PromoteToPersistMiddleware(keys=["telemetry_count"])
        ephemeral = EphemeralMiddleware({"telemetry_count": 42})
        # Position [0] = outer = last on response.
        manager = await manager_factory(middleware=[promoter, ephemeral])
        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        metadata = _assistant_metadata(manager)
        assert metadata["telemetry_count"] == 42
        assert "_persist" not in metadata

    @pytest.mark.asyncio
    async def test_promote_middleware_skips_missing_keys(self, manager_factory):
        """Missing allowlisted keys do not error and do not land on the node."""
        promoter = PromoteToPersistMiddleware(
            keys=["not_written", "also_not_written"]
        )
        manager = await manager_factory(middleware=[promoter])
        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        metadata = _assistant_metadata(manager)
        assert "not_written" not in metadata
        assert "also_not_written" not in metadata
        # Canonical fields still present.
        assert "model" in metadata
        assert "provider" in metadata

    @pytest.mark.asyncio
    async def test_promote_middleware_preserves_native_persist_writer(
        self, manager_factory
    ):
        """Native ``_persist`` writer takes precedence over the promoter."""
        promoter = PromoteToPersistMiddleware(keys=["shared"])
        native = PersistingMiddleware({"shared": "native"})
        ephemeral = EphemeralMiddleware({"shared": "flat"})
        # Ordering (position [0] = outer = last on response):
        # - response: ephemeral runs first (writes flat "shared"=flat),
        #             native runs next (writes _persist["shared"]="native"),
        #             promoter runs last (setdefault skips — already set).
        manager = await manager_factory(
            middleware=[promoter, native, ephemeral]
        )
        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        metadata = _assistant_metadata(manager)
        assert metadata["shared"] == "native"

    @pytest.mark.asyncio
    async def test_promote_middleware_captures_provider_metadata_regardless_of_position(
        self, manager_factory
    ):
        """Provider writes pre-date middleware, so promoter position is
        irrelevant for provider-sourced keys."""
        # Promoter intentionally a sole middleware — its position is
        # irrelevant for provider-sourced keys because those are already
        # in response.metadata before any middleware runs.
        promoter = PromoteToPersistMiddleware(keys=["model_info"])
        manager = await manager_factory(middleware=[promoter])

        # Pre-script the provider so the LLMResponse carries flat
        # ``model_info`` in its metadata before any middleware runs.
        scripted = LLMResponse(
            content="scripted content",
            model="echo-model",
            finish_reason="stop",
            metadata={"model_info": {"family": "llama", "params": "7B"}},
        )
        manager.llm.set_responses([scripted])

        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        metadata = _assistant_metadata(manager)
        assert metadata["model_info"] == {"family": "llama", "params": "7B"}

    @pytest.mark.asyncio
    async def test_promote_middleware_wrong_position_misses_middleware_writes(
        self, manager_factory
    ):
        """Regression guard: promoter at [-1] misses middleware writes.

        Onion-reversal on response: ``middleware=[ephemeral, promoter]``
        runs promoter FIRST on response (before ephemeral writes) and
        ephemeral LAST. The promoter captures nothing because ephemeral's
        flat key isn't in ``response.metadata`` yet when the promoter runs.
        Documents the position-[0] requirement.
        """
        ephemeral = EphemeralMiddleware({"x": 1})
        promoter = PromoteToPersistMiddleware(keys=["x"])
        # Wrong position: promoter at [-1].
        manager = await manager_factory(middleware=[ephemeral, promoter])
        await manager.add_message(role="user", content="Hello")
        await manager.complete()

        metadata = _assistant_metadata(manager)
        assert "x" not in metadata

    @pytest.mark.asyncio
    async def test_promote_middleware_on_stream_complete(self, manager_factory):
        """Promoter operates on ``stream_complete()`` too: flat keys written
        by another middleware during response processing are promoted into
        the persisted assistant-node metadata."""
        promoter = PromoteToPersistMiddleware(keys=["telemetry_count"])
        ephemeral = EphemeralMiddleware({"telemetry_count": 42})
        # Position [0] = outer = last on response.
        manager = await manager_factory(middleware=[promoter, ephemeral])
        await manager.add_message(role="user", content="Hello")
        async for _chunk in manager.stream_complete():
            pass

        metadata = _assistant_metadata(manager)
        assert metadata["telemetry_count"] == 42
        assert "_persist" not in metadata
