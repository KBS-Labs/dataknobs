"""Tests for the subsystem backend registries and their typed configs.

Covers the data-driven dispatch added to the memory, knowledge-base, and
grounded-source factories: built-in backend registration, 3rd-party
extensibility via the ``register_*_backend`` wrappers, injected-collaborator
threading, round-trip behaviour of the typed sub-configs,
structured-config-consumer parity for the subsystem classes, and the
config-vs-pre-built construction contracts.

The exhaustive behavioural coverage of each factory (every backend's output
and key handling) lives in ``test_memory.py``, ``test_composite_memory.py``,
``test_knowledge.py``, and ``test_grounded_reasoning.py``; this module
focuses on the registry surface and the typed configs.

Real constructs only — ``EchoProvider`` (via the ``echo`` provider key) and
in-memory vector stores. No mocks.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from dataknobs_bots.knowledge import (
    RAGKnowledgeBase,
    RAGKnowledgeBaseConfig,
    create_knowledge_base_from_config,
    is_knowledge_base_backend_registered,
    list_knowledge_base_backends,
)
from dataknobs_bots.knowledge.sources import (
    create_source_from_config,
    is_source_backend_registered,
    list_source_backends,
)
from dataknobs_bots.memory import (
    BufferMemory,
    BufferMemoryConfig,
    CompositeMemory,
    CompositeMemoryConfig,
    Memory,
    SummaryMemoryConfig,
    VectorMemory,
    VectorMemoryConfig,
    create_memory_from_config,
    is_memory_backend_registered,
    list_memory_backends,
    memory_backends,
    register_memory_backend,
)
from dataknobs_bots.memory.summary import SummaryMemory
from dataknobs_bots.reasoning.grounded_config import GroundedSourceConfig
from dataknobs_common.testing import (
    assert_structured_config_consumer,
    assert_structured_config_roundtrip,
)
from dataknobs_data.vector.stores import VectorStoreFactory
from dataknobs_llm.llm import LLMProviderFactory


async def _make_store_and_embedder() -> tuple[Any, Any]:
    """Build an initialized in-memory vector store + echo embedder."""
    store = VectorStoreFactory().create(backend="memory", dimensions=8)
    await store.initialize()
    provider = LLMProviderFactory(is_async=True).create(
        {"provider": "echo", "model": "test"}
    )
    await provider.initialize()
    return store, provider


# ===========================================================================
# Built-in registration
# ===========================================================================


class TestBuiltinRegistration:
    """The built-in backends are registered (lazily) on first access."""

    def test_memory_builtins_registered(self) -> None:
        # Subset, not exact-equality: ``memory_backends`` is a process-global
        # singleton, so an exact ``==`` would be fragile if any other test
        # registered a backend without cleanup (order-dependent under
        # pytest-randomly).
        assert set(list_memory_backends()) >= {
            "buffer",
            "composite",
            "summary",
            "vector",
        }
        for name in ("buffer", "vector", "summary", "composite"):
            assert is_memory_backend_registered(name)

    def test_knowledge_builtins_registered(self) -> None:
        assert "rag" in list_knowledge_base_backends()
        assert is_knowledge_base_backend_registered("rag")

    def test_source_builtins_registered(self) -> None:
        # Subset rather than exact-equality — see the note on the memory
        # built-ins test (shared process-global singleton).
        assert set(list_source_backends()) >= {"database", "vector_kb"}
        assert is_source_backend_registered("vector_kb")
        assert is_source_backend_registered("database")


# ===========================================================================
# 3rd-party extensibility
# ===========================================================================


class _CountingMemory(Memory):
    """Minimal real Memory implementation for registration tests."""

    def __init__(self, label: str = "x") -> None:
        self.label = label
        self.messages: list[dict[str, Any]] = []

    async def add_message(
        self, content: str, role: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self.messages.append({"content": content, "role": role})

    async def get_context(self, current_message: str) -> list[dict[str, Any]]:
        return list(self.messages)

    async def clear(self) -> None:
        self.messages.clear()


def _build_counting(config: dict[str, Any], **_: Any) -> Memory:
    return _CountingMemory(label=config.get("label", "x"))


@pytest.fixture
def counting_backend() -> Iterator[None]:
    """Register a custom memory backend and clean it up afterwards."""
    register_memory_backend("counting", _build_counting)
    try:
        yield
    finally:
        memory_backends.unregister("counting")


class TestExtensibility:
    """Custom backends register and dispatch through the public factory."""

    @pytest.mark.asyncio
    async def test_custom_memory_backend_dispatches(
        self, counting_backend: None
    ) -> None:
        assert is_memory_backend_registered("counting")
        memory = await create_memory_from_config(
            {"type": "counting", "label": "hello"}
        )
        assert isinstance(memory, _CountingMemory)
        assert memory.label == "hello"

    def test_duplicate_registration_requires_override(
        self, counting_backend: None
    ) -> None:
        from dataknobs_common.exceptions import OperationError

        with pytest.raises(OperationError, match="already registered"):
            register_memory_backend("counting", _build_counting)
        # override succeeds
        register_memory_backend("counting", _build_counting, override=True)

    @pytest.mark.asyncio
    async def test_custom_backend_used_in_composite(
        self, counting_backend: None
    ) -> None:
        """A custom backend works as a composite child (recursion path)."""
        memory = await create_memory_from_config(
            {
                "type": "composite",
                "strategies": [
                    {"type": "buffer", "max_messages": 5},
                    {"type": "counting", "label": "child"},
                ],
            }
        )
        assert isinstance(memory, CompositeMemory)
        assert len(memory.strategies) == 2
        assert any(isinstance(s, _CountingMemory) for s in memory.strategies)


# ===========================================================================
# Error contract (preserved across the registry migration)
# ===========================================================================


class TestErrorContract:
    """Unknown discriminators still raise ValueError (public contract)."""

    @pytest.mark.asyncio
    async def test_unknown_memory_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown memory type"):
            await create_memory_from_config({"type": "__nope__"})

    @pytest.mark.asyncio
    async def test_unknown_kb_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown knowledge base type"):
            await create_knowledge_base_from_config({"type": "__nope__"})

    @pytest.mark.asyncio
    async def test_unknown_source_type_raises_value_error(self) -> None:
        config = GroundedSourceConfig(source_type="__nope__", name="s")
        with pytest.raises(ValueError, match="Unknown grounded source type"):
            await create_source_from_config(config)

    @pytest.mark.asyncio
    async def test_nested_unknown_type_surfaces_value_error(self) -> None:
        """A bad composite child surfaces ValueError, not OperationError."""
        with pytest.raises(ValueError, match="Unknown memory type"):
            await create_memory_from_config(
                {
                    "type": "composite",
                    "strategies": [
                        {"type": "buffer", "max_messages": 5},
                        {"type": "__nope__"},
                    ],
                }
            )

    @pytest.mark.asyncio
    async def test_null_source_type_raises_friendly_value_error(self) -> None:
        """A null ``type:`` (``source_type`` is ``None``) surfaces the
        documented unknown-type message — not the registry's internal
        "key is required when config_key is not configured"."""
        config = GroundedSourceConfig.from_dict({"type": None, "name": "s"})
        assert config.source_type is None
        with pytest.raises(ValueError, match="Unknown grounded source type"):
            await create_source_from_config(config)

    @pytest.mark.asyncio
    async def test_empty_source_type_raises_friendly_value_error(self) -> None:
        config = GroundedSourceConfig(source_type="", name="s")
        with pytest.raises(ValueError, match="Unknown grounded source type"):
            await create_source_from_config(config)


# ===========================================================================
# Native error-type preservation (non-ValueError backend failures)
# ===========================================================================


class TestBackendErrorTypePreserved:
    """A backend's native dataknobs error reaches the caller as its own type.

    The registry wraps factory exceptions in ``OperationError``; the public
    factory unwraps the original cause so a ``ResourceError`` (e.g. a vector
    backend that fails to connect) is surfaced as ``ResourceError`` rather
    than masked as a generic ``OperationError``.
    """

    @pytest.mark.asyncio
    async def test_resource_error_from_backend_not_masked(self) -> None:
        from dataknobs_common.exceptions import ResourceError

        def _failing(config: dict[str, Any], **_: Any) -> Memory:
            raise ResourceError("backend connection refused")

        register_memory_backend("failing", _failing)
        try:
            with pytest.raises(ResourceError, match="connection refused"):
                await create_memory_from_config({"type": "failing"})
        finally:
            memory_backends.unregister("failing")


# ===========================================================================
# Injected collaborator threading
# ===========================================================================


class TestCollaboratorThreading:
    """Injected collaborators reach the backends that consume them."""

    @pytest.mark.asyncio
    async def test_summary_adopts_injected_llm(self) -> None:
        from dataknobs_llm.llm import LLMProviderFactory

        provider = LLMProviderFactory(is_async=True).create(
            {"provider": "echo", "model": "test"}
        )
        memory = await create_memory_from_config(
            {"type": "summary", "recent_window": 4}, llm_provider=provider
        )
        # The injected provider is adopted; the memory does not own it.
        assert memory.llm_provider is provider
        assert memory._owns_llm_provider is False

    @pytest.mark.asyncio
    async def test_summary_dedicated_llm_owns_lifecycle(self) -> None:
        memory = await create_memory_from_config(
            {
                "type": "summary",
                "recent_window": 4,
                "llm": {"provider": "echo", "model": "dedicated"},
            }
        )
        assert memory._owns_llm_provider is True

    @pytest.mark.asyncio
    async def test_vector_kb_source_threads_knowledge_base(self) -> None:
        kb = await create_knowledge_base_from_config(
            {
                "type": "rag",
                "vector_store": {"backend": "memory", "dimensions": 384},
                "embedding_provider": "echo",
                "embedding_model": "test",
            }
        )
        config = GroundedSourceConfig(source_type="vector_kb", name="docs")
        source = await create_source_from_config(config, knowledge_base=kb)
        assert source.name == "docs"


# ===========================================================================
# Typed sub-config round-trips
# ===========================================================================


class TestConfigRoundTrips:
    """Each new typed sub-config round-trips through from_dict/to_dict."""

    def test_buffer_config_roundtrip(self) -> None:
        assert_structured_config_roundtrip(BufferMemoryConfig(max_messages=7))

    def test_summary_config_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            SummaryMemoryConfig(recent_window=6, summary_prompt="x")
        )
        assert_structured_config_roundtrip(
            SummaryMemoryConfig(llm={"provider": "echo", "model": "m"})
        )

    def test_composite_config_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            CompositeMemoryConfig(
                strategies=[{"type": "buffer", "max_messages": 5}],
                primary_index=0,
            )
        )

    def test_vector_config_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            VectorMemoryConfig(
                backend="memory",
                dimension=8,
                embedding={"provider": "echo", "model": "m"},
                max_results=3,
                default_filter={"user_id": "u1"},
                immutable_metadata_keys=["user_id"],
            )
        )

    def test_rag_config_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            RAGKnowledgeBaseConfig(
                vector_store={"backend": "memory", "dimensions": 8},
                embedding={"provider": "echo", "model": "m"},
                chunking={"max_chunk_size": 300},
                documents_path="./docs",
            )
        )

    def test_composite_primary_alias(self) -> None:
        """The documented ``primary`` key maps to ``primary_index``."""
        cfg = CompositeMemoryConfig.from_dict(
            {"strategies": [{"type": "buffer"}], "primary": 3}
        )
        assert cfg.primary_index == 3

    def test_grounded_source_config_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            GroundedSourceConfig(
                source_type="database",
                name="courses",
                weight=2,
                options={"backend": "memory", "content_field": "body"},
            )
        )

    def test_grounded_source_legacy_flat_shape(self) -> None:
        """``type`` aliases ``source_type`` and flat keys collect into options."""
        cfg = GroundedSourceConfig.from_dict(
            {
                "type": "database",
                "name": "courses",
                "weight": 3,
                "backend": "sqlite",
                "content_field": "description",
            }
        )
        assert cfg.source_type == "database"
        assert cfg.weight == 3
        assert cfg.options == {
            "backend": "sqlite",
            "content_field": "description",
        }


# ===========================================================================
# Structured-config-consumer parity (drift guard)
# ===========================================================================


class TestStructuredConfigConsumerParity:
    """The subsystem classes correctly apply the structured-config pattern.

    Each memory/knowledge class mixes in
    :class:`~dataknobs_common.structured_config.StructuredConfigConsumer`
    and is registered directly (no hand-rolled builder). The parity guard
    pins: ``CONFIG_CLS`` is declared and is a ``StructuredConfig`` subclass,
    the dataclass field set matches the construction surface, the mixin
    precedes ``Memory`` / ``KnowledgeBase`` in the MRO, an async
    ``from_config`` override delegates to ``from_config_async`` (Check 5),
    and the ``_ainit`` / ``_adopt_components`` collaborator hooks declare
    their params keyword-only with defaults (Check 7). ``ignore_params``
    lists the injected collaborators that are not config fields.
    """

    def test_buffer_consumer(self) -> None:
        assert_structured_config_consumer(BufferMemory)

    def test_vector_consumer(self) -> None:
        assert_structured_config_consumer(
            VectorMemory,
            ignore_params={"vector_store", "embedding_provider"},
        )

    def test_summary_consumer(self) -> None:
        assert_structured_config_consumer(
            SummaryMemory,
            ignore_params={"llm_provider", "prompt_resolver"},
        )

    def test_composite_consumer(self) -> None:
        assert_structured_config_consumer(
            CompositeMemory,
            ignore_params={
                "strategies",
                "primary_index",
                "llm_provider",
                "prompt_resolver",
            },
        )

    def test_rag_consumer(self) -> None:
        assert_structured_config_consumer(
            RAGKnowledgeBase,
            ignore_params={
                "vector_store",
                "embedding_provider",
                "chunker",
                "merger_config",
                "formatter_config",
            },
        )


# ===========================================================================
# Typed-config acceptance by the factories
# ===========================================================================


class TestFactoryAcceptsDictAndDefaults:
    """The factories accept plain dicts and apply documented defaults."""

    @pytest.mark.asyncio
    async def test_default_memory_type_is_buffer(self) -> None:
        memory = await create_memory_from_config({})
        assert isinstance(memory, BufferMemory)
        assert memory.max_messages == 10

    @pytest.mark.asyncio
    async def test_default_kb_type_is_rag(self) -> None:
        kb = await create_knowledge_base_from_config(
            {
                "vector_store": {"backend": "memory", "dimensions": 384},
                "embedding_provider": "echo",
                "embedding_model": "test",
            }
        )
        from dataknobs_bots.knowledge import RAGKnowledgeBase

        assert isinstance(kb, RAGKnowledgeBase)


# ===========================================================================
# Construction-citizenship contracts (config vs. pre-built collaborators)
# ===========================================================================


class TestConstructionCitizenship:
    """The subsystem classes construct uniformly through the mixin lifecycle.

    Two construction shapes per class: ``from_config(_async)`` builds (and,
    where applicable, owns) its collaborators via ``_ainit``;
    ``from_components`` adopts caller-owned collaborators and short-circuits
    ``_ainit`` via the ``_prebuilt`` flag. Real constructs only — in-memory
    vector store + echo embedder.
    """

    @pytest.mark.asyncio
    async def test_vector_from_components_adopts_without_building(self) -> None:
        store, provider = await _make_store_and_embedder()
        mem = VectorMemory.from_components(
            {"max_results": 2},
            vector_store=store,
            embedding_provider=provider,
        )
        # Adopts the exact instances, marks them not-owned, skips _ainit.
        assert mem.vector_store is store
        assert mem.embedding_provider is provider
        assert mem._owns_vector_store is False
        assert mem._owns_embedding_provider is False
        assert mem._prebuilt is True
        assert mem.max_results == 2

    @pytest.mark.asyncio
    async def test_vector_from_config_builds_and_owns(self) -> None:
        mem = await VectorMemory.from_config(
            {
                "backend": "memory",
                "dimension": 8,
                "embedding_provider": "echo",
                "embedding_model": "test",
            }
        )
        assert mem.vector_store is not None
        assert mem.embedding_provider is not None
        assert mem._owns_vector_store is True
        assert mem._owns_embedding_provider is True

    @pytest.mark.asyncio
    async def test_rag_from_components_adopts_without_building(self) -> None:
        store, provider = await _make_store_and_embedder()
        kb = RAGKnowledgeBase.from_components(
            {"chunking": {"max_chunk_size": 123}},
            vector_store=store,
            embedding_provider=provider,
        )
        assert kb.vector_store is store
        assert kb.embedding_provider is provider
        assert kb._prebuilt is True
        # Config-derived (synchronous) collaborator still built in _setup.
        assert kb.chunking_config["max_chunk_size"] == 123

    @pytest.mark.asyncio
    async def test_summary_from_config_delegator_runs_ainit(self) -> None:
        # The async ``from_config`` delegator runs ``_ainit`` so a dedicated
        # ``llm`` section yields an owned, initialized provider.
        mem = await SummaryMemory.from_config(
            {"recent_window": 4, "llm": {"provider": "echo", "model": "test"}}
        )
        assert mem.recent_window == 4
        assert mem._owns_llm_provider is True
        assert mem.llm_provider is not None

    @pytest.mark.asyncio
    async def test_composite_from_config_delegator_runs_ainit(self) -> None:
        # The async ``from_config`` delegator runs ``_ainit`` so child specs
        # are recursively built through the factory.
        composite = await CompositeMemory.from_config(
            {
                "strategies": [
                    {"type": "buffer", "max_messages": 5},
                    {"type": "buffer", "max_messages": 10},
                ],
                "primary": 1,
            }
        )
        assert isinstance(composite, CompositeMemory)
        assert len(composite.strategies) == 2
        assert composite.primary.max_messages == 10
