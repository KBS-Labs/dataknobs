"""Tests for CompositeMemory and VectorMemory scoping."""

import pytest

from dataknobs_bots.memory import (
    BufferMemory,
    CompositeMemory,
    VectorMemory,
    create_memory_from_config,
)
from dataknobs_bots.memory.base import Memory
from dataknobs_data.vector.stores import VectorStoreFactory
from dataknobs_llm.llm import LLMProviderFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_vector_memory(
    *,
    similarity_threshold: float = 0.0,
    default_metadata: dict | None = None,
    default_filter: dict | None = None,
) -> VectorMemory:
    """Create a VectorMemory backed by an in-memory vector store + EchoProvider."""
    factory = VectorStoreFactory()
    store = factory.create(backend="memory", dimensions=384)
    await store.initialize()

    llm_factory = LLMProviderFactory(is_async=True)
    provider = llm_factory.create({"provider": "echo", "model": "test"})
    await provider.initialize()

    return VectorMemory(
        vector_store=store,
        embedding_provider=provider,
        max_results=10,
        similarity_threshold=similarity_threshold,
        default_metadata=default_metadata,
        default_filter=default_filter,
    )


class _FailingMemory(Memory):
    """A memory that raises on every operation. Used for degradation tests."""

    async def add_message(self, content, role, metadata=None):  # type: ignore[override]
        raise RuntimeError("add_message intentionally broken")

    async def get_context(self, current_message):  # type: ignore[override]
        raise RuntimeError("get_context intentionally broken")

    async def clear(self):
        raise RuntimeError("clear intentionally broken")

    async def close(self):
        raise RuntimeError("close intentionally broken")


# ===========================================================================
# VectorMemory scoping
# ===========================================================================


class TestVectorMemoryScoping:
    """Tests for VectorMemory default_metadata and default_filter."""

    @pytest.mark.asyncio
    async def test_default_metadata_merged_on_add(self):
        """default_metadata is present in every stored vector's metadata."""
        mem = await _make_vector_memory(default_metadata={"user_id": "u1"})

        await mem.add_message("hello", "user")

        # Retrieve via search — threshold 0 means everything matches
        context = await mem.get_context("hello")
        assert len(context) == 1
        assert context[0]["metadata"]["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_caller_metadata_overrides_default(self):
        """Caller-supplied metadata wins over default_metadata for same keys."""
        mem = await _make_vector_memory(default_metadata={"user_id": "default"})

        await mem.add_message("hello", "user", metadata={"user_id": "override"})

        context = await mem.get_context("hello")
        assert len(context) == 1
        assert context[0]["metadata"]["user_id"] == "override"

    @pytest.mark.asyncio
    async def test_no_defaults_preserves_existing_behavior(self):
        """Without defaults, VectorMemory behaves identically to before."""
        mem = await _make_vector_memory()

        await mem.add_message("hello", "user")

        context = await mem.get_context("hello")
        assert len(context) == 1
        # Should have base metadata but no extra keys
        md = context[0]["metadata"]
        assert "content" in md
        assert "role" in md
        assert "timestamp" in md
        assert "id" in md

    @pytest.mark.asyncio
    async def test_default_metadata_cannot_overwrite_base_fields(self):
        """default_metadata with reserved keys does not clobber system fields."""
        mem = await _make_vector_memory(
            default_metadata={"content": "WRONG", "role": "WRONG", "tenant": "t1"},
        )
        await mem.add_message("hello", "user")

        context = await mem.get_context("hello")
        assert len(context) >= 1
        md = context[0]["metadata"]
        # Base fields must win over defaults
        assert md["content"] == "hello"
        assert md["role"] == "user"
        # Non-colliding default still present
        assert md["tenant"] == "t1"

    @pytest.mark.asyncio
    async def test_from_config_passes_defaults(self):
        """from_config() with default_metadata and default_filter constructs correctly."""
        config = {
            "backend": "memory",
            "dimension": 384,
            "embedding_provider": "echo",
            "embedding_model": "test",
            "default_metadata": {"tenant": "t1"},
            "default_filter": {"tenant": "t1"},
        }
        mem = await VectorMemory.from_config(config)
        assert mem._default_metadata == {"tenant": "t1"}
        assert mem._default_filter == {"tenant": "t1"}

    @pytest.mark.asyncio
    async def test_default_filter_scopes_get_context(self):
        """default_filter isolates reads so each tenant sees only its own data."""
        # Use a shared store for both tenants, with threshold=-1.0 so
        # deterministic EchoProvider embeddings always pass the score check.
        mem_u1 = await _make_vector_memory(
            similarity_threshold=-1.0,
            default_filter={"user_id": "u1"},
            default_metadata={"user_id": "u1"},
        )
        mem_u2 = await _make_vector_memory(
            similarity_threshold=-1.0,
            default_filter={"user_id": "u2"},
            default_metadata={"user_id": "u2"},
        )

        # Both share the same backing store so filtering is meaningful
        mem_u2.vector_store = mem_u1.vector_store

        await mem_u1.add_message("user one message", "user")
        await mem_u2.add_message("user two message", "user")

        # u1 should only see its own message
        ctx_u1 = await mem_u1.get_context("query")
        contents_u1 = [m["content"] for m in ctx_u1]
        assert "user one message" in contents_u1
        assert "user two message" not in contents_u1

        # u2 should only see its own message
        ctx_u2 = await mem_u2.get_context("query")
        contents_u2 = [m["content"] for m in ctx_u2]
        assert "user two message" in contents_u2
        assert "user one message" not in contents_u2


# ===========================================================================
# CompositeMemory — basics
# ===========================================================================


class TestCompositeMemoryBasics:
    """Core CompositeMemory behavior."""

    @pytest.mark.asyncio
    async def test_add_message_forwards_to_all(self):
        """add_message() is forwarded to every strategy."""
        buf1 = BufferMemory(max_messages=10)
        buf2 = BufferMemory(max_messages=10)
        composite = CompositeMemory([buf1, buf2])

        await composite.add_message("hello", "user")

        ctx1 = await buf1.get_context("x")
        ctx2 = await buf2.get_context("x")
        assert len(ctx1) == 1
        assert len(ctx2) == 1
        assert ctx1[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_get_context_primary_first(self):
        """Primary strategy results appear before secondary results."""
        buf_primary = BufferMemory(max_messages=10)
        buf_secondary = BufferMemory(max_messages=10)

        # Add different messages to each directly
        await buf_primary.add_message("primary-msg", "user")
        await buf_secondary.add_message("secondary-msg", "assistant")

        composite = CompositeMemory([buf_primary, buf_secondary])

        context = await composite.get_context("x")
        assert len(context) == 2
        assert context[0]["content"] == "primary-msg"
        assert context[1]["content"] == "secondary-msg"

    @pytest.mark.asyncio
    async def test_get_context_deduplicates(self):
        """Same message in multiple strategies appears only once."""
        buf1 = BufferMemory(max_messages=10)
        buf2 = BufferMemory(max_messages=10)
        composite = CompositeMemory([buf1, buf2])

        # Add same message via composite — both buffers get it
        await composite.add_message("duplicate", "user")

        context = await composite.get_context("x")
        contents = [m["content"] for m in context]
        assert contents.count("duplicate") == 1

    @pytest.mark.asyncio
    async def test_get_context_preserves_unique(self):
        """Different messages from each strategy all appear."""
        buf1 = BufferMemory(max_messages=10)
        buf2 = BufferMemory(max_messages=10)

        await buf1.add_message("from-buf1", "user")
        await buf2.add_message("from-buf2", "assistant")

        composite = CompositeMemory([buf1, buf2])

        context = await composite.get_context("x")
        contents = [m["content"] for m in context]
        assert "from-buf1" in contents
        assert "from-buf2" in contents

    @pytest.mark.asyncio
    async def test_clear_clears_all(self):
        """clear() clears all strategies."""
        buf1 = BufferMemory(max_messages=10)
        buf2 = BufferMemory(max_messages=10)
        composite = CompositeMemory([buf1, buf2])

        await composite.add_message("hello", "user")
        await composite.clear()

        ctx1 = await buf1.get_context("x")
        ctx2 = await buf2.get_context("x")
        assert len(ctx1) == 0
        assert len(ctx2) == 0

    @pytest.mark.asyncio
    async def test_pop_messages_delegates_to_primary(self):
        """pop_messages() only affects the primary strategy."""
        buf_primary = BufferMemory(max_messages=10)
        buf_secondary = BufferMemory(max_messages=10)
        composite = CompositeMemory([buf_primary, buf_secondary])

        await composite.add_message("msg1", "user")
        await composite.add_message("msg2", "assistant")

        removed = await composite.pop_messages(1)
        assert len(removed) == 1
        assert removed[0]["content"] == "msg2"

        # Primary lost the message
        primary_ctx = await buf_primary.get_context("x")
        assert len(primary_ctx) == 1

        # Secondary still has both
        secondary_ctx = await buf_secondary.get_context("x")
        assert len(secondary_ctx) == 2

    @pytest.mark.asyncio
    async def test_pop_messages_not_supported_by_primary(self):
        """If primary raises NotImplementedError, it propagates."""
        vec = await _make_vector_memory()
        buf = BufferMemory(max_messages=10)

        # VectorMemory is primary — it doesn't support pop
        composite = CompositeMemory([vec, buf])

        with pytest.raises(NotImplementedError, match="VectorMemory"):
            await composite.pop_messages(1)


# ===========================================================================
# CompositeMemory — graceful degradation
# ===========================================================================


class TestCompositeMemoryDegradation:
    """CompositeMemory continues operating when a strategy fails."""

    @pytest.mark.asyncio
    async def test_add_message_continues_on_strategy_failure(self):
        """One strategy raises, others still receive the message."""
        buf = BufferMemory(max_messages=10)
        failing = _FailingMemory()
        composite = CompositeMemory([buf, failing])

        # Should not raise
        await composite.add_message("hello", "user")

        ctx = await buf.get_context("x")
        assert len(ctx) == 1

    @pytest.mark.asyncio
    async def test_get_context_continues_on_strategy_failure(self):
        """One strategy raises, results from others still returned."""
        buf = BufferMemory(max_messages=10)
        await buf.add_message("buffered", "user")

        failing = _FailingMemory()
        composite = CompositeMemory([buf, failing])

        context = await composite.get_context("x")
        assert len(context) == 1
        assert context[0]["content"] == "buffered"

    @pytest.mark.asyncio
    async def test_clear_continues_on_strategy_failure(self):
        """One strategy's clear fails, others still cleared."""
        buf = BufferMemory(max_messages=10)
        failing = _FailingMemory()
        composite = CompositeMemory([buf, failing])

        await composite.add_message("hello", "user")

        # Should not raise even though failing.clear() throws
        await composite.clear()

        ctx = await buf.get_context("x")
        assert len(ctx) == 0

    @pytest.mark.asyncio
    async def test_close_continues_on_strategy_failure(self):
        """One strategy's close fails, others still closed."""
        buf = BufferMemory(max_messages=10)
        failing = _FailingMemory()
        composite = CompositeMemory([buf, failing])

        # Should not raise even though failing.close() throws
        await composite.close()


# ===========================================================================
# CompositeMemory — validation
# ===========================================================================


class TestCompositeMemoryValidation:
    """Constructor validation for CompositeMemory."""

    def test_empty_strategies_raises(self):
        with pytest.raises(ValueError, match="at least one strategy"):
            CompositeMemory([])

    def test_invalid_primary_index_raises(self):
        buf = BufferMemory(max_messages=10)
        with pytest.raises(ValueError, match="out of range"):
            CompositeMemory([buf], primary_index=5)

    def test_negative_primary_index_raises(self):
        buf = BufferMemory(max_messages=10)
        with pytest.raises(ValueError, match="out of range"):
            CompositeMemory([buf], primary_index=-1)

    def test_properties(self):
        buf1 = BufferMemory(max_messages=10)
        buf2 = BufferMemory(max_messages=10)
        composite = CompositeMemory([buf1, buf2], primary_index=1)

        assert composite.primary is buf2
        assert len(composite.strategies) == 2
        assert composite.strategies[0] is buf1
        assert composite.strategies[1] is buf2


# ===========================================================================
# CompositeMemory — factory (create_memory_from_config)
# ===========================================================================


class TestCompositeMemoryFromConfig:
    """Tests for creating CompositeMemory via the factory."""

    @pytest.mark.asyncio
    async def test_composite_from_config_buffer_plus_buffer(self):
        """Config with two buffer strategies creates CompositeMemory."""
        config = {
            "type": "composite",
            "strategies": [
                {"type": "buffer", "max_messages": 5},
                {"type": "buffer", "max_messages": 10},
            ],
        }
        memory = await create_memory_from_config(config)
        assert isinstance(memory, CompositeMemory)
        assert len(memory.strategies) == 2
        assert all(isinstance(s, BufferMemory) for s in memory.strategies)

    @pytest.mark.asyncio
    async def test_composite_from_config_with_primary(self):
        """primary key selects the correct strategy."""
        config = {
            "type": "composite",
            "primary": 1,
            "strategies": [
                {"type": "buffer", "max_messages": 5},
                {"type": "buffer", "max_messages": 20},
            ],
        }
        memory = await create_memory_from_config(config)
        assert isinstance(memory, CompositeMemory)
        assert isinstance(memory.primary, BufferMemory)
        assert memory.primary.max_messages == 20

    @pytest.mark.asyncio
    async def test_composite_from_config_empty_strategies_raises(self):
        """Empty strategies list raises ValueError."""
        config = {
            "type": "composite",
            "strategies": [],
        }
        with pytest.raises(ValueError, match="at least one strategy"):
            await create_memory_from_config(config)

    @pytest.mark.asyncio
    async def test_composite_from_config_no_strategies_key_raises(self):
        """Missing strategies key raises ValueError."""
        config = {"type": "composite"}
        with pytest.raises(ValueError, match="at least one strategy"):
            await create_memory_from_config(config)


# ===========================================================================
# CompositeMemory — providers() / set_provider()
# ===========================================================================


class TestCompositeMemoryProviders:
    """Tests for provider aggregation across sub-strategies."""

    @pytest.mark.asyncio
    async def test_providers_aggregates_from_sub_strategies(self):
        """providers() returns union of all sub-strategy providers."""
        vec = await _make_vector_memory()
        buf = BufferMemory(max_messages=10)
        composite = CompositeMemory([buf, vec])

        providers = composite.providers()
        # VectorMemory exposes its embedding provider
        from dataknobs_bots.bot.base import PROVIDER_ROLE_MEMORY_EMBEDDING

        assert PROVIDER_ROLE_MEMORY_EMBEDDING in providers
        assert providers[PROVIDER_ROLE_MEMORY_EMBEDDING] is vec.embedding_provider

    @pytest.mark.asyncio
    async def test_set_provider_forwards_to_sub_strategies(self):
        """set_provider() is forwarded; returns True if any accepted."""
        vec = await _make_vector_memory()
        buf = BufferMemory(max_messages=10)
        composite = CompositeMemory([buf, vec])

        from dataknobs_bots.bot.base import PROVIDER_ROLE_MEMORY_EMBEDDING

        new_provider = object()
        result = composite.set_provider(PROVIDER_ROLE_MEMORY_EMBEDDING, new_provider)
        assert result is True
        assert vec.embedding_provider is new_provider

    def test_set_provider_returns_false_for_unknown_role(self):
        buf = BufferMemory(max_messages=10)
        composite = CompositeMemory([buf])

        result = composite.set_provider("nonexistent_role", object())
        assert result is False
