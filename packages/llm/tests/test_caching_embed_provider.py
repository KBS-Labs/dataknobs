"""Tests for CachingEmbedProvider and cache backends."""

import struct
import tempfile
from pathlib import Path

import pytest

from dataknobs_llm import EchoProvider, LLMProviderFactory
from dataknobs_llm.llm.providers.caching import (
    CachingEmbedProvider,
    EmbeddingCache,
    MemoryEmbeddingCache,
    SqliteEmbeddingCache,
    _cache_key,
    create_caching_provider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_echo_provider() -> EchoProvider:
    factory = LLMProviderFactory(is_async=True)
    return factory.create({"provider": "echo", "model": "test-embed"})


# ---------------------------------------------------------------------------
# Cache key tests
# ---------------------------------------------------------------------------


class TestCacheKey:
    """Verify cache key determinism and collision avoidance."""

    def test_same_input_same_key(self):
        assert _cache_key("model", "text") == _cache_key("model", "text")

    def test_different_text_different_key(self):
        assert _cache_key("model", "a") != _cache_key("model", "b")

    def test_different_model_different_key(self):
        assert _cache_key("m1", "text") != _cache_key("m2", "text")

    def test_no_collision_with_separator(self):
        """("model", "text") != ("mode", "ltext")."""
        assert _cache_key("model", "text") != _cache_key("mode", "ltext")


# ---------------------------------------------------------------------------
# MemoryEmbeddingCache
# ---------------------------------------------------------------------------


class TestMemoryEmbeddingCache:
    """Verify MemoryEmbeddingCache stores and retrieves correctly."""

    @pytest.mark.asyncio
    async def test_get_miss(self):
        cache = MemoryEmbeddingCache()
        assert await cache.get("model", "text") is None

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        cache = MemoryEmbeddingCache()
        vec = [1.0, 2.0, 3.0]
        await cache.put("model", "hello", vec)
        result = await cache.get("model", "hello")
        assert result == vec

    @pytest.mark.asyncio
    async def test_get_batch(self):
        cache = MemoryEmbeddingCache()
        await cache.put("m", "a", [1.0])
        await cache.put("m", "c", [3.0])

        results = await cache.get_batch("m", ["a", "b", "c"])
        assert results == [[1.0], None, [3.0]]

    @pytest.mark.asyncio
    async def test_put_batch(self):
        cache = MemoryEmbeddingCache()
        await cache.put_batch("m", ["x", "y"], [[1.0], [2.0]])

        assert await cache.get("m", "x") == [1.0]
        assert await cache.get("m", "y") == [2.0]

    @pytest.mark.asyncio
    async def test_clear(self):
        cache = MemoryEmbeddingCache()
        await cache.put("m", "a", [1.0])
        assert await cache.count() == 1
        await cache.clear()
        assert await cache.count() == 0

    @pytest.mark.asyncio
    async def test_count(self):
        cache = MemoryEmbeddingCache()
        assert await cache.count() == 0
        await cache.put("m", "a", [1.0])
        await cache.put("m", "b", [2.0])
        assert await cache.count() == 2


# ---------------------------------------------------------------------------
# SqliteEmbeddingCache
# ---------------------------------------------------------------------------


class TestSqliteEmbeddingCache:
    """Verify SqliteEmbeddingCache persistence and operations."""

    @pytest.mark.asyncio
    async def test_put_get_roundtrip(self, tmp_path: Path):
        cache = SqliteEmbeddingCache(tmp_path / "embed.db")
        await cache.initialize()
        try:
            vec = [0.1, 0.2, 0.3, 0.4]
            await cache.put("model", "hello world", vec)
            result = await cache.get("model", "hello world")
            assert result is not None
            assert len(result) == 4
            # float32 precision: compare within tolerance
            for a, b in zip(result, vec):
                assert abs(a - b) < 1e-6
        finally:
            await cache.close()

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, tmp_path: Path):
        """Data survives cache close + reopen."""
        db_path = tmp_path / "persist.db"

        # Write
        cache1 = SqliteEmbeddingCache(db_path)
        await cache1.initialize()
        await cache1.put("m", "text", [1.0, 2.0])
        await cache1.close()

        # Read from fresh instance
        cache2 = SqliteEmbeddingCache(db_path)
        await cache2.initialize()
        try:
            result = await cache2.get("m", "text")
            assert result is not None
            assert len(result) == 2
        finally:
            await cache2.close()

    @pytest.mark.asyncio
    async def test_clear(self, tmp_path: Path):
        cache = SqliteEmbeddingCache(tmp_path / "clear.db")
        await cache.initialize()
        try:
            await cache.put("m", "a", [1.0])
            await cache.put("m", "b", [2.0])
            assert await cache.count() == 2
            await cache.clear()
            assert await cache.count() == 0
        finally:
            await cache.close()

    @pytest.mark.asyncio
    async def test_batch_operations(self, tmp_path: Path):
        cache = SqliteEmbeddingCache(tmp_path / "batch.db")
        await cache.initialize()
        try:
            await cache.put_batch("m", ["a", "b"], [[1.0], [2.0]])
            results = await cache.get_batch("m", ["a", "c", "b"])
            assert len(results) == 3
            assert results[0] is not None  # "a" cached
            assert results[1] is None  # "c" not cached
            assert results[2] is not None  # "b" cached
        finally:
            await cache.close()

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, tmp_path: Path):
        cache = SqliteEmbeddingCache(tmp_path / "upsert.db")
        await cache.initialize()
        try:
            await cache.put("m", "text", [1.0])
            await cache.put("m", "text", [9.0])
            result = await cache.get("m", "text")
            assert result is not None
            assert abs(result[0] - 9.0) < 1e-6
        finally:
            await cache.close()


# ---------------------------------------------------------------------------
# CachingEmbedProvider — embed() caching
# ---------------------------------------------------------------------------


class TestCachingEmbedProviderEmbed:
    """Verify cache hit/miss behavior for embed()."""

    @pytest.mark.asyncio
    async def test_cache_miss_delegates_to_inner(self):
        """First embed() calls inner provider and stores result."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        result = await provider.embed("hello")

        assert isinstance(result, list)
        assert len(result) > 0
        assert inner.embed_call_count == 1
        assert await cache.count() == 1

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self):
        """Second embed() returns cached vector without calling inner."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        vec1 = await provider.embed("hello")
        assert inner.embed_call_count == 1

        vec2 = await provider.embed("hello")
        assert inner.embed_call_count == 1  # No additional call — cache hit
        assert vec1 == vec2

    @pytest.mark.asyncio
    async def test_cache_miss_batch(self):
        """Batch embed() delegates all texts on full miss."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        results = await provider.embed(["hello", "world"])
        assert isinstance(results, list)
        assert len(results) == 2
        assert inner.embed_call_count == 1  # Single batch call to inner
        assert await cache.count() == 2

    @pytest.mark.asyncio
    async def test_partial_cache_hit_batch(self):
        """Batch with mix of cached + uncached texts."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        # Cache "hello" first
        await provider.embed("hello")
        assert inner.embed_call_count == 1

        # Batch with one hit ("hello"), one miss ("world")
        results = await provider.embed(["hello", "world"])
        assert inner.embed_call_count == 2  # One additional call for "world" only
        assert len(results) == 2
        assert await cache.count() == 2

    @pytest.mark.asyncio
    async def test_different_model_different_key(self):
        """Same text with different model = cache miss."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        await provider.embed("hello")
        assert inner.embed_call_count == 1

        # Change the model on inner config
        inner.config.model = "different-model"
        await provider.embed("hello")
        assert inner.embed_call_count == 2  # Different model → cache miss
        assert await cache.count() == 2

    @pytest.mark.asyncio
    async def test_embed_before_initialize_raises(self):
        """embed() raises ResourceError if called before initialize()."""
        from dataknobs_common.exceptions import ResourceError

        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        # Do NOT call initialize()
        with pytest.raises(ResourceError, match="not initialized"):
            await provider.embed("hello")

    @pytest.mark.asyncio
    async def test_single_text_returns_flat_list(self):
        """Single text input returns list[float], not list[list[float]]."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        result = await provider.embed("hello")
        # Should be flat list of floats
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_batch_text_returns_nested_list(self):
        """Batch text input returns list[list[float]]."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        result = await provider.embed(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)


# ---------------------------------------------------------------------------
# CachingEmbedProvider — passthrough methods
# ---------------------------------------------------------------------------


class TestCachingEmbedProviderPassthrough:
    """Verify that non-embed methods delegate to inner."""

    @pytest.mark.asyncio
    async def test_complete_passthrough(self):
        """complete() delegates to inner unchanged."""
        inner = _create_echo_provider()
        inner.set_responses(["test response"])
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        result = await provider.complete("hello")
        assert result.content == "test response"
        assert inner.call_count == 1

    @pytest.mark.asyncio
    async def test_stream_complete_passthrough(self):
        """stream_complete() delegates to inner unchanged."""
        inner = _create_echo_provider()
        inner.set_responses(["streamed"])
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        chunks = []
        async for chunk in provider.stream_complete("hello"):
            chunks.append(chunk)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_function_call_passthrough(self):
        """function_call() delegates to inner unchanged."""
        import warnings

        from dataknobs_llm import LLMMessage

        inner = _create_echo_provider()
        inner.set_responses(["fn result"])
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        messages = [LLMMessage(role="user", content="test")]
        functions = [{"name": "test_fn", "parameters": {}}]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = await provider.function_call(messages, functions)
        assert result is not None


# ---------------------------------------------------------------------------
# CachingEmbedProvider — config/capabilities forwarding
# ---------------------------------------------------------------------------


class TestCachingEmbedProviderForwarding:
    """Verify config and capabilities delegate to inner."""

    @pytest.mark.asyncio
    async def test_config_forwarding(self):
        """provider.config returns inner's config."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)

        assert provider.config is inner.config
        assert provider.config.model == "test-embed"

    @pytest.mark.asyncio
    async def test_capabilities_forwarding(self):
        """get_capabilities() returns inner's capabilities."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)

        caps = provider.get_capabilities()
        assert caps == inner.get_capabilities()

    @pytest.mark.asyncio
    async def test_validate_model_forwarding(self):
        """validate_model() delegates to inner (handles sync/async)."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)

        result = await provider.validate_model()
        assert result is True


# ---------------------------------------------------------------------------
# CachingEmbedProvider — lifecycle
# ---------------------------------------------------------------------------


class TestCachingEmbedProviderLifecycle:
    """Verify initialize/close manage both inner and cache."""

    @pytest.mark.asyncio
    async def test_initialize_inits_both(self):
        """initialize() initializes inner provider and cache."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)

        assert not provider.is_initialized
        await provider.initialize()
        assert provider.is_initialized
        assert inner.is_initialized

    @pytest.mark.asyncio
    async def test_close_closes_both(self):
        """close() closes inner provider and cache."""
        inner = _create_echo_provider()
        cache = MemoryEmbeddingCache()
        provider = CachingEmbedProvider(inner, cache)
        await provider.initialize()

        await provider.close()
        assert not provider.is_initialized
        assert inner.close_count == 1


# ---------------------------------------------------------------------------
# create_caching_provider factory
# ---------------------------------------------------------------------------


class TestCreateCachingProvider:
    """Verify the convenience factory function."""

    @pytest.mark.asyncio
    async def test_memory_backend(self):
        inner = _create_echo_provider()
        provider = await create_caching_provider(
            inner, cache_backend="memory"
        )
        try:
            assert provider.is_initialized
            result = await provider.embed("test")
            assert isinstance(result, list)
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_sqlite_backend(self, tmp_path: Path):
        inner = _create_echo_provider()
        provider = await create_caching_provider(
            inner, cache_path=tmp_path / "test.db", cache_backend="sqlite"
        )
        try:
            assert provider.is_initialized
            result = await provider.embed("test")
            assert isinstance(result, list)
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_sqlite_requires_path(self):
        inner = _create_echo_provider()
        with pytest.raises(ValueError, match="cache_path is required"):
            await create_caching_provider(inner, cache_backend="sqlite")

    @pytest.mark.asyncio
    async def test_unknown_backend_raises(self):
        inner = _create_echo_provider()
        with pytest.raises(ValueError, match="Unknown cache backend"):
            await create_caching_provider(inner, cache_backend="redis")
