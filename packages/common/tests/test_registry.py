"""Tests for the registry pattern."""

import asyncio
import time
from dataclasses import dataclass
from threading import Thread

import pytest

from dataknobs_common.exceptions import NotFoundError, OperationError
from dataknobs_common.registry import AsyncRegistry, CachedRegistry, Registry


@dataclass
class Tool:
    """Test tool class."""
    name: str
    description: str


class TestRegistry:
    """Test basic Registry functionality."""

    def test_create_registry(self):
        """Test creating a registry."""
        registry = Registry[str]("test_registry")
        assert registry.name == "test_registry"
        assert registry.count() == 0

    def test_register_item(self):
        """Test registering an item."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")

        assert registry.count() == 1
        assert registry.has("key1")
        assert registry.get("key1") == "value1"

    def test_register_with_metadata(self):
        """Test registering with metadata."""
        registry = Registry[Tool]("tools", enable_metrics=True)
        tool = Tool("calculator", "Basic calculator")

        registry.register("calc", tool, metadata={"version": "1.0"})

        metrics = registry.get_metrics("calc")
        assert "registered_at" in metrics
        assert metrics["metadata"]["version"] == "1.0"

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate key raises error."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")

        with pytest.raises(OperationError) as exc_info:
            registry.register("key1", "value2")

        assert "already registered" in str(exc_info.value)

    def test_register_duplicate_with_overwrite(self):
        """Test overwriting existing item."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")
        registry.register("key1", "value2", allow_overwrite=True)

        assert registry.get("key1") == "value2"

    def test_unregister_item(self):
        """Test unregistering an item."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")

        item = registry.unregister("key1")

        assert item == "value1"
        assert registry.count() == 0
        assert not registry.has("key1")

    def test_unregister_nonexistent_raises_error(self):
        """Test unregistering non-existent item raises error."""
        registry = Registry[str]("test")

        with pytest.raises(NotFoundError) as exc_info:
            registry.unregister("nonexistent")

        assert "not found" in str(exc_info.value).lower()

    def test_get_item(self):
        """Test getting an item."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")

        assert registry.get("key1") == "value1"

    def test_get_nonexistent_raises_error(self):
        """Test getting non-existent item raises error."""
        registry = Registry[str]("test")

        with pytest.raises(NotFoundError) as exc_info:
            registry.get("nonexistent")

        error = exc_info.value
        assert "not found" in str(error).lower()
        assert "available_keys" in error.context

    def test_get_optional(self):
        """Test getting item with optional return."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")

        assert registry.get_optional("key1") == "value1"
        assert registry.get_optional("nonexistent") is None

    def test_has_item(self):
        """Test checking if item exists."""
        registry = Registry[str]("test")

        assert not registry.has("key1")

        registry.register("key1", "value1")
        assert registry.has("key1")

    def test_list_keys(self):
        """Test listing all keys."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")
        registry.register("key2", "value2")

        keys = registry.list_keys()
        assert set(keys) == {"key1", "key2"}

    def test_list_items(self):
        """Test listing all items."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")
        registry.register("key2", "value2")

        items = registry.list_items()
        assert set(items) == {"value1", "value2"}

    def test_items(self):
        """Test getting all key-item pairs."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")
        registry.register("key2", "value2")

        items = registry.items()
        assert set(items) == {("key1", "value1"), ("key2", "value2")}

    def test_count(self):
        """Test counting items."""
        registry = Registry[str]("test")
        assert registry.count() == 0

        registry.register("key1", "value1")
        assert registry.count() == 1

        registry.register("key2", "value2")
        assert registry.count() == 2

        registry.unregister("key1")
        assert registry.count() == 1

    def test_clear(self):
        """Test clearing all items."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")
        registry.register("key2", "value2")

        registry.clear()

        assert registry.count() == 0
        assert not registry.has("key1")
        assert not registry.has("key2")

    def test_metrics_disabled_by_default(self):
        """Test that metrics are disabled by default."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")

        metrics = registry.get_metrics()
        assert metrics == {}

    def test_metrics_enabled(self):
        """Test metrics when enabled."""
        registry = Registry[str]("test", enable_metrics=True)
        registry.register("key1", "value1")

        metrics = registry.get_metrics("key1")
        assert "registered_at" in metrics
        assert isinstance(metrics["registered_at"], float)

    def test_thread_safety(self):
        """Test thread-safe operations."""
        registry = Registry[int]("test")

        def register_items(start, end):
            for i in range(start, end):
                registry.register(f"key{i}", i)

        threads = [
            Thread(target=register_items, args=(0, 100)),
            Thread(target=register_items, args=(100, 200)),
            Thread(target=register_items, args=(200, 300)),
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert registry.count() == 300

    def test_generic_typing(self):
        """Test generic type usage."""
        # String registry
        str_registry = Registry[str]("strings")
        str_registry.register("key", "value")
        assert isinstance(str_registry.get("key"), str)

        # Tool registry
        tool_registry = Registry[Tool]("tools")
        tool = Tool("calc", "Calculator")
        tool_registry.register("calc", tool)
        assert isinstance(tool_registry.get("calc"), Tool)

    def test_unregister_with_metrics(self):
        """Test that unregister removes metrics when enabled."""
        registry = Registry[str]("test", enable_metrics=True)
        registry.register("key1", "value1", metadata={"version": "1.0"})

        # Verify metrics exist
        metrics = registry.get_metrics("key1")
        assert "registered_at" in metrics

        # Unregister and verify metrics are removed
        registry.unregister("key1")
        metrics = registry.get_metrics("key1")
        assert metrics == {}

    def test_clear_with_metrics(self):
        """Test that clear removes all metrics when enabled."""
        registry = Registry[str]("test", enable_metrics=True)
        registry.register("key1", "value1")
        registry.register("key2", "value2")

        # Verify metrics exist
        all_metrics = registry.get_metrics()
        assert "key1" in all_metrics
        assert "key2" in all_metrics

        # Clear and verify metrics are removed
        registry.clear()
        all_metrics = registry.get_metrics()
        assert all_metrics == {}

    def test_get_all_metrics(self):
        """Test getting all metrics without specifying a key."""
        registry = Registry[str]("test", enable_metrics=True)
        registry.register("key1", "value1", metadata={"v": "1.0"})
        registry.register("key2", "value2", metadata={"v": "2.0"})

        # Get all metrics
        all_metrics = registry.get_metrics()
        assert "key1" in all_metrics
        assert "key2" in all_metrics
        assert all_metrics["key1"]["metadata"]["v"] == "1.0"
        assert all_metrics["key2"]["metadata"]["v"] == "2.0"

    def test_len_method(self):
        """Test using len() on registry."""
        registry = Registry[str]("test")
        assert len(registry) == 0

        registry.register("key1", "value1")
        assert len(registry) == 1

        registry.register("key2", "value2")
        assert len(registry) == 2

    def test_contains_method(self):
        """Test using 'in' operator on registry."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")

        assert "key1" in registry
        assert "key2" not in registry

    def test_iter_method(self):
        """Test iterating over registry items."""
        registry = Registry[str]("test")
        registry.register("key1", "value1")
        registry.register("key2", "value2")
        registry.register("key3", "value3")

        items = list(registry)
        assert len(items) == 3
        assert "value1" in items
        assert "value2" in items
        assert "value3" in items


class TestCachedRegistry:
    """Test CachedRegistry functionality."""

    def test_create_cached_registry(self):
        """Test creating a cached registry."""
        registry = CachedRegistry[str]("test", cache_ttl=60)
        assert registry.name == "test"
        assert registry.count() == 0

    def test_get_cached_first_time(self):
        """Test getting cached item for first time calls factory."""
        registry = CachedRegistry[str]("test", cache_ttl=60)
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"value{call_count}"

        result = registry.get_cached("key1", factory)

        assert result == "value1"
        assert call_count == 1

    def test_get_cached_returns_cached_value(self):
        """Test that cached value is returned without calling factory."""
        registry = CachedRegistry[str]("test", cache_ttl=60)
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"value{call_count}"

        result1 = registry.get_cached("key1", factory)
        result2 = registry.get_cached("key1", factory)

        assert result1 == result2 == "value1"
        assert call_count == 1  # Factory called only once

    def test_get_cached_expires_after_ttl(self):
        """Test that cache expires after TTL."""
        registry = CachedRegistry[str]("test", cache_ttl=0.1)  # 100ms TTL
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"value{call_count}"

        result1 = registry.get_cached("key1", factory)
        time.sleep(0.15)  # Wait for expiration
        result2 = registry.get_cached("key1", factory)

        assert result1 == "value1"
        assert result2 == "value2"
        assert call_count == 2  # Factory called twice

    def test_get_cached_force_refresh(self):
        """Test force refresh bypasses cache."""
        registry = CachedRegistry[str]("test", cache_ttl=60)
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"value{call_count}"

        result1 = registry.get_cached("key1", factory)
        result2 = registry.get_cached("key1", factory, force_refresh=True)

        assert result1 == "value1"
        assert result2 == "value2"
        assert call_count == 2

    def test_invalidate_cache_single_key(self):
        """Test invalidating single cache entry."""
        registry = CachedRegistry[str]("test", cache_ttl=60)
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"value{call_count}"

        registry.get_cached("key1", factory)
        registry.invalidate_cache("key1")
        registry.get_cached("key1", factory)

        assert call_count == 2

    def test_invalidate_cache_all(self):
        """Test invalidating all cache entries."""
        registry = CachedRegistry[str]("test", cache_ttl=60)

        registry.get_cached("key1", lambda: "value1")
        registry.get_cached("key2", lambda: "value2")

        stats_before = registry.get_cache_stats()
        assert stats_before["size"] == 2

        registry.invalidate_cache()

        stats_after = registry.get_cache_stats()
        assert stats_after["size"] == 0

    def test_cache_stats(self):
        """Test cache statistics."""
        registry = CachedRegistry[str]("test", cache_ttl=60, max_cache_size=100)

        # Initial stats
        stats = registry.get_cache_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

        # Cause cache miss
        registry.get_cached("key1", lambda: "value1")
        stats = registry.get_cache_stats()
        assert stats["misses"] == 1
        assert stats["size"] == 1

        # Cause cache hit
        registry.get_cached("key1", lambda: "value1")
        stats = registry.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_eviction(self):
        """Test LRU cache eviction when max size exceeded."""
        registry = CachedRegistry[str]("test", cache_ttl=60, max_cache_size=10)

        # Fill cache beyond limit
        for i in range(15):
            registry.get_cached(f"key{i}", lambda i=i: f"value{i}")

        stats = registry.get_cache_stats()
        # Cache should have evicted oldest entries
        assert stats["size"] < 15

    def test_cached_registry_inherits_from_registry(self):
        """Test that CachedRegistry has Registry functionality."""
        registry = CachedRegistry[str]("test", cache_ttl=60)

        # Can use regular registry operations
        registry.register("key1", "value1")
        assert registry.get("key1") == "value1"
        assert registry.count() == 1


@pytest.mark.asyncio
class TestAsyncRegistry:
    """Test AsyncRegistry functionality."""

    async def test_create_async_registry(self):
        """Test creating an async registry."""
        registry = AsyncRegistry[str]("test")
        assert registry.name == "test"
        assert await registry.count() == 0

    async def test_async_register_item(self):
        """Test registering an item asynchronously."""
        registry = AsyncRegistry[str]("test")
        await registry.register("key1", "value1")

        assert await registry.count() == 1
        assert await registry.has("key1")
        assert await registry.get("key1") == "value1"

    async def test_async_register_duplicate_raises_error(self):
        """Test that registering duplicate key raises error."""
        registry = AsyncRegistry[str]("test")
        await registry.register("key1", "value1")

        with pytest.raises(OperationError):
            await registry.register("key1", "value2")

    async def test_async_unregister_item(self):
        """Test unregistering an item asynchronously."""
        registry = AsyncRegistry[str]("test")
        await registry.register("key1", "value1")

        item = await registry.unregister("key1")

        assert item == "value1"
        assert await registry.count() == 0

    async def test_async_get_optional(self):
        """Test async get_optional."""
        registry = AsyncRegistry[str]("test")
        await registry.register("key1", "value1")

        assert await registry.get_optional("key1") == "value1"
        assert await registry.get_optional("nonexistent") is None

    async def test_async_list_operations(self):
        """Test async list operations."""
        registry = AsyncRegistry[str]("test")
        await registry.register("key1", "value1")
        await registry.register("key2", "value2")

        keys = await registry.list_keys()
        assert set(keys) == {"key1", "key2"}

        items = await registry.list_items()
        assert set(items) == {"value1", "value2"}

        pairs = await registry.items()
        assert set(pairs) == {("key1", "value1"), ("key2", "value2")}

    async def test_async_clear(self):
        """Test async clear."""
        registry = AsyncRegistry[str]("test")
        await registry.register("key1", "value1")
        await registry.register("key2", "value2")

        await registry.clear()

        assert await registry.count() == 0

    async def test_async_metrics(self):
        """Test async metrics."""
        registry = AsyncRegistry[str]("test", enable_metrics=True)
        await registry.register("key1", "value1")

        metrics = await registry.get_metrics("key1")
        assert "registered_at" in metrics

    async def test_async_metrics_disabled(self):
        """Test async metrics when disabled."""
        registry = AsyncRegistry[str]("test", enable_metrics=False)
        await registry.register("key1", "value1")

        metrics = await registry.get_metrics()
        assert metrics == {}

    async def test_async_concurrent_operations(self):
        """Test concurrent async operations."""
        registry = AsyncRegistry[int]("test")

        async def register_items(start, end):
            for i in range(start, end):
                await registry.register(f"key{i}", i)

        # Run concurrent registrations
        await asyncio.gather(
            register_items(0, 100),
            register_items(100, 200),
            register_items(200, 300),
        )

        assert await registry.count() == 300

    async def test_async_unregister_nonexistent_raises_error(self):
        """Test that unregistering non-existent item raises error."""
        registry = AsyncRegistry[str]("test")

        with pytest.raises(NotFoundError) as exc_info:
            await registry.unregister("nonexistent")

        assert "not found" in str(exc_info.value).lower()

    async def test_async_unregister_with_metrics(self):
        """Test that unregister removes metrics when enabled."""
        registry = AsyncRegistry[str]("test", enable_metrics=True)
        await registry.register("key1", "value1", metadata={"version": "1.0"})

        # Verify metrics exist
        metrics = await registry.get_metrics("key1")
        assert "registered_at" in metrics

        # Unregister and verify metrics are removed
        await registry.unregister("key1")
        metrics = await registry.get_metrics("key1")
        assert metrics == {}

    async def test_async_get_nonexistent_raises_error(self):
        """Test that getting non-existent item raises error."""
        registry = AsyncRegistry[str]("test")

        with pytest.raises(NotFoundError) as exc_info:
            await registry.get("nonexistent")

        error = exc_info.value
        assert "not found" in str(error).lower()
        assert "available_keys" in error.context

    async def test_async_clear_with_metrics(self):
        """Test that clear removes all metrics when enabled."""
        registry = AsyncRegistry[str]("test", enable_metrics=True)
        await registry.register("key1", "value1")
        await registry.register("key2", "value2")

        # Verify metrics exist
        all_metrics = await registry.get_metrics()
        assert "key1" in all_metrics
        assert "key2" in all_metrics

        # Clear and verify metrics are removed
        await registry.clear()
        all_metrics = await registry.get_metrics()
        assert all_metrics == {}

    async def test_async_get_all_metrics(self):
        """Test getting all metrics without specifying a key."""
        registry = AsyncRegistry[str]("test", enable_metrics=True)
        await registry.register("key1", "value1", metadata={"v": "1.0"})
        await registry.register("key2", "value2", metadata={"v": "2.0"})

        # Get all metrics
        all_metrics = await registry.get_metrics()
        assert "key1" in all_metrics
        assert "key2" in all_metrics
        assert all_metrics["key1"]["metadata"]["v"] == "1.0"
        assert all_metrics["key2"]["metadata"]["v"] == "2.0"

    async def test_async_len_method(self):
        """Test using len() on async registry."""
        registry = AsyncRegistry[str]("test")
        # Note: __len__ is synchronous
        assert len(registry) == 0

        await registry.register("key1", "value1")
        assert len(registry) == 1

        await registry.register("key2", "value2")
        assert len(registry) == 2

    async def test_async_contains_method(self):
        """Test using 'in' operator on async registry."""
        registry = AsyncRegistry[str]("test")
        await registry.register("key1", "value1")

        # Note: __contains__ is synchronous
        assert "key1" in registry
        assert "key2" not in registry

    async def test_async_iter_method(self):
        """Test iterating over async registry items."""
        registry = AsyncRegistry[str]("test")
        await registry.register("key1", "value1")
        await registry.register("key2", "value2")
        await registry.register("key3", "value3")

        # Note: __iter__ is synchronous but returns a snapshot
        items = list(registry)
        assert len(items) == 3
        assert "value1" in items
        assert "value2" in items
        assert "value3" in items


class TestCustomRegistry:
    """Test creating custom registries."""

    def test_custom_registry_extension(self):
        """Test extending Registry for specific use cases."""
        class ToolRegistry(Registry[Tool]):
            def __init__(self):
                super().__init__("tools", enable_metrics=True)

            def register_tool(self, tool: Tool):
                self.register(
                    tool.name,
                    tool,
                    metadata={"description": tool.description}
                )

            def get_by_prefix(self, prefix: str):
                return [
                    tool for key, tool in self.items()
                    if key.startswith(prefix)
                ]

        registry = ToolRegistry()
        tool1 = Tool("calc_add", "Addition")
        tool2 = Tool("calc_sub", "Subtraction")
        tool3 = Tool("search", "Search")

        registry.register_tool(tool1)
        registry.register_tool(tool2)
        registry.register_tool(tool3)

        calc_tools = registry.get_by_prefix("calc_")
        assert len(calc_tools) == 2
        assert tool1 in calc_tools
        assert tool2 in calc_tools
