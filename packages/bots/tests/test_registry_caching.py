"""Tests for CachingRegistryManager and ConfigCachingManager."""

import asyncio

import pytest

from dataknobs_bots.registry import (
    ConfigCachingManager,
    InMemoryBackend,
    ResolvedConfig,
)
from dataknobs_common.events import Event, EventType, InMemoryEventBus


class TestResolvedConfig:
    """Tests for ResolvedConfig data class."""

    def test_create_resolved_config(self):
        """Test creating a resolved config."""
        config = ResolvedConfig(
            config_id="test-1",
            raw_config={"key": "value"},
            resolved_config={"key": "resolved"},
            environment_name="production",
        )

        assert config.config_id == "test-1"
        assert config.raw_config == {"key": "value"}
        assert config.resolved_config == {"key": "resolved"}
        assert config.environment_name == "production"

    def test_get_value(self):
        """Test getting values from resolved config."""
        config = ResolvedConfig(
            config_id="test-1",
            raw_config={},
            resolved_config={"llm": {"provider": "anthropic"}, "memory": {}},
        )

        assert config.get("llm") == {"provider": "anthropic"}
        assert config.get("missing", "default") == "default"
        assert config["llm"] == {"provider": "anthropic"}

    def test_to_dict(self):
        """Test converting to dict (deep copy)."""
        original = {"nested": {"key": "value"}}
        config = ResolvedConfig(
            config_id="test-1",
            raw_config={},
            resolved_config=original,
        )

        result = config.to_dict()
        assert result == original

        # Verify it's a deep copy
        result["nested"]["key"] = "modified"
        assert config.resolved_config["nested"]["key"] == "value"


class TestConfigCachingManager:
    """Tests for ConfigCachingManager using real implementations."""

    @pytest.fixture
    async def backend(self):
        """Create a fresh backend for each test."""
        backend = InMemoryBackend()
        await backend.initialize()
        yield backend
        await backend.close()

    @pytest.fixture
    async def manager(self, backend):
        """Create a fresh manager for each test."""
        manager = ConfigCachingManager(backend=backend, cache_ttl=10)
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_initialize(self, manager):
        """Test manager initialization."""
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_get_or_create_new_config(self, manager, backend):
        """Test loading a new configuration."""
        await backend.register("test-1", {"llm": {"provider": "anthropic"}})

        resolved = await manager.get_or_create("test-1")

        assert isinstance(resolved, ResolvedConfig)
        assert resolved.config_id == "test-1"
        assert resolved["llm"] == {"provider": "anthropic"}

    @pytest.mark.asyncio
    async def test_get_or_create_cached(self, manager, backend):
        """Test returning cached configuration."""
        await backend.register("test-1", {"key": "value"})

        resolved1 = await manager.get_or_create("test-1")
        resolved2 = await manager.get_or_create("test-1")

        # Should be the same object (cached)
        assert resolved1 is resolved2

    @pytest.mark.asyncio
    async def test_get_or_create_force_refresh(self, manager, backend):
        """Test force refresh bypasses cache."""
        await backend.register("test-1", {"key": "value"})

        resolved1 = await manager.get_or_create("test-1")
        resolved2 = await manager.get_or_create("test-1", force_refresh=True)

        # Should be different objects
        assert resolved1 is not resolved2

    @pytest.mark.asyncio
    async def test_get_or_create_expired(self, backend):
        """Test expired cache creates new config."""
        manager = ConfigCachingManager(backend=backend, cache_ttl=0)
        await manager.initialize()

        try:
            await backend.register("test-1", {"key": "value"})

            resolved1 = await manager.get_or_create("test-1")
            await asyncio.sleep(0.01)  # Let it expire
            resolved2 = await manager.get_or_create("test-1")

            assert resolved1 is not resolved2
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_get_or_create_not_found(self, manager):
        """Test KeyError when registration not found."""
        with pytest.raises(KeyError) as excinfo:
            await manager.get_or_create("nonexistent")

        assert "nonexistent" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_invalidate(self, manager, backend):
        """Test invalidating a cached config."""
        await backend.register("test-1", {"key": "value"})

        await manager.get_or_create("test-1")
        assert manager.is_cached("test-1")

        result = await manager.invalidate("test-1")

        assert result is True
        assert not manager.is_cached("test-1")

    @pytest.mark.asyncio
    async def test_invalidate_not_cached(self, manager):
        """Test invalidating non-cached config returns False."""
        result = await manager.invalidate("not-cached")
        assert result is False

    @pytest.mark.asyncio
    async def test_invalidate_all(self, manager, backend):
        """Test invalidating all cached configs."""
        await backend.register("test-1", {"key": "v1"})
        await backend.register("test-2", {"key": "v2"})

        await manager.get_or_create("test-1")
        await manager.get_or_create("test-2")
        assert manager.cache_size == 2

        count = await manager.invalidate_all()

        assert count == 2
        assert manager.cache_size == 0

    @pytest.mark.asyncio
    async def test_cache_eviction(self, backend):
        """Test LRU-style eviction when cache is full."""
        manager = ConfigCachingManager(
            backend=backend,
            max_cache_size=2,
            cache_ttl=300,
        )
        await manager.initialize()

        try:
            await backend.register("test-1", {"key": "v1"})
            await backend.register("test-2", {"key": "v2"})
            await backend.register("test-3", {"key": "v3"})

            await manager.get_or_create("test-1")
            await asyncio.sleep(0.01)  # Ensure time difference
            await manager.get_or_create("test-2")
            await asyncio.sleep(0.01)
            await manager.get_or_create("test-3")

            # test-1 should have been evicted (oldest)
            assert manager.cache_size == 2
            assert not manager.is_cached("test-1")
            assert manager.is_cached("test-2")
            assert manager.is_cached("test-3")
        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, manager, backend):
        """Test cache statistics."""
        await backend.register("test-1", {"key": "v1"})
        await manager.get_or_create("test-1")

        stats = manager.get_cache_stats()

        assert stats["size"] == 1
        assert stats["max_size"] == manager._max_cache_size
        assert stats["ttl_seconds"] == manager._cache_ttl
        assert stats["valid_count"] == 1
        assert stats["expired_count"] == 0

    @pytest.mark.asyncio
    async def test_get_resolved_config(self, manager, backend):
        """Test convenience method for getting resolved dict."""
        await backend.register("test-1", {"llm": {"provider": "anthropic"}})

        config_dict = await manager.get_resolved_config("test-1")

        assert isinstance(config_dict, dict)
        assert config_dict == {"llm": {"provider": "anthropic"}}

    @pytest.mark.asyncio
    async def test_get_raw_config(self, manager, backend):
        """Test getting raw config bypasses cache."""
        await backend.register("test-1", {"key": "value"})

        raw = await manager.get_raw_config("test-1")

        assert raw == {"key": "value"}
        # Should not be cached (raw bypasses cache)
        assert not manager.is_cached("test-1")

    @pytest.mark.asyncio
    async def test_config_key_extraction(self, backend):
        """Test extracting config by key."""
        manager = ConfigCachingManager(
            backend=backend,
            config_key="bot",
        )
        await manager.initialize()

        try:
            await backend.register(
                "test-1",
                {
                    "metadata": {"name": "Test Bot"},
                    "bot": {"llm": {"provider": "anthropic"}},
                },
            )

            resolved = await manager.get_or_create("test-1")

            # Should have extracted just the "bot" section
            assert resolved["llm"] == {"provider": "anthropic"}
            # Raw should still have full config
            assert resolved.raw_config["metadata"] == {"name": "Test Bot"}
        finally:
            await manager.close()


class TestConfigCachingManagerWithEventBus:
    """Tests for event-driven cache invalidation."""

    @pytest.fixture
    async def event_bus(self):
        """Create event bus for testing."""
        bus = InMemoryEventBus()
        await bus.connect()
        yield bus
        await bus.close()

    @pytest.fixture
    async def manager(self, event_bus):
        """Create manager with event bus."""
        backend = InMemoryBackend()
        await backend.initialize()

        manager = ConfigCachingManager(
            backend=backend,
            event_bus=event_bus,
            event_topic="test:configs",
        )
        await manager.initialize()
        yield manager
        await manager.close()
        await backend.close()

    @pytest.mark.asyncio
    async def test_event_subscription(self, manager):
        """Test that manager subscribes to events."""
        assert manager._subscription is not None

    @pytest.mark.asyncio
    async def test_publish_invalidation(self, manager, event_bus):
        """Test publishing invalidation events."""
        events_received = []

        async def handler(event):
            events_received.append(event)

        await event_bus.subscribe("test:configs", handler)

        await manager.publish_invalidation("test-1", EventType.UPDATED)

        # Allow event to be processed
        await asyncio.sleep(0.01)

        assert len(events_received) == 1
        assert events_received[0].payload["instance_id"] == "test-1"
        assert events_received[0].type == EventType.UPDATED

    @pytest.mark.asyncio
    async def test_event_invalidates_cache(self, manager, event_bus):
        """Test that receiving events invalidates cache."""
        backend = manager._backend
        await backend.register("test-1", {"key": "value"})

        await manager.get_or_create("test-1")
        assert manager.is_cached("test-1")

        # Simulate receiving an invalidation event
        event = Event(
            type=EventType.UPDATED,
            topic="test:configs",
            payload={"instance_id": "test-1"},
        )
        await event_bus.publish("test:configs", event)

        # Allow event to be processed
        await asyncio.sleep(0.01)

        assert not manager.is_cached("test-1")

    @pytest.mark.asyncio
    async def test_ignores_irrelevant_events(self, manager, event_bus):
        """Test that irrelevant events are ignored."""
        backend = manager._backend
        await backend.register("test-1", {"key": "value"})

        await manager.get_or_create("test-1")
        assert manager.is_cached("test-1")

        # Event without instance_id in payload
        event = Event(
            type=EventType.UPDATED,
            topic="test:configs",
            payload={},  # No instance_id
        )
        await event_bus.publish("test:configs", event)

        await asyncio.sleep(0.01)

        # Should still be cached
        assert manager.is_cached("test-1")

    @pytest.mark.asyncio
    async def test_ignores_create_events(self, manager, event_bus):
        """Test that CREATED events don't invalidate cache."""
        backend = manager._backend
        await backend.register("test-1", {"key": "value"})

        await manager.get_or_create("test-1")

        event = Event(
            type=EventType.CREATED,
            topic="test:configs",
            payload={"instance_id": "test-1"},
        )
        await event_bus.publish("test:configs", event)

        await asyncio.sleep(0.01)

        # Should still be cached (CREATED doesn't invalidate)
        assert manager.is_cached("test-1")


class TestConfigCachingManagerDistributed:
    """Integration tests for distributed cache invalidation."""

    @pytest.mark.asyncio
    async def test_distributed_invalidation_simulation(self):
        """Test simulating distributed cache invalidation."""
        # Create shared event bus (simulates Redis/Postgres in production)
        event_bus = InMemoryEventBus()
        await event_bus.connect()

        # Create two managers sharing the same event bus
        backend1 = InMemoryBackend()
        await backend1.initialize()
        backend2 = InMemoryBackend()
        await backend2.initialize()

        manager1 = ConfigCachingManager(
            backend=backend1,
            event_bus=event_bus,
            event_topic="registry",
        )
        manager2 = ConfigCachingManager(
            backend=backend2,
            event_bus=event_bus,
            event_topic="registry",
        )

        await manager1.initialize()
        await manager2.initialize()

        try:
            # Register on both backends (simulating shared database)
            await backend1.register("shared-config", {"key": "v1"})
            await backend2.register("shared-config", {"key": "v1"})

            # Cache on both managers
            await manager1.get_or_create("shared-config")
            await manager2.get_or_create("shared-config")

            assert manager1.is_cached("shared-config")
            assert manager2.is_cached("shared-config")

            # Manager1 publishes invalidation (e.g., config was updated)
            await manager1.publish_invalidation("shared-config", EventType.UPDATED)

            # Allow event propagation
            await asyncio.sleep(0.05)

            # Both caches should be invalidated
            assert not manager1.is_cached("shared-config")
            assert not manager2.is_cached("shared-config")

        finally:
            await manager1.close()
            await manager2.close()
            await backend1.close()
            await backend2.close()
            await event_bus.close()

    @pytest.mark.asyncio
    async def test_config_update_workflow(self):
        """Test a realistic config update workflow."""
        event_bus = InMemoryEventBus()
        await event_bus.connect()

        backend = InMemoryBackend()
        await backend.initialize()

        manager = ConfigCachingManager(
            backend=backend,
            event_bus=event_bus,
            event_topic="configs",
        )
        await manager.initialize()

        try:
            # Initial registration
            await backend.register("my-bot", {"llm": {"provider": "openai"}})

            # Get config (cached)
            config1 = await manager.get_or_create("my-bot")
            assert config1["llm"]["provider"] == "openai"

            # Update registration in backend
            await backend.register("my-bot", {"llm": {"provider": "anthropic"}})

            # Old value still cached
            config2 = await manager.get_or_create("my-bot")
            assert config2["llm"]["provider"] == "openai"
            assert config1 is config2

            # Publish invalidation (e.g., from API endpoint)
            await manager.publish_invalidation("my-bot", EventType.UPDATED)
            await asyncio.sleep(0.01)

            # Now get fresh config
            config3 = await manager.get_or_create("my-bot")
            assert config3["llm"]["provider"] == "anthropic"
            assert config3 is not config1

        finally:
            await manager.close()
            await backend.close()
            await event_bus.close()
