"""Tests for RegistryPoller and HotReloadManager."""

import asyncio

import pytest

from dataknobs_bots.registry import (
    ConfigCachingManager,
    HotReloadManager,
    InMemoryBackend,
    RegistryPoller,
    ReloadMode,
)
from dataknobs_common.events import EventType, InMemoryEventBus


class TestRegistryPoller:
    """Tests for RegistryPoller."""

    @pytest.fixture
    async def backend(self):
        """Create a fresh backend for each test."""
        backend = InMemoryBackend()
        await backend.initialize()
        yield backend
        await backend.close()

    @pytest.fixture
    async def event_bus(self):
        """Create event bus for testing."""
        bus = InMemoryEventBus()
        await bus.connect()
        yield bus
        await bus.close()

    @pytest.fixture
    async def poller(self, backend, event_bus):
        """Create a poller for testing."""
        poller = RegistryPoller(
            backend=backend,
            event_bus=event_bus,
            poll_interval=0.1,  # Fast for tests
        )
        yield poller
        if poller.is_running:
            await poller.stop()

    @pytest.mark.asyncio
    async def test_poll_interval_property(self, poller):
        """Test poll interval getter/setter."""
        assert poller.poll_interval == 0.1

        poller.poll_interval = 1.0
        assert poller.poll_interval == 1.0

        with pytest.raises(ValueError):
            poller.poll_interval = 0

    @pytest.mark.asyncio
    async def test_start_stop(self, poller):
        """Test starting and stopping the poller."""
        assert not poller.is_running

        await poller.start()
        assert poller.is_running

        await poller.stop()
        assert not poller.is_running

    @pytest.mark.asyncio
    async def test_detect_new_registration(self, poller, backend):
        """Test detecting new registrations."""
        # Start with no registrations
        await poller.start()
        await asyncio.sleep(0.05)  # Let it take initial snapshot

        # Add a registration
        await backend.register("test-1", {"key": "value"})

        # Poll and check for changes
        changes = await poller.poll_once()

        assert "test-1" in changes
        assert changes["test-1"] == EventType.CREATED

    @pytest.mark.asyncio
    async def test_detect_deleted_registration(self, poller, backend):
        """Test detecting deleted registrations."""
        # Start with one registration
        await backend.register("test-1", {"key": "value"})
        await poller.start()
        await asyncio.sleep(0.05)

        # Delete the registration
        await backend.unregister("test-1")

        # Poll and check for changes
        changes = await poller.poll_once()

        assert "test-1" in changes
        assert changes["test-1"] == EventType.DELETED

    @pytest.mark.asyncio
    async def test_detect_updated_registration(self, poller, backend):
        """Test detecting updated registrations."""
        # Start with one registration
        await backend.register("test-1", {"key": "value"})
        await poller.start()
        await asyncio.sleep(0.05)

        # Update the registration
        await backend.register("test-1", {"key": "updated"})

        # Poll and check for changes
        changes = await poller.poll_once()

        assert "test-1" in changes
        assert changes["test-1"] == EventType.UPDATED

    @pytest.mark.asyncio
    async def test_no_changes_detected(self, poller, backend):
        """Test that no changes are reported when nothing changed."""
        await backend.register("test-1", {"key": "value"})
        await poller.start()
        await asyncio.sleep(0.05)

        # Poll without making changes
        changes = await poller.poll_once()

        assert len(changes) == 0

    @pytest.mark.asyncio
    async def test_change_callback(self, poller, backend):
        """Test that change callbacks are invoked."""
        callback_events = []

        def callback(instance_id, event_type):
            callback_events.append((instance_id, event_type))

        poller.add_change_callback(callback)

        await poller.start()
        await asyncio.sleep(0.05)

        # Add a registration
        await backend.register("test-1", {"key": "value"})
        await poller.poll_once()

        assert len(callback_events) == 1
        assert callback_events[0] == ("test-1", EventType.CREATED)

    @pytest.mark.asyncio
    async def test_remove_change_callback(self, poller, backend):
        """Test removing change callbacks."""
        callback_events = []

        def callback(instance_id, event_type):
            callback_events.append((instance_id, event_type))

        poller.add_change_callback(callback)
        poller.remove_change_callback(callback)

        await poller.start()
        await backend.register("test-1", {"key": "value"})
        await poller.poll_once()

        assert len(callback_events) == 0

    @pytest.mark.asyncio
    async def test_async_change_callback(self, poller, backend):
        """Test async change callbacks."""
        callback_events = []

        async def async_callback(instance_id, event_type):
            callback_events.append((instance_id, event_type))

        poller.add_change_callback(async_callback)

        await poller.start()
        await backend.register("test-1", {"key": "value"})
        await poller.poll_once()

        assert len(callback_events) == 1

    @pytest.mark.asyncio
    async def test_publishes_events(self, poller, backend, event_bus):
        """Test that changes are published to event bus."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        await event_bus.subscribe("registry:changes", handler)

        await poller.start()
        await asyncio.sleep(0.05)

        await backend.register("test-1", {"key": "value"})
        await poller.poll_once()

        # Allow event to be processed
        await asyncio.sleep(0.01)

        assert len(received_events) == 1
        assert received_events[0].payload["instance_id"] == "test-1"
        assert received_events[0].type == EventType.CREATED


class TestHotReloadManager:
    """Tests for HotReloadManager."""

    @pytest.fixture
    async def backend(self):
        """Create a fresh backend for each test."""
        backend = InMemoryBackend()
        await backend.initialize()
        yield backend
        await backend.close()

    @pytest.fixture
    async def event_bus(self):
        """Create event bus for testing."""
        bus = InMemoryEventBus()
        await bus.connect()
        yield bus
        await bus.close()

    @pytest.fixture
    async def caching_manager(self, backend, event_bus):
        """Create a caching manager for testing."""
        manager = ConfigCachingManager(
            backend=backend,
            event_bus=event_bus,
            cache_ttl=300,
        )
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_default_mode_with_event_bus(self, caching_manager, event_bus):
        """Test that mode defaults to EVENT_DRIVEN when event_bus provided."""
        hot_reload = HotReloadManager(
            caching_manager=caching_manager,
            event_bus=event_bus,
            auto_start=False,
        )

        assert hot_reload.mode == ReloadMode.EVENT_DRIVEN

    @pytest.mark.asyncio
    async def test_default_mode_without_event_bus(self, caching_manager):
        """Test that mode defaults to POLLING without event_bus."""
        hot_reload = HotReloadManager(
            caching_manager=caching_manager,
            auto_start=False,
        )

        assert hot_reload.mode == ReloadMode.POLLING

    @pytest.mark.asyncio
    async def test_explicit_mode(self, caching_manager, event_bus):
        """Test explicit mode setting."""
        hot_reload = HotReloadManager(
            caching_manager=caching_manager,
            event_bus=event_bus,
            mode=ReloadMode.HYBRID,
            auto_start=False,
        )

        assert hot_reload.mode == ReloadMode.HYBRID

    @pytest.mark.asyncio
    async def test_initialize_and_close(self, caching_manager, backend):
        """Test initialization and cleanup."""
        hot_reload = HotReloadManager(
            caching_manager=caching_manager,
            backend=backend,
            mode=ReloadMode.POLLING,
            poll_interval=0.1,
            auto_start=False,
        )

        await hot_reload.initialize()
        assert hot_reload.is_running is False  # auto_start=False

        await hot_reload.start()
        assert hot_reload.is_running is True

        await hot_reload.close()
        assert hot_reload.is_running is False

    @pytest.mark.asyncio
    async def test_auto_start(self, caching_manager, backend):
        """Test auto-start on initialize."""
        hot_reload = HotReloadManager(
            caching_manager=caching_manager,
            backend=backend,
            mode=ReloadMode.POLLING,
            poll_interval=0.1,
            auto_start=True,
        )

        try:
            await hot_reload.initialize()
            assert hot_reload.is_running is True
        finally:
            await hot_reload.close()

    @pytest.mark.asyncio
    async def test_manual_reload(self, caching_manager, backend):
        """Test manual reload of an instance."""
        await backend.register("test-1", {"key": "value"})

        hot_reload = HotReloadManager(
            caching_manager=caching_manager,
            backend=backend,
            mode=ReloadMode.POLLING,
            auto_start=False,
        )
        await hot_reload.initialize()

        try:
            # First access caches the config
            config1 = await caching_manager.get_or_create("test-1")
            assert config1["key"] == "value"

            # Update in backend
            await backend.register("test-1", {"key": "updated"})

            # Cache still has old value
            config2 = await caching_manager.get_or_create("test-1")
            assert config2["key"] == "value"
            assert config1 is config2  # Same cached object

            # Manual reload
            config3 = await hot_reload.reload("test-1")
            assert config3["key"] == "updated"
            assert config3 is not config1  # New object

            assert hot_reload.reload_count == 1
        finally:
            await hot_reload.close()

    @pytest.mark.asyncio
    async def test_reload_all(self, caching_manager, backend):
        """Test reloading all cached instances."""
        await backend.register("test-1", {"key": "v1"})
        await backend.register("test-2", {"key": "v2"})

        hot_reload = HotReloadManager(
            caching_manager=caching_manager,
            backend=backend,
            mode=ReloadMode.POLLING,
            auto_start=False,
        )
        await hot_reload.initialize()

        try:
            # Cache both
            await caching_manager.get_or_create("test-1")
            await caching_manager.get_or_create("test-2")

            assert caching_manager.cache_size == 2

            # Reload all
            count = await hot_reload.reload_all()

            assert count == 2
            assert caching_manager.cache_size == 0
        finally:
            await hot_reload.close()

    @pytest.mark.asyncio
    async def test_reload_callback(self, caching_manager, backend):
        """Test reload callbacks are invoked."""
        await backend.register("test-1", {"key": "value"})

        callback_events = []

        def callback(instance_id):
            callback_events.append(instance_id)

        hot_reload = HotReloadManager(
            caching_manager=caching_manager,
            backend=backend,
            mode=ReloadMode.POLLING,
            auto_start=False,
        )
        hot_reload.add_reload_callback(callback)
        await hot_reload.initialize()

        try:
            await hot_reload.reload("test-1")
            assert callback_events == ["test-1"]
        finally:
            await hot_reload.close()

    @pytest.mark.asyncio
    async def test_get_stats(self, caching_manager, backend):
        """Test statistics reporting."""
        await backend.register("test-1", {"key": "value"})

        hot_reload = HotReloadManager(
            caching_manager=caching_manager,
            backend=backend,
            mode=ReloadMode.POLLING,
            poll_interval=1.0,
            auto_start=False,
        )
        await hot_reload.initialize()

        try:
            await hot_reload.start()

            stats = hot_reload.get_stats()

            assert stats["mode"] == "polling"
            assert stats["running"] is True
            assert stats["reload_count"] == 0
            assert stats["poll_interval"] == 1.0
            assert stats["poller_running"] is True
        finally:
            await hot_reload.close()


class TestHotReloadIntegration:
    """Integration tests for hot-reload with polling."""

    @pytest.mark.asyncio
    async def test_polling_detects_changes_automatically(self):
        """Test that polling mode detects changes automatically."""
        backend = InMemoryBackend()
        await backend.initialize()

        event_bus = InMemoryEventBus()
        await event_bus.connect()

        caching_manager = ConfigCachingManager(
            backend=backend,
            event_bus=event_bus,
            cache_ttl=300,
        )
        await caching_manager.initialize()

        try:
            # Register initial config
            await backend.register("test-bot", {"version": 1})

            # Create hot-reload manager in polling mode
            hot_reload = HotReloadManager(
                caching_manager=caching_manager,
                event_bus=event_bus,
                backend=backend,
                mode=ReloadMode.POLLING,
                poll_interval=0.1,
                auto_start=True,
            )
            await hot_reload.initialize()

            try:
                # Cache the config
                config1 = await caching_manager.get_or_create("test-bot")
                assert config1["version"] == 1

                # Update the config in backend
                await backend.register("test-bot", {"version": 2})

                # Wait for poll cycle to detect and invalidate
                await asyncio.sleep(0.2)

                # Cache should be invalidated, next access gets fresh
                assert not caching_manager.is_cached("test-bot")

                # Get fresh config
                config2 = await caching_manager.get_or_create("test-bot")
                assert config2["version"] == 2

            finally:
                await hot_reload.close()

        finally:
            await caching_manager.close()
            await event_bus.close()
            await backend.close()

    @pytest.mark.asyncio
    async def test_hybrid_mode_setup(self):
        """Test that hybrid mode initializes both event handling and polling."""
        backend = InMemoryBackend()
        await backend.initialize()

        event_bus = InMemoryEventBus()
        await event_bus.connect()

        caching_manager = ConfigCachingManager(
            backend=backend,
            event_bus=event_bus,
        )
        await caching_manager.initialize()

        try:
            hot_reload = HotReloadManager(
                caching_manager=caching_manager,
                event_bus=event_bus,
                backend=backend,
                mode=ReloadMode.HYBRID,
                poll_interval=1.0,
                auto_start=False,
            )
            await hot_reload.initialize()

            try:
                await hot_reload.start()

                stats = hot_reload.get_stats()
                assert stats["mode"] == "hybrid"
                assert stats["poller_running"] is True

            finally:
                await hot_reload.close()

        finally:
            await caching_manager.close()
            await event_bus.close()
            await backend.close()
