"""Tests for BotRegistry with pluggable backends."""

import asyncio

import pytest

from dataknobs_bots import BotContext
from dataknobs_bots.bot import (
    BotRegistry,
    InMemoryBotRegistry,
    create_memory_registry,
)
from dataknobs_bots.registry import InMemoryBackend, PortabilityError


class TestBotRegistry:
    """Tests for BotRegistry multi-tenant support."""

    @pytest.fixture
    async def registry(self):
        """Create a fresh registry for each test."""
        reg = InMemoryBotRegistry(validate_on_register=False)
        await reg.initialize()
        yield reg
        await reg.close()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test registry initialization."""
        backend = InMemoryBackend()
        registry = BotRegistry(backend=backend, cache_ttl=300, max_cache_size=100)

        assert registry.cache_ttl == 300
        assert registry.max_cache_size == 100
        assert len(registry._cache) == 0
        assert registry.backend is backend

    @pytest.mark.asyncio
    async def test_initialize_and_close(self):
        """Test initialize and close lifecycle."""
        registry = InMemoryBotRegistry()

        await registry.initialize()
        await registry.register(
            "test-bot",
            {"llm": {"provider": "echo"}},
            skip_validation=True,
        )

        await registry.close()
        assert len(registry._cache) == 0

    @pytest.mark.asyncio
    async def test_get_bot(self, registry):
        """Test getting a bot for a client."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        bot = await registry.get_bot("client-1")

        assert bot is not None
        assert len(registry._cache) == 1

    @pytest.mark.asyncio
    async def test_bot_caching(self, registry):
        """Test that bots are cached."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Get bot twice
        bot1 = await registry.get_bot("client-1")
        bot2 = await registry.get_bot("client-1")

        # Should be same instance (cached)
        assert bot1 is bot2
        assert len(registry._cache) == 1

    @pytest.mark.asyncio
    async def test_cache_expiry(self):
        """Test cache expiration."""
        registry = InMemoryBotRegistry(
            cache_ttl=0,  # Immediate expiry
            validate_on_register=False,
        )
        await registry.initialize()

        try:
            await registry.register(
                "client-1",
                {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
            )

            # Get bot
            bot1 = await registry.get_bot("client-1")

            # Wait a tiny bit for cache to expire
            await asyncio.sleep(0.01)

            # Get bot again - should create new instance
            bot2 = await registry.get_bot("client-1")

            # Should be different instances
            assert bot1 is not bot2
        finally:
            await registry.close()

    @pytest.mark.asyncio
    async def test_force_refresh(self, registry):
        """Test forcing cache refresh."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Get bot and cache it
        bot1 = await registry.get_bot("client-1")

        # Force refresh
        bot2 = await registry.get_bot("client-1", force_refresh=True)

        # Should be different instances
        assert bot1 is not bot2

    @pytest.mark.asyncio
    async def test_multiple_clients(self, registry):
        """Test managing multiple clients."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )
        await registry.register(
            "client-2",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "memory": {"type": "buffer", "max_messages": 5},
            },
        )

        # Get bots for different clients
        bot1 = await registry.get_bot("client-1")
        bot2 = await registry.get_bot("client-2")

        assert bot1 is not bot2
        assert bot1.memory is None
        assert bot2.memory is not None
        assert len(registry._cache) == 2

    @pytest.mark.asyncio
    async def test_register_returns_registration(self, registry):
        """Test that register returns a Registration object."""
        reg = await registry.register(
            "new-client",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        assert reg.bot_id == "new-client"
        assert reg.status == "active"
        assert reg.created_at is not None

    @pytest.mark.asyncio
    async def test_register_invalidates_cache(self, registry):
        """Test that registering updates invalidate cache."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Get and cache bot
        bot1 = await registry.get_bot("client-1")

        # Update configuration
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "memory": {"type": "buffer", "max_messages": 10},
            },
        )

        # Get bot again - should be new instance with new config
        bot2 = await registry.get_bot("client-1")

        assert bot1 is not bot2
        assert bot2.memory is not None

    @pytest.mark.asyncio
    async def test_unregister(self, registry):
        """Test unregistering a bot."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Get and cache bot
        await registry.get_bot("client-1")
        assert len(registry._cache) == 1

        # Unregister
        result = await registry.unregister("client-1")

        assert result is True
        assert len(registry._cache) == 0

        # Should not be able to get bot anymore
        with pytest.raises(KeyError, match="No bot configuration found"):
            await registry.get_bot("client-1")

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, registry):
        """Test unregistering a bot that doesn't exist."""
        result = await registry.unregister("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_deactivate(self, registry):
        """Test deactivating a bot."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Get and cache bot
        await registry.get_bot("client-1")

        # Deactivate
        result = await registry.deactivate("client-1")

        assert result is True
        assert len(registry._cache) == 0

        # Exists should return False
        assert await registry.exists("client-1") is False

        # Registration should still be retrievable
        reg = await registry.get_registration("client-1")
        assert reg is not None
        assert reg.status == "inactive"

    @pytest.mark.asyncio
    async def test_exists(self, registry):
        """Test checking if bot exists."""
        assert await registry.exists("client-1") is False

        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        assert await registry.exists("client-1") is True

    @pytest.mark.asyncio
    async def test_list_bots(self, registry):
        """Test listing all bot IDs."""
        await registry.register(
            "client-1",
            {"llm": {"provider": "echo"}},
        )
        await registry.register(
            "client-2",
            {"llm": {"provider": "echo"}},
        )

        bots = await registry.list_bots()

        assert len(bots) == 2
        assert "client-1" in bots
        assert "client-2" in bots

    @pytest.mark.asyncio
    async def test_count(self, registry):
        """Test counting registrations."""
        assert await registry.count() == 0

        await registry.register(
            "client-1",
            {"llm": {"provider": "echo"}},
        )
        await registry.register(
            "client-2",
            {"llm": {"provider": "echo"}},
        )

        assert await registry.count() == 2

    @pytest.mark.asyncio
    async def test_get_cached_bots(self, registry):
        """Test getting list of cached bots."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )
        await registry.register(
            "client-2",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Get bots to cache them
        await registry.get_bot("client-1")
        await registry.get_bot("client-2")

        # Check cached bots
        cached = registry.get_cached_bots()
        assert "client-1" in cached
        assert "client-2" in cached
        assert len(cached) == 2

    @pytest.mark.asyncio
    async def test_clear_cache(self, registry):
        """Test clearing all cached bots."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )
        await registry.register(
            "client-2",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Cache some bots
        await registry.get_bot("client-1")
        await registry.get_bot("client-2")
        assert len(registry._cache) == 2

        # Clear cache
        registry.clear_cache()
        assert len(registry._cache) == 0

        # Registrations should still exist
        assert await registry.count() == 2

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test cache eviction when max size reached."""
        registry = InMemoryBotRegistry(
            max_cache_size=10,
            validate_on_register=False,
        )
        await registry.initialize()

        try:
            # Register many bots
            for i in range(15):
                await registry.register(
                    f"client-{i}",
                    {
                        "llm": {"provider": "echo", "model": "test"},
                        "conversation_storage": {"backend": "memory"},
                    },
                )

            # Load more bots than cache can hold
            for i in range(15):
                await registry.get_bot(f"client-{i}")

            # Cache should have evicted oldest entries
            assert len(registry._cache) <= 10
        finally:
            await registry.close()

    @pytest.mark.asyncio
    async def test_concurrent_access(self, registry):
        """Test concurrent access to registry."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Concurrent requests
        async def get_and_use_bot():
            bot = await registry.get_bot("client-1")
            context = BotContext(
                conversation_id=f"conv-{asyncio.current_task().get_name()}",
                client_id="client-1",
            )
            return await bot.chat("Hello", context)

        # Run multiple concurrent requests
        responses = await asyncio.gather(*[get_and_use_bot() for _ in range(5)])

        # All should succeed
        assert len(responses) == 5
        assert all(r is not None for r in responses)

    @pytest.mark.asyncio
    async def test_missing_client(self, registry):
        """Test error handling for missing client."""
        with pytest.raises(KeyError, match="No bot configuration found"):
            await registry.get_bot("non-existent")

    @pytest.mark.asyncio
    async def test_get_config(self, registry):
        """Test getting stored configuration."""
        config = {
            "llm": {"provider": "echo", "model": "test"},
            "conversation_storage": {"backend": "memory"},
        }
        await registry.register("client-1", config)

        stored = await registry.get_config("client-1")
        assert stored == config

    @pytest.mark.asyncio
    async def test_get_config_nonexistent(self, registry):
        """Test getting config for nonexistent bot."""
        config = await registry.get_config("nonexistent")
        assert config is None

    @pytest.mark.asyncio
    async def test_get_registration(self, registry):
        """Test getting full registration."""
        await registry.register(
            "client-1",
            {"llm": {"provider": "echo"}},
        )

        reg = await registry.get_registration("client-1")

        assert reg is not None
        assert reg.bot_id == "client-1"
        assert reg.status == "active"
        assert reg.created_at is not None


class TestBotRegistryPortability:
    """Tests for portability validation in BotRegistry."""

    @pytest.fixture
    async def registry_with_validation(self):
        """Create registry with portability validation enabled."""
        reg = InMemoryBotRegistry(validate_on_register=True)
        await reg.initialize()
        yield reg
        await reg.close()

    @pytest.mark.asyncio
    async def test_portable_config_accepted(self, registry_with_validation):
        """Test that portable configs are accepted."""
        # Config without local paths or localhost URLs
        reg = await registry_with_validation.register(
            "test-bot",
            {
                "bot": {
                    "llm": {"provider": "openai", "model": "gpt-4"},
                    "conversation_storage": {"backend": "memory"},
                }
            },
        )
        assert reg.bot_id == "test-bot"

    @pytest.mark.asyncio
    async def test_resource_references_accepted(self, registry_with_validation):
        """Test that configs with $resource refs are accepted."""
        reg = await registry_with_validation.register(
            "test-bot",
            {
                "bot": {
                    "llm": {"$resource": "default", "type": "llm_providers"},
                    "conversation_storage": {"$resource": "db", "type": "databases"},
                }
            },
        )
        assert reg.bot_id == "test-bot"

    @pytest.mark.asyncio
    async def test_local_path_rejected(self, registry_with_validation):
        """Test that configs with local paths are rejected."""
        with pytest.raises(PortabilityError, match="macOS home directory"):
            await registry_with_validation.register(
                "test-bot",
                {
                    "storage": {"path": "/Users/dev/data"},
                },
            )

    @pytest.mark.asyncio
    async def test_localhost_rejected(self, registry_with_validation):
        """Test that configs with localhost are rejected."""
        with pytest.raises(PortabilityError, match="localhost"):
            await registry_with_validation.register(
                "test-bot",
                {
                    "database": {"host": "localhost:5432"},
                },
            )

    @pytest.mark.asyncio
    async def test_skip_validation(self, registry_with_validation):
        """Test skipping portability validation."""
        # Should succeed even with local path
        reg = await registry_with_validation.register(
            "test-bot",
            {
                "storage": {"path": "/Users/dev/data"},
            },
            skip_validation=True,
        )
        assert reg.bot_id == "test-bot"

    @pytest.mark.asyncio
    async def test_validation_disabled_in_registry(self):
        """Test registry-level validation disable."""
        registry = InMemoryBotRegistry(validate_on_register=False)
        await registry.initialize()

        try:
            # Should succeed even with local path
            reg = await registry.register(
                "test-bot",
                {
                    "storage": {"path": "/Users/dev/data"},
                },
            )
            assert reg.bot_id == "test-bot"
        finally:
            await registry.close()


class TestBotRegistryLegacyMethods:
    """Tests for legacy compatibility methods."""

    @pytest.fixture
    async def registry(self):
        """Create a fresh registry for each test."""
        reg = InMemoryBotRegistry(validate_on_register=False)
        await reg.initialize()
        yield reg
        await reg.close()

    @pytest.mark.asyncio
    async def test_register_client(self, registry):
        """Test legacy register_client method."""
        # Legacy method still works
        await registry.register_client(
            "new-client",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Should be able to get bot
        bot = await registry.get_bot("new-client")
        assert bot is not None

    @pytest.mark.asyncio
    async def test_remove_client(self, registry):
        """Test legacy remove_client method."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Cache the bot
        await registry.get_bot("client-1")

        # Legacy method still works
        await registry.remove_client("client-1")

        # Should be removed
        with pytest.raises(KeyError):
            await registry.get_bot("client-1")

    @pytest.mark.asyncio
    async def test_get_cached_clients(self, registry):
        """Test legacy get_cached_clients method."""
        await registry.register(
            "client-1",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        await registry.get_bot("client-1")

        # Legacy method still works
        cached = registry.get_cached_clients()
        assert "client-1" in cached


class TestCreateMemoryRegistry:
    """Tests for create_memory_registry factory."""

    @pytest.mark.asyncio
    async def test_creates_with_inmemory_backend(self):
        """Test factory creates registry with InMemoryBackend."""
        registry = create_memory_registry()
        await registry.initialize()

        try:
            assert isinstance(registry.backend, InMemoryBackend)
        finally:
            await registry.close()

    @pytest.mark.asyncio
    async def test_passes_through_options(self):
        """Test factory passes options to BotRegistry."""
        registry = create_memory_registry(
            cache_ttl=600,
            max_cache_size=500,
            validate_on_register=False,
            config_key="custom",
        )

        assert registry.cache_ttl == 600
        assert registry.max_cache_size == 500
        assert registry._validate_on_register is False
        assert registry._config_key == "custom"

    @pytest.mark.asyncio
    async def test_for_testing_workflow(self):
        """Test typical testing workflow."""
        registry = create_memory_registry(validate_on_register=False)
        await registry.initialize()

        try:
            # Register test bot
            await registry.register(
                "test-bot",
                {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
            )

            # Get bot
            bot = await registry.get_bot("test-bot")
            assert bot is not None

            # Count
            assert await registry.count() == 1
        finally:
            await registry.close()


class TestInMemoryBotRegistry:
    """Tests for InMemoryBotRegistry class."""

    @pytest.mark.asyncio
    async def test_uses_inmemory_backend(self):
        """Test InMemoryBotRegistry uses InMemoryBackend."""
        registry = InMemoryBotRegistry()
        assert isinstance(registry.backend, InMemoryBackend)

    @pytest.mark.asyncio
    async def test_passes_through_options(self):
        """Test constructor passes options to base class."""
        registry = InMemoryBotRegistry(
            cache_ttl=600,
            max_cache_size=500,
            validate_on_register=False,
            config_key="custom",
        )

        assert registry.cache_ttl == 600
        assert registry.max_cache_size == 500
        assert registry._validate_on_register is False
        assert registry._config_key == "custom"

    @pytest.mark.asyncio
    async def test_repr(self):
        """Test InMemoryBotRegistry repr."""
        registry = InMemoryBotRegistry()
        repr_str = repr(registry)

        assert "InMemoryBotRegistry" in repr_str
        assert "cached=0" in repr_str
        # Should not show backend (it's implied by the class name)
        assert "InMemoryBackend" not in repr_str

    @pytest.mark.asyncio
    async def test_repr_with_cached_bots(self):
        """Test repr shows cached count."""
        registry = InMemoryBotRegistry(validate_on_register=False)
        await registry.initialize()

        try:
            await registry.register(
                "bot-1",
                {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
            )
            await registry.get_bot("bot-1")

            repr_str = repr(registry)
            assert "InMemoryBotRegistry" in repr_str
            assert "cached=1" in repr_str
        finally:
            await registry.close()

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clear() removes all registrations and cache."""
        registry = InMemoryBotRegistry(validate_on_register=False)
        await registry.initialize()

        try:
            # Register and cache some bots
            await registry.register(
                "bot-1",
                {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
            )
            await registry.register(
                "bot-2",
                {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
            )
            await registry.get_bot("bot-1")
            await registry.get_bot("bot-2")

            assert await registry.count() == 2
            assert len(registry.get_cached_bots()) == 2

            # Clear everything
            await registry.clear()

            assert await registry.count() == 0
            assert len(registry.get_cached_bots()) == 0
        finally:
            await registry.close()


class TestCreateMemoryRegistry:
    """Tests for create_memory_registry factory."""

    @pytest.mark.asyncio
    async def test_creates_inmemory_bot_registry(self):
        """Test factory creates InMemoryBotRegistry."""
        registry = create_memory_registry()
        await registry.initialize()

        try:
            assert isinstance(registry, InMemoryBotRegistry)
            assert isinstance(registry.backend, InMemoryBackend)
        finally:
            await registry.close()

    @pytest.mark.asyncio
    async def test_passes_through_options(self):
        """Test factory passes options to InMemoryBotRegistry."""
        registry = create_memory_registry(
            cache_ttl=600,
            max_cache_size=500,
            validate_on_register=False,
            config_key="custom",
        )

        assert registry.cache_ttl == 600
        assert registry.max_cache_size == 500
        assert registry._validate_on_register is False
        assert registry._config_key == "custom"

    @pytest.mark.asyncio
    async def test_for_testing_workflow(self):
        """Test typical testing workflow."""
        registry = create_memory_registry(validate_on_register=False)
        await registry.initialize()

        try:
            # Register test bot
            await registry.register(
                "test-bot",
                {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
            )

            # Get bot
            bot = await registry.get_bot("test-bot")
            assert bot is not None

            # Count
            assert await registry.count() == 1
        finally:
            await registry.close()


class TestBotRegistryRepr:
    """Tests for BotRegistry string representation."""

    @pytest.mark.asyncio
    async def test_repr_without_environment(self):
        """Test repr without environment."""
        backend = InMemoryBackend()
        registry = BotRegistry(backend=backend)
        repr_str = repr(registry)

        assert "BotRegistry" in repr_str
        assert "InMemoryBackend" in repr_str
        assert "cached=0" in repr_str

    @pytest.mark.asyncio
    async def test_repr_with_cached_bots(self):
        """Test repr shows cached count."""
        backend = InMemoryBackend()
        registry = BotRegistry(backend=backend, validate_on_register=False)
        await registry.initialize()

        try:
            await registry.register(
                "bot-1",
                {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
            )
            await registry.get_bot("bot-1")

            repr_str = repr(registry)
            assert "BotRegistry" in repr_str
            assert "cached=1" in repr_str
        finally:
            await registry.close()
