"""Tests for BotRegistry."""

import asyncio

import pytest

from dataknobs_bots import BotContext
from dataknobs_bots.bot import BotRegistry
from dataknobs_config import Config


class TestBotRegistry:
    """Tests for BotRegistry multi-tenant support."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test registry initialization."""
        config = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        registry = BotRegistry(config, cache_ttl=300, max_cache_size=100)
        assert registry.cache_ttl == 300
        assert registry.max_cache_size == 100
        assert len(registry._cache) == 0

    @pytest.mark.asyncio
    async def test_get_bot(self):
        """Test getting a bot for a client."""
        config = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        registry = BotRegistry(config)
        bot = await registry.get_bot("client-1")

        assert bot is not None
        assert len(registry._cache) == 1

    @pytest.mark.asyncio
    async def test_bot_caching(self):
        """Test that bots are cached."""
        config = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        registry = BotRegistry(config, cache_ttl=300)

        # Get bot twice
        bot1 = await registry.get_bot("client-1")
        bot2 = await registry.get_bot("client-1")

        # Should be same instance (cached)
        assert bot1 is bot2
        assert len(registry._cache) == 1

    @pytest.mark.asyncio
    async def test_cache_expiry(self):
        """Test cache expiration."""
        config = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        # Very short TTL for testing
        registry = BotRegistry(config, cache_ttl=0)

        # Get bot
        bot1 = await registry.get_bot("client-1")

        # Wait a tiny bit for cache to expire
        await asyncio.sleep(0.01)

        # Get bot again - should create new instance
        bot2 = await registry.get_bot("client-1")

        # Should be different instances
        assert bot1 is not bot2

    @pytest.mark.asyncio
    async def test_force_refresh(self):
        """Test forcing cache refresh."""
        config = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        registry = BotRegistry(config, cache_ttl=300)

        # Get bot and cache it
        bot1 = await registry.get_bot("client-1")

        # Force refresh
        bot2 = await registry.get_bot("client-1", force_refresh=True)

        # Should be different instances
        assert bot1 is not bot2

    @pytest.mark.asyncio
    async def test_multiple_clients(self):
        """Test managing multiple clients."""
        config = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
                "client-2": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                    "memory": {"type": "buffer", "max_messages": 5},
                },
            }
        }

        registry = BotRegistry(config)

        # Get bots for different clients
        bot1 = await registry.get_bot("client-1")
        bot2 = await registry.get_bot("client-2")

        assert bot1 is not bot2
        assert bot1.memory is None
        assert bot2.memory is not None
        assert len(registry._cache) == 2

    @pytest.mark.asyncio
    async def test_register_client(self):
        """Test registering a new client dynamically."""
        config = {"bots": {}}

        registry = BotRegistry(config)

        # Register new client
        await registry.register_client(
            "new-client",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Should be able to get bot for new client
        bot = await registry.get_bot("new-client")
        assert bot is not None

    @pytest.mark.asyncio
    async def test_register_client_invalidates_cache(self):
        """Test that registering updates invalidate cache."""
        config = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        registry = BotRegistry(config)

        # Get and cache bot
        bot1 = await registry.get_bot("client-1")

        # Update configuration
        await registry.register_client(
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
    async def test_remove_client(self):
        """Test removing a client."""
        config = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        registry = BotRegistry(config)

        # Get and cache bot
        await registry.get_bot("client-1")
        assert len(registry._cache) == 1

        # Remove client
        await registry.remove_client("client-1")

        # Cache should be cleared
        assert len(registry._cache) == 0

    @pytest.mark.asyncio
    async def test_get_cached_clients(self):
        """Test getting list of cached clients."""
        config = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
                "client-2": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
            }
        }

        registry = BotRegistry(config)

        # Get bots
        await registry.get_bot("client-1")
        await registry.get_bot("client-2")

        # Check cached clients
        cached = registry.get_cached_clients()
        assert "client-1" in cached
        assert "client-2" in cached
        assert len(cached) == 2

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing all cached bots."""
        config = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
                "client-2": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
            }
        }

        registry = BotRegistry(config)

        # Cache some bots
        await registry.get_bot("client-1")
        await registry.get_bot("client-2")
        assert len(registry._cache) == 2

        # Clear cache
        registry.clear_cache()
        assert len(registry._cache) == 0

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test cache eviction when max size reached."""
        config = {
            "bots": {
                f"client-{i}": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
                for i in range(15)
            }
        }

        # Small cache size for testing
        registry = BotRegistry(config, max_cache_size=10)

        # Load more bots than cache can hold
        for i in range(15):
            await registry.get_bot(f"client-{i}")

        # Cache should have evicted oldest entries
        assert len(registry._cache) <= 10

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent access to registry."""
        config = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        registry = BotRegistry(config)

        # Concurrent requests
        async def get_and_use_bot():
            bot = await registry.get_bot("client-1")
            context = BotContext(
                conversation_id=f"conv-{asyncio.current_task().get_name()}",
                client_id="client-1",
            )
            return await bot.chat("Hello", context)

        # Run multiple concurrent requests
        responses = await asyncio.gather(
            *[get_and_use_bot() for _ in range(5)]
        )

        # All should succeed
        assert len(responses) == 5
        assert all(r is not None for r in responses)

    @pytest.mark.asyncio
    async def test_missing_client(self):
        """Test error handling for missing client."""
        config = {"bots": {}}

        registry = BotRegistry(config)

        # Try to get non-existent client
        with pytest.raises(KeyError, match="No bot configuration found"):
            await registry.get_bot("non-existent")


class TestBotRegistryWithConfig:
    """Tests for BotRegistry using Config objects instead of plain dicts.

    These tests ensure the registry works correctly with the Config class,
    which uses a different internal structure (Dict[str, List[Dict]]).
    """

    @pytest.mark.asyncio
    async def test_get_bot_with_config_object(self):
        """Test getting a bot using a Config object."""
        # Config transforms {"bots": {"client-1": {...}}}
        # into {"bots": [{"client-1": {...}}]}
        config_dict = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        config = Config(config_dict)
        registry = BotRegistry(config)
        bot = await registry.get_bot("client-1")

        assert bot is not None
        assert len(registry._cache) == 1

    @pytest.mark.asyncio
    async def test_bot_caching_with_config_object(self):
        """Test bot caching with Config objects."""
        config_dict = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        config = Config(config_dict)
        registry = BotRegistry(config, cache_ttl=300)

        # Get bot twice
        bot1 = await registry.get_bot("client-1")
        bot2 = await registry.get_bot("client-1")

        # Should be same instance (cached)
        assert bot1 is bot2
        assert len(registry._cache) == 1

    @pytest.mark.asyncio
    async def test_register_client_with_config_object(self):
        """Test registering a new client with Config object."""
        config = Config({"bots": {}})
        registry = BotRegistry(config)

        # Register new client
        await registry.register_client(
            "new-client",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Should be able to get bot for new client
        bot = await registry.get_bot("new-client")
        assert bot is not None

    @pytest.mark.asyncio
    async def test_register_client_update_with_config_object(self):
        """Test updating existing client with Config object."""
        config_dict = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        config = Config(config_dict)
        registry = BotRegistry(config)

        # Get and cache bot
        bot1 = await registry.get_bot("client-1")
        assert bot1.memory is None

        # Update configuration
        await registry.register_client(
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
    async def test_remove_client_with_config_object(self):
        """Test removing a client with Config object."""
        config_dict = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                }
            }
        }

        config = Config(config_dict)
        registry = BotRegistry(config)

        # Get and cache bot
        await registry.get_bot("client-1")
        assert len(registry._cache) == 1

        # Remove client
        await registry.remove_client("client-1")

        # Cache should be cleared
        assert len(registry._cache) == 0

        # Should not be able to get bot anymore
        with pytest.raises(KeyError, match="No bot configuration found"):
            await registry.get_bot("client-1")

    @pytest.mark.asyncio
    async def test_multiple_clients_with_config_object(self):
        """Test managing multiple clients with Config object."""
        config_dict = {
            "bots": {
                "client-1": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                },
                "client-2": {
                    "llm": {"provider": "echo", "model": "test"},
                    "conversation_storage": {"backend": "memory"},
                    "memory": {"type": "buffer", "max_messages": 5},
                },
            }
        }

        config = Config(config_dict)
        registry = BotRegistry(config)

        # Get bots for different clients
        bot1 = await registry.get_bot("client-1")
        bot2 = await registry.get_bot("client-2")

        assert bot1 is not bot2
        assert bot1.memory is None
        assert bot2.memory is not None
        assert len(registry._cache) == 2

    @pytest.mark.asyncio
    async def test_missing_client_with_config_object(self):
        """Test error handling for missing client with Config object."""
        config = Config({"bots": {}})
        registry = BotRegistry(config)

        # Try to get non-existent client
        with pytest.raises(KeyError, match="No bot configuration found"):
            await registry.get_bot("non-existent")

    @pytest.mark.asyncio
    async def test_config_without_bots_section(self):
        """Test error when Config object has no bots section."""
        config = Config({"other_section": {}})
        registry = BotRegistry(config)

        # Try to get client when bots section doesn't exist
        with pytest.raises(KeyError, match="No bot configuration found"):
            await registry.get_bot("client-1")

    @pytest.mark.asyncio
    async def test_register_client_creates_bots_section(self):
        """Test that register_client creates bots section if missing."""
        config = Config({"other_section": {}})
        registry = BotRegistry(config)

        # Register client - should create bots section
        await registry.register_client(
            "new-client",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Should be able to get bot now
        bot = await registry.get_bot("new-client")
        assert bot is not None

    @pytest.mark.asyncio
    async def test_get_bot_by_name_field(self):
        """Test getting bot using name field in Config object."""
        # Create a Config with bot stored by name field
        config = Config({})
        registry = BotRegistry(config)

        # Manually construct the internal Config structure
        # where bot config has "name" field instead of being keyed by client_id
        config._data["bots"] = [
            {
                "name": "client-with-name",
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            }
        ]

        # Should be able to get bot by name
        bot = await registry.get_bot("client-with-name")
        assert bot is not None


class TestBotRegistryEdgeCases:
    """Test edge cases and error conditions for BotRegistry."""

    @pytest.mark.asyncio
    async def test_plain_dict_without_bots_key(self):
        """Test error when plain dict has no bots key."""
        config = {"other_section": {}}
        registry = BotRegistry(config)

        # Try to get client when bots key doesn't exist
        with pytest.raises(KeyError, match="No bot configuration found"):
            await registry.get_bot("client-1")

    @pytest.mark.asyncio
    async def test_plain_dict_bots_not_dict(self):
        """Test error when bots value is not a dict."""
        config = {"bots": ["not", "a", "dict"]}
        registry = BotRegistry(config)

        # Try to get client when bots is not a dict
        with pytest.raises(ValueError, match="Invalid bots configuration format"):
            await registry.get_bot("client-1")

    @pytest.mark.asyncio
    async def test_register_client_with_invalid_bots_format(self):
        """Test registering client when bots is wrong format."""
        config = {"bots": "invalid"}
        registry = BotRegistry(config)

        # Register should fix the format
        await registry.register_client(
            "new-client",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Should be able to get bot now
        bot = await registry.get_bot("new-client")
        assert bot is not None

    @pytest.mark.asyncio
    async def test_register_client_creates_bots_key(self):
        """Test that register_client creates bots key if missing."""
        config = {}
        registry = BotRegistry(config)

        # Register client - should create bots key
        await registry.register_client(
            "new-client",
            {
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
            },
        )

        # Should be able to get bot now
        bot = await registry.get_bot("new-client")
        assert bot is not None
        assert "bots" in config
        assert "new-client" in config["bots"]
