"""Multi-tenant bot registry with caching."""

import asyncio
import time
from typing import Any

from dataknobs_config import Config

from .base import DynaBot


class BotRegistry:
    """Multi-tenant bot registry with caching.

    The BotRegistry manages multiple bot instances for different clients/tenants.
    It provides:
    - Automatic bot creation from configuration
    - LRU-style caching with TTL
    - Thread-safe access
    - Dynamic configuration updates

    This enables:
    - Multi-tenant SaaS platforms
    - A/B testing with different bot configurations
    - Horizontal scaling with stateless bot instances

    Attributes:
        config: Configuration object
        cache_ttl: Time-to-live for cached bots in seconds
        max_cache_size: Maximum number of bots to cache
        _cache: Internal cache mapping client_id to (bot, timestamp)
        _lock: Asyncio lock for thread-safe access

    Example:
        ```python
        from dataknobs_config import Config

        # Load configuration
        config = Config("config/bots.yaml")

        # Create registry
        registry = BotRegistry(config, cache_ttl=300)

        # Get bot for a client
        bot = await registry.get_bot("client-123")

        # Use the bot
        response = await bot.chat(message, context)
        ```
    """

    def __init__(
        self,
        config: Config | dict[str, Any],
        cache_ttl: int = 300,
        max_cache_size: int = 1000,
    ):
        """Initialize bot registry.

        Args:
            config: Configuration object or dictionary
            cache_ttl: Cache time-to-live in seconds (default: 300)
            max_cache_size: Maximum cached bots (default: 1000)
        """
        # Store config as-is (don't wrap dicts in Config since Config
        # transforms the structure in ways that make nested access difficult)
        self.config = config

        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self._cache: dict[str, tuple[DynaBot, float]] = {}
        self._lock = asyncio.Lock()

    def _get_bot_config(self, client_id: str) -> dict[str, Any]:
        """Get bot configuration for a client.

        Args:
            client_id: Client/tenant identifier

        Returns:
            Bot configuration dictionary

        Raises:
            KeyError: If no configuration exists for the client
        """
        if isinstance(self.config, dict):
            # Plain dict: {"bots": {"client-1": {...}, ...}}
            if "bots" not in self.config:
                raise KeyError(f"No bot configuration found for client: {client_id}")
            bots = self.config["bots"]

            if isinstance(bots, dict):
                bot_config = bots.get(client_id)
            else:
                raise ValueError(f"Invalid bots configuration format: {type(bots)}")

            if bot_config is None:
                raise KeyError(f"No bot configuration found for client: {client_id}")
            return bot_config
        else:
            # Config object: _data is Dict[str, List[Dict]]
            if "bots" not in self.config._data:
                raise KeyError(f"No bot configuration found for client: {client_id}")

            bots_list = self.config._data["bots"]
            # Find bot in list by matching client_id key or name field
            for bot_dict in bots_list:
                if client_id in bot_dict:
                    return bot_dict[client_id]
                elif bot_dict.get("name") == client_id:
                    return bot_dict

            raise KeyError(f"No bot configuration found for client: {client_id}")

    def _set_bot_config(self, client_id: str, bot_config: dict[str, Any]) -> None:
        """Set bot configuration for a client.

        Args:
            client_id: Client/tenant identifier
            bot_config: Bot configuration dictionary
        """
        if isinstance(self.config, dict):
            # Plain dict format
            if "bots" not in self.config:
                self.config["bots"] = {}

            if not isinstance(self.config["bots"], dict):
                self.config["bots"] = {}

            self.config["bots"][client_id] = bot_config
        else:
            # Config object format
            if "bots" not in self.config._data:
                self.config._data["bots"] = []

            bots_list = self.config._data["bots"]
            # Find and update existing, or append new
            found = False
            for i, bot_dict in enumerate(bots_list):
                if client_id in bot_dict or bot_dict.get("name") == client_id:
                    bots_list[i] = {client_id: bot_config}
                    found = True
                    break

            if not found:
                bots_list.append({client_id: bot_config})

    def _remove_bot_config(self, client_id: str) -> None:
        """Remove bot configuration for a client.

        Args:
            client_id: Client/tenant identifier
        """
        if isinstance(self.config, dict):
            # Plain dict format
            if "bots" in self.config and isinstance(self.config["bots"], dict):
                self.config["bots"].pop(client_id, None)
        else:
            # Config object format
            if "bots" in self.config._data:
                bots_list = self.config._data["bots"]
                self.config._data["bots"] = [
                    bot_dict for bot_dict in bots_list
                    if client_id not in bot_dict and bot_dict.get("name") != client_id
                ]

    async def get_bot(self, client_id: str, force_refresh: bool = False) -> DynaBot:
        """Get bot instance for a client.

        Bots are cached for performance. If a cached bot exists and hasn't
        expired, it's returned. Otherwise, a new bot is created from configuration.

        Args:
            client_id: Client/tenant identifier
            force_refresh: If True, bypass cache and create fresh bot

        Returns:
            DynaBot instance for the client

        Raises:
            KeyError: If no configuration exists for the client
            ValueError: If bot configuration is invalid

        Example:
            ```python
            # Get cached bot
            bot = await registry.get_bot("client-123")

            # Force refresh (e.g., after config change)
            bot = await registry.get_bot("client-123", force_refresh=True)
            ```
        """
        async with self._lock:
            # Check cache
            if not force_refresh and client_id in self._cache:
                bot, cached_at = self._cache[client_id]
                if time.time() - cached_at < self.cache_ttl:
                    return bot

            # Load bot configuration using helper
            bot_config = self._get_bot_config(client_id)

            # Create bot from configuration
            bot = await DynaBot.from_config(bot_config)

            # Cache the bot
            self._cache[client_id] = (bot, time.time())

            # Evict old entries if cache is full
            if len(self._cache) > self.max_cache_size:
                self._evict_oldest()

            return bot

    async def register_client(
        self, client_id: str, bot_config: dict[str, Any]
    ) -> None:
        """Register or update a client's bot configuration.

        This allows dynamic registration of new clients or updating
        existing client configurations at runtime.

        Args:
            client_id: Client/tenant identifier
            bot_config: Bot configuration dictionary

        Example:
            ```python
            # Register new client
            await registry.register_client("new-client", {
                "llm": {"provider": "openai", "model": "gpt-4"},
                "conversation_storage": {"backend": "postgres"},
                "memory": {"type": "buffer", "max_messages": 10}
            })

            # Bot is now available
            bot = await registry.get_bot("new-client")
            ```
        """
        async with self._lock:
            # Update configuration using helper
            self._set_bot_config(client_id, bot_config)

            # Invalidate cache for this client
            if client_id in self._cache:
                del self._cache[client_id]

    async def remove_client(self, client_id: str) -> None:
        """Remove a client from the registry.

        Args:
            client_id: Client/tenant identifier

        Example:
            ```python
            # Remove client
            await registry.remove_client("old-client")
            ```
        """
        async with self._lock:
            # Remove from cache
            if client_id in self._cache:
                del self._cache[client_id]

            # Remove from config using helper
            self._remove_bot_config(client_id)

    def get_cached_clients(self) -> list[str]:
        """Get list of currently cached client IDs.

        Returns:
            List of client IDs with cached bots

        Example:
            ```python
            clients = registry.get_cached_clients()
            print(f"Cached bots: {clients}")
            ```
        """
        return list(self._cache.keys())

    def clear_cache(self) -> None:
        """Clear all cached bots.

        Example:
            ```python
            # Clear cache after config update
            registry.clear_cache()
            ```
        """
        self._cache.clear()

    def _evict_oldest(self) -> None:
        """Evict oldest cache entries when cache is full.

        Removes 10% of the oldest entries to make room for new ones.
        """
        # Sort by timestamp (oldest first)
        sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])

        # Remove oldest 10%
        num_to_remove = max(1, len(sorted_items) // 10)
        for client_id, _ in sorted_items[:num_to_remove]:
            del self._cache[client_id]
