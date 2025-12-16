"""Multi-tenant bot registry with pluggable storage backends.

This module provides a registry for managing bot configurations and instances
across multiple tenants. It combines:
- Pluggable storage backends (via RegistryBackend protocol)
- Environment-aware configuration resolution
- Portability validation for cross-environment deployments
- Bot instance caching with TTL

Example:
    ```python
    from dataknobs_bots.bot import BotRegistry
    from dataknobs_bots.registry import InMemoryBackend

    # Create registry with in-memory storage
    registry = BotRegistry(
        backend=InMemoryBackend(),
        environment="production",
    )
    await registry.initialize()

    # Register a portable bot configuration
    await registry.register("my-bot", {
        "bot": {
            "llm": {"$resource": "default", "type": "llm_providers"},
            "conversation_storage": {"$resource": "db", "type": "databases"},
        }
    })

    # Get bot instance (resolves $resource references)
    bot = await registry.get_bot("my-bot")
    response = await bot.chat(message, context)

    # Cleanup
    await registry.close()
    ```
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..registry import InMemoryBackend, RegistryBackend, validate_portability
from .base import DynaBot

if TYPE_CHECKING:
    from dataknobs_config import EnvironmentConfig

    from ..registry import Registration

logger = logging.getLogger(__name__)


class BotRegistry:
    """Multi-tenant bot registry with caching and environment support.

    The BotRegistry manages multiple bot instances for different clients/tenants.
    It provides:
    - Pluggable storage backends via RegistryBackend protocol
    - Environment-aware configuration resolution
    - Portability validation to ensure configs work across environments
    - LRU-style caching with TTL for bot instances
    - Thread-safe access

    This enables:
    - Multi-tenant SaaS platforms
    - A/B testing with different bot configurations
    - Horizontal scaling with stateless bot instances
    - Cross-environment deployment with portable configs

    Attributes:
        backend: Storage backend for configurations
        environment: Environment for $resource resolution
        cache_ttl: Time-to-live for cached bots in seconds
        max_cache_size: Maximum number of bots to cache

    Example:
        ```python
        from dataknobs_bots.bot import BotRegistry
        from dataknobs_bots.registry import InMemoryBackend

        # Create registry
        registry = BotRegistry(
            backend=InMemoryBackend(),
            environment="production",
            cache_ttl=300,
        )
        await registry.initialize()

        # Register portable configuration
        await registry.register("client-123", {
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers"},
            }
        })

        # Get bot for a client
        bot = await registry.get_bot("client-123")

        # Use the bot
        response = await bot.chat(message, context)
        ```
    """

    def __init__(
        self,
        backend: RegistryBackend | None = None,
        environment: EnvironmentConfig | str | None = None,
        env_dir: str | Path = "config/environments",
        cache_ttl: int = 300,
        max_cache_size: int = 1000,
        validate_on_register: bool = True,
        config_key: str = "bot",
    ):
        """Initialize bot registry.

        Args:
            backend: Storage backend for configurations.
                If None, uses InMemoryBackend.
            environment: Environment name or EnvironmentConfig for
                $resource resolution. If None, configs are used as-is
                without environment resolution.
            env_dir: Directory containing environment config files.
                Only used if environment is a string name.
            cache_ttl: Cache time-to-live in seconds (default: 300)
            max_cache_size: Maximum cached bots (default: 1000)
            validate_on_register: If True, validate config portability
                when registering (default: True)
            config_key: Key within config containing bot configuration.
                Defaults to "bot". Used during environment resolution.
        """
        self._backend = backend or InMemoryBackend()
        self._env_dir = Path(env_dir)
        self._cache_ttl = cache_ttl
        self._max_cache_size = max_cache_size
        self._validate_on_register = validate_on_register
        self._config_key = config_key

        # Bot instance cache: bot_id -> (DynaBot, cached_timestamp)
        self._cache: dict[str, tuple[DynaBot, float]] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

        # Load environment config if specified
        self._environment: EnvironmentConfig | None = None
        if environment is not None:
            try:
                from dataknobs_config import EnvironmentConfig as EnvConfig

                if isinstance(environment, str):
                    self._environment = EnvConfig.load(environment, env_dir)
                else:
                    self._environment = environment
                logger.info(f"BotRegistry using environment: {self._environment.name}")
            except ImportError:
                logger.warning(
                    "dataknobs_config not installed, environment-aware features disabled"
                )

    @property
    def backend(self) -> RegistryBackend:
        """Get the storage backend."""
        return self._backend

    @property
    def environment(self) -> EnvironmentConfig | None:
        """Get current environment config, or None if not environment-aware."""
        return self._environment

    @property
    def environment_name(self) -> str | None:
        """Get current environment name, or None if not environment-aware."""
        return self._environment.name if self._environment else None

    @property
    def cache_ttl(self) -> int:
        """Get cache TTL in seconds."""
        return self._cache_ttl

    @property
    def max_cache_size(self) -> int:
        """Get maximum cache size."""
        return self._max_cache_size

    async def initialize(self) -> None:
        """Initialize the registry and backend.

        Must be called before using the registry.
        """
        if not self._initialized:
            await self._backend.initialize()
            self._initialized = True
            logger.info("BotRegistry initialized")

    async def close(self) -> None:
        """Close the registry and backend.

        Clears the bot cache and closes the storage backend.
        """
        async with self._lock:
            self._cache.clear()
        await self._backend.close()
        self._initialized = False
        logger.info("BotRegistry closed")

    async def register(
        self,
        bot_id: str,
        config: dict[str, Any],
        status: str = "active",
        skip_validation: bool = False,
    ) -> Registration:
        """Register or update a bot configuration.

        Stores a portable configuration in the backend. By default, validates
        that the configuration is portable (no resolved local values).

        Args:
            bot_id: Unique bot identifier
            config: Bot configuration dictionary (should be portable)
            status: Registration status (default: active)
            skip_validation: If True, skip portability validation

        Returns:
            Registration object with metadata

        Raises:
            PortabilityError: If config is not portable and validation is enabled

        Example:
            ```python
            # Register with portable config
            reg = await registry.register("support-bot", {
                "bot": {
                    "llm": {"$resource": "default", "type": "llm_providers"},
                }
            })
            print(f"Registered at: {reg.created_at}")

            # Update existing registration
            reg = await registry.register("support-bot", new_config)
            print(f"Updated at: {reg.updated_at}")
            ```
        """
        # Validate portability if enabled
        if self._validate_on_register and not skip_validation:
            validate_portability(config)

        # Store in backend
        registration = await self._backend.register(bot_id, config, status)

        # Invalidate cache for this bot
        async with self._lock:
            if bot_id in self._cache:
                del self._cache[bot_id]
                logger.debug(f"Invalidated cache for bot: {bot_id}")

        logger.info(f"Registered bot: {bot_id}")
        return registration

    async def get_bot(
        self,
        bot_id: str,
        force_refresh: bool = False,
    ) -> DynaBot:
        """Get bot instance for a client.

        Bots are cached for performance. If a cached bot exists and hasn't
        expired, it's returned. Otherwise, a new bot is created from the
        stored configuration with environment resolution applied.

        Args:
            bot_id: Bot identifier
            force_refresh: If True, bypass cache and create fresh bot

        Returns:
            DynaBot instance for the client

        Raises:
            KeyError: If no registration exists for the bot_id
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
            if not force_refresh and bot_id in self._cache:
                bot, cached_at = self._cache[bot_id]
                if time.time() - cached_at < self._cache_ttl:
                    logger.debug(f"Returning cached bot: {bot_id}")
                    return bot

            # Load configuration from backend
            config = await self._backend.get_config(bot_id)
            if config is None:
                raise KeyError(f"No bot configuration found for: {bot_id}")

            # Create bot with environment resolution if configured
            if self._environment is not None:
                logger.debug(f"Creating bot with environment resolution: {bot_id}")
                bot = await DynaBot.from_environment_aware_config(
                    config,
                    environment=self._environment,
                    env_dir=self._env_dir,
                    config_key=self._config_key,
                )
            else:
                # Traditional path - use config as-is
                # Extract bot config if wrapped in config_key
                bot_config = config.get(self._config_key, config)
                logger.debug(f"Creating bot without environment resolution: {bot_id}")
                bot = await DynaBot.from_config(bot_config)

            # Cache the bot
            self._cache[bot_id] = (bot, time.time())
            logger.info(f"Created bot: {bot_id}")

            # Evict old entries if cache is full
            if len(self._cache) > self._max_cache_size:
                self._evict_oldest()

            return bot

    async def get_config(self, bot_id: str) -> dict[str, Any] | None:
        """Get stored configuration for a bot.

        Returns the portable configuration as stored, without
        environment resolution applied.

        Args:
            bot_id: Bot identifier

        Returns:
            Configuration dict if found, None otherwise
        """
        return await self._backend.get_config(bot_id)

    async def get_registration(self, bot_id: str) -> Registration | None:
        """Get full registration including metadata.

        Args:
            bot_id: Bot identifier

        Returns:
            Registration if found, None otherwise
        """
        return await self._backend.get(bot_id)

    async def unregister(self, bot_id: str) -> bool:
        """Remove a bot registration (hard delete).

        Args:
            bot_id: Bot identifier

        Returns:
            True if removed, False if not found
        """
        # Remove from cache
        async with self._lock:
            if bot_id in self._cache:
                del self._cache[bot_id]

        result = await self._backend.unregister(bot_id)
        if result:
            logger.info(f"Unregistered bot: {bot_id}")
        return result

    async def deactivate(self, bot_id: str) -> bool:
        """Deactivate a bot registration (soft delete).

        Args:
            bot_id: Bot identifier

        Returns:
            True if deactivated, False if not found
        """
        # Remove from cache
        async with self._lock:
            if bot_id in self._cache:
                del self._cache[bot_id]

        result = await self._backend.deactivate(bot_id)
        if result:
            logger.info(f"Deactivated bot: {bot_id}")
        return result

    async def exists(self, bot_id: str) -> bool:
        """Check if an active bot registration exists.

        Args:
            bot_id: Bot identifier

        Returns:
            True if registration exists and is active
        """
        return await self._backend.exists(bot_id)

    async def list_bots(self) -> list[str]:
        """List all active bot IDs.

        Returns:
            List of active bot identifiers
        """
        return await self._backend.list_ids()

    async def count(self) -> int:
        """Count active bot registrations.

        Returns:
            Number of active registrations
        """
        return await self._backend.count()

    def get_cached_bots(self) -> list[str]:
        """Get list of currently cached bot IDs.

        Returns:
            List of bot IDs with cached instances
        """
        return list(self._cache.keys())

    def clear_cache(self) -> None:
        """Clear all cached bot instances.

        Does not affect stored registrations.
        """
        self._cache.clear()
        logger.debug("Cleared bot cache")

    def _evict_oldest(self) -> None:
        """Evict oldest cache entries when cache is full.

        Removes 10% of the oldest entries to make room for new ones.
        """
        # Sort by timestamp (oldest first)
        sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])

        # Remove oldest 10%
        num_to_remove = max(1, len(sorted_items) // 10)
        for bot_id, _ in sorted_items[:num_to_remove]:
            del self._cache[bot_id]
        logger.debug(f"Evicted {num_to_remove} bots from cache")

    # Legacy compatibility methods

    async def register_client(
        self, client_id: str, bot_config: dict[str, Any]
    ) -> None:
        """Register or update a client's bot configuration.

        .. deprecated::
            Use :meth:`register` instead.

        Args:
            client_id: Client/tenant identifier
            bot_config: Bot configuration dictionary
        """
        await self.register(client_id, bot_config)

    async def remove_client(self, client_id: str) -> None:
        """Remove a client from the registry.

        .. deprecated::
            Use :meth:`unregister` instead.

        Args:
            client_id: Client/tenant identifier
        """
        await self.unregister(client_id)

    def get_cached_clients(self) -> list[str]:
        """Get list of currently cached client IDs.

        .. deprecated::
            Use :meth:`get_cached_bots` instead.

        Returns:
            List of client IDs with cached bots
        """
        return self.get_cached_bots()

    def __repr__(self) -> str:
        """String representation."""
        env = f", environment={self._environment.name!r}" if self._environment else ""
        return (
            f"BotRegistry(backend={self._backend!r}, "
            f"cached={len(self._cache)}{env})"
        )


class InMemoryBotRegistry(BotRegistry):
    """BotRegistry with in-memory storage backend.

    A convenience subclass that uses InMemoryBackend for storage,
    suitable for testing, CLIs, and single-instance deployments.

    Unlike the base BotRegistry which accepts a pluggable backend,
    this class always uses in-memory storage and doesn't require
    external dependencies like databases.

    Example:
        ```python
        from dataknobs_bots.bot import InMemoryBotRegistry

        # For testing - no environment resolution
        registry = InMemoryBotRegistry(validate_on_register=False)
        await registry.initialize()

        await registry.register("test-bot", {"llm": {"provider": "echo"}})
        bot = await registry.get_bot("test-bot")

        # For development with environment
        registry = InMemoryBotRegistry(environment="development")
        await registry.initialize()
        ```
    """

    def __init__(
        self,
        environment: EnvironmentConfig | str | None = None,
        env_dir: str | Path = "config/environments",
        cache_ttl: int = 300,
        max_cache_size: int = 1000,
        validate_on_register: bool = True,
        config_key: str = "bot",
    ):
        """Initialize in-memory bot registry.

        Args:
            environment: Environment name or EnvironmentConfig for
                $resource resolution. If None, configs are used as-is
                without environment resolution.
            env_dir: Directory containing environment config files.
                Only used if environment is a string name.
            cache_ttl: Cache time-to-live in seconds (default: 300)
            max_cache_size: Maximum cached bots (default: 1000)
            validate_on_register: If True, validate config portability
                when registering (default: True)
            config_key: Key within config containing bot configuration.
                Defaults to "bot". Used during environment resolution.
        """
        super().__init__(
            backend=InMemoryBackend(),
            environment=environment,
            env_dir=env_dir,
            cache_ttl=cache_ttl,
            max_cache_size=max_cache_size,
            validate_on_register=validate_on_register,
            config_key=config_key,
        )

    async def clear(self) -> None:
        """Clear all registrations and cached bots.

        Convenience method for test cleanup that clears both the
        backend storage and the bot instance cache.

        Example:
            ```python
            # In tests - reset between test cases
            await registry.clear()
            assert await registry.count() == 0
            ```
        """
        await self._backend.clear()
        self._cache.clear()
        logger.debug("Cleared all registrations and cache")

    def __repr__(self) -> str:
        """String representation."""
        env = f", environment={self._environment.name!r}" if self._environment else ""
        return f"InMemoryBotRegistry(cached={len(self._cache)}{env})"


def create_memory_registry(
    environment: EnvironmentConfig | str | None = None,
    env_dir: str | Path = "config/environments",
    cache_ttl: int = 300,
    max_cache_size: int = 1000,
    validate_on_register: bool = True,
    config_key: str = "bot",
) -> InMemoryBotRegistry:
    """Create an InMemoryBotRegistry.

    Convenience factory for creating in-memory registries suitable for
    testing, CLIs, or single-instance deployments.

    Args:
        environment: Environment name or EnvironmentConfig for
            $resource resolution. If None, configs are used as-is.
        env_dir: Directory containing environment config files.
        cache_ttl: Cache time-to-live in seconds (default: 300)
        max_cache_size: Maximum cached bots (default: 1000)
        validate_on_register: If True, validate config portability
        config_key: Key within config containing bot configuration

    Returns:
        InMemoryBotRegistry instance

    Example:
        ```python
        from dataknobs_bots.bot import create_memory_registry

        registry = create_memory_registry(validate_on_register=False)
        await registry.initialize()

        await registry.register("test-bot", {"llm": {"provider": "echo"}})
        bot = await registry.get_bot("test-bot")
        ```
    """
    return InMemoryBotRegistry(
        environment=environment,
        env_dir=env_dir,
        cache_ttl=cache_ttl,
        max_cache_size=max_cache_size,
        validate_on_register=validate_on_register,
        config_key=config_key,
    )
