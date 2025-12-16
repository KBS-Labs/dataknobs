"""Bot manager for multi-tenant bot instances.

.. deprecated::
    This module is deprecated. Use :class:`dataknobs_bots.bot.BotRegistry` instead,
    which provides the same functionality plus persistent storage backends,
    environment-aware configuration resolution, and TTL-based caching.

    For simple in-memory usage, use :class:`dataknobs_bots.bot.InMemoryBotRegistry`.

    Migration example::

        # Old (deprecated)
        from dataknobs_bots import BotManager
        manager = BotManager()
        bot = await manager.get_or_create("my-bot", config)

        # New (recommended)
        from dataknobs_bots.bot import InMemoryBotRegistry
        registry = InMemoryBotRegistry(validate_on_register=False)
        await registry.initialize()
        await registry.register("my-bot", config)
        bot = await registry.get_bot("my-bot")
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from .base import DynaBot

if TYPE_CHECKING:
    from dataknobs_config import EnvironmentAwareConfig, EnvironmentConfig

logger = logging.getLogger(__name__)

_DEPRECATION_MESSAGE = (
    "BotManager is deprecated and will be removed in a future version. "
    "Use BotRegistry or InMemoryBotRegistry instead, which provide persistent "
    "storage backends, environment-aware resolution, and TTL caching. "
    "See dataknobs_bots.bot.BotRegistry for details."
)


@runtime_checkable
class ConfigLoader(Protocol):
    """Protocol for configuration loaders with a load method."""

    def load(self, bot_id: str) -> dict[str, Any]:
        """Load configuration for a bot."""
        ...


@runtime_checkable
class AsyncConfigLoader(Protocol):
    """Protocol for async configuration loaders."""

    async def load(self, bot_id: str) -> dict[str, Any]:
        """Load configuration for a bot asynchronously."""
        ...


ConfigLoaderType = (
    ConfigLoader
    | AsyncConfigLoader
    | Callable[[str], dict[str, Any]]
    | Callable[[str], Any]  # For async callables
)


class BotManager:
    """Manages multiple DynaBot instances for multi-tenancy.

    .. deprecated::
        Use :class:`BotRegistry` or :class:`InMemoryBotRegistry` instead.

    BotManager handles:
    - Bot instance creation and caching
    - Client-level isolation
    - Configuration loading and validation
    - Bot lifecycle management
    - Environment-aware resource resolution (optional)

    Each client/tenant gets its own bot instance, which can serve multiple users.
    The underlying DynaBot architecture ensures conversation isolation through
    BotContext with different conversation_ids.

    Attributes:
        bots: Cache of bot_id -> DynaBot instances
        config_loader: Optional configuration loader (sync or async)
        environment_name: Current environment name (if environment-aware)

    Example:
        ```python
        # Basic usage with inline configuration
        manager = BotManager()
        bot = await manager.get_or_create("my-bot", config={
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "conversation_storage": {"backend": "memory"},
        })

        # With environment-aware configuration
        manager = BotManager(environment="production")
        bot = await manager.get_or_create("my-bot", config={
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers"},
                "conversation_storage": {"$resource": "db", "type": "databases"},
            }
        })

        # With config loader function
        def load_config(bot_id: str) -> dict:
            return load_yaml(f"configs/{bot_id}.yaml")

        manager = BotManager(config_loader=load_config)
        bot = await manager.get_or_create("my-bot")

        # List active bots
        active_bots = manager.list_bots()
        ```
    """

    def __init__(
        self,
        config_loader: ConfigLoaderType | None = None,
        environment: EnvironmentConfig | str | None = None,
        env_dir: str | Path = "config/environments",
    ):
        """Initialize BotManager.

        Args:
            config_loader: Optional configuration loader.
                Can be:
                - An object with a `.load(bot_id)` method (sync or async)
                - A callable function: bot_id -> config_dict (sync or async)
                - None (configurations must be provided explicitly)
            environment: Environment name or EnvironmentConfig for resource resolution.
                If None, environment-aware features are disabled unless
                an EnvironmentAwareConfig is passed to get_or_create().
                If a string, loads environment config from env_dir.
            env_dir: Directory containing environment config files.
                Only used if environment is a string name.
        """
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

        self._bots: dict[str, DynaBot] = {}
        self._config_loader = config_loader
        self._env_dir = Path(env_dir)

        # Load environment config if specified
        self._environment: EnvironmentConfig | None = None
        if environment is not None:
            try:
                from dataknobs_config import EnvironmentConfig

                if isinstance(environment, str):
                    self._environment = EnvironmentConfig.load(environment, env_dir)
                else:
                    self._environment = environment
                logger.info(f"Initialized BotManager with environment: {self._environment.name}")
            except ImportError:
                logger.warning(
                    "dataknobs_config not installed, environment-aware features disabled"
                )
        else:
            logger.info("Initialized BotManager")

    @property
    def environment_name(self) -> str | None:
        """Get current environment name, or None if not environment-aware."""
        return self._environment.name if self._environment else None

    @property
    def environment(self) -> EnvironmentConfig | None:
        """Get current environment config, or None if not environment-aware."""
        return self._environment

    async def get_or_create(
        self,
        bot_id: str,
        config: dict[str, Any] | EnvironmentAwareConfig | None = None,
        use_environment: bool | None = None,
        config_key: str = "bot",
    ) -> DynaBot:
        """Get existing bot or create new one.

        Args:
            bot_id: Bot identifier (e.g., "customer-support", "sales-assistant")
            config: Optional bot configuration. Can be:
                - dict with resolved values (traditional)
                - dict with $resource references (requires environment)
                - EnvironmentAwareConfig instance
                If not provided and config_loader is set, will load configuration.
            use_environment: Whether to use environment-aware resolution.
                - True: Use environment for $resource resolution
                - False: Use config as-is (no resolution)
                - None (default): Auto-detect based on whether manager has
                  an environment configured or config is EnvironmentAwareConfig
            config_key: Key within config containing bot configuration.
                       Defaults to "bot". Set to None to use root config.
                       Only used when use_environment is True.

        Returns:
            DynaBot instance

        Raises:
            ValueError: If config is None and no config_loader is set

        Example:
            ```python
            # Traditional usage (no environment resolution)
            manager = BotManager()
            bot = await manager.get_or_create("support-bot", config={
                "llm": {"provider": "openai", "model": "gpt-4"},
                "conversation_storage": {"backend": "memory"},
            })

            # Environment-aware usage with $resource references
            manager = BotManager(environment="production")
            bot = await manager.get_or_create("support-bot", config={
                "bot": {
                    "llm": {"$resource": "default", "type": "llm_providers"},
                    "conversation_storage": {"$resource": "db", "type": "databases"},
                }
            })

            # Explicit environment resolution control
            bot = await manager.get_or_create(
                "support-bot",
                config=my_config,
                use_environment=True,
                config_key="bot"
            )
            ```
        """
        # Return cached bot if exists
        if bot_id in self._bots:
            logger.debug(f"Returning cached bot: {bot_id}")
            return self._bots[bot_id]

        # Load configuration if not provided
        if config is None:
            if self._config_loader is None:
                raise ValueError(
                    f"No configuration provided for bot '{bot_id}' "
                    "and no config_loader is set"
                )
            config = await self._load_config(bot_id)

        # Determine whether to use environment resolution
        is_env_aware_config = False
        try:
            from dataknobs_config import EnvironmentAwareConfig

            is_env_aware_config = isinstance(config, EnvironmentAwareConfig)
        except ImportError:
            pass

        should_use_environment = use_environment
        if should_use_environment is None:
            # Auto-detect: use environment if manager has one or config is EnvironmentAwareConfig
            should_use_environment = self._environment is not None or is_env_aware_config

        # Create new bot
        logger.info(f"Creating new bot: {bot_id} (environment_aware={should_use_environment})")

        if should_use_environment:
            bot = await DynaBot.from_environment_aware_config(
                config,
                environment=self._environment,
                env_dir=self._env_dir,
                config_key=config_key,
            )
        else:
            # Traditional path - use config as-is
            bot = await DynaBot.from_config(config)

        # Cache and return
        self._bots[bot_id] = bot
        return bot

    async def get(self, bot_id: str) -> DynaBot | None:
        """Get bot without creating if doesn't exist.

        Args:
            bot_id: Bot identifier

        Returns:
            DynaBot instance if exists, None otherwise
        """
        return self._bots.get(bot_id)

    async def remove(self, bot_id: str) -> bool:
        """Remove bot instance.

        Args:
            bot_id: Bot identifier

        Returns:
            True if bot was removed, False if didn't exist
        """
        if bot_id in self._bots:
            logger.info(f"Removing bot: {bot_id}")
            del self._bots[bot_id]
            return True
        return False

    async def reload(self, bot_id: str) -> DynaBot:
        """Reload bot instance with fresh configuration.

        Args:
            bot_id: Bot identifier

        Returns:
            New DynaBot instance

        Raises:
            ValueError: If no config_loader is set
        """
        if self._config_loader is None:
            raise ValueError("Cannot reload without config_loader")

        # Remove existing bot
        await self.remove(bot_id)

        # Create new one
        return await self.get_or_create(bot_id)

    def list_bots(self) -> list[str]:
        """List all active bot IDs.

        Returns:
            List of bot identifiers
        """
        return list(self._bots.keys())

    def get_bot_count(self) -> int:
        """Get count of active bots.

        Returns:
            Number of active bot instances
        """
        return len(self._bots)

    async def _load_config(self, bot_id: str) -> dict[str, Any]:
        """Load configuration for bot using config_loader.

        Supports both synchronous and asynchronous config loaders.
        Handles both callable loaders and objects with a load() method.

        Args:
            bot_id: Bot identifier

        Returns:
            Bot configuration dictionary
        """
        logger.debug(f"Loading configuration for bot: {bot_id}")

        if callable(self._config_loader):
            # Handle callable config loader (function)
            if inspect.iscoroutinefunction(self._config_loader):
                # Async function
                result = await self._config_loader(bot_id)
                return dict(result) if isinstance(result, dict) else {}
            else:
                # Sync function - run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._config_loader, bot_id)
                return dict(result) if isinstance(result, dict) else {}
        else:
            # Assume it's an object with a load method
            load_method = self._config_loader.load  # type: ignore

            if inspect.iscoroutinefunction(load_method):
                # Async method
                result = await load_method(bot_id)
                return dict(result) if isinstance(result, dict) else {}
            else:
                # Sync method - run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, load_method, bot_id)
                return dict(result) if isinstance(result, dict) else {}

    async def clear_all(self) -> None:
        """Clear all bot instances.

        Useful for testing or when restarting the service.
        """
        logger.info("Clearing all bot instances")
        self._bots.clear()

    def get_portable_config(
        self,
        config: dict[str, Any] | EnvironmentAwareConfig,
    ) -> dict[str, Any]:
        """Get portable configuration for storage.

        Extracts portable config (with $resource references intact,
        environment variables unresolved) suitable for storing in
        registries or databases.

        Args:
            config: Configuration to make portable.
                Can be dict or EnvironmentAwareConfig.

        Returns:
            Portable configuration dictionary

        Example:
            ```python
            manager = BotManager(environment="production")

            # Get portable config from EnvironmentAwareConfig
            portable = manager.get_portable_config(env_aware_config)

            # Store in registry (portable across environments)
            await registry.store(bot_id, portable)
            ```
        """
        return DynaBot.get_portable_config(config)

    def __repr__(self) -> str:
        """String representation."""
        bots = ", ".join(self._bots.keys())
        env = f", environment={self._environment.name!r}" if self._environment else ""
        return f"BotManager(bots=[{bots}], count={len(self._bots)}{env})"
