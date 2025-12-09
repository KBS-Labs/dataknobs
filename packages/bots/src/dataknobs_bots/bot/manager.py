"""Bot manager for multi-tenant bot instances."""

import asyncio
import inspect
import logging
from typing import Any, Callable, Protocol, runtime_checkable

from .base import DynaBot

logger = logging.getLogger(__name__)


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

    BotManager handles:
    - Bot instance creation and caching
    - Client-level isolation
    - Configuration loading and validation
    - Bot lifecycle management

    Each client/tenant gets its own bot instance, which can serve multiple users.
    The underlying DynaBot architecture ensures conversation isolation through
    BotContext with different conversation_ids.

    Attributes:
        bots: Cache of bot_id -> DynaBot instances
        config_loader: Optional configuration loader (sync or async)

    Example:
        ```python
        # With inline configuration
        manager = BotManager()
        bot = await manager.get_or_create("my-bot", config={
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "conversation_storage": {"backend": "memory"},
        })

        # With config loader function
        def load_config(bot_id: str) -> dict:
            return load_yaml(f"configs/{bot_id}.yaml")

        manager = BotManager(config_loader=load_config)
        bot = await manager.get_or_create("my-bot")

        # With ConfigLoader instance
        loader = MyConfigLoader("./configs")
        manager = BotManager(config_loader=loader)

        # List active bots
        active_bots = manager.list_bots()
        ```
    """

    def __init__(
        self,
        config_loader: ConfigLoaderType | None = None,
    ):
        """Initialize BotManager.

        Args:
            config_loader: Optional configuration loader.
                Can be:
                - An object with a `.load(bot_id)` method (sync or async)
                - A callable function: bot_id -> config_dict (sync or async)
                - None (configurations must be provided explicitly)
        """
        self._bots: dict[str, DynaBot] = {}
        self._config_loader = config_loader
        logger.info("Initialized BotManager")

    async def get_or_create(
        self, bot_id: str, config: dict[str, Any] | None = None
    ) -> DynaBot:
        """Get existing bot or create new one.

        Args:
            bot_id: Bot identifier (e.g., "customer-support", "sales-assistant")
            config: Optional bot configuration. If not provided and config_loader
                   is set, will attempt to load configuration.

        Returns:
            DynaBot instance

        Raises:
            ValueError: If config is None and no config_loader is set

        Example:
            ```python
            manager = BotManager()
            bot = await manager.get_or_create("support-bot", config={...})
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

        # Create new bot
        logger.info(f"Creating new bot: {bot_id}")
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

    def __repr__(self) -> str:
        """String representation."""
        bots = ", ".join(self._bots.keys())
        return f"BotManager(bots=[{bots}], count={len(self._bots)})"
