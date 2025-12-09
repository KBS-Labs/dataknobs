"""Dependency injection helpers for FastAPI.

This module provides singleton management and FastAPI dependency injection
for bot-related services.

Example:
    ```python
    from fastapi import FastAPI
    from dataknobs_bots.api.dependencies import (
        init_bot_manager,
        BotManagerDep,
    )

    app = FastAPI()

    @app.on_event("startup")
    async def startup():
        # Initialize with a config loader
        init_bot_manager(config_loader=my_loader)

    @app.post("/chat/{bot_id}")
    async def chat(
        bot_id: str,
        message: str,
        manager: BotManagerDep,
    ):
        bot = await manager.get_or_create(bot_id)
        return await bot.chat(message, context)
    ```
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from dataknobs_bots.bot.manager import BotManager, ConfigLoaderType


logger = logging.getLogger(__name__)


class _BotManagerSingleton:
    """Singleton container for BotManager instance.

    Using a class-based approach avoids global statement warnings
    while maintaining singleton semantics.
    """

    _instance: BotManager | None = None

    @classmethod
    def get(cls) -> BotManager:
        """Get the singleton instance, creating with defaults if needed."""
        if cls._instance is None:
            cls._instance = BotManager()
            logger.info("Created default BotManager singleton (no config loader)")
        return cls._instance

    @classmethod
    def init(
        cls,
        config_loader: ConfigLoaderType | None = None,
        **kwargs: Any,
    ) -> BotManager:
        """Initialize the singleton with configuration."""
        cls._instance = BotManager(config_loader=config_loader, **kwargs)
        logger.info("Initialized BotManager singleton")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        cls._instance = None
        logger.info("Reset BotManager singleton")


def get_bot_manager() -> BotManager:
    """Get or create BotManager singleton instance.

    Returns:
        BotManager instance

    Note:
        Call `init_bot_manager()` during app startup to configure
        the singleton before using this dependency.
    """
    return _BotManagerSingleton.get()


def init_bot_manager(
    config_loader: ConfigLoaderType | None = None,
    **kwargs: Any,
) -> BotManager:
    """Initialize the BotManager singleton with configuration.

    Call this during application startup to configure the singleton.

    Args:
        config_loader: Optional configuration loader for bots
        **kwargs: Additional arguments passed to BotManager

    Returns:
        Configured BotManager instance

    Example:
        ```python
        @app.on_event("startup")
        async def startup():
            init_bot_manager(
                config_loader=MyConfigLoader("./configs")
            )
        ```
    """
    return _BotManagerSingleton.init(config_loader=config_loader, **kwargs)


def reset_bot_manager() -> None:
    """Reset the BotManager singleton.

    Useful for testing or when reconfiguring the application.
    """
    _BotManagerSingleton.reset()


# Dependency function for FastAPI
def _get_bot_manager_dep() -> BotManager:
    """Dependency function for FastAPI."""
    return get_bot_manager()


# Type alias for FastAPI dependency injection
# Usage: async def endpoint(manager: BotManagerDep):
try:
    from fastapi import Depends

    BotManagerDep = Annotated[BotManager, Depends(_get_bot_manager_dep)]
except ImportError:
    # FastAPI not installed - provide a placeholder
    BotManagerDep = BotManager  # type: ignore
