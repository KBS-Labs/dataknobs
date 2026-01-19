"""Dependency injection helpers for FastAPI.

This module provides singleton management and FastAPI dependency injection
for bot-related services.

Example using BotRegistry (recommended):
    ```python
    from fastapi import FastAPI
    from dataknobs_bots.api.dependencies import (
        init_bot_registry,
        BotRegistryDep,
    )

    app = FastAPI()

    @app.on_event("startup")
    async def startup():
        # Initialize registry with environment
        await init_bot_registry(environment="production")

    @app.post("/chat/{bot_id}")
    async def chat(
        bot_id: str,
        message: str,
        registry: BotRegistryDep,
    ):
        bot = await registry.get_bot(bot_id)
        return await bot.chat(message, context)
    ```

Legacy example using BotManager (deprecated):
    ```python
    from dataknobs_bots.api.dependencies import (
        init_bot_manager,
        BotManagerDep,
    )

    @app.on_event("startup")
    async def startup():
        init_bot_manager(config_loader=my_loader)

    @app.post("/chat/{bot_id}")
    async def chat(bot_id: str, manager: BotManagerDep):
        bot = await manager.get_or_create(bot_id)
        return await bot.chat(message, context)
    ```
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from dataknobs_bots.bot.registry import BotRegistry, InMemoryBotRegistry

if TYPE_CHECKING:
    from dataknobs_config import EnvironmentConfig

    from dataknobs_bots.bot.manager import ConfigLoaderType

logger = logging.getLogger(__name__)


# =============================================================================
# BotRegistry Singleton (Recommended)
# =============================================================================


class _BotRegistrySingleton:
    """Singleton container for BotRegistry instance.

    Using a class-based approach avoids global statement warnings
    while maintaining singleton semantics.
    """

    _instance: BotRegistry | None = None
    _initialized: bool = False

    @classmethod
    def get(cls) -> BotRegistry:
        """Get the singleton instance, creating with defaults if needed.

        Note: You must call init() first for async initialization.
        """
        if cls._instance is None:
            cls._instance = InMemoryBotRegistry(validate_on_register=False)
            logger.info("Created default BotRegistry singleton")
        return cls._instance

    @classmethod
    async def init(
        cls,
        environment: EnvironmentConfig | str | None = None,
        env_dir: str | Path = "config/environments",
        cache_ttl: int = 300,
        validate_on_register: bool = True,
        **kwargs: Any,
    ) -> BotRegistry:
        """Initialize the singleton with configuration.

        Args:
            environment: Environment name or config for $resource resolution
            env_dir: Directory containing environment config files
            cache_ttl: Cache time-to-live in seconds
            validate_on_register: Validate config portability on register
            **kwargs: Additional arguments passed to InMemoryBotRegistry

        Returns:
            Configured and initialized BotRegistry instance
        """
        cls._instance = InMemoryBotRegistry(
            environment=environment,
            env_dir=env_dir,
            cache_ttl=cache_ttl,
            validate_on_register=validate_on_register,
            **kwargs,
        )
        await cls._instance.initialize()
        cls._initialized = True
        logger.info("Initialized BotRegistry singleton")
        return cls._instance

    @classmethod
    async def reset(cls) -> None:
        """Reset the singleton instance."""
        if cls._instance is not None and cls._initialized:
            await cls._instance.close()
        cls._instance = None
        cls._initialized = False
        logger.info("Reset BotRegistry singleton")


def get_bot_registry() -> BotRegistry:
    """Get or create BotRegistry singleton instance.

    Returns:
        BotRegistry instance

    Note:
        Call `await init_bot_registry()` during app startup to configure
        the singleton before using this dependency.
    """
    return _BotRegistrySingleton.get()


async def init_bot_registry(
    environment: EnvironmentConfig | str | None = None,
    env_dir: str | Path = "config/environments",
    cache_ttl: int = 300,
    validate_on_register: bool = True,
    **kwargs: Any,
) -> BotRegistry:
    """Initialize the BotRegistry singleton with configuration.

    Call this during application startup to configure the singleton.

    Args:
        environment: Environment name or EnvironmentConfig for $resource resolution
        env_dir: Directory containing environment config files
        cache_ttl: Cache time-to-live in seconds (default: 300)
        validate_on_register: Validate config portability on register (default: True)
        **kwargs: Additional arguments passed to InMemoryBotRegistry

    Returns:
        Configured and initialized BotRegistry instance

    Example:
        ```python
        @app.on_event("startup")
        async def startup():
            await init_bot_registry(
                environment="production",
                cache_ttl=600,
            )
        ```
    """
    return await _BotRegistrySingleton.init(
        environment=environment,
        env_dir=env_dir,
        cache_ttl=cache_ttl,
        validate_on_register=validate_on_register,
        **kwargs,
    )


async def reset_bot_registry() -> None:
    """Reset the BotRegistry singleton.

    Useful for testing or when reconfiguring the application.
    """
    await _BotRegistrySingleton.reset()


# Dependency function for FastAPI
def _get_bot_registry_dep() -> BotRegistry:
    """Dependency function for FastAPI."""
    return get_bot_registry()


# =============================================================================
# BotManager Singleton (Deprecated - use BotRegistry instead)
# =============================================================================


class _BotManagerSingleton:
    """Singleton container for BotManager instance.

    .. deprecated::
        Use _BotRegistrySingleton and BotRegistry instead.

    Using a class-based approach avoids global statement warnings
    while maintaining singleton semantics.
    """

    _instance: Any = None  # BotManager type, but imported lazily

    @classmethod
    def get(cls) -> Any:
        """Get the singleton instance, creating with defaults if needed."""
        if cls._instance is None:
            # Import lazily to avoid deprecation warning at module load
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from dataknobs_bots.bot.manager import BotManager

                cls._instance = BotManager()
            logger.info("Created default BotManager singleton (no config loader)")
        return cls._instance

    @classmethod
    def init(
        cls,
        config_loader: ConfigLoaderType | None = None,
        **kwargs: Any,
    ) -> Any:
        """Initialize the singleton with configuration."""
        # Import lazily to avoid deprecation warning at module load
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from dataknobs_bots.bot.manager import BotManager

            cls._instance = BotManager(config_loader=config_loader, **kwargs)
        logger.info("Initialized BotManager singleton")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        cls._instance = None
        logger.info("Reset BotManager singleton")


def get_bot_manager() -> Any:
    """Get or create BotManager singleton instance.

    .. deprecated::
        Use :func:`get_bot_registry` instead.

    Returns:
        BotManager instance

    Note:
        Call `init_bot_manager()` during app startup to configure
        the singleton before using this dependency.
    """
    warnings.warn(
        "get_bot_manager() is deprecated. Use get_bot_registry() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _BotManagerSingleton.get()


def init_bot_manager(
    config_loader: ConfigLoaderType | None = None,
    **kwargs: Any,
) -> Any:
    """Initialize the BotManager singleton with configuration.

    .. deprecated::
        Use :func:`init_bot_registry` instead.

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
    warnings.warn(
        "init_bot_manager() is deprecated. Use init_bot_registry() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _BotManagerSingleton.init(config_loader=config_loader, **kwargs)


def reset_bot_manager() -> None:
    """Reset the BotManager singleton.

    .. deprecated::
        Use :func:`reset_bot_registry` instead.

    Useful for testing or when reconfiguring the application.
    """
    warnings.warn(
        "reset_bot_manager() is deprecated. Use reset_bot_registry() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _BotManagerSingleton.reset()


# Dependency function for FastAPI (deprecated)
def _get_bot_manager_dep() -> Any:
    """Dependency function for FastAPI (deprecated)."""
    return _BotManagerSingleton.get()


# Type aliases for FastAPI dependency injection
try:
    from fastapi import Depends

    # Recommended: BotRegistry dependency
    # Usage: async def endpoint(registry: BotRegistryDep):
    BotRegistryDep = Annotated[BotRegistry, Depends(_get_bot_registry_dep)]

    # Deprecated: BotManager dependency
    # Usage: async def endpoint(manager: BotManagerDep):
    BotManagerDep = Annotated[Any, Depends(_get_bot_manager_dep)]
except ImportError:
    # FastAPI not installed - provide placeholders
    BotRegistryDep = BotRegistry  # type: ignore
    BotManagerDep = Any  # type: ignore
