"""Configuration caching manager with environment-aware resolution."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

from dataknobs_common.events import EventBus

from .backend import RegistryBackend
from .caching import CachingRegistryManager

if TYPE_CHECKING:
    from dataknobs_config import EnvironmentConfig

logger = logging.getLogger(__name__)


class ResolvedConfig:
    """A resolved configuration with metadata.

    This class wraps a resolved configuration dictionary along with
    metadata about how it was resolved.

    Attributes:
        config_id: The identifier for this configuration
        raw_config: The original unresolved configuration
        resolved_config: The configuration after $resource resolution
        environment_name: The environment used for resolution (or None)
    """

    def __init__(
        self,
        config_id: str,
        raw_config: dict[str, Any],
        resolved_config: dict[str, Any],
        environment_name: str | None = None,
    ) -> None:
        """Initialize a resolved configuration.

        Args:
            config_id: Unique identifier
            raw_config: Original configuration before resolution
            resolved_config: Configuration after $resource resolution
            environment_name: Name of environment used for resolution
        """
        self.config_id = config_id
        self.raw_config = raw_config
        self.resolved_config = resolved_config
        self.environment_name = environment_name

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the resolved configuration.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            The configuration value
        """
        return self.resolved_config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get a value from the resolved configuration."""
        return self.resolved_config[key]

    def to_dict(self) -> dict[str, Any]:
        """Return the resolved configuration as a dict."""
        return copy.deepcopy(self.resolved_config)


class ConfigCachingManager(CachingRegistryManager[ResolvedConfig]):
    """Caching manager for environment-resolved configurations.

    This manager loads configurations from a RegistryBackend, optionally
    resolves `$resource` references using an EnvironmentConfig, and caches
    the resolved configurations with TTL and event-driven invalidation.

    Use Cases:
    - Caching bot configurations with resolved LLM providers
    - Caching database connection configs with resolved credentials
    - Any scenario where configuration resolution is expensive

    Args:
        backend: Storage backend for raw configurations
        environment: EnvironmentConfig for $resource resolution (optional)
        config_key: Key within config containing the main configuration
        event_bus: Event bus for distributed cache invalidation
        event_topic: Topic for cache invalidation events
        cache_ttl: Cache time-to-live in seconds
        max_cache_size: Maximum cached configurations

    Example:
        ```python
        from dataknobs_bots.registry import ConfigCachingManager, InMemoryBackend
        from dataknobs_config import EnvironmentConfig

        # Load environment
        env = EnvironmentConfig.load("production")

        # Create manager
        manager = ConfigCachingManager(
            backend=InMemoryBackend(),
            environment=env,
            config_key="bot",
        )
        await manager.initialize()

        # Get resolved config (cached)
        resolved = await manager.get_or_create("my-bot")
        llm_config = resolved["llm"]  # $resource already resolved
        ```
    """

    def __init__(
        self,
        backend: RegistryBackend,
        environment: EnvironmentConfig | None = None,
        config_key: str | None = None,
        event_bus: EventBus | None = None,
        event_topic: str = "registry:configs",
        cache_ttl: int = 300,
        max_cache_size: int = 1000,
    ) -> None:
        """Initialize the configuration caching manager.

        Args:
            backend: Storage backend for configurations
            environment: EnvironmentConfig for $resource resolution
            config_key: Key to extract from config (e.g., "bot")
            event_bus: Event bus for distributed invalidation
            event_topic: Topic for invalidation events
            cache_ttl: Cache TTL in seconds
            max_cache_size: Maximum cached configs
        """
        super().__init__(
            backend=backend,
            event_bus=event_bus,
            event_topic=event_topic,
            cache_ttl=cache_ttl,
            max_cache_size=max_cache_size,
        )
        self._environment = environment
        self._config_key = config_key

    @property
    def environment(self) -> EnvironmentConfig | None:
        """Get the environment configuration."""
        return self._environment

    @property
    def environment_name(self) -> str | None:
        """Get the environment name, if configured."""
        return self._environment.name if self._environment else None

    async def _create_instance(
        self,
        instance_id: str,
        config: dict[str, Any],
    ) -> ResolvedConfig:
        """Create a resolved configuration from raw config.

        If an environment is configured, resolves $resource references.
        Otherwise, returns the config as-is.

        Args:
            instance_id: Configuration identifier
            config: Raw configuration from backend

        Returns:
            ResolvedConfig with resolved values
        """
        raw_config = copy.deepcopy(config)

        # Extract config_key if specified
        if self._config_key and self._config_key in config:
            config_to_resolve = config[self._config_key]
        else:
            config_to_resolve = config

        # Resolve $resource references if environment is configured
        if self._environment is not None:
            resolved_config = self._resolve_resources(config_to_resolve)
            logger.debug(
                "Resolved config %s with environment %s",
                instance_id,
                self._environment.name,
            )
        else:
            resolved_config = copy.deepcopy(config_to_resolve)
            logger.debug("Created config %s (no environment resolution)", instance_id)

        return ResolvedConfig(
            config_id=instance_id,
            raw_config=raw_config,
            resolved_config=resolved_config,
            environment_name=self.environment_name,
        )

    async def _destroy_instance(self, instance: ResolvedConfig) -> None:
        """Clean up a resolved configuration.

        Configurations are just data structures, so no cleanup is needed.
        This method exists to satisfy the abstract interface.

        Args:
            instance: The resolved configuration to clean up
        """
        # Configurations don't need cleanup - they're just data
        logger.debug("Released config %s from cache", instance.config_id)

    def _resolve_resources(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve $resource references in a configuration.

        Recursively walks the config and resolves any $resource references
        using the configured environment.

        Args:
            config: Configuration with potential $resource references

        Returns:
            Configuration with $resource references resolved
        """
        if self._environment is None:
            return copy.deepcopy(config)

        def resolve_value(value: Any) -> Any:
            if isinstance(value, dict):
                # Check for $resource reference
                if "$resource" in value and "type" in value:
                    resource_id = value["$resource"]
                    resource_type = value["type"]
                    resolved = self._environment.get_resource(resource_type, resource_id)
                    if resolved is not None:
                        return resolved
                    # If resource not found, return original
                    logger.warning(
                        "Resource not found: %s/%s",
                        resource_type,
                        resource_id,
                    )
                    return value
                # Recursively resolve nested dicts
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value

        return resolve_value(config)

    async def get_resolved_config(self, config_id: str) -> dict[str, Any]:
        """Get the resolved configuration as a dictionary.

        Convenience method that returns just the resolved dict.

        Args:
            config_id: Configuration identifier

        Returns:
            Resolved configuration dictionary

        Raises:
            KeyError: If configuration not found
        """
        resolved = await self.get_or_create(config_id)
        return resolved.to_dict()

    async def get_raw_config(self, config_id: str) -> dict[str, Any] | None:
        """Get the raw (unresolved) configuration from backend.

        Bypasses the cache and returns the original configuration.

        Args:
            config_id: Configuration identifier

        Returns:
            Raw configuration dict or None if not found
        """
        return await self._backend.get_config(config_id)
