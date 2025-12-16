"""Configuration binding resolver for resource instantiation.

This module provides the ConfigBindingResolver class that resolves logical
resource references to concrete instances using registered factories.

Example:
    ```python
    from dataknobs_config import EnvironmentConfig, ConfigBindingResolver

    # Load environment config
    env = EnvironmentConfig.load("production")

    # Create resolver
    resolver = ConfigBindingResolver(env)

    # Register factories for resource types
    resolver.register_factory("databases", DatabaseFactory())
    resolver.register_factory("vector_stores", VectorStoreFactory())

    # Resolve a logical reference to a concrete instance
    db = resolver.resolve("databases", "conversations")

    # Or with async factories
    vector_store = await resolver.resolve_async("vector_stores", "knowledge")
    ```
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable

from .environment_config import EnvironmentConfig
from .inheritance import substitute_env_vars

logger = logging.getLogger(__name__)


class BindingResolverError(Exception):
    """Error during resource binding resolution."""

    pass


class FactoryNotFoundError(BindingResolverError):
    """Factory not registered for resource type."""

    pass


@runtime_checkable
class ResourceFactory(Protocol):
    """Protocol for resource factories that can create from config.

    Factories must implement either create() or __call__() method.
    """

    def create(self, **config: Any) -> Any:
        """Create a resource instance from configuration.

        Args:
            **config: Configuration parameters for the resource

        Returns:
            Created resource instance
        """
        ...


@runtime_checkable
class AsyncResourceFactory(Protocol):
    """Protocol for async resource factories.

    Factories that create resources asynchronously should implement
    create_async() method.
    """

    async def create_async(self, **config: Any) -> Any:
        """Create a resource instance asynchronously.

        Args:
            **config: Configuration parameters for the resource

        Returns:
            Created resource instance
        """
        ...


class ConfigBindingResolver:
    """Resolves logical resource bindings to concrete instances.

    Works with EnvironmentConfig to:
    1. Look up resource configurations by logical name
    2. Resolve environment variables in configurations
    3. Instantiate resources using registered factories
    4. Cache instances for reuse

    Attributes:
        environment: The EnvironmentConfig for resource lookup
    """

    def __init__(
        self,
        environment: EnvironmentConfig,
        resolve_env_vars: bool = True,
    ):
        """Initialize the binding resolver.

        Args:
            environment: Environment configuration for resource lookup
            resolve_env_vars: Whether to resolve env vars before instantiation
        """
        self._environment = environment
        self._resolve_env_vars = resolve_env_vars
        self._factories: dict[str, ResourceFactory | Callable[..., Any]] = {}
        self._cache: dict[tuple[str, str], Any] = {}

    @property
    def environment(self) -> EnvironmentConfig:
        """Get the environment configuration."""
        return self._environment

    def register_factory(
        self,
        resource_type: str,
        factory: ResourceFactory | Callable[..., Any],
    ) -> None:
        """Register a factory for a resource type.

        Args:
            resource_type: Type of resource (e.g., "databases", "vector_stores")
            factory: Factory instance or callable that creates resources.
                    Must have create(**config) method or be callable.
        """
        self._factories[resource_type] = factory
        logger.debug(f"Registered factory for resource type: {resource_type}")

    def unregister_factory(self, resource_type: str) -> None:
        """Unregister a factory for a resource type.

        Args:
            resource_type: Type of resource to unregister

        Raises:
            KeyError: If factory not registered
        """
        if resource_type not in self._factories:
            raise KeyError(f"No factory registered for: {resource_type}")
        del self._factories[resource_type]
        logger.debug(f"Unregistered factory for resource type: {resource_type}")

    def has_factory(self, resource_type: str) -> bool:
        """Check if a factory is registered for a resource type.

        Args:
            resource_type: Type of resource

        Returns:
            True if factory is registered
        """
        return resource_type in self._factories

    def get_registered_types(self) -> list[str]:
        """Get all registered resource types.

        Returns:
            List of resource type names with registered factories
        """
        return list(self._factories.keys())

    def resolve(
        self,
        resource_type: str,
        logical_name: str,
        use_cache: bool = True,
        **overrides: Any,
    ) -> Any:
        """Resolve a logical resource reference to a concrete instance.

        Args:
            resource_type: Type of resource ("databases", "vector_stores", etc.)
            logical_name: Logical name from app config
            use_cache: Whether to return cached instance if available
            **overrides: Config overrides for this resolution

        Returns:
            Instantiated resource

        Raises:
            FactoryNotFoundError: If no factory registered for resource type
            BindingResolverError: If resource creation fails
        """
        cache_key = (resource_type, logical_name)

        # Check cache
        if use_cache and cache_key in self._cache:
            logger.debug(f"Returning cached {resource_type}[{logical_name}]")
            return self._cache[cache_key]

        # Get config from environment
        config = self._get_resolved_config(resource_type, logical_name, overrides)

        # Get factory for this type
        factory = self._get_factory(resource_type)

        # Create instance
        try:
            instance = self._create_instance(factory, config)
        except Exception as e:
            raise BindingResolverError(
                f"Failed to create {resource_type}[{logical_name}]: {e}"
            ) from e

        # Cache the instance
        if use_cache:
            self._cache[cache_key] = instance
            logger.debug(f"Cached {resource_type}[{logical_name}]")

        return instance

    async def resolve_async(
        self,
        resource_type: str,
        logical_name: str,
        use_cache: bool = True,
        **overrides: Any,
    ) -> Any:
        """Async version of resolve for async factories.

        If the factory has a create_async method, it will be used.
        Otherwise, falls back to synchronous create.

        Args:
            resource_type: Type of resource
            logical_name: Logical name from app config
            use_cache: Whether to return cached instance
            **overrides: Config overrides for this resolution

        Returns:
            Instantiated resource

        Raises:
            FactoryNotFoundError: If no factory registered for resource type
            BindingResolverError: If resource creation fails
        """
        cache_key = (resource_type, logical_name)

        # Check cache
        if use_cache and cache_key in self._cache:
            logger.debug(f"Returning cached {resource_type}[{logical_name}]")
            return self._cache[cache_key]

        # Get config from environment
        config = self._get_resolved_config(resource_type, logical_name, overrides)

        # Get factory for this type
        factory = self._get_factory(resource_type)

        # Create instance (async if supported)
        try:
            instance = await self._create_instance_async(factory, config)
        except Exception as e:
            raise BindingResolverError(
                f"Failed to create {resource_type}[{logical_name}]: {e}"
            ) from e

        # Cache the instance
        if use_cache:
            self._cache[cache_key] = instance
            logger.debug(f"Cached {resource_type}[{logical_name}]")

        return instance

    def _get_resolved_config(
        self,
        resource_type: str,
        logical_name: str,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Get and resolve configuration for a resource.

        Args:
            resource_type: Type of resource
            logical_name: Logical name
            overrides: Config overrides

        Returns:
            Resolved configuration
        """
        config = self._environment.get_resource(resource_type, logical_name)

        # Apply overrides
        if overrides:
            config = config.copy()
            config.update(overrides)

        # Resolve environment variables
        if self._resolve_env_vars:
            config = substitute_env_vars(config)

        return config

    def _get_factory(
        self, resource_type: str
    ) -> ResourceFactory | Callable[..., Any]:
        """Get the factory for a resource type.

        Args:
            resource_type: Type of resource

        Returns:
            Factory instance

        Raises:
            FactoryNotFoundError: If no factory registered
        """
        if resource_type not in self._factories:
            raise FactoryNotFoundError(
                f"No factory registered for resource type: {resource_type}. "
                f"Registered types: {list(self._factories.keys())}"
            )
        return self._factories[resource_type]

    def _create_instance(
        self,
        factory: ResourceFactory | Callable[..., Any],
        config: dict[str, Any],
    ) -> Any:
        """Create an instance using a factory.

        Args:
            factory: Factory to use
            config: Configuration for the resource

        Returns:
            Created instance
        """
        # Try create method first
        if hasattr(factory, "create"):
            return factory.create(**config)

        # Try build method
        if hasattr(factory, "build"):
            return factory.build(**config)

        # Try calling directly (for callable factories)
        if callable(factory):
            return factory(**config)

        raise BindingResolverError(
            f"Factory {factory} has no create(), build(), or __call__ method"
        )

    async def _create_instance_async(
        self,
        factory: ResourceFactory | Callable[..., Any],
        config: dict[str, Any],
    ) -> Any:
        """Create an instance using a factory (async).

        Args:
            factory: Factory to use
            config: Configuration for the resource

        Returns:
            Created instance
        """
        # Try async create method first
        if hasattr(factory, "create_async"):
            return await factory.create_async(**config)

        # Fall back to sync methods
        return self._create_instance(factory, config)

    def get_cached(
        self,
        resource_type: str,
        logical_name: str,
    ) -> Any | None:
        """Get a cached instance if it exists.

        Args:
            resource_type: Type of resource
            logical_name: Logical name

        Returns:
            Cached instance or None
        """
        return self._cache.get((resource_type, logical_name))

    def is_cached(self, resource_type: str, logical_name: str) -> bool:
        """Check if an instance is cached.

        Args:
            resource_type: Type of resource
            logical_name: Logical name

        Returns:
            True if cached
        """
        return (resource_type, logical_name) in self._cache

    def clear_cache(self, resource_type: str | None = None) -> None:
        """Clear cached resource instances.

        Args:
            resource_type: Specific resource type to clear, or None for all
        """
        if resource_type:
            # Clear only for specific type
            keys_to_remove = [k for k in self._cache if k[0] == resource_type]
            for key in keys_to_remove:
                del self._cache[key]
            logger.debug(f"Cleared cache for resource type: {resource_type}")
        else:
            self._cache.clear()
            logger.debug("Cleared all cached resources")

    def cache_instance(
        self,
        resource_type: str,
        logical_name: str,
        instance: Any,
    ) -> None:
        """Manually cache an instance.

        Useful for pre-populating cache or caching externally created instances.

        Args:
            resource_type: Type of resource
            logical_name: Logical name
            instance: Instance to cache
        """
        self._cache[(resource_type, logical_name)] = instance
        logger.debug(f"Manually cached {resource_type}[{logical_name}]")


class SimpleFactory:
    """Simple factory that creates instances of a class.

    Example:
        ```python
        from myapp import DatabaseConnection

        resolver.register_factory(
            "databases",
            SimpleFactory(DatabaseConnection)
        )
        ```
    """

    def __init__(self, cls: type, **default_kwargs: Any):
        """Initialize with a class to instantiate.

        Args:
            cls: Class to instantiate
            **default_kwargs: Default kwargs merged with config
        """
        self._cls = cls
        self._defaults = default_kwargs

    def create(self, **config: Any) -> Any:
        """Create an instance.

        Args:
            **config: Configuration merged with defaults

        Returns:
            Created instance
        """
        merged = {**self._defaults, **config}
        return self._cls(**merged)


class CallableFactory:
    """Factory that wraps a callable.

    Example:
        ```python
        def create_database(backend, connection_string, **kwargs):
            if backend == "postgres":
                return PostgresDB(connection_string, **kwargs)
            elif backend == "sqlite":
                return SQLiteDB(connection_string, **kwargs)

        resolver.register_factory(
            "databases",
            CallableFactory(create_database)
        )
        ```
    """

    def __init__(self, func: Callable[..., Any], **default_kwargs: Any):
        """Initialize with a callable.

        Args:
            func: Callable that creates resources
            **default_kwargs: Default kwargs merged with config
        """
        self._func = func
        self._defaults = default_kwargs

    def create(self, **config: Any) -> Any:
        """Create an instance.

        Args:
            **config: Configuration merged with defaults

        Returns:
            Created instance
        """
        merged = {**self._defaults, **config}
        return self._func(**merged)


class AsyncCallableFactory:
    """Factory that wraps an async callable.

    Example:
        ```python
        async def create_database(backend, connection_string, **kwargs):
            db = DatabaseConnection(backend, connection_string)
            await db.connect()
            return db

        resolver.register_factory(
            "databases",
            AsyncCallableFactory(create_database)
        )
        ```
    """

    def __init__(self, func: Callable[..., Any], **default_kwargs: Any):
        """Initialize with an async callable.

        Args:
            func: Async callable that creates resources
            **default_kwargs: Default kwargs merged with config
        """
        self._func = func
        self._defaults = default_kwargs

    async def create_async(self, **config: Any) -> Any:
        """Create an instance asynchronously.

        Args:
            **config: Configuration merged with defaults

        Returns:
            Created instance
        """
        merged = {**self._defaults, **config}
        return await self._func(**merged)

    def create(self, **config: Any) -> Any:
        """Sync create is not supported for async factories.

        Raises:
            RuntimeError: Always, use create_async instead
        """
        raise RuntimeError(
            "AsyncCallableFactory requires async context. "
            "Use resolve_async() instead of resolve()."
        )
