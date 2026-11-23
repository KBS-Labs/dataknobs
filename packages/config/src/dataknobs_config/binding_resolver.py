# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

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

import inspect
import logging
from typing import Any, Callable, Protocol, runtime_checkable

from dataknobs_common.exceptions import _is_written_by_dataknobs, _written_by_dataknobs

from .builders import _is_consumer_class
from .environment_aware import resolve_resource_references
from .environment_config import EnvironmentConfig

logger = logging.getLogger(__name__)


class BindingResolverError(Exception):
    """Error during resource binding resolution.

    When a factory fails, the message names the resource and the factory
    error's type, never its text -- a factory builds a backend from deployment
    config, and its text can carry the connection URL it was handed. The
    factory's error is on ``__cause__``, for a server-side log.
    """

    pass


class FactoryNotFoundError(BindingResolverError):
    """Factory not registered for resource type."""

    pass


def _creation_failed(
    resource_type: str, logical_name: str, error: Exception
) -> BindingResolverError:
    """The error for a factory that raised: the resource, and what of the error may be shown.

    A refusal dataknobs wrote -- an unknown provider name, say -- names only
    what the config's author wrote, so its words follow. Anything else is
    named by its type, because a factory builds from deployment config and
    its text can carry the connection URL it was handed; the original is on
    ``__cause__`` either way.
    """
    if _is_written_by_dataknobs(error):
        return BindingResolverError(f"Failed to create {resource_type}[{logical_name}]: {error}")
    return BindingResolverError(
        f"Failed to create {resource_type}[{logical_name}] ({type(error).__name__})"
    )


def _refuse_awaitable(result: Any, factory: object, resource_type: str, logical_name: str) -> None:
    """Refuse an awaitable a factory returned to :meth:`ConfigBindingResolver.resolve`.

    A synchronous caller cannot await, so the only honest answers are the
    instance or an error. Returning the coroutine was the third: it reached
    the caller as the resolved resource, was cached as one, and failed far
    from here, if at all -- an un-awaited coroutine raises nothing and warns at
    garbage collection, naming the factory rather than the resolver.

    Raised before anything is cached, and the awaitable is closed so refusing
    it does not itself emit that warning. The same rule as
    :meth:`~dataknobs_common.registry.PluginRegistry._refuse_awaitable`, in the
    resolver's error type.
    """
    if not inspect.isawaitable(result):
        return
    # A coroutine is closed; a Task or Future has no ``close`` and would go on
    # running -- holding a half-opened connection, or logging an exception
    # nobody retrieves -- so it is cancelled instead.
    close = getattr(result, "close", None)
    cancel = getattr(result, "cancel", None)
    if callable(close):
        close()
    elif callable(cancel):
        cancel()
    # Named by its own name when it has one -- a function's or a class's,
    # which is its author's code -- else by its type. Never its ``repr``,
    # which is its author's text and can carry the connection URL it holds.
    name = getattr(factory, "__qualname__", None)
    if not isinstance(name, str):
        name = type(factory).__name__
    raise _written_by_dataknobs(
        BindingResolverError(
            f"Factory {name} for {resource_type}[{logical_name}] "
            f"is asynchronous; use resolve_async() instead of resolve()."
        )
    )


@runtime_checkable
class ResourceFactory(Protocol):
    """Protocol for resource factories that can create from config.

    The usual shape. The resolver also accepts ``build(**config)``, a plain
    callable, and a class or other object with ``from_config`` -- see
    :meth:`ConfigBindingResolver.register_factory`.
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

    **Missing resources raise, deliberately.** This API takes a ``(type,
    name)`` pair and nothing else, so there is no reference to read a policy
    off and no inline defaults to degrade to -- resolving a name this
    environment does not define can only fail. That makes this **the strict
    policy**, not an accident of implementation.

    The lenient, declarable behaviour belongs to the ``$resource`` reference
    syntax, where a reference can carry ``$required`` and inline defaults; see
    :meth:`EnvironmentAwareConfig.resolve_for_build`.

    Everything below the entry point is the same code:
    :func:`~dataknobs_config.resolve_resource_references` resolves the config
    this looks up, so a reference *nested* inside a resource is read here
    exactly as it is there -- same marker validation, same precedence chain,
    same exception types, same cycle guard. The two APIs differ in one thing
    only: what happens at the top, where this has no reference to read a
    policy off and so can only be strict.

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
            resolve_env_vars: Whether to substitute env vars before
                instantiation. Applies only to environments that have not
                already been substituted — an ``EnvironmentConfig`` loaded
                via :meth:`EnvironmentConfig.load` /
                :meth:`EnvironmentConfig.from_dict` records that it was
                (:attr:`EnvironmentConfig.substituted`) and is not
                substituted again here, because substitution is not
                idempotent for a value whose own text contains ``${...}``.
                Load-bearing for dataclass-constructed instances. To opt
                out of all substitution, pass ``substitute_vars=False`` to
                ``EnvironmentConfig.load`` and ``resolve_env_vars=False``
                here.
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
            factory: What builds the resource: a factory object with
                ``create(**config)`` (and optionally ``create_async``) or
                ``build(**config)``, a callable taking the config as keyword
                arguments, or a class or other object with
                ``from_config(config)`` (and optionally ``from_config_async``)
                and none of the factory methods. :meth:`_constructor` states
                the precedence.
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
            ResourceNotFoundError: If this environment does not define the
                resource. Strict by design -- see the class docstring
            FactoryNotFoundError: If no factory registered for resource type
            BindingResolverError: If resource creation fails, or the factory
                is asynchronous -- it returned an awaitable, which this
                method cannot await; call :meth:`resolve_async` instead.
                Nothing is cached in either case.
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

        # Find how to call the factory first, so a factory with no callable
        # surface is refused in words of the resolver's own, then call it.
        construct = self._constructor(factory, can_await=False)
        try:
            instance = construct(config)
        except Exception as e:
            raise _creation_failed(resource_type, logical_name, e) from e
        _refuse_awaitable(instance, factory, resource_type, logical_name)

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
        """Async version of resolve, which also builds from async factories.

        Construction precedence (see :meth:`_constructor`): a registered
        class's ``from_config_async``, then a factory's ``create_async``, then
        what :meth:`resolve` uses -- ``from_config``, ``create``, ``build``,
        or the factory itself. Whatever the chosen surface returns is awaited
        when it is awaitable, so a plain ``async def`` factory, or one whose
        ``create`` or ``build`` is ``async def``, resolves to its instance.

        Args:
            resource_type: Type of resource
            logical_name: Logical name from app config
            use_cache: Whether to return cached instance
            **overrides: Config overrides for this resolution

        Returns:
            Instantiated resource

        Raises:
            ResourceNotFoundError: If this environment does not define the
                resource. Strict by design -- see the class docstring
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

        # Create instance (async if supported). As in ``resolve``, the
        # factory's surface is chosen before the call.
        construct = self._constructor(factory, can_await=True)
        try:
            instance = construct(config)
            if inspect.isawaitable(instance):
                instance = await instance
        except Exception as e:
            raise _creation_failed(resource_type, logical_name, e) from e

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

        Raises:
            ResourceNotFoundError: If the resource is absent, or if one a
                reference nested inside it names is. No ``defaults`` are
                passed, which under :meth:`EnvironmentConfig.get_resource` is
                the strict policy -- and it is the right one here, because
                this API is reached without a reference that could have
                declared otherwise. A reference *nested* in the resource does
                have one, and it is read there as anywhere else.
            ConfigError: If a nested reference is malformed, names a resource
                that does not declare a capability it ``$requires``, or
                reaches itself
        """
        config = self._environment.get_resource(resource_type, logical_name)

        # Each source is expanded and its references resolved on its own
        # terms, before the two are merged beyond distinguishing. That split
        # is the reason for two calls rather than one over the merged dict:
        # the environment's values may already be expanded while the caller's
        # overrides never are, and a single flag cannot be true of both.
        #
        # Substitution is NOT idempotent: a value whose own text contains
        # ${...} would be expanded a second time, replacing it with whatever
        # unrelated variable its content happened to name.
        # EnvironmentConfig.load() and from_dict() substitute by default and
        # record it on ``substituted``, so the environment's pass runs only
        # for directly-constructed instances (or ones loaded with
        # substitute_vars=False), where it is load-bearing.
        #
        # Resolving references here is what stops this API disagreeing with
        # the reference resolver about how far a binding reaches. A resource
        # may itself carry a `$resource` block, and handing that to a factory
        # as a literal `{"$resource": ...}` keyword argument is the same
        # silent degrade the reference vocabulary exists to close -- one
        # layer down, where nothing was looking.
        #
        # Annotated because the resolver walks an arbitrary config tree and so
        # is declared ``Any``; a resource entry is a mapping, and this method
        # is where that is promised.
        resolved: dict[str, Any] = resolve_resource_references(
            config,
            self._environment,
            substitute=self._resolve_env_vars and not self._environment.substituted,
            strict_resources=True,
        )

        # Overrides are a separate source -- caller-supplied and never
        # previously substituted -- so they get their own single pass
        # regardless of the environment's provenance. Whether they should be
        # substituted at all is a separate question (199-FU2); this preserves
        # the behaviour they have always had.
        if overrides:
            resolved_overrides: dict[str, Any] = resolve_resource_references(
                overrides,
                self._environment,
                substitute=self._resolve_env_vars,
                strict_resources=True,
            )
            resolved = {**resolved, **resolved_overrides}

        return resolved

    def _get_factory(self, resource_type: str) -> ResourceFactory | Callable[..., Any]:
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

    def _constructor(
        self,
        # `object`, not the registered type: the resolver accepts surfaces
        # (`build`, `from_config`) that type does not describe.
        factory: object,
        *,
        can_await: bool,
    ) -> Callable[[dict[str, Any]], Any]:
        """How to build from ``factory``, for a caller that can or cannot await.

        The one statement of the precedence both :meth:`resolve` and
        :meth:`resolve_async` use, so the two cannot drift again:

        * a **consumer class** -- a class with ``from_config`` and none of
          ``create``, ``create_async`` or ``build`` -- is built through
          ``from_config_async(config)`` when the caller can await and it has
          one, else ``from_config(config)``;
        * any other factory through ``create_async(**config)`` when the caller
          can await, then ``create(**config)``, ``build(**config)``, or
          ``factory(**config)``;
        * last, an object that is not a class and has none of those -- a
          registered singleton, a module -- through ``from_config_async(config)``
          when the caller can await and it has one, else ``from_config(config)``.

        That is ``ObjectBuilder``'s order for a ``factory:`` reference.

        The consumer-class test is ``ObjectBuilder``'s
        (:func:`~dataknobs_config.builders._is_consumer_class`), and only a
        class passes it: ``ConfigurableBase`` gives every subclass a
        ``from_config`` classmethod, so a factory *instance* deriving from it
        answers ``hasattr`` too, and its ``create`` is what builds the
        resource. ``from_config`` / ``from_config_async`` take the whole
        config; the rest take it as keyword arguments.

        The result may be an awaitable. :meth:`resolve_async` awaits it;
        :meth:`resolve` refuses it (:func:`_refuse_awaitable`).

        Raises:
            BindingResolverError: When ``factory`` has none of these.
        """
        # `hasattr` first is what lets the type checker see the attribute;
        # `_is_consumer_class` is what decides.
        if (
            can_await
            and hasattr(factory, "from_config_async")
            and _is_consumer_class(factory, "from_config_async")
        ):
            from_config_async: Callable[[dict[str, Any]], Any] = factory.from_config_async
            return from_config_async
        if hasattr(factory, "from_config") and _is_consumer_class(factory):
            from_config: Callable[[dict[str, Any]], Any] = factory.from_config
            return from_config
        if can_await and hasattr(factory, "create_async"):
            create_async = factory.create_async
            return lambda config: create_async(**config)
        if hasattr(factory, "create"):
            create = factory.create
            return lambda config: create(**config)
        if hasattr(factory, "build"):
            build = factory.build
            return lambda config: build(**config)
        if callable(factory):
            call = factory
            return lambda config: call(**config)
        # An object that is not a class -- a registered singleton, a module --
        # with only the consumer protocol, after the factory protocol and the
        # `callable` fallback, as `ObjectBuilder` orders it.
        if can_await and hasattr(factory, "from_config_async"):
            object_from_config_async: Callable[[dict[str, Any]], Any] = factory.from_config_async
            return object_from_config_async
        if hasattr(factory, "from_config"):
            object_from_config: Callable[[dict[str, Any]], Any] = factory.from_config
            return object_from_config
        # Named by its type: a factory object's ``repr`` is its author's, and
        # can carry the connection URL it holds.
        raise BindingResolverError(
            f"Factory {type(factory).__name__} has none of from_config(), "
            f"from_config_async(), create(), create_async(), build(), or __call__"
        )

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
        # Marked as dataknobs' words, so the resolver's bounded error shows them
        # rather than only "(RuntimeError)".
        raise _written_by_dataknobs(
            RuntimeError(
                "AsyncCallableFactory requires async context. "
                "Use resolve_async() instead of resolve()."
            )
        )
