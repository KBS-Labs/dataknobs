"""Registry-extensible factory for distributed-lock backends.

``create_lock()`` resolves the ``backend`` config key through this
registry. Out-of-tree consumers can add a custom
:class:`~dataknobs_common.locks.lock.DistributedLock` backend
(Redis, etcd, ZooKeeper, …) without forking DataKnobs::

    from dataknobs_common.locks import lock_backends, create_lock

    def _make_redis_lock(config):
        from my_pkg.redis_lock import RedisLock
        return RedisLock(url=config["url"])

    lock_backends.register("redis", _make_redis_lock)
    lock = create_lock({"backend": "redis", "url": "..."})

The registry is a :class:`~dataknobs_common.registry.PluginRegistry` —
the shared config-driven factory abstraction also used by
``event_bus_backends`` and the bots-side ``memory_backends`` /
``knowledge_base_backends`` / ``source_backends``. Resolution of the
``backend`` discriminator, the not-found error shape ("Unknown lock
backend: <name>. Available backends: …"), the ``ValueError`` exception
class, and the lazy-init flow live in :class:`PluginRegistry`; this
module declares the per-domain knobs (kind label, validate_type, default
backend) and the two built-in backend factories.

Each built-in wrapper imports its concrete backend *lazily* (inside the
factory call) so importing this module never pulls optional backend
dependencies (asyncpg) at module load time, preserving the
``dependencies = []`` base install.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dataknobs_common.registry import PluginRegistry

from .lock import DistributedLock

LockFactory = Callable[[dict[str, Any]], DistributedLock]
"""A backend factory: maps a config dict to a :class:`DistributedLock`.

Preserved as a public typealias for out-of-tree consumers that annotate
their factory closures. The registry holds factories of this shape; no
behavioural difference from the underlying ``Callable``.
"""

lock_backends: PluginRegistry[DistributedLock] = PluginRegistry(
    name="lock_backends",
    validate_type=DistributedLock,
    config_key="backend",
    config_key_default="memory",
    not_found_kind="lock backend",
    not_found_exception=ValueError,
)
"""Registry of named :data:`LockFactory` callables.

Register a custom backend with
``lock_backends.register("name", factory)`` and select it via
``create_lock({"backend": "name", ...})``. The registry conforms
to :class:`~dataknobs_common.registry.BackendRegistry` for ``isinstance``
checks.
"""


def _create_in_process_lock(config: dict[str, Any]) -> DistributedLock:
    from .memory import InProcessLock

    return InProcessLock()


def _create_postgres_lock(config: dict[str, Any]) -> DistributedLock:
    # Lazy import keeps ``factory.py`` (and ``dataknobs_common.locks``)
    # importable without asyncpg — the optional ``postgres`` extra is
    # only required when this backend is actually selected.
    from .postgres import PostgresAdvisoryLock

    # ``from_config`` routes the dict through ``PostgresLockConfig.from_dict``
    # → ``_normalize_dict`` → ``normalize_postgres_connection_config``,
    # identical resolution to direct ``PostgresAdvisoryLock(config=config)``.
    return PostgresAdvisoryLock.from_config(config)


lock_backends.register("memory", _create_in_process_lock)
lock_backends.register("postgres", _create_postgres_lock)


def create_lock(config: dict[str, Any]) -> DistributedLock:
    """Create a distributed lock from configuration.

    Factory function that creates the appropriate
    :class:`DistributedLock` implementation based on the ``backend`` key
    in the config. Backends are resolved through the
    :data:`lock_backends` registry, so out-of-tree consumers can
    register and select a custom backend without forking DataKnobs:

        ```python
        from dataknobs_common.locks import lock_backends, create_lock

        lock_backends.register("redis", my_redis_lock_factory)
        lock = create_lock({"backend": "redis", "url": "..."})
        ```

    Args:
        config: Configuration dict with a ``backend`` key (default
            ``"memory"``) and backend-specific options.

    Returns:
        A :class:`DistributedLock` instance.

    Raises:
        ValueError: If the backend is not registered. The message lists
            all registered backends (including consumer-registered ones).
        OperationError: If the backend factory raises during construction
            (invalid config, missing required fields, etc.). Wraps the
            originating exception via ``__cause__``.

    Example:
        ```python
        # In-process lock (default)
        lock = create_lock({"backend": "memory"})
        lock = create_lock({})  # equivalent — "memory" is the default
        ```
    """
    return lock_backends.create(config=config)


async def create_lock_async(config: dict[str, Any]) -> DistributedLock:
    """Async-symmetric counterpart to :func:`create_lock`.

    For backends whose construction is asynchronous (eager-connecting
    asyncpg pools, etcd / ZooKeeper sessions, …). Today every built-in
    backend constructs synchronously, so this function returns the same
    instance type as :func:`create_lock`; the surface is shipped for API
    symmetry and consumer-extensibility (an out-of-tree backend's
    ``from_config_async`` is detected and awaited via
    :meth:`PluginRegistry.create_async`).

    Args:
        config: Configuration dict with a ``backend`` key and
            backend-specific options.

    Returns:
        A :class:`DistributedLock` instance.

    Raises:
        ValueError: If the backend is not registered. The message lists
            all registered backends (including consumer-registered ones).
        OperationError: If the backend factory raises during construction
            (invalid config, missing required fields, etc.). Wraps the
            originating exception via ``__cause__``. Same behaviour as
            the sync :func:`create_lock`.
    """
    return await lock_backends.create_async(config=config)
