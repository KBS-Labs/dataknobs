"""Registry-extensible factory for distributed-lock backends.

``create_lock()`` resolves the ``backend`` config key through this
registry instead of a sealed ``if/elif`` chain. Out-of-tree consumers
can add a custom :class:`~dataknobs_common.locks.lock.DistributedLock`
backend (Redis, etcd, ZooKeeper, …) without forking DataKnobs::

    from dataknobs_common.locks import lock_backends, create_lock

    def _make_redis_lock(config):
        from my_pkg.redis_lock import RedisLock
        return RedisLock(url=config["url"])

    lock_backends.register("redis", _make_redis_lock)
    lock = create_lock({"backend": "redis", "url": "..."})

This is the exact structural mirror of
:data:`dataknobs_common.events.event_bus_backends`; the two stay
consistent so the pattern is learned once and applied everywhere.

Each built-in wrapper imports its concrete backend *lazily* (inside the
factory call) so importing this module never pulls optional backend
dependencies at module load time, preserving the ``dependencies = []``
base install.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dataknobs_common.registry import Registry

from .lock import DistributedLock

LockFactory = Callable[[dict[str, Any]], DistributedLock]
"""A backend factory: maps a config dict to a :class:`DistributedLock`."""

lock_backends: Registry[LockFactory] = Registry(name="lock_backends")
"""Registry of named :data:`LockFactory` callables.

Register a custom backend with
``lock_backends.register("name", factory)`` and select it via
``create_lock({"backend": "name", ...})``.
"""


def _create_in_process_lock(config: dict[str, Any]) -> DistributedLock:
    from .memory import InProcessLock

    return InProcessLock()


lock_backends.register("memory", _create_in_process_lock)
# Note: the "postgres" backend (PostgresAdvisoryLock) is not yet
# registered alongside locks/postgres.py. Until then an unknown "postgres"
# backend resolves to the same ValueError as any other unknown backend,
# keeping this change behaviour-identical for the single built-in.


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

    Example:
        ```python
        # In-process lock (default)
        lock = create_lock({"backend": "memory"})
        lock = create_lock({})  # equivalent — "memory" is the default
        ```
    """
    backend = config.get("backend", "memory")
    # NOTE: Registry.get() raises NotFoundError; only get_optional()
    # returns None. This must mirror create_event_bus() exactly — the
    # two factories share one corrected resolution pattern.
    factory = lock_backends.get_optional(backend)
    if factory is None:
        available = ", ".join(sorted(lock_backends.list_keys()))
        raise ValueError(
            f"Unknown lock backend: {backend}. "
            f"Available backends: {available}"
        )
    return factory(config)
