"""Distributed lock abstraction for cross-replica critical sections.

The missing third member of DataKnobs' concurrency-primitive set
(``RateLimiter``, ``EventBus``, **lock**). Provides a
:class:`DistributedLock` protocol, an :class:`InProcessLock` default
(single-process, the right behaviour for the common single-replica
case and the testing construct), and a registry-extensible
:func:`create_lock` factory mirroring
:func:`dataknobs_common.events.create_event_bus`.

Example:
    ```python
    from dataknobs_common.locks import create_lock

    # Single-process default
    lock = create_lock({"backend": "memory"})

    async with lock.hold("ingest:my-domain") as acquired:
        if acquired:
            ...  # critical section — serialized per key

    await lock.close()
    ```

Multi-replica deployments inject a cross-replica backend (e.g. a
Postgres advisory-lock backend, registry-pluggable via
:data:`lock_backends`) — application code is unchanged.
"""

from __future__ import annotations

from .config import PostgresLockConfig
from .factory import LockFactory, create_lock, lock_backends
from .lock import DistributedLock
from .memory import InProcessLock
from .postgres import PostgresAdvisoryLock

__all__ = [
    # Protocol
    "DistributedLock",
    # Factory
    "create_lock",
    # Plugin registry
    "lock_backends",
    "LockFactory",
    # Default / testing implementation
    "InProcessLock",
    # Cross-replica implementation
    "PostgresAdvisoryLock",
    # Typed config
    "PostgresLockConfig",
]
