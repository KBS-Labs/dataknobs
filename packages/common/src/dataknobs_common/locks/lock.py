"""``DistributedLock`` protocol and the shared ``hold()`` mechanic.

Mirrors the established concurrency-primitive style in
``dataknobs-common`` (``RateLimiter`` in :mod:`dataknobs_common.ratelimit`,
``EventBus`` in :mod:`dataknobs_common.events`): a ``@runtime_checkable``
async :class:`~typing.Protocol` keyed by an opaque string, plus an
ergonomic async-context-manager helper that is the primary API.

This is the missing third member of the set
(``RateLimiter``, ``EventBus``, **lock**). The lock is *named* mutual
exclusion: :class:`~dataknobs_common.locks.memory.InProcessLock` (the
default) scopes "all holders" to one process; a cross-replica backend
(registry-pluggable via :data:`~dataknobs_common.locks.factory.lock_backends`)
scopes it to every process pointing at the same backing store.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol, runtime_checkable


@runtime_checkable
class DistributedLock(Protocol):
    """Named mutual exclusion, optionally spanning processes/replicas.

    Implementations guarantee that, for a given ``key``, at most one
    holder across all processes sharing the lock's backing store holds
    it at a time. :class:`InProcessLock` scopes "all processes" to one
    process (behaviour-identical to a bare :class:`asyncio.Lock` per
    key); a Postgres advisory-lock backend scopes it to every process
    pointing at the same database.

    The key is opaque and namespacing is the caller's responsibility
    (e.g. ``f"ingest:{domain_id}"``).

    Example:
        ```python
        from dataknobs_common.locks import create_lock

        lock = create_lock({"backend": "memory"})
        async with lock.hold("ingest:my-domain") as acquired:
            if acquired:
                ...  # critical section
        await lock.close()
        ```
    """

    async def acquire(self, key: str, *, timeout: float | None = None) -> bool:
        """Acquire ``key``, blocking until held or ``timeout``.

        Returns ``True`` on acquisition. Returns ``False`` if a finite
        ``timeout`` elapsed first. ``timeout=None`` blocks indefinitely
        and always returns ``True``.

        A failed acquisition is an expected control-flow outcome (skip
        or retry), **not** an exception — this deliberately differs from
        :meth:`RateLimiter.acquire`, where exhaustion is exceptional and
        raises ``TimeoutError``. Lock contention is routine.
        """
        ...

    async def release(self, key: str) -> None:
        """Release ``key``. No-op if not held by this instance."""
        ...

    def hold(
        self, key: str, *, timeout: float | None = None
    ) -> AbstractAsyncContextManager[bool]:
        """Async context manager wrapping acquire/release.

        ``async with lock.hold(key) as got:`` — ``got`` is the
        :meth:`acquire` result; :meth:`release` runs on exit even on
        error. This is the primary API; :meth:`acquire`/:meth:`release`
        are the low-level escape hatch.
        """
        ...

    async def close(self) -> None:
        """Release backing resources (connections, tasks)."""
        ...


@asynccontextmanager
async def _hold(
    lock: DistributedLock, key: str, timeout: float | None
) -> AsyncIterator[bool]:
    """Shared ``hold()`` implementation reused by every backend.

    ``acquire`` → yield the acquired flag → ``release`` in ``finally``
    (only if it was actually acquired). Implemented once here per the
    shared-behaviour-extraction mandate: every backend's ``hold()`` is
    ``return _hold(self, key, timeout)`` — the acquire/release/finally
    contract is never re-derived per backend, so it cannot drift
    between implementations.
    """
    got = await lock.acquire(key, timeout=timeout)
    try:
        yield got
    finally:
        if got:
            await lock.release(key)
