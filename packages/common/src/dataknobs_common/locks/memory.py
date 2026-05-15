"""``InProcessLock`` — the default single-process :class:`DistributedLock`.

Behaviour-identical to a bare :class:`asyncio.Lock` per key, so it is
the zero-config, zero-dependency default and the right choice for the
common single-replica case. It is also the **testing construct** for
the lock primitive — use it instead of mocking a lock (a fake that
appends to a list has the same blindness as ``MagicMock``).

For multi-replica deployments inject a cross-replica backend (e.g. a
Postgres advisory-lock backend) instead; see
:data:`~dataknobs_common.locks.factory.lock_backends`.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager

from .lock import _hold


class InProcessLock:
    """An :class:`asyncio.Lock`-per-key map with reference-counted eviction.

    Mutual exclusion is scoped to **this process**. The key→lock map is
    reference-count evicted so it cannot grow unbounded as distinct keys
    come and go (closing the never-evicting per-domain lock-map leak
    that the previous inline ``IngestOrchestrator`` implementation had).

    The eviction bookkeeping is the one subtle part — which is exactly
    why it belongs in a single reviewed shared primitive rather than
    being re-derived inline per consumer.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._refs: dict[str, int] = {}
        self._guard = asyncio.Lock()  # protects ``_locks`` / ``_refs``

    async def acquire(self, key: str, *, timeout: float | None = None) -> bool:
        """Acquire ``key``. See :meth:`DistributedLock.acquire`."""
        async with self._guard:
            lk = self._locks.get(key)
            if lk is None:
                lk = self._locks[key] = asyncio.Lock()
            self._refs[key] = self._refs.get(key, 0) + 1
        try:
            if timeout is None:
                await lk.acquire()
                return True
            try:
                await asyncio.wait_for(lk.acquire(), timeout)
                return True
            except asyncio.TimeoutError:
                return False
        finally:
            # Drop this caller's reference; evict the entry only when no
            # other caller references it AND the lock is not held. While
            # a caller holds the lock, ``lk.locked()`` is True so the
            # entry survives until ``release``.
            async with self._guard:
                self._refs[key] -= 1
                if self._refs[key] == 0 and not lk.locked():
                    self._locks.pop(key, None)
                    self._refs.pop(key, None)

    async def release(self, key: str) -> None:
        """Release ``key``. No-op if not held. See the protocol."""
        async with self._guard:
            lk = self._locks.get(key)
            if lk is None or not lk.locked():
                return
            lk.release()
            # Read + release + eviction check are one await-free guarded
            # section: a waiter woken by ``lk.release()`` above cannot
            # run until we exit, so its acquire-start ref increment
            # (also taken under ``_guard``) is still counted here.
            # ``_refs == 0`` therefore means "no holder, acquirer, or
            # waiter references this key", which is exactly when
            # eviction is safe — with no reliance on event-loop
            # scheduling order or ``asyncio.Lock`` FIFO fairness.
            if self._refs.get(key, 0) == 0:
                self._locks.pop(key, None)
                self._refs.pop(key, None)

    def hold(
        self, key: str, *, timeout: float | None = None
    ) -> AbstractAsyncContextManager[bool]:
        """Async CM wrapping acquire/release. See the protocol."""
        return _hold(self, key, timeout)

    async def close(self) -> None:
        """No-op — :class:`InProcessLock` holds no external resources."""
        return None
