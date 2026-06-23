"""Base classes for generic connection pool management."""

from __future__ import annotations

import asyncio
import atexit
import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, TYPE_CHECKING
from weakref import WeakValueDictionary

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


logger = logging.getLogger(__name__)


@dataclass
class _PoolEntry:
    """Manager-owned record for one shared pool on one event loop.

    ``refcount`` is the number of live holders that obtained this pool via
    :meth:`ConnectionPoolManager.get_pool` and have not yet
    :meth:`~ConnectionPoolManager.release_pool`-d it. The pool is closed
    and evicted when the count reaches zero.
    """

    pool: Any
    close_func: Callable | None = None
    refcount: int = 1


class PoolProtocol(Protocol):
    """Protocol for connection pools."""

    async def acquire(self):
        """Acquire a connection from the pool."""
        ...

    async def close(self):
        """Close the pool."""
        ...


PoolType = TypeVar('PoolType', bound=PoolProtocol)


class BasePoolConfig:
    """Base configuration for connection pools."""

    @abstractmethod
    def to_connection_string(self) -> str:
        """Convert configuration to a connection string."""
        ...

    @abstractmethod
    def to_hash_key(self) -> tuple:
        """Create a hashable key for this configuration."""
        ...


class ConnectionPoolManager(Generic[PoolType]):
    """Generic connection pool manager that handles pools per event loop.

    This class ensures that each event loop gets its own connection pool,
    preventing cross-loop usage errors that can occur with async connections.

    Pools are *shared* by config across holders on one event loop:
    :meth:`get_pool` reference-counts each hand-out and :meth:`release_pool`
    tears the pool down only when the last holder releases. The manager is
    used concurrently, so every mutation of an entry that spans an ``await``
    (cold create, validate-rebuild, release-close) is serialized per loop by
    the create lock and removes the entry from the live map *before* awaiting
    the close — a concurrent :meth:`get_pool` can never observe a pool that is
    mid-close.

    Reference-counting does **not** absolve the application of calling
    :meth:`close_all` during shutdown: pools still open at interpreter exit
    (when a loop is running and cannot be driven from the synchronous
    ``atexit`` hook) are abandoned, not closed.
    """

    def __init__(self):
        """Initialize the connection pool manager."""
        # Map of (config_hash, loop_id) -> _PoolEntry.
        self._pools: dict[tuple, _PoolEntry] = {}
        # Weak references to event loops for cleanup
        self._loop_refs: WeakValueDictionary = WeakValueDictionary()
        # Per-loop locks guarding the entry-mutation critical sections
        # (cold create, validate-rebuild, release-close). Keyed on loop_id
        # (an int, never a loop reference, so it cannot hold a loop alive).
        # Mutation is await-free -> atomic under asyncio; an idle lock is
        # evicted once its loop's last pool is gone.
        self._create_locks: dict[int, asyncio.Lock] = {}
        # Register cleanup on exit
        atexit.register(self._cleanup_on_exit)

    def _create_lock_for(self, loop_id: int) -> asyncio.Lock:
        """Return (lazily creating) the create-lock for ``loop_id``.

        Mutation is await-free -> atomic under asyncio's single-threaded
        model, so this needs no lock of its own. The Lock is bound to the
        running loop on first use.
        """
        lock = self._create_locks.get(loop_id)
        if lock is None:
            lock = asyncio.Lock()
            self._create_locks[loop_id] = lock
        return lock

    async def get_pool(
        self,
        config: BasePoolConfig,
        create_pool_func: Callable[[BasePoolConfig], Awaitable[PoolType]],
        validate_pool_func: Callable[[PoolType], Awaitable[None]] | None = None,
        close_pool_func: Callable[[PoolType], Awaitable[None]] | None = None
    ) -> PoolType:
        """Get or create a connection pool for the current event loop.

        The returned pool is *shared* by config across holders on this
        event loop: each hand-out increments a holder count, and the pool
        is torn down only when the last holder calls :meth:`release_pool`.

        Args:
            config: Pool configuration
            create_pool_func: Async function to create a new pool
            validate_pool_func: Optional async function to validate an existing pool
            close_pool_func: Optional async function to close a pool

        Returns:
            Pool instance for the current event loop
        """
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        config_hash = hash(config.to_hash_key())
        pool_key = (config_hash, loop_id)

        # Fast path: warm reuse without taking the create lock. The
        # ``refcount += 1`` is await-free between the membership check and
        # the assignment, so it is atomic under asyncio and needs no lock.
        stale_pool: Any = None  # the pool object we proved invalid below
        if pool_key in self._pools:
            entry = self._pools[pool_key]
            if validate_pool_func is None:
                entry.refcount += 1
                return entry.pool
            try:
                await validate_pool_func(entry.pool)
            except Exception as e:
                logger.warning(
                    "Pool for loop %s is invalid: %s. Recreating.",
                    loop_id, e,
                )
                stale_pool = entry.pool
                # Fall through to the locked create/rebuild region.
            else:
                # Validation succeeded — but ``validate`` awaited, so a
                # concurrent release (last holder) may have closed and
                # evicted this entry, or a rebuild may have replaced it,
                # while we were parked. Commit only if it is still the live
                # entry; otherwise fall through to the locked region to
                # re-resolve (``stale_pool`` stays None — it was not proven
                # invalid, just superseded).
                if self._pools.get(pool_key) is entry:
                    entry.refcount += 1
                    return entry.pool

        # Slow path: create (or rebuild) under the per-loop lock so two
        # concurrent cold-key callers cannot both create a pool and corrupt
        # the holder accounting (the create race re-triggers the very
        # cross-holder close cascade the refcount exists to prevent).
        async with self._create_lock_for(loop_id):
            carried = 0  # holder count carried across a validate-rebuild
            if pool_key in self._pools:
                # Holding the lock, the entry cannot be evicted or replaced
                # mid-await here: release-close and rebuild also hold the
                # lock, and the only lock-free mutator (warm no-validator
                # reuse) merely increments the *same* entry object. So the
                # post-validate re-store below is safe without a re-check.
                entry = self._pools[pool_key]
                if entry.pool is stale_pool:
                    # We already proved this exact pool invalid in the fast
                    # path — don't re-run the (potentially side-effectful)
                    # validator. Rebuild, carrying existing holders forward.
                    carried = entry.refcount
                    await self._close_pool(pool_key, close_pool_func)
                elif validate_pool_func is not None:
                    # A *different* entry appeared while we waited for the
                    # lock (another coroutine rebuilt it) — validate it.
                    try:
                        await validate_pool_func(entry.pool)
                        entry.refcount += 1
                        self._pools[pool_key] = entry
                        return entry.pool
                    except Exception as e:
                        logger.warning(
                            "Pool for loop %s is invalid: %s. Recreating.",
                            loop_id, e,
                        )
                        carried = entry.refcount  # existing holders persist
                        await self._close_pool(pool_key, close_pool_func)
                else:
                    entry.refcount += 1
                    self._pools[pool_key] = entry
                    return entry.pool

            logger.info("Creating new connection pool for loop %s", loop_id)
            pool = await create_pool_func(config)
            self._pools[pool_key] = _PoolEntry(
                pool, close_pool_func, refcount=carried + 1,
            )
            self._loop_refs[loop_id] = loop
            return pool

    async def release_pool(self, config: BasePoolConfig) -> None:
        """Release one holder's claim on the shared pool for this config.

        Decrements the holder count for ``config``'s pool on the current
        event loop. When the last holder releases (count reaches zero), the
        pool is closed (via its registered close-func or ``close()``) and
        evicted from the manager. A no-op when no pool is registered for
        the config (idempotent — safe to call from a double ``close``).

        This is the close path for manager-shared pools: a holder's
        ``close()`` signals "I am done", not "tear the pool down". The pool
        dies only when no holder remains.
        """
        loop_id = id(asyncio.get_running_loop())
        pool_key = (hash(config.to_hash_key()), loop_id)
        # Serialize against cold-create / validate-rebuild on the same loop
        # (which also hold this lock): a release that interleaved with an
        # in-flight create could otherwise lose this decrement or race the
        # eviction. The decrement-and-decision below is await-free, so a
        # lock-free warm reuse can only observe the entry fully present
        # (and bump it) or fully evicted (and rebuild) — never mid-close.
        async with self._create_lock_for(loop_id):
            if pool_key not in self._pools:
                return
            entry = self._pools[pool_key]
            entry.refcount -= 1
            if entry.refcount < 0:
                # Pop-at-zero means the public API cannot normally drive a
                # live entry negative; surface a latent double-release loudly
                # rather than closing one hand-out early and silently.
                logger.warning(
                    "Pool holder count for loop %s went negative (%d): more "
                    "release_pool() calls than get_pool() hand-outs.",
                    loop_id, entry.refcount,
                )
            if entry.refcount <= 0:
                await self._close_pool(pool_key)  # evicts + closes

    def _evict_entry(self, pool_key: tuple) -> _PoolEntry | None:
        """Pop a pool entry out of the live map (await-free, atomic).

        Removing the entry *before* the close is awaited is what makes the
        close atomic with respect to :meth:`get_pool`: a concurrent warm
        reuse can never re-grab a pool that is mid-close, and a missing key
        forces the reuse through the per-loop create lock. Also evicts the
        loop's idle create-lock once its last pool is gone. Returns the
        detached entry, or None if the key was absent.
        """
        entry = self._pools.pop(pool_key, None)
        if entry is None:
            return None
        self._maybe_evict_create_lock(pool_key[1])
        return entry

    def _maybe_evict_create_lock(self, loop_id: int) -> None:
        """Drop the loop's create-lock once it has no pools and is idle.

        Mirrors the ``WeakValueDictionary`` discipline for ``_loop_refs`` —
        don't accrete per-loop state. Never evicts a lock that is currently
        held (a lock only has parked waiters while held), so a rebuild /
        release holding the lock, or a cold-create parked on it, is never
        pulled out from under an in-flight critical section (which would
        re-admit the cold-create race). An in-use lock simply lingers until
        a later idle teardown or :meth:`close_all` reclaims it.
        """
        if any(key[1] == loop_id for key in self._pools):
            return
        lock = self._create_locks.get(loop_id)
        if lock is not None and lock.locked():
            return
        self._create_locks.pop(loop_id, None)

    async def _close_pool(self, pool_key: tuple, close_func: Callable | None = None):
        """Close and remove a pool, ignoring the refcount (force teardown).

        The entry is evicted from the live map (:meth:`_evict_entry`)
        *before* the close is awaited, so the close is atomic with respect
        to a concurrent :meth:`get_pool`.
        """
        entry = self._evict_entry(pool_key)
        if entry is None:
            return
        await self._close_entry(entry, close_func)

    async def _close_entry(self, entry: _PoolEntry, close_func: Callable | None = None):
        """Await the close on an already-detached entry (never touches the map)."""
        pool = entry.pool
        close_func = close_func or entry.close_func
        try:
            # Check if we have a running event loop
            try:
                loop = asyncio.get_running_loop()
                if loop.is_closed():
                    # Event loop is closed, skip async cleanup
                    return
            except RuntimeError:
                # No running event loop, skip async cleanup
                return

            if close_func:
                await close_func(pool)
            elif hasattr(pool, 'close'):
                await pool.close()
        except RuntimeError as e:
            # Silently ignore "Event loop is closed" errors
            if "Event loop is closed" not in str(e):
                logger.error("Error closing pool: %s", e)
        except Exception as e:
            logger.error("Error closing pool: %s", e)

    async def remove_pool(self, config: BasePoolConfig) -> bool:
        """Force-remove a pool for the current event loop, ignoring holders.

        Unlike :meth:`release_pool`, this tears the pool down regardless of
        the holder count (admin / test path).

        Args:
            config: Pool configuration

        Returns:
            True if pool was removed, False if not found
        """
        loop_id = id(asyncio.get_running_loop())
        config_hash = hash(config.to_hash_key())
        pool_key = (config_hash, loop_id)

        if pool_key in self._pools:
            await self._close_pool(pool_key)
            return True
        return False

    async def close_all(self):
        """Close all connection pools (force teardown, ignores holders)."""
        for pool_key in list(self._pools.keys()):
            await self._close_pool(pool_key)
        # _close_pool evicts each loop's idle lock as its last pool goes;
        # clear any stragglers (e.g. locks held/created but never evicted).
        self._create_locks.clear()

    def get_pool_count(self) -> int:
        """Get the number of active pools."""
        return len(self._pools)

    def get_pool_info(self) -> dict[str, Any]:
        """Get information about all active pools."""
        info = {}
        for (config_hash, loop_id), entry in self._pools.items():
            key = f"config_{config_hash}_loop_{loop_id}"
            info[key] = {
                "loop_id": loop_id,
                "config_hash": config_hash,
                "pool": str(entry.pool)
            }
        return info

    def _cleanup_on_exit(self):
        """Cleanup function called on program exit."""
        if not self._pools:
            return

        pool_count = len(self._pools)
        logger.debug("Cleaning up %d connection pool(s) on exit", pool_count)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — create temporary loop for synchronous cleanup
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.close_all())
            except Exception:
                logger.exception("Error closing connection pools during exit")
            finally:
                loop.close()
        else:
            # Running loop exists — cannot reliably await from synchronous
            # atexit, so the still-open pools are abandoned (not closed).
            # Reference-counting does not cover this: the application must
            # call close_all() during shutdown to close pools cleanly.
            logger.warning(
                "%d connection pool(s) not closed before exit. "
                "Ensure close_all() is called during shutdown.",
                pool_count,
            )
            self._pools.clear()
            self._create_locks.clear()
