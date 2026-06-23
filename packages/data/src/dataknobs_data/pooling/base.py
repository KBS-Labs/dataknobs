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
    """

    def __init__(self):
        """Initialize the connection pool manager."""
        # Map of (config_hash, loop_id) -> _PoolEntry. Legacy shapes (a
        # bare pool, or a (pool, close_func) 2-tuple) may still be injected
        # directly by tests / pre-refcount callers; read paths normalize
        # them through ``_as_entry``.
        self._pools: dict[tuple, Any] = {}
        # Weak references to event loops for cleanup
        self._loop_refs: WeakValueDictionary = WeakValueDictionary()
        # Per-loop locks guarding the cold-key create critical section.
        # Keyed on loop_id (an int, never a loop reference, so it cannot
        # hold a loop alive). Mutation is await-free -> atomic under
        # asyncio; evicted alongside a loop's last pool.
        self._create_locks: dict[int, asyncio.Lock] = {}
        # Register cleanup on exit
        atexit.register(self._cleanup_on_exit)

    @staticmethod
    def _as_entry(raw: Any) -> _PoolEntry:
        """Normalize a stored value to a :class:`_PoolEntry`.

        Tolerates the two legacy shapes some tests / pre-refcount callers
        inject directly: a ``(pool, close_func)`` 2-tuple and a bare pool.
        """
        if isinstance(raw, _PoolEntry):
            return raw
        if isinstance(raw, tuple):  # legacy (pool, close_func)
            pool, close_func = raw
            return _PoolEntry(pool, close_func, refcount=1)
        return _PoolEntry(raw, None, refcount=1)  # legacy bare pool

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
            entry = self._as_entry(self._pools[pool_key])
            if validate_pool_func is None:
                entry.refcount += 1
                self._pools[pool_key] = entry  # persist (normalizes legacy)
                return entry.pool
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
                stale_pool = entry.pool
                # Fall through to the locked create/rebuild region.

        # Slow path: create (or rebuild) under the per-loop lock so two
        # concurrent cold-key callers cannot both create a pool and corrupt
        # the holder accounting (the create race re-triggers the very
        # cross-holder close cascade the refcount exists to prevent).
        async with self._create_lock_for(loop_id):
            carried = 0  # holder count carried across a validate-rebuild
            if pool_key in self._pools:
                entry = self._as_entry(self._pools[pool_key])
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
        pool_key = (
            hash(config.to_hash_key()),
            id(asyncio.get_running_loop()),
        )
        if pool_key not in self._pools:
            return
        entry = self._as_entry(self._pools[pool_key])
        entry.refcount -= 1
        if entry.refcount <= 0:
            await self._close_pool(pool_key)  # closes + dels
        else:
            self._pools[pool_key] = entry  # persist decremented count

    async def _close_pool(self, pool_key: tuple, close_func: Callable | None = None):
        """Close and remove a pool, ignoring the refcount (force teardown)."""
        if pool_key in self._pools:
            entry = self._as_entry(self._pools[pool_key])
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
                    logger.error(f"Error closing pool: {e}")
            except Exception as e:
                logger.error(f"Error closing pool: {e}")
            finally:
                del self._pools[pool_key]
                # Evict the loop's create-lock once its last pool is gone
                # (mirror the WeakValueDictionary discipline for _loop_refs:
                # don't accrete per-loop state).
                loop_id = pool_key[1]
                if not any(key[1] == loop_id for key in self._pools):
                    self._create_locks.pop(loop_id, None)

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
        # _close_pool evicts each loop's lock as its last pool goes; clear
        # any stragglers (e.g. locks created but never used).
        self._create_locks.clear()

    def get_pool_count(self) -> int:
        """Get the number of active pools."""
        return len(self._pools)

    def get_pool_info(self) -> dict[str, Any]:
        """Get information about all active pools."""
        info = {}
        for (config_hash, loop_id), pool_entry in self._pools.items():
            pool = self._as_entry(pool_entry).pool

            key = f"config_{config_hash}_loop_{loop_id}"
            info[key] = {
                "loop_id": loop_id,
                "config_hash": config_hash,
                "pool": str(pool)
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
            # Running loop exists — cannot reliably await from synchronous atexit.
            # Application should have called close_all() during shutdown.
            logger.warning(
                "%d connection pool(s) not closed before exit. "
                "Ensure close_all() is called during shutdown.",
                pool_count,
            )
            self._pools.clear()
            self._create_locks.clear()
