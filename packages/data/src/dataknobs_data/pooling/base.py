"""Base classes for generic connection pool management."""

from __future__ import annotations

import asyncio
import atexit
import logging
from abc import abstractmethod
from typing import Any, Generic, Protocol, TypeVar, TYPE_CHECKING
from weakref import WeakValueDictionary

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


logger = logging.getLogger(__name__)


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
        # Map of (config_hash, loop_id) -> pool or (pool, close_func)
        self._pools: dict[tuple, PoolType | tuple[PoolType, Callable | None]] = {}
        # Weak references to event loops for cleanup
        self._loop_refs: WeakValueDictionary = WeakValueDictionary()
        # Register cleanup on exit
        atexit.register(self._cleanup_on_exit)

    async def get_pool(
        self,
        config: BasePoolConfig,
        create_pool_func: Callable[[BasePoolConfig], Awaitable[PoolType]],
        validate_pool_func: Callable[[PoolType], Awaitable[None]] | None = None,
        close_pool_func: Callable[[PoolType], Awaitable[None]] | None = None
    ) -> PoolType:
        """Get or create a connection pool for the current event loop.
        
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

        # Check if we already have a pool for this config and loop
        if pool_key in self._pools:
            pool_entry = self._pools[pool_key]
            # Handle both old and new format
            if isinstance(pool_entry, tuple):
                pool, _ = pool_entry
            else:
                # Non-tuple format (backward compatibility)
                pool = pool_entry

            # Validate the pool if validation function provided
            if validate_pool_func:
                try:
                    await validate_pool_func(pool)
                    return pool
                except Exception as e:
                    logger.warning(f"Pool for loop {loop_id} is invalid: {e}. Creating new one.")
                    await self._close_pool(pool_key, close_pool_func)
            else:
                return pool

        # Create new pool
        logger.info(f"Creating new connection pool for loop {loop_id}")
        pool = await create_pool_func(config)

        # Store pool and loop reference with close function
        self._pools[pool_key] = (pool, close_pool_func)
        self._loop_refs[loop_id] = loop

        return pool

    async def _close_pool(self, pool_key: tuple, close_func: Callable | None = None):
        """Close and remove a pool."""
        if pool_key in self._pools:
            pool_entry = self._pools[pool_key]
            # Handle both old format (pool) and new format (pool, close_func)
            if isinstance(pool_entry, tuple):
                pool, stored_close_func = pool_entry
                close_func = close_func or stored_close_func
            else:
                # Non-tuple format (backward compatibility)
                pool = pool_entry

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

    async def remove_pool(self, config: BasePoolConfig) -> bool:
        """Remove a pool for the current event loop.
        
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
        """Close all connection pools."""
        for pool_key in list(self._pools.keys()):
            await self._close_pool(pool_key)

    def get_pool_count(self) -> int:
        """Get the number of active pools."""
        return len(self._pools)

    def get_pool_info(self) -> dict[str, Any]:
        """Get information about all active pools."""
        info = {}
        for (config_hash, loop_id), pool_entry in self._pools.items():
            # Handle both old and new format
            if isinstance(pool_entry, tuple):
                pool, _ = pool_entry
            else:
                # Non-tuple format (backward compatibility)
                pool = pool_entry

            key = f"config_{config_hash}_loop_{loop_id}"
            info[key] = {
                "loop_id": loop_id,
                "config_hash": config_hash,
                "pool": str(pool)
            }
        return info

    def _cleanup_on_exit(self):
        """Cleanup function called on program exit."""
        if self._pools:
            logger.debug(f"Cleaning up {len(self._pools)} connection pools on exit")
            # Try to get any running loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, try to create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.close_all())
                finally:
                    loop.close()
            else:
                # There's a running loop, schedule cleanup
                asyncio.create_task(self.close_all())
