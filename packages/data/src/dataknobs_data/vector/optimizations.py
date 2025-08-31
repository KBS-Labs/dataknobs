"""Vector store optimization and performance enhancements."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from .types import DistanceMetric


logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch operations."""

    size: int = 100
    max_queue_size: int = 10000
    flush_interval: float = 1.0  # seconds
    parallel_workers: int = 4
    retry_on_failure: bool = True
    max_retries: int = 3


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pooling."""

    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    recycle_timeout: float = 3600.0


class BatchProcessor:
    """Handles batch processing of vector operations."""

    def __init__(self, config: BatchConfig | None = None):
        """Initialize the batch processor.
        
        Args:
            config: Batch configuration
        """
        self.config = config or BatchConfig()
        self.queue: deque = deque(maxlen=self.config.max_queue_size)
        self.lock = Lock()
        self.processing = False
        self._flush_task: asyncio.Task | None = None

    async def add(self, item: Any, callback: Callable | None = None) -> None:
        """Add an item to the batch queue.
        
        Args:
            item: Item to process
            callback: Optional callback when item is processed
        """
        should_flush = False
        with self.lock:
            self.queue.append((item, callback))
            # Check if we should flush
            if len(self.queue) >= self.config.size:
                should_flush = True

        # Flush outside of lock to avoid deadlock
        if should_flush:
            await self.flush()

    async def flush(self) -> int:
        """Process all items in the queue.
        
        Returns:
            Number of items processed
        """
        items_to_process = []

        with self.lock:
            # Get batch of items
            batch_size = min(len(self.queue), self.config.size)
            for _ in range(batch_size):
                if self.queue:
                    items_to_process.append(self.queue.popleft())

        if not items_to_process:
            return 0

        # Process items in parallel if configured
        if self.config.parallel_workers > 1:
            return await self._process_parallel(items_to_process)
        else:
            return await self._process_sequential(items_to_process)

    async def _process_sequential(self, items: list[tuple]) -> int:
        """Process items sequentially.
        
        Args:
            items: List of (item, callback) tuples
            
        Returns:
            Number of items processed
        """
        processed = 0
        for item, callback in items:
            try:
                if callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(item)
                    else:
                        callback(item)
                processed += 1
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                if self.config.retry_on_failure:
                    # Re-queue for retry
                    with self.lock:
                        self.queue.append((item, callback))

        return processed

    async def _process_parallel(self, items: list[tuple]) -> int:
        """Process items in parallel.
        
        Args:
            items: List of (item, callback) tuples
            
        Returns:
            Number of items processed
        """
        # Split items into chunks for parallel processing
        chunk_size = len(items) // self.config.parallel_workers
        if chunk_size == 0:
            chunk_size = 1

        chunks = [
            items[i:i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]

        # Process chunks in parallel
        tasks = [
            self._process_sequential(chunk)
            for chunk in chunks
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful processes
        processed = sum(
            r for r in results
            if isinstance(r, int)
        )

        return processed

    async def start_auto_flush(self) -> None:
        """Start automatic flushing at intervals."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._auto_flush_loop())

    async def stop_auto_flush(self) -> None:
        """Stop automatic flushing."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

    async def _auto_flush_loop(self) -> None:
        """Background task for automatic flushing."""
        while True:
            try:
                await asyncio.sleep(self.config.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-flush: {e}")


class VectorOptimizer:
    """Optimizes vector operations for better performance."""

    @staticmethod
    def optimize_batch_size(
        num_vectors: int,
        vector_dim: int,
        available_memory: int = 1024 * 1024 * 1024  # 1GB default
    ) -> int:
        """Calculate optimal batch size based on available resources.
        
        Args:
            num_vectors: Total number of vectors
            vector_dim: Dimension of each vector
            available_memory: Available memory in bytes
            
        Returns:
            Optimal batch size
        """
        # Estimate memory per vector (float32 = 4 bytes)
        bytes_per_vector = vector_dim * 4

        # Add overhead for metadata and indexing (estimate 50% overhead)
        bytes_per_vector = int(bytes_per_vector * 1.5)

        # Calculate max vectors that fit in memory
        max_batch = available_memory // bytes_per_vector

        # Apply reasonable limits
        min_batch = 10
        max_reasonable = 10000

        optimal = min(max_batch, max_reasonable, num_vectors)
        optimal = max(optimal, min_batch)

        return optimal

    @staticmethod
    def select_index_type(
        num_vectors: int,
        vector_dim: int,
        metric: DistanceMetric
    ) -> dict[str, Any]:
        """Select optimal index type based on dataset characteristics.
        
        Args:
            num_vectors: Number of vectors
            vector_dim: Vector dimensions
            metric: Distance metric
            
        Returns:
            Index configuration
        """
        config = {"metric": metric}

        # Small datasets: use flat index for exact search
        if num_vectors < 10000:
            config["type"] = "flat"
            return config

        # Medium datasets: use IVF
        if num_vectors < 1000000:
            # Calculate optimal number of clusters
            nlist = int(np.sqrt(num_vectors))
            nlist = min(max(nlist, 100), 4096)

            config["type"] = "ivfflat"
            config["nlist"] = nlist
            config["nprobe"] = min(nlist // 10, 64)
            return config

        # Large datasets: use HNSW
        config["type"] = "hnsw"
        config["m"] = 16  # Number of connections
        config["ef_construction"] = 200
        config["ef_search"] = 50

        return config

    @staticmethod
    def optimize_search_params(
        index_type: str,
        recall_target: float = 0.95
    ) -> dict[str, Any]:
        """Optimize search parameters for target recall.
        
        Args:
            index_type: Type of index
            recall_target: Target recall rate (0-1)
            
        Returns:
            Optimized search parameters
        """
        params = {}

        if index_type == "flat":
            # Flat index is always exact
            return params

        elif index_type == "ivfflat":
            # Adjust nprobe based on recall target
            if recall_target >= 0.99:
                params["nprobe"] = 128
            elif recall_target >= 0.95:
                params["nprobe"] = 64
            elif recall_target >= 0.90:
                params["nprobe"] = 32
            else:
                params["nprobe"] = 16

        elif index_type == "hnsw":
            # Adjust ef_search based on recall target
            if recall_target >= 0.99:
                params["ef_search"] = 200
            elif recall_target >= 0.95:
                params["ef_search"] = 100
            elif recall_target >= 0.90:
                params["ef_search"] = 50
            else:
                params["ef_search"] = 32

        return params


class ConnectionPool:
    """Manages a pool of connections for vector stores."""

    def __init__(self,
                 factory: Callable,
                 config: ConnectionPoolConfig | None = None):
        """Initialize the connection pool.
        
        Args:
            factory: Function to create new connections
            config: Pool configuration
        """
        self.factory = factory
        self.config = config or ConnectionPoolConfig()
        self.available: deque = deque()
        self.in_use: set = set()
        self.lock = Lock()
        self._closed = False

    async def acquire(self) -> Any:
        """Acquire a connection from the pool.
        
        Returns:
            A connection object
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        with self.lock:
            # Try to get an available connection
            while self.available:
                conn = self.available.popleft()
                # TODO: Check if connection is still valid
                self.in_use.add(conn)
                return conn

            # Create new connection if under limit
            if len(self.in_use) < self.config.max_connections:
                conn = await self.factory()
                self.in_use.add(conn)
                return conn

        # Wait for a connection to become available
        retry_count = 0
        while retry_count < 100:  # Avoid infinite loop
            await asyncio.sleep(0.1)
            with self.lock:
                if self.available:
                    conn = self.available.popleft()
                    self.in_use.add(conn)
                    return conn
            retry_count += 1

        raise TimeoutError("Could not acquire connection from pool")

    async def release(self, conn: Any) -> None:
        """Release a connection back to the pool.
        
        Args:
            conn: Connection to release
        """
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
                if not self._closed:
                    self.available.append(conn)

    async def close(self) -> None:
        """Close all connections in the pool."""
        self._closed = True

        with self.lock:
            # Close all connections
            all_conns = list(self.available) + list(self.in_use)
            self.available.clear()
            self.in_use.clear()

        # Close connections (if they have close method)
        for conn in all_conns:
            if hasattr(conn, 'close'):
                try:
                    if asyncio.iscoroutinefunction(conn.close):
                        await conn.close()
                    else:
                        conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")


class QueryOptimizer:
    """Optimizes vector queries for better performance."""

    @staticmethod
    def should_use_index(
        num_vectors: int,
        k: int,
        filter_selectivity: float = 1.0
    ) -> bool:
        """Determine if index should be used for query.
        
        Args:
            num_vectors: Total number of vectors
            k: Number of results to return
            filter_selectivity: Estimated filter selectivity (0-1)
            
        Returns:
            True if index should be used
        """
        # If we're retrieving most vectors, scan might be faster
        if k / num_vectors > 0.1:
            return False

        # If filter is very selective, scan filtered results
        if filter_selectivity < 0.01:
            return False

        # Otherwise use index
        return True

    @staticmethod
    def optimize_reranking(
        initial_k: int,
        final_k: int,
        rerank_factor: float = 3.0
    ) -> int:
        """Calculate optimal number of candidates for reranking.
        
        Args:
            initial_k: Initial number of results
            final_k: Final number of results after reranking
            rerank_factor: Multiplier for candidates
            
        Returns:
            Optimal number of candidates
        """
        candidates = int(final_k * rerank_factor)

        # Apply reasonable limits
        min_candidates = final_k * 2
        max_candidates = min(initial_k, final_k * 10)

        candidates = max(candidates, min_candidates)
        candidates = min(candidates, max_candidates)

        return candidates


# Export main classes
__all__ = [
    "BatchConfig",
    "BatchProcessor",
    "ConnectionPool",
    "ConnectionPoolConfig",
    "QueryOptimizer",
    "VectorOptimizer",
]
