"""Utility functions for I/O operations.

This module provides utility functions for common I/O patterns.
"""

import asyncio
from typing import (
    Any, Dict, List, Union, AsyncIterator, Iterator,
    Callable, TypeVar, Awaitable
)
from functools import reduce

from .base import IOConfig, IOFormat, IOProvider
from .adapters import (
    FileIOAdapter, DatabaseIOAdapter, HTTPIOAdapter
)

T = TypeVar('T')


def create_io_provider(
    config: IOConfig,
    is_async: bool = True
) -> IOProvider:
    """Create appropriate I/O provider based on configuration.
    
    Args:
        config: I/O configuration
        is_async: Whether to create async provider
        
    Returns:
        Appropriate I/O provider instance
    """
    # Determine adapter based on format and source
    if config.format == IOFormat.DATABASE:
        adapter = DatabaseIOAdapter()
    elif config.format == IOFormat.API or (isinstance(config.source, str) and config.source.startswith(('http://', 'https://'))):
        adapter = HTTPIOAdapter()
    elif isinstance(config.source, dict):
        adapter = DatabaseIOAdapter()
    else:
        adapter = FileIOAdapter()
        
    return adapter.create_provider(config, is_async)


def batch_iterator(
    iterable: Iterator[T],
    batch_size: int
) -> Iterator[List[T]]:
    """Create batches from an iterator.
    
    Args:
        iterable: Source iterator
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


async def async_batch_iterator(
    iterable: AsyncIterator[T],
    batch_size: int
) -> AsyncIterator[List[T]]:
    """Create batches from an async iterator.
    
    Args:
        iterable: Source async iterator
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    batch = []
    async for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def transform_pipeline(
    *transforms: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Create a synchronous transformation pipeline.
    
    Args:
        *transforms: Transformation functions to apply in sequence
        
    Returns:
        Combined transformation function
    """
    def pipeline(data: Any) -> Any:
        return reduce(lambda d, f: f(d), transforms, data)
    return pipeline


def async_transform_pipeline(
    *transforms: Union[Callable[[Any], Any], Callable[[Any], Awaitable[Any]]]
) -> Callable[[Any], Awaitable[Any]]:
    """Create an asynchronous transformation pipeline.
    
    Args:
        *transforms: Transformation functions (sync or async) to apply in sequence
        
    Returns:
        Combined async transformation function
    """
    async def pipeline(data: Any) -> Any:
        result = data
        for transform in transforms:
            if asyncio.iscoroutinefunction(transform):
                result = await transform(result)
            else:
                result = transform(result)
        return result
    return pipeline


class IORouter:
    """Routes data between multiple I/O providers based on conditions."""
    
    def __init__(self):
        self.routes = []
        
    def add_route(
        self,
        condition: Callable[[Any], bool],
        provider: IOProvider,
        transform: Callable[[Any], Any] | None = None
    ):
        """Add a routing rule.
        
        Args:
            condition: Function to determine if route should be used
            provider: I/O provider for this route
            transform: Optional transformation to apply
        """
        self.routes.append({
            'condition': condition,
            'provider': provider,
            'transform': transform or (lambda x: x)
        })
        
    async def route(self, data: Any) -> List[Any]:
        """Route data to appropriate providers.
        
        Args:
            data: Data to route
            
        Returns:
            Results from all matching routes
        """
        results = []
        for route in self.routes:
            if route['condition'](data):
                transformed = route['transform'](data)
                if hasattr(route['provider'], 'write'):
                    if asyncio.iscoroutinefunction(route['provider'].write):
                        await route['provider'].write(transformed)
                    else:
                        route['provider'].write(transformed)
                results.append(transformed)
        return results


class IOBuffer:
    """Buffer for I/O operations with overflow handling."""
    
    def __init__(
        self,
        max_size: int = 10000,
        overflow_handler: Callable[[List[Any]], None] | None = None
    ):
        """Initialize buffer.
        
        Args:
            max_size: Maximum buffer size
            overflow_handler: Function to handle overflow
        """
        self.max_size = max_size
        self.overflow_handler = overflow_handler
        self.buffer = []
        self._lock = asyncio.Lock()
        
    async def add(self, item: Any) -> None:
        """Add item to buffer.
        
        Args:
            item: Item to add
        """
        async with self._lock:
            self.buffer.append(item)
            if len(self.buffer) >= self.max_size:
                await self._handle_overflow()
                
    async def flush(self) -> List[Any]:
        """Flush and return buffer contents.
        
        Returns:
            Buffer contents
        """
        async with self._lock:
            items = self.buffer.copy()
            self.buffer.clear()
            return items
            
    async def _handle_overflow(self) -> None:
        """Handle buffer overflow."""
        if self.overflow_handler:
            overflow_items = self.buffer[:self.max_size // 2]
            self.buffer = self.buffer[self.max_size // 2:]
            if asyncio.iscoroutinefunction(self.overflow_handler):
                await self.overflow_handler(overflow_items)
            else:
                self.overflow_handler(overflow_items)


class IOMetrics:
    """Track metrics for I/O operations."""
    
    def __init__(self):
        self.metrics = {
            'read_count': 0,
            'write_count': 0,
            'bytes_read': 0,
            'bytes_written': 0,
            'errors': 0,
            'retries': 0,
            'duration_ms': 0
        }
        
    def record_read(self, bytes_read: int = 0):
        """Record read operation."""
        self.metrics['read_count'] += 1
        self.metrics['bytes_read'] += bytes_read
        
    def record_write(self, bytes_written: int = 0):
        """Record write operation."""
        self.metrics['write_count'] += 1
        self.metrics['bytes_written'] += bytes_written
        
    def record_error(self):
        """Record error."""
        self.metrics['errors'] += 1
        
    def record_retry(self):
        """Record retry."""
        self.metrics['retries'] += 1
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
        
    def reset(self):
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = 0


async def retry_io_operation(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> T:
    """Retry an I/O operation with exponential backoff.
    
    Args:
        operation: Operation to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Exceptions to catch and retry
        
    Returns:
        Result of successful operation
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                raise
                
    raise last_exception  # type: ignore


def parallel_io_executor(
    providers: List[IOProvider],
    max_workers: int = 4
) -> 'ParallelIOExecutor':
    """Create a parallel I/O executor.
    
    Args:
        providers: List of I/O providers
        max_workers: Maximum concurrent workers
        
    Returns:
        Parallel I/O executor instance
    """
    return ParallelIOExecutor(providers, max_workers)


class ParallelIOExecutor:
    """Execute I/O operations in parallel."""
    
    def __init__(self, providers: List[IOProvider], max_workers: int = 4):
        self.providers = providers
        self.max_workers = max_workers
        
    async def read_all(self, **kwargs) -> List[Any]:
        """Read from all providers in parallel.
        
        Returns:
            Results from all providers
        """
        tasks = []
        for provider in self.providers:
            if hasattr(provider, 'read'):
                if asyncio.iscoroutinefunction(provider.read):
                    tasks.append(provider.read(**kwargs))
                    
        if tasks:
            return await asyncio.gather(*tasks)
        return []
        
    async def write_all(self, data: Any, **kwargs) -> None:
        """Write to all providers in parallel.
        
        Args:
            data: Data to write
        """
        tasks = []
        for provider in self.providers:
            if hasattr(provider, 'write'):
                if asyncio.iscoroutinefunction(provider.write):
                    tasks.append(provider.write(data, **kwargs))
                    
        if tasks:
            await asyncio.gather(*tasks)
