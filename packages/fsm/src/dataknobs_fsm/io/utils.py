"""Utility functions for I/O operations.

This module provides utility functions for common I/O patterns.
"""

import asyncio
import logging
from typing import Any, Dict, List, Union, AsyncIterator, Iterator, Callable, TypeVar, Awaitable
from functools import reduce

from dataknobs_common.callbacks import run_callback, run_callback_off_loop

from .base import IOAdapter, IOConfig, IOFormat, IOProvider
from .adapters import FileIOAdapter, DatabaseIOAdapter, HTTPIOAdapter

logger = logging.getLogger(__name__)

T = TypeVar("T")


def create_io_provider(config: IOConfig, is_async: bool = True) -> IOProvider:
    """Create appropriate I/O provider based on configuration.

    Args:
        config: I/O configuration
        is_async: Whether to create async provider

    Returns:
        Appropriate I/O provider instance
    """
    # Determine adapter based on format and source. Annotated at the base:
    # the branches below assign three different adapters, and inferring the
    # type from the first makes the other two errors.
    adapter: IOAdapter
    if config.format == IOFormat.DATABASE:
        adapter = DatabaseIOAdapter()
    elif config.format == IOFormat.API or (
        isinstance(config.source, str) and config.source.startswith(("http://", "https://"))
    ):
        adapter = HTTPIOAdapter()
    elif isinstance(config.source, dict):
        adapter = DatabaseIOAdapter()
    else:
        adapter = FileIOAdapter()

    return adapter.create_provider(config, is_async)


def batch_iterator(iterable: Iterator[T], batch_size: int) -> Iterator[List[T]]:
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
    iterable: AsyncIterator[T], batch_size: int
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


def transform_pipeline(*transforms: Callable[[Any], Any]) -> Callable[[Any], Any]:
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
    *transforms: Union[Callable[[Any], Any], Callable[[Any], Awaitable[Any]]],
) -> Callable[[Any], Awaitable[Any]]:
    """Create an asynchronous transformation pipeline.

    A synchronous transform runs **on the event loop**, unlike the provider
    writes and the overflow flush elsewhere in this module. A transform is
    named for computing rather than for doing, and it runs once per item,
    so it would pay for a thread hop on every record; the surface decides,
    and this surface says inline. A transform that really does block belongs
    behind :func:`asyncio.to_thread` in the caller's own transform.

    Args:
        *transforms: Transformation functions (sync or async) to apply in sequence

    Returns:
        Combined async transformation function
    """

    async def pipeline(data: Any) -> Any:
        result = data
        for transform in transforms:
            result = await run_callback(transform, result)
        return result

    return pipeline


class IORouter:
    """Routes data between multiple I/O providers based on conditions."""

    def __init__(self) -> None:
        self.routes: List[Dict[str, Any]] = []

    def add_route(
        self,
        condition: Callable[[Any], bool] | Callable[[Any], Awaitable[bool]],
        provider: IOProvider,
        transform: Callable[[Any], Any] | Callable[[Any], Awaitable[Any]] | None = None,
    ) -> None:
        """Add a routing rule.

        Args:
            condition: Decides whether this route takes the data. May be sync
                or async; an async one is awaited before its answer is read.
            provider: I/O provider for this route
            transform: Optional transformation to apply. May be sync or async.
        """
        self.routes.append(
            {"condition": condition, "provider": provider, "transform": transform or (lambda x: x)}
        )

    async def route(self, data: Any) -> List[Any]:
        """Route data to appropriate providers.

        The condition and the transform run inline and the provider's write
        is offloaded, which is the same rule the rest of this module follows:
        a predicate and a per-record transform compute, and a write is I/O
        whether the provider spells it ``async`` or not.

        Args:
            data: Data to route

        Returns:
            Results from all matching routes
        """
        results = []
        for route in self.routes:
            # `run_callback`, not a bare call: both of these are supplied by
            # the caller and may be async, and a coroutine is truthy -- so an
            # unjudged condition matches every record, and an unjudged
            # transform is written to the provider in place of the data.
            if await run_callback(route["condition"], data):
                transformed = await run_callback(route["transform"], data)
                if hasattr(route["provider"], "write"):
                    await run_callback_off_loop(route["provider"].write, transformed)
                else:
                    # The transformed value still joins `results`, so a caller
                    # reading the return value cannot tell that nothing was
                    # written. Say so rather than dropping it in silence --
                    # a provider registered on a route and unable to accept a
                    # write is a wiring mistake, not a routing decision.
                    logger.warning(
                        "Route provider %r has no write(); data matched the route "
                        "and was transformed but not written",
                        type(route["provider"]).__name__,
                    )
                results.append(transformed)
        return results


class IOBuffer:
    """Buffer for I/O operations with overflow handling."""

    def __init__(
        self,
        max_size: int = 10000,
        overflow_handler: Callable[[List[Any]], None]
        | Callable[[List[Any]], Awaitable[None]]
        | None = None,
    ):
        """Initialize buffer.

        Args:
            max_size: Maximum buffer size. Values below 2 still drain one item
                at a time rather than never draining.
            overflow_handler: Function to handle overflow. Sync or async ---
                a synchronous one is run off the event loop, since a flush
                writes somewhere. It is called with the items already removed
                from the buffer; if it raises, they are put back, so the
                overflow is never lost between the two.
        """
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")
        self.max_size = max_size
        self.overflow_handler = overflow_handler
        self.buffer: List[Any] = []
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
        """Handle buffer overflow.

        The handler is the only thing standing between the overflow and
        losing it: the items are off the buffer before it is called, and no
        copy is kept. So it is dispatched by
        :func:`~dataknobs_common.callbacks.run_callback_off_loop`, which is
        right on both counts here -- it judges a callable object correctly,
        where ``iscoroutinefunction`` reports one as sync and silently
        discards the coroutine, and it keeps a handler that writes the
        overflow somewhere off the event loop.

        Two consequences of that same no-copy design, which the reasoning
        above describes and the code did not honour:

        **A failed flush must not consume the overflow.** A handler writing to
        a full disk or a refused socket raises, and the items it was given are
        already gone from the buffer. They go back, at the front, so the
        exception reaches the caller with the data still held rather than
        merely reported.

        **The drain has to move at least one item.** ``max_size // 2`` is zero
        at ``max_size=1``, which called the handler with an empty list on
        every subsequent ``add`` while the buffer kept everything ---
        unbounded growth in the component whose contract is to bound it.
        """
        if not self.overflow_handler:
            return

        drain = max(1, self.max_size // 2)
        overflow_items = self.buffer[:drain]
        self.buffer = self.buffer[drain:]
        try:
            await run_callback_off_loop(self.overflow_handler, overflow_items)
        except BaseException:
            # Back to the front, so a retry drains the oldest first and the
            # buffer's ordering survives a failed flush.
            self.buffer = overflow_items + self.buffer
            raise


class IOMetrics:
    """Track metrics for I/O operations."""

    def __init__(self) -> None:
        self.metrics = {
            "read_count": 0,
            "write_count": 0,
            "bytes_read": 0,
            "bytes_written": 0,
            "errors": 0,
            "retries": 0,
            "duration_ms": 0,
        }

    def record_read(self, bytes_read: int = 0) -> None:
        """Record read operation."""
        self.metrics["read_count"] += 1
        self.metrics["bytes_read"] += bytes_read

    def record_write(self, bytes_written: int = 0) -> None:
        """Record write operation."""
        self.metrics["write_count"] += 1
        self.metrics["bytes_written"] += bytes_written

    def record_error(self) -> None:
        """Record error."""
        self.metrics["errors"] += 1

    def record_retry(self) -> None:
        """Record retry."""
        self.metrics["retries"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()

    def reset(self) -> None:
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = 0


async def retry_io_operation(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
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


def parallel_io_executor(providers: List[IOProvider], max_workers: int = 4) -> "ParallelIOExecutor":
    """Create a parallel I/O executor.

    Args:
        providers: List of I/O providers
        max_workers: Maximum concurrent workers

    Returns:
        Parallel I/O executor instance
    """
    return ParallelIOExecutor(providers, max_workers)


class ParallelIOExecutor:
    """Execute I/O operations in parallel, up to ``max_workers`` at a time."""

    def __init__(self, providers: List[IOProvider], max_workers: int = 4) -> None:
        """Initialize the executor.

        Args:
            providers: I/O providers to drive. Sync and async alike.
            max_workers: Maximum provider calls in flight at once.

        Raises:
            ValueError: If ``max_workers`` is below 1, which would deadlock.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        self.providers = providers
        self.max_workers = max_workers
        # One semaphore for the executor's lifetime rather than one per call,
        # so two concurrent `read_all`s share the bound instead of getting one
        # each. Safe to build outside a running loop on the supported Pythons:
        # a Semaphore has not bound itself to a loop at construction since 3.10.
        self._semaphore = asyncio.Semaphore(max_workers)

    async def _bounded(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Dispatch one provider call, holding a slot for its duration.

        Wraps :func:`~dataknobs_common.callbacks.run_callback_off_loop` rather
        than replacing it: the judgement about the callable is still made
        there, and this adds only the bound.

        Without the bound, every synchronous provider becomes a
        ``to_thread`` submission issued at once into the loop's *default*
        executor --- which is process-wide and shared with every other
        offload in the application, this package's own included. Fanning out
        unboundedly there is not merely ignoring a knob; it is spending
        somebody else's thread pool.
        """
        async with self._semaphore:
            return await run_callback_off_loop(callback, *args, **kwargs)

    async def read_all(self, **kwargs: Any) -> List[Any]:
        """Read from all providers in parallel.

        Both kinds of provider participate. ``SyncIOProvider`` is as much an
        :class:`~dataknobs_fsm.io.base.IOProvider` as ``AsyncIOProvider`` is,
        and this class is annotated to take either; a synchronous read is
        offloaded to a worker thread rather than run on the event loop, so
        the providers still proceed concurrently and a slow disk read does
        not stall the others --- up to ``max_workers`` of them at a time.

        Returns:
            Results from all providers
        """
        tasks = [
            self._bounded(provider.read, **kwargs)
            for provider in self.providers
            if hasattr(provider, "read")
        ]

        if tasks:
            return await asyncio.gather(*tasks)
        return []

    async def write_all(self, data: Any, **kwargs: Any) -> None:
        """Write to all providers in parallel.

        As with :meth:`read_all`, a synchronous provider is written to on a
        worker thread rather than skipped. This method returns ``None`` either
        way, so a provider silently receiving nothing was undetectable from
        the outside.

        Args:
            data: Data to write
        """
        tasks = [
            self._bounded(provider.write, data, **kwargs)
            for provider in self.providers
            if hasattr(provider, "write")
        ]

        if tasks:
            await asyncio.gather(*tasks)
