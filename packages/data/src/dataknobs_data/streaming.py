"""Streaming support for database operations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from .query import Query
    from .records import Record


@dataclass
class StreamConfig:
    """Configuration for streaming operations."""

    batch_size: int = 1000
    prefetch: int = 2  # Number of batches to prefetch
    timeout: float | None = None
    on_error: Callable[[Exception, Record], bool] | None = None  # Return True to continue

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.prefetch < 0:
            raise ValueError("prefetch must be non-negative")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive if specified")


@dataclass
class StreamResult:
    """Result of streaming operation."""

    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0
    total_batches: int = 0  # Number of batches processed
    failed_indices: list[int] = field(default_factory=list)  # Indices of failed records

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.successful / self.total_processed) * 100

    def add_error(self, record_id: str | None, error: Exception, index: int | None = None) -> None:
        """Add an error to the result.
        
        Args:
            record_id: ID of the record that failed
            error: The exception that occurred
            index: Optional index of the failed record in the original batch
        """
        self.errors.append({
            "record_id": record_id,
            "error": str(error),
            "type": type(error).__name__,
            "index": index
        })
        if index is not None:
            self.failed_indices.append(index)

    def merge(self, other: StreamResult) -> None:
        """Merge another result into this one."""
        self.total_processed += other.total_processed
        self.successful += other.successful
        self.failed += other.failed
        self.errors.extend(other.errors)
        self.duration += other.duration
        self.total_batches += other.total_batches
        self.failed_indices.extend(other.failed_indices)

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"StreamResult(processed={self.total_processed}, "
            f"successful={self.successful}, failed={self.failed}, "
            f"success_rate={self.success_rate:.1f}%, "
            f"duration={self.duration:.2f}s)"
        )


def process_batch_with_fallback(
    batch: list[Record],
    batch_create_func: Callable[[list[Record]], list[str]],
    single_create_func: Callable[[Record], str],
    result: StreamResult,
    config: StreamConfig,
    on_quit_signal: Callable[[], None] | None = None,
    batch_index: int = 0
) -> bool:
    """Process a batch with graceful fallback to individual record creation.
    
    When a batch operation fails, this function will retry each record individually
    to identify which specific records are causing the failure, allowing successful
    records to be processed while only failing the problematic ones.
    
    Args:
        batch: List of records to process
        batch_create_func: Function to create a batch of records
        single_create_func: Function to create a single record
        result: StreamResult to update with statistics
        config: Stream configuration
        on_quit_signal: Optional callback when quitting is signaled
        
    Returns:
        True to continue processing, False to quit streaming
    """
    try:
        # Try batch creation first
        ids = batch_create_func(batch)
        result.successful += len(ids)
        result.total_processed += len(batch)
        result.total_batches += 1
        return True
    except Exception:
        # Batch failed, try individual records to identify failures
        result.total_batches += 1
        for i, record in enumerate(batch):
            result.total_processed += 1
            record_index = batch_index * config.batch_size + i
            try:
                single_create_func(record)
                result.successful += 1
            except Exception as record_error:
                # This specific record failed
                result.failed += 1
                # Safely get record ID if available
                record_id = record.id if record and hasattr(record, 'id') else None
                result.add_error(record_id, record_error, record_index)

                if config.on_error:
                    # Call error handler
                    if not config.on_error(record_error, record):
                        # Handler returned False, quit streaming
                        if on_quit_signal:
                            on_quit_signal()
                        return False
                else:
                    # No error handler, quit on first error
                    if on_quit_signal:
                        on_quit_signal()
                    return False

    return True


async def async_process_batch_with_fallback(
    batch: list[Record],
    batch_create_func: Callable,  # Async callable
    single_create_func: Callable,  # Async callable
    result: StreamResult,
    config: StreamConfig,
    on_quit_signal: Callable[[], None] | None = None,
    batch_index: int = 0
) -> bool:
    """Async version of process_batch_with_fallback.
    
    When a batch operation fails, this function will retry each record individually
    to identify which specific records are causing the failure, allowing successful
    records to be processed while only failing the problematic ones.
    
    Args:
        batch: List of records to process
        batch_create_func: Async function to create a batch of records
        single_create_func: Async function to create a single record
        result: StreamResult to update with statistics
        config: Stream configuration
        on_quit_signal: Optional callback when quitting is signaled
        
    Returns:
        True to continue processing, False to quit streaming
    """
    try:
        # Try batch creation first
        ids = await batch_create_func(batch)
        result.successful += len(ids)
        result.total_processed += len(batch)
        result.total_batches += 1
        return True
    except Exception:
        # Batch failed, try individual records to identify failures
        result.total_batches += 1
        for i, record in enumerate(batch):
            result.total_processed += 1
            record_index = batch_index * config.batch_size + i
            try:
                await single_create_func(record)
                result.successful += 1
            except Exception as record_error:
                # This specific record failed
                result.failed += 1
                # Safely get record ID if available
                record_id = record.id if record and hasattr(record, 'id') else None
                result.add_error(record_id, record_error, record_index)

                if config.on_error:
                    # Call error handler
                    if not config.on_error(record_error, record):
                        # Handler returned False, quit streaming
                        if on_quit_signal:
                            on_quit_signal()
                        return False
                else:
                    # No error handler, quit on first error
                    if on_quit_signal:
                        on_quit_signal()
                    return False

    return True


class StreamProcessor:
    """Base class for stream processing utilities."""

    @staticmethod
    def batch_iterator(
        iterator: Iterator[Record],
        batch_size: int
    ) -> Iterator[list[Record]]:
        """Convert a record iterator into batches."""
        batch = []
        for record in iterator:
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def list_to_iterator(records: list[Record]) -> Iterator[Record]:
        """Convert a list of records to an iterator.
        
        Args:
            records: List of records
            
        Yields:
            Individual records from the list
        """
        for record in records:
            yield record

    @staticmethod
    async def list_to_async_iterator(records: list[Record]) -> AsyncIterator[Record]:
        """Convert a list of records to an async iterator.
        
        This adapter allows synchronous lists to be used with async streaming APIs.
        
        Args:
            records: List of records
            
        Yields:
            Individual records from the list
        """
        for record in records:
            yield record

    @staticmethod
    async def iterator_to_async_iterator(iterator: Iterator[Record]) -> AsyncIterator[Record]:
        """Convert a synchronous iterator to an async iterator.
        
        This adapter allows synchronous iterators to be used with async streaming APIs.
        
        Args:
            iterator: Synchronous iterator of records
            
        Yields:
            Individual records from the iterator
        """
        for record in iterator:
            yield record

    @staticmethod
    async def async_batch_iterator(
        iterator: AsyncIterator[Record],
        batch_size: int
    ) -> AsyncIterator[list[Record]]:
        """Convert an async record iterator into batches."""
        batch = []
        async for record in iterator:
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def filter_stream(
        iterator: Iterator[Record],
        predicate: Callable[[Record], bool]
    ) -> Iterator[Record]:
        """Filter records in a stream."""
        for record in iterator:
            if predicate(record):
                yield record

    @staticmethod
    async def async_filter_stream(
        iterator: AsyncIterator[Record],
        predicate: Callable[[Record], bool]
    ) -> AsyncIterator[Record]:
        """Filter records in an async stream."""
        async for record in iterator:
            if predicate(record):
                yield record

    @staticmethod
    def transform_stream(
        iterator: Iterator[Record],
        transform: Callable[[Record], Record | None]
    ) -> Iterator[Record]:
        """Transform records in a stream, filtering out None results."""
        for record in iterator:
            result = transform(record)
            if result is not None:
                yield result

    @staticmethod
    async def async_transform_stream(
        iterator: AsyncIterator[Record],
        transform: Callable[[Record], Record | None]
    ) -> AsyncIterator[Record]:
        """Transform records in an async stream, filtering out None results."""
        async for record in iterator:
            result = transform(record)
            if result is not None:
                yield result


class StreamingMixin:
    """Mixin class providing common streaming functionality for sync databases."""

    def _default_stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> Iterator[Record]:
        """Default implementation of stream_read using search method.
        
        This provides a simple streaming wrapper around the search method
        that most backends can use without modification.
        """
        config = config or StreamConfig()

        # Use search to get all matching records
        if query:
            records = self.search(query)
        else:
            # If no query, get all records
            from .query import Query
            records = self.search(Query())

        # Yield records in batches for consistency
        for i in range(0, len(records), config.batch_size):
            batch = records[i:i + config.batch_size]
            for record in batch:
                yield record

    def _default_stream_write(
        self,
        records: Iterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Default implementation of stream_write using create_batch method.
        
        This provides batch writing functionality with graceful fallback
        to individual record creation when batches fail.
        """
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        quitting = False
        batch_index = 0

        batch = []
        for record in records:
            batch.append(record)

            if len(batch) >= config.batch_size:
                # Write batch with graceful fallback
                continue_processing = process_batch_with_fallback(
                    batch,
                    self.create_batch,
                    self.create,
                    result,
                    config,
                    batch_index=batch_index
                )

                if not continue_processing:
                    quitting = True
                    break

                batch = []
                batch_index += 1

        # Write remaining batch
        if batch and not quitting:
            process_batch_with_fallback(
                batch,
                self.create_batch,
                self.create,
                result,
                config,
                batch_index=batch_index
            )

        result.duration = time.time() - start_time
        return result


class AsyncStreamingMixin:
    """Mixin class providing common streaming functionality for async databases."""

    async def _default_stream_read(
        self,
        query: Query | None = None,
        config: StreamConfig | None = None
    ) -> AsyncIterator[Record]:
        """Default implementation of async stream_read using search method.
        
        This provides a simple streaming wrapper around the search method
        that most backends can use without modification.
        """
        config = config or StreamConfig()

        # Use search to get all matching records
        if query:
            records = await self.search(query)
        else:
            # If no query, get all records
            from .query import Query
            records = await self.search(Query())

        # Yield records in batches for consistency
        for i in range(0, len(records), config.batch_size):
            batch = records[i:i + config.batch_size]
            for record in batch:
                yield record

    async def _default_stream_write(
        self,
        records: AsyncIterator[Record],
        config: StreamConfig | None = None
    ) -> StreamResult:
        """Default implementation of async stream_write using create_batch method.
        
        This provides batch writing functionality with graceful fallback
        to individual record creation when batches fail.
        """
        config = config or StreamConfig()
        result = StreamResult()
        start_time = time.time()
        quitting = False
        batch_index = 0

        batch = []
        async for record in records:
            batch.append(record)

            if len(batch) >= config.batch_size:
                # Write batch with graceful fallback
                continue_processing = await async_process_batch_with_fallback(
                    batch,
                    self.create_batch,
                    self.create,
                    result,
                    config,
                    batch_index=batch_index
                )

                if not continue_processing:
                    quitting = True
                    break

                batch = []
                batch_index += 1

        # Write remaining batch
        if batch and not quitting:
            await async_process_batch_with_fallback(
                batch,
                self.create_batch,
                self.create,
                result,
                config,
                batch_index=batch_index
            )

        result.duration = time.time() - start_time
        return result
