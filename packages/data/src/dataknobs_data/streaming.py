"""Streaming support for database operations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from dataknobs_common.structured_config import StructuredConfig

from .exceptions import DuplicateRecordError, OperationError


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator
    from .query import Query
    from .records import Record


class ConflictPolicy(str, Enum):
    """How a streaming/batched write resolves an id that already exists.

    - ``INSERT`` (default) — fail closed on a colliding id, exactly as a plain
      ``create()`` does. Correct for a virgin target; a re-run into a populated
      target records the colliding ids as failures.
    - ``UPSERT`` — overwrite the target row so it matches the source. Idempotent
      by definition; a colliding id cannot fail.
    - ``SKIP`` — leave the existing row untouched and count the colliding id as
      skipped (not failed). An idempotent top-up that migrates only what is not
      already present.
    """

    INSERT = "insert"
    UPSERT = "upsert"
    SKIP = "skip"


@dataclass(frozen=True)
class StreamConfig(StructuredConfig):
    """Configuration for streaming operations."""

    batch_size: int = 1000
    prefetch: int = 2  # Number of batches to prefetch
    timeout: float | None = None
    on_error: Callable[[Exception, Record], bool] | None = None  # Return True to continue
    on_conflict: ConflictPolicy = ConflictPolicy.INSERT

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.prefetch < 0:
            raise ValueError("prefetch must be non-negative")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive if specified")
        # Coerce a raw string (e.g. from a config dict) to the enum and reject
        # an unknown value loudly, rather than silently defaulting to INSERT.
        # Frozen dataclass: assign through object.__setattr__.
        object.__setattr__(self, "on_conflict", ConflictPolicy(self.on_conflict))


@dataclass
class StreamResult:
    """Result of streaming operation."""

    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0  # Records left untouched under ConflictPolicy.SKIP
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
        self.skipped += other.skipped
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


def _account_batch_shortfall(
    result: StreamResult,
    batch: list[Record],
    written: int,
    config: StreamConfig,
    on_quit_signal: Callable[[], None] | None,
) -> bool:
    """Record a partial-batch failure when the batch verb wrote fewer than all.

    A batch write func returns the ids it actually wrote. Most backends are
    all-or-nothing (they raise on any failure, caught by the caller's per-record
    fallback), but a bulk backend can partially succeed — Elasticsearch's bulk
    API reports per-item errors, so its ``create_batch`` / ``upsert_batch``
    return only the ids that succeeded. Without this accounting the unconfirmed
    records would silently vanish (``successful < total_processed``, ``failed``
    unchanged). Counting the shortfall as failed keeps the invariant
    ``total_processed == successful + failed + skipped`` honest.

    The individual failing records are not available at this layer (the bulk
    verb reports only the count of confirmed ids), so the shortfall is routed
    through ``config.on_error`` once as an aggregate ``OperationError`` with a
    ``None`` record. This keeps the batch path on the same stop/continue
    contract as the per-record fallback below: a configured handler decides
    (return ``False`` to abort), and with no handler the stream quits on the
    first failing batch, exactly as a single per-record failure would. Without
    this, a caller's ``on_error`` veto would be silently unreachable for a bulk
    partial failure.

    Returns:
        True to continue processing, False to quit streaming.
    """
    shortfall = len(batch) - written
    if shortfall <= 0:
        return True
    result.failed += shortfall
    error = OperationError(
        f"batch write confirmed {written} of {len(batch)} records; "
        f"{shortfall} failed to write"
    )
    result.add_error(None, error)
    if config.on_error and config.on_error(error, None):
        # Handler explicitly opted to keep streaming.
        return True
    # No handler (fail-stop default) or handler vetoed — quit the stream.
    if on_quit_signal:
        on_quit_signal()
    return False


def process_batch_with_fallback(
    batch: list[Record],
    batch_create_func: Callable[[list[Record]], list[str]] | None,
    single_create_func: Callable[[Record], str],
    result: StreamResult,
    config: StreamConfig,
    on_quit_signal: Callable[[], None] | None = None,
    batch_index: int = 0,
    *,
    skip_on_duplicate: bool = False,
) -> bool:
    """Process a batch with graceful fallback to individual record writes.

    When a batch operation fails, this function retries each record individually
    to identify which specific records are causing the failure, allowing
    successful records to be processed while only failing the problematic ones.

    Args:
        batch: List of records to process
        batch_create_func: Function to write a batch of records, or ``None`` to
            skip the batch attempt and write every record individually. The
            ``SKIP`` policy always passes ``None`` (a whole-batch verb cannot
            skip individual duplicates while inserting the rest); ``INSERT`` and
            ``UPSERT`` pass their native bulk verb when the backend has one
            (``insert_batch_func`` / ``upsert_batch_func``), else ``None``.
        single_create_func: Function to write a single record
        result: StreamResult to update with statistics
        config: Stream configuration
        on_quit_signal: Optional callback when quitting is signaled
        batch_index: Index of this batch (for computing global record indices)
        skip_on_duplicate: When True, a per-record ``DuplicateRecordError`` is
            counted as a skip (``result.skipped``) and processing continues,
            rather than counting a failure and quitting. Implements
            ``ConflictPolicy.SKIP``.

    Returns:
        True to continue processing, False to quit streaming
    """
    if batch_create_func is not None:
        try:
            # Try batch creation first
            ids = batch_create_func(batch)
            written = len(ids)
            result.successful += written
            result.total_processed += len(batch)
            result.total_batches += 1
            return _account_batch_shortfall(
                result, batch, written, config, on_quit_signal
            )
        except Exception:
            # Batch failed, fall back to per-record writes below.
            pass

    # Per-record path: either the policy has no batch verb, or the batch failed.
    result.total_batches += 1
    for i, record in enumerate(batch):
        result.total_processed += 1
        record_index = batch_index * config.batch_size + i
        try:
            single_create_func(record)
            result.successful += 1
        except Exception as record_error:
            if skip_on_duplicate and isinstance(record_error, DuplicateRecordError):
                # Idempotent top-up: leave the existing row, count a skip.
                result.skipped += 1
                continue
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
    batch_create_func: Callable | None,  # Async callable, or None to skip the batch attempt
    single_create_func: Callable,  # Async callable
    result: StreamResult,
    config: StreamConfig,
    on_quit_signal: Callable[[], None] | None = None,
    batch_index: int = 0,
    *,
    skip_on_duplicate: bool = False,
) -> bool:
    """Async version of process_batch_with_fallback.

    When a batch operation fails, this function retries each record individually
    to identify which specific records are causing the failure, allowing
    successful records to be processed while only failing the problematic ones.

    Args:
        batch: List of records to process
        batch_create_func: Async function to write a batch of records, or
            ``None`` to skip the batch attempt and write every record
            individually. The ``SKIP`` policy always passes ``None``;
            ``INSERT`` and ``UPSERT`` pass their native bulk verb when the
            backend has one, else ``None``.
        single_create_func: Async function to write a single record
        result: StreamResult to update with statistics
        config: Stream configuration
        on_quit_signal: Optional callback when quitting is signaled
        batch_index: Index of this batch (for computing global record indices)
        skip_on_duplicate: When True, a per-record ``DuplicateRecordError`` is
            counted as a skip (``result.skipped``) and processing continues,
            rather than counting a failure and quitting (``ConflictPolicy.SKIP``).

    Returns:
        True to continue processing, False to quit streaming
    """
    if batch_create_func is not None:
        try:
            # Try batch creation first
            ids = await batch_create_func(batch)
            written = len(ids)
            result.successful += written
            result.total_processed += len(batch)
            result.total_batches += 1
            return _account_batch_shortfall(
                result, batch, written, config, on_quit_signal
            )
        except Exception:
            # Batch failed, fall back to per-record writes below.
            pass

    # Per-record path: either the policy has no batch verb, or the batch failed.
    result.total_batches += 1
    for i, record in enumerate(batch):
        result.total_processed += 1
        record_index = batch_index * config.batch_size + i
        try:
            await single_create_func(record)
            result.successful += 1
        except Exception as record_error:
            if skip_on_duplicate and isinstance(record_error, DuplicateRecordError):
                # Idempotent top-up: leave the existing row, count a skip.
                result.skipped += 1
                continue
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


def resolve_conflict_write(
    on_conflict: ConflictPolicy,
    *,
    insert_batch_func: Callable | None,
    single_create_func: Callable,
    upsert_func: Callable,
    upsert_batch_func: Callable | None = None,
) -> tuple[Callable | None, Callable, bool]:
    """Map a write-conflict policy to concrete ``(batch, single, skip)`` funcs.

    Returns the triple ``process_batch_with_fallback`` consumes —
    ``(batch_write_func, single_write_func, skip_on_duplicate)`` — for the given
    policy. This is the single shared definition every ``stream_write`` path uses
    so the policy behaves identically across backends; only the backend-specific
    ``insert_batch_func`` / ``upsert_batch_func`` (the native bulk fast-paths)
    differ per caller.

    - ``INSERT`` — the backend's native batch fast-path plus a ``create``
      fallback; a colliding id fails closed. Byte-identical to prior behavior.
    - ``UPSERT`` — the backend's native ``upsert_batch`` fast-path (when the
      caller supplies one) plus a per-record ``upsert`` fallback. Overwrite is
      idempotent, so a colliding id cannot fail and partial-batch fallback is
      benign. When ``upsert_batch_func`` is ``None`` (the back-compat default)
      every record goes through ``upsert`` one at a time, as before.
    - ``SKIP`` — no batch attempt (a whole-batch verb cannot skip individual
      dupes while inserting the rest); ``create`` per record, and a
      ``DuplicateRecordError`` is counted as a skip rather than a failure.
    """
    if on_conflict == ConflictPolicy.UPSERT:
        return upsert_batch_func, upsert_func, False
    if on_conflict == ConflictPolicy.SKIP:
        return None, single_create_func, True
    return insert_batch_func, single_create_func, False


def run_stream_write(
    records: Iterator[Record],
    *,
    batch_write_func: Callable | None,
    single_write_func: Callable,
    skip_on_duplicate: bool,
    config: StreamConfig,
) -> StreamResult:
    """Drive a sync ``stream_write``: accumulate batches, write each with fallback.

    The shared batch-accumulation loop used by ``StreamingMixin`` and by the
    backends whose ``stream_write`` needs a per-record conflict-aware path. The
    caller resolves the write funcs (usually via :func:`resolve_conflict_write`)
    and hands them in, so the loop itself is backend- and policy-agnostic.
    """
    result = StreamResult()
    start_time = time.time()
    quitting = False
    batch_index = 0

    batch: list[Record] = []
    for record in records:
        batch.append(record)
        if len(batch) >= config.batch_size:
            if not process_batch_with_fallback(
                batch,
                batch_write_func,
                single_write_func,
                result,
                config,
                batch_index=batch_index,
                skip_on_duplicate=skip_on_duplicate,
            ):
                quitting = True
                break
            batch = []
            batch_index += 1

    if batch and not quitting:
        process_batch_with_fallback(
            batch,
            batch_write_func,
            single_write_func,
            result,
            config,
            batch_index=batch_index,
            skip_on_duplicate=skip_on_duplicate,
        )

    result.duration = time.time() - start_time
    return result


async def async_run_stream_write(
    records: AsyncIterator[Record],
    *,
    batch_write_func: Callable | None,
    single_write_func: Callable,
    skip_on_duplicate: bool,
    config: StreamConfig,
) -> StreamResult:
    """Async counterpart of :func:`run_stream_write`."""
    result = StreamResult()
    start_time = time.time()
    quitting = False
    batch_index = 0

    batch: list[Record] = []
    async for record in records:
        batch.append(record)
        if len(batch) >= config.batch_size:
            if not await async_process_batch_with_fallback(
                batch,
                batch_write_func,
                single_write_func,
                result,
                config,
                batch_index=batch_index,
                skip_on_duplicate=skip_on_duplicate,
            ):
                quitting = True
                break
            batch = []
            batch_index += 1

    if batch and not quitting:
        await async_process_batch_with_fallback(
            batch,
            batch_write_func,
            single_write_func,
            result,
            config,
            batch_index=batch_index,
            skip_on_duplicate=skip_on_duplicate,
        )

    result.duration = time.time() - start_time
    return result


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
        """Default implementation of stream_write.

        Honors ``config.on_conflict``: INSERT uses the ``create_batch``
        fast-path with a ``create`` per-record fallback; UPSERT uses the
        ``upsert_batch`` fast-path with an ``upsert`` per-record fallback; SKIP
        writes per-record via ``create`` (see :func:`resolve_conflict_write`).
        """
        config = config or StreamConfig()
        batch_write_func, single_write_func, skip_on_duplicate = resolve_conflict_write(
            config.on_conflict,
            insert_batch_func=self.create_batch,
            single_create_func=self.create,
            upsert_func=self.upsert,
            upsert_batch_func=self.upsert_batch,
        )
        return run_stream_write(
            records,
            batch_write_func=batch_write_func,
            single_write_func=single_write_func,
            skip_on_duplicate=skip_on_duplicate,
            config=config,
        )


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
        """Default implementation of async stream_write.

        Honors ``config.on_conflict``: INSERT uses the ``create_batch``
        fast-path with a ``create`` per-record fallback; UPSERT uses the
        ``upsert_batch`` fast-path with an ``upsert`` per-record fallback; SKIP
        writes per-record via ``create`` (see :func:`resolve_conflict_write`).
        """
        config = config or StreamConfig()
        batch_write_func, single_write_func, skip_on_duplicate = resolve_conflict_write(
            config.on_conflict,
            insert_batch_func=self.create_batch,
            single_create_func=self.create,
            upsert_func=self.upsert,
            upsert_batch_func=self.upsert_batch,
        )
        return await async_run_stream_write(
            records,
            batch_write_func=batch_write_func,
            single_write_func=single_write_func,
            skip_on_duplicate=skip_on_duplicate,
            config=config,
        )
