"""Streaming support for database operations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

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


class BatchWriteAccountant(Protocol):
    """Sink for the outcomes of a batch-first-with-per-record-fallback write.

    :func:`drive_batch_with_fallback` reports every outcome here; the concrete
    accountant owns BOTH the accounting (which counters/records to bump) AND the
    stop mechanism. A method returning ``False`` stops the drive loop gracefully
    (the streaming path signals a quit that way); raising stops it hard (the
    migration path re-raises to abort immediately). Decoupling the loop from any
    one result/config type lets the same batch-first-fallback decision serve
    both the streaming writer and the batched migrator without duplication.
    """

    def on_batch_start(self) -> None:
        """Called once per drive invocation, before the batch verb is attempted."""

    def on_batch_success(self, batch: list[Record], written_ids: list[str]) -> bool:
        """Account a (possibly partial) successful batch write.

        ``written_ids`` are the ids the batch verb confirmed. A bulk backend
        that partially succeeds (e.g. Elasticsearch per-item errors) confirms
        fewer ids than ``len(batch)``; the shortfall is the accountant's to
        record as failed. Return ``False`` (or raise) to stop.
        """

    def on_record_success(self, record: Record) -> None:
        """Account one record written on the per-record fallback path."""

    def on_record_skip(self, record: Record) -> None:
        """Account one record left untouched under skip-on-duplicate."""

    def on_record_failure(
        self, record: Record | None, error: Exception, local_index: int
    ) -> bool:
        """Account one per-record failure; return ``False`` (or raise) to stop.

        ``local_index`` is the record's position within ``batch`` (0-based), for
        accountants that compute a global record index.
        """


class StreamResultAccountant:
    """:class:`BatchWriteAccountant` that tallies into a :class:`StreamResult`.

    Byte-for-byte reproduces the streaming write path's accounting: the
    partial-batch shortfall folds into :meth:`on_batch_success` (a bulk verb that
    confirms fewer ids than the batch counts the shortfall as failed and routes
    one aggregate ``OperationError`` through ``config.on_error``), and the
    per-record fallback records each id with its global index
    (``batch_index * batch_size + local_index``). Stop = return ``False`` and
    fire the optional quit signal, exactly as the per-record path did.
    """

    def __init__(
        self,
        result: StreamResult,
        config: StreamConfig,
        on_quit_signal: Callable[[], None] | None = None,
        batch_index: int = 0,
    ) -> None:
        self._result = result
        self._config = config
        self._on_quit_signal = on_quit_signal
        self._batch_index = batch_index

    def on_batch_start(self) -> None:
        self._result.total_batches += 1

    def on_batch_success(self, batch: list[Record], written_ids: list[str]) -> bool:
        written = len(written_ids)
        self._result.successful += written
        self._result.total_processed += len(batch)
        shortfall = len(batch) - written
        if shortfall <= 0:
            return True
        # A bulk backend confirmed fewer ids than the batch. The individual
        # failing records are not available here (the verb reports only the
        # confirmed count), so route the shortfall through ``on_error`` once as
        # an aggregate error, on the same stop/continue contract as a per-record
        # failure — keeping ``total_processed == successful + failed + skipped``.
        self._result.failed += shortfall
        error = OperationError(
            f"batch write confirmed {written} of {len(batch)} records; "
            f"{shortfall} failed to write"
        )
        self._result.add_error(None, error)
        if self._config.on_error and self._config.on_error(error, None):
            return True
        if self._on_quit_signal:
            self._on_quit_signal()
        return False

    def on_record_success(self, record: Record) -> None:
        self._result.total_processed += 1
        self._result.successful += 1

    def on_record_skip(self, record: Record) -> None:
        self._result.total_processed += 1
        self._result.skipped += 1

    def on_record_failure(
        self, record: Record | None, error: Exception, local_index: int
    ) -> bool:
        self._result.total_processed += 1
        self._result.failed += 1
        record_id = record.id if record is not None and hasattr(record, "id") else None
        record_index = self._batch_index * self._config.batch_size + local_index
        self._result.add_error(record_id, error, record_index)
        if self._config.on_error:
            if not self._config.on_error(error, record):
                if self._on_quit_signal:
                    self._on_quit_signal()
                return False
            return True
        # No error handler — quit on first error, matching the batch path.
        if self._on_quit_signal:
            self._on_quit_signal()
        return False


def drive_batch_with_fallback(
    batch: list[Record],
    batch_write_func: Callable[[list[Record]], list[str]] | None,
    single_write_func: Callable[[Record], Any],
    accountant: BatchWriteAccountant,
    *,
    skip_on_duplicate: bool = False,
) -> bool:
    """Write a batch, falling back to per-record writes for error attribution.

    Attempts the native batch verb first (when ``batch_write_func`` is not
    ``None``); on any batch-level failure, retries each record individually so a
    failing record is isolated from the successful ones. Every outcome is
    reported to ``accountant``, which owns the accounting and the stop mechanism
    (return ``False`` — or raise — to stop). This is the single shared loop both
    the streaming writer (via :class:`StreamResultAccountant`) and the batched
    migrator drive, so the batch-first-fallback decision exists once.

    Returns:
        True to continue processing, False to stop.
    """
    accountant.on_batch_start()
    if batch_write_func is not None:
        try:
            written_ids = batch_write_func(batch)
        except Exception:
            # Batch failed — fall through to the per-record path below.
            pass
        else:
            return accountant.on_batch_success(batch, written_ids)

    for i, record in enumerate(batch):
        try:
            single_write_func(record)
        except Exception as error:
            if skip_on_duplicate and isinstance(error, DuplicateRecordError):
                accountant.on_record_skip(record)
                continue
            if not accountant.on_record_failure(record, error, i):
                return False
        else:
            accountant.on_record_success(record)
    return True


async def async_drive_batch_with_fallback(
    batch: list[Record],
    batch_write_func: Callable[[list[Record]], Any] | None,  # awaitable -> list[str]
    single_write_func: Callable[[Record], Any],  # awaitable
    accountant: BatchWriteAccountant,
    *,
    skip_on_duplicate: bool = False,
) -> bool:
    """Async counterpart of :func:`drive_batch_with_fallback`.

    The write funcs are awaited; the accountant methods stay synchronous, so a
    single accountant type serves both the sync and async drives.
    """
    accountant.on_batch_start()
    if batch_write_func is not None:
        try:
            written_ids = await batch_write_func(batch)
        except Exception:
            # Batch failed — fall through to the per-record path below.
            pass
        else:
            return accountant.on_batch_success(batch, written_ids)

    for i, record in enumerate(batch):
        try:
            await single_write_func(record)
        except Exception as error:
            if skip_on_duplicate and isinstance(error, DuplicateRecordError):
                accountant.on_record_skip(record)
                continue
            if not accountant.on_record_failure(record, error, i):
                return False
        else:
            accountant.on_record_success(record)
    return True


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
    """Write a batch into a :class:`StreamResult`, with per-record fallback.

    A thin :class:`StreamResult`-typed adapter over
    :func:`drive_batch_with_fallback`: it builds a :class:`StreamResultAccountant`
    for ``(result, config, on_quit_signal, batch_index)`` and drives the shared
    batch-first-with-per-record-fallback loop.

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
    accountant = StreamResultAccountant(result, config, on_quit_signal, batch_index)
    return drive_batch_with_fallback(
        batch,
        batch_create_func,
        single_create_func,
        accountant,
        skip_on_duplicate=skip_on_duplicate,
    )


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
    """Async version of :func:`process_batch_with_fallback`.

    A thin :class:`StreamResult`-typed adapter over
    :func:`async_drive_batch_with_fallback`; the write funcs are awaited while
    the shared :class:`StreamResultAccountant` accounting stays synchronous.

    Returns:
        True to continue processing, False to quit streaming
    """
    accountant = StreamResultAccountant(result, config, on_quit_signal, batch_index)
    return await async_drive_batch_with_fallback(
        batch,
        batch_create_func,
        single_create_func,
        accountant,
        skip_on_duplicate=skip_on_duplicate,
    )


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
