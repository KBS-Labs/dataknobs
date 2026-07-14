"""Enhanced data migrator with streaming support.
"""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

from dataknobs_data.query import Query
from dataknobs_data.streaming import (
    ConflictPolicy,
    StreamConfig,
    build_shortfall_error,
    drive_batch_with_fallback,
    resolve_conflict_write,
)

from .migration import Migration
from .progress import MigrationProgress
from .transformer import Transformer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from dataknobs_data.database import AsyncDatabase, SyncDatabase
    from dataknobs_data.records import Record


class MigrationProgressAccountant:
    """:class:`BatchWriteAccountant` tallying into a :class:`MigrationProgress`.

    Drives the migrator's batched write path through the same
    batch-first-with-per-record-fallback loop the streaming writer uses
    (:func:`dataknobs_data.streaming.drive_batch_with_fallback`). Stop is hard: a
    per-record failure with no ``on_error`` handler — or a handler that vetoes —
    re-raises immediately, so the batch's remaining records are not written,
    preserving the migrator's fail-fast contract. A handler returning ``True``
    records the failure and continues.
    """

    def __init__(
        self,
        progress: MigrationProgress,
        on_error: Callable[[Exception, Record | None], bool] | None = None,
    ) -> None:
        self._progress = progress
        self._on_error = on_error

    def on_batch_start(self) -> None:
        # MigrationProgress tracks no per-batch counter.
        return None

    def on_batch_success(self, batch: list[Record], written_ids: list[str]) -> bool:
        for record_id in written_ids:
            self._progress.record_success(record_id)
        shortfall = len(batch) - len(written_ids)
        if shortfall <= 0:
            return True
        # A bulk backend confirmed fewer ids than the batch; the individual
        # failing records are not available here, so record the shortfall as one
        # aggregate failure (count-correct, single error entry) and offer the
        # same aggregate error to ``on_error`` on the same stop contract as a
        # per-record failure. ``on_error`` receives ``None`` for the record here
        # because no single record maps to the aggregate shortfall.
        error = build_shortfall_error(len(written_ids), len(batch))
        self._progress.record_batch_failure(shortfall, str(error), error)
        if self._on_error and self._on_error(error, None):
            return True
        raise error

    def on_record_success(self, record: Record) -> None:
        self._progress.record_success(record.id)

    def on_record_skip(self, record: Record) -> None:
        self._progress.record_skip("Already present in target", record.id)

    def on_record_failure(
        self, record: Record | None, error: Exception, local_index: int
    ) -> bool:
        record_id = record.id if record is not None else None
        self._progress.record_failure(str(error), record_id, error)
        if self._on_error and self._on_error(error, record):
            return True
        raise error


class Migrator:
    """Data migration orchestrator with streaming support.
    
    Provides memory-efficient migration between databases using streaming,
    with support for transformations, progress tracking, and parallel processing.
    """

    def migrate(
        self,
        source: SyncDatabase,
        target: SyncDatabase,
        transform: Transformer | Migration | None = None,
        query: Query | None = None,
        batch_size: int = 1000,
        on_progress: Callable[[MigrationProgress], None] | None = None,
        on_error: Callable[[Exception, Record | None], bool] | None = None,
        on_conflict: ConflictPolicy | str = ConflictPolicy.INSERT,
    ) -> MigrationProgress:
        """Migrate data between databases with optional transformation.

        Args:
            source: Source database
            target: Target database
            transform: Optional transformer or migration to apply
            query: Optional query to filter source records
            batch_size: Number of records to process per batch
            on_progress: Optional callback for progress updates
            on_error: Optional error handler ``(exc, record) -> bool`` (return
                True to continue, False/omit to abort). ``record`` is the failing
                record, or ``None`` for an aggregate batch shortfall (a bulk verb
                that confirmed fewer ids than it was given, where no single
                record maps to the error).
            on_conflict: How to resolve an id that already exists in the target
                (``"insert"`` fails closed on a collision — the default;
                ``"upsert"`` overwrites the target row; ``"skip"`` leaves the
                existing row and counts the id as skipped). Accepts a
                ``ConflictPolicy`` or its string value.

        Returns:
            MigrationProgress with final statistics
        """
        on_conflict = ConflictPolicy(on_conflict)
        progress = MigrationProgress().start()

        # Get total count for progress tracking
        all_records = source.search(query or Query())
        progress.total = len(all_records)

        batch = []
        for original_record in all_records:
            try:
                # Apply transformation if provided
                record = original_record
                if transform is not None:
                    if isinstance(transform, Transformer):
                        original_id = record.id  # Preserve ID before transformation
                        transformed = transform.transform(record)
                        if transformed is None:
                            # Record filtered out
                            progress.record_skip("Filtered by transformer", original_id)
                            continue
                        record = transformed
                    elif isinstance(transform, Migration):
                        record = transform.apply(record)
            except Exception as e:
                # Transform-stage failure only. Write failures are accounted
                # inside _write_batch (below, outside this block) via the
                # accountant, so keeping the flush out of this try prevents a
                # write failure from being recorded — and offered to on_error —
                # a second time here.
                record_id = original_record.id if hasattr(original_record, "id") else None
                progress.record_failure(str(e), record_id, e)
                if on_error and on_error(e, original_record):
                    continue  # Handler says continue - skip this record
                raise  # Handler says stop, or no handler - abort immediately

            batch.append(record)

            # Process batch when full. _write_batch accounts every outcome
            # through its accountant and re-raises to abort on an unhandled
            # failure; that exception propagates directly (not through the
            # transform handler above), so a write failure is neither recorded
            # nor offered to on_error twice.
            if len(batch) >= batch_size:
                self._write_batch(target, batch, progress, on_error, on_conflict)
                batch = []
                if on_progress:
                    on_progress(progress)

        # Process final batch
        if batch:
            self._write_batch(target, batch, progress, on_error, on_conflict)

        progress.finish()

        if on_progress:
            on_progress(progress)

        return progress

    def migrate_stream(
        self,
        source: SyncDatabase,
        target: SyncDatabase,
        transform: Transformer | Migration | None = None,
        query: Query | None = None,
        config: StreamConfig | None = None,
        on_progress: Callable[[MigrationProgress], None] | None = None
    ) -> MigrationProgress:
        """Stream-based migration for memory efficiency.
        
        Never loads full dataset into memory.
        
        Args:
            source: Source database with streaming support
            target: Target database with streaming support
            transform: Optional transformer or migration to apply
            query: Optional query to filter source records
            config: Streaming configuration
            on_progress: Optional callback for progress updates
            
        Returns:
            MigrationProgress with final statistics
        """
        config = config or StreamConfig()
        progress = MigrationProgress().start()

        # Estimate total (if possible)
        try:
            progress.total = source.count(query)
        except Exception:
            # Count not available, will track as we go
            pass

        # Create streaming pipeline
        def transform_stream(records: Iterator[Record]) -> Iterator[Record]:
            """Apply transformation to streaming records."""
            for record in records:
                # Count `processed` exactly once per record, at the point its
                # outcome is decided: a yield below (pass-through), record_skip
                # (filtered), or record_failure (transform error). record_skip
                # and record_failure both increment `processed`, so a
                # pre-increment here double-counted filtered / errored records.
                try:
                    if transform is not None:
                        if isinstance(transform, Transformer):
                            original_id = record.id  # Preserve ID before transformation
                            transformed = transform.transform(record)
                            if transformed:
                                progress.processed += 1
                                yield transformed
                            else:
                                progress.record_skip("Filtered by transformer", original_id)
                        elif isinstance(transform, Migration):
                            applied = transform.apply(record)
                            progress.processed += 1
                            yield applied
                    else:
                        progress.processed += 1
                        yield record
                except Exception as e:
                    if config.on_error and config.on_error(e, record):
                        progress.record_failure(str(e), record.id if hasattr(record, 'id') else None, e)
                        continue
                    else:
                        progress.record_failure(str(e), record.id if hasattr(record, 'id') else None, e)
                        raise

        # Stream from source through transformation to target
        source_stream = source.stream_read(query, config)
        transformed_stream = transform_stream(source_stream)

        # Write stream to target
        result = target.stream_write(transformed_stream, config)

        # Update progress from result
        # Note: processed was already tracked in transform_stream
        # Result contains only write successes/failures
        progress.succeeded += result.successful
        progress.failed += result.failed
        progress.skipped += result.skipped
        progress.errors.extend(result.errors)

        progress.finish()

        if on_progress:
            on_progress(progress)

        return progress

    def migrate_parallel(
        self,
        source: SyncDatabase,
        target: SyncDatabase,
        transform: Transformer | Migration | None = None,
        partitions: int = 4,
        partition_field: str = "partition_id",
        on_progress: Callable[[MigrationProgress], None] | None = None,
        on_conflict: ConflictPolicy | str = ConflictPolicy.INSERT,
    ) -> MigrationProgress:
        """Parallel streaming migration.

        Partition data and migrate in parallel streams.

        Args:
            source: Source database
            target: Target database
            transform: Optional transformer or migration
            partitions: Number of parallel partitions
            partition_field: Field to use for partitioning
            on_progress: Optional callback for progress updates
            on_conflict: Conflict policy for an id already present in the target
                (insert = fail closed, upsert = overwrite, skip = leave and
                count as skipped). Applied to every partition stream.

        Returns:
            Combined MigrationProgress
        """
        on_conflict = ConflictPolicy(on_conflict)

        def migrate_partition(partition_id: int) -> MigrationProgress:
            """Migrate a single partition."""
            query = Query().filter(partition_field, "=", partition_id)
            return self.migrate_stream(
                source, target, transform, query,
                config=StreamConfig(on_conflict=on_conflict),
            )

        total_progress = MigrationProgress().start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=partitions) as executor:
            futures = [
                executor.submit(migrate_partition, i)
                for i in range(partitions)
            ]

            for future in concurrent.futures.as_completed(futures):
                partition_progress = future.result()
                total_progress.merge(partition_progress)

                if on_progress:
                    on_progress(total_progress)

        total_progress.finish()
        return total_progress

    async def migrate_async(
        self,
        source: AsyncDatabase,
        target: AsyncDatabase,
        transform: Transformer | Migration | None = None,
        query: Query | None = None,
        config: StreamConfig | None = None,
        on_progress: Callable[[MigrationProgress], None] | None = None
    ) -> MigrationProgress:
        """Async stream-based migration.
        
        Args:
            source: Async source database
            target: Async target database
            transform: Optional transformer or migration
            query: Optional query to filter source records
            config: Streaming configuration
            on_progress: Optional callback for progress updates
            
        Returns:
            MigrationProgress with final statistics
        """
        config = config or StreamConfig()
        progress = MigrationProgress().start()

        # Estimate total (if possible)
        try:
            progress.total = await source.count(query)
        except Exception:
            pass

        # Create async streaming pipeline
        async def transform_stream(records):
            """Apply transformation to async streaming records."""
            async for record in records:
                # Count `processed` exactly once per record, at the point its
                # outcome is decided: a yield below (pass-through), record_skip
                # (filtered), or record_failure (transform error). record_skip
                # and record_failure both increment `processed`, so a
                # pre-increment here double-counted filtered / errored records.
                try:
                    if transform is not None:
                        if isinstance(transform, Transformer):
                            original_id = record.id  # Preserve ID before transformation
                            transformed = transform.transform(record)
                            if transformed:
                                progress.processed += 1
                                yield transformed
                            else:
                                progress.record_skip("Filtered by transformer", original_id)
                        elif isinstance(transform, Migration):
                            applied = transform.apply(record)
                            progress.processed += 1
                            yield applied
                    else:
                        progress.processed += 1
                        yield record
                except Exception as e:
                    if config.on_error and config.on_error(e, record):
                        progress.record_failure(str(e), record.id if hasattr(record, 'id') else None, e)
                        continue
                    else:
                        progress.record_failure(str(e), record.id if hasattr(record, 'id') else None, e)
                        raise

        # Stream from source through transformation to target
        source_stream = source.stream_read(query, config)
        transformed_stream = transform_stream(source_stream)

        # Write stream to target
        result = await target.stream_write(transformed_stream, config)

        # Update progress from result
        # Note: processed was already tracked in transform_stream
        # Result contains only write successes/failures
        progress.succeeded += result.successful
        progress.failed += result.failed
        progress.skipped += result.skipped
        progress.errors.extend(result.errors)

        progress.finish()

        if on_progress:
            on_progress(progress)

        return progress

    def _write_batch(
        self,
        target: SyncDatabase,
        batch: list[Record],
        progress: MigrationProgress,
        on_error: Callable[[Exception, Record | None], bool] | None = None,
        on_conflict: ConflictPolicy = ConflictPolicy.INSERT,
    ) -> None:
        """Write a batch of records to target database.

        The write rides the target's native bulk verbs — ``create_batch`` for
        INSERT (fail-closed on a colliding id) and ``upsert_batch`` for UPSERT —
        with a graceful per-record fallback that preserves per-id progress
        accounting and ``on_error`` semantics when the batch verb fails. SKIP
        stays per-record: a whole-batch verb cannot skip one duplicate while
        inserting the rest, so it writes each record with ``create`` and counts a
        colliding id as skipped. This is the same conflict-policy resolution and
        batch-first-fallback loop the streaming write path uses
        (:func:`dataknobs_data.streaming.resolve_conflict_write` /
        :func:`dataknobs_data.streaming.drive_batch_with_fallback`).

        The INSERT bulk fast-path is used only when the target's ``create_batch``
        is atomic on raise (:meth:`SyncDatabase._insert_batch_atomic`). On a
        backend whose bulk create is non-atomic (Elasticsearch's per-item bulk
        API, or the ABC per-record ``create_batch`` loop S3 inherits), INSERT
        routes per-record via ``create`` — because the fallback would otherwise
        re-write the records the failed bulk call already durably wrote and count
        them as spurious duplicate failures. This mirrors what those backends'
        own ``stream_write`` already does (``insert_batch_func=None``), so the
        batched migrator and the streaming writer agree per backend.

        Args:
            target: Target database
            batch: Batch of records to write
            progress: Progress tracker to update
            on_error: Optional error handler
            on_conflict: Conflict policy for an id already present in the target
                (insert = fail closed, upsert = overwrite, skip = leave and
                count as skipped).
        """
        for record in batch:
            # Mint ids for id-less records up front so the bulk verbs honor
            # ``record.id`` (mirrors the per-record path's pre-write assignment).
            if not record.id:
                record.generate_id()

        # Only offer the native ``create_batch`` fast-path for INSERT when it is
        # atomic on raise; otherwise ``None`` forces per-record ``create`` so the
        # fallback never re-writes a non-atomic bulk verb's partial success.
        insert_batch_func = (
            target.create_batch if target._insert_batch_atomic() else None
        )
        batch_write_func, single_write_func, skip_on_duplicate = resolve_conflict_write(
            on_conflict,
            insert_batch_func=insert_batch_func,
            single_create_func=target.create,
            upsert_func=target.upsert,
            upsert_batch_func=target.upsert_batch,
        )
        drive_batch_with_fallback(
            batch,
            batch_write_func,
            single_write_func,
            MigrationProgressAccountant(progress, on_error),
            skip_on_duplicate=skip_on_duplicate,
        )

    def validate_migration(
        self,
        source: SyncDatabase,
        target: SyncDatabase,
        query: Query | None = None,
        sample_size: int | None = None
    ) -> tuple[bool, list[str]]:
        """Validate that migration was successful.

        The source-vs-target count check is meaningful only for a from-empty
        ``insert`` migration. Under ``upsert`` it holds only if the source ids
        are a superset of the target's, and under a partial-target ``skip``
        re-run the counts can legitimately differ (the target already held extra
        rows). The per-id sample check (each source id present in the target)
        remains valid for every conflict policy.

        Args:
            source: Source database
            target: Target database
            query: Optional query used for migration
            sample_size: Optional number of records to sample for validation

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Get counts
        source_records = source.search(query or Query())
        target_records = target.search(Query())

        source_count = len(source_records)
        target_count = len(target_records)

        if source_count != target_count:
            issues.append(
                f"Record count mismatch: source={source_count}, target={target_count}"
            )

        # Sample validation
        if sample_size:
            sample = source_records[:sample_size]
        else:
            sample = source_records

        for source_record in sample:
            if source_record.id:
                target_record = target.read(source_record.id)
                if not target_record:
                    issues.append(f"Record {source_record.id} not found in target")

        return len(issues) == 0, issues
