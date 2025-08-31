"""Enhanced data migrator with streaming support.
"""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

from dataknobs_data.query import Query
from dataknobs_data.streaming import StreamConfig

from .migration import Migration
from .progress import MigrationProgress
from .transformer import Transformer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from dataknobs_data.database import AsyncDatabase, SyncDatabase
    from dataknobs_data.records import Record
    

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
        on_error: Callable[[Exception, Record], bool] | None = None
    ) -> MigrationProgress:
        """Migrate data between databases with optional transformation.
        
        Args:
            source: Source database
            target: Target database
            transform: Optional transformer or migration to apply
            query: Optional query to filter source records
            batch_size: Number of records to process per batch
            on_progress: Optional callback for progress updates
            on_error: Optional error handler (return True to continue)
            
        Returns:
            MigrationProgress with final statistics
        """
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

                batch.append(record)

                # Process batch when full
                if len(batch) >= batch_size:
                    self._write_batch(target, batch, progress, on_error)
                    batch = []

                    if on_progress:
                        on_progress(progress)

            except Exception as e:
                progress.record_failure(str(e), record.id if hasattr(record, 'id') else None, e)
                if on_error:
                    if not on_error(e, record):
                        # Handler says stop - re-raise to stop processing immediately
                        raise
                    # Handler says continue - keep going
                else:
                    # No handler - stop processing immediately
                    raise

        # Process final batch
        if batch:
            self._write_batch(target, batch, progress, on_error)

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
                progress.processed += 1  # Track that we've processed this record
                try:
                    if transform is not None:
                        if isinstance(transform, Transformer):
                            original_id = record.id  # Preserve ID before transformation
                            transformed = transform.transform(record)
                            if transformed:
                                yield transformed
                            else:
                                progress.record_skip("Filtered by transformer", original_id)
                        elif isinstance(transform, Migration):
                            yield transform.apply(record)
                    else:
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
        on_progress: Callable[[MigrationProgress], None] | None = None
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
            
        Returns:
            Combined MigrationProgress
        """
        def migrate_partition(partition_id: int) -> MigrationProgress:
            """Migrate a single partition."""
            query = Query().filter(partition_field, "=", partition_id)
            return self.migrate_stream(source, target, transform, query)

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
                progress.processed += 1  # Track that we've processed this record
                try:
                    if transform is not None:
                        if isinstance(transform, Transformer):
                            original_id = record.id  # Preserve ID before transformation
                            transformed = transform.transform(record)
                            if transformed:
                                yield transformed
                            else:
                                progress.record_skip("Filtered by transformer", original_id)
                        elif isinstance(transform, Migration):
                            yield transform.apply(record)
                    else:
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
        on_error: Callable[[Exception, Record], bool] | None = None
    ) -> None:
        """Write a batch of records to target database.
        
        Args:
            target: Target database
            batch: Batch of records to write
            progress: Progress tracker to update
            on_error: Optional error handler
        """
        for record in batch:
            try:
                # Ensure record has an ID
                if not record.id:
                    record.generate_id()

                target.create(record)
                progress.record_success(record.id)
            except Exception as e:
                progress.record_failure(str(e), record.id, e)
                if on_error:
                    if not on_error(e, record):
                        # Handler says stop - re-raise to stop processing immediately
                        raise
                    # Handler says continue - keep going
                else:
                    # No handler - stop processing immediately
                    raise

    def validate_migration(
        self,
        source: SyncDatabase,
        target: SyncDatabase,
        query: Query | None = None,
        sample_size: int | None = None
    ) -> tuple[bool, list[str]]:
        """Validate that migration was successful.
        
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
