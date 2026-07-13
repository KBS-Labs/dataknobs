"""
Extended tests for migrator module to improve coverage.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Iterator

from dataknobs_data.records import Record
from dataknobs_data.backends.memory import SyncMemoryDatabase as MemoryDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.query import Query
from dataknobs_data.streaming import ConflictPolicy, StreamConfig, StreamResult

from dataknobs_data.migration import (
    Migrator,
    Transformer,
    Migration,
    AddField,
    RenameField,
    MigrationProgress,
)


class TestMigratorAdvanced:
    """Advanced tests for Migrator class."""
    
    def test_migrate_with_skip_logic(self):
        """Test migration with records that get filtered out."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add test records
        for i in range(10):
            record = Record({"id": i, "value": i * 10, "keep": i % 2 == 0})
            source.create(record)
        
        # Custom transformer that filters out odd records
        # Note: Must override transform() to return None for filtering
        class FilteringTransformer(Transformer):
            def transform(self, record):
                # First apply any rules from parent
                result = super().transform(record)
                # Then apply filtering logic
                if result and result.get_value("keep"):
                    return result
                return None  # This will cause the record to be skipped
        
        migrator = Migrator()
        progress = migrator.migrate(
            source=source,
            target=target,
            transform=FilteringTransformer(),
            batch_size=3
        )
        
        # Should have 5 succeeded (even numbers) and 5 skipped (odd numbers)
        assert progress.succeeded == 5
        assert progress.skipped == 5
        
        # Verify only even records in target
        target_records = target.search(Query())
        assert len(target_records) == 5
        assert all(r.get_value("keep") for r in target_records)
    
    def test_migrate_with_error_no_handler(self):
        """Test migration stops on error when no error handler provided."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add test records
        for i in range(5):
            record = Record({"id": i, "value": i})
            source.create(record)
        
        # Mock target that fails on third record
        call_count = [0]  # Use list to avoid nonlocal issues
        original_create = target.create
        
        def failing_create(record):
            call_count[0] += 1
            if call_count[0] == 3:
                raise ValueError("Database error")
            return original_create(record)
        
        target.create = failing_create
        
        migrator = Migrator()
        
        # Test should now raise since no error handler is provided
        with pytest.raises(ValueError, match="Database error"):
            progress = migrator.migrate(
                source=source,
                target=target,
                batch_size=1  # Process one at a time to control error timing
            )
    
    def test_migrate_with_error_handler_continues(self):
        """Test migration continues when error handler returns True."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add test records
        for i in range(5):
            record = Record({"id": i, "value": i})
            source.create(record)
        
        # Mock target that fails on third record
        call_count = [0]
        original_create = target.create
        
        def failing_create(record):
            call_count[0] += 1
            if call_count[0] == 3:
                raise ValueError("Database error")
            return original_create(record)
        
        target.create = failing_create
        
        # Error handler that continues
        def error_handler(error, record):
            return True  # Continue processing
        
        migrator = Migrator()
        progress = migrator.migrate(
            source=source,
            target=target,
            batch_size=1,  # Process one at a time
            on_error=error_handler
        )
        
        # Should process all records: 2 before error, 1 error, 2 after
        assert progress.succeeded == 4  # All except the one that failed
        assert progress.failed == 1  # The one that errored
        assert len(progress.errors) == 1
        assert "Database error" in str(progress.errors[0])
    
    def test_migrate_stream_basic(self):
        """Test stream-based migration."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add test records
        for i in range(100):
            record = Record({"id": i, "value": i * 2})
            source.create(record)
        
        # Configure streaming
        config = StreamConfig(batch_size=10, prefetch=2)
        
        migrator = Migrator()
        progress = migrator.migrate_stream(
            source=source,
            target=target,
            config=config
        )
        
        # All records should be migrated
        assert progress.succeeded == 100
        assert progress.failed == 0
        
        # Verify target has all records
        target_records = target.search(Query())
        assert len(target_records) == 100
    
    def test_migrate_stream_with_transform(self):
        """Test stream migration with transformation."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add test records
        for i in range(50):
            record = Record({"old_id": i, "data": f"item_{i}"})
            source.create(record)
        
        # Create transformer
        transformer = Transformer()
        transformer.map("old_id", "new_id", lambda x: x * 100)
        transformer.add("processed", True)
        
        config = StreamConfig(batch_size=5)
        
        migrator = Migrator()
        progress = migrator.migrate_stream(
            source=source,
            target=target,
            transform=transformer,
            config=config
        )
        
        assert progress.succeeded == 50
        
        # Verify transformation applied
        target_records = target.search(Query())
        assert all("new_id" in r.fields for r in target_records)
        assert all("old_id" not in r.fields for r in target_records)
        assert all(r.get_value("processed") is True for r in target_records)
    
    def test_migrate_stream_with_filter(self):
        """Test stream migration with query filter."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add mixed records
        for i in range(30):
            record = Record({
                "id": i,
                "type": "important" if i % 3 == 0 else "regular",
                "value": i
            })
            source.create(record)
        
        # Only migrate important records
        query = Query().filter("type", "=", "important")
        
        migrator = Migrator()
        progress = migrator.migrate_stream(
            source=source,
            target=target,
            query=query
        )
        
        # Should have 10 important records (0, 3, 6, 9, ...)
        assert progress.succeeded == 10
        
        target_records = target.search(Query())
        assert len(target_records) == 10
        assert all(r.get_value("type") == "important" for r in target_records)
    
    def test_migrate_stream_with_errors(self):
        """Test stream migration with error handling."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add test records
        for i in range(20):
            record = Record({"id": i, "value": i})
            source.create(record)
        
        # Transformer that fails on specific values
        class FailingTransformer(Transformer):
            def transform(self, record):
                value = record.get_value("value")
                if value in [5, 10, 15]:
                    raise ValueError(f"Cannot process {value}")
                # Apply parent rules then return
                return super().transform(record)
        
        # Error handler that continues
        def error_handler(error, record):
            return True  # Continue processing
        
        config = StreamConfig(on_error=error_handler)
        
        migrator = Migrator()
        progress = migrator.migrate_stream(
            source=source,
            target=target,
            transform=FailingTransformer(),
            config=config
        )
        
        # The transformer errors are caught in the stream transform
        # Should have 17 succeeded (20 - 3 errors)
        assert progress.succeeded == 17
        assert progress.failed == 3
    
    def test_migrate_stream_with_migration(self):
        """Test stream migration with Migration object."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add test records
        for i in range(25):
            record = Record({
                "user_id": i,
                "username": f"user_{i}",
                "old_field": "deprecated"
            })
            source.create(record)
        
        # Create migration
        migration = Migration("v1", "v2")
        migration.add(RenameField("user_id", "id"))
        migration.add(AddField("version", "2.0"))
        
        migrator = Migrator()
        progress = migrator.migrate_stream(
            source=source,
            target=target,
            transform=migration
        )
        
        assert progress.succeeded == 25
        
        # Verify migration applied
        target_records = target.search(Query())
        assert all("id" in r.fields for r in target_records)
        assert all("user_id" not in r.fields for r in target_records)
        assert all(r.get_value("version") == "2.0" for r in target_records)
    
    def test_migrate_parallel(self):
        """Test parallel migration with partitions."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add partitioned data
        for partition in range(4):
            for i in range(25):
                record = Record({
                    "id": f"p{partition}_r{i}",
                    "partition_id": partition,
                    "value": i
                })
                source.create(record)
        
        migrator = Migrator()
        progress = migrator.migrate_parallel(
            source=source,
            target=target,
            partitions=4,
            partition_field="partition_id"
        )
        
        # Should migrate all 100 records (4 partitions * 25 records)
        assert progress.succeeded == 100
        assert progress.failed == 0
        
        # Verify all partitions migrated
        target_records = target.search(Query())
        assert len(target_records) == 100
        
        # Check partition distribution
        partition_counts = {}
        for record in target_records:
            p_id = record.get_value("partition_id")
            partition_counts[p_id] = partition_counts.get(p_id, 0) + 1
        
        assert len(partition_counts) == 4
        assert all(count == 25 for count in partition_counts.values())
    
    def test_migrate_parallel_with_transform(self):
        """Test parallel migration with transformation."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add partitioned data
        for partition in range(2):
            for i in range(10):
                record = Record({
                    "partition_id": partition,
                    "old_value": i
                })
                source.create(record)
        
        # Transformer to apply
        transformer = Transformer()
        transformer.map("old_value", "new_value", lambda x: x * 10)
        transformer.add("migrated", True)
        
        migrator = Migrator()
        progress = migrator.migrate_parallel(
            source=source,
            target=target,
            transform=transformer,
            partitions=2
        )
        
        assert progress.succeeded == 20
        
        # Verify transformation
        target_records = target.search(Query())
        assert all("new_value" in r.fields for r in target_records)
        assert all(r.get_value("migrated") is True for r in target_records)
    
    @pytest.mark.asyncio
    async def test_migrate_async_basic(self):
        """Test async migration."""
        source = AsyncMemoryDatabase()
        target = AsyncMemoryDatabase()
        
        # Add test records
        for i in range(50):
            record = Record({"id": i, "data": f"async_{i}"})
            await source.create(record)
        
        migrator = Migrator()
        progress = await migrator.migrate_async(
            source=source,
            target=target
        )
        
        assert progress.succeeded == 50
        assert progress.failed == 0
        
        # Verify migration
        target_records = await target.search(Query())
        assert len(target_records) == 50
    
    @pytest.mark.asyncio
    async def test_migrate_async_with_transform(self):
        """Test async migration with transformation."""
        source = AsyncMemoryDatabase()
        target = AsyncMemoryDatabase()
        
        # Add test records
        for i in range(30):
            record = Record({
                "original_id": i,
                "value": i * 5
            })
            await source.create(record)
        
        # Create transformer
        transformer = Transformer()
        transformer.rename("original_id", "new_id")
        transformer.map("value", "doubled_value", lambda x: x * 2)
        
        config = StreamConfig(batch_size=10)
        
        migrator = Migrator()
        progress = await migrator.migrate_async(
            source=source,
            target=target,
            transform=transformer,
            config=config
        )
        
        assert progress.succeeded == 30
        
        # Verify transformation
        target_records = await target.search(Query())
        assert all("new_id" in r.fields for r in target_records)
        assert all("original_id" not in r.fields for r in target_records)
        # Values should be doubled: original * 5 * 2
        for i, record in enumerate(sorted(target_records, key=lambda r: r.get_value("new_id"))):
            expected = i * 5 * 2
            assert record.get_value("doubled_value") == expected
    
    @pytest.mark.asyncio
    async def test_migrate_async_with_errors(self):
        """Test async migration error handling."""
        source = AsyncMemoryDatabase()
        target = AsyncMemoryDatabase()
        
        # Add test records
        for i in range(15):
            record = Record({"id": i, "fail": i == 7})
            await source.create(record)
        
        # Transformer that fails on certain records
        class AsyncFailingTransformer(Transformer):
            def transform(self, record):
                if record.get_value("fail"):
                    raise ValueError("Intentional failure")
                return super().transform(record)
        
        # Error handler
        def handle_error(error, record):
            return True  # Continue
        
        config = StreamConfig(on_error=handle_error)
        
        migrator = Migrator()
        progress = await migrator.migrate_async(
            source=source,
            target=target,
            transform=AsyncFailingTransformer(),
            config=config
        )
        
        # Should have 14 succeeded (15 - 1 error)
        assert progress.succeeded == 14
        assert progress.failed == 1
    
    def test_write_batch_error_handling(self):
        """Test _write_batch error handling."""
        target = MemoryDatabase()
        
        # Create batch with one problematic record
        batch = [
            Record({"id": 1, "value": "good"}),
            Record({"id": 2, "value": "bad"}),  # This will fail
            Record({"id": 3, "value": "good"}),
        ]
        
        # Mock target to fail on specific record
        original_create = target.create
        def failing_create(record):
            if record.get_value("value") == "bad":
                raise ValueError("Cannot create bad record")
            return original_create(record)
        
        target.create = failing_create
        
        progress = MigrationProgress().start()
        
        # Test without error handler - should raise
        migrator = Migrator()
        with pytest.raises(ValueError, match="Cannot create bad record"):
            migrator._write_batch(target, batch, progress)
        
        # Should have processed 1 successfully before error
        assert progress.succeeded == 1
        assert progress.failed == 1
        
        # Test with error handler that continues. Use a fresh target so the
        # "good" records are genuine inserts: create() is an atomic insert and
        # re-creating an id already written above would (correctly) raise a
        # DuplicateRecordError rather than silently overwrite.
        target2 = MemoryDatabase()
        original_create2 = target2.create

        def failing_create2(record):
            if record.get_value("value") == "bad":
                raise ValueError("Cannot create bad record")
            return original_create2(record)

        target2.create = failing_create2

        progress2 = MigrationProgress().start()

        def error_handler(error, record):
            return True  # Continue processing

        migrator._write_batch(target2, batch, progress2, on_error=error_handler)

        # Should process all, with one failure
        assert progress2.succeeded == 2
        assert progress2.failed == 1
    
    def test_validate_migration_with_sample(self):
        """Test migration validation with sampling."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add records to both databases
        for i in range(100):
            record = Record({"value": i})
            record.id = f"id_{i}"
            source.create(record)
            target.create(record)
        
        migrator = Migrator()
        
        # Test with full validation
        is_valid, issues = migrator.validate_migration(source, target)
        assert is_valid
        assert len(issues) == 0
        
        # Test with sample validation
        is_valid, issues = migrator.validate_migration(
            source, target, 
            sample_size=10
        )
        assert is_valid
        assert len(issues) == 0
        
        # Add extra record to source to create mismatch
        extra = Record({"value": 999})
        extra.id = "extra_id"
        source.create(extra)
        
        is_valid, issues = migrator.validate_migration(source, target)
        assert not is_valid
        assert any("count mismatch" in issue.lower() for issue in issues)
        
        # Test with missing record in target
        # First, recreate target with missing record
        target2 = MemoryDatabase()
        target_records = target.search(Query())
        
        # Copy all but first record to new target
        for r in target_records[1:]:
            target2.create(r)
        
        is_valid, issues = migrator.validate_migration(
            source, target2,
            sample_size=50
        )
        assert not is_valid  # Should detect the mismatch
    
    def test_migrate_with_progress_callback(self):
        """Test progress callback functionality."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Add test records
        for i in range(40):
            record = Record({"id": i})
            source.create(record)
        
        # Track progress updates
        progress_history = []
        
        def track_progress(progress):
            progress_history.append({
                "processed": progress.processed,
                "succeeded": progress.succeeded,
                "percent": progress.percent
            })
        
        migrator = Migrator()
        final_progress = migrator.migrate(
            source=source,
            target=target,
            batch_size=10,
            on_progress=track_progress
        )
        
        assert final_progress.succeeded == 40
        assert len(progress_history) > 0
        
        # Verify progress increased monotonically
        for i in range(1, len(progress_history)):
            assert progress_history[i]["processed"] >= progress_history[i-1]["processed"]
    
    def test_migrate_stream_without_count(self):
        """Test stream migration when count is not available."""
        source = MemoryDatabase()
        target = MemoryDatabase()
        
        # Mock source to raise exception on count
        original_count = source.count
        def failing_count(query):
            raise NotImplementedError("Count not supported")
        
        source.count = failing_count
        
        # Add test records
        for i in range(15):
            record = Record({"id": i})
            source.create(record)
        
        migrator = Migrator()
        progress = migrator.migrate_stream(
            source=source,
            target=target
        )
        
        # Should still migrate successfully without total count
        assert progress.succeeded == 15
        assert progress.failed == 0
        assert progress.processed == 15
        
        # When count is unavailable, total should be None or 0
        assert progress.total is None or progress.total == 0
        
        # Progress percent should handle missing total gracefully
        assert progress.percent == 0.0 or progress.percent == 100.0
        
        # Verify all records were migrated
        assert target.count(Query()) == 15

        # Restore original count
        source.count = original_count


def _seed(db, ids, value, *, partition=0):
    """Seed a sync backend with records carrying a stable id + partition_id."""
    for i in ids:
        db.create(Record({"v": value, "partition_id": partition}, id=str(i)))


async def _aseed(db, ids, value, *, partition=0):
    """Seed an async backend with records carrying a stable id + partition_id."""
    for i in ids:
        await db.create(Record({"v": value, "partition_id": partition}, id=str(i)))


def _run_sync(method, src, tgt, policy):
    """Dispatch a conflict-policy migration through each sync ``migrate*`` path."""
    m = Migrator()
    if method == "migrate":
        return m.migrate(src, tgt, on_conflict=policy)
    if method == "migrate_stream":
        return m.migrate_stream(src, tgt, config=StreamConfig(on_conflict=policy))
    if method == "migrate_parallel":
        return m.migrate_parallel(src, tgt, partitions=1, on_conflict=policy)
    raise AssertionError(f"unknown method {method}")


class TestMigratorConflictPolicy:
    """Conflict policy (insert / upsert / skip) across all four migrate paths.

    Built on real in-process memory backends (no mocks). ``create()`` is an
    atomic insert post-tightening, so a re-run into a populated target has no
    idempotent path without a policy — these tests pin the default (strict
    insert) starting point and prove upsert/skip are threaded through every
    public ``migrate*`` method and both shared write helpers.
    """

    def test_batched_insert_records_collision_as_failure(self):
        """RED/pin: default (insert) fails closed on a colliding id (batched)."""
        src, tgt = MemoryDatabase(), MemoryDatabase()
        _seed(src, [1, 2, 3], "src")
        _seed(tgt, [2], "old")
        # No policy => strict insert. Continue past the failure (on_error → True)
        # so it is recorded rather than aborting the run.
        progress = Migrator().migrate(src, tgt, on_error=lambda e, r: True)
        assert progress.failed == 1
        assert progress.succeeded == 2
        assert progress.skipped == 0
        # The pre-existing target row is untouched.
        assert tgt.read("2").get_value("v") == "old"

    def test_streaming_insert_overwrites_via_batch_fastpath(self):
        """Pin the documented streaming fast-path asymmetry.

        ``create_batch`` overwrites a colliding id instead of raising, so the
        streaming INSERT fast-path does NOT fail closed the way batched
        ``migrate()`` does. This is pre-existing behavior; the test guards that
        the default policy did not accidentally change it.
        """
        src, tgt = MemoryDatabase(), MemoryDatabase()
        _seed(src, [1, 2, 3], "src")
        _seed(tgt, [2], "old")
        progress = Migrator().migrate_stream(src, tgt)  # default INSERT
        assert progress.failed == 0
        assert tgt.read("2").get_value("v") == "src"  # overwritten by create_batch

    def test_batched_upsert_overwrites(self):
        """UPSERT overwrites the target row and counts the id as a success."""
        src, tgt = MemoryDatabase(), MemoryDatabase()
        _seed(src, [1, 2, 3], "src")
        _seed(tgt, [2], "old")
        progress = Migrator().migrate(src, tgt, on_conflict="upsert")
        assert progress.failed == 0
        assert progress.skipped == 0
        assert progress.succeeded == 3
        assert tgt.read("2").get_value("v") == "src"

    def test_batched_skip_is_idempotent(self):
        """SKIP leaves the existing row and re-runs converge to all-skipped."""
        src, tgt = MemoryDatabase(), MemoryDatabase()
        _seed(src, [1, 2, 3], "src")
        _seed(tgt, [2], "old")
        m = Migrator()

        first = m.migrate(src, tgt, on_conflict="skip")
        assert first.skipped == 1
        assert first.failed == 0
        assert first.succeeded == 2
        assert tgt.read("2").get_value("v") == "old"  # untouched

        # Second run: every id already present => all skipped, nothing failed.
        second = m.migrate(src, tgt, on_conflict="skip")
        assert second.skipped == 3
        assert second.failed == 0
        assert second.succeeded == 0
        # Rows are byte-identical to the first run.
        assert tgt.read("1").get_value("v") == "src"
        assert tgt.read("2").get_value("v") == "old"

    @pytest.mark.parametrize("method", ["migrate", "migrate_stream", "migrate_parallel"])
    def test_all_sync_methods_upsert(self, method):
        """Every sync path honors upsert (no failures; target matches source)."""
        src, tgt = MemoryDatabase(), MemoryDatabase()
        _seed(src, [1, 2, 3], "src")
        _seed(tgt, [2], "old")
        progress = _run_sync(method, src, tgt, "upsert")
        assert progress.failed == 0
        assert tgt.read("2").get_value("v") == "src"
        assert tgt.read("1") is not None
        assert tgt.read("3") is not None

    @pytest.mark.parametrize("method", ["migrate", "migrate_stream", "migrate_parallel"])
    def test_all_sync_methods_skip(self, method):
        """Every sync path honors skip (colliding id skipped, old row kept)."""
        src, tgt = MemoryDatabase(), MemoryDatabase()
        _seed(src, [1, 2, 3], "src")
        _seed(tgt, [2], "old")
        progress = _run_sync(method, src, tgt, "skip")
        assert progress.failed == 0
        assert progress.skipped >= 1
        assert tgt.read("2").get_value("v") == "old"  # skip kept the old row
        assert tgt.read("1").get_value("v") == "src"

    @pytest.mark.asyncio
    async def test_async_upsert(self):
        """migrate_async honors upsert via config.on_conflict."""
        src, tgt = AsyncMemoryDatabase(), AsyncMemoryDatabase()
        await _aseed(src, [1, 2, 3], "src")
        await _aseed(tgt, [2], "old")
        progress = await Migrator().migrate_async(
            src, tgt, config=StreamConfig(on_conflict="upsert")
        )
        assert progress.failed == 0
        assert (await tgt.read("2")).get_value("v") == "src"

    @pytest.mark.asyncio
    async def test_async_skip(self):
        """migrate_async honors skip via config.on_conflict."""
        src, tgt = AsyncMemoryDatabase(), AsyncMemoryDatabase()
        await _aseed(src, [1, 2, 3], "src")
        await _aseed(tgt, [2], "old")
        progress = await Migrator().migrate_async(
            src, tgt, config=StreamConfig(on_conflict="skip")
        )
        assert progress.failed == 0
        assert progress.skipped >= 1
        assert (await tgt.read("2")).get_value("v") == "old"

    def test_streaming_skip_accounting(self):
        """StreamResult carries `skipped` and folds into progress.skipped."""
        src, tgt = MemoryDatabase(), MemoryDatabase()
        _seed(src, [1, 2, 3], "src")
        _seed(tgt, [2], "old")
        progress = Migrator().migrate_stream(
            src, tgt, config=StreamConfig(on_conflict="skip")
        )
        assert progress.skipped == 1

        # StreamResult itself carries the field and merges it.
        result = StreamResult(skipped=2)
        result.merge(StreamResult(skipped=3))
        assert result.skipped == 5

    def test_str_coercion_and_invalid_rejected(self):
        """A bare string resolves; an unknown policy fails closed, not silent."""
        src, tgt = MemoryDatabase(), MemoryDatabase()
        _seed(src, [1], "src")
        progress = Migrator().migrate(src, tgt, on_conflict="upsert")
        assert progress.failed == 0
        assert ConflictPolicy("skip") is ConflictPolicy.SKIP
        with pytest.raises(ValueError):
            StreamConfig(on_conflict="nonsense")
