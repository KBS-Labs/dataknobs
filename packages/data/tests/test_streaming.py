"""Tests for streaming API functionality."""

import asyncio
import time
from typing import AsyncIterator, Iterator

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.query import Query
from dataknobs_data.records import Record
from dataknobs_data.streaming import StreamConfig, StreamProcessor, StreamResult


class TestStreamConfig:
    """Test StreamConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StreamConfig()
        assert config.batch_size == 1000
        assert config.prefetch == 2
        assert config.timeout is None
        assert config.on_error is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        def error_handler(e, r):
            return True
        
        config = StreamConfig(
            batch_size=500,
            prefetch=5,
            timeout=30.0,
            on_error=error_handler
        )
        assert config.batch_size == 500
        assert config.prefetch == 5
        assert config.timeout == 30.0
        assert config.on_error == error_handler
    
    def test_invalid_batch_size(self):
        """Test that invalid batch size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            StreamConfig(batch_size=0)
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            StreamConfig(batch_size=-1)
    
    def test_invalid_prefetch(self):
        """Test that invalid prefetch raises error."""
        with pytest.raises(ValueError, match="prefetch must be non-negative"):
            StreamConfig(prefetch=-1)
    
    def test_invalid_timeout(self):
        """Test that invalid timeout raises error."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            StreamConfig(timeout=0)
        
        with pytest.raises(ValueError, match="timeout must be positive"):
            StreamConfig(timeout=-1)


class TestStreamResult:
    """Test StreamResult dataclass."""
    
    def test_default_result(self):
        """Test default result values."""
        result = StreamResult()
        assert result.total_processed == 0
        assert result.successful == 0
        assert result.failed == 0
        assert result.errors == []
        assert result.duration == 0.0
        assert result.success_rate == 0.0
    
    def test_success_rate(self):
        """Test success rate calculation."""
        result = StreamResult()
        
        # No records processed
        assert result.success_rate == 0.0
        
        # All successful
        result.total_processed = 100
        result.successful = 100
        assert result.success_rate == 100.0
        
        # Half successful
        result.successful = 50
        assert result.success_rate == 50.0
        
        # None successful
        result.successful = 0
        assert result.success_rate == 0.0
    
    def test_add_error(self):
        """Test adding errors to result."""
        result = StreamResult()
        
        # Add error with record ID
        result.add_error("rec1", ValueError("Invalid value"))
        assert len(result.errors) == 1
        assert result.errors[0]["record_id"] == "rec1"
        assert result.errors[0]["error"] == "Invalid value"
        assert result.errors[0]["type"] == "ValueError"
        
        # Add error without record ID
        result.add_error(None, TypeError("Type error"))
        assert len(result.errors) == 2
        assert result.errors[1]["record_id"] is None
        assert result.errors[1]["type"] == "TypeError"
    
    def test_merge_results(self):
        """Test merging results."""
        result1 = StreamResult(
            total_processed=50,
            successful=45,
            failed=5,
            duration=10.0
        )
        result1.add_error("rec1", ValueError("Error 1"))
        
        result2 = StreamResult(
            total_processed=30,
            successful=28,
            failed=2,
            duration=5.0
        )
        result2.add_error("rec2", TypeError("Error 2"))
        
        result1.merge(result2)
        
        assert result1.total_processed == 80
        assert result1.successful == 73
        assert result1.failed == 7
        assert result1.duration == 15.0
        assert len(result1.errors) == 2
    
    def test_string_representation(self):
        """Test string representation of result."""
        result = StreamResult(
            total_processed=100,
            successful=95,
            failed=5,
            duration=12.5
        )
        
        str_repr = str(result)
        assert "processed=100" in str_repr
        assert "successful=95" in str_repr
        assert "failed=5" in str_repr
        assert "success_rate=95.0%" in str_repr
        assert "duration=12.50s" in str_repr


class TestStreamProcessor:
    """Test StreamProcessor utilities."""
    
    def test_batch_iterator(self):
        """Test batching records."""
        records = [Record() for _ in range(10)]
        batches = list(StreamProcessor.batch_iterator(iter(records), batch_size=3))
        
        assert len(batches) == 4  # 3 + 3 + 3 + 1
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1
    
    def test_filter_stream(self):
        """Test filtering records in stream."""
        records = []
        for i in range(10):
            record = Record()
            record.set_field("value", i)
            records.append(record)
        
        # Filter even values
        filtered = list(StreamProcessor.filter_stream(
            iter(records),
            lambda r: r.get_value("value") % 2 == 0
        ))
        
        assert len(filtered) == 5
        assert all(r.get_value("value") % 2 == 0 for r in filtered)
    
    def test_transform_stream(self):
        """Test transforming records in stream."""
        records = []
        for i in range(5):
            record = Record()
            record.set_field("value", i)
            records.append(record)
        
        # Double values, filter out odd ones
        def transform(record):
            value = record.get_value("value")
            if value % 2 == 0:
                new_record = Record()
                new_record.set_field("value", value * 2)
                return new_record
            return None
        
        transformed = list(StreamProcessor.transform_stream(
            iter(records),
            transform
        ))
        
        assert len(transformed) == 3  # 0, 2, 4
        assert transformed[0].get_value("value") == 0
        assert transformed[1].get_value("value") == 4
        assert transformed[2].get_value("value") == 8
    
    @pytest.mark.asyncio
    async def test_async_batch_iterator(self):
        """Test async batching records."""
        async def record_generator():
            for i in range(10):
                yield Record()
        
        batches = []
        async for batch in StreamProcessor.async_batch_iterator(
            record_generator(),
            batch_size=3
        ):
            batches.append(batch)
        
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[3]) == 1
    
    @pytest.mark.asyncio
    async def test_async_filter_stream(self):
        """Test async filtering records."""
        async def record_generator():
            for i in range(10):
                record = Record()
                record.set_field("value", i)
                yield record
        
        filtered = []
        async for record in StreamProcessor.async_filter_stream(
            record_generator(),
            lambda r: r.get_value("value") % 2 == 0
        ):
            filtered.append(record)
        
        assert len(filtered) == 5
        assert all(r.get_value("value") % 2 == 0 for r in filtered)
    
    @pytest.mark.asyncio
    async def test_async_transform_stream(self):
        """Test async transforming records."""
        async def record_generator():
            for i in range(5):
                record = Record()
                record.set_field("value", i)
                yield record
        
        def transform(record):
            value = record.get_value("value")
            if value % 2 == 0:
                new_record = Record()
                new_record.set_field("value", value * 2)
                return new_record
            return None
        
        transformed = []
        async for record in StreamProcessor.async_transform_stream(
            record_generator(),
            transform
        ):
            transformed.append(record)
        
        assert len(transformed) == 3
        assert transformed[0].get_value("value") == 0
        assert transformed[1].get_value("value") == 4
        assert transformed[2].get_value("value") == 8


class TestSyncMemoryDatabaseStreaming:
    """Test synchronous memory database streaming."""
    
    def setup_method(self):
        """Set up test database."""
        self.db = SyncMemoryDatabase()
        
        # Add test records
        self.records = []
        self.record_ids = []
        for i in range(100):
            record = Record(id=str(i).zfill(3))  # Use zero-padded IDs for consistent string comparison
            record.set_field("value", i * 10)
            record.set_field("category", "A" if i % 2 == 0 else "B")
            record_id = self.db.create(record)
            self.records.append(record)
            self.record_ids.append(record_id)
    
    def test_stream_read_all(self):
        """Test streaming all records."""
        config = StreamConfig(batch_size=10)
        streamed = list(self.db.stream_read(config=config))
        
        assert len(streamed) == 100
        # Records should be deep copies
        assert all(r is not orig for r, orig in zip(streamed, self.records))
    
    def test_stream_read_with_query(self):
        """Test streaming with query filter."""
        query = Query().filter("category", "=", "A")
        config = StreamConfig(batch_size=5)
        
        streamed = list(self.db.stream_read(query, config))
        
        assert len(streamed) == 50
        assert all(r.get_value("category") == "A" for r in streamed)
    
    def test_stream_write(self):
        """Test streaming records into database."""
        new_db = SyncMemoryDatabase()
        
        def record_generator():
            for i in range(50):
                record = Record()
                record.set_field("id", i)
                record.set_field("value", i * 100)
                yield record
        
        config = StreamConfig(batch_size=10)
        result = new_db.stream_write(record_generator(), config)
        
        assert result.total_processed == 50
        assert result.successful == 50
        assert result.failed == 0
        assert result.success_rate == 100.0
        assert new_db.count() == 50
    
    def test_stream_write_with_errors(self):
        """Test stream write error handling."""
        db = SyncMemoryDatabase()
        
        def bad_generator():
            for i in range(10):
                if i == 5:
                    # This will cause issues if the database validates
                    yield None  # Invalid record
                else:
                    record = Record()
                    record.set_field("id", i)
                    yield record
        
        # Mock create_batch to fail on None
        original_create_batch = db.create_batch
        def failing_create_batch(records):
            if any(r is None for r in records):
                raise ValueError("Invalid record")
            return original_create_batch(records)
        
        db.create_batch = failing_create_batch
        
        config = StreamConfig(batch_size=3)
        result = db.stream_write(bad_generator(), config)
        
        # The batch containing None should fail
        assert result.failed > 0
        assert len(result.errors) > 0
    
    def test_stream_transform(self):
        """Test stream transformation."""
        def double_values(record):
            new_record = record.copy()
            new_record.set_field("value", record.get_value("value") * 2)
            return new_record
        
        config = StreamConfig(batch_size=10)
        transformed = list(self.db.stream_transform(
            transform=double_values,
            config=config
        ))
        
        assert len(transformed) == 100
        for i, record in enumerate(transformed):
            expected_value = i * 10 * 2
            assert record.get_value("value") == expected_value
    
    def test_stream_transform_with_filter(self):
        """Test stream transformation with filtering."""
        def filter_and_double(record):
            if record.get_value("category") == "A":
                new_record = record.copy()
                new_record.set_field("value", record.get_value("value") * 2)
                return new_record
            return None  # Filter out category B
        
        query = Query().filter("id", "<", "020")  # Use string comparison with zero-padded value
        config = StreamConfig(batch_size=5)
        
        transformed = list(self.db.stream_transform(
            query=query,
            transform=filter_and_double,
            config=config
        ))
        
        # Only category A records with id < 20
        assert len(transformed) == 10
        assert all(r.get_value("category") == "A" for r in transformed)


@pytest.mark.asyncio
class TestAsyncMemoryDatabaseStreaming:
    """Test async memory database streaming."""
    
    async def setup_records(self):
        """Set up test database with records."""
        self.db = AsyncMemoryDatabase()
        
        self.records = []
        for i in range(100):
            record = Record()
            record.set_field("id", i)
            record.set_field("value", i * 10)
            record.set_field("category", "A" if i % 2 == 0 else "B")
            await self.db.create(record)
            self.records.append(record)
    
    async def test_stream_read_all(self):
        """Test async streaming all records."""
        await self.setup_records()
        
        config = StreamConfig(batch_size=10)
        streamed = []
        async for record in self.db.stream_read(config=config):
            streamed.append(record)
        
        assert len(streamed) == 100
    
    async def test_stream_read_with_query(self):
        """Test async streaming with query filter."""
        await self.setup_records()
        
        query = Query().filter("category", "=", "B")
        config = StreamConfig(batch_size=5)
        
        streamed = []
        async for record in self.db.stream_read(query, config):
            streamed.append(record)
        
        assert len(streamed) == 50
        assert all(r.get_value("category") == "B" for r in streamed)
    
    async def test_stream_write(self):
        """Test async streaming records into database."""
        new_db = AsyncMemoryDatabase()
        
        async def record_generator():
            for i in range(50):
                record = Record()
                record.set_field("id", i)
                record.set_field("value", i * 100)
                yield record
        
        config = StreamConfig(batch_size=10)
        result = await new_db.stream_write(record_generator(), config)
        
        assert result.total_processed == 50
        assert result.successful == 50
        assert result.failed == 0
        assert await new_db.count() == 50
    
    async def test_stream_transform(self):
        """Test async stream transformation."""
        await self.setup_records()
        
        def halve_values(record):
            new_record = record.copy()
            new_record.set_field("value", record.get_value("value") // 2)
            return new_record
        
        config = StreamConfig(batch_size=10)
        transformed = []
        async for record in self.db.stream_transform(
            transform=halve_values,
            config=config
        ):
            transformed.append(record)
        
        assert len(transformed) == 100
        for i, record in enumerate(transformed):
            expected_value = (i * 10) // 2
            assert record.get_value("value") == expected_value
    
    async def test_error_handling_with_callback(self):
        """Test error handling with custom callback."""
        db = AsyncMemoryDatabase()
        errors_seen = []
        
        def error_handler(error, record):
            errors_seen.append(error)
            return True  # Continue processing
        
        async def bad_generator():
            for i in range(10):
                if i == 5:
                    yield None  # Will cause error
                else:
                    record = Record()
                    record.set_field("id", i)
                    yield record
        
        # Mock create_batch to fail on None
        original_create_batch = db.create_batch
        async def failing_create_batch(records):
            if any(r is None for r in records):
                raise ValueError("Invalid record")
            return await original_create_batch(records)
        
        db.create_batch = failing_create_batch
        
        config = StreamConfig(
            batch_size=3,
            on_error=error_handler
        )
        
        result = await db.stream_write(bad_generator(), config)
        
        # Error handler should have been called
        assert len(errors_seen) > 0
        assert result.failed > 0


class TestStreamingIntegration:
    """Integration tests for streaming between databases."""
    
    def test_sync_memory_to_memory_migration(self):
        """Test streaming migration between sync memory databases."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        # Populate source
        for i in range(100):
            record = Record()
            record.set_field("id", i)
            record.set_field("value", i * 10)
            source.create(record)
        
        # Stream from source to target
        config = StreamConfig(batch_size=10)
        result = target.stream_write(
            source.stream_read(config=config),
            config=config
        )
        
        assert result.total_processed == 100
        assert result.successful == 100
        assert target.count() == 100
    
    def test_filtered_migration(self):
        """Test streaming migration with filtering."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        # Populate source
        for i in range(100):
            record = Record()
            record.set_field("id", i)
            record.set_field("value", i * 10)
            record.set_field("active", i % 3 != 0)
            source.create(record)
        
        # Stream only active records
        query = Query().filter("active", "=", True)
        config = StreamConfig(batch_size=10)
        
        result = target.stream_write(
            source.stream_read(query, config),
            config
        )
        
        # 66 records should be active (not divisible by 3)
        # Numbers 0-99: 0,3,6,9...99 are divisible by 3 (34 numbers), so 100-34=66
        assert result.successful == 66
        assert target.count() == 66
    
    def test_transformed_migration(self):
        """Test streaming migration with transformation."""
        source = SyncMemoryDatabase()
        target = SyncMemoryDatabase()
        
        # Populate source
        for i in range(50):
            record = Record()
            record.set_field("id", i)
            record.set_field("value", i)
            record.set_field("category", "old")
            source.create(record)
        
        # Transform records during migration
        def transform(record):
            new_record = Record()
            new_record.set_field("id", record.get_value("id"))
            new_record.set_field("value", record.get_value("value") * 100)
            new_record.set_field("category", "new")
            new_record.set_field("migrated", True)
            return new_record
        
        config = StreamConfig(batch_size=10)
        result = target.stream_write(
            source.stream_transform(transform=transform, config=config),
            config
        )
        
        assert result.successful == 50
        assert target.count() == 50
        
        # Verify transformation
        migrated = target.search(Query())
        assert all(r.get_value("category") == "new" for r in migrated)
        assert all(r.get_value("migrated") is True for r in migrated)
    
    @pytest.mark.asyncio
    async def test_async_memory_to_memory_migration(self):
        """Test async streaming migration."""
        source = AsyncMemoryDatabase()
        target = AsyncMemoryDatabase()
        
        # Populate source
        for i in range(100):
            record = Record()
            record.set_field("id", i)
            record.set_field("data", f"record_{i}")
            await source.create(record)
        
        # Stream from source to target
        config = StreamConfig(batch_size=20)
        result = await target.stream_write(
            source.stream_read(config=config),
            config=config
        )
        
        assert result.total_processed == 100
        assert result.successful == 100
        assert await target.count() == 100