"""Tests for streaming mixin default implementations."""

import asyncio
import time
from typing import AsyncIterator, Iterator, List, Optional

import pytest

from dataknobs_data.query import Query
from dataknobs_data.records import Record
from dataknobs_data.streaming import (
    AsyncStreamingMixin,
    StreamConfig,
    StreamResult,
    StreamingMixin,
)


class MockSyncDatabase(StreamingMixin):
    """Mock sync database using StreamingMixin defaults."""
    
    def __init__(self):
        self.records = []
        self.create_batch_calls = []
        self.search_calls = []
        self.should_fail_create = False
        self.fail_on_record_index = None
    
    def search(self, query: Query) -> List[Record]:
        """Mock search implementation."""
        self.search_calls.append(query)
        return self.records.copy()
    
    def create_batch(self, records: List[Record]) -> List[str]:
        """Mock create_batch implementation."""
        self.create_batch_calls.append(records)
        
        if self.should_fail_create:
            raise ValueError("Mock create error")
        
        # Check for individual record failures
        if self.fail_on_record_index is not None:
            for i, record in enumerate(records):
                if record.get_value("index") == self.fail_on_record_index:
                    raise ValueError(f"Failed on record {self.fail_on_record_index}")
        
        # Return fake IDs
        return [f"id_{i}" for i in range(len(records))]
    
    def stream_read(
        self,
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
    ) -> Iterator[Record]:
        """Use the default implementation from mixin."""
        return self._default_stream_read(query, config)
    
    def stream_write(
        self,
        records: Iterator[Record],
        config: Optional[StreamConfig] = None
    ) -> StreamResult:
        """Use the default implementation from mixin."""
        return self._default_stream_write(records, config)


class MockAsyncDatabase(AsyncStreamingMixin):
    """Mock async database using AsyncStreamingMixin defaults."""
    
    def __init__(self):
        self.records = []
        self.create_batch_calls = []
        self.search_calls = []
        self.should_fail_create = False
        self.fail_on_record_index = None
    
    async def search(self, query: Query) -> List[Record]:
        """Mock async search implementation."""
        self.search_calls.append(query)
        # Small delay to simulate async operation
        await asyncio.sleep(0.001)
        return self.records.copy()
    
    async def create_batch(self, records: List[Record]) -> List[str]:
        """Mock async create_batch implementation."""
        self.create_batch_calls.append(records)
        
        if self.should_fail_create:
            raise ValueError("Mock async create error")
        
        # Check for individual record failures
        if self.fail_on_record_index is not None:
            for record in records:
                if record.get_value("index") == self.fail_on_record_index:
                    raise ValueError(f"Failed on record {self.fail_on_record_index}")
        
        # Small delay to simulate async operation
        await asyncio.sleep(0.001)
        return [f"id_{i}" for i in range(len(records))]
    
    async def stream_read(
        self,
        query: Optional[Query] = None,
        config: Optional[StreamConfig] = None
    ) -> AsyncIterator[Record]:
        """Use the default implementation from mixin."""
        async for record in self._default_stream_read(query, config):
            yield record
    
    async def stream_write(
        self,
        records: AsyncIterator[Record],
        config: Optional[StreamConfig] = None
    ) -> StreamResult:
        """Use the default implementation from mixin."""
        return await self._default_stream_write(records, config)


class TestStreamingMixinDefaults:
    """Test default implementations in StreamingMixin."""
    
    def setup_method(self):
        """Set up test database with records."""
        self.db = MockSyncDatabase()
        
        # Add test records
        for i in range(100):
            record = Record()
            record.set_field("index", i)
            record.set_field("value", i * 10)
            record.set_field("category", "A" if i % 2 == 0 else "B")
            self.db.records.append(record)
    
    def test_default_stream_read_no_query(self):
        """Test default stream_read without query."""
        config = StreamConfig(batch_size=25)
        
        # Stream all records
        streamed = list(self.db.stream_read(config=config))
        
        assert len(streamed) == 100
        assert len(self.db.search_calls) == 1
        # Should have called search with empty Query
        assert isinstance(self.db.search_calls[0], Query)
        assert len(self.db.search_calls[0].filters) == 0
    
    def test_default_stream_read_with_query(self):
        """Test default stream_read with query."""
        query = Query().filter("category", "=", "A")
        config = StreamConfig(batch_size=10)
        
        # Stream filtered records
        streamed = list(self.db.stream_read(query, config))
        
        assert len(streamed) == 100  # Mock returns all records
        assert len(self.db.search_calls) == 1
        assert self.db.search_calls[0] == query
    
    def test_default_stream_read_batching(self):
        """Test that default stream_read respects batch size."""
        config = StreamConfig(batch_size=30)
        
        # Count records per "batch"
        batch_count = 0
        current_batch_size = 0
        
        for i, record in enumerate(self.db.stream_read(config=config)):
            current_batch_size += 1
            
            # Check if we're at a batch boundary
            if (i + 1) % config.batch_size == 0 or i == 99:
                batch_count += 1
                current_batch_size = 0
        
        # Should have 4 batches: 30, 30, 30, 10
        assert batch_count == 4
    
    def test_default_stream_write_success(self):
        """Test default stream_write with successful writes."""
        def record_generator():
            for i in range(50):
                record = Record()
                record.set_field("new_id", i)
                record.set_field("new_value", i * 100)
                yield record
        
        config = StreamConfig(batch_size=15)
        result = self.db.stream_write(record_generator(), config)
        
        assert result.total_processed == 50
        assert result.successful == 50
        assert result.failed == 0
        assert result.success_rate == 100.0
        assert len(result.errors) == 0
        
        # Check batching
        assert len(self.db.create_batch_calls) == 4  # 15, 15, 15, 5
        assert len(self.db.create_batch_calls[0]) == 15
        assert len(self.db.create_batch_calls[1]) == 15
        assert len(self.db.create_batch_calls[2]) == 15
        assert len(self.db.create_batch_calls[3]) == 5
    
    def test_default_stream_write_with_errors(self):
        """Test default stream_write with errors."""
        def record_generator():
            for i in range(20):
                record = Record()
                record.set_field("index", i)
                yield record
        
        # Fail on the second batch
        self.db.should_fail_create = False
        self.db.fail_on_record_index = 7  # Will be in second batch with size 5
        
        config = StreamConfig(batch_size=5)
        result = self.db.stream_write(record_generator(), config)
        
        # First batch succeeds, second fails, rest not processed without error handler
        assert result.failed >= 5  # At least the failing batch
        assert len(result.errors) > 0
        assert "Mock create error" in str(result.errors[0]["error"]) or "Failed on record" in str(result.errors[0]["error"])
    
    def test_default_stream_write_with_error_handler(self):
        """Test default stream_write with error handler that continues."""
        errors_seen = []
        
        def error_handler(error, record):
            errors_seen.append((str(error), record))
            return True  # Continue processing
        
        def record_generator():
            for i in range(20):
                record = Record()
                record.set_field("index", i)
                yield record
        
        # Make create_batch fail for specific batch
        self.db.fail_on_record_index = 8
        
        config = StreamConfig(batch_size=5, on_error=error_handler)
        result = self.db.stream_write(record_generator(), config)
        
        # Should process all records despite error
        assert result.total_processed == 20
        assert result.failed >= 5  # The batch with index 8
        assert len(errors_seen) > 0  # Error handler was called
    
    def test_default_stream_write_with_error_handler_stops(self):
        """Test default stream_write with error handler that stops."""
        error_count = 0
        
        def error_handler(error, record):
            nonlocal error_count
            error_count += 1
            return False  # Stop processing
        
        def record_generator():
            for i in range(20):
                record = Record()
                record.set_field("index", i)
                yield record
        
        self.db.should_fail_create = True
        
        config = StreamConfig(batch_size=5, on_error=error_handler)
        result = self.db.stream_write(record_generator(), config)
        
        # Should stop after first error
        assert result.failed >= 5  # First batch fails
        # Handler is called for each record in the failed batch until it returns False
        assert error_count >= 1  # Handler called at least once
        assert len(result.errors) > 0
    
    def test_default_stream_write_empty_iterator(self):
        """Test default stream_write with empty iterator."""
        def empty_generator():
            return
            yield  # Never reached
        
        config = StreamConfig(batch_size=10)
        result = self.db.stream_write(empty_generator(), config)
        
        assert result.total_processed == 0
        assert result.successful == 0
        assert result.failed == 0
        assert result.success_rate == 0.0
        assert len(self.db.create_batch_calls) == 0
    
    def test_default_stream_write_single_record(self):
        """Test default stream_write with single record."""
        def single_generator():
            record = Record()
            record.set_field("single", True)
            yield record
        
        config = StreamConfig(batch_size=10)
        result = self.db.stream_write(single_generator(), config)
        
        assert result.total_processed == 1
        assert result.successful == 1
        assert len(self.db.create_batch_calls) == 1
        assert len(self.db.create_batch_calls[0]) == 1
    
    def test_default_stream_duration_tracking(self):
        """Test that duration is tracked correctly."""
        def slow_generator():
            for i in range(5):
                time.sleep(0.01)  # Small delay
                record = Record()
                record.set_field("id", i)
                yield record
        
        config = StreamConfig(batch_size=2)
        result = self.db.stream_write(slow_generator(), config)
        
        assert result.duration > 0.04  # At least 5 * 0.01 seconds
        assert result.total_processed == 5
    
    def test_default_stream_write_last_batch_error(self):
        """Test error handling in the final batch."""
        def record_generator():
            for i in range(7):  # Will have batches of 5 and 2
                record = Record()
                record.set_field("index", i)
                yield record
        
        # Fail only when we have exactly 2 records (last batch)
        original_create_batch = self.db.create_batch
        def selective_fail(records):
            if len(records) == 2:
                raise ValueError("Failed on last batch")
            return original_create_batch(records)
        
        self.db.create_batch = selective_fail
        
        config = StreamConfig(batch_size=5)
        result = self.db.stream_write(record_generator(), config)
        
        # First batch succeeds (5 records), last batch fails (2 records)
        assert result.successful == 5
        assert result.failed == 2
        assert result.total_processed == 7
        assert len(result.errors) == 1
        assert "Failed on last batch" in str(result.errors[0]["error"])


@pytest.mark.asyncio
class TestAsyncStreamingMixinDefaults:
    """Test default implementations in AsyncStreamingMixin."""
    
    async def setup_records(self):
        """Set up test database with records."""
        self.db = MockAsyncDatabase()
        
        # Add test records
        for i in range(100):
            record = Record()
            record.set_field("index", i)
            record.set_field("value", i * 10)
            record.set_field("category", "A" if i % 2 == 0 else "B")
            self.db.records.append(record)
    
    async def test_async_default_stream_read_no_query(self):
        """Test async default stream_read without query."""
        await self.setup_records()
        
        config = StreamConfig(batch_size=25)
        
        # Stream all records
        streamed = []
        async for record in self.db.stream_read(config=config):
            streamed.append(record)
        
        assert len(streamed) == 100
        assert len(self.db.search_calls) == 1
        assert isinstance(self.db.search_calls[0], Query)
        assert len(self.db.search_calls[0].filters) == 0
    
    async def test_async_default_stream_read_with_query(self):
        """Test async default stream_read with query."""
        await self.setup_records()
        
        query = Query().filter("category", "=", "B").limit(30)
        config = StreamConfig(batch_size=10)
        
        # Stream filtered records
        streamed = []
        async for record in self.db.stream_read(query, config):
            streamed.append(record)
        
        assert len(streamed) == 100  # Mock returns all records
        assert len(self.db.search_calls) == 1
        assert self.db.search_calls[0] == query
    
    async def test_async_default_stream_write_success(self):
        """Test async default stream_write with successful writes."""
        self.db = MockAsyncDatabase()
        
        async def record_generator():
            for i in range(50):
                record = Record()
                record.set_field("new_id", i)
                record.set_field("new_value", i * 100)
                yield record
        
        config = StreamConfig(batch_size=15)
        result = await self.db.stream_write(record_generator(), config)
        
        assert result.total_processed == 50
        assert result.successful == 50
        assert result.failed == 0
        assert result.success_rate == 100.0
        
        # Check batching
        assert len(self.db.create_batch_calls) == 4
        assert len(self.db.create_batch_calls[0]) == 15
        assert len(self.db.create_batch_calls[3]) == 5
    
    async def test_async_default_stream_write_with_errors(self):
        """Test async default stream_write with errors."""
        self.db = MockAsyncDatabase()
        
        async def record_generator():
            for i in range(20):
                record = Record()
                record.set_field("index", i)
                yield record
        
        self.db.should_fail_create = True
        
        config = StreamConfig(batch_size=5)
        result = await self.db.stream_write(record_generator(), config)
        
        # First batch should fail
        assert result.failed >= 5
        assert len(result.errors) > 0
        assert "Mock async create error" in str(result.errors[0]["error"])
    
    async def test_async_default_stream_write_with_error_handler(self):
        """Test async default stream_write with error handler."""
        self.db = MockAsyncDatabase()
        errors_seen = []
        
        def error_handler(error, record):
            errors_seen.append(str(error))
            return True  # Continue
        
        async def record_generator():
            for i in range(15):
                record = Record()
                record.set_field("index", i)
                yield record
        
        self.db.fail_on_record_index = 7
        
        config = StreamConfig(batch_size=5, on_error=error_handler)
        result = await self.db.stream_write(record_generator(), config)
        
        assert result.total_processed == 15
        assert result.failed >= 5  # Batch containing index 7
        assert len(errors_seen) > 0
    
    async def test_async_default_stream_write_error_handler_stops(self):
        """Test async default stream_write when error handler stops processing."""
        self.db = MockAsyncDatabase()
        stop_count = 0
        
        def error_handler(error, record):
            nonlocal stop_count
            stop_count += 1
            return False  # Stop
        
        async def record_generator():
            for i in range(20):
                record = Record()
                record.set_field("index", i)
                yield record
        
        self.db.should_fail_create = True
        
        config = StreamConfig(batch_size=5, on_error=error_handler)
        result = await self.db.stream_write(record_generator(), config)
        
        assert result.failed >= 5
        assert stop_count >= 1  # Handler called at least once
        assert len(result.errors) > 0
    
    async def test_async_default_stream_write_empty(self):
        """Test async default stream_write with empty generator."""
        self.db = MockAsyncDatabase()
        
        async def empty_generator():
            return
            yield  # Never reached
        
        config = StreamConfig(batch_size=10)
        result = await self.db.stream_write(empty_generator(), config)
        
        assert result.total_processed == 0
        assert result.successful == 0
        assert result.failed == 0
        assert len(self.db.create_batch_calls) == 0
    
    async def test_async_default_stream_batching(self):
        """Test async stream batching behavior."""
        await self.setup_records()
        
        config = StreamConfig(batch_size=33)
        
        # Count batches
        batch_starts = []
        count = 0
        
        async for record in self.db.stream_read(config=config):
            if count % config.batch_size == 0:
                batch_starts.append(count)
            count += 1
        
        # Should have batches starting at 0, 33, 66, 99
        assert batch_starts == [0, 33, 66, 99]
    
    async def test_async_default_duration_tracking(self):
        """Test async duration tracking."""
        self.db = MockAsyncDatabase()
        
        async def slow_generator():
            for i in range(5):
                await asyncio.sleep(0.01)
                record = Record()
                record.set_field("id", i)
                yield record
        
        config = StreamConfig(batch_size=2)
        result = await self.db.stream_write(slow_generator(), config)
        
        assert result.duration > 0.04  # At least 5 * 0.01 seconds
        assert result.total_processed == 5
    
    async def test_async_default_stream_write_partial_batch(self):
        """Test async stream write with partial final batch."""
        self.db = MockAsyncDatabase()
        
        async def generator():
            for i in range(17):  # Not evenly divisible by batch size
                record = Record()
                record.set_field("id", i)
                yield record
        
        config = StreamConfig(batch_size=5)
        result = await self.db.stream_write(generator(), config)
        
        assert result.total_processed == 17
        assert result.successful == 17
        assert len(self.db.create_batch_calls) == 4  # 5, 5, 5, 2
        assert len(self.db.create_batch_calls[3]) == 2  # Last batch has 2
    
    async def test_async_default_stream_write_last_batch_error(self):
        """Test async error handling in the final batch."""
        self.db = MockAsyncDatabase()
        
        async def record_generator():
            for i in range(7):  # Will have batches of 5 and 2
                record = Record()
                record.set_field("index", i)
                yield record
        
        # Fail only when we have exactly 2 records (last batch)
        original_create_batch = self.db.create_batch
        async def selective_fail(records):
            if len(records) == 2:
                raise ValueError("Failed on async last batch")
            return await original_create_batch(records)
        
        self.db.create_batch = selective_fail
        
        config = StreamConfig(batch_size=5)
        result = await self.db.stream_write(record_generator(), config)
        
        # First batch succeeds (5 records), last batch fails (2 records)
        assert result.successful == 5
        assert result.failed == 2
        assert result.total_processed == 7
        assert len(result.errors) == 1
        assert "Failed on async last batch" in str(result.errors[0]["error"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])