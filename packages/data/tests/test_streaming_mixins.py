"""Tests for streaming mixin default implementations."""

import asyncio
import time
from typing import AsyncIterator, Iterator, List, Optional

import pytest

from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.query import Query
from dataknobs_data.records import Record
from dataknobs_data.streaming import (
    StreamConfig,
    StreamResult,
)


class TestStreamingMixinDefaults:
    """Test the default implementations in StreamingMixin."""
    
    def test_default_stream_read(self):
        """Test default stream_read implementation."""
        # Create a memory database with some records
        db = SyncMemoryDatabase()
        
        # Add some test records
        records = []
        for i in range(5):
            record = Record(data={"index": i, "value": f"test_{i}"})
            db.create(record)
            records.append(record)
        
        # Test streaming all records
        config = StreamConfig(batch_size=2)
        streamed = list(db.stream_read(config=config))
        
        assert len(streamed) == 5
        for i, record in enumerate(streamed):
            assert record.get_value("index") == i
    
    def test_default_stream_read_with_query(self):
        """Test default stream_read with a query."""
        db = SyncMemoryDatabase()
        
        # Add test records
        for i in range(10):
            record = Record(data={"index": i, "value": f"test_{i}"})
            db.create(record)
        
        # Create a query for even numbers
        query = Query().filter("index", ">=", 5)
        
        # Stream with query
        config = StreamConfig(batch_size=2)
        streamed = list(db.stream_read(query=query, config=config))
        
        assert len(streamed) == 5
        for record in streamed:
            assert record.get_value("index") >= 5
    
    def test_default_stream_write_success(self):
        """Test default stream_write implementation with successful writes."""
        db = SyncMemoryDatabase()
        
        # Create records to stream
        def record_generator():
            for i in range(5):
                yield Record(data={"index": i, "value": f"test_{i}"})
        
        # Stream write
        config = StreamConfig(batch_size=2)
        result = db.stream_write(record_generator(), config=config)
        
        assert result.total_processed == 5
        assert result.successful == 5
        assert result.failed == 0
        assert len(result.errors) == 0
        assert result.success_rate == 100.0
        
        # Verify records were created
        all_records = db.search(Query())
        assert len(all_records) == 5
    
    def test_default_stream_write_with_batch_failure(self):
        """Test stream_write with batch failure and individual retry."""
        # We'll need to create a custom database that can fail on specific batches
        class TestableMemoryDB(SyncMemoryDatabase):
            def __init__(self):
                super().__init__()
                self.batch_attempt = 0
                
            def create_batch(self, records: List[Record]) -> List[str]:
                self.batch_attempt += 1
                # Fail on first batch attempt to trigger fallback
                if self.batch_attempt == 1:
                    raise ValueError("Batch creation failed")
                return super().create_batch(records)
        
        db = TestableMemoryDB()
        
        # Create records
        def record_generator():
            for i in range(3):
                yield Record(data={"index": i})
        
        config = StreamConfig(batch_size=3)
        result = db.stream_write(record_generator(), config=config)
        
        # Should succeed via individual fallback
        assert result.total_processed == 3
        assert result.successful == 3
        assert result.failed == 0
    
    def test_default_stream_write_with_individual_failures(self):
        """Test stream_write with some individual record failures."""
        class TestableMemoryDB(SyncMemoryDatabase):
            def create(self, record: Record) -> str:
                # Fail on specific records
                if record.get_value("fail"):
                    raise ValueError(f"Record {record.get_value('index')} failed")
                return super().create(record)
            
            def create_batch(self, records: List[Record]) -> List[str]:
                # Always fail batch to force individual processing
                raise ValueError("Batch failed")
        
        db = TestableMemoryDB()
        
        # Create mixed records
        def record_generator():
            yield Record(data={"index": 0, "fail": False})
            yield Record(data={"index": 1, "fail": True})  # Will fail
            yield Record(data={"index": 2, "fail": False})
            yield Record(data={"index": 3, "fail": True})  # Will fail
            yield Record(data={"index": 4, "fail": False})
        
        # Use error handler that continues on error
        config = StreamConfig(
            batch_size=2,
            on_error=lambda error, record: True  # Continue on error
        )
        result = db.stream_write(record_generator(), config=config)
        
        assert result.total_processed == 5
        assert result.successful == 3
        assert result.failed == 2
        assert len(result.errors) == 2
    
    def test_default_stream_write_stop_on_error(self):
        """Test stream_write stops on error when no error handler."""
        class TestableMemoryDB(SyncMemoryDatabase):
            def create(self, record: Record) -> str:
                if record.get_value("index") == 1:
                    raise ValueError("Stop here")
                return super().create(record)
            
            def create_batch(self, records: List[Record]) -> List[str]:
                # Force individual processing
                raise ValueError("Batch failed")
        
        db = TestableMemoryDB()
        
        def record_generator():
            for i in range(5):
                yield Record(data={"index": i})
        
        # No error handler - should stop on first error
        config = StreamConfig(batch_size=2)
        result = db.stream_write(record_generator(), config=config)
        
        # Should process 2 records (0 succeeds, 1 fails and stops)
        assert result.total_processed == 2
        assert result.successful == 1
        assert result.failed == 1


class TestAsyncStreamingMixinDefaults:
    """Test the default implementations in AsyncStreamingMixin."""
    
    @pytest.mark.asyncio
    async def test_async_default_stream_read(self):
        """Test async default stream_read implementation."""
        db = AsyncMemoryDatabase()
        
        # Add test records
        for i in range(5):
            record = Record(data={"index": i, "value": f"test_{i}"})
            await db.create(record)
        
        # Stream all records
        config = StreamConfig(batch_size=2)
        streamed = []
        async for record in db.stream_read(config=config):
            streamed.append(record)
        
        assert len(streamed) == 5
        for i, record in enumerate(streamed):
            assert record.get_value("index") == i
    
    @pytest.mark.asyncio
    async def test_async_default_stream_write_success(self):
        """Test async default stream_write implementation."""
        db = AsyncMemoryDatabase()
        
        # Create async record generator
        async def record_generator():
            for i in range(5):
                yield Record(data={"index": i, "value": f"test_{i}"})
        
        # Stream write
        config = StreamConfig(batch_size=2)
        result = await db.stream_write(record_generator(), config=config)
        
        assert result.total_processed == 5
        assert result.successful == 5
        assert result.failed == 0
        
        # Verify records were created
        all_records = await db.search(Query())
        assert len(all_records) == 5
    
    @pytest.mark.asyncio
    async def test_async_stream_write_with_batch_failure(self):
        """Test async stream_write with batch failure."""
        class TestableAsyncMemoryDB(AsyncMemoryDatabase):
            def __init__(self):
                super().__init__()
                self.batch_attempt = 0
                
            async def create_batch(self, records: List[Record]) -> List[str]:
                self.batch_attempt += 1
                if self.batch_attempt == 1:
                    raise ValueError("Batch failed")
                return await super().create_batch(records)
        
        db = TestableAsyncMemoryDB()
        
        async def record_generator():
            for i in range(3):
                yield Record(data={"index": i})
        
        config = StreamConfig(batch_size=3)
        result = await db.stream_write(record_generator(), config=config)
        
        # Should succeed via fallback
        assert result.total_processed == 3
        assert result.successful == 3
        assert result.failed == 0
    
    @pytest.mark.asyncio
    async def test_async_stream_write_with_individual_failures(self):
        """Test async stream_write with individual failures."""
        class TestableAsyncMemoryDB(AsyncMemoryDatabase):
            async def create(self, record: Record) -> str:
                if record.get_value("fail"):
                    raise ValueError(f"Record failed")
                return await super().create(record)
            
            async def create_batch(self, records: List[Record]) -> List[str]:
                # Force individual processing
                raise ValueError("Batch failed")
        
        db = TestableAsyncMemoryDB()
        
        async def record_generator():
            yield Record(data={"index": 0, "fail": False})
            yield Record(data={"index": 1, "fail": True})
            yield Record(data={"index": 2, "fail": False})
        
        config = StreamConfig(
            batch_size=2,
            on_error=lambda error, record: True  # Continue
        )
        result = await db.stream_write(record_generator(), config=config)
        
        assert result.total_processed == 3
        assert result.successful == 2
        assert result.failed == 1


class TestStreamResultMerge:
    """Test StreamResult merge functionality."""
    
    def test_merge_results(self):
        """Test merging stream results."""
        result1 = StreamResult()
        result1.total_processed = 10
        result1.successful = 8
        result1.failed = 2
        result1.duration = 1.5
        result1.add_error("id1", ValueError("Error 1"))
        
        result2 = StreamResult()
        result2.total_processed = 5
        result2.successful = 4
        result2.failed = 1
        result2.duration = 0.5
        result2.add_error("id2", ValueError("Error 2"))
        
        # Merge result2 into result1
        result1.merge(result2)
        
        assert result1.total_processed == 15
        assert result1.successful == 12
        assert result1.failed == 3
        assert result1.duration == 2.0
        assert len(result1.errors) == 2
        assert result1.success_rate == 80.0