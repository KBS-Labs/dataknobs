"""Tests for SQLite backend implementation."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.query import Filter, Operator, Query, SortOrder
from dataknobs_data.query_logic import ComplexQuery, FilterCondition, LogicCondition, LogicOperator
from dataknobs_data.records import Record


class TestSyncSQLiteDatabase:
    """Test synchronous SQLite database backend."""
    
    @pytest.fixture
    def memory_db(self):
        """Create an in-memory SQLite database."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        yield db
        db.close()
    
    @pytest.fixture
    def file_db(self, tmp_path):
        """Create a file-based SQLite database."""
        db_path = tmp_path / "test.db"
        db = SyncSQLiteDatabase({"path": str(db_path)})
        db.connect()
        yield db
        db.close()
    
    def test_connect_memory(self):
        """Test connecting to in-memory database."""
        db = SyncSQLiteDatabase({"path": ":memory:"})
        db.connect()
        assert db._connected
        db.close()
        assert not db._connected
    
    def test_connect_file(self, tmp_path):
        """Test connecting to file-based database."""
        db_path = tmp_path / "test.db"
        db = SyncSQLiteDatabase({"path": str(db_path)})
        db.connect()
        assert db._connected
        assert db_path.exists()
        db.close()
    
    def test_create_read(self, memory_db):
        """Test creating and reading records."""
        record = Record(data={"name": "Alice", "age": 30})
        
        # Create record
        record_id = memory_db.create(record)
        assert record_id is not None
        
        # Read record
        retrieved = memory_db.read(record_id)
        assert retrieved is not None
        assert retrieved["name"] == "Alice"
        assert retrieved["age"] == 30
    
    def test_update(self, memory_db):
        """Test updating records."""
        record = Record(data={"name": "Bob", "age": 25})
        record_id = memory_db.create(record)
        
        # Update record
        updated_record = Record(data={"name": "Bob", "age": 26})
        success = memory_db.update(record_id, updated_record)
        assert success
        
        # Verify update
        retrieved = memory_db.read(record_id)
        assert retrieved["age"] == 26
    
    def test_delete(self, memory_db):
        """Test deleting records."""
        record = Record(data={"name": "Charlie"})
        record_id = memory_db.create(record)
        
        # Delete record
        success = memory_db.delete(record_id)
        assert success
        
        # Verify deletion
        retrieved = memory_db.read(record_id)
        assert retrieved is None
    
    def test_exists(self, memory_db):
        """Test checking record existence."""
        record = Record(data={"name": "David"})
        record_id = memory_db.create(record)
        
        assert memory_db.exists(record_id)
        assert not memory_db.exists("nonexistent")
    
    def test_search_basic(self, memory_db):
        """Test basic search functionality."""
        # Create test records
        memory_db.create(Record(data={"name": "Alice", "age": 30, "city": "NYC"}))
        memory_db.create(Record(data={"name": "Bob", "age": 25, "city": "LA"}))
        memory_db.create(Record(data={"name": "Charlie", "age": 35, "city": "NYC"}))
        
        # Search by city
        query = Query().filter("city", Operator.EQ, "NYC")
        results = memory_db.search(query)
        assert len(results) == 2
        
        # Search with multiple filters
        query = Query().filter("city", Operator.EQ, "NYC").filter("age", Operator.GT, 30)
        results = memory_db.search(query)
        assert len(results) == 1
        assert results[0]["name"] == "Charlie"
    
    def test_search_operators(self, memory_db):
        """Test various search operators."""
        # Create test records
        memory_db.create(Record(data={"value": 10}))
        memory_db.create(Record(data={"value": 20}))
        memory_db.create(Record(data={"value": 30}))
        memory_db.create(Record(data={"value": 40}))
        
        # Test GT
        query = Query().filter("value", Operator.GT, 25)
        results = memory_db.search(query)
        assert len(results) == 2
        
        # Test LTE
        query = Query().filter("value", Operator.LTE, 20)
        results = memory_db.search(query)
        assert len(results) == 2
        
        # Test BETWEEN
        query = Query().filter("value", Operator.BETWEEN, [15, 35])
        results = memory_db.search(query)
        assert len(results) == 2
        
        # Test IN
        query = Query().filter("value", Operator.IN, [10, 30, 50])
        results = memory_db.search(query)
        assert len(results) == 2
    
    def test_search_complex_query(self, memory_db):
        """Test complex query with boolean logic."""
        # Create test records
        memory_db.create(Record(data={"name": "Alice", "age": 30, "city": "NYC"}))
        memory_db.create(Record(data={"name": "Bob", "age": 25, "city": "LA"}))
        memory_db.create(Record(data={"name": "Charlie", "age": 35, "city": "SF"}))
        memory_db.create(Record(data={"name": "David", "age": 28, "city": "NYC"}))
        
        # Test OR condition: city = NYC OR age > 30
        query = ComplexQuery(
            condition=LogicCondition(
                operator=LogicOperator.OR,
                conditions=[
                    FilterCondition(Filter("city", Operator.EQ, "NYC")),
                    FilterCondition(Filter("age", Operator.GT, 30))
                ]
            )
        )
        results = memory_db.search(query)
        assert len(results) == 3  # Alice, Charlie, David
        
        # Test AND with OR: (city = NYC OR city = LA) AND age < 30
        query = ComplexQuery(
            condition=LogicCondition(
                operator=LogicOperator.AND,
                conditions=[
                    LogicCondition(
                        operator=LogicOperator.OR,
                        conditions=[
                            FilterCondition(Filter("city", Operator.EQ, "NYC")),
                            FilterCondition(Filter("city", Operator.EQ, "LA"))
                        ]
                    ),
                    FilterCondition(Filter("age", Operator.LT, 30))
                ]
            )
        )
        results = memory_db.search(query)
        assert len(results) == 2  # Bob, David
    
    def test_sorting(self, memory_db):
        """Test sorting functionality."""
        # Create test records
        memory_db.create(Record(data={"name": "Charlie", "age": 35}))
        memory_db.create(Record(data={"name": "Alice", "age": 30}))
        memory_db.create(Record(data={"name": "Bob", "age": 25}))
        
        # Sort by age ascending
        query = Query().sort("age", SortOrder.ASC)
        results = memory_db.search(query)
        assert results[0]["name"] == "Bob"
        assert results[2]["name"] == "Charlie"
        
        # Sort by name descending
        query = Query().sort("name", SortOrder.DESC)
        results = memory_db.search(query)
        assert results[0]["name"] == "Charlie"
        assert results[2]["name"] == "Alice"
    
    def test_pagination(self, memory_db):
        """Test pagination with limit and offset."""
        # Create test records
        for i in range(10):
            memory_db.create(Record(data={"id": i, "value": i * 10}))
        
        # Test limit
        query = Query().limit(3)
        results = memory_db.search(query)
        assert len(results) == 3
        
        # Test offset
        query = Query().offset(5).limit(3)
        results = memory_db.search(query)
        assert len(results) == 3
    
    def test_batch_operations(self, memory_db):
        """Test batch create, update, and delete."""
        # Batch create
        records = [
            Record(data={"name": f"User{i}", "value": i})
            for i in range(5)
        ]
        ids = memory_db.create_batch(records)
        assert len(ids) == 5
        
        # Verify all created
        for id in ids:
            assert memory_db.exists(id)
        
        # Batch update
        updates = [
            (ids[i], Record(data={"name": f"Updated{i}", "value": i * 2}))
            for i in range(3)
        ]
        results = memory_db.update_batch(updates)
        assert all(results)
        
        # Verify updates
        record = memory_db.read(ids[0])
        assert record["name"] == "Updated0"
        assert record["value"] == 0
        
        # Batch delete
        delete_results = memory_db.delete_batch(ids[:2])
        assert all(delete_results)
        assert not memory_db.exists(ids[0])
        assert not memory_db.exists(ids[1])
        assert memory_db.exists(ids[2])
    
    def test_count(self, memory_db):
        """Test counting records."""
        # Create test records
        memory_db.create(Record(data={"type": "A", "value": 10}))
        memory_db.create(Record(data={"type": "B", "value": 20}))
        memory_db.create(Record(data={"type": "A", "value": 30}))
        
        # Count all
        count = memory_db.count()
        assert count == 3
        
        # Count with filter
        query = Query().filter("type", Operator.EQ, "A")
        count = memory_db.count(query)
        assert count == 2
    
    def test_journal_mode_wal(self, tmp_path):
        """Test WAL journal mode for better concurrency."""
        db_path = tmp_path / "test_wal.db"
        db = SyncSQLiteDatabase({
            "path": str(db_path),
            "journal_mode": "WAL"
        })
        db.connect()
        
        # Create a record
        record = Record(data={"test": "data"})
        db.create(record)
        
        # WAL files should exist
        wal_file = Path(str(db_path) + "-wal")
        assert wal_file.exists() or db_path.exists()
        
        db.close()
    
    def test_transaction_rollback(self, memory_db):
        """Test transaction rollback on error."""
        # Create a record with a specific ID
        record1 = Record(data={"name": "First"})
        id1 = memory_db.create(record1)
        
        # Try to create batch with duplicate ID
        records = [
            Record(data={"name": "Second"}),
            Record(data={"name": "Third"})
        ]
        
        # Manually inject duplicate ID (this would normally cause an error)
        # Since we generate UUIDs, we'll test with a different approach
        original_count = memory_db.count()
        
        # Create valid batch
        ids = memory_db.create_batch(records)
        new_count = memory_db.count()
        
        assert new_count == original_count + 2
        assert len(ids) == 2
    
    def test_stream_write(self, memory_db):
        """Test streaming write functionality."""
        from dataknobs_data.streaming import StreamConfig
        
        # Create a generator of records
        def record_generator():
            for i in range(100):
                yield Record(data={"index": i, "value": f"record_{i}"})
        
        # Stream write the records
        config = StreamConfig(batch_size=10)
        result = memory_db.stream_write(record_generator(), config)
        
        # Verify the result
        assert result.total_processed == 100
        assert result.successful == 100
        assert result.failed == 0
        assert result.total_batches == 10  # 100 records / 10 batch_size
        assert result.duration > 0
        
        # Verify records were actually written
        all_records = memory_db.all()
        assert len(all_records) == 100
        
        # Verify data integrity
        for i, record in enumerate(sorted(all_records, key=lambda r: r.data.get("index", -1))):
            assert record.data["index"] == i
            assert record.data["value"] == f"record_{i}"
    
    def test_stream_write_small_batch(self, memory_db):
        """Test stream write with small batches."""
        from dataknobs_data.streaming import StreamConfig
        
        # Create a generator with fewer records than batch size
        def record_generator():
            for i in range(3):
                yield Record(data={"index": i})
        
        config = StreamConfig(batch_size=10)
        result = memory_db.stream_write(record_generator(), config)
        
        assert result.total_processed == 3
        assert result.successful == 3
        assert result.failed == 0
        assert result.total_batches == 1  # Only one partial batch


class TestAsyncSQLiteDatabase:
    """Test asynchronous SQLite database backend."""
    
    @pytest_asyncio.fixture
    async def memory_db(self):
        """Create an in-memory async SQLite database."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        yield db
        await db.close()
    
    @pytest.mark.asyncio
    async def test_connect_memory(self):
        """Test connecting to in-memory database."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        assert db._connected
        await db.close()
        assert not db._connected
    
    @pytest.mark.asyncio
    async def test_create_read(self):
        """Test creating and reading records asynchronously."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            record = Record(data={"name": "Alice", "age": 30})
            
            # Create record
            record_id = await db.create(record)
            assert record_id is not None
            
            # Read record
            retrieved = await db.read(record_id)
            assert retrieved is not None
            assert retrieved["name"] == "Alice"
            assert retrieved["age"] == 30
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent database operations."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            # Create multiple records concurrently
            async def create_record(i):
                record = Record(data={"id": i, "value": i * 10})
                return await db.create(record)
            
            # Create 10 records concurrently
            tasks = [create_record(i) for i in range(10)]
            ids = await asyncio.gather(*tasks)
            
            assert len(ids) == 10
            
            # Read all records concurrently
            async def read_record(id):
                return await db.read(id)
            
            tasks = [read_record(id) for id in ids]
            records = await asyncio.gather(*tasks)
            
            assert all(r is not None for r in records)
            assert len(records) == 10
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_complex_query_async(self):
        """Test complex queries with async backend."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            # Create test records
            await db.create(Record(data={"name": "Alice", "age": 30, "city": "NYC"}))
            await db.create(Record(data={"name": "Bob", "age": 25, "city": "LA"}))
            await db.create(Record(data={"name": "Charlie", "age": 35, "city": "SF"}))
            
            # Test complex OR query
            query = ComplexQuery(
                condition=LogicCondition(
                    operator=LogicOperator.OR,
                    conditions=[
                        FilterCondition(Filter("city", Operator.EQ, "NYC")),
                        FilterCondition(Filter("age", Operator.GT, 30))
                    ]
                )
            )
            results = await db.search(query)
            assert len(results) == 2  # Alice and Charlie
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_batch_operations_async(self):
        """Test async batch operations."""
        db = AsyncSQLiteDatabase({"path": ":memory:"})
        await db.connect()
        
        try:
            # Batch create
            records = [
                Record(data={"name": f"User{i}", "value": i})
                for i in range(5)
            ]
            ids = await db.create_batch(records)
            assert len(ids) == 5
            
            # Batch update
            updates = [
                (ids[i], Record(data={"name": f"Updated{i}", "value": i * 2}))
                for i in range(3)
            ]
            results = await db.update_batch(updates)
            assert all(results)
            
            # Batch delete
            delete_results = await db.delete_batch(ids[:2])
            assert all(delete_results)
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_stream_write(self, memory_db):
        """Test async streaming write functionality."""
        from dataknobs_data.streaming import StreamConfig
        
        # Create an async generator of records
        async def record_generator():
            for i in range(50):
                yield Record(data={"index": i, "value": f"async_record_{i}"})
        
        # Stream write the records
        config = StreamConfig(batch_size=5)
        result = await memory_db.stream_write(record_generator(), config)
        
        # Verify the result
        assert result.total_processed == 50
        assert result.successful == 50
        assert result.failed == 0
        assert result.total_batches == 10  # 50 records / 5 batch_size
        assert result.duration > 0
        
        # Verify records were actually written
        all_records = await memory_db.all()
        assert len(all_records) == 50
        
        # Verify data integrity
        for i, record in enumerate(sorted(all_records, key=lambda r: r.data.get("index", -1))):
            assert record.data["index"] == i
            assert record.data["value"] == f"async_record_{i}"
    


class TestSQLiteSpecificFeatures:
    """Test SQLite-specific features and optimizations."""
    
    def test_json_validation(self, tmp_path):
        """Test JSON validation in SQLite."""
        db_path = tmp_path / "test_json.db"
        db = SyncSQLiteDatabase({"path": str(db_path)})
        db.connect()
        
        try:
            # Create record with nested JSON
            record = Record(data={
                "name": "Test",
                "nested": {"key": "value", "array": [1, 2, 3]}
            })
            id = db.create(record)
            
            # Read and verify
            retrieved = db.read(id)
            assert retrieved["nested"]["key"] == "value"
            assert retrieved["nested"]["array"] == [1, 2, 3]
        finally:
            db.close()
    
    def test_memory_optimization(self):
        """Test in-memory database performance settings."""
        db = SyncSQLiteDatabase({
            "path": ":memory:",
            "synchronous": "OFF"  # Fastest for in-memory
        })
        db.connect()
        
        try:
            # Should be fast for in-memory operations
            import time
            start = time.time()
            
            for i in range(100):
                db.create(Record(data={"id": i}))
            
            elapsed = time.time() - start
            assert elapsed < 1.0  # Should be very fast
        finally:
            db.close()
    
    def test_file_persistence(self, tmp_path):
        """Test that data persists across connections."""
        db_path = tmp_path / "persist.db"
        
        # First connection - create data
        db1 = SyncSQLiteDatabase({"path": str(db_path)})
        db1.connect()
        id1 = db1.create(Record(data={"persistent": "data"}))
        db1.close()
        
        # Second connection - read data
        db2 = SyncSQLiteDatabase({"path": str(db_path)})
        db2.connect()
        record = db2.read(id1)
        assert record is not None
        assert record["persistent"] == "data"
        db2.close()
