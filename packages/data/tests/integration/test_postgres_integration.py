"""Integration tests for PostgreSQL backend with real database."""

import asyncio
import concurrent.futures
import os
import uuid

import pytest

from dataknobs_data import AsyncDatabase, Query, Record, SyncDatabase
from dataknobs_data.query import Filter, Operator, SortOrder

# pytestmark = pytest.mark.integration

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="PostgreSQL tests require TEST_POSTGRES=true and a running PostgreSQL instance"
)


class TestPostgresIntegration:
    """Integration tests for PostgreSQL backend."""

    def test_connection_and_table_creation(self, postgres_test_db):
        """Test that we can connect and create tables."""
        db = SyncDatabase.from_backend("postgres", postgres_test_db)
        
        # The table should be created automatically
        # Try to create a record to verify
        record = Record({"test": "data"})
        id = db.create(record)
        
        assert id is not None
        assert db.exists(id)
        
        # Cleanup
        db.delete(id)
        db.close()

    def test_full_crud_cycle(self, postgres_test_db):
        """Test complete CRUD operations with real database."""
        db = SyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Create
        record = Record({
            "name": "Test User",
            "age": 30,
            "active": True,
            "balance": 1234.56,
            "tags": ["test", "integration"],
        })
        id = db.create(record)
        assert id is not None
        
        # Read
        retrieved = db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "Test User"
        assert retrieved.get_value("age") == 30
        assert retrieved.get_value("active") is True
        assert retrieved.get_value("balance") == 1234.56
        assert retrieved.get_value("tags") == ["test", "integration"]
        
        # Update
        updated_record = Record({
            "name": "Updated User",
            "age": 31,
            "active": False,
            "balance": 5678.90,
            "tags": ["updated"],
        })
        success = db.update(id, updated_record)
        assert success is True
        
        retrieved = db.read(id)
        assert retrieved.get_value("name") == "Updated User"
        assert retrieved.get_value("age") == 31
        assert retrieved.get_value("active") is False
        
        # Delete
        success = db.delete(id)
        assert success is True
        assert db.read(id) is None
        
        db.close()

    def test_batch_operations_with_sample_data(self, postgres_test_db, sample_records):
        """Test batch operations with sample dataset."""
        db = SyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Create batch
        ids = db.create_batch(sample_records)
        assert len(ids) == len(sample_records)
        
        # Verify all created
        for id in ids:
            assert db.exists(id)
        
        # Read batch
        retrieved = db.read_batch(ids)
        assert len(retrieved) == len(ids)
        assert all(r is not None for r in retrieved)
        
        # Verify data integrity
        for i, record in enumerate(retrieved):
            original = sample_records[i]
            assert record.get_value("name") == original.get_value("name")
            assert record.get_value("age") == original.get_value("age")
            assert record.metadata == original.metadata
        
        # Delete batch
        results = db.delete_batch(ids)
        assert all(results)
        
        # Verify all deleted
        retrieved = db.read_batch(ids)
        assert all(r is None for r in retrieved)
        
        db.close()

    def test_complex_queries(self, postgres_test_db, sample_records):
        """Test complex query operations."""
        db = SyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Insert sample data
        ids = db.create_batch(sample_records)
        
        # Test 1: Filter by department
        query = Query().filter("department", Operator.EQ, "Engineering")
        results = db.search(query)
        assert len(results) == 3
        names = {r.get_value("name") for r in results}
        assert names == {"Alice Johnson", "Charlie Brown", "Eve Anderson"}
        
        # Test 2: Range query on salary
        query = Query().filter("salary", Operator.GT, 100000)
        results = db.search(query)
        assert len(results) == 2
        names = {r.get_value("name") for r in results}
        assert names == {"Charlie Brown", "Eve Anderson"}
        
        # Test 3: Combined filters
        query = (Query()
            .filter("department", Operator.EQ, "Engineering")
            .filter("active", Operator.EQ, True)
            .filter("salary", Operator.GTE, 100000))
        results = db.search(query)
        assert len(results) == 1
        assert results[0].get_value("name") == "Eve Anderson"
        
        # Test 4: LIKE pattern matching
        query = Query().filter("email", Operator.LIKE, "%@example.com")
        results = db.search(query)
        assert len(results) == 5  # All have @example.com
        
        # Test 5: IN operator
        query = Query().filter("department", Operator.IN, ["Engineering", "HR"])
        results = db.search(query)
        assert len(results) == 4
        
        # Test 6: NOT_IN operator
        query = Query().filter("department", Operator.NOT_IN, ["Engineering", "HR"])
        results = db.search(query)
        assert len(results) == 1
        assert results[0].get_value("department") == "Marketing"
        
        # Cleanup
        db.delete_batch(ids)
        db.close()

    def test_sorting_and_pagination(self, postgres_test_db, sample_records):
        """Test sorting and pagination features."""
        db = SyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Insert sample data
        ids = db.create_batch(sample_records)
        
        # Test 1: Sort by age ascending
        query = Query().sort("age", SortOrder.ASC)
        results = db.search(query)
        ages = [r.get_value("age") for r in results]
        assert ages == sorted(ages)
        
        # Test 2: Sort by salary descending
        query = Query().sort("salary", SortOrder.DESC)
        results = db.search(query)
        salaries = [r.get_value("salary") for r in results]
        assert salaries == sorted(salaries, reverse=True)
        
        # Test 3: Pagination with limit
        query = Query().sort("name", SortOrder.ASC).limit(3)
        results = db.search(query)
        assert len(results) == 3
        names = [r.get_value("name") for r in results]
        assert names == ["Alice Johnson", "Bob Smith", "Charlie Brown"]
        
        # Test 4: Pagination with offset
        query = Query().sort("name", SortOrder.ASC).offset(2).limit(2)
        results = db.search(query)
        assert len(results) == 2
        names = [r.get_value("name") for r in results]
        assert names == ["Charlie Brown", "Diana Prince"]
        
        # Test 5: Multiple sort criteria
        query = Query().sort("department", SortOrder.ASC).sort("salary", SortOrder.DESC)
        results = db.search(query)
        
        # Verify Engineering department records are sorted by salary descending
        eng_records = [r for r in results if r.get_value("department") == "Engineering"]
        eng_salaries = [r.get_value("salary") for r in eng_records]
        assert eng_salaries == sorted(eng_salaries, reverse=True)
        
        # Cleanup
        db.delete_batch(ids)
        db.close()

    def test_metadata_persistence(self, postgres_test_db):
        """Test that metadata is properly stored and retrieved."""
        db = SyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Create record with metadata
        record = Record(
            data={"name": "Test", "value": 123},
            metadata={
                "created_by": "integration_test",
                "version": 2,
                "tags": ["test", "postgres"],
                "nested": {"key": "value"},
            }
        )
        id = db.create(record)
        
        # Retrieve and verify metadata
        retrieved = db.read(id)
        assert retrieved.metadata == record.metadata
        assert retrieved.metadata["created_by"] == "integration_test"
        assert retrieved.metadata["version"] == 2
        assert retrieved.metadata["tags"] == ["test", "postgres"]
        assert retrieved.metadata["nested"]["key"] == "value"
        
        # Update with new metadata
        updated_record = Record(
            data={"name": "Updated", "value": 456},
            metadata={"version": 3, "updated": True}
        )
        db.update(id, updated_record)
        
        retrieved = db.read(id)
        assert retrieved.metadata["version"] == 3
        assert retrieved.metadata["updated"] is True
        
        # Cleanup
        db.delete(id)
        db.close()

    def test_concurrent_operations(self, postgres_test_db):
        """Test concurrent database operations."""
        db = SyncDatabase.from_backend("postgres", postgres_test_db)
        
        def create_record(index):
            record = Record({"index": index, "data": f"concurrent_{index}"})
            return db.create(record)
        
        def read_record(id):
            return db.read(id)
        
        # Create records concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_record, i) for i in range(10)]
            ids = [f.result() for f in futures]
        
        assert len(ids) == 10
        assert all(id is not None for id in ids)
        
        # Read records concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(read_record, id) for id in ids]
            records = [f.result() for f in futures]
        
        assert len(records) == 10
        assert all(r is not None for r in records)
        
        # Verify data integrity
        indices = {r.get_value("index") for r in records}
        assert indices == set(range(10))
        
        # Cleanup
        db.delete_batch(ids)
        db.close()

    def test_transaction_isolation(self, postgres_test_db):
        """Test transaction isolation and consistency."""
        db1 = SyncDatabase.from_backend("postgres", postgres_test_db)
        db2 = SyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Create a record in db1
        record = Record({"name": "Transaction Test", "value": 100})
        id = db1.create(record)
        
        # Should be immediately visible in db2 (after commit)
        retrieved = db2.read(id)
        assert retrieved is not None
        assert retrieved.get_value("value") == 100
        
        # Update in db1
        updated = Record({"name": "Updated Transaction", "value": 200})
        db1.update(id, updated)
        
        # Should see update in db2
        retrieved = db2.read(id)
        assert retrieved.get_value("value") == 200
        
        # Cleanup
        db1.delete(id)
        db1.close()
        db2.close()

    def test_special_characters_and_unicode(self, postgres_test_db):
        """Test handling of special characters and Unicode."""
        db = SyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Create record with special characters
        record = Record({
            "name": "Test's \"Special\" Name",
            "description": "Line 1\nLine 2\tTabbed",
            "unicode": "Hello 世界 🌍 مرحبا мир",
            "symbols": "!@#$%^&*(){}[]|\\:;\"'<>,.?/",
            "json_field": {"nested": {"key": "value with 'quotes'"}},
        })
        id = db.create(record)
        
        # Retrieve and verify
        retrieved = db.read(id)
        assert retrieved.get_value("name") == "Test's \"Special\" Name"
        assert retrieved.get_value("description") == "Line 1\nLine 2\tTabbed"
        assert retrieved.get_value("unicode") == "Hello 世界 🌍 مرحبا мир"
        assert retrieved.get_value("symbols") == "!@#$%^&*(){}[]|\\:;\"'<>,.?/"
        assert retrieved.get_value("json_field")["nested"]["key"] == "value with 'quotes'"
        
        # Search with special characters
        query = Query().filter("name", Operator.LIKE, "%Special%")
        results = db.search(query)
        assert len(results) == 1
        
        # Cleanup
        db.delete(id)
        db.close()

    def test_count_operations(self, postgres_test_db, sample_records):
        """Test count operations."""
        db = SyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Initially empty
        assert db.count() == 0
        
        # Insert sample data
        ids = db.create_batch(sample_records)
        
        # Count all
        assert db.count() == len(sample_records)
        
        # Count with query
        query = Query().filter("department", Operator.EQ, "Engineering")
        assert db.count(query) == 3
        
        query = Query().filter("active", Operator.EQ, True)
        assert db.count(query) == 4
        
        # Cleanup
        deleted = db.clear()
        assert deleted == len(sample_records)
        assert db.count() == 0
        
        db.close()


@pytest.mark.asyncio
class TestPostgresAsyncIntegration:
    """Async integration tests for PostgreSQL backend."""

    async def test_async_crud_operations(self, postgres_test_db):
        """Test async CRUD operations."""
        db = await AsyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Create
        record = Record({
            "name": "Async Test",
            "value": 42,
            "async": True,
        })
        id = await db.create(record)
        assert id is not None
        
        # Read
        retrieved = await db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "Async Test"
        assert retrieved.get_value("async") is True
        
        # Update
        updated = Record({
            "name": "Async Updated",
            "value": 84,
            "async": False,
        })
        success = await db.update(id, updated)
        assert success is True
        
        retrieved = await db.read(id)
        assert retrieved.get_value("value") == 84
        
        # Delete
        success = await db.delete(id)
        assert success is True
        assert await db.read(id) is None
        
        await db.close()

    async def test_async_batch_operations(self, postgres_test_db, sample_records):
        """Test async batch operations."""
        db = await AsyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Create batch
        ids = await db.create_batch(sample_records)
        assert len(ids) == len(sample_records)
        
        # Read batch
        retrieved = await db.read_batch(ids)
        assert all(r is not None for r in retrieved)
        
        # Delete batch
        results = await db.delete_batch(ids)
        assert all(results)
        
        await db.close()

    async def test_async_concurrent_operations(self, postgres_test_db):
        """Test concurrent async operations."""
        db = await AsyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Create multiple records concurrently
        tasks = []
        for i in range(10):
            record = Record({"index": i, "type": "async_concurrent"})
            tasks.append(db.create(record))
        
        ids = await asyncio.gather(*tasks)
        assert len(ids) == 10
        
        # Read all concurrently
        tasks = [db.read(id) for id in ids]
        records = await asyncio.gather(*tasks)
        assert all(r is not None for r in records)
        
        # Verify data
        indices = {r.get_value("index") for r in records}
        assert indices == set(range(10))
        
        # Cleanup
        await db.delete_batch(ids)
        await db.close()

    async def test_async_search_operations(self, postgres_test_db, sample_records):
        """Test async search operations."""
        db = await AsyncDatabase.from_backend("postgres", postgres_test_db)
        
        # Insert data
        ids = await db.create_batch(sample_records)
        
        # Search
        query = Query().filter("department", Operator.EQ, "Engineering")
        results = await db.search(query)
        assert len(results) == 3
        
        # Count
        count = await db.count(query)
        assert count == 3
        
        # Cleanup
        await db.clear()
        await db.close()
