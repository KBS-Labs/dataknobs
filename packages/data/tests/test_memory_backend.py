"""Tests for Memory backend implementation."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest_plugins = ("pytest_asyncio",)

from dataknobs_data import AsyncDatabase, Query, Record, SyncDatabase


class TestSyncMemoryDatabase:
    """Test synchronous memory database."""

    def test_create_database(self):
        """Test creating a memory database."""
        db = SyncDatabase.from_backend("memory")
        assert db is not None
        assert hasattr(db, "_storage")
        assert hasattr(db, "_lock")

    def test_crud_operations(self):
        """Test basic CRUD operations."""
        db = SyncDatabase.from_backend("memory")

        # Create
        record = Record({"name": "Test", "value": 42})
        id = db.create(record)
        assert id is not None
        assert isinstance(id, str)

        # Read
        retrieved = db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "Test"
        assert retrieved.get_value("value") == 42

        # Update
        updated_record = Record({"name": "Updated", "value": 100})
        success = db.update(id, updated_record)
        assert success is True

        retrieved = db.read(id)
        assert retrieved.get_value("name") == "Updated"
        assert retrieved.get_value("value") == 100

        # Delete
        success = db.delete(id)
        assert success is True

        retrieved = db.read(id)
        assert retrieved is None

        # Operations on non-existent records
        assert db.read("non-existent") is None
        assert db.update("non-existent", record) is False
        assert db.delete("non-existent") is False

    def test_exists(self):
        """Test existence checking."""
        db = SyncDatabase.from_backend("memory")

        record = Record({"test": "data"})
        id = db.create(record)

        assert db.exists(id) is True
        assert db.exists("non-existent") is False

        db.delete(id)
        assert db.exists(id) is False

    def test_upsert(self):
        """Test upsert operation."""
        db = SyncDatabase.from_backend("memory")

        # Insert new record
        record1 = Record({"name": "New"})
        id = db.upsert("custom-id", record1)
        assert id == "custom-id"
        assert db.exists("custom-id")

        # Update existing record
        record2 = Record({"name": "Updated"})
        id = db.upsert("custom-id", record2)
        assert id == "custom-id"

        retrieved = db.read("custom-id")
        assert retrieved.get_value("name") == "Updated"

    def test_search_with_filters(self):
        """Test searching with filters."""
        db = SyncDatabase.from_backend("memory")

        # Create test data
        db.create(Record({"name": "Alice", "age": 25, "status": "active"}))
        db.create(Record({"name": "Bob", "age": 30, "status": "active"}))
        db.create(Record({"name": "Charlie", "age": 35, "status": "inactive"}))
        db.create(Record({"name": "David", "age": 28, "status": "active"}))

        # Search with single filter
        query = Query().filter("age", ">", 28)
        results = db.search(query)
        assert len(results) == 2
        ages = [r.get_value("age") for r in results]
        assert all(age > 28 for age in ages)

        # Search with multiple filters
        query = Query().filter("age", ">=", 28).filter("status", "=", "active")
        results = db.search(query)
        assert len(results) == 2
        for r in results:
            assert r.get_value("age") >= 28
            assert r.get_value("status") == "active"

        # Search with IN operator
        query = Query().filter("name", "in", ["Alice", "Bob"])
        results = db.search(query)
        assert len(results) == 2
        names = [r.get_value("name") for r in results]
        assert "Alice" in names
        assert "Bob" in names

        # Search with LIKE operator
        query = Query().filter("name", "like", "C%")
        results = db.search(query)
        assert len(results) == 1
        assert results[0].get_value("name") == "Charlie"

    def test_search_with_sorting(self):
        """Test searching with sorting."""
        db = SyncDatabase.from_backend("memory")

        # Create test data
        db.create(Record({"name": "Charlie", "score": 85}))
        db.create(Record({"name": "Alice", "score": 92}))
        db.create(Record({"name": "Bob", "score": 78}))

        # Sort by score ascending
        query = Query().sort_by("score", "asc")
        results = db.search(query)
        scores = [r.get_value("score") for r in results]
        assert scores == [78, 85, 92]

        # Sort by score descending
        query = Query().sort_by("score", "desc")
        results = db.search(query)
        scores = [r.get_value("score") for r in results]
        assert scores == [92, 85, 78]

        # Sort by name
        query = Query().sort_by("name", "asc")
        results = db.search(query)
        names = [r.get_value("name") for r in results]
        assert names == ["Alice", "Bob", "Charlie"]

        # Multiple sort criteria
        db.create(Record({"name": "David", "score": 85}))
        query = Query().sort_by("score", "desc").sort_by("name", "asc")
        results = db.search(query)
        data = [(r.get_value("score"), r.get_value("name")) for r in results]
        expected = [(92, "Alice"), (85, "Charlie"), (85, "David"), (78, "Bob")]
        assert data == expected

    def test_search_with_pagination(self):
        """Test searching with pagination."""
        db = SyncDatabase.from_backend("memory")

        # Create test data
        for i in range(10):
            db.create(Record({"id": i, "value": f"item_{i}"}))

        # Test limit
        query = Query().sort_by("id", "asc").set_limit(3)
        results = db.search(query)
        assert len(results) == 3
        ids = [r.get_value("id") for r in results]
        assert ids == [0, 1, 2]

        # Test offset
        query = Query().sort_by("id", "asc").set_offset(5)
        results = db.search(query)
        assert len(results) == 5
        ids = [r.get_value("id") for r in results]
        assert ids == [5, 6, 7, 8, 9]

        # Test limit with offset
        query = Query().sort_by("id", "asc").set_limit(3).set_offset(4)
        results = db.search(query)
        assert len(results) == 3
        ids = [r.get_value("id") for r in results]
        assert ids == [4, 5, 6]

    def test_search_with_projection(self):
        """Test searching with field projection."""
        db = SyncDatabase.from_backend("memory")

        # Create test data
        db.create(
            Record(
                {"name": "Test", "age": 30, "email": "test@example.com", "phone": "123-456-7890"}
            )
        )

        # Search with projection
        query = Query().select("name", "email")
        results = db.search(query)
        assert len(results) == 1

        record = results[0]
        assert "name" in record
        assert "email" in record
        assert "age" not in record
        assert "phone" not in record

    def test_search_by_id_field(self):
        """Test searching by the special 'id' field."""
        db = SyncDatabase.from_backend("memory")
        
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=str(i), data={"value": i * 10})
            db.create(record)
        
        # Test filtering by ID with various operators
        from dataknobs_data.query import Operator
        
        # Test GT (greater than)
        query = Query().filter("id", Operator.GT, "2").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "3"
        assert results[1].id == "4"
        
        # Test LT (less than)
        query = Query().filter("id", Operator.LT, "2").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "0"
        assert results[1].id == "1"
        
        # Test EQ (equal)
        query = Query().filter("id", Operator.EQ, "3")
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "3"
        
        # Test BETWEEN
        query = Query().filter("id", Operator.BETWEEN, ["1", "3"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert [r.id for r in results] == ["1", "2", "3"]
        
        # Test IN
        query = Query().filter("id", Operator.IN, ["0", "2", "4"])
        results = db.search(query)
        assert len(results) == 3
        assert set(r.id for r in results) == {"0", "2", "4"}
        
        # Test combined filters (id AND value)
        query = Query().filter("id", Operator.GT, "1").filter("value", Operator.LTE, 30)
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
    
    def test_batch_operations(self):
        """Test batch operations."""
        db = SyncDatabase.from_backend("memory")

        # Batch create
        records = [Record({"name": f"Record_{i}", "value": i}) for i in range(5)]
        ids = db.create_batch(records)
        assert len(ids) == 5
        assert all(isinstance(id, str) for id in ids)

        # Batch read
        retrieved = db.read_batch(ids)
        assert len(retrieved) == 5
        assert all(r is not None for r in retrieved)
        values = [r.get_value("value") for r in retrieved]
        assert values == [0, 1, 2, 3, 4]

        # Batch delete
        results = db.delete_batch(ids[:3])
        assert results == [True, True, True]

        # Verify deletion
        remaining = db.read_batch(ids)
        assert remaining[0] is None
        assert remaining[1] is None
        assert remaining[2] is None
        assert remaining[3] is not None
        assert remaining[4] is not None

    def test_count_and_clear(self):
        """Test count and clear operations."""
        db = SyncDatabase.from_backend("memory")

        # Create test data
        for i in range(5):
            db.create(Record({"value": i, "even": i % 2 == 0}))

        # Count all
        assert db.count() == 5

        # Count with query
        query = Query().filter("even", "=", True)
        assert db.count(query) == 3

        # Clear
        deleted = db.clear()
        assert deleted == 5
        assert db.count() == 0

    def test_context_manager(self):
        """Test using database as context manager."""
        with SyncDatabase.from_backend("memory") as db:
            id = db.create(Record({"test": "data"}))
            assert db.exists(id)

        # Database should still work after context exit (memory doesn't need closing)
        assert db.exists(id)

    def test_thread_safety(self):
        """Test thread safety of sync memory database."""
        db = SyncDatabase.from_backend("memory")
        created_ids = []
        lock = threading.Lock()

        def create_records(thread_id):
            for i in range(10):
                record = Record({"thread": thread_id, "index": i})
                id = db.create(record)
                with lock:
                    created_ids.append(id)

        # Create records from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for thread_id in range(5):
                future = executor.submit(create_records, thread_id)
                futures.append(future)

            for future in futures:
                future.result()

        # Verify all records were created
        assert len(created_ids) == 50
        assert db.count() == 50

        # Verify all records can be read
        for id in created_ids:
            assert db.exists(id)
            assert db.read(id) is not None


class TestAsyncMemoryDatabase:
    """Test asynchronous memory database."""

    @pytest.mark.asyncio
    async def test_async_crud_operations(self):
        """Test async CRUD operations."""
        db = await AsyncDatabase.from_backend("memory")

        # Create
        record = Record({"name": "AsyncTest", "value": 99})
        id = await db.create(record)
        assert id is not None

        # Read
        retrieved = await db.read(id)
        assert retrieved.get_value("name") == "AsyncTest"

        # Update
        updated = Record({"name": "AsyncUpdated", "value": 100})
        success = await db.update(id, updated)
        assert success is True

        # Delete
        success = await db.delete(id)
        assert success is True

        # Verify deletion
        assert await db.read(id) is None

    @pytest.mark.asyncio
    async def test_async_search_by_id_field(self):
        """Test async searching by the special 'id' field."""
        db = await AsyncDatabase.from_backend("memory")
        
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=str(i), data={"value": i * 10})
            await db.create(record)
        
        # Test filtering by ID with various operators
        from dataknobs_data.query import Operator
        
        # Test GT (greater than)
        query = Query().filter("id", Operator.GT, "2").sort_by("id", "asc")
        results = await db.search(query)
        assert len(results) == 2
        assert results[0].id == "3"
        assert results[1].id == "4"
        
        # Test LT (less than)
        query = Query().filter("id", Operator.LT, "2").sort_by("id", "asc")
        results = await db.search(query)
        assert len(results) == 2
        assert results[0].id == "0"
        assert results[1].id == "1"
        
        # Test EQ (equal)
        query = Query().filter("id", Operator.EQ, "3")
        results = await db.search(query)
        assert len(results) == 1
        assert results[0].id == "3"
        
        # Test combined filters (id AND value)
        query = Query().filter("id", Operator.GT, "1").filter("value", Operator.LTE, 30)
        results = await db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"2", "3"}
    
    @pytest.mark.asyncio
    async def test_async_search(self):
        """Test async search operations."""
        db = await AsyncDatabase.from_backend("memory")

        # Create test data
        await db.create(Record({"type": "A", "value": 1}))
        await db.create(Record({"type": "B", "value": 2}))
        await db.create(Record({"type": "A", "value": 3}))

        # Search
        query = Query().filter("type", "=", "A")
        results = await db.search(query)
        assert len(results) == 2

        for r in results:
            assert r.get_value("type") == "A"

    @pytest.mark.asyncio
    async def test_async_batch_operations(self):
        """Test async batch operations."""
        db = await AsyncDatabase.from_backend("memory")

        # Batch create
        records = [Record({"id": i}) for i in range(3)]
        ids = await db.create_batch(records)
        assert len(ids) == 3

        # Batch read
        retrieved = await db.read_batch(ids)
        assert len(retrieved) == 3
        assert all(r is not None for r in retrieved)

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager."""
        async with await AsyncDatabase.from_backend("memory") as db:
            id = await db.create(Record({"async": "context"}))
            assert await db.exists(id)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent async operations."""
        db = await AsyncDatabase.from_backend("memory")

        async def create_records(task_id):
            ids = []
            for i in range(10):
                record = Record({"task": task_id, "index": i})
                id = await db.create(record)
                ids.append(id)
            return ids

        # Create records concurrently
        tasks = [create_records(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Verify all records were created
        all_ids = [id for ids in results for id in ids]
        assert len(all_ids) == 50
        assert await db.count() == 50
