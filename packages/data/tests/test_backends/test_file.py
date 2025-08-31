"""Tests for file backend implementation."""

import asyncio
import os
import tempfile

import pytest
import pytest_asyncio

from dataknobs_data import AsyncDatabase, SyncDatabase
from dataknobs_data.query import Operator, Query, SortOrder
from dataknobs_data.records import Record


class TestFileDatabase:
    """Test async FileDatabase implementation."""

    @pytest_asyncio.fixture
    async def temp_file(self):
        """Create a temporary file for testing."""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
        # Also remove lock file if it exists
        lock_file = path + ".lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)

    @pytest_asyncio.fixture
    async def db_json(self, temp_file):
        """Create a JSON file database."""
        db = await AsyncDatabase.from_backend("file", {"path": temp_file})
        yield db
        await db.close()

    @pytest_asyncio.fixture
    async def db_csv(self):
        """Create a CSV file database."""
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        db = await AsyncDatabase.from_backend("file", {"path": path})
        yield db
        await db.close()
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
        lock_file = path + ".lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)

    @pytest_asyncio.fixture
    async def db_gzip(self):
        """Create a gzipped JSON file database."""
        fd, path = tempfile.mkstemp(suffix=".json.gz")
        os.close(fd)
        db = await AsyncDatabase.from_backend("file", {"path": path})
        yield db
        await db.close()
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
        lock_file = path + ".lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)

    @pytest.mark.asyncio
    async def test_create_and_read(self, db_json):
        """Test creating and reading a record."""
        record = Record({"name": "Alice", "age": 30})

        # Create record
        record_id = await db_json.create(record)
        assert record_id is not None

        # Read record
        retrieved = await db_json.read(record_id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "Alice"
        assert retrieved.get_value("age") == 30

    @pytest.mark.asyncio
    async def test_update(self, db_json):
        """Test updating a record."""
        record = Record({"name": "Bob", "age": 25})
        record_id = await db_json.create(record)

        # Update record
        updated_record = Record({"name": "Bob", "age": 26})
        success = await db_json.update(record_id, updated_record)
        assert success is True

        # Verify update
        retrieved = await db_json.read(record_id)
        assert retrieved.get_value("age") == 26

    @pytest.mark.asyncio
    async def test_delete(self, db_json):
        """Test deleting a record."""
        record = Record({"name": "Charlie", "age": 35})
        record_id = await db_json.create(record)

        # Delete record
        success = await db_json.delete(record_id)
        assert success is True

        # Verify deletion
        retrieved = await db_json.read(record_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_exists(self, db_json):
        """Test checking if a record exists."""
        record = Record({"name": "Diana", "age": 28})
        record_id = await db_json.create(record)

        # Check existence
        exists = await db_json.exists(record_id)
        assert exists is True

        # Check non-existent record
        exists = await db_json.exists("non-existent-id")
        assert exists is False

    @pytest.mark.asyncio
    async def test_upsert(self, db_json):
        """Test upsert operation."""
        # Insert new record
        record = Record({"name": "Eve", "age": 32})
        record_id = await db_json.upsert("test-id", record)
        assert record_id == "test-id"

        # Update existing record
        updated_record = Record({"name": "Eve", "age": 33})
        record_id = await db_json.upsert("test-id", updated_record)
        assert record_id == "test-id"

        # Verify update
        retrieved = await db_json.read("test-id")
        assert retrieved.get_value("age") == 33

    @pytest.mark.asyncio
    async def test_search_with_filters(self, db_json):
        """Test searching with filters."""
        # Create test records
        await db_json.create(Record({"name": "Alice", "age": 30, "city": "NYC"}))
        await db_json.create(Record({"name": "Bob", "age": 25, "city": "LA"}))
        await db_json.create(Record({"name": "Charlie", "age": 35, "city": "NYC"}))

        # Search with filter
        query = Query().filter("city", Operator.EQ, "NYC")
        results = await db_json.search(query)

        assert len(results) == 2
        cities = [r.get_value("city") for r in results]
        assert all(city == "NYC" for city in cities)

    @pytest.mark.asyncio
    async def test_search_with_sorting(self, db_json):
        """Test searching with sorting."""
        # Create test records
        await db_json.create(Record({"name": "Alice", "age": 30}))
        await db_json.create(Record({"name": "Bob", "age": 25}))
        await db_json.create(Record({"name": "Charlie", "age": 35}))

        # Search with sorting
        query = Query().sort("age", SortOrder.ASC)
        results = await db_json.search(query)

        ages = [r.get_value("age") for r in results]
        assert ages == [25, 30, 35]

    @pytest.mark.asyncio
    async def test_search_with_pagination(self, db_json):
        """Test searching with pagination."""
        # Create test records
        for i in range(10):
            await db_json.create(Record({"index": i}))

        # Search with limit and offset
        query = Query().sort("index", SortOrder.ASC).limit(3).offset(2)
        results = await db_json.search(query)

        assert len(results) == 3
        indices = [r.get_value("index") for r in results]
        assert indices == [2, 3, 4]

    @pytest.mark.asyncio
    async def test_batch_operations(self, db_json):
        """Test batch operations."""
        # Create batch
        records = [Record({"name": f"User{i}", "index": i}) for i in range(5)]
        ids = await db_json.create_batch(records)
        assert len(ids) == 5

        # Read batch
        retrieved = await db_json.read_batch(ids)
        assert len(retrieved) == 5
        assert all(r is not None for r in retrieved)

        # Delete batch
        results = await db_json.delete_batch(ids)
        assert all(results)

    @pytest.mark.asyncio
    async def test_csv_format(self, db_csv):
        """Test CSV format support."""
        # Create records
        await db_csv.create(Record({"name": "Alice", "age": 30}))
        await db_csv.create(Record({"name": "Bob", "age": 25}))

        # Search all
        query = Query()
        results = await db_csv.search(query)
        assert len(results) == 2

        # Verify CSV file exists
        assert os.path.exists(db_csv.filepath)
        assert db_csv.filepath.endswith(".csv")

    @pytest.mark.asyncio
    async def test_gzip_compression(self, db_gzip):
        """Test gzip compression support."""
        # Create records
        await db_gzip.create(Record({"name": "Alice", "data": "x" * 1000}))
        await db_gzip.create(Record({"name": "Bob", "data": "y" * 1000}))

        # Search all
        query = Query()
        results = await db_gzip.search(query)
        assert len(results) == 2

        # Verify gzipped file exists
        assert os.path.exists(db_gzip.filepath)
        assert db_gzip.filepath.endswith(".gz")

    @pytest.mark.asyncio
    async def test_persistence(self, temp_file):
        """Test data persistence across database instances."""
        # Create and save data
        db1 = await AsyncDatabase.from_backend("file", {"path": temp_file})
        record_id = await db1.create(Record({"name": "Persistent", "value": 42}))
        await db1.close()

        # Load data in new instance
        db2 = await AsyncDatabase.from_backend("file", {"path": temp_file})
        retrieved = await db2.read(record_id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "Persistent"
        assert retrieved.get_value("value") == 42
        await db2.close()

    @pytest.mark.asyncio
    async def test_concurrent_access(self, db_json):
        """Test concurrent access to the database."""

        # Create multiple records concurrently
        async def create_record(index):
            record = Record({"index": index})
            return await db_json.create(record)

        # Create 10 records concurrently
        tasks = [create_record(i) for i in range(10)]
        ids = await asyncio.gather(*tasks)

        assert len(ids) == 10
        assert len(set(ids)) == 10  # All IDs should be unique

        # Verify all records exist
        for record_id in ids:
            exists = await db_json.exists(record_id)
            assert exists is True


class TestSyncFileDatabase:
    """Test synchronous FileDatabase implementation."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
        lock_file = path + ".lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)

    @pytest.fixture
    def db_json(self, temp_file):
        """Create a JSON file database."""
        db = SyncDatabase.from_backend("file", {"path": temp_file})
        yield db
        db.close()

    def test_create_and_read(self, db_json):
        """Test creating and reading a record."""
        record = Record({"name": "Alice", "age": 30})

        # Create record
        record_id = db_json.create(record)
        assert record_id is not None

        # Read record
        retrieved = db_json.read(record_id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "Alice"
        assert retrieved.get_value("age") == 30

    def test_update(self, db_json):
        """Test updating a record."""
        record = Record({"name": "Bob", "age": 25})
        record_id = db_json.create(record)

        # Update record
        updated_record = Record({"name": "Bob", "age": 26})
        success = db_json.update(record_id, updated_record)
        assert success is True

        # Verify update
        retrieved = db_json.read(record_id)
        assert retrieved.get_value("age") == 26

    def test_delete(self, db_json):
        """Test deleting a record."""
        record = Record({"name": "Charlie", "age": 35})
        record_id = db_json.create(record)

        # Delete record
        success = db_json.delete(record_id)
        assert success is True

        # Verify deletion
        retrieved = db_json.read(record_id)
        assert retrieved is None

    def test_search_with_filters(self, db_json):
        """Test searching with filters."""
        # Create test records
        db_json.create(Record({"name": "Alice", "age": 30, "city": "NYC"}))
        db_json.create(Record({"name": "Bob", "age": 25, "city": "LA"}))
        db_json.create(Record({"name": "Charlie", "age": 35, "city": "NYC"}))

        # Search with filter
        query = Query().filter("city", Operator.EQ, "NYC")
        results = db_json.search(query)

        assert len(results) == 2
        cities = [r.get_value("city") for r in results]
        assert all(city == "NYC" for city in cities)

    def test_batch_operations(self, db_json):
        """Test batch operations."""
        # Create batch
        records = [Record({"name": f"User{i}", "index": i}) for i in range(5)]
        ids = db_json.create_batch(records)
        assert len(ids) == 5

        # Read batch
        retrieved = db_json.read_batch(ids)
        assert len(retrieved) == 5
        assert all(r is not None for r in retrieved)

        # Delete batch
        results = db_json.delete_batch(ids)
        assert all(results)

    def test_persistence(self, temp_file):
        """Test data persistence across database instances."""
        # Create and save data
        db1 = SyncDatabase.from_backend("file", {"path": temp_file})
        record_id = db1.create(Record({"name": "Persistent", "value": 42}))
        db1.close()

        # Load data in new instance
        db2 = SyncDatabase.from_backend("file", {"path": temp_file})
        retrieved = db2.read(record_id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "Persistent"
        assert retrieved.get_value("value") == 42
        db2.close()

    def test_format_detection(self):
        """Test automatic format detection from file extension."""
        # Test JSON
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        db = SyncDatabase.from_backend("file", {"path": path})
        assert db.format == "json"
        db.close()
        os.remove(path)

        # Test CSV
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        db = SyncDatabase.from_backend("file", {"path": path})
        assert db.format == "csv"
        db.close()
        os.remove(path)

        # Test gzipped JSON
        fd, path = tempfile.mkstemp(suffix=".json.gz")
        os.close(fd)
        db = SyncDatabase.from_backend("file", {"path": path})
        assert db.format == "json"
        assert db.compression == "gzip"
        db.close()
        os.remove(path)
