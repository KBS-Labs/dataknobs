"""Tests for Elasticsearch backend implementation."""

import os
import time
import uuid

import pytest

from dataknobs_data import AsyncDatabase, Query, Record, SyncDatabase
from dataknobs_data.query import Filter, Operator, SortOrder, SortSpec

# Skip all tests if Elasticsearch is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_ELASTICSEARCH", "").lower() == "true",
    reason="Elasticsearch tests require TEST_ELASTICSEARCH=true and a running Elasticsearch instance"
)


@pytest.fixture
def elasticsearch_config():
    """Elasticsearch configuration for testing."""
    return {
        "host": os.environ.get("ELASTICSEARCH_HOST", "localhost"),
        "port": int(os.environ.get("ELASTICSEARCH_PORT", 9200)),
        "index": f"test_records_{uuid.uuid4().hex[:8]}",  # Unique index per test
        "refresh": True,  # Immediate refresh for testing
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
    }


@pytest.fixture
def sync_db(elasticsearch_config):
    """Create a synchronous Elasticsearch database for testing."""
    db = SyncDatabase.from_backend("elasticsearch", elasticsearch_config)
    yield db
    # Cleanup: delete the test index
    try:
        db.clear()
        # Delete the index completely
        db.es_index.delete()
    except Exception:
        pass
    db.close()


@pytest.fixture
async def async_db(elasticsearch_config):
    """Create an asynchronous Elasticsearch database for testing."""
    db = await AsyncDatabase.from_backend("elasticsearch", elasticsearch_config)
    yield db
    # Cleanup
    try:
        await db.clear()
        # Delete the index completely
        db._sync_db.es_index.delete()
    except Exception:
        pass
    await db.close()


class TestSyncElasticsearchDatabase:
    """Test synchronous Elasticsearch database."""

    def test_create_database(self, elasticsearch_config):
        """Test creating an Elasticsearch database."""
        db = SyncDatabase.from_backend("elasticsearch", elasticsearch_config)
        assert db is not None
        assert hasattr(db, "es_index")
        assert hasattr(db, "index_name")
        assert db.es_index.exists()
        db.close()

    def test_crud_operations(self, sync_db):
        """Test basic CRUD operations."""
        # Create
        record = Record({"name": "Test", "value": 42, "active": True})
        id = sync_db.create(record)
        assert id is not None
        assert isinstance(id, str)

        # Read
        retrieved = sync_db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "Test"
        assert retrieved.get_value("value") == 42
        assert retrieved.get_value("active") is True

        # Update
        updated_record = Record({"name": "Updated", "value": 100, "active": False})
        success = sync_db.update(id, updated_record)
        assert success is True

        retrieved = sync_db.read(id)
        assert retrieved.get_value("name") == "Updated"
        assert retrieved.get_value("value") == 100
        assert retrieved.get_value("active") is False

        # Delete
        success = sync_db.delete(id)
        assert success is True

        retrieved = sync_db.read(id)
        assert retrieved is None

        # Operations on non-existent records
        assert sync_db.read("non-existent") is None
        assert sync_db.update("non-existent", record) is False
        assert sync_db.delete("non-existent") is False

    def test_exists(self, sync_db):
        """Test existence checking."""
        record = Record({"test": "data"})
        id = sync_db.create(record)

        assert sync_db.exists(id) is True
        assert sync_db.exists("non-existent") is False

        sync_db.delete(id)
        assert sync_db.exists(id) is False

    def test_upsert(self, sync_db):
        """Test upsert operation."""
        # Insert new record
        record1 = Record({"name": "New"})
        custom_id = f"custom-{uuid.uuid4().hex[:8]}"
        id = sync_db.upsert(custom_id, record1)
        assert id == custom_id
        assert sync_db.exists(custom_id)

        # Update existing record
        record2 = Record({"name": "Updated"})
        id = sync_db.upsert(custom_id, record2)
        assert id == custom_id

        retrieved = sync_db.read(custom_id)
        assert retrieved.get_value("name") == "Updated"

    def test_search_with_filters(self, sync_db):
        """Test searching with filters."""
        # Create test data
        sync_db.create(Record({"name": "Alice", "age": 25, "status": "active"}))
        sync_db.create(Record({"name": "Bob", "age": 30, "status": "active"}))
        sync_db.create(Record({"name": "Charlie", "age": 35, "status": "inactive"}))
        sync_db.create(Record({"name": "David", "age": 28, "status": "active"}))

        # Small delay for indexing (even with refresh=true)
        time.sleep(0.5)

        # Test equality filter
        query = Query().filter("status", Operator.EQ, "active")
        results = sync_db.search(query)
        assert len(results) == 3
        assert all(r.get_value("status") == "active" for r in results)

        # Test greater than filter
        query = Query().filter("age", Operator.GT, 28)
        results = sync_db.search(query)
        assert len(results) == 2
        assert all(r.get_value("age") > 28 for r in results)

        # Test LIKE filter (wildcard in Elasticsearch)
        query = Query().filter("name", Operator.LIKE, "C%")
        results = sync_db.search(query)
        assert len(results) == 1
        assert results[0].get_value("name") == "Charlie"

        # Test IN filter
        query = Query().filter("age", Operator.IN, [25, 30])
        results = sync_db.search(query)
        assert len(results) == 2
        names = {r.get_value("name") for r in results}
        assert names == {"Alice", "Bob"}

        # Test combined filters
        query = Query().filter("status", Operator.EQ, "active").filter("age", Operator.GTE, 28)
        results = sync_db.search(query)
        assert len(results) == 2
        names = {r.get_value("name") for r in results}
        assert names == {"Bob", "David"}

    def test_search_with_sorting(self, sync_db):
        """Test searching with sorting."""
        # Create test data
        sync_db.create(Record({"name": "Alice", "age": 25}))
        sync_db.create(Record({"name": "Bob", "age": 30}))
        sync_db.create(Record({"name": "Charlie", "age": 35}))

        # Small delay for indexing
        time.sleep(0.5)

        # Sort by age ascending
        query = Query().sort("age", SortOrder.ASC)
        results = sync_db.search(query)
        ages = [r.get_value("age") for r in results]
        assert ages == sorted(ages)

        # Sort by age descending
        query = Query().sort("age", SortOrder.DESC)
        results = sync_db.search(query)
        ages = [r.get_value("age") for r in results]
        assert ages == sorted(ages, reverse=True)

        # Sort by name (keyword field)
        query = Query().sort("name", SortOrder.ASC)
        results = sync_db.search(query)
        names = [r.get_value("name") for r in results]
        assert names == sorted(names)

    def test_search_with_pagination(self, sync_db):
        """Test searching with pagination."""
        # Create test data
        for i in range(10):
            sync_db.create(Record({"index": i, "value": f"item_{i}"}))

        # Small delay for indexing
        time.sleep(0.5)

        # Test limit
        query = Query().limit(3)
        results = sync_db.search(query)
        assert len(results) == 3

        # Test offset
        query = Query().offset(5).limit(3)
        results = sync_db.search(query)
        assert len(results) == 3

        # Test limit + offset + sort for deterministic results
        query = Query().sort("index", SortOrder.ASC).limit(3).offset(2)
        results = sync_db.search(query)
        assert len(results) == 3
        indices = [r.get_value("index") for r in results]
        assert indices == [2, 3, 4]

    def test_batch_operations(self, sync_db):
        """Test batch operations."""
        # Create batch
        records = [
            Record({"batch": 1, "item": i})
            for i in range(5)
        ]
        ids = sync_db.create_batch(records)
        assert len(ids) == 5
        assert all(isinstance(id, str) for id in ids)

        # Small delay for indexing
        time.sleep(0.5)

        # Read batch
        retrieved = sync_db.read_batch(ids)
        assert len(retrieved) == 5
        assert all(r is not None for r in retrieved)

        # Delete batch
        results = sync_db.delete_batch(ids)
        assert all(results)

        # Verify deletion
        retrieved = sync_db.read_batch(ids)
        assert all(r is None for r in retrieved)

    def test_count_and_clear(self, sync_db):
        """Test count and clear operations."""
        # Create test data
        for i in range(5):
            sync_db.create(Record({"index": i}))

        # Small delay for indexing
        time.sleep(0.5)

        # Test count all
        count = sync_db.count()
        assert count == 5

        # Test count with query
        query = Query().filter("index", Operator.GTE, 3)
        count = sync_db.count(query)
        assert count == 2

        # Test clear
        deleted_count = sync_db.clear()
        assert deleted_count == 5

        # Small delay for deletion to propagate
        time.sleep(0.5)

        # Verify cleared
        count = sync_db.count()
        assert count == 0

    def test_metadata_handling(self, sync_db):
        """Test metadata storage and retrieval."""
        record = Record(
            data={"name": "Test"},
            metadata={"created_by": "user1", "version": 1}
        )
        id = sync_db.create(record)

        retrieved = sync_db.read(id)
        assert retrieved.metadata["created_by"] == "user1"
        assert retrieved.metadata["version"] == 1

    def test_special_characters(self, sync_db):
        """Test handling of special characters in data."""
        record = Record({
            "name": "Test's \"quoted\" value",
            "json": {"nested": {"key": "value"}},
            "unicode": "Hello 世界 🌍",
        })
        id = sync_db.create(record)

        retrieved = sync_db.read(id)
        assert retrieved.get_value("name") == "Test's \"quoted\" value"
        assert retrieved.get_value("json") == {"nested": {"key": "value"}}
        assert retrieved.get_value("unicode") == "Hello 世界 🌍"

    def test_exists_filter(self, sync_db):
        """Test EXISTS and NOT_EXISTS filters."""
        # Create test data with different fields
        sync_db.create(Record({"name": "Alice", "email": "alice@example.com"}))
        sync_db.create(Record({"name": "Bob"}))  # No email
        sync_db.create(Record({"name": "Charlie", "email": "charlie@example.com", "phone": "123"}))

        # Small delay for indexing
        time.sleep(0.5)

        # Test EXISTS filter
        query = Query().filter("email", Operator.EXISTS)
        results = sync_db.search(query)
        assert len(results) == 2
        names = {r.get_value("name") for r in results}
        assert names == {"Alice", "Charlie"}

        # Test NOT_EXISTS filter
        query = Query().filter("email", Operator.NOT_EXISTS)
        results = sync_db.search(query)
        assert len(results) == 1
        assert results[0].get_value("name") == "Bob"

    def test_regex_filter(self, sync_db):
        """Test regular expression filter."""
        # Create test data
        sync_db.create(Record({"email": "alice@example.com"}))
        sync_db.create(Record({"email": "bob@test.org"}))
        sync_db.create(Record({"email": "charlie@example.org"}))

        # Small delay for indexing
        time.sleep(0.5)

        # Test regex filter
        query = Query().filter("email", Operator.REGEX, ".*@example\\.(com|org)")
        results = sync_db.search(query)
        assert len(results) == 2
        emails = {r.get_value("email") for r in results}
        assert emails == {"alice@example.com", "charlie@example.org"}


@pytest.mark.asyncio
class TestAsyncElasticsearchDatabase:
    """Test asynchronous Elasticsearch database."""

    async def test_async_crud_operations(self, async_db):
        """Test async CRUD operations."""
        # Create
        record = Record({"name": "AsyncTest", "value": 42})
        id = await async_db.create(record)
        assert id is not None

        # Read
        retrieved = await async_db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "AsyncTest"

        # Update
        updated_record = Record({"name": "AsyncUpdated", "value": 100})
        success = await async_db.update(id, updated_record)
        assert success is True

        retrieved = await async_db.read(id)
        assert retrieved.get_value("name") == "AsyncUpdated"

        # Delete
        success = await async_db.delete(id)
        assert success is True

        retrieved = await async_db.read(id)
        assert retrieved is None

    async def test_async_search(self, async_db):
        """Test async search operations."""
        # Create test data
        await async_db.create(Record({"type": "async", "value": 1}))
        await async_db.create(Record({"type": "async", "value": 2}))
        await async_db.create(Record({"type": "sync", "value": 3}))

        # Small delay for indexing
        time.sleep(0.5)

        # Search
        query = Query().filter("type", Operator.EQ, "async")
        results = await async_db.search(query)
        assert len(results) == 2
        assert all(r.get_value("type") == "async" for r in results)

    async def test_async_context_manager(self, elasticsearch_config):
        """Test async context manager."""
        async with await AsyncDatabase.from_backend("elasticsearch", elasticsearch_config) as db:
            record = Record({"context": "manager"})
            id = await db.create(record)
            assert await db.exists(id)