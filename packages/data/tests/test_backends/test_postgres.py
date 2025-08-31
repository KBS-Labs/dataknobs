"""Tests for PostgreSQL backend implementation."""

import os
import uuid

import pytest
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from dataknobs_data import AsyncDatabase, Query, Record, SyncDatabase
from dataknobs_data.query import Filter, Operator, SortOrder, SortSpec

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="PostgreSQL tests require TEST_POSTGRES=true and a running PostgreSQL instance"
)


@pytest.fixture(scope="session")
def ensure_test_database():
    """Ensure the test database exists."""
    if not os.environ.get("TEST_POSTGRES", "").lower() == "true":
        return
    
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", 5432))
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    db_name = os.environ.get("POSTGRES_DB", "test_dataknobs")
    
    # Connect to postgres database to create test database
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()
        
        if not exists:
            # Create the database
            cur.execute(f"CREATE DATABASE {db_name}")
        
        cur.close()
        conn.close()
    except Exception as e:
        # If we can't create the database, the tests will fail anyway
        print(f"Warning: Could not ensure test database exists: {e}")
    
    yield
    
    # Cleanup is optional - we can leave the test database for future runs


@pytest.fixture
def postgres_config(ensure_test_database):
    """PostgreSQL configuration for testing."""
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", 5432)),
        "database": os.environ.get("POSTGRES_DB", "test_dataknobs"),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
        "table": f"test_records_{uuid.uuid4().hex[:8]}",  # Unique table per test
        "schema": "public",
    }


@pytest.fixture
def sync_db(postgres_config):
    """Create a synchronous PostgreSQL database for testing."""
    db = SyncDatabase.from_backend("postgres", postgres_config)
    yield db
    # Cleanup: drop the test table
    try:
        db.clear()
    except Exception:
        pass
    db.close()


@pytest.fixture
async def async_db(postgres_config):
    """Create an asynchronous PostgreSQL database for testing."""
    db = await AsyncDatabase.from_backend("postgres", postgres_config)
    yield db
    # Cleanup
    try:
        await db.clear()
    except Exception:
        pass
    await db.close()


class TestSyncPostgresDatabase:
    """Test synchronous PostgreSQL database."""

    def test_create_database(self, postgres_config):
        """Test creating a PostgreSQL database."""
        db = SyncDatabase.from_backend("postgres", postgres_config)
        assert db is not None
        assert hasattr(db, "db")  # Has PostgresDB instance
        assert hasattr(db, "table_name")
        assert hasattr(db, "schema_name")
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

        # Test LIKE filter
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

        # Sort by age ascending
        query = Query().sort("age", SortOrder.ASC)
        results = sync_db.search(query)
        ages = [r.get_value("age") for r in results]
        assert ages == [25, 30, 35]

        # Sort by age descending
        query = Query().sort("age", SortOrder.DESC)
        results = sync_db.search(query)
        ages = [r.get_value("age") for r in results]
        assert ages == [35, 30, 25]

        # Sort by name
        query = Query().sort("name", SortOrder.ASC)
        results = sync_db.search(query)
        names = [r.get_value("name") for r in results]
        assert names == ["Alice", "Bob", "Charlie"]

    def test_search_with_pagination(self, sync_db):
        """Test searching with pagination."""
        # Create test data
        for i in range(10):
            sync_db.create(Record({"index": i, "value": f"item_{i}"}))

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


@pytest.mark.asyncio
class TestAsyncPostgresDatabase:
    """Test asynchronous PostgreSQL database."""

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

        # Search
        query = Query().filter("type", Operator.EQ, "async")
        results = await async_db.search(query)
        assert len(results) == 2
        assert all(r.get_value("type") == "async" for r in results)

    async def test_async_context_manager(self, postgres_config):
        """Test async context manager."""
        async with await AsyncDatabase.from_backend("postgres", postgres_config) as db:
            record = Record({"context": "manager"})
            id = await db.create(record)
            assert await db.exists(id)