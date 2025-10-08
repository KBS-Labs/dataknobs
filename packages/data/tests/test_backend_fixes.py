"""Test backend fixes for storage_id population, update persistence, and connection strings.

These tests verify the fixes for issues documented in:
sandbox/packages/sandbox-focus/docs/active/dataknobs-data-backend-issues.md
"""

import os
import pytest

# Check if PostgreSQL tests should run
TEST_POSTGRES = os.getenv("TEST_POSTGRES", "false").lower() == "true"
skip_postgres = pytest.mark.skipif(
    not TEST_POSTGRES,
    reason="PostgreSQL tests skipped. Set TEST_POSTGRES=true to run."
)


@skip_postgres
@pytest.mark.asyncio
async def test_postgres_search_populates_storage_id():
    """Test that PostgreSQL search() populates storage_id from database ID.

    Issue #1: Search Returns Records with storage_id=None
    Fix: Populate record.storage_id from row['id'] in search()
    """
    from dataknobs_data.backends.postgres import AsyncPostgresDatabase
    from dataknobs_data.query import Query
    from dataknobs_data.records import Record

    # Create backend with connection string (also tests Issue #3)
    db_name = os.environ.get("POSTGRES_DB", "dataknobs_test")
    backend = AsyncPostgresDatabase({
        "connection_string": f"postgresql://postgres:postgres@localhost:5432/{db_name}"
    })

    try:
        await backend.connect()

        # Clear any existing records from previous test runs
        await backend.clear()

        # Create a record
        record = Record(data={"test": "data", "status": "pending"})
        created_id = await backend.create(record)

        # Search for it
        query = Query().filter("test", "==", "data")
        results = await backend.search(query)

        # Verify storage_id is populated
        assert len(results) == 1, "Should find exactly one record"
        assert results[0].storage_id is not None, "storage_id should not be None"
        assert results[0].storage_id == created_id, "storage_id should match created ID"

        print(f"✓ Test passed: storage_id={results[0].storage_id}")

    finally:
        await backend.close()


@skip_postgres
@pytest.mark.asyncio
async def test_postgres_update_persists_changes():
    """Test that PostgreSQL update() persists changes to the database.

    Issue #2: Update Method Fails to Persist Changes
    Root cause: Wrong ID being used (sync_id instead of PostgreSQL UUID)
    Fix: Issue #1 fix provides correct storage_id to use for updates
    """
    from dataknobs_data.backends.postgres import AsyncPostgresDatabase
    from dataknobs_data.query import Query
    from dataknobs_data.records import Record

    db_name = os.environ.get("POSTGRES_DB", "dataknobs_test")
    backend = AsyncPostgresDatabase({
        "host": "localhost",
        "port": 5432,
        "database": db_name,
        "user": "postgres",
        "password": "postgres"
    })

    try:
        await backend.connect()

        # Clear any existing records from previous test runs
        await backend.clear()

        # Create initial record
        record = Record(data={"status": "pending", "count": 0})
        record_id = await backend.create(record)

        # Search for it to get the storage_id
        query = Query().filter("status", "==", "pending")
        results = await backend.search(query)

        assert len(results) == 1
        found_record = results[0]

        # Update using the storage_id from search results
        updated_record = Record(data={"status": "completed", "count": 5})
        success = await backend.update(found_record.storage_id, updated_record)

        assert success, "Update should succeed"

        # Verify changes persisted
        verify_query = Query().filter("status", "==", "completed")
        verify_results = await backend.search(verify_query)

        assert len(verify_results) == 1, "Should find updated record"
        assert verify_results[0].data["status"] == "completed", "Status should be updated"
        assert verify_results[0].data["count"] == 5, "Count should be updated"

        print(f"✓ Test passed: Update persisted successfully")

    finally:
        await backend.close()


@skip_postgres
@pytest.mark.asyncio
async def test_postgres_connection_string():
    """Test that PostgreSQL backend accepts connection strings.

    Issue #3: PostgreSQL Backend Does Not Accept Connection Strings
    Fix: Added connection_string parameter to PostgresPoolConfig.from_dict()
    """
    from dataknobs_data.backends.postgres import AsyncPostgresDatabase
    from dataknobs_data.records import Record

    # Test with connection string
    db_name = os.environ.get("POSTGRES_DB", "dataknobs_test")
    connection_string = f"postgresql://postgres:postgres@localhost:5432/{db_name}"
    backend = AsyncPostgresDatabase({
        "connection_string": connection_string
    })

    try:
        await backend.connect()

        # Clear any existing records from previous test runs
        await backend.clear()

        # Should successfully connect and be usable
        record = Record(data={"test": "connection_string"})
        record_id = await backend.create(record)

        assert record_id is not None, "Should create record successfully"

        # Verify we can read it back
        read_record = await backend.read(record_id)
        assert read_record is not None, "Should read record successfully"
        assert read_record.data["test"] == "connection_string"

        print(f"✓ Test passed: Connection string works")

    finally:
        await backend.close()


def test_sqlite_search_populates_storage_id():
    """Test that SQLite search() also populates storage_id correctly."""
    from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
    from dataknobs_data.query import Query
    from dataknobs_data.records import Record

    backend = SyncSQLiteDatabase({"path": ":memory:"})

    try:
        backend.connect()

        # Create a record
        record = Record(data={"test": "data"})
        created_id = backend.create(record)

        # Search for it
        query = Query().filter("test", "==", "data")
        results = backend.search(query)

        # Verify storage_id is populated
        assert len(results) == 1
        assert results[0].storage_id is not None
        assert results[0].storage_id == created_id

        print(f"✓ SQLite test passed: storage_id={results[0].storage_id}")

    finally:
        backend.close()


@pytest.mark.asyncio
async def test_sqlite_async_search_populates_storage_id():
    """Test that async SQLite search() also populates storage_id correctly."""
    from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
    from dataknobs_data.query import Query
    from dataknobs_data.records import Record

    backend = AsyncSQLiteDatabase({"path": ":memory:"})

    try:
        await backend.connect()

        # Create a record
        record = Record(data={"test": "data"})
        created_id = await backend.create(record)

        # Search for it
        query = Query().filter("test", "==", "data")
        results = await backend.search(query)

        # Verify storage_id is populated
        assert len(results) == 1
        assert results[0].storage_id is not None
        assert results[0].storage_id == created_id

        print(f"✓ Async SQLite test passed: storage_id={results[0].storage_id}")

    finally:
        await backend.close()


if __name__ == "__main__":
    import asyncio

    print("Running backend fix tests...\n")

    # Run sync tests
    print("Testing SQLite (sync)...")
    test_sqlite_search_populates_storage_id()

    # Run async tests
    print("\nTesting SQLite (async)...")
    asyncio.run(test_sqlite_async_search_populates_storage_id())

    print("\n\nPostgreSQL tests require a running PostgreSQL instance.")
    print("To run PostgreSQL tests:")
    print("  pytest tests/test_backend_fixes.py::test_postgres_search_populates_storage_id")
    print("  pytest tests/test_backend_fixes.py::test_postgres_update_persists_changes")
    print("  pytest tests/test_backend_fixes.py::test_postgres_connection_string")
