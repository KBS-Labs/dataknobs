"""Test backend fixes for storage_id population, update persistence, and connection strings.

Each test names the behaviour it pins; the three were reported together
by a downstream user of the Postgres backend.
"""

import pytest
from dataknobs_common.testing import postgres_dsn, requires_real_postgres

# Async backends only, so asyncpg is the driver these need.
skip_postgres = requires_real_postgres


@skip_postgres
@pytest.mark.asyncio
async def test_postgres_search_populates_storage_id(postgres_connection_params):
    """Test that PostgreSQL search() populates storage_id from database ID.

    Search must populate ``record.storage_id`` from the row's ``id``.
    """
    from dataknobs_data.backends.postgres import AsyncPostgresDatabase
    from dataknobs_data.query import Query
    from dataknobs_data.records import Record

    # Create backend with connection string (also covers the
    # connection-string acceptance path below).
    backend = AsyncPostgresDatabase({"connection_string": postgres_dsn(postgres_connection_params)})

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
async def test_postgres_update_persists_changes(postgres_connection_params):
    """Test that PostgreSQL update() persists changes to the database.

    ``update()`` must write through using the storage_id, not the
    caller-side sync_id. This is the individual-keys config path; the
    two tests either side of it use ``connection_string``.
    """
    from dataknobs_data.backends.postgres import AsyncPostgresDatabase
    from dataknobs_data.query import Query
    from dataknobs_data.records import Record

    backend = AsyncPostgresDatabase(dict(postgres_connection_params))

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

        print("✓ Test passed: Update persisted successfully")

    finally:
        await backend.close()


@skip_postgres
@pytest.mark.asyncio
async def test_postgres_connection_string(postgres_connection_params):
    """Test that PostgreSQL backend accepts connection strings.

    ``PostgresPoolConfig.from_dict()`` must accept a ``connection_string``
    rather than requiring individual keys.
    """
    from dataknobs_data.backends.postgres import AsyncPostgresDatabase
    from dataknobs_data.records import Record

    # Test with connection string
    connection_string = postgres_dsn(postgres_connection_params)
    backend = AsyncPostgresDatabase({"connection_string": connection_string})

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

        print("✓ Test passed: Connection string works")

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
