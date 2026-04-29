"""Integration tests: unusual-but-valid table names work after identifier quoting."""

import pytest
import pytest_asyncio

from dataknobs_data.records import Record


# In-process SQLite + DuckDB tests (no service required)
def test_sqlite_quoted_identifier_round_trip(tmp_path):
    """SQLite: unusual table name works correctly after quoting."""
    from dataknobs_data.backends.sqlite import SyncSQLiteDatabase

    db_path = tmp_path / "test.db"
    db = SyncSQLiteDatabase({"path": str(db_path), "table": "MyWeirdTable"})
    db.connect()
    try:
        record = Record(id="x1", data={"v": 42})
        db.create(record)
        fetched = db.read("x1")
        assert fetched is not None
        assert fetched.get_value("v") == 42
    finally:
        db.disconnect()


def test_duckdb_quoted_identifier_round_trip(tmp_path):
    """DuckDB: unusual table name works correctly after quoting."""
    from dataknobs_data.backends.duckdb import SyncDuckDBDatabase

    db_path = tmp_path / "test.db"
    db = SyncDuckDBDatabase({"path": str(db_path), "table": "MyWeirdTable"})
    db.connect()
    try:
        record = Record(id="x1", data={"v": 42})
        db.create(record)
        fetched = db.read("x1")
        assert fetched is not None
        assert fetched.get_value("v") == 42
    finally:
        db.disconnect()


@pytest.mark.asyncio
async def test_async_sqlite_quoted_identifier_round_trip(tmp_path):
    """AsyncSQLiteDatabase: mixed-case table name works for create/read/update_batch/delete_batch/count."""
    from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

    db_path = tmp_path / "test.db"
    db = AsyncSQLiteDatabase({"path": str(db_path), "table": "MyWeirdTable"})
    await db.connect()
    try:
        r1 = Record(id="a1", data={"v": 1})
        r2 = Record(id="a2", data={"v": 2})
        await db.create(r1)
        await db.create(r2)

        fetched = await db.read("a1")
        assert fetched is not None
        assert fetched.get_value("v") == 1

        # update_batch exercises line 342
        r1_updated = Record(id="a1", data={"v": 99})
        results = await db.update_batch([("a1", r1_updated)])
        assert results == [True]

        # delete_batch exercises line 370
        deleted = await db.delete_batch(["a2"])
        assert deleted == [True]

        # count() with no query exercises _count_all (line 404)
        n = await db.count()
        assert n == 1
    finally:
        await db.close()
