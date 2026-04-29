"""Integration tests: unusual-but-valid table names work after identifier quoting."""

import pytest

from dataknobs_data.backends.memory import SyncMemoryDatabase
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
