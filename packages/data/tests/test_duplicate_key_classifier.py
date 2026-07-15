"""The SQL duplicate-id classifier distinguishes a primary-key collision from
other column-constraint violations.

``create()`` maps a colliding id to ``DuplicateRecordError``, but the raw driver
exceptions (``sqlite3.IntegrityError`` / ``duckdb.ConstraintException``) also
fire for ``NOT NULL`` and ``CHECK`` violations on the ``data`` / ``metadata``
columns. ``is_duplicate_key_error`` must return True only for the id collision,
so those other violations are not mislabeled as duplicate ids.

Exercised against **real** driver exceptions (no mocks): each test provokes an
actual constraint violation and feeds the resulting exception to the classifier,
pinning it against the exact error text each engine emits.
"""

from __future__ import annotations

import sqlite3

import duckdb
import pytest

from dataknobs_data.backends.sql_base import is_duplicate_key_error

# The record-table shape both engines use: id PRIMARY KEY, data NOT NULL, plus a
# CHECK constraint (mirrors the SQLite record schema).
_SQLITE_DDL = (
    "CREATE TABLE t ("
    "  id VARCHAR(255) PRIMARY KEY,"
    "  data TEXT NOT NULL CHECK (json_valid(data)),"
    "  metadata TEXT CHECK (metadata IS NULL OR json_valid(metadata))"
    ")"
)
_DUCKDB_DDL = (
    "CREATE TABLE t ("
    "  id VARCHAR PRIMARY KEY,"
    "  data JSON NOT NULL,"
    "  metadata JSON"
    ")"
)


def _sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(_SQLITE_DDL)
    conn.execute("INSERT INTO t (id, data, metadata) VALUES ('a', '{}', NULL)")
    return conn


def test_sqlite_primary_key_collision_is_duplicate() -> None:
    conn = _sqlite_conn()
    with pytest.raises(sqlite3.IntegrityError) as exc:
        conn.execute("INSERT INTO t (id, data, metadata) VALUES ('a', '{}', NULL)")
    assert is_duplicate_key_error(exc.value) is True


def test_sqlite_not_null_violation_is_not_duplicate() -> None:
    conn = _sqlite_conn()
    with pytest.raises(sqlite3.IntegrityError) as exc:
        conn.execute("INSERT INTO t (id, data, metadata) VALUES ('b', NULL, NULL)")
    assert is_duplicate_key_error(exc.value) is False


def test_sqlite_check_violation_is_not_duplicate() -> None:
    conn = _sqlite_conn()
    with pytest.raises(sqlite3.IntegrityError) as exc:
        conn.execute("INSERT INTO t (id, data, metadata) VALUES ('c', 'not json', NULL)")
    assert is_duplicate_key_error(exc.value) is False


def _duckdb_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute(_DUCKDB_DDL)
    conn.execute("INSERT INTO t (id, data, metadata) VALUES ('a', '{}', NULL)")
    return conn


def test_duckdb_primary_key_collision_is_duplicate() -> None:
    conn = _duckdb_conn()
    with pytest.raises(duckdb.ConstraintException) as exc:
        conn.execute("INSERT INTO t (id, data, metadata) VALUES ('a', '{}', NULL)")
    assert is_duplicate_key_error(exc.value) is True


def test_duckdb_not_null_violation_is_not_duplicate() -> None:
    conn = _duckdb_conn()
    with pytest.raises(duckdb.ConstraintException) as exc:
        conn.execute("INSERT INTO t (id, data, metadata) VALUES ('b', NULL, NULL)")
    assert is_duplicate_key_error(exc.value) is False
