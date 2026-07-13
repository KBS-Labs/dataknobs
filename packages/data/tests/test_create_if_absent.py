"""Atomic create-if-absent contract across the in-process backends.

``create()`` is a defined atomic insert: a colliding id fails closed with
``DuplicateRecordError`` rather than silently overwriting an existing record.
Before this contract the backends disagreed — memory and file silently
overwrote the existing record, while SQLite and DuckDB raised a bare
``ValueError``. This module pins the unified behavior on every in-process
backend (memory, file, SQLite, DuckDB), for both the sync and async variants.

``DuplicateRecordError`` subclasses ``ValueError``, so the SQLite/DuckDB
callers that previously caught ``ValueError`` on a duplicate id keep working.

Backends that require an external service (Postgres, S3, Elasticsearch) are
covered under ``tests/integration/`` behind their service markers.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase


# ---------------------------------------------------------------------------
# Sync backends
# ---------------------------------------------------------------------------
@pytest.fixture(params=["memory", "file", "sqlite", "duckdb"])
def sync_db(request: pytest.FixtureRequest) -> Iterator[object]:
    """A connected sync backend, one per in-process backend family."""
    kind = request.param
    with tempfile.TemporaryDirectory() as d:
        db: object
        if kind == "memory":
            db = SyncMemoryDatabase()
        elif kind == "file":
            db = SyncFileDatabase({"path": str(Path(d) / "records.json")})
        elif kind == "sqlite":
            db = SyncSQLiteDatabase({"path": str(Path(d) / "records.db")})
            db.connect()
        else:
            db = SyncDuckDBDatabase({"path": str(Path(d) / "records.duckdb"), "table": "records"})
            db.connect()
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()


def test_sync_duplicate_create_raises(sync_db: object) -> None:
    """A second create() on the same id fails closed with DuplicateRecordError."""
    sync_db.create(Record({"v": 1}, id="dup"))
    with pytest.raises(DuplicateRecordError) as excinfo:
        sync_db.create(Record({"v": 2}, id="dup"))
    assert excinfo.value.id == "dup"


def test_sync_duplicate_create_is_valueerror(sync_db: object) -> None:
    """DuplicateRecordError subclasses ValueError (catch-compat)."""
    sync_db.create(Record({"v": 1}, id="dup"))
    with pytest.raises(ValueError):
        sync_db.create(Record({"v": 2}, id="dup"))


def test_sync_no_clobber_on_collision(sync_db: object) -> None:
    """The original record survives a colliding create (no overwrite)."""
    sync_db.create(Record({"v": "winner"}, id="dup"))
    with pytest.raises(DuplicateRecordError):
        sync_db.create(Record({"v": "loser"}, id="dup"))
    assert sync_db.read("dup").get_value("v") == "winner"


def test_sync_distinct_ids_still_create(sync_db: object) -> None:
    """Distinct ids create normally — the tightening is collision-only."""
    sync_db.create(Record({"v": 1}, id="a"))
    sync_db.create(Record({"v": 2}, id="b"))
    assert sync_db.read("a").get_value("v") == 1
    assert sync_db.read("b").get_value("v") == 2


# ---------------------------------------------------------------------------
# Async backends
# ---------------------------------------------------------------------------
@pytest.fixture(params=["memory", "file", "sqlite", "duckdb"])
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[object]:
    """A connected async backend, one per in-process backend family."""
    kind = request.param
    with tempfile.TemporaryDirectory() as d:
        db: object
        if kind == "memory":
            db = AsyncMemoryDatabase()
        elif kind == "file":
            db = AsyncFileDatabase({"path": str(Path(d) / "records.json")})
        elif kind == "sqlite":
            db = AsyncSQLiteDatabase({"path": str(Path(d) / "records.db")})
            await db.connect()
        else:
            db = AsyncDuckDBDatabase({"path": str(Path(d) / "records.duckdb"), "table": "records"})
            await db.connect()
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                await close()


@pytest.mark.asyncio
async def test_async_duplicate_create_raises(async_db: object) -> None:
    """A second create() on the same id fails closed with DuplicateRecordError."""
    await async_db.create(Record({"v": 1}, id="dup"))
    with pytest.raises(DuplicateRecordError) as excinfo:
        await async_db.create(Record({"v": 2}, id="dup"))
    assert excinfo.value.id == "dup"


@pytest.mark.asyncio
async def test_async_duplicate_create_is_valueerror(async_db: object) -> None:
    """DuplicateRecordError subclasses ValueError (catch-compat)."""
    await async_db.create(Record({"v": 1}, id="dup"))
    with pytest.raises(ValueError):
        await async_db.create(Record({"v": 2}, id="dup"))


@pytest.mark.asyncio
async def test_async_no_clobber_on_collision(async_db: object) -> None:
    """The original record survives a colliding create (no overwrite)."""
    await async_db.create(Record({"v": "winner"}, id="dup"))
    with pytest.raises(DuplicateRecordError):
        await async_db.create(Record({"v": "loser"}, id="dup"))
    got = await async_db.read("dup")
    assert got.get_value("v") == "winner"


@pytest.mark.asyncio
async def test_async_concurrent_create_exactly_one_wins(async_db: object) -> None:
    """Concurrent create() of one id: exactly one succeeds, the rest raise."""
    import asyncio

    results = await asyncio.gather(
        async_db.create(Record({"who": "a"}, id="race")),
        async_db.create(Record({"who": "b"}, id="race")),
        async_db.create(Record({"who": "c"}, id="race")),
        return_exceptions=True,
    )
    successes = [r for r in results if not isinstance(r, BaseException)]
    duplicates = [r for r in results if isinstance(r, DuplicateRecordError)]
    assert len(successes) == 1
    assert len(duplicates) == 2
    # The surviving record is the winner's, and it is readable.
    got = await async_db.read("race")
    assert got is not None
    assert got.get_value("who") in {"a", "b", "c"}
