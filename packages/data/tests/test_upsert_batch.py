"""The ``upsert_batch`` batch write-verb contract across the in-process backends.

``upsert_batch(records)`` is the batch sibling of ``create_batch``: it inserts
new records and overwrites existing ones in a single call, honors a
caller-supplied ``record.id`` (minting one only when absent), and returns the
ids in input order. Unlike ``create_batch`` it never fails on a colliding id —
overwrite is the defined behavior — and it carries no version check (batch CAS
is a separate concern).

This pins the unified behavior on every in-process backend (memory, file,
SQLite, DuckDB), for both the sync and async variants. Backends that require an
external service (Postgres, S3, Elasticsearch) are covered under
``tests/integration/`` behind their service markers.

This is the batch sibling of ``test_upsert_enhancements.py`` (single
``upsert()``).
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from dataknobs_data import ConflictPolicy, Record, StreamConfig
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.streaming import resolve_conflict_write


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
            db = SyncDuckDBDatabase(
                {"path": str(Path(d) / "records.duckdb"), "table": "records"}
            )
            db.connect()
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()


def test_sync_upsert_batch_inserts_new(sync_db: object) -> None:
    """upsert_batch inserts new records and honors caller-supplied ids."""
    ids = sync_db.upsert_batch(
        [Record({"v": 1}, id="x"), Record({"v": 2}, id="y")]
    )
    assert ids == ["x", "y"]
    assert sync_db.read("x").get_value("v") == 1
    assert sync_db.read("y").get_value("v") == 2


def test_sync_upsert_batch_overwrites_existing(sync_db: object) -> None:
    """A colliding id is overwritten (not raised, not skipped) in one call."""
    sync_db.create(Record({"v": "old"}, id="dup"))
    ids = sync_db.upsert_batch(
        [
            Record({"v": "new"}, id="dup"),  # overwrite the existing row
            Record({"v": 3}, id="fresh"),  # and insert a new one
        ]
    )
    assert ids == ["dup", "fresh"]
    assert sync_db.read("dup").get_value("v") == "new"
    assert sync_db.read("fresh").get_value("v") == 3


def test_sync_upsert_batch_empty_is_noop(sync_db: object) -> None:
    """An empty batch returns an empty id list and writes nothing."""
    assert sync_db.upsert_batch([]) == []


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
            db = AsyncDuckDBDatabase(
                {"path": str(Path(d) / "records.duckdb"), "table": "records"}
            )
            await db.connect()
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                await close()


@pytest.mark.asyncio
async def test_async_upsert_batch_inserts_new(async_db: object) -> None:
    """upsert_batch inserts new records and honors caller-supplied ids."""
    ids = await async_db.upsert_batch(
        [Record({"v": 1}, id="x"), Record({"v": 2}, id="y")]
    )
    assert ids == ["x", "y"]
    assert (await async_db.read("x")).get_value("v") == 1
    assert (await async_db.read("y")).get_value("v") == 2


@pytest.mark.asyncio
async def test_async_upsert_batch_overwrites_existing(async_db: object) -> None:
    """A colliding id is overwritten (not raised, not skipped) in one call."""
    await async_db.create(Record({"v": "old"}, id="dup"))
    ids = await async_db.upsert_batch(
        [
            Record({"v": "new"}, id="dup"),
            Record({"v": 3}, id="fresh"),
        ]
    )
    assert ids == ["dup", "fresh"]
    assert (await async_db.read("dup")).get_value("v") == "new"
    assert (await async_db.read("fresh")).get_value("v") == 3


@pytest.mark.asyncio
async def test_async_upsert_batch_empty_is_noop(async_db: object) -> None:
    """An empty batch returns an empty id list and writes nothing."""
    assert await async_db.upsert_batch([]) == []


# ---------------------------------------------------------------------------
# Streaming adoption: the UPSERT policy now routes through upsert_batch
# ---------------------------------------------------------------------------
def test_resolver_upsert_uses_batch_func_when_supplied() -> None:
    """UPSERT returns the supplied ``upsert_batch_func`` as the batch verb."""
    sentinel_batch = object()
    batch, single, skip = resolve_conflict_write(
        ConflictPolicy.UPSERT,
        insert_batch_func=None,
        single_create_func="create",
        upsert_func="upsert",
        upsert_batch_func=sentinel_batch,
    )
    assert batch is sentinel_batch  # native bulk verb taken
    assert single == "upsert"  # per-record fallback is upsert
    assert skip is False


def test_resolver_upsert_backcompat_without_batch_func() -> None:
    """Without an ``upsert_batch_func`` UPSERT stays per-record (back-compat)."""
    batch, single, skip = resolve_conflict_write(
        ConflictPolicy.UPSERT,
        insert_batch_func="ins",
        single_create_func="create",
        upsert_func="upsert",
    )
    assert batch is None  # no batch attempt, as before
    assert single == "upsert"
    assert skip is False


def test_streaming_upsert_overwrites_via_batch_path(sync_db: object) -> None:
    """A streaming ``upsert`` write overwrites colliding ids without failure.

    Exercises the UPSERT → ``upsert_batch`` streaming path end-to-end: an id
    already present is overwritten, and no record is counted as a failure.
    """
    sync_db.create(Record({"v": "old"}, id="a"))
    records = [
        Record({"v": "new-a"}, id="a"),  # collides — overwrite
        Record({"v": "b"}, id="b"),  # new
        Record({"v": "c"}, id="c"),  # new
    ]
    result = sync_db.stream_write(
        iter(records), StreamConfig(batch_size=10, on_conflict=ConflictPolicy.UPSERT)
    )
    assert result.failed == 0
    assert sync_db.read("a").get_value("v") == "new-a"
    assert sync_db.read("b").get_value("v") == "b"
    assert sync_db.read("c").get_value("v") == "c"
