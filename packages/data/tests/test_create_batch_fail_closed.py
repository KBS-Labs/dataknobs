"""Atomic, fail-closed ``create_batch`` contract on the memory and file backends.

``create_batch()`` on the memory and file backends now honors the same
atomic-insert contract as ``create()``: a colliding id — against an existing
record or a duplicate within the same batch — raises ``DuplicateRecordError``
before *any* record in the batch is written, and the record's own id is
preserved. Previously memory and sync-file silently overwrote, and async-file
minted a fresh id and discarded ``record.id``.

Scope note: this covers only the two backends whose ``create_batch`` was brought
into line here. The SQLite / DuckDB / PostgreSQL / Elasticsearch bulk
``create_batch`` (and the S3 / PostgreSQL / Elasticsearch streaming
``_write_batch``) still mint a fresh id per record and do NOT fail closed; that
broader batch-verb tightening is tracked as a separate follow-up. What this PR
*does* guarantee end-to-end is that the **streaming INSERT** path fails closed
on all four in-process backend families (memory + file via this ``create_batch``
fix; SQLite + DuckDB because their ``stream_write`` routes INSERT through
per-record ``create()``) — see ``test_migrator_extended.py``.

This is the batch sibling of ``test_create_if_absent.py`` (single ``create()``).
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase


# ---------------------------------------------------------------------------
# Sync backends
# ---------------------------------------------------------------------------
@pytest.fixture(params=["memory", "file"])
def sync_db(request: pytest.FixtureRequest) -> Iterator[object]:
    """A connected sync backend whose create_batch is fail-closed."""
    kind = request.param
    with tempfile.TemporaryDirectory() as d:
        db: object
        if kind == "memory":
            db = SyncMemoryDatabase()
        else:
            db = SyncFileDatabase({"path": str(Path(d) / "records.json")})
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()


def test_sync_create_batch_duplicate_raises(sync_db: object) -> None:
    """create_batch fails closed when a record collides with an existing id."""
    sync_db.create(Record({"v": 1}, id="dup"))
    with pytest.raises(DuplicateRecordError) as excinfo:
        sync_db.create_batch([Record({"v": 2}, id="dup")])
    assert excinfo.value.id == "dup"


def test_sync_create_batch_is_atomic_on_collision(sync_db: object) -> None:
    """A batch containing one collision writes NONE of its records."""
    sync_db.create(Record({"v": "old"}, id="dup"))
    with pytest.raises(DuplicateRecordError):
        sync_db.create_batch(
            [
                Record({"v": 1}, id="new1"),
                Record({"v": 2}, id="dup"),  # collides — whole batch must abort
                Record({"v": 3}, id="new2"),
            ]
        )
    # Nothing from the batch was persisted; the existing row is untouched.
    assert sync_db.read("new1") is None
    assert sync_db.read("new2") is None
    assert sync_db.read("dup").get_value("v") == "old"


def test_sync_create_batch_within_batch_duplicate_raises(sync_db: object) -> None:
    """Two records sharing an id within one batch fail closed (nothing written)."""
    with pytest.raises(DuplicateRecordError):
        sync_db.create_batch(
            [Record({"v": 1}, id="same"), Record({"v": 2}, id="same")]
        )
    assert sync_db.read("same") is None


def test_sync_create_batch_preserves_ids(sync_db: object) -> None:
    """Distinct ids create normally and the given ids are honored (no minting)."""
    ids = sync_db.create_batch([Record({"v": 1}, id="x"), Record({"v": 2}, id="y")])
    assert ids == ["x", "y"]
    assert sync_db.read("x").get_value("v") == 1
    assert sync_db.read("y").get_value("v") == 2


# ---------------------------------------------------------------------------
# Async backends
# ---------------------------------------------------------------------------
@pytest.fixture(params=["memory", "file"])
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[object]:
    """A connected async backend whose create_batch is fail-closed."""
    kind = request.param
    with tempfile.TemporaryDirectory() as d:
        db: object
        if kind == "memory":
            db = AsyncMemoryDatabase()
        else:
            db = AsyncFileDatabase({"path": str(Path(d) / "records.json")})
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                await close()


@pytest.mark.asyncio
async def test_async_create_batch_duplicate_raises(async_db: object) -> None:
    """create_batch fails closed when a record collides with an existing id."""
    await async_db.create(Record({"v": 1}, id="dup"))
    with pytest.raises(DuplicateRecordError) as excinfo:
        await async_db.create_batch([Record({"v": 2}, id="dup")])
    assert excinfo.value.id == "dup"


@pytest.mark.asyncio
async def test_async_create_batch_is_atomic_on_collision(async_db: object) -> None:
    """A batch containing one collision writes NONE of its records."""
    await async_db.create(Record({"v": "old"}, id="dup"))
    with pytest.raises(DuplicateRecordError):
        await async_db.create_batch(
            [
                Record({"v": 1}, id="new1"),
                Record({"v": 2}, id="dup"),  # collides — whole batch must abort
                Record({"v": 3}, id="new2"),
            ]
        )
    assert await async_db.read("new1") is None
    assert await async_db.read("new2") is None
    got = await async_db.read("dup")
    assert got.get_value("v") == "old"


@pytest.mark.asyncio
async def test_async_create_batch_preserves_ids(async_db: object) -> None:
    """Distinct ids create normally and the given ids are honored (no minting).

    Regression: async file create_batch previously minted a fresh id per record
    and discarded ``record.id``.
    """
    ids = await async_db.create_batch(
        [Record({"v": 1}, id="x"), Record({"v": 2}, id="y")]
    )
    assert ids == ["x", "y"]
    assert (await async_db.read("x")).get_value("v") == 1
    assert (await async_db.read("y")).get_value("v") == 2
