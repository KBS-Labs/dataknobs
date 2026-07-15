"""Seamless monotonic-key allocation contract across backends.

``allocate`` / ``allocate_sync`` wrap the atomic ``create()`` in a bounded
create-on-conflict loop: a caller-supplied ``build`` callable does a fresh read,
computes the next monotonic key, and returns a ``Record`` under it; on a
colliding id the helper re-runs ``build`` and retries, so concurrent allocators
each land a distinct next key instead of one failing closed.

Reproduce-first framing: ``test_*_direct_create_race_fails_closed`` pins the
friction the helper closes — two allocators that ``create()`` the same computed
key directly (no retry) leave one raising ``DuplicateRecordError`` with its
record unwritten. The ``allocate`` tests then show the same two allocators both
succeed under distinct consecutive keys.

The in-process backends (memory, file, SQLite, DuckDB) are covered here.
Service-backed backends (Postgres, S3, Elasticsearch) ride the same helper —
it composes over the shared ``create()`` contract — and are exercised by the
create-semantics integration suite; no allocation-specific service test is
needed because the helper adds no backend-specific path.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from dataknobs_data import DuplicateRecordError, Record, allocate, allocate_sync
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

_STEM = "doc"
_SYNC_PARAMS = ["memory", "file", "sqlite", "duckdb"]
_ASYNC_PARAMS = ["memory", "file", "sqlite", "duckdb"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=_SYNC_PARAMS)
def sync_db(request: pytest.FixtureRequest) -> Iterator[object]:
    """A connected sync backend, one per realized in-process backend family."""
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


@pytest.fixture(params=_ASYNC_PARAMS)
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[object]:
    """A connected async backend, one per realized in-process backend family."""
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


# ---------------------------------------------------------------------------
# build closures — a genuine fresh read + next-key computation each call
# ---------------------------------------------------------------------------
def _sync_highest_version(db: object, stem: str) -> int:
    """Probe consecutive ``<stem>-v{N}`` keys until a gap; return the highest N."""
    n = 0
    while db.exists(f"{stem}-v{n + 1}"):
        n += 1
    return n


async def _async_highest_version(db: object, stem: str) -> int:
    n = 0
    while await db.exists(f"{stem}-v{n + 1}"):
        n += 1
    return n


def _sync_build(db: object, stem: str = _STEM) -> Record:
    """Fresh read → next version → record under ``<stem>-v{N}``."""
    n = _sync_highest_version(db, stem)
    return Record({"body": f"content-{n + 1}"}, id=f"{stem}-v{n + 1}")


async def _async_build(db: object, stem: str = _STEM) -> Record:
    n = await _async_highest_version(db, stem)
    return Record({"body": f"content-{n + 1}"}, id=f"{stem}-v{n + 1}")


# ---------------------------------------------------------------------------
# RED — direct create races fail closed (pins the friction the helper closes)
# ---------------------------------------------------------------------------
def test_sync_direct_create_race_fails_closed(sync_db: object) -> None:
    """Two allocators computing the same key from a shared pre-read snapshot and
    creating directly (no retry): one wins, the other fails closed and its
    record is not written.
    """
    # Both read the same empty state and compute the same next key.
    record_a = _sync_build(sync_db)
    record_b = _sync_build(sync_db)
    assert record_a.id == record_b.id == f"{_STEM}-v1"

    sync_db.create(record_a)
    with pytest.raises(DuplicateRecordError) as excinfo:
        sync_db.create(record_b)
    assert excinfo.value.id == f"{_STEM}-v1"

    # Only the winner's record is stored; the loser's body never landed.
    stored = sync_db.read(f"{_STEM}-v1")
    assert stored.get_value("body") == "content-1"
    assert sync_db.exists(f"{_STEM}-v2") is False


@pytest.mark.asyncio
async def test_async_direct_create_race_fails_closed(async_db: object) -> None:
    record_a = await _async_build(async_db)
    record_b = await _async_build(async_db)
    assert record_a.id == record_b.id == f"{_STEM}-v1"

    await async_db.create(record_a)
    with pytest.raises(DuplicateRecordError) as excinfo:
        await async_db.create(record_b)
    assert excinfo.value.id == f"{_STEM}-v1"

    stored = await async_db.read(f"{_STEM}-v1")
    assert stored.get_value("body") == "content-1"
    assert await async_db.exists(f"{_STEM}-v2") is False


# ---------------------------------------------------------------------------
# GREEN — seamless allocation through allocate()
# ---------------------------------------------------------------------------
def test_sync_allocate_seamless_sequential(sync_db: object) -> None:
    """Repeated allocation through the helper lands consecutive keys."""
    ids = [allocate_sync(sync_db, build=lambda: _sync_build(sync_db)) for _ in range(3)]
    assert ids == [f"{_STEM}-v1", f"{_STEM}-v2", f"{_STEM}-v3"]
    for n, id in enumerate(ids, start=1):
        assert sync_db.read(id).get_value("body") == f"content-{n}"


@pytest.mark.asyncio
async def test_async_allocate_seamless_concurrent(async_db: object) -> None:
    """Two allocators via asyncio.gather each land a distinct consecutive key;
    the loser's collision is retried into the next slot rather than raised.
    """
    ids = await asyncio.gather(
        allocate(async_db, build=lambda: _async_build(async_db)),
        allocate(async_db, build=lambda: _async_build(async_db)),
    )
    assert sorted(ids) == [f"{_STEM}-v1", f"{_STEM}-v2"]
    assert await async_db.exists(f"{_STEM}-v1") is True
    assert await async_db.exists(f"{_STEM}-v2") is True
    bodies = {(await async_db.read(id)).get_value("body") for id in ids}
    assert bodies == {"content-1", "content-2"}


# ---------------------------------------------------------------------------
# Bounded exhaustion + retry-count (deterministic, real counting closures)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_async_allocate_exhausts_after_max_attempts(async_db: object) -> None:
    """A build that always returns the same colliding id exhausts the bound and
    re-raises, calling build exactly max_attempts times — never infinite.
    """
    await async_db.create(Record({"body": "x"}, id="fixed"))
    calls = 0

    async def build() -> Record:
        nonlocal calls
        calls += 1
        return Record({"body": "y"}, id="fixed")

    with pytest.raises(DuplicateRecordError):
        await allocate(async_db, build=build, max_attempts=3)
    assert calls == 3


@pytest.mark.asyncio
async def test_async_allocate_recovers_on_second_attempt(async_db: object) -> None:
    """A build that collides once then computes a fresh key succeeds on attempt
    two, proving a collision triggers a re-build rather than aborting.
    """
    await async_db.create(Record({"body": "x"}, id="taken"))
    calls = 0

    async def build() -> Record:
        nonlocal calls
        calls += 1
        id = "taken" if calls == 1 else "free"
        return Record({"body": "y"}, id=id)

    result = await allocate(async_db, build=build, max_attempts=3)
    assert result == "free"
    assert calls == 2


def test_sync_allocate_exhausts_after_max_attempts(sync_db: object) -> None:
    sync_db.create(Record({"body": "x"}, id="fixed"))
    calls = 0

    def build() -> Record:
        nonlocal calls
        calls += 1
        return Record({"body": "y"}, id="fixed")

    with pytest.raises(DuplicateRecordError):
        allocate_sync(sync_db, build=build, max_attempts=3)
    assert calls == 3


def test_sync_allocate_recovers_on_second_attempt(sync_db: object) -> None:
    sync_db.create(Record({"body": "x"}, id="taken"))
    calls = 0

    def build() -> Record:
        nonlocal calls
        calls += 1
        id = "taken" if calls == 1 else "free"
        return Record({"body": "y"}, id=id)

    result = allocate_sync(sync_db, build=build, max_attempts=3)
    assert result == "free"
    assert calls == 2


# ---------------------------------------------------------------------------
# Single-writer byte-identity — one attempt, same stored record as direct create
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_async_allocate_single_writer_one_attempt(async_db: object) -> None:
    calls = 0

    async def build() -> Record:
        nonlocal calls
        calls += 1
        return Record({"body": "only"}, id=f"{_STEM}-v1")

    result = await allocate(async_db, build=build)
    assert result == f"{_STEM}-v1"
    assert calls == 1
    assert (await async_db.read(f"{_STEM}-v1")).get_value("body") == "only"


def test_sync_allocate_single_writer_one_attempt(sync_db: object) -> None:
    calls = 0

    def build() -> Record:
        nonlocal calls
        calls += 1
        return Record({"body": "only"}, id=f"{_STEM}-v1")

    result = allocate_sync(sync_db, build=build)
    assert result == f"{_STEM}-v1"
    assert calls == 1
    assert sync_db.read(f"{_STEM}-v1").get_value("body") == "only"


# ---------------------------------------------------------------------------
# max_attempts < 1 guard — fail loud, never silently loop or succeed
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_async_allocate_rejects_nonpositive_max_attempts(async_db: object) -> None:
    calls = 0

    async def build() -> Record:
        nonlocal calls
        calls += 1
        return Record({"body": "y"}, id="k")

    with pytest.raises(ValueError, match="max_attempts"):
        await allocate(async_db, build=build, max_attempts=0)
    assert calls == 0  # guarded before any build/create


def test_sync_allocate_rejects_nonpositive_max_attempts(sync_db: object) -> None:
    calls = 0

    def build() -> Record:
        nonlocal calls
        calls += 1
        return Record({"body": "y"}, id="k")

    with pytest.raises(ValueError, match="max_attempts"):
        allocate_sync(sync_db, build=build, max_attempts=-1)
    assert calls == 0
