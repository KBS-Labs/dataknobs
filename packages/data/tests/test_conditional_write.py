"""Optimistic-concurrency (conditional write) contract across backends.

``update()`` / ``upsert()`` / ``delete()`` accept an optional
``expected_version`` token read back from ``get_version(id)``. When supplied,
the write is a compare-and-set: it proceeds only if the record's current token
still matches, otherwise it raises ``ConcurrencyError`` instead of
last-writer-wins (a conditional ``delete`` of a missing record returns
``False`` — an absent id never conflicts). Omitting ``expected_version``
preserves the unconditional write, byte-identical to prior behavior.

Reproduce-first framing: ``test_*_unconditional_update_still_clobbers`` pins the
*old* behavior CAS guards against (a blind second write silently overwrites the
first). The conditional tests then show that reading a token and passing it back
converts that clobber into a raised ``ConcurrencyError``.

The in-process backends (memory, file, SQLite, DuckDB) are covered here. Memory
mints an ABA-safe monotonic counter token; file/SQLite/DuckDB use the
content-hash default. Service-backed backends (Postgres, S3, Elasticsearch) are
covered under ``tests/integration/`` behind their service markers.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from dataknobs_data import ConcurrencyError, Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

# In-process backend families. Memory mints an ABA-safe monotonic counter
# token; file/SQLite/DuckDB use the content-hash default. All four serialize
# conditional writes within a single instance, so the concurrent test runs
# across all of them.
_SYNC_PARAMS = ["memory", "file", "sqlite", "duckdb"]
_ASYNC_PARAMS = ["memory", "file", "sqlite", "duckdb"]


# ---------------------------------------------------------------------------
# Sync backends
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


def _seed(db: object, id: str = "k") -> str:
    """Create a record and return its id."""
    db.create(Record({"v": 0}, id=id))
    return id


def test_sync_get_version_absent_is_none(sync_db: object) -> None:
    """get_version returns None for an id that does not exist."""
    assert sync_db.get_version("missing") is None


def test_sync_get_version_present(sync_db: object) -> None:
    """A stored record has a non-None opaque token."""
    id = _seed(sync_db)
    assert sync_db.get_version(id) is not None


def test_sync_unconditional_update_still_clobbers(sync_db: object) -> None:
    """Without expected_version, a blind write overwrites (the guarded behavior)."""
    id = _seed(sync_db)
    assert sync_db.update(id, Record({"v": 1}, id=id)) is True
    assert sync_db.update(id, Record({"v": 2}, id=id)) is True
    assert sync_db.read(id).get_value("v") == 2


def test_sync_conditional_update_fresh_token_succeeds(sync_db: object) -> None:
    """A conditional update with the current token succeeds and bumps the token."""
    id = _seed(sync_db)
    token = sync_db.get_version(id)
    assert sync_db.update(id, Record({"v": 1}, id=id), expected_version=token) is True
    assert sync_db.read(id).get_value("v") == 1
    assert sync_db.get_version(id) != token


def test_sync_conditional_update_stale_token_raises(sync_db: object) -> None:
    """A second writer holding the pre-update token loses with ConcurrencyError."""
    id = _seed(sync_db)
    stale = sync_db.get_version(id)
    # Writer A commits, advancing the token.
    sync_db.update(id, Record({"v": "A"}, id=id), expected_version=stale)
    # Writer B still holds the stale token.
    with pytest.raises(ConcurrencyError) as excinfo:
        sync_db.update(id, Record({"v": "B"}, id=id), expected_version=stale)
    assert excinfo.value.context["id"] == id
    assert excinfo.value.context["expected_version"] == stale
    # A's write survives; B's is rejected.
    assert sync_db.read(id).get_value("v") == "A"


def test_sync_token_roundtrip(sync_db: object) -> None:
    """Re-reading the token after a conditional write lets the next write proceed."""
    id = _seed(sync_db)
    t1 = sync_db.get_version(id)
    sync_db.update(id, Record({"v": 1}, id=id), expected_version=t1)
    t2 = sync_db.get_version(id)
    assert t2 != t1
    assert sync_db.update(id, Record({"v": 2}, id=id), expected_version=t2) is True


def test_sync_conditional_upsert_stale_raises(sync_db: object) -> None:
    """A conditional upsert with a stale token raises rather than overwriting."""
    id = _seed(sync_db)
    stale = sync_db.get_version(id)
    sync_db.update(id, Record({"v": "A"}, id=id), expected_version=stale)
    with pytest.raises(ConcurrencyError):
        sync_db.upsert(id, Record({"v": "B"}, id=id), expected_version=stale)
    assert sync_db.read(id).get_value("v") == "A"


def test_sync_conditional_upsert_absent_raises(sync_db: object) -> None:
    """A conditional upsert never inserts: an absent target is a conflict."""
    with pytest.raises(ConcurrencyError) as excinfo:
        sync_db.upsert("ghost", Record({"v": 1}, id="ghost"), expected_version="1")
    assert excinfo.value.context["actual_version"] is None


def test_sync_unconditional_upsert_still_inserts(sync_db: object) -> None:
    """Omitting expected_version keeps upsert's insert-or-update behavior."""
    rid = sync_db.upsert("new", Record({"v": 7}, id="new"))
    assert sync_db.read(rid).get_value("v") == 7


def test_sync_conditional_delete_fresh_token_succeeds(sync_db: object) -> None:
    """A conditional delete with the current token removes the record."""
    id = _seed(sync_db)
    token = sync_db.get_version(id)
    assert sync_db.delete(id, expected_version=token) is True
    assert sync_db.read(id) is None


def test_sync_conditional_delete_stale_token_raises(sync_db: object) -> None:
    """A conditional delete with a stale token raises and leaves the record."""
    id = _seed(sync_db)
    stale = sync_db.get_version(id)
    # An interleaved writer advances the token.
    sync_db.update(id, Record({"v": "A"}, id=id), expected_version=stale)
    with pytest.raises(ConcurrencyError) as excinfo:
        sync_db.delete(id, expected_version=stale)
    assert excinfo.value.context["id"] == id
    assert excinfo.value.context["expected_version"] == stale
    # The stale-token delete did not remove the record.
    assert sync_db.read(id) is not None


def test_sync_conditional_delete_missing_returns_false(sync_db: object) -> None:
    """A conditional delete of an absent id returns False (never conflicts)."""
    assert sync_db.delete("ghost", expected_version="whatever") is False


def test_sync_unconditional_delete_still_works(sync_db: object) -> None:
    """Omitting expected_version keeps delete's unconditional behavior."""
    id = _seed(sync_db)
    assert sync_db.delete(id) is True
    assert sync_db.read(id) is None


def test_sync_memory_delete_recreate_is_aba_safe() -> None:
    """Memory's monotonic token survives a delete->recreate ABA cycle.

    Reproduce-first: the old per-key counter reset to 1 on ``create`` and was
    popped on ``delete``, so a token read before a delete->recreate cycle
    (value ``"1"``) matched the recreated record's token (also ``"1"``) — a
    stale conditional write silently succeeded. The record is recreated with
    *identical* content here so a content-hash token would also collide; only a
    never-reused monotonic sequence detects the change, which is exactly memory's
    stated guarantee.
    """
    db = SyncMemoryDatabase()
    db.create(Record({"v": 0}, id="k"))
    stale = db.get_version("k")
    db.delete("k")
    db.create(Record({"v": 0}, id="k"))  # identical content, same id
    with pytest.raises(ConcurrencyError):
        db.update("k", Record({"v": 9}, id="k"), expected_version=stale)
    # The recreated record is untouched by the stale-token write.
    assert db.read("k").get_value("v") == 0


# ---------------------------------------------------------------------------
# Async backends
# ---------------------------------------------------------------------------
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


async def _aseed(db: object, id: str = "k") -> str:
    await db.create(Record({"v": 0}, id=id))
    return id


@pytest.mark.asyncio
async def test_async_get_version_absent_is_none(async_db: object) -> None:
    """get_version returns None for an id that does not exist."""
    assert await async_db.get_version("missing") is None


@pytest.mark.asyncio
async def test_async_conditional_update_fresh_token_succeeds(async_db: object) -> None:
    """A conditional update with the current token succeeds and bumps the token."""
    id = await _aseed(async_db)
    token = await async_db.get_version(id)
    assert await async_db.update(id, Record({"v": 1}, id=id), expected_version=token) is True
    got = await async_db.read(id)
    assert got.get_value("v") == 1
    assert await async_db.get_version(id) != token


@pytest.mark.asyncio
async def test_async_conditional_update_stale_token_raises(async_db: object) -> None:
    """A second writer holding the pre-update token loses with ConcurrencyError."""
    id = await _aseed(async_db)
    stale = await async_db.get_version(id)
    await async_db.update(id, Record({"v": "A"}, id=id), expected_version=stale)
    with pytest.raises(ConcurrencyError) as excinfo:
        await async_db.update(id, Record({"v": "B"}, id=id), expected_version=stale)
    assert excinfo.value.context["id"] == id
    got = await async_db.read(id)
    assert got.get_value("v") == "A"


@pytest.mark.asyncio
async def test_async_unconditional_update_still_clobbers(async_db: object) -> None:
    """Without expected_version, a blind write overwrites (the guarded behavior)."""
    id = await _aseed(async_db)
    await async_db.update(id, Record({"v": 1}, id=id))
    await async_db.update(id, Record({"v": 2}, id=id))
    got = await async_db.read(id)
    assert got.get_value("v") == 2


@pytest.mark.asyncio
async def test_async_concurrent_conditional_update_one_wins(async_db: object) -> None:
    """Concurrent conditional updates on one token: exactly one wins."""
    id = await _aseed(async_db)
    token = await async_db.get_version(id)
    results = await asyncio.gather(
        async_db.update(id, Record({"who": "a"}, id=id), expected_version=token),
        async_db.update(id, Record({"who": "b"}, id=id), expected_version=token),
        async_db.update(id, Record({"who": "c"}, id=id), expected_version=token),
        return_exceptions=True,
    )
    successes = [r for r in results if r is True]
    conflicts = [r for r in results if isinstance(r, ConcurrencyError)]
    assert len(successes) == 1
    assert len(conflicts) == 2
    got = await async_db.read(id)
    assert got.get_value("who") in {"a", "b", "c"}


@pytest.mark.asyncio
async def test_async_conditional_delete_fresh_token_succeeds(async_db: object) -> None:
    """A conditional delete with the current token removes the record."""
    id = await _aseed(async_db)
    token = await async_db.get_version(id)
    assert await async_db.delete(id, expected_version=token) is True
    assert await async_db.read(id) is None


@pytest.mark.asyncio
async def test_async_conditional_delete_stale_token_raises(async_db: object) -> None:
    """A conditional delete with a stale token raises and leaves the record."""
    id = await _aseed(async_db)
    stale = await async_db.get_version(id)
    await async_db.update(id, Record({"v": "A"}, id=id), expected_version=stale)
    with pytest.raises(ConcurrencyError) as excinfo:
        await async_db.delete(id, expected_version=stale)
    assert excinfo.value.context["id"] == id
    assert await async_db.read(id) is not None


@pytest.mark.asyncio
async def test_async_conditional_delete_missing_returns_false(async_db: object) -> None:
    """A conditional delete of an absent id returns False (never conflicts)."""
    assert await async_db.delete("ghost", expected_version="whatever") is False


@pytest.mark.asyncio
async def test_async_concurrent_conditional_delete_one_wins(async_db: object) -> None:
    """Concurrent conditional deletes on one token: exactly one wins."""
    id = await _aseed(async_db)
    token = await async_db.get_version(id)
    results = await asyncio.gather(
        async_db.delete(id, expected_version=token),
        async_db.delete(id, expected_version=token),
        async_db.delete(id, expected_version=token),
        return_exceptions=True,
    )
    successes = [r for r in results if r is True]
    # The losers either raise ConcurrencyError (record still present at the
    # stale token) or return False (record already gone) — never a silent
    # second successful delete.
    assert len(successes) == 1
    assert await async_db.read(id) is None


@pytest.mark.asyncio
async def test_async_memory_delete_recreate_is_aba_safe() -> None:
    """Memory's monotonic token survives a delete->recreate ABA cycle (async)."""
    db = AsyncMemoryDatabase()
    await db.create(Record({"v": 0}, id="k"))
    stale = await db.get_version("k")
    await db.delete("k")
    await db.create(Record({"v": 0}, id="k"))  # identical content, same id
    with pytest.raises(ConcurrencyError):
        await db.update("k", Record({"v": 9}, id="k"), expected_version=stale)
    got = await db.read("k")
    assert got.get_value("v") == 0
