"""Every ``create()`` / ``create_batch()`` mint fallback routes through ``_generate_id()``.

The write-keying rule is ``storage_id = record.id or <mint>``. A caller-supplied
``record.id`` is always honored; the ``<mint>`` fallback — used only when the
record carries no id — resolves through a single overridable hook,
``Database._generate_id()``, on the base ``SyncDatabase`` / ``AsyncDatabase``.
Overriding that one hook changes minted storage ids on **every** create path,
uniformly across all backends (the mint analogue of the honor-case invariant
pinned by ``test_create_payload_id_resolution.py``).

Real backends, no mocks: per-family sentinel subclasses override ``_generate_id``
to a recognizable prefix, and every in-process backend family (memory, file,
sqlite, duckdb — sync + async) is asserted to mint via the override for both
``create()`` and ``create_batch()``, while a caller-supplied id still bypasses
the hook. Service-backed backends (Postgres, S3, Elasticsearch) are covered
under ``tests/integration/`` behind their service markers.
"""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

BACKENDS = ["memory", "file", "sqlite", "duckdb"]

_PREFIX = "MINT-SENTINEL-"


def _sentinel_id() -> str:
    """A recognizable, collision-free minted id."""
    return f"{_PREFIX}{uuid.uuid4().hex}"


class _SentinelMixin:
    """Overrides the single mint hook a consumer would override for a custom
    id scheme. Mixed in first so its ``_generate_id`` wins the MRO.
    """

    def _generate_id(self) -> str:
        return _sentinel_id()


# One real sentinel subclass per in-process backend family, sync + async.
class _SentinelSyncMemory(_SentinelMixin, SyncMemoryDatabase):
    pass


class _SentinelSyncFile(_SentinelMixin, SyncFileDatabase):
    pass


class _SentinelSyncSQLite(_SentinelMixin, SyncSQLiteDatabase):
    pass


class _SentinelSyncDuckDB(_SentinelMixin, SyncDuckDBDatabase):
    pass


class _SentinelAsyncMemory(_SentinelMixin, AsyncMemoryDatabase):
    pass


class _SentinelAsyncFile(_SentinelMixin, AsyncFileDatabase):
    pass


class _SentinelAsyncSQLite(_SentinelMixin, AsyncSQLiteDatabase):
    pass


class _SentinelAsyncDuckDB(_SentinelMixin, AsyncDuckDBDatabase):
    pass


def _make_sync_db(kind: str, root: Path) -> object:
    if kind == "memory":
        return _SentinelSyncMemory()
    if kind == "file":
        return _SentinelSyncFile({"path": str(root / "records.json")})
    if kind == "sqlite":
        db = _SentinelSyncSQLite({"path": str(root / "records.db")})
        db.connect()
        return db
    db = _SentinelSyncDuckDB({"path": str(root / "records.duckdb"), "table": "records"})
    db.connect()
    return db


async def _make_async_db(kind: str, root: Path) -> object:
    if kind == "memory":
        return _SentinelAsyncMemory()
    if kind == "file":
        return _SentinelAsyncFile({"path": str(root / "records.json")})
    if kind == "sqlite":
        db = _SentinelAsyncSQLite({"path": str(root / "records.db")})
        await db.connect()
        return db
    db = _SentinelAsyncDuckDB({"path": str(root / "records.duckdb"), "table": "records"})
    await db.connect()
    return db


# ---------------------------------------------------------------------------
# Sync backends
# ---------------------------------------------------------------------------
@pytest.fixture(params=BACKENDS)
def sync_db(request: pytest.FixtureRequest) -> Iterator[object]:
    with tempfile.TemporaryDirectory() as d:
        db = _make_sync_db(request.param, Path(d))
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()


def test_sync_create_mints_via_hook(sync_db: object) -> None:
    """A record with no id mints its storage id through the sentinel hook."""
    new_id = sync_db.create(Record({"v": 1}))
    assert new_id.startswith(_PREFIX)
    assert sync_db.read(new_id).get_value("v") == 1


def test_sync_create_honors_caller_id_over_hook(sync_db: object) -> None:
    """A caller-supplied id is honored; the mint hook is not consulted."""
    new_id = sync_db.create(Record({"v": 1}, id="explicit"))
    assert new_id == "explicit"
    assert not new_id.startswith(_PREFIX)
    assert sync_db.read("explicit").get_value("v") == 1


def test_sync_create_batch_mints_via_hook(sync_db: object) -> None:
    """Every id-less record in a batch mints through the sentinel hook."""
    ids = sync_db.create_batch([Record({"v": i}) for i in range(3)])
    assert len(ids) == 3
    assert all(rid.startswith(_PREFIX) for rid in ids)
    assert len(set(ids)) == 3  # distinct minted ids


# ---------------------------------------------------------------------------
# Async backends
# ---------------------------------------------------------------------------
@pytest.fixture(params=BACKENDS)
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[object]:
    with tempfile.TemporaryDirectory() as d:
        db = await _make_async_db(request.param, Path(d))
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                result = close()
                if result is not None:
                    await result


async def test_async_create_mints_via_hook(async_db: object) -> None:
    new_id = await async_db.create(Record({"v": 1}))
    assert new_id.startswith(_PREFIX)
    got = await async_db.read(new_id)
    assert got.get_value("v") == 1


async def test_async_create_honors_caller_id_over_hook(async_db: object) -> None:
    new_id = await async_db.create(Record({"v": 1}, id="explicit"))
    assert new_id == "explicit"
    assert not new_id.startswith(_PREFIX)
    got = await async_db.read("explicit")
    assert got.get_value("v") == 1


async def test_async_create_batch_mints_via_hook(async_db: object) -> None:
    ids = await async_db.create_batch([Record({"v": i}) for i in range(3)])
    assert len(ids) == 3
    assert all(rid.startswith(_PREFIX) for rid in ids)
    assert len(set(ids)) == 3
