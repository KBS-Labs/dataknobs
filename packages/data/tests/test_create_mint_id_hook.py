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


def test_sync_upsert_mints_via_hook(sync_db: object) -> None:
    """A record with no id upserts under a storage id minted via the hook."""
    new_id = sync_db.upsert(Record({"v": 1}))
    assert new_id.startswith(_PREFIX)
    assert sync_db.read(new_id).get_value("v") == 1


def test_sync_upsert_honors_explicit_id_over_hook(sync_db: object) -> None:
    """An explicit ``upsert(id, record)`` id wins; the hook is not consulted."""
    new_id = sync_db.upsert("explicit", Record({"v": 1}))
    assert new_id == "explicit"
    assert not new_id.startswith(_PREFIX)
    assert sync_db.read("explicit").get_value("v") == 1


def test_sync_upsert_batch_mints_via_hook(sync_db: object) -> None:
    """Every id-less record in an upsert batch mints through the sentinel hook."""
    ids = sync_db.upsert_batch([Record({"v": i}) for i in range(3)])
    assert len(ids) == 3
    assert all(rid.startswith(_PREFIX) for rid in ids)
    assert len(set(ids)) == 3


def test_sync_upsert_falsy_id_mints_matching_create_and_batch(sync_db: object) -> None:
    """A falsy ("") id is treated as absent and minted — not keyed under "".

    Pins the single-upsert convergence: ``upsert(Record(id=""))`` now mints via
    the hook (matching ``create`` and ``upsert_batch``) instead of keying the
    record under the empty string.
    """
    upsert_id = sync_db.upsert(Record({"v": 1}, id=""))
    assert upsert_id.startswith(_PREFIX)
    assert sync_db.read(upsert_id).get_value("v") == 1
    assert sync_db.read("") is None  # nothing keyed under ""

    # Parity with create() and upsert_batch() on the same falsy input.
    create_id = sync_db.create(Record({"v": 2}, id=""))
    assert create_id.startswith(_PREFIX)
    batch_ids = sync_db.upsert_batch([Record({"v": 3}, id="")])
    assert batch_ids[0].startswith(_PREFIX)


def test_sync_upsert_record_form_does_not_mutate_caller(sync_db: object) -> None:
    """``upsert(record)`` resolves the id on a copy; the caller's record is untouched.

    Reproduce-first: before the copy-first fix, ``_resolve_upsert_id`` stamped the
    minted id onto the caller's record in place, so ``rec.storage_id`` changed. The
    resolved id is available as ``upsert()``'s return value.
    """
    rec = Record({"v": 1})  # no id
    before = rec.storage_id
    new_id = sync_db.upsert(rec)
    assert new_id  # a fresh id was minted and returned
    assert rec.storage_id == before  # caller record NOT stamped
    assert sync_db.read(new_id).get_value("v") == 1


def test_sync_upsert_id_form_does_not_mutate_caller(sync_db: object) -> None:
    """``upsert(id, record)`` keys under the explicit id without stamping the caller.

    Reproduce-first: before the fix, the base method stamped ``id`` onto the
    caller's record (``record.storage_id = id``); after the fix it stamps a copy.
    """
    rec = Record({"v": 1})  # no id
    before = rec.storage_id
    new_id = sync_db.upsert("explicit", rec)
    assert new_id == "explicit"
    assert rec.storage_id == before  # caller record NOT stamped to "explicit"
    assert sync_db.read("explicit").get_value("v") == 1


def test_sync_write_methods_do_not_mutate_caller_record(sync_db: object) -> None:
    """Invariant lock: batch/create write methods never mutate the caller's record.

    Green both before and after the fix (create/create_batch/upsert_batch are
    already copy-first); pins that single ``upsert`` now matches these siblings.
    """
    for write in (
        sync_db.create,
        lambda r: sync_db.create_batch([r]),
        lambda r: sync_db.upsert_batch([r]),
    ):
        rec = Record({"v": 1})
        before = rec.storage_id
        write(rec)
        assert rec.storage_id == before


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


async def test_async_upsert_mints_via_hook(async_db: object) -> None:
    new_id = await async_db.upsert(Record({"v": 1}))
    assert new_id.startswith(_PREFIX)
    got = await async_db.read(new_id)
    assert got.get_value("v") == 1


async def test_async_upsert_honors_explicit_id_over_hook(async_db: object) -> None:
    new_id = await async_db.upsert("explicit", Record({"v": 1}))
    assert new_id == "explicit"
    assert not new_id.startswith(_PREFIX)
    got = await async_db.read("explicit")
    assert got.get_value("v") == 1


async def test_async_upsert_batch_mints_via_hook(async_db: object) -> None:
    ids = await async_db.upsert_batch([Record({"v": i}) for i in range(3)])
    assert len(ids) == 3
    assert all(rid.startswith(_PREFIX) for rid in ids)
    assert len(set(ids)) == 3


async def test_async_upsert_falsy_id_mints_matching_create_and_batch(
    async_db: object,
) -> None:
    """A falsy ("") id is treated as absent and minted — not keyed under ""."""
    upsert_id = await async_db.upsert(Record({"v": 1}, id=""))
    assert upsert_id.startswith(_PREFIX)
    got = await async_db.read(upsert_id)
    assert got.get_value("v") == 1
    assert await async_db.read("") is None  # nothing keyed under ""

    # Parity with create() and upsert_batch() on the same falsy input.
    create_id = await async_db.create(Record({"v": 2}, id=""))
    assert create_id.startswith(_PREFIX)
    batch_ids = await async_db.upsert_batch([Record({"v": 3}, id="")])
    assert batch_ids[0].startswith(_PREFIX)


async def test_async_upsert_record_form_does_not_mutate_caller(async_db: object) -> None:
    """``upsert(record)`` resolves the id on a copy; the caller's record is untouched."""
    rec = Record({"v": 1})  # no id
    before = rec.storage_id
    new_id = await async_db.upsert(rec)
    assert new_id  # a fresh id was minted and returned
    assert rec.storage_id == before  # caller record NOT stamped
    got = await async_db.read(new_id)
    assert got.get_value("v") == 1


async def test_async_upsert_id_form_does_not_mutate_caller(async_db: object) -> None:
    """``upsert(id, record)`` keys under the explicit id without stamping the caller."""
    rec = Record({"v": 1})  # no id
    before = rec.storage_id
    new_id = await async_db.upsert("explicit", rec)
    assert new_id == "explicit"
    assert rec.storage_id == before  # caller record NOT stamped to "explicit"
    got = await async_db.read("explicit")
    assert got.get_value("v") == 1


async def test_async_write_methods_do_not_mutate_caller_record(async_db: object) -> None:
    """Invariant lock: batch/create write methods never mutate the caller's record."""
    rec_c = Record({"v": 1})
    before_c = rec_c.storage_id
    await async_db.create(rec_c)
    assert rec_c.storage_id == before_c

    rec_cb = Record({"v": 2})
    before_cb = rec_cb.storage_id
    await async_db.create_batch([rec_cb])
    assert rec_cb.storage_id == before_cb

    rec_ub = Record({"v": 3})
    before_ub = rec_ub.storage_id
    await async_db.upsert_batch([rec_ub])
    assert rec_ub.storage_id == before_ub
