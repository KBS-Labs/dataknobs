"""Single ``create()`` resolves a payload ``id`` field the same as ``create_batch()``.

``create()`` and ``create_batch()`` — sync and async, on every backend — key a
record's storage id off ``record.id`` (the 5-step resolution): a caller-supplied
id is honored, a fresh uuid is minted only when the record carries no id, and a
colliding id fails closed with ``DuplicateRecordError``. ``upsert`` /
``on_conflict="upsert"`` remain the insert-or-overwrite path.

The sibling ``test_create_if_absent.py`` and ``test_create_batch_fail_closed*.py``
already pin this for a record whose id arrives via the ``id=`` constructor kwarg
(which sets ``storage_id``). This module pins the complementary and previously
divergent case: a record whose id is a **data field** — ``Record({"id": "x"})``
— carrying no ``storage_id``. For that record ``record.id`` resolves to ``"x"``
while ``has_storage_id()`` is ``False``; the two write-keying chokepoints (the
base ``_prepare_record_for_storage`` helper and the SQL ``build_create_query``
builder) key off ``record.id`` so the payload id is the storage key on every
backend and method, matching ``create_batch()``.

Backends that require an external service (Postgres, S3, Elasticsearch) are
covered under ``tests/integration/`` behind their service markers; Postgres and
Elasticsearch already honored ``record.id`` on single ``create()``, so the
service-side gap is S3 (see ``integration/test_create_if_absent_s3.py``).
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest

from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

BACKENDS = ["memory", "file", "sqlite", "duckdb"]


def _make_sync_db(kind: str, root: Path) -> object:
    """Build and connect one in-process sync backend in the existing dir ``root``."""
    if kind == "memory":
        return SyncMemoryDatabase()
    if kind == "file":
        return SyncFileDatabase({"path": str(root / "records.json")})
    if kind == "sqlite":
        db = SyncSQLiteDatabase({"path": str(root / "records.db")})
        db.connect()
        return db
    db = SyncDuckDBDatabase({"path": str(root / "records.duckdb"), "table": "records"})
    db.connect()
    return db


async def _make_async_db(kind: str, root: Path) -> object:
    """Build and connect one in-process async backend in the existing dir ``root``."""
    if kind == "memory":
        return AsyncMemoryDatabase()
    if kind == "file":
        return AsyncFileDatabase({"path": str(root / "records.json")})
    if kind == "sqlite":
        db = AsyncSQLiteDatabase({"path": str(root / "records.db")})
        await db.connect()
        return db
    db = AsyncDuckDBDatabase({"path": str(root / "records.duckdb"), "table": "records"})
    await db.connect()
    return db


# ---------------------------------------------------------------------------
# Sync backends
# ---------------------------------------------------------------------------
@pytest.fixture(params=BACKENDS)
def sync_db(request: pytest.FixtureRequest) -> Iterator[object]:
    """A connected sync backend, one per in-process backend family."""
    with tempfile.TemporaryDirectory() as d:
        db = _make_sync_db(request.param, Path(d))
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()


@pytest.fixture(params=BACKENDS)
def sync_db_factory(
    request: pytest.FixtureRequest,
) -> Iterator[Callable[[], object]]:
    """Yield a factory building fresh, independent sync dbs of one backend kind.

    Parity tests need two isolated stores of the *same* backend so the same
    record can be routed through ``create()`` on one and ``create_batch()`` on
    the other. All built dbs are closed on teardown.
    """
    kind = request.param
    with tempfile.TemporaryDirectory() as d:
        built: list[object] = []
        counter = {"n": 0}

        def factory() -> object:
            counter["n"] += 1
            sub = Path(d) / f"db{counter['n']}"
            sub.mkdir()
            db = _make_sync_db(kind, sub)
            built.append(db)
            return db

        try:
            yield factory
        finally:
            for db in built:
                close = getattr(db, "close", None)
                if callable(close):
                    close()


def test_sync_create_honors_payload_id_field(sync_db: object) -> None:
    """A payload ``id`` data field is the storage key (not a minted uuid)."""
    returned = sync_db.create(Record({"id": "x", "v": 1}))
    assert returned == "x"
    assert sync_db.read("x").get_value("v") == 1


def test_sync_create_honors_payload_record_id_field(sync_db: object) -> None:
    """A payload ``record_id`` field is honored the same as ``id`` (5-step fallback)."""
    returned = sync_db.create(Record({"record_id": "rid", "v": 2}))
    assert returned == "rid"
    assert sync_db.read("rid").get_value("v") == 2


def test_sync_create_mints_when_no_id_present(sync_db: object) -> None:
    """A record carrying no id at all still gets a minted, readable storage id."""
    returned = sync_db.create(Record({"v": 3}))
    assert returned  # non-empty minted id
    assert sync_db.read(returned).get_value("v") == 3


def test_sync_create_payload_id_fails_closed_on_collision(sync_db: object) -> None:
    """A second create() with the same payload id fails closed (no overwrite)."""
    sync_db.create(Record({"id": "x", "v": "winner"}))
    with pytest.raises(DuplicateRecordError) as excinfo:
        sync_db.create(Record({"id": "x", "v": "loser"}))
    assert excinfo.value.id == "x"
    assert sync_db.read("x").get_value("v") == "winner"


def test_sync_create_and_create_batch_agree_on_payload_id(
    sync_db_factory: Callable[[], object],
) -> None:
    """The same record keys to the same storage id via create() and create_batch()."""
    db_single = sync_db_factory()
    db_batch = sync_db_factory()
    single_id = db_single.create(Record({"id": "x", "v": 1}))
    batch_ids = db_batch.create_batch([Record({"id": "x", "v": 1})])
    assert single_id == "x"
    assert batch_ids == ["x"]


def test_sync_explicit_storage_id_kwarg_still_honored(sync_db: object) -> None:
    """The ``id=`` / ``storage_id=`` kwarg path keeps its key (regression guard)."""
    returned = sync_db.create(Record({"v": 1}, id="chosen"))
    assert returned == "chosen"
    assert sync_db.read("chosen").get_value("v") == 1


def test_sync_read_back_preserves_business_id_field(sync_db: object) -> None:
    """Honoring the payload id for keying does not strip it from the record data."""
    sync_db.create(Record({"id": "x", "v": 1}))
    got = sync_db.read("x")
    assert got.get_value("id") == "x"
    assert got.get_value("v") == 1


# ---------------------------------------------------------------------------
# Async backends
# ---------------------------------------------------------------------------
@pytest.fixture(params=BACKENDS)
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[object]:
    """A connected async backend, one per in-process backend family."""
    with tempfile.TemporaryDirectory() as d:
        db = await _make_async_db(request.param, Path(d))
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                await close()


@pytest.fixture(params=BACKENDS)
async def async_db_factory(
    request: pytest.FixtureRequest,
) -> AsyncIterator[Callable[[], object]]:
    """Yield an async factory building fresh, independent async dbs of one kind.

    ``factory()`` returns an awaitable resolving to a connected db (async
    construction cannot happen inside a plain callable). All built dbs are
    closed on teardown.
    """
    kind = request.param
    with tempfile.TemporaryDirectory() as d:
        built: list[object] = []
        counter = {"n": 0}

        async def factory() -> object:
            counter["n"] += 1
            sub = Path(d) / f"db{counter['n']}"
            sub.mkdir()
            db = await _make_async_db(kind, sub)
            built.append(db)
            return db

        try:
            yield factory
        finally:
            for db in built:
                close = getattr(db, "close", None)
                if callable(close):
                    await close()


@pytest.mark.asyncio
async def test_async_create_honors_payload_id_field(async_db: object) -> None:
    """A payload ``id`` data field is the storage key (not a minted uuid)."""
    returned = await async_db.create(Record({"id": "x", "v": 1}))
    assert returned == "x"
    assert (await async_db.read("x")).get_value("v") == 1


@pytest.mark.asyncio
async def test_async_create_honors_payload_record_id_field(async_db: object) -> None:
    """A payload ``record_id`` field is honored the same as ``id`` (5-step fallback)."""
    returned = await async_db.create(Record({"record_id": "rid", "v": 2}))
    assert returned == "rid"
    assert (await async_db.read("rid")).get_value("v") == 2


@pytest.mark.asyncio
async def test_async_create_mints_when_no_id_present(async_db: object) -> None:
    """A record carrying no id at all still gets a minted, readable storage id."""
    returned = await async_db.create(Record({"v": 3}))
    assert returned  # non-empty minted id
    assert (await async_db.read(returned)).get_value("v") == 3


@pytest.mark.asyncio
async def test_async_create_payload_id_fails_closed_on_collision(
    async_db: object,
) -> None:
    """A second create() with the same payload id fails closed (no overwrite)."""
    await async_db.create(Record({"id": "x", "v": "winner"}))
    with pytest.raises(DuplicateRecordError) as excinfo:
        await async_db.create(Record({"id": "x", "v": "loser"}))
    assert excinfo.value.id == "x"
    assert (await async_db.read("x")).get_value("v") == "winner"


@pytest.mark.asyncio
async def test_async_create_and_create_batch_agree_on_payload_id(
    async_db_factory: Callable[[], object],
) -> None:
    """The same record keys to the same storage id via create() and create_batch()."""
    db_single = await async_db_factory()
    db_batch = await async_db_factory()
    single_id = await db_single.create(Record({"id": "x", "v": 1}))
    batch_ids = await db_batch.create_batch([Record({"id": "x", "v": 1})])
    assert single_id == "x"
    assert batch_ids == ["x"]


@pytest.mark.asyncio
async def test_async_explicit_storage_id_kwarg_still_honored(async_db: object) -> None:
    """The ``id=`` / ``storage_id=`` kwarg path keeps its key (regression guard)."""
    returned = await async_db.create(Record({"v": 1}, id="chosen"))
    assert returned == "chosen"
    assert (await async_db.read("chosen")).get_value("v") == 1


@pytest.mark.asyncio
async def test_async_read_back_preserves_business_id_field(async_db: object) -> None:
    """Honoring the payload id for keying does not strip it from the record data."""
    await async_db.create(Record({"id": "x", "v": 1}))
    got = await async_db.read("x")
    assert got.get_value("id") == "x"
    assert got.get_value("v") == 1
