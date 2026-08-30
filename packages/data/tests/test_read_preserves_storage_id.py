"""A record a database hands back carries the id it was read under.

Every read path has to satisfy this, because ``record.id`` is what the caller
writes back with. A read that drops it turns the natural round trip ---
``read``, edit, ``update(record.id, record)`` --- into a write under ``None``,
or, for callers that fall back to ``create()``, into a **duplicate** of the
record that was meant to be replaced.

The base class already owns the answer: :meth:`Database._prepare_record_from_storage`
stamps the storage id onto whatever came out of the store. The bug this module
pins is a read path that does ``record.copy(deep=True)`` instead of calling it
--- which is invisible on a backend that happens to keep the id inside the
serialized payload, and fatal on one that does not.

Measured before the fix, on the same corpus:

===========================  ==========  ================
Backend                      ``read``    ``read_batch``
===========================  ==========  ================
``SyncMemoryDatabase``       **dropped** **dropped**
every other in-process       ok          ok
===========================  ==========  ================

So this is a contract module rather than a memory-backend module. The sync
memory backend is where it was broken today; the parametrization is what stops
the next backend breaking it unnoticed, and what caught that ``file``'s
``read_batch`` was passing by accident rather than by construction.

Backends needing an external service (Postgres, S3, Elasticsearch) are covered
under ``tests/integration/`` by the same rule.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.query import Query

BACKENDS = ["memory", "file", "sqlite", "duckdb"]


def _make_sync_db(kind: str, root: Path) -> Any:
    """Build and connect one in-process sync backend under ``root``."""
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


async def _make_async_db(kind: str, root: Path) -> Any:
    """Build and connect one in-process async backend under ``root``."""
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


@pytest.fixture(params=BACKENDS)
def sync_db(request: pytest.FixtureRequest) -> Iterator[Any]:
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
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[Any]:
    """A connected async backend, one per in-process backend family."""
    with tempfile.TemporaryDirectory() as d:
        db = await _make_async_db(request.param, Path(d))
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                await close()


class TestEveryReadPathCarriesTheId:
    """The id is on the record, whichever call produced it."""

    def test_sync_read(self, sync_db: Any) -> None:
        rid = sync_db.create(Record(data={"title": "hello"}))
        record = sync_db.read(rid)
        assert record is not None
        assert record.id == rid, "read() dropped the id it was called with"
        assert record.has_storage_id()

    def test_sync_read_batch(self, sync_db: Any) -> None:
        first = sync_db.create(Record(data={"title": "one"}))
        second = sync_db.create(Record(data={"title": "two"}))
        records = sync_db.read_batch([first, second])
        assert [r.id for r in records if r is not None] == [first, second]

    def test_sync_search(self, sync_db: Any) -> None:
        rid = sync_db.create(Record(data={"title": "hello"}))
        results = sync_db.search(Query())
        assert [r.id for r in results] == [rid]

    async def test_async_read(self, async_db: Any) -> None:
        rid = await async_db.create(Record(data={"title": "hello"}))
        record = await async_db.read(rid)
        assert record is not None
        assert record.id == rid, "read() dropped the id it was called with"
        assert record.has_storage_id()

    async def test_async_read_batch(self, async_db: Any) -> None:
        first = await async_db.create(Record(data={"title": "one"}))
        second = await async_db.create(Record(data={"title": "two"}))
        records = await async_db.read_batch([first, second])
        assert [r.id for r in records if r is not None] == [first, second]

    async def test_async_search(self, async_db: Any) -> None:
        rid = await async_db.create(Record(data={"title": "hello"}))
        results = await async_db.search(Query())
        assert [r.id for r in results] == [rid]


class TestTheRoundTripThatDependsOnIt:
    """Why the id matters: read, edit, write back under the same key."""

    def test_sync_read_edit_update_replaces_rather_than_duplicates(self, sync_db: Any) -> None:
        """The canonical caller. Without the id this writes nothing, or twice.

        ``bulk_embed_and_store`` is exactly this shape --- ``if record.id and
        exists(record.id): update(...) else: create(...)`` --- so a dropped id
        sends it down the ``create`` branch and stores a second copy of a
        record it was asked to update.
        """
        rid = sync_db.create(Record(data={"title": "before"}))

        record = sync_db.read(rid)
        assert record is not None
        record.set_value("title", "after")

        assert record.id, "nothing to update under: read() returned no id"
        assert sync_db.update(record.id, record) is True

        assert len(sync_db.all()) == 1, "the update stored a duplicate"
        assert sync_db.read(rid).get_value("title") == "after"

    async def test_async_read_edit_update_replaces_rather_than_duplicates(
        self, async_db: Any
    ) -> None:
        rid = await async_db.create(Record(data={"title": "before"}))

        record = await async_db.read(rid)
        assert record is not None
        record.set_value("title", "after")

        assert record.id, "nothing to update under: read() returned no id"
        assert await async_db.update(record.id, record) is True

        assert len(await async_db.all()) == 1, "the update stored a duplicate"
        assert (await async_db.read(rid)).get_value("title") == "after"


class TestACallerSuppliedIdIsNotOverwritten:
    """Restoring the id must not *replace* one the payload already carries.

    ``_prepare_record_from_storage`` stamps only when the record has no storage
    id, so a record whose id came out of its own payload keeps it. Pinning this
    is what makes the fix a restoration rather than an override.
    """

    def test_sync_payload_id_survives(self, sync_db: Any) -> None:
        sync_db.create(Record({"id": "chosen", "title": "hello"}))
        record = sync_db.read("chosen")
        assert record is not None
        assert record.id == "chosen"
