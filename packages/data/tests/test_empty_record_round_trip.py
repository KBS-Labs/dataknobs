"""A record that is stored can be read back, including one with no fields.

``Database._prepare_record_from_storage`` is the shared helper every read path
routes through --- it stamps the storage id onto whatever came out of the store
and returns ``None`` when nothing did. It decided which case it was on
``if record:``, and ``Record`` defines ``__len__``, so **a record with no fields
is falsy**. An empty record was therefore stored successfully and read back as
``None``, on every backend at once, because they all share the helper.

The incoherence is visible without knowing any of that: ``exists`` and ``read``
disagree. Measured before the fix, on three backends:

==========  ==========  =========  =========
Backend     ``create``  ``exists`` ``read``
==========  ==========  =========  =========
memory      an id       ``True``   ``None``
file        an id       ``True``   ``None``
sqlite      an id       ``True``   ``None``
==========  ==========  =========  =========

An empty record is not exotic --- a placeholder created before its fields are
known, or one whose fields have all been cleared, is one. And a caller cannot
tell this apart from "no such record", so the natural recovery is to create it
again, which is how one lost record becomes two.

The fix is ``is not None`` in the helper. These cells are a contract over the
backends rather than a test of the helper, because the helper is not what a
caller touches and because a backend that stops calling it would pass a test
written against it while failing this one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_data import Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

BACKENDS = ["memory", "file", "sqlite", "duckdb"]


def _make_sync_db(kind: str, root: Path) -> Any:
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
    with tempfile.TemporaryDirectory() as d:
        db = await _make_async_db(request.param, Path(d))
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                await close()


class TestAnEmptyRecordIsStillARecord:
    """``exists`` and ``read`` have to agree, whatever the record holds."""

    def test_sync_read(self, sync_db: Any) -> None:
        rid = sync_db.create(Record(data={}))
        assert sync_db.exists(rid) is True
        assert sync_db.read(rid) is not None, "stored, reported present, read back as absent"

    async def test_async_read(self, async_db: Any) -> None:
        rid = await async_db.create(Record(data={}))
        assert await async_db.exists(rid) is True
        assert await async_db.read(rid) is not None

    def test_sync_read_batch(self, sync_db: Any) -> None:
        """A batch read is not a weaker contract than a single one."""
        rid = sync_db.create(Record(data={}))
        assert sync_db.read_batch([rid])[0] is not None

    async def test_async_read_batch(self, async_db: Any) -> None:
        rid = await async_db.create(Record(data={}))
        assert (await async_db.read_batch([rid]))[0] is not None

    def test_the_id_is_still_stamped(self, sync_db: Any) -> None:
        """The helper's actual job, on the record that used to skip it."""
        rid = sync_db.create(Record(data={}))
        assert sync_db.read(rid).id == rid


class TestAMissingRecordIsStillMissing:
    """The guard on the fix.

    ``is not None`` is a wider condition than ``if record``, so the risk it
    introduces is at the other end: an id nothing is stored under must go on
    answering ``None`` rather than an empty record.
    """

    def test_sync_unknown_id(self, sync_db: Any) -> None:
        assert sync_db.read("no-such-id") is None
        assert sync_db.exists("no-such-id") is False

    async def test_async_unknown_id(self, async_db: Any) -> None:
        assert await async_db.read("no-such-id") is None
        assert await async_db.exists("no-such-id") is False

    def test_sync_unknown_id_in_a_batch(self, sync_db: Any) -> None:
        rid = sync_db.create(Record(data={"a": 1}))
        assert sync_db.read_batch([rid, "no-such-id"])[1] is None


class TestANonEmptyRecordIsUnaffected:
    """The path that always worked, asserted so the fix cannot quietly move it."""

    def test_sync_round_trip(self, sync_db: Any) -> None:
        rid = sync_db.create(Record(data={"colour": "red"}))
        assert sync_db.read(rid).get_value("colour") == "red"

    async def test_async_round_trip(self, async_db: Any) -> None:
        rid = await async_db.create(Record(data={"colour": "red"}))
        assert (await async_db.read(rid)).get_value("colour") == "red"
