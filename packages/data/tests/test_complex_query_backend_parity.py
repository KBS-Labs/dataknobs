"""Every backend accepts the ``ComplexQuery`` its own signature promises to.

``SyncDatabase.search`` and ``AsyncDatabase.search`` both declare
``query: Query | ComplexQuery``, and the base class carries the fallback that
makes the second half true --- ``_search_with_complex_query``, which converts
the query where it can and filters in memory where it cannot. Eleven of the
fourteen backend ``search`` implementations dispatch to it on the first line.

Three did not, and narrowed their own signature to ``Query`` instead: both file
backends and both S3 backends. A narrowed override is the shape of the defect
rather than the defect itself --- what it costs is that the body then reads
``query.filters``, an attribute ``ComplexQuery`` does not have. Measured before
the fix, same corpus and the same `OR` query:

=========================  ==============================================
Backend                    ``search(ComplexQuery.OR([...]))``
=========================  ==============================================
``SyncMemoryDatabase``     ``['red', 'blue']``
``SyncFileDatabase``       ``AttributeError: 'ComplexQuery' object has no
                           attribute 'filters'``
``AsyncFileDatabase``      the same ``AttributeError``
=========================  ==============================================

So this is a parity module rather than a file-backend module: the base class
declares one contract for every backend, and the way to keep a fourth from
drifting out of it is to assert it of all of them at once. The S3 pair takes
the same fix and is pinned beside the rest of the S3 suite, which needs a
running service; everything here is in-process and runs by default.
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
from dataknobs_data.query import Query
from dataknobs_data.query_logic import ComplexQuery

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

BACKENDS = ["memory", "file", "sqlite", "duckdb"]

#: Three records, so a two-branch `OR` selects a strict subset and an empty
#: result and a full result are both distinguishable from the right answer.
COLOURS = ["red", "green", "blue"]


def _or_query() -> ComplexQuery:
    """Two colours of the three, which no single ``Query`` filter expresses."""
    return ComplexQuery.OR(
        [Query().filter("colour", "=", "red"), Query().filter("colour", "=", "blue")]
    )


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
    """A connected sync backend holding one record per colour."""
    with tempfile.TemporaryDirectory() as d:
        db = _make_sync_db(request.param, Path(d))
        for colour in COLOURS:
            db.create(Record(data={"colour": colour}))
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()


@pytest.fixture(params=BACKENDS)
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[Any]:
    """A connected async backend holding one record per colour."""
    with tempfile.TemporaryDirectory() as d:
        db = await _make_async_db(request.param, Path(d))
        for colour in COLOURS:
            await db.create(Record(data={"colour": colour}))
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                await close()


def _colours(records: list[Record]) -> set[str]:
    return {record.get_value("colour") for record in records}


class TestABooleanQueryIsAnswered:
    """The half of the declared type that three backends did not implement."""

    def test_sync_or(self, sync_db: Any) -> None:
        assert _colours(sync_db.search(_or_query())) == {"red", "blue"}

    async def test_async_or(self, async_db: Any) -> None:
        assert _colours(await async_db.search(_or_query())) == {"red", "blue"}

    def test_sync_and(self, sync_db: Any) -> None:
        """An `AND` of two conditions on the same record, which one colour meets."""
        query = ComplexQuery.AND(
            [Query().filter("colour", "=", "red"), Query().filter("colour", "!=", "blue")]
        )
        assert _colours(sync_db.search(query)) == {"red"}

    async def test_async_and(self, async_db: Any) -> None:
        query = ComplexQuery.AND(
            [Query().filter("colour", "=", "red"), Query().filter("colour", "!=", "blue")]
        )
        assert _colours(await async_db.search(query)) == {"red"}


class TestTheSimpleQueryPathIsUnchanged:
    """The guard on the fix: dispatching on type must not divert a plain query.

    A `ComplexQuery` reaches `search` through an `isinstance` branch added at
    the top of each body. This asserts the branch is not taken for the type
    that was always handled, which is the regression the fix could introduce
    and the one nothing else here would catch.
    """

    def test_sync_plain_query(self, sync_db: Any) -> None:
        found = sync_db.search(Query().filter("colour", "=", "green"))
        assert _colours(found) == {"green"}

    async def test_async_plain_query(self, async_db: Any) -> None:
        found = await async_db.search(Query().filter("colour", "=", "green"))
        assert _colours(found) == {"green"}

    def test_sync_empty_query_returns_everything(self, sync_db: Any) -> None:
        assert _colours(sync_db.search(Query())) == set(COLOURS)

    async def test_async_empty_query_returns_everything(self, async_db: Any) -> None:
        assert _colours(await async_db.search(Query())) == set(COLOURS)
