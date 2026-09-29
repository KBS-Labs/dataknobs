"""``count(query)`` counts the whole match, on every backend.

A caller passes one ``Query`` to ``search()`` for a page and to ``count()`` for
the total, so a count ignores the query's ``limit``, ``offset`` and sort. The
backends used to disagree. Measured before the fix, three records, each query
below:

=====================  ======  ====  ===============  ======  ======
Query                  memory  file  sync Postgres    sqlite  DuckDB
=====================  ======  ====  ===============  ======  ======
``limit=1``            1       1     1                3       3
filter, ``limit=1``    1       1     1                3       3
filter, ``offset=2``   1       1     1                3       3
=====================  ======  ====  ===============  ======  ======

sqlite, DuckDB and Elasticsearch push a count down with the filters alone.
Memory, file, S3 and Postgres inherited the base class's ``count``, which was
``len(search(query))`` and so counted the page.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from dataknobs_common.testing import (
    requires_localstack,
    requires_postgres,
    requires_real_elasticsearch,
)

from dataknobs_data import Filter, Operator, Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.query import Query

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

BACKENDS = [
    "memory",
    "file",
    "sqlite",
    "duckdb",
    pytest.param("postgres", marks=requires_postgres),
    pytest.param("s3", marks=requires_localstack),
    pytest.param("elasticsearch", marks=requires_real_elasticsearch),
]

#: Three records that all match the filter below, so a page of one and the
#: whole match are distinguishable, and so is an offset past two of them.
RECORDS = [{"kind": "a", "n": n} for n in range(3)]

_MATCH = Filter("kind", Operator.EQ, "a")

CASES: list[tuple[str, Query]] = [
    ("limit-alone", Query(limit_value=1)),
    ("filter-and-limit", Query(filters=[_MATCH], limit_value=1)),
    ("filter-and-offset", Query(filters=[_MATCH], offset_value=2)),
    (
        "filter-limit-offset-sort",
        Query(filters=[_MATCH], limit_value=1, offset_value=1).sort_by("n"),
    ),
]


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> Iterator[tuple[str, dict[str, Any]]]:
    """One backend's kind and constructor config.

    Service configs are resolved here, in a sync fixture, because the S3
    bucket fixture runs its own event loop and cannot be reached from inside
    an async one.
    """
    kind = request.param
    if kind == "postgres":
        for config in request.getfixturevalue("make_postgres_test_db")("test_count_paging_"):
            yield kind, config
    elif kind == "s3":
        for config in request.getfixturevalue("make_localstack_s3_bucket")(
            "dataknobs-count-paging"
        ):
            yield kind, {**config, "prefix": f"cp-{uuid.uuid4().hex[:10]}/"}
    elif kind == "elasticsearch":
        for config in request.getfixturevalue("make_elasticsearch_test_index")(
            "test_count_paging_"
        ):
            yield kind, config
    else:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            yield (
                kind,
                {
                    "memory": {},
                    "file": {"path": str(root / "records.json")},
                    "sqlite": {"path": str(root / "records.db")},
                    "duckdb": {"path": str(root / "records.duckdb"), "table": "records"},
                }[kind],
            )


def _sync_class(kind: str) -> Any:
    if kind == "postgres":
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        return SyncPostgresDatabase
    if kind == "s3":
        from dataknobs_data.backends.s3 import SyncS3Database

        return SyncS3Database
    if kind == "elasticsearch":
        from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase

        return SyncElasticsearchDatabase
    return {
        "memory": SyncMemoryDatabase,
        "file": SyncFileDatabase,
        "sqlite": SyncSQLiteDatabase,
        "duckdb": SyncDuckDBDatabase,
    }[kind]


def _async_class(kind: str) -> Any:
    if kind == "postgres":
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        return AsyncPostgresDatabase
    if kind == "s3":
        from dataknobs_data.backends.s3_async import AsyncS3Database

        return AsyncS3Database
    if kind == "elasticsearch":
        from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase

        return AsyncElasticsearchDatabase
    return {
        "memory": AsyncMemoryDatabase,
        "file": AsyncFileDatabase,
        "sqlite": AsyncSQLiteDatabase,
        "duckdb": AsyncDuckDBDatabase,
    }[kind]


@pytest.fixture
def sync_db(backend: tuple[str, dict[str, Any]]) -> Iterator[Any]:
    """A connected sync backend holding the three records."""
    kind, config = backend
    db = _sync_class(kind)(config)
    db.connect()
    try:
        for data in RECORDS:
            db.create(Record(data=dict(data)))
        yield db
    finally:
        if kind == "s3":
            db.clear()
        db.close()


@pytest.fixture
async def async_db(backend: tuple[str, dict[str, Any]]) -> AsyncIterator[Any]:
    """A connected async backend holding the three records."""
    kind, config = backend
    db = _async_class(kind)(config)
    await db.connect()
    try:
        for data in RECORDS:
            await db.create(Record(data=dict(data)))
        yield db
    finally:
        if kind == "s3":
            await db.clear()
        await db.close()


@pytest.mark.parametrize("query", [pytest.param(q, id=c) for c, q in CASES])
class TestACountIgnoresPaging:
    """Every backend counts the whole match, whatever page the query names."""

    def test_sync(self, sync_db: Any, query: Query) -> None:
        assert sync_db.count(query) == len(RECORDS)

    async def test_async(self, async_db: Any, query: Query) -> None:
        assert await async_db.count(query) == len(RECORDS)

    def test_the_page_is_still_a_page_sync(self, sync_db: Any, query: Query) -> None:
        """The same query still pages a search: the count is what differs."""
        assert len(sync_db.search(query)) < len(RECORDS)

    async def test_the_page_is_still_a_page_async(self, async_db: Any, query: Query) -> None:
        assert len(await async_db.search(query)) < len(RECORDS)


class TestACountLeavesTheCallersQueryAlone:
    """Counting a paged query does not clear the caller's paging."""

    def test_sync(self, sync_db: Any) -> None:
        query = Query(filters=[_MATCH], limit_value=1, offset_value=1).sort_by("n")
        sync_db.count(query)
        assert (query.limit_value, query.offset_value, len(query.sort_specs)) == (1, 1, 1)

    async def test_async(self, async_db: Any) -> None:
        query = Query(filters=[_MATCH], limit_value=1, offset_value=1).sort_by("n")
        await async_db.count(query)
        assert (query.limit_value, query.offset_value, len(query.sort_specs)) == (1, 1, 1)
