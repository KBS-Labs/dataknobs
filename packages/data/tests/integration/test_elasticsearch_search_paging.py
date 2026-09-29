"""An Elasticsearch search returns the page its ``Query`` names, whatever its size.

A ``Query`` with no ``limit`` means every match. Measured before the fix,
against a live node:

=================================  ==============================  =====================
Query                              sync                            async
=================================  ==============================  =====================
none, 12 records                   10 records (the default size)   12 records
a filter and ``offset=2``          the match past two              ``BadRequestError``:
                                                                   ``from + size`` 10002
                                                                   passes the window
=================================  ==============================  =====================

So the twins disagreed, and neither could read past ``index.max_result_window``.
Both now drive one plan: a bounded read inside the window is one ``from`` /
``size`` request, anything else pages with ``search_after``. The second half of
this module sets a small window and page size, so the paging path runs on the
node with a handful of records rather than ten thousand.

The in-memory backend is the oracle.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from dataknobs_common.testing import requires_real_elasticsearch

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.backends.memory import SyncMemoryDatabase

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

#: More than Elasticsearch's default page of ten; every fifth record is kind b.
RECORDS = [(f"r{n:02d}", {"number": n, "kind": "b" if n % 5 == 0 else "a"}) for n in range(25)]

#: Settings small enough that the corpus crosses them: a window of 10 and
#: pages of 4 records.
SMALL = {"max_result_window": 10, "search_page_size": 4}

_KIND_A = Filter("kind", Operator.EQ, "a")


def _by_number(query: Query) -> Query:
    return query.sort_by("number")


#: ``(case id, query, ordered)``. An unsorted result is compared as a set.
CASES: list[tuple[str, Query, bool]] = [
    ("unbounded", Query(), False),
    ("unbounded-filter", Query(filters=[_KIND_A]), False),
    ("offset-alone", Query(filters=[_KIND_A], offset_value=2), False),
    ("sorted-unbounded", _by_number(Query()), True),
    ("sorted-offset-alone", _by_number(Query(offset_value=7)), True),
    ("sorted-limit-offset-inside", _by_number(Query(limit_value=3, offset_value=2)), True),
    ("sorted-limit-offset-across", _by_number(Query(limit_value=6, offset_value=8)), True),
    ("sorted-offset-past-a-page", _by_number(Query(filters=[_KIND_A], offset_value=9)), True),
    ("sorted-limit-a-page", _by_number(Query(limit_value=4, offset_value=10)), True),
    ("sorted-by-id-desc", Query(offset_value=3).sort_by("id", "desc"), True),
    ("limit-zero", Query(limit_value=0, offset_value=20), True),
    ("offset-past-the-end", _by_number(Query(offset_value=30)), True),
]


def _expected(query: Query) -> list[str]:
    db = SyncMemoryDatabase()
    for record_id, data in RECORDS:
        db.create(Record(data=dict(data), id=record_id))
    return [record.id for record in db.search(query) if record.id is not None]


def _check(found: list[Record], query: Query, ordered: bool) -> None:
    """An unsorted query fixes how many records come back, not which.

    With no sort, which records an offset skips is each backend's own order,
    so an unordered case checks the count, that nothing repeats, and that
    every record returned matches.
    """
    ids = [record.id for record in found]
    expected = _expected(query)
    if ordered:
        assert ids == expected
        return
    whole_match = set(_expected(Query(filters=list(query.filters))))
    assert len(ids) == len(expected)
    assert len(set(ids)) == len(ids)
    assert set(ids) <= whole_match


@pytest.fixture
def es_config(make_elasticsearch_test_index: Any) -> Iterator[dict[str, Any]]:
    yield from make_elasticsearch_test_index("test_search_paging_")


def _sync_db(config: dict[str, Any]) -> Iterator[SyncElasticsearchDatabase]:
    db = SyncElasticsearchDatabase(config)
    db.connect()
    try:
        for record_id, data in RECORDS:
            db.create(Record(data=dict(data), id=record_id))
        yield db
    finally:
        db.close()


async def _async_db(config: dict[str, Any]) -> AsyncIterator[AsyncElasticsearchDatabase]:
    db = AsyncElasticsearchDatabase(config)
    await db.connect()
    try:
        for record_id, data in RECORDS:
            await db.create(Record(data=dict(data), id=record_id))
        yield db
    finally:
        await db.close()


@pytest.fixture(params=["default", "small"])
def sync_db(
    request: pytest.FixtureRequest, es_config: dict[str, Any]
) -> Iterator[SyncElasticsearchDatabase]:
    extra = SMALL if request.param == "small" else {}
    yield from _sync_db({**es_config, **extra})


@pytest.fixture(params=["default", "small"])
async def async_db(
    request: pytest.FixtureRequest, es_config: dict[str, Any]
) -> AsyncIterator[AsyncElasticsearchDatabase]:
    extra = SMALL if request.param == "small" else {}
    async for db in _async_db({**es_config, **extra}):
        yield db


@pytest.mark.parametrize(("query", "ordered"), [pytest.param(q, o, id=c) for c, q, o in CASES])
class TestASearchReturnsThePageItsQueryNames:
    """Each twin returns what the in-memory backend returns for the same query."""

    def test_sync(self, sync_db: SyncElasticsearchDatabase, query: Query, ordered: bool) -> None:
        _check(sync_db.search(query), query, ordered)

    async def test_async(
        self, async_db: AsyncElasticsearchDatabase, query: Query, ordered: bool
    ) -> None:
        _check(await async_db.search(query), query, ordered)


def test_sync_stream_read_reads_every_record(sync_db: SyncElasticsearchDatabase) -> None:
    """The sync stream reads through ``search``, so it was cut at ten as well."""
    assert {record.id for record in sync_db.stream_read()} == {rid for rid, _ in RECORDS}


@pytest.fixture
def unmapped_config(es_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """An index created bare, so ``id`` is dynamically mapped as text.

    That is what an index created outside the backend gets, and a sort on
    ``id`` there is refused; the paged read must not depend on one.
    """
    from elasticsearch import Elasticsearch

    client = Elasticsearch([f"http://{es_config['host']}:{es_config['port']}"])
    try:
        client.indices.create(index=es_config["index"])
        yield {**es_config, **SMALL}
    finally:
        client.close()


def test_sync_pages_an_index_without_the_backends_mapping(
    unmapped_config: dict[str, Any],
) -> None:
    for db in _sync_db(unmapped_config):
        assert len(db.search(Query())) == len(RECORDS)


async def test_async_pages_an_index_without_the_backends_mapping(
    unmapped_config: dict[str, Any],
) -> None:
    async for db in _async_db(unmapped_config):
        assert len(await db.search(Query(offset_value=3))) == len(RECORDS) - 3
