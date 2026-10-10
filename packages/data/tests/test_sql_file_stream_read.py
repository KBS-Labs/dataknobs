"""``stream_read`` on SQLite and DuckDB pages in a total order, and honours the query.

Each page is a ``search`` of its own, so the pages agree with one another only
if every page sorts the rows the same way. A sort that ties leaves the order of
the tied rows to the engine, page by page, so the key breaks the tie: every row
is read once. A stream also returns what ``search`` would -- its limit, its
offset, its projection -- one batch at a time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from dataknobs_common.async_iter import aclosing_iter
from dataknobs_data.factory import AsyncDatabaseFactory, DatabaseFactory
from dataknobs_data.query import RESERVED_KEY_FIELD, Query, SortOrder, SortSpec
from dataknobs_data.records import Record
from dataknobs_data.streaming import StreamConfig, stream_page

BATCH = 4
#: Three full pages and one more row, in two groups, so a sort by group ties.
#: ``n`` is text, padded so its order is the same on every engine: the subject
#: here is paging, not how an engine orders a JSON number.
ROWS = [
    Record({"group": "ab"[n % 2], "n": f"{n:02d}"}, storage_id=f"r{n:02d}")
    for n in range(3 * BATCH + 1)
]

BACKENDS = ["sqlite", "duckdb"]


@pytest.fixture(params=[(b, t) for b in BACKENDS for t in ("sync", "async")], ids="-".join)
def stream(request: pytest.FixtureRequest) -> Iterator[Any]:
    """``stream(query, config)`` over a fresh JSON-layout store holding :data:`ROWS`."""
    backend, twin = request.param
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    if twin == "sync":
        db = DatabaseFactory().create(backend=backend, path=":memory:")
        db.connect()
        db.create_batch([r.copy(deep=True) for r in ROWS])
        yield lambda query, config: list(db.stream_read(query, config))
        db.close()
        return
    loop = asyncio.new_event_loop()
    adb = AsyncDatabaseFactory().create(backend=backend, path=":memory:")
    loop.run_until_complete(adb.connect())
    loop.run_until_complete(adb.create_batch([r.copy(deep=True) for r in ROWS]))

    async def collect(query: Query, config: StreamConfig) -> list[Record]:
        async with aclosing_iter(adb.stream_read(query, config)) as records:
            return [record async for record in records]

    yield lambda query, config: loop.run_until_complete(collect(query, config))
    loop.run_until_complete(adb.close())
    loop.close()


def _ids(records: list[Record]) -> list[str]:
    return [str(r.storage_id) for r in records]


def test_a_sort_that_ties_reads_every_row_once(stream: Any) -> None:
    seen = stream(Query(sort_specs=[SortSpec("group")]), StreamConfig(batch_size=BATCH))
    assert sorted(_ids(seen)) == sorted(str(r.storage_id) for r in ROWS)
    groups = [r.get_value("group") for r in seen]
    assert groups == sorted(groups), "the query's own sort still leads"


def test_no_sort_reads_every_row_once(stream: Any) -> None:
    seen = stream(Query(), StreamConfig(batch_size=BATCH))
    assert sorted(_ids(seen)) == sorted(str(r.storage_id) for r in ROWS)


def test_a_limit_is_honoured(stream: Any) -> None:
    """Bug: each page set its own limit over the caller's, so a stream asked
    for five rows returned all of them.
    """
    seen = stream(Query(sort_specs=[SortSpec("n")], limit_value=5), StreamConfig(batch_size=BATCH))
    assert [r.get_value("n") for r in seen] == ["00", "01", "02", "03", "04"]


def test_an_offset_is_honoured(stream: Any) -> None:
    """Bug: each page set its own offset over the caller's, so the first rows
    the caller skipped were streamed anyway.
    """
    query = Query(sort_specs=[SortSpec("n", SortOrder.DESC)], offset_value=3, limit_value=6)
    seen = stream(query, StreamConfig(batch_size=BATCH))
    assert [r.get_value("n") for r in seen] == ["09", "08", "07", "06", "05", "04"]


def test_a_limit_on_a_page_boundary_stops_there(stream: Any) -> None:
    query = Query(sort_specs=[SortSpec("n")], limit_value=2 * BATCH)
    seen = stream(query, StreamConfig(batch_size=BATCH))
    assert [r.get_value("n") for r in seen] == [f"{n:02d}" for n in range(2 * BATCH)]


# --- the page --------------------------------------------------------------------------


def test_a_page_breaks_ties_on_the_key() -> None:
    query = Query(sort_specs=[SortSpec("group")])
    page = stream_page(query, streamed=0, batch_size=BATCH)
    assert page is not None
    assert [(s.field, s.order) for s in page.sort_specs] == [
        ("group", SortOrder.ASC),
        (RESERVED_KEY_FIELD, SortOrder.ASC),
    ]
    assert (page.offset_value, page.limit_value) == (0, BATCH)
    assert [s.field for s in query.sort_specs] == ["group"], "the caller's query is not changed"


def test_a_page_sorted_by_the_key_is_not_given_it_twice() -> None:
    query = Query(sort_specs=[SortSpec(RESERVED_KEY_FIELD, SortOrder.DESC)])
    page = stream_page(query, streamed=BATCH, batch_size=BATCH)
    assert page is not None
    assert [(s.field, s.order) for s in page.sort_specs] == [(RESERVED_KEY_FIELD, SortOrder.DESC)]
    assert page.offset_value == BATCH


def test_the_page_after_the_last_is_none() -> None:
    query = Query(offset_value=2, limit_value=5)
    assert stream_page(query, streamed=4, batch_size=BATCH) == Query(
        sort_specs=[SortSpec(RESERVED_KEY_FIELD)], offset_value=6, limit_value=1
    )
    assert stream_page(query, streamed=5, batch_size=BATCH) is None
