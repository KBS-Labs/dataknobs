"""A SQLite or DuckDB stream finds each page after the last row it read.

A stream reads a page at a time, each its own statement, so a write can land
between two pages. A page found by its offset counts the rows ahead of it, so a
row added or removed ahead of the stream's position moves every row after it:
one is read twice, or never. A page found by the last row's sort keys moves
with nothing but the rows it is after. A row present for the whole stream is
read once, and a row written meanwhile is read if it sorts after the stream's
position.

For a table nobody writes, the stream still returns exactly what ``search``
does, in its order, for any sort, offset, limit and batch size.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from dataknobs_common.async_iter import aclosing_iter
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.factory import AsyncDatabaseFactory, DatabaseFactory
from dataknobs_data.query import (
    RESERVED_KEY_FIELD,
    Filter,
    Operator,
    Query,
    SortOrder,
    SortSpec,
    is_storage_key_field,
)
from dataknobs_data.records import Record
from dataknobs_data.streaming import StreamConfig, stream_page

BACKENDS = ["sqlite", "duckdb"]
TWINS = ["sync", "async"]


class Store:
    """One twin of one backend, holding ``rows``, behind one synchronous surface."""

    def __init__(self, backend: str, twin: str, rows: list[Record]) -> None:
        if backend == "duckdb":
            pytest.importorskip("duckdb")
        self._loop: asyncio.AbstractEventLoop | None = None
        copies = [r.copy(deep=True) for r in rows]
        if twin == "sync":
            self.db: Any = DatabaseFactory().create(backend=backend, path=":memory:")
            self.db.connect()
            self.db.create_batch(copies)
        else:
            self._loop = asyncio.new_event_loop()
            self.db = AsyncDatabaseFactory().create(backend=backend, path=":memory:")
            self._run(self.db.connect())
            self._run(self.db.create_batch(copies))

    def _run(self, result: Any) -> Any:
        if self._loop is not None and inspect.isawaitable(result):
            return self._loop.run_until_complete(result)
        return result

    def search(self, query: Query) -> list[Record]:
        return self._run(self.db.search(query))

    def stream(
        self,
        query: Query,
        config: StreamConfig,
        *,
        write_after: int | None = None,
        write: Callable[[Any], Any] | None = None,
    ) -> list[Record]:
        """The records ``stream_read`` returns, ``write(db)`` run once ``write_after`` are read."""
        if self._loop is None:
            seen: list[Record] = []
            for record in self.db.stream_read(query, config):
                seen.append(record)
                if write is not None and len(seen) == write_after:
                    write(self.db)
            return seen

        async def collect() -> list[Record]:
            seen: list[Record] = []
            async with aclosing_iter(self.db.stream_read(query, config)) as records:
                async for record in records:
                    seen.append(record)
                    if write is not None and len(seen) == write_after:
                        result = write(self.db)
                        if inspect.isawaitable(result):
                            await result
            return seen

        assert self._loop is not None
        return self._loop.run_until_complete(collect())

    def close(self) -> None:
        self._run(self.db.close())
        if self._loop is not None:
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
            self._loop.close()


def _ids(records: list[Record]) -> list[str]:
    return [str(r.storage_id) for r in records]


# --- a write between two pages ---------------------------------------------------------

#: Ten rows, ``n = 0, 10, ..., 90``, read three to a page by ``n``.
TENS = [Record({"n": n}, storage_id=f"r{n:02d}") for n in range(0, 100, 10)]
BY_N = Query(sort_specs=[SortSpec("n")])
THREE = StreamConfig(batch_size=3)
#: The fourth record is the first of the second page, so the write lands
#: between the second page and the third.
FOURTH = 4


@pytest.fixture(params=[(b, t) for b in BACKENDS for t in TWINS], ids="-".join)
def tens(request: pytest.FixtureRequest) -> Iterator[Store]:
    backend, twin = request.param
    store = Store(backend, twin, TENS)
    yield store
    store.close()


def test_a_row_inserted_ahead_of_the_stream_moves_nothing(tens: Store) -> None:
    """Bug: the third page skipped six rows, and the row inserted ahead of
    them made the sixth of those the fourth row read again: ``n = 50`` came
    twice.
    """
    seen = tens.stream(
        BY_N,
        THREE,
        write_after=FOURTH,
        write=lambda db: db.create(Record({"n": -1}, storage_id="ahead")),
    )
    assert _ids(seen) == _ids(TENS)


def test_a_row_deleted_behind_the_stream_moves_nothing(tens: Store) -> None:
    """Bug: with ``n = 0`` gone, the third page's offset of six began a row
    late, and ``n = 60`` was never read.
    """
    seen = tens.stream(BY_N, THREE, write_after=FOURTH, write=lambda db: db.delete("r00"))
    assert _ids(seen) == _ids(TENS)


def test_a_row_written_after_the_stream_position_is_read(tens: Store) -> None:
    seen = tens.stream(
        BY_N,
        THREE,
        write_after=FOURTH,
        write=lambda db: db.create(Record({"n": 55}, storage_id="later")),
    )
    assert _ids(seen) == [*_ids(TENS[:6]), "later", *_ids(TENS[6:])]


def test_a_descending_stream_moves_with_its_own_direction(tens: Store) -> None:
    """Ahead of a descending stream is a larger value."""
    seen = tens.stream(
        Query(sort_specs=[SortSpec("n", SortOrder.DESC)]),
        THREE,
        write_after=FOURTH,
        write=lambda db: db.create(Record({"n": 1000}, storage_id="ahead")),
    )
    assert _ids(seen) == _ids(TENS[::-1])


# --- an unwritten table: the stream is the search --------------------------------------

#: Ties, missing values, mixed kinds and a nested field, so every rule a key
#: list carries meets a page boundary somewhere.
MIXED = [
    Record(data, storage_id=f"m{i:02d}")
    for i, data in enumerate(
        [
            {"a": 1, "b": "x", "c": 1.5, "info": {"rank": 3}},
            {"a": 2, "b": "y", "info": {"rank": 1}},
            {"a": 1},
            {"a": 3, "b": "x", "c": -2.0, "info": {"rank": 3}},
            {"a": 2, "b": "z", "c": 10, "info": {"rank": 2}},
            {"b": "w", "c": 0},
            {"a": 1, "b": "x", "c": 1e30, "info": {"rank": 1}},
            {"a": 2.5, "b": "é"},
            {"a": 10, "b": 'a"b', "c": 2**53 + 1},
            {"a": 2, "b": "y", "c": 2**53, "info": {"rank": 2}},
            {"a": None, "b": "x"},
            {"a": -1, "b": "B", "c": -0.5, "info": {"rank": 1}},
            {"a": 2, "c": 2**70},
        ]
    )
]

QUERIES = {
    "a asc, b desc": Query(sort_specs=[SortSpec("a"), SortSpec("b", SortOrder.DESC)]),
    "b, missing in some": Query(sort_specs=[SortSpec("b")]),
    "c desc": Query(sort_specs=[SortSpec("c", SortOrder.DESC)]),
    "key desc": Query(sort_specs=[SortSpec(RESERVED_KEY_FIELD, SortOrder.DESC)]),
    "nested, then a desc": Query(sort_specs=[SortSpec("info.rank"), SortSpec("a", SortOrder.DESC)]),
    "a desc, offset and limit": Query(
        sort_specs=[SortSpec("a", SortOrder.DESC)], offset_value=2, limit_value=7
    ),
    "no sort, offset": Query(offset_value=3),
    "filtered, b desc, projected": Query(
        filters=[Filter("a", Operator.GTE, 2)],
        sort_specs=[SortSpec("b", SortOrder.DESC)],
        fields=["b"],
    ),
}


def _with_key(query: Query) -> Query:
    """``query``, its ties broken by the key: the order a stream reads in."""
    total = query.copy()
    if not any(is_storage_key_field(s.field) for s in total.sort_specs):
        total.sort_specs.append(SortSpec(RESERVED_KEY_FIELD))
    return total


@pytest.fixture(scope="module", params=[(b, t) for b in BACKENDS for t in TWINS], ids="-".join)
def mixed(request: pytest.FixtureRequest) -> Iterator[Store]:
    backend, twin = request.param
    store = Store(backend, twin, MIXED)
    yield store
    store.close()


@pytest.mark.parametrize("batch", [1, 2, 100])
@pytest.mark.parametrize("name", list(QUERIES))
def test_a_stream_reads_what_search_does(mixed: Store, name: str, batch: int) -> None:
    query = QUERIES[name]
    expected = mixed.search(_with_key(query))
    seen = mixed.stream(query, StreamConfig(batch_size=batch))
    assert _ids(seen) == _ids(expected)
    assert [r.to_dict() for r in seen] == [r.to_dict() for r in expected]


# --- each page after the first is found by its keys, not counted to --------------------


def test_no_page_after_the_first_is_counted_to() -> None:
    """Bug: each page skipped every row the stream had read, so reading ``n``
    rows ``b`` at a time scanned ``n**2 / 2b`` of them.
    """
    sent: list[str] = []
    store = Store("sqlite", "sync", MIXED)
    try:
        store.db.conn.set_trace_callback(sent.append)
        seen = store.stream(Query(offset_value=1), StreamConfig(batch_size=2))
        store.db.conn.set_trace_callback(None)
    finally:
        store.close()
    pages = [s for s in sent if s.lstrip().upper().startswith("SELECT")]
    assert len(seen) == len(MIXED) - 1
    assert len(pages) == len(MIXED) // 2 + 1
    assert "OFFSET 1" in pages[0]
    assert all("OFFSET" not in page for page in pages[1:])


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_a_page_after_a_row_has_no_offset(dialect: str) -> None:
    """The query's offset places the first page; the last row read places the rest."""
    builder = SQLQueryBuilder("records", dialect=dialect)
    query = Query(
        sort_specs=[SortSpec("a"), SortSpec(RESERVED_KEY_FIELD)], offset_value=3, limit_value=2
    )
    first_page, later_page = (stream_page(query, n, batch_size=1) for n in (0, 1))
    assert first_page is not None and later_page is not None
    first, _, keys = builder.build_page_query(first_page)
    assert "OFFSET 3" in first
    assert keys == len(builder.page_keys(query))
    later, params, _ = builder.build_page_query(later_page, after=["k"] * keys)
    assert "OFFSET" not in later
    assert params == ["k"] * len(params) and params, "the last row's keys are bound, not rendered"
