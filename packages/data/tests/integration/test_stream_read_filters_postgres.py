"""``stream_read`` applies the filters it was given, on both Postgres twins.

``search`` and ``stream_read`` take the same :class:`Query`, so a caller who
swaps one for the other to bound memory has every right to the same rows. On
these two backends they disagreed: both ``stream_read`` implementations
open-coded their own WHERE construction and only ever emitted a clause for
``Operator.EQ``, so every other operator was dropped in silence and the
stream returned rows the caller had filtered out.

The async twin carried a second, louder half. Its placeholder counter advanced
once per *filter* while its argument list grew only for EQ, so one non-EQ
filter ahead of an EQ one shifted every later placeholder past its argument --
the SQL named a ``$N`` nothing had bound.

Both are asserted against a real server, because the fix routes these methods
through ``SQLQueryBuilder`` and what that changes is the SQL the server sees.
An in-process assertion on a generated string would restate the fix rather
than test it.

Requires a running Postgres; the module skips when unavailable.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import Record
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase
from dataknobs_data.query import Filter, Operator, Query

pytestmark = requires_postgres

ROWS = [
    Record({"name": "ada", "age": 36, "team": "core"}, metadata={"work_order_id": "W-1"}, id="r1"),
    Record({"name": "bela", "age": 41, "team": "core"}, metadata={"work_order_id": "W-2"}, id="r2"),
    Record({"name": "cyd", "age": 29, "team": "edge"}, metadata={"work_order_id": "W-1"}, id="r3"),
]


@pytest.fixture
def sync_pg(make_postgres_test_db) -> Generator[SyncPostgresDatabase, None, None]:
    for pg in make_postgres_test_db("test_stream_filters_"):
        db = SyncPostgresDatabase(pg)
        db.connect()
        for record in ROWS:
            db.create(record)
        try:
            yield db
        finally:
            db.close()


@pytest.fixture
async def async_pg(make_postgres_test_db) -> AsyncGenerator[AsyncPostgresDatabase, None]:
    for pg in make_postgres_test_db("test_stream_filters_async_"):
        db = AsyncPostgresDatabase(pg)
        await db.connect()
        for record in ROWS:
            await db.create(record)
        try:
            yield db
        finally:
            await db.close()


def _names(records) -> list[str]:
    return sorted(r.get_value("name") for r in records)


# --------------------------------------------------------------------------
# A non-EQ filter reaches the server
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        pytest.param(Operator.GT, 30, ["ada", "bela"], id="gt"),
        pytest.param(Operator.LTE, 36, ["ada", "cyd"], id="lte"),
        pytest.param(Operator.NEQ, "ada", ["bela", "cyd"], id="neq"),
        pytest.param(Operator.IN, ["ada", "cyd"], ["ada", "cyd"], id="in"),
    ],
)
def test_sync_stream_read_applies_a_non_eq_filter(
    sync_pg: SyncPostgresDatabase, operator: Operator, value: object, expected: list[str]
) -> None:
    """Dropped silently before: the stream returned every row in the table."""
    field = "name" if operator in (Operator.NEQ, Operator.IN) else "age"
    query = Query(filters=[Filter(field, operator, value)])

    streamed = _names(sync_pg.stream_read(query))

    assert streamed == expected
    assert streamed == _names(sync_pg.search(query))


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        pytest.param(Operator.GT, 30, ["ada", "bela"], id="gt"),
        pytest.param(Operator.LTE, 36, ["ada", "cyd"], id="lte"),
        pytest.param(Operator.NEQ, "ada", ["bela", "cyd"], id="neq"),
        pytest.param(Operator.IN, ["ada", "cyd"], ["ada", "cyd"], id="in"),
    ],
)
async def test_async_stream_read_applies_a_non_eq_filter(
    async_pg: AsyncPostgresDatabase, operator: Operator, value: object, expected: list[str]
) -> None:
    """Same claim on the twin, which also had the placeholder shift below."""
    field = "name" if operator in (Operator.NEQ, Operator.IN) else "age"
    query = Query(filters=[Filter(field, operator, value)])

    streamed = _names([record async for record in async_pg.stream_read(query)])

    assert streamed == expected
    assert streamed == _names(await async_pg.search(query))


# --------------------------------------------------------------------------
# A non-EQ filter ahead of an EQ one
# --------------------------------------------------------------------------


def test_sync_a_non_eq_filter_does_not_displace_the_eq_one_after_it(
    sync_pg: SyncPostgresDatabase,
) -> None:
    """Both clauses apply, and they apply to the fields they name."""
    query = Query(filters=[Filter("age", Operator.GT, 30), Filter("team", Operator.EQ, "core")])

    assert _names(sync_pg.stream_read(query)) == ["ada", "bela"]


async def test_async_a_non_eq_filter_does_not_displace_the_eq_one_after_it(
    async_pg: AsyncPostgresDatabase,
) -> None:
    """The placeholder-shift half, which failed loudly rather than quietly.

    The counter advanced for the ``GT`` while nothing was bound for it, so the
    ``EQ`` emitted ``$2`` against a one-argument list.
    """
    query = Query(filters=[Filter("age", Operator.GT, 30), Filter("team", Operator.EQ, "core")])

    streamed = [record async for record in async_pg.stream_read(query)]

    assert _names(streamed) == ["ada", "bela"]


# --------------------------------------------------------------------------
# The EQ path both twins did support
# --------------------------------------------------------------------------


def test_sync_eq_still_agrees_with_search(sync_pg: SyncPostgresDatabase) -> None:
    """The control: routing through the builder must not move the case that worked."""
    query = Query(filters=[Filter("team", Operator.EQ, "core")])

    assert _names(sync_pg.stream_read(query)) == ["ada", "bela"]
    assert _names(sync_pg.stream_read(query)) == _names(sync_pg.search(query))


async def test_async_eq_still_agrees_with_search(async_pg: AsyncPostgresDatabase) -> None:
    """The control, on the twin."""
    query = Query(filters=[Filter("team", Operator.EQ, "core")])

    streamed = _names([record async for record in async_pg.stream_read(query)])

    assert streamed == ["ada", "bela"]
    assert streamed == _names(await async_pg.search(query))


def test_sync_an_unfiltered_stream_is_still_the_whole_table(
    sync_pg: SyncPostgresDatabase,
) -> None:
    """No query is no WHERE, which the builder must also get right."""
    assert _names(sync_pg.stream_read()) == ["ada", "bela", "cyd"]


async def test_async_an_unfiltered_stream_is_still_the_whole_table(
    async_pg: AsyncPostgresDatabase,
) -> None:
    """No query is no WHERE, on the twin."""
    assert _names([record async for record in async_pg.stream_read()]) == [
        "ada",
        "bela",
        "cyd",
    ]


def test_sync_a_dotted_metadata_field_streams_the_rows_search_returns(
    sync_pg: SyncPostgresDatabase,
) -> None:
    """A nested field is one grammar, not two.

    ``stream_read`` pre-flighted its filter fields against the *whole-identifier*
    grammar while the builder it now calls validates each dot-separated segment.
    So ``metadata.work_order_id`` answered rows through ``search`` and raised
    ``ValueError`` through ``stream_read`` -- on one backend, from one ``Query``.
    """
    query = Query(filters=[Filter("metadata.work_order_id", Operator.EQ, "W-1")])

    assert _names(sync_pg.stream_read(query)) == ["ada", "cyd"]
    assert _names(sync_pg.stream_read(query)) == _names(sync_pg.search(query))


async def test_async_a_dotted_metadata_field_streams_the_rows_search_returns(
    async_pg: AsyncPostgresDatabase,
) -> None:
    """The same nested field, on the twin."""
    query = Query(filters=[Filter("metadata.work_order_id", Operator.EQ, "W-1")])

    streamed = _names([record async for record in async_pg.stream_read(query)])

    assert streamed == ["ada", "cyd"]
    assert streamed == _names(await async_pg.search(query))


def test_sync_an_unsafe_field_is_still_refused_before_the_query_runs(
    sync_pg: SyncPostgresDatabase,
) -> None:
    """Widening the grammar to dotted paths must not widen it to anything else.

    The control on the fix: the pre-flight still refuses a field that would
    reach a JSONB key position unquoted, and still refuses it on first
    iteration rather than at the server.
    """
    query = Query(filters=[Filter("x'; DROP TABLE records;--", Operator.EQ, "x")])

    with pytest.raises(ValueError, match="Invalid field name segment"):
        next(iter(sync_pg.stream_read(query)))
