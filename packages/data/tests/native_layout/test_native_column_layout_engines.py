"""A native table answers every filter as ``Filter.matches`` answers it, on every engine.

A table with ordinary typed columns is read through ``NativeColumnLayout``.
Each statement the builder renders is executed against a real table on SQLite,
DuckDB and PostgreSQL (through both asyncpg and psycopg2), and the rows it
returns are compared with the oracle: ``Filter.matches`` over the records the
layout returns.

Rendered with no knowledge of the column's type, the same filters raise on
DuckDB and PostgreSQL (``'abc'`` against an integer), answer wrongly with no
error on SQLite and psycopg2 (``'3'`` matching an integer ``3``), and asyncpg
sends ``3.5`` to an integer column as ``3``. So each column answers only for
values of the kinds its type holds, and a bound compared in the column's own
type is sent with that type.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from dataknobs_data.backends.column_layout import NativeColumnLayout
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.backends.sql_types import sql_types
from dataknobs_data.query import Filter, Operator, Query, SortOrder, SortSpec
from dataknobs_data.query_logic import (
    ComplexQuery,
    FilterCondition,
    LogicCondition,
    LogicOperator,
)
from dataknobs_data.records import Record
from dataknobs_data.schema import DatabaseSchema

from _native_tables import ENGINES, Engine, Table, engine_for

U1 = uuid.UUID("12345678-1234-5678-1234-567812345678")
U2 = uuid.UUID("87654321-4321-8765-4321-876543218765")
U3 = uuid.UUID("11111111-2222-3333-4444-555555555555")

SCHEMA = DatabaseSchema.from_dict(
    {
        "fields": {
            "k": {"type": "string", "sql_type": "uuid"},
            "shape": "string",
            "size": "integer",
            "weight": "float",
            "solid": "boolean",
            "made": "datetime",
            "seen": {"type": "datetime", "sql_type": "timestamptz"},
            "note": "text",
        }
    }
)
COLUMNS = list(SCHEMA.fields)

#: Row 3 is ``NULL`` in every column but the key.
ROWS: list[dict[str, Any]] = [
    {
        "k": U1,
        "shape": "apple",
        "size": 3,
        "weight": 1.5,
        "solid": True,
        "made": datetime(2024, 1, 1),
        "seen": datetime(2024, 1, 1, 12, tzinfo=UTC),
        "note": "2024-03-01T10:00:00",
    },
    {
        "k": U2,
        "shape": "Banana",
        "size": 5,
        "weight": 2.5,
        "solid": False,
        "made": datetime(2024, 6, 1),
        "seen": datetime(2024, 6, 1, 23, 30, tzinfo=timezone(timedelta(hours=-2))),
        "note": "not a time",
    },
    {"k": U3, **dict.fromkeys(COLUMNS[1:])},
]

SQL_TYPES = {
    "k": ("uuid", "UUID", "TEXT"),
    "shape": ("text", "VARCHAR", "TEXT"),
    "size": ("integer", "INTEGER", "INTEGER"),
    "weight": ("double precision", "DOUBLE", "REAL"),
    "solid": ("boolean", "BOOLEAN", "BOOLEAN"),
    "made": ("timestamp", "TIMESTAMP", "TIMESTAMP"),
    "seen": ("timestamptz", "TIMESTAMPTZ", "TIMESTAMPTZ"),
    "note": ("text", "VARCHAR", "TEXT"),
}

SHAPES = Table(
    "shapes",
    {
        name: dict(zip(("postgres", "duckdb", "sqlite"), types, strict=True))
        for name, types in SQL_TYPES.items()
    },
    [[row[c] for c in COLUMNS] for row in ROWS],
)


@pytest.fixture(scope="module", params=ENGINES)
def engine(request: pytest.FixtureRequest) -> Iterator[Engine]:
    yield from engine_for(request, [SHAPES])


LAYOUT = NativeColumnLayout(SCHEMA, id_column="k")


@pytest.fixture(scope="module")
def records(engine: Engine) -> dict[str, Record]:
    """Every row, as the layout reads it, by key."""
    builder = engine.builder("shapes", LAYOUT)
    rows = engine.fetch(*builder.build_search_query(Query()))
    found = [builder.record_from_row(row) for row in rows]
    return {record.storage_id: record for record in found if record.storage_id}


def _oracle_value(field: str, value: Any, op: Operator) -> Any:
    """The bound as a record holds it: the key column's type canonicalises a uuid."""
    if field not in ("k", "id"):
        return value
    bind = sql_types.get("uuid").bind
    if op in (Operator.IN, Operator.NOT_IN, Operator.BETWEEN, Operator.NOT_BETWEEN):
        return [bind(v) for v in value]
    return bind(value)


def _expected(records: dict[str, Record], spec: Filter) -> set[str]:
    oracle = Filter(spec.field, spec.operator, _oracle_value(spec.field, spec.value, spec.operator))
    return {
        key
        for key, record in records.items()
        if oracle.matches(key if spec.field == "id" else record.get_value(spec.field))
    }


def _found(engine: Engine, builder: SQLQueryBuilder, sql: str, params: list[Any]) -> set[str]:
    return {str(builder.record_from_row(row).storage_id) for row in engine.fetch(sql, params)}


FILTERS = [
    # Integer: strings, booleans and fractions never cross into it.
    Filter("size", Operator.EQ, "abc"),
    Filter("size", Operator.EQ, "3"),
    Filter("size", Operator.EQ, 3.5),
    Filter("size", Operator.EQ, True),
    Filter("size", Operator.LIKE, "3%"),
    Filter("size", Operator.GTE, 3.5),
    Filter("size", Operator.GT, 3.5),
    Filter("size", Operator.BETWEEN, [3.5, 5.5]),
    Filter("size", Operator.EQ, 5.0000000001),
    Filter("size", Operator.EQ, 2**40),
    Filter("size", Operator.IN, [3.5, 5]),
    Filter("size", Operator.IN, [3, "5", True]),
    Filter("size", Operator.NEQ, 3),
    Filter("size", Operator.NOT_IN, [3]),
    Filter("size", Operator.GT, 2.5),
    Filter("size", Operator.EQ, 3.0),
    Filter("size", Operator.EQ, Decimal("3")),
    Filter("size", Operator.NOT_BETWEEN, [4, 10]),
    # Float.
    Filter("weight", Operator.EQ, "1.5"),
    Filter("weight", Operator.EQ, 1.5),
    Filter("weight", Operator.GT, 2),
    Filter("weight", Operator.IN, [1.5, 2]),
    # Text: ordered by code point, and only against strings.
    Filter("shape", Operator.EQ, 5),
    Filter("shape", Operator.GT, 3),
    Filter("shape", Operator.GT, "B"),
    Filter("shape", Operator.LT, "b"),
    Filter("shape", Operator.LIKE, "A%"),
    Filter("shape", Operator.NOT_LIKE, "a%"),
    Filter("shape", Operator.STARTS_WITH, "B"),
    Filter("shape", Operator.IN, ["apple", "x"]),
    Filter("shape", Operator.NEQ, "apple"),
    Filter("shape", Operator.EQ, "APPLE"),
    # Boolean: never a number, and not a string.
    Filter("solid", Operator.EQ, 1),
    Filter("solid", Operator.EQ, "true"),
    Filter("solid", Operator.EQ, True),
    Filter("solid", Operator.NEQ, False),
    Filter("solid", Operator.GT, False),
    Filter("solid", Operator.IN, [True, 1]),
    # A naive time: a string bound naming a time is read as one.
    Filter("made", Operator.GT, "2024-03-01"),
    Filter("made", Operator.EQ, "2024-01-01T00:00:00"),
    Filter("made", Operator.GT, date(2024, 3, 1)),
    Filter("made", Operator.EQ, date(2024, 1, 1)),
    Filter("made", Operator.EQ, datetime(2024, 1, 1)),
    Filter("made", Operator.GT, datetime(2024, 1, 1, tzinfo=UTC)),
    Filter("made", Operator.LT, "not a time"),
    Filter("made", Operator.NEQ, "not a time"),
    Filter("made", Operator.BETWEEN, [date(2023, 1, 1), "2024-03-31"]),
    Filter("made", Operator.IN, ["2024-06-01T00:00:00", date(2024, 1, 1)]),
    # A zoned instant, read in UTC.
    Filter("seen", Operator.GT, datetime(2024, 3, 1, tzinfo=UTC)),
    Filter("seen", Operator.EQ, datetime(2024, 1, 1, 12, tzinfo=UTC)),
    Filter("seen", Operator.EQ, "2024-01-01T12:00:00Z"),
    Filter("seen", Operator.EQ, "2024-01-01T13:00:00+01:00"),
    Filter("seen", Operator.GTE, date(2024, 6, 2)),
    Filter("seen", Operator.LT, date(2024, 6, 2)),
    Filter("seen", Operator.LT, datetime(2024, 3, 1)),
    # Text that names a time, against a time bound.
    Filter("note", Operator.GT, date(2024, 1, 1)),
    Filter("note", Operator.EQ, datetime(2024, 3, 1, 10)),
    Filter("note", Operator.GT, "m"),
    # A uuid: canonical text, compared in its own type where it can be.
    Filter("k", Operator.EQ, "not-a-uuid"),
    Filter("k", Operator.EQ, U1),
    Filter("k", Operator.EQ, str(U1).upper()),
    Filter("k", Operator.IN, [U1, "x"]),
    Filter("k", Operator.NEQ, U1),
    Filter("k", Operator.GT, "2"),
    Filter("k", Operator.LIKE, "1234%"),
    Filter("k", Operator.STARTS_WITH, "8"),
    Filter("id", Operator.EQ, str(U2)),
    Filter("id", Operator.EQ, "not-a-uuid"),
    # Presence.
    Filter("size", Operator.EXISTS, None),
    Filter("size", Operator.NOT_EXISTS, None),
]


@pytest.mark.parametrize("spec", FILTERS, ids=lambda f: f"{f.field} {f.operator.value} {f.value!r}")
def test_a_filter_answers_as_the_oracle_does(
    engine: Engine, records: dict[str, Record], spec: Filter
) -> None:
    builder = engine.builder("shapes", LAYOUT)
    found = _found(engine, builder, *builder.build_search_query(Query(filters=[spec])))
    assert found == _expected(records, spec)


@pytest.mark.parametrize(
    "spec", FILTERS, ids=lambda f: f"NOT {f.field} {f.operator.value} {f.value!r}"
)
def test_not_over_a_leaf_answers_as_the_oracle_does(
    engine: Engine, records: dict[str, Record], spec: Filter
) -> None:
    """A leaf is NULL only where the oracle answers False, so NOT over it is right.

    The row whose columns are all NULL is the one this pins.
    """
    builder = engine.builder("shapes", LAYOUT)
    query = ComplexQuery(condition=LogicCondition(LogicOperator.NOT, [FilterCondition(spec)]))
    found = _found(engine, builder, *builder.build_complex_search_query(query))
    assert found == set(records) - _expected(records, spec)


def test_a_count_agrees_with_the_search(engine: Engine, records: dict[str, Record]) -> None:
    builder = engine.builder("shapes", LAYOUT)
    for spec in FILTERS:
        query = Query(filters=[spec])
        rows = engine.fetch(*builder.build_count_query(query))
        assert next(iter(rows[0].values())) == len(_expected(records, spec)), spec


def test_a_read_value_has_its_declared_type(records: dict[str, Record]) -> None:
    record = records[str(U1)]
    assert record.get_value("k") == str(U1)
    assert type(record.get_value("shape")) is str
    assert type(record.get_value("size")) is int
    assert type(record.get_value("weight")) is float
    assert record.get_value("solid") is True
    made = record.get_value("made")
    assert isinstance(made, datetime) and made.tzinfo is None
    assert made == datetime(2024, 1, 1)
    seen = records[str(U2)].get_value("seen")
    assert isinstance(seen, datetime) and seen.utcoffset() == timedelta(0)
    assert seen == datetime(2024, 6, 2, 1, 30, tzinfo=UTC)
    assert records[str(U3)].get_value("size") is None


def test_a_sort_follows_the_oracle(engine: Engine, records: dict[str, Record]) -> None:
    """Text by code point, and every other type by its own order."""
    builder = engine.builder("shapes", LAYOUT)
    present = [Filter("size", Operator.EXISTS, None)]
    for field in ("shape", "size", "weight", "made", "seen", "k"):
        for order in (SortOrder.ASC, SortOrder.DESC):
            query = Query(filters=present, sort_specs=[SortSpec(field, order)])
            rows = engine.fetch(*builder.build_search_query(query))
            got = [builder.record_from_row(row).get_value(field) for row in rows]
            expected = sorted(
                (r.get_value(field) for r in records.values() if r.get_value("size") is not None),
                reverse=order == SortOrder.DESC,
            )
            assert got == expected, (field, order)


def test_a_read_finds_its_key_and_refuses_nothing(engine: Engine) -> None:
    builder = engine.builder("shapes", LAYOUT)
    found = [builder.record_from_row(r) for r in engine.fetch(*builder.build_read_query(str(U2)))]
    assert [r.storage_id for r in found] == [str(U2)]
    assert engine.fetch(*builder.build_read_query("not-a-uuid")) == []
    assert engine.fetch(*builder.build_exists_query(str(U1).upper())) != []
    assert engine.fetch(*builder.build_exists_query("not-a-uuid")) == []


SCOPED = NativeColumnLayout(SCHEMA, id_column="k", scope=[Filter("shape", Operator.EQ, "apple")])


def test_the_scope_reaches_every_read(engine: Engine) -> None:
    builder = engine.builder("shapes", SCOPED)
    everything = _found(engine, builder, *builder.build_search_query(Query()))
    assert everything == {str(U1)}
    count = engine.fetch(*builder.build_count_query(Query()))
    assert next(iter(count[0].values())) == 1
    assert engine.fetch(*builder.build_read_query(str(U2))) == []
    assert engine.fetch(*builder.build_exists_query(str(U2))) == []
    assert engine.fetch(*builder.build_read_query(str(U1))) != []

    # An OR in the caller's condition cannot reach outside the scope.
    either = ComplexQuery(
        condition=LogicCondition(
            LogicOperator.OR,
            [
                FilterCondition(Filter("shape", Operator.EQ, "Banana")),
                FilterCondition(Filter("size", Operator.GT, 0)),
            ],
        )
    )
    assert _found(engine, builder, *builder.build_complex_search_query(either)) == {str(U1)}
    # NOT over everything else still stays inside it.
    neither = ComplexQuery(
        condition=LogicCondition(
            LogicOperator.NOT, [FilterCondition(Filter("shape", Operator.EQ, "apple"))]
        )
    )
    assert _found(engine, builder, *builder.build_complex_search_query(neither)) == set()

    where, params = builder.build_where_clause(None)
    rows = engine.fetch(
        f"SELECT {SCOPED.select_list(builder)} FROM {builder.qualified_table} WHERE TRUE{where}",
        params,
    )
    assert {builder.record_from_row(r).storage_id for r in rows} == {str(U1)}


def test_an_integer_past_64_bits_matches_nothing_it_cannot_equal(
    engine: Engine, records: dict[str, Record], request: pytest.FixtureRequest
) -> None:
    """No integer column holds it, and nothing raises."""
    if engine.name == "sqlite":
        request.applymarker(
            pytest.mark.xfail(
                raises=OverflowError,
                strict=True,
                reason="SQLite's driver cannot bind an integer past 64 bits, under the "
                "JSON layout as here",
            )
        )
    builder = engine.builder("shapes", LAYOUT)
    for spec in (
        Filter("size", Operator.EQ, 2**70),
        Filter("size", Operator.LT, 2**70),
        Filter("size", Operator.NEQ, -(2**70)),
    ):
        found = _found(engine, builder, *builder.build_search_query(Query(filters=[spec])))
        assert found == _expected(records, spec), spec
