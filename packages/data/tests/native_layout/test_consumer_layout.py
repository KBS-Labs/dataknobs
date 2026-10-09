"""A layout a consumer writes, on the builder's public surface alone.

``ColumnLayout`` is an extension point: a table that neither shipped layout
reads gets its own, built from :class:`SQLQueryBuilder`'s public clause
primitives. :class:`TextTableLayout` below is one, for a legacy table that
holds every value as text, and it answers every filter as ``Filter.matches``
answers it over the records it returns, on every engine. That is the proof the
surface is complete: it reaches for nothing ``_``-prefixed, and neither does
:class:`NativeColumnLayout`, the first layout built on it.

A layout other than the JSON layout reads only. The builder's write
statements are the JSON layout's, so one claiming it may write is refused when
a builder is given it, rather than handed ``INSERT`` statements naming columns
its table does not have.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any, ClassVar

import pytest
from dataknobs_common.exceptions import OperationError, ValidationError

from dataknobs_data.backends.column_layout import ColumnLayout, JsonbLayout, NativeColumnLayout
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.backends.sql_types import TIME_READINGS
from dataknobs_data.query import Filter, Operator, Query, SortOrder, SortSpec
from dataknobs_data.records import Record

from _native_tables import ENGINES, Engine, Table, engine_for


class TextTableLayout(ColumnLayout):
    """A table whose every column is text, its key among them.

    A value is the string the column holds, so a bound relates to it as a
    string does: a string compares with it, and a time bound reads the text as
    the time it names, as ``Filter.matches`` reads a string. A number or a
    boolean relates to none of it.
    """

    def __init__(self, columns: Sequence[str], key: str) -> None:
        self._columns = tuple(columns)
        self._key = key

    def _quoted(self, field: str) -> str:
        name = self._key if field == "id" else field
        if name not in self._columns:
            raise ValidationError(f"no column {field!r}", context={"column": field})
        return f'"{name}"'

    def filter_clause(
        self, builder: SQLQueryBuilder, spec: Filter, param_start: int
    ) -> tuple[str, list[Any]]:
        column = self._quoted(spec.field)
        op = spec.operator
        if op == Operator.EXISTS:
            return f"{column} IS NOT NULL", []
        if op == Operator.NOT_EXISTS:
            return f"{column} IS NULL", []
        if op in builder.STRING_ONLY_OPERATORS:
            return builder.operator_clause(column, op, spec.value, param_start)

        def expr_for(reading: str | None) -> tuple[str | None, str] | None:
            if reading == "string":
                return None, column
            if reading in TIME_READINGS:
                assert reading is not None
                names_time, value = builder.time_reading(column, reading)
                return names_time, f"CASE WHEN {names_time} THEN {value} END"
            return None

        return builder.typed_clause(op, spec.value, param_start, expr_for, f"{column} IS NOT NULL")

    def sort_keys(self, builder: SQLQueryBuilder, field: str) -> list[str]:
        return [builder.code_point_order(self._quoted(field))]

    def select_list(self, builder: SQLQueryBuilder) -> str:
        return ", ".join(f'"{c}"' for c in self._columns)

    def key_clause(
        self, builder: SQLQueryBuilder, record_id: str, param_start: int
    ) -> tuple[str, list[Any]]:
        return f'"{self._key}" = {builder.param_placeholder(param_start)}', [record_id]

    def record_from_row(self, row: Mapping[str, Any]) -> Record:
        return Record({c: row[c] for c in self._columns}, storage_id=row[self._key])


COLUMNS = ("k", "name", "made", "n")
TEXTS = Table(
    "texts",
    {c: {"postgres": "text", "duckdb": "VARCHAR", "sqlite": "TEXT"} for c in COLUMNS},
    [
        ["a1", "apple", "2024-01-01T10:00:00", "3"],
        ["a2", "Banana", "not a time", "10"],
        ["a3", None, "2024-06-01T00:00:00+02:00", None],
    ],
)
LAYOUT = TextTableLayout(COLUMNS, key="k")


@pytest.fixture(scope="module", params=ENGINES)
def engine(request: pytest.FixtureRequest) -> Iterator[Engine]:
    yield from engine_for(request, [TEXTS])


def _records(engine: Engine, query: Query) -> list[Record]:
    builder = engine.builder("texts", LAYOUT)
    return [
        builder.record_from_row(row) for row in engine.fetch(*builder.build_search_query(query))
    ]


FILTERS = [
    Filter("name", Operator.EQ, "apple"),
    Filter("name", Operator.GT, "B"),
    Filter("name", Operator.NEQ, "apple"),
    Filter("name", Operator.NOT_IN, ["x"]),
    Filter("name", Operator.LIKE, "a%"),
    Filter("name", Operator.STARTS_WITH, "B"),
    Filter("name", Operator.EQ, 3),
    Filter("name", Operator.EXISTS, None),
    Filter("name", Operator.NOT_EXISTS, None),
    Filter("n", Operator.GT, "2"),
    Filter("n", Operator.EQ, 3),
    Filter("n", Operator.IN, ["3", 10]),
    Filter("made", Operator.GT, date(2024, 1, 1)),
    Filter("made", Operator.EQ, datetime(2024, 1, 1, 10)),
    Filter("made", Operator.GT, datetime(2024, 5, 31, tzinfo=UTC)),
    Filter("made", Operator.NEQ, "not a time"),
    Filter("id", Operator.EQ, "a2"),
    Filter("id", Operator.NOT_IN, ["a1"]),
]


@pytest.mark.parametrize("spec", FILTERS, ids=lambda f: f"{f.field} {f.operator.value} {f.value!r}")
def test_a_layout_on_the_public_surface_answers_as_the_oracle_does(
    engine: Engine, spec: Filter
) -> None:
    every = _records(engine, Query())
    expected = {
        r.storage_id
        for r in every
        if spec.matches(r.storage_id if spec.field == "id" else r.get_value(spec.field))
    }
    assert {r.storage_id for r in _records(engine, Query(filters=[spec]))} == expected


def test_a_layout_on_the_public_surface_sorts_reads_and_counts(engine: Engine) -> None:
    present = Filter("name", Operator.EXISTS, None)
    for order in (SortOrder.ASC, SortOrder.DESC):
        got = [
            r.get_value("name")
            for r in _records(
                engine, Query(filters=[present], sort_specs=[SortSpec("name", order)])
            )
        ]
        assert got == sorted(["apple", "Banana"], reverse=order == SortOrder.DESC)

    builder = engine.builder("texts", LAYOUT)
    rows = engine.fetch(*builder.build_read_query("a2"))
    assert [builder.record_from_row(r).get_value("name") for r in rows] == ["Banana"]
    assert engine.count(*builder.build_count_query(Query(filters=[present]))) == 2


@pytest.mark.parametrize("layout", [NativeColumnLayout, TextTableLayout])
def test_a_layout_reaches_for_nothing_private_on_the_builder(layout: type[ColumnLayout]) -> None:
    """The public surface is complete only while no layout needs more than it."""
    assert re.findall(r"builder\._\w+", inspect.getsource(layout)) == []


class _WritableText(TextTableLayout):
    writable: ClassVar[bool] = True


def test_a_layout_that_is_not_the_json_layout_is_refused_when_it_claims_to_write() -> None:
    """The builder's write statements name ``id``, ``data`` and ``metadata``."""
    with pytest.raises(ValidationError, match="only the JSON layout writes"):
        SQLQueryBuilder("texts", dialect="sqlite", layout=_WritableText(COLUMNS, key="k"))


def test_a_layout_reads_only_unless_it_says_otherwise() -> None:
    builder = SQLQueryBuilder("texts", dialect="sqlite", layout=LAYOUT)
    with pytest.raises(OperationError, match="read-only layout"):
        builder.build_delete_query("a1")


def test_a_json_layout_of_ones_own_still_writes() -> None:
    class Audited(JsonbLayout):
        pass

    builder = SQLQueryBuilder("records", dialect="sqlite", layout=Audited())
    sql, _ = builder.build_delete_query("a1")
    assert sql.startswith("DELETE")
