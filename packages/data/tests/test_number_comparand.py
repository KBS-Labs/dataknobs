"""What a number bound is compared with, where a column cannot hold it exactly.

:func:`~dataknobs_data.backends.sql_types.comparand` replaces a bound the
column's domain cannot hold with its neighbour on the side the operator
needs, or with a constant answer. The rule is checked here against every
value of a small stand-in domain, then on the three domains the JSON layout
reads numbers in; :meth:`SQLQueryBuilder.typed_clause` is checked to apply it
through its ``bind`` hook, and to call a hook written before the hook took the
operator as it was written.
"""

from __future__ import annotations

import math
import sys
import tempfile
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import Query, Record, SyncDatabase
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.backends.sql_types import (
    DOUBLE,
    HUGEINT,
    INT64_OR_DOUBLE,
    MATCHES_ALL,
    MATCHES_NONE,
    NumberDomain,
    comparand,
)
from dataknobs_data.query import Filter, Operator, SortOrder, SortSpec

ORDERINGS = {
    Operator.GT: lambda x, b: x > b,
    Operator.GTE: lambda x, b: x >= b,
    Operator.LT: lambda x, b: x < b,
    Operator.LTE: lambda x, b: x <= b,
    Operator.EQ: lambda x, b: x == b,
    Operator.IN: lambda x, b: x == b,
}

#: A domain small enough to enumerate: the integers -3..3.
SMALL = NumberDomain(integers=range(-3, 4))
SMALL_VALUES = list(range(-3, 4))


def _answers(domain_values: list[Any], op: Operator, compared: Any) -> list[Any]:
    if compared is MATCHES_NONE:
        return []
    if compared is MATCHES_ALL:
        return list(domain_values)
    return [x for x in domain_values if ORDERINGS[op](x, compared)]


@pytest.mark.parametrize("op", list(ORDERINGS))
@pytest.mark.parametrize(
    "bound",
    [-10, -3, -2.5, Decimal("-0.5"), 0, Fraction(1, 3), 2.0, 3, 3.5, 10, math.inf, -math.inf],
)
def test_the_comparand_answers_every_value_of_the_domain_as_the_bound_does(
    op: Operator, bound: Any
) -> None:
    """Exact, between two values, and past either end."""
    compared = comparand(SMALL, op, bound)
    want = [x for x in SMALL_VALUES if ORDERINGS[op](x, bound)]
    assert _answers(SMALL_VALUES, op, compared) == want


@pytest.mark.parametrize(
    ("domain", "op", "bound", "expected"),
    [
        # A bound the domain holds is sent as its own value.
        (INT64_OR_DOUBLE, Operator.EQ, 2**53 + 1, 2**53 + 1),
        (INT64_OR_DOUBLE, Operator.EQ, 2**70, float(2**70)),
        (INT64_OR_DOUBLE, Operator.EQ, Decimal("2.5"), 2.5),
        (HUGEINT, Operator.EQ, 2**70 + 1, 2**70 + 1),
        (HUGEINT, Operator.EQ, 2.0, 2),
        (DOUBLE, Operator.EQ, 2**53, 2.0**53),
        # One SQLite cannot bind lies between a 64-bit integer or a double.
        (INT64_OR_DOUBLE, Operator.EQ, 2**70 + 1, MATCHES_NONE),
        (INT64_OR_DOUBLE, Operator.GT, 2**70 + 1, float(2**70)),
        (INT64_OR_DOUBLE, Operator.GTE, 2**70 + 1, math.nextafter(float(2**70), math.inf)),
        (INT64_OR_DOUBLE, Operator.LT, 2**63, float(2**63)),
        (
            INT64_OR_DOUBLE,
            Operator.GT,
            -(2**63) - 1,
            math.nextafter(float(-(2**63)), -math.inf),
        ),
        # A fraction never equals an integer, and orders between two.
        (HUGEINT, Operator.IN, Decimal("2.5"), MATCHES_NONE),
        (HUGEINT, Operator.GT, Decimal("2.5"), 2),
        (HUGEINT, Operator.LT, Decimal("2.5"), 3),
        # Past 128 bits every integer is on one side: the last is the one
        # neighbour, and the other side of it is constant.
        (HUGEINT, Operator.GT, 10**40, 2**127 - 1),
        (HUGEINT, Operator.GTE, 10**40, MATCHES_NONE),
        (HUGEINT, Operator.LT, 10**40, MATCHES_ALL),
        (HUGEINT, Operator.LTE, 10**40, 2**127 - 1),
        (HUGEINT, Operator.GTE, -(10**40), -(2**127)),
        (HUGEINT, Operator.GT, -(10**40), MATCHES_ALL),
        # An integer a double cannot hold.
        (DOUBLE, Operator.EQ, 2**53 + 1, MATCHES_NONE),
        (DOUBLE, Operator.GT, 2**53 + 1, 2.0**53),
        (DOUBLE, Operator.GTE, 2**53 + 1, 2.0**53 + 2),
        # Past every double.
        (DOUBLE, Operator.GT, 10**400, sys.float_info.max),
        (DOUBLE, Operator.GTE, 10**400, MATCHES_NONE),
        (DOUBLE, Operator.LTE, 10**400, sys.float_info.max),
    ],
)
def test_the_comparand_on_each_domain_the_json_layout_reads(
    domain: NumberDomain, op: Operator, bound: Any, expected: Any
) -> None:
    compared = comparand(domain, op, bound)
    assert compared == expected
    assert type(compared) is type(expected)


def test_no_single_comparand_answers_a_range() -> None:
    """A ``BETWEEN`` is compared one side at a time, as ``GTE`` and ``LTE``."""
    with pytest.raises(ValueError, match="split it first"):
        comparand(HUGEINT, Operator.BETWEEN, 2)


def _clause(builder: SQLQueryBuilder, op: Operator, value: Any, bind: Any) -> tuple[str, list]:
    return builder.typed_clause(
        op,
        value,
        1,
        lambda reading: (None, "n") if reading == "number" else None,
        "n IS NOT NULL",
        bind=bind,
    )


def test_a_bind_hook_is_given_the_operator_each_bound_is_compared_by() -> None:
    builder = SQLQueryBuilder("t", dialect="sqlite", param_style="qmark")
    seen: list[tuple[Any, Any, Operator]] = []

    def bind(reading: str | None, bound: Any, op: Operator) -> Any:
        seen.append((reading, bound, op))
        return builder.bind_comparand(reading, bound, op)

    _sql, params = _clause(builder, Operator.BETWEEN, [2**70 + 1, 2**71 + 1], bind)
    assert seen == [
        ("number", 2**70 + 1, Operator.GTE),
        ("number", 2**71 + 1, Operator.LTE),
    ]
    assert params == [math.nextafter(float(2**70), math.inf), float(2**71)]


def test_a_bind_hook_taking_no_operator_is_called_as_it_was_written() -> None:
    """A layout written before the hook took the operator keeps working."""
    builder = SQLQueryBuilder("t", dialect="sqlite", param_style="qmark")
    seen: list[tuple[Any, Any]] = []

    def bind(reading: str | None, bound: Any) -> Any:
        seen.append((reading, bound))
        return builder.bind_bound(reading, bound)

    sql, params = _clause(builder, Operator.GT, 7, bind)
    assert seen == [("number", 7)]
    assert (sql, params) == ("n > ?", [7])


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        (Operator.EQ, 2**70 + 1, ("FALSE", [])),
        (Operator.NEQ, 2**70 + 1, ("n IS NOT NULL", [])),
        (Operator.LT, 10**400, ("TRUE", [])),
        (Operator.IN, [2**70 + 1, 5], ("n IN (SELECT value FROM json_each(?))", ["[5]"])),
        (Operator.BETWEEN, [10**400, 10**401], ("FALSE", [])),
        (Operator.NOT_BETWEEN, [10**400, 10**401], ("n IS NOT NULL", [])),
    ],
)
def test_a_constant_comparand_renders_as_its_answer(
    op: Operator, value: Any, expected: tuple[str, list]
) -> None:
    builder = SQLQueryBuilder("t", dialect="sqlite", param_style="qmark")
    assert _clause(builder, op, value, None) == expected


#: (id, value) of every JSON kind, a ``null`` and a missing value.
MIXED: list[tuple[str, Any]] = [
    ("string", "b"),
    ("string-2", "a"),
    ("number", 10),
    ("number-2", 9.5),
    ("boolean", True),
    ("boolean-2", False),
    ("array", [1, 2]),
    ("object", {"a": 1}),
    ("null", None),
]
#: Postgres's ``jsonb`` order for the kinds, ascending, then no value.
MIXED_ASCENDING = [
    "string-2",
    "string",
    "number-2",
    "number",
    "boolean-2",
    "boolean",
    "array",
    "object",
]


def _mixed_order(kind: str, config: dict[str, Any]) -> dict[SortOrder, list[str]]:
    db = SyncDatabase.from_backend(kind, config=config)
    try:
        for row_id, value in MIXED:
            db.create(Record({"v": value}, storage_id=row_id))
        db.create(Record({"u": 1}, storage_id="missing"))
        return {
            order: [str(r.storage_id) for r in db.search(Query(sort_specs=[SortSpec("v", order)]))]
            for order in SortOrder
        }
    finally:
        db.close()


def _assert_kind_order(got: dict[SortOrder, list[str]]) -> None:
    assert got[SortOrder.ASC][:8] == MIXED_ASCENDING
    assert got[SortOrder.DESC][:8] == MIXED_ASCENDING[::-1]
    for order in SortOrder:
        assert set(got[order][8:]) == {"null", "missing"}


def test_duckdb_sorts_a_field_mixing_kinds_by_postgres_kind_order() -> None:
    """Each kind sorts together, in ``jsonb``'s order, and no value sorts last."""
    with tempfile.TemporaryDirectory() as d:
        config = {"path": str(Path(d) / "r.duckdb"), "table": "records"}
        _assert_kind_order(_mixed_order("duckdb", config))


@requires_postgres
def test_postgres_sorts_a_field_mixing_kinds_in_the_same_order(
    make_postgres_test_db: Any,
) -> None:
    for config in make_postgres_test_db("test_mixed_"):
        _assert_kind_order(_mixed_order("postgres", config))


def test_a_filter_and_the_comparand_agree_on_a_stored_neighbour() -> None:
    """``GT`` a bound just past ``2**70`` keeps the float ``2**70``'s neighbour out."""
    with tempfile.TemporaryDirectory() as d:
        db = SyncDatabase.from_backend("sqlite", config={"path": str(Path(d) / "r.db")})
        try:
            db.create(Record({"n": float(2**70)}, storage_id="at"))
            db.create(Record({"n": math.nextafter(float(2**70), math.inf)}, storage_id="after"))
            found = db.search(Query(filters=[Filter("n", Operator.GT, 2**70 + 1)]))
        finally:
            db.close()
    assert [r.storage_id for r in found] == ["after"]
