"""A comparison matches only values of its bound's JSON type, on every backend.

Schemaless storage makes a field holding values of several types ordinary,
and ``Filter.matches`` compares a value only with a bound of its own kind: a
number with a number, a string with a string, a boolean with a boolean. A
negated operator (``NEQ``, ``NOT_IN``, ``NOT_BETWEEN``) therefore matches every
present value of another kind, and no operator matches a missing value or a
JSON ``null``.

The SQL push-down disagreed three ways. A string bound compared the text
projection, so it matched numbers and booleans, and SQLite ordered every
number below every string. A numeric or boolean bound cast every value in the
field, so one string made the whole query raise on DuckDB and PostgreSQL. And a
membership list was cast by its first member, so ``IN [5, '7']`` raised or
coerced ``'7'`` to match ``7``.

The oracle had one defect of its own: it followed Python, where ``True == 1``,
so a boolean compared equal to and ordered against numbers. JSON keeps them
apart, as PostgreSQL, DuckDB and Elasticsearch store them, and so does
``Filter.matches`` now.
"""

from __future__ import annotations

import re
import tempfile
import uuid
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from dataknobs_common.testing import requires_localstack, requires_postgres

from dataknobs_data import AsyncDatabase, Query, Record, SyncDatabase
from dataknobs_data import query as query_module
from dataknobs_data.query import (
    NAIVE_TIMESTAMP_SHAPE,
    ZONED_TIMESTAMP_SHAPE,
    Filter,
    Operator,
    _values_equal,
    read_timestamp,
)
from dataknobs_data.query_logic import (
    ComplexQuery,
    FilterCondition,
    LogicCondition,
    LogicOperator,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Record id -> the value of ``t``. ``MISSING`` has no ``t`` at all; it holds
#: another field, because a record with no fields is a different subject.
ROWS: dict[str, Any] = {
    "n5": 5,
    "n75": 7.5,
    "n1": 1,
    "n0": 0,
    "bT": True,
    "bF": False,
    "sA": "A",
    "sc": "c",
    "s5": "5",
    "s1": "1",
    "sTrue": "true",
    "nul": None,
    "obj": {"a": 1},
    "arr": [1],
    "d24": "2024-01-02T03:04:05",
    "d23": "2023-06-01T00:00:00",
    "dRed": "red-2024",
    # The same day as NOON, earlier; text compares it by its separator.
    "dAm": "2024-01-01T10:00:00",
    "dAmSpace": "2024-01-01 10:00:00",
    "dMs": "2024-01-01T12:00:00.250000",
    # A zone: never ordered against the naive NOON.
    "dZ": "2024-01-02T03:04:05Z",
    "dOffset": "2024-01-02T03:04:05+01:00",
}
MISSING = "miss"
IDS = [*ROWS, MISSING]

NOON = datetime(2024, 1, 1, 12, 0, 0)

_SCALAR_BOUNDS: list[Any] = [5, 1, 0, 7.5, True, False, "B", "5", "true"]
_RANGES: list[list[Any]] = [
    [0, 6],
    [False, True],
    ["B", "d"],
    [0, "z"],
    [datetime(2023, 1, 1), NOON],
]
_MEMBERS: list[list[Any]] = [
    [5, "c"],
    [1],
    [True],
    [False, 0],
    ["5"],
    [5, "7"],
    ["true", True, 1],
    [None, 5],
    [],
    [datetime(2024, 1, 1, 10, 0, 0), "c"],
]

FILTERS: list[Filter] = [
    *(
        Filter("t", op, bound)
        for op in (Operator.EQ, Operator.NEQ, Operator.GT, Operator.GTE, Operator.LT, Operator.LTE)
        for bound in _SCALAR_BOUNDS
    ),
    *(
        Filter("t", op, bound)
        for op in (Operator.EQ, Operator.NEQ, Operator.GT, Operator.GTE, Operator.LT, Operator.LTE)
        for bound in (NOON, datetime(2024, 1, 1, 10, 0, 0))
    ),
    *(
        Filter("t", op, bounds)
        for op in (Operator.BETWEEN, Operator.NOT_BETWEEN)
        for bounds in _RANGES
    ),
    *(Filter("t", op, members) for op in (Operator.IN, Operator.NOT_IN) for members in _MEMBERS),
    # The storage key is a string, so a bound of another kind matches it only negated.
    *(
        Filter("id", op, bound)
        for op in (Operator.EQ, Operator.NEQ, Operator.GT, Operator.LT)
        for bound in (5, True, "n5")
    ),
    Filter("id", Operator.IN, [5, "n5"]),
    Filter("id", Operator.NOT_IN, [5, "n5"]),
]

#: Filters asked again under a ``NOT``: a value of another kind is unmatched by
#: each, so the ``NOT`` matches it.
NEGATED: list[Filter] = [
    Filter("t", Operator.EQ, 5),
    Filter("t", Operator.GT, "B"),
    Filter("t", Operator.LT, True),
    Filter("t", Operator.IN, [5, "c"]),
    Filter("t", Operator.NEQ, 1),
    Filter("t", Operator.GT, NOON),
]

#: A search's answer: the matching ids, or what it raised.
Answer = list[str] | str

BACKENDS = [
    "memory",
    "file",
    "sqlite",
    "duckdb",
    pytest.param("postgres", marks=requires_postgres),
    pytest.param("s3", marks=requires_localstack),
]


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> Iterator[tuple[str, dict[str, Any]]]:
    """One backend's kind and constructor config, resolved in a sync fixture."""
    kind = request.param
    if kind == "postgres":
        yield from (
            (kind, c) for c in request.getfixturevalue("make_postgres_test_db")("test_jtype_")
        )
    elif kind == "s3":
        for config in request.getfixturevalue("make_localstack_s3_bucket")("dataknobs-jtype"):
            yield kind, {**config, "prefix": f"jtype-{uuid.uuid4().hex[:10]}/"}
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


def _records() -> list[Record]:
    return [Record({"t": value}, storage_id=row_id) for row_id, value in ROWS.items()] + [
        Record({"other": 1}, storage_id=MISSING)
    ]


def _expected(spec: Filter) -> list[str]:
    if spec.field == "id":
        return sorted(row_id for row_id in IDS if spec.matches(row_id))
    return sorted(row_id for row_id, value in ROWS.items() if spec.matches(value))


def _expected_negated(spec: Filter) -> list[str]:
    return sorted(row_id for row_id in IDS if row_id not in _expected(spec))


def _negated(spec: Filter) -> ComplexQuery:
    return ComplexQuery(
        condition=LogicCondition(operator=LogicOperator.NOT, conditions=[FilterCondition(spec)])
    )


def _disagreements(
    answers: list[tuple[Filter, Answer]],
    negated: list[tuple[Filter, Answer]],
) -> dict[str, object]:
    wrong: dict[str, object] = {}
    for spec, got in answers:
        if got != (want := _expected(spec)):
            wrong[f"{spec.field} {spec.operator.value} {spec.value!r}"] = {"got": got, "want": want}
    for spec, got in negated:
        if got != (want := _expected_negated(spec)):
            key = f"NOT {spec.field} {spec.operator.value} {spec.value!r}"
            wrong[key] = {"got": got, "want": want}
    return wrong


def _ids(found: list[Record]) -> list[str]:
    return sorted(str(r.storage_id) for r in found)


def _raised(exc: Exception) -> str:
    return f"raises {type(exc).__name__}"


def test_every_backend_matches_only_the_bounds_json_type_sync(
    backend: tuple[str, dict[str, Any]],
) -> None:
    """Each filter answers as ``Filter.matches`` does, on every backend."""
    kind, config = backend
    db = SyncDatabase.from_backend(kind, config=config)

    def run(query: Query | ComplexQuery) -> Answer:
        try:
            return _ids(db.search(query))
        except Exception as exc:  # a raise is an answer, and never the oracle's
            return _raised(exc)

    try:
        for record in _records():
            db.create(record)
        answers = [(spec, run(Query(filters=[spec]))) for spec in FILTERS]
        negated = [(spec, run(_negated(spec))) for spec in NEGATED]
    finally:
        if kind == "s3":
            db.clear()
        db.close()
    assert _disagreements(answers, negated) == {}


async def test_every_backend_matches_only_the_bounds_json_type_async(
    backend: tuple[str, dict[str, Any]],
) -> None:
    kind, config = backend
    db = await AsyncDatabase.from_backend(kind, config=config)

    async def run(query: Query | ComplexQuery) -> Answer:
        try:
            return _ids(await db.search(query))
        except Exception as exc:  # a raise is an answer, and never the oracle's
            return _raised(exc)

    try:
        for record in _records():
            await db.create(record)
        answers = [(spec, await run(Query(filters=[spec]))) for spec in FILTERS]
        negated = [(spec, await run(_negated(spec))) for spec in NEGATED]
    finally:
        if kind == "s3":
            await db.clear()
        await db.close()
    assert _disagreements(answers, negated) == {}


class TestABooleanIsNotANumber:
    """``Filter.matches`` keeps booleans and numbers apart, as JSON does."""

    @pytest.mark.parametrize(
        ("spec", "value"),
        [
            (Filter("t", Operator.EQ, 1), True),
            (Filter("t", Operator.EQ, True), 1),
            (Filter("t", Operator.EQ, 0.0), False),
            (Filter("t", Operator.GT, 0), True),
            (Filter("t", Operator.LT, 5), False),
            (Filter("t", Operator.GTE, True), 1),
            (Filter("t", Operator.BETWEEN, [0, 6]), True),
            (Filter("t", Operator.IN, [1]), True),
            (Filter("t", Operator.IN, [True]), 1),
            (Filter("t", Operator.IN, {1, 2}), True),
        ],
    )
    def test_never_matches_across_the_two(self, spec: Filter, value: Any) -> None:
        assert not spec.matches(value)

    @pytest.mark.parametrize(
        ("spec", "value"),
        [
            (Filter("t", Operator.NEQ, 1), True),
            (Filter("t", Operator.NOT_IN, [True]), 1),
            (Filter("t", Operator.NOT_BETWEEN, [0, 6]), False),
        ],
    )
    def test_a_negation_matches_across_the_two(self, spec: Filter, value: Any) -> None:
        assert spec.matches(value)

    @pytest.mark.parametrize(
        ("spec", "value"),
        [
            (Filter("t", Operator.EQ, True), True),
            (Filter("t", Operator.EQ, 5), 5.0),
            (Filter("t", Operator.LT, True), False),
            (Filter("t", Operator.IN, [False, 2]), False),
            (Filter("t", Operator.IN, {1, 2}), 1),
            (Filter("t", Operator.BETWEEN, [0, 6]), 5.5),
        ],
    )
    def test_each_still_matches_its_own_kind(self, spec: Filter, value: Any) -> None:
        assert spec.matches(value)


@pytest.mark.parametrize(
    ("spec", "value", "expected"),
    [
        (Filter("t", Operator.LT, "B"), "2024-01-02T03:04:05", True),
        (Filter("t", Operator.GT, "B"), "2024-01-02T03:04:05", False),
        (Filter("t", Operator.GT, "2024-01-01"), "2024-01-02T03:04:05", True),
    ],
)
def test_a_date_string_against_one_that_is_not_compares_as_text(
    spec: Filter, value: str, expected: bool
) -> None:
    """The value was parsed before the bound failed to, leaving a datetime
    ordered against a string, which raised and read as no match either way.
    """
    assert spec.matches(value) is expected


#: A day past its month's end: the one string of timestamp shape naming no
#: real time that the shape check cannot refuse.
IMPOSSIBLE = "2024-02-30T00:00:00"


@requires_postgres
def test_a_postgres_datetime_bound_over_an_impossible_timestamp_still_raises(
    make_postgres_test_db: Any,
) -> None:
    """The stated residual of the datetime guard, held so its fix is noticed.

    A value that is not a string, or not of timestamp shape, is left out before
    the cast, and the shape bounds each field (month 01-12, day 01-31, hour
    00-23, minute and second 00-59). A day past its month's end reaches the cast,
    and PostgreSQL 15 has no cast that answers ``NULL`` instead of raising
    (``pg_input_is_valid`` arrives in 16). ``Filter.matches`` does not match it.
    """
    for config in make_postgres_test_db("test_jtype_"):
        db = SyncDatabase.from_backend("postgres", config=config)
        try:
            db.create(Record({"t": IMPOSSIBLE}, storage_id="x"))
            db.create(Record({"t": "2024-01-02T03:04:05"}, storage_id="y"))
            assert not Filter("t", Operator.GT, NOON).matches(IMPOSSIBLE)
            with pytest.raises(Exception, match="out of range"):
                db.search(Query(filters=[Filter("t", Operator.GT, NOON)]))
        finally:
            db.close()


def _answers(
    kind: str, config: dict[str, Any], values: dict[str, Any], filters: list[Filter]
) -> dict[str, dict[str, Answer]]:
    """Each filter's answer over ``values`` where it disagrees with the oracle."""
    db = SyncDatabase.from_backend(kind, config=config)
    wrong: dict[str, dict[str, Answer]] = {}
    try:
        for row_id, value in values.items():
            db.create(Record({"t": value}, storage_id=row_id))
        db.create(Record({"other": 1}, storage_id=MISSING))
        for spec in filters:
            try:
                got: Answer = _ids(db.search(Query(filters=[spec])))
            except Exception as exc:  # a raise is an answer, and never the oracle's
                got = _raised(exc)
            if spec.field == "id":
                want = sorted(r for r in [*values, MISSING] if spec.matches(r))
            else:
                want = sorted(r for r, value in values.items() if spec.matches(value))
            if got != want:
                wrong[f"{spec.operator.value} {spec.value!r}"] = {"got": got, "want": want}
    finally:
        if kind == "s3":
            db.clear()
        db.close()
    return wrong


#: Strings of timestamp shape naming no real time, beside two that name one.
#: SQLite reads a day past the month's end as the next month and keeps hour 24;
#: DuckDB and PostgreSQL read hour 24 as the next midnight. ``Filter.matches``
#: reads none of them as a time, so each is a string of another kind.
NOT_A_TIME: dict[str, Any] = {
    "month13": "2024-13-45T00:00:00",
    "feb30": "2024-02-30T12:00:00",
    "hour24": "2024-01-01T24:00:00",
    "minute60": "2024-01-01T23:60:00",
    "mar1": "2024-03-01T12:00:00",
    "jan2": "2024-01-02T00:00:00",
}
_MAR1_NOON = datetime(2024, 3, 1, 12, 0, 0)
_JAN2 = datetime(2024, 1, 2)
NOT_A_TIME_FILTERS: list[Filter] = [
    *(
        Filter("t", op, bound)
        for op in (Operator.EQ, Operator.NEQ, Operator.GT, Operator.LT)
        for bound in (_MAR1_NOON, _JAN2, NOON)
    ),
    *(Filter("t", op, [_MAR1_NOON, _JAN2]) for op in (Operator.IN, Operator.NOT_IN)),
    *(
        Filter("t", op, [NOON, datetime(2024, 12, 31)])
        for op in (Operator.BETWEEN, Operator.NOT_BETWEEN)
    ),
]


def test_a_string_naming_no_real_time_is_not_a_timestamp(
    backend: tuple[str, dict[str, Any]],
) -> None:
    """Of timestamp shape, it is a string: unmatched, and matched by a negation.

    PostgreSQL 15 raises on a day past the month's end instead, the residual
    pinned below, so that row is left out there.
    """
    kind, config = backend
    values = {k: v for k, v in NOT_A_TIME.items() if not (kind == "postgres" and k == "feb30")}
    assert _answers(kind, config, values, NOT_A_TIME_FILTERS) == {}


#: A range that is not two bounds: one, three, a string, a set (no order), a
#: mapping. SQL raised a driver error, used two of three bounds, or took a
#: two-character string's characters as the bounds; ``Filter.matches`` answered
#: it silently. It is refused where the filter is built instead, as a
#: membership value that is not a list of candidates already is.
MALFORMED_RANGES: list[Any] = [5, None, [1], [1, 2, 3], "ab", {1, 2}, {"lo": 1, "hi": 2}]


@pytest.mark.parametrize("op", [Operator.BETWEEN, Operator.NOT_BETWEEN])
@pytest.mark.parametrize("bounds", MALFORMED_RANGES)
def test_a_malformed_range_is_refused_when_the_filter_is_built(op: Operator, bounds: Any) -> None:
    with pytest.raises(ValueError, match="needs two bounds"):
        Filter("t", op, bounds)
    with pytest.raises(ValueError, match="needs two bounds"):
        Filter.from_dict({"field": "t", "operator": op.value, "value": bounds})
    with pytest.raises(ValueError, match="needs two bounds"):
        Query().filter("t", op, bounds)


@pytest.mark.parametrize("op", [Operator.BETWEEN, Operator.NOT_BETWEEN])
@pytest.mark.parametrize("bounds", [[1, 5], (1, 5), ["a", "m"], [None, 5]])
def test_a_list_or_tuple_of_two_bounds_is_a_range(op: Operator, bounds: Any) -> None:
    assert Filter("t", op, bounds).value == bounds


class _CountedMember:
    """A member that counts how often it is compared for equality."""

    compared = 0

    def __init__(self, n: int) -> None:
        self.n = n

    def __hash__(self) -> int:
        return hash(("counted", self.n))

    def __eq__(self, other: object) -> bool:
        type(self).compared += 1
        return isinstance(other, _CountedMember) and other.n == self.n


@pytest.mark.parametrize("value", [5, 5.5, True, "a-slug-or-uuid", "plain", date(2024, 1, 1)])
def test_membership_in_a_set_is_a_lookup_not_a_scan(value: Any) -> None:
    """A set of members is probed by hash, whatever the value's kind.

    Only a member that could equal the value across kinds is compared one by
    one: a date or datetime, or a string naming one. None is here.
    """
    members = frozenset(_CountedMember(i) for i in range(1000))
    spec = Filter("t", Operator.IN, members)
    _CountedMember.compared = 0
    for _ in range(3):
        assert not spec.matches(value)
    assert _CountedMember.compared == 0


@pytest.mark.parametrize(
    ("members", "value", "expected"),
    [
        (frozenset({1, 2}), True, False),
        (frozenset({True}), 1, False),
        (frozenset({True}), True, True),
        (frozenset({1.0}), 1, True),
        (frozenset({"2024-01-01"}), date(2024, 1, 1), True),
        (frozenset({"2024-01-01T00:00:00"}), date(2024, 1, 1), True),
        (frozenset({datetime(2024, 1, 1)}), "2024-01-01T00:00:00", True),
        (frozenset({date(2024, 1, 1)}), datetime(2024, 1, 1), True),
        (frozenset({"x", ("a",)}), ("a",), True),
        ([[1, 2]], [1, 2], True),
        (["a-b", 3], "a-b", True),
    ],
)
def test_membership_answers_as_equality_does(members: Any, value: Any, expected: bool) -> None:
    spec = Filter("t", Operator.IN, members)
    assert spec.matches(value) is expected
    assert spec.matches(value) is any(_values_equal(value, member) for member in members)


#: Bounds of every kind ``value_kind`` names, and the strings that fall either
#: side of "names a time". ``Filter.matches`` reads only ISO extended form as a
#: time, so the basic, week, hour-only and comma forms, and year 0000, are
#: strings. The row keyed ``2024-01-01`` is there for the ``id`` filters.
EVERY_KIND: dict[str, Any] = {
    "dOnly": "2024-01-01",
    "dT": "2024-01-01T00:00:00",
    "dNoon": "2024-01-01T12:00:00",
    "basic": "20240101",
    "week": "2024-W01-1",
    "hourOnly": "2024-01-01T12",
    "comma": "2024-01-01T12:30:00,5",
    "y0": "0000-01-01",
    "n5": 5,
    "n55": 5.5,
    "bT": True,
    "x": "x",
    "2024-01-01": "a key that names a time",
}
_EVERY_BOUND: list[Any] = [
    date(2024, 1, 1),
    datetime(2023, 1, 1),
    Decimal("5"),
    Decimal("5.5"),
    np.int64(5),
    np.float64(5.5),
    np.True_,
    None,
    float("nan"),
]
EVERY_KIND_FILTERS: list[Filter] = [
    *(
        Filter("t", op, bound)
        for op in (Operator.EQ, Operator.NEQ, Operator.GT, Operator.LT)
        for bound in _EVERY_BOUND
    ),
    *(
        Filter("t", op, bounds)
        for op in (Operator.BETWEEN, Operator.NOT_BETWEEN)
        for bounds in (
            [date(2023, 1, 1), datetime(2025, 1, 1)],
            [None, None],
            [Decimal(1), 10],
        )
    ),
    *(
        Filter("t", op, members)
        for op in (Operator.IN, Operator.NOT_IN)
        for members in (
            [date(2024, 1, 1), "x"],
            [None, float("nan")],
            [np.True_],
            [Decimal("5")],
        )
    ),
    *(
        Filter("id", op, bound)
        for op in (Operator.EQ, Operator.NEQ, Operator.GT, Operator.LT)
        for bound in (datetime(2000, 1, 1), date(2024, 1, 1), None)
    ),
]


def test_every_kind_of_bound_answers_as_the_oracle_does(
    backend: tuple[str, dict[str, Any]],
) -> None:
    """A date, Decimal, numpy scalar, ``None`` or NaN bound, and ``id`` against a time.

    Each took the untyped comparison: a date or Decimal raised on DuckDB, a
    ``None`` or NaN left a negation ``NULL`` and so matched nothing, and ``id``
    against a datetime matched every key on SQLite and raised on DuckDB.
    """
    kind, config = backend
    assert _answers(kind, config, EVERY_KIND, EVERY_KIND_FILTERS) == {}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2024-01-02", datetime(2024, 1, 2)),
        ("2024-01-02T10:30", datetime(2024, 1, 2, 10, 30)),
        ("2024-01-02 10:30:05.25", datetime(2024, 1, 2, 10, 30, 5, 250000)),
        ("2024-01-02T10:30:05Z", datetime(2024, 1, 2, 10, 30, 5, tzinfo=UTC)),
        (
            "2024-01-02 10:30-05:00",
            datetime(2024, 1, 2, 10, 30, tzinfo=timezone(-timedelta(hours=5))),
        ),
        ("2024-01-02Z", None),
        ("2024-01-02T10:30+24:00", None),
        ("2024-01-02T10:30+0500", None),
        ("2024-01-02T10:30z", None),
        ("20240102", None),
        ("2024-W01-2", None),
        ("2024-01-02T10", None),
        ("2024-01-02T10:30:05,5", None),
        ("0000-01-01", None),
        ("2024-02-30", None),
        ("2024-01-01T24:00:00", None),
        ("2024-01-01", datetime(2024, 1, 1)),
    ],
)
def test_one_reading_of_a_timestamp(text: str, expected: datetime | None) -> None:
    """The oracle's reading, and the shapes the SQL backends test against.

    The shape admits a day past its month's end, which no regular expression
    refuses cheaply; each engine refuses it by its own reading of the date.
    """
    assert read_timestamp(text) == expected
    naive = expected is not None and expected.tzinfo is None
    zoned = expected is not None and expected.tzinfo is not None
    assert (re.match(NAIVE_TIMESTAMP_SHAPE, text) is not None) is (naive or text == "2024-02-30")
    assert (re.match(ZONED_TIMESTAMP_SHAPE, text) is not None) is zoned


_POOL: list[Any] = [
    True,
    False,
    np.True_,
    1,
    1.0,
    0,
    5,
    Decimal("5"),
    np.int64(5),
    float("nan"),
    None,
    "x",
    "a-b",
    "2024-01-01",
    "2024-01-01T00:00:00",
    "2024-01-01T00:00:00+00:00",
    "20240101",
    date(2024, 1, 1),
    datetime(2024, 1, 1),
    datetime(2024, 1, 1, 12),
    datetime(2024, 1, 1, tzinfo=UTC),
    ("a",),
    ["a"],
]


@pytest.mark.parametrize("member", _POOL, ids=repr)
def test_membership_is_equality_against_some_member(member: Any) -> None:
    """``IN [m]`` matches a value exactly when ``EQ m`` does, for every pairing."""
    for value in _POOL:
        if value is None:
            continue  # a missing value matches no operator but EXISTS
        want = Filter("t", Operator.EQ, member).matches(value)
        assert Filter("t", Operator.IN, [member]).matches(value) is want, (value, member)


def test_a_date_value_is_not_compared_member_by_member(monkeypatch: pytest.MonkeyPatch) -> None:
    """A date against a set of strings is a hash probe, not a parse of each."""
    calls = 0
    real = query_module._values_equal

    def counted(a: Any, b: Any) -> bool:
        nonlocal calls
        calls += 1
        return real(a, b)

    monkeypatch.setattr(query_module, "_values_equal", counted)
    spec = Filter("t", Operator.IN, frozenset(str(uuid.uuid4()) for _ in range(1000)))
    for _ in range(3):
        assert not spec.matches(date(2024, 1, 1))
    assert calls == 0


def test_a_filter_keeps_its_own_copy_of_a_list(tmp_path: Path) -> None:
    """Appending to the caller's list changes neither the filter nor its answer."""
    members = ["a"]
    spec = Filter("t", Operator.IN, members)
    before = hash(spec)
    assert not spec.matches("b")
    members.append("b")
    bounds = [1, 5]
    ranged = Filter("t", Operator.BETWEEN, bounds)
    bounds.append(9)
    assert spec.value == ["a"] and hash(spec) == before
    assert not spec.matches("b")
    assert ranged.matches(3)
    db = SyncDatabase.from_backend("sqlite", config={"path": str(tmp_path / "r.db")})
    try:
        db.create(Record({"t": "b"}, storage_id="b"))
        db.create(Record({"t": 3}, storage_id="three"))
        assert db.search(Query(filters=[spec])) == []
        assert _ids(db.search(Query(filters=[ranged]))) == ["three"]
    finally:
        db.close()


#: Strings that name times in forms that order differently as text and as
#: times: a date alone, a space for the ``T``, a fraction, and an offset.
TIME_STRINGS: dict[str, Any] = {
    "dOnly": "2024-01-01",
    "sp10": "2024-01-01 10:00:00",
    "t09": "2024-01-01T09:00:00",
    "t0930f": "2024-01-01T09:30:00.5",
    "off": "2024-01-01T08:00:00+05:00",
    "x": "x",
}
TIME_STRING_FILTERS: list[Filter] = [
    *(
        Filter("t", op, bound)
        for op in (Operator.EQ, Operator.GT, Operator.GTE, Operator.LT, Operator.LTE)
        for bound in ("2024-01-01T00:00:00", "2024-01-01T09:30:00", "2024-01-01")
    ),
    *(
        Filter("t", op, ["2024-01-01T00:00:00", "2024-01-01T09:30:00"])
        for op in (Operator.BETWEEN, Operator.NOT_BETWEEN)
    ),
]


@pytest.mark.parametrize(
    ("spec", "value", "expected"),
    [
        (Filter("t", Operator.GTE, "2024-01-01T00:00:00"), "2024-01-01", False),
        (Filter("t", Operator.LTE, "2024-01-01T00:00:00"), "2024-01-01", True),
        (Filter("t", Operator.EQ, "2024-01-01T00:00:00"), "2024-01-01", False),
        (Filter("t", Operator.GT, "2024-01-01T09:30:00"), "2024-01-01 10:00:00", False),
        (Filter("t", Operator.GT, "2024-01-01T00:00:00"), "2024-01-01T08:00:00+05:00", True),
    ],
)
def test_two_strings_compare_as_text(spec: Filter, value: str, expected: bool) -> None:
    """A string bound orders a string value by code point, whatever either names.

    The bound's kind decides: a ``date`` or ``datetime`` bound reads a stored
    string as the time it names; a string bound compares text.
    """
    assert spec.matches(value) is expected


def test_a_memory_filter_agrees_with_the_memory_sort() -> None:
    """``t > b`` selects exactly what an ascending sort puts after ``b``."""
    db = SyncDatabase.from_backend("memory", config={})
    for row_id, value in TIME_STRINGS.items():
        db.create(Record({"t": value}, storage_id=row_id))
    order = [r.storage_id for r in db.search(Query().sort_by("t", "asc"))]
    for row_id, bound in TIME_STRINGS.items():
        after = set(order[order.index(row_id) + 1 :])
        got = {r.storage_id for r in db.search(Query(filters=[Filter("t", Operator.GT, bound)]))}
        assert got == after, bound


def test_two_strings_compare_as_text_on_every_backend(
    backend: tuple[str, dict[str, Any]],
) -> None:
    kind, config = backend
    assert _answers(kind, config, TIME_STRINGS, TIME_STRING_FILTERS) == {}
    # The oracle itself: every answer is the text answer.
    for spec in TIME_STRING_FILTERS:
        for value in TIME_STRINGS.values():
            assert spec.matches(value) is _text_answer(spec, value), (spec, value)


def _text_answer(spec: Filter, value: str) -> bool:
    bound = spec.value
    return {
        Operator.EQ: lambda: value == bound,
        Operator.GT: lambda: value > bound,
        Operator.GTE: lambda: value >= bound,
        Operator.LT: lambda: value < bound,
        Operator.LTE: lambda: value <= bound,
        Operator.BETWEEN: lambda: bound[0] <= value <= bound[1],
        Operator.NOT_BETWEEN: lambda: not bound[0] <= value <= bound[1],
    }[spec.operator]()


_PLUS5 = timezone(timedelta(hours=5))
_MINUS5 = timezone(timedelta(hours=-5))

#: Naive and zoned times, and strings a zone makes no time. The zoned rows at
#: 05:00Z name one instant in three spellings; ``zEve`` is 2024-01-01 in UTC
#: and 2023-12-31 on its own clock; ``zMidnight`` is its own day's midnight.
ZONED: dict[str, Any] = {
    "nAm": "2024-01-01T10:00:00",
    "nEve": "2023-12-31T20:00:00",
    "nDate": "2024-01-01",
    "zPlus5": "2024-01-01T10:00:00+05:00",
    "zZ": "2024-01-01T05:00:00Z",
    "zSpace": "2024-01-01 05:00Z",
    "zMinus5": "2024-01-01T23:00:00-05:00",
    "zEve": "2023-12-31T22:00:00-05:00",
    "zMidnight": "2024-01-01T00:00:00+05:00",
    "zFrac": "2024-01-01T05:00:00.250000+00:00",
    "zDateOnly": "2024-01-01Z",
    "zHour24": "2024-01-01T10:00:00+24:00",
    "zNoColon": "2024-01-01T10:00:00+0500",
    "zLower": "2024-01-01T05:00:00z",
    "zFeb30": "2024-02-30T10:00:00Z",
    # In UTC, a day either side of the years Python and SQLite share.
    "zYear1": "0001-01-01T01:00:00+05:00",
    "zYear1Late": "0001-01-01T20:00:00-05:00",
    "zYear10000": "9999-12-31T23:00:00-05:00",
    "2024-01-01T05:00:00Z": "a key that names a zoned time",
    "x": "x",
}
_AWARE = datetime(2024, 1, 1, 5, tzinfo=UTC)
_ZONED_BOUNDS: list[Any] = [
    _AWARE,
    datetime(2024, 1, 1, 10, tzinfo=_PLUS5),  # the same instant, another zone
    datetime(2024, 1, 1, 5, 0, 0, 250000, tzinfo=UTC),
    datetime(2024, 1, 1, 10),
    date(2024, 1, 1),
    datetime(1, 1, 1, 1, tzinfo=_PLUS5),
    datetime(1, 1, 1, 20, tzinfo=_MINUS5),
    datetime(9999, 12, 31, 20, tzinfo=_MINUS5),
]
ZONED_FILTERS: list[Filter] = [
    *(
        Filter("t", op, bound)
        for op in (
            Operator.EQ,
            Operator.NEQ,
            Operator.GT,
            Operator.GTE,
            Operator.LT,
            Operator.LTE,
        )
        for bound in _ZONED_BOUNDS
    ),
    *(
        Filter("t", op, bounds)
        for op in (Operator.BETWEEN, Operator.NOT_BETWEEN)
        for bounds in (
            [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, 6, tzinfo=_MINUS5)],
            # A date and an aware bound: each value's own day, then its instant.
            [date(2024, 1, 1), datetime(2024, 1, 1, 6, tzinfo=UTC)],
            [date(2023, 12, 31), datetime(2024, 1, 1, 12)],
            [datetime(2023, 1, 1), _AWARE],
        )
    ),
    *(
        Filter("t", op, members)
        for op in (Operator.IN, Operator.NOT_IN)
        for members in (
            [_AWARE, datetime(2024, 1, 1, 10)],
            [date(2024, 1, 1), datetime(2024, 1, 2, 4, tzinfo=UTC)],
            [datetime(2024, 1, 1, 10, tzinfo=_PLUS5), "x"],
        )
    ),
    *(
        Filter("id", op, bound)
        for op in (Operator.EQ, Operator.NEQ, Operator.GT, Operator.LT)
        for bound in (_AWARE, date(2024, 1, 1))
    ),
]


def test_an_aware_bound_matches_zoned_times_by_instant(
    backend: tuple[str, dict[str, Any]],
) -> None:
    """Naive and aware times are different kinds, as Python has them.

    An aware bound matches only zoned values, by instant; a naive bound only
    naive values; a date bound both, each by the value's own wall-clock day;
    and a negated operator matches a value of the other kind. SQL read only
    naive strings as times, so an aware bound matched naive values there ---
    SQLite dropping the bound's zone, DuckDB and PostgreSQL reading the value
    in the session time zone --- and never a zoned one.

    PostgreSQL 15 raises on a day past the month's end, the residual pinned
    above, so that row is left out there.
    """
    kind, config = backend
    values = {k: v for k, v in ZONED.items() if not (kind == "postgres" and k == "zFeb30")}
    assert _answers(kind, config, values, ZONED_FILTERS) == {}


#: Session time zones far from UTC and from each other.
_SESSION_ZONES = ["Pacific/Kiritimati", "Pacific/Pago_Pago"]


@pytest.mark.parametrize("zone", _SESSION_ZONES)
def test_duckdb_answers_the_same_in_every_session_time_zone(tmp_path: Path, zone: str) -> None:
    """DuckDB read a naive value against an aware bound in its session zone."""
    db = SyncDatabase.from_backend("duckdb", config={"path": str(tmp_path / "r.duckdb")})
    try:
        assert db.conn is not None
        db.conn.execute(f"SET TimeZone = '{zone}'")
        assert _zoned_disagreements(db) == {}
    finally:
        db.close()


@requires_postgres
@pytest.mark.parametrize("zone", _SESSION_ZONES)
def test_postgres_answers_the_same_in_every_session_time_zone(
    make_postgres_test_db: Any, monkeypatch: pytest.MonkeyPatch, zone: str
) -> None:
    """PostgreSQL read a naive value against an aware bound in its session zone.

    libpq sets the session's ``TimeZone`` from ``PGTZ`` when it connects.
    """
    monkeypatch.setenv("PGTZ", zone)
    for config in make_postgres_test_db("test_jtype_"):
        db = SyncDatabase.from_backend("postgres", config=config)
        try:
            assert _zoned_disagreements(db) == {}
        finally:
            db.close()


def _zoned_disagreements(db: SyncDatabase) -> dict[str, Answer]:
    values = {k: v for k, v in ZONED.items() if k != "zFeb30"}
    for row_id, value in values.items():
        db.create(Record({"t": value}, storage_id=row_id))
    wrong: dict[str, Answer] = {}
    for spec in ZONED_FILTERS:
        if spec.field != "t":
            continue
        want = sorted(r for r, value in values.items() if spec.matches(value))
        if (got := _ids(db.search(Query(filters=[spec])))) != want:
            wrong[f"{spec.operator.value} {spec.value!r}"] = got
    return wrong
