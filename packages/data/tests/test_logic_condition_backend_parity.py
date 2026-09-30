"""A ``ComplexQuery`` answers as ``LogicCondition.matches`` does, on every backend.

``NOT`` is the complement of the condition it wraps. A record without the
field does not match ``colour == 'blue'``, so it matches
``NOT(colour == 'blue')``. That is a different question from
``colour != 'blue'``, which asks for a value that is not ``'blue'`` and so
never matches a missing value or a JSON ``null``.

The SQL push-down rendered ``NOT`` as ``NOT (<clause>)``. A positive
comparison on a missing field is ``NULL`` in SQL, and ``NOT NULL`` is
``NULL``, so the record was dropped on SQLite, DuckDB and PostgreSQL and kept
everywhere else. Measured before the fix, over records with a colour, with a
``null`` one and with none: ``NOT(colour == 'blue')`` returned the red and
``null`` records on SQL, and the record without a colour as well in memory.

Three more disagreements came from the same composition:

* ``NOT`` over several conditions matches a record none of them match, and
  both SQL and Elasticsearch negated only the first.
* An empty group rendered as no condition, so ``OR[]`` matched every record
  where it matches none, and ``AND[x, OR[]]`` answered as ``x``.
* ``NOT`` with no conditions raised ``IndexError`` on SQL.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from dataknobs_common.testing import (
    requires_localstack,
    requires_postgres,
    requires_real_elasticsearch,
)

from dataknobs_data import AsyncDatabase, Record, SyncDatabase
from dataknobs_data.query import Filter, Operator
from dataknobs_data.query_logic import (
    ComplexQuery,
    Condition,
    FilterCondition,
    LogicCondition,
    LogicOperator,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Record id -> its data. One field per kind, so a backend with a per-field
#: mapping holds each. ``nul`` has every field as a JSON ``null``; ``miss``
#: has none of them; ``part`` has one.
ROWS: dict[str, dict[str, Any]] = {
    "a": {"s": "red", "n": 5, "b": True, "t": "2024-01-02T03:04:05"},
    "z": {"s": "blue", "n": 1, "b": False, "t": "2023-06-01T00:00:00"},
    "nul": {"s": None, "n": None, "b": None, "t": None},
    "miss": {"other": 1},
    "part": {"s": "red"},
}

T0 = datetime(2024, 1, 1, 0, 0, 0)


def _f(field: str, op: Operator, value: Any) -> FilterCondition:
    return FilterCondition(Filter(field, op, value))


def _not(*conditions: Condition) -> LogicCondition:
    return LogicCondition(operator=LogicOperator.NOT, conditions=list(conditions))


def _and(*conditions: Condition) -> LogicCondition:
    return LogicCondition(operator=LogicOperator.AND, conditions=list(conditions))


def _or(*conditions: Condition) -> LogicCondition:
    return LogicCondition(operator=LogicOperator.OR, conditions=list(conditions))


BLUE = _f("s", Operator.EQ, "blue")
RED = _f("s", Operator.EQ, "red")

CASES: dict[str, Condition] = {
    # NOT over one comparison of each kind.
    "not string eq": _not(BLUE),
    "not number gt": _not(_f("n", Operator.GT, 3)),
    "not boolean eq": _not(_f("b", Operator.EQ, True)),
    "not timestamp gt": _not(_f("t", Operator.GT, T0)),
    "not membership": _not(_f("s", Operator.IN, ["blue", "green"])),
    "not empty membership": _not(_f("s", Operator.IN, [])),
    "not between": _not(_f("n", Operator.BETWEEN, [0, 3])),
    # NOT around a negated operator, which never matches a missing value.
    "not neq": _not(_f("s", Operator.NEQ, "blue")),
    # Nesting.
    "not not": _not(_not(BLUE)),
    "not not not": _not(_not(_not(BLUE))),
    "not and": _not(_and(BLUE, _f("n", Operator.EQ, 1))),
    "not or": _not(_or(BLUE, RED)),
    "and of nots": _and(_not(BLUE), _not(_f("n", Operator.GT, 3))),
    "or of not": _or(_not(BLUE), _f("n", Operator.EQ, 1)),
    "not across fields": _not(_or(_f("n", Operator.GT, 3), _f("b", Operator.EQ, False))),
    # NOT over several conditions: none of them matches.
    "not of two": _not(BLUE, RED),
    "not of two, one field missing": _not(BLUE, _f("n", Operator.EQ, 5)),
    # Empty groups: AND of nothing holds, OR of nothing does not, NOT of nothing holds.
    "empty or": _or(),
    "empty and": _and(),
    "and with empty or": _and(RED, _or()),
    "or with empty and": _or(BLUE, _and()),
    "not empty and": _not(_and()),
    "not empty or": _not(_or()),
    "not of nothing": _not(),
}

BACKENDS = [
    "memory",
    "file",
    "sqlite",
    "duckdb",
    pytest.param("postgres", marks=requires_postgres),
    pytest.param("s3", marks=requires_localstack),
    pytest.param("elasticsearch", marks=requires_real_elasticsearch),
]


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> Iterator[tuple[str, dict[str, Any]]]:
    """One backend's kind and constructor config, resolved in a sync fixture."""
    kind = request.param
    if kind == "postgres":
        yield from (
            (kind, c) for c in request.getfixturevalue("make_postgres_test_db")("test_not_")
        )
    elif kind == "s3":
        for config in request.getfixturevalue("make_localstack_s3_bucket")("dataknobs-not"):
            yield kind, {**config, "prefix": f"not-{uuid.uuid4().hex[:10]}/"}
    elif kind == "elasticsearch":
        yield from (
            (kind, c) for c in request.getfixturevalue("make_elasticsearch_test_index")("test_not_")
        )
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
    return [Record(data, storage_id=row_id) for row_id, data in ROWS.items()]


def _expected(condition: Condition) -> list[str]:
    return sorted(r.storage_id or "" for r in _records() if condition.matches(r))


def _ids(found: list[Record]) -> list[str]:
    return sorted(str(r.storage_id) for r in found)


#: Elasticsearch renders a negated operator as a bare ``must_not``, which keeps
#: a document without the field, so ``NOT`` around one leaves that document
#: out. That is a defect of the negated operator rather than of ``NOT``, so its
#: answer is modelled here rather than skipped, and this entry goes when it is
#: fixed.
KNOWN: dict[str, dict[str, list[str]]] = {
    "elasticsearch": {"not neq": ["z"]},
}


def _disagreements(kind: str, answers: dict[str, list[str] | str]) -> dict[str, object]:
    known = KNOWN.get(kind, {})
    return {
        name: {"got": got, "want": want}
        for name, got in answers.items()
        if got != (want := known.get(name, _expected(CASES[name])))
    }


def test_every_backend_answers_boolean_logic_as_matches_does_sync(
    backend: tuple[str, dict[str, Any]],
) -> None:
    kind, config = backend
    db = SyncDatabase.from_backend(kind, config=config)
    answers: dict[str, list[str] | str] = {}
    try:
        for record in _records():
            db.create(record)
        for name, condition in CASES.items():
            try:
                answers[name] = _ids(db.search(ComplexQuery(condition=condition)))
            except Exception as exc:  # a raise is an answer, and never the oracle's
                answers[name] = f"raises {type(exc).__name__}"
    finally:
        if kind == "s3":
            db.clear()
        db.close()
    assert _disagreements(kind, answers) == {}


async def test_every_backend_answers_boolean_logic_as_matches_does_async(
    backend: tuple[str, dict[str, Any]],
) -> None:
    kind, config = backend
    db = await AsyncDatabase.from_backend(kind, config=config)
    answers: dict[str, list[str] | str] = {}
    try:
        for record in _records():
            await db.create(record)
        for name, condition in CASES.items():
            try:
                answers[name] = _ids(await db.search(ComplexQuery(condition=condition)))
            except Exception as exc:  # a raise is an answer, and never the oracle's
                answers[name] = f"raises {type(exc).__name__}"
    finally:
        if kind == "s3":
            await db.clear()
        await db.close()
    assert _disagreements(kind, answers) == {}


class TestTheOracle:
    """What the parity tests hold every backend to, stated where it can be read."""

    def test_not_equal_is_not_the_negation_of_equal(self) -> None:
        """``!=`` asks for a value; ``NOT(==)`` asks that none be known to equal."""
        assert _expected(_f("s", Operator.NEQ, "blue")) == ["a", "part"]
        assert _expected(_not(BLUE)) == ["a", "miss", "nul", "part"]

    def test_a_missing_field_and_a_null_answer_alike(self) -> None:
        for condition in (BLUE, _not(BLUE), _f("s", Operator.NEQ, "blue")):
            assert condition.matches(Record(ROWS["miss"])) is condition.matches(Record(ROWS["nul"]))

    def test_empty_groups(self) -> None:
        everything = sorted(ROWS)
        assert _expected(_and()) == everything
        assert _expected(_or()) == []
        assert _expected(_not()) == everything
        assert _expected(_not(_and())) == []

    def test_not_of_several_matches_none_of_them(self) -> None:
        assert _expected(_not(BLUE, RED)) == ["miss", "nul"]
