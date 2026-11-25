"""Elasticsearch answers a comparison as ``Filter.matches`` does, on both twins.

A negated operator asks for a value that is not matched, so it never matches a
record without one. Elasticsearch's bare ``must_not`` kept that record. A bound
of a kind the field does not hold relates to none of its values; Elasticsearch
read ``"3"`` as a number and refused ``"red"`` or ``True``.

One field per kind, because an index maps a field once. Two differences are
left out: a bound that is not a string against a string field, where the index
holds ``5`` and ``"5"`` alike, and a field of lists, which Elasticsearch
compares element by element.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import date, datetime
from typing import Any

import pytest
from dataknobs_common.testing import requires_real_elasticsearch
from elasticsearch import BadRequestError

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

ROWS: dict[str, dict[str, Any]] = {
    "a": {"s": "red", "n": 5, "f": 7.5, "b": True, "o": {"k": "x"}},
    "z": {"s": "Blue", "n": 1, "f": 0.5, "b": False, "o": {"k": "y"}},
    "five": {"s": "5", "n": 0},
    "long": {"s": "x" * 9000},
    "nul": {"s": None, "n": None, "f": None, "b": None, "o": {"k": None}},
    "miss": {"other": 1},
}
NOON = datetime(2024, 1, 1, 12)
NAN = float("nan")

#: Bounds of every kind, for a field holding numbers or booleans or nothing.
ANY_KIND: list[Any] = [5, 0, 7.5, True, False, "red", "5", "true", NOON, date(2024, 1, 1)]
NEVER: list[Any] = [None, NAN]
ANY_RANGE: list[list[Any]] = [
    [0, 6],
    [False, True],
    ["A", "z"],
    ["0", "9"],
    [0, "z"],
    [None, 5],
    [datetime(2023, 1, 1), NOON],
]
ANY_MEMBERS: list[Any] = [
    [],
    [None],
    [NAN],
    [None, "red"],
    [None, 5],
    ("red",),
    {"red"},
    {"red": 1}.keys(),
    [5, "red"],
    ["red", 5],
    ["5"],
    [True],
    [1],
    [1, True],
    [5.0],
    [NOON],
]

SCALAR_OPS = [
    Operator.EQ,
    Operator.NEQ,
    Operator.GT,
    Operator.GTE,
    Operator.LT,
    Operator.LTE,
]
RANGE_OPS = [Operator.BETWEEN, Operator.NOT_BETWEEN]
MEMBERSHIP_OPS = [Operator.IN, Operator.NOT_IN]
PATTERN_OPS = [Operator.LIKE, Operator.NOT_LIKE, Operator.STARTS_WITH]


def _only_strings(members: Any) -> bool:
    return all(isinstance(m, str) or m is None for m in members)


def _filters() -> list[Filter]:
    found: list[Filter] = []
    for field in ("n", "f", "b", "absent"):
        found += [Filter(field, op, v) for op in SCALAR_OPS for v in ANY_KIND + NEVER]
        found += [Filter(field, op, v) for op in RANGE_OPS for v in ANY_RANGE]
        found += [Filter(field, op, v) for op in MEMBERSHIP_OPS for v in ANY_MEMBERS]
        found += [Filter(field, op, v) for op in PATTERN_OPS for v in ("r%", "5", "r")]
    for field in ("s", "o.k", "id"):
        strings = ["red", "5", "x", "a"]
        found += [Filter(field, op, v) for op in SCALAR_OPS for v in strings + NEVER]
        found += [
            Filter(field, op, v)
            for op in RANGE_OPS
            for v in (["A", "r"], ["0", "9"], [None, "z"], [0, "z"], [NAN, "z"])
        ]
        found += [
            Filter(field, op, v)
            for op in MEMBERSHIP_OPS
            for v in [*filter(_only_strings, ANY_MEMBERS), ["x", "a"], {"a", "z"}]
        ]
        found += [Filter(field, op, v) for op in PATTERN_OPS for v in ("r%", "5", "x", "a%")]
    found += [Filter("id", op, v) for op in SCALAR_OPS for v in (5, 0, True, False)]
    found += [Filter("id", op, v) for op in RANGE_OPS for v in ([0, 6], [False, True])]
    found += [Filter("id", op, v) for op in MEMBERSHIP_OPS for v in ([5, "a"], [True])]
    return found


#: A string past the keyword sub-field's limit has no value to compare with. A
#: positive comparison does not match it, so a negated one does.
TOO_LONG = {"long"}


def _value(row_id: str, field: str) -> Any:
    value: Any = row_id if field == "id" else ROWS[row_id]
    if field != "id":
        for part in field.split("."):
            value = value.get(part) if isinstance(value, dict) else None
    return value


NEGATED = {Operator.NEQ, Operator.NOT_IN, Operator.NOT_BETWEEN, Operator.NOT_LIKE}


def _expected(spec: Filter) -> list[str]:
    matched = {row_id for row_id in ROWS if spec.matches(_value(row_id, spec.field))}
    if spec.field == "s":
        matched = matched | TOO_LONG if spec.operator in NEGATED else matched - TOO_LONG
    return sorted(matched)


@pytest.fixture
def config(make_elasticsearch_test_index: Any) -> Iterator[dict[str, Any]]:
    yield from make_elasticsearch_test_index("test_compare_")


def _records() -> list[Record]:
    return [Record(data, storage_id=row_id) for row_id, data in ROWS.items()]


async def test_both_twins_answer_as_the_matcher_does(config: dict[str, Any]) -> None:
    sync_db = SyncElasticsearchDatabase(config)
    await asyncio.to_thread(sync_db.connect)
    async_db = AsyncElasticsearchDatabase(config)
    await async_db.connect()
    wrong: dict[str, Any] = {}
    try:
        for record in _records():
            await asyncio.to_thread(sync_db.create, record)
        for spec in _filters():
            query = Query(filters=[spec])
            for name in ("sync", "async"):
                got: Any
                try:
                    if name == "sync":
                        found = await asyncio.to_thread(sync_db.search, query)
                    else:
                        found = await async_db.search(query)
                    got = sorted(str(r.storage_id) for r in found)
                except Exception as exc:  # a raise is an answer, and never the matcher's
                    got = f"raises {type(exc).__name__}"
                if got != (want := _expected(spec)):
                    wrong[f"{name} {spec.field} {spec.operator.name} {spec.value!r}"] = (got, want)
    finally:
        await async_db.close()
        await asyncio.to_thread(sync_db.close)
    assert wrong == {}


def test_the_expectations_are_not_all_alike() -> None:
    """Written out, so they cannot move together with the matcher."""
    assert _expected(Filter("n", Operator.NEQ, 5)) == ["five", "z"]
    assert _expected(Filter("n", Operator.NEQ, "5")) == ["a", "five", "z"]
    assert _expected(Filter("n", Operator.GTE, "5")) == []
    assert _expected(Filter("n", Operator.NOT_LIKE, "5")) == []
    assert _expected(Filter("s", Operator.NOT_IN, [])) == ["a", "five", "long", "z"]
    assert _expected(Filter("o.k", Operator.NEQ, "x")) == ["z"]


#: Types a custom mapping can give a field, which say what it cannot hold. A date
#: field can hold a number (epoch milliseconds), so a number is not among them.
TYPED: dict[str, dict[str, Any]] = {
    "t": {"day": "2024-01-01T10:00:00", "ip": "10.0.0.1"},
    "epoch": {"day": 1704103200000},
    "miss": {"other": 1},
}
CANNOT_HOLD: dict[str, list[Any]] = {"day": [True, False], "ip": [True, 5, 0.5, NOON]}


async def test_a_bound_a_date_or_ip_field_cannot_hold_relates_to_nothing(
    config: dict[str, Any],
) -> None:
    mapped = {"day": {"type": "date"}, "ip": {"type": "ip"}}
    config["mappings"] = {"properties": {"data": {"properties": mapped}}}
    sync_db = SyncElasticsearchDatabase(config)
    await asyncio.to_thread(sync_db.connect)
    async_db = AsyncElasticsearchDatabase(config)
    await async_db.connect()
    specs = [
        spec
        for field, bounds in CANNOT_HOLD.items()
        for bound in bounds
        for spec in (
            *(Filter(field, op, bound) for op in SCALAR_OPS),
            *(Filter(field, op, [bound]) for op in MEMBERSHIP_OPS),
            *(Filter(field, op, [bound, bound]) for op in RANGE_OPS),
        )
    ]
    wrong: dict[str, Any] = {}
    try:
        for row_id, data in TYPED.items():
            await asyncio.to_thread(sync_db.create, Record(data, storage_id=row_id))
        for spec in specs:
            query = Query(filters=[spec])
            want = sorted(i for i, row in TYPED.items() if spec.matches(row.get(spec.field)))
            for name in ("sync", "async"):
                got: Any
                try:
                    if name == "sync":
                        found = await asyncio.to_thread(sync_db.search, query)
                    else:
                        found = await async_db.search(query)
                    got = sorted(str(r.storage_id) for r in found)
                except Exception as exc:  # a raise is an answer, and never the matcher's
                    got = f"raises {type(exc).__name__}"
                if got != want:
                    wrong[f"{name} {spec.field} {spec.operator.name} {spec.value!r}"] = (got, want)
        kept = await async_db.search(Query(filters=[Filter("day", Operator.GT, 0)]))
        assert "epoch" in {r.storage_id for r in kept}
    finally:
        await async_db.close()
        await asyncio.to_thread(sync_db.close)
    assert wrong == {}


@pytest.mark.parametrize("operator", [Operator.EQ, Operator.NEQ])
@pytest.mark.parametrize("field", ["n", "s"])
def test_equality_with_a_list_is_refused_rather_than_missed(
    config: dict[str, Any], field: str, operator: Operator
) -> None:
    """A number field can hold ``[1]``, so answering nothing would lose it."""
    db = SyncElasticsearchDatabase(config)
    db.connect()
    try:
        db.create(Record({"n": [1], "s": ["a"]}, storage_id="a"))
        with pytest.raises(BadRequestError):
            db.search(Query(filters=[Filter(field, operator, [1] if field == "n" else ["a"])]))
    finally:
        db.close()
