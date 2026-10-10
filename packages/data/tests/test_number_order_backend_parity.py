"""Every backend compares and sorts a JSON number as a number, exactly.

``Filter.matches`` compares Python numbers exactly: ``2**53 + 1`` is not
``2**53``, and the float ``1e30`` is not the integer ``10**30``. The in-memory
sort orders them the same way. Two SQL engines did not:

- DuckDB read a JSON field for a sort as its JSON text, so ``[9, 12, 100]``
  sorted ascending as ``[100, 12, 9]``, and read every number for a filter as a
  ``DOUBLE``, so ``2**70`` equalled ``2**70 + 1``.
- SQLite cannot bind an integer outside 64 bits, so a filter with such a bound
  raised ``OverflowError`` for every operator but membership.

Each backend is asked the same comparisons and sorts and held to the oracle's
answers. Where an engine cannot hold a stored value exactly, the oracle is
asked about the value the engine holds (:func:`_held`), so a change to either
limit fails here too.
"""

from __future__ import annotations

import tempfile
import uuid
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from dataknobs_common.testing import requires_localstack, requires_postgres

from dataknobs_data import AsyncDatabase, Query, Record, SyncDatabase
from dataknobs_data.query import Filter, Operator, SortOrder, SortSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

#: (id, n). Integers either side of 2**53, 2**63, 2**64 and 2**70, floats that
#: equal or neighbour some of them, and an integer past 128 bits.
ROWS: list[tuple[str, int | float]] = [
    ("zero", 0),
    ("five", 5),
    ("minus-five", -5),
    ("tiny", 1e-5),
    ("minus-half", -0.5),
    ("two-and-a-half", 2.5),
    ("ten-float", 10.0),
    ("two-53", 2**53),
    ("two-53-plus-1", 2**53 + 1),
    ("two-62", 2**62),
    ("minus-two-62", -(2**62)),
    ("int64-max", 2**63 - 1),
    ("two-64", 2**64),
    ("two-70", 2**70),
    ("two-70-plus-1", 2**70 + 1),
    ("two-70-float", float(2**70)),
    ("e30-float", 1e30),
    ("minus-e30-float", -1e30),
    ("ten-40", 10**40),
]
IDS = [row_id for row_id, _ in ROWS]
N_OF = dict(ROWS)

BOUNDS: list[Any] = [
    0,
    2.5,
    Decimal("2.5"),
    2**53,
    2**53 + 1,
    2**63 - 1,
    2**63,
    -(2**63) - 1,
    2**70,
    2**70 + 1,
    -(2**70),
    10**30,
    1e30,
    10**40,
    1e300,
]
MEMBERS: list[list[Any]] = [
    [2**70, 5],
    [2**70 + 1],
    [2**53, 10**30, 2.5],
    [2**63, 1e30, -(2**70)],
]
RANGES: list[list[Any]] = [
    [2**53, 2**70],
    [2.5, 2**63],
    [-(2**70), 0],
    [1e30, 10**40],
    [2**53 + 1, 2**70 + 1],
]

FILTERS: list[Filter] = (
    [
        Filter("n", op, bound)
        for op in (
            Operator.EQ,
            Operator.NEQ,
            Operator.GT,
            Operator.GTE,
            Operator.LT,
            Operator.LTE,
        )
        for bound in BOUNDS
    ]
    + [Filter("n", op, members) for op in (Operator.IN, Operator.NOT_IN) for members in MEMBERS]
    + [
        Filter("n", op, bounds)
        for op in (Operator.BETWEEN, Operator.NOT_BETWEEN)
        for bounds in RANGES
    ]
)

SORTS = [SortSpec("n", order) for order in SortOrder]

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
            (kind, c) for c in request.getfixturevalue("make_postgres_test_db")("test_number_")
        )
    elif kind == "s3":
        for config in request.getfixturevalue("make_localstack_s3_bucket")("dataknobs-number"):
            yield kind, {**config, "prefix": f"number-{uuid.uuid4().hex[:10]}/"}
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


def _held(kind: str, value: float) -> int | float | Decimal:
    """The number an engine holds for a stored value, where it cannot hold it exactly.

    - SQLite reads a JSON integer outside 64 bits as a ``REAL``.
    - DuckDB reads a JSON integer outside 128 bits as a ``DOUBLE``.
    - PostgreSQL's ``jsonb`` holds a float as the decimal Python wrote for it,
      so the float ``1e30`` is held as the integer ``10**30``. How a float is
      stored there is its own question, not this test's: the model pins it so
      that a change to it fails here.
    """
    if kind == "sqlite" and isinstance(value, int) and not -(2**63) <= value < 2**63:
        return float(value)
    if kind == "duckdb" and isinstance(value, int) and not -(2**127) <= value < 2**127:
        return float(value)
    if kind == "postgres" and isinstance(value, float):
        return Decimal(repr(value))
    return value


def _sent(kind: str, flavour: str, bound: Any) -> Any:
    """A bound as the engine compares it, where it is not the bound itself.

    psycopg2 sends a ``float`` as the decimal Python writes for it, as
    ``jsonb`` holds a stored one; asyncpg sends its exact value.
    """
    if isinstance(bound, list):
        return [_sent(kind, flavour, b) for b in bound]
    if kind == "postgres" and flavour == "sync" and isinstance(bound, float):
        return Decimal(repr(bound))
    return bound


def _expected_filter(kind: str, flavour: str, spec: Filter) -> list[str]:
    asked = Filter(spec.field, spec.operator, _sent(kind, flavour, spec.value))
    return sorted(row_id for row_id, value in ROWS if asked.matches(_held(kind, value)))


def _disagreements(
    kind: str,
    flavour: str,
    filtered: list[tuple[Filter, Answer]],
    ordered: list[tuple[SortSpec, Answer]],
) -> dict[str, object]:
    wrong: dict[str, object] = {}
    for spec, got in filtered:
        want = _expected_filter(kind, flavour, spec)
        if got != want:
            wrong[f"n {spec.operator.value} {spec.value!r}"] = {"got": got, "want": want}
    for sort, got in ordered:
        # Equal numbers (2**70 and its float) may come back in either order,
        # so the order is compared as the numbers held, not as ids.
        want_numbers = sorted(
            (_held(kind, v) for v in N_OF.values()), reverse=sort.order == SortOrder.DESC
        )
        got_numbers = got if isinstance(got, str) else [_held(kind, N_OF[i]) for i in got]
        if got_numbers != want_numbers:
            wrong[f"sort n {sort.order.value}"] = {"got": got, "want": want_numbers}
    return wrong


def _ids(found: list[Record]) -> list[str]:
    return [str(r.storage_id) for r in found]


def _raised(exc: Exception) -> str:
    return f"raises {type(exc).__name__}: {exc}"


def _sorted(answer: Answer) -> Answer:
    """A filter's matches as a set; the order they came back in is not asked."""
    return answer if isinstance(answer, str) else sorted(answer)


def test_every_backend_compares_and_sorts_numbers_exactly_sync(
    backend: tuple[str, dict[str, Any]],
) -> None:
    """Comparison and sort agree with the oracle on every backend.

    DuckDB sorted by JSON text and compared through ``DOUBLE``; SQLite raised
    for a bound outside 64 bits.
    """
    kind, config = backend
    db = SyncDatabase.from_backend(kind, config=config)

    def run(query: Query) -> Answer:
        try:
            return _ids(db.search(query))
        except Exception as exc:  # a raise is an answer, and never the oracle's
            return _raised(exc)

    try:
        for row_id, value in ROWS:
            db.create(Record({"n": value}, storage_id=row_id))
        filtered = [(f, _sorted(run(Query(filters=[f])))) for f in FILTERS]
        ordered = [(s, run(Query(sort_specs=[s]))) for s in SORTS]
    finally:
        if kind == "s3":
            db.clear()
        db.close()
    assert _disagreements(kind, "sync", filtered, ordered) == {}


async def test_every_backend_compares_and_sorts_numbers_exactly_async(
    backend: tuple[str, dict[str, Any]],
) -> None:
    kind, config = backend
    db = await AsyncDatabase.from_backend(kind, config=config)

    async def run(query: Query) -> Answer:
        try:
            return _ids(await db.search(query))
        except Exception as exc:  # a raise is an answer, and never the oracle's
            return _raised(exc)

    try:
        for row_id, value in ROWS:
            await db.create(Record({"n": value}, storage_id=row_id))
        filtered = [(f, _sorted(await run(Query(filters=[f])))) for f in FILTERS]
        ordered = [(s, await run(Query(sort_specs=[s]))) for s in SORTS]
    finally:
        if kind == "s3":
            await db.clear()
        await db.close()
    assert _disagreements(kind, "async", filtered, ordered) == {}
