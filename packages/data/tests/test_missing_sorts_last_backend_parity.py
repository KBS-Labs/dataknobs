"""A record with no value for the sorted field sorts last, in both directions.

A field a record lacks, and a field holding ``null``, are both "no value",
and every backend puts those records after every record that has one, whether
the sort is ascending or descending. Before, each backend placed them its own
way: the in-memory sort raised ``TypeError`` over a sparse number field (it
read a missing value as ``""``), SQLite put them first ascending, and
PostgreSQL first descending, while DuckDB already put them last.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from dataknobs_common.testing import (
    requires_localstack,
    requires_postgres,
    requires_real_elasticsearch,
)

from dataknobs_data import AsyncDatabase, Query, Record, SyncDatabase
from dataknobs_data.query import SortOrder, SortSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

#: (id, data). Two records with no ``n`` and no ``t``: one lacks the fields,
#: one holds ``null`` for both.
ROWS: list[tuple[str, dict[str, Any]]] = [
    ("k0", {"t": "b", "n": 2}),
    ("k1", {"u": 1}),
    ("k2", {"t": "a", "n": 1}),
    ("k3", {"t": None, "n": None}),
    ("k4", {"t": "c", "n": 10}),
]
#: The records that have each field, in ascending order.
PRESENT_ASCENDING = ["k2", "k0", "k4"]
NO_VALUE = {"k1", "k3"}

SORTS = [SortSpec(field, order) for field in ("t", "n") for order in SortOrder]

#: A search's answer: the ids in order, or what it raised.
Answer = list[str] | str

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
            (kind, c) for c in request.getfixturevalue("make_postgres_test_db")("test_missing_")
        )
    elif kind == "s3":
        for config in request.getfixturevalue("make_localstack_s3_bucket")("dataknobs-missing"):
            yield kind, {**config, "prefix": f"missing-{uuid.uuid4().hex[:10]}/"}
    elif kind == "elasticsearch":
        yield from (
            (kind, c)
            for c in request.getfixturevalue("make_elasticsearch_test_index")("test_missing_")
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


def _known_answer(kind: str, flavour: str, sort: SortSpec) -> str | None:
    """A backend's answer where a defect outside this test's subject decides it."""
    if kind == "elasticsearch" and flavour == "async" and sort.field == "t":
        # The async backend sorts a text field without its keyword
        # sub-field, and Elasticsearch refuses the request.
        return "raises BadRequestError"
    if kind == "elasticsearch" and flavour == "sync" and sort.field == "n":
        # The sync backend sorts a field on its keyword sub-field unless its
        # name is on a fixed list of numeric names, and a number field has
        # none, so Elasticsearch refuses the request.
        return "raises BadRequestError"
    return None


def _disagreements(kind: str, flavour: str, ordered: list[tuple[SortSpec, Answer]]) -> dict:
    wrong: dict[str, object] = {}
    for sort, got in ordered:
        known = _known_answer(kind, flavour, sort)
        if known is not None:
            if got != known:
                wrong[f"sort {sort.field} {sort.order.value}"] = {"got": got, "want": known}
            continue
        present = PRESENT_ASCENDING[:: -1 if sort.order == SortOrder.DESC else 1]
        # The records with no value come last; their order among themselves
        # is not asked.
        if isinstance(got, str) or got[:3] != present or set(got[3:]) != NO_VALUE:
            want = f"{present} then {sorted(NO_VALUE)} in either order"
            wrong[f"sort {sort.field} {sort.order.value}"] = {"got": got, "want": want}
    return wrong


def _ids(found: list[Record]) -> list[str]:
    return [str(r.storage_id) for r in found]


def _raised(exc: Exception) -> str:
    return f"raises {type(exc).__name__}"


def test_a_record_with_no_value_sorts_last_sync(backend: tuple[str, dict[str, Any]]) -> None:
    kind, config = backend
    db = SyncDatabase.from_backend(kind, config=config)

    def run(query: Query) -> Answer:
        try:
            return _ids(db.search(query))
        except Exception as exc:  # a raise is an answer, and never the oracle's
            return _raised(exc)

    try:
        for row_id, data in ROWS:
            db.create(Record(data, storage_id=row_id))
        ordered = [(s, run(Query(sort_specs=[s]))) for s in SORTS]
    finally:
        if kind == "s3":
            db.clear()
        db.close()
    assert _disagreements(kind, "sync", ordered) == {}


async def test_a_record_with_no_value_sorts_last_async(
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
        for row_id, data in ROWS:
            await db.create(Record(data, storage_id=row_id))
        ordered = [(s, await run(Query(sort_specs=[s]))) for s in SORTS]
    finally:
        if kind == "s3":
            await db.clear()
        await db.close()
    assert _disagreements(kind, "async", ordered) == {}
