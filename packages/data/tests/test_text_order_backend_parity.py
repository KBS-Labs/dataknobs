"""Every backend orders text by code point, as ``Filter.matches`` does.

``Filter.matches`` compares Python strings, and the in-memory sort orders
them the same way, so ``'Banana' < 'apple'`` and ``'Zebra' < 'éclair'``.
SQLite (``BINARY``) and DuckDB agree. PostgreSQL compared and sorted text
under the database's collation, so on a database created as ``en_US.utf8``
``name > 'B'`` left out ``'apple'`` and an ascending sort put ``'apple'``
first, with no error. The same held for the ``id`` column.

Each backend is asked the same comparisons and sorts, on a data field and on
the storage key, and its answers are held to the oracle's. PostgreSQL is also
asked twice more: on a table created before ``id`` was declared
``COLLATE "C"``, which must answer the same, and whether the primary-key index
still serves a range and sort on ``id`` for a table created now.
"""

from __future__ import annotations

import re
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
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.query import Filter, Operator, SortOrder, SortSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

#: (id, t). Each column holds pairs whose collation order and code-point order
#: disagree: case, a space or underscore a collation skips, and accents.
ROWS = [
    ("Apple", "apple"),
    ("banana", "Banana"),
    ("Cherry", "cherry"),
    ("zebra", "Zebra"),
    ("ab", "ab"),
    ("a_b", "a b"),
    ("Zed", "_x"),
    ("dune", "éclair"),
    ("Echo", "Émile"),
]
IDS = [row_id for row_id, _ in ROWS]
VALUES = [value for _, value in ROWS]
T_OF = dict(ROWS)

THRESHOLDS = ["B", "a", "b", "Z", "_", "é", "É", "a b", "ab"]
RANGES = [("B", "b"), ("a", "é"), ("Z", "a"), ("_", "c")]

FILTERS: list[Filter] = [
    Filter(field, op, threshold)
    for field in ("t", "id")
    for op in (Operator.GT, Operator.GTE, Operator.LT, Operator.LTE)
    for threshold in THRESHOLDS
] + [
    Filter(field, op, list(bounds))
    for field in ("t", "id")
    for op in (Operator.BETWEEN, Operator.NOT_BETWEEN)
    for bounds in RANGES
]

SORTS = [SortSpec(field, order) for field in ("t", "id") for order in SortOrder]

#: A search's answer: the matching ids, or what it raised.
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
            (kind, c) for c in request.getfixturevalue("make_postgres_test_db")("test_order_")
        )
    elif kind == "s3":
        for config in request.getfixturevalue("make_localstack_s3_bucket")("dataknobs-order"):
            yield kind, {**config, "prefix": f"order-{uuid.uuid4().hex[:10]}/"}
    elif kind == "elasticsearch":
        yield from (
            (kind, c)
            for c in request.getfixturevalue("make_elasticsearch_test_index")("test_order_")
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


def _expected_filter(spec: Filter) -> list[str]:
    return sorted(
        row_id for row_id, value in ROWS if spec.matches(row_id if spec.field == "id" else value)
    )


def _expected_sort(spec: SortSpec) -> list[str]:
    reverse = spec.order == SortOrder.DESC
    if spec.field == "id":
        return sorted(IDS, reverse=reverse)
    return sorted(IDS, key=T_OF.__getitem__, reverse=reverse)


def _analyzed_range(spec: Filter) -> list[str]:
    """What a range answers when run against a text field's analyzed terms.

    A record matches when any of its lowercased terms is in range; the bound
    itself is compared as given.
    """

    def in_range(term: str) -> bool:
        if spec.operator in (Operator.BETWEEN, Operator.NOT_BETWEEN):
            lower, upper = spec.value
            return bool(lower <= term <= upper)
        return bool(Filter("t", spec.operator, spec.value).matches(term))

    hits = {
        row_id
        for row_id, value in ROWS
        if any(in_range(term) for term in re.findall(r"\w+", value.lower()))
    }
    if spec.operator == Operator.NOT_BETWEEN:
        hits = set(IDS) - hits
    return sorted(hits)


def _known_answer(kind: str, flavour: str, spec: Filter | SortSpec) -> Answer | None:
    """A backend's answer where a defect outside this test's subject decides it.

    The defect's answer replaces the oracle's, so the backend is still held to
    an exact answer: the test fails when a defect is fixed, and the entry
    here goes with it, as it does when a defect changes shape.
    """
    if kind != "elasticsearch" or spec.field != "t":
        return None
    if isinstance(spec, Filter):
        # A range with a string bound runs against the analyzed field.
        return _analyzed_range(spec)
    if flavour == "async":
        # The async backend sorts a text field without its keyword
        # sub-field, and Elasticsearch refuses the request.
        return "raises BadRequestError"
    return None


def _disagreements(
    kind: str,
    flavour: str,
    filtered: list[tuple[Filter, Answer]],
    ordered: list[tuple[SortSpec, Answer]],
) -> dict[str, object]:
    wrong: dict[str, object] = {}
    for spec, got in filtered:
        known = _known_answer(kind, flavour, spec)
        want = _expected_filter(spec) if known is None else known
        if got != want:
            wrong[f"{spec.field} {spec.operator.value} {spec.value!r}"] = {"got": got, "want": want}
    for sort, got in ordered:
        known = _known_answer(kind, flavour, sort)
        want = _expected_sort(sort) if known is None else known
        if got != want:
            wrong[f"sort {sort.field} {sort.order.value}"] = {"got": got, "want": want}
    return wrong


def _ids(found: list[Record]) -> list[str]:
    return [str(r.storage_id) for r in found]


def _raised(exc: Exception) -> str:
    return f"raises {type(exc).__name__}"


def _search_all(db: SyncDatabase, kind: str) -> dict[str, object]:
    def run(query: Query) -> Answer:
        try:
            return _ids(db.search(query))
        except Exception as exc:  # a raise is an answer, and never the oracle's
            return _raised(exc)

    filtered = [(f, _sorted(run(Query(filters=[f])))) for f in FILTERS]
    ordered = [(s, run(Query(sort_specs=[s]))) for s in SORTS]
    return _disagreements(kind, "sync", filtered, ordered)


async def _asearch_all(db: AsyncDatabase, kind: str) -> dict[str, object]:
    async def run(query: Query) -> Answer:
        try:
            return _ids(await db.search(query))
        except Exception as exc:  # a raise is an answer, and never the oracle's
            return _raised(exc)

    filtered = [(f, _sorted(await run(Query(filters=[f])))) for f in FILTERS]
    ordered = [(s, await run(Query(sort_specs=[s]))) for s in SORTS]
    return _disagreements(kind, "async", filtered, ordered)


def _sorted(answer: Answer) -> Answer:
    """A filter's matches as a set; the order they came back in is not asked."""
    return answer if isinstance(answer, str) else sorted(answer)


def test_every_backend_orders_text_by_code_point_sync(
    backend: tuple[str, dict[str, Any]],
) -> None:
    """Comparison and sort agree with the oracle on every backend.

    PostgreSQL compared and sorted under the database collation.
    """
    kind, config = backend
    db = SyncDatabase.from_backend(kind, config=config)
    try:
        for row_id, value in ROWS:
            db.create(Record({"t": value}, storage_id=row_id))
        wrong = _search_all(db, kind)
    finally:
        if kind == "s3":
            db.clear()
        db.close()
    assert wrong == {}


async def test_every_backend_orders_text_by_code_point_async(
    backend: tuple[str, dict[str, Any]],
) -> None:
    kind, config = backend
    db = await AsyncDatabase.from_backend(kind, config=config)
    try:
        for row_id, value in ROWS:
            await db.create(Record({"t": value}, storage_id=row_id))
        wrong = await _asearch_all(db, kind)
    finally:
        if kind == "s3":
            await db.clear()
        await db.close()
    assert wrong == {}


@pytest.fixture
def postgres_config(make_postgres_test_db: Any) -> Iterator[dict[str, Any]]:
    yield from make_postgres_test_db("test_order_")


def _execute(config: dict[str, Any], *statements: str) -> list[str]:
    """Run statements in one session; the last one's rows, first column."""
    import psycopg2

    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        dbname=config["database"],
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)
            return [str(row[0]) for row in cursor.fetchall()] if cursor.description else []
    finally:
        conn.close()


def _id_range_and_sort_plan(config: dict[str, Any]) -> str:
    """The plan for a range and sort on ``id``, with the alternatives refused.

    On a table this small the planner prefers a scan and a sort whatever the
    index can do, so sequential scans, bitmap scans and sorts are all refused:
    what is left shows whether the index *can* serve the query, which is an
    index condition on ``id`` and no separate sort.
    """
    builder = SQLQueryBuilder(config["table"], config["schema"], dialect="postgres")
    sql, _ = builder.build_search_query(
        Query(filters=[Filter("id", Operator.GT, "b")], sort_specs=[SortSpec("id")])
    )
    return "\n".join(
        _execute(
            config,
            "SET enable_seqscan = off",
            "SET enable_bitmapscan = off",
            "SET enable_sort = off",
            f"PREPARE q(text) AS {sql}",
            "EXPLAIN EXECUTE q('b')",
        )
    )


def _served_by_the_index(plan: str) -> bool:
    return "Index Cond: (id > " in plan and "Sort" not in plan


@requires_postgres
def test_a_postgres_table_created_now_keeps_the_primary_key_index(
    postgres_config: dict[str, Any],
) -> None:
    """``id`` is declared ``COLLATE "C"``, so the code-point range uses the index."""
    db = SyncDatabase.from_backend("postgres", config=postgres_config)
    try:
        db.create(Record({"t": "x"}, storage_id="a"))
    finally:
        db.close()
    plan = _id_range_and_sort_plan(postgres_config)
    assert _served_by_the_index(plan), plan


@requires_postgres
def test_a_postgres_table_created_before_answers_by_code_point_too(
    postgres_config: dict[str, Any],
) -> None:
    """An ``id`` column under the database collation answers the same.

    The table is created first with the earlier declaration, which the backend
    then opens as it finds it. The answers are the contract; that the index no
    longer serves the range is the stated cost, and doubles as the control for
    the test above.
    """
    table = f'"{postgres_config["schema"]}"."{postgres_config["table"]}"'
    _execute(
        postgres_config,
        f"CREATE TABLE {table} (id TEXT PRIMARY KEY, data JSONB NOT NULL, metadata JSONB,"
        " created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
        " updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    )
    db = SyncDatabase.from_backend("postgres", config=postgres_config)
    try:
        for row_id, value in ROWS:
            db.create(Record({"t": value}, storage_id=row_id))
        wrong = _search_all(db, "postgres")
    finally:
        db.close()
    assert wrong == {}
    plan = _id_range_and_sort_plan(postgres_config)
    assert not _served_by_the_index(plan), plan
