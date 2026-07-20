"""Record identifiers as first-class query targets: literal-prefix predicate.

``Operator.STARTS_WITH`` is a **literal, case-sensitive** prefix match — unlike
``LIKE``, a ``_`` or ``%`` in the prefix is matched verbatim, not as a wildcard.
``Filter("id", ...)`` resolves to the record's *storage key* on every backend, so
a store that encodes hierarchy into its keys (``artifacts/{owner}/{path}/{name}``)
can query *through* the key — a subtree scan is one ``Filter(...)`` instead of a
coarse fetch-everything-then-filter-in-Python walk.

Reproduce-first framing:

- ``test_like_underscore_is_a_false_positive`` pins the wildcard hazard
  ``STARTS_WITH`` exists to close: ``LIKE "a_b/%"`` also matches ``axb/1`` because
  ``_`` is a wildcard. ``STARTS_WITH "a_b/"`` matches only the literal-prefix key.
- ``test_starts_with_is_case_sensitive`` pins the parity constraint behind the
  per-dialect SQL translation (SQLite ``LIKE`` is case-*insensitive* for ASCII, so
  it cannot back this operator — the SQLite path uses a case-sensitive range scan).

The in-process backend families (memory, file, SQLite, DuckDB) are covered here,
sync and async. Service-backed backends (Postgres, S3, Elasticsearch) — including
the two async backends whose ``id`` filtering this change realigns — are covered
under ``tests/integration/`` behind their service markers.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from dataknobs_data import Filter, Operator, Query, Record, SortOrder, SortSpec
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sql_base import (
    SQLQueryBuilder,
    escape_like_prefix,
    prefix_upper_bound,
)
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.query_logic import ComplexQuery, FilterCondition, LogicCondition, LogicOperator

_IN_PROCESS = ["memory", "file", "sqlite", "duckdb"]
# ComplexQuery (boolean AND/OR) is supported by the memory and SQL backends;
# the file backend's search() only understands a flat ``Query`` (pre-existing).
_COMPLEX_QUERY_BACKENDS = ["memory", "sqlite", "duckdb"]


def _build_sync_db(kind: str, d: str) -> object:
    if kind == "memory":
        return SyncMemoryDatabase()
    if kind == "file":
        return SyncFileDatabase({"path": str(Path(d) / "records.json")})
    if kind == "sqlite":
        db = SyncSQLiteDatabase({"path": str(Path(d) / "records.db")})
        db.connect()
        return db
    db = SyncDuckDBDatabase({"path": str(Path(d) / "records.duckdb"), "table": "records"})
    db.connect()
    return db


# ---------------------------------------------------------------------------
# Fixtures — one connected instance per in-process backend family
# ---------------------------------------------------------------------------
@pytest.fixture(params=_IN_PROCESS)
def sync_db(request: pytest.FixtureRequest) -> Iterator[object]:
    with tempfile.TemporaryDirectory() as d:
        db = _build_sync_db(request.param, d)
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()


@pytest.fixture(params=_COMPLEX_QUERY_BACKENDS)
def sync_cq_db(request: pytest.FixtureRequest) -> Iterator[object]:
    """Backends whose search() supports a boolean ComplexQuery."""
    with tempfile.TemporaryDirectory() as d:
        db = _build_sync_db(request.param, d)
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                close()


@pytest.fixture(params=_IN_PROCESS)
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[object]:
    kind = request.param
    with tempfile.TemporaryDirectory() as d:
        db: object
        if kind == "memory":
            db = AsyncMemoryDatabase()
        elif kind == "file":
            db = AsyncFileDatabase({"path": str(Path(d) / "records.json")})
        elif kind == "sqlite":
            db = AsyncSQLiteDatabase({"path": str(Path(d) / "records.db")})
            await db.connect()
        else:
            db = AsyncDuckDBDatabase({"path": str(Path(d) / "records.duckdb"), "table": "records"})
            await db.connect()
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                await close()


@pytest.fixture(params=_COMPLEX_QUERY_BACKENDS)
async def async_cq_db(request: pytest.FixtureRequest) -> AsyncIterator[object]:
    """Async backends whose search() supports a boolean ComplexQuery."""
    kind = request.param
    with tempfile.TemporaryDirectory() as d:
        db: object
        if kind == "memory":
            db = AsyncMemoryDatabase()
        elif kind == "sqlite":
            db = AsyncSQLiteDatabase({"path": str(Path(d) / "records.db")})
            await db.connect()
        else:
            db = AsyncDuckDBDatabase({"path": str(Path(d) / "records.duckdb"), "table": "records"})
            await db.connect()
        try:
            yield db
        finally:
            close = getattr(db, "close", None)
            if callable(close):
                await close()


def _seed_keys(db: object, keys: list[str]) -> None:
    """Create one record per storage key (the key IS the record id)."""
    for k in keys:
        db.create(Record({"payload": k}, id=k))


async def _aseed_keys(db: object, keys: list[str]) -> None:
    for k in keys:
        await db.create(Record({"payload": k}, id=k))


def _ids(records: list[Record]) -> set[str]:
    return {r.id for r in records}


# ---------------------------------------------------------------------------
# 1. LIKE false positive vs literal STARTS_WITH (the motivating hazard)
# ---------------------------------------------------------------------------
def test_like_underscore_is_a_false_positive(sync_db: object) -> None:
    """LIKE treats ``_`` as a wildcard; STARTS_WITH matches it literally.

    Reproduce-first: ``LIKE "a_b/%"`` matches BOTH ``a_b/1`` and ``axb/1`` — the
    ``_`` is a single-char wildcard. That false positive is exactly why a literal
    prefix predicate is needed.
    """
    _seed_keys(sync_db, ["a_b/1", "axb/1", "a_b/2"])

    like_hits = _ids(sync_db.search(Query(filters=[Filter("id", Operator.LIKE, "a_b/%")])))
    assert like_hits == {"a_b/1", "axb/1", "a_b/2"}  # axb/1 is the wildcard false positive

    prefix_hits = _ids(
        sync_db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "a_b/")]))
    )
    assert prefix_hits == {"a_b/1", "a_b/2"}  # literal: axb/1 excluded


def test_starts_with_literal_percent(sync_db: object) -> None:
    """A ``%`` in the prefix is matched literally, not as a wildcard."""
    _seed_keys(sync_db, ["100%/done", "100x/done", "100%/pending"])
    hits = _ids(
        sync_db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "100%/")]))
    )
    assert hits == {"100%/done", "100%/pending"}


# ---------------------------------------------------------------------------
# 2. Cross-backend id parity (STARTS_WITH pushes down / scans identically)
# ---------------------------------------------------------------------------
def test_id_prefix_subtree_scan(sync_db: object) -> None:
    """A key-prefix scan returns exactly the subtree under that prefix."""
    _seed_keys(
        sync_db,
        [
            "artifacts/alice/report/final",
            "artifacts/alice/report/draft",
            "artifacts/alice/notes/1",
            "artifacts/bob/report/final",
        ],
    )
    hits = _ids(
        sync_db.search(
            Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/alice/report/")])
        )
    )
    assert hits == {"artifacts/alice/report/final", "artifacts/alice/report/draft"}


def test_id_operators_resolve_to_storage_key(sync_db: object) -> None:
    """EQ / IN / NEQ / STARTS_WITH on ``id`` all resolve to the storage key."""
    _seed_keys(sync_db, ["orders/1", "orders/2", "orders/3", "carts/1"])

    assert _ids(sync_db.search(Query(filters=[Filter("id", Operator.EQ, "orders/2")]))) == {
        "orders/2"
    }
    assert _ids(
        sync_db.search(Query(filters=[Filter("id", Operator.IN, ["orders/1", "carts/1"])]))
    ) == {"orders/1", "carts/1"}
    assert _ids(
        sync_db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "orders/")]))
    ) == {"orders/1", "orders/2", "orders/3"}


@pytest.mark.asyncio
async def test_async_id_prefix_subtree_scan(async_db: object) -> None:
    """Async in-process backends push down / scan the id prefix identically."""
    await _aseed_keys(
        async_db,
        ["artifacts/alice/a", "artifacts/alice/b", "artifacts/bob/a"],
    )
    hits = _ids(
        await async_db.search(
            Query(filters=[Filter("id", Operator.STARTS_WITH, "artifacts/alice/")])
        )
    )
    assert hits == {"artifacts/alice/a", "artifacts/alice/b"}


@pytest.mark.asyncio
async def test_async_like_underscore_is_a_false_positive(async_db: object) -> None:
    """The literal-prefix guarantee holds on the async backends too."""
    await _aseed_keys(async_db, ["a_b/1", "axb/1"])
    prefix_hits = _ids(
        await async_db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "a_b/")]))
    )
    assert prefix_hits == {"a_b/1"}


# ---------------------------------------------------------------------------
# 3. Case-sensitivity contract (guards the SQLite-LIKE hazard)
# ---------------------------------------------------------------------------
def test_starts_with_is_case_sensitive(sync_db: object) -> None:
    """STARTS_WITH is case-sensitive on every backend (matches str.startswith).

    SQLite ``LIKE`` is case-insensitive for ASCII, so the SQLite path uses a
    case-sensitive range scan instead; this test pins that all backends agree.
    """
    _seed_keys(sync_db, ["ABC/1", "abc/1", "ABx/1"])
    upper = _ids(sync_db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "AB")])))
    assert upper == {"ABC/1", "ABx/1"}  # NOT abc/1
    lower = _ids(sync_db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "abc")])))
    assert lower == {"abc/1"}


def test_starts_with_empty_prefix_matches_all(sync_db: object) -> None:
    """An empty prefix matches every record (never an invalid/unbounded clause)."""
    _seed_keys(sync_db, ["x/1", "y/2", "z/3"])
    hits = _ids(sync_db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, "")])))
    assert hits == {"x/1", "y/2", "z/3"}


def test_starts_with_on_data_field(sync_db: object) -> None:
    """STARTS_WITH also works on an ordinary (non-id) string data field."""
    sync_db.create(Record({"path": "docs/intro"}, id="1"))
    sync_db.create(Record({"path": "docs/advanced"}, id="2"))
    sync_db.create(Record({"path": "images/logo"}, id="3"))
    hits = _ids(
        sync_db.search(Query(filters=[Filter("path", Operator.STARTS_WITH, "docs/")]))
    )
    assert hits == {"1", "2"}


# ---------------------------------------------------------------------------
# 4. Secondary-key recipe — promote a non-key id to an indexed field
# ---------------------------------------------------------------------------
def test_secondary_key_lookup_via_field(sync_db: object) -> None:
    """Looking up by an id that is not the storage key = filter the data field."""
    sync_db.create(Record({"sku": "SKU-100", "name": "widget"}, id="row-1"))
    sync_db.create(Record({"sku": "SKU-200", "name": "gadget"}, id="row-2"))
    hits = sync_db.search(Query(filters=[Filter("sku", Operator.EQ, "SKU-200")]))
    assert _ids(hits) == {"row-2"}
    assert hits[0].get_value("name") == "gadget"


def test_data_id_field_is_shadowed_by_filter_id(sync_db: object) -> None:
    """Reserved-name footgun: a ``data`` field named ``id`` is shadowed.

    ``Filter("id", ...)`` resolves to the record's *storage key* on every
    backend, so a value stored under ``data["id"]`` is unreachable by query — the
    filter matches the storage key and silently returns no rows. This pins the
    hazard the API reference now documents at the secondary-identifier recipe, so
    a future refactor cannot quietly change the semantics without a test failing.
    """
    # Storage key is "row-1"/"row-2" (the ``id=`` arg); the ``data["id"]`` value
    # is an ordinary field that happens to collide with the reserved name.
    sync_db.create(Record({"id": "node-abc", "name": "widget"}, id="row-1"))
    sync_db.create(Record({"id": "node-xyz", "name": "gadget"}, id="row-2"))

    # Negative direction: filtering on the data value matches nothing — the
    # filter compared against the storage key, never ``data["id"]``.
    assert sync_db.search(Query(filters=[Filter("id", Operator.EQ, "node-abc")])) == []

    # Proof the filter resolved to the storage key: filtering on the key hits.
    assert _ids(sync_db.search(Query(filters=[Filter("id", Operator.EQ, "row-1")]))) == {
        "row-1"
    }


@pytest.mark.asyncio
async def test_async_data_id_field_is_shadowed_by_filter_id(async_db: object) -> None:
    """Async twin of the shadowing pin — the async in-process backends (where the
    181 audit found the sync/async drift) agree that ``data["id"]`` is shadowed."""
    await async_db.create(Record({"id": "node-abc", "name": "widget"}, id="row-1"))
    await async_db.create(Record({"id": "node-xyz", "name": "gadget"}, id="row-2"))

    assert (
        await async_db.search(Query(filters=[Filter("id", Operator.EQ, "node-abc")]))
    ) == []
    assert _ids(
        await async_db.search(Query(filters=[Filter("id", Operator.EQ, "row-1")]))
    ) == {"row-1"}


# ---------------------------------------------------------------------------
# 4b. Sort on the reserved id field resolves to the storage key (parity + shadow)
# ---------------------------------------------------------------------------
def test_sort_by_id_orders_by_storage_key(sync_db: object) -> None:
    """``SortSpec("id", ...)`` orders by the storage key on every backend.

    The sort translation consults the same reserved-name policy as filtering, so
    ordering by ``id`` follows the storage key uniformly — the parity that guards
    against a sort site drifting away from the filter sites.
    """
    _seed_keys(sync_db, ["c", "a", "b"])
    asc = [r.id for r in sync_db.search(Query(sort_specs=[SortSpec("id", SortOrder.ASC)]))]
    assert asc == ["a", "b", "c"]
    desc = [r.id for r in sync_db.search(Query(sort_specs=[SortSpec("id", SortOrder.DESC)]))]
    assert desc == ["c", "b", "a"]


def test_data_id_field_is_not_orderable_by_sort_id(sync_db: object) -> None:
    """Negative direction for sort: a ``data`` field named ``id`` is shadowed.

    ``SortSpec("id", ...)`` orders by the storage key, never by a value stored
    under ``data["id"]`` — pinning the shadowing hazard on the ordering path too.
    """
    # Storage keys sort ascending as row-a, row-b; the ``data["id"]`` values sort
    # the opposite way, so an order driven by ``data["id"]`` would be reversed.
    sync_db.create(Record({"id": "zzz", "name": "widget"}, id="row-a"))
    sync_db.create(Record({"id": "aaa", "name": "gadget"}, id="row-b"))

    asc = [r.id for r in sync_db.search(Query(sort_specs=[SortSpec("id", SortOrder.ASC)]))]
    assert asc == ["row-a", "row-b"]  # storage-key order, NOT data["id"] order


@pytest.mark.asyncio
async def test_async_sort_by_id_orders_by_storage_key(async_db: object) -> None:
    """Async twin of the sort parity pin — the async in-process backends order by
    the storage key when sorting on the reserved ``id`` field."""
    await _aseed_keys(async_db, ["c", "a", "b"])
    asc = [
        r.id
        for r in await async_db.search(Query(sort_specs=[SortSpec("id", SortOrder.ASC)]))
    ]
    assert asc == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 5. Pushdown proof — the SQL builder emits a pushed-down predicate, not a scan
# ---------------------------------------------------------------------------
def test_sql_pushdown_postgres_duckdb_like_escape() -> None:
    """postgres/duckdb compile STARTS_WITH to LIKE ... ESCAPE with an escaped prefix."""
    for dialect in ("postgres", "duckdb"):
        qb = SQLQueryBuilder("records", dialect=dialect, param_style="numeric")
        sql, params = qb.build_search_query(
            Query(filters=[Filter("id", Operator.STARTS_WITH, "a_b/")])
        )
        assert "id LIKE" in sql
        assert "ESCAPE" in sql
        # The '_' in the prefix is escaped so it matches literally, plus trailing %.
        assert params == ["a\\_b/%"]


def test_sql_pushdown_sqlite_range() -> None:
    """SQLite compiles STARTS_WITH to a case-sensitive half-open range on the id column."""
    qb = SQLQueryBuilder("records", dialect="sqlite", param_style="qmark")
    sql, params = qb.build_search_query(
        Query(filters=[Filter("id", Operator.STARTS_WITH, "orders/")])
    )
    assert "id >=" in sql and "id <" in sql
    assert "LIKE" not in sql  # NOT LIKE — SQLite LIKE is case-insensitive
    assert params == ["orders/", prefix_upper_bound("orders/")]


def test_sql_pushdown_sqlite_empty_prefix_is_always_true() -> None:
    """An empty prefix on sqlite emits an always-true clause with no bound params."""
    qb = SQLQueryBuilder("records", dialect="sqlite", param_style="qmark")
    sql, params = qb.build_search_query(
        Query(filters=[Filter("id", Operator.STARTS_WITH, "")])
    )
    assert "IS NOT NULL" in sql
    assert params == []


def test_sql_pushdown_sqlite_all_maximal_prefix_keeps_lower_bound() -> None:
    """Reproduce-first: a non-empty all-maximal prefix stays lower-bounded.

    ``prefix_upper_bound(chr(0x10FFFF))`` is ``None`` (no exclusive upper bound
    exists), but the prefix is non-empty, so the SQLite clause must remain
    ``id >= <prefix>`` — one bound param — not the empty-prefix ``IS NOT NULL``
    that would return every row.
    """
    qb = SQLQueryBuilder("records", dialect="sqlite", param_style="qmark")
    sql, params = qb.build_search_query(
        Query(filters=[Filter("id", Operator.STARTS_WITH, chr(0x10FFFF))])
    )
    assert "id >=" in sql
    assert "IS NOT NULL" not in sql
    assert params == [chr(0x10FFFF)]


def test_starts_with_all_maximal_prefix_is_lower_bounded(sync_db: object) -> None:
    """Reproduce-first: a non-empty all-``U+10FFFF`` prefix must not degrade to
    match-all — it stays lower-bounded, matching only keys at/after it.

    The SQLite range path previously emitted a bare ``IS NOT NULL`` for both the
    empty prefix and a non-empty all-maximal prefix, so ``STARTS_WITH
    chr(0x10FFFF)`` returned every row. Verified across every in-process backend
    for cross-backend parity with ``str.startswith``.
    """
    maximal = chr(0x10FFFF)
    sync_db.create(Record({"n": 1}, id=maximal + "z"))
    sync_db.create(Record({"n": 2}, id=maximal))
    sync_db.create(Record({"n": 3}, id="normal"))
    got = _ids(sync_db.search(Query(filters=[Filter("id", Operator.STARTS_WITH, maximal)])))
    assert got == {maximal + "z", maximal}


def test_string_only_operators_guard_non_string_json_fields() -> None:
    """The SQL push-down carries a JSON-string-type guard for string-only
    operators on a JSON data field, so a non-string value cannot match via text
    projection — parity with the in-memory ``isinstance(str)`` contract."""
    guards = {
        "postgres": "jsonb_typeof(data->'code') = 'string'",
        "sqlite": "json_type(data, '$.code') = 'text'",
        "duckdb": "json_type(data, '$.code') = 'VARCHAR'",
    }
    styles = {"postgres": "numeric", "sqlite": "qmark", "duckdb": "qmark"}
    for dialect, guard in guards.items():
        qb = SQLQueryBuilder("records", dialect=dialect, param_style=styles[dialect])
        for op in (Operator.LIKE, Operator.NOT_LIKE, Operator.REGEX, Operator.STARTS_WITH):
            sql, _ = qb.build_search_query(Query(filters=[Filter("code", op, "5")]))
            assert guard in sql, (dialect, op, sql)


def test_id_string_only_operators_are_unguarded() -> None:
    """The 'id' real column is always a string — no JSON type guard is added."""
    for dialect, style in (("postgres", "numeric"), ("sqlite", "qmark"), ("duckdb", "qmark")):
        qb = SQLQueryBuilder("records", dialect=dialect, param_style=style)
        sql, _ = qb.build_search_query(
            Query(filters=[Filter("id", Operator.STARTS_WITH, "x")])
        )
        assert "jsonb_typeof" not in sql
        assert "json_type" not in sql


def test_string_only_operators_skip_non_string_values(sync_db: object) -> None:
    """Reproduce-first: STARTS_WITH / LIKE on a JSON data field match only string
    values — a numeric value whose text form shares the prefix does not match.

    The SQL text projection (``data->>'f'`` / ``json_extract``) coerced a numeric
    value to text, so SQLite/DuckDB matched ``500`` for ``STARTS_WITH "500"`` while
    the in-memory matcher (``isinstance(str)``) did not. The JSON-string-type guard
    aligns every in-process backend.
    """
    sync_db.create(Record({"code": 500}, id="numeric"))
    sync_db.create(Record({"code": "500-alpha"}, id="string"))
    assert _ids(
        sync_db.search(Query(filters=[Filter("code", Operator.STARTS_WITH, "500")]))
    ) == {"string"}
    assert _ids(
        sync_db.search(Query(filters=[Filter("code", Operator.LIKE, "500%")]))
    ) == {"string"}
    # NOT_LIKE polarity: the numeric value must NOT resurface just because its
    # text form ('500') is unlike 'zzz%'. A non-string value never matches a
    # string-only operator in *either* direction (in-memory returns False for
    # NOT_LIKE too), so only the genuinely-unlike string is returned.
    assert _ids(
        sync_db.search(Query(filters=[Filter("code", Operator.NOT_LIKE, "zzz%")]))
    ) == {"string"}


# ---------------------------------------------------------------------------
# 6. Byte-identity regression — existing operators unchanged; async S3 gaps closed
# ---------------------------------------------------------------------------
def test_existing_operators_unchanged(sync_db: object) -> None:
    """EQ / IN / BETWEEN / LIKE keep working exactly as before."""
    for i in range(1, 6):
        sync_db.create(Record({"n": i, "tag": f"t{i}"}, id=f"r{i}"))
    assert _ids(sync_db.search(Query(filters=[Filter("n", Operator.EQ, 3)]))) == {"r3"}
    assert _ids(sync_db.search(Query(filters=[Filter("n", Operator.BETWEEN, [2, 4])]))) == {
        "r2",
        "r3",
        "r4",
    }
    assert _ids(sync_db.search(Query(filters=[Filter("tag", Operator.IN, ["t1", "t5"])]))) == {
        "r1",
        "r5",
    }


# ---------------------------------------------------------------------------
# 7. Escape / bound helper units — the edge-case surface of the SQL translation
# ---------------------------------------------------------------------------
def test_escape_like_prefix() -> None:
    assert escape_like_prefix("plain") == "plain"
    assert escape_like_prefix("a_b") == "a\\_b"
    assert escape_like_prefix("50%") == "50\\%"
    assert escape_like_prefix("a\\b") == "a\\\\b"
    # Order matters: the backslash is escaped first so a literal '%' does not
    # get double-escaped.
    assert escape_like_prefix("x%_\\y") == "x\\%\\_\\\\y"


def test_prefix_upper_bound() -> None:
    assert prefix_upper_bound("") is None
    assert prefix_upper_bound("abc") == "abd"
    assert prefix_upper_bound("ab/") == "ab0"  # '/' (0x2f) -> '0' (0x30)
    # A trailing max code point drops and the preceding point increments.
    assert prefix_upper_bound("a" + chr(0x10FFFF)) == "b"
    # All-maximal prefix has no finite bound.
    assert prefix_upper_bound(chr(0x10FFFF)) is None


def test_prefix_upper_bound_is_a_correct_boundary() -> None:
    """Every string with the prefix is < the bound; the next sibling is not."""
    prefix = "orders/"
    bound = prefix_upper_bound(prefix)
    assert bound is not None
    assert prefix < bound
    assert (prefix + "\U0010ffff") < bound  # any suffix stays under the bound
    assert bound <= "orders0"  # 'orders0' sorts at/after the bound → excluded


# ---------------------------------------------------------------------------
# 8. ComplexQuery coverage — STARTS_WITH inside AND/OR
# ---------------------------------------------------------------------------
def test_starts_with_in_complex_query(sync_cq_db: object) -> None:
    """STARTS_WITH composes inside a boolean ComplexQuery (in-memory + SQL push)."""
    sync_cq_db.create(Record({"path": "docs/a", "n": 1}, id="1"))
    sync_cq_db.create(Record({"path": "docs/b", "n": 9}, id="2"))
    sync_cq_db.create(Record({"path": "img/c", "n": 1}, id="3"))
    cq = ComplexQuery(
        condition=LogicCondition(
            operator=LogicOperator.AND,
            conditions=[
                FilterCondition(Filter("path", Operator.STARTS_WITH, "docs/")),
                FilterCondition(Filter("n", Operator.EQ, 1)),
            ],
        )
    )
    assert _ids(sync_cq_db.search(cq)) == {"1"}


# ---------------------------------------------------------------------------
# 8b. Reserved ``id`` inside a boolean ComplexQuery resolves to the storage key
#
# A top-level OR/NOT ComplexQuery cannot collapse to a simple Query, so the
# memory backend falls into the shared in-memory scan path
# (``AsyncDatabase._search_with_complex_query`` + ``FilterCondition.matches``),
# while the SQL backends translate natively. These tests pin that BOTH routes
# resolve the reserved ``id`` field to the storage key — the parity the flat-Query
# path already guarantees. Reproduce-first: before the scan path consulted the
# reserved-name policy, the memory parametrization failed (it resolved ``id`` to
# the shadowed ``data["id"]``) while the SQL parametrizations passed.
# ---------------------------------------------------------------------------
def _or_with_id(id_value: str) -> ComplexQuery:
    """An OR that can't simplify: ``id == id_value`` OR an always-miss clause."""
    return ComplexQuery(
        condition=LogicCondition(
            operator=LogicOperator.OR,
            conditions=[
                FilterCondition(Filter("id", Operator.EQ, id_value)),
                FilterCondition(Filter("name", Operator.EQ, "__never__")),
            ],
        )
    )


def test_id_filter_in_complex_query_resolves_to_storage_key(sync_cq_db: object) -> None:
    """``Filter("id", ...)`` inside a boolean ComplexQuery targets the storage key.

    The scan-path memory backend and the native-SQL backends must agree; before
    the fix the memory parametrization resolved ``id`` to ``data["id"]`` and
    returned no rows for the storage-key value.
    """
    sync_cq_db.create(Record({"id": "node-abc", "name": "widget"}, id="row-1"))
    sync_cq_db.create(Record({"id": "node-xyz", "name": "gadget"}, id="row-2"))

    # Positive: the storage key resolves and matches.
    assert _ids(sync_cq_db.search(_or_with_id("row-1"))) == {"row-1"}
    # Negative: the shadowed ``data["id"]`` value matches nothing.
    assert sync_cq_db.search(_or_with_id("node-abc")) == []


def test_sort_by_id_in_complex_query_orders_by_storage_key(sync_cq_db: object) -> None:
    """``SortSpec("id", ...)`` on a ComplexQuery orders by the storage key.

    The condition matches via a non-id field so the ordering is isolated from the
    filter path; before the fix the scan-path sort keyed on ``data["id"]`` and
    reversed the result on the memory backend.
    """
    # Storage keys sort ascending row-a, row-b; the ``data["id"]`` values sort the
    # opposite way, so an order driven by ``data["id"]`` would be reversed.
    sync_cq_db.create(Record({"grp": "g", "id": "zzz"}, id="row-a"))
    sync_cq_db.create(Record({"grp": "g", "id": "aaa"}, id="row-b"))
    cq = ComplexQuery(
        condition=LogicCondition(
            operator=LogicOperator.OR,
            conditions=[
                FilterCondition(Filter("grp", Operator.EQ, "g")),
                FilterCondition(Filter("name", Operator.EQ, "__never__")),
            ],
        ),
        sort_specs=[SortSpec("id", SortOrder.ASC)],
    )
    assert [r.id for r in sync_cq_db.search(cq)] == ["row-a", "row-b"]


@pytest.mark.asyncio
async def test_async_id_filter_in_complex_query_resolves_to_storage_key(
    async_cq_db: object,
) -> None:
    """Async twin: the reserved ``id`` inside a ComplexQuery targets the storage key.

    Covers the async scan path (``AsyncDatabase._search_with_complex_query``) that
    the async memory backend falls into for a boolean query.
    """
    await async_cq_db.create(Record({"id": "node-abc", "name": "widget"}, id="row-1"))
    await async_cq_db.create(Record({"id": "node-xyz", "name": "gadget"}, id="row-2"))

    assert _ids(await async_cq_db.search(_or_with_id("row-1"))) == {"row-1"}
    assert (await async_cq_db.search(_or_with_id("node-abc"))) == []


@pytest.mark.asyncio
async def test_async_sort_by_id_in_complex_query_orders_by_storage_key(
    async_cq_db: object,
) -> None:
    """Async twin of the ComplexQuery sort-parity pin."""
    await async_cq_db.create(Record({"grp": "g", "id": "zzz"}, id="row-a"))
    await async_cq_db.create(Record({"grp": "g", "id": "aaa"}, id="row-b"))
    cq = ComplexQuery(
        condition=LogicCondition(
            operator=LogicOperator.OR,
            conditions=[
                FilterCondition(Filter("grp", Operator.EQ, "g")),
                FilterCondition(Filter("name", Operator.EQ, "__never__")),
            ],
        ),
        sort_specs=[SortSpec("id", SortOrder.ASC)],
    )
    assert [r.id for r in await async_cq_db.search(cq)] == ["row-a", "row-b"]


# ---------------------------------------------------------------------------
# 9. Round-trip — the operator serializes and reconstructs
# ---------------------------------------------------------------------------
def test_operator_round_trip() -> None:
    f = Filter("id", Operator.STARTS_WITH, "artifacts/")
    d = f.to_dict()
    assert d["operator"] == "starts_with"
    back = Filter.from_dict(d)
    assert back.operator is Operator.STARTS_WITH
    assert back.field == "id"
    assert back.value == "artifacts/"


def test_query_filter_string_operator_map() -> None:
    """The fluent string form resolves to the STARTS_WITH operator."""
    q = Query().filter("id", "starts_with", "x/")
    assert q.filters[0].operator is Operator.STARTS_WITH


def test_matches_non_string_is_false() -> None:
    """STARTS_WITH against a non-string value is a no-match (mirrors LIKE)."""
    assert Filter("id", Operator.STARTS_WITH, "1").matches(123) is False
    assert Filter("id", Operator.STARTS_WITH, "x").matches(None) is False
