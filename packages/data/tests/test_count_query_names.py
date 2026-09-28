"""A pushdown count is built from the query, not cut out of a search's SQL.

sqlite and DuckDB answer ``count(query)`` with one ``SELECT COUNT(*)``. The
statement used to be derived from the search SQL: ``SELECT *`` replaced, then
the text cut at the first ``ORDER BY``, ``LIMIT`` or ``OFFSET`` found anywhere
in it. Field paths and the table name are interpolated into that text, so a
name containing one of those words lost the rest of the statement. Measured
before the fix, through ``count()``, while ``search()`` with the same filter
answered correctly:

===============================  =====================================================
Case                             sqlite / DuckDB ``count()``
===============================  =====================================================
filter on field ``LIMIT_x``      ``unrecognized token`` / ``unterminated quoted string``
filter on ``metadata.OFFSET_k``  the same
any filter, table ``LIMITS``     ``unrecognized token``: the statement ends at ``FROM "``
===============================  =====================================================

The match was case-sensitive, so only an upper-case name triggered it. The
in-memory backend counts by matching records and is the oracle here.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_data import Filter, Operator, Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.query import Query

if TYPE_CHECKING:
    from collections.abc import Iterator

BACKENDS = ["sqlite", "duckdb"]

#: Two records carry the upper-case field and one does not, so a count of the
#: whole table, of nothing, and of the right subset are all distinguishable.
RECORDS: list[tuple[dict[str, Any], dict[str, Any]]] = [
    ({"LIMIT_x": 1, "y": 1}, {"OFFSET_k": "a"}),
    ({"LIMIT_x": 2, "y": 1}, {}),
    ({"y": 2}, {"OFFSET_k": "b"}),
]

#: ``(case id, table name, query)``. The last case uses an ordinary field and
#: an upper-case table name.
CASES: list[tuple[str, str, Query]] = [
    ("data-field", "records", Query(filters=[Filter("LIMIT_x", Operator.EQ, 1)])),
    (
        "metadata-field",
        "records",
        Query(filters=[Filter("metadata.OFFSET_k", Operator.EXISTS)]),
    ),
    ("table-name", "LIMITS", Query(filters=[Filter("y", Operator.EQ, 1)])),
]


def _oracle(query: Query) -> int:
    db = SyncMemoryDatabase()
    for data, metadata in RECORDS:
        db.create(Record(data=dict(data), metadata=dict(metadata)))
    return db.count(query)


@pytest.fixture
def root() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _sync_db(kind: str, root: Path, table: str) -> Any:
    if kind == "sqlite":
        return SyncSQLiteDatabase({"path": str(root / "records.db"), "table": table})
    return SyncDuckDBDatabase({"path": str(root / "records.duckdb"), "table": table})


def _async_db(kind: str, root: Path, table: str) -> Any:
    if kind == "sqlite":
        return AsyncSQLiteDatabase({"path": str(root / "records.db"), "table": table})
    return AsyncDuckDBDatabase({"path": str(root / "records.duckdb"), "table": table})


@pytest.mark.parametrize(("table", "query"), [pytest.param(t, q, id=c) for c, t, q in CASES])
@pytest.mark.parametrize("kind", BACKENDS)
class TestACountIsNotCutAtAKeywordInsideAName:
    """Each pushdown count equals the in-memory count for the same query."""

    def test_the_oracle_counts_a_strict_subset(self, kind: str, table: str, query: Query) -> None:
        assert 0 < _oracle(query) < len(RECORDS)

    def test_sync(self, kind: str, table: str, query: Query, root: Path) -> None:
        db = _sync_db(kind, root, table)
        db.connect()
        try:
            for data, metadata in RECORDS:
                db.create(Record(data=dict(data), metadata=dict(metadata)))
            assert db.count(query) == _oracle(query)
        finally:
            db.close()

    async def test_async(self, kind: str, table: str, query: Query, root: Path) -> None:
        db = _async_db(kind, root, table)
        await db.connect()
        try:
            for data, metadata in RECORDS:
                await db.create(Record(data=dict(data), metadata=dict(metadata)))
            assert await db.count(query) == _oracle(query)
        finally:
            await db.close()


async def test_the_async_oracle_agrees() -> None:
    """The async in-memory backend gives the same counts as the sync one."""
    db = AsyncMemoryDatabase()
    for data, metadata in RECORDS:
        await db.create(Record(data=dict(data), metadata=dict(metadata)))
    for _, _, query in CASES:
        assert await db.count(query) == _oracle(query)


#: Every dialect and placeholder style a backend constructs the builder with.
BUILDERS = [
    ("postgres", "numeric"),
    ("postgres", "pyformat"),
    ("sqlite", "qmark"),
    ("duckdb", "qmark"),
]


@pytest.mark.parametrize(("dialect", "param_style"), BUILDERS)
class TestTheCountSelectsWhatTheSearchSelects:
    """At builder level the count's ``WHERE`` is the search's, byte for byte."""

    def test_where_and_params_match(self, dialect: str, param_style: str) -> None:
        # A zero-bind clause (``IN []``) ahead of bound ones, so a numbering
        # slip in either statement shows up as a placeholder mismatch.
        query = Query(
            filters=[
                Filter("LIMIT_x", Operator.IN, []),
                Filter("y", Operator.EQ, 1),
                Filter("metadata.OFFSET_k", Operator.GT, 2),
            ],
            limit_value=5,
            offset_value=1,
        ).sort_by("y")
        builder = SQLQueryBuilder("LIMITS", dialect=dialect, param_style=param_style)
        table = builder.qualified_table

        count_sql, count_params = builder.build_count_query(query)
        search_sql, search_params = builder.build_search_query(query)
        count_head = f"SELECT COUNT(*) FROM {table} WHERE "
        assert count_sql.startswith(count_head)
        where = count_sql[len(count_head) :]

        assert search_sql.startswith(f"SELECT * FROM {table} WHERE {where} ORDER BY ")
        assert count_params == search_params
        assert builder.build_where_clause(query) == (f" AND {where}", count_params)

    def test_an_unfiltered_count_is_the_whole_table(self, dialect: str, param_style: str) -> None:
        builder = SQLQueryBuilder("LIMITS", dialect=dialect, param_style=param_style)
        expected = (f"SELECT COUNT(*) FROM {builder.qualified_table}", [])
        assert builder.build_count_query(Query(limit_value=1, offset_value=1)) == expected
        assert builder.build_count_query(None) == expected


@pytest.mark.parametrize(("dialect", "param_style"), BUILDERS)
def test_an_offset_without_a_limit_is_valid_sql(dialect: str, param_style: str) -> None:
    """SQLite takes ``OFFSET`` only after a ``LIMIT``; the others take it alone."""
    builder = SQLQueryBuilder("records", dialect=dialect, param_style=param_style)
    sql, _ = builder.build_search_query(Query(offset_value=2))
    paging = "LIMIT -1 OFFSET 2" if dialect == "sqlite" else "OFFSET 2"
    assert sql == f"SELECT * FROM {builder.qualified_table} {paging}"
