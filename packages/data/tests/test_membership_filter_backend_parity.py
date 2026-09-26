"""An ``IN`` / ``NOT IN`` filter means what ``Filter.matches`` means, on every backend.

The memory matcher is the contract: nothing is in an empty list, and a record
whose field has no value never matches (``Filter.matches`` returns ``False``
for a ``None`` value before it looks at the operator). The SQL builder used to
render the list verbatim, and three things followed from that. Measured before
the fix, over ``red``, ``green``, a ``None`` colour and a record with no colour:

==========================  ===========  ==================  =================
Filter                      memory       sqlite              duckdb / postgres
==========================  ===========  ==================  =================
``IN []``                   nothing      nothing             syntax error
``NOT IN []``               red, green   **all four**        syntax error
``NOT IN [None, "green"]``  red          **nothing**         **nothing**
==========================  ===========  ==================  =================

The last row is SQL's three-valued logic: ``x NOT IN (NULL, ...)`` is ``NULL``
for every ``x``. A ``None`` member can never match a record, so it is dropped,
and a list left empty takes the empty-list answer. An empty ``NOT IN`` is
``IS NOT NULL`` rather than ``TRUE``, because ``TRUE`` is sqlite's wrong
answer above.

Every case asserts the backend's answer against ``Filter.matches`` over the
same corpus **and** against a hard-coded expectation, so a drift in the oracle
fails as well. sqlite's empty ``NOT IN`` is the case that proves the module
reads results: it never raised.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import Filter, Operator, Record
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.query import RESERVED_KEY_FIELD, Query
from dataknobs_data.query_logic import ComplexQuery

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

BACKENDS = [
    "memory",
    "file",
    "sqlite",
    "duckdb",
    pytest.param("postgres", marks=requires_postgres),
]

#: Each record is keyed by what distinguishes it. The ``n`` records carry no
#: colour and the colour records carry no ``n``, so every filter below also
#: sees records that lack its field entirely.
CORPUS: dict[str, dict[str, Any]] = {
    "red": {"colour": "red"},
    "green": {"colour": "green"},
    "colour-null": {"colour": None},
    "no-colour": {"other": 1},
    "n5": {"n": 5},
    "n7": {"n": 7},
    "n-null": {"n": None},
}

#: ``(case id, filter, expected record ids)``. The expectation is written out
#: rather than derived, so it cannot move together with the oracle.
CASES: list[tuple[str, Filter, set[str]]] = [
    ("in-empty", Filter("colour", Operator.IN, []), set()),
    ("not-in-empty", Filter("colour", Operator.NOT_IN, []), {"red", "green"}),
    ("not-in-none-member", Filter("colour", Operator.NOT_IN, [None, "green"]), {"red"}),
    ("in-only-none", Filter("colour", Operator.IN, [None]), set()),
    ("not-in-only-none", Filter("colour", Operator.NOT_IN, [None]), {"red", "green"}),
    ("in-none-then-number", Filter("n", Operator.IN, [None, 5]), {"n5"}),
    ("not-in-none-then-number", Filter("n", Operator.NOT_IN, [None, 5]), {"n7"}),
    ("in-tuple", Filter("colour", Operator.IN, ("red",)), {"red"}),
    ("key-in-empty", Filter(RESERVED_KEY_FIELD, Operator.IN, []), set()),
    ("key-not-in-empty", Filter(RESERVED_KEY_FIELD, Operator.NOT_IN, []), set(CORPUS)),
]


def _oracle(filter_spec: Filter) -> set[str]:
    """The ids ``Filter.matches`` selects from the corpus."""
    selected = set()
    for record_id, data in CORPUS.items():
        if filter_spec.field == RESERVED_KEY_FIELD:
            value: Any = record_id
        else:
            value = data.get(filter_spec.field)
        if filter_spec.matches(value):
            selected.add(record_id)
    return selected


def _complex_cases() -> list[tuple[str, ComplexQuery | Query]]:
    """An empty membership clause followed by one that binds a parameter.

    The empty clause binds nothing, so the placeholder after it has to be
    numbered as if it were not there. Both the flat and the boolean query
    builders thread the numbering, so both are asserted.
    """
    return [
        (
            "complex-and",
            ComplexQuery.AND(
                [
                    Query().filter("colour", "not_in", []),
                    Query().filter("colour", "in", ["red"]),
                ]
            ),
        ),
        (
            "flat-and",
            Query().filter("colour", "not_in", []).filter("colour", "in", ["red"]),
        ),
    ]


def _ids(records: list[Record]) -> set[str]:
    return {record.id for record in records if record.id is not None}


def _make_sync_db(kind: str, root: Path, request: pytest.FixtureRequest) -> Iterator[Any]:
    """Yield one connected sync backend, closing it afterwards."""
    if kind == "memory":
        yield SyncMemoryDatabase()
        return
    if kind == "file":
        yield SyncFileDatabase({"path": str(root / "records.json")})
        return
    if kind == "postgres":
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        for config in request.getfixturevalue("make_postgres_test_db")("test_membership_"):
            db: Any = SyncPostgresDatabase(config)
            db.connect()
            try:
                yield db
            finally:
                db.close()
        return
    if kind == "sqlite":
        db = SyncSQLiteDatabase({"path": str(root / "records.db")})
    else:
        db = SyncDuckDBDatabase({"path": str(root / "records.duckdb"), "table": "records"})
    db.connect()
    try:
        yield db
    finally:
        db.close()


async def _make_async_db(
    kind: str, root: Path, request: pytest.FixtureRequest
) -> AsyncIterator[Any]:
    """Yield one connected async backend, closing it afterwards."""
    if kind == "memory":
        yield AsyncMemoryDatabase()
        return
    if kind == "file":
        yield AsyncFileDatabase({"path": str(root / "records.json")})
        return
    if kind == "postgres":
        from dataknobs_data.backends.postgres import AsyncPostgresDatabase

        for config in request.getfixturevalue("make_postgres_test_db")("test_membership_async_"):
            db: Any = AsyncPostgresDatabase(config)
            await db.connect()
            try:
                yield db
            finally:
                await db.close()
        return
    if kind == "sqlite":
        db = AsyncSQLiteDatabase({"path": str(root / "records.db")})
    else:
        db = AsyncDuckDBDatabase({"path": str(root / "records.duckdb"), "table": "records"})
    await db.connect()
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture(params=BACKENDS)
def sync_db(request: pytest.FixtureRequest) -> Iterator[Any]:
    """A connected sync backend holding the corpus."""
    with tempfile.TemporaryDirectory() as d:
        for db in _make_sync_db(request.param, Path(d), request):
            for record_id, data in CORPUS.items():
                db.create(Record(data=dict(data), id=record_id))
            yield db


@pytest.fixture(params=BACKENDS)
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[Any]:
    """A connected async backend holding the corpus."""
    with tempfile.TemporaryDirectory() as d:
        async for db in _make_async_db(request.param, Path(d), request):
            for record_id, data in CORPUS.items():
                await db.create(Record(data=dict(data), id=record_id))
            yield db


@pytest.mark.parametrize(
    ("filter_spec", "expected"),
    [pytest.param(f, e, id=case_id) for case_id, f, e in CASES],
)
class TestAMembershipFilterIsAnsweredAsTheMatcherAnswersIt:
    """Each backend returns the records ``Filter.matches`` selects."""

    def test_the_oracle_is_the_expectation(self, filter_spec: Filter, expected: set[str]) -> None:
        assert _oracle(filter_spec) == expected

    def test_sync(self, sync_db: Any, filter_spec: Filter, expected: set[str]) -> None:
        assert _ids(sync_db.search(Query(filters=[filter_spec]))) == expected

    async def test_async(self, async_db: Any, filter_spec: Filter, expected: set[str]) -> None:
        assert _ids(await async_db.search(Query(filters=[filter_spec]))) == expected


@pytest.mark.parametrize(
    "query",
    [pytest.param(q, id=case_id) for case_id, q in _complex_cases()],
)
class TestAnEmptyMembershipClauseBindsNothing:
    """A clause that binds no parameter leaves the next placeholder's number alone."""

    def test_sync(self, sync_db: Any, query: ComplexQuery | Query) -> None:
        assert _ids(sync_db.search(query)) == {"red"}

    async def test_async(self, async_db: Any, query: ComplexQuery | Query) -> None:
        assert _ids(await async_db.search(query)) == {"red"}


DIALECTS = [
    ("postgres", "numeric"),
    ("postgres", "pyformat"),
    ("sqlite", "qmark"),
    ("duckdb", "qmark"),
    ("standard", "qmark"),
]


@pytest.mark.parametrize(("dialect", "param_style"), DIALECTS)
class TestTheBuilderRendersOnlyMembersThatCanMatch:
    """What each dialect is sent, including ``standard``, which no backend runs."""

    @staticmethod
    def _render(dialect: str, param_style: str, filter_spec: Filter) -> tuple[str, list[Any]]:
        builder = SQLQueryBuilder("records", dialect=dialect, param_style=param_style)
        return builder.build_search_query(Query(filters=[filter_spec]))

    @pytest.mark.parametrize(
        ("filter_spec", "bound"),
        [
            (Filter("colour", Operator.IN, []), []),
            (Filter("colour", Operator.NOT_IN, []), []),
            (Filter("colour", Operator.IN, [None]), []),
            (Filter("colour", Operator.NOT_IN, [None]), []),
            (Filter("colour", Operator.NOT_IN, [None, "green"]), ["green"]),
            (Filter("colour", Operator.IN, ["red", None, "green"]), ["red", "green"]),
        ],
    )
    def test_no_empty_list_and_no_none_is_bound(
        self, dialect: str, param_style: str, filter_spec: Filter, bound: list[Any]
    ) -> None:
        sql, params = self._render(dialect, param_style, filter_spec)
        assert "()" not in sql
        assert params == bound

    def test_an_empty_in_is_false(self, dialect: str, param_style: str) -> None:
        sql, _ = self._render(dialect, param_style, Filter("colour", Operator.IN, []))
        assert sql.endswith("WHERE FALSE")

    def test_an_empty_not_in_is_the_field_having_a_value(
        self, dialect: str, param_style: str
    ) -> None:
        sql, _ = self._render(dialect, param_style, Filter("colour", Operator.NOT_IN, []))
        assert sql.endswith("IS NOT NULL")
        assert "NOT IN" not in sql


@pytest.mark.parametrize(
    ("dialect", "param_style", "cast"),
    [("postgres", "numeric", "::numeric"), ("duckdb", "qmark", "AS DOUBLE")],
)
@pytest.mark.parametrize("operator", [Operator.IN, Operator.NOT_IN])
def test_the_cast_is_chosen_from_a_member_that_can_match(
    dialect: str, param_style: str, cast: str, operator: Operator
) -> None:
    """A leading ``None`` used to leave a numeric field compared as text."""
    builder = SQLQueryBuilder("records", dialect=dialect, param_style=param_style)
    sql, params = builder.build_search_query(Query(filters=[Filter("n", operator, [None, 5])]))
    assert cast in sql
    assert params == [5]
