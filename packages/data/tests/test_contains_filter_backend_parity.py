# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""``CONTAINS`` means what ``Filter.matches`` means, on every backend.

A field holding a list equals no single value, so no other operator finds a
record by one element of its list. Every answer is compared with the matcher's
over the same corpus, and the matcher's with a hard-coded expectation.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from dataknobs_common.testing import requires_postgres, requires_real_elasticsearch

from dataknobs_data import Filter, Operator, Record
from dataknobs_data.backends.column_layout import NativeColumnLayout
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.exceptions import ValidationError
from dataknobs_data.fields import FieldType, VectorField
from dataknobs_data.query import RESERVED_KEY_FIELD, Query
from dataknobs_data.query_logic import (
    ComplexQuery,
    FilterCondition,
    LogicCondition,
    LogicOperator,
)
from dataknobs_data.schema import DatabaseSchema, FieldSchema

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

BACKENDS = [
    "memory",
    "file",
    "sqlite",
    "duckdb",
    pytest.param("postgres", marks=requires_postgres),
    pytest.param("elasticsearch", marks=requires_real_elasticsearch),
]

#: Elasticsearch maps a field once, so each kind of element has its own.
CORPUS: dict[str, dict[str, Any]] = {
    "ab": {"t": ["a", "b"]},
    "b": {"t": ["b"]},
    "quoted": {"t": ['"a"', "a b", "a%", "it's"]},
    "upper": {"t": ["A"]},
    "empty": {"t": []},
    "nulls": {"t": [None]},
    "null": {"t": None},
    "none": {"other": 1},
    "ints": {"n": [1, 2, 9007199254740993]},
    "reals": {"n": [0.1, 2.0]},
    "one": {"n": 1},
    "flags": {"f": [True]},
    "unset": {"f": [False]},
    "times": {"w": ["2024-01-02T00:00:00", "2024-01-03T04:05:06+02:00", "soon"]},
    "in-object": {"o": {"t": ["a"]}},
    "object-in-list": {"l": [{"k": "a"}]},
}
#: What one Elasticsearch field cannot hold together.
MIXED: dict[str, dict[str, Any]] = {
    "scalar": {"m": "a"},
    "list": {"m": ["a"]},
    "substring": {"m": "xax"},
    "nested": {"m": [["a"], ["b", "a"]]},
    "mapping": {"m": {"a": "a"}},
    "kinds": {"m": [1, "1", True, None, {"k": 1}, [1]]},
}

C, NC = Operator.CONTAINS, Operator.NOT_CONTAINS
STRINGS = {"ab", "b", "quoted", "upper"}
EMPTIES = {"empty", "nulls"}

CASES: list[tuple[str, Filter, set[str]]] = [
    ("string", Filter("t", C, "a"), {"ab"}),
    ("shared", Filter("t", C, "b"), {"ab", "b"}),
    ("not", Filter("t", NC, "b"), {"quoted", "upper"} | EMPTIES),
    ("absent", Filter("t", C, "zzz"), set()),
    ("not-absent", Filter("t", NC, "zzz"), STRINGS | EMPTIES),
    ("quotes", Filter("t", C, '"a"'), {"quoted"}),
    ("apostrophe", Filter("t", C, "it's"), {"quoted"}),
    ("wildcard-is-literal", Filter("t", C, "a%"), {"quoted"}),
    ("case", Filter("t", C, "A"), {"upper"}),
    ("empty-string", Filter("t", C, ""), set()),
    ("none", Filter("t", C, None), set()),
    ("not-none", Filter("t", NC, None), STRINGS | EMPTIES),
    ("nan", Filter("n", C, float("nan")), set()),
    ("int", Filter("n", C, 1), {"ints"}),
    ("int-as-real", Filter("n", C, 2.0), {"ints", "reals"}),
    ("past-a-double", Filter("n", C, 9007199254740993), {"ints"}),
    ("beside-it", Filter("n", C, 9007199254740992), set()),
    ("real", Filter("n", C, 0.1), {"reals"}),
    ("numpy", Filter("n", C, np.int64(1)), {"ints"}),
    ("decimal", Filter("n", C, Decimal("2")), {"ints", "reals"}),
    ("fraction-no-int-equals", Filter("n", C, 1.5), set()),
    ("another-kind", Filter("n", C, "1"), set()),
    ("boolean-is-no-number", Filter("n", C, True), set()),
    ("true", Filter("f", C, True), {"flags"}),
    ("false", Filter("f", C, False), {"unset"}),
    ("not-true", Filter("f", NC, True), {"unset"}),
    ("number-is-no-boolean", Filter("f", C, 1), set()),
    ("naive-time", Filter("w", C, datetime(2024, 1, 2)), {"times"}),
    ("date", Filter("w", C, date(2024, 1, 2)), {"times"}),
    ("instant", Filter("w", C, datetime(2024, 1, 3, 2, 5, 6, tzinfo=UTC)), {"times"}),
    ("other-time", Filter("w", C, datetime(2024, 1, 4)), set()),
    ("path", Filter("o.t", C, "a"), {"in-object"}),
    ("not-a-field", Filter("zz", C, "a"), set()),
    ("not-in-not-a-field", Filter("zz", NC, "a"), set()),
    ("key", Filter(RESERVED_KEY_FIELD, C, "ab"), set()),
    ("not-in-key", Filter(RESERVED_KEY_FIELD, NC, "ab"), set(CORPUS)),
]
MIXED_CASES: list[tuple[str, Filter, set[str]]] = [
    ("only-a-list", Filter("m", C, "a"), {"list"}),
    ("a-value-that-is-no-list", Filter("m", NC, "a"), set(MIXED) - {"list"}),
    ("number-among-kinds", Filter("m", C, 1), {"kinds"}),
    ("string-among-kinds", Filter("m", C, "1"), {"kinds"}),
    ("boolean-among-kinds", Filter("m", C, True), {"kinds"}),
    ("inside-a-mapping", Filter("m.a", C, "a"), set()),
]
#: Where Elasticsearch differs: a time is text there.
ELASTICSEARCH: dict[str, set[str]] = {
    # The corpus has the field mapped as a ``long``, which holds no fraction.
    "real": set(),
    "naive-time": {"times"},
    "date": set(),
    "instant": set(),
}


def _value(record_id: str, corpus: dict[str, dict[str, Any]], field: str) -> Any:
    if field == RESERVED_KEY_FIELD:
        return record_id
    value: Any = corpus[record_id]
    for part in field.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _oracle(spec: Filter, corpus: dict[str, dict[str, Any]]) -> set[str]:
    return {i for i in corpus if spec.matches(_value(i, corpus, spec.field))}


def _records(corpus: dict[str, dict[str, Any]]) -> list[Record]:
    return [Record(data=dict(data), id=record_id) for record_id, data in corpus.items()]


def _ids(records: list[Record]) -> set[str]:
    return {record.id for record in records if record.id is not None}


def _expected(db: Any, case: str, expected: set[str]) -> set[str]:
    if _is_elasticsearch(db):
        return ELASTICSEARCH.get(case, expected)
    return expected


def _is_elasticsearch(db: Any) -> bool:
    return "Elasticsearch" in type(db).__name__


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

        for config in request.getfixturevalue("make_postgres_test_db")("test_contains_"):
            db: Any = SyncPostgresDatabase(config)
            db.connect()
            try:
                yield db
            finally:
                db.close()
        return
    if kind == "elasticsearch":
        from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase

        for config in request.getfixturevalue("make_elasticsearch_test_index")("test_contains_"):
            db = SyncElasticsearchDatabase(config)
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

        for config in request.getfixturevalue("make_postgres_test_db")("test_contains_async_"):
            db: Any = AsyncPostgresDatabase(config)
            await db.connect()
            try:
                yield db
            finally:
                await db.close()
        return
    if kind == "elasticsearch":
        from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase

        for config in request.getfixturevalue("make_elasticsearch_test_index")("test_contains_"):
            db = AsyncElasticsearchDatabase(config)
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
            db.create_batch(_records(CORPUS))
            yield db


@pytest.fixture(params=BACKENDS)
async def async_db(request: pytest.FixtureRequest) -> AsyncIterator[Any]:
    """A connected async backend holding the corpus."""
    with tempfile.TemporaryDirectory() as d:
        async for db in _make_async_db(request.param, Path(d), request):
            await db.create_batch(_records(CORPUS))
            yield db


class TestContainsIsAnsweredAsTheMatcherAnswersIt:
    """A backend is asked every case at once: only the questions differ."""

    @pytest.mark.parametrize(("case", "spec", "expected"), CASES, ids=[c[0] for c in CASES])
    def test_the_oracle_is_the_expectation(
        self, case: str, spec: Filter, expected: set[str]
    ) -> None:
        assert _oracle(spec, CORPUS) == expected

    def test_sync(self, sync_db: Any) -> None:
        wrong = {}
        for case, spec, expected in CASES:
            query = Query(filters=[spec])
            answer = _expected(sync_db, case, expected)
            found, counted = _ids(sync_db.search(query)), sync_db.count(query)
            if (found, counted) != (answer, len(answer)):
                wrong[case] = (found, counted, answer)
        assert wrong == {}

    async def test_async(self, async_db: Any) -> None:
        wrong = {}
        for case, spec, expected in CASES:
            answer = _expected(async_db, case, expected)
            found = _ids(await async_db.search(Query(filters=[spec])))
            if found != answer:
                wrong[case] = (found, answer)
        assert wrong == {}


class TestOnlyAListContains:
    """A value on its own contains nothing, whatever the list's elements are."""

    @pytest.mark.parametrize(
        ("case", "spec", "expected"), MIXED_CASES, ids=[c[0] for c in MIXED_CASES]
    )
    def test_the_oracle_is_the_expectation(
        self, case: str, spec: Filter, expected: set[str]
    ) -> None:
        assert _oracle(spec, MIXED) == expected

    def test_sync(self, sync_db: Any) -> None:
        if _is_elasticsearch(sync_db):
            pytest.skip("one Elasticsearch field cannot hold these")
        sync_db.create_batch(_records(MIXED))
        wrong = {}
        for case, spec, expected in MIXED_CASES:
            found = _ids(sync_db.search(Query(filters=[spec])))
            if found != expected:
                wrong[case] = (found, expected)
        assert wrong == {}

    async def test_async(self, async_db: Any) -> None:
        if _is_elasticsearch(async_db):
            pytest.skip("one Elasticsearch field cannot hold these")
        await async_db.create_batch(_records(MIXED))
        wrong = {}
        for case, spec, expected in MIXED_CASES:
            found = _ids(await async_db.search(Query(filters=[spec])))
            if found != expected:
                wrong[case] = (found, expected)
        assert wrong == {}


class TestContainsAmongOtherConditions:
    """It binds as many parameters as it names, whatever comes before and after."""

    def test_between_other_filters(self, sync_db: Any) -> None:
        query = (
            Query()
            .filter("t", Operator.EXISTS)
            .filter("id", "!=", "b")
            .filter("t", "contains", "b")
            .filter("id", "in", ["ab", "b"])
        )
        assert _ids(sync_db.search(query)) == {"ab"}

    def test_both_at_once(self, sync_db: Any) -> None:
        query = Query().filter("t", "contains", "b").filter("t", "not contains", "a")
        assert _ids(sync_db.search(query)) == {"b"}

    def test_or(self, sync_db: Any) -> None:
        query = ComplexQuery.OR(
            [Query().filter("t", "contains", "a"), Query().filter("n", "contains", 2)]
        )
        assert _ids(sync_db.search(query)) == {"ab", "ints", "reals"}

    def test_not_matches_a_record_without_the_field(self, sync_db: Any) -> None:
        spec = FilterCondition(Filter("t", C, "b"))
        query = ComplexQuery(condition=LogicCondition(LogicOperator.NOT, [spec]))
        assert _ids(sync_db.search(query)) == set(CORPUS) - {"ab", "b"}


class TestTheOperand:
    @pytest.mark.parametrize("operator", [C, NC])
    @pytest.mark.parametrize(
        "value", [["a"], ("a",), {"a"}, frozenset(), {"a": 1}, {}.keys(), np.array([1])]
    )
    def test_a_collection_is_refused(self, operator: Operator, value: Any) -> None:
        with pytest.raises(ValueError, match="one value"):
            Filter("t", operator, value)

    @pytest.mark.parametrize("value", ["a", b"a", 1, None, True, date(2024, 1, 1)])
    def test_one_value_is_taken(self, value: Any) -> None:
        spec = Filter("t", C, value)
        assert Filter.from_dict(spec.to_dict()) == spec
        assert hash(Filter.from_dict(spec.to_dict())) == hash(spec)


class TestTheMatcher:
    @pytest.mark.parametrize(
        "held", [["a"], ("a",), np.array(["a", "b"]), [None, "a"], [["x"], "a"]]
    )
    def test_what_holds_elements(self, held: Any) -> None:
        assert Filter("t", C, "a").matches(held)
        assert not Filter("t", NC, "a").matches(held)

    @pytest.mark.parametrize("held", ["a", "xax", {"a": 1}, 5, b"a", [], [["a"]], np.array([])])
    def test_what_does_not(self, held: Any) -> None:
        assert not Filter("t", C, "a").matches(held)
        assert Filter("t", NC, "a").matches(held)

    def test_a_missing_value(self) -> None:
        assert not Filter("t", C, "a").matches(None)
        assert not Filter("t", NC, "a").matches(None)

    def test_a_vector(self) -> None:
        field = VectorField(np.array([1.0, 2.0]), name="v")
        assert Filter("v", C, 2).matches(field.value)
        assert not Filter("v", C, 3).matches(field.value)

    def test_rows_of_an_array_are_not_elements_equal_to_a_number(self) -> None:
        assert not Filter("v", C, 1).matches(np.array([[1, 1], [1, 2]]))


class TestANativeColumnHoldsOneValue:
    @staticmethod
    def _builder(scope: list[Filter] | None = None) -> SQLQueryBuilder:
        schema = DatabaseSchema(
            fields={
                "k": FieldSchema("k", FieldType.STRING),
                "s": FieldSchema("s", FieldType.STRING),
                "j": FieldSchema("j", FieldType.JSON),
            }
        )
        layout = NativeColumnLayout(schema, id_column="k", scope=scope or [])
        return SQLQueryBuilder("t", dialect="sqlite", layout=layout)

    def test_so_it_contains_nothing(self) -> None:
        builder = self._builder()
        assert builder._build_filter_clause(Filter("s", C, "a"), 1) == ("FALSE", [])
        assert builder._build_filter_clause(Filter("s", NC, "a"), 1) == ('"s" IS NOT NULL', [])

    def test_a_structured_column_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="only EXISTS"):
            self._builder()._build_filter_clause(Filter("j", C, "a"), 1)

    @pytest.mark.parametrize(
        ("operator", "why"), [(C, "can match no row"), (NC, "excludes no row")]
    )
    def test_a_scope_naming_one_is_refused(self, operator: Operator, why: str) -> None:
        with pytest.raises(ValidationError, match=why):
            self._builder([Filter("s", operator, "a")])


def test_a_dialect_that_cannot_read_json_refuses_it() -> None:
    with pytest.raises(ValueError, match="CONTAINS"):
        SQLQueryBuilder("t", dialect="standard")._build_filter_clause(Filter("t", C, "a"), 1)


@pytest.mark.parametrize("dialect", ["postgres", "sqlite", "duckdb"])
def test_a_field_name_cannot_carry_sql(dialect: str) -> None:
    with pytest.raises(ValueError, match="field"):
        SQLQueryBuilder("t", dialect=dialect)._build_filter_clause(Filter("a') --", C, "a"), 1)
