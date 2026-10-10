"""What a declared column can hold, and what a filter over it answers: written-out cases.

Each case states the rows it selects, so a change to the oracle cannot silently
move the builder with it, and also checks those rows against the oracle,
``Filter.matches`` over the records the layout returns. The cases cover, for
each column type, a value the engine would refuse, one it would silently
coerce, and one it holds; they run on every engine the native layout renders
for. The tables are authored for these tests and model nothing.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any

import pytest
from _native_tables import ENGINES, Engine, Table, engine_for
from dataknobs_common.exceptions import ValidationError

from dataknobs_data import SqlType, sql_types
from dataknobs_data.backends.column_layout import NativeColumnLayout
from dataknobs_data.query import Filter, Operator, Query, SortSpec
from dataknobs_data.schema import DatabaseSchema

U1, U2, U3 = (uuid.UUID(int=n) for n in (0xA1, 0xA2, 0xA3))
R1, R2, R3 = str(U1), str(U2), str(U3)


def _types(postgres: str, duckdb: str, sqlite: str) -> dict[str, str]:
    return {"postgres": postgres, "duckdb": duckdb, "sqlite": sqlite}


#: One column per declared type. Row 3 is ``NULL`` everywhere but its key, so
#: every answer has a row no comparison may match.
SAMPLE = Table(
    "sample",
    {
        "u": _types("uuid", "UUID", "TEXT"),
        "s": _types("text", "VARCHAR", "TEXT"),
        "n": _types("integer", "INTEGER", "INTEGER"),
        "x": _types("double precision", "DOUBLE", "REAL"),
        "b": _types("boolean", "BOOLEAN", "BOOLEAN"),
        "t": _types("timestamp", "TIMESTAMP", "TIMESTAMP"),
        "j": _types("jsonb", "JSON", "TEXT"),
    },
    [
        (U1, "three", 3, 1.5, True, datetime(2024, 1, 1), '{"k": 1}'),
        (U2, "five", 5, 2.5, False, datetime(2024, 6, 1), '{"k": 2}'),
        (U3, None, None, None, None, None, None),
    ],
)

SAMPLE_FIELDS: dict[str, Any] = {
    "u": {"type": "string", "metadata": {"sql_type": "uuid"}},
    "s": "string",
    "n": "integer",
    "x": "float",
    "b": "boolean",
    "t": "datetime",
    "j": "json",
}

#: Mixed case, which the sample table deliberately avoids, for text order.
WORDS = [(uuid.UUID(int=0xB0 + i), word) for i, word in enumerate(("apple", "Banana", "cherry"))]
WORD = Table(
    "word", {"id": _types("uuid", "UUID", "TEXT"), "w": _types("text", "VARCHAR", "TEXT")}, WORDS
)

#: A zoned column: the sample table's ``t`` is a naive timestamp.
FIRST, SECOND = uuid.UUID(int=0xC1), uuid.UUID(int=0xC2)
EVENT = Table(
    "event",
    {
        "id": _types("uuid", "UUID", "TEXT"),
        "at": _types("timestamptz", "TIMESTAMPTZ", "TIMESTAMPTZ"),
    },
    [(FIRST, datetime(2024, 1, 1, tzinfo=UTC)), (SECOND, datetime(2024, 6, 1, 12, tzinfo=UTC))],
)


@pytest.fixture(scope="module", params=ENGINES)
def engine(request: pytest.FixtureRequest) -> Iterator[Engine]:
    yield from engine_for(request, [SAMPLE, WORD, EVENT])


def _layout(
    fields: dict[str, Any] | None = None, scope: list[Filter] | None = None
) -> NativeColumnLayout:
    schema = DatabaseSchema.from_dict({"fields": fields or SAMPLE_FIELDS})
    return NativeColumnLayout(schema, id_column="u", scope=scope or [])


def _search(engine: Engine, table: str, layout: NativeColumnLayout, query: Query) -> list[Any]:
    builder = engine.builder(table, layout)
    return [
        builder.record_from_row(row) for row in engine.fetch(*builder.build_search_query(query))
    ]


def _ids(records: list[Any]) -> set[str]:
    return {str(record.storage_id) for record in records}


def _canonical(value: Any) -> Any:
    """The uuid column's ``bind``, as the oracle must see it: the canonical text."""
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        return value


def _oracle(engine: Engine, table: str, layout: NativeColumnLayout, spec: Filter) -> set[str]:
    """The rows ``Filter.matches`` selects over the records this layout returns."""
    if spec.field in ("u", "id"):
        value = (
            [_canonical(member) for member in spec.value]
            if isinstance(spec.value, list)
            else _canonical(spec.value)
        )
        spec = Filter(spec.field, spec.operator, value)
    return {
        str(record.storage_id)
        for record in _search(engine, table, layout, Query())
        if spec.matches(record.get_value(spec.field))
    }


#: (field, operator, value, the rows it selects).
CASES: list[tuple[str, Operator, Any, set[str]]] = [
    # integer: a value the column cannot hold, and asyncpg's truncation of 3.5 to 3
    ("n", Operator.EQ, "abc", set()),
    ("n", Operator.EQ, "3", set()),
    ("n", Operator.EQ, 3, {R1}),
    ("n", Operator.EQ, 3.0, {R1}),
    ("n", Operator.EQ, 3.5, set()),
    ("n", Operator.EQ, 5.0000000001, set()),
    ("n", Operator.EQ, 2**40, set()),
    ("n", Operator.EQ, 2**70, set()),
    ("n", Operator.NEQ, 3, {R2}),
    ("n", Operator.NEQ, "abc", {R1, R2}),
    ("n", Operator.GT, 2.5, {R1, R2}),
    ("n", Operator.GTE, 3.5, {R2}),
    ("n", Operator.GT, 3.5, {R2}),
    ("n", Operator.LT, 5, {R1}),
    ("n", Operator.GTE, 3.0, {R1, R2}),
    ("n", Operator.LT, 2**70, {R1, R2}),
    ("n", Operator.GT, "abc", set()),
    ("n", Operator.BETWEEN, [3.5, 5.5], {R2}),
    ("n", Operator.BETWEEN, [3, 5], {R1, R2}),
    ("n", Operator.NOT_BETWEEN, [3.5, 5.5], {R1}),
    ("n", Operator.NOT_BETWEEN, ["a", "b"], {R1, R2}),
    ("n", Operator.IN, [3, "x", 5.5], {R1}),
    ("n", Operator.NOT_IN, [3, "x"], {R2}),
    ("n", Operator.LIKE, "3%", set()),
    ("n", Operator.NOT_LIKE, "3%", set()),
    # float
    ("x", Operator.EQ, "1.5", set()),
    ("x", Operator.EQ, 1.5, {R1}),
    ("x", Operator.EQ, 2, set()),
    ("x", Operator.GT, 2, {R2}),
    ("x", Operator.LTE, 1.5, {R1}),
    # string
    ("s", Operator.EQ, 5, set()),
    ("s", Operator.GT, 3, set()),
    ("s", Operator.EQ, "three", {R1}),
    ("s", Operator.GT, "five", {R1}),
    ("s", Operator.LT, "T", set()),
    ("s", Operator.BETWEEN, ["T", "u"], {R1, R2}),
    ("s", Operator.BETWEEN, ["g", "u"], {R1}),
    ("s", Operator.LIKE, "th%", {R1}),
    ("s", Operator.NOT_LIKE, "th%", {R2}),
    ("s", Operator.STARTS_WITH, "fi", {R2}),
    ("s", Operator.REGEX, "^f", {R2}),
    ("s", Operator.IN, ["five", 7], {R2}),
    # boolean: false orders below true, as in Python and the engines
    ("b", Operator.EQ, True, {R1}),
    ("b", Operator.EQ, "true", set()),
    ("b", Operator.NEQ, True, {R2}),
    ("b", Operator.GT, False, {R1}),
    ("b", Operator.LTE, True, {R1, R2}),
    ("b", Operator.BETWEEN, [False, True], {R1, R2}),
    # datetime: a string is the time ``read_timestamp`` reads, a date its
    # midnight, and an aware bound relates to no naive value
    ("t", Operator.GT, "2024-03-01", {R2}),
    ("t", Operator.EQ, "2024-01-01T00:00:00", {R1}),
    ("t", Operator.EQ, "2024-01-01", {R1}),
    ("t", Operator.EQ, "20240101", set()),
    ("t", Operator.NEQ, "20240101", {R1, R2}),
    ("t", Operator.EQ, "2024-02-30", set()),
    ("t", Operator.IN, ["2024-06-01T00:00:00", "x"], {R2}),
    ("t", Operator.EQ, datetime(2024, 1, 1), {R1}),
    ("t", Operator.GT, datetime(2024, 3, 1), {R2}),
    ("t", Operator.GT, date(2024, 3, 1), {R2}),
    ("t", Operator.EQ, date(2024, 1, 1), {R1}),
    ("t", Operator.BETWEEN, [date(2024, 1, 1), "2024-03-01"], {R1}),
    ("t", Operator.EQ, datetime(2024, 1, 1, tzinfo=UTC), set()),
    ("t", Operator.NEQ, datetime(2024, 1, 1, tzinfo=UTC), {R1, R2}),
    ("t", Operator.GT, "2024-03-01T00:00:00+00:00", set()),
    ("t", Operator.LIKE, "2024%", set()),
    # uuid: canonical text either way in, and its text for the string family
    ("u", Operator.EQ, "not-a-uuid", set()),
    ("u", Operator.EQ, U1, {R1}),
    ("u", Operator.EQ, R1.upper(), {R1}),
    ("u", Operator.LIKE, "00000000%", {R1, R2, R3}),
    ("u", Operator.IN, [R2, "x"], {R2}),
    ("u", Operator.GT, R1, {R2, R3}),
    ("u", Operator.LT, "not-a-uuid", {R1, R2, R3}),
    ("u", Operator.GT, "not-a-uuid", set()),
    # presence, on every kind of column, including one that is otherwise refused
    ("n", Operator.EXISTS, None, {R1, R2}),
    ("t", Operator.NOT_EXISTS, None, {R3}),
    ("j", Operator.EXISTS, None, {R1, R2}),
]


@pytest.mark.parametrize(
    ("field", "op", "value", "expected"),
    CASES,
    ids=[f"{field} {op.name} {value!r}" for field, op, value, _ in CASES],
)
def test_every_answer_is_the_oracles(
    engine: Engine,
    field: str,
    op: Operator,
    value: Any,
    expected: set[str],
) -> None:
    """No case raises; each selects what ``Filter.matches`` selects."""
    layout = _layout()
    spec = Filter(field, op, value)

    found = _ids(_search(engine, "sample", layout, Query(filters=[spec])))

    assert found == expected
    assert found == _oracle(engine, "sample", layout, spec)
    builder = engine.builder("sample", layout)
    assert engine.count(*builder.build_count_query(Query(filters=[spec]))) == len(expected)


# --- once deviations, now parity ---------------------------------------------------


def test_bool_and_int_are_separate_domains(engine: Engine) -> None:
    """A boolean never equals or orders against a number, here or in the oracle."""
    layout = _layout()
    for field, op, value in (
        ("n", Operator.EQ, True),
        ("b", Operator.EQ, 1),
        ("b", Operator.GT, 0),
        ("n", Operator.LT, True),
        ("b", Operator.IN, [1, 0]),
    ):
        spec = Filter(field, op, value)
        found = _ids(_search(engine, "sample", layout, Query(filters=[spec])))
        assert found == set() == _oracle(engine, "sample", layout, spec)
    spec = Filter("b", Operator.NEQ, 1)
    found = _ids(_search(engine, "sample", layout, Query(filters=[spec])))
    assert found == {R1, R2} == _oracle(engine, "sample", layout, spec)


def test_like_folds_case(engine: Engine) -> None:
    """``LIKE`` ignores case, as in memory."""
    layout = _layout()
    for op in (Operator.LIKE, Operator.NOT_LIKE):
        spec = Filter("s", op, "TH%")
        found = _ids(_search(engine, "sample", layout, Query(filters=[spec])))
        assert found == _oracle(engine, "sample", layout, spec)
    assert _ids(
        _search(engine, "sample", layout, Query(filters=[Filter("s", Operator.LIKE, "TH%")]))
    ) == {R1}


# --- refusals ------------------------------------------------------------------------


def test_a_structured_column_is_refused_by_name(engine: Engine) -> None:
    """``json`` has no one equality across engines, so a ``FALSE`` would be a wrong answer."""
    builder = engine.builder("sample", _layout())
    for op, value in ((Operator.EQ, {"k": 1}), (Operator.LIKE, "%k%"), (Operator.GT, 1)):
        with pytest.raises(ValidationError, match=r"'j'.*json"):
            builder.build_search_query(Query(filters=[Filter("j", op, value)]))


def test_a_scope_no_row_can_satisfy_is_refused() -> None:
    """``n = 'abc'`` as a scope would be a store that is always empty: a config error."""
    with pytest.raises(ValidationError, match="scope"):
        _layout(scope=[Filter("n", Operator.EQ, "abc")])


# --- the registry --------------------------------------------------------------------


def test_sql_types_are_an_open_registry(engine: Engine) -> None:
    """A consumer registers a type by name; an unregistered name is refused with the known ones.

    A ``FieldType`` name is not a registered ``sql_type``: the built-in answers
    are keyed by ``FieldType``, so ``integer`` cannot be re-registered to change
    every integer column.
    """
    for name in ("uuidd", "integer"):
        fields = {**SAMPLE_FIELDS, "s": {"type": "string", "metadata": {"sql_type": name}}}
        with pytest.raises(ValidationError, match=f"sql_type: {name}`.*uuid"):
            _layout(fields)

    # A text column holding only lower-case strings: a bound it cannot hold binds nothing.
    sql_types.register(
        "lowercase_text",
        SqlType(
            kinds=frozenset({"string"}),
            text="stored",
            bind=lambda v: v if not isinstance(v, str) or v == v.lower() else None,
        ),
    )
    try:
        fields = {
            **SAMPLE_FIELDS,
            "s": {"type": "string", "metadata": {"sql_type": "lowercase_text"}},
        }
        layout = _layout(fields)
        assert _ids(
            _search(engine, "sample", layout, Query(filters=[Filter("s", Operator.EQ, "five")]))
        ) == {R2}
        assert (
            _search(engine, "sample", layout, Query(filters=[Filter("s", Operator.EQ, "FIVE")]))
            == []
        )
        # Its values still order as text, as ``Filter.matches`` orders them.
        spec = Filter("s", Operator.GT, "a")
        found = _ids(_search(engine, "sample", layout, Query(filters=[spec])))
        assert found == {R1, R2} == _oracle(engine, "sample", layout, spec)
    finally:
        sql_types.unregister("lowercase_text")


# --- text order ----------------------------------------------------------------------


def test_text_orders_and_sorts_by_code_point(engine: Engine) -> None:
    """``'B' < 'apple'``, as in Python, and not as a locale collation has it."""
    schema = DatabaseSchema.from_dict(
        {"fields": {"id": {"type": "string", "metadata": {"sql_type": "uuid"}}, "w": "string"}}
    )
    layout = NativeColumnLayout(schema, id_column="id")

    def words(query: Query) -> list[str]:
        return [str(record.get_value("w")) for record in _search(engine, "word", layout, query)]

    assert words(Query(sort_specs=[SortSpec("w")])) == ["Banana", "apple", "cherry"]
    assert sorted(words(Query(filters=[Filter("w", Operator.LT, "a")]))) == ["Banana"]
    assert sorted(words(Query(filters=[Filter("w", Operator.GT, "B")]))) == [
        "Banana",
        "apple",
        "cherry",
    ]


# --- an aware column ----------------------------------------------------------------


def test_a_timestamptz_column_relates_aware_times(engine: Engine) -> None:
    """``sql_type: timestamptz``: aware bounds relate, naive ones do not, as in the oracle.

    A record holds the instant in UTC, so a date is that day's UTC midnight and
    a zoned string the instant it names.
    """
    schema = DatabaseSchema.from_dict(
        {
            "fields": {
                "id": {"type": "string", "metadata": {"sql_type": "uuid"}},
                "at": {"type": "datetime", "metadata": {"sql_type": "timestamptz"}},
            }
        }
    )
    layout = NativeColumnLayout(schema, id_column="id")
    one, two = str(FIRST), str(SECOND)

    for op, value, expected in (
        (Operator.EQ, datetime(2024, 1, 1, tzinfo=UTC), {one}),
        (Operator.EQ, "2024-06-01T13:00:00+01:00", {two}),
        (Operator.EQ, "2024-01-01T00:00:00Z", {one}),
        (Operator.EQ, date(2024, 1, 1), {one}),
        (Operator.GT, date(2024, 3, 1), {two}),
        (Operator.EQ, datetime(2024, 1, 1), set()),
        (Operator.GT, "2024-03-01", set()),
        (Operator.NEQ, datetime(2024, 1, 1), {one, two}),
    ):
        spec = Filter("at", op, value)
        found = _ids(_search(engine, "event", layout, Query(filters=[spec])))
        assert found == expected, spec
        assert found == _oracle(engine, "event", layout, spec), spec
