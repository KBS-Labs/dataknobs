"""The native column layout, at the builder: what it refuses, and how it numbers.

Engine behaviour is in ``test_native_column_layout_engines.py``. This module
needs no database: it checks what the builder renders, and what it refuses
before any SQL is built.
"""

from __future__ import annotations

import inspect
import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from dataknobs_common.exceptions import OperationError, ValidationError

from dataknobs_data import NATIVE_FIELD_KEYS, SQL_TYPE_KEY, SqlType, sql_types
from dataknobs_data.backends.column_layout import (
    JsonbLayout,
    NativeColumnLayout,
    read_layout_config,
)
from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.database import extract_schema_from_config
from dataknobs_data.fields import FieldType
from dataknobs_data.query import Filter, Operator, Query, SortSpec
from dataknobs_data.query_logic import (
    ComplexQuery,
    FilterCondition,
    LogicCondition,
    LogicOperator,
)
from dataknobs_data.records import Record
from dataknobs_data.schema import DatabaseSchema, FieldSchema

SCHEMA = DatabaseSchema.from_dict(
    {
        "fields": {
            "k": {"type": "string", "sql_type": "uuid"},
            "shape": "string",
            "size": "integer",
            "solid": "boolean",
            "made": "datetime",
            "payload": "json",
        }
    },
    keys=NATIVE_FIELD_KEYS,
)


def _layout(scope: list[Filter] | None = None) -> NativeColumnLayout:
    return NativeColumnLayout(SCHEMA, id_column="k", scope=scope or [])


def _builder(
    dialect: str = "postgres", style: str = "numeric", layout: Any = None
) -> SQLQueryBuilder:
    return SQLQueryBuilder("shapes", dialect=dialect, param_style=style, layout=layout or _layout())


# -- refused before SQL ---------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda b: b.build_search_query(Query(filters=[Filter("colour", Operator.EQ, "red")])),
        lambda b: b.build_search_query(Query(sort_specs=[SortSpec("colour")])),
        lambda b: b.build_complex_search_query(
            ComplexQuery(
                condition=LogicCondition(
                    LogicOperator.OR,
                    [
                        FilterCondition(Filter("shape", Operator.EQ, "apple")),
                        FilterCondition(Filter("colour", Operator.EQ, "red")),
                    ],
                )
            )
        ),
        lambda b: b.build_count_query(Query(filters=[Filter("colour", Operator.EXISTS)])),
        lambda b: b.build_where_clause(Query(filters=[Filter("shape.inner", Operator.EQ, 1)])),
    ],
    ids=["filter", "sort", "or-leaf", "count", "dotted"],
)
def test_an_undeclared_column_is_refused_before_sql(build: Any) -> None:
    with pytest.raises(ValidationError, match="no declared column") as caught:
        build(_builder())
    assert caught.value.context["table"] == "shapes"
    assert caught.value.context["declared"] == sorted(SCHEMA.fields)


def test_an_undeclared_scope_or_key_column_is_refused_at_construction() -> None:
    with pytest.raises(ValidationError, match="no declared column 'colour'"):
        _layout([Filter("colour", Operator.EQ, "red")])
    with pytest.raises(ValidationError, match="id_column 'key' is not a declared column"):
        NativeColumnLayout(SCHEMA, id_column="key")
    with pytest.raises(ValidationError, match="declares none"):
        NativeColumnLayout(DatabaseSchema(), id_column="k")


@pytest.mark.parametrize(
    ("declaration", "named"),
    [
        ("float", "float"),
        ("boolean", "boolean"),
        ("datetime", "datetime"),
        ({"type": "datetime", "metadata": {SQL_TYPE_KEY: "timestamptz"}}, "timestamptz"),
        ("json", "json"),
    ],
    ids=["float", "boolean", "datetime", "timestamptz", "json"],
)
def test_a_key_column_whose_text_differs_by_engine_is_refused(declaration: Any, named: str) -> None:
    """A record's key is the column's text, which a read must find the row by again.

    Each engine writes a float, a boolean or a time as different text, and
    none writes it as Python does, so a key read back would not find its row.
    """
    schema = DatabaseSchema.from_dict({"fields": {"k": declaration, "shape": "string"}})
    with pytest.raises(ValidationError, match=f"id_column 'k' is declared {named}") as caught:
        NativeColumnLayout(schema, id_column="k")
    assert caught.value.context["id_column"] == "k"


@pytest.mark.parametrize("declaration", ["string", "text", "integer"])
def test_a_key_column_with_one_text_on_every_engine_is_taken(declaration: str) -> None:
    schema = DatabaseSchema.from_dict({"fields": {"k": declaration, "shape": "string"}})
    assert NativeColumnLayout(schema, id_column="k").id_column == "k"


def test_an_integer_key_is_compared_in_its_own_type_only_for_its_own_text() -> None:
    """``"10"`` is an integer's text, so it is sent as a ``bigint`` and the index serves it.

    ``"010"`` is no integer's text, so the column is compared as text, where it
    matches nothing.
    """
    schema = DatabaseSchema.from_dict({"fields": {"n": "integer", "shape": "string"}})
    builder = _builder(layout=NativeColumnLayout(schema, id_column="n"))
    sql, params = builder.build_read_query("10")
    assert sql.endswith('WHERE "n" = CAST($1 AS bigint)')
    assert params == [10]
    sql, params = builder.build_read_query("010")
    assert sql.endswith('WHERE CAST("n" AS TEXT) = $1')
    assert params == ["010"]


@pytest.mark.parametrize(
    "scope",
    [
        Filter("shape", Operator.EQ, 5),
        Filter("size", Operator.IN, ["a", "b"]),
        Filter("size", Operator.LIKE, "3%"),
        Filter("solid", Operator.EQ, 1),
        Filter("made", Operator.GT, "not a time"),
    ],
    ids=lambda f: f"{f.field} {f.operator.value} {f.value!r}",
)
def test_a_scope_that_can_match_no_row_is_refused(scope: Filter) -> None:
    """Rendered, it would be FALSE in every statement: a store that is always empty."""
    with pytest.raises(ValidationError, match="can match no row"):
        _layout([scope])


@pytest.mark.parametrize(
    "scope",
    [
        Filter("shape", Operator.NEQ, 5),
        Filter("size", Operator.NOT_IN, ["a", "b"]),
        Filter("size", Operator.NOT_BETWEEN, ["a", "b"]),
        Filter("made", Operator.NEQ, "not a time"),
    ],
    ids=lambda f: f"{f.field} {f.operator.value} {f.value!r}",
)
def test_a_negated_scope_whose_value_its_column_cannot_hold_is_refused_as_excluding_nothing(
    scope: Filter,
) -> None:
    """A negation of a value no row holds is every present row: it scopes nothing.

    ``Filter.matches`` answers it True for every value of the column, so it
    could match every row; refusing it is right, but not as matching none.
    """
    with pytest.raises(ValidationError, match="excludes no row the column has a value in"):
        _layout([scope])


def test_a_scope_that_can_match_is_taken() -> None:
    layout = _layout(
        [
            Filter("shape", Operator.IN, ["apple", 5]),
            Filter("made", Operator.GT, "2024-01-01"),
            Filter("size", Operator.NOT_EXISTS),
        ]
    )
    assert len(layout.scope) == 3


def test_a_structured_column_is_refused_for_all_but_presence() -> None:
    builder = _builder()
    for op, value in ((Operator.EQ, {"a": 1}), (Operator.LIKE, "%a%"), (Operator.IN, [1])):
        with pytest.raises(ValidationError, match="only EXISTS and NOT_EXISTS"):
            builder.build_search_query(Query(filters=[Filter("payload", op, value)]))
    with pytest.raises(ValidationError, match="no order a sort can follow"):
        builder.build_search_query(Query(sort_specs=[SortSpec("payload")]))
    sql, _ = builder.build_search_query(Query(filters=[Filter("payload", Operator.EXISTS)]))
    assert sql.endswith('WHERE "payload" IS NOT NULL')


def test_the_standard_dialect_is_refused() -> None:
    with pytest.raises(ValidationError, match="renders for"):
        SQLQueryBuilder("shapes", dialect="standard", layout=_layout())


def test_an_unregistered_sql_type_is_refused_listing_the_registered() -> None:
    schema = DatabaseSchema(
        fields={"k": FieldSchema("k", FieldType.STRING, metadata={SQL_TYPE_KEY: "inet"})}
    )
    with pytest.raises(ValidationError, match="'uuid'") as caught:
        NativeColumnLayout(schema, id_column="k")
    assert "timestamptz" in caught.value.context["registered"]
    bad = DatabaseSchema(
        fields={"k": FieldSchema("k", FieldType.STRING, metadata={SQL_TYPE_KEY: ""})}
    )
    with pytest.raises(ValidationError, match="name of a registered SQL type"):
        NativeColumnLayout(bad, id_column="k")


def test_a_registered_sql_type_is_honoured() -> None:
    name = "test_citext_for_native_layout"
    sql_types.register(name, SqlType(kinds=frozenset({"string"}), text="stored"))
    try:
        schema = DatabaseSchema(
            fields={
                "k": FieldSchema("k", FieldType.STRING),
                "tag": FieldSchema("tag", FieldType.STRING, metadata={SQL_TYPE_KEY: name}),
            }
        )
        builder = _builder(layout=NativeColumnLayout(schema, id_column="k"))
        sql, params = builder.build_search_query(Query(filters=[Filter("tag", Operator.EQ, "A")]))
        assert sql.endswith('WHERE "tag" = $1') and params == ["A"]
        sql, params = builder.build_search_query(Query(filters=[Filter("tag", Operator.EQ, 1)]))
        assert sql.endswith("WHERE FALSE") and params == []
    finally:
        sql_types.unregister(name)


def test_an_sql_type_declares_only_kinds_it_can_hold() -> None:
    with pytest.raises(ValueError, match="holds one or more of"):
        SqlType(kinds=frozenset({"vector"}))
    with pytest.raises(ValueError, match="text is one of"):
        SqlType(kinds=frozenset({"string"}), text="both")  # type: ignore[arg-type]


def test_a_python_built_schema_passing_a_member_works() -> None:
    schema = DatabaseSchema(
        fields={
            "k": FieldSchema("k", FieldType.STRING),
            "size": FieldSchema("size", FieldType.INTEGER),
        }
    )
    builder = _builder(layout=NativeColumnLayout(schema, id_column="k"))
    sql, params = builder.build_search_query(Query(filters=[Filter("size", Operator.GT, 3)]))
    assert sql == 'SELECT "k", "size" FROM "shapes" WHERE "size" > CAST($1 AS bigint)'
    assert params == [3]


# -- what is selected, and where the key goes ------------------------------------


def test_the_select_list_is_the_declared_columns() -> None:
    sql, _ = _builder().build_search_query(Query())
    assert sql == 'SELECT "k", "shape", "size", "solid", "made", "payload" FROM "shapes"'


def test_the_storage_key_routes_to_id_column() -> None:
    builder = _builder()
    key = "12345678-1234-5678-1234-567812345678"
    sql, params = builder.build_search_query(Query(filters=[Filter("id", Operator.EQ, key)]))
    assert sql.endswith('WHERE "k" = CAST($1 AS uuid)') and params == [key]
    sql, params = builder.build_read_query(key.upper())
    assert sql.endswith('WHERE "k" = CAST($1 AS uuid)') and params == [key]
    sql, params = builder.build_read_query("not-a-uuid")
    assert sql.endswith('WHERE CAST("k" AS TEXT) = $1') and params == ["not-a-uuid"]


def test_a_value_its_column_cannot_hold_binds_nothing() -> None:
    builder = _builder()
    cases = {
        Filter("size", Operator.EQ, "3"): "FALSE",
        Filter("size", Operator.NEQ, "3"): '"size" IS NOT NULL',
        Filter("solid", Operator.EQ, 1): "FALSE",
        Filter("shape", Operator.GT, 3): "FALSE",
        Filter("size", Operator.NOT_LIKE, "3%"): "FALSE",
        Filter("made", Operator.EQ, "not a time"): "FALSE",
    }
    for spec, clause in cases.items():
        sql, params = builder.build_search_query(Query(filters=[spec]))
        assert sql.endswith(f"WHERE {clause}"), spec
        assert params == [], spec


# -- writes ----------------------------------------------------------------------

WRITES: dict[str, tuple[Any, ...]] = {
    "build_create_query": (Record({"shape": "apple"}),),
    "build_update_query": ("r1", Record({"shape": "apple"})),
    "build_delete_query": ("r1",),
    "build_batch_create_queries": ([Record({"shape": "apple"})],),
    "build_batch_upsert_queries": ([Record({"shape": "apple"})],),
    "build_batch_update_queries": ([("r1", Record({"shape": "apple"}))],),
    "build_batch_update_rows": ([("r1", Record({"shape": "apple"}))],),
    "build_batch_delete_query": (["r1"],),
    "build_batch_create_query": ([Record({"shape": "apple"})],),
    "build_batch_upsert_query": ([Record({"shape": "apple"})],),
    "build_batch_update_query": ([("r1", Record({"shape": "apple"}))],),
    "build_existing_ids_query": (["r1"],),
}

READS = {
    "build_read_query",
    "build_exists_query",
    "build_search_query",
    "build_complex_search_query",
    "build_count_query",
    "build_where_clause",
}


@pytest.mark.parametrize("name", sorted(WRITES))
def test_every_write_is_refused_under_a_read_only_layout(name: str) -> None:
    with pytest.raises(OperationError, match="read-only layout") as caught:
        getattr(_builder(), name)(*WRITES[name])
    assert caught.value.context == {"table": "shapes", "operation": name}


def test_every_public_builder_method_is_a_scoped_read_or_a_refused_write() -> None:
    """A method added later must be classified, so it cannot skip the scope or the refusal."""
    public = {
        name
        for name, _ in inspect.getmembers(SQLQueryBuilder, inspect.isfunction)
        if name.startswith("build_")
    }
    assert public == READS | set(WRITES)
    for name in WRITES:
        assert getattr(getattr(SQLQueryBuilder, name), "writes", False), name
    for name in READS:
        assert not getattr(getattr(SQLQueryBuilder, name), "writes", False), name


def test_the_json_layout_still_writes() -> None:
    builder = SQLQueryBuilder("records", dialect="postgres")
    assert isinstance(builder.layout, JsonbLayout)
    sql, _ = builder.build_delete_query("r1")
    assert sql == 'DELETE FROM "records" WHERE id = $1'


# -- the scope, and numbering ------------------------------------------------------

SCOPE = [Filter("shape", Operator.EQ, "apple"), Filter("size", Operator.GT, 1)]


def _placeholders(sql: str, style: str) -> list[str]:
    pattern = r"\$\d+" if style == "numeric" else r"%\(p\d+\)s"
    return re.findall(pattern, sql)


def _expected_placeholders(count: int, style: str, start: int = 1) -> list[str]:
    if style == "numeric":
        return [f"${n}" for n in range(start, start + count)]
    return [f"%(p{n - 1})s" for n in range(start, start + count)]


@pytest.mark.parametrize("style", ["numeric", "pyformat"])
def test_the_scope_is_numbered_first_and_every_placeholder_after_it(style: str) -> None:
    """A zero-bind clause and a dropped member ahead of bound ones shift nothing."""
    builder = _builder(style=style, layout=_layout(SCOPE))
    filters = [
        Filter("size", Operator.EQ, "3"),  # FALSE: binds nothing
        Filter("shape", Operator.IN, ["x", 5]),  # 5 is dropped
        Filter("size", Operator.BETWEEN, [1, 9]),
        Filter("made", Operator.GT, datetime(2024, 1, 1)),
    ]
    sql, params = builder.build_search_query(Query(filters=filters))
    assert params == ["apple", 1, ["x"], 1, 9, datetime(2024, 1, 1)]
    assert _placeholders(sql, style) == _expected_placeholders(len(params), style)
    assert sql.index('"shape" = ') < sql.index("FALSE")

    complex_query = ComplexQuery(
        condition=LogicCondition(
            LogicOperator.OR,
            [FilterCondition(f) for f in filters],
        )
    )
    sql, params = builder.build_complex_search_query(complex_query)
    assert params[:2] == ["apple", 1]
    assert _placeholders(sql, style) == _expected_placeholders(len(params), style)
    assert re.search(r"WHERE .* AND \(\(", sql), "the condition sits under the scope"

    where, params = builder.build_where_clause(Query(filters=filters), param_start=3)
    assert where.startswith(" AND ")
    assert _placeholders(where, style) == _expected_placeholders(len(params), style, start=3)

    for build in (builder.build_read_query, builder.build_exists_query):
        sql, params = build("12345678-1234-5678-1234-567812345678")
        assert params == ["apple", 1, "12345678-1234-5678-1234-567812345678"]
        assert _placeholders(sql, style) == _expected_placeholders(3, style)


def test_a_scoped_query_with_no_filters_still_carries_the_scope() -> None:
    builder = _builder(layout=_layout(SCOPE))
    where, params = builder.build_where_clause(None)
    assert where == ' AND "shape" = $1 AND "size" > CAST($2 AS bigint)'
    assert params == ["apple", 1]
    sql, _ = builder.build_count_query(None)
    assert sql == 'SELECT COUNT(*) FROM "shapes" WHERE "shape" = $1 AND "size" > CAST($2 AS bigint)'
    sql, _ = builder.build_complex_search_query(ComplexQuery(condition=None))
    assert sql.endswith('WHERE "shape" = $1 AND "size" > CAST($2 AS bigint)')


def test_a_placeholder_is_typed_by_its_bounds() -> None:
    def where(dialect: str, style: str, spec: Filter) -> tuple[str, list[Any]]:
        sql, params = _builder(dialect, style).build_search_query(Query(filters=[spec]))
        return sql.split("WHERE ")[1], params

    assert where("postgres", "numeric", Filter("size", Operator.IN, [3, 5])) == (
        '"size" = ANY(CAST($1 AS bigint[]))',
        [[3, 5]],
    )
    assert where("postgres", "numeric", Filter("size", Operator.EQ, 2**70)) == (
        '"size" = CAST($1 AS numeric)',
        [2**70],
    )
    assert where("sqlite", "qmark", Filter("size", Operator.GTE, 3)) == ('"size" >= ?', [3])
    assert where("duckdb", "qmark", Filter("size", Operator.GTE, 3)) == ('"size" >= ?', [3])
    assert where(
        "duckdb", "qmark", Filter("k", Operator.IN, ["12345678-1234-5678-1234-567812345678"])
    ) == ('"k" IN (CAST(? AS UUID))', ["12345678-1234-5678-1234-567812345678"])


def test_an_integer_column_is_sent_a_number_it_compares_with_exactly() -> None:
    """A float bound would be compared by rounding the column past 2**53.

    A whole one is sent as its ``int``; a fractional one as the value halfway
    between the integers it lies between, which compares with each of them as
    the bound does and which no engine rounds onto one.
    """

    def where(dialect: str, style: str, spec: Filter) -> tuple[str, list[Any]]:
        sql, params = _builder(dialect, style).build_search_query(Query(filters=[spec]))
        return sql.split("WHERE ")[1], params

    assert where("postgres", "numeric", Filter("size", Operator.EQ, float(2**60))) == (
        '"size" = CAST($1 AS bigint)',
        [2**60],
    )
    assert where("postgres", "numeric", Filter("size", Operator.GTE, 3.25)) == (
        '"size" >= CAST($1 AS numeric)',
        [Decimal("3.5")],
    )
    assert where("postgres", "numeric", Filter("size", Operator.IN, [2**53, 0.5])) == (
        '"size" = ANY(CAST($1 AS numeric[]))',
        [[2**53, Decimal("0.5")]],
    )
    assert where("duckdb", "qmark", Filter("size", Operator.LT, -2.75)) == (
        '"size" < CAST(? AS DECIMAL(38,1))',
        [-2.5],
    )
    assert where("sqlite", "qmark", Filter("size", Operator.LT, -2.75)) == ('"size" < ?', [-2.5])
    assert where("postgres", "numeric", Filter("size", Operator.LT, float("inf"))) == (
        '"size" < CAST($1 AS double precision)',
        [float("inf")],
    )
    # A float column holds what a float bound names: it is sent as it is.
    weighed = NativeColumnLayout(
        DatabaseSchema.from_dict({"fields": {"k": "string", "weight": "float"}}), id_column="k"
    )
    sql, params = _builder(layout=weighed).build_search_query(
        Query(filters=[Filter("weight", Operator.GTE, 3.25)])
    )
    assert sql.endswith('WHERE "weight" >= CAST($1 AS double precision)')
    assert params == [3.25]


def test_a_zoned_column_takes_a_date_as_its_midnight_in_utc() -> None:
    schema = DatabaseSchema.from_dict(
        {"fields": {"k": "string", "seen": {"type": "datetime", "sql_type": "timestamptz"}}},
        keys=NATIVE_FIELD_KEYS,
    )
    builder = _builder(layout=NativeColumnLayout(schema, id_column="k"))
    sql, params = builder.build_search_query(
        Query(filters=[Filter("seen", Operator.GTE, datetime(2024, 6, 2).date())])
    )
    assert sql.endswith('WHERE "seen" >= $1')
    assert params == [datetime(2024, 6, 2, tzinfo=UTC)]


# -- read_layout_config --------------------------------------------------------------


def test_read_layout_config_reads_each_layout() -> None:
    assert isinstance(read_layout_config({}, None), JsonbLayout)
    assert isinstance(read_layout_config({"layout": "jsonb"}, None), JsonbLayout)
    layout = read_layout_config(
        {
            "layout": "native",
            "id_column": "k",
            "scope": [{"field": "shape", "operator": "=", "value": "apple"}],
        },
        SCHEMA,
    )
    assert isinstance(layout, NativeColumnLayout)
    assert layout.scope == (Filter("shape", Operator.EQ, "apple"),)


@pytest.mark.parametrize(
    ("config", "schema", "message", "context"),
    [
        (
            {"layout": "rows"},
            None,
            r"`layout:` is one of \['jsonb', 'native'\]",
            {"layout": "rows"},
        ),
        (
            {"id_column": "k"},
            None,
            "`id_column:` is read only with `layout: native`",
            {"key": "id_column"},
        ),
        ({"scope": []}, None, "`scope:` is read only with `layout: native`", {"key": "scope"}),
        ({}, SCHEMA, "declare `sql_type`, which only `layout: native` reads", {"columns": ["k"]}),
        ({"layout": "native"}, SCHEMA, "needs `id_column:`", {"id_column": None}),
        (
            {"layout": "native", "id_column": "k", "scope": "shape = apple"},
            SCHEMA,
            "is a list of filters",
            {"got": "str"},
        ),
        (
            {"layout": "native", "id_column": "k", "scope": [{"field": "shape"}]},
            SCHEMA,
            "is not a filter",
            {"entry": {"field": "shape"}},
        ),
        ({"layout": "native", "id_column": "k"}, None, "needs a `schema:`", {}),
        (
            {"layout": "native", "id_column": "colour"},
            SCHEMA,
            "id_column 'colour' is not a declared column",
            {"id_column": "colour"},
        ),
    ],
)
def test_read_layout_config_refuses_by_name(
    config: dict[str, Any], schema: Any, message: str, context: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match=message) as caught:
        read_layout_config(config, schema, origin="database 'shapes'", context={"source_id": "s"})
    assert str(caught.value).startswith("database 'shapes': ")
    assert caught.value.context["source_id"] == "s"
    for key, value in context.items():
        assert caught.value.context[key] == value


# -- the schema reader ----------------------------------------------------------------


def test_sql_type_reads_as_a_shorthand_and_an_explicit_entry_wins() -> None:
    schema = DatabaseSchema.from_dict(
        {
            "fields": [
                {"name": "k", "type": "string", "sql_type": "uuid"},
                {
                    "name": "j",
                    "type": "string",
                    "sql_type": "uuid",
                    "metadata": {"sql_type": "inet"},
                },
            ]
        },
        keys=NATIVE_FIELD_KEYS,
    )
    assert schema.fields["k"].metadata[SQL_TYPE_KEY] == "uuid"
    assert schema.fields["j"].metadata[SQL_TYPE_KEY] == "inet"


def test_the_default_door_refuses_the_sql_type_shorthand() -> None:
    """Only a native layout reads ``sql_type``, so a door that is not one refuses it.

    Loaded, it would read as a column type a memory, file or JSON-layout store
    honours, and none does.
    """
    declared = {"fields": {"k": {"type": "string", "sql_type": "uuid"}}}
    with pytest.raises(ValidationError, match=r"declares \['sql_type'\]"):
        DatabaseSchema.from_dict(declared)
    with pytest.raises(ValidationError, match=r"declares \['sql_type'\]"):
        extract_schema_from_config(declared)
    schema = DatabaseSchema.from_dict(declared, keys=NATIVE_FIELD_KEYS)
    assert schema.fields["k"].metadata[SQL_TYPE_KEY] == "uuid"


@pytest.mark.parametrize(
    ("declaration", "spelled"),
    [
        ({"type": "string", "sql_type": 5}, "`sql_type: 5`"),
        ({"type": "string", "sql_type": ""}, "`sql_type: ''`"),
        ({"type": "string", "metadata": {"sql_type": ["uuid"]}}, "`metadata.sql_type: ['uuid']`"),
    ],
)
def test_a_sql_type_that_is_not_a_name_is_refused(
    declaration: dict[str, Any], spelled: str
) -> None:
    with pytest.raises(ValidationError, match=re.escape(spelled)):
        DatabaseSchema.from_dict({"fields": {"k": declaration}}, keys=NATIVE_FIELD_KEYS)
