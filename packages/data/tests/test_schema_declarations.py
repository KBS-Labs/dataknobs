"""A schema declaration is read, or it is refused by name.

A ``schema:`` reaches ``DatabaseSchema`` through three doors -- a database
config (every backend's ``_normalize_dict``), ``DatabaseSchema.from_dict``
directly, and an ontology binding's ``schema:`` -- and they share one reader of
field declarations, in either spelling:

- a mapping, ``{fields: {<column>: <type> | {type: ..., ...}}}``;
- a list of rows, ``{fields: [{name: <column>, type: <type>}, ...]}``, or the
  bare list, which is what an ontology binding writes.

Every case below used to load as *no schema*, or to fail with an exception that
named nothing. A declaration that silently declares nothing is the worst of
those, because every check keyed on it then passes over nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common.exceptions import ConfigurationError, ValidationError
from dataknobs_common.ontology import OntologyConfig

from dataknobs_data.backends.config import MemoryDatabaseConfig, PostgresDatabaseConfig
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.fields import FieldType
from dataknobs_data.ontology import OntologyRegistry
from dataknobs_data.database import AsyncDatabase, SyncDatabase
from dataknobs_data.schema import DatabaseSchema, FieldSchema, read_field_declarations

ROWS: list[dict[str, Any]] = [
    {"name": "sku", "type": "string"},
    {"name": "price", "type": "float", "required": True},
]

MEMORY_DOORS = [SyncMemoryDatabase, AsyncMemoryDatabase]


def _schema_via(door: type, schema: Any) -> DatabaseSchema:
    """The schema a memory backend built from a ``schema:`` config value."""
    return door({"schema": schema}).schema


def _assert_rows_read(schema: DatabaseSchema) -> None:
    assert set(schema.fields) == {"sku", "price"}
    assert schema.fields["sku"].type is FieldType.STRING
    assert schema.fields["price"].type is FieldType.FLOAT
    assert schema.fields["price"].required is True


# --------------------------------------------------------------------------
# The two spellings, read the same way
# --------------------------------------------------------------------------


def test_fields_may_be_a_list_of_rows() -> None:
    """The ontology's spelling under `fields:` used to raise `AttributeError`."""
    _assert_rows_read(DatabaseSchema.from_dict({"fields": ROWS}))
    config = MemoryDatabaseConfig.from_dict({"schema": {"fields": ROWS}})
    assert config.schema is not None
    _assert_rows_read(config.schema)


@pytest.mark.parametrize("door", MEMORY_DOORS)
def test_a_bare_list_of_rows_is_a_schema(door: type) -> None:
    """What an ontology binding writes, written in a database config, used to be dropped."""
    _assert_rows_read(_schema_via(door, ROWS))


def test_both_spellings_build_the_same_schema() -> None:
    as_rows = DatabaseSchema.from_dict({"fields": ROWS})
    as_mapping = DatabaseSchema.from_dict(
        {"fields": {"sku": "string", "price": {"type": "float", "required": True}}}
    )
    assert as_rows == as_mapping


def test_a_field_with_no_type_is_a_string_in_both_spellings() -> None:
    """The ontology reader's rule, which the mapping spelling used to `KeyError` on."""
    from_mapping = DatabaseSchema.from_dict({"fields": {"sku": {"required": True}}})
    from_rows = DatabaseSchema.from_dict({"fields": [{"name": "sku", "required": True}]})
    for schema in (from_mapping, from_rows):
        assert schema.fields["sku"].type is FieldType.STRING
        assert schema.fields["sku"].required is True


def test_to_dict_round_trips_through_from_dict() -> None:
    """What `to_dict` writes -- the mapping spelling, `name` repeated -- is read back."""
    schema = DatabaseSchema(metadata={"owner": "catalog"})
    schema.add_field(FieldSchema(name="content", type=FieldType.TEXT, required=True))
    schema.add_field(FieldSchema(name="score", type=FieldType.FLOAT, default=0.0))
    schema.add_vector_field("embedding", dimensions=384, source_field="content")
    assert DatabaseSchema.from_dict(schema.to_dict()) == schema


# --------------------------------------------------------------------------
# What used to declare nothing, and is refused by name
# --------------------------------------------------------------------------


@pytest.mark.parametrize("door", MEMORY_DOORS)
def test_columns_at_the_top_level_are_refused_pointing_at_fields(door: type) -> None:
    """`{name: string}` used to load as a schema with no fields at all."""
    with pytest.raises(ValidationError) as excinfo:
        _schema_via(door, {"name": "string", "size": "integer"})
    message = str(excinfo.value)
    assert "`fields:`" in message
    assert "'name'" in message and "'size'" in message


@pytest.mark.parametrize("door", MEMORY_DOORS)
@pytest.mark.parametrize("value", ["text", 5, True])
def test_a_scalar_schema_is_refused(door: type, value: Any) -> None:
    """A scalar used to load as *no schema*, which every check then passed over."""
    with pytest.raises(ValidationError) as excinfo:
        _schema_via(door, value)
    assert type(value).__name__ in str(excinfo.value)


def test_a_field_that_is_neither_a_type_nor_a_mapping_is_refused_naming_it() -> None:
    """`{fields: {x: 5}}` used to drop `x` and keep the rest."""
    with pytest.raises(ValidationError) as excinfo:
        DatabaseSchema.from_dict({"fields": {"sku": "string", "price": 5}})
    assert "'price'" in str(excinfo.value)


def test_an_unknown_type_is_a_validation_error_naming_the_field_and_the_types() -> None:
    """It used to be a bare `ValueError` naming neither."""
    with pytest.raises(ValidationError) as excinfo:
        DatabaseSchema.from_dict({"fields": {"price": "money"}})
    message = str(excinfo.value)
    assert "'price'" in message and "'money'" in message
    assert all(member.value in message for member in FieldType)


def test_a_repeated_name_is_refused() -> None:
    """The last row used to win without a word."""
    with pytest.raises(ValidationError) as excinfo:
        DatabaseSchema.from_dict(
            {"fields": [{"name": "sku", "type": "string"}, {"name": "sku", "type": "integer"}]}
        )
    assert "'sku'" in str(excinfo.value)


@pytest.mark.parametrize("name", ["", 5, None])
def test_a_row_names_its_field_with_a_non_empty_string(name: Any) -> None:
    with pytest.raises(ValidationError):
        DatabaseSchema.from_dict({"fields": [{"name": name, "type": "string"}]})


def test_a_row_with_no_name_is_refused() -> None:
    with pytest.raises(ValidationError):
        DatabaseSchema.from_dict({"fields": [{"type": "string"}]})


def test_a_mapping_entry_whose_name_disagrees_with_its_key_is_refused() -> None:
    with pytest.raises(ValidationError) as excinfo:
        DatabaseSchema.from_dict({"fields": {"sku": {"name": "id", "type": "string"}}})
    assert "'sku'" in str(excinfo.value) and "'id'" in str(excinfo.value)


@pytest.mark.parametrize(
    "fields",
    [
        {"price": {"type": "float", "requird": True}},
        [{"name": "price", "type": "float", "requird": True}],
    ],
)
def test_an_unknown_field_key_is_refused_listing_the_known_ones(fields: Any) -> None:
    """A misspelt key used to be ignored, so `requird: true` declared nothing."""
    with pytest.raises(ValidationError) as excinfo:
        DatabaseSchema.from_dict({"fields": fields})
    message = str(excinfo.value)
    assert "'requird'" in message
    assert "'required'" in message


@pytest.mark.parametrize("required", ["no", 1, 0.0])
def test_required_is_a_boolean(required: Any) -> None:
    """`required: "no"` is truthy, so the value that means *off* turned it on.

    `null` is not here: an explicit `null` reads as the key left out, which for
    `required` is `false` (see `test_an_explicit_null_reads_as_absent`).
    """
    with pytest.raises(ValidationError):
        DatabaseSchema.from_dict({"fields": {"price": {"type": "float", "required": required}}})


def test_fields_that_are_neither_a_mapping_nor_a_list_are_refused() -> None:
    with pytest.raises(ValidationError):
        DatabaseSchema.from_dict({"fields": "sku"})


@pytest.mark.parametrize("key", [5, None, ""])
def test_a_mapping_key_names_its_field_with_a_non_empty_string(key: Any) -> None:
    """A key used to be `str()`-ed before its name was checked, so `{5: integer}`
    declared a field called `"5"` and `{~: string}` one called `"None"` -- the
    names the row spelling refuses.
    """
    with pytest.raises(ValidationError) as excinfo:
        DatabaseSchema.from_dict({"fields": {key: "string"}})
    assert repr(key) in str(excinfo.value)


@pytest.mark.parametrize(
    ("declaration", "field", "expected"),
    [
        ({"fields": {"sku": {"type": None}}}, "type", FieldType.STRING),
        ({"fields": [{"name": "sku", "type": None}]}, "type", FieldType.STRING),
        ({"fields": {"sku": {"type": "string", "required": None}}}, "required", False),
        ({"fields": {"sku": {"type": "string", "metadata": None}}}, "metadata", {}),
        ({"fields": {"sku": {"name": None, "type": "string"}}}, "name", "sku"),
    ],
)
def test_an_explicit_null_reads_as_absent(
    declaration: dict[str, Any], field: str, expected: Any
) -> None:
    """YAML's `type:` with no value is `null`. It used to be refused as the type
    `None`, while leaving the key out read as `string` -- and before this reader,
    the ontology door read that `null` as `string` too.
    """
    schema = DatabaseSchema.from_dict(declaration)
    assert getattr(schema.fields["sku"], field) == expected


def test_a_null_fields_or_metadata_is_an_empty_one() -> None:
    schema = DatabaseSchema.from_dict({"fields": None, "metadata": None})
    assert schema.fields == {}
    assert schema.metadata == {}


def test_a_misspelt_key_is_refused_even_when_null() -> None:
    """Only a key a field takes reads `null` as absent; `requird:` is still a typo."""
    with pytest.raises(ValidationError) as excinfo:
        DatabaseSchema.from_dict({"fields": {"sku": {"type": "string", "requird": None}}})
    assert "'requird'" in str(excinfo.value)


@pytest.mark.parametrize("fields", [{"sku": None}, [None]])
def test_a_null_field_is_still_refused(fields: Any) -> None:
    """`null` stands in for a *key* left out. A field declared as nothing is not
    a field left out; it is a declaration with nothing in it.
    """
    with pytest.raises(ValidationError):
        DatabaseSchema.from_dict({"fields": fields})


@pytest.mark.parametrize(
    "declaration",
    [{"metadata": 5}, {"fields": {"sku": {"type": "string", "metadata": "pii"}}}],
)
def test_metadata_is_a_mapping_at_either_level(declaration: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DatabaseSchema.from_dict(declaration)


def test_a_field_type_member_is_read_as_its_type_in_both_spellings() -> None:
    """What a Python caller hands over is a `FieldType`, not its string."""
    as_mapping = DatabaseSchema.from_dict({"fields": {"sku": FieldType.INTEGER}})
    as_rows = DatabaseSchema.from_dict({"fields": [{"name": "sku", "type": FieldType.INTEGER}]})
    assert as_mapping.fields["sku"].type is FieldType.INTEGER
    assert as_rows == as_mapping


def test_a_key_set_the_reader_cannot_read_is_a_programming_error() -> None:
    """A door may narrow the keys a field takes, never widen them: a key the
    reader does not read would load and be discarded.
    """
    with pytest.raises(ValueError, match="'description'"):
        read_field_declarations({"sku": "string"}, keys=frozenset({"name", "type", "description"}))


def test_an_empty_declaration_is_still_an_empty_schema() -> None:
    """The refusals are about declarations that say something; `{}` says nothing."""
    assert DatabaseSchema.from_dict({}).fields == {}
    assert DatabaseSchema.from_dict({"metadata": {"owner": "catalog"}}).metadata == {
        "owner": "catalog"
    }


# --------------------------------------------------------------------------
# `enum`, a type name in any case, and a caller that says where it read from
# --------------------------------------------------------------------------


def test_enum_folds_into_metadata_in_both_spellings() -> None:
    """`DatabaseSource.get_schema()` reads `metadata["enum"]`; `enum:` is its shorthand."""
    as_mapping = DatabaseSchema.from_dict(
        {"fields": {"dept": {"type": "string", "enum": ["CS", "Math"]}}}
    )
    as_rows = DatabaseSchema.from_dict(
        {"fields": [{"name": "dept", "type": "string", "enum": ["CS", "Math"]}]}
    )
    assert as_mapping.fields["dept"].metadata == {"enum": ["CS", "Math"]}
    assert as_rows == as_mapping


def test_an_explicit_metadata_enum_wins_over_the_shorthand() -> None:
    """The same precedence `dimensions` and `source_field` already have."""
    schema = DatabaseSchema.from_dict(
        {"fields": {"dept": {"enum": ["CS"], "metadata": {"enum": ["Math"]}}}}
    )
    assert schema.fields["dept"].metadata["enum"] == ["Math"]


@pytest.mark.parametrize("enum", ["CS", 5, {"CS": 1}])
def test_an_enum_that_is_not_a_list_is_refused_naming_the_field(enum: Any) -> None:
    """A string would reach the filter schema as `list("CS")`, one letter a value."""
    with pytest.raises(ValidationError) as excinfo:
        DatabaseSchema.from_dict({"fields": {"dept": {"enum": enum}}})
    assert "'dept'" in str(excinfo.value)


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("String", FieldType.STRING),
        ("TEXT", FieldType.TEXT),
        ("Sparse_Vector", FieldType.SPARSE_VECTOR),
    ],
)
def test_a_type_name_is_read_in_any_case(declared: str, expected: FieldType) -> None:
    """Every type's name is its member name lowercased, so `String` can only mean `string`."""
    as_mapping = DatabaseSchema.from_dict({"fields": {"sku": declared}})
    as_rows = DatabaseSchema.from_dict({"fields": [{"name": "sku", "type": declared}]})
    assert as_mapping.fields["sku"].type is expected
    assert as_rows == as_mapping


@pytest.mark.parametrize(
    "declaration",
    [
        {"columns": {"sku": "string"}},
        {"metadata": 5},
        {"fields": {"sku": "money"}},
        {"fields": [{"type": "string"}]},
    ],
)
def test_from_dict_names_its_origin_in_every_refusal(declaration: dict[str, Any]) -> None:
    """A caller that knows where the declaration came from says so once, and
    every refusal -- the top-level ones included -- carries it.
    """
    with pytest.raises(ValidationError) as excinfo:
        DatabaseSchema.from_dict(
            declaration, origin="source 'courses'", context={"source": "courses"}
        )
    assert str(excinfo.value).startswith("source 'courses': ")
    assert excinfo.value.context["source"] == "courses"


# --------------------------------------------------------------------------
# The ontology door reads through the same reader
# --------------------------------------------------------------------------


def _ontology(schema: Any) -> OntologyConfig:
    return OntologyConfig(
        id="catalog",
        entity_types=[{"id": "Product"}],
        sources=[
            {
                "id": "products",
                "kind": "record",
                "entity_projection": {
                    "table": "products",
                    "id": "sku",
                    "name": "title",
                    "type": {"const": "Product"},
                },
                "schema": schema,
            }
        ],
    )


async def _load(schema: Any) -> None:
    registry = OntologyRegistry.from_components(
        config=_ontology(schema), database=AsyncMemoryDatabase()
    )
    try:
        await registry.load()
    finally:
        await registry.close()


async def test_an_ontology_row_with_a_misspelt_key_is_refused_naming_the_binding() -> None:
    """`typ: integer` used to load as a string column, the typo unread."""
    with pytest.raises(ValidationError) as excinfo:
        await _load([{"name": "sku", "typ": "integer"}, {"name": "title"}])
    message = str(excinfo.value)
    assert "'products'" in message
    assert "'typ'" in message


@pytest.mark.parametrize(
    "extra",
    [
        {"required": True},
        {"default": "x"},
        {"metadata": {"sql_type": "uuid"}},
        {"dimensions": 3},
        {"source_field": "title"},
        {"enum": ["a", "b"]},
        {"description": "the product code"},
    ],
)
async def test_an_ontology_row_takes_only_what_the_registry_reads(extra: dict[str, Any]) -> None:
    """A binding uses a declaration's column names and types and nothing else.

    A key it would load and then discard reads as a key it honours --
    `required: true` there would refuse no record -- so it is refused naming
    the binding, as an unknown key always was.
    """
    with pytest.raises(ValidationError) as excinfo:
        await _load([{"name": "sku", "type": "string", **extra}, {"name": "title"}])
    message = str(excinfo.value)
    assert "'products'" in message
    assert repr(next(iter(extra))) in message
    assert "['name', 'type']" in message


async def test_an_ontology_row_with_a_null_type_is_a_string_column() -> None:
    """The ontology reader read `type:` with no value as `string` before it
    shared a reader, and still does.
    """
    await _load([{"name": "sku", "type": None}, {"name": "title"}])


async def test_an_ontology_unknown_type_still_names_the_binding() -> None:
    with pytest.raises(ValidationError) as excinfo:
        await _load([{"name": "sku", "type": "money"}, {"name": "title"}])
    message = str(excinfo.value)
    assert "'products'" in message and "'money'" in message


async def test_an_ontology_schema_is_still_a_list() -> None:
    """The ontology's published form is rows; the mapping spelling is not admitted there."""
    with pytest.raises(ValidationError) as excinfo:
        await _load({"fields": {"sku": "string", "title": "string"}})
    assert "'products'" in str(excinfo.value)


# --------------------------------------------------------------------------
# `DatabaseSchema.create` reads its options through the same reader
# --------------------------------------------------------------------------


def test_create_refuses_an_option_a_field_does_not_take() -> None:
    """A misspelt `requird` used to be ignored here, as it was in a config."""
    with pytest.raises(ValidationError) as excinfo:
        DatabaseSchema.create(price=(FieldType.FLOAT, {"requird": True}))
    assert "'price'" in str(excinfo.value) and "'requird'" in str(excinfo.value)


def test_create_refuses_a_required_that_is_not_a_boolean() -> None:
    with pytest.raises(ValidationError):
        DatabaseSchema.create(price=(FieldType.FLOAT, {"required": "no"}))


def test_create_refuses_a_type_among_its_options() -> None:
    """The tuple's first element is the type; a second one in the options was ignored."""
    with pytest.raises(ValidationError) as excinfo:
        DatabaseSchema.create(price=(FieldType.FLOAT, {"type": "string"}))
    assert "'price'" in str(excinfo.value)


def test_create_and_from_dict_agree_on_which_dimensions_wins() -> None:
    """`create` let the shorthand beat an explicit `metadata.dimensions`, and the
    reader lets the explicit one win: one declaration, two schemas.
    """
    created = DatabaseSchema.create(
        embedding=(FieldType.VECTOR, {"dimensions": 384, "metadata": {"dimensions": 768}})
    )
    read = DatabaseSchema.from_dict(
        {
            "fields": {
                "embedding": {
                    "type": "vector",
                    "dimensions": 384,
                    "metadata": {"dimensions": 768},
                }
            }
        }
    )
    assert created == read
    assert created.fields["embedding"].get_dimensions() == 768


def test_create_leaves_the_callers_metadata_alone() -> None:
    """The shorthands used to be written into the caller's own `metadata` dict."""
    metadata = {"owner": "catalog"}
    DatabaseSchema.create(embedding=(FieldType.VECTOR, {"metadata": metadata, "dimensions": 384}))
    assert metadata == {"owner": "catalog"}


def test_create_still_reads_what_it_documents() -> None:
    schema = DatabaseSchema.create(
        content=FieldType.TEXT,
        embedding=(FieldType.VECTOR, {"dimensions": 384, "source_field": "content"}),
        score=(FieldType.FLOAT, {"required": True, "default": 0.0}),
    )
    assert schema.fields["content"].type is FieldType.TEXT
    assert schema.fields["embedding"].get_dimensions() == 384
    assert schema.fields["embedding"].get_source_field() == "content"
    assert schema.fields["score"].required is True
    assert schema.fields["score"].default == 0.0


@pytest.mark.parametrize("definition", ["text", (FieldType.TEXT,), 5])
def test_create_still_raises_value_error_for_a_definition_it_cannot_read(
    definition: Any,
) -> None:
    """A Python call's shape is `create`'s own check, and its type is unchanged."""
    with pytest.raises(ValueError):
        DatabaseSchema.create(content=definition)


# --------------------------------------------------------------------------
# A backend a consumer writes, constructed the legacy way
# --------------------------------------------------------------------------


def _unused(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError("construction only")


def _legacy_backend(base: type) -> type:
    """A backend subclassing the base directly, not `StructuredConfigConsumer`:
    the only kind that still reaches the base's dict-construction branch.
    """
    return type(f"Legacy{base.__name__}", (base,), dict.fromkeys(base.__abstractmethods__, _unused))


@pytest.mark.parametrize("base", [SyncDatabase, AsyncDatabase])
def test_a_legacy_backend_reads_its_schema_through_the_same_reader(base: type) -> None:
    backend = _legacy_backend(base)
    _assert_rows_read(backend({"schema": ROWS}).schema)
    with pytest.raises(ValidationError):
        backend({"schema": {"sku": "string"}})


# --------------------------------------------------------------------------
# The Postgres overload is untouched
# --------------------------------------------------------------------------


def test_postgres_still_reads_a_scalar_schema_as_its_namespace() -> None:
    """The scalar refused on every other backend is the SQL schema name here."""
    assert PostgresDatabaseConfig.from_dict({"schema": "reporting"}).schema_name == "reporting"


def test_postgres_still_refuses_a_structural_schema_by_configuration() -> None:
    """Pinned, so that admitting one is a decision a change makes on purpose."""
    with pytest.raises(ConfigurationError) as excinfo:
        PostgresDatabaseConfig.from_dict({"schema": {"fields": {"sku": "string"}}})
    assert "string identifier" in str(excinfo.value)
