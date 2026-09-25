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
from dataknobs_data.schema import DatabaseSchema, FieldSchema

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


@pytest.mark.parametrize("required", ["no", 1, None])
def test_required_is_a_boolean(required: Any) -> None:
    """`required: "no"` is truthy, so the value that means *off* turned it on."""
    with pytest.raises(ValidationError):
        DatabaseSchema.from_dict({"fields": {"price": {"type": "float", "required": required}}})


def test_fields_that_are_neither_a_mapping_nor_a_list_are_refused() -> None:
    with pytest.raises(ValidationError):
        DatabaseSchema.from_dict({"fields": "sku"})


def test_an_empty_declaration_is_still_an_empty_schema() -> None:
    """The refusals are about declarations that say something; `{}` says nothing."""
    assert DatabaseSchema.from_dict({}).fields == {}
    assert DatabaseSchema.from_dict({"metadata": {"owner": "catalog"}}).metadata == {
        "owner": "catalog"
    }


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


async def test_an_ontology_row_may_carry_what_a_database_field_carries() -> None:
    """One declaration: a row takes the keys a database config's field takes."""
    await _load(
        [
            {"name": "sku", "type": "string", "required": True, "metadata": {"sql_type": "uuid"}},
            {"name": "title", "type": "string"},
        ]
    )


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
