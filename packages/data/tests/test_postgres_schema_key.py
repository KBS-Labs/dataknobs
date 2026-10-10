"""What ``schema`` means to the Postgres backend, decided by the value's type.

Every backend reads ``schema`` as the declared fields. Postgres also reads it
as its SQL namespace, so the value says which it is: a string is the namespace
(``schema_name``), and a mapping, a list or a ``DatabaseSchema`` is the
declared fields, read as every other backend reads them.

Two defects came from the overload. A config holding declared fields could not
be read back from its own ``to_dict``: the dict it wrote was taken for a
namespace and refused. And a namespace in the configuration mapping was lost
when the declared fields came as a keyword, because the two were merged under
one key before the config could tell them apart.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dataknobs_common.exceptions import ConfigurationError, ValidationError
from dataknobs_data.backends.config import PostgresDatabaseConfig
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase
from dataknobs_data.fields import FieldType
from dataknobs_data.query import Filter, Operator
from dataknobs_data.schema import DatabaseSchema

TWINS = [AsyncPostgresDatabase, SyncPostgresDatabase]

FIELDS = DatabaseSchema.create(name=FieldType.STRING, size=FieldType.INTEGER)


def _names(schema: DatabaseSchema | None) -> list[str]:
    assert schema is not None
    return sorted(schema.fields)


def test_a_config_holding_declared_fields_reads_back_from_its_own_dict() -> None:
    config = PostgresDatabaseConfig(schema=FIELDS, schema_name="reporting", table="t")
    again = PostgresDatabaseConfig.from_dict(config.to_dict())

    assert _names(again.schema) == ["name", "size"]
    assert again.schema_name == "reporting"
    assert again == config


def test_a_native_config_reads_back_from_its_own_dict() -> None:
    config = PostgresDatabaseConfig.from_dict(
        {
            "layout": "native",
            "id_column": "name",
            "scope": [
                Filter("size", Operator.GT, 1),
                {"field": "name", "operator": "!=", "value": "x"},
            ],
            "schema": {"fields": {"name": "string", "size": "integer"}},
        }
    )
    assert PostgresDatabaseConfig.from_dict(config.to_dict()) == config
    as_json = json.loads(json.dumps(config.to_json_dict()))  # what a config file holds
    again = PostgresDatabaseConfig.from_dict(as_json)

    assert again == config
    assert again.layout == "native" and again.id_column == "name"


@pytest.mark.parametrize(
    "declared",
    [
        {"fields": {"name": "string", "size": "integer"}},
        [{"name": "name", "type": "string"}, {"name": "size", "type": "integer"}],
    ],
)
def test_declared_fields_come_from_configuration(declared: Any) -> None:
    """The form every other backend reads, which Postgres refused as a namespace."""
    for cls in TWINS:
        db = cls({"schema": declared})
        assert _names(db.schema) == ["name", "size"]
        assert db.schema_name == "public"


def test_a_string_schema_is_still_the_namespace() -> None:
    for cls in TWINS:
        assert cls({"schema": "reporting"}).schema_name == "reporting"


def test_a_namespace_and_declared_fields_given_two_ways_are_both_kept() -> None:
    """The namespace in the mapping and the fields as a keyword, and the other way round."""
    for cls in TWINS:
        db = cls({"schema": "reporting"}, schema=FIELDS)
        assert db.schema_name == "reporting"
        assert _names(db.schema) == ["name", "size"]

        db = cls({"schema": {"fields": {"name": "string", "size": "integer"}}}, schema="reporting")
        assert db.schema_name == "reporting"
        assert _names(db.schema) == ["name", "size"]


def test_a_keyword_still_wins_over_the_same_meaning_in_the_mapping() -> None:
    """Merging is unchanged where both sides say the same kind of thing."""
    for cls in TWINS:
        assert cls({"schema": "a"}, schema="b").schema_name == "b"
        assert cls({"table": "a"}, table="b").table_name == "b"


def test_a_value_that_is_neither_is_refused() -> None:
    with pytest.raises((ConfigurationError, ValidationError)):
        PostgresDatabaseConfig.from_dict({"schema": 5})


# --- native mode creates nothing ---------------------------------------------------


def test_native_mode_creates_neither_table_nor_database_by_default() -> None:
    config = PostgresDatabaseConfig.from_dict(
        {"layout": "native", "id_column": "name", "schema": {"fields": {"name": "string"}}}
    )
    assert config.auto_create_table is False
    assert config.ensure_database is False


@pytest.mark.parametrize("key", ["auto_create_table", "ensure_database", "vector_enabled"])
def test_native_mode_refuses_to_be_told_to_create_anything(key: str) -> None:
    with pytest.raises(ConfigurationError, match=key):
        PostgresDatabaseConfig.from_dict(
            {
                "layout": "native",
                "id_column": "name",
                "schema": {"fields": {"name": "string"}},
                key: True,
            }
        )


def test_the_json_layout_still_creates_its_table_by_default() -> None:
    config = PostgresDatabaseConfig.from_dict({})
    assert config.auto_create_table is True
    assert config.ensure_database is True
