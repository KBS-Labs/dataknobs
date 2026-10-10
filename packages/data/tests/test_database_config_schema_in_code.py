"""A backend config built in code reads its ``schema`` as one read from a mapping does.

``DatabaseConfig`` documents that a mapping or a list of field rows is
converted to a ``DatabaseSchema`` at construction and any other value is
refused. Only ``from_dict`` did that: a config built by calling the class kept
whatever it was given, so ``schema`` held a ``dict`` -- or ``42`` -- and a
backend reading the declared fields at construction failed with
``AttributeError: 'dict' object has no attribute 'fields'``.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from typing import Any

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_data.backends import config as config_module
from dataknobs_data.backends.config import (
    AsyncDuckDBDatabaseConfig,
    AsyncSQLiteDatabaseConfig,
    DatabaseConfig,
    FileDatabaseConfig,
    MemoryDatabaseConfig,
    PostgresDatabaseConfig,
    SyncDuckDBDatabaseConfig,
    SyncSQLiteDatabaseConfig,
)
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.file import AsyncFileDatabase, SyncFileDatabase
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.backends.postgres import AsyncPostgresDatabase, SyncPostgresDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.database import AsyncDatabase, SyncDatabase
from dataknobs_data.schema import DatabaseSchema

SCHEMA = {"fields": {"k": "string"}}

#: Every ``DatabaseConfig`` the module defines, the base included.
CONFIGS = sorted(
    (
        cls
        for cls in vars(config_module).values()
        if inspect.isclass(cls)
        and issubclass(cls, DatabaseConfig)
        and cls.__module__ == config_module.__name__
    ),
    key=lambda cls: cls.__name__,
)


def _required(cls: type) -> dict[str, Any]:
    """The one value a config refuses to default: an S3 bucket."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {"bucket": "a-bucket"} if "bucket" in names else {}


def _name(cls: type) -> str:
    return cls.__name__


@pytest.mark.parametrize("cls", CONFIGS, ids=_name)
@pytest.mark.parametrize(
    "schema", [SCHEMA, [{"name": "k", "type": "string"}]], ids=["mapping", "rows"]
)
def test_a_schema_given_in_code_is_read_into_a_database_schema(cls: type, schema: Any) -> None:
    config = cls(schema=schema, **_required(cls))
    assert isinstance(config.schema, DatabaseSchema)
    assert list(config.schema.fields) == ["k"]
    # The schema, not the whole config: ``from_dict`` also resolves a Postgres
    # connection from ``POSTGRES_*`` / ``DATABASE_URL``, which a config built
    # in code deliberately does not.
    assert config.schema == cls.from_dict({"schema": schema, **_required(cls)}).schema


@pytest.mark.parametrize("cls", CONFIGS, ids=_name)
@pytest.mark.parametrize("value", [42, "fields"])
def test_a_schema_that_declares_nothing_is_refused_in_code(cls: type, value: Any) -> None:
    with pytest.raises(ValidationError):
        cls(schema=value, **_required(cls))


NATIVE_CONFIGS = [
    SyncSQLiteDatabaseConfig,
    AsyncSQLiteDatabaseConfig,
    SyncDuckDBDatabaseConfig,
    AsyncDuckDBDatabaseConfig,
    PostgresDatabaseConfig,
]


@pytest.mark.parametrize("cls", NATIVE_CONFIGS, ids=_name)
def test_a_native_column_type_is_read_in_code_as_from_a_mapping(cls: type) -> None:
    native = {"fields": {"k": {"type": "string", "sql_type": "uuid"}}}
    # A file-backed store names the file the table is in; nothing opens it here.
    where = (
        {"path": "owned-elsewhere.db"}
        if "path" in {f.name for f in dataclasses.fields(cls)}
        else {}
    )
    config = cls(table="t", layout="native", id_column="k", schema=native, **where)
    assert config.schema.fields["k"].metadata == {"sql_type": "uuid"}
    with pytest.raises(ValidationError, match="sql_type"):
        cls(schema=native)


@pytest.mark.parametrize(
    ("db_cls", "cfg_cls", "where"),
    [
        (SyncSQLiteDatabase, SyncSQLiteDatabaseConfig, ":memory:"),
        (AsyncSQLiteDatabase, AsyncSQLiteDatabaseConfig, ":memory:"),
        (SyncDuckDBDatabase, SyncDuckDBDatabaseConfig, ":memory:"),
        (AsyncDuckDBDatabase, AsyncDuckDBDatabaseConfig, ":memory:"),
        (SyncPostgresDatabase, PostgresDatabaseConfig, None),
        (AsyncPostgresDatabase, PostgresDatabaseConfig, None),
        (SyncMemoryDatabase, MemoryDatabaseConfig, None),
        (AsyncMemoryDatabase, MemoryDatabaseConfig, None),
        (SyncFileDatabase, FileDatabaseConfig, "file"),
        (AsyncFileDatabase, FileDatabaseConfig, "file"),
    ],
    ids=lambda v: (
        v.__name__ if isinstance(v, type) and issubclass(v, (SyncDatabase, AsyncDatabase)) else ""
    ),
)
def test_a_backend_given_a_config_built_in_code_holds_its_declared_fields(
    db_cls: type, cfg_cls: type, where: str | None, tmp_path: Path
) -> None:
    """Bug: SQLite, DuckDB and Postgres raised ``AttributeError`` at construction."""
    location = (
        {} if where is None else {"path": where if where != "file" else str(tmp_path / "db.json")}
    )
    db = db_cls(cfg_cls(schema=SCHEMA, **location))
    assert isinstance(db.schema, DatabaseSchema)
    assert list(db.schema.fields) == ["k"]


def test_a_postgres_namespace_given_as_schema_in_code_is_refused_naming_schema_name() -> None:
    """A string ``schema`` is the SQL namespace when a mapping is read, and is
    routed to ``schema_name`` there. Built in code it is refused, naming the
    field that takes it, rather than with the generic refusal of a schema that
    declares nothing.
    """
    assert PostgresDatabaseConfig.from_dict({"schema": "reporting"}).schema_name == "reporting"
    with pytest.raises(ValidationError, match="schema_name"):
        PostgresDatabaseConfig(schema="reporting")
