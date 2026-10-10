"""A SQLite or DuckDB configuration names a table somebody else's file holds.

The keys a native table is read by -- ``layout``, ``id_column``, ``scope`` and a
column's ``sql_type`` -- read as Postgres reads them. What a file-backed
store adds is what it must not do to that file: create it, change its journal,
or open it for writing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dataknobs_common.exceptions import ConfigurationError, ValidationError
from dataknobs_data.backends.config import (
    AsyncDuckDBDatabaseConfig,
    AsyncSQLiteDatabaseConfig,
    DuckDBDatabaseConfigBase,
    PostgresDatabaseConfig,
    SQLiteDatabaseConfigBase,
    SyncDuckDBDatabaseConfig,
    SyncSQLiteDatabaseConfig,
)
from dataknobs_data.query import Filter, Operator

SQLITE = [SyncSQLiteDatabaseConfig, AsyncSQLiteDatabaseConfig]
DUCKDB = [SyncDuckDBDatabaseConfig, AsyncDuckDBDatabaseConfig]
CONFIGS = SQLITE + DUCKDB


def native(**overrides: object) -> dict[str, Any]:
    config: dict[str, Any] = {
        "path": "owned-elsewhere.db",
        "table": "tickets",
        "layout": "native",
        "id_column": "id",
        "schema": {
            "fields": {
                "id": {"type": "string", "sql_type": "uuid"},
                "status": "string",
            }
        },
        "scope": [{"field": "status", "operator": "!=", "value": "closed"}],
    }
    config.update(overrides)
    return config


def _name(cls: type) -> str:
    return cls.__name__


@pytest.mark.parametrize("cls", CONFIGS, ids=_name)
def test_a_native_config_reads_back_from_its_own_dict(cls: type) -> None:
    config = cls.from_dict(native(scope=[Filter("status", Operator.NEQ, "closed")]))
    assert config.layout == "native" and config.id_column == "id"
    assert config.scope == [{"field": "status", "operator": "!=", "value": "closed"}]
    assert config.schema.fields["id"].metadata == {"sql_type": "uuid"}

    assert cls.from_dict(config.to_dict()) == config
    as_json = json.loads(json.dumps(config.to_json_dict()))  # what a config file holds
    assert cls.from_dict(as_json) == config


@pytest.mark.parametrize("cls", CONFIGS, ids=_name)
def test_a_sql_type_is_still_refused_under_the_json_layout(cls: type) -> None:
    with pytest.raises(ValidationError, match="sql_type"):
        cls.from_dict({"schema": {"fields": {"id": {"type": "string", "sql_type": "uuid"}}}})


@pytest.mark.parametrize("cls", CONFIGS, ids=_name)
def test_native_mode_creates_no_table_by_default(cls: type) -> None:
    assert cls.from_dict(native()).auto_create_table is False


@pytest.mark.parametrize("cls", CONFIGS, ids=_name)
def test_the_json_layout_still_creates_its_table_by_default(cls: type) -> None:
    assert cls.from_dict({}).auto_create_table is True
    assert cls.from_dict({"auto_create_table": "false"}).auto_create_table is False


@pytest.mark.parametrize("cls", CONFIGS, ids=_name)
@pytest.mark.parametrize("given", ["false", None, False])
def test_native_mode_told_false_or_nothing_is_not_refused(cls: type, given: Any) -> None:
    assert cls.from_dict(native(auto_create_table=given)).auto_create_table is False


@pytest.mark.parametrize("cls", CONFIGS, ids=_name)
def test_native_mode_refuses_to_be_told_to_create_its_table(cls: type) -> None:
    with pytest.raises(ConfigurationError, match="auto_create_table: true"):
        cls.from_dict(native(auto_create_table=True))


@pytest.mark.parametrize("cls", SQLITE, ids=_name)
def test_sqlite_native_mode_refuses_vectors(cls: type) -> None:
    with pytest.raises(ConfigurationError, match="vector_enabled: true"):
        cls.from_dict(native(vector_enabled=True))


@pytest.mark.parametrize("cls", CONFIGS, ids=_name)
@pytest.mark.parametrize("path", [":memory:", ""])
def test_native_mode_refuses_a_database_with_no_file(cls: type, path: str) -> None:
    """A fresh in-memory or temporary database holds no table somebody else made."""
    with pytest.raises(ConfigurationError, match="path"):
        cls.from_dict(native(path=path))


@pytest.mark.parametrize("cls", CONFIGS, ids=_name)
def test_native_mode_needs_a_path(cls: type) -> None:
    config = native()
    del config["path"]
    with pytest.raises(ConfigurationError, match="path"):
        cls.from_dict(config)


@pytest.mark.parametrize("cls", SQLITE, ids=_name)
def test_sqlite_native_mode_refuses_a_journal_mode(cls: type) -> None:
    """The journal mode is written into the file, so setting it changes the owner's file."""
    with pytest.raises(ConfigurationError, match="journal_mode"):
        cls.from_dict(native(journal_mode="WAL"))


@pytest.mark.parametrize("cls", DUCKDB, ids=_name)
def test_duckdb_native_mode_opens_read_only(cls: type) -> None:
    assert cls.from_dict(native()).read_only is True
    assert cls.from_dict(native(read_only=True)).read_only is True
    assert cls.from_dict(native(read_only="true")).read_only is True


@pytest.mark.parametrize("cls", DUCKDB, ids=_name)
@pytest.mark.parametrize("given", [False, "false"])
def test_duckdb_native_mode_refuses_to_open_for_writing(cls: type, given: Any) -> None:
    with pytest.raises(ConfigurationError, match="read_only: false"):
        cls.from_dict(native(read_only=given))


@pytest.mark.parametrize("cls", DUCKDB, ids=_name)
def test_duckdb_json_layout_read_only_is_unchanged(cls: type) -> None:
    assert cls.from_dict({}).read_only is False
    assert cls.from_dict({"read_only": "true"}).read_only is True


@pytest.mark.parametrize("base", [SQLiteDatabaseConfigBase, DuckDBDatabaseConfigBase])
def test_a_config_built_directly_resolves_the_same_way(base: type) -> None:
    """The defaults come from the layout however the config was built."""
    cls = SQLITE[0] if base is SQLiteDatabaseConfigBase else DUCKDB[0]
    built = cls(path="owned-elsewhere.db", layout="native", id_column="id")
    assert built.auto_create_table is False
    assert cls().auto_create_table is True


@pytest.mark.parametrize("cls", [*CONFIGS, PostgresDatabaseConfig], ids=_name)
@pytest.mark.parametrize("given", [{}, {"auto_create_table": "false"}, "native"])
def test_the_resolved_switches_are_read_as_bools(cls: type, given: Any) -> None:
    """A backend reads each switch as the bool the config resolved it to."""
    if given == "native":
        given = native()
        if cls is PostgresDatabaseConfig:
            del given["path"]
    config = cls.from_dict(given)
    assert config.creates_table is config.auto_create_table
    assert isinstance(config.creates_table, bool)
    if cls in DUCKDB:
        assert config.opens_read_only is config.read_only
        assert isinstance(config.opens_read_only, bool)


@pytest.mark.parametrize("cls", CONFIGS, ids=_name)
def test_scope_keys_are_refused_under_the_json_layout(cls: type) -> None:
    """``read_layout_config`` refuses them; the config only carries them."""
    from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
    from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
    from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

    backend = {
        SyncSQLiteDatabaseConfig: SyncSQLiteDatabase,
        AsyncSQLiteDatabaseConfig: AsyncSQLiteDatabase,
        SyncDuckDBDatabaseConfig: SyncDuckDBDatabase,
        AsyncDuckDBDatabaseConfig: AsyncDuckDBDatabase,
    }[cls]
    with pytest.raises(ValidationError, match="scope"):
        backend({"scope": [{"field": "status", "operator": "=", "value": "open"}]})


@pytest.mark.parametrize("cls", [*CONFIGS, PostgresDatabaseConfig], ids=_name)
@pytest.mark.parametrize("layout", ["NATIVE", "json", "columns"])
def test_a_layout_the_backend_does_not_read_is_refused_by_the_config(
    cls: type, layout: str
) -> None:
    """Bug: the config compared ``layout`` with ``"native"`` and took anything
    else for the JSON layout, so ``NATIVE`` made a config that would create
    its table, refused only once a backend was built from it.
    """
    with pytest.raises(ValidationError, match="layout"):
        cls.from_dict({"layout": layout})
    with pytest.raises(ValidationError, match="layout"):
        cls(layout=layout)


def test_the_layout_annotation_names_the_layouts_the_backend_reads() -> None:
    """The annotation carries the vocabulary, so a config can be built from it.

    ``Literal["jsonb", "native"] | None``: the layouts the backend reads, and
    ``None`` for YAML's ``null``, which reads as the default.
    """
    import typing

    from dataknobs_data.backends.column_layout import LAYOUTS
    from dataknobs_data.backends.config import ColumnLayoutConfig

    literal, none = typing.get_args(typing.get_type_hints(ColumnLayoutConfig)["layout"])
    assert typing.get_args(literal) == LAYOUTS
    assert none is type(None)
    assert SyncSQLiteDatabaseConfig.from_dict({"layout": None}).layout == "jsonb"
