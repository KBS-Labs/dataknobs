"""Unit tests for SQLTableManager.get_table_exists_sql(), _coerce_bool, and
cross-backend coercion parity."""

import pytest

from dataknobs_data.backends.duckdb import SyncDuckDBDatabase
from dataknobs_data.backends.postgres import SyncPostgresDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sql_base import SQLTableManager


class TestGetTableExistsSql:
    def test_postgres_dialect(self):
        mgr = SQLTableManager("records", schema_name="public", dialect="postgres")
        sql, params = mgr.get_table_exists_sql()
        assert "information_schema.tables" in sql
        assert params == ("public", "records")

    def test_postgres_default_schema(self):
        mgr = SQLTableManager("records", dialect="postgres")
        sql, params = mgr.get_table_exists_sql()
        assert params == ("public", "records")

    def test_sqlite_dialect(self):
        mgr = SQLTableManager("records", dialect="sqlite")
        sql, params = mgr.get_table_exists_sql()
        assert "sqlite_master" in sql
        assert params == ("records",)

    def test_sqlite_ignores_schema(self):
        mgr = SQLTableManager("records", schema_name="ignored", dialect="sqlite")
        sql, params = mgr.get_table_exists_sql()
        assert "sqlite_master" in sql
        assert params == ("records",)

    def test_duckdb_dialect(self):
        mgr = SQLTableManager("records", dialect="duckdb")
        sql, params = mgr.get_table_exists_sql()
        assert "information_schema.tables" in sql
        assert params == ("main", "records")

    def test_duckdb_custom_schema(self):
        mgr = SQLTableManager("records", schema_name="myschema", dialect="duckdb")
        sql, params = mgr.get_table_exists_sql()
        assert params == ("myschema", "records")

    def test_standard_dialect_fallback(self):
        mgr = SQLTableManager("records", dialect="standard")
        sql, params = mgr.get_table_exists_sql()
        assert "information_schema.tables" in sql
        assert params == ("records",)


class TestCoerceBool:
    def test_true_bool(self):
        assert SQLTableManager._coerce_bool(True) is True

    def test_false_bool(self):
        assert SQLTableManager._coerce_bool(False) is False

    def test_string_true(self):
        assert SQLTableManager._coerce_bool("true") is True
        assert SQLTableManager._coerce_bool("True") is True
        assert SQLTableManager._coerce_bool("TRUE") is True

    def test_string_false(self):
        assert SQLTableManager._coerce_bool("false") is False
        assert SQLTableManager._coerce_bool("False") is False
        assert SQLTableManager._coerce_bool("FALSE") is False

    def test_string_zero(self):
        assert SQLTableManager._coerce_bool("0") is False

    def test_string_no(self):
        assert SQLTableManager._coerce_bool("no") is False
        assert SQLTableManager._coerce_bool("NO") is False

    def test_string_empty(self):
        assert SQLTableManager._coerce_bool("") is False

    def test_string_other_truthy(self):
        assert SQLTableManager._coerce_bool("1") is True
        assert SQLTableManager._coerce_bool("yes") is True

    def test_none_returns_default_true(self):
        assert SQLTableManager._coerce_bool(None) is True

    def test_none_returns_custom_default(self):
        assert SQLTableManager._coerce_bool(None, default=False) is False

    def test_int_truthy(self):
        assert SQLTableManager._coerce_bool(1) is True

    def test_int_falsy(self):
        assert SQLTableManager._coerce_bool(0) is False


class TestAutoCreateTableCoercionParity:
    """All SQL backends must agree on the same auto_create_table coercion result.

    Regression guard: the Postgres backend previously used an allowlist
    (only "true"/"1"/"yes" → True) while SQLite and DuckDB used the shared
    denylist (everything except "false"/"0"/"no"/"" → True). This test
    ensures the three backends stay in sync.
    """

    @pytest.mark.parametrize("raw,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("", False),
        (True, True),
        (False, False),
    ])
    def test_auto_create_table_coercion_parity(self, raw: str | bool, expected: bool) -> None:
        sqlite_db = SyncSQLiteDatabase({"auto_create_table": raw, "path": ":memory:"})
        duckdb_db = SyncDuckDBDatabase({"auto_create_table": raw, "path": ":memory:"})
        postgres_db = SyncPostgresDatabase({"auto_create_table": raw})
        assert sqlite_db.auto_create_table == expected, f"SQLite: {raw!r}"
        assert duckdb_db.auto_create_table == expected, f"DuckDB: {raw!r}"
        assert postgres_db.auto_create_table == expected, f"Postgres: {raw!r}"


class TestPostgresIdentifierValidation:
    """_init_postgres_attributes validates table_name and schema_name at construction.

    Malformed identifiers are rejected before any connection is made, so
    the injection surface in CREATE TABLE f-strings is closed at the boundary.
    """

    def test_valid_table_and_schema(self) -> None:
        db = SyncPostgresDatabase({"table": "records", "schema": "public"})
        assert db.table_name == "records"
        assert db.schema_name == "public"

    @pytest.mark.parametrize("bad_table", [
        "x; DROP TABLE y",
        "a-b",
        "1abc",
        "",
        "my table",
        "my.table",
    ])
    def test_invalid_table_name_raises_at_construction(self, bad_table: str) -> None:
        with pytest.raises(ValueError, match="table_name"):
            SyncPostgresDatabase({"table": bad_table})

    @pytest.mark.parametrize("bad_schema", [
        "my-schema",
        "123schema",
        "",
        "schema name",
    ])
    def test_invalid_schema_name_raises_at_construction(self, bad_schema: str) -> None:
        with pytest.raises(ValueError, match="schema_name"):
            SyncPostgresDatabase({"schema": bad_schema})
