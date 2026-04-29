"""Unit tests for SQLTableManager.get_table_exists_sql() and coerce_bool."""

import pytest

from dataknobs_data.backends.sql_base import SQLTableManager


class TestGetTableExistsSql:
    def test_postgres_dialect(self):
        mgr = SQLTableManager("records", schema_name="public", dialect="postgres", param_style="numeric")
        sql, params = mgr.get_table_exists_sql()
        assert "information_schema.tables" in sql
        assert params == ("public", "records")

    def test_postgres_default_schema(self):
        mgr = SQLTableManager("records", dialect="postgres", param_style="numeric")
        _sql, params = mgr.get_table_exists_sql()
        assert params == ("public", "records")

    def test_postgres_qmark_raises(self):
        """qmark placeholders are invalid for postgres — caught at construction time."""
        with pytest.raises(ValueError, match="param_style='qmark' is not valid for dialect='postgres'"):
            SQLTableManager("records", dialect="postgres")

    def test_postgres_qmark_raises_explicit(self):
        with pytest.raises(ValueError, match="param_style='qmark'"):
            SQLTableManager("records", dialect="postgres", param_style="qmark")

    def test_sqlite_dialect(self):
        mgr = SQLTableManager("records", dialect="sqlite")
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
        _sql, params = mgr.get_table_exists_sql()
        assert params == ("myschema", "records")

    def test_returns_parameterized_not_interpolated(self):
        """Table name must NOT be embedded directly in the SQL string."""
        mgr = SQLTableManager("my_records", dialect="sqlite")
        sql, params = mgr.get_table_exists_sql()
        assert "my_records" not in sql
        assert "my_records" in params

    # R1-04: placeholder-style assertions for all dialects

    def test_postgres_numeric_placeholder(self):
        mgr = SQLTableManager(
            "records", schema_name="public", dialect="postgres", param_style="numeric"
        )
        sql, params = mgr.get_table_exists_sql()
        assert "$1" in sql
        assert "$2" in sql
        assert "?" not in sql
        assert params == ("public", "records")

    def test_postgres_pyformat_placeholder(self):
        mgr = SQLTableManager(
            "records", schema_name="public", dialect="postgres", param_style="pyformat"
        )
        sql, params = mgr.get_table_exists_sql()
        assert "%(schema)s" in sql
        assert "%(table)s" in sql
        assert isinstance(params, dict)
        assert params == {"schema": "public", "table": "records"}

    def test_duckdb_qmark_placeholder(self):
        mgr = SQLTableManager("records", dialect="duckdb")
        sql, params = mgr.get_table_exists_sql()
        assert "?" in sql
        assert "$" not in sql
        assert isinstance(params, tuple)

    def test_postgres_not_interpolated(self):
        """Schema and table must not be embedded in the SQL string for postgres."""
        mgr = SQLTableManager(
            "my_records", schema_name="my_schema", dialect="postgres", param_style="numeric"
        )
        sql, params = mgr.get_table_exists_sql()
        assert "my_records" not in sql
        assert "my_schema" not in sql
        assert "my_schema" in params
        assert "my_records" in params

    @pytest.mark.parametrize(
        "dialect,param_style,expected_in_sql,expected_param_type",
        [
            ("postgres", "numeric", "$1", tuple),
            ("postgres", "pyformat", "%(schema)s", dict),
            ("duckdb", "qmark", "?", tuple),
            ("sqlite", "qmark", "?", tuple),
        ],
    )
    def test_placeholder_style_per_dialect(
        self,
        dialect: str,
        param_style: str,
        expected_in_sql: str,
        expected_param_type: type,
    ) -> None:
        mgr = SQLTableManager(
            "records", schema_name="public", dialect=dialect, param_style=param_style
        )
        sql, params = mgr.get_table_exists_sql()
        assert expected_in_sql in sql
        assert isinstance(params, expected_param_type)


class TestCoerceBool:
    def test_true_passthrough(self):
        assert SQLTableManager.coerce_bool(True) is True

    def test_false_passthrough(self):
        assert SQLTableManager.coerce_bool(False) is False

    def test_string_false(self):
        assert SQLTableManager.coerce_bool("false") is False
        assert SQLTableManager.coerce_bool("False") is False
        assert SQLTableManager.coerce_bool("FALSE") is False

    def test_string_zero(self):
        assert SQLTableManager.coerce_bool("0") is False

    def test_string_no(self):
        assert SQLTableManager.coerce_bool("no") is False
        assert SQLTableManager.coerce_bool("NO") is False

    def test_empty_string(self):
        assert SQLTableManager.coerce_bool("") is False

    def test_string_true(self):
        assert SQLTableManager.coerce_bool("true") is True
        assert SQLTableManager.coerce_bool("True") is True
        assert SQLTableManager.coerce_bool("yes") is True
        assert SQLTableManager.coerce_bool("1") is True

    def test_int_zero(self):
        assert SQLTableManager.coerce_bool(0) is False

    def test_int_one(self):
        assert SQLTableManager.coerce_bool(1) is True

    def test_none_returns_default_true(self):
        assert SQLTableManager.coerce_bool(None) is True
        assert SQLTableManager.coerce_bool(None, default=True) is True

    def test_none_returns_default_false(self):
        assert SQLTableManager.coerce_bool(None, default=False) is False
