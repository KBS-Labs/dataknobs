"""Tests for PostgresTableManager static helpers."""

from dataknobs_data.backends.postgres_mixins import PostgresTableManager


def test_get_table_exists_sql_returns_parameterized():
    sql, params = PostgresTableManager.get_table_exists_sql("public", "records")
    assert "$1" in sql and "$2" in sql
    assert "'" not in sql  # no string literal interpolation of schema/table names
    assert params == ("public", "records")


def test_get_create_table_sql_quotes_identifiers():
    sql = PostgresTableManager.get_create_table_sql("public", "MyTable")
    assert '"MyTable"' in sql
    assert '"public"' in sql
    # Index names are also quoted
    assert '"idx_MyTable_data"' in sql or '"idx_MyTable_metadata"' in sql
