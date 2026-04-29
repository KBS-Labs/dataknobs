"""Reproduce-first tests for unquoted DataFrame column names in sql_utils.py (A6).

PostgresDB.upload() builds an INSERT statement using raw df.columns — no quoting.
A column named "My Column" produces "INSERT INTO t (My Column) VALUES ..." which
is invalid SQL.  _create_table -> psql_schema_line has the same gap.

Fix: apply quote_ident() to each column name.  The extract of _psql_schema_line and
_build_insert_columns as @staticmethods makes the SQL-building logic unit-testable
without a live postgres connection.
"""
import numpy as np
import pandas as pd
import pytest

from dataknobs_utils.sql_utils import PostgresDB


class TestPsqlSchemaLineColumnQuoting:
    """_psql_schema_line must produce quoted column identifiers."""

    def test_space_in_column_quoted(self):
        """Bug: 'My Column integer' is invalid SQL — column name needs double-quoting."""
        df = pd.DataFrame({"My Column": pd.array([1], dtype="int64")})
        line = PostgresDB._psql_schema_line(df, "My Column")
        assert '"My Column"' in line
        assert "integer" in line

    def test_reserved_word_column_quoted(self):
        df = pd.DataFrame({"user": ["alice"]})
        line = PostgresDB._psql_schema_line(df, "user")
        assert '"user"' in line

    def test_plain_column_still_quoted(self):
        df = pd.DataFrame({"name": ["a"]})
        line = PostgresDB._psql_schema_line(df, "name")
        assert line == '"name" varchar(1)'

    def test_integer_dtype(self):
        df = pd.DataFrame({"count": pd.array([1, 2, 3], dtype="int64")})
        line = PostgresDB._psql_schema_line(df, "count")
        assert line == '"count" integer'

    def test_float_dtype(self):
        df = pd.DataFrame({"score": [1.0, 2.0]})
        line = PostgresDB._psql_schema_line(df, "score")
        assert line == '"score" real'

    def test_float32_dtype(self):
        df = pd.DataFrame({"score": np.array([1.0, 2.0], dtype=np.float32)})
        line = PostgresDB._psql_schema_line(df, "score")
        assert line == '"score" real'

    def test_nullable_float_dtype(self):
        df = pd.DataFrame({"score": pd.array([1.0, None], dtype="Float64")})
        line = PostgresDB._psql_schema_line(df, "score")
        assert line == '"score" real'

    def test_nullable_integer_dtype(self):
        df = pd.DataFrame({"count": pd.array([1, None], dtype="Int64")})
        line = PostgresDB._psql_schema_line(df, "count")
        assert line == '"count" integer'

    def test_empty_dataframe_varchar(self):
        df = pd.DataFrame({"tag": pd.Series([], dtype="string")})
        line = PostgresDB._psql_schema_line(df, "tag")
        assert '"tag" varchar(1)' == line

    def test_empty_dataframe_numpy_varchar(self):
        df = pd.DataFrame({"tag": np.array([], dtype=object)})
        line = PostgresDB._psql_schema_line(df, "tag")
        assert '"tag" varchar(1)' == line


class TestBuildInsertColumns:
    """_build_insert_columns must return a properly-quoted comma-separated column list."""

    def test_space_in_column_name(self):
        result = PostgresDB._build_insert_columns(["My Column", "age"])
        assert result == '"My Column", "age"'

    def test_reserved_word(self):
        result = PostgresDB._build_insert_columns(["user", "name"])
        assert result == '"user", "name"'

    def test_plain_identifiers(self):
        result = PostgresDB._build_insert_columns(["id", "value"])
        assert result == '"id", "value"'

    def test_single_column(self):
        result = PostgresDB._build_insert_columns(["data"])
        assert result == '"data"'
