"""Tests for SQLQueryBuilder dot-notation field path support.

Dot-notation (e.g. ``"metadata.work_order_id"``, ``"config.timeout"``) is
interpreted as nested JSON path traversal by all SQL backends.  This matches
the semantics of ``Record.get_value()``.

.. important::

   JSON keys that literally contain a dot (e.g. ``{"my.field": 1}``) cannot
   be addressed through the query interface.  This is a deliberate design
   choice documented in ``SQLQueryBuilder._build_json_field_expr``.
"""

from __future__ import annotations

import pytest

from dataknobs_data.backends.sql_base import SQLQueryBuilder
from dataknobs_data.query import Filter, Operator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _builder(dialect: str, param_style: str = "numeric") -> SQLQueryBuilder:
    """Create a query builder for the given dialect."""
    return SQLQueryBuilder(
        table_name="records",
        dialect=dialect,
        param_style=param_style,
    )


DIALECTS = ["postgres", "sqlite", "duckdb"]


# ---------------------------------------------------------------------------
# _build_json_field_expr — column routing and nested path extraction
# ---------------------------------------------------------------------------

class TestBuildJsonFieldExpr:
    """Tests for the low-level JSON field expression builder."""

    # -- Postgres ----------------------------------------------------------

    def test_postgres_simple_field(self) -> None:
        b = _builder("postgres")
        assert b._build_json_field_expr("name") == "data->>'name'"

    def test_postgres_simple_field_metadata_column(self) -> None:
        b = _builder("postgres")
        assert b._build_json_field_expr("version", column="metadata") == "metadata->>'version'"

    def test_postgres_nested_one_level(self) -> None:
        b = _builder("postgres")
        assert b._build_json_field_expr("config.timeout") == "data->'config'->>'timeout'"

    def test_postgres_nested_two_levels(self) -> None:
        b = _builder("postgres")
        expr = b._build_json_field_expr("config.retry.max_attempts")
        assert expr == "data->'config'->'retry'->>'max_attempts'"

    def test_postgres_metadata_column_nested(self) -> None:
        b = _builder("postgres")
        expr = b._build_json_field_expr("tenant.region", column="metadata")
        assert expr == "metadata->'tenant'->>'region'"

    # -- SQLite ------------------------------------------------------------

    def test_sqlite_simple_field(self) -> None:
        b = _builder("sqlite")
        assert b._build_json_field_expr("name") == "json_extract(data, '$.name')"

    def test_sqlite_nested_field(self) -> None:
        b = _builder("sqlite")
        assert b._build_json_field_expr("config.timeout") == "json_extract(data, '$.config.timeout')"

    def test_sqlite_metadata_column(self) -> None:
        b = _builder("sqlite")
        expr = b._build_json_field_expr("version", column="metadata")
        assert expr == "json_extract(metadata, '$.version')"

    # -- DuckDB ------------------------------------------------------------

    def test_duckdb_simple_field(self) -> None:
        b = _builder("duckdb")
        assert b._build_json_field_expr("name") == "json_extract_string(data, '$.name')"

    def test_duckdb_nested_field(self) -> None:
        b = _builder("duckdb")
        expr = b._build_json_field_expr("config.timeout")
        assert expr == "json_extract_string(data, '$.config.timeout')"

    def test_duckdb_metadata_column(self) -> None:
        b = _builder("duckdb")
        expr = b._build_json_field_expr("version", column="metadata")
        assert expr == "json_extract_string(metadata, '$.version')"

    def test_invalid_column_raises_value_error(self) -> None:
        b = _builder("postgres")
        with pytest.raises(ValueError, match="column must be one of"):
            b._build_json_field_expr("name", column="users")

    def test_postgres_as_text_false(self) -> None:
        """as_text=False uses -> instead of ->> for Postgres."""
        b = _builder("postgres")
        expr = b._build_json_field_expr("name", as_text=False)
        assert expr == "data->'name'"

    def test_postgres_nested_as_text_false(self) -> None:
        b = _builder("postgres")
        expr = b._build_json_field_expr("config.timeout", as_text=False)
        assert expr == "data->'config'->'timeout'"


# ---------------------------------------------------------------------------
# _build_filter_clause — metadata.* routing
# ---------------------------------------------------------------------------

class TestMetadataFieldRouting:
    """Filters prefixed with ``metadata.`` target the metadata column."""

    @pytest.mark.parametrize("dialect", DIALECTS)
    def test_metadata_field_routes_to_metadata_column(self, dialect: str) -> None:
        """metadata.work_order_id should query the metadata column, not data."""
        b = _builder(dialect, param_style="numeric" if dialect == "postgres" else "qmark")
        f = Filter("metadata.work_order_id", Operator.EQ, "WO-001")
        clause, params = b._build_filter_clause(f, 1)

        # The clause must reference the metadata column
        assert "metadata" in clause
        # Must NOT use the data column as source (check for "data->" or
        # "(data," which would indicate the data column is being queried).
        # Note: "metadata" contains "data" as a substring, so we check
        # for patterns specific to the data column being used as source.
        assert clause.startswith("metadata") or clause.startswith("json_extract")
        # Verify the actual extraction targets metadata, not data
        assert "json_extract(data," not in clause
        assert "json_extract_string(data," not in clause
        assert params == ["WO-001"]

    def test_postgres_metadata_single_key(self) -> None:
        b = _builder("postgres")
        f = Filter("metadata.tenant_id", Operator.EQ, "T-1")
        clause, _ = b._build_filter_clause(f, 1)
        assert clause == "metadata->>'tenant_id' = $1"

    def test_postgres_metadata_nested_key(self) -> None:
        b = _builder("postgres")
        f = Filter("metadata.tenant.region", Operator.EQ, "us-east")
        clause, _ = b._build_filter_clause(f, 1)
        assert clause == "metadata->'tenant'->>'region' = $1"

    def test_sqlite_metadata_key(self) -> None:
        b = _builder("sqlite", param_style="qmark")
        f = Filter("metadata.tenant_id", Operator.EQ, "T-1")
        clause, _ = b._build_filter_clause(f, 1)
        assert clause == "json_extract(metadata, '$.tenant_id') = ?"

    def test_duckdb_metadata_key(self) -> None:
        b = _builder("duckdb", param_style="qmark")
        f = Filter("metadata.tenant_id", Operator.EQ, "T-1")
        clause, _ = b._build_filter_clause(f, 1)
        assert clause == "json_extract_string(metadata, '$.tenant_id') = ?"


# ---------------------------------------------------------------------------
# _build_filter_clause — dot-notation in the data column
# ---------------------------------------------------------------------------

class TestDataFieldDotNotation:
    """Dot-notation in non-metadata fields targets the data column."""

    def test_postgres_nested_data_field(self) -> None:
        b = _builder("postgres")
        f = Filter("config.timeout", Operator.EQ, "30")
        clause, _ = b._build_filter_clause(f, 1)
        assert clause == "data->'config'->>'timeout' = $1"

    def test_postgres_deep_nested_data_field(self) -> None:
        b = _builder("postgres")
        f = Filter("config.retry.max_attempts", Operator.EQ, "3")
        clause, _ = b._build_filter_clause(f, 1)
        assert clause == "data->'config'->'retry'->>'max_attempts' = $1"

    def test_sqlite_nested_data_field(self) -> None:
        b = _builder("sqlite", param_style="qmark")
        f = Filter("config.timeout", Operator.EQ, "30")
        clause, _ = b._build_filter_clause(f, 1)
        assert clause == "json_extract(data, '$.config.timeout') = ?"

    def test_duckdb_nested_data_field(self) -> None:
        b = _builder("duckdb", param_style="qmark")
        f = Filter("config.timeout", Operator.EQ, "30")
        clause, _ = b._build_filter_clause(f, 1)
        assert clause == "json_extract_string(data, '$.config.timeout') = ?"


# ---------------------------------------------------------------------------
# Type casting with dot-notation fields
# ---------------------------------------------------------------------------

class TestTypeCastingWithDotNotation:
    """Type casts are applied correctly to dot-notation field expressions."""

    def test_postgres_metadata_numeric_gt(self) -> None:
        b = _builder("postgres")
        f = Filter("metadata.version", Operator.GT, 2)
        clause, params = b._build_filter_clause(f, 1)
        assert clause == "(metadata->>'version')::numeric > $1"
        assert params == [2]

    def test_postgres_data_nested_numeric_eq(self) -> None:
        b = _builder("postgres")
        f = Filter("config.timeout", Operator.EQ, 30)
        clause, params = b._build_filter_clause(f, 1)
        assert clause == "(data->'config'->>'timeout')::numeric = $1"
        assert params == [30]

    def test_postgres_metadata_boolean_eq(self) -> None:
        b = _builder("postgres")
        f = Filter("metadata.active", Operator.EQ, True)
        clause, _ = b._build_filter_clause(f, 1)
        assert clause == "(metadata->>'active')::boolean = $1"

    def test_duckdb_metadata_numeric(self) -> None:
        b = _builder("duckdb", param_style="qmark")
        f = Filter("metadata.version", Operator.GTE, 3)
        clause, _ = b._build_filter_clause(f, 1)
        assert clause == "CAST(json_extract_string(metadata, '$.version') AS DOUBLE) >= ?"

    def test_sqlite_no_cast_needed(self) -> None:
        """SQLite json_extract returns typed values — no CAST is applied."""
        b = _builder("sqlite", param_style="qmark")
        f = Filter("metadata.version", Operator.GT, 2)
        clause, _ = b._build_filter_clause(f, 1)
        # json_extract already returns the correct type in SQLite
        assert clause == "json_extract(metadata, '$.version') > ?"


# ---------------------------------------------------------------------------
# ID field is unaffected by dot-notation logic
# ---------------------------------------------------------------------------

class TestIdFieldUnchanged:
    """The ``id`` field still maps to the ``id`` column directly."""

    @pytest.mark.parametrize("dialect", DIALECTS)
    def test_id_field_not_json_extracted(self, dialect: str) -> None:
        ps = "numeric" if dialect == "postgres" else "qmark"
        b = _builder(dialect, param_style=ps)
        f = Filter("id", Operator.EQ, "abc-123")
        clause, params = b._build_filter_clause(f, 1)
        placeholder = "$1" if dialect == "postgres" else "?"
        assert clause == f"id = {placeholder}"
        assert params == ["abc-123"]


# ---------------------------------------------------------------------------
# Operators work correctly through the refactored path
# ---------------------------------------------------------------------------

class TestOperatorsThroughRefactoredPath:
    """All operators produce correct SQL through the refactored methods."""

    def test_in_operator_with_metadata(self) -> None:
        b = _builder("postgres")
        f = Filter("metadata.region", Operator.IN, ["us-east", "us-west"])
        clause, params = b._build_filter_clause(f, 1)
        assert clause == "metadata->>'region' IN ($1, $2)"
        assert params == ["us-east", "us-west"]

    def test_between_operator_with_nested_data(self) -> None:
        b = _builder("postgres")
        f = Filter("stats.score", Operator.BETWEEN, [80, 100])
        clause, params = b._build_filter_clause(f, 1)
        assert clause == "(data->'stats'->>'score')::numeric BETWEEN $1 AND $2"
        assert params == [80, 100]

    def test_like_operator_with_metadata(self) -> None:
        b = _builder("postgres")
        f = Filter("metadata.description", Operator.LIKE, "%test%")
        clause, params = b._build_filter_clause(f, 1)
        assert clause == "metadata->>'description' LIKE $1"
        assert params == ["%test%"]

    def test_exists_operator_with_metadata(self) -> None:
        b = _builder("postgres")
        f = Filter("metadata.optional_field", Operator.EXISTS, True)
        clause, params = b._build_filter_clause(f, 1)
        assert clause == "metadata->>'optional_field' IS NOT NULL"
        assert params == []

    def test_regex_operator_postgres(self) -> None:
        b = _builder("postgres")
        f = Filter("metadata.tag", Operator.REGEX, "^v[0-9]+")
        clause, params = b._build_filter_clause(f, 1)
        assert clause == "metadata->>'tag' ~ $1"
        assert params == ["^v[0-9]+"]


# ---------------------------------------------------------------------------
# Sort expressions with dot-notation
# ---------------------------------------------------------------------------

class TestSortExprDotNotation:
    """Sort expressions handle dot-notation and metadata.* routing."""

    def test_postgres_simple_sort(self) -> None:
        b = _builder("postgres")
        assert b._build_sort_expr("name") == "data->'name'"

    def test_postgres_metadata_sort(self) -> None:
        b = _builder("postgres")
        assert b._build_sort_expr("metadata.version") == "metadata->'version'"

    def test_postgres_nested_data_sort(self) -> None:
        b = _builder("postgres")
        # sort uses -> (preserves jsonb type) for all segments
        assert b._build_sort_expr("config.timeout") == "data->'config'->'timeout'"

    def test_postgres_id_sort(self) -> None:
        b = _builder("postgres")
        assert b._build_sort_expr("id") == "id"

    def test_sqlite_metadata_sort(self) -> None:
        b = _builder("sqlite")
        assert b._build_sort_expr("metadata.version") == "json_extract(metadata, '$.version')"

    def test_duckdb_nested_sort(self) -> None:
        b = _builder("duckdb")
        expr = b._build_sort_expr("config.timeout")
        assert expr == "json_extract_string(data, '$.config.timeout')"


# ---------------------------------------------------------------------------
# Full search query integration
# ---------------------------------------------------------------------------

class TestSearchQueryIntegration:
    """End-to-end tests for build_search_query with dot-notation filters."""

    def test_postgres_metadata_filter_in_search_query(self) -> None:
        from dataknobs_data.query import Query

        b = _builder("postgres")
        query = Query().filter("metadata.tenant_id", "=", "T-1")
        sql, params = b.build_search_query(query)

        assert "WHERE metadata->>'tenant_id' = $1" in sql
        assert params == ["T-1"]

    def test_postgres_mixed_data_and_metadata_filters(self) -> None:
        from dataknobs_data.query import Query

        b = _builder("postgres")
        query = (
            Query()
            .filter("status", "=", "active")
            .filter("metadata.tenant_id", "=", "T-1")
        )
        sql, params = b.build_search_query(query)

        assert "data->>'status' = $1" in sql
        assert "metadata->>'tenant_id' = $2" in sql
        assert params == ["active", "T-1"]

    def test_sqlite_metadata_filter_in_search_query(self) -> None:
        from dataknobs_data.query import Query

        b = _builder("sqlite", param_style="qmark")
        query = Query().filter("metadata.tenant_id", "=", "T-1")
        sql, params = b.build_search_query(query)

        assert "WHERE json_extract(metadata, '$.tenant_id') = ?" in sql
        assert params == ["T-1"]
