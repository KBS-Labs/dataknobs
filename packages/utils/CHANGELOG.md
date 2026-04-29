# Changelog

All notable changes to the dataknobs-utils package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Security

- **`PostgresDB.upload()` and `_create_table` now quote DataFrame column names** with `quote_ident()`. Column names were previously joined raw into INSERT and CREATE TABLE statements, allowing columns named with spaces, reserved words, or special characters to produce invalid SQL. `psql_schema_line` has been extracted as `PostgresDB._psql_schema_line(df, col)` (a `@staticmethod`) and `_build_insert_columns(columns)` has been added as a `@staticmethod` to make the SQL-building logic directly testable. The `isinstance(dtype, np.dtype)` guard replaces the broader `hasattr(dtype, "type")` check, fixing a pre-existing crash with pandas `StringDtype` columns. The float-subtype check is broadened from `np.float64` to `np.floating` so `float32` and other numpy float subtypes map to `real` instead of falling through to `varchar`. Pandas nullable numeric types (`Float32Dtype`, `Float64Dtype`, `Int64Dtype`, etc.) are now detected via `pd.api.types.is_float_dtype` / `is_integer_dtype` rather than silently producing `varchar`. `str.len().max(skipna=True) or 1` replaces `max(str.len())` to handle empty DataFrames without raising `ValueError`.

### Added

- `quote_ident(name, dialect="postgres")` in `dataknobs_utils.sql_utils`: production-grade SQL identifier quoting returning double-quoted identifiers (`"name"` with internal `"` escaped as `""`). Supports `postgres`, `sqlite`, and `duckdb` dialects (all use the same SQL-standard rule). Applied internally to `table_head()`, `upload()`, and `_create_table()` in `PostgresDB`-derived classes. Now raises `ValueError` for unsupported dialects. Removed dead `psycopg2` delegation that silently fell through to the pure-Python rule on every call.
