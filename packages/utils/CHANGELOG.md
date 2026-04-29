# Changelog

All notable changes to the dataknobs-utils package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `quote_ident(name, dialect="postgres")` in `dataknobs_utils.sql_utils`: production-grade SQL identifier quoting returning double-quoted identifiers (`"name"` with internal `"` escaped as `""`). Supports `postgres`, `sqlite`, and `duckdb` dialects (all use the same SQL-standard rule). Applied internally to `table_head()`, `upload()`, and `_create_table()` in `PostgresDB`-derived classes. Now raises `ValueError` for unsupported dialects. Removed dead `psycopg2` delegation that silently fell through to the pure-Python rule on every call.
