# SQL Utilities

The `dataknobs_utils.sql_utils` module provides utilities for working with SQL databases, including identifier quoting and DataFrame-backed query helpers.

## `quote_ident(name, dialect="postgres")`

Returns `name` as a properly double-quoted SQL identifier.

```python
from dataknobs_utils.sql_utils import quote_ident

quote_ident("public")        # '"public"'
quote_ident("MyTable")       # '"MyTable"'
quote_ident("user")          # '"user"'  (reserved word — safe after quoting)
quote_ident('weird"name')    # '"weird""name"'
```

All three supported dialects (`postgres`, `sqlite`, `duckdb`) use the same SQL-standard rule: surround the identifier with `"`, escaping any internal `"` as `""`.

### Rules and edge cases

- Non-empty string required — raises `ValueError` for empty string or non-string input
- Does **not** split qualified names — `"schema.table"` → `'"schema.table"'` (one identifier, not two)
- Caller is responsible for splitting a qualified name and quoting each part separately
- All three dialects use the same pure-Python SQL-standard rule: wrap with `"`, escape internal `"` as `""`

### Not for test code

Use `dataknobs_common.testing.safe_sql_ident` in test fixtures (regex allowlist). Use `quote_ident` in production code where legitimate mixed-case, reserved-word, or hyphenated identifiers must be supported.

## PostgreSQL Utilities

`PostgresDB` and related classes (from the legacy `sql_utils` module) provide DataFrame-backed query helpers for use with psycopg2 connections.

### Schema inference in `upload()`

`PostgresDB.upload(table_name, df)` creates the table if it does not exist, deriving each column's SQL type from the DataFrame's dtype:

| pandas dtype | PostgreSQL type |
|---|---|
| `bool`, `BooleanDtype` | `boolean` |
| any integer, including nullable `Int8`…`Int64` | `integer` |
| any float, including nullable `Float32`/`Float64` | `real` |
| `datetime64[ns]` | `timestamp` |
| `datetime64[ns, tz]` | `timestamptz` |
| `timedelta64[ns]` | `interval` |
| anything else (`object`, `string`, `category`, …) | `varchar(n)` |

The mapping is decided by `pandas.api.types` predicates, so nullable extension dtypes land on the same type as their numpy counterparts.

`n` for a `varchar` column is the width of the widest **rendered** value — `upload()` sends `str(value)` for each cell, so the width that has to fit is the string form. Nulls are excluded from the measurement, and a column with no values yields `varchar(1)`.

Two consequences are worth knowing before relying on it:

- **`integer` is PostgreSQL's 4-byte `integer`, not `bigint`.** A DataFrame `int64` column holding values beyond ±2<sup>31</sup> creates a column its own data does not fit.
- **Null handling is the caller's.** `upload()` renders every cell with `str()`, which turns `NaN`/`None`/`pd.NA` into the text `'nan'`/`'<NA>'`. A typed column rejects those, so drop or fill nulls before uploading.

The generated schema is a convenience for scratch and analysis tables. Create the table yourself when you need precise types, constraints, or indexes.
