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
| any `datetime64`, any resolution | `timestamp` |
| any `datetime64` with a timezone | `timestamptz` |
| any `timedelta64` | `interval` |
| anything else (`object`, `string`, `category`, …) | `varchar(n)` |

The mapping is decided by `pandas.api.types` predicates, so nullable extension dtypes land on the same type as their numpy counterparts.

`n` for a `varchar` column is the width of the widest **rendered** value — `upload()` sends `str(value)` for each cell, so the width that has to fit is the string form, measured with that same `str`. A column with no values, or one whose values are all the empty string, yields `varchar(1)`: PostgreSQL will not accept a `varchar(0)` declaration. Nulls are excluded from the measurement — see below for why widening for them would not help.

The table is what `CREATE TABLE` emits. Four things to know before relying on it, because in each the value `upload()` then sends does not arrive intact in the column it just made — three are rejected outright, one is accepted and rounded:

- **`integer` is PostgreSQL's 4-byte `integer`, not `bigint`.** A DataFrame `int64` column holding values beyond ±2<sup>31</sup> creates a column its own data does not fit. The INSERT fails.
- **`real` is 4-byte `float4`, about 7 significant digits.** `float64` is pandas' default float dtype, and a value carrying more precision than that is silently rounded on the way in — `1.2345678901234567` reads back as `1.2345679`. This one does not fail; it loses data quietly.
- **A `timedelta64[ns]` column does not upload.** `upload()` renders it with `str()`, which follows the column's own resolution and produces `'86400000000000 nanoseconds'`; PostgreSQL's `interval` has no unit finer than a microsecond and rejects that literal. Every coarser resolution loads — `'86400 seconds'`, `'86400000 milliseconds'` and `'86400000000 microseconds'` are all accepted, and microsecond is what `pd.to_timedelta` produces on pandas 3.x. Cast with `.astype("timedelta64[us]")` if your data is nanosecond-resolution.
- **Null handling is the caller's.** `upload()` renders every cell with `str()`, which turns `NaN`/`None`/`pd.NA` into the text `'nan'`/`'<NA>'`. A typed column rejects those at any width, so drop or fill nulls before uploading. A nullable extension column is affected even in its non-null cells: `to_records()` upcasts `Int64` to float when a null is present, so `1` is sent as `'1.0'` into the `integer` column the ladder chose for it.

The generated schema is a convenience for scratch and analysis tables. Create the table yourself when you need precise types, constraints, or indexes.
