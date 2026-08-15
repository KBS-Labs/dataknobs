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
| integer narrower than 64-bit (`int8`…`int32`, `Int8`…`Int32`) | `integer` |
| 64-bit integer (`int64`, nullable `Int64`) | `bigint` |
| 32-bit float (`float32`, nullable `Float32`) | `real` |
| 64-bit float (`float64`, nullable `Float64`) | `double precision` |
| any `datetime64`, any resolution | `timestamp` |
| any `datetime64` with a timezone | `timestamptz` |
| any `timedelta64` | `interval` |
| anything else (`object`, `string`, `category`, …) | `varchar(n)` |

The mapping is decided by `pandas.api.types` predicates, so nullable extension dtypes land on the same type as their numpy counterparts. **Numeric width comes from the dtype, not from the family**: `integer` is 4-byte and `real` carries about 7 significant digits, while pandas defaults to 64 bits for both — so an ordinary `int64` column would overflow an `integer`, and an ordinary `float64` would be silently rounded into a `real`. A column that genuinely fits the narrow type keeps it.

`n` for a `varchar` column is the width of the widest **rendered** value: a column the ladder types as `varchar` is written with `str(value)`, so the width that has to fit is the string form, measured with that same `str`. A column with no values, or one whose values are all the empty string, yields `varchar(1)`: PostgreSQL will not accept a `varchar(0)` declaration. Nulls are excluded, and arrive as SQL `NULL`, which occupies no width.

### Values sent by `upload()`

A column with a real SQL type is written as a **typed parameter** through psycopg2's own adaptation — an integer as an integer, a timestamp as a `datetime`, a timedelta as a `timedelta`. Only the `varchar` fallback goes through `str()`, because that branch is the ladder's catch-all for dtypes with no SQL type of their own, and psycopg2 cannot adapt an arbitrary object.

Two consequences worth knowing:

- **Nulls become SQL `NULL`.** `NaN`, `None`, `pd.NA` and `NaT` all arrive as `NULL` rather than as the text `'nan'`/`'<NA>'`, in typed and `varchar` columns alike. You do not need to drop or fill nulls before uploading.
- **Each column keeps its own dtype.** Values are gathered per column rather than per row, so a nullable `Int64` is not upcast to float by the presence of a null or of a float column beside it.

### Column labels

Every column label must be a non-empty string, since it becomes a SQL identifier. A DataFrame built without column names carries pandas' default *integer* labels (`pd.DataFrame([[1, 2]])` has labels `0` and `1`), and `upload()` refuses it with a message naming each offending position and its type. Set them first:

```python
df.columns = ["first", "second"]
```

The generated schema is a convenience for scratch and analysis tables. Create the table yourself when you need precise types, constraints, or indexes.

### Connections

A `PostgresDB` holds **one connection per thread**, opened on first use and reused across `query()`, `execute()` and `upload()`. Close it when you are done — the class is also a context manager:

```python
from dataknobs_utils.sql_utils import PostgresDB

with PostgresDB(host="localhost", db="analytics") as db:
    df = db.query("SELECT * FROM events LIMIT 10")
```

`get_conn()` returns the calling thread's connection; do not close it yourself — call `PostgresDB.close()`, which is idempotent and closes every connection the object opened, on any thread. A connection closed underneath the object (a dropped server-side connection, say) is replaced on the next call rather than handed back closed.

Reuse is per thread rather than per object because psycopg2's `with connection:` transaction block is not re-entrant: two threads entering it on one connection raise `ProgrammingError: the connection cannot be re-entered recursively`, and beneath that they would share a single transaction, where either thread's commit commits the other's uncommitted work. psycopg2's threadsafety level 2 permits passing a connection between threads; it does not make a transaction block shareable. Sharing one `PostgresDB` across a thread pool is therefore safe, and costs one connection per worker.

A connector passed in explicitly belongs to its caller and is left open by `PostgresDB.close()`, so two objects can share one connector without either closing it out from under the other:

```python
connector = DotenvPostgresConnector(host="localhost", db="analytics")
db = PostgresDB(connector)   # db.close() will NOT close `connector`
```

Note that psycopg2's `with connection:` block commits or rolls back a **transaction** — it does not close the connection. That is why closing is explicit here.

### `PostgresRecordFetcher`

Fetches rows by ID, with `fields_to_retrieve` naming the **columns** to return — the same meaning the parameter has on the other `RecordFetcher` implementations, which use it for pandas column selection. Every identifier is quoted, so mixed-case and reserved-word column names work, and a value that is not a column name is rejected rather than becoming part of the statement. `ids` entries must be numbers.
