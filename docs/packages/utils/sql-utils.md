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
| signed integer narrower than 64-bit (`int8`…`int32`, `Int8`…`Int32`) | `integer` |
| unsigned integer narrower than 32-bit (`uint8`, `uint16`, `UInt8`, `UInt16`) | `integer` |
| 64-bit signed integer (`int64`, nullable `Int64`) | `bigint` |
| 32-bit unsigned integer (`uint32`, nullable `UInt32`) | `bigint` |
| 64-bit unsigned integer (`uint64`, nullable `UInt64`) | `numeric(20)` |
| 32-bit float (`float32`, nullable `Float32`) | `real` |
| 64-bit float (`float64`, nullable `Float64`) | `double precision` |
| any `datetime64`, any resolution | `timestamp` |
| any `datetime64` with a timezone | `timestamptz` |
| any `timedelta64` | `interval` |
| anything else (`object`, `string`, `category`, …) | `varchar(n)` |

The mapping is decided by `pandas.api.types` predicates, so nullable extension dtypes land on the same type as their numpy counterparts.

**The SQL type comes from the dtype's own range, not from its family.** `integer` is 4-byte and `real` carries about 7 significant digits, while pandas defaults to 64 bits for both — so an ordinary `int64` column would overflow an `integer`, and an ordinary `float64` would be silently rounded into a `real`. A column that genuinely fits the narrow type keeps it.

For integers the range depends on signedness as well as width, because **PostgreSQL has no unsigned integer types**. `integer` is a *signed* int4 stopping at 2³¹−1, so a `uint32` — which runs to 2³²−1 — needs a `bigint` even though both are four bytes wide. `uint64` overflows `bigint` for the same reason and has no wider integer type to be promoted into, so it becomes `numeric(20)`, which holds all 20 digits of 2⁶⁴−1.

`n` for a `varchar` column is the width of the widest **rendered** value: a column the ladder types as `varchar` is written with `str(value)`, so the width that has to fit is the string form, measured with that same `str`. A column with no values, or one whose values are all the empty string, yields `varchar(1)`: PostgreSQL will not accept a `varchar(0)` declaration. Nulls are excluded, and arrive as SQL `NULL`, which occupies no width.

### Values sent by `upload()`

A column with a real SQL type is written as a **typed parameter** through psycopg2's own adaptation — an integer as an integer, a timestamp as a `datetime`, a timedelta as a `timedelta`. Only the `varchar` fallback goes through `str()`, because that branch is the ladder's catch-all for dtypes with no SQL type of their own, and psycopg2 cannot adapt an arbitrary object.

Two consequences worth knowing:

- **Nulls become SQL `NULL`.** `NaN`, `None`, `pd.NA` and `NaT` all arrive as `NULL` rather than as the text `'nan'`/`'<NA>'`, in typed and `varchar` columns alike. You do not need to drop or fill nulls before uploading.
- **Each column keeps its own dtype.** Values are gathered per column rather than per row, so a nullable `Int64` is not upcast to float by the presence of a null or of a float column beside it.
- **A cell holding a container is text like any other `varchar` value.** An `object` column of lists or arrays is rendered with `str()` and measured with the same `str()`, so it round-trips as its Python repr.
- **Nanosecond timestamps are truncated to microseconds.** PostgreSQL's `timestamp` and `interval` have no unit finer than a microsecond, so a `datetime64[ns]` column carrying sub-microsecond data emits pandas' `Discarding nonzero nanoseconds in conversion` warning per affected cell. Round the column first if the warning is noise to you.

### Column labels

Every column label must be a non-empty string, since it becomes a SQL identifier. A DataFrame built without column names carries pandas' default *integer* labels (`pd.DataFrame([[1, 2]])` has labels `0` and `1`), and `upload()` refuses it with a message naming each offending position and its type. Set them first:

```python
df.columns = ["first", "second"]
```

Labels must also be **unique**. Two columns of one name have no distinguishable destination in the INSERT, so `upload()` refuses the frame up front and names the repeated label with its positions, rather than failing later inside the value conversion.

The generated schema is a convenience for scratch and analysis tables. Create the table yourself when you need precise types, constraints, or indexes.

### Connections

A `PostgresDB` holds **one connection per thread**, opened on first use and reused across `query()`, `execute()` and `upload()`. Close it when you are done — the class is also a context manager:

```python
from dataknobs_utils.sql_utils import PostgresDB

with PostgresDB(host="localhost", db="analytics") as db:
    df = db.query("SELECT * FROM events LIMIT 10")
```

`get_conn()` returns the calling thread's connection; do not close it yourself — call `PostgresDB.close()`, which is idempotent and closes every connection the object opened, on any thread.

A connection closed underneath the object is replaced on the next call rather than handed back closed. That covers a server-side drop — a `pg_terminate_backend`, an idle timeout, a pooler eviction — as well as a caller closing what it was handed. It has to be checked explicitly, because psycopg2's `connection.closed` reports only what *this process* did to the connection: a backend killed by the server leaves it reading 0 until something tries to use the connection and fails. The check reads the socket instead, costs no round trip, and is why a `PostgresDB` left idle overnight still works in the morning.

A thread that exits releases its connection without waiting for `close()`, so a pool that cycles workers does not accumulate backends.

Reuse is per thread rather than per object because psycopg2's `with connection:` transaction block is not re-entrant: two threads entering it on one connection raise `ProgrammingError: the connection cannot be re-entered recursively`, and beneath that they would share a single transaction, where either thread's commit commits the other's uncommitted work. psycopg2's threadsafety level 2 permits passing a connection between threads; it does not make a transaction block shareable. Sharing one `PostgresDB` across a thread pool is therefore safe, and costs one connection per worker.

Two consequences of reuse to know about:

- **Do not hold `get_conn()` in a `with` block across a `query()`, `execute()` or `upload()` call on the same thread.** Those enter `with conn` themselves, and the transaction block is not re-entrant, so the inner entry raises `ProgrammingError: the connection cannot be re-entered recursively`. Each acquisition used to return a private connection, which made the combination safe; it is not any more. Use `get_conn()` on its own, or the wrapper methods on their own.
- **`close()` is a shutdown operation, not a barrier.** It does not wait for other threads to become idle; a thread closed out from under mid-statement sees `InterfaceError: connection already closed`. Join your workers before closing.

A connector passed in explicitly belongs to its caller and is left open by `PostgresDB.close()`, so two objects can share one connector without either closing it out from under the other:

```python
connector = DotenvPostgresConnector(host="localhost", db="analytics")
db = PostgresDB(connector)   # db.close() will NOT close `connector`
```

The context-manager form follows the same rule, so `with PostgresDB(connector) as db:` closes nothing on exit — the connector's owner is still responsible for it.

Note that psycopg2's `with connection:` block commits or rolls back a **transaction** — it does not close the connection. That is why closing is explicit here.

### `PostgresRecordFetcher`

Fetches rows by ID, with `fields_to_retrieve` naming the **columns** to return — the same meaning the parameter has on the other `RecordFetcher` implementations, which use it for pandas column selection. Every identifier is quoted, so mixed-case and reserved-word column names work, and a value that is not a column name is rejected rather than becoming part of the statement.

`ids` entries must be finite numbers, and are sent as bound parameters. An empty `ids` returns an empty frame without a round trip.
