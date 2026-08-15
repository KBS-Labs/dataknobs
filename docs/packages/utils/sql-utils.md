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

Consequences worth knowing:

- **Nulls become SQL `NULL`.** `NaN`, `None`, `pd.NA` and `NaT` all arrive as `NULL` rather than as the text `'nan'`/`'<NA>'`, in typed and `varchar` columns alike. You do not need to drop or fill nulls before uploading.
- **Each column keeps its own dtype.** Values are gathered per column rather than per row, so a nullable `Int64` is not upcast to float by the presence of a null or of a float column beside it.
- **A cell holding a container is text like any other `varchar` value.** An `object` column of lists or arrays is rendered with `str()` and measured with the same `str()`, so it round-trips as its Python repr.
- **Sub-microsecond precision is lost, and the two families lose it differently.** Neither `timestamp` nor `interval` has a unit finer than a microsecond. A `datetime64[ns]` column is **truncated**, and pandas says so — `Discarding nonzero nanoseconds in conversion`, once per affected cell. A `timedelta64[ns]` column is **rounded to nearest**, and says nothing at all. Round the column yourself if either matters.

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

**The transaction on a `get_conn()` connection is yours.** `query()`, `execute()` and `upload()` run on a *separate* connection, so a wrapper call cannot commit or roll back work you have left open — and equally, they will not commit it for you. A transaction belongs to a connection, and those methods each enter `with conn`, which commits on the way out; sharing one connection between you and them would mean an ordinary `query()` committing your uncommitted INSERT, with nothing nested and nothing raised. So a thread that calls `get_conn()` holds two connections: yours, and the one the wrappers share. A thread that never calls it holds one.

A connection closed underneath the object is replaced on the next call rather than handed back closed. That covers a server-side drop — a `pg_terminate_backend`, an idle timeout, a pooler eviction — as well as a caller closing what it was handed.

This needs a round trip, and there is no way around it. psycopg2's `connection.closed` reports only what *this process* did to the connection, so a backend the server killed leaves it reading 0. The socket cannot settle it either: readable means EOF, an error, *or* data, and a terminated backend sends an error message before closing — so peeking finds a byte there exactly as a pending `NOTIFY` would. A check built on readability discards healthy connections and drops their `LISTEN` subscriptions; one built on `closed` misses the case it exists for. So the connection is asked, with `SELECT 1`:

| | per call |
|---|---|
| no reuse (before) | 4.1 ms — full TCP+auth handshake |
| reuse + validation (now) | 0.29 ms |
| reuse without validation | ~0 |

What the probe asks is whether the **server answered**, not whether the statement succeeded — the two come apart in both directions. A live server refuses every statement inside an aborted transaction (`InFailedSqlTransaction`) and cancels one that trips `statement_timeout` (`QueryCanceled`); in both cases the connection is open and reusable, and the reply itself is the proof of life. Conversely a local status of "in a failed transaction" says nothing about the backend, which `idle_in_transaction_session_timeout` reaps exactly as it reaps an open one. `connection.closed` separates the two, because psycopg2 sets it when the transport failed and leaves it alone when the server merely refused — so an aborted transaction you left behind is handed back for you to unwind, and a killed one is replaced.

Validation is on by default and costs about 7% of what reuse saves. Pass `validate_on_reuse=False` to `PostgresDB` or `DotenvPostgresConnector` to trade it back where connections are known not to sit idle long enough to be dropped; the free local checks still run, so a connection libpq has already given up on is never handed back. The probe runs under `autocommit` when the connection is idle, so it never hands one back idle-in-transaction, and it joins rather than disturbs a transaction you already have open.

A thread that exits releases its connections without waiting for `close()`, so a pool that cycles workers does not accumulate backends.

Reuse is per thread rather than per object because psycopg2's `with connection:` transaction block is not re-entrant: two threads entering it on one connection raise `ProgrammingError: the connection cannot be re-entered recursively`, and beneath that they would share a single transaction, where either thread's commit commits the other's uncommitted work. psycopg2's threadsafety level 2 permits passing a connection between threads; it does not make a transaction block shareable. Each thread's connections are its own, so concurrent `query()` / `execute()` / `upload()` calls through one `PostgresDB` are safe and cost one connection per worker. (The cached table list behind `table_names` and `tables_df` is not itself synchronised; the worst case is a redundant lookup.)

Two things to know:

- **Do not nest `with` blocks on the connection you were handed.** psycopg2's transaction block is not re-entrant, so entering `with conn:` twice on the same connection raises `ProgrammingError: the connection cannot be re-entered recursively`. Interleaving is fine — a `query()`, `execute()` or `upload()` inside your own `with get_conn()` block runs on the wrappers' connection and cannot touch your transaction. That is what the separate lane buys.
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

`ids` entries must be **integers**, and are sent as bound parameters. The check is `operator.index`, not `int()`: `int()` accepts `"5"` and silently truncates `1.9` to `1`, which returns a different, wrong row rather than the empty result such a value used to produce. Numpy integers — what a DataFrame column yields — are still accepted.

An empty `ids` returns an empty frame **with the columns a populated one would have**, matching the sibling fetchers. That costs a round trip, and deliberately: the columns come from the server, and deriving them locally would mean reimplementing `SELECT *`. A zero-column frame would make `got["id"]` raise `KeyError` on a result that merely has no rows.
