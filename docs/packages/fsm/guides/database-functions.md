# Database Function Library

The built-in database functions
(`dataknobs_fsm.functions.library.database`) let an FSM state read from and
write to any `dataknobs-data` `AsyncDatabase` backend without blocking the
event loop. Each function is an `ITransformFunction` you wire into a state's
`transform` slot; it acquires its database through the state's declared
`async_database` resource (injected into `FunctionContext.resources`).

## The `async_database` resource

Declare the backend as a state resource and reference it by name from the
function:

```python
config = {
    "resources": [
        {"name": "target_db", "type": "async_database",
         "config": {"type": "file", "path": "/data/out.json"}},
    ],
    "states": [
        {"name": "load", "is_start": True,
         "resources": ["target_db"],
         "functions": {"transform": {"type": "registered", "name": "load"}}},
        {"name": "done", "is_end": True},
    ],
    "arcs": [{"from": "load", "to": "done", "name": "loaded"}],
}
```

The resource wraps `AsyncDatabaseResourceAdapter`, so every operation
(`upsert`, `bulk_insert`, `commit_batch`, `execute_query`) is a real coroutine
that runs off the event loop.

## Record identity

Three of the functions need to know a record's **identity** — the stable id it
is stored under — to upsert, detect duplicates, or re-commit idempotently.
Identity is consumer-specific, so it is configured through a small, swappable
strategy. Every identity-bearing function accepts **at most one** of:

| Sugar | Resolves to | Use when |
|---|---|---|
| `key_columns=[...]` | `KeyColumnsIdentity` | The id is a join of named key columns (the common case). |
| `id_fn=callable` | `CallableIdentity` | The id is computed (hash, encoding, a single raw column, a natural key). |
| `identity=<RecordIdentity>` | your own strategy | Full control — implement the `RecordIdentity` protocol (`derive(row) -> str | None`). |

Supplying more than one raises `ConfigurationError`. Supplying none means "no
caller-defined identity" — the backend assigns ids on create.

`KeyColumnsIdentity` joins key-column values with an ASCII unit separator
(`\x1f`) rather than a printable character, so composite keys cannot collide
the way a `"_"` join lets them (`{"a": "x_y", "b": "z"}` and
`{"a": "x", "b": "y_z"}` derive **distinct** ids). Override `sep=` only when the
storage id must follow a specific printable format.

```python
from dataknobs_fsm.functions.library.identity import (
    KeyColumnsIdentity, CallableIdentity,
)

KeyColumnsIdentity(["tenant", "order_id"])          # join two columns
CallableIdentity(lambda row: f"user:{row['email']}")  # custom scheme
```

## `DatabaseUpsert`

Upserts the incoming record(s) keyed by the configured identity. Reads
`data["records"]`, else `data["record"]`, else the whole `data` dict.

```python
DatabaseUpsert(
    resource_name="target_db",
    table="orders",
    key_columns=["id"],            # or id_fn= / identity=
    value_columns=None,            # None → persist the whole row
    on_conflict="update",          # "update" | "ignore" | "error"
)
```

`on_conflict` controls behaviour when the derived id already exists: `update`
(write through), `ignore` (skip), `error` (raise).

## `DatabaseBulkInsert`

Inserts `data["records"]` in chunks, honouring `on_duplicate`:

```python
DatabaseBulkInsert(
    resource_name="target_db",
    table="events",
    columns=None,                  # None → all columns from each row
    chunk_size=1000,
    on_duplicate="ignore",         # "error" | "ignore" | "update"
    key_columns=["id"],            # required for "ignore"/"update"
)
```

`on_duplicate` is evaluated against the configured identity:

- `error` — raise if a row's id already exists.
- `ignore` — skip existing rows (not counted in `inserted_count`).
- `update` — overwrite existing rows.

`ignore` and `update` **require** an identity (`key_columns` / `id_fn` /
`identity`) — without one there is nothing to detect duplicates against, so the
constructor raises `ConfigurationError` rather than silently degrading to
create-only. With no identity at all, the function is a plain create (rows get
backend-assigned ids and `on_duplicate` is not evaluated).

## `BatchCommit`

Persists `data["batch"]` and clears it. Without an identity it writes the batch
via `create_batch` (all-or-nothing on transactional backends); with an identity
it upserts each row under its derived id, so re-committing the same batch is
idempotent.

```python
BatchCommit(
    resource_name="target_db",
    batch_size=1000,
    key_columns=["id"],            # optional → idempotent re-commit
    atomicity="best_effort",       # "best_effort" | "require"
)
```

### Atomicity policy

`atomicity` is the consumer's choice, not a fixed behaviour:

- `best_effort` (default) — proceed on any backend. On a non-transactional
  backend (`memory`, `file`, `s3`, `elasticsearch`) the write is not
  all-or-nothing; this is logged at DEBUG, not silently implied to be atomic.
- `require` — raise `CapabilityNotSupportedError` when the backend cannot
  guarantee all-or-nothing (so a consumer who *needs* atomicity gets a loud,
  actionable failure instead of a partial write under a false promise). It
  succeeds atomically on transactional backends (`sqlite`, `postgres`,
  `duckdb`).

The legacy `use_transaction=` flag is a back-compat alias:
`use_transaction=True` maps to `atomicity="require"`, `False` to
`"best_effort"`.

> **Atomic idempotent commit** (`atomicity="require"` *together with* an
> identity) is not yet supported — the per-row upsert path is not batch-atomic
> without a connection-scoped transaction primitive, so it raises
> `CapabilityNotSupportedError`. Use create-mode (`require`, no identity) for
> all-or-nothing batch writes today.

## Read functions

`DatabaseFetch` and `DatabaseQuery` read records through the resource's
`execute_query`. They accept a `dataknobs-data` `Query` (or `None` for all
rows); raw SQL strings are not supported by the `AsyncDatabase` abstraction.
