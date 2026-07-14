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

The same functions work when the resource is declared on an **arc** instead of
a state — declare `resources` on the arc and reference the function from the
arc's `transform`. Both the arc transform and an arc condition receive the
arc's resources through `FunctionContext.resources`. See
[Resources in FSM functions](resources.md#resources-in-fsm-functions-the-injection-contract)
for the full injection contract, including name-based vs role-based access.

```python
config = {
    "resources": [
        {"name": "target_db", "type": "async_database",
         "config": {"type": "file", "path": "/data/out.json"}},
    ],
    "states": [
        {"name": "start", "is_start": True},
        {"name": "done", "is_end": True},
    ],
    "arcs": [
        {"from": "start", "to": "done", "name": "loaded",
         "resources": ["target_db"],
         "transform": {"type": "registered", "name": "load"}},
    ],
}
```

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

A key column that is **missing from the row or `None`** raises
`ValidationError` — an absent component has no well-defined value, and
rendering it as the literal `"None"` would let every such row collide. Use
`CallableIdentity` if your composite key legitimately has nullable or sparse
components.

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

`ignore` and `error` **require** an identity (`key_columns` / `id_fn` /
`identity`) — without an id there is no conflict to detect, so the constructor
raises `ConfigurationError` rather than silently degrading to create-only. The
default `update` with no identity is a legitimate plain create (rows get
backend-assigned ids).

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
via `create_batch`; with an identity it upserts each row under its derived id
via `upsert_batch`, so re-committing the same batch is idempotent. Both paths
are all-or-nothing on transactional backends.

```python
BatchCommit(
    resource_name="target_db",
    batch_size=1000,               # max rows per commit under best_effort
    key_columns=["id"],            # optional → idempotent re-commit
    atomicity="best_effort",       # "best_effort" | "require"
)
```

`batch_size` bounds how many rows are sent per `commit_batch` call under
`best_effort`, so a very large batch need not be held or transmitted as a
single unit (it must be a positive integer). It is **not applied under
`require`**: a required-atomic commit is issued as one all-or-nothing batch
(chunking would only make each chunk atomic, not the whole), so the whole batch
is committed in a single call regardless of `batch_size`.

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

`atomicity="require"` is honored on **both** paths on transactional backends:
create-mode (via `create_batch`) and the idempotent-upsert path (via
`upsert_batch`) are each written as a single all-or-nothing batch statement. On
a non-transactional backend `require` raises `CapabilityNotSupportedError` on
either path.

The `require` capability check is sourced from the data layer's
`AsyncDatabase.supports_transactions()` flag — `True` on `sqlite`, `postgres`,
and `duckdb`; `False` on `memory`, `file`, `s3`, and `elasticsearch`.

## `DatabaseTransaction`

Stages writes across FSM states and commits or rolls them back as a unit. The
`begin` action opens a buffered transaction (via
`AsyncDatabase.begin_transaction()`) and stows the handle on
`data["_transaction"]`; later states stage writes on the handle, and `commit`
flushes them (returning `committed_count`) while `rollback` discards them.
Because writes are buffered, a failure *before* `commit` persists nothing on any
backend.

Commit atomicity follows the buffered transaction's `is_atomic` flag: a single
same-kind batch (all creates, all upserts, or all deletes) is all-or-nothing on
a transactional backend, but a buffer spanning **more than one kind** (e.g.
mixed create/delete, or create + upsert) commits as a sequence of independent
batches and can partially persist if one fails mid-flush (see the data
package's [Transactions](../../data/transactions.md) guide). A `commit` reaching a state
with no active handle — a missing or failed prior `begin` — is logged at WARNING
and commits nothing, rather than reporting a phantom success; a handle-less
`rollback` is a quiet no-op.

```python
DatabaseTransaction(
    resource_name="target_db",
    action="begin",            # "begin" | "commit" | "rollback"
    on_unsupported="strict",   # "strict" | "emulate"
)
```

`on_unsupported` is the isolation policy for `begin` on a backend that cannot
guarantee atomic commit (`supports_transactions()` is `False`):

- `strict` (default, fail-closed) — raise `CapabilityNotSupportedError` rather
  than imply atomicity the backend cannot give.
- `emulate` — proceed with best-effort buffer-and-flush (writes still deferred,
  so a pre-commit failure persists nothing, but the flush is not crash-safe
  atomic).

The buffered transaction defers all writes until commit; it does **not** provide
in-transaction isolation or read-your-writes (staged writes are invisible to
reads until commit). For connection-scoped isolation, branch on
`supports_transactions()` and use a backend-native transaction directly.

> **The `transaction:` config block is not a database-atomicity knob.** A
> `transaction: {strategy: batch|manual, ...}` block configures an in-memory
> `TransactionManager` (commit-trigger / batching coordination) that the
> execution engines do **not** consult to drive database commit/rollback —
> configuring it logs a warning at build time. For database atomicity use
> `DatabaseTransaction`, `BatchCommit(atomicity="require")`, or the
> `AsyncDatabase.transaction()` primitive directly.

## Read functions

`DatabaseFetch` and `DatabaseQuery` read records through the resource's
`execute_query`. They accept a `dataknobs-data` `Query` (or `None` for all
rows); raw SQL strings are not supported by the `AsyncDatabase` abstraction.
