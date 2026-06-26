# Transactions

`AsyncDatabase` provides a **buffered transaction** primitive: writes are
staged and applied together on commit, and discarded if the block raises.

```python
async with db.transaction() as tx:           # default policy="strict"
    await tx.create(Record({"name": "a"}))
    await tx.upsert(record_id, Record({"name": "b"}))
# both applied together on clean exit; if the block raised, neither
```

## Guarantees

A buffered transaction makes two promises:

1. **Universal rollback.** Because every write is buffered, raising before
   commit persists nothing — on *any* backend, transactional or not.
2. **Atomic commit on transactional backends.** The commit flush replays the
   buffer through the backend's atomic batch primitives (`create_batch` /
   `delete_batch`), coalescing consecutive same-kind operations into a single
   batch call. On backends whose batch operations run inside a backend
   transaction — **SQLite, Postgres, DuckDB** — each coalesced batch is
   all-or-nothing.

It deliberately does **not** provide in-transaction isolation or
read-your-writes: buffered writes are invisible to reads (`db.read`) until
commit, and concurrent readers never observe a partially-applied transaction
because nothing is written until the flush. Consumers needing connection-scoped
isolation should branch on `supports_transactions()` and use a backend-native
transaction directly.

## `supports_transactions()`

```python
if db.supports_transactions():
    async with db.transaction():
        ...
else:
    ...  # roll your own, or accept best-effort
```

Returns `True` for the transactional backends (`sqlite`, `postgres`, `duckdb`)
and `False` for the rest (`memory`, `file`, `s3`, `elasticsearch`). It reports
whether the **commit flush** is crash-safe atomic — not whether a transaction
can be opened (the buffer-and-flush works on every backend).

## Policy on non-transactional backends

The `policy` argument decides what happens when the backend cannot guarantee an
atomic commit:

- `"strict"` (default, fail-closed) — raise `CapabilityNotSupportedError`,
  rather than imply atomicity the backend cannot deliver.
- `"emulate"` — proceed with best-effort buffer-and-flush. Writes are still
  deferred (so a pre-commit failure persists nothing), but the flush itself is
  not crash-safe atomic.

```python
async with db.transaction(policy="emulate") as tx:   # ok on memory/file
    await tx.create(Record({"name": "a"}))
```

## Explicit begin / commit / rollback

When the begin and commit happen at separate call sites (for example an FSM
staging writes across states), use the explicit form. The returned
`BufferedTransaction` exposes the same staging methods plus `commit()` /
`rollback()`:

```python
tx = await db.begin_transaction(policy="emulate")
await tx.create(Record({"name": "a"}))
# ... later ...
await tx.commit()     # or await tx.rollback()
```

`commit()` returns `{"affected_rows": <count>}` and is idempotent (a second
`commit`/`rollback` is a no-op); staging a write after the transaction is closed
raises `OperationError`.
