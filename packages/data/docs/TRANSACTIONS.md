# Transactions

`AsyncDatabase` provides a **buffered transaction** primitive: writes are
staged and applied together on commit, and discarded if the block raises.

```python
async with db.transaction() as tx:           # default policy="strict"
    await tx.create(Record({"name": "a"}))
    await tx.create(Record({"name": "b"}))
# both applied together on clean exit; if the block raised, neither
```

## Guarantees

A buffered transaction makes two promises:

1. **Universal rollback.** Because every write is buffered, raising before
   commit persists nothing — on *any* backend, transactional or not.
2. **Atomic commit of a single same-kind batch on transactional backends.** The
   commit flush replays the buffer through the backend's atomic batch
   primitives (`create_batch` / `delete_batch`), coalescing consecutive
   same-kind operations into a single batch call. When the staged buffer
   reduces to *one* such call — all creates, or all deletes, with no upserts —
   that call is all-or-nothing on a backend whose batch operations run inside a
   backend transaction (**SQLite, Postgres, DuckDB**).

### What is *not* all-or-nothing: mixed and upsert buffers

A buffer that mixes creates **and** deletes, or that contains any **upsert**,
commits as a **sequence of independent backend batches** — create/delete runs
flush as separate calls, and upserts apply one row at a time (the abstraction
has no batch-upsert primitive). If a later batch fails mid-flush, earlier
batches have **already committed and stay persisted** — a partial commit, with
no compensating rollback (the earlier writes are already durable).

Branch on **`is_atomic`** to know which case you have:

```python
tx = await db.begin_transaction()
await tx.create(Record({"name": "a"}))
await tx.create(Record({"name": "b"}))
assert tx.is_atomic            # single create_batch → all-or-nothing
await tx.delete(other_id)
assert not tx.is_atomic        # now create + delete → two independent batches
```

`is_atomic` is `True` only when the backend supports transactions **and** the
currently staged ops reduce to a single coalesced same-kind batch. It is
computed from the live buffer, so staging more writes can flip it — read it
immediately before `commit()` when you need to rely on all-or-nothing across
the whole commit. For genuine cross-operation atomicity today, branch on
`is_atomic` / `supports_transactions()` and use a backend-native transaction.

It deliberately does **not** provide in-transaction isolation or
read-your-writes: buffered writes are invisible to reads (`db.read`) until
commit, and concurrent readers never observe a partially-applied transaction
because nothing is written until the flush. Do **not** commit two buffered
transactions concurrently against a single-connection backend (e.g. aiosqlite):
the per-batch `BEGIN`/`COMMIT` boundaries the backend issues can interleave.
Consumers needing connection-scoped isolation should branch on
`supports_transactions()` and use a backend-native transaction directly.

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
whether a **coalesced batch** in the commit flush is crash-safe atomic — not
whether a transaction can be opened (the buffer-and-flush works on every
backend), and not whether a *whole* mixed/upsert commit is atomic (see
`is_atomic` above for that).

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
