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
2. **Atomic commit on transactional backends — any composition.** The commit
   flush replays the buffer through the backend's atomic batch primitives
   (`create_batch` / `upsert_batch` / `delete_batch`), coalescing consecutive
   same-kind operations into a single batch call. On a transactional backend
   (**SQLite, Postgres, DuckDB**) the *whole* flush runs inside **one** native
   transaction, so a commit is all-or-nothing regardless of composition — a
   single same-kind batch (all creates, all upserts, or all deletes) **and** a
   mixed buffer spanning several kinds (e.g. creates *and* deletes, or creates
   *and* upserts) alike. A mid-flush failure rolls the whole commit back.

### Multi-kind buffers commit atomically on transactional backends

A buffer spanning **more than one kind** (e.g. creates **and** deletes, or
creates **and** upserts) commits inside the same single native transaction as a
single-kind buffer: every coalesced batch runs on one pinned connection, so a
mid-flush failure rolls the **whole** commit back — no partial persistence.

On a **non-transactional** backend (`memory`, `file`, `s3`, `elasticsearch`)
there is no native transaction to span the flush, so a multi-kind buffer commits
as a **sequence of independent batches**: if a later batch fails mid-flush,
earlier batches have already been applied and stay applied. Open the transaction
with the default `policy="strict"` there to fail closed rather than assume an
atomicity the backend cannot deliver.

Branch on **`is_atomic`** to know whether a commit is all-or-nothing:

```python
tx = await db.begin_transaction()
await tx.create(Record({"name": "a"}))
await tx.delete(other_id)
assert tx.is_atomic            # transactional backend → whole commit atomic
```

`is_atomic` is `True` whenever the backend supports transactions: it reflects
the backend capability directly (any composition is atomic there) and is stable
across staging. On a non-transactional backend it is `False`.

It deliberately does **not** provide in-transaction isolation or
read-your-writes: buffered writes are invisible to reads (`db.read`) until
commit, and concurrent readers never observe a partially-applied transaction
because nothing is written until the flush. Do **not** commit two buffered
transactions concurrently against a single-connection backend (e.g. aiosqlite):
the per-batch `BEGIN`/`COMMIT` boundaries the backend issues can interleave.
Connection-scoped isolation / read-your-writes is not provided — the public API
exposes no connection-scoped transaction beyond this buffered form. For a
read-modify-write invariant use optimistic concurrency (`update` / `upsert` with
`expected_version`) or serialize the conflicting work yourself.

## `supports_transactions()`

```python
if db.supports_transactions():
    async with db.transaction():
        ...
else:
    ...  # roll your own, or accept best-effort
```

Returns `True` for the transactional backends (`sqlite`, `postgres`, `duckdb`)
and `False` for the rest (`memory`, `file`, `s3`, `elasticsearch`). On a
transactional backend the commit flush is crash-safe atomic for **any** buffer
composition — single-kind or multi-kind alike. It reports the backend
capability, not whether a transaction can be opened (the buffer-and-flush works
on every backend); `is_atomic` reports the same thing for a given handle.

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
