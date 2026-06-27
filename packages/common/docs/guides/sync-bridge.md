# Async ↔ Sync Bridges

DataKnobs is async-first, but synchronous public APIs still need to drive
async-first implementations. Crossing that boundary naively is a common
source of bugs. `dataknobs-common` ships a small bridge for **each
direction**:

| Direction | Primitive | When to reach for it |
|---|---|---|
| sync → async | [`aiter_sync_in_thread`](#sync-async-driving-a-blocking-iterator-from-async) | An `async def` must consume a *lazy, blocking* sync iterator without stalling the loop |
| async → sync | [`SyncLoopBridge` / `run_coro_sync`](#async-sync-running-a-coroutine-from-sync-code) | A synchronous function must run a coroutine to completion and return its result |

Both run the foreign-coloured work on a dedicated worker/loop thread, so
neither blocks nor re-enters the caller's event loop.

## async → sync: running a coroutine from sync code

A synchronous wrapper around an async implementation must run a coroutine
to completion and return its value. The obvious tools fail in the one case
that matters most:

- `asyncio.run(coro)` and `loop.run_until_complete(coro)` **raise (or
  deadlock)** when a loop is *already running on the calling thread* —
  exactly what happens when your synchronous wrapper is itself called from
  inside async code.
- `nest_asyncio` patches around this but is rejected by the dependency bar.

`SyncLoopBridge` is the structural fix: it owns a private event loop on a
**dedicated daemon thread**, so a coroutine handed to `run()` always
executes on a loop that is *not* the caller's. The caller blocks on the
result like any synchronous call, and the "loop already running" footgun is
avoided by construction.

### Quick start

```python
from dataknobs_common import SyncLoopBridge

# Long-lived bridge owned by a synchronous wrapper (one daemon thread).
bridge = SyncLoopBridge()
try:
    result = bridge.run(some_async_function(arg))   # blocks, returns the value
finally:
    bridge.close()                                  # stops the loop, joins the thread

# Or as a context manager:
with SyncLoopBridge() as bridge:
    result = bridge.run(some_async_function(arg))
```

For a one-off call that does not justify owning a bridge, `run_coro_sync`
spins one up, runs the coroutine, and tears it down:

```python
from dataknobs_common import run_coro_sync

result = run_coro_sync(some_async_function(arg))
```

### Behavior

- **Returns the coroutine's value**; an exception it raises is re-raised in
  the caller with its **original traceback** preserved.
- **Safe from inside a running event loop** — the coroutine runs on the
  bridge's separate loop, so there is no re-entrancy and no deadlock.
- **Clean teardown** — `close()` stops the loop and joins the thread; it is
  idempotent and supported via the context-manager protocol. The loop
  thread is a `daemon`, so it can never block process exit.
- **Reusable** — a single bridge serves many `run()` calls. `run()` after
  `close()` raises `RuntimeError`.

### Cost and reuse

Each `SyncLoopBridge` (and each `run_coro_sync` call) costs one daemon
thread for its lifetime. When a synchronous component makes repeated calls,
**own a long-lived bridge and reuse it** rather than calling
`run_coro_sync` per call or spawning a bridge per call.

## sync → async: driving a blocking iterator from async

The counterpart, [`aiter_sync_in_thread`](api.md), drives a *lazy,
blocking* synchronous iterator on a worker thread and hands items to an
async consumer across a bounded queue — so the iterator's setup and every
step happen off the event loop, memory stays bounded (backpressure), and
abandoned iteration releases the source and joins the thread. Reach for it
when an `async def` must iterate a blocking generator it cannot rewrite as
async (a streaming file/format parser, a sync DB cursor, a paginated SDK
iterator).
