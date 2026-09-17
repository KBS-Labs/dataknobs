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
- **Bounded wait** — `run(coro, timeout=...)` (and `run_coro_sync(coro,
  timeout=...)`) raise `TimeoutError` if the coroutine does not finish in
  time. The timed-out coroutine is asked to cancel (best-effort) and the
  bridge stays usable. With no `timeout` the wait is unbounded.
- **Clean teardown** — `close()` stops the loop and joins the thread; it is
  idempotent and supported via the context-manager protocol. Concurrent
  closers all block until teardown completes. The loop thread is a
  `daemon`, so it can never block process exit.
- **Reusable, and concurrency-safe for submission** — a single bridge serves
  many `run()` calls, including concurrent calls from multiple threads.
  `run()` after `close()` raises `RuntimeError`.

> **Interrupt/timeout semantics.** If the calling thread is interrupted
> (`KeyboardInterrupt`) or times out while blocked in `run()`, the
> interrupt/timeout reaches the *caller*, but the coroutine keeps running on
> the bridge loop until it completes or its best-effort cancellation takes
> effect — it is not abandoned mid-flight. Likewise, ordering between `run()`
> and `close()` is the caller's responsibility: a `run()` issued strictly
> after `close()` raises `RuntimeError`, but a `run()` that *races* an
> in-flight `close()` from another thread is undefined — quiesce `run()`
> callers before closing, and pass a `timeout` if you need a guaranteed upper
> bound on the wait. Do **not** call `close()` from inside a coroutine
> running on the bridge (it would have to join its own thread); that raises
> `RuntimeError`.

### Cost and reuse

Each `SyncLoopBridge` (and each `run_coro_sync` call) costs one daemon
thread for its lifetime. When a synchronous component makes repeated calls,
**own a long-lived bridge and reuse it** rather than calling
`run_coro_sync` per call or spawning a bridge per call.

### Whoever owns the object owns its loop

Thread cost is the *cheap* half of choosing a scope. The other half is that
**an object can bind itself to the first loop it runs on, and then only that
loop will do.** An `asyncpg` pool acquired by `connect()` belongs to the loop
that acquired it; a second loop finds it unusable, with an error that names
the connection rather than the loop —
`InterfaceError: cannot perform operation: another operation is in progress`.

So a wrapper's bridge scope is bounded above by its object's lifetime, and
the object's lifetime belongs to whoever *owns* it:

| Who connects the object | Where the loop has to come from |
|---|---|
| the wrapper, on first use | the wrapper's own bridge — held for as long as it holds the object |
| the caller, before handing it in | the **caller's** bridge, passed as `bridge=` |

The second row is the one a wrapper cannot fix for itself. A wrapper handed
an already-connected object has no way to reach the loop that connected it,
so `bridge=` is not only a way to save a thread — for such an object it is
the only way the wrapper can work at all. `BatchOperations`
(`dataknobs-data`) is the worked example: it takes a database it does not
own, so it scopes its own bridge to one operation and documents `bridge=` as
required for any backend that binds.

Uncontended `asyncio` primitives do **not** bind, which is why this is easy
to miss: `asyncio.Lock.acquire` reaches `_get_loop` only when it has to wait,
so an in-memory store guarded by one survives any amount of loop churn and a
test suite built on it reports green.

### `SyncBridgeAdapter` — the wrapper shape, declared once

A synchronous wrapper over an asynchronous object is the common case for a
long-lived bridge, and three of them in this workspace wrote the same surface
by hand before it had a name: `SyncTextEmbedder` (`dataknobs-data`),
`BridgedEntityResolver` (here) and `SyncProviderAdapter` (`dataknobs-llm`).
`SyncBridgeAdapter` is that surface. A subclass supplies only what varies —
the object it wraps, the methods that forward to it, and, if it *owns* that
object, how to close it:

```python
from dataknobs_common import SyncBridgeAdapter

class SyncThing(SyncBridgeAdapter):
    BRIDGE_THREAD_NAME = "dk-sync-thing"   # required: see below

    def __init__(self, inner, **kwargs):
        super().__init__(**kwargs)
        self._inner = inner

    def do(self, x):
        return self._run(self._inner.do(x))
```

What a subclass gets for free:

| Member | What it is for |
|---|---|
| `bridge=` | run on a bridge the caller owns, so several wrappers cost one thread. `close()` then leaves it running |
| `timeout=` | an upper bound on a blocking wait a synchronous caller cannot otherwise cancel |
| `_run(coro)` | the one place the bridge is reached, so forwarding methods cannot disagree about which loop they run on |
| `_run_teardown(coro)` | the same, for a `_close_inner` override — the one call allowed after the wrapper is marked closed, since `_run` refuses there |
| `close()` / `aclose()` | teardown from sync and from async code. Exactly one caller of either tears down; the rest wait for it, so a holder closed from two threads at once cannot stop the loop under its own teardown |
| `with` / `async with` | the reliable teardown form, one per kind of holder |
| lazy construction | the thread is allocated on first `_run`, so building one to read a model id or a capability set costs nothing |

The wrapped object is deliberately **not** stored by the base: each of the
three names it differently and one exposes it publicly, so a base that owned it
would force a rename on a published attribute to buy nothing.

Override `_close_inner()` / `_aclose_inner()` only if the wrapper **owns** what
it wraps. Two of the three are handed an object the caller keeps and must not
close; the third is what a factory returns, so nothing else can close it. That
is ownership, not drift, which is why it is a hook rather than a shared body.
An override reaches the wrapped object through `_run_teardown(coro)`, not
`_run(coro)`: the hook runs with the wrapper already marked closed, and `_run`
refuses there by design.

**`aclose()` is a claim about which thread waits, not about which loop runs the
teardown.** What it guarantees is that the *holder's* loop is free; where the
teardown belongs is the subclass's decision, and the answer is not always "this
loop". A wrapper whose object holds loop-bound state — an `aiohttp` session
opened by an `initialize()` that went through `_run`, and therefore bound to
the *bridge's* loop — must still close it there, with
`await asyncio.to_thread(self._close_inner)`. Awaiting the object directly is
right only when nothing it holds is tied to a loop. `SyncProviderAdapter` is
the worked example: closing a provider from the holder's loop makes
`AsyncLLMProvider.close`'s `asyncio.gather` over its in-flight tasks raise
`got Future ... attached to a different loop`.

Both context-manager protocols are present because both teardowns are. A
wrapper whose purpose is to be *called* synchronously is routinely built and
torn down by async code that hands it to a `def` site in a worker thread, so
both kinds of holder are real:

```python
async with SyncThing(inner) as sync:               # teardown is awaited
    await asyncio.to_thread(run_sync_pipeline, sync)

with SyncThing(inner) as sync:                     # teardown goes via the bridge
    run_sync_pipeline(sync)
```

Neither entry is a mistake, so neither is refused. An async holder that writes
the synchronous form gets the bridged teardown, which is a cost rather than a
bug — unlike `AsyncLLMProvider`, where sync entry is *always* wrong and
`__enter__` therefore raises.

> **Quiesce the callers before the block ends.** `asyncio.to_thread` cannot
> cancel the thread it started. If the holding task is cancelled — a client
> disconnects, a `TaskGroup` sibling fails, a shutdown timeout fires — the
> `async with` body unwinds *while the worker is still inside a `_run` call*,
> and teardown stops the loop out from under it. This is
> [`SyncLoopBridge`'s own rule](#behavior) reaching the wrapper: a `run` that
> races an in-flight `close` is undefined. Pass `timeout=` — it is the only
> upper bound a blocked worker has — and join or cancel-and-await your workers
> before leaving the block.

Teardown itself is not guaranteed to be free of I/O, in either form. Ending the
bridge joins its thread, and the loop drains its async generators on the way
down — so an abandoned stream's `finally` runs inside that join. Usually
microseconds; a holder that cannot afford even that wraps `aclose()` in
`asyncio.to_thread` as well.

`BRIDGE_THREAD_NAME` is **required** — a subclass that omits it raises
`TypeError` at class-creation time. It is also a diagnostic label rather than a
way out of the leak guard: the bridge registers every name it runs under, so
`assert_no_leaked_bridge_threads()` watches a subclass's name too. The reason
it is not merely defaulted is that the default would be the shared
`dk-sync-loop-bridge` that `run_coro_sync`'s throwaway bridges already use, so
a subclass that forgot would report a real registered name belonging to
something else: the diagnostic silently inverted rather than absent.

## sync → async: driving a blocking iterator from async

The counterpart, [`aiter_sync_in_thread`](https://kbs-labs.github.io/dataknobs/packages/common/api/), drives a *lazy,
blocking* synchronous iterator on a worker thread and hands items to an
async consumer across a bounded queue — so the iterator's setup and every
step happen off the event loop, memory stays bounded (backpressure), and
abandoned iteration releases the source and joins the thread. Reach for it
when an `async def` must iterate a blocking generator it cannot rewrite as
async (a streaming file/format parser, a sync DB cursor, a paginated SDK
iterator).
