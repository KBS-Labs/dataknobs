# Distributed Locks

The lock abstraction provides **named mutual exclusion** for critical
sections in the DataKnobs ecosystem. It is the third member of the
concurrency-primitive set alongside the [Rate Limiter](ratelimit.md)
and the [Event Bus](events.md): one protocol, a zero-dependency
in-process default, and a registry-extensible factory so the backing
mechanism is selected by configuration without code changes.

## Overview

The lock abstraction lets you:

- **Serialize a critical section by key** — at most one holder per
  `key` at a time
- **Choose blocking, timed, or context-managed** acquisition
- **Switch backends** without changing application code (single
  process today; cross-replica via a registered backend)
- **Scale from single-process** (in-memory) **to multi-replica** (a
  registry-pluggable distributed backend)

## Installation

The in-process lock is included in `dataknobs-common`:

```bash
pip install dataknobs-common
```

No extra dependency is required for the default backend. The
cross-replica Postgres backend reuses the existing optional `postgres`
extra (shared with `PostgresEventBus` — no new dependency):

```bash
pip install 'dataknobs-common[postgres]'
```

## Quick Start

```python
import asyncio
from dataknobs_common.locks import create_lock


async def main() -> None:
    lock = create_lock({"backend": "memory"})

    # Primary API: the async context manager.
    async with lock.hold("ingest:my-domain") as acquired:
        if acquired:
            ...  # critical section — serialized per key

    # Low-level escape hatch.
    if await lock.acquire("ingest:my-domain", timeout=5.0):
        try:
            ...
        finally:
            await lock.release("ingest:my-domain")

    await lock.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## Core Concepts

### Keys

A lock key is an **opaque string**; namespacing is the caller's
responsibility (e.g. `f"ingest:{domain_id}"`). Different keys are
independent — they never serialize against each other.

### `hold()` vs `acquire()`/`release()`

| API | Behaviour | Use When |
|-----|-----------|----------|
| `async with lock.hold(key) as got:` | Acquires on enter, releases on exit (even on error). `got` is the acquire result. | The default — almost always |
| `await lock.acquire(key, timeout=...)` | Returns `True` if acquired, `False` if a finite `timeout` elapsed. `timeout=None` blocks indefinitely and always returns `True`. | You need the acquire/release boundaries to span more than one lexical scope |
| `await lock.release(key)` | Releases `key`. No-op if not held by this instance. | Pairs with a manual `acquire()` |

A failed acquisition returns `False` — it is **not** an exception.
This deliberately differs from `RateLimiter.acquire`, which raises
`TimeoutError`: rate-limit exhaustion is exceptional, while lock
contention is a routine, expected control-flow outcome (skip or
retry).

### `close()`

Releases backing resources (connections, tasks). For `InProcessLock`
it is a no-op; always call it on shutdown so code is portable across
backends.

## Backend Selection

Choose your backend based on deployment topology:

### In-Process (Default — Development & Single-Replica Production)

Single-process mutual exclusion, no external dependencies.
Behaviour-identical to a bare `asyncio.Lock` per key, with the
key→lock map reference-count evicted so it cannot grow unbounded.

```python
lock = create_lock({"backend": "memory"})
lock = create_lock({})  # equivalent — "memory" is the default
```

**Use when:**

- Unit testing (this is the **testing construct** — use it instead of
  mocking a lock)
- Local development
- Single-replica production deployments (one process owns the
  critical section)

### Multi-Replica Deployments

When more than one process/replica can enter the same critical
section concurrently (e.g. two ingest workers behind a queue), a
process-local lock is **insufficient** — two replicas each acquire
their own `asyncio.Lock` and both proceed. A **cross-replica backend**
is required.

#### Postgres (Built-in Cross-Replica Backend)

`PostgresAdvisoryLock` provides mutual exclusion across **every
process pointing at the same database** using a **session-scoped**
`pg_advisory_lock` on a dedicated connection per held key. Select it
purely by config — application code is unchanged:

```python
lock = create_lock({
    "backend": "postgres",
    "connection_string": "postgresql://user:pass@host:5432/db",
})
```

The connection is resolved by the *same*
[Postgres connection config](postgres-config.md) helper
`PostgresEventBus` uses, so `connection_string`, individual
host/port/database/user/password keys, `DATABASE_URL`, and
`POSTGRES_*` env-var fallbacks all work identically:

```python
lock = create_lock({"backend": "postgres"})  # resolves from env
```

**Properties:**

- **Session-level, not transaction-level.** A critical section (e.g.
  an ingest) routinely outlives any single transaction;
  `pg_advisory_xact_lock` would release at the first commit inside the
  section, so the session-scoped form is used.
- **Liveness (guaranteed).** If a holding replica crashes, its
  Postgres session dies and the lock is released automatically — a
  dead replica can never wedge a domain.
- **Not a fencing token.** Advisory locks bound concurrency, not
  ordering. Mutual exclusion fully meets the orchestrator's need;
  fencing/leasing is a deliberately out-of-scope, distinct
  abstraction.
- **Stable keyspace.** The opaque string key is mapped to the signed
  64-bit id via Python `blake2b` (not Postgres `hashtext`, whose
  algorithm is not contractually stable across major versions), so a
  key means the same thing after a DB upgrade.
- **No new dependency.** `asyncpg` is the existing optional `postgres`
  extra, lazily imported — importing `dataknobs_common.locks` never
  requires it.

Other cross-replica backends (Redis, etcd, ZooKeeper, …) are
registry-pluggable (see
[Custom Backends](#custom-backends-plugin-registry)); only add a
specific one when a real consumer lacks Postgres. Application code
does not change — only the `create_lock({...})` config does.

### Custom Backends (Plugin Registry)

`create_lock()` resolves the `backend` key through the
`lock_backends` registry. You can register your own
`DistributedLock` implementation (Redis, etcd, ZooKeeper, …) and
select it by name — no fork of DataKnobs required:

```python
from dataknobs_common.locks import lock_backends, create_lock


def _make_redis_lock(config: dict) -> "DistributedLock":
    from my_pkg.redis_lock import RedisLock
    return RedisLock(url=config["url"])


# Register once at startup (e.g. in your package __init__).
lock_backends.register("redis", _make_redis_lock)

# Now selectable like any built-in backend.
lock = create_lock({"backend": "redis", "url": "redis://host:6379"})
```

A backend factory is any `Callable[[dict], DistributedLock]`.
Registering a key that already exists raises `OperationError` unless
you pass `allow_overwrite=True`. The built-in `memory` backend is
registered automatically. Passing `allow_overwrite=True` for `memory`
*will* replace the built-in backend process-wide — supported but
strongly discouraged; prefer a distinct backend name. Selecting an
unregistered backend raises `ValueError` listing every registered
backend (including consumer-registered ones).

This is the exact structural mirror of the Event Bus
[`event_bus_backends`](events.md#custom-backends-plugin-registry)
registry — the pattern is learned once and applied everywhere.

## Usage Patterns

### Per-Resource Serialization

```python
# Serialize all work for a given tenant/domain without blocking others.
async with lock.hold(f"ingest:{domain_id}"):
    await reindex(domain_id)
```

### Best-Effort (Skip If Busy)

```python
async with lock.hold("compaction", timeout=0) as got:
    if not got:
        return  # someone else is compacting; skip this tick
    await compact()
```

### Manual Span

```python
if await lock.acquire("leader", timeout=None):
    try:
        await run_as_leader()
    finally:
        await lock.release("leader")
```

## Testing

For testing, use `InProcessLock` directly — it is the **testing
construct** (like `InMemoryEventBus` / `InMemoryRateLimiter`). Do not
mock or fake a lock: a fake that appends to a list has the same
blindness as `MagicMock` — the whole point of a lock is its
concurrency behaviour.

```python
import asyncio
from dataknobs_common.locks import InProcessLock


async def test_critical_section_is_serialized() -> None:
    lock = InProcessLock()
    order: list[str] = []

    async def worker(tag: str) -> None:
        async with lock.hold("k"):
            order.append(f"enter-{tag}")
            await asyncio.sleep(0.01)
            order.append(f"exit-{tag}")

    await asyncio.gather(worker("a"), worker("b"))

    # Strict non-overlap.
    assert order in (
        ["enter-a", "exit-a", "enter-b", "exit-b"],
        ["enter-b", "exit-b", "enter-a", "exit-a"],
    )
    await lock.close()
```

The cross-replica `PostgresAdvisoryLock` is **not** faked in its own
tests either — a fake has the same blindness as a mock, and the entire
point of that backend is cross-process behaviour. It is exercised
against a real Postgres with two independent instances simulating two
replicas (gated by `@requires_postgres` + `TEST_POSTGRES=true`); start
the service with `bin/dk up`.

## API Reference

### DistributedLock Protocol

```python
@runtime_checkable
class DistributedLock(Protocol):
    async def acquire(
        self, key: str, *, timeout: float | None = None
    ) -> bool:
        """Acquire key. True if acquired, False if a finite timeout
        elapsed first. timeout=None blocks indefinitely (always True).
        """

    async def release(self, key: str) -> None:
        """Release key. No-op if not held by this instance."""

    def hold(
        self, key: str, *, timeout: float | None = None
    ) -> AbstractAsyncContextManager[bool]:
        """Async context manager wrapping acquire/release. Primary API."""

    async def close(self) -> None:
        """Release backing resources (connections, tasks)."""
```

### Factory Function

```python
def create_lock(config: dict) -> DistributedLock:
    """Create a distributed lock from configuration.

    Backends are resolved through the ``lock_backends`` registry, so
    custom backends registered by consumers are selectable too.

    Args:
        config: Configuration dict with:
            - backend: a registered backend name ("memory" or any
              consumer-registered key); default "memory"
            - Additional backend-specific options

    Returns:
        DistributedLock implementation

    Raises:
        ValueError: If the backend is not registered (message lists all
            registered backends).
    """
```

### Plugin Registry

```python
lock_backends: Registry[LockFactory]
# LockFactory = Callable[[dict[str, Any]], DistributedLock]

lock_backends.register("name", factory)   # add a backend
lock_backends.list_keys()                  # registered backend names
```

See [Custom Backends](#custom-backends-plugin-registry) for usage.

## Configuration Reference

### Memory Backend

```python
{"backend": "memory"}
```

No additional keys. `{}` is equivalent — `"memory"` is the default.

### Postgres Backend

```python
{"backend": "postgres", "connection_string": "postgresql://..."}
```

Accepts the full [Postgres connection config](postgres-config.md)
shape: `connection_string`, individual `host`/`port`/`database`/
`user`/`password` keys, `DATABASE_URL`, or `POSTGRES_*` env-var
fallbacks. Requires the `postgres` extra
(`pip install 'dataknobs-common[postgres]'`).

## Usage Across DataKnobs

| Package | Component | How It Uses the Lock |
|---------|-----------|----------------------|
| `dataknobs-bots` | `IngestOrchestrator` | Per-domain serialization of ingest triggers, keyed `ingest:<domain_id>`. Defaults to `InProcessLock`; multi-replica deployments inject a cross-replica lock. |

## Module Exports

```python
from dataknobs_common.locks import (
    # Protocol
    DistributedLock,
    # Factory
    create_lock,
    # Plugin registry
    lock_backends,
    LockFactory,
    # Default / testing implementation
    InProcessLock,
    # Cross-replica implementation
    PostgresAdvisoryLock,
)

# Also re-exported from the top-level namespace:
from dataknobs_common import DistributedLock, create_lock, InProcessLock
```
