# Knowledge-layer observability

The knowledge layer emits in-process events for ingest lifecycle and
backend state writes through the `CallbackRegistry` substrate. Consumers
register callbacks for live observability and, by composing a
`CallbackRegistry` with an `EventBus`, fan those events out across
replicas — no bespoke metrics interface required.

## Event topics

The canonical topic names live in `dataknobs_bots.knowledge.events` as
`Final[str]` constants. Import and subscribe to them; the literal
strings are part of the cross-service contract.

| Constant | Topic | Fired by |
|---|---|---|
| `INGEST_DOMAIN_START` | `ingest:domain:start` | `KnowledgeIngestionManager` at the head of every `ingest()` / `ingest_changes()` |
| `INGEST_DOMAIN_END` | `ingest:domain:end` | `KnowledgeIngestionManager` at the tail of every run (success **and** failure) |
| `INGEST_METADATA_WRITE` | `ingest:metadata:write` | every backend, after a metadata state write |
| `INGEST_SNAPSHOT_WRITE` | `ingest:snapshot:write` | every backend, after a snapshot state write |

Payloads are `dict[str, Any]`. The start/end payloads carry
`domain_id`, lifecycle stats, and `tenant_id` (only when the manager is
tenant-bound). State-write payloads carry `domain_id`, `key`, `kind`
(a `KnowledgeKeyKind`), and `byte_size`.

## Ingest lifecycle events

Register callbacks on the manager's `lifecycle_callbacks` registry:

```python
from dataknobs_bots.knowledge import (
    INGEST_DOMAIN_END,
    INGEST_DOMAIN_START,
    KnowledgeIngestionManager,
)

manager = KnowledgeIngestionManager(source=backend, destination=rag)

manager.lifecycle_callbacks.register(
    INGEST_DOMAIN_START,
    lambda event: log.info("ingest started: %s", event["domain_id"]),
)
manager.lifecycle_callbacks.register(
    INGEST_DOMAIN_END,
    lambda event: metrics.observe(event["domain_id"], event["status"]),
)

await manager.ingest("my-domain")
```

The end event fires even when the run raises — its `status` is
`"failed"` rather than `"completed"`, so a single callback observes the
whole lifecycle.

## Cross-replica fan-out

Construct the manager with an `event_bus` and the lifecycle registry
auto-composes `also_publish_to(event_bus)` — every lifecycle event is
also published to the bus for other replicas to consume. In-process
callbacks still run.

```python
from dataknobs_common.events import InMemoryEventBus

bus = InMemoryEventBus()
await bus.connect()

manager = KnowledgeIngestionManager(
    source=backend, destination=rag, event_bus=bus, tenant_id="acme"
)

# A second replica subscribes to the shared bus.
async def on_ingest_end(event):
    ...  # event.payload has domain_id, tenant_id, status, ...

await bus.subscribe(INGEST_DOMAIN_END, on_ingest_end)
await manager.ingest("my-domain")
```

Fan-out is observability and never breaks the operation it observes: a
failed bus `publish` (a transient broker/network hiccup on a real
Postgres/SQS/Redis bus) is logged and swallowed, so the ingest completes
normally and a genuine ingestion error is never masked by a publish
error. A consumer that needs publish failures to be fatal — a durable
audit trail, say — opts out per-target with
`also_publish_to(bus, isolate_errors=False)`. `asyncio.CancelledError`
always propagates regardless.

## Per-tenant filtering

When several tenants publish to the same topic, wrap a callback in
`TenantFilteredCallback` to receive only one tenant's events:

```python
from dataknobs_bots.knowledge import INGEST_DOMAIN_END, TenantFilteredCallback

manager.lifecycle_callbacks.register(
    INGEST_DOMAIN_END,
    TenantFilteredCallback(acme_handler, tenant_id="acme"),
)
```

Events without a `tenant_id` key (single-tenant payloads) are dropped by
the filter.

## Backend state-write events

Every backend fires `INGEST_METADATA_WRITE` / `INGEST_SNAPSHOT_WRITE`
after each state write on its `state_write_callbacks` registry. Compose
with an `EventBus` the same way as the lifecycle registry:

```python
from dataknobs_bots.knowledge import INGEST_METADATA_WRITE

backend.state_write_callbacks.register(
    INGEST_METADATA_WRITE,
    lambda event: audit.record(event["domain_id"], event["key"]),
)
backend.state_write_callbacks.also_publish_to(bus, topic_prefix="kb:")
```

The registry is constructed lazily on first access, so backends that no
consumer observes pay nothing.

## Subscribing to content changes

`subscribe_to_changes` composes a backend's `key_pattern()` with
`EventBus.subscribe()`. It defaults to `KnowledgeKeyKind.CONTENT` — the
consumer-controlled writes external event sources should observe, never
the DK-managed `_metadata.json` / `_snapshots/` writes (subscribing to
those during ingest creates a positive feedback loop).

```python
from dataknobs_bots.knowledge.storage import KnowledgeKeyKind

# Returns an EventBus.Subscription; await sub.cancel() to tear down.
sub = await backend.subscribe_to_changes(
    bus, domain_id="my-domain", handler=on_content_change
)

# Or the async-context-manager variant (cancels on exit):
async with backend.changes_subscription(
    bus, kinds={KnowledgeKeyKind.CONTENT}, handler=on_content_change
) as sub:
    ...
```

To subscribe to more than one kind, call once per kind — a single
fnmatch pattern cannot express multiple kinds.

## Tool-execution observability

`ExecutionTracker` (in `dataknobs-llm`) exposes the same pattern: every
`record(...)` / `record_async(...)` fires the `execution:record` topic
on its `execution_callbacks` registry with a
`{tool_name, success, duration_ms, error}` payload. A `priority=-100`
guard callback under `ErrorPolicy.RAISE` can abort the recording path
(e.g. on a cost ceiling).

For EventBus fan-out, compose `execution_callbacks.also_publish_to(bus)`
and drive recording through `record_async` — the async variant fires via
`fire_async`, so bus delivery is awaited correctly from inside a running
loop. The `ToolRegistry.execute_tool` path already uses `record_async`,
so a tool registry with tracking enabled gets cross-replica fan-out for
free. (Sync `record(...)` with fan-out composed inside a running loop is
rejected — use `record_async` there.)
