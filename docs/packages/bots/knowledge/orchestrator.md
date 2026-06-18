# IngestOrchestrator

`IngestOrchestrator` is the subscriber-side primitive for event-driven
knowledge-base ingestion. It listens on an `EventBus` trigger topic and
dispatches each event to the `KnowledgeIngestionManager` — to
`ingest_changes`, `ingest`, or `ingest_if_changed` depending on the
payload (see [Trigger-Event Payload](#trigger-event-payload)).

It is intentionally small: trigger adapters (S3 event bridges, SQS
consumers, cron schedulers, webhook handlers) are deployment-specific
and remain the consumer's responsibility. The orchestrator handles the
generic receive side.

## Constructor

```python
from dataknobs_bots.knowledge import IngestOrchestrator

IngestOrchestrator(
    ingestion_manager: KnowledgeIngestionManager | None,
    event_bus: EventBus,
    trigger_topic: str = "knowledge:trigger",
    lock: DistributedLock | None = None,
    lock_config: dict | None = None,
    manager_resolver: IngestionManagerResolver | None = None,
)
```

See [Concurrency & Locking](#concurrency-locking) for `lock` /
`lock_config`, and [Multi-tenant routing](#multi-tenant-routing) for
`manager_resolver` (the per-tenant alternative to a single static
`ingestion_manager`). Exactly one of `ingestion_manager` or
`manager_resolver` is required — pass an explicit `None` for the
manager when using a resolver.

## Lifecycle

```python
orch = IngestOrchestrator(manager, bus)
await orch.start()     # subscribe to trigger_topic
# ...
await orch.stop()      # cancel the subscription
```

Both `start()` and `stop()` are idempotent — repeated calls are safe.
`orch.is_running` reflects whether a subscription is active.

## Trigger-Event Payload

```python
{
    "domain_id": str,             # required
    "tenant_id": str | None,      # optional; routes to the per-tenant
                                  # manager when a resolver is configured
    "since_version": str | None,  # optional
    "force_full": bool | None,    # optional
    "last_version": str | None,   # optional
}
```

- **`domain_id`** is required. Missing payload skips dispatch and logs
  a WARNING (`dataknobs_bots.knowledge.orchestration`).
- **`tenant_id`** is optional. It is read but unused on the static
  single-tenant path; when a `manager_resolver` is configured it is
  passed to the resolver to select the per-tenant manager and it scopes
  the serialization lock key (see
  [Multi-tenant routing](#multi-tenant-routing)). A present-but-non-string
  `tenant_id` fails closed: the trigger is skipped with a WARNING rather
  than routed or coerced, since a misidentified tenant is a cross-tenant
  data leak.
- The remaining keys select the ingest entry point, checked in this
  order (so a `since_version` present alongside `force_full` takes the
  delta path — the more specific intent wins):
  1. **`since_version`** present (truthy) → `ingest_changes(domain_id,
     since_version)` — a per-file delta re-ingest of only what changed
     since that canonical snapshot id.
  2. **`force_full`** truthy → `ingest(domain_id,
     swap_mode=IngestSwapMode.CLEAR_FIRST)` — an unconditional full
     re-ingest.
  3. otherwise → `ingest_if_changed(domain_id,
     last_version=last_version)` (the default; `last_version` absent ⇒
     the manager always runs, treating the version as unknown).

  Every path is serialized through the same tenant-scoped per-domain
  lock (`ingest:{tenant_id or '-'}:<domain_id>`); only the dispatched
  method differs.

Example trigger event:

```python
from dataknobs_common.events import Event, EventType

event = Event(
    type=EventType.UPDATED,
    topic="knowledge:trigger",
    payload={"domain_id": "my-domain", "last_version": "v42"},
)
await bus.publish("knowledge:trigger", event)
```

## Error Containment

Exceptions raised by the manager are logged at `ERROR` via
`logger.exception` and suppressed so the subscription keeps serving
subsequent events. This is the expected behavior for subscriber loops
— a poisoned event must not tear down the pipeline.

Completion events published by the manager (topic
`ingest:domain:start` / `ingest:domain:end`) are unaffected by this
behavior — they still fire on every ingest run.

## Concurrency & Locking

Triggers for the same `(tenant_id, domain_id)` are serialized through
a `dataknobs_common.locks.DistributedLock` keyed
`ingest:{tenant_id or '-'}:<domain_id>`; different domains (and
different tenants sharing a domain) ingest in parallel. The lock's
*scope* is exactly the serialization scope. Single-tenant deployments
emit no `tenant_id`, so the key degrades to a stable
`ingest:-:<domain_id>` — one key per domain, serialization behaviour
identical to prior releases.

- **Default (neither `lock` nor `lock_config`)** — a process-local
  `InProcessLock`. Sufficient for single-replica deployments and
  behaviour-identical to prior releases.
- **`lock=`** — inject a pre-built `DistributedLock`.
- **`lock_config=`** — configuration-driven: the orchestrator
  resolves the lock through the shared `create_lock` factory, so a
  deployment selects the backend by config without writing code.

`lock` and `lock_config` are mutually exclusive (passing both raises
`ValueError`); an unknown `lock_config` backend raises `ValueError`
(fail closed).

> **Multi-replica deployments MUST configure a cross-replica lock.**
> A process-local lock cannot prevent two replicas from ingesting the
> same domain concurrently and racing on the vector store.

```python
from dataknobs_bots.knowledge import IngestOrchestrator

# Configuration-driven cross-replica lock (no lock code in your app):
orch = IngestOrchestrator(
    manager,
    bus,
    lock_config={"backend": "postgres", "connection_string": dsn},
)

# Equivalent with a pre-built lock:
from dataknobs_common.locks import create_lock

orch = IngestOrchestrator(
    manager,
    bus,
    lock=create_lock({"backend": "postgres", "connection_string": dsn}),
)
```

A built-in Postgres advisory-lock backend ships as the `"postgres"`
lock backend (`{"backend": "postgres", "connection_string": dsn}`);
additional cross-replica backends can be registered via
`dataknobs_common.locks.lock_backends`.

## Multi-tenant routing

A single-tenant deployment passes one static `ingestion_manager` and
every trigger dispatches to it. A **multi-tenant** deployment instead
injects a `manager_resolver` and emits `tenant_id` in the trigger
payload; the orchestrator resolves the correct per-tenant manager —
its own KB backend prefix, `vector_partition`, and embedder — for each
event. `ingestion_manager` and `manager_resolver` are mutually
exclusive; exactly one is required (passing both, or neither, raises
`ValueError`).

`IngestionManagerResolver` is an async, keyword-only,
`@runtime_checkable` protocol:

```python
from dataknobs_bots.knowledge import (
    IngestionManagerResolver,  # the protocol
    IngestOrchestrator,
    KnowledgeIngestionManager,
)

# Mapping tenant_id -> backend prefix / vector_partition / embedder is
# CONSUMER policy; caching is the resolver's responsibility (a plain
# dict here). The orchestrator never inspects how the manager is built.
_managers: dict[str, KnowledgeIngestionManager] = {}

async def resolve(*, tenant_id: str | None, domain_id: str):
    mgr = _managers.get(tenant_id)
    if mgr is None:
        mgr = build_manager_for_tenant(tenant_id)  # your wiring
        _managers[tenant_id] = mgr
    return mgr

orch = IngestOrchestrator(None, bus, manager_resolver=resolve)
await orch.start()

await bus.publish(
    "knowledge:trigger",
    Event(
        type=EventType.UPDATED,
        topic="knowledge:trigger",
        payload={"domain_id": "shared", "tenant_id": "acme"},
    ),
)
```

A resolver that raises is caught by the same log-don't-raise guard as
a failing manager (see [Error Containment](#error-containment)) — the
subscription keeps serving subsequent events.

The serialization lock key is tenant-scoped
(`ingest:{tenant_id or '-'}:<domain_id>`), so two tenants sharing a
`domain_id` no longer false-share one lock under the cross-replica
`"postgres"` backend (see [Concurrency & Locking](#concurrency-locking)
— read the two correctness boundaries together).

> **A multi-tenant deployment MUST pass `manager_resolver=` and emit
> `tenant_id` in the trigger payload.** This is the exact analogue of
> the multi-replica lock rule above: routing one tenant's trigger
> through another tenant's static manager is a cross-tenant data leak.
> When no `tenant_id` is present the resolver receives
> `tenant_id=None` and may apply its own default-tenant policy.

A built-in caching/eviction resolver wrapper is intentionally **not**
provided — eviction policy, connection lifecycle on evict, and
config-change invalidation are deployment concerns. Cache inside your
own resolver (the dict above) until a shared helper is warranted.

## Stateless by Design

The orchestrator does not track last-seen versions across restarts.
Consumers needing version persistence can either:

1. Include `last_version` in every trigger event (trigger-adapter
   responsibility), or
2. Wire a status store into the `KnowledgeIngestionManager` so the
   current version is sourced from there.

## Complete Example

```python
from dataknobs_common.events import Event, EventType, InMemoryEventBus
from dataknobs_bots.knowledge import (
    IngestOrchestrator,
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
    create_knowledge_backend,
)

async def main():
    bus = InMemoryEventBus()
    await bus.connect()

    backend = create_knowledge_backend("file", {"path": "./kb"})
    await backend.initialize()

    kb = await RAGKnowledgeBase.from_config({
        "vector_store": {"backend": "memory", "dimensions": 384},
        "embedding_provider": "echo",
        "embedding_model": "test",
    })

    manager = KnowledgeIngestionManager(
        source=backend,
        destination=kb,
        event_bus=bus,
    )
    orch = IngestOrchestrator(manager, bus)
    await orch.start()

    # In production, a trigger adapter publishes these events.
    await bus.publish(
        "knowledge:trigger",
        Event(
            type=EventType.UPDATED,
            topic="knowledge:trigger",
            payload={"domain_id": "my-domain"},
        ),
    )

    # ... completion events fire on the "ingest:domain:end" topic ...

    await orch.stop()
    await bus.close()
```

## Trigger Adapter Sketches

The orchestrator deliberately excludes trigger adapters. Below are
reference sketches consumers can adapt.

### AWS Lambda — S3 event → EventBus

```python
# AWS Lambda handler receives an S3 event; extract the bucket/key and
# publish a trigger. Topic, domain mapping, and version stamping are
# consumer-specific.
async def handler(s3_event):
    for record in s3_event["Records"]:
        key = record["s3"]["object"]["key"]
        domain_id = key.split("/", 1)[0]
        await bus.publish(
            "knowledge:trigger",
            Event(
                type=EventType.UPDATED,
                topic="knowledge:trigger",
                payload={"domain_id": domain_id},
            ),
        )
```

### SQS Consumer — SQS message → EventBus

```python
async def consume_sqs(sqs_client, queue_url):
    while running:
        messages = await sqs_client.receive_messages(queue_url, MaxNumberOfMessages=10)
        for msg in messages:
            payload = json.loads(msg["Body"])
            await bus.publish("knowledge:trigger", Event(
                type=EventType.UPDATED,
                topic="knowledge:trigger",
                payload=payload,
            ))
            await sqs_client.delete_message(queue_url, msg["ReceiptHandle"])
```

### Cron-Style — Scheduled tick → EventBus

```python
async def cron_tick(interval_seconds: float):
    while running:
        for domain_id in known_domains():
            await bus.publish("knowledge:trigger", Event(
                type=EventType.UPDATED,
                topic="knowledge:trigger",
                payload={"domain_id": domain_id},
            ))
        await asyncio.sleep(interval_seconds)
```

These are illustrations. The orchestrator itself ships no trigger
adapters.

## Testing

Use `InMemoryEventBus` and `InMemoryKnowledgeBackend` for unit tests:

```python
from dataknobs_common.events import InMemoryEventBus
from dataknobs_bots.knowledge.storage import InMemoryKnowledgeBackend

bus = InMemoryEventBus()
await bus.connect()
backend = InMemoryKnowledgeBackend()
await backend.initialize()
await backend.create_kb("d1")
await backend.put_file("d1", "intro.md", b"# Intro\n")

kb = await RAGKnowledgeBase.from_config(test_config)
manager = KnowledgeIngestionManager(source=backend, destination=kb, event_bus=bus)
orch = IngestOrchestrator(manager, bus)
await orch.start()

# Subscribe to completion events for assertions
completions = []
async def capture(event):
    completions.append(event)
await bus.subscribe("ingest:domain:end", capture)

await bus.publish("knowledge:trigger", Event(
    type=EventType.UPDATED,
    topic="knowledge:trigger",
    payload={"domain_id": "d1"},
))
# ...await completion, assert on chunks_created, etc.
```

## Related

- [Knowledge Base Ingestion Guide](ingestion-guide.md) — all three
  ingestion paths end-to-end
- [RAG Ingestion & Hybrid Search](../guides/rag-ingestion.md) —
  `load_from_directory`, progress callbacks, hybrid search
- [Events](../../common/events.md) — `EventBus`, `Event`, `EventType`,
  and `InMemoryEventBus`
