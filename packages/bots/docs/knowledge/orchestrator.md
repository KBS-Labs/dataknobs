# IngestOrchestrator

`IngestOrchestrator` is the subscriber-side primitive for event-driven
knowledge-base ingestion. It listens on an `EventBus` trigger topic and
dispatches each event to `KnowledgeIngestionManager.ingest_if_changed`.

It is intentionally small: trigger adapters (S3 event bridges, SQS
consumers, cron schedulers, webhook handlers) are deployment-specific
and remain the consumer's responsibility. The orchestrator handles the
generic receive side.

## Constructor

```python
from dataknobs_bots.knowledge import IngestOrchestrator

IngestOrchestrator(
    ingestion_manager: KnowledgeIngestionManager,
    event_bus: EventBus,
    trigger_topic: str = "knowledge:trigger",
)
```

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
    "last_version": str | None,   # optional
}
```

- **`domain_id`** is required. Missing payload skips dispatch and logs
  a WARNING (`dataknobs_bots.knowledge.orchestration`).
- **`last_version`**, when present, is forwarded to
  `ingest_if_changed(domain_id, last_version=...)`. When absent, the
  manager always runs (treating as "version unknown").

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
`knowledge:ingestion`) are unaffected by this behavior — they still
fire on successful ingests.

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

    # ... completion events fire on the "knowledge:ingestion" topic ...

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
await bus.subscribe("knowledge:ingestion", capture)

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
- [KnowledgeIngestionManager](../RAG_INGESTION.md) — the dispatch target
- [Events](../../../common/docs/guides/events.md) — `EventBus`,
  `Event`, `EventType`, and `InMemoryEventBus`
