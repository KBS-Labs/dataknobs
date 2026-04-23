# Knowledge Base Ingestion Guide

This guide covers the three supported paths for populating a
`RAGKnowledgeBase`: local-directory ingest, storage-backend ingest, and
event-driven ingest. All three paths drive the same
`DirectoryProcessor` pipeline, so `KnowledgeBaseConfig` richness
(patterns, exclude patterns, per-pattern chunking, streaming JSON) is
available regardless of path.

## Choosing a Path

| Path | When to use |
|---|---|
| **Local directory** (`load_from_directory`) | Dev setups, on-disk corpora, tests |
| **Storage backend** (`ingest_from_backend`) | S3-hosted corpora, in-memory backends, multi-tenant deployments with a shared backend |
| **Event-driven** (`IngestOrchestrator`) | Hot-reload on upload events (S3 → SQS, cron triggers, webhook-delivered updates) |

## Path 1 — Local Directory

```python
from dataknobs_bots.knowledge import (
    RAGKnowledgeBase,
    KnowledgeBaseConfig,
    FilePatternConfig,
)

kb = await RAGKnowledgeBase.from_config({
    "vector_store": {"backend": "memory", "dimensions": 384},
    "embedding_provider": "echo",
    "embedding_model": "test",
})

config = KnowledgeBaseConfig(
    name="docs",
    patterns=[
        FilePatternConfig(pattern="**/*.md"),
        FilePatternConfig(
            pattern="api/*.json",
            text_fields=["title", "description"],
        ),
    ],
    exclude_patterns=["**/drafts/**"],
)

stats = await kb.load_from_directory("./docs", config=config)
print(stats["total_files"], stats["total_chunks"])
```

If `config` is omitted, `load_from_directory` reads
`knowledge_base.(json|yaml|yml)` from the directory when present;
otherwise it uses `KnowledgeBaseConfig` defaults.

## Path 2 — Storage Backend

Use `ingest_from_backend` for corpora stored in a
`KnowledgeResourceBackend` — file, in-memory, or S3.

```python
from dataknobs_bots.knowledge import (
    RAGKnowledgeBase,
    create_knowledge_backend,
)

backend = create_knowledge_backend("s3", {
    "bucket": "my-kb-bucket",
    "prefix": "domains/",
    # Region + credentials — see dataknobs-data S3 reference
})
await backend.initialize()

kb = await RAGKnowledgeBase.from_config(rag_config)
stats = await kb.ingest_from_backend(backend, "my-domain")
```

### Config from `_metadata/`

When `config` is `None`, `ingest_from_backend` attempts to load
`_metadata/knowledge_base.yaml`, `.yml`, or `.json` (in that order)
from the backend's `domain_id` namespace. Falls back to
`KnowledgeBaseConfig(name=domain_id)` when no metadata document is
present.

```python
# Stored alongside the domain's content:
#   my-domain/_metadata/knowledge_base.yaml
#   my-domain/docs/intro.md
#   my-domain/docs/api.md
```

### S3 Credentials & Region

S3 session configuration (region, credentials) follows the canonical
`S3SessionConfig` contract — passing keys through the backend config
dict is equivalent to setting them on the session. See the
`dataknobs-data` reference for the full key list and environment
precedence.

## Path 3 — Event-Driven (Manager + Orchestrator)

For hot-reload deployments, wire a `KnowledgeIngestionManager` with an
`EventBus` and subscribe an `IngestOrchestrator` to a trigger topic.

```python
from dataknobs_common.events import Event, EventType, InMemoryEventBus
from dataknobs_bots.knowledge import (
    IngestOrchestrator,
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
    create_knowledge_backend,
)

bus = InMemoryEventBus()
await bus.connect()

backend = create_knowledge_backend("file", {"path": "./kb"})
await backend.initialize()

kb = await RAGKnowledgeBase.from_config(rag_config)

manager = KnowledgeIngestionManager(
    source=backend,
    destination=kb,
    event_bus=bus,
)
orchestrator = IngestOrchestrator(manager, bus)
await orchestrator.start()

# A trigger adapter (e.g. S3-event bridge, SQS consumer, cron task)
# publishes trigger events. For illustration, publish directly:
await bus.publish(
    "knowledge:trigger",
    Event(
        type=EventType.UPDATED,
        topic="knowledge:trigger",
        payload={"domain_id": "my-domain"},
    ),
)

# Manager publishes a "knowledge:ingestion" completion event when done.
# ...

await orchestrator.stop()
await bus.close()
```

Trigger adapters (S3 event → bus, SQS → bus, cron → bus, webhook →
bus) are **deployment-specific** and remain the consumer's
responsibility. `IngestOrchestrator` is the generic receive side.

See [IngestOrchestrator](orchestrator.md) for trigger-event payload
shape, subscription lifecycle, and error containment.

## Skip-if-Changed — `ingest_if_changed`

`KnowledgeIngestionManager.ingest_if_changed(domain_id, last_version=None)`
checks the backend's `has_changes_since()` before re-ingesting. When
`last_version` is provided and the backend reports no changes, it
returns `None` and skips the ingest.

This is the method `IngestOrchestrator` calls on each trigger. Trigger
payloads may include `last_version` for per-domain version tracking;
orchestrators remain stateless across restarts.

```python
result = await manager.ingest_if_changed("my-domain", last_version="v42")
if result is None:
    print("No changes since v42 — skipped")
else:
    print(f"Ingested {result.chunks_created} chunks")
```

## Status Tracking

`KnowledgeIngestionManager.ingest()` transitions the domain's
`ingestion_status` on the backend:

- `"ingesting"` before processing
- `"ready"` on success
- `"error"` on failure (with the error message)

Backends expose `get_info(domain_id)` to read the current status.

## Completion Events

When an `event_bus` is wired into the manager, every successful ingest
publishes a `knowledge:ingestion` event:

```python
Event(
    type=EventType.UPDATED,
    topic="knowledge:ingestion",
    payload={
        "domain_id": "my-domain",
        "files_processed": 12,
        "chunks_created": 47,
        "status": "ready",
    },
)
```

Consumers subscribe on the same bus to react — invalidate query
caches, refresh routing tables, emit metrics, etc.

## Summary

| Path | Entry point | Driven by |
|---|---|---|
| Local directory | `RAGKnowledgeBase.load_from_directory` | `LocalDocumentSource` |
| Any backend | `RAGKnowledgeBase.ingest_from_backend` | `BackendDocumentSource` |
| Event-driven | `IngestOrchestrator` + `KnowledgeIngestionManager.ingest_if_changed` | The above plus `EventBus` subscription |

All three share the same `DirectoryProcessor` pipeline — patterns,
excludes, per-pattern chunking, streaming JSON apply in every case.

## Related

- [IngestOrchestrator](orchestrator.md) — event-driven subscriber API
- [DocumentSource](../../../xization/docs/ingestion/document-source.md)
  — async protocol underlying the unified pipeline
- [DirectoryProcessor](../../../xization/docs/ingestion/directory-processor.md)
  — async-primary processor
- [RAG Ingestion (historical reference)](../RAG_INGESTION.md) —
  `load_from_directory`, hybrid search, progress callbacks
