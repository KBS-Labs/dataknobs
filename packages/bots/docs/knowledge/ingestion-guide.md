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
`AwsSessionConfig` contract — passing keys through the backend config
dict is equivalent to setting them on the session. See the
`dataknobs-data` reference for the full key list and environment
precedence.

> The former name `S3SessionConfig` remains importable as a deprecated
> alias for `AwsSessionConfig`.

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

# Manager fires an "ingest:domain:end" completion event when done.
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

`last_version` must be a **canonical snapshot id** — the value returned
by `manager.get_current_version(domain_id)` (equivalently
`backend.get_checksum(domain_id)`). It is *not* the monotonic
`KnowledgeBaseInfo.version` counter; that counter is retained for
cache-invalidation/display only and must not be passed here. Capture
the snapshot id, persist it, and pass it back on the next trigger:

```python
version = await manager.get_current_version("my-domain")
# ... persist `version` somewhere durable, then later:
result = await manager.ingest_if_changed("my-domain", last_version=version)
if result is None:
    print("No changes — skipped")
else:
    print(f"Ingested {result.chunks_created} chunks")
```

This is the method `IngestOrchestrator` calls on each trigger. Trigger
payloads may include `last_version` for per-domain version tracking;
orchestrators remain stateless across restarts.

For a **file-level delta** (which files were added / modified /
deleted, not just "something changed"), use
`backend.list_changes_since(domain_id, version)`, which returns a
frozen `ChangeSet` (`added` / `modified` / `deleted` / `version`, with
`is_empty`). `has_changes_since` is its degenerate case. If `version`
predates the backend's snapshot retention the backend raises
`InvalidVersionError`; catch it and fall back to a full re-ingest.
All three in-tree backends retain per-version snapshots so
`list_changes_since` returns a minimal diff: `InMemoryKnowledgeBackend`
(in-process map), `FileKnowledgeBackend` (a `{path: checksum}` file
written to `<domain>/_snapshots/<version>.json` after every mutation),
and `S3KnowledgeBackend` (a snapshot object per version, or — with
`change_detection_mode="s3_versioning"` — the metadata object's own S3
version history, which requires bucket versioning and otherwise falls
back to a full re-ingest). An out-of-tree backend that does not
override `_load_snapshot` still gets correct change *detection* (every
current file reported `added` when the version differs) via the
version-equality short-circuit.

## Per-File Delta — `ingest_changes`

`ingest_if_changed` is binary: it re-ingests the *whole* domain when
anything changed. For large corpora where a single file changed (an
S3 `PutObject` on one of 100 files), that is a 100× embedding
amplification. `KnowledgeIngestionManager.ingest_changes` re-embeds
only what actually changed:

```python
version = await manager.get_current_version("my-domain")
# ... persist `version`; later, on a change trigger:
result = await manager.ingest_changes("my-domain", version)
print(result.files_processed, result.files_deleted)
```

`ingest_changes(domain_id, since_version, *, progress_callback=None,
config=None)` diffs the source against `since_version` (a
`get_current_version` / `get_checksum` value — **not** the monotonic
`KnowledgeBaseInfo.version` counter), then:

- purges the chunks of **deleted and modified** files (modified files
  must shed their stale chunks before re-embedding),
- re-embeds only the **added and modified** files,
- routes both through the *same* internal apply path as a full
  `ingest()`, so swap semantics never diverge between the full-domain
  and per-file routes.

`result.files_processed` counts added + modified; the new
`result.files_deleted` carries the removed-file count (`0` for a full
`ingest`). If `since_version` predates the backend's snapshot
retention the backend raises `InvalidVersionError`; `ingest_changes`
catches it, logs a warning, and falls back to a full re-ingest — it
never silently skips an update.

The lower-level seam is
`RAGKnowledgeBase.ingest_from_backend(file_filter=)`: an optional
keyword-only `Callable[[KnowledgeFile], bool]` predicate, evaluated
after the pattern match, that restricts enumeration to a subset of
files while reusing the full pattern/chunking pipeline. `None`
(default) enumerates every matching file (unchanged behavior).

## Zero-Downtime Re-Ingest — `swap_mode`

`ingest()` and `ingest_changes()` accept a keyword-only
`swap_mode: IngestSwapMode`:

| Mode | Behavior |
|---|---|
| `CLEAR_FIRST` | Delete the (domain-scoped) chunks, then ingest. Historical default; a concurrent reader sees a brief zero-results window. |
| `APPEND` | Ingest without a preceding full-domain clear. Per-file deletions requested by `ingest_changes` still happen. |
| `TOMBSTONE` | Crash-safe swap (below). Recommended production default for multi-replica re-ingest. |

```python
from dataknobs_bots.knowledge import IngestSwapMode

await manager.ingest("my-domain", swap_mode=IngestSwapMode.TOMBSTONE)
# Per-file delta, same crash-safe guarantee scoped to changed files:
await manager.ingest_changes(
    "my-domain", version, swap_mode=IngestSwapMode.TOMBSTONE
)
```

### What `TOMBSTONE` guarantees

1. The existing (scoped) chunks are marked `_stale`; reads stop
   seeing them immediately (the shared read filter for `query()`
   and `hybrid_query()` hides `_stale` chunks).
2. Status flips to `IngestionStatus.SWAPPING` and a fresh
   per-swap generation token is persisted on the domain.
3. The new generation is ingested with **distinct, generation-keyed
   chunk ids** (stamped with the token), so it never overwrites the
   tombstoned old rows in place — both generations coexist
   physically. New chunks are not `_stale`, so they are read-visible
   as soon as they commit.
4. On a clean commit the tombstoned old generation is physically
   retired. On a raised error **or** a partial-error ingest the
   rollback drops exactly the new generation by its token, restores
   the modified files' old generation to visibility, and
   unconditionally purges files deleted at the source (never
   resurrected).

Because the new generation has distinct ids, the old generation is
never overwritten or deleted until the new one commits cleanly — it
is fully restorable throughout. A crash mid-swap leaves the domain
in `IngestionStatus.SWAPPING` with the old generation
tombstoned-but-intact; it is recovered automatically (see below),
never silently lost. A transient in-swap window does remain (between
steps 1 and 3 a reader briefly sees neither generation); closing it
requires a generation pointer-flip — a future swap mode. `TOMBSTONE`
trades that window for a far simpler crash-safe mechanism. It is
honored identically by all in-tree vector stores (Memory, FAISS,
PgVector, Chroma).

### Reading during a swap

`query()` and `hybrid_query()` hide tombstoned chunks by default on
both the vector and hybrid (native and client-side) paths. Pass
`include_stale=True` to a query to see them (introspection /
debugging). Consumers reading through `service.py` or the retrieval
helpers inherit the exclusion automatically.

### Recovering an interrupted swap

If the process is killed between the upsert and the commit (a
SIGKILL bypasses Python-level rollback), the domain is left in
`IngestionStatus.SWAPPING`: the old generation is tombstoned but
fully intact, and orphan new-generation chunks carrying the crashed
swap's token may be present. Recovery is automatic — the next
`ingest()` or `ingest_changes()` for that domain reconciles **before**
applying anything: it restores the previous generation to visibility
and drops exactly the crashed swap's orphans by the persisted token.

For a domain that will not be re-ingested soon, trigger recovery
explicitly:

```python
recovered = await manager.reconcile("my-domain")
# True if an interrupted swap was reconciled; False if there was
# nothing to do (idempotent — safe to call unconditionally).
```

### `clear_existing=` is deprecated

`KnowledgeIngestionManager.ingest(clear_existing=)` is deprecated in
favor of `swap_mode=`. `clear_existing=True` maps to `CLEAR_FIRST`,
`False` to `APPEND`; passing it emits a `DeprecationWarning`. With
neither argument the default is unchanged (`CLEAR_FIRST`).

## Rate-Limiting the Embedder

Each chunk is embedded with one provider call. Against a local
Ollama embedder that is free; against a rate-limited hosted
embedding API a large corpus can burst past the provider's limit and
fail the whole ingest. Pass an optional
`dataknobs_common.ratelimit.RateLimiter` to pace the ingest-path
embed calls — every per-chunk embed is preceded by
`await rate_limiter.acquire("embed")`.

```python
from dataknobs_common.ratelimit import create_rate_limiter
from dataknobs_bots.knowledge import KnowledgeIngestionManager

# e.g. 60 embeds/minute to stay under a hosted provider's quota
limiter = create_rate_limiter({"rates": [{"limit": 60, "interval": 60}]})

manager = KnowledgeIngestionManager(
    source=backend,
    destination=rag,
    rate_limiter=limiter,
)
# Every ingest()/ingest_changes()/ingest_if_changed() now paces embeds.
```

The same `rate_limiter=` is available keyword-only on
`RAGKnowledgeBase.ingest_from_backend(...)` for direct callers.
`None` (the default) is unchanged behaviour — no pacing — so local
Ollama deployments need no configuration. Pacing applies to the
ingest path only; query/hybrid-search embedding is not throttled.

## Status Tracking

`KnowledgeIngestionManager` transitions the domain's
`ingestion_status` on the backend:

- `IngestionStatus.INGESTING` before processing
- `IngestionStatus.SWAPPING` during a `TOMBSTONE` swap (old + new
  generations coexist; reads are served from the new one). A domain
  left here by a crash carries the in-flight swap token on
  `KnowledgeBaseInfo.generation` and is auto-reconciled by the next
  ingest (or `manager.reconcile(domain_id)`).
- `IngestionStatus.READY` on success
- `IngestionStatus.ERROR` on failure (with the error message)

`set_ingestion_status` accepts an `IngestionStatus` member (the
typed, preferred form) or its string value; an unrecognized string
raises `ValidationError` (a `dataknobs_common.exceptions`
`DataknobsError`, not a `ValueError` subclass), carrying the list of
accepted values. Backends expose `get_info(domain_id)` to read the
current status.

## Completion Events

When an `event_bus` is wired into the manager, every successful ingest
fires an `ingest:domain:end` event (fanned out to the bus):

```python
Event(
    type=EventType.CUSTOM,
    topic="ingest:domain:end",
    payload={
        "domain_id": "my-domain",
        "files_processed": 12,
        "chunks_created": 47,
        "files_deleted": 0,
        "status": "completed",
        "completed_at": "2026-06-17T12:00:00+00:00",
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
