# Changelog

All notable changes to the dataknobs-bots package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- `RegistryBackend.peek_config(bot_id)` — non-mutating sibling of
  `get_config`. Returns the stored config dict without updating
  `last_accessed_at`, for inspection / audit / bookkeeping reads
  that should not register as user activity. Implemented on
  `InMemoryBackend`, `DataKnobsRegistryAdapter`, and
  `HTTPRegistryBackend`. The HTTP backend has no client-side
  activity state, so its `peek_config` delegates to `get_config`;
  servers that want to distinguish a non-touching peek may define
  their own contract (header, query parameter, or sibling
  endpoint) — this client deliberately does not impose one.

### Changed
- `BotRegistry.get_config()` now routes through
  `RegistryBackend.peek_config` rather than `get_config`.
  Inspection-style reads no longer bump `last_accessed_at`;
  consumers needing the touching behavior should use
  `BotRegistry.get_bot()`, which is the user-facing resolution
  path.
- `BotRegistry.get_bot()` now touches the backend on every call
  (cache hit and miss alike) so `last_accessed_at` reliably
  reflects user activity. Previously the backend `get_config`
  was issued only on the cache-miss branch, which produced an
  inverted activity signal — hot bots (always cache hits) never
  updated, cold bots updated only on TTL expiry. The change adds
  one backend read per `get_bot` call; for the HTTP backend that
  is one extra round trip per call, for the
  `DataKnobsRegistryAdapter` it is one extra `db.read` plus the
  pre-existing `db.update` that `get_config` already performed.
- `ConfigCachingManager.get_raw_config()` now routes through
  `RegistryBackend.peek_config`. Bypassing the cache also bypasses
  the activity bump, matching the inspection-path role the method
  already documents.
- `CachingRegistryManager.get_or_create()` cache-miss reads now
  route through `RegistryBackend.peek_config`. Previously
  `last_accessed_at` was bumped only on cache misses (cache hits
  bypass the backend), producing an inverted activity signal —
  hot bots never updated, cold bots updated only on TTL expiry.
  Storage timestamps now reflect direct backend reads only;
  user-activity tracking for `CachingRegistryManager` consumers
  belongs at the `get_or_create` caller (or higher) — if your
  deployment relied on cache-miss-as-activity, call
  `backend.get_config()` directly in the request-handling path,
  or migrate the call site to `BotRegistry.get_bot()` (which now
  bumps unconditionally).
- Non-UTF-8 backend bytes for a knowledge-base config raise
  `IngestionConfigError` from
  `RAGKnowledgeBase._load_kb_config_from_backend`. Previously a
  stray `UnicodeDecodeError` could escape this path.

### Internal
- `RAGKnowledgeBase._load_kb_config_from_backend` uses
  `dataknobs_common.config_loading.parse_yaml_or_json` for the
  bytes → dict parse. Surface is `IngestionConfigError`.

## v0.6.18 - 2026-05-06

## v0.6.17 - 2026-04-29

### Added
- `RAGKnowledgeBase.ingest_from_backend(backend, domain_id,
  config=None, progress_callback=None, extra_metadata=None)` —
  unified ingest for any `KnowledgeResourceBackend` (file, memory,
  S3) with full `KnowledgeBaseConfig` support: patterns, exclude
  patterns, per-pattern chunking overrides, streaming JSON/JSONL.
  When `config` is `None`, auto-loads
  `knowledge_base.(yaml|yml|json)` from the domain root (falling
  back to `_metadata/knowledge_base.*`); a malformed config raises
  `IngestionConfigError`. `extra_metadata` is merged onto every
  chunk — `KnowledgeIngestionManager` uses this to thread
  `{"domain_id": domain_id}` onto chunks so multi-tenant queries
  can filter on it.
- `IngestOrchestrator` (`dataknobs_bots.knowledge.orchestration`) —
  subscriber-side primitive that listens on an `EventBus` trigger
  topic and dispatches to
  `KnowledgeIngestionManager.ingest_if_changed`. Concurrent triggers
  for the same `domain_id` are serialized via a per-domain
  `asyncio.Lock`; different domains proceed in parallel. Stateless
  across restarts; trigger adapters (S3/SQS/cron → bus) remain
  consumer responsibility.
- `BackendDocumentSource` (re-exported from
  `dataknobs_xization.ingestion`) — adapts any
  `KnowledgeResourceBackend` to the `DocumentSource` protocol.
  Derives a common literal prefix from configured patterns and
  passes it to `backend.list_files(prefix=...)` so S3 (and any
  other prefix-aware backend) can avoid listing the whole bucket.
- `KnowledgeIngestionManager.ingest_if_changed(domain_id,
  last_version=None)` returning `IngestionResult | None` —
  returns `None` (and skips the ingest) when `last_version` is
  supplied and the backend reports no changes.
- `S3KnowledgeBackend` accepts a pre-built
  `session_config: S3SessionConfig` kwarg for sharing a single S3
  configuration across multiple backends.

### Changed
- `KnowledgeIngestionManager.ingest()` delegates to
  `RAGKnowledgeBase.ingest_from_backend` and threads
  `{"domain_id": domain_id}` into per-chunk metadata so downstream
  queries can filter by tenant.
- `S3KnowledgeBackend` `region` default flipped from `"us-east-1"`
  to `None`; client routes through `create_boto3_s3_client`. See
  `dataknobs-data` notes above for the behavior-change details and
  migration guidance.
- `S3KnowledgeBackend.from_config` accepts both `region` and
  `region_name` keys (parity with `SyncS3Database` /
  `AsyncS3Database`).
