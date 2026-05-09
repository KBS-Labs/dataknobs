# Changelog

All notable changes to the dataknobs-bots package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **`VectorMemory(immutable_metadata_keys=...)`** — declares which
  `default_metadata` keys cannot be overridden by caller-supplied
  `metadata` on `add_message()`. Use for tenant-scoping identifiers
  (e.g. `immutable_metadata_keys=["user_id"]` paired with
  `default_metadata={"user_id": "..."}`). Caller-attempted overrides
  are logged as warnings and the configured value is preserved.
  Plumbed through `VectorMemory.from_config()`.

- **`VectorMemory.clear(filter_metadata=...)`** — filter-aware
  clear. When called with no args on a `VectorMemory` constructed
  with `default_filter=...`, the default filter is auto-applied,
  making `mem.clear()` symmetric with `mem.get_context()` for
  tenant-scoped instances. Pass `filter_metadata=...` explicitly to
  scope a clear to a different subset (e.g. one
  category/conversation within a tenant).

- **`RAGKnowledgeBase.clear(filter=...)`** — filter-aware clear,
  passing through to the underlying `VectorStore.clear(filter=)`.

### Fixed

- **`RAGKnowledgeBase._embed_and_store_chunks` no longer lets
  caller `metadata` overwrite system-controlled chunk fields**
  (`text`, `source`, `chunk_index`, `document_type`,
  `source_path`). Pre-fix, an ingest call passing
  `metadata={"text": "tampered"}` could silently corrupt stored
  chunks; the bug was reachable through every public ingest entry
  point. Caller-supplied values for system fields are now logged
  as warnings via `dataknobs_common.metadata.enforce_immutable_keys`
  and the system value is preserved.

- **`KnowledgeIngestionManager.ingest(domain_id, clear_existing=True)`
  no longer wipes other domains' chunks in a shared vector store.**
  Pre-fix, the manager called the underlying `VectorStore.clear()`
  with no filter, so refreshing one domain in a multi-tenant store
  removed every other domain's chunks silently. Post-fix, the clear
  is scoped by `domain_id` via
  `RAGKnowledgeBase.clear(filter={"domain_id": domain_id})`.
  Consumer-side workarounds (e.g. defaulting `clear_first=False`
  to dodge the issue) can be reverted on upgrade.

- **`RAGKnowledgeBase._embed_and_store_chunks` chunk IDs are now
  scoped by `domain_id` when present in the threaded metadata.**
  Pre-fix, the chunk-id stem was derived purely from
  `Path(source_file).stem`, so two chunks at the same relative
  filename across different domains (e.g. `domain-a/doc.md` and
  `domain-b/doc.md`) collided on a shared store and the second
  ingest upserted over the first. Post-fix, the chunk-id prefix
  becomes `f"{domain_id}\x1f{stem}"` whenever `domain_id` is in the
  caller-supplied metadata (which `KnowledgeIngestionManager`
  threads automatically). The record-separator (`\x1f`) between
  `domain_id` and `stem` rules out snake_case-domain collisions
  (`my` + `team_doc` vs `my_team` + `doc` would otherwise both
  produce `my_team_doc` under `_`). Single-domain consumers
  (no `domain_id` threaded) see **no change** — chunk IDs keep the
  historical `f"{stem}_{index}"` form, so re-ingest into existing
  populated stores remains idempotent.

- **`RAGKnowledgeBase.ingest_from_backend` no longer threads the
  redundant `source` and `filename` keys** that
  `KnowledgeBaseConfig.get_metadata` adds, into
  `_embed_and_store_chunks`. The chunk-build step already receives
  the more-precise `source_file` (display URI) and `source_path`
  (relative path) explicitly; dropping the redundant copies stops
  the new immutable-key enforcer from emitting a spurious warning
  on every legitimate ingest.

### Changed

- **`VectorMemory.clear()` semantics on tenant-scoped instances.**
  When `default_filter` is set, `clear()` (no args) now removes
  only the matching tenant's vectors, not the entire store. The
  pre-fix unscoped behavior was a documented gap (Brief 118
  sub-issue 8b); the docs steered consumers away from production
  `clear()` because it could not respect tenant scoping. This is
  a behavior change for tenant-scoped instances — consumers who
  genuinely want to wipe all tenants from a shared store should
  call `mem.vector_store.clear()` directly (bypassing the
  `VectorMemory` wrapper).

- **`VectorMemory.clear(filter_metadata=...)` now AND-composes
  with `default_filter` instead of replacing it.** Pre-fix, an
  explicit `filter_metadata` argument took full precedence over
  the memory's `default_filter`, allowing a tenant-scoped instance
  to wipe other tenants' rows in a shared store via an explicit
  override (e.g. tenant-A's memory could call
  `clear(filter_metadata={"user_id": "B"})` and remove tenant B's
  data). Post-fix the filters AND-compose, so explicit filters
  narrow WITHIN the tenant scope and never escape it. On key
  collision (caller passes a key that conflicts with the default)
  the merged filter contains contradictory clauses and matches
  nothing — the clear is a no-op rather than a cross-tenant wipe.

- **`KnowledgeBase` ABC now declares `clear(filter=...)`** with a
  default `NotImplementedError`. `RAGKnowledgeBase` overrides it
  with the filter-aware delete path. Subclasses that don't support
  deletion get a clean error rather than being silently
  mis-driven by managers like `KnowledgeIngestionManager`.

### Fixed

- **`MarkdownChunker.ChunkMetadata.to_dict()` no longer lets
  `custom` overwrite structured fields.** Pre-fix, `to_dict` ended
  with `**self.custom`, so a custom entry sharing a key with a
  structured field (`headings`, `chunk_index`, `chunk_size`,
  etc.) silently overwrote the structured value in the serialized
  dict — same vulnerability class as the pre-118 `_create_chunk`
  `node_type` defense, but covering the entire system-field
  surface. Post-fix, `**self.custom` is unpacked first so
  structured fields win.

- **`RAGKnowledgeBase._embed_and_store_chunks` chunk-id separator
  switched from `_` to `\x1f` (ASCII unit separator)** to
  eliminate snake-case-domain collisions. Pre-fix, the
  underscore-joined prefix caused
  `domain_id="my"`+file `team_doc.md` to collide with
  `domain_id="my_team"`+file `doc.md` (both produced
  `my_team_doc_0`). The unit-separator character cannot appear in
  domain IDs or file stems, so collisions are structurally
  impossible. Chunk IDs are not part of any documented public
  surface, so this is a safe internal change.

- **`RAGKnowledgeBase._embed_and_store_chunks` strips redundant
  `source` / `filename` keys from caller metadata at the shared
  layer.** Pre-fix, the strip lived only in
  `ingest_from_backend`, so direct callers of
  `load_markdown_text(metadata={"source": "..."})` still
  triggered a spurious immutable-key warning even though the
  caller's `source` was a redundant copy of the explicit
  `source_file` argument (different views of the same file). Now
  every entry point benefits.

- **Immutable-key warnings are emitted once per offense, not once
  per chunk.** Pre-fix, the per-chunk loop in
  `_embed_and_store_chunks` invoked `enforce_immutable_keys` on
  every chunk, so an N-chunk document with one bad metadata blob
  emitted N identical warnings. Post-fix, the helper is invoked
  with `caller=metadata` on the first chunk only (warning
  emission) and `caller=None` on subsequent chunks (silent
  enforcement) — one warning per offense.

### Migration

- Callers who currently rely on `default_metadata` for tenant
  scoping should add `immutable_metadata_keys=[...]` matching the
  scoping keys. Existing callers who do not set
  `immutable_metadata_keys` see no behavior change for
  `add_message` — caller metadata still wins on every key (the
  pre-fix default).
- Callers who relied on `VectorMemory.clear(filter_metadata=...)`
  as a "broader" wipe than `default_filter` (e.g. a tenant-A memory
  passing `filter_metadata={"category": "X"}` expecting to wipe
  category X across ALL tenants in the shared store) must update
  their code: the explicit filter now narrows WITHIN the tenant
  scope. For an all-tenants wipe, drop down to the underlying
  vector store: `mem.vector_store.clear(filter={"category": "X"})`.
- Callers of `RAGKnowledgeBase` ingest methods who passed
  caller-`metadata` containing `text`/`source`/`chunk_index`/
  `document_type`/`source_path` (a bug-shaped pattern) must update
  their code: those keys are now system-controlled and caller
  values are logged as warnings and discarded.
- **`VectorMemory.clear()` on tenant-scoped instances now
  auto-applies `default_filter`.** Code that called `clear()` to
  wipe an entire shared store (regardless of tenant scoping) will
  now wipe only the calling tenant's slice. Consumers who meant
  the all-tenants wipe should call `mem.vector_store.clear()`
  directly.
- **`KnowledgeIngestionManager.ingest(clear_existing=True)` is now
  safe in shared stores.** Workarounds that flipped
  `clear_existing` to `False` to avoid cross-domain wipes can be
  reverted on upgrade.

### Security
- Bumped minimum `jinja2` requirement from `>=3.1.0` to `>=3.1.6`
  to exclude versions affected by GHSA-cpwx-vrp4-4pq7,
  GHSA-gmj6-6f8f-6699, GHSA-h75v-3vvj-5mfj, and GHSA-q2x7-8rv6-6q7h.

### Added
- `EnsureIngestionResult.duration_seconds` property — counterpart
  to `IngestionResult.duration_seconds`. Computes
  `completed_at - started_at` in seconds. Returns `float` (not
  `float | None`): `EnsureIngestionResult.completed_at` is typed
  as `datetime` with a construction-time default factory, so a
  terminal result's duration is always defined.
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
- `EnsureIngestionResult.completed_at` is typed as `datetime`
  (non-optional) with a construction-time default factory. Every
  terminal state — skip, error, success — produced by
  `KnowledgeIngestionService.ensure_ingested`,
  `KnowledgeIngestionService.ingest_from_config`, and
  `AutoIngestionMixin._ensure_knowledge_base_ingested` carries a
  real timestamp; consumers that serialize via `to_dict()` see a
  consistent `"completed_at"` on every result. The
  ``IngestionResult`` → ``EnsureIngestionResult`` boundary in
  `from_ingestion_result` coalesces a not-yet-completed source
  (`IngestionResult.completed_at is None`) to
  `datetime.now(timezone.utc)` rather than weakening the
  invariant.
- `EnsureIngestionResult.to_dict()` now serializes `started_at`
  (ISO format), `completed_at` (ISO format), and
  `duration_seconds` — bringing it into shape parity with
  `IngestionResult.to_dict()`. Strict superset of prior keys; no
  removed keys.

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
