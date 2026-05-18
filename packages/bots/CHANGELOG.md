# Changelog

All notable changes to the dataknobs-bots package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- **`KnowledgeResourceBackend.list_changes_since(domain_id, version)
  -> ChangeSet`** — file-level diff (added / modified / deleted +
  the current canonical version) between the current knowledge base
  and the snapshot identified by `version` (a `get_checksum()`
  value). `has_changes_since` is now its degenerate case
  (`not (await list_changes_since(...)).is_empty`) rather than a
  separately-implemented sibling.
- **`ChangeSet`** (frozen dataclass: `added` / `modified` /
  `deleted` / `version`, with `is_empty`) and
  **`InvalidVersionError`** (raised when a version predates a
  backend's snapshot retention; consumers fall back to a full
  re-ingest) — exported from `dataknobs_bots.knowledge` and
  `dataknobs_bots.knowledge.storage`.
- **`KnowledgeResourceBackendMixin`** — the shared canonical
  change-detection algorithm (`get_checksum` / `has_changes_since`
  / `list_changes_since` over `list_files()` plus a `_load_snapshot`
  seam). All in-tree backends inherit it; out-of-tree backends mix
  it in for correct behaviour for free. All three in-tree backends
  retain per-version snapshots so `list_changes_since` is a minimal
  file-level diff: `InMemoryKnowledgeBackend` (in-process map),
  `FileKnowledgeBackend` (`_snapshots/<version>.json` written after
  every mutation), and `S3KnowledgeBackend` (snapshot objects, or the
  metadata object's own S3 version history — see
  `change_detection_mode` below). An out-of-tree backend that does
  not override `_load_snapshot` still gets correct (full, non-minimal)
  change *detection* via the version-equality short-circuit.
- **`S3KnowledgeBackend(change_detection_mode=...)`** (also via
  `from_config`, default `"snapshot"`) selects how per-version
  snapshots are resolved: `"snapshot"` writes a small
  `{path: checksum}` object under `{domain}/_snapshots/<version>.json`
  after every mutation (self-contained, any bucket); `"s3_versioning"`
  writes no extra objects and instead walks the metadata object's own
  S3 version history (`ListObjectVersions`) — requires bucket
  versioning enabled, and with it disabled a stale version safely
  falls back to a full re-ingest. An unrecognized mode raises
  `ValueError` (fail closed).
- **`IngestOrchestrator` trigger-payload dispatch.** The trigger
  event payload now selects the ingest entry point: `since_version`
  → `ingest_changes` (per-file delta), `force_full` →
  `ingest(swap_mode=CLEAR_FIRST)` (full re-ingest), otherwise the
  unchanged `ingest_if_changed(last_version)` default. `since_version`
  takes precedence over `force_full`. Payloads using only
  `domain_id` / `last_version` are byte-for-byte unchanged.
- **`IngestionStatus.SWAPPING`** — set by the `TOMBSTONE` swap path
  while the new generation is written; a crash here leaves the
  domain in this state with the in-flight token recoverable.
- **Interrupted-swap auto-reconciliation + `KnowledgeIngestionManager.
  reconcile(domain_id) -> bool`.** A process crash between the upsert
  and the commit of a `TOMBSTONE` swap leaves the domain in
  `SWAPPING` with the old generation tombstoned-but-intact and orphan
  new-generation chunks possibly present. The next `ingest()` /
  `ingest_changes()` for that domain now reconciles *before* applying
  anything — restoring the previous generation to visibility and
  dropping exactly the crashed swap's orphans by its persisted
  token — so residue never accumulates and unrelated files are never
  left hidden. `reconcile()` exposes the same recovery as an
  idempotent one-shot for domains that will not be re-ingested soon
  (returns `True` if it reconciled, `False` if there was nothing to
  do). Backed by a new `KnowledgeBaseInfo.generation: str | None`
  field (round-trips through `to_dict`/`from_dict`) and a kw-only
  `generation=` parameter on `KnowledgeResourceBackend.
  set_ingestion_status` (always written through, so any non-SWAPPING
  transition clears a stale token); implemented by the in-memory,
  file, and S3 backends.
- **`KnowledgeIngestionManager.ingest_changes(domain_id,
  since_version, *, progress_callback=None, config=None)`** —
  per-file delta re-ingest. Diffs the source against
  `since_version` (a `get_checksum`/`get_current_version` value),
  purges chunks for deleted *and* modified files, then re-embeds
  only the added/modified files through the same internal apply
  path as a full `ingest()` — so swap semantics cannot diverge
  between the full-domain and per-file routes. An S3 `PutObject`
  on one file in a 100-file corpus now re-embeds one file, not
  the whole corpus. If `since_version` predates the backend's
  snapshot retention (`InvalidVersionError`) it falls back to a
  full re-ingest after a warning — never a silent skip.
- **`IngestionResult.files_deleted`** — count of source files
  whose chunks were removed because the file no longer exists at
  the source (populated by `ingest_changes`; `0` for a full
  `ingest`). Included in `to_dict()` and the `knowledge:ingestion`
  event payload.
- **`RAGKnowledgeBase.ingest_from_backend(file_filter=)`** —
  optional keyword-only `Callable[[KnowledgeFile], bool]`
  predicate, evaluated after the pattern match, restricting
  enumeration to a subset of the backend's files. `None`
  (default) is unchanged behavior. This is the seam
  `ingest_changes` uses to re-embed only the changed files
  through the full pattern/chunking pipeline.
- **`IngestSwapMode`** (`CLEAR_FIRST` / `APPEND` / `TOMBSTONE`)
  plus a keyword-only `swap_mode=` on
  `KnowledgeIngestionManager.ingest()` and `ingest_changes()`
  (exported as `dataknobs_bots.knowledge.IngestSwapMode`).
  `TOMBSTONE` is a crash-safe re-ingest: the existing (scoped)
  chunks are marked `_stale` (hidden from reads), the new
  generation is ingested under distinct generation-keyed chunk
  ids so it never overwrites the old rows, and the old
  generation is physically retired **only on a clean commit** —
  on a raised error or partial-error ingest the rollback drops
  the new generation by its token and restores the old one. The
  old generation is never overwritten or deleted before the new
  one commits, so a crash, a raised error, or a racing
  same-domain re-ingest always leaves a fully restorable
  previous generation (unlike the `CLEAR_FIRST`
  delete-then-insert). A crash mid-swap leaves the domain in
  `IngestionStatus.SWAPPING`, auto-reconciled by the next ingest
  (or `KnowledgeIngestionManager.reconcile`). Honored identically
  by all in-tree vector stores (Memory, FAISS, PgVector, Chroma);
  `ingest_changes(swap_mode=TOMBSTONE)` scopes the swap to
  exactly the changed/deleted files. A transient in-swap read
  window remains (closing it needs a generation pointer-flip,
  a future mode).
- **`RAGKnowledgeBase.query(..., include_stale=False)`** and
  **`hybrid_query(..., include_stale=False)`** — a single shared
  read chokepoint hides chunks tombstoned by an in-progress
  `TOMBSTONE` swap on **both** read paths (vector search and
  hybrid, native and client-side fusion); `include_stale=True`
  returns them. `service.py` / retrieval inherit this through
  `query` / `hybrid_query`.
- **`RAGKnowledgeBase.update_metadata_where(filter, set_)`** —
  delegates to the vector store's filter-keyed bulk metadata
  merge; the destination-side primitive the `TOMBSTONE` swap
  uses to mark (and, on rollback, un-mark) a generation without
  enumerating ids.

### Changed

- **`KnowledgeBaseInfo.version`** is now documented as a
  cache-invalidation / display counter only and is **no longer the
  change-detection key** (it is still incremented on every change).
  Change detection uses the canonical content snapshot
  (`get_checksum`). **`KnowledgeIngestionManager.get_current_version()`**
  consequently returns the canonical snapshot identity (a
  `get_checksum` value), not the monotonic counter — so capturing
  it and passing it back to `ingest_if_changed(last_version=...)`
  is now a correct round-trip.
- **`IngestOrchestrator(__init__)`** accepts a new optional
  `lock: DistributedLock | None = None` parameter. Per-domain
  serialization of ingest triggers is now backed by an injected
  `dataknobs_common.locks.DistributedLock` (keyed
  `ingest:<domain_id>`) instead of an internal `asyncio.Lock`. The
  default is `InProcessLock()` — process-local and
  behaviour-identical to prior releases for single-replica
  deployments. Multi-replica deployments must inject a cross-replica
  lock; a process-local lock cannot serialize across replicas. A
  built-in Postgres advisory-lock backend ships in a follow-up phase;
  until then register a cross-replica backend via
  `dataknobs_common.locks.lock_backends`.
- **`KnowledgeResourceBackend.set_ingestion_status`** accepts
  `IngestionStatus | str` (Protocol + memory / file / S3
  backends). The typed enum is the preferred form; legacy
  string values still work and are normalized internally. An
  unrecognized status string now raises
  `dataknobs_common.exceptions.ValidationError` (was a bare
  `ValueError`) — the message enumerates the accepted values, and
  the type is a `DataknobsError`, **not** a `ValueError` subclass,
  so a bare `except ValueError` no longer silently swallows an
  invalid-status bug. Domain-not-found still raises `ValueError`.
  No in-tree caller catches `ValueError` around status
  normalization, so this is contract-tightening only.
- **`RAGKnowledgeBase.count()` excludes tombstoned chunks by
  default.** A mid-`TOMBSTONE`-swap `count(filter)` previously
  delegated straight to the store and reported old+new (≈double)
  while `query()`/`hybrid_query()` only returned the new
  generation. `count()` now returns the read-visible count
  (`count(filter) − count(filter ∧ _stale=True)`, two store-agnostic
  counts); the new kw-only `include_stale=True` restores the prior
  single delegated count (every stored chunk). The numbers differ
  **only** while a swap is in flight; outside a swap there are no
  `_stale` chunks and the result is unchanged.

### Deprecated

- **`KnowledgeIngestionManager.ingest(clear_existing=)`** — pass
  `swap_mode=` (`IngestSwapMode`) instead. `clear_existing=True`
  maps to `CLEAR_FIRST`, `False` to `APPEND`; passing the
  argument emits a `DeprecationWarning`. With neither argument
  set the default is unchanged (`CLEAR_FIRST`), so existing
  callers that omit it are unaffected.

### Fixed

- **`get_checksum()` → `has_changes_since()` round-trip no longer
  spuriously re-ingests.** `has_changes_since` (and so
  `KnowledgeIngestionManager.ingest_if_changed`) previously compared
  the monotonic `KnowledgeBaseInfo.version` counter while
  `get_checksum()` returned a content-snapshot hash — different
  value spaces, so a consumer pairing the two (the intuitive,
  documented usage) always saw "changed" and re-ingested the entire
  domain on every check. Both now derive from the canonical content
  snapshot, so an unchanged knowledge base correctly reports no
  changes across all in-tree backends (memory / file / S3).
- **`IngestOrchestrator` multi-replica race made honest.** The
  previous `asyncio.Lock`-per-domain provided no protection across
  processes, yet the class docstring implied per-domain
  serialization unconditionally. The docstring now states the
  serialization scope is exactly the scope of the injected lock and
  that multi-replica deployments must inject a cross-replica lock.
- **`IngestOrchestrator` per-domain lock-map leak.** The internal
  `dict[str, asyncio.Lock]` was never evicted, so every distinct
  `domain_id` grew it unbounded for the lifetime of the
  orchestrator. The injected `InProcessLock` reference-count evicts
  its key map, closing the leak.
- **`IngestSwapMode.TOMBSTONE` re-ingest is now genuinely
  crash-safe.** Chunk ids were deterministic, so a re-embedded
  file's new chunks upserted *over* the tombstoned old rows in
  place — clearing their `_stale` mark and destroying the old
  generation the instant the new one was written. TOMBSTONE was a
  no-op for the dominant re-ingest case (any file whose content
  changed), a mid-swap crash or partial-error left freshly written
  chunks live with no `_stale` key (leaked partial generation), and
  an `ingest_changes` rollback un-tombstoned the *whole* swap scope
  — resurrecting files that had been deleted at the source. Each
  swap now mints a `uuid4` generation token folded into the new
  chunks' ids and stamped on their metadata (`_generation`), so the
  two generations coexist physically until a clean commit. Rollback
  (raised failure *or* partial error) drops exactly the new
  generation by its token, restores the modified files' old
  generation to visibility, and unconditionally purges files
  deleted at the source (never resurrected). On a clean commit the
  old generation is physically retired. APPEND / CLEAR_FIRST id
  derivation is byte-for-byte unchanged (the token is opt-in by
  presence), so single-domain consumers and existing populated
  stores are unaffected.
- **Native hybrid fusion no longer under-returns mid-swap.**
  `hybrid_query(fusion_strategy="native")` requested exactly `k`
  from the store's `hybrid_search` and *then* dropped tombstoned
  rows, so when `_stale` chunks ranked in the top `k` it returned
  fewer than `k` visible results during a `TOMBSTONE` swap. Both the
  vector and native-hybrid read paths now share a single
  `_fetch_drop_stale_truncate` helper that over-fetches
  `k * _STALE_OVERFETCH` before the stale gate and truncates to `k`,
  so a swap in progress no longer shrinks native-fusion result
  sets. (`_is_stale`'s `None`-guard was also tightened from a
  truthiness check to an explicit `is not None` — same result for
  every real input, but it correctly documents that the guard
  protects against a metadata-less row, not an empty dict.)

## v0.6.20 - 2026-05-13

### Added

- **`Registration.metadata`** — `dict[str, Any]` field on
  `dataknobs_bots.registry.Registration` for cross-cutting context
  (`tenant_id`, audit info, feature flags) that lands in the storage
  backend's ``metadata`` column rather than mixed into the config
  payload.  Round-trips through `to_dict` / `from_dict` and the HTTP
  wire protocol.

- **`RegistryBackend.register(..., metadata=...)`** — kw-only
  parameter routes caller-supplied metadata to the backend's
  metadata channel.  Implemented by `InMemoryBackend`,
  `DataKnobsRegistryAdapter`, and `HTTPRegistryBackend`.

- **Registry filter / pagination surface** on `RegistryBackend`:
  - `list_all(*, status=None, filter_metadata=None, sort=None,
    limit=None, offset=None)` — list with optional status equality,
    equality filter over the metadata column, sort spec, and
    limit/offset pagination.
  - `list_active(...)` / `list_inactive(...)` — symmetric
    convenience wrappers over `list_all` with the status pinned.
  - `count_all(*, status=None, filter_metadata=None)` — routed
    through `AsyncDatabase.count(query)` so backends with pushdown
    counts (`SELECT COUNT(*) WHERE ...`) benefit transparently.
  - `count(*, filter_metadata=None)` / `count_inactive(...)` —
    pinned-status counterparts.
  - `stream(*, status=None, filter_metadata=None, config=None)` —
    async-iterator surface for large tenant populations, yields
    `Registration` instances one at a time.

- **`BotRegistry` surfaces the new metadata / filter / pagination
  surface** so consumers don't drop to ``registry._backend``:
  - ``register(..., metadata=...)`` threads ``metadata`` to the
    backend's metadata channel.
  - ``list_bots(*, filter_metadata=None, sort=None, limit=None,
    offset=None)`` — no-kwarg form returns active bot IDs as
    before; any kwarg routes through ``list_active`` for pushdown
    filtering.
  - ``list_registrations(*, status=None, filter_metadata=None,
    sort=None, limit=None, offset=None)`` — new method surfacing
    full `Registration` objects (timestamps / status / metadata).
  - ``count(*, filter_metadata=None)`` — tenant-scoped counts.

- **`HTTPRegistryBackend` wire-protocol extensions** — optional
  query parameters on `GET /configs`:
  `?filter_metadata=<URL-encoded JSON object>` (sorted keys for
  deterministic cache lines), `?status=<value>`,
  `?sort=<field>[:asc|desc]` (repeatable; wire order is tie-break
  order), `?limit=<int>`, `?offset=<int>`.  Schema is **additive
  optional**: servers that recognize a parameter honor it; servers
  that don't ignore it and return the broader list.  The client
  defensively re-applies idempotent filters (`filter_metadata`,
  `status`, `sort`) after parsing the response; `limit`/`offset`
  are intentionally NOT re-applied client-side (re-offsetting a
  server-paginated window would drop live rows).

- **`POST /configs/{bot_id}/deactivate`** — new server-side
  endpoint that routes directly to ``RegistryBackend.deactivate``.
  Lets HTTP clients soft-delete without first issuing
  ``GET /configs/{bot_id}`` (which bumps ``last_accessed_at``).
  Returns ``204 No Content`` on success or ``404 Not Found``.

- **`create_registry_router(backend)`** — reference FastAPI router
  in `dataknobs_bots.registry.server` exposing `RegistryBackend` as
  the wire protocol that `HTTPRegistryBackend` speaks.  Consumers
  can stand up a config service backed by any `RegistryBackend`
  (`InMemoryBackend`, `DataKnobsRegistryAdapter` over
  Postgres/SQLite/S3, …) with one line of glue.  FastAPI is an
  optional dependency: importing the module without it installed
  succeeds; calling `create_registry_router` raises `ImportError`
  with an install hint (`pip install 'dataknobs-bots[server]'`).
  Protocol is pinned on both sides by client and server test
  suites — drift breaks both.

- **`ArtifactRegistry.query`** — kw-only `filter_metadata=`,
  `sort=`, `limit=`, `offset=` parameters.  Filter / sort push down
  to the database query so SQL backends can use indexes.  Pagination
  is applied **after** the latest-pointer dedup pass (dual-write
  storage shape — pre-dedup row count diverges from post-dedup
  artifact count, so a pushdown ``LIMIT`` is unsafe).  Existing
  positional parameters (`artifact_type`, `status`, `tags`,
  `filters`) unchanged.

- **`ArtifactRegistry.count`** — new method mirroring `query`
  parameter-for-parameter (minus sort/limit/offset).  Equivalent
  to ``len(await registry.query(...))`` after dedup.

- **`RubricRegistry.list_all` / `RubricRegistry.get_for_target`** —
  kw-only `filter_metadata=`, `sort=`, `limit=`, `offset=`.  Same
  post-dedup pagination policy as `ArtifactRegistry.query` (same
  dual-write storage shape).

- **`RubricRegistry.count_for_target` / `RubricRegistry.count_all`**
  — new count methods mirroring the corresponding list/get methods.

- **`GeneratorRegistry.list_definitions`** — kw-only
  `filter_metadata=`, `sort=`, `limit=`, `offset=`.  Unlike the
  dual-write registries, `GeneratorRegistry` writes a single row
  per generator id — no pointer/snapshot divergence — so
  limit/offset push down to the database directly.

- **`GeneratorRegistry.count_definitions`** — new method that routes
  through `AsyncKeyedRecordStore.count`, letting backends with
  pushdown counts skip row materialization.

### Changed

- **`DataKnobsRegistryAdapter`, `ArtifactRegistry`, `RubricRegistry`,
  and `GeneratorRegistry` now compose `AsyncKeyedRecordStore`** (from
  `dataknobs-data`) instead of building `Record(...)` instances
  inline.  The store's
  ``(T) -> (data, metadata)`` serializer signature makes the
  metadata channel part of the function's type, so a future change
  to a model can't accidentally drop the metadata channel without a
  type-visible diff at the serializer site.  Public surface
  preserved; the `DataKnobsRegistryAdapter` stored shape differs —
  see Migration below.

### Fixed

- **`DataKnobsRegistryAdapter` now persists caller-provided
  metadata to the `Record.metadata` column.**  Previously the
  metadata column was always empty (there was no
  `Registration.metadata` field), rendering `metadata.X` filters
  and the Postgres metadata GIN index unreachable.  Multi-tenant
  consumers can now use `filter_metadata={"tenant_id": ...}` to
  scope `list_active` / `list_all` queries.

- **`ArtifactRegistry` and `RubricRegistry` now persist artifact /
  rubric `metadata` to the `Record.metadata` column** (latent
  defect — no consumer had hit it yet).

- **`GeneratorRegistry` no longer silently routes definition
  fields into the `data` column under a `metadata` variable
  name.**  The pre-fix code passed a local variable named
  ``metadata`` positionally to ``Record(...)``, but ``Record(...)``'s
  first positional is ``data`` — so the schema/version/id fields
  landed in the data column and the record's metadata column was
  never populated.  Migrating to `AsyncKeyedRecordStore` removes
  the inline `Record(...)` call, so the variable-name shadow
  cannot recur and `GeneratorDefinition.metadata` lands in the
  correct column.

- **`DataKnobsRegistryAdapter.count()` no longer materializes
  every active row** to compute its result.  It now routes through
  `_db.count(query)`, so backends with `SELECT COUNT(*)` pushdown
  return without row materialization.

- **`HTTPRegistryBackend.register` and `.deactivate` no longer
  issue touching reads.**  Previously both methods called
  ``await self.get(bot_id)`` first — the corresponding
  ``GET /configs/{bot_id}`` route bumps ``last_accessed_at`` per
  the `get` protocol contract, so every re-register and every
  soft-delete contaminated the user-activity signal that timestamp
  is supposed to carry.  `register` now issues a single
  ``PUT /configs/{bot_id}`` (upsert); `deactivate` calls the new
  ``POST /configs/{bot_id}/deactivate`` endpoint.

- **`ArtifactRegistry.revise` / `set_status` / `submit_for_review`
  are now serialized per artifact id**, closing an in-process
  read-modify-write race.  Two concurrent ``revise(id, …)`` callers
  could both read ``v1.0.0``, both compute ``v1.0.1``, and both
  write the same snapshot key — last-write wins and the losing
  revision silently disappeared.  A per-id ``asyncio.Lock`` now
  wraps each read-modify-write flow.  **Scope:** in-process only.
  Two processes writing to the same backing database still race;
  the multi-process fix (optimistic-version / row-lock check at
  the database layer) is tracked as a separate work item.

- Bumped minimum `pyyaml` requirement from `>=6.0` to `>=6.0.2` to
  exclude versions that lack cp312/cp313 wheels and fail to build
  from source against modern Cython.  Surfaced by the floor
  resolve step in the `dependency-update` workflow.

### Migration

- **Stored record shape for `DataKnobsRegistryAdapter` changed.**
  Pre-migration, every field of the `Registration` was written into
  the ``data`` column and the record's metadata column was always
  empty (there was no ``Registration.metadata`` field).
  Post-migration, `Registration.metadata` is written to the
  record's ``metadata`` column.  Existing deployments must rewrite
  stored rows once before the new `filter_metadata=` / metadata
  pushdown will see anything (the column is empty on pre-migration
  rows).

- **Wire-protocol change is additive.** `Registration.to_dict()`
  / `from_dict()` gained a ``metadata`` key.  Old clients that
  ignore unknown keys keep working against new servers; old
  servers that omit the key produce ``metadata={}`` on the new
  client via ``data.get("metadata") or {}``.  No coordinated
  upgrade is required, but until both sides understand the key,
  the metadata channel is effectively absent on that consumer.

- **New `ArtifactRegistry.query` parameters (`filter_metadata=`,
  `sort=`, `limit=`, `offset=`) are kw-only.**  This is the
  contract for the new surface; positional usage of the
  established parameters (`artifact_type`, `status`, `tags`,
  `filters`) is unchanged.

## v0.6.19 - 2026-05-09

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
