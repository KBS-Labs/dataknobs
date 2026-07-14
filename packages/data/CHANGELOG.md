# Changelog

All notable changes to the dataknobs-data package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- `create()` is now a defined atomic insert across every backend: a colliding
  id raises `DuplicateRecordError` instead of silently overwriting the existing
  record (memory, file, S3, Elasticsearch) or raising a bare `ValueError`
  (SQLite, DuckDB). This removes the racy `exists()`-then-`create()` workaround
  consumers needed for collision-safe inserts. `DuplicateRecordError` subclasses
  `ValueError`, so existing callers that caught the former `ValueError` on a
  duplicate id are unaffected. On S3 the guarantee is enforced with a
  conditional PUT (`If-None-Match`) and therefore holds against any S3
  implementation that honors conditional writes (real AWS S3, recent
  LocalStack); both a pre-existing key (412) and a concurrent conditional-write
  race (409) fail closed as `DuplicateRecordError`, while older stores that
  ignore the header degrade to last-writer-wins. `create_batch()` honors this
  contract on some but not all backends — see the `create_batch()` and streaming
  entries below for the per-backend detail.
- On the SQL backends (SQLite, DuckDB), `create()` now distinguishes a
  duplicate-id collision from other column-constraint violations: only a
  primary-key collision raises `DuplicateRecordError`, while a `NOT NULL` or
  `CHECK` violation on the stored data surfaces as `RecordValidationError`
  rather than being mislabeled as a duplicate id.

### Added

- `DuplicateRecordError` (exported from `dataknobs_data`), raised by `create()`
  on a duplicate id. Subclasses both the data-layer `ConcurrencyError` and
  `ValueError`; carries the colliding id in `.id` and `context={"id": ...}`.
- `ConcurrencyError.__init__` accepts an optional keyword-only `context` mapping
  (backward-compatible), so concurrency conflicts can carry structured detail.
- `get_version(id) -> str | None` on `AsyncDatabase` / `SyncDatabase` and every
  backend: returns an opaque, backend-local optimistic-concurrency token for a
  stored record (or `None` if the id does not exist). The token is native where
  the store provides one — an in-memory per-instance monotonic sequence,
  PostgreSQL `xmin`, Elasticsearch `_seq_no`/`_primary_term`, S3 `ETag` — and a
  deterministic content hash of the stored record on the file, SQLite, and
  DuckDB backends. The in-memory token is ABA-safe on every path, including a
  delete→recreate at the same id (the sequence value is never reused). Treat it
  as opaque; it is not comparable across backends.
- An opt-in, keyword-only `expected_version` parameter on `update()`,
  `upsert()`, and `delete()` across both base contracts and all 14 backends.
  Passing a token read from `get_version()` turns the write into a
  compare-and-set: it proceeds only if the record's current token still matches,
  otherwise it raises `ConcurrencyError` (carrying `id` / `expected_version` /
  `actual_version` in `.context`) instead of last-writer-wins. The
  compare-and-set is enforced atomically where the store supports it —
  PostgreSQL `WHERE ... AND xmin = …`, Elasticsearch `if_seq_no`/`if_primary_term`,
  S3 `If-Match` (on both the conditional PUT and the conditional DELETE) — and
  the in-process content-hash backends serialize the check within a single
  connection/instance. A conditional `update()` never inserts (a missing record
  returns `False`); a conditional `delete()` never conflicts on an absent id (a
  missing record returns `False`); a conditional `upsert()` never inserts (a
  missing record is itself a conflict and raises). Omitting `expected_version`
  leaves all three operations byte-identical to prior behavior (unconditional
  last-writer-wins). `upsert()` applies the update through the backend's own
  atomic guard and acts on its result, so a concurrent delete cannot make it
  report success without writing.
- The database backends advertise their optional consistency features through
  the `CapabilityContract` surface: `AsyncDatabase` / `SyncDatabase` and every
  concrete backend report `Capability.CONDITIONAL_WRITE`. A consumer can query
  `db.supports(Capability.CONDITIONAL_WRITE)` (or use `require_capability`)
  before relying on `expected_version` compare-and-set, instead of knowing the
  backend matrix out-of-band. The advertisement is uniform because every
  backend enforces the contract; the ABA nuance of the content-hash backends
  is documented, not encoded as a separate capability.
- `Migrator` gains an `on_conflict` policy (`insert` / `upsert` / `skip`) for
  idempotent re-runs into a populated target. `insert` (the default) fails
  closed on a colliding id as before; `upsert` overwrites the target row;
  `skip` leaves the existing row and counts the id as skipped. The policy is
  threaded through all four migrate methods — `migrate()` and
  `migrate_parallel()` take it directly, `migrate_stream()` / `migrate_async()`
  read it from `StreamConfig`. Default behavior is unchanged.
- `ConflictPolicy` enum and `StreamConfig.on_conflict` field (exported from
  `dataknobs_data` and `dataknobs_data.migration`) carry the policy on the
  streaming path; every backend's `stream_write` honors it. `StreamResult`
  gains a `skipped` counter. The `insert` fast-path uses the backend's native
  batch write; `upsert` uses the native `upsert_batch` bulk verb (see below)
  with a per-record `upsert` fallback; `skip` writes one record at a time (a
  whole-batch verb cannot skip individual dupes while inserting the rest). An
  unknown `on_conflict` value is rejected when the `StreamConfig` is built.
- `upsert_batch(records)` on `AsyncDatabase` / `SyncDatabase` and every backend
  — the batch sibling of `create_batch`, with upsert (insert-or-overwrite)
  semantics: it honors a caller-supplied `record.id` (minting one only when
  absent), overwrites a colliding id (never raised, never skipped), returns ids
  in input order, and carries no version check (a whole batch cannot carry one
  optimistic-concurrency token). Native bulk fast-paths where the store has one
  — a single `INSERT ... ON CONFLICT (id) DO UPDATE` on SQLite, DuckDB, and
  PostgreSQL; a bulk index-by-id on Elasticsearch; a single file-rewrite (file)
  / single-lock pass (memory) — and the per-record ABC-default loop (per-key
  PUT) on S3, which has no cheaper bulk verb. The streaming `upsert` policy and
  the FSM `DatabaseResource.commit_batch` identity path both adopt it for batch
  throughput. `BufferedTransaction` gains a matching `upsert_batch` staging
  method (it still flushes staged upserts row-by-row).
- `create_batch()` on the **memory and file** backends now fails closed on a
  colliding id, matching single `create()`: a colliding id — against an existing
  record or a duplicate within the same batch — raises `DuplicateRecordError`
  before any record is written (the batch is all-or-nothing), and the record's
  own id is honored. Memory and sync-file previously overwrote on collision;
  async-file previously minted a fresh id and discarded `record.id`.
- As a result, the **streaming INSERT** path (`migrate_stream` /
  `migrate_async` under the default `ConflictPolicy.INSERT`) now fails closed on
  a colliding id — recording it as a failure and preserving the source id — on
  the **memory and file** backends (via the `create_batch` fix above). A re-run
  into a populated memory/file target records the colliding ids as failures
  rather than silently overwriting them, matching the batched `migrate()` path.
- **Known gap:** streaming INSERT into a **SQLite, DuckDB, PostgreSQL, S3, or
  Elasticsearch** target is still *not* fail-closed — their streaming batch
  write mints a fresh id per record, so a colliding source id is written as a
  new row under a new id (the source id is not preserved). Use `upsert` / `skip`
  for idempotent re-runs into those targets, or the batched `migrate()` path
  (single `create()`) for id-preserving, collision-safe inserts. Closing this
  gap is tracked as a follow-up (bringing each backend's bulk batch write into
  line with the `create()` contract).

### Fixed

- The async Elasticsearch backend's `create_batch()` and `upsert_batch()` now
  reconcile the bulk response per item, so a record whose bulk operation failed
  (e.g. a mapping error or version conflict) is no longer reported as written. A
  partial bulk failure previously returned every input id as successful — the id
  list is now filtered to the operations that actually succeeded, matching the
  sync backend's `_execute_bulk_index` reconciliation (extracted here into a
  shared `_extract_bulk_index_ids` helper used by both async bulk paths).
- Streaming now accounts a partial-batch failure honestly. When a batch write
  verb confirms fewer ids than the batch it was given — which a bulk backend
  can do (Elasticsearch reports per-item errors, so its `create_batch` /
  `upsert_batch` return only the ids that succeeded) — the unconfirmed records
  are counted as `failed` instead of silently vanishing from the tally, so
  `StreamResult.total_processed == successful + failed + skipped` holds. The
  shortfall is routed through `StreamConfig.on_error` once as an aggregate
  error (with a `None` record, since per-item identity is not available at this
  layer), placing the batch path on the same stop/continue contract as the
  per-record fallback: a configured handler decides whether to continue, and
  with no handler the stream quits on the first failing batch — the same
  fail-stop default a per-record failure already gets.

### Notes

- The file/SQLite/DuckDB content-hash token is subject to the classic ABA
  limitation: an A→B→A mutation cycle yields the original token, so a stale
  conditional write in that exact scenario is not detected. The backends with a
  native monotonic version (memory counter, PostgreSQL `xmin`, Elasticsearch
  `_seq_no`, S3 `ETag`) are ABA-safe. The in-process content-hash backends
  enforce the compare-and-set within a single connection/instance; conditional
  writes are not hardened across separate processes/connections.

## v0.5.5 - 2026-07-07

### Changed

- The AWS session helper is now AWS-generic rather than S3-named, and has
  been relocated to `dataknobs-common` so every AWS consumer across the
  stack shares one implementation. The normalized session config is
  `AwsSessionConfig` (region, credentials, endpoint, retry/pool tuning)
  and now lives in `dataknobs_common.aws` alongside
  `create_aioboto3_session` and `clear_aioboto3_session_cache`. The
  S3-specific surface — `S3PoolConfig`, `create_boto3_s3_client`,
  `validate_s3_session` — stays in `dataknobs_data.pooling.s3`, which
  re-exports the generic names for import stability.
  `create_aioboto3_session` gains a keyword-only `warm_service` parameter
  (default `"s3"`) selecting which service's botocore data files are
  warmed off the event loop; the process-wide warmed-session cache is
  keyed by session kwargs **and** `warm_service`, so distinct services key
  to distinct warmed sessions.

### Deprecated

- `S3SessionConfig` is a deprecated alias for `AwsSessionConfig`.
  Importing it from `dataknobs_data.pooling.s3` emits a
  `DeprecationWarning`; the `dataknobs_data.pooling` package-root alias
  resolves without one (a permanent compatibility alias). Import
  `AwsSessionConfig` from `dataknobs_common.aws` (or the
  `dataknobs_data.pooling` package root) instead.

### Security

- Acknowledged GHSA-f4j7-r4q5-qw2c / PYSEC-2026-311 (CVSS 9.3, pre-auth
  code injection via the `/api/v2/...` endpoint) against the
  `chromadb>=1.0.0` floor (optional Chroma backend), flagged at the
  floor resolve by the `dependency-update` workflow. Affects chromadb
  1.0.0–1.5.9 with no upstream fix. Not exposed in this package:
  `ChromaVectorStore` uses `chromadb.Client` / `chromadb.PersistentClient`
  only (embedded/persistent modes), never `HttpClient` or server mode.
  The inline floor comment in `pyproject.toml` records the rationale.

## v0.5.4 - 2026-06-29

### Added

- **`AsyncDatabase` gains a buffered transaction capability.** A new
  `async with db.transaction(policy="strict"|"emulate") as tx:` context manager
  (and an explicit `await db.begin_transaction(...)` / `tx.commit()` /
  `tx.rollback()` form for staging writes across call sites) defers every write
  (`tx.create` / `tx.create_batch` / `tx.upsert` / `tx.delete`) until commit.
  Two guarantees: an exception before commit persists nothing on *any* backend
  (universal rollback), and on backends whose batch operations run inside a
  backend transaction — SQLite, Postgres, DuckDB — a commit whose staged ops
  reduce to a single coalesced same-kind batch (all creates, or all deletes,
  with no upserts) is all-or-nothing. A **mixed** create/delete or
  **upsert**-containing buffer commits as a *sequence* of independent batches
  and can partially persist if one fails mid-flush; the `tx.is_atomic` property
  reports — from the currently staged ops — which case applies, so a consumer
  needing cross-operation atomicity can branch on it. A new
  `db.supports_transactions()` flag reports which backends wrap a coalesced
  batch in a backend transaction; the three transactional backends return
  `True`, the rest (`memory`, `file`, `s3`, `elasticsearch`) return `False`.
  The `policy` argument chooses what happens on a non-transactional backend:
  `"strict"` (default, fail-closed) raises `CapabilityNotSupportedError`;
  `"emulate"` proceeds with best-effort buffer-and-flush. The handle does
  **not** provide in-transaction isolation or read-your-writes — buffered
  writes are invisible to reads until commit; consumers needing connection-
  scoped isolation should branch on `supports_transactions()` and use a
  backend-native transaction. The new `BufferedTransaction` type and the
  `VALID_TRANSACTION_POLICIES` tuple are exported from `dataknobs_data`.

## v0.5.3 - 2026-06-23

### Fixed

- **`ConnectionPoolManager` now reference-counts pools shared by DSN
  across instances on an event loop.** A new async `release_pool(config)`
  closes and evicts a pool only when its last holder releases.
  `AsyncPostgresDatabase`, `AsyncElasticsearchDatabase`, and
  `AsyncS3Database` `close()` now *release* their shared resource instead
  of either hard-closing it (Postgres — which closed the pool out from
  under sibling instances on the same DSN, so a sibling's `close()` broke
  the others' live connections) or never reclaiming it (Elasticsearch —
  whose pooled client was leaked until process exit under instance churn).
  Concurrent first-time connects on a cold key are serialized by a
  per-event-loop create lock so exactly one pool is created and the holder
  count stays sound under concurrency. Single-holder teardown is
  unchanged; the public `close()` signatures are unchanged.

## v0.5.2 - 2026-06-22

### Added

- **`PgVectorStore` can run against an externally supplied connection
  pool.** Build the store with
  `PgVectorStore.from_components(config, pool=shared_pool)` to hand it an
  asyncpg pool you manage. In that mode `initialize()` runs only the
  schema/table setup against the pool (it does not create one), and
  `close()` leaves the pool open for you to manage and retains the store's
  reference to it — so one pool can back several stores that are opened,
  closed, and reopened independently. Pool ownership is fixed at
  construction: re-initializing a self-owned store after `close()` rebuilds
  its pool, while an injected-pool store reuses the same caller-owned pool
  on reopen (it never fabricates one). The config / connection-string /
  `VectorStoreFactory` path is unchanged: it builds and owns its own pool
  and closes + drops it on `close()`.

### Fixed

- **The aioboto3 session-warm now pre-loads botocore's paginator model so
  `AsyncS3Database.stream_read` (and any aioboto3 paginator consumer) never
  stats for `paginators-1.json` on the event loop on first use.** Client
  creation loads the service model but not the paginator model, so the warm
  also builds a throwaway `list_objects_v2` paginator on its private
  worker-thread loop — the consumer's first real paginator build then reuses
  the session's loader cache instead of blocking the loop. The same pre-load
  also covers the knowledge S3 backend's paginating paths: every aioboto3
  client is built from one cached session (`_SESSION_CACHE`), so warming that
  session's loader once covers all of its paginator consumers.
- **`AsyncSQLiteDatabase.connect` and `AsyncDuckDBDatabase.connect` no
  longer block the event loop creating the database directory.** For a
  file-based database each created the parent directory with a synchronous
  `mkdir` on the running loop; the `mkdir` is now offloaded via
  `asyncio.to_thread`. Behavior is unchanged.
- **`SyncS3Database` and `AsyncS3Database` now sort search results
  correctly when a sort field holds a falsy value such as a numeric `0`.**
  The async backend's inline sort key coerced any falsy value (`0`,
  `False`, `""`) to an empty string, so sorting a numeric field whose
  values included `0` raised
  `TypeError: '<' not supported between instances of 'str' and 'int'`.
  Both S3 backends now apply sorting, `offset`/`limit` (including
  `limit=0`), and field projection through the shared
  `process_search_results` helper, so result ordering is consistent with
  every other backend and the duplicated per-backend logic is gone.

- **The async file database, the in-memory, Chroma, and FAISS vector
  stores, and the shared aioboto3 session factory perform their I/O
  without blocking the event loop.** Each held a synchronous, blocking
  transport behind an `async def`, stalling the loop for the duration of
  the call: `AsyncFileDatabase` ran its locked file load/save (including
  the inter-process `FileLock` acquire) on the loop on every CRUD
  operation, plus its temp-file cleanup on `close()`;
  `MemoryVectorStore.save`/`load` ran their `pickle` disk I/O on the loop
  (and `initialize` an `os.path.exists` stat before loading);
  `ChromaVectorStore` drove the synchronous chromadb client/collection
  directly; `FaissVectorStore.save`/`load` did blocking `faiss` index +
  pickle disk I/O; and `create_aioboto3_session` blocked on session
  construction plus aiobotocore's first-client botocore-data load. All
  now offload their blocking work via `asyncio.to_thread`, and the
  aioboto3 factory additionally warms the session's botocore caches
  off-loop so the first client creation by any consumer
  (`AsyncS3Database`, the SQS event bus, S3-backed knowledge storage) is a
  cache hit. Warmed sessions are cached process-wide by config, so
  consumers that build a session per instance rather than once at startup
  (e.g. a multi-tenant registry loading several runtime configs against
  the same bucket) warm once instead of once per instance. The async and
  sync file backends now share a single synchronous load/save
  implementation. FAISS in-memory `add`/`search`
  remain on the loop — they are CPU-bound and release the GIL internally,
  so offloading them buys nothing. No public signatures changed and no new
  runtime dependency was added (`asyncio.to_thread` is stdlib).

- **`MemoryVectorStore.save` and `FaissVectorStore.save` persist a
  consistent snapshot when a write runs concurrently with the save.**
  Because the disk write is offloaded to a worker thread, each `save()`
  now copies its in-memory state — the vectors / metadata / timestamp
  dicts, plus a clone of the FAISS index — on the event loop *before*
  handing off, so a `save()` that overlaps an `add_vectors` /
  `delete_vectors` records the state as of the `save()` call rather than a
  partially-mutated mix observed mid-serialization. `MemoryVectorStore.save`
  additionally handles a `persist_path` with no directory component (a bare
  filename), which previously failed with `FileNotFoundError`.

### Changed

- The `AsyncDatabase` and `VectorStore` base classes now document an
  async-transport contract — implementations use an async transport or
  offload blocking calls off the event loop (`asyncio.to_thread` /
  `aiter_sync_in_thread`), never blocking `open()` / `os` disk I/O behind
  an `async def`. ruff's `ASYNC` lint family now enforces this for the
  package.
- **`VectorStore.close()` now documents a backing-resource ownership
  contract.** A store that built its own backing resource (connection
  pool, client, session) closes it; a store handed an externally supplied
  resource leaves it open and releases only per-store state. Stores that
  build their backing resource internally (in-memory, FAISS, Chroma)
  satisfy this trivially; `PgVectorStore` honors the contract for its
  caller-supplied connection pool. No behavior change for stores that
  build their own resources.

## v0.5.1 - 2026-06-08

## v0.5.0 - 2026-05-26

### Changed

- **`StreamConfig` is now a frozen `StructuredConfig`.** It gains
  `from_dict()` / `to_dict()` and round-tripping; its existing
  `__post_init__` validation (`batch_size > 0`, `prefetch >= 0`,
  positive `timeout`) is preserved and now also fires on the
  `from_dict()` path. All `StreamConfig(...)` constructors are
  unchanged, but instances are immutable — construct a modified copy
  with `dataclasses.replace(...)` instead of assigning fields.
  `StreamResult` (runtime data) is unaffected.
- **All four vector stores now construct through typed configuration
  dataclasses.** `MemoryVectorStore`, `FaissVectorStore`,
  `ChromaVectorStore`, and `PgVectorStore` each grow a
  `<Backend>VectorStoreConfig` frozen dataclass (a
  `dataknobs_common.structured_config.StructuredConfig` subclass, in
  `dataknobs_data.vector.stores.config`) and are built via the
  `StructuredConfigConsumer` mixin. As a result, **`store.config` is now
  the typed config object, not a dict** — read fields as attributes
  (`store.config.dimensions`) rather than dict lookups. Every existing
  construction shape is preserved: `Backend(config_dict)`,
  `Backend.from_config(config_dict)`, and the `VectorStoreFactory` all
  continue to accept the same dict keys (projected onto the typed
  config), and a typed config may now be passed directly. The common
  keys (dimensions, metric, persistence, batch size, parameter
  sub-dicts, `domain_id`, and a nested `timestamps` config) live on the
  shared `VectorStoreConfig` base; each backend's leaf config adds only
  its own keys. Per-field validation (`id_type`, `index_type`,
  identifier shape, timestamp format) and pgvector connection
  resolution (`connection_string` / `DATABASE_URL` / `POSTGRES_*`)
  surface at construction exactly as before. Mixing a typed `config=`
  with loose keyword arguments raises `TypeError`.
- **The empty-list filter contract is now documented and enforced
  across backends.** An empty-list filter value (`{key: []}`) is an
  unsatisfiable predicate — it matches no record on any vector-store
  backend. This was already true (it backs the deliberate no-op
  `VectorMemory.clear()` uses for tenant isolation) but rested on four
  independent implementations with no shared test; a parametrized
  cross-backend conformance test now guards it so a regression in any
  one backend's filter translation is caught.
- **All 14 database backends now construct through typed configuration
  dataclasses.** Every `SyncDatabase` / `AsyncDatabase` backend (memory,
  sqlite, postgres, elasticsearch, s3, duckdb, file — sync and async)
  grows a `<Backend>DatabaseConfig` frozen dataclass (a
  `dataknobs_common.structured_config.StructuredConfig` subclass) and is
  built via the `StructuredConfigConsumer` mixin. As a result,
  **`db.config` is now the typed config object, not a dict** — read
  fields as attributes (`db.config.table`) rather than dict lookups
  (`db.config["table"]`). Every existing construction shape is preserved:
  `Backend(config_dict)`, `Backend.from_config(config_dict)`, and the
  `database_factory` / `async_database_factory` registries all continue
  to accept the same dict keys (projected onto the typed config), and a
  typed config may now be passed directly. Mixing a typed `config=` with
  loose keyword arguments raises `TypeError`.
- **The sync and async Postgres backends now share one configuration**
  (`PostgresDatabaseConfig`), the union of their parameters. This
  corrects prior drift where only the async backend honored `ssl`
  (see Fixed). `command_timeout` and the pool-size knobs
  (`min_pool_size` / `max_pool_size`) remain async-only — psycopg2 has
  no connect-time equivalent.
- **The sync and async S3 backends now emit a single bucket-required
  error message** (`"S3 backend requires 'bucket' in configuration"`);
  the sync backend previously raised a different string. Both report the
  same message now that bucket validation lives in the shared config.
- **Credential fields are redacted from config `repr`.** Building on the
  `StructuredConfig._SENSITIVE_FIELDS` mechanism in `dataknobs-common`,
  the backend and vector-store configs mask their credentials as `'***'`
  in `repr(config)` (and therefore in logs, tracebacks, and pytest
  failure output): `PostgresDatabaseConfig.password`,
  `AsyncElasticsearchDatabaseConfig.api_key` / `.basic_auth`,
  `S3DatabaseConfigBase.aws_access_key_id` / `.aws_secret_access_key` /
  `.aws_session_token` (inherited by both S3 backend configs),
  `PgVectorStoreConfig.connection_string`, and
  `ChromaVectorStoreConfig.openai_api_key`. `to_dict()` is never redacted,
  so round-trip construction is unaffected.

### Fixed

- **Sync Postgres backend now honors `ssl` configuration.** Previously
  only `AsyncPostgresDatabase` applied `ssl`; `SyncPostgresDatabase`
  silently ignored it. The sync backend now translates the asyncpg-native
  `ssl` value to a psycopg2 `sslmode` (`str` → that mode, `True` →
  `"require"`, `False` → `"disable"`); an unsupported value such as an
  `ssl.SSLContext` raises `ConfigurationError` rather than silently
  connecting without TLS. (Requires `dataknobs-utils` with the new
  `sslmode` connector parameter.)

### Security

- Bumped minimum `pyarrow` requirement (extra: `parquet`) from
  `>=17.0.0` to `>=23.0.1` to exclude PYSEC-2026-113 (CVSS 7.0),
  flagged at the floor resolve by the `dependency-update` workflow.
  The bump preserves the prior sweep of PYSEC-2023-238 (CVSS 9.8) and
  PYSEC-2024-161 (both fixed by 17.0.0).

## v0.4.20 - 2026-05-20

### Fixed
- **`TestS3Backend` LocalStack bucket provisioning** —
  `tests/examples/test_vector_multi_backend.py::TestS3Backend` no
  longer assumes `test-bucket` pre-exists on the LocalStack volume.
  Both `test_s3_sync_backend` and `test_s3_async_backend` now depend
  on the shared `make_localstack_s3_bucket` fixture from
  `dataknobs_common.testing.localstack_fixtures`, which idempotently
  creates the bucket on session entry. Inlined `localstack_host`
  detection blocks removed in favour of the resolved
  `endpoint_url` the fixture provides. Only affects opt-in
  (`TEST_S3=true`) test runs.

## v0.4.19 - 2026-05-18

### Added

- **`FaissVectorStore` timestamp exposure** — `FaissVectorStore` now
  tracks `created_at`/`updated_at` per vector and accepts
  `include_timestamps=True` on `get_vectors()` and `search()`, at
  parity with `MemoryVectorStore` and `PgVectorStore`. Timestamps are
  carried across upserts (created preserved, updated refreshed),
  evicted with the row on delete/`clear`, and persisted in the FAISS
  sidecar pickle (legacy indexes without the side-car load empty and
  surface `None` until the next write — same pre-migration semantics
  as the other backends). Only `ChromaVectorStore` remains deferred.

- **`VectorStore.update_metadata_where(filter, set_) -> int`** — the
  filter-keyed sibling of the id-keyed `update_metadata`. Bulk-*merges*
  `set_` into the metadata of every vector matching `filter` (same
  four-quadrant filter shape as `clear` / `count` / `search`; `None`
  matches all), preserving unrelated metadata keys, and returns the
  number of rows affected. Implemented on **all four in-tree stores**:
  `MemoryVectorStore`, `FaissVectorStore` (side-car merge — FAISS
  filtering is post-retrieval, there is no index to invalidate),
  `PgVectorStore` (`metadata = metadata || $::jsonb`), and
  `ChromaVectorStore` (fetch-merge-`update`). The ABC default raises
  `NotImplementedError` — the contract for **out-of-tree**
  implementers only, so an unported backend fails loudly rather than
  silently mis-applying a zero-downtime swap; it is never reached by
  a backend DataKnobs ships. This is the store-layer primitive behind
  `dataknobs-bots`' `IngestSwapMode.TOMBSTONE` re-ingest.

- **`AsyncS3Database.region`** — public attribute exposing the resolved
  region (`None` when the config relies on the boto default chain), at
  parity with the long-standing `SyncS3Database.region`. Lets callers
  inspect region resolution without reaching into the internal pool
  config.

### Changed

- **Review before upgrade.** `PgVectorStore` now validates the
  existing `embedding` column's vector dimensionality at
  initialization (when the table already exists and
  `auto_create_table=True`). A mismatch between the stored
  `vector(N)` and the configured `dimensions` now raises
  `ConfigurationError` at `initialize()` — naming both dimensions —
  instead of deferring to an opaque `asyncpg.DataError` at the first
  insert. The guard is read-only (it reads
  `pg_attribute.atttypmod`; no schema is altered or dropped).
  Consumers that (incorrectly) relied on the silent
  `CREATE TABLE IF NOT EXISTS` dimension shadow must drop/migrate the
  mismatched table or reconfigure `dimensions`.

### Fixed

- **`ChromaVectorStore` works against chromadb 1.x and no longer
  corrupts non-scalar metadata.** chromadb's metadata contract is
  scalar-only: it rejects an empty/`None` metadata dict, and — the
  dangerous case — *silently accepts* a list/dict-valued metadata
  value then corrupts it, bleeding the value positionally across
  unrelated collections that share chromadb's process-wide in-memory
  `System`. Every list/dict value (including `[]`) is now encoded to a
  reversible JSON sentinel at the Chroma boundary and restored on read
  (the legacy empty-list sentinel still decodes), so chromadb only ever
  stores scalars and the metadata round-trip — `{"k": []}`,
  `{"k": [...]}`, nested dicts — matches `MemoryVectorStore`/
  `FaissVectorStore` with no cross-store contamination. chromadb result
  fields (now numpy arrays) are coerced before truthiness/indexing,
  fixing `get_vectors`/`search` silently returning no rows. List
  filter values are post-filtered (chromadb's where-engine returns
  zero rows for any predicate against list-valued metadata) unless the
  key is declared in `scalar_metadata_keys`; four-quadrant results are
  unchanged. The `chromadb` floor is now `>=1.0.0`.

- **`MemoryVectorStore`/`FaissVectorStore` now own ingested
  metadata** (copy-on-ingest, parity with `PgVectorStore`/
  `ChromaVectorStore` which already serialize on write). Callers may
  safely reuse or mutate the dict they passed to `add_vectors`
  without corrupting store state, and store-internal keys (`_stale`,
  injected timestamps) no longer leak onto the caller's dict.
  (Behavior already in effect since the config-level `domain_id`
  symmetry change via `VectorStoreBase._apply_domain_default`; this
  entry documents the guarantee and adds a cross-backend conformance
  test.)

- **`FaissVectorStore.get_vectors()` returns stored vectors and
  metadata for every index type.** Previously it returned
  `(None, None)` for all ids on `ivfflat`/`ivfpq` indexes
  (auto-selected for embedding dimensions ≥ 100 — the 384/768/1024
  production case); `flat`/`hnsw` were unaffected. The store now keeps
  the authoritative vectors in an internal side-car (same key space
  as its metadata/timestamp stores) and serves `get_vectors` from
  there instead of FAISS reconstruct-by-id, which is not usable for
  IVF without a maintained direct map that this faiss build refuses
  to combine with `remove_ids`. The FAISS index is retained for
  similarity `search`; `get_vectors`, `delete_vectors`, upsert,
  `clear`, and save/reload stay correct for IVF across re-ingest and
  clear/repopulate cycles. A resolved id whose internal id has no
  stored vector (post-delete reuse race) is logged at WARNING rather
  than being silently indistinguishable from an absent id.
  **Migration:** an index persisted by an earlier `dataknobs-data`
  has no stored-vector side-car, so `get_vectors` returns `None` (and
  empty timestamps) for its ids until rebuilt — re-add the vectors
  (or re-ingest) once; `search` is unaffected, and new indexes need
  no action.
- **`FaissVectorStore` no longer crashes when an IVF store's first
  batch is smaller than `nlist`.** Previously a sub-`nlist` first
  `add_vectors` on an `ivfflat`/`ivfpq` store raised
  `RuntimeError: ... 'is_trained' failed` (the train-skip path fell
  through to `add_with_ids` on an untrained IVF index). The store now
  serves a temporary flat index until the corpus reaches `nlist`,
  then trains the real IVF and migrates to it from the side-car —
  search and `get_vectors` stay correct throughout. The deferred
  state is persisted, so a save/reload before the threshold resumes
  correctly.
- **`FaissVectorStore` IVF search now honors the configured
  `nprobe`.** The index is wrapped in `IndexIDMap2`, which does not
  proxy `nprobe`, so the setting never reached the underlying IVF and
  every `ivfflat`/`ivfpq` search ran at FAISS's default `nprobe=1`
  regardless of `search_params.nprobe` — silently degrading recall.
  `search()` now unwraps the inner index and applies `nprobe` there.

### Changed

- **`PgVectorStore` default `schema` changed from `"edubot"` to
  `"public"`** (the PostgreSQL default). **Review before upgrade:**
  deployments that relied on the implicit default were writing to a
  schema named after an unrelated project; after upgrade they will
  use `public`. To retain prior behavior, set `schema="edubot"`
  explicitly in the store config. No in-tree consumer relied on the
  implicit default.

- **`MemoryVectorStore`, `FaissVectorStore`, and `ChromaVectorStore`
  now honor a config-level `domain_id`** (matching `PgVectorStore`).
  A store constructed with `{"domain_id": "x", ...}` defaults
  `domain_id="x"` into the metadata of vectors added without one and
  AND-composes `domain_id="x"` into the effective filter for
  `search()`, `count()`, `clear()`, and `update_metadata_where()`.
  `clear()` (no filter) on a tenant-scoped store now deletes only
  that tenant's rows rather than wiping the whole collection, and an
  out-of-scope explicit `domain_id` filter resolves to a no-match.
  `PgVectorStore` behavior is unchanged (its SQL predicate already
  enforced this). **Review before upgrade:** consumers that
  previously set `domain_id` in the store config on Memory/FAISS/
  Chroma (where it was silently a no-op) now get real tenant
  isolation — `count()`/`search()`/`clear()` will scope to that
  tenant. One residual cross-backend divergence remains and is
  documented in `VECTOR_FILTER_SEMANTICS.md`: an *explicit*
  `filter={"domain_id": "x"}` is a metadata-key match on Memory/
  FAISS/Chroma but a JSONB-containment probe on PgVector (which
  stores the configured tenant in a column, not in JSONB) — rely on
  config-level scoping, not explicit `domain_id` filters, for
  backend-portable isolation.

## v0.4.18 - 2026-05-13

### Added

- **`AsyncKeyedRecordStore[T]` / `SyncKeyedRecordStore[T]`** — generic
  id-keyed persistence over `AsyncDatabase` / `SyncDatabase` for
  registry / pointer-table use cases.  Encapsulates the `Record`
  two-column (`data`, `metadata`) shape *by construction*: the
  serializer signature is ``(T) -> tuple[dict, dict]`` rather than
  ``(T) -> Record``, so the metadata channel is part of the function's
  type and cannot be silently dropped.  Surface: `put`, `get`,
  `exists`, `delete`, `put_batch`, `get_batch`, `delete_batch`,
  `list`, `count`, `stream`, `search`.  Filter channels —
  `filter_data` and `filter_metadata` — both routed through the
  existing `metadata.X` field-path convention so JSONB pushdown
  works on Postgres / SQLite / DuckDB and `Record.get_value`
  traversal works on memory / file backends.  Exported from
  `dataknobs_data` package root.  Composed by
  `DataKnobsRegistryAdapter`, `ArtifactRegistry`, `RubricRegistry`,
  and `GeneratorRegistry` in `dataknobs-bots`, and by
  `UnifiedDatabaseStorage.save_step` in `dataknobs-fsm`, as the
  single Record-construction site for those registries.

### Changed

- **`limit=0` now produces an empty result across every backend**,
  consistent with Python slice semantics (``limit=None`` →
  unlimited, ``limit=0`` → empty).  Previously the pagination paths
  used truthy-checks (``if query.limit:`` / ``if query.offset:``),
  so ``limit=0`` was silently treated as "no limit".  ``offset=0``
  is now also documented as a no-op rather than a slice that copies
  the full list.

  **Migration:** Audit consumers that pass ``limit=0`` explicitly.
  Any caller that relied on the truthy-check to silently mean
  "unlimited" will now receive an empty result; pass
  ``limit=None`` (or omit the argument) for unlimited semantics.

## v0.4.17 - 2026-05-09

### Added

- **`VectorStore.clear(filter=...)`** — filter-aware clear, now
  supported across all four backend implementations
  (`MemoryVectorStore`, `FaissVectorStore`, `ChromaVectorStore`,
  `PgVectorStore`). When `filter` is `None` (default), behavior is
  unchanged — all vectors are removed. When provided, only vectors
  whose metadata matches the filter are removed; non-matching
  vectors are preserved. The filter shape matches `search()` and
  `count()`; each backend reuses its existing filter-translation
  infrastructure (`_match_metadata_filter` for memory/FAISS,
  `_partition_filter_for_chroma` for Chroma,
  `_build_jsonb_filter_sql` for pgvector).

  This closes a long-standing gap where multi-tenant shared stores
  could not perform per-tenant cleanup without scanning IDs in the
  consumer. `KnowledgeIngestionManager` (in `dataknobs-bots`) now
  uses this to scope its automatic clear-before-reingest by
  `domain_id`.

- **`ChromaVectorStore` accepts `scalar_metadata_keys`** — opt-in
  declaration of metadata keys whose stored values are guaranteed
  scalar (never list-valued). For declared keys with scalar filter
  values, `_partition_filter_for_chroma` pushes a Chroma-native
  `$eq` predicate instead of post-filtering in Python.
  `count(filter=...)` then fetches IDs only (no metadata
  materialization) when the filter pushes down fully — eliminating
  the memory-bound trade-off documented in
  `VECTOR_FILTER_SEMANTICS.md` for the common multi-tenant scoping
  pattern (e.g. `{"domain_id": "x"}`). Backward compat preserved:
  keys not declared keep the conservative post-filter behavior.

- **`VECTOR_FILTER_SEMANTICS.md` documents the pgvector
  config-level `domain_id` swap asymmetry** — when runtime-swapping
  between vector-store backends, `PgVectorStore`'s config-level
  `domain_id` scopes `clear()` automatically while the other three
  backends do not. The doc gives explicit guidance for swap-safe
  consumers.

### Fixed

- **`FaissVectorStore.add_vectors` no longer leaks orphan metadata
  on upsert.** Pre-fix, re-adding an external ID overwrote
  `id_map[ext_id]` without removing the prior internal ID's entries
  from the FAISS index or `metadata_store`, leaving silent residuals
  that filtered `clear()` could not reach (it walks `id_map`).
  Post-fix, the prior internal ID is evicted from FAISS and
  `metadata_store` before the new mapping is assigned.

### Migration

- **No source-compat break.** `await store.clear()` continues to
  work and continues to remove all vectors.
- **Backend-specific note (FAISS).** FAISS has no native filtered
  delete; filtered clear iterates `metadata_store` to collect
  matching IDs and delegates to `delete_vectors(ids)`. This is O(N)
  over stored vectors — acceptable for typical KB sizes, but
  workloads at scale where filtered clear is hot should prefer
  pgvector or Chroma where filtered delete is native.

### Security
- Bumped minimum `duckdb` requirement (extra: `duckdb`) from `>=0.9.0`
  to `>=1.1.0` to exclude versions affected by PYSEC-2024-25 (CVSS
  9.8) and PYSEC-2024-203. This is a major-version bump (0.x → 1.x);
  the public DuckDB API used by `SyncDuckDBDatabase` /
  `AsyncDuckDBDatabase` (connection management, `execute`, `query`,
  Arrow result conversion) is stable across this range.
- Bumped minimum `pyarrow` requirement (extra: `parquet`) from
  `>=14.0.0` to `>=17.0.0` to exclude versions affected by
  PYSEC-2023-238 (CVSS 9.8) and PYSEC-2024-161.

### Changed

- **`validate_database_name()` now raises `ConfigurationError`
  instead of `ValueError`** for consistent exception typing across
  the postgres identifier-validation surface (the new
  `validate_pg_identifier` already raised `ConfigurationError`,
  and config-shape errors belong to the
  `dataknobs_common.exceptions` hierarchy).  External callers that
  catch `ValueError` specifically must update to catch
  `ConfigurationError` (or its base `DataknobsError`).
  `validate_database_name` is internal infrastructure with no
  publicly documented `ValueError` contract, so this is a small
  behavior change rather than a breaking API change.

### Fixed

- **`PostgresBaseConfig._parse_postgres_config` now raises
  `ConfigurationError` when the `table` or `schema` config key is
  not a valid string identifier**, instead of silently propagating
  non-string values through `quote_ident()` and producing broken
  SQL at first query.  Defense-in-depth — the canonical fix for
  the FSM-side `schema`-key collision lives in `dataknobs-fsm`
  (Item 117).  This validator catches misuse from any future
  consumer that accidentally injects a non-identifier value via
  either key.  The same identifier shape (`^[a-zA-Z_][a-zA-Z0-9_]*$`)
  used by `validate_database_name` is enforced for both keys
  through the public `validate_pg_identifier` helper in
  `dataknobs_data.backends.postgres_mixins`.
- **`PgVectorStore` now validates `schema` and `table_name`
  identifiers at construction**, closing the third Postgres
  consumer's parallel hazard.  `PgVectorStore._parse_backend_config`
  reads these keys directly (it does not flow through
  `_parse_postgres_config`), so the records-backend fix above did
  not cover it.  Both consumers now use the same
  `validate_pg_identifier` helper, so a malformed identifier is
  caught with a clear `ConfigurationError` at construction
  regardless of which Postgres consumer the application uses.
- **`AsyncPostgresDatabase.update_batch()` no longer raises
  `PostgresSyntaxError` from a duplicated `RETURNING id` clause**.
  The query was built by `SQLQueryBuilder.build_batch_update_query`,
  which already appends ` RETURNING id` for the postgres dialect
  (sql_base.py:559-561), and then `update_batch` appended
  ` RETURNING id` a second time at postgres.py:1484 — producing
  invalid SQL ending in `RETURNING id RETURNING id`. asyncpg
  rejected the query before any row was updated. Pre-existing
  latent bug uncovered while reviewing PR #303 — no prior test
  exercised `AsyncPostgresDatabase.update_batch` against a real
  Postgres (only sqlite/duckdb async `update_batch` had coverage).
  Fix removes the second `RETURNING id` append; the builder's
  output is already postgres-ready. Sync `SyncPostgresDatabase.
  update_batch` was already correct (its comment "query now
  includes RETURNING clause" was accurate).
- **`AsyncPostgresDatabase.stream_read()` no longer raises
  `TypeError: 'async for' requires an object with __aiter__ method,
  got Cursor`**. The cursor was constructed via
  ``cursor = await conn.cursor(sql, *params)``, which returns an
  asyncpg ``Cursor`` (intended for the explicit-fetch API
  ``await cur.fetch(n)``) and is not an async iterator; the
  subsequent ``async for row in cursor`` then failed before yielding
  the first row. Pre-existing bug uncovered by the new
  ``test_async_stream_read_preserves_record_id`` parity test added
  for the Item 114 fix above (no prior test exercised
  ``AsyncPostgresDatabase.stream_read`` against a real Postgres). Fix
  iterates the ``CursorFactory`` returned by
  ``conn.cursor(sql, *params)`` directly — matching the asyncpg
  pattern used elsewhere in this file. ``stream_read`` now actually
  streams rows from Postgres for the first time.
- **`AsyncPostgresDatabase.read()`, `search()`, `vector_search()`,
  `stream_read()`, and `_text_search_for_hybrid()` now return records
  with populated `record.id` / `record.storage_id`**, matching the
  sync `SyncPostgresDatabase` behavior. The async `_row_to_record`
  previously copy-pasted the sync serializer body but dropped the
  `ensure_record_id` step, so `await db.read(id)` returned records
  where `record.id` / `record.storage_id` were whatever was in the
  JSON payload (typically `None`) — silently differing from
  `db.read(id)` on the sync backend for the same on-disk row.
  `search()` was the only async call site that compensated explicitly
  (`record.storage_id = str(row['id'])` after `_row_to_record`); the
  other four were silently broken. The fix delegates the async
  `_row_to_record` to the shared
  `SQLRecordSerializer.row_to_record(dict(row))` static helper that
  the sync sibling already uses, so all five async call sites now
  populate the id uniformly. Strictly information-additive — the
  sync backend has always returned the populated id, so consumers
  working against both backends already handle the populated case.

### Changed

- **`SQLRecordSerializer.record_to_row(record, id=None)`** added as
  the outbound counterpart to the existing
  `SQLRecordSerializer.row_to_record(row)` static. Centralizes the
  `id` / `data` / `metadata` row shape so sync and async SQL
  backends do not duplicate the body and silently drift — the same
  shape that produced the inbound `_row_to_record` divergence.
- `SyncPostgresDatabase._record_to_row` and
  `AsyncPostgresDatabase._record_to_row` are now one-line delegations
  to `SQLRecordSerializer.record_to_row`. Behavior is unchanged
  (both bodies were functionally identical pre-consolidation); the
  consolidation eliminates the parallel-implementation drift surface.
- **All four redundant `record.storage_id = str(row['id'])`
  assignments in SQL backend `search()` paths have been removed.**
  After the `_row_to_record`-delegation fix above, every SQL
  backend's search path now goes through
  `SQLRecordSerializer.row_to_record` (directly, or via
  `SQLQueryBuilder.row_to_record`), which calls `ensure_record_id`
  before returning — so the post-call explicit assignments were
  no-ops. Cleanups: `AsyncPostgresDatabase.search()` (postgres.py:
  1362-1363), `SyncPostgresDatabase.search()` (postgres.py:419-420),
  `AsyncSQLiteDatabase.search()` (sqlite_async.py:271-272),
  `AsyncDuckDBDatabase.search()` (duckdb.py:404-405), and
  `SyncDuckDBDatabase.search()` (duckdb.py:952). Behavior unchanged
  (each was a redundant double-write of the same value); the
  cleanup eliminates the future-confusion surface ("why does this
  set storage_id when `_row_to_record` already does?").
- New unit-test module `tests/test_backends/test_sql_record_
  serializer.py` covers `SQLRecordSerializer.row_to_record` and
  `SQLRecordSerializer.record_to_row` directly (round-trip,
  id-population, metadata serialization edge cases). The new
  helpers were previously covered only transitively via integration
  tests requiring a live Postgres.
- `packages/data/docs/RECORD_SERIALIZATION.md` documents the new
  `record_to_row` static and the inbound/outbound boundary
  contract, with a forward-reference to the Item 114 cautionary
  tale.

## v0.4.16 - 2026-04-29

### Security

- **`get_vector_extraction_sql`, `_build_text_field_concat`, and both `stream_read` implementations now validate field names** against `[A-Za-z_][A-Za-z0-9_]*` before embedding them in SQL string literals (JSONB key positions). All four functions previously accepted input like `"field'name"` or `"'; DROP TABLE;--"`, which breaks SQL syntax or enables injection in the string-literal position where `quote_ident()` does not apply. In `stream_read`, validation fires before the connection check so it raises `ValueError` on first iteration without requiring a live database. Invalid names raise `ValueError`; valid names are unchanged. The shared `validate_field_name(field)` helper in `sql_base` centralises the check and error message; `postgres.py` calls it via the public function rather than reaching into the private `_FIELD_NAME_RE` regex.

### Changed

- **Identifier quoting in all SQL backends**: `SyncPostgresDatabase`, `AsyncPostgresDatabase`, `SyncSQLiteDatabase`, `AsyncSQLiteDatabase`, `SyncDuckDBDatabase`, `AsyncDuckDBDatabase`, `PgVectorStore`, and `PostgresTableManager` now internally quote schema and table names using `quote_ident()` from `dataknobs_utils`. Any valid SQL identifier (mixed-case, reserved words, etc.) is now accepted without pre-quoting. Existing consumers using plain `[a-z_][a-z0-9_]*` names see no behavior change. Vector column names in `AsyncPostgresDatabase` are also fully quoted (`_ensure_vector_column` ALTER TABLE, `vector_search`, `hybrid_search`). `AsyncPostgresDatabase.stream_write()` uses asyncpg's `schema_name=` keyword so the table name is not double-quoted. `postgres_vector.py` helper functions (`build_vector_index_sql`, `get_vector_count_sql`) now accept pre-quoted identifier arguments.

- **`PostgresTableManager.get_table_exists_sql()`** added as a new static method returning `(sql, params)` tuple with `$1`/`$2` parameter binding.

### Added
- `auto_create_table` config option on all SQL-style relational database
  backends — `Sync/AsyncPostgresDatabase`, `Sync/AsyncSQLiteDatabase`,
  `Sync/AsyncDuckDBDatabase`. Default is `True` (no behaviour change for
  existing consumers). When `False`, `connect()` verifies the records table
  exists and raises `RuntimeError` if it doesn't, enabling
  Alembic/Flyway/Sqitch-managed schemas with DML-only application roles.
  Mirrors the existing `PgVectorStore.auto_create_table` contract.
- `SQLTableManager.get_table_exists_sql()` — dialect-aware parameterized
  table-existence query supporting qmark (`?`), numeric (`$1`/`$2`), and
  pyformat (`%(name)s`) placeholder styles. Used internally by all SQL
  backends; both Postgres backends now delegate to this shared helper
  (`SyncPostgresDatabase` with `param_style="pyformat"`,
  `AsyncPostgresDatabase` with `param_style="numeric"`) replacing the
  separate `PostgresTableManager.get_table_exists_sql()` static method.
- `SQLTableManager.coerce_bool()` — public shared helper for coercing
  YAML/env string values (`"false"`, `"0"`, `"no"`) to Python `bool`.
  `None` returns the `default` parameter (``True`` by default). Replaces
  per-backend inline coercion logic for consistent edge-case handling.
  **Behaviour change for `ensure_database`:** the previous inline coercion
  used an allowlist (`"true"`, `"1"`, `"yes"` → `True`; all other strings
  → `False`). `coerce_bool` uses a blocklist (`"false"`, `"0"`, `"no"`, `""`
  → `False`; all other strings → `True`). Unrecognised strings such as
  `"on"` or `"enabled"` now correctly enable the feature rather than
  silently disabling it.
- `SQLTableManager.__init__` now accepts a `param_style` keyword argument
  (`"qmark"` default, `"numeric"` for asyncpg, `"pyformat"` for psycopg2)
  controlling which placeholder style `get_table_exists_sql()` emits.

## v0.4.15

### Breaking
- Individual keys now override the same field from a
  `connection_string` (restoring the historical
  `_parse_postgres_config` precedence). A caller passing
  `{"connection_string": "postgresql://.../dbA", "database": "dbB"}`
  now connects to `dbB`, not `dbA`. Pre-Unreleased releases had
  briefly inverted this.

### Added
- `S3SessionConfig` and `create_boto3_s3_client` in
  `dataknobs_data.pooling.s3` — single canonical layer for boto3 /
  aioboto3 S3 client construction. Used by `SyncS3Database`,
  `AsyncS3Database` (via `S3PoolConfig.to_session_config()`), and
  `S3KnowledgeBackend`. `S3SessionConfig.from_dict` accepts both
  `region`/`region_name` and both legacy (`access_key_id`,
  `max_workers`, `max_retries`) and canonical (`aws_access_key_id`,
  `max_pool_connections`, `max_attempts`) key shapes so one config
  dict feeds every S3 construct.
- **`PgVectorStore` now tracks `updated_at`** on each row. The schema
  gains `updated_at TIMESTAMP DEFAULT NOW()`, refreshed to `NOW()`
  on every upsert (same-ID `add_vectors`) and on `update_metadata`;
  `created_at` is preserved on upsert. Pre-existing tables gain the
  column via idempotent `ALTER TABLE ADD COLUMN IF NOT EXISTS`
  during `initialize()` when `auto_create_table=True`; pre-existing
  rows keep `updated_at IS NULL` until re-ingested — treat `NULL` as
  "not re-ingested since the column was added." Consumers with
  `auto_create_table=False` must apply the ALTER manually (SQL in
  the `pgvector-backend.md` doc). Memory (in-process) tracks the
  same (created, updated) tuple and preserves it through the pickle
  round-trip.
- **`VectorStore` timestamp exposure** — `include_timestamps=True`
  on `get_vectors()` and `search()` returns `_created_at` /
  `_updated_at` in the metadata dict. Format (`iso` / `epoch` /
  `datetime`) and key names are configurable via a new `timestamps`
  config block on every `VectorStoreBase` subclass. Supported by
  `MemoryVectorStore` and `PgVectorStore`;
  `FaissVectorStore` and `ChromaVectorStore` do **not** yet accept
  the `include_timestamps` kwarg (calling it raises `TypeError`) —
  deferred per Item 36 follow-ups. Collision policy: consumer
  metadata values for a configured timestamp key always win; a
  WARNING is logged once per process per colliding key. See
  `packages/data/docs/vector-timestamps.md` for the full contract.

### Changed
- **Behavior change (Defect A):** `PgVectorStore` `id_type` default
  changed from `"uuid"` to `"text"`. RAG consumers passing chunk ids
  such as `"01-fundamentals_0"` now work out-of-the-box. **No data
  migration is required** — `CREATE TABLE IF NOT EXISTS` is a no-op
  on existing tables. **A config update IS required** for pre-flip
  consumers whose tables use a UUID `id` column: add
  `id_type: "uuid"` to the store config, otherwise inserts and
  lookups will fail with a guided `ValueError` pointing at the fix.
- `PgVectorStore`, `PostgresPoolConfig`, `AsyncPostgresDatabase`, and
  `SyncPostgresDatabase` now accept individual `host`/`port`/
  `database`/`user`/`password` keys plus `POSTGRES_*` env-var
  fallbacks. `DATABASE_URL` env fallback now works uniformly across
  all postgres-using constructs (previously only PgVectorStore).
- `SyncPostgresDatabase._open_connection` no longer uses
  `DotenvPostgresConnector` directly; the connection path goes through
  `normalize_postgres_connection_config`, which reads `.env` /
  `.project_vars` files as an additional env fallback layer
  (preserving the retired connector's auto-loading behavior for
  developers who keep secrets in those files).
- **Behavior change:** `SyncS3Database` and `S3KnowledgeBackend` no
  longer default `region` to `"us-east-1"`. Both now defer to
  boto's resolution chain (`AWS_DEFAULT_REGION` env →
  `~/.aws/config` → IMDS → `us-east-1` terminal fallback) when no
  region is configured. Consumers who set `AWS_DEFAULT_REGION`
  previously had it silently overridden — it is now honored.
  Consumers explicitly passing `region: "us-east-1"` see no change.
  Consumers with no AWS config anywhere still terminate at
  `us-east-1` (boto's fallback), preserving existing behavior.
- `S3PoolConfig.from_dict` now accepts `region` (in addition to
  `region_name`), so the same config dict feeds sync and async S3
  paths without rename.
- `SyncS3Database._ensure_bucket_exists` now resolves the effective
  region from `client.meta.region_name` when `region` is unset, so
  bucket creation correctly applies `LocationConstraint` for
  env-derived regions.
- `S3SessionConfig.to_client_kwargs()` and `validate_s3_session`
  automatically add `use_ssl=False` when `endpoint_url` starts with
  `http://` (LocalStack, MinIO, dev S3-compatible servers),
  preserving the previous `SyncS3Database` behavior. `https://`
  endpoints leave `use_ssl` unset so boto's default (`True`) applies
  — a slight tightening of the prior code, which set `use_ssl=False`
  for *all* custom endpoints regardless of scheme. Callers can
  override either case via
  `extra_client_kwargs={"use_ssl": ...}`.

### Fixed
- **Defect C:** `asyncpg.DataError` raised by a `PgVectorStore` when
  the configured `id_type` disagrees with the actual id value or
  column type is now wrapped as a guided `ValueError`. Both
  directions are covered: `id_type="uuid"` + non-UUID id, and
  `id_type="text"` + UUID-typed column (the common post-Defect-A
  migration case). The message names the offending id, the table, and
  the exact config or schema change required.
- `PgVectorStore.delete_vectors` now validates ids client-side when
  `id_type="uuid"` so a bulk delete containing one malformed id
  surfaces that specific id in the error instead of dumping the
  full list.
- `validate_s3_session` no longer passes an empty `endpoint_url`
  kwarg to boto when none is configured.
