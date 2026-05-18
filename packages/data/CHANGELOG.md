# Changelog

All notable changes to the dataknobs-data package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
