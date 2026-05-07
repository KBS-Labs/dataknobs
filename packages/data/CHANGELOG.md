# Changelog

All notable changes to the dataknobs-data package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

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
