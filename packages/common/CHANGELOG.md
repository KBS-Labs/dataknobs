# Changelog

All notable changes to the dataknobs-common package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Removed

- Removed the `dataknobs_common_postgres` and `dataknobs_common_elasticsearch`
  pytest11 fixture plugins. The fixtures (`postgres_connection_params`,
  `ensure_postgres_ready`, `make_postgres_test_db`, `elasticsearch_connection_params`,
  `ensure_elasticsearch_ready`, `make_elasticsearch_test_index`) are no longer
  auto-discovered by pytest. Projects that relied on plugin-autodiscovered fixtures
  should import the helpers directly from `dataknobs_common.testing`
  (`wait_for_postgres`, `wait_for_elasticsearch`) or configure equivalent fixtures
  in their own `conftest.py`. Consumer audit (sandbox, edubot) confirmed no active
  usage of these fixtures.

### Test Infrastructure
- `dataknobs_common.testing` is now a package (was a single file). All
  existing imports continue to work unchanged via re-exports from
  `__init__.py`.
- New `dataknobs_common.testing.postgres_fixtures` pytest11 plugin
  exposing shared session-scoped `postgres_connection_params` /
  `ensure_postgres_ready` fixtures plus a `make_postgres_test_db(prefix)`
  factory fixture and a `wait_for_postgres()` helper. Consumers wrap the
  factory with a thin per-prefix fixture (e.g. `yield from
  make_postgres_test_db("test_records_")`) instead of duplicating
  `wait_for_postgres` / connection params / database creation /
  table-cleanup boilerplate. Lazy `psycopg2` import — no hard dep added
  to `dataknobs-common`.
- New `dataknobs_common.testing.elasticsearch_fixtures` pytest11 plugin
  exposing parallel `elasticsearch_connection_params` /
  `ensure_elasticsearch_ready` fixtures plus a
  `make_elasticsearch_test_index(prefix)` factory fixture and a
  `wait_for_elasticsearch()` helper. Lazy `requests` and
  `dataknobs_utils.elasticsearch_utils` imports. Index-cleanup teardown
  tightened from a bare `except Exception: pass` to specific
  `ConnectionError` / `ValueError` swallowing with `logger.warning` —
  unexpected exceptions now propagate.
- Both plugins are registered as pytest11 entry points in
  `pyproject.toml`, so any package depending on `dataknobs-common`
  automatically gets the fixtures via pytest plugin discovery — no
  consumer-side `conftest.py` imports required.

## v1.3.10

### Added
- `normalize_postgres_connection_config` — canonical postgres connection
  config normalizer used by every postgres-using construct in dataknobs
  (PgVectorStore, Sync/AsyncPostgresDatabase, PostgresPoolConfig,
  PostgresEventBus). Accepts `connection_string`, individual host/
  port/database/user/password keys, `DATABASE_URL` env var,
  `POSTGRES_*` env vars, and values from `.env` / `.project_vars`
  files (when `python-dotenv` is installed). Explicit config always
  wins over env; individual keys always override the same field from
  a `connection_string`.

### Changed
- `PostgresEventBus` now accepts the unified config dict (individual
  keys, env-var fallbacks) in addition to the legacy positional
  `connection_string` argument. `create_event_bus({"backend":
  "postgres", ...})` passes the full config through unchanged.
