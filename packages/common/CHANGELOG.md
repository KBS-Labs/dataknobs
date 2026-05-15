# Changelog

All notable changes to the dataknobs-common package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- `dataknobs_common.events.event_bus_backends` — a registry-extensible
  plugin point for `create_event_bus()`. Out-of-tree consumers register
  a custom `EventBus` backend
  (`event_bus_backends.register("name", factory)`, where a factory is
  `Callable[[dict], EventBus]`) and select it via
  `create_event_bus({"backend": "name", ...})` without forking
  DataKnobs. Exported from `dataknobs_common.events` along with the
  `EventBusFactory` type alias. The built-in `memory`/`postgres`/`redis`
  backends and the `create_event_bus()` signature are unchanged.

### Changed
- `create_event_bus()` now resolves backends through
  `event_bus_backends` instead of a sealed `if/elif` chain. Behaviour is
  identical for the three built-in backends; the unknown-backend
  `ValueError` now lists all registered backends (including
  consumer-registered ones) instead of a hard-coded
  `memory, postgres, redis`.

## v1.3.12 - 2026-05-09

### Added
- `dataknobs_common.metadata.enforce_immutable_keys` — primitive for
  "layered-merge with a designated immutable source for some keys."
  Used by `VectorMemory` (tenant-scope enforcement), `RAGKnowledgeBase`
  (chunk-text protection), and the markdown chunker (node-classification
  protection). Mutates and returns the merged target dict; emits a
  WARNING when a caller-supplied value differed from the source value
  for an immutable key, naming the key. Re-exported from the top-level
  `dataknobs_common` namespace. See the `dataknobs-bots` and
  `dataknobs-xization` 0.x changelog entries for the consumer-side
  fixes built on this helper. The helper's caller-vs-source equality
  check is array-safe: numpy arrays, lists, and other non-scalar
  values do not raise `ValueError` from element-wise comparison's
  ambiguous truth value.
- `dataknobs_common.config_loading` module with `find_config_file()`,
  `load_yaml_or_json()`, and `parse_yaml_or_json()` helpers, plus a
  `ConfigLoadError` exception hierarchy
  (`ConfigParseError`, `ConfigShapeError`,
  `ConfigUnsupportedFormatError`, `ConfigYAMLNotInstalledError`).
  Consolidates the YAML/JSON file→dict and bytes→dict
  parse-and-validate chain previously duplicated across nine sites
  in five packages (`dataknobs_config`, `dataknobs_xization`,
  `dataknobs_fsm`, `dataknobs_bots`, `dataknobs_llm`). PyYAML is
  lazy-imported — no hard dependency added to `dataknobs-common`.
  `find_config_file` adds a leading dot automatically when callers
  pass extensions without one (`"yaml"` → `".yaml"`).
  `parse_yaml_or_json` wraps `UnicodeDecodeError` from non-UTF-8
  byte input as `ConfigParseError`, so consumers reading from
  binary backends never see the stdlib decode error leak past the
  helper. The helpers are also re-exported from the top-level
  `dataknobs_common` namespace.

## v1.3.11 - 2026-04-29

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
