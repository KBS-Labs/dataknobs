# Changelog

All notable changes to the dataknobs-fsm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- **`UnifiedDatabaseStorage._setup_backend` now reads the backend
  selection from `StorageConfig.backend`** (the canonical enum)
  instead of `connection_params['type']` (a redundant string copy
  with a silent default of `'memory'`).  Previously, callers that
  did not redundantly populate `connection_params['type']` were
  silently downgraded to `AsyncMemoryDatabase` regardless of which
  `StorageBackend` they requested — a silent data-loss bug for
  production deployments using Postgres backing.  The same fix is
  applied at the second site in `get_statistics()`, which previously
  returned `backend_type='unknown'` under the same conditions.
- **`UnifiedDatabaseStorage._setup_backend` no longer injects a
  `schema` payload into `connection_params`**, eliminating a
  config-key collision with `AsyncPostgresDatabase`'s Postgres
  schema-name parameter.  Previously, instantiating a Postgres
  backend through the factory path crashed during `CREATE TABLE`
  with `PostgresSyntaxError: syntax error at or near "="` — the
  FSM's `DatabaseSchema` object was being interpolated as the
  Postgres schema name.  The bug had been latent since
  `UnifiedDatabaseStorage` first registered the Postgres backend
  because dataknobs's own integration tests bypassed
  `_setup_backend` by injecting a pre-built `AsyncPostgresDatabase`
  via the `database=` kwarg.
- **`FileStorage` now forwards `StorageConfig.compression` to the
  data backend's `compression` config key (was: `compress`)**, so
  enabling compression actually compresses the on-disk file.
  `AsyncFileDatabase` reads `connection_params["compression"]`
  (string `"gzip"` or `None`); FSM was injecting an unrecognized
  `compress` key with a boolean value, which the data backend
  silently ignored, leaving file storage uncompressed regardless
  of `StorageConfig.compression`.
- **`UnifiedDatabaseStorage`'s deprecation warning now uses
  `stacklevel=3`**, attributing the warning to the user's
  `await storage.initialize()` call site instead of internal
  `dataknobs_fsm.storage.base` code.  The previous `stacklevel=2`
  pointed at framework code, which made the migration target
  invisible.
- **`fsm history list` and `fsm history show-execution` CLI
  commands now actually work**.  Previously both commands had
  multiple compounding bugs hidden behind `# type: ignore`:
  `FileStorage(Path(...))` mis-constructed the storage (constructor
  expects a `StorageConfig`), `ExecutionHistory(storage)` misused
  the history dataclass as a manager, and the `query_history` /
  `get_execution` methods called on it do not exist on any class
  in the codebase.  The display code further read keys
  (`execution_id`, `success`, `states`, `arcs`) that have never
  been part of `BaseHistoryStorage`'s actual return shape.
  Both commands have been rewritten to call the real
  `BaseHistoryStorage` API (`query_histories` / `load_history` /
  `load_steps`) with timestamp formatting, status colorization,
  and a working `--verbose` step listing.  The on-disk location
  is now `~/.fsm/history.json` (a single JSON file managed by
  `AsyncFileDatabase`); the previous code tried to use the bare
  `~/.fsm/history` directory path as if it were a file, which
  failed with `IsADirectoryError` whenever the directory existed.
  Behavioral CLI tests added in
  `test_cli_real.py::TestHistoryCLICommands`.
- **`fsm history list` / `show-execution` status display is now
  consistent for in-progress runs.**  Status is derived from
  `(end_time, failed_steps)` via a shared `_derive_history_status`
  helper, so a run with `end_time=None` is reported as
  `in_progress` (cyan) instead of contradicting itself with
  `End: In progress` alongside `Status: completed`.  `_status_style`
  also colorises `in_progress`/`running`.  Behavioral coverage
  added in `test_cli_real.py::test_show_execution_in_progress_run_status_consistent`
  and `::test_list_in_progress_run_shows_in_progress_status`.
- **`FileStorage` class docstring corrected.**  Previously
  advertised "Directory-based organization", "File rotation
  policies", and "Indexing via metadata files" — none of which
  `AsyncFileDatabase` provides.  The docstring now describes the
  real single-file behavior and the actual config knobs (`path`,
  `format`, `compression`).

### Deprecated

- Passing `'type'` in `connection_params` to `UnifiedDatabaseStorage`
  is deprecated; `StorageConfig.backend` is the source of truth.
  The legacy alias is honored with a `DeprecationWarning` and will
  be removed in the next minor release.  `InMemoryStorage` and
  `FileStorage` no longer auto-inject the deprecated `'type'` key
  internally — they were only doing so to feed the buggy parent
  lookup, and the canonical enum drives selection now.

### Removed

- The unused `UnifiedDatabaseStorage._create_steps_schema()` and
  `_create_history_schema()` methods have been deleted, along with
  the corresponding `record_schema` injection into
  `connection_params`.  Both methods constructed `DatabaseSchema`
  descriptors but were never consumed by any backend — history
  records carry their fields as `history_data` JSON payloads
  rather than typed columns, and step records share the database
  (and implicit schema) of history records via `_db is _steps_db`
  semantics.  No external callers expected: both methods were
  private (leading-underscore names).

### Migration

- **Callers who never passed `connection_params['type']`** were
  silently getting in-memory storage even when `StorageConfig.backend`
  said otherwise.  After this fix, they will get the backend they
  asked for.  This is a behavior change — any downstream that
  relied on the in-memory fall-back was relying on the bug.
- **Callers who pre-build `AsyncPostgresDatabase` and inject via
  `UnifiedDatabaseStorage(config, database=db)`** are unaffected;
  the factory path is bypassed entirely.  This was the
  in-production workaround for both bugs and remains supported.
- **Callers using the redundant `connection_params['type']` key**
  continue to work but receive a `DeprecationWarning`.  Remove
  the key; rely on `StorageConfig.backend`.
