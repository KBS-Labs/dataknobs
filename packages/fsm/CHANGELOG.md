# Changelog

All notable changes to the dataknobs-fsm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## v0.1.19 - 2026-05-13

### Added

- **`UnifiedDatabaseStorage.save_step(metadata=...)`** — new kw-arg
  routes caller-supplied cross-cutting context (tenant_id,
  correlation_id, audit info) to the underlying record's
  ``metadata`` column.  The kwarg is **consumer-supplied**: the
  FSM engine does not populate it during execution, so consumers
  wrap the storage call from their own execution path or extend
  ``UnifiedDatabaseStorage`` in a subclass to inject these fields
  uniformly.  Composes
  ``AsyncKeyedRecordStore[_StepRecord]`` from `dataknobs-data` as
  the single Record-construction site, so the metadata channel is
  part of the serializer signature.  Persisted metadata is
  filterable end-to-end via the ``metadata.X`` dot-notation
  field-path convention (JSONB pushdown on Postgres; JSON-extract
  pushdown on SQLite and DuckDB; ``Record.get_value`` traversal on
  memory / file).  See ``packages/fsm/docs/FSM_CONFIG_GUIDE.md``
  for usage examples.

- **``load_steps`` filter / pagination kwargs** —
  ``filter_metadata=`` (kw-only `Mapping[str, Any] | None`),
  ``sort=`` (kw-only `list[SortSpec] | None`), ``limit=`` (kw-only
  `int | None`; ``limit=0`` honors Python-slice semantics → empty
  result), and ``offset=`` (kw-only `int | None`) are now
  first-class on `IHistoryStorage.load_steps`.  Surface mirrors
  the bots-registry layer (`ArtifactRegistry.query(...)`,
  `GeneratorRegistry.list_definitions(...)`,
  `RubricRegistry.list_all(...)`) so consumers composing FSM
  history with bot registries see one consistent pagination /
  filter shape.  Positional ``filters=`` (data-column equality)
  remains for back-compat; the two channels AND-combine.

- **``query_histories`` filter / sort kwargs** — ``filter_metadata=``
  (kw-only) is a symmetry kwarg for callers who'd otherwise write
  ``filters={"metadata.X": V}``; both routes AND-combine when
  supplied together.  ``sort=`` (kw-only) overrides the default
  ``start_time DESC`` ordering when the caller needs a different
  multi-key sort, pushed down to the database query.  ``filters=``
  is now optional (`None` default, previously required); positional
  ``limit=100``/``offset=0`` defaults are preserved.

### Security
- Added explicit floors `markdown>=3.8.1` (GHSA-5wmx-573v-2qwq, XSS,
  CVSS 7.5) and `pymdown-extensions>=10.16.1` (GHSA-r6h4-mm7h-8pmq,
  CVSS 2.7) to the `dev` extra. Both are transitive via
  `mkdocs-material`, but `mkdocs-material`'s own constraint
  (`markdown~=3.2`) permits the vulnerable `markdown` version, so an
  explicit direct-dep floor in `dataknobs-fsm[dev]` is required for
  fresh consumer installs to land on a non-vulnerable resolve.

### Fixed
- Bumped minimum `pyyaml` requirement from `>=6.0.0` to `>=6.0.2` to
  exclude versions that lack cp312/cp313 wheels and fail to build from
  source against modern Cython (`'build_ext' object has no attribute
  'cython_sources'`). Surfaced by the floor resolve step in the
  `dependency-update` workflow.

## v0.1.18 - 2026-05-09

### Security
- Bumped minimum `aiohttp` requirement (extra: `http`) from `>=3.9.0`
  to `>=3.13.4` to exclude 22 known CVEs (highest CVSS 9.1:
  GHSA-63hf-3vf5-4wqf), including CVE-2024-23334 / GHSA-5m98-qgg9-wh84.
- Bumped minimum `httpx` requirement (extra: `http`) from `>=0.25.0`
  to `>=0.27.0` to sweep transitive `h11<0.16` (GHSA-vqfr-h8mv-ghfj,
  CVSS 9.1) — `httpx>=0.27` requires `httpcore` 1.x, which requires
  `h11>=0.16`.
- Bumped minimum `langchain` requirement (extra: `llm`) from `>=0.1.0`
  to `>=1.0.0` to exclude 17 known CVEs across `langchain`,
  `langchain-community`, and `langchain-core` (highest CVSS 9.3:
  GHSA-c67j-w6g6-q2cm — DoS in MathDocumentExtractor; CVSS 10.0:
  PYSEC-2025-70 in `langchain-community`). The `langchain` extra is
  declared as a convenience for downstream consumers; `dataknobs_fsm`
  itself does not import langchain, so the major-version bump
  (0.x → 1.x) has no impact on `dataknobs-fsm`'s API.

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
