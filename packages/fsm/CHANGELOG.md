# Changelog

All notable changes to the dataknobs-fsm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- New `async_database` FSM resource type (backed by
  `AsyncDatabaseResourceAdapter`) so a state transform can `await`
  non-blocking `upsert` / `execute_query` against any `dataknobs-data`
  `AsyncDatabase` backend.
- Record-identity strategy for the database function library
  (`dataknobs_fsm.functions.library.identity`): a `RecordIdentity` protocol
  with `KeyColumnsIdentity` (collision-safe unit-separator join) and
  `CallableIdentity` reference implementations. `DatabaseUpsert`,
  `DatabaseBulkInsert`, and `BatchCommit` accept `key_columns=` / `id_fn=` /
  `identity=` to control how a row maps to its storage id. `KeyColumnsIdentity`
  raises `ValidationError` for a key column that is missing or `None` rather
  than rendering it as the literal `"None"` (which would let sparse rows
  collide).
- `BatchCommit` gains an `atomicity` policy (`"best_effort"` / `"require"`).
  `"require"` raises `CapabilityNotSupportedError` on a backend that cannot
  guarantee an all-or-nothing batch instead of writing a partial batch under a
  false promise. The legacy `use_transaction=` flag is now an alias
  (`True` → `"require"`). `batch_size` bounds the rows per commit under
  `best_effort`; under `require` the batch is committed as a single
  all-or-nothing unit (chunking would defeat the atomicity guarantee).

### Fixed

- `DatabaseBulkInsert.on_duplicate` is now honored. Previously the adapter
  always created records and ignored the parameter; `"error"` / `"ignore"` /
  `"update"` now take effect against the configured record identity, and a
  duplicate-detecting policy (`"ignore"` / `"update"`) with no identity raises
  `ConfigurationError` rather than silently degrading to create-only.
- `DatabaseUpsert.on_conflict` is fail-closed in the same way: `"error"` /
  `"ignore"` with no identity, and an unknown `on_conflict` value, now raise
  `ConfigurationError` instead of silently behaving like create-only / update.
  The default `"update"` with no identity remains a plain create.
- `BatchCommit` now persists its batch through the real `commit_batch` atomic
  primitive. It previously called a `resource.transaction()` method that no
  adapter implemented, so it raised `AttributeError` on first use.
- The async execution engine now acquires a state's declared `resources`
  into the transform `FunctionContext`, so registered async transforms
  receive their injected resources (matching the synchronous engine).
- `StepResult` (advanced API) gains a `failed_states` field listing the
  states whose transform raised during a stepped record's execution.
- States gain a `run_on_failure` flag (`StateDefinition.run_on_failure`, and the
  `run_on_failure:` state config key). A state declared with
  `run_on_failure=True` runs its transforms even after an upstream transform
  failed — the per-state opt-out for recovery / compensation / cleanup /
  dead-letter states that must execute despite a prior failure. It re-enables
  the transforms only; the record is still reported as a failure.
- States gain an `emit_output` flag (`StateDefinition.emit_output`, and the
  `emit_output:` state config key, default `True`). An **end** state marked
  `emit_output=False` has its records excluded from the output in every
  processing mode — the streaming sink skips non-emitting terminals just as the
  batch/whole writers only write records that reach an emitting terminal. Used
  to keep "processed but not part of the output" records (e.g. filtered or
  rejected) out of the result.
- `AsyncSimpleFSM.process_stream` gains an `output_format` parameter (default
  `'auto'`, deriving the format from the sink extension) so a caller can pin
  the streaming output format independently of the sink filename.
- `AsyncSimpleFSM.get_state(name)` exposes a state definition by name, so
  consumers can inspect a state's attributes (e.g. `emit_output`) without
  reaching into the private FSM handle.

### Changed

- A record whose **state transform raises** is now reported as a failed
  record by the execution engines: it still traverses to a final state, but
  `execute()` returns `success=False` (the failure is recorded in
  `context.failed_states` and surfaced by
  `BaseExecutionEngine.finalize_single_result`). Previously such a record was
  reported as `success=True`. This is a cross-cutting behavioral change that
  affects **every** execution-engine consumer (sync and async), not only ETL.
- Once a record has failed a state transform, its remaining and downstream
  state transforms are **skipped** rather than run against the indeterminate
  pre-failure data. This stops a later state (e.g. an ETL `load` upsert) from
  persisting a record whose transform already failed, while traversal still
  continues so the record is accounted as an error. States that must run
  despite a prior failure (recovery / compensation / cleanup / dead-letter)
  opt out with `run_on_failure=True`. A transform failure on a parallel, batch,
  or (isolated) sub-network sub-path is propagated back to the parent context, so
  it gates the parent's downstream-transform skip and persistence decision too —
  an isolated sub-network whose transform raised no longer reports the parent
  record as a success.
- `FileProcessor` streaming mode now runs on the same async execution engine as
  its batch and whole-file modes (`AsyncStreamExecutor` drives the async engine
  directly instead of running the synchronous engine in a thread pool). All
  three modes share one execution path, so filters, validators, and transforms
  behave identically regardless of mode, and async state transforms are awaited
  in streaming mode.
- `FileProcessor.process()` returns the same metrics shape — with the same
  per-terminal classification — in every mode. STREAM mode now populates
  `records_processed` / `records_written` / `skipped` / `errors` (it previously
  exposed only the streaming executor's `total_processed` / `successful` /
  `failed` and left the unified keys at 0), and classifies each non-emitting
  terminal identically to BATCH/WHOLE: validation rejections (the `error`
  terminal) count as `errors`, filtered records count as `skipped`. (Previously
  STREAM inferred `skipped` as a `total - failed - written` remainder, which
  swept validation rejections into `skipped` and left `errors` at 0.)
  `records_processed` counts clean terminals only (written + skipped) across all
  modes; `lines_read` remains tracked on the BATCH read path only. The async
  streaming executor reports clean non-emitting records bucketed by terminal
  name via a new `excluded_by_state` field on its result / `process_stream`
  return, so any consumer can apply its own per-terminal accounting.
- A terminal's `emit_output` flag is now the single source of truth for output
  emission in the batch and whole-file modes too — both writers resolve
  `emit_output` from the final state rather than matching a hardcoded `complete`
  name, so they apply exactly the policy the streaming sink already used.

### Fixed

- `AdvancedFSM.execute_step_sync` / `execute_step_async` now report
  `success=False` (with the offending state in `StepResult.failed_states`)
  when a step enters a state whose transform raised, instead of reporting a
  successful step. `run_until_breakpoint` / `run_until_breakpoint_sync` stop
  on such a step. Previously the step-driver API silently reported success at
  a final state even when a state transform had failed.
- `DatabaseETL` no longer upserts a record whose `transform` step raised: the
  failed record is counted as an `error` and skipped at `load`, rather than
  being written to the target with its pre-failure (untransformed) data.
- `DatabaseETL.run()` now persists records to the target database. Each
  extracted record has its `field_mappings` and `transformations` applied
  and is upserted into `target_db`, and `run()` flushes and closes the
  target so the rows are durable. The returned metrics (`extracted` /
  `transformed` / `loaded` / `errors`) reflect the records actually
  processed. Previously records traversed skeleton states without a load
  step, the user `transformations` callables were never applied, and the
  metrics were hollow. (The `validate` and `enrich` stages remain
  passthroughs pending their config contracts.)
- `AsyncBatchExecutor` drives the async execution engine directly instead
  of running the synchronous engine in a thread pool, so async state
  transforms are awaited — they previously leaked unawaited coroutines and
  never ran.
- Registered interface functions (e.g. `ITransformFunction` instances)
  supplied via `custom_functions=` are now detected as async and awaited
  correctly, instead of being mistaken for synchronous callables.
- `FileProcessor` now processes records end-to-end. Previously every record
  dead-ended at the `filter` state and was reported as errored/failed even for
  a pure passthrough config, and batch mode never wrote its output. The FSM now
  connects only the *enabled* stages into a single chain to `write → complete`
  (so no stage dead-ends), batch mode writes its output, and configured
  `filters` / `transformations` / `aggregations` / `validation_schema` actually
  execute — they are wired through the FSM's `custom_functions=` channel and
  referenced from state `functions` blocks (transform / aggregate) and arc
  conditions (filter / validate) instead of unresolvable inline-code names.
  Filtered records are excluded from the output and counted as `skipped`;
  records that fail validation or a transform are excluded and counted as
  `errors`.
- `FileProcessor` STREAM mode now honors an explicitly configured `format` /
  `output_format` instead of always auto-detecting from the file extension, so
  (for example) a `.log`-extensioned file declared as `format=JSON` has its
  lines parsed as JSON rather than wrapped as `{'text': line}`.
- `FileProcessor`'s `validate` / `filter` gates route passing records
  deterministically: the conditional arc is given a higher `priority` than its
  unconditional fall-through, so routing no longer depends on arc declaration
  order.
- `create_batch_file_processor` no longer raises `TypeError` on construction —
  it passed a non-existent `batch_size` field to `FileProcessingConfig`; the
  batch size is now applied to the config's `chunk_size`.

## v0.2.3 - 2026-06-23

## v0.2.2 - 2026-06-22

### Security

- Bumped the minimum `langchain` requirement in the `llm` extra from
  `>=1.0.0` to `>=1.3.9` to exclude versions affected by
  GHSA-gr75-jv2w-4656 (CVSS 5.1), which affects 1.0.0–1.3.8 and is fixed
  in 1.3.9. `langchain` is declared only as a convenience extra and is
  not imported by this package. Surfaced by the floor-resolve audit in
  the `dependency-update` workflow.

### Changed

- ruff's `ASYNC` lint family (`flake8-async`) is now enforced for this
  package, so blocking I/O on the event loop inside `async def` code is
  caught at lint time. See the `async-transport` authoring rule.

### Fixed

- `FileProcessor` and `DatabaseETL` now run their FSM pipelines on the
  active event loop's async engine instead of driving a synchronous FSM
  wrapper from their `async` methods, so awaiting `FileProcessor.process()`
  and `DatabaseETL.run()` no longer stalls the loop on the wrapper's
  blocking sync-to-async bridge. `DatabaseETL.run()` additionally builds
  its source database through the async database factory (it previously
  raised on every call and could not execute), and `FileProcessor`'s
  streaming mode passes the input/output paths to the streaming executor;
  both now run end-to-end.
- The FSM file-processing and streaming utilities now perform their
  file reads and writes off the event loop, so awaiting them from an
  async context no longer stalls the loop. The lazy chunk/line readers
  (`StreamingFileReader`, the `read_*_file` helpers in
  `utils/file_utils`, and `FileProcessor._read_batches`) stream their
  blocking `open()` + iteration on a worker thread via
  `aiter_sync_in_thread`, preserving bounded-memory streaming; the
  whole-file readers/writers (`read_json_file`,
  `FileProcessor._process_whole`/`_write_output`, the `ChunkReader`
  format readers, `FileAppender`'s buffered writes, `StreamingFileWriter`'s
  buffered open/flush/close, and the `AsyncSimpleFSM.process_stream`
  JSON-sink whole-file cleanup) offload their one-shot disk I/O via
  `asyncio.to_thread`. Public async surfaces are unchanged.
- `AsyncSimpleFSM.process_stream` now accepts a `Path` source/sink in
  addition to `str` (a `Path` previously fell through to the async-iterator
  branch and failed), and `FileProcessor.process()` now raises
  `NotImplementedError` when `compression` is configured rather than
  silently emitting uncompressed output — no execution path writes
  compressed output, so the option was being silently dropped.

### Security

- Bumped minimum `aiohttp` requirement (extra: `http`) from
  `>=3.13.4` to `>=3.14.1` to extend the prior `<=3.13.3` CVE sweep
  (highest CVSS 9.1: GHSA-63hf-3vf5-4wqf) through the full `<3.14.x`
  floor-resolve advisory set. The two named highs are
  GHSA-hg6j-4rv6-33pg (CVSS 6.6, cross-origin redirect cookie
  leakage on the per-request `cookies=` kwarg) and
  GHSA-jg22-mg44-37j8 (CVSS 6.4, `CookieJar.load()`
  deserialization); both were already triaged unreachable from this
  codebase (outbound HTTP uses header-based auth, the advisory's
  safe pattern, and `CookieJar.load()` is never invoked) but
  bumping clears the floor-resolve audit regardless. Fixes land
  across 3.14.0 and 3.14.1, hence `>=3.14.1` as the floor. The
  bump was previously blocked by `aioresponses 0.7.8` not passing
  the `stream_writer` kwarg to `aiohttp.ClientResponse` introduced
  in aiohttp 3.14; unblocked by the workspace move off
  `aioresponses` to an in-process `aiohttp.web` test server in the
  bots package.

## v0.2.1 - 2026-06-08

## v0.2.0 - 2026-05-26

### Changed

- The pattern-family runtime configs — `CircuitBreakerConfig`,
  `FallbackConfig`, `CompensationConfig`, `BulkheadConfig`,
  `ErrorRecoveryConfig` (`patterns.error_recovery`), `APIEndpoint` +
  `APIOrchestrationConfig` (`patterns.api_orchestration`), `ETLConfig`
  (`patterns.etl`), and `FileProcessingConfig` (`patterns.file_processing`) —
  are now frozen `StructuredConfig` subclasses. They gain `from_dict()` /
  `to_dict()` and symmetric round-tripping, and are **immutable** (use
  `dataclasses.replace(...)` to derive a modified copy). `ErrorRecoveryConfig`
  rebuilds its five nested sub-configs (including the `dataknobs_common`
  `RetryConfig`) as typed instances from a nested mapping, and
  `APIOrchestrationConfig` rebuilds its `endpoints` list as typed `APIEndpoint`
  instances. Configs carrying live callables round-trip by identity, so
  `to_dict()` on such a config is for in-process round-tripping, not JSON
  serialization. `CompensationConfig.compensation_actions` now defaults to an
  empty list (previously a required field). `FileProcessor` format
  auto-detection now resolves onto the processor rather than writing back to
  the (now immutable) config, which keeps its caller-supplied "auto-detect"
  value; the resolved values are exposed as the read-only
  `FileProcessor.resolved_format` / `resolved_output_format` properties.
  Existing constructor call sites are unaffected; the Pydantic FSM
  loader schema (`config/schema.py`) is the separate declarative layer and is
  unchanged.

- The resources/IO/storage/streaming/functions runtime configs — `PoolConfig`
  (`resources.pool`), `IOConfig` (`io.base`), `StreamConfig` (`streaming.core`),
  `ResourceConfig` (`functions.base`), and `StorageConfig` (`storage.base`) —
  are now frozen `StructuredConfig` subclasses, gaining `from_dict()` /
  `to_dict()` and symmetric round-tripping, and are **immutable** (use
  `dataclasses.replace(...)`). Their `Enum` fields (`IOConfig.mode`/`format`,
  `StorageConfig.backend`) load from raw strings and survive a JSON round-trip
  via `to_json_dict()`; `IOConfig.error_handler` and other live-callable fields
  round-trip by identity (in-process, not JSON). `StorageConfig` was converted
  from a plain class to a frozen dataclass (its `get_mode_config()` helper is
  retained); the in-memory and file storage backends now build a local working
  copy of `connection_params` and reconstruct the config via
  `dataclasses.replace(...)` instead of mutating the caller's config in place.
  Existing constructor call sites are unaffected; the Pydantic FSM loader
  schema (`config/schema.py`) remains the separate declarative layer.

- The FSM pattern/runtime consumers built from those configs — `CircuitBreaker`,
  `Bulkhead`, `ErrorRecoveryWorkflow` (`patterns.error_recovery`),
  `APIOrchestrator` (`patterns.api_orchestration`), `DatabaseETL`
  (`patterns.etl`), `FileProcessor` (`patterns.file_processing`),
  `StreamContext`, `AsyncStreamContext` (`streaming.core`), and `ResourcePool`
  (`resources.pool`) — now build through `StructuredConfigConsumer`. Each gains
  a uniform construction surface: a dict-dispatch `Cls.from_config({...})`
  classmethod alongside the existing typed-config constructor, and a typed
  read-only `self.config` property. The previous typed-config and
  `config=None`/all-default constructor calls are unchanged. `ResourcePool`
  additionally carries a required `provider` collaborator (a live resource
  provider, not config data); it keeps its back-compat
  `ResourcePool(provider, config=None)` positional shortcut — the provider
  travels through the mixin's collaborator channel while the config flows onto
  `self.config` — and `ResourcePool.from_config(config, provider=...)` delivers
  the provider alongside the config (mirroring `PostgresEventBus`'s
  `connection_string` positional shortcut).

### Security

- `APIEndpoint.headers` (`patterns.api_orchestration`) is masked as `'***'`
  in `repr()` via `_SENSITIVE_FIELDS`. The mapping routinely carries
  credentials (`Authorization`, `X-Api-Key`, `Cookie`) whose key names are
  not in the `StructuredConfig` interior default set, so the whole field is
  masked by name. Display-only — `to_dict()` round-trips the real value.

- Bumped minimum `pymdown-extensions` requirement (docs dev
  dependency) from `>=10.16.1` to `>=10.21.3` to exclude
  GHSA-62q4-447f-wv8h (CVSS 4.3), flagged at the floor resolve by the
  `dependency-update` workflow. The floor preserves the prior sweep of
  GHSA-r6h4-mm7h-8pmq (CVSS 2.7, 10.16.1). `pymdown-extensions` is a
  transitive dependency of `mkdocs-material` whose own constraint
  permits the vulnerable version, so an explicit floor is required; the
  identical floor in the workspace-root docs dev dependencies was
  bumped in lockstep.

## v0.1.21 - 2026-05-20

## v0.1.20 - 2026-05-18

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
