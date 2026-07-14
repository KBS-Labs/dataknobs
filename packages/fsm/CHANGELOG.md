# Changelog

All notable changes to the dataknobs-fsm package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- `DatabaseResource.commit_batch`'s idempotent identity path now writes through
  the data layer's `upsert_batch` batch verb instead of a per-row `upsert` loop,
  so it uses the backend's native bulk upsert where one exists (a single
  `ON CONFLICT DO UPDATE` on SQLite/DuckDB/PostgreSQL, a bulk index on
  Elasticsearch) and a per-record loop elsewhere. Each row is still upserted
  under its derived id (stamped onto `storage_id`, which takes priority over any
  `id` field in the row), and a `None` derivation still mints/resolves its own
  id. On transactional backends (SQLite/DuckDB/PostgreSQL) that whole-batch
  upsert is a single all-or-nothing statement, so `BatchCommit` /
  `commit_batch` now honor `atomicity="require"` (and the `use_transaction=True`
  alias) on this idempotent-upsert path too — committing atomically instead of
  rejecting it, matching the create-mode path — and reject `require` only on
  non-transactional backends.

### Removed

- **Breaking:** removed the strategy-based FSM transaction coordinator — the
  `dataknobs_fsm.core.transactions` module (`TransactionManager`,
  `TransactionStrategy`, and the Single/Batch/Manual managers), the
  `transaction` configuration block (`TransactionConfig`),
  `AdvancedFSM.configure_transactions`, and the unused `on_transaction_*`
  callbacks on `ExecutionHook`. It configured an in-memory coordinator that the
  execution engines never consulted to drive database commit/rollback, so it
  delivered no database atomicity. A leftover `transaction:` block in an
  existing configuration is now ignored (a warning is logged at load time).
  Database atomicity is provided by the `AsyncDatabase.transaction()` primitive,
  the `DatabaseTransaction` function, and `BatchCommit(atomicity="require")`.

### Fixed

- Corrected the FSM transaction-mode documentation to the actual supported
  modes (`NONE`/`PER_RECORD`/`PER_BATCH`/`PER_SESSION`/`DISTRIBUTED`); the
  `transaction_mode` setting selects in-memory logical bookkeeping only and does
  not by itself drive database commit/rollback.

## v0.2.5 - 2026-07-07

### Security

- Acknowledged GHSA-f4j7-r4q5-qw2c / PYSEC-2026-311 (CVSS 9.3, pre-auth
  code injection via the `/api/v2/...` endpoint) against the
  `chromadb>=1.0.0` floor, flagged at the floor resolve by the
  `dependency-update` workflow. Affects chromadb 1.0.0–1.5.9 with no
  upstream fix. Not exposed via the dataknobs-data `ChromaVectorStore`
  client used here (embedded/persistent modes only; no `HttpClient` or
  server mode). The inline floor comment in `pyproject.toml` records the
  rationale.

## v0.2.4 - 2026-06-29

### Fixed

- **An `IStateTestFunction` instance used as an arc condition is now dispatched
  on every path.** A bare interface instance injected via the low-level
  `AsyncExecutionEngine(custom_functions=...)` merge was stored un-normalized and
  called as `instance(data, context)` — raising `TypeError`, since the
  interface's logic lives on `.test()`, not `__call__`. The arc-condition paths
  (`AsyncExecutionEngine._evaluate_arc` and `ArcExecution.can_execute_async`) now
  normalize such an instance to its bound `.test` method (sync or `async def`)
  before invoking it, matching the normalization the function manager and config
  builder already applied on their paths. Registered-predicate and
  config-authored arc conditions were unaffected.
- **A rejecting initial-state pre-validator now reports *why* it rejected.** When
  the start state's pre-validator failed, `execute()` returned the generic
  "Failed to enter initial state 'X'", discarding the specific
  "Pre-validation failed for state 'X'" reason recorded during entry. The
  initial-state entry now surfaces the specific reason to the caller.
- **Config-authored `builtin` and `custom` function references now resolve to a
  working function and run.** A `{"type": "builtin", "name":
  "transformers.map_fields", "params": {...}}` (or `{"type": "custom", "module":
  ..., "name": ..., "params": {...}}`) reference now materializes the library
  class/factory with its `params` and adapts it to the engine's invocation
  contract — previously the reference was bound with `functools.partial`, so the
  engine called the *constructor/factory* with the record and the transform never
  ran. Built-in functions are referenced by their introspected names
  (`validators.<Name>` / `transformers.<Name>`); a built-in/custom validator
  gates state entry when declared under `pre_validators`. The bare-string
  state-sugar shorthand still resolves only to `registered`/`inline` (use the
  dict form for `builtin`/`custom`). The config guide's built-in example (which
  named a non-existent function) is corrected and every function-type example is
  verified against the resolver.
- The **synchronous FSM APIs that ran on the standalone sync engine now run
  async transforms correctly.** `FSM.execute`, the sync batch/stream executors,
  and `AdvancedFSM.execute_step_sync` now execute on the single async engine, so
  an `async def` transform (e.g. every built-in database transform) is awaited
  rather than being invoked and discarded as an un-awaited coroutine. Sync FSMs
  whose states/arcs used async transforms previously silently skipped that work.
  (`SimpleFSM.process` already ran on the async engine, so it was unaffected.)
- **Synchronous push arcs now enter the sub-network.** Driving a push arc
  through a synchronous entry point (`FSM.execute`, sync batch/stream,
  `AdvancedFSM.execute_step_sync`) previously flat-traversed it — the
  sub-network was never entered. These paths now run the async engine's full
  push/pop subflow lifecycle (isolation, `data_mapping`/`result_mapping`,
  pre-validators), matching the documented intent.
- A push fired from a **regularly-entered** parent state now inherits that
  state's resources into the sub-network (inheritance seeds from
  `current_state_resources`, which the async regular-transition and
  initial-state entries now populate by routing through the shared state-entry
  path). A state's acquired resources are now **released on every exit** — a
  regular transition, a subflow pop (including the pushing state's resources
  held through the subflow), and **run completion** (the final/terminal state
  the run ends on, which is never "left", releases its resources when the run
  finishes) — closing a held-resource leak.
- **Batch and stream items now enter their initial state at full parity** with
  single-record execution. Each batch item / streamed record previously seeded
  its fresh context with a bare `set_state`, skipping the initial state's
  pre-validators, resource allocation, and **initial-state transforms**; they now
  route initial entry through the shared state-entry path, so a start-state
  transform runs for every item/record.
- **The synchronous `timeout=` now bounds the wait across the whole Simple
  API.** It is honored via the bridge (which cancels the in-flight coroutine and
  returns), instead of the previous `ThreadPoolExecutor` that blocked on
  `shutdown(wait=True)` until the coroutine finished anyway — so a slow run was
  cut short only nominally while the caller waited it out. `SimpleFSM.process`,
  `SimpleFSM.process_batch` / `process_stream`, and the `process_file` /
  `batch_process` module helpers all now bound the wait through the bridge. The
  `process_file` / `batch_process` helpers also gained a `custom_functions=`
  parameter (forwarded to `create_fsm`), so a caller can register transforms on
  the FSM the helper builds.
- The async execution engine now executes **push arcs**. Previously a push arc
  was treated as a flat transition on the async path, so the sub-network was
  never entered (`SimpleFSM.process()` runs on the async engine, so it was
  affected too). `AsyncExecutionEngine` now pushes the target sub-network onto
  the context stack, isolates the sub-network's data view via the shared
  `DataIsolationMode.apply` helper (the single source of truth — `copy` /
  `reference` / `serialize`), enters the sub-network's initial state, and pops
  back to the parent's return state when the sub-network reaches a final state —
  matching the synchronous engine's subflow lifecycle.
- The async subflow state entry now runs the sub-network initial state's
  **pre-validators** and allocates its own **state resources**, at parity with
  the synchronous engine — previously the async path only set the state and ran
  transforms, so sub-network pre-validators never ran and state resources were
  never allocated. A rejecting pre-validator now fails the push and rolls it
  back.
- Push-arc **`result_mapping`** is now applied when a subflow completes (mapping
  the sub-network's result fields back onto the parent's pre-push data). It was
  previously inert on both engines — the pop did not have the originating arc,
  and the parent-data snapshot it relied on was never recorded.
- A **nested** subflow that returns to a state which is itself a final state of
  the parent sub-network now unwinds every completed level in one step, instead
  of finalizing the whole run prematurely inside the parent sub-network.
- A push whose initial state cannot be resolved (an unknown `network:state`
  target) now leaves the context unchanged — the parent's data is no longer
  replaced by an orphaned isolated copy. The target is resolved before the push
  is committed; a state-entry failure after commit rolls back cleanly.

### Changed

- **FSM execution now runs on a single async engine; the synchronous APIs are
  thin wrappers over it.** All synchronous entry points drive the one
  `AsyncExecutionEngine` through an async→sync bridge
  (`dataknobs_common.SyncLoopBridge`) rather than a parallel synchronous engine.
  The public sync signatures and semantics are unchanged. Explicit-lifecycle
  objects — `SimpleFSM` and `AdvancedFSM.execute_step_sync` (repeated stepping) —
  share one long-lived bridge per FSM (obtained via `FSM.get_sync_bridge()`,
  released by `FSM.close()` / `SimpleFSM.close()` / `AdvancedFSM.close()`);
  `SimpleFSM` dropped its private event-loop thread in favor of it. The stateless one-shot surfaces —
  `FSM.execute` and the sync batch/stream executors — instead scope a throwaway
  bridge to the operation, so they need no `close()` and leave no
  process-lifetime thread behind. The async engine's regular-transition and
  initial-state entries now route through the shared `enter_state`, so state
  pre-validators and resource allocation behave identically on every entry path.
- The push-arc subflow lifecycle is now driven by shared, color-free helpers on
  `BaseExecutionEngine` (target parsing, initial-state resolution, data-mapping
  application, push commit, rollback, result mapping, and subflow-final-state
  detection), in addition to the `apply_data_mapping` / `apply_result_mapping`
  helpers (previously private to the sync engine). The synchronous and
  asynchronous engines now share one implementation of the push/pop logic and
  cannot drift; the per-push state needed for result mapping and rollback is
  tracked on a `SubflowFrame` stack on `ExecutionContext`.

### Removed

- **The standalone synchronous execution engine has been removed.** FSM
  execution now runs entirely on the single `AsyncExecutionEngine`; the public
  sync APIs remain and run on it through the async→sync bridge (see Changed). The
  deleted internals were the `ExecutionEngine` class
  (`dataknobs_fsm.execution.engine`) and the `NetworkExecutor`
  (`dataknobs_fsm.execution.network`), along with the synchronous, `*_async`-paired
  methods on `ArcExecution` — `execute()`, `can_execute()`, and the
  `execute_push()` stub (use `execute_async()` / `can_execute_async()`). The
  unused `ArcExecution.execute_with_transaction()` wrapper and `FSM.get_engine()`
  were also removed (`FSM.get_async_engine()` is the engine accessor). The
  `TraversalStrategy` enum moved from `dataknobs_fsm.execution.engine` to
  `dataknobs_fsm.execution.common`; it is still re-exported from
  `dataknobs_fsm.execution`, so `from dataknobs_fsm.execution import
  TraversalStrategy` is unaffected.

### Added

- `AdvancedFSM` gained a **lifecycle close**: `close()` (sync), `aclose()`
  (async), and sync/async context-manager support (`with` / `async with`).
  These stop and join the FSM's shared async→sync bridge thread (created lazily
  by repeated `execute_step_sync` stepping) and release the resource manager —
  so an `AdvancedFSM` that only ever stepped synchronously can release its
  bridge thread instead of leaving it alive until process exit.
- Push arcs now honor config-authored **`data_mapping`** and **`result_mapping`**
  (`PushArcConfig.data_mapping` / `result_mapping`). These thread through the
  builder to the runtime `PushArc`; previously the fields could not be expressed
  in config and were dropped at build time, so `result_mapping` was inert end to
  end.

### Added

- The ETL `validate` stage is now a real per-record gate. `ETLConfig.validation_schema`
  accepts a friendly dict schema (`{field: {required, type, min, max, pattern}}`),
  a library `IValidationFunction`, or a callable predicate `record -> bool`; a
  record that fails is diverted to a non-loading `rejected` terminal (never
  written to the target) and counted in a new `rejected` metric, distinct from
  `errors`. By default rejections do not trip `error_threshold` (validation is a
  data-quality filter, not a pipeline outage); the new
  `ETLConfig.reject_counts_as_error` (default `False`) opts them in for a strict
  gate. To validate against a reference table, set
  `ETLConfig.validation_resources` (`{name: {"type": ..., "config": ...}}`):
  each entry is registered as an FSM resource and bound on the `valid` arc, so a
  resource-reading `validation_schema` predicate resolves it from its
  `FunctionContext` (`require_resource(name)` / `resource_for_role(name)`).
  Setting `validation_resources` without `validation_schema` raises
  `InvalidConfigurationError` (the resources would never be bound to a gate).
- `dataknobs_fsm.functions.library.validators.build_record_validator(spec)`
  normalizes any of three validation-spec forms — a friendly dict schema, a
  library `IValidationFunction`, or a callable predicate (sync or async) — into
  the `(record, context) -> bool` gate the engine invokes as an arc condition.
  The ETL and file-processing patterns build their `validate` gate through it,
  so the friendly validation vocabulary is identical across both.
- The ETL `enrich` stage is now a real per-record step. `ETLConfig.enrichment_sources`
  is a list of enrichers applied in order between `transform` and `load`; each is
  a computed field→value map (static or callable values), a reference-table
  lookup (`{"database": <backend cfg>, "match": {record_field: reference_field},
  "fields": [...], "overwrite": bool}` — the looked-up fields are merged into the
  record), a library `ITransformFunction`, or a callable `record -> dict`. The
  reference lookup compiles to a `dataknobs-data` `Query` (backend-agnostic — no
  raw SQL) and reads through a non-blocking `async_database` resource. The new
  `ETLConfig.enrichment_on_missing` (`"ignore"` default / `"null"` / `"error"`)
  controls how a missed reference lookup is handled. An enrichment failure counts
  as an `error`; enrichment adds no new terminal or metric key. Per-record API
  lookups and multi-row fan-out joins are not yet wired (an `api` source is
  rejected at config validation).
- `dataknobs_fsm.functions.library.enrichers.build_record_enricher(spec)`
  normalizes any of four enrichment-spec forms — a field→value map, a
  reference-table lookup (`LookupMergeEnricher`, exported alongside it), an
  `ITransformFunction`, or a callable (sync or async) — into the
  `(record, context) -> dict` step the engine applies in the enrich stage. The
  computed and lookup forms share one collision decision (`_enrichment_collides`)
  and one write primitive (`merge_enrichment_field`), so they cannot diverge on
  `overwrite` handling. A reference-lookup spec is validated eagerly: a `match`
  join with no source key (a malformed lookup that would otherwise be mis-read as
  a field→value map), `overwrite` without explicit `fields` (a blanket merge-all
  could clobber the record's own key columns), and `on_missing="null"` without
  `fields` (nothing to null) are all rejected at construction rather than
  silently mis-enriching or no-op'ing at run time.
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
- `DatabaseTransaction` gains an `on_unsupported` isolation policy
  (`"strict"` / `"emulate"`) for the `begin` action: on a non-transactional
  backend `"strict"` (default, fail-closed) raises
  `CapabilityNotSupportedError` and `"emulate"` proceeds with best-effort
  buffer-and-flush. `AsyncDatabaseResourceAdapter.begin_transaction()` opens a
  buffered transaction on the backing `AsyncDatabase` (its `commit` / `rollback`
  flush / discard the staged writes), so an FSM can stage writes in one state
  and commit/roll-back in another. `commit` returns the flushed row count as
  `committed_count`; commit atomicity follows the handle's `is_atomic` flag (a
  single same-kind batch is all-or-nothing on a transactional backend, a mixed
  or upsert buffer commits as independent batches). A `commit` reaching a state
  with no active handle (a missing or failed prior `begin`) is logged at WARNING
  and commits nothing instead of reporting a phantom success; a handle-less
  `rollback` is a quiet no-op. `on_unsupported` is validated against the data
  layer's exported `VALID_TRANSACTION_POLICIES`, and a reserved `savepoint=`
  argument warns on use.
- Arc resource injection. An FSM arc may declare `resources`, and its transform
  **and** its condition (pre-test) receive them through
  `FunctionContext.resources` — on both the async and sync engines. A resource
  is acquired once for the scope of the arc invocation (the sync engine acquires
  before its retry loop and reuses the handles across attempts, matching the
  async engine) and released afterward, with no acquire timeout (arc resources
  carry no per-resource `timeout_seconds`). Condition delivery covers both the
  callable and the `IStateTestFunction` (`test(data, context)`) interface forms.
  `FunctionContext` gains `require_resource(name)` (name-based, raising a clear
  error when the resource was not declared) and `resource_for_role(role)`
  (role-based, resolving an arc's `{role: name}` map, also exposed at
  `metadata["resource_roles"]`) so one function can be reused across arcs that
  bind the same role to different resources. The built-in database function
  library now works on arcs, not just states.
- `ExecutionContext.transform_context_factory` is now honored on the async
  engine's state and arc **transform** paths (it was previously applied only on
  the synchronous arc path, so it was silently ignored for every transform run
  through the async engine). Arc conditions receive resources and the role map
  but keep the plain context (the factory's documented scope is transforms) —
  uniformly on every condition path, both engines (async `_evaluate_arc`, sync
  `_evaluate_pre_test`, and the sync `ArcExecution.can_execute` /
  `can_execute_async` used by the network engine).
- An arc's `resources` may be declared as a `{role: name}` map in config
  (`resources: {database: primary_db}`), not only a list of names. This makes
  role-based access (`FunctionContext.resource_for_role`) reachable directly
  from YAML/dict config; a list (`resources: [primary_db]`) still produces the
  identity `{name: name}` map.

### Fixed

- An arc condition that raises an *unexpected* error (a missing/down resource,
  a validator bug, a failing reference lookup) now surfaces as a record error
  instead of silently de-selecting the arc. Both engines previously swallowed
  every condition exception to `False`, which routed the record to the
  fall-through arc — so a validation gate whose reference table was down
  rejected every row while reporting `errors == 0`, hiding an infrastructure
  outage as a clean data-quality drop. Conditions now distinguish a soft reject
  from a hard failure: returning falsy (or raising `ValidationError`, the
  explicit "record is invalid" signal) de-selects the arc; any other exception
  propagates and the record is counted as an error (tripping `error_threshold`).
  The sync engine's `execute()` gained the same per-record error wrapper the
  async engine already had, so the behaviour is identical across engines.
- The friendly dict validation schema now separates presence from value:
  `required` (or the literal `True` shorthand) governs whether an *absent* field
  rejects, while `type` / `min` / `max` / `pattern` apply only when the field is
  *present*. So `{"score": {"min": 0}}` means "if present, score must be >= 0"
  (an absent optional field passes; add `"required": True` to also demand
  presence), and a *present* value that cannot satisfy a numeric bound (e.g. a
  string against `min`) rejects rather than raising `TypeError`. This replaces
  the promoted `_make_validator` behaviour where a missing field defaulted to
  `0` (so a `min` bound silently depended on whether `0` satisfied it). Shared
  by the ETL and file-processing gates.
- The ETL "error threshold exceeded" message now includes the rejected count
  when `reject_counts_as_error` is set, so it no longer reads a confusing
  "0 errors" when excess rejections (not errors) tripped the gate.
- Arc transforms that are `ITransformFunction` instances (such as the database
  functions) now run on the async engine. The async arc path previously invoked
  the function as a plain `(data, context)` callable against the raw
  `ExecutionContext` — an interface transform is not directly callable and never
  reached its resources, so an arc-referenced database function failed. Interface
  transforms are now dispatched deterministically with the resource-bearing
  context, and an `ExecutionResult` or `None` return is coalesced the same way as
  on the synchronous arc path.
- A resource declared on a network-level `{from, to}` arc is no longer dropped
  during config normalization. The loader copied only a subset of arc fields to
  the generated state-level arc and omitted `resources`, so an arc's declared
  resources silently never reached it.
- Sync arc conditions and transforms key their resources by **name**, matching
  the async engine. `ArcExecution` previously keyed arc resources by the
  role/type (the declaration key), so a function reading `resources["<name>"]`
  missed on a hand-built `{role: name}` arc; `ArcExecution.can_execute` /
  `can_execute_async` and the sync `ExecutionEngine._evaluate_pre_test` now also
  acquire the arc's declared resources for the condition (they previously built a
  resourceless context). The arc resource-release path now releases the
  arc-acquired resources it actually tracks (it previously read an attribute that
  was never populated, leaking the acquisitions).
- A raising async arc condition now de-selects only that arc instead of failing
  the whole FSM run, matching the synchronous engine. The async engine evaluates
  arc conditions as concurrent tasks; a predicate that raised (for example,
  `require_resource()` after a failed concurrent acquire) propagated out of the
  evaluator and aborted the run. The condition evaluator now treats a raising
  predicate as "arc unavailable", the same contract as the sync engine's
  `_evaluate_pre_test`.
- `DatabaseTransaction` now drives a real transaction. It previously called a
  `resource.begin_transaction()` method that no adapter implemented, so it
  raised `AttributeError` on first use; it now opens a buffered transaction
  through the new `AsyncDatabase.transaction()` capability. Construction
  validates `action` and `on_unsupported`, and `CapabilityNotSupportedError`
  surfaces unwrapped (not masked as a generic `TransformError`).
- `BatchCommit` / `commit_batch` now source their atomicity guarantee from the
  data-layer `AsyncDatabase.supports_transactions()` flag, replacing the interim
  per-backend allowlist with the canonical capability.
- The formerly silent transaction no-op sites are reconciled.
  `ExecutionContext.{start,commit,rollback}_transaction` no longer call a
  `hasattr`-guarded `self.database.<method>()` — which silently no-op'd on
  backends without the method and, once `AsyncDatabase.begin_transaction` became
  an async coroutine, would have invoked it un-awaited (a silent miss). They
  keep their in-memory logical bookkeeping and DEBUG-log the decoupling.
  `DatabaseStreamSink._commit_transaction` likewise drops its broken
  `self.database.commit()` call and the `except Exception: pass` that masked it.
- A non-default `transaction.strategy` (`batch` / `manual`) now logs a warning
  at build time. The in-memory `TransactionManager` it configures is not
  consulted by the execution engines to drive database commit/rollback, so the
  knob would otherwise silently fail to deliver database atomicity. Use the
  `DatabaseTransaction` function, `BatchCommit(atomicity="require")`, or
  `AsyncDatabase.transaction()` for database atomicity.
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

- Push-arc data isolation declared in FSM config
  (`copy`/`reference`/`serialize`) is now threaded through to the runtime push
  arc and honored by the executors that traverse a push arc's sub-network;
  previously the configured value was silently dropped at build time and the arc
  always carried the deep-copy default. The push-arc isolation config value
  (`PushArcConfig.data_isolation`) now uses the isolation enum
  (`DataIsolationMode`) rather than the state-level data-handling enum, so it
  expresses exactly the modes the runtime honors: `serialize` is newly
  expressible, and `direct` — which never had push-arc isolation semantics and
  was being dropped — is no longer accepted and raises a validation error at load
  on both the typed and the dict/YAML config paths (use `StateConfig.data_mode`
  for state-level DIRECT handling). Isolation is applied through a single shared
  `DataIsolationMode.apply` helper, so every executor isolates identically and
  `serialize` consistently uses the project JSON encoder (serializing the
  FSM-specific types stdlib `json` rejects: `FSMData`, `ExecutionResult`, and any
  object exposing `to_dict()`/`__json__()`). The public `NetworkExecutor` now
  honors all three modes when it runs a push arc's full sub-network (each mode
  runs the sub-network in a fresh execution context; only the data crossing the
  boundary varies, and `max_depth` is enforced across nested push arcs). The default high-level
  engines do not yet execute push arcs through a sub-network traversal (the async
  engine treats a push arc as a flat transition; the synchronous
  `ExecutionEngine.execute()` does not traverse sub-networks), so isolation takes
  effect wherever a sub-network is actually traversed and wiring it into those
  high-level engines remains future work.

  *Migration note for programmatic consumers:* `PushArcConfig.data_isolation` is
  now a `DataIsolationMode` member, not a `DataHandlingMode` member. Code that
  compared it against `DataHandlingMode.COPY`/`.REFERENCE` should compare against
  the `DataIsolationMode` members of the same name instead.
- `FileProcessingConfig.validation_schema` now also accepts a library
  `IValidationFunction` or a callable predicate, not only a dict schema (the
  three forms the ETL pattern accepts), via the shared `build_record_validator`.
  The friendly dict-schema behavior is unchanged.
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
- `ETLConfig.enrichment_sources`, previously accepted but silently ignored (a
  documented per-record passthrough), is now wired as a real enrich stage
  between `transform` and `load`. A source that was inert before will now run;
  a malformed source — a `database` source with no `match` join spec,
  `overwrite` without an explicit `fields` list, or `on_missing="null"` without
  `fields` — raises `InvalidConfigurationError` at `ETLConfig` construction
  instead of no-op'ing. Migration: add a `match` (and `fields`) to each
  reference source you intended to run, or remove sources you did not.

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
  metrics were hollow. (The `validate` and `enrich` stages are now real
  per-record steps — see *Added*.)
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
