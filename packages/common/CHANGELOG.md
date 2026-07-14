# Changelog

All notable changes to the dataknobs-common package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `sweep_stale_test_indices(host, port, *, prefixes=("test_",),
  min_age_seconds=None, now_ms=None)` in
  `dataknobs_common.testing.elasticsearch_fixtures`: an age-gated, best-effort
  reclamation of stale `test_*` Elasticsearch indices. Lists matching indices
  (a read) and deletes, by exact name, each one older than the threshold
  (default 300s, overridable via the `min_age_seconds` argument or the
  `DK_ES_TEST_INDEX_MAX_AGE_SECONDS` env var). Age-gating makes it safe under
  pytest-xdist — an in-flight index is never touched — and it never raises.
- `is_elasticsearch_available()` and the `requires_elasticsearch` pytest skip
  marker (`dataknobs_common.testing`), mirroring the existing
  `is_postgres_available` / `requires_postgres` pair, for gating
  Elasticsearch-dependent tests on a reachable cluster.
- `Capability.CONDITIONAL_WRITE` (consistency family): advertises that a
  backend enforces compare-and-set on `update`/`upsert`. A version token —
  read via `get_version` on a record store, or `get_state_version` on a
  metadata store — is passed back as `expected_version`, and a stale token
  raises `ConcurrencyError` instead of silent last-writer-wins. This is the
  single, layer-neutral identifier for the conditional-write contract, shared
  by the data-record backends and the knowledge-storage backends.

### Changed

- The shared `dataknobs_common_elasticsearch` pytest plugin now sweeps stale
  `test_*` indices once at session start (from `ensure_elasticsearch_ready`,
  after the readiness wait). Indices leaked by runs killed before fixture
  teardown no longer accumulate on the persistent dev/CI cluster volume and
  exhaust the single-node shard budget. The sweep is age-gated and best-effort,
  so it never deletes an in-flight index nor fails an otherwise-green run.

### Removed

- `Capability.TRANSACTIONAL_METADATA`. The conditional-write /
  optimistic-concurrency contract it described is now advertised under the
  layer-neutral `Capability.CONDITIONAL_WRITE`; the former name read as
  metadata-document-specific and did not fit a general record store. This is
  a breaking change for any code referencing the member by name.

## v1.5.2 - 2026-07-07

### Added

- `dataknobs_common.aws`: the shared AWS session-construction layer,
  exported from the package root. `AwsSessionConfig` is the normalized,
  frozen session config (region, credentials, endpoint, retry/pool
  tuning) accepting both boto-native and dataknobs-legacy alias keys;
  `create_aioboto3_session(config, *, warm_service="s3")` builds an
  `aioboto3.Session` and warms its botocore data-file caches on a worker
  thread (via `asyncio.to_thread`) so the first client creation by any
  consumer is a cache hit rather than a blocking data-file load on the
  event loop, with a process-wide cache keyed by session kwargs **and**
  `warm_service`; `clear_aioboto3_session_cache()` resets it (test
  isolation). This is the single source of truth every AWS consumer
  routes through — `SqsEventBus` here, and the S3 data backends and
  `S3KnowledgeBackend` — with future AWS services (e.g. a
  `bedrock-runtime` LLM provider) routing through the same surface. It
  lives in `dataknobs-common` (the lowest layer) precisely so
  `SqsEventBus` can share it without an illegal upward dependency on
  `dataknobs-data`. `aioboto3` is lazy-imported inside the factory's
  worker thread, so the base install stays `dependencies = []`; the new
  `aws` optional extra (`pip install 'dataknobs-common[aws]'`) declares
  the transport.
- `AwsSessionConfig` now validates credential completeness at
  construction (fail closed): a partial explicit credential — an access
  key without its secret (or vice versa), or a session token without
  both — raises `ConfigurationError` naming the missing field, instead of
  botocore's deferred `PartialCredentialsError` at first client use (which
  for the offloaded async factory would surface from a worker thread).
  Supplying neither credential stays valid and defers to boto's default
  chain. This also closes a silent-fallback hazard in the former SQS
  session path, where a lone credential was dropped and the caller
  silently authenticated as a different ambient identity.
- `dataknobs_common.testing.is_bedrock_available()` and the
  `requires_bedrock` pytest marker. The probe is conservative and never
  makes a paid API call: it gates on the `DK_TEST_BEDROCK` opt-in env var
  (truthy) **and** resolvable AWS credentials via a lazy botocore session,
  returning `False` (skip, never fail) otherwise. Amazon Bedrock has no
  faithful local emulator, so live tests are opt-in only; the marker joins
  `requires_ollama` / `requires_localstack` for cross-package reuse.
- `AwsSessionConfig.to_session_client_kwargs(*, connect_timeout=None,
  read_timeout=None)` — the session-based counterpart to
  `to_client_kwargs()` for consumers that open `session.client(service,
  ...)`. Since credentials and region ride on the session, it returns only
  the per-client knobs: a `botocore.config.Config` carrying retry / pool
  tuning plus explicit timeouts, `endpoint_url` with the `http://` →
  `use_ssl=False` inference, and the `extra_client_kwargs` passthrough. It
  is the single builder every async consumer that opens a client from a
  session shares (`SqsEventBus`, the Bedrock LLM provider), so retry / pool
  / endpoint / SSL / timeout handling cannot drift per consumer.
  `to_boto_config_kwargs()` gained optional `connect_timeout` /
  `read_timeout` keywords (omitted when unset).

### Changed

- `SqsEventBus.connect()` now builds its session through the shared
  `create_aioboto3_session(warm_service="sqs")` factory instead of an
  inline `aioboto3.Session(...)`. This fixes a blocking-I/O-on-the-event-
  loop issue (the first `session.client("sqs")` synchronously loaded
  botocore's bundled data files on the loop) and removes the local
  session-kwarg reinvention that stood in for an illegal `dataknobs-data`
  import. Session/region/credential normalization is now shared with
  every other AWS consumer via `AwsSessionConfig`, and its per-client
  kwargs (endpoint, timeouts, retry / pool `Config`) now route through the
  shared `AwsSessionConfig.to_session_client_kwargs()` builder rather than
  a local one — adding the `http://` → `use_ssl=False` inference for
  LocalStack / MinIO endpoints.

## v1.5.1 - 2026-06-29

### Added
- `SyncLoopBridge` and the `run_coro_sync` convenience function in
  `dataknobs_common.sync_bridge` (exported from the package root). They run
  a coroutine to completion from synchronous code on a private background
  event-loop thread, returning the coroutine's value or re-raising its
  exception with the traceback preserved. Because the loop runs on a
  separate thread, both are safe to call from inside an already-running
  event loop — the `asyncio.run()` / `loop.run_until_complete()` "loop
  already running" footgun (and the `nest_asyncio` workaround the dependency
  bar rejects) is avoided by construction. Both accept an optional `timeout`
  that raises `TimeoutError` and best-effort-cancels the coroutine if it
  overruns, and concurrent `run()` calls from multiple threads are
  supported. Calling `run()` from within a coroutine already running on the
  bridge's own loop raises `RuntimeError` (symmetric with `close()`) instead
  of self-deadlocking. This is the async→sync counterpart to the existing
  sync→async `aiter_sync_in_thread`. Use a long-lived `SyncLoopBridge` for a
  synchronous wrapper that makes repeated calls; use `run_coro_sync` for
  one-off calls.

## v1.5.0 - 2026-06-22

### Added
- `TenantContext` Protocol plus `SingleTenantContext`,
  `BoundTenantContext`, `PrefixedTenantContext`, and
  `SharedCorpusTenantContext` reference implementations in
  `dataknobs_common.tenancy` (exported from the package root). A
  `TenantContext` describes the tenant scope of a backend operation;
  backends use it to compute distributed-lock keys (`lock_key(operation)`)
  and state-storage prefixes (`state_key_prefix()`) for per-tenant
  isolation, and to compare scopes (`matches(other)`). `SingleTenantContext`
  is the backwards-compatible default — its empty state prefix preserves the
  on-disk state layout of single-tenant code written before the context
  surface existed, its lock keys follow a stable canonical format, and it
  accepts `str` equality for incremental migration. Multi-tenant impls
  compose tenant + domain into lock keys and state prefixes. Build a context
  from a config mapping with
  `create_tenant_context({...})` or from `DOMAIN_ID` / `TENANT_ID`
  environment variables with `tenant_context_from_env()`.
- `Capability.STATE_BRIDGE_INBOX_ONLY` and
  `Capability.STATE_BRIDGE_BIDIRECTIONAL` members (with a `state_bridge`
  entry in `CAPABILITY_FAMILIES`). A host advertises these to declare how
  it bridges named-key state between components — consume-on-read inbox
  semantics versus symmetric read/write.
- `ScopeProjector` Protocol plus `IdentityProjector`, `ReadOnlyProjector`,
  `WhitelistProjector`, `ChainedProjector`, `CallableProjector`, and
  `CachedProjector` reference implementations in `dataknobs_common.scope`
  (exported from the package root). A `ScopeProjector` projects a runtime
  state-bearing value into a `Mapping[str, Any]` suitable for a safe-eval
  scope, a template-evaluation context, or any key-lookup-by-name surface.
  Consumers compose projectors to expose state read-only
  (`ReadOnlyProjector`), filter to a declared key set
  (`WhitelistProjector`), chain projections with later-wins merge
  (`ChainedProjector`), or memoize an expensive projection
  (`CachedProjector`). The `Capability.SCOPE_PROJECTOR_READ_ONLY` member
  lets a host advertise that its eval scope is read-only-wrapped.
- `aiter_sync_in_thread(make_iter, *, max_buffer=32)` in
  `dataknobs_common.async_iter` (exported from the package root). Drives a
  *lazy, blocking* synchronous iterator on a worker thread and pumps its
  items to an async consumer across a bounded queue, so an `async def`
  can iterate a sync generator it cannot rewrite as async — a streaming
  file/format parser, a sync DB cursor, a paginated SDK iterator —
  without stalling the event loop. Unlike `asyncio.to_thread(list, it)`
  it does not buffer the whole iterator: the producer blocks once
  `max_buffer` items are unconsumed (backpressure), so streaming is
  preserved. Abandoned iteration (`break` / exception / cancellation)
  signals the producer to stop, `close()`s the source iterator (releasing
  its file handle / cursor), and joins the worker thread — no leaked
  thread, no dangling handle; the worker is a daemon so it can never
  block process exit. Exceptions from the factory or during iteration
  propagate to the consumer. The waiting consumer parks on an
  `asyncio.Event` (no executor polling), so many concurrent streams do
  not consume the default thread-pool, and the teardown thread-join is
  cancellation-shielded. `max_buffer` must be `>= 1` (a `0`/unbounded
  queue would defeat backpressure) and defaults to `32`.
- `assert_no_blocking()` context manager and the `no_blocking` pytest
  fixture in `dataknobs_common.testing` (auto-discovered via the
  `dataknobs_common_blocking` pytest11 plugin). A runtime detector that
  fails a test if a blocking syscall (`time.sleep`, `open`, blocking
  socket/os reads and writes, the sync `sqlite3` driver, …) runs on the
  event loop inside the block — turning an otherwise-invisible
  loop-blocking defect into a deterministic, reproduce-first test
  failure. Wrap the awaited operation under test: the block fails against
  a backend that blocks the loop and passes once it uses an async
  transport or offloads via `asyncio.to_thread`. Backed by the
  `blockbuster` dev/test-only dependency (never imported by runtime
  code); `is_blockbuster_available()` and `blocking_error_type()` gate
  and assert against it, and the ready-made `requires_blockbuster`
  skip marker decorates such tests so they skip cleanly when the
  dependency is absent (preferred over a hand-rolled `skipif`).
  Consumers guarding their own async backends add `blockbuster` to
  their dev dependencies and get the construct for free.
- `close_if_owned(resource, owns, *, on_error=None)` and its synchronous
  counterpart `close_if_owned_sync(...)` in `dataknobs_common.lifecycle`
  (also re-exported from the top-level namespace). The canonical
  owned-vs-injected teardown guard: a collaborator is closed only when
  the holder owns it (`owns` is True), it is not None, and it exposes a
  `close()` method — an injected collaborator is left open for its owner.
  Pass `on_error` to error-isolate the close (catch the exception and
  hand it to a callback, e.g. a logger) so one failing subsystem in a
  teardown cascade does not abort the rest; `asyncio.CancelledError`
  always propagates. Used across the data, bots, llm, and fsm packages
  to keep the ownership-gated close idiom consistent.
- Three `Capability` identifiers in the observability family:
  `INGEST_EVENT_PUBLICATION`, `BACKEND_STATE_OBSERVABILITY`, and
  `EXECUTION_TRACKING`. Knowledge-ingestion, backend-state, and
  tool-execution surfaces advertise these so consumers can query
  support before composing observability callbacks.
- `CallbackRegistry[CallbackT]` and the `CallbackOrdering` Protocol in
  `dataknobs_common.callbacks` (also re-exported from the top-level
  namespace). In-process named-callback dispatch with sync and async
  fire, pluggable ordering, and per-registry error policy. Built-in
  orderings: `FIFOOrdering` (default; insertion order),
  `PriorityOrdering` (numeric priority, lower fires first, FIFO
  tiebreaker), `StageOrdering` (named stages; unknown stages sort to
  the end), and `CompositeOrdering` (first non-zero compare wins).
  Error policies: `ErrorPolicy.RAISE` (re-raise the first failure,
  abort the rest), `ErrorPolicy.LOG_AND_CONTINUE` (default — log and
  continue), `ErrorPolicy.LOG_AND_RAISE_AT_END` (run every callback
  then raise `BatchedCallbackError` carrying every failure). Testing
  constructs `CapturingCallbackRegistry` (real dispatch plus
  per-fire `(topic, payload)` capture) and `RecordingCallbackRegistry`
  (records without dispatching to registered callbacks) ship in the
  same module so consumer test suites import them like `EchoProvider`.
  Calling `clear()` drains entries in place — the registry instance,
  its ordering, and its error policy survive across clears so external
  references stay valid.
- Compose a `CallbackRegistry` with any
  `dataknobs_common.events.EventBus` via
  `registry.also_publish_to(bus, topic_prefix="…")`: every fired
  payload is additionally published to the bus under
  `topic_prefix + topic`, wrapped in an `Event` carrying
  `EventType.CUSTOM`. Local callbacks still run; the bus is the
  cross-replica observability substrate. Multiple fan-out targets
  compose by calling `also_publish_to` once per target.
  Fan-out is non-load-bearing observability: by default a failed bus
  `publish` is logged and swallowed so it can neither abort nor mask
  the operation being observed. A consumer needing publish failures to
  be fatal opts out per-target with
  `also_publish_to(bus, isolate_errors=False)`;
  `asyncio.CancelledError` always propagates regardless.
  `registry.supports_event_bus_emission()` exposes whether at least
  one fan-out target is configured without reaching into private
  state. Calling `fire()` (sync) with fan-out configured from inside
  a running event loop raises `TypeError`; consumers must use
  `fire_async` to guarantee the bus publish completes before dispatch
  returns and to surface any publish failure. The no-running-loop
  branch of `fire()` drives fan-out to completion via `asyncio.run`
  before returning. `RecordingCallbackRegistry` (testing double)
  implements the full duck-typed surface — `register` / `unregister`
  / `clear` / `set_ordering` / `callback_count` / `fire` /
  `fire_async` — so injecting it where a real `CallbackRegistry` is
  expected does not raise `AttributeError`.
- `BackendRegistry[T_co]` runtime_checkable Protocol in
  `dataknobs_common.registry` (also re-exported from
  `dataknobs_common`). Stable `isinstance` target for "is this thing a
  registry-like object?" across both the item-shape `Registry` and the
  factory-shape `PluginRegistry`. Deliberately minimal — only `name`
  (property), `has(key)`, `list_keys()`, and `unregister(key)` —
  covering the four methods every registry-like adopter must offer.
  Both `Registry` and `PluginRegistry` structurally conform without
  any inheritance refactor. Joins `ResourceResolver`, `Discriminator`,
  and `CapabilityContract` as the cross-cutting Protocols re-exported
  from the top-level namespace. Consumers needing
  registry-kind-specific methods (`create` / `create_async` /
  `get_factory` for `PluginRegistry`; `get_metrics` / `list_items` /
  `items` / `count` / `clear` for `Registry`) should `isinstance`
  against the concrete class.
- `PluginRegistry.create()` and `PluginRegistry.create_async()` shape
  their not-found error via two opt-in constructor kwargs:
  `not_found_kind: str | None = None` and
  `not_found_exception: type[Exception] = NotFoundError`. The default
  shape is unchanged (`NotFoundError("Plugin '<key>' not registered",
  context={...})` — the principled `DataknobsError`-rooted form
  consumers catch programmatically). Setting
  `not_found_kind="event bus backend"` plus
  `not_found_exception=ValueError` produces
  `ValueError("Unknown event bus backend: <key>. Available backends:
  <sorted-keys>")` for domain shims preserving a historical
  `ValueError` contract. Non-`DataknobsError` exception classes
  receive the message only (no `context=` kwarg, which would crash
  stdlib exceptions).
- `PluginRegistry.register()` accepts `allow_overwrite=` as a
  keyword-only alias for `override=`, matching the
  `Registry.register()` spelling. When explicitly `True` or `False`,
  `allow_overwrite` wins; `None` (the default) leaves `override`
  unmodified. Positional registrations and `override=`-keyword
  registrations behave identically to before this alias was added.
- `PluginRegistry.has(key)` delegates to `is_registered(key)`. Added so
  `PluginRegistry` structurally conforms to the new `BackendRegistry`
  Protocol.
- `PluginRegistry` documents the per-input-shape registry split
  convention in its class docstring (with a worked example): when
  adopting `PluginRegistry` for a Protocol parameterized by input
  shape (`ResourceResolver[KeyT, ValueT]`, `Discriminator[InputT,
  KindT]`), prefer N typed registries — one per concrete input shape —
  over one flat registry with `validate_type=Any`. The typed
  `validate_type=` is load-bearing under consumer-extensibility: an
  out-of-tree backend conforming to the wrong Protocol shape would
  silently register and only fail at use-time without the constraint.
- Declarative capability advertisement in `dataknobs_common.capabilities`
  (also re-exported from `dataknobs_common`). The `Capability` enum
  declares stable identifiers for cross-cutting optional features
  organized into families: tenancy (`TENANT_SCOPED_CHUNKS`,
  `TENANT_SCOPED_LOCKS`, `TENANT_SCOPED_STATE`,
  `PER_TENANT_RATE_LIMITS`), observability
  (`EVENT_BUS_EMISSION`, `CALLBACK_REGISTRY`, `METRICS_EMISSION`),
  consistency (`SNAPSHOT_ISOLATION`, `TRANSACTIONAL_METADATA`,
  `STREAMING_READS`, `STREAMING_WRITES`), and composition
  (`KEY_PATTERN_FILTERING`, `CHANGE_SUBSCRIPTION`). Members are
  `str`-typed for stable serialization and JSON-friendly logging. The
  `CAPABILITY_FAMILIES` mapping (also re-exported) gives consumers a
  precomputed `family → frozenset[Capability]` lookup so "all tenancy
  capabilities" is a one-liner rather than a hand-rolled set. Classes
  advertise via the `CapabilityContract` protocol — `CapabilityMixin`
  reads from a
  `SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]]`
  declaration (the `CapabilityLike = Capability | str` alias is also
  re-exported, so consumer-defined raw-string capabilities require no
  `# type: ignore`); `DynamicCapabilityMixin` computes the capability
  set from `__init__` state via the `_compute_instance_capabilities()`
  override hook with a cache invalidated through
  `_invalidate_capability_cache()`. Subclasses of
  `DynamicCapabilityMixin` are responsible for their own
  `super().__init__(...)` chain and MUST call
  `_init_capability_cache()` from their `__init__` — the mixin does
  not forward `__init__` args through cooperative multiple inheritance,
  avoiding the `*args, **kwargs` → `object.__init__` collision when an
  adopter pairs the mixin with another base class. List the mixin
  FIRST among bases. `require_capability(host, capability)` is a
  pre-call guard raising `CapabilityNotSupportedError`;
  `CapabilityNotSupportedError` extends `OperationError` (and thereby
  `DataknobsError`), records the offending capability identifier and
  host class name on `context` for structured logging, and exposes the
  same data via the `.capability` / `.host` attribute pair.
  Consumer-defined capabilities are supported via raw strings —
  `supports()` and `require_capability()` accept either `Capability`
  enum members or arbitrary strings.
- Composing reference implementations in `dataknobs_common.discriminator`
  (also re-exported from `dataknobs_common`): `CallableDiscriminator`
  (wraps a `Callable[[InputT], KindT]` for ad-hoc classifier construction),
  `MappingDiscriminator` (fast lookup against a static `Mapping` with a
  `default` fallback; `frozen=True, eq=False` so identity hashing
  applies — auto-generated value equality would make instances
  unhashable because the typical `mapping` argument is a `dict`),
  `MultiFieldDiscriminator` (classifies multiple fields of a
  mapping-shaped payload via per-field discriminators — missing fields
  surface as `None` in the result dict so consumers can distinguish
  "absent" from "classified as None"; result key order matches
  `field_discriminators` iteration order, which is insertion-ordered
  when a `dict` literal is passed and unspecified for hand-rolled
  `Mapping` impls without an ordered iteration contract), and
  `ChainedDiscriminator` (tries each inner discriminator in order; the
  first result that differs from `default` wins). Async siblings
  `AsyncCallableDiscriminator` and `AsyncChainedDiscriminator` round
  out the surface; the latter mixes sync and async inner
  discriminators in a single chain. Useful for event-routing pipelines
  that classify multiple aspects of a payload through one composable
  Protocol surface, and for layered classification (cheap rule →
  expensive fallback).
- Generic resource-lookup-by-key Protocols
  (`ResourceResolver[KeyT, ValueT]` and
  `AsyncResourceResolver[KeyT, ValueT]`) in `dataknobs_common.resolver`,
  also re-exported from `dataknobs_common`. Single `resolve(key)` method
  returning the value or `None`; idempotent at the protocol level and
  non-mutating on the input key. Reference implementations include
  `MappingResolver` (wraps a `Mapping`), `CallableResolver`,
  `DefaultingResolver` (substitutes a default for `None`; the
  `resolve()` return type is tightened to `_ValueT` rather than the
  Protocol's `_ValueT | None` since this wrapper's whole purpose is to
  eliminate the `None` case for consumers),
  `CachedResolver` (LRU; `None` returns are NOT cached so transient
  misses don't persist), `CompositeResolver` (first non-`None` wins),
  `NullResolver` (always returns `None`), and async siblings
  `AsyncCallableResolver` and `AsyncCachedResolver` (asyncio-Lock-guarded
  insertion). Five vector-store partition resolvers
  (`NullPartitionResolver`, `MetadataKeyPartitionResolver`,
  `TemporalPartitionResolver`, `CallablePartitionResolver`,
  `JoiningPartitionResolver`) cover the common partition shapes:
  per-tenant, per-content-type, temporal bucketing
  (`year`/`quarter`/`month` — bucket validated at construction),
  arbitrary callable, and composite combinations joined by a separator.
  `JoiningPartitionResolver` is distinct from `CompositeResolver`:
  it CONCATENATES every inner resolver's output, while
  `CompositeResolver` ALTERNATES (first-non-`None` wins).
  `MetadataKeyPartitionResolver` rejects non-scalar metadata values
  (list, dict, set, custom objects) by returning `default` rather than
  silently `str()`-coercing them into garbage partition names.
- `Discriminator[InputT, KindT]` and `AsyncDiscriminator[InputT, KindT]`
  Protocols (`@runtime_checkable`, in `dataknobs_common.discriminator`,
  also re-exported from `dataknobs_common`). One generic shape for
  value-to-kind classification across the codebase — backend-key
  classifiers, ML-backed label classifiers, and payload-field
  discriminators can all satisfy a single Protocol shape rather than
  N specific interfaces. `InputT` is contravariant and `KindT` is
  covariant, matching the substitution rules a consumer would expect
  when composing classification pipelines. Implementations MUST be
  deterministic (same input → same kind within a process) and MUST NOT
  mutate the input value.
- `dataknobs_common.expressions.SAFE_BUILTINS` gains
  `frozenset`, `sum`, `any`, `all`, and `reversed`. These extend the
  safe-builtin allowlist used by `safe_eval` / `safe_eval_value`
  and shared across every config-authored expression context
  (wizard transition conditions, field derivation expressions, and
  any other consumer that builds on the helper). Rationale:
  `any` / `all` unlock the idiomatic predicate shape used by the
  wizard `intent_confirm:` primitive's synthesized `on_no_match`
  condition (`not any(data.get(k) for k in [...])`); `sum` covers
  the common `sum(1 for x in items if cond)` counting / aggregation
  pattern in derivation expressions; `reversed` and `frozenset`
  round out the symmetry with the existing `sorted` and the
  existing `str`/`int`/`list`/`dict`/`tuple`/`set` type-constructor
  group. The allowlist still excludes `getattr`, `setattr`,
  `delattr`, `globals`, `locals`, `compile`, `eval`, `exec`,
  `__import__`, `open`, and `breakpoint`.
- `StructuredConfigConsumer.forwardable_components()` and the
  `INTERNAL_COMPONENTS: ClassVar[frozenset[str]]` ClassVar — the
  documented mixin contract for composing classes that build child
  consumers from a registry. A subclass declares the names of
  collaborators it consumes itself (default: empty); the helper returns
  a fresh dict of `self.components` minus those names, ready to spread
  into a child's construction call. Strictly additive — the empty
  default makes it a no-op for the 30+ existing adopters that don't
  compose children. First in-tree adopter is `WizardReasoning` (see the
  `dataknobs-bots` CHANGELOG); the pattern is the documented extension
  contract for consumer-authored composing strategies (see the
  `dataknobs-bots` USER_GUIDE "Building your own composing strategy"
  subsection).
- `dataknobs_common.testing.assert_dataclass_config_matches_ctor` and
  `assert_factory_kwargs_match_ctor` now default-ignore the
  `_forwarded_components` ctor parameter (alongside the existing `self`,
  `config`, and `_components`). Mixin adopters with a back-compat
  positional ctor shape (the `WizardReasoning` / `ResourcePool` pattern)
  can use the documented `_forwarded_components` plumbing keyword to
  capture `from_config`'s `**kwargs` and route them into the mixin's
  `_components` channel without polluting the parity test with a
  per-class allowlist entry.
- `dataknobs_common.events.create_event_bus_async(config)` — async
  counterpart to `create_event_bus`. Dispatches through
  `event_bus_backends.create_async(config=config)` so out-of-tree
  backends exposing `from_config_async` are detected and awaited. Today
  every built-in backend constructs synchronously, so the async shim
  returns the same instance type as the sync shim; the surface is
  shipped for API symmetry and consumer-extensibility.
- `dataknobs_common.locks.create_lock_async(config)` — async
  counterpart to `create_lock`. Dispatches through
  `lock_backends.create_async(config=config)` so out-of-tree backends
  exposing `from_config_async` are detected and awaited. Today every
  built-in backend constructs synchronously, so the async shim returns
  the same instance type as the sync shim; the surface is shipped for
  API symmetry and consumer-extensibility.
- `dataknobs_common.ratelimit.rate_limiter_backends` —
  `PluginRegistry[RateLimiter]` exposing the consumer-extensibility
  surface that previously did not exist for rate limiters. Previously,
  the only way to add a backend was to fork
  `dataknobs_common.ratelimit.limiter` and edit its `if/elif` chain.
  Register a custom backend with
  `rate_limiter_backends.register("name", factory)` and select it via
  `create_rate_limiter({"backend": "name", ...})`. Factory signature is
  `(config: dict, *, parsed: RateLimiterConfig) -> RateLimiter` — the
  parsed `RateLimiterConfig` is forwarded from the shim so backends
  don't re-parse the rate / category config. Conforms to
  `BackendRegistry`. Also re-exported from the top-level
  `dataknobs_common` namespace.
- `dataknobs_common.ratelimit.create_rate_limiter_async(config)` —
  async counterpart to `create_rate_limiter`. Performs the same
  top-level rate / category normalization as the sync shim ahead of the
  registry dispatch, then dispatches through
  `rate_limiter_backends.create_async(config=config, parsed=parsed)` so
  out-of-tree backends exposing `from_config_async` are detected and
  awaited. Today every built-in backend constructs synchronously, so
  the async shim returns the same instance type as the sync shim; the
  surface is shipped for API symmetry and consumer-extensibility. Also
  re-exported from the top-level `dataknobs_common` namespace.
- `dataknobs_common.resolver.resolver_backends` —
  `PluginRegistry[ResourceResolver[Any, Any]]` exposing a
  consumer-extensibility surface for generic key→value resolvers.
  Built-in factories: `"mapping"`, `"callable"`, `"composite"`,
  `"defaulting"`, `"cached"`, `"null"` (one per in-tree reference
  implementation). Factory signature is
  `(config: dict[str, Any]) -> ResourceResolver[Any, Any]`. Register a
  custom backend with `resolver_backends.register("name", factory)` and
  select it via `resolver_backends.create({"backend": "name", ...})` —
  or, when the discriminator field is unset, the default `"mapping"`
  factory is selected (the `config_key_default` semantic), letting a
  `{"mapping": {...}}` config skip the discriminator entirely.
  `validate_type=ResourceResolver` pins the registered backends to the
  `ResourceResolver` Protocol so a structurally non-conforming factory
  return fails fast at `create()`-time rather than at use-time.
  Conforms to `BackendRegistry`. Also re-exported from the top-level
  `dataknobs_common` namespace.
- `dataknobs_common.resolver.partition_resolver_backends` —
  `PluginRegistry[Any]` exposing a consumer-extensibility surface for
  vector-store partition resolvers (record→partition-name lookups).
  Built-in factories: `"null"` (default), `"metadata_key"`,
  `"temporal"`, `"callable"`, `"joining"`. Distinct namespace from
  `resolver_backends` per the per-input-shape split convention
  documented on `PluginRegistry` — partition resolvers key on a record
  rather than a `KeyT`, and no declared partition Protocol exists, so
  `validate_type=` is intentionally unset (a flat registry with
  `validate_type=ResourceResolver` would silently accept either input
  shape and only fail at use-time). Register a custom backend with
  `partition_resolver_backends.register("name", factory)` and select it
  via `partition_resolver_backends.create({"backend": "name", ...})`.
  Conforms to `BackendRegistry`. Also re-exported from the top-level
  `dataknobs_common` namespace. Neither resolver registry ships a
  standalone `create_resolver()` / `create_partition_resolver()`
  convenience shim (asymmetric with the `create_event_bus()` /
  `create_lock()` / `create_rate_limiter()` shape from earlier in the
  series) — the registries are surfaced directly. There is no top-level
  rate / category parsing step to perform ahead of dispatch (which is
  what justified the shim in the rate-limiter case), and a
  `{"mapping": {...}}` config can already skip the discriminator via
  `config_key_default="mapping"`, so a shim would add no value over
  `resolver_backends.create({...})`.

### Changed
- ruff's `ASYNC` lint family (`flake8-async`) is now enforced for this
  package, so blocking I/O on the event loop inside `async def` code is
  caught at lint time; the `assert_no_blocking()` construct proves the
  runtime behavior. See the `async-transport` authoring rule.
- **Breaking:** Backend-factory construction errors raised by
  `create_event_bus()` / `create_event_bus_async()` are now wrapped in
  `OperationError` (with the original exception preserved via
  `__cause__`) — previously they propagated unwrapped as the originating
  type. Each built-in backend raises its own exception type from its
  construction path: `SqsEventBus` with a missing `queue_url` raises
  `ValueError` from `SqsEventBusConfig.__post_init__`; `PostgresEventBus`
  raises `ConfigurationError` from `normalize_postgres_connection_config`
  when no connection config is resolvable, or `ValueError` from
  `PostgresEventBusConfig.__post_init__` for an empty-after-sanitization
  `channel_prefix`; `InMemoryEventBus` and `RedisEventBus` have no
  construction-time validation today (defaulted fields, no
  `__post_init__`), so they don't surface this change in practice — but
  if a future validation is added or an out-of-tree backend's factory
  raises during construction, the same wrapping applies. The
  unknown-backend path is NOT affected — it still raises `ValueError`
  with the same `"Unknown event bus backend: <key>. Available backends:
  …"` message text, and it is raised *before* the `OperationError`
  wrapper engages, so it can never surface as `OperationError`.
  Consumers catching the originating type around `create_event_bus()`
  to catch *construction* failures should switch to catching the
  common base `DataknobsError` (which covers `OperationError` and
  `ConfigurationError`) — the unknown-backend `ValueError` propagates
  separately and needs no special-casing to distinguish. Tests
  asserting on specific exception types around invalid construction
  config need the same update.
- `dataknobs_common.events.event_bus_backends` is now a
  `PluginRegistry[EventBus]` (was `Registry[EventBusFactory]`). The
  registration surface (`event_bus_backends.register("name", factory)`),
  the `create_event_bus({"backend": "name", ...})` resolution shape,
  the unknown-backend error message text, and the `ValueError`
  exception class (for the unknown-backend path) are unchanged. The
  `EventBusFactory` typealias is preserved. Consumers checking "is
  this a registry-like thing?" via
  `isinstance(event_bus_backends, Registry)` should switch to
  `isinstance(event_bus_backends, BackendRegistry)` — the shared
  runtime_checkable Protocol both `Registry` and `PluginRegistry`
  structurally conform to. The `event_bus_backends.unregister(name)`
  return value is now `None` (was the previously-registered factory);
  existing in-tree usages discard the return value.
- `create_event_bus_async` is now re-exported from the top-level
  `dataknobs_common` namespace (was only reachable via
  `dataknobs_common.events`) for symmetry with the existing top-level
  `create_event_bus` re-export.
- **Breaking:** Backend-factory construction errors raised by
  `create_lock()` / `create_lock_async()` are now wrapped in
  `OperationError` (with the original exception preserved via
  `__cause__`) — previously they propagated unwrapped as the
  originating type. Each built-in backend raises its own exception
  type from its construction path: `PostgresAdvisoryLock` raises
  `ConfigurationError` from `normalize_postgres_connection_config`
  when no connection config is resolvable, or `ValueError` from
  `_validate_url_component` (via the normalizer) when `host` or
  `database` contain shell-unsafe characters; `InProcessLock` has no
  construction-time validation today (defaulted fields, no
  `__post_init__`), so it doesn't surface this change in practice —
  but if a future validation is added or an out-of-tree backend's
  factory raises during construction, the same wrapping applies. The
  unknown-backend path is NOT affected — it still raises `ValueError`
  with the same `"Unknown lock backend: <key>. Available backends:
  …"` message text, and it is raised *before* the `OperationError`
  wrapper engages, so it can never surface as `OperationError`.
  Consumers catching the originating type around `create_lock()` to
  catch *construction* failures should switch to catching the common
  base `DataknobsError` (which covers `OperationError` and
  `ConfigurationError`) — the unknown-backend `ValueError` propagates
  separately and needs no special-casing to distinguish. Tests
  asserting on specific exception types around invalid construction
  config need the same update.
- `dataknobs_common.locks.lock_backends` is now a
  `PluginRegistry[DistributedLock]` (was `Registry[LockFactory]`). The
  registration surface (`lock_backends.register("name", factory)`),
  the `create_lock({"backend": "name", ...})` resolution shape, the
  unknown-backend error message text, and the `ValueError` exception
  class (for the unknown-backend path) are unchanged. The
  `LockFactory` typealias is preserved. Consumers checking "is this a
  registry-like thing?" via `isinstance(lock_backends, Registry)`
  should switch to `isinstance(lock_backends, BackendRegistry)` — the
  shared runtime_checkable Protocol both `Registry` and
  `PluginRegistry` structurally conform to. The
  `lock_backends.unregister(name)` return value is now `None` (was
  the previously-registered factory); existing in-tree usages discard
  the return value.
- `create_lock_async` is re-exported from the top-level
  `dataknobs_common` namespace alongside the existing
  `create_lock` re-export, matching the events-side symmetry.
- **Breaking:** Backend-factory construction errors raised by
  `create_rate_limiter()` / `create_rate_limiter_async()` are now
  wrapped in `OperationError` (with the original exception preserved
  via `__cause__`) — previously they propagated unwrapped as the
  originating type. Each built-in backend raises its own exception type
  from its construction path: `PyrateRateLimiter` raises `ImportError`
  when `pyrate-limiter` is not installed, or `ValueError` from its
  bucket-config validation. `InMemoryRateLimiter` has no
  construction-time validation today (no `__post_init__`), so it
  doesn't surface this change in practice — but if a future validation
  is added or an out-of-tree backend's factory raises during
  construction, the same wrapping applies. The top-level
  rate / category normalization (`_parse_config`) runs *before* the
  registry dispatch, so missing-`rates` and malformed-rate-dict errors
  still propagate as `ValueError` unwrapped (they're caught before any
  backend factory engages). The unknown-backend path is similarly NOT
  affected — it still raises `ValueError` with the same `"Unknown rate
  limiter backend: <key>. Available backends: …"` message text, and is
  raised by the registry *before* the `OperationError` wrapper engages.
  Consumers catching the originating type around `create_rate_limiter()`
  to catch *construction* failures (e.g. missing optional pyrate
  dependency) should switch to catching the common base `DataknobsError`
  (which covers `OperationError`) — the normalization `ValueError` and
  the unknown-backend `ValueError` continue to propagate separately and
  need no special-casing to distinguish.
- `dataknobs_common.ratelimit.create_rate_limiter` now dispatches the
  `backend` key through the new `rate_limiter_backends` registry
  instead of an inline `if/elif` chain. The shim continues to run the
  top-level rate / category normalization (`_parse_config`) ahead of
  the registry dispatch, so backend factories receive both the raw
  `config` dict and the parsed `RateLimiterConfig` via the keyword
  `parsed=`. The default backend (`"memory"`) and the public function
  signature are unchanged. The unknown-backend error message text
  (`"Unknown rate limiter backend: <key>. Available backends: …"`) and
  the `ValueError` exception class are unchanged.

### Fixed
- `PyrateRateLimiter.try_acquire` (and therefore `acquire`, which polls
  it) now performs its bucket I/O without blocking the event loop when
  configured with a blocking bucket backend (`sqlite` / `redis` /
  `postgres`). Those backends drive a synchronous bucket transport
  (disk / socket) — both the per-acquire bucket operation and the lazy
  first-call bucket construction previously ran on the loop, stalling it.
  The synchronous call is now offloaded onto a worker thread via
  `asyncio.to_thread`. The offload is gated on the bucket backend, decided
  once at construction: the default `memory` bucket is pure in-memory and
  stays on the loop, so the common hot path keeps its zero-thread-dispatch
  behaviour. Limit enforcement, ordering, and the async-bucket contract are
  unchanged; no public signature changed and no new runtime dependency was
  added.
- `PyrateRateLimiter.close()` and `reset()` now release the bucket transport
  they own instead of only dropping the in-memory bucket registry. For the
  `sqlite` backend the connection this limiter opened is closed (previously
  it leaked until garbage collection); for `redis` the client this limiter
  built is closed. A caller-supplied `postgres` pool is deliberately left
  open (the caller owns it). The release runs off the event loop via
  `asyncio.to_thread` for blocking backends, matching `try_acquire`.
- `PyrateRateLimiter.acquire()` now backs off its poll interval
  exponentially (capped at 1s) instead of polling every 50ms. For blocking
  bucket backends each poll dispatches a worker thread, so a long wait under
  many concurrent waiters no longer risks saturating the shared default
  executor; short waits stay responsive (the first poll is still 50ms) and
  the timeout deadline is still honoured promptly.
- `dataknobs_common.expressions._validate_ast` now blocks
  `.format()` and `.format_map()` calls on any attribute. The
  format-spec mini-language performs runtime attribute access via
  `{N.attr}` syntax that bypasses the existing AST-level dunder
  check — `'{0.__class__}'.format(())` would have reached the tuple
  class, and the same chain extends to the classic
  `__bases__[0].__subclasses__()` sandbox-escape vector. f-strings
  are unaffected: their substitutions go through normal AST
  validation, so `f'{x.__class__}'` is still blocked by the dunder
  check and the legitimate `f'value is {x}'` shape still works.
  The blocked-call error message recommends f-strings as the
  replacement.

## v1.4.0 - 2026-05-26

### Added
- `dataknobs_common.structured_config` module — typed configuration
  meta-abstraction. `StructuredConfig` is a frozen-dataclass base with
  auto-derived `from_dict()` / `to_dict()` via `dataclasses.fields()`
  introspection; override `_normalize_dict(cls, raw)` for pre-projection
  dict-shape normalization (e.g., `normalize_postgres_connection_config`
  routing). Per-class invariants live in `__post_init__`; per-class
  normalization in `_normalize_dict` — never override `from_dict`
  directly. `from_dict` recurses into fields whose declared type is (or
  contains) a `StructuredConfig` subclass — `SubCfg`, `SubCfg | None`,
  `list[SubCfg]`, `tuple[SubCfg, ...]`, `set[SubCfg]`,
  `frozenset[SubCfg]`, `dict[K, SubCfg]`, and `dict[K, list[SubCfg]]` are
  all rebuilt from their raw dict shape (recursion is bounded by the
  static field-type graph; values already typed pass through;
  polymorphic selection — a discriminator key, or a union of several
  concrete sub-configs — stays in the subsystem registry / object-graph
  layer and passes through uncoerced). `from_dict` likewise rebuilds
  `Enum`-typed fields from a raw member value (`{"mode": "fast"}` →
  `Mode.FAST`) through the same container / `| None` shapes, so configs
  load cleanly from YAML/JSON where enums arrive as strings; an
  unrecognised value passes through for the constructor to reject.
  `to_dict()` is the in-process form (enum members verbatim, callables by
  identity), and the companion `to_json_dict()` renders enum members as
  their `.value` at every depth so
  `from_dict(json.loads(json.dumps(cfg.to_json_dict())))` round-trips for
  configs whose other fields are JSON-native.
  `StructuredConfigConsumer[ConfigT]` is the generic
  mixin for classes constructed from a `StructuredConfig` subclass: it
  provides `__init__(config: ConfigT | Mapping | None, **kwargs)`
  typed/dict/loose dispatch (mixing typed `config=` with loose kwargs
  raises `TypeError`), a typed `self.config` property, a
  `cls.from_config(config)` one-liner, a `cls.from_config_async(config)`
  async entry point (sync assemble + `await _ainit()`), and `_setup()`
  (sync) / `_ainit()` (async) subclass hooks. `from_config_async` is the
  canonical async-construction entry — the only path that runs `_ainit`;
  the base `from_config` is synchronous and never runs it. An object
  whose canonical construction is async but that must keep a public
  `await X.from_config(...)` API may override `from_config` with a
  one-line async delegator (`return await cls.from_config_async(config,
  **components)`) — the blessed counterpart to the back-compat `__init__`
  shortcut, routing through `_coerce_config` / `_setup` / `_ainit` so the
  override is lifecycle-faithful rather than returning a half-built
  instance. `__init__` calls
  `super().__init__()` so the mixin composes into a cooperative
  multiple-inheritance hierarchy — list it first among the bases so its
  `__init__` is the construction entry point. The four event-bus
  backends are the first adopters; downstream packages (data, vector
  stores, bots) follow over subsequent releases.
- `dataknobs_common.serialization.jsonify(value)` — public recursive
  utility that replaces every `Enum` member with its `.value`, descending
  through `dict` / `list` / `tuple` containers and into each member's
  `.value` (so an enum-of-enum normalises fully), and passing all other
  values (callables, `type` objects, `set`s) through untouched. Lossless
  and narrow — the counterpart to the lossy, dropping `sanitize_for_json`.
  It is the engine behind `StructuredConfig.to_json_dict()` and is
  re-exported at the top level as `dataknobs_common.jsonify`.
- `StructuredConfig` automatically redacts credential fields from
  `repr`. A subclass declares its credential field names in a
  `_SENSITIVE_FIELDS: ClassVar[frozenset[str]]` and the base masks each
  non-empty value as `'***'` in `repr(config)` — no per-subclass
  `__repr__` needed. The redacting repr is installed by
  `__init_subclass__` (before the subclass's `@dataclass` decorator
  runs, so a dataclass-generated plaintext repr is never produced and an
  intermediate base's generated repr cannot shadow it), making the
  redaction inheritance-safe at every level of the config hierarchy.
  `None` and `""` render verbatim (an unset credential is not a secret);
  `to_dict()` is never redacted, so round-trip is preserved. This keeps
  secrets out of logs that interpolate `repr(config)`, tracebacks, and
  pytest failure output. In `dataknobs-common`, `RedisEventBusConfig`
  (`password`), `PostgresEventBusConfig` (`connection_string`),
  `SqsEventBusConfig` (`aws_access_key_id` / `aws_secret_access_key`,
  migrated off the prior `repr=False` field flag), and `PostgresLockConfig`
  (`connection_string`) adopt it. Redaction also descends into raw
  `Mapping`/`list` field values, masking any interior key in a module
  default set (`api_key`, `password`, `connection_string`, the compound
  `*_secret*` / `*_token` credential names, and the AWS keys) unioned with
  the class's `_SENSITIVE_FIELDS`. This closes the gap where a credential
  nested inside an intentionally-untyped polymorphic section (a
  `vector_store` / `embedding` / `llm` dict held raw because its schema is
  owned by another package and dispatched by a discriminator key) printed in
  cleartext. The default set holds only unambiguous credential names — the
  bare generics `token` and `secret` are excluded, since a false positive
  there would mask a benign key inside a third-party opaque section the
  consumer can neither rename nor remove. Interior matching is exact and
  case-insensitive (a benign `access_token_expiry` is untouched), truthy-only,
  and depth-bounded — and likewise display-only, so `to_dict()` and
  round-trip are unchanged. `register_sensitive_interior_key(*names)` extends
  the interior set process-globally (add-only — redaction is fail-closed, so
  configuration can never switch masking off) for consumers whose custom
  opaque sections use a non-default credential key. The interior-descent depth
  bound defaults to 6 (ample for the ~2-3-level real shapes); a subclass with
  an unusually deep raw section may raise it via the `_MAX_REDACT_DEPTH`
  ClassVar but never lower it below the floor — a below-floor or non-int value
  raises `ValueError` at class definition (lowering would shrink the masked
  region).
- `dataknobs_common.testing.assert_structured_config_consumer(cls)` —
  unified parity guard combining structural checks: `CONFIG_CLS`
  declared, is a `StructuredConfig` subclass, dataclass field set
  matches the consumer ctor surface, the mixin precedes other bases in
  the MRO (so its `__init__` is the entry point), entry-point symmetry
  (an overridden `from_config_async` routes through `_coerce_config`; an
  overridden `from_config` routes through `_coerce_config` when sync, or
  delegates to `from_config_async` when async — pinning the blessed
  async-canonical delegator), and (optional)
  the registry factory delegates to `from_config`. Bundles the
  `assert_dataclass_config_matches_ctor` /
  `assert_factory_kwargs_match_ctor` checks for adopters of the
  structured-config primitives.
- `dataknobs_common.testing.assert_config_attribute_access_matches_dataclass(consumer_cls, config_cls)`
  — AST drift guard for the *body-access* direction: asserts every
  `self.config.<attr>` a consumer reads (walked across its full MRO, so
  inherited base-class reads are covered) is a field or attribute of its
  typed config dataclass. Complements
  `assert_dataclass_config_matches_ctor` (which guards the ctor-surface
  direction). Config *methods* (`clone`, `generation_params`) are valid
  reads; the walk is scoped to `self.<config_attr>.<attr>` (configurable
  via `config_attr=`) so reads off other types — a dataknobs `Config`, a
  dict — are not false-flagged. Does not instantiate the consumer, so
  optional-dependency consumers (provider SDKs, asyncpg) are audited
  without those installed.
- `dataknobs_common.testing.assert_structured_config_roundtrip(cfg)`
  — property assertion that `type(cfg).from_dict(cfg.to_dict()) == cfg`.
  Holds for flat configs and for nested configs alike: `to_dict`
  recurses via `asdict` and `from_dict` recurses back into the matching
  field types, so the two are symmetric for every statically-typed
  nesting shape (no `_normalize_dict` override required for nesting).
  Eliminates per-consumer round-trip boilerplate.
- Collaborator injection on `StructuredConfigConsumer`. The
  construction contract distinguishes *config* (scalar knobs, nested
  sub-configs — flows through `CONFIG_CLS` onto `self.config`) from
  *injected collaborators* (objects the orchestrating parent supplies:
  a shared knowledge base, a bot's main LLM, a pre-built store). The
  classmethod entry points accept collaborators through a keyword
  channel distinct from config — `from_config(config, **components)`
  and `from_config_async(config, **components)` — landing them on the
  read-only `self.components` mapping (never folded into
  `self.config`); the async `_ainit` hook receives them as keyword
  arguments through signature-aware delivery — a hook gets the
  collaborators it declares (or all, when it declares `**kwargs`), so a
  no-arg or narrowly-typed override is never crashed by an undeclared
  injected collaborator (which stays reachable on `self.components`). A
  consumer that consumes a collaborator declares it keyword-only with a
  default — `_ainit(self, *, <name>=None)` — so the zero-injection path
  is safe. The collaborator-free call sites (`from_config(config)`,
  `cls(config)`) are unchanged. `from_components(config=None,
  **collaborators)` covers the dual-input shape — assemble directly from
  pre-built collaborators (binding them via the `_adopt_components` hook
  and setting `self._prebuilt`) instead of building from config — so a
  parent that already holds fully-built children can wire an object
  graph without each class hand-rolling a config-vs-collaborators
  constructor. Called without a `config` snapshot against a `CONFIG_CLS`
  that has required fields, it raises a clear `ValueError` naming the
  class and the remedy rather than a cryptic dataclass `TypeError`.
- `PluginRegistry.create_async(key=None, config=None, **kwargs)` — the
  async counterpart to `create()`. Identical key resolution and factory
  lookup (both share an extracted `_resolve_factory` prologue), but the
  factory result is awaited before the `validate_type` guard runs, so
  the guard checks the resolved instance rather than a coroutine.
  Prefers a `from_config_async` classmethod when present, else awaits an
  awaitable `from_config`/factory result; a purely synchronous factory
  works unchanged. Lets registries dispatch asynchronously-constructed
  plugins (eager-connecting backends, LLM-warmed components,
  ingest-on-build knowledge bases) with the type guard intact.
  `assert_structured_config_consumer` gains a collaborator-hook check:
  an overridden `_ainit` / `_adopt_components` that names parameters
  must declare them keyword-only with defaults, so the framework's
  component delivery is safe and the zero-collaborator path cannot
  crash on a required positional.
- `dataknobs_common.events.config` module — structured config
  dataclasses for every event bus backend: `MemoryEventBusConfig`,
  `RedisEventBusConfig`, `PostgresEventBusConfig`,
  `SqsEventBusConfig`. Each is a frozen `@dataclass` with a
  `from_dict(config: dict)` classmethod and is the single source of
  truth for available kwargs on its backend. Mirrors the
  `LLMConfig` / `RateLimiterConfig` / `RetryConfig` /
  `VectorConfig` pattern used elsewhere in dataknobs. Adding a new
  ctor knob is a dataclass-field addition; the registry factory
  consumes the dict wholesale via `from_dict`, eliminating the
  per-factory allowlist edits that previously caused silent drift.
- `<EventBus>.from_config(config: dict | <EventBus>Config)`
  classmethod on every event bus — the recommended programmatic
  construction path alongside the existing kwarg-positional and
  typed-config init shapes. A config of the wrong `StructuredConfig`
  subclass raises a clear `TypeError` (naming the expected and received
  type) rather than an opaque error.
- `<EventBus>.config` property on every event bus — read-only access
  to the underlying typed config dataclass. Pairs with the new
  `SqsEventBus.require_topic_attribute` shortcut property that maps
  to `bus.config.require_topic_attribute` (kwarg name = config-dict
  key = property name = CHANGELOG vocabulary, single token across
  the public API).
- `dataknobs_common.testing.assert_dataclass_config_matches_ctor`,
  `assert_factory_kwargs_match_ctor`,
  `assert_ctor_reads_documented_keys` — structural drift-guard
  helpers, one per factory-pattern shape used in dataknobs
  registries. Per-registry parity tests in `dataknobs-common`,
  `dataknobs-data`, and `dataknobs-llm` import these to assert that
  every registered factory's ctor surface is reachable from the
  documented config dict.

### Fixed
- `sanitize_for_json` / `validate_json_safe` now treat `Enum` members as
  representable, normalising them to `.value` (recursively) instead of the
  prior inconsistent handling: a plain `Enum` was silently **dropped**
  (data loss — e.g. an enum stored in persisted wizard `state.data`), while
  `IntEnum` / `StrEnum` slipped through the primitive gate and leaked the
  enum *instance* into the supposedly JSON-safe output. Both kinds now
  collapse to a plain `str` / `int`; an enum whose `.value` is itself
  non-representable is still dropped / flagged, since the value is recursed.
- `create_event_bus({"backend": "sqs", "require_topic_attribute":
  False, ...})` now forwards the flag to `SqsEventBus`. Previously
  the registry factory's explicit-allowlist enumeration dropped the
  parameter, so the config-driven entry point silently received the
  constructor default `True` regardless of the config dict. Direct
  `SqsEventBus(...)` callers were unaffected. The new
  structured-config refactor removes this drift mode entirely:
  every kwarg is a dataclass field consumed wholesale by
  `from_dict`, so future ctor additions propagate through the
  registry without per-knob factory edits.

### Changed
- Event-bus backends (`InMemoryEventBus`, `RedisEventBus`,
  `PostgresEventBus`, `SqsEventBus`) now inherit from
  `StructuredConfigConsumer[<Backend>EventBusConfig]` and declare
  `CONFIG_CLS`. Each backend's typed config dataclass inherits from
  `StructuredConfig`, so its `from_dict` is auto-derived rather than
  hand-rolled; `PostgresEventBusConfig` keeps a `_normalize_dict`
  override that routes through
  `normalize_postgres_connection_config`. All documented construction
  shapes are preserved byte-for-byte (typed config, dict via
  `config=`, `from_config`, loose kwargs, `PostgresEventBus`'s legacy
  positional `connection_string` / `channel_prefix`); construction
  dispatch (typed config vs. dict vs. loose kwargs) is handled once by
  the inherited `StructuredConfigConsumer.__init__` rather than per
  backend.
- `dataknobs_common.testing.assert_dataclass_config_matches_ctor` and
  `assert_factory_kwargs_match_ctor` now recognise a ctor that
  declares `**kwargs` (the `StructuredConfigConsumer` pattern) and
  treat all dataclass fields as accepted by construction. This
  prevents false-positive drift reports against consumers that route
  every field through the variadic into `from_dict`. Behaviour
  against hand-rolled ctors that enumerate each kwarg is unchanged.
- The four `_create_*_bus` registry factories collapse to one-line
  `cls.from_config(config)` wrappers. Behaviour is unchanged for
  every existing caller; the public `create_event_bus(...)` entry
  point is unmodified. Every existing call shape continues to work:
  `SqsEventBus(queue_url=...)` (loose kwargs),
  `SqsEventBus(SqsEventBusConfig(...))` (typed),
  `SqsEventBus.from_config({...})` (factory classmethod),
  `create_event_bus({"backend": "sqs", ...})` (registry factory).
  Mixing a typed config with loose kwargs raises `TypeError` —
  ambiguity is surfaced loudly rather than resolved by implicit
  precedence.
- `PostgresAdvisoryLock` exposes a typed `PostgresLockConfig` (new,
  exported from `dataknobs_common.locks`) via a read-only `lock.config`,
  and is a `StructuredConfigConsumer[PostgresLockConfig]`.
  `PostgresLockConfig` is a `StructuredConfig` whose `_normalize_dict`
  routes every input shape through
  `normalize_postgres_connection_config(require=True)` — the same
  canonical DSN resolution `PostgresEventBus` uses (`connection_string`,
  individual host/port/database/user/password keys, `DATABASE_URL`,
  `POSTGRES_*` env-var fallbacks) — and lists `connection_string` in
  `_SENSITIVE_FIELDS`, so the password-bearing DSN is masked as `'***'`
  in `repr(config)`. Construction shapes: legacy positional
  `PostgresAdvisoryLock(connection_string=...)`, loose dict
  `PostgresAdvisoryLock(config={...})`, typed
  `PostgresAdvisoryLock(config=PostgresLockConfig(...))`, and
  `PostgresAdvisoryLock.from_config(...)`; the `_create_postgres_lock`
  registry factory is a one-line `from_config(config)` wrapper. Mixing a
  typed config with the legacy positional raises `TypeError`. Lock
  semantics (session-scoped `pg_advisory_lock`, the `blake2b` key
  mapping, the connect/wait timeout policy) are unchanged.
- `RetryConfig`, `RateLimit`, and `RateLimiterConfig` are now
  `StructuredConfig` subclasses — they gain `from_dict()` / `to_dict()`
  and symmetric round-tripping, including `RateLimiterConfig`'s nested
  `default_rates: list[RateLimit]` and `categories: dict[str,
  list[RateLimit]]` shapes, which rebuild into typed `RateLimit`
  instances through the base recursion (no `_normalize_dict` override).
  `RetryConfig` and `RateLimiterConfig` are now frozen (immutable);
  construct a modified copy with `dataclasses.replace(...)`. `RateLimit`
  was already frozen. Field names, defaults, and all existing
  constructor call shapes are unchanged. `RetryConfig.to_dict()` on a
  config carrying live callable hooks (`retry_on_result`, `on_retry`,
  `on_failure`) or exception types remains an in-process dict (those
  fields round-trip by object identity) and is not JSON-serializable —
  matching the pre-existing semantics of those runtime-hook fields.

## v1.3.14 - 2026-05-20

### Added
- `dataknobs_common.testing.get_localstack_endpoint(host=None, port=None) -> str` —
  public helper that resolves the LocalStack edge endpoint URL
  (e.g. `http://localhost:4566`) suitable for `endpoint_url=` in
  `boto3` / `aioboto3` clients. Pairs with `is_localstack_available`:
  both share a single resolution chain (explicit args →
  `LOCALSTACK_ENDPOINT` → `AWS_ENDPOINT_URL` →
  `LOCALSTACK_HOST` / `LOCALSTACK_PORT` → Docker-aware default).
  Scheme-less env values fall through to defaults rather than emit a
  malformed URL. Consumer copies of the resolution helper can now
  delete in favour of this one.
- `dataknobs_common.testing.ensure_localstack_s3_bucket(bucket, endpoint=None, *, region="us-east-1")` —
  async helper that idempotently creates an S3 bucket on a LocalStack
  edge endpoint (head-then-create; swallows the
  `BucketAlreadyOwnedByYou` / `BucketAlreadyExists` race a concurrent
  setup may produce). Lazy-imports `aioboto3`; install the `sqs`
  extra to pull it in.
- `dataknobs_common.testing.localstack_fixtures` pytest11 plugin
  (auto-discovered by any package depending on `dataknobs-common`):
  `localstack_endpoint` (session-scoped str) and
  `make_localstack_s3_bucket` (factory fixture). Consumers wire a
  per-test bucket with
  `yield from make_localstack_s3_bucket("my-bucket")`. The fixture
  ensures the bucket exists before the test runs and yields a config
  dict (`bucket`, `endpoint_url`, `region`, `access_key_id`,
  `secret_access_key`) shaped for spread into a dataknobs S3 backend
  constructor. No teardown — LocalStack persistence handles the
  bucket's lifetime; tests still wipe object contents themselves.
- `SqsEventBus.require_topic_attribute` constructor parameter
  (single-topic bridge mode). When set to `False`, messages arriving
  on the queue without the configured topic attribute are dispatched
  to the bus's single subscription instead of being released back to
  the queue. Use this mode for queues fed by AWS-native event sources
  that cannot set arbitrary SQS message attributes (EventBridge → SQS
  targets, S3 → SQS bucket notifications, raw SNS → SQS delivery).
  The bus is dedicated to a single topic — `subscribe()` raises
  `ValueError` if a second subscription is attempted in this mode.
  Default remains `True` — existing consumers see no behaviour
  change. Message bodies that are valid JSON but not
  `Event.to_dict()`-shaped are delivered as synthesised
  `Event(type=EventType.CUSTOM, topic=<the subscription's topic>,
  payload=<decoded body>, event_id="sqs:<MessageId>",
  metadata={"sqs_message_id": ..., "sqs_synthesised": True})` events
  with one WARNING log per synthesis. The `event_id` is derived from
  the stable SQS `MessageId` so handlers can key idempotency on it
  across at-least-once redeliveries.

### Changed
- `is_localstack_available()` now delegates endpoint resolution to
  `get_localstack_endpoint` and gains Docker-aware host detection.
  Inside a container (`/.dockerenv` or `DOCKER_CONTAINER` set), with
  no `LOCALSTACK_*` / `AWS_ENDPOINT_URL` env var configured, the
  probe targets `localstack:4566` instead of `localhost:4566`.
  Matches the existing precedent in `postgres_connection_params` /
  `elasticsearch_connection_params`. All other env-driven paths are
  unchanged.

## v1.3.13 - 2026-05-18

### Added
- `dataknobs_common.testing.postgres_fixtures` gains two pytest11
  fixtures (auto-discovered by any package depending on
  `dataknobs-common`; dev/test only — no runtime/consumer
  propagation): `make_pgvector_test_table` — a factory mirroring
  `make_postgres_test_db` that yields a per-test `PgVectorStore`
  config dict and **drops the table before yielding** (the pre-drop
  defeats the `CREATE TABLE IF NOT EXISTS` dimension shadow a killed
  prior session can leave behind) as well as on teardown; and
  `_sweep_orphan_test_tables` — a session-scoped autouse sweep of
  leaked `public.test_*` tables that is fail-closed and opt-in (no-op
  unless `DK_SWEEP_ORPHAN_TEST_TABLES=true`, refuses unless the
  connected DB name is on a test-DB allowlist, drops per-table in
  autocommit so a large leaked backlog cannot exhaust
  `max_locks_per_transaction`).
- `dataknobs_common.testing.requires_real_postgres` — a pytest skip
  mark for behavioural tests that need a live Postgres: skips unless
  the server is reachable, `TEST_POSTGRES=true`, and `asyncpg` is
  installed. A single shared gate for opt-in real-Postgres tests
  across packages (no per-file re-derivation).
- `pytest-randomly` is now a dev/test dependency (root
  `[dependency-groups] dev`; no runtime/consumer propagation). Test
  order is randomized each run and the seed is printed in the pytest
  header; `bin/test.sh` notes the replay/disable flags
  (`--randomly-seed=last`, `--randomly-seed=<N>`, `-p no:randomly`) and
  its `--help` documents them. Reproducible-order is the general
  lever for order-dependent flakes.
- `dataknobs_common.events.event_bus_backends` — a registry-extensible
  plugin point for `create_event_bus()`. Out-of-tree consumers register
  a custom `EventBus` backend
  (`event_bus_backends.register("name", factory)`, where a factory is
  `Callable[[dict], EventBus]`) and select it via
  `create_event_bus({"backend": "name", ...})` without forking
  DataKnobs. Exported from `dataknobs_common.events` along with the
  `EventBusFactory` type alias. The built-in `memory`/`postgres`/`redis`
  backends and the `create_event_bus()` signature are unchanged.
- `dataknobs_common.events.SqsEventBus` — an AWS SQS-backed `EventBus`
  (the built-in `"sqs"` backend). Single queue with the topic carried
  in a configurable message attribute (default `"topic"`); subscribers
  long-poll and filter by exact match. At-least-once delivery —
  handlers must be idempotent; a handler that raises is not acked and
  the message is redelivered after the queue's visibility timeout.
  FIFO queues (`queue_url` ending `.fifo`) get per-topic
  `MessageGroupId` ordering. Wildcard `pattern` subscriptions are
  unsupported and raise `NotImplementedError`. Selectable via
  `create_event_bus({"backend": "sqs", "queue_url": ...})`. Requires
  the optional `aioboto3` dependency: `pip install
  'dataknobs-common[sqs]'`; it is lazy-imported, so the base install
  stays dependency-free and importing `dataknobs_common.events` never
  pulls `aioboto3` (the top-level `SqsEventBus` symbol is a PEP 562
  lazy export). Added the `requires_localstack` pytest marker and
  `is_localstack_available()` probe (`dataknobs_common.testing`) for
  gating the real-LocalStack behavioural tests.
- `postgres` and `redis` optional-dependency extras
  (`pip install 'dataknobs-common[postgres]'` /
  `'dataknobs-common[redis]'`) pulling `asyncpg` / `redis`. `[postgres]`
  serves `PostgresEventBus`; `[redis]` serves both `RedisEventBus` and
  the pyrate Redis-bucket rate limiter. These complete the optional
  EventBus-backend install matrix alongside `[sqs]`; the base install
  remains `dependencies = []` (all three drivers stay lazy-imported).
  The backends' `ImportError` guidance now points at the extra.
- `dataknobs_common.locks` — distributed lock abstraction; the third
  member of the concurrency-primitive set alongside `RateLimiter` and
  `EventBus`. A `@runtime_checkable` `DistributedLock` protocol
  (`acquire`/`release`/`hold`/`close`; `acquire` returns `bool` and
  does not raise on timeout — lock contention is routine, not
  exceptional), an `InProcessLock` default (single-process, zero
  dependency, reference-count evicted key map so it cannot leak; also
  the testing construct — use instead of mocking a lock), and a
  registry-extensible `create_lock()` factory backed by the
  `lock_backends` registry. Out-of-tree consumers register a custom
  cross-replica backend (`lock_backends.register("name", factory)`,
  factory `Callable[[dict], DistributedLock]`) and select it via
  `create_lock({"backend": "name", ...})` without forking DataKnobs —
  the exact structural mirror of `event_bus_backends`. Exported from
  `dataknobs_common.locks` and re-exported at the top-level
  `dataknobs_common` namespace along with the `LockFactory` type alias.
  Two backends are built in: `memory` (`InProcessLock`) and `postgres`
  (`PostgresAdvisoryLock` — session-scoped `pg_advisory_lock` on a
  dedicated connection per held key, cross-replica mutual exclusion for
  every process on the same database). The Postgres backend resolves
  its DSN through the shared `normalize_postgres_connection_config`
  (same path as `PostgresEventBus`: `connection_string`, individual
  keys, `DATABASE_URL`, `POSTGRES_*` env), maps the opaque key to a
  signed 64-bit id via `blake2b` (upgrade-stable, unlike Postgres
  `hashtext`), is liveness-safe (a crashed holder's session death frees
  the lock) and explicitly not a fencing token. `asyncpg` is the
  existing optional `postgres` extra, lazily imported, so
  importing `dataknobs_common.locks` stays dependency-free and the base
  install remains `dependencies = []`. An unknown backend raises
  `ValueError` listing the registered backends (including
  consumer-registered ones).

### Changed
- `compute_backoff_delay()` is now a public pure function in
  `dataknobs_common.retry` (also re-exported from the top-level
  `dataknobs_common` namespace). It encapsulates the back-off delay math
  for every `BackoffStrategy` (FIXED/LINEAR/EXPONENTIAL/JITTER/
  DECORRELATED) including the `max_delay` cap. `RetryExecutor` is
  unchanged for callers — it now delegates its internal delay
  computation to this function so the math has a single home shared with
  the internal event-bus supervised-loop helper.
- The `SqsEventBus` and `RedisEventBus` listener loops now back off with
  exponential delay **plus jitter** and **escalate** under sustained
  failure (capped), instead of a flat 1-second retry. A broker/region
  blip no longer makes every listener (across replicas) wake on the same
  1-second boundary and re-hammer a degraded backend in lockstep; a
  recovered listener resets to the base delay. Both backends now share a
  single internal supervised-loop helper, so back-off behaviour is
  consistent and has one home.
- `create_event_bus()` now resolves backends through
  `event_bus_backends` instead of a sealed `if/elif` chain. Behaviour is
  identical for the three built-in backends; the unknown-backend
  `ValueError` now lists all registered backends (including
  consumer-registered ones) instead of a hard-coded
  `memory, postgres, redis`.

### Fixed
- `PostgresEventBus` now reconnects a dropped dedicated LISTEN
  connection. Previously, if that connection failed the notification
  callback simply stopped firing and the bus **silently stopped
  delivering events** with no error surfaced to subscribers. A
  supervised watchdog now probes the LISTEN connection's liveness and,
  on a drop, re-opens it and re-registers every active channel, so
  delivery resumes.
- `RedisEventBus` now re-establishes its pub/sub connection on
  connection loss instead of retrying a dead one forever. Each listener
  iteration owns rebuilding the pub/sub and re-subscribing every active
  channel and pattern before reading, so delivery resumes after a
  dropped connection.
- `SqsEventBus` no longer starves a topic's consumer on a shared
  single queue. A subscriber that receives a message for a *different*
  topic now returns it to the queue immediately (visibility reset to 0)
  instead of leaving it hidden for the full visibility timeout.
  Previously, with multiple topic subscribers on one queue, a
  subscriber could repeatedly receive-and-park another topic's message,
  delaying or starving the subscriber that actually handles it; the
  release is best-effort and never disrupts the poll loop.

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
