# Changelog

All notable changes to the dataknobs-common package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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
