# Changelog

All notable changes to the dataknobs-common package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
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
  typed-config init shapes.
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
