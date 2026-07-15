# Testing Utilities

Test utilities for dataknobs packages including service availability checks, pytest markers, and configuration factories.

## Table of Contents

- [Service Availability Checks](#service-availability-checks)
- [Pytest Markers](#pytest-markers)
- [Configuration Factories](#configuration-factories)
- [File Helpers](#file-helpers)
- [Factory Parity Helpers](#factory-parity-helpers)
- [Async Blocking Detection](#async-blocking-detection)
- [Shared Integration Fixtures: Postgres and Elasticsearch](#shared-integration-fixtures-postgres-and-elasticsearch)
- [Usage Examples](#usage-examples)

---

## Service Availability Checks

Functions to check if external services and packages are available.

### Ollama

```python
from dataknobs_common import is_ollama_available, is_ollama_model_available

# Check if Ollama service is running
if is_ollama_available():
    print("Ollama is available")

# Check if a specific model is available
if is_ollama_model_available("nomic-embed-text"):
    print("Embedding model ready")

if is_ollama_model_available("llama3.1:8b"):
    print("LLM model ready")
```

### Packages

```python
from dataknobs_common import (
    is_faiss_available,
    is_chromadb_available,
    is_package_available,
)

# Check vector store packages
if is_faiss_available():
    print("FAISS is installed")

if is_chromadb_available():
    print("ChromaDB is installed")

# Check any package
if is_package_available("torch"):
    print("PyTorch is installed")
```

### Services

```python
from dataknobs_common import is_redis_available

# Check Redis connection
if is_redis_available(host="localhost", port=6379):
    print("Redis is available")
```

### LocalStack

`get_localstack_endpoint()` resolves the LocalStack edge endpoint as
a fully-qualified URL — suitable for `endpoint_url=` in `boto3` /
`aioboto3` clients — and shares the resolution chain with
`is_localstack_available()` so the probe and the URL form cannot
drift.

```python
from dataknobs_common.testing import (
    get_localstack_endpoint,
    is_localstack_available,
    requires_localstack,
)

# URL form for SDK clients
endpoint = get_localstack_endpoint()  # "http://localhost:4566"

# Skip a test when LocalStack is not running
@requires_localstack
async def test_against_localstack():
    import aioboto3
    session = aioboto3.Session(region_name="us-east-1")
    async with session.client("sqs", endpoint_url=endpoint) as sqs:
        ...
```

Resolution order — highest priority first:

1. Explicit `host` / `port` arguments (each independent).
2. `LOCALSTACK_ENDPOINT` (full URL; scheme optional).
3. `AWS_ENDPOINT_URL` (full URL; same scheme handling).
4. `LOCALSTACK_HOST` + `LOCALSTACK_PORT` env vars.
5. Default: `http://localhost:4566`, or `http://localstack:4566`
   when running inside a Docker container (detected via
   `/.dockerenv` or `DOCKER_CONTAINER`).

#### S3 Bucket Provisioning Fixture

For LocalStack S3 tests, the `make_localstack_s3_bucket` pytest11
fixture idempotently ensures a named bucket exists before the test
runs and yields a config dict shaped for the dataknobs S3 backends.
Auto-discovered from `dataknobs-common` — no `conftest.py` import
required.

```python
import pytest
from dataknobs_common.testing import requires_localstack
from dataknobs_data import AsyncDatabaseFactory


@pytest.fixture
def s3_test_bucket(make_localstack_s3_bucket):
    """Ensure ``my-test-bucket`` exists on LocalStack for this test."""
    yield from make_localstack_s3_bucket("my-test-bucket")


@requires_localstack
@pytest.mark.asyncio
async def test_s3_roundtrip(s3_test_bucket):
    db = AsyncDatabaseFactory().create(
        backend="s3",
        bucket=s3_test_bucket["bucket"],
        endpoint_url=s3_test_bucket["endpoint_url"],
    )
    await db.connect()
    try:
        ...  # Use the database
    finally:
        await db.clear()  # Wipe object contents; bucket persists
```

The fixture does **not** delete the bucket on teardown — LocalStack
persists it across the session and tests are expected to wipe their
own object contents (typically via `db.clear()`). The factory pattern
keeps the bucket name caller-controlled and works for both sync and
async test bodies (the `asyncio.run` wrapping happens in the fixture
setup phase, outside any per-test event loop).

For a one-off, non-fixture flow, call the underlying async helper
directly:

```python
from dataknobs_common.testing import ensure_localstack_s3_bucket

await ensure_localstack_s3_bucket("my-bucket")  # idempotent
```

The helper lazy-imports `aioboto3`; install the `sqs` extra to pull
it in.

---

## Pytest Markers

Pre-configured pytest markers to skip tests when dependencies are unavailable.

### Basic Markers

```python
import pytest
from dataknobs_common import (
    requires_ollama,
    requires_faiss,
    requires_chromadb,
    requires_redis,
)

@requires_ollama
def test_with_ollama():
    """Skipped if Ollama is not available."""
    # Your test code
    pass

@requires_faiss
def test_with_faiss():
    """Skipped if FAISS is not installed."""
    pass

@requires_chromadb
def test_with_chromadb():
    """Skipped if ChromaDB is not installed."""
    pass

@requires_redis
def test_with_redis():
    """Skipped if Redis is not available."""
    pass
```

### Dynamic Markers

```python
from dataknobs_common import requires_package, requires_ollama_model

@requires_package("torch")
def test_with_pytorch():
    """Skipped if PyTorch is not installed."""
    import torch
    # Test code
    pass

@requires_ollama_model("llama3.1:8b")
def test_with_llama():
    """Skipped if specific model is not available."""
    # Test code
    pass
```

### Manual Skipif

You can also use the availability functions directly:

```python
import pytest
from dataknobs_common import is_ollama_available

@pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama not available"
)
def test_ollama_integration():
    # Your test
    pass
```

---

## Configuration Factories

Pre-built configuration dictionaries for testing.

### Bot Configuration

```python
from dataknobs_common import get_test_bot_config

# Basic test config with echo LLM
config = get_test_bot_config()
# Returns:
# {
#     "llm": {"provider": "echo", "model": "test", "temperature": 0.7},
#     "conversation_storage": {"backend": "memory"}
# }

# With real LLM (for integration tests)
config = get_test_bot_config(use_echo_llm=False)

# With memory
config = get_test_bot_config(include_memory=True)

# With system prompt
config = get_test_bot_config(
    system_prompt="You are a test assistant."
)

# Full customization
config = get_test_bot_config(
    use_echo_llm=True,
    use_in_memory_storage=True,
    include_memory=True,
    system_prompt="Custom prompt"
)
```

### RAG Configuration

```python
from dataknobs_common import get_test_rag_config

# Basic RAG config
rag_config = get_test_rag_config()

# With FAISS backend (for persistence)
rag_config = get_test_rag_config(use_in_memory_store=False)

# Custom embedding
rag_config = get_test_rag_config(
    embedding_provider="openai",
    embedding_model="text-embedding-3-small"
)
```

---

## File Helpers

Create test files for ingestion testing.

### Markdown Files

```python
from dataknobs_common import create_test_markdown_files

def test_document_ingestion(tmp_path):
    # Create test markdown files
    files = create_test_markdown_files(tmp_path)

    # files contains paths to two markdown documents
    assert len(files) == 2

    # Use for testing
    for file_path in files:
        # Process documents
        pass
```

### JSON Files

```python
from dataknobs_common import create_test_json_files

def test_json_processing(tmp_path):
    # Create test JSON files
    files = create_test_json_files(tmp_path)

    # files contains paths to two JSON documents
    assert len(files) == 2
```

---

## Factory Parity Helpers

AST-based drift guards that pin the parity between a registry's factory
function and its target class's constructor. Use them in a per-registry
parity test to catch the original failure mode that motivated the
helpers: a factory that enumerates a fixed allowlist of kwargs and
silently drops the next ctor knob added to the target class.

The five AST-based helpers (all below except
`assert_structured_config_roundtrip`) are import-only — they read source
via `inspect` and parse it with `ast`, so backends and consumers with
optional runtime dependencies (asyncpg, aioboto3, provider SDKs, ...) can
be parity-tested without those dependencies installed.
`assert_structured_config_roundtrip` is a runtime property assertion that
takes a config instance.

The first four guards (and the `assert_structured_config_consumer` bundle)
pin the **ctor-surface** direction — every config field has a matching
constructor parameter. `assert_config_attribute_access_matches_dataclass`
pins the orthogonal **body-access** direction — every attribute the
consumer *reads* off its typed config actually exists on the config type.

### `assert_dataclass_config_matches_ctor`

Use when the ctor consumes a typed `@dataclass` config. Asserts every
dataclass field is a ctor `__init__` parameter, and every ctor
parameter (minus `self`/`config`/`*args`/`**kwargs`) is a dataclass
field.

```python
from dataknobs_common.testing import assert_dataclass_config_matches_ctor
from dataknobs_common.events import RedisEventBusConfig
from dataknobs_common.events.redis import RedisEventBus

def test_redis_dataclass_matches_ctor() -> None:
    assert_dataclass_config_matches_ctor(RedisEventBusConfig, RedisEventBus)
```

Pass `ignore_params={...}` for ctor params that are intentionally not
config fields (internal-only kwargs).

### `assert_factory_kwargs_match_ctor`

Use when a registry entry is a free-function factory that names its
kwargs. AST-walks the factory body for `Target(...)` or
`Target.from_config(...)` call sites and asserts:

1. Every kwarg the factory passes is a valid parameter of `Target.__init__`
2. Every `Target.__init__` parameter is forwarded by the factory (or
   explicitly ignored)

The check is symmetric — it catches both directions of drift (factory
passing an unknown kwarg, factory missing a known kwarg). When the
factory delegates to `from_config`, the "missing kwargs" check is
satisfied automatically (the dataclass is the kwarg-coverage source of
truth) — pair it with `assert_dataclass_config_matches_ctor`.

```python
from dataknobs_common.testing import assert_factory_kwargs_match_ctor
from dataknobs_common.events.registry import _create_redis_bus
from dataknobs_common.events.redis import RedisEventBus

def test_redis_factory_kwargs_match_ctor() -> None:
    assert_factory_kwargs_match_ctor(_create_redis_bus, RedisEventBus)
```

Pass `ignore_kwargs={...}` for required positionals the consumer is
expected to supply, or knobs without a sensible config-dict default.

### `assert_ctor_reads_documented_keys`

Use when the ctor takes a config dict and reads keys via
`config.get("X")`, `config["X"]`, or `config.pop("X")` inside its body
(vector stores, postgres lock, data backends post-merge-into-base-init).
Asserts every documented key is read somewhere in the ctor body.

```python
from dataknobs_common.testing import assert_ctor_reads_documented_keys
from dataknobs_data.vector.stores.pgvector import PgVectorStore

def test_pgvector_reads_documented_keys() -> None:
    assert_ctor_reads_documented_keys(
        PgVectorStore,
        documented_keys={"connection_string", "table", "dimensions"},
    )
```

Pass `config_param="..."` to name the dict parameter when it isn't the
default `"config"`.

### `assert_structured_config_consumer`

Use for a class that mixes in `StructuredConfigConsumer`. Bundles the
structural checks into one call:

1. The class declares a `CONFIG_CLS` ClassVar.
2. `CONFIG_CLS` is a `StructuredConfig` subclass.
3. The dataclass field set matches the consumer ctor surface (via
   `assert_dataclass_config_matches_ctor`, ignoring the implicit `self` /
   `config` / `**kwargs`).
4. **MRO ordering** — when the consumer relies on the mixin's `__init__`
   (does not define its own), the resolved `__init__` must be
   `StructuredConfigConsumer.__init__`, proving the mixin precedes any
   other base with a competing `__init__`. A consumer that overrides
   `__init__` (the back-compat positional shortcut) is exempt.
5. **Entry-point symmetry** — an overridden `from_config_async` must
   route through `_coerce_config` (the same guard `from_config` uses); an
   overridden `from_config` must route through `_coerce_config` when sync,
   or delegate to `from_config_async` when async (the blessed
   async-canonical delegator, so `_coerce_config` and the `_ainit`
   lifecycle both run rather than returning a half-built instance).
6. (Optional) when `expected_factory` is passed, the registry factory
   delegates to `from_config` (via `assert_factory_kwargs_match_ctor`).
7. **Collaborator-hook safety** — an overridden `_ainit` /
   `_adopt_components` that names collaborator parameters must declare
   them keyword-only with defaults, so the framework's signature-aware
   delivery is safe and the zero-collaborator construction path cannot
   crash on a required positional.

```python
from dataknobs_common.testing import assert_structured_config_consumer
from dataknobs_common.events.redis import RedisEventBus
from dataknobs_common.events.registry import _create_redis_bus

def test_redis_is_structured_config_consumer() -> None:
    assert_structured_config_consumer(
        RedisEventBus, expected_factory=_create_redis_bus
    )
```

Pass `ignore_params={...}` for back-compat positional shortcuts that
intentionally live outside the dataclass surface.

### `assert_structured_config_roundtrip`

Use to pin the serialization round-trip of a `StructuredConfig`
instance: asserts `type(cfg).from_dict(cfg.to_dict()) == cfg`. Holds for
flat configs and for nested configs alike — `from_dict` recurses into
fields typed as a `StructuredConfig` subclass (or a `list`/`tuple`/`set`/
`frozenset`/`dict` of one), so `to_dict`/`from_dict` are symmetric for
every statically-typed nesting shape with no `_normalize_dict` override
required.

```python
from dataknobs_common.testing import assert_structured_config_roundtrip
from dataknobs_common.events import RedisEventBusConfig

def test_redis_config_roundtrips() -> None:
    assert_structured_config_roundtrip(
        RedisEventBusConfig(host="h", port=6379)
    )
```

### `assert_config_attribute_access_matches_dataclass`

The body-access counterpart to `assert_dataclass_config_matches_ctor`.
Where the ctor-surface guard proves every config *field* has a matching
constructor parameter, this proves every attribute the consumer *reads*
off its typed config actually lives on the config type. The drift it
catches: a consumer reads `self.config.foo` where `foo` is neither a
dataclass field nor any attribute/method of the config type — an
`AttributeError` waiting for the first time that (often provider-specific,
un-CI'd) code path runs.

It AST-walks every class in the consumer's `__mro__` (so inherited
base-class reads are covered, not just the leaf body) for
`self.<config_attr>.<name>` accesses, and asserts each `<name>` is a field
of the config *or* any attribute/method resolvable on it (`dir(config_cls)`
— so config helper methods like `clone`, `generation_params`, `to_dict`
are valid reads and don't false-positive). It does not instantiate the
consumer, so provider classes with optional SDK dependencies are audited
without those installed.

```python
from dataknobs_common.testing import (
    assert_config_attribute_access_matches_dataclass,
)
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers import OllamaProvider

def test_ollama_reads_match_llm_config() -> None:
    assert_config_attribute_access_matches_dataclass(
        OllamaProvider, LLMConfig
    )
```

The walk is scoped to `self.<config_attr>.<name>` deliberately: bare
`config.<name>` reads from method parameters are *not* walked, because
they routinely operate on other types (a dataknobs `Config`, a plain dict)
and would produce unavoidable false positives. Pass `config_attr="..."`
when the typed config lives on a differently-named attribute, and
`ignore_attrs=frozenset({...})` for documented intentional off-config
reads.

### When to Use Which

| Factory pattern | Helper |
|---|---|
| Ctor consumes a typed `@dataclass` config | `assert_dataclass_config_matches_ctor` |
| Factory names kwargs in its body — `cls(k=cfg.get("k"))` or `cls.from_config(cfg)` | `assert_factory_kwargs_match_ctor` |
| Ctor reads `config.get("X")` / `config["X"]` directly | `assert_ctor_reads_documented_keys` |
| Class mixes in `StructuredConfigConsumer` | `assert_structured_config_consumer` |
| Consumer reads `self.config.<attr>` in its body | `assert_config_attribute_access_matches_dataclass` |
| Round-trip a `StructuredConfig` instance — `from_dict(to_dict())` | `assert_structured_config_roundtrip` |

When a registry uses the dataclass + `from_config` pattern, pair the
first two — together they pin both the dataclass↔ctor parity and the
factory↔ctor parity, so drift in either direction fails the test. For a
`StructuredConfigConsumer` adopter, `assert_structured_config_consumer`
bundles them into a single call.

---

## Async Blocking Detection

An `async def` method promises to keep the event loop free while it awaits.
A synchronous, blocking transport invoked from inside it — a sync `boto3`
client, `open()`, `time.sleep`, a blocking socket read — breaks that
promise: the loop stalls for the duration of the call and every other task
on it is starved. The defect is *non-functional* (the method still returns
the right value), so ordinary outcome assertions never catch it.

`assert_no_blocking()` turns that invisible defect into a deterministic,
reproduce-first test failure. It activates a runtime detector
([`blockbuster`](https://pypi.org/project/blockbuster/)) that patches the
common blocking syscalls to raise when they run on a live event loop. Wrap
the awaited operation under test:

```python
from dataknobs_common.testing import assert_no_blocking

async def test_put_does_not_block(backend):
    with assert_no_blocking():
        await backend.put_file("kb", "doc.md", b"...")
```

The block **fails** against a backend that blocks the loop and **passes**
once it uses an async transport (`aioboto3`, `asyncpg`, `aiohttp`) or
offloads the blocking call via `asyncio.to_thread`. Scope the block tightly
around the `await` under test — not synchronous setup (building a temp
directory, constructing fixtures) which may legitimately block.

`assert_no_blocking()` is the *runtime* proof; ruff's `ASYNC` lint family
(`flake8-async`, enabled repo-wide) is the *static* counterpart — it flags
blocking `open()` / `os` / `time.sleep` / sync-HTTP calls inside an
`async def` at lint time. The authoring contract both enforce lives in
`.claude/rules/async-transport.md`.

Detection only fires while an event loop is running, so `assert_no_blocking`
is meaningful inside an `async` test (or any frame with a running loop); in
a synchronous frame it is a no-op.

`blockbuster` is a **dev/test-only** dependency — never imported by shipped
runtime code. The imports are lazy, so importing `dataknobs_common.testing`
never requires it. Consumers guarding their own async backends add
`blockbuster` to their dev dependencies and get the construct for free.

| Helper | Role |
|---|---|
| `assert_no_blocking()` | Context manager; raises the detector's `BlockingError` if a blocking syscall runs on the loop in the block. Raises `RuntimeError` if `blockbuster` is not installed (fails loud, never silently passes). |
| `no_blocking` (fixture) | Wraps the whole test in `assert_no_blocking()`. Auto-discovered via the `dataknobs_common_blocking` pytest11 plugin; skips when `blockbuster` is absent. Prefer the context manager for precise scoping. |
| `requires_blockbuster` | Ready-made `pytest.mark.skipif` marker — decorate a test that calls `assert_no_blocking()` so it skips cleanly when `blockbuster` is absent. Prefer this over hand-rolling the skipif. |
| `is_blockbuster_available()` | `bool` — underlies `requires_blockbuster`; call directly only for a custom skip condition. |
| `blocking_error_type()` | Returns `blockbuster`'s `BlockingError` type (lazily), so a self-test can `pytest.raises(blocking_error_type())` without importing `blockbuster` directly. |

```python
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

@requires_blockbuster
async def test_put_does_not_block(backend):
    with assert_no_blocking():
        await backend.put_file("kb", "doc.md", b"...")
```

---

## Shared Integration Fixtures: Postgres and Elasticsearch

`dataknobs-common` ships pytest11 plugins that expose Postgres and
Elasticsearch infrastructure fixtures to every package depending on it.
Plugin discovery is automatic — no `conftest.py` imports needed. Consumers
wrap a factory fixture with their own per-prefix fixture so each package
controls its table/index naming.

### Available Fixtures

| Fixture | Scope | Provided by |
|---|---|---|
| `postgres_connection_params` | session | `dataknobs_common.testing.postgres_fixtures` |
| `ensure_postgres_ready` | session | `dataknobs_common.testing.postgres_fixtures` |
| `make_postgres_test_db` | function | `dataknobs_common.testing.postgres_fixtures` |
| `elasticsearch_connection_params` | session | `dataknobs_common.testing.elasticsearch_fixtures` |
| `ensure_elasticsearch_ready` | session | `dataknobs_common.testing.elasticsearch_fixtures` |
| `make_elasticsearch_test_index` | function | `dataknobs_common.testing.elasticsearch_fixtures` |

The non-fixture helpers `wait_for_postgres()` and `wait_for_elasticsearch()`
are also importable from `dataknobs_common.testing`, along with
`sweep_stale_test_indices()` (see [Session-Start Index Sweep](#session-start-index-sweep))
and the availability probes / skip markers `is_postgres_available()` /
`requires_postgres`, `is_elasticsearch_available()` / `requires_elasticsearch`.

### Environment Variables

Postgres fixtures read (defaults shown):

- `POSTGRES_HOST` — `postgres` in Docker, `localhost` otherwise
- `POSTGRES_PORT` — `5432`
- `POSTGRES_USER` — `postgres`
- `POSTGRES_PASSWORD` — `postgres`
- `POSTGRES_DB` — `dataknobs_test`
- `DOCKER_CONTAINER` — any truthy value forces the `postgres` host default

Elasticsearch fixtures read:

- `ELASTICSEARCH_HOST` — `elasticsearch` in Docker, `localhost` otherwise
- `ELASTICSEARCH_PORT` — `9200`
- `DOCKER_CONTAINER` — any truthy value forces the `elasticsearch` host default
- `DK_ES_TEST_INDEX_MAX_AGE_SECONDS` — staleness threshold (seconds) for the
  session-start index sweep; default `300`

`get_localstack_endpoint` and `is_localstack_available` read:

- `LOCALSTACK_ENDPOINT` — full URL; overrides host/port
- `AWS_ENDPOINT_URL` — full URL fallback when `LOCALSTACK_ENDPOINT` is unset
- `LOCALSTACK_HOST` — `localstack` in Docker, `localhost` otherwise
- `LOCALSTACK_PORT` — `4566`
- `DOCKER_CONTAINER` — any truthy value forces the `localstack` host default

Docker detection also checks for `/.dockerenv`.

### Factory Fixture Pattern

`make_postgres_test_db` and `make_elasticsearch_test_index` are factory
fixtures: they return a callable that, when invoked with a table/index
prefix, yields a clean per-test config and tears down the resource on
completion. Consumers wrap them with a thin per-prefix fixture:

```python
# packages/<your-pkg>/tests/integration/conftest.py
import pytest


@pytest.fixture
def postgres_test_db(make_postgres_test_db):
    yield from make_postgres_test_db("test_conversations_")


@pytest.fixture
def elasticsearch_test_index(make_elasticsearch_test_index):
    yield from make_elasticsearch_test_index("test_records_")
```

The `yield from` bridge threads exception cleanup through both generator
layers, so teardown runs even when the test body raises.

The yielded Postgres config has the same shape as
`postgres_connection_params` plus `table` and `schema` keys. The yielded
Elasticsearch config has the same shape as `elasticsearch_connection_params`
plus `index` and `refresh=True` keys.

### Usage in Tests

```python
from dataknobs_common.testing import requires_postgres


@requires_postgres
def test_create_record(postgres_test_db):
    # postgres_test_db is the dict yielded by the consumer wrapper
    # Each test gets its own table; the table is dropped on teardown.
    ...
```

### Why a Factory Fixture?

The table/index prefix differs between packages (`test_records_` for
`dataknobs-data`, `test_conversations_` for `dataknobs-llm`, etc.). Two
alternatives were considered and rejected:

- **Parameterize a single shared fixture.** Forces every consumer to
  re-declare `@pytest.fixture(params=[...])` indirect parameterization in
  its own `conftest.py`, defeating the point of the shared plugin.
- **Hardcode a prefix in `dataknobs-common`.** Every package would need a
  rename and the per-package `test_*` namespace convention would be lost.

The factory pattern keeps the prefix consumer-controlled with a one-line
wrapper.

### Cleanup Behavior

Postgres cleanup unconditionally re-opens a connection and runs
`DROP TABLE IF EXISTS … CASCADE` under `safe_sql_ident()`. Cleanup
exceptions propagate so test failures aren't masked by silent teardown
errors.

Elasticsearch cleanup is best-effort: `ConnectionError` and `ValueError`
are logged at `WARNING` and swallowed (the test result is preserved); any
other exception propagates to surface unexpected failures.

### Session-Start Index Sweep

Per-test teardown deletes each test's index — but only when the run reaches
teardown. A run killed mid-test (Ctrl-C, crash, timeout) leaks its
uniquely-suffixed `test_*` index. Because the dev/CI cluster uses a
persistent data volume, those leaks accumulate across runs; each holds a
shard, and a single-node cluster's `cluster.max_shards_per_node` ceiling
(default 1000) eventually rejects all new index creation, reddening the whole
Elasticsearch suite.

To reclaim that residue, `ensure_elasticsearch_ready` calls
`sweep_stale_test_indices()` once at session start (after the readiness
wait). The sweep lists `test_*` indices (a read) and deletes, **by exact
name**, each one whose ES `creation.date` is older than the staleness
threshold:

```python
from dataknobs_common.testing import sweep_stale_test_indices

# Defaults: prefixes=("test_",), threshold from
# DK_ES_TEST_INDEX_MAX_AGE_SECONDS or 300s.
deleted = sweep_stale_test_indices(host, port)
```

Two properties make it safe:

- **Age-gating.** Session fixtures run once per worker, so under
  pytest-xdist a blanket "delete all matching" would race a concurrent
  worker's live index. An in-flight index is seconds old and is never swept;
  accumulated residue is minutes-to-months old and always is. Deletion is by
  exact name because ES rejects a wildcard `DELETE` under
  `action.destructive_requires_name`.
- **Best-effort.** Any request/connection/parse error is logged at `WARNING`
  and the function returns the names deleted so far; it never raises, so a
  cleanup hiccup can't fail an otherwise-green run.

The `docker-compose.override.yml` Elasticsearch service also sets
`cluster.max_shards_per_node=3000` — headroom for a single run that spikes
many concurrent indices before the next sweep, not the fix itself.

---

## Usage Examples

### Complete Test Setup

```python
# conftest.py
import pytest
from dataknobs_common import (
    is_ollama_available,
    get_test_bot_config,
    create_test_markdown_files,
)

@pytest.fixture
def bot_config():
    """Provide test bot configuration."""
    return get_test_bot_config(
        include_memory=True,
        system_prompt="Test assistant"
    )

@pytest.fixture
def test_documents(tmp_path):
    """Create test documents."""
    return create_test_markdown_files(tmp_path)

# Skip entire module if Ollama unavailable
pytestmark = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama required for integration tests"
)
```

### Unit Test Example

```python
# test_bot.py
import pytest
from dataknobs_bots import DynaBot, BotContext
from dataknobs_common import get_test_bot_config, requires_ollama

class TestDynaBot:
    @pytest.mark.asyncio
    async def test_chat_with_echo(self):
        """Test chat with echo provider (no external deps)."""
        config = get_test_bot_config()
        bot = await DynaBot.from_config(config)

        context = BotContext(
            conversation_id="test-1",
            client_id="test"
        )

        response = await bot.chat("Hello", context)
        assert response is not None

    @requires_ollama
    @pytest.mark.asyncio
    async def test_chat_with_ollama(self):
        """Test chat with Ollama (skipped if unavailable)."""
        config = get_test_bot_config(use_echo_llm=False)
        config["llm"]["provider"] = "ollama"
        config["llm"]["model"] = "gemma3:1b"

        bot = await DynaBot.from_config(config)
        # ...
```

### Integration Test Example

```python
# test_rag_integration.py
import pytest
from dataknobs_common import (
    requires_ollama,
    requires_ollama_model,
    get_test_bot_config,
    get_test_rag_config,
    create_test_markdown_files,
)

@requires_ollama
@requires_ollama_model("nomic-embed-text")
class TestRAGIntegration:
    @pytest.fixture
    def rag_bot_config(self, tmp_path):
        """Create RAG bot configuration with test docs."""
        docs = create_test_markdown_files(tmp_path)

        config = get_test_bot_config(use_echo_llm=False)
        config["llm"]["provider"] = "ollama"
        config["llm"]["model"] = "gemma3:1b"
        config["knowledge_base"] = get_test_rag_config()
        config["knowledge_base"]["documents_path"] = str(tmp_path)

        return config

    @pytest.mark.asyncio
    async def test_rag_query(self, rag_bot_config):
        """Test RAG retrieval."""
        # Test implementation
        pass
```

---

## Best Practices

1. **Use Echo LLM for Unit Tests**: Avoid external dependencies in unit tests.

2. **Mark Integration Tests**: Use markers to identify tests requiring services.

3. **Organize by Dependency**: Group tests by their external requirements.

4. **CI/CD Considerations**: Run integration tests only when services available.

5. **Fixture Composition**: Combine configuration factories with pytest fixtures.

---

## See Also

- [pytest documentation](https://docs.pytest.org/)
- [pytest11 plugin entry-points](https://docs.pytest.org/en/stable/how-to/writing_plugins.html#making-your-plugin-installable-by-others)
