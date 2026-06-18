# Dynamic Registration

Dynamic registration enables runtime management of bot configurations and knowledge resources. This system provides:

- **Registry backends** for storing bot configurations
- **Caching with TTL** for efficient instance management
- **Hot reload** for updating bots without restarts
- **Knowledge storage** for managing raw knowledge files
- **Ingestion pipelines** for loading knowledge into RAG

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  BotManager  │    │ HotReload    │    │  Ingestion   │     │
│   │              │    │   Manager    │    │   Manager    │     │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
└──────────┼───────────────────┼───────────────────┼──────────────┘
           │                   │                   │
┌──────────▼───────────────────▼───────────────────▼──────────────┐
│                    Caching & Invalidation                        │
│   ┌──────────────────────────────────────────────────────┐     │
│   │              CachingRegistryManager                   │     │
│   │   - TTL-based expiration                             │     │
│   │   - LRU eviction                                     │     │
│   │   - Event-driven invalidation                        │     │
│   └──────────────────────────┬───────────────────────────┘     │
└──────────────────────────────┼──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      Registry Backends                           │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│   │  Memory    │  │  DataKnobs │  │    HTTP    │               │
│   │  Backend   │  │  Adapter   │  │  Backend   │               │
│   └────────────┘  └─────┬──────┘  └────────────┘               │
│                         │                                       │
│   ┌─────────────────────▼─────────────────────────────┐        │
│   │              dataknobs_data backends               │        │
│   │   PostgreSQL │ SQLite │ S3 │ File                 │        │
│   └───────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                        Event Bus                                 │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│   │  In-Memory │  │ PostgreSQL │  │   Redis    │               │
│   │            │  │ LISTEN/    │  │  Pub/Sub   │               │
│   │            │  │ NOTIFY     │  │            │               │
│   └────────────┘  └────────────┘  └────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

## Registry Backends

### Overview

Registry backends store bot configurations. Choose based on your deployment:

| Backend | Use Case | Features |
|---------|----------|----------|
| `InMemoryBackend` | Testing, development | No persistence |
| `DataKnobsRegistryAdapter` | Production | PostgreSQL, SQLite, S3, File |
| `HTTPRegistryBackend` | External config service | REST API integration |

### InMemoryBackend

Simple dict-based storage for testing:

```python
from dataknobs_bots.registry import InMemoryBackend, Registration

backend = InMemoryBackend()
await backend.initialize()

# Register a bot
registration = await backend.register("my-bot", {
    "llm": {"provider": "anthropic", "model": "claude-3-sonnet"},
    "memory": {"type": "buffer", "max_messages": 10}
})
print(f"Registered at: {registration.created_at}")

# Get config (registers an access — bumps last_accessed_at)
config = await backend.get_config("my-bot")

# Peek config (no-touch read — for inspection / audit / bookkeeping
# paths that should not register as user activity)
config = await backend.peek_config("my-bot")

# List all active
active = await backend.list_active()

# Deactivate (soft delete)
await backend.deactivate("my-bot")

# Unregister (hard delete)
await backend.unregister("my-bot")
```

### DataKnobsRegistryAdapter

Wraps any `dataknobs_data` database backend:

```python
from dataknobs_bots.registry import DataKnobsRegistryAdapter

# Create with explicit backend type
adapter = DataKnobsRegistryAdapter(
    backend_type="postgres",
    backend_config={
        "host": "localhost",
        "database": "myapp",
        "table": "bot_configs"
    }
)
await adapter.initialize()

# Or with existing database instance
from dataknobs_data.backends.memory import AsyncMemoryDatabase
db = AsyncMemoryDatabase()
await db.connect()
adapter = DataKnobsRegistryAdapter(database=db)
```

Supported backend types:

| Type | Backend Class | Use Case |
|------|---------------|----------|
| `memory` | `AsyncMemoryDatabase` | Testing |
| `postgres` | `AsyncPostgresDatabase` | Production |
| `sqlite` | `AsyncSQLiteDatabase` | Local persistence |
| `file` | `SyncFileDatabase` | Simple file storage |
| `s3` | `SyncS3Database` | Cloud storage |

### HTTPRegistryBackend

Fetch configurations from REST APIs:

```python
from dataknobs_bots.registry import HTTPRegistryBackend

backend = HTTPRegistryBackend(
    base_url="https://config-service.internal/api/v1",
    auth_header="Bearer token123",
    timeout=30,
)
await backend.initialize()

# Get config
config = await backend.get_config("my-bot")

# Peek config: identical wire call to get_config — the HTTP backend
# has no client-side activity state to skip, so non-mutation is
# satisfied trivially at the client surface. Server-side activity
# tracking remains the server's contract; this client deliberately
# does not impose a wire-protocol distinction (e.g. ?peek=true).
config = await backend.peek_config("my-bot")

# List all
configs = await backend.list_all()
```

Expected API contract:

```
GET  /bots/{bot_id}     → {"bot_id": "...", "config": {...}, "status": "active"}
GET  /bots              → [{"bot_id": "...", ...}, ...]
POST /bots              → Create registration
PUT  /bots/{bot_id}     → Update registration
DELETE /bots/{bot_id}   → Delete registration
```

#### Filter, sort, and pagination over the wire

`HTTPRegistryBackend.list_all` pushes the same filter / pagination
surface as the other backends to the server as optional query
parameters on `GET /configs` (all omitted when `None`):

| Parameter | Wire form | Notes |
|-----------|-----------|-------|
| `filter_metadata` | `?filter_metadata=<JSON object>` | Serialized with `sort_keys=True` so the query string is deterministic (server cache lines, request logs). |
| `status` | `?status=<value>` | Equality filter on the `status` column. |
| `sort` | `?sort=<field>[:asc|desc]` (repeatable) | Wire order of repeated values is preserved as the tie-break order. |
| `limit` | `?limit=<int>` | Server-side row limit. |
| `offset` | `?offset=<int>` | Server-side offset. |

The wire schema is **additive optional**: servers that recognize a
parameter SHOULD honor it; servers that don't ignore it and return
the broader, unfiltered list.  For correctness on legacy servers, the
client defensively re-applies idempotent filters (`filter_metadata`,
`status`, `sort`) after parsing the response.  `limit`/`offset` are
**not** re-applied — that would corrupt server-side pagination by
re-offsetting an already-offset window.  Servers that don't honor
pagination return the full list and the caller iterates more rows
than requested; this is the documented degradation mode for
non-conforming servers.

### Reference server: `create_registry_router`

`create_registry_router(backend)` is a one-line glue that mounts the
same wire protocol `HTTPRegistryBackend` speaks, backed by any
`RegistryBackend` (`InMemoryBackend`, `DataKnobsRegistryAdapter` over
Postgres/SQLite/S3, …).  Useful when you want a config service whose
storage you can swap without rewriting route handlers, and when you
want to keep client and server in lockstep on protocol changes
(client tests and server tests pin the wire shape from both ends).

```python
from fastapi import FastAPI
from dataknobs_bots.registry import (
    DataKnobsRegistryAdapter,
    create_registry_router,
)

app = FastAPI()
adapter = DataKnobsRegistryAdapter(
    backend_type="postgres",
    backend_config={
        "host": "localhost",
        "database": "myapp",
        "table": "bot_configs",
    },
)

@app.on_event("startup")
async def startup() -> None:
    await adapter.initialize()
    app.include_router(
        create_registry_router(adapter),
        prefix="/api/v1",
    )
```

**Efficiency note.** When the underlying backend is
`DataKnobsRegistryAdapter` over Postgres, `filter_metadata` is pushed
down to a `metadata->>'<key>' = $1` predicate (text extraction +
equality) — server-side filtering rather than client-side scanning,
but **not** an indexed lookup by default.  Postgres's
`auto_create_table=True` path creates a GIN index over the `metadata`
JSONB column using the default `jsonb_ops` opclass, which optimizes
containment / existence operators (`@>`, `?`, `?&`, `?|`) and **does
not** accelerate `->>`-style text extractions.  Consumers who need
O(index lookup) on a hot `filter_metadata` key (e.g. `tenant_id`)
should add an expression index in their own migrations, for example:

```sql
CREATE INDEX idx_bot_configs_tenant_id
  ON bot_configs ((metadata->>'tenant_id'));
```

Backends without a JSONB column fall back to per-row matching via the
`metadata.X` field-path translation in their respective query
translators; this is also correct but unindexed.

**Optional dependency.** FastAPI is optional: importing the module
without FastAPI installed succeeds, but calling
`create_registry_router` raises `ImportError` with an install hint.

### Reading configs: `get_config` vs `peek_config`

Every `RegistryBackend` exposes two read APIs that differ only in
whether they register as user activity:

| Method | Updates `last_accessed_at`? | Use for |
|--------|-----------------------------|---------|
| `get_config(bot_id)` | Yes (where the backend tracks activity locally) | User-facing reads — the canonical "this bot was used" signal |
| `peek_config(bot_id)` | No (at the client surface) | Inspection, audit, hot-reload polling, admin tooling — reads that should not register as user activity |

The non-mutation guarantee for `peek_config` scopes to **backend-local
activity-tracking state**. Backends without local activity tracking
(notably `HTTPRegistryBackend`, where the server owns its own
access-tracking semantics) MAY satisfy the contract by delegating to
`get_config` — the guarantee then holds trivially at the client
surface. Servers that want to distinguish a non-touching peek define
that distinction in their own contract; the HTTP client does not
impose one on the wire.

`BotRegistry.get_config()` routes through `peek_config` for inspection
paths; use `BotRegistry.get_bot()` for user-facing bot resolution
(which always registers an access).

### Filter, Sort, and Pagination

Every `RegistryBackend` exposes a uniform kw-only filter / sort /
pagination surface on `list_all`, `list_active`, and `list_inactive`,
plus matching `count_all` / `count` / `count_inactive` for cheap-count
paths.  This is symmetric with the surfaces on `ArtifactRegistry`,
`RubricRegistry`, `GeneratorRegistry`, and FSM's
`load_steps`/`query_histories`, so a consumer composing multiple
registries sees one shape:

```python
from dataknobs_data import SortSpec, SortOrder

# Filter on the metadata channel (routes to `metadata.X`):
acme_bots = await backend.list_active(
    filter_metadata={"tenant_id": "acme"},
)

# Sort + paginate.  Sort pushes down to the database query so SQL
# backends can use indexes.
recent = await backend.list_active(
    sort=[SortSpec(field="created_at", order=SortOrder.DESC)],
    limit=20,
    offset=0,
)

# Count for cheap-count / pagination footers — routes through
# `AsyncDatabase.count(query)` so backends with pushdown counts
# (`SELECT COUNT(*) WHERE ...`) benefit transparently.
n = await backend.count(filter_metadata={"tenant_id": "acme"})

# Stream large tenant populations one Registration at a time.
async for reg in backend.stream(filter_metadata={"tenant_id": "acme"}):
    process(reg)
```

#### Soft-delete symmetry: active vs inactive

| Method | Returns |
|--------|---------|
| `list_active(...)` / `count(...)` | Registrations with `status="active"` |
| `list_inactive(...)` / `count_inactive(...)` | Registrations with `status="inactive"` (soft-deleted) |
| `list_all(status=None, ...)` / `count_all(status=None, ...)` | Both (or pass `status="active"` / `status="inactive"` to filter) |

Soft-deleted records are retained for audit / reactivation.  Use
`unregister(bot_id)` for hard delete.

### Cross-cutting metadata on `Registration`

`Registration.metadata` is a `dict[str, Any]` (default empty) that
backends route to their `metadata` column rather than mixing into
`config`.  Use it for cross-cutting context — `tenant_id`, audit info,
feature flags — that should be independently filterable without
disturbing the bot's portable configuration:

```python
await backend.register(
    "support-bot",
    {"llm": {"provider": "ollama", "model": "llama3.2"}},
    metadata={"tenant_id": "acme", "env": "prod"},
)

# Now filterable independently of `config`:
prod_acme = await backend.list_active(
    filter_metadata={"tenant_id": "acme", "env": "prod"},
)
```

The `Registration.to_dict` / `from_dict` JSON shape preserves
`metadata` so it round-trips through the HTTP wire and any
caller-side dump.

#### Same surface on `BotRegistry`

The same metadata channel, filter, and pagination kwargs are exposed
on the higher-level `BotRegistry` wrapper — consumers don't have to
drop to `registry._backend` to use them:

```python
from dataknobs_bots.bot import BotRegistry

registry = BotRegistry(backend=backend)
await registry.initialize()

# Register with metadata
await registry.register(
    "support-bot",
    config,
    metadata={"tenant_id": "acme"},
)

# List active bot IDs filtered by tenant
ids = await registry.list_bots(filter_metadata={"tenant_id": "acme"})

# Or get full Registrations (timestamps, status, metadata)
regs = await registry.list_registrations(
    filter_metadata={"tenant_id": "acme"},
)

# Count by tenant
n = await registry.count(filter_metadata={"tenant_id": "acme"})
```

> **Migration note (DataKnobsRegistryAdapter).** Existing deployments
> on Postgres / SQLite / S3 / file backends must rewrite stored rows
> once to populate the new `metadata` column — pre-migration rows
> have an empty metadata column and won't match `filter_metadata`
> until re-registered.  See `packages/bots/CHANGELOG.md`'s Unreleased
> entry for the full migration note.

### Factory Function

Use `create_registry_backend` for configuration-driven creation:

```python
from dataknobs_bots.registry import create_registry_backend

# Memory backend
backend = create_registry_backend("memory", {})

# PostgreSQL backend
backend = create_registry_backend("postgres", {
    "host": "localhost",
    "database": "myapp",
    "table": "bot_configs",
})

# HTTP backend
backend = create_registry_backend("http", {
    "base_url": "https://config-service/api/v1",
    "auth_token": "secret",
})

await backend.initialize()
```

## Caching Registry Manager

`CachingRegistryManager` provides TTL-based caching with event-driven invalidation:

```python
from dataknobs_bots.registry import CachingRegistryManager, InMemoryBackend

class MyBotManager(CachingRegistryManager):
    """Custom manager with caching."""

    async def _create_instance(self, bot_id: str, config: dict) -> MyBot:
        """Create bot instance from config."""
        return await MyBot.from_config(bot_id, config)

    async def _destroy_instance(self, instance: MyBot) -> None:
        """Cleanup bot instance."""
        await instance.close()

# Usage
backend = InMemoryBackend()
await backend.initialize()

manager = MyBotManager(
    backend=backend,
    event_bus=event_bus,  # Optional, for event-driven invalidation
    cache_ttl=300,        # 5 minutes
    max_cache_size=1000,
)
await manager.initialize()

# Get or create (uses cache)
bot = await manager.get_or_create("my-bot")

# Invalidate specific bot
await manager.invalidate("my-bot")

# Invalidate all cached instances
await manager.invalidate_all()
```

### Features

- **TTL-based expiration**: Cached instances expire after configured time
- **LRU eviction**: Least recently used items evicted when max size reached
- **Event-driven invalidation**: Automatic invalidation on EventBus events
- **Lazy loading**: Only creates instances on demand
- **Configurable refresh**: Control when instances are refreshed

### ConfigCachingManager

For caching resolved configurations (without creating instances):

```python
from dataknobs_bots.registry import ConfigCachingManager, ResolvedConfig

manager = ConfigCachingManager(
    backend=backend,
    cache_ttl=300,
)
await manager.initialize()

# Get resolved config
resolved: ResolvedConfig = await manager.get("my-bot")
print(f"Config: {resolved.config}")
print(f"Version: {resolved.version}")
```

## Hot Reload

Hot reload enables updating bot configurations without application restarts.

### RegistryPoller

Detects changes in backends that don't support push notifications:

```python
from dataknobs_bots.registry import RegistryPoller

async def on_change(bot_id: str, old_config: dict, new_config: dict):
    print(f"Config changed for {bot_id}")
    await invalidate_bot(bot_id)

poller = RegistryPoller(
    backend=backend,
    poll_interval=60,  # seconds
    on_change=on_change,
)
await poller.start()

# Later
await poller.stop()
```

### HotReloadManager

Coordinates polling, events, and cache invalidation:

```python
from dataknobs_bots.registry import HotReloadManager, ReloadMode

hot_reload = HotReloadManager(
    backend=backend,
    caching_manager=bot_manager,
    event_bus=event_bus,         # Optional
    mode=ReloadMode.POLLING,     # or ReloadMode.EVENT_DRIVEN, ReloadMode.HYBRID
    polling_interval=60,
)
await hot_reload.initialize()

# Subscribes to events automatically (if event_bus provided)
# Starts poller if mode includes polling
# Invalidates cache on changes

await hot_reload.shutdown()
```

### Reload Modes

| Mode | Description | Use When |
|------|-------------|----------|
| `POLLING` | Periodic polling only | HTTP backend, no event bus |
| `EVENT_DRIVEN` | Event-driven only | Have event bus, backends support push |
| `HYBRID` | Both polling and events | Belt-and-suspenders reliability |

## Knowledge Resource Storage

Store raw knowledge files (markdown, JSON, etc.) before vector embedding.

### Quick Start

```python
from dataknobs_bots.knowledge import (
    create_knowledge_backend,
    KnowledgeFile,
    IngestionStatus,
)

# Create backend
backend = create_knowledge_backend("file", {"path": "./data/knowledge"})
await backend.initialize()

# Create knowledge base
info = await backend.create_kb("my-domain", metadata={"description": "My KB"})

# Upload files
file_info = await backend.put_file(
    "my-domain",
    "content/intro.md",
    b"# Introduction\n\nWelcome to our knowledge base!",
    content_type="text/markdown",
)

# List files
files = await backend.list_files("my-domain", prefix="content/")

# Get file content
content = await backend.get_file("my-domain", "content/intro.md")

# Check for changes
checksum = await backend.get_checksum("my-domain")
has_changes = await backend.has_changes_since("my-domain", previous_checksum)

# Update ingestion status
await backend.set_ingestion_status("my-domain", IngestionStatus.READY)

# Cleanup
await backend.delete_file("my-domain", "content/intro.md")
await backend.delete_kb("my-domain")
await backend.close()
```

### Storage Backends

#### InMemoryKnowledgeBackend

For testing:

```python
backend = create_knowledge_backend("memory", {})
```

#### FileKnowledgeBackend

Local filesystem storage:

```python
backend = create_knowledge_backend("file", {
    "path": "./data/knowledge"
})
```

Directory structure:

```
./data/knowledge/
├── my-domain/
│   ├── _metadata.json      # KB metadata
│   ├── content/
│   │   ├── intro.md
│   │   └── guide.md
│   └── data/
│       └── faq.json
└── other-domain/
    └── ...
```

#### S3KnowledgeBackend

Production S3 storage. `region` (or its `region_name` alias) is
optional — when omitted, boto's default chain
(`AWS_DEFAULT_REGION` env, `~/.aws/config`, IMDS, then `us-east-1`
terminal fallback) resolves the value. See
[S3 Session Configuration](../../data/s3-session.md) for the full
chain and unified config shape:

```python
# Defer to boto's chain (recommended for env-driven deployments).
backend = create_knowledge_backend("s3", {
    "bucket": "my-bucket",
    "prefix": "knowledge/",
})

# Or pin explicitly when overriding the deployment env.
backend = create_knowledge_backend("s3", {
    "bucket": "my-bucket",
    "prefix": "knowledge/",
    "region": "eu-west-1",
    "endpoint_url": "http://localhost:4566",  # LocalStack / MinIO
})
```

### Streaming Large Files

```python
stream = await backend.stream_file("my-domain", "large-file.bin")
async for chunk in stream:
    process(chunk)
```

### Models

#### KnowledgeFile

```python
@dataclass
class KnowledgeFile:
    path: str                    # Relative path within KB
    content_type: str            # MIME type
    size_bytes: int              # File size in bytes
    checksum: str                # MD5/SHA256 hash
    uploaded_at: datetime        # When uploaded/modified
    metadata: dict[str, Any]     # Optional custom metadata
```

#### KnowledgeBaseInfo

```python
@dataclass
class KnowledgeBaseInfo:
    domain_id: str               # Unique identifier
    file_count: int              # Number of files
    total_size_bytes: int        # Total size of all files
    last_updated: datetime       # When last modified
    version: str                 # Incremented on changes
    ingestion_status: IngestionStatus
    ingestion_error: str | None  # Error message if status is ERROR
    vector_store_path: str | None  # Path to persisted vector store
    metadata: dict[str, Any]     # Custom metadata
```

#### IngestionStatus

```python
class IngestionStatus(Enum):
    PENDING = "pending"       # Awaiting ingestion
    INGESTING = "ingesting"   # Currently processing
    READY = "ready"           # Ready for queries
    ERROR = "error"           # Ingestion failed
```

## Knowledge Ingestion Manager

Coordinates loading files from storage into RAG knowledge base.

### Basic Usage

```python
from dataknobs_bots.knowledge import (
    create_knowledge_backend,
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
)

# Create components
file_backend = create_knowledge_backend("file", {"path": "./data/kb"})
await file_backend.initialize()

rag_kb = await RAGKnowledgeBase.from_config({
    "vector_store": {"backend": "memory", "dimensions": 384},
    "embedding_provider": "echo",
})

# Create ingestion manager
manager = KnowledgeIngestionManager(
    source=file_backend,
    destination=rag_kb,
    event_bus=event_bus,  # Optional, publishes ingestion events
)

# Ingest all files
result = await manager.ingest("my-domain", clear_existing=True)
print(f"Processed {result.files_processed} files")
print(f"Created {result.chunks_created} chunks")
if result.errors:
    print(f"Errors: {result.errors}")

# `clear_existing=True` is safe in shared multi-tenant stores: the
# clear is scoped to the supplied domain via
# `RAGKnowledgeBase.clear(filter={"domain_id": "my-domain"})`, so
# re-ingesting one domain leaves chunks from other domains intact.

# Ingest only if changed
result = await manager.ingest_if_changed("my-domain", last_version="abc123")
if result is None:
    print("No changes detected")
```

### IngestionResult

```python
@dataclass
class IngestionResult:
    domain_id: str
    files_processed: int
    chunks_created: int
    files_skipped: int
    errors: list[dict]
    started_at: datetime
    completed_at: datetime | None

    @property
    def success(self) -> bool:
        """True if no errors occurred."""

    @property
    def duration_seconds(self) -> float | None:
        """Duration of ingestion in seconds."""
```

### Supported File Types

| Extension | Type | Processing |
|-----------|------|------------|
| `.md`, `.markdown` | Markdown | Direct to chunks |
| `.json` | JSON | Transform to markdown, then chunk |
| `.yaml`, `.yml` | YAML | Transform to markdown, then chunk |
| `.csv` | CSV | Convert to markdown table |
| `.txt` | Text | Direct to chunks |
| `.gz` | Compressed | Auto-decompress, then process |

## Complete Example

### Bot Manager with Caching and Hot Reload

```python
import asyncio
from dataknobs_bots.registry import (
    CachingRegistryManager,
    HotReloadManager,
    InMemoryBackend,
    ReloadMode,
)
from dataknobs_common.events import create_event_bus, Event, EventType


class Bot:
    def __init__(self, bot_id: str, config: dict):
        self.bot_id = bot_id
        self.config = config

    async def chat(self, message: str) -> str:
        return f"Bot {self.bot_id} received: {message}"

    async def close(self):
        pass


class BotManager(CachingRegistryManager[Bot]):
    async def _create_instance(self, bot_id: str, config: dict) -> Bot:
        return Bot(bot_id, config)

    async def _destroy_instance(self, instance: Bot) -> None:
        await instance.close()


async def main():
    # Setup event bus
    event_bus = create_event_bus({"backend": "memory"})
    await event_bus.connect()

    # Setup registry backend
    backend = InMemoryBackend()
    await backend.initialize()

    # Register a bot config
    await backend.register("support-bot", {
        "llm": {"provider": "anthropic", "model": "claude-3-sonnet"},
        "system_prompt": "You are a helpful support agent."
    })

    # Create bot manager with caching
    manager = BotManager(
        backend=backend,
        event_bus=event_bus,
        cache_ttl=300,
    )
    await manager.initialize()

    # Setup hot reload
    hot_reload = HotReloadManager(
        backend=backend,
        caching_manager=manager,
        event_bus=event_bus,
        mode=ReloadMode.EVENT_DRIVEN,
    )
    await hot_reload.initialize()

    # Get bot (creates and caches)
    bot = await manager.get_or_create("support-bot")
    response = await bot.chat("Hello!")
    print(response)

    # Update config - hot reload will invalidate cache
    await backend.update("support-bot", {
        "llm": {"provider": "anthropic", "model": "claude-3-opus"},
        "system_prompt": "You are a helpful support agent."
    })

    # Publish update event (triggers invalidation)
    await event_bus.publish("registry:bots", Event(
        type=EventType.UPDATED,
        topic="registry:bots",
        payload={"bot_id": "support-bot"}
    ))

    # Small delay for event processing
    await asyncio.sleep(0.1)

    # Next get_or_create will create new bot with updated config
    bot = await manager.get_or_create("support-bot")

    # Cleanup
    await hot_reload.shutdown()
    await event_bus.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### Knowledge Base with Hot Reload

```python
import asyncio
from dataknobs_bots.knowledge import (
    create_knowledge_backend,
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
    IngestionStatus,
)
from dataknobs_common.events import create_event_bus, Event, EventType


async def main():
    # Setup
    event_bus = create_event_bus({"backend": "memory"})
    await event_bus.connect()

    kb_backend = create_knowledge_backend("file", {"path": "./data/kb"})
    await kb_backend.initialize()

    rag_kb = await RAGKnowledgeBase.from_config({
        "vector_store": {"backend": "memory", "dimensions": 384},
        "embedding_provider": "echo",
    })

    # Create ingestion manager
    ingestion = KnowledgeIngestionManager(
        source=kb_backend,
        destination=rag_kb,
        event_bus=event_bus,
    )

    # Subscribe to ingestion events
    async def on_ingestion(event: Event) -> None:
        if event.type == EventType.CUSTOM:
            print(f"Ingestion complete: {event.payload}")

    await event_bus.subscribe("ingest:domain:end", on_ingestion)

    # Create knowledge base and upload files
    await kb_backend.create_kb("support", metadata={"description": "Support docs"})

    await kb_backend.put_file(
        "support",
        "faq.md",
        b"# FAQ\n\n## How do I reset my password?\n\nGo to settings...",
        content_type="text/markdown",
    )

    # Ingest
    result = await ingestion.ingest("support")
    print(f"Ingested {result.files_processed} files, {result.chunks_created} chunks")

    # Query
    results = await rag_kb.search("reset password", k=3)
    for r in results:
        print(f"- {r.content[:100]}...")

    # Cleanup
    await event_bus.close()
    await kb_backend.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Reference

### Environment-Based Configuration

```yaml
# configs/environments/development.yaml
name: development

resources:
  databases:
    registry:
      backend: memory
  event_buses:
    default:
      backend: memory
  knowledge:
    backend: file
    path: ${DATA_DIR:./data}/knowledge

settings:
  log_level: DEBUG
```

```yaml
# configs/environments/production.yaml
name: production

resources:
  databases:
    registry:
      backend: postgres
      host: ${RDS_HOST}
      database: ${RDS_DATABASE}
      table: registrations
  event_buses:
    default:
      backend: redis
      host: ${ELASTICACHE_HOST}
      ssl: true
  knowledge:
    backend: s3
    bucket: ${KNOWLEDGE_BUCKET}
    prefix: knowledge/
    # `region` omitted — boto's default chain resolves
    # AWS_DEFAULT_REGION env / ~/.aws/config / IMDS at connect time.
    # Pin explicitly here only when you need to override the env.

settings:
  log_level: WARNING
```

### Application Config with $resource References

```yaml
# configs/domains/my-bot.yaml
domain:
  id: my-bot
  name: My Bot

bot:
  registry:
    $resource: registry
    type: databases

  events:
    $resource: default
    type: event_buses

  knowledge:
    $resource: knowledge
    type: knowledge
```

## Module Exports

### dataknobs_bots.registry

```python
from dataknobs_bots.registry import (
    # Models
    Registration,
    # Protocols
    RegistryBackend,
    # Backends
    InMemoryBackend,
    DataKnobsRegistryAdapter,
    HTTPRegistryBackend,
    # Caching
    CachingRegistryManager,
    ConfigCachingManager,
    ResolvedConfig,
    # Hot Reload
    RegistryPoller,
    HotReloadManager,
    ReloadMode,
    # Factory
    create_registry_backend,
    # Portability
    PortabilityError,
    validate_portability,
    has_resource_references,
    is_portable,
)
```

### dataknobs_bots.knowledge (storage)

```python
from dataknobs_bots.knowledge import (
    # Storage backends
    KnowledgeResourceBackend,
    InMemoryKnowledgeBackend,
    FileKnowledgeBackend,
    S3KnowledgeBackend,
    create_knowledge_backend,
    # Models
    KnowledgeFile,
    KnowledgeBaseInfo,
    IngestionStatus,
    # Ingestion
    KnowledgeIngestionManager,
    IngestionResult,
)
```
