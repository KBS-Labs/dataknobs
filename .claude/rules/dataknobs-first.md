# Dataknobs-First Development

## Before Implementing New Utilities

When you need common functionality, check existing dataknobs packages first.

### 1. Consult the Reference Table

See `~/.claude/rules/dataknobs-reference.md` for the complete verified lookup table of all classes, import paths, and roles. Use it to find existing constructs before writing new code.

### 2. Key Constructs by Category

**Database Backends** (`dataknobs_data`):
- Base classes: `SyncDatabase`, `AsyncDatabase`
- In-memory: `SyncMemoryDatabase`, `AsyncMemoryDatabase`
- Production: `SyncPostgresDatabase`, `SyncSQLiteDatabase`, `SyncElasticsearchDatabase`, `SyncS3Database`, `SyncDuckDBDatabase`, `SyncFileDatabase` (+ async variants)
- Factories: `database_factory`, `async_database_factory` (create by backend key: `"memory"`, `"sqlite"`, `"postgres"`, etc.)

**Vector Stores** (`dataknobs_data`):
- Base: `VectorStore`
- Implementations: `MemoryVectorStore`, `FaissVectorStore`, `ChromaVectorStore`, `PgVectorStore`

**LLM Providers** (`dataknobs_llm`):
- Base: `AsyncLLMProvider` (`from dataknobs_llm.llm.base import AsyncLLMProvider`)
- Implementations: `OllamaProvider`, `OpenAIProvider`, `AnthropicProvider`, `HuggingFaceProvider`, `EchoProvider`
- Factory: `LLMProviderFactory` (create by provider key: `"ollama"`, `"openai"`, `"echo"`, etc.)

**Configuration** (`dataknobs_config`):
- `Config` - Modular config from YAML/JSON/dict with `get()`, `set()`, `build_object()`, `get_instance()`
- `ConfigurableBase` - Base for configurable objects
- `EnvironmentConfig` - Environment-specific bindings
- `ConfigBindingResolver` - Resolve logical names to concrete instances

**Registries** (`dataknobs_common`):
- `Registry[T]` - Thread-safe generic registry
- `AsyncRegistry[T]` - Async-safe variant
- `PluginRegistry[T]` - With factory support and lazy instantiation

**Events** (`dataknobs_common`):
- `EventBus` protocol with `InMemoryEventBus`, PostgreSQL, and Redis backends
- Factory: `create_event_bus("memory")`, `"postgres"`, `"redis"`

### 3. If Utility Doesn't Exist

Ask these questions:
1. Is this generally reusable across projects?
2. Does it fit an existing dataknobs package?
3. Should it be a new dataknobs utility?

**If YES to reusability:**
- Add to the appropriate dataknobs package
- Design abstraction first, then implementation
- Include tests using real constructs (not mocks)
- Update both documentation locations

**If project-specific:**
- Implement in the current project
- Consider if patterns emerge for future extraction to dataknobs

### 4. Using Dataknobs in Tests

Use real testing constructs, not mocks:

```python
# GOOD: Real implementations
from dataknobs_llm import EchoProvider
from dataknobs_data.backends.memory import AsyncMemoryDatabase

provider = EchoProvider()
db = AsyncMemoryDatabase()

# BAD: Mocks hide integration issues
provider = MagicMock(spec=AsyncLLMProvider)
db = MagicMock(spec=AsyncDatabase)
```
