# DataKnobs KV Package Design Plan

## Overview
A key/value store abstraction package built on top of the dataknobs-data package, providing a simple and powerful interface for managing key/value pairs with metadata across multiple storage backends.

## Core Principles
1. **Hierarchical Keys**: Support for structured, dot-notation keys
2. **Multi-Type Values**: Handle various content types (JSON, text, binary, etc.)
3. **Metadata Rich**: Associate metadata with every key/value pair
4. **Backend Agnostic**: Leverage dataknobs-data for storage flexibility
5. **Pattern Matching**: Powerful key pattern matching and globbing

## Architecture

### 1. Key Structure

#### Key Design
```python
Key = {
    "path": str,              # Full key path (e.g., "app.config.database.host")
    "segments": List[str],    # Path segments ["app", "config", "database", "host"]
    "namespace": Optional[str], # Optional namespace for isolation
    "metadata": Dict[str, Any]  # Key-specific metadata
}
```

#### Key Features
- Hierarchical structure with dot notation
- Namespace isolation for multi-tenancy
- Pattern matching with wildcards (*, **, ?)
- Key validation and normalization
- Case sensitivity options

### 2. Value Types

#### Value Structure
```python
Value = {
    "data": Any,              # The actual value
    "type": ValueType,        # Type identifier
    "encoding": Optional[str], # Encoding for binary data
    "metadata": Dict[str, Any], # Value-specific metadata
    "checksum": Optional[str],  # Data integrity check
}

ValueType = Enum(
    "STRING",    # Plain text
    "JSON",      # JSON object
    "BINARY",    # Binary data
    "NUMBER",    # Numeric value
    "BOOLEAN",   # True/False
    "DOCUMENT",  # PDF, Word, etc.
    "IMAGE",     # Image files
    "CUSTOM"     # Custom types
)
```

### 3. KV Store Interface

#### Core Operations
```python
class KVStore:
    # Basic operations
    async def get(self, key: str) -> Optional[Value]
    async def set(self, key: str, value: Any, metadata: Dict = None) -> bool
    async def delete(self, key: str) -> bool
    async def exists(self, key: str) -> bool
    
    # Batch operations
    async def get_many(self, keys: List[str]) -> Dict[str, Value]
    async def set_many(self, items: Dict[str, Any]) -> Dict[str, bool]
    async def delete_many(self, keys: List[str]) -> Dict[str, bool]
    
    # Pattern operations
    async def keys(self, pattern: str = "*") -> List[str]
    async def get_pattern(self, pattern: str) -> Dict[str, Value]
    async def delete_pattern(self, pattern: str) -> int
    
    # Metadata operations
    async def get_metadata(self, key: str) -> Optional[Dict]
    async def set_metadata(self, key: str, metadata: Dict) -> bool
    
    # Advanced operations
    async def compare_and_swap(self, key: str, old_value: Any, new_value: Any) -> bool
    async def increment(self, key: str, delta: int = 1) -> int
    async def append(self, key: str, value: str) -> str
```

### 4. Pattern Matching

#### Pattern Syntax
- `*` - Matches any characters except dots
- `**` - Matches any characters including dots
- `?` - Matches single character
- `[abc]` - Matches any character in set
- `{foo,bar}` - Matches alternatives

#### Examples
```python
# Match all config keys
store.keys("app.config.*")

# Match all database keys at any level
store.keys("**.database.**")

# Match specific pattern
store.keys("user.?.settings")

# Match alternatives
store.keys("app.{dev,prod}.config")
```

### 5. Metadata Management

#### Metadata Types
```python
SystemMetadata = {
    "created_at": datetime,
    "updated_at": datetime,
    "accessed_at": datetime,
    "ttl": Optional[int],      # Time to live in seconds
    "version": int,             # Version number
    "checksum": str,            # Data integrity
    "size": int,                # Value size in bytes
    "content_type": str,        # MIME type
}

UserMetadata = Dict[str, Any]  # Custom user metadata
```

### 6. Storage Backend Integration

#### Backend Mapping
```python
# KV record in dataknobs-data format
Record = {
    "id": str,           # Key hash for indexing
    "key": str,          # Full key path
    "namespace": str,    # Namespace
    "value": Any,        # Serialized value
    "value_type": str,   # Type identifier
    "metadata": Dict,    # Combined metadata
    "expires_at": Optional[datetime],
}
```

## API Design

### Basic Usage
```python
from dataknobs_kv import KVStore

# Create store with specific backend
store = KVStore(backend="memory")  # or "file", "postgres", etc.

# Basic operations
await store.set("app.name", "MyApp")
value = await store.get("app.name")
await store.delete("app.name")

# With metadata
await store.set("user.123.profile", 
    {"name": "John", "age": 30},
    metadata={"tags": ["vip", "active"]}
)

# Pattern matching
config_keys = await store.keys("app.config.*")
all_configs = await store.get_pattern("app.config.*")
```

### Hierarchical Operations
```python
# Set nested configuration
await store.set("app.config.database.host", "localhost")
await store.set("app.config.database.port", 5432)
await store.set("app.config.database.name", "mydb")

# Get all database config
db_config = await store.get_pattern("app.config.database.*")

# Delete entire subtree
await store.delete_pattern("app.config.database.**")
```

### TTL and Expiration
```python
# Set with TTL
await store.set("session.abc123", session_data, ttl=3600)  # 1 hour

# Check if expired
if await store.exists("session.abc123"):
    data = await store.get("session.abc123")

# Cleanup expired keys
expired_count = await store.cleanup_expired()
```

### Atomic Operations
```python
# Compare and swap
success = await store.compare_and_swap(
    "counter", 
    old_value=5, 
    new_value=6
)

# Increment counter
new_value = await store.increment("page.views", delta=1)

# Append to string
result = await store.append("log.entries", "\nNew entry")
```

### Namespace Isolation
```python
# Create namespaced stores
user_store = KVStore(backend="postgres", namespace="user_data")
system_store = KVStore(backend="postgres", namespace="system")

# Operations are isolated
await user_store.set("config", user_config)
await system_store.set("config", system_config)

# Different values for same key
user_config = await user_store.get("config")
system_config = await system_store.get("config")
```

## Implementation Phases

### Phase 1: Core Structure (Week 1)
1. Key management system
2. Value type definitions
3. Metadata structure
4. Basic KVStore interface

### Phase 2: Backend Integration (Week 1-2)
1. Integration with dataknobs-data
2. Record mapping
3. Serialization/deserialization
4. Backend configuration

### Phase 3: Basic Operations (Week 2)
1. Get/Set/Delete operations
2. Exists checking
3. Batch operations
4. Error handling

### Phase 4: Pattern Matching (Week 2-3)
1. Pattern parser
2. Key matching algorithm
3. Pattern-based operations
4. Glob support

### Phase 5: Advanced Features (Week 3)
1. TTL and expiration
2. Atomic operations
3. Namespace support
4. Metadata management

### Phase 6: Performance (Week 4)
1. Caching layer
2. Batch optimizations
3. Index management
4. Connection pooling

## Testing Strategy

### Unit Tests
- Key validation and parsing
- Pattern matching
- Value serialization
- Metadata handling

### Integration Tests
- Backend compatibility
- Cross-backend operations
- Namespace isolation
- Expiration handling

### Performance Tests
- Operation latency
- Batch operation efficiency
- Pattern matching speed
- Memory usage

## Dependencies
```toml
[project]
dependencies = [
    "dataknobs-data>=0.1.0",
    "pydantic>=2.0.0",
    "python-dateutil>=2.8.0",
]

[project.optional-dependencies]
cache = ["redis>=4.0.0", "diskcache>=5.0.0"]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-benchmark>=4.0.0",
]
```

## File Structure
```
packages/kv/
├── README.md
├── DESIGN_PLAN.md (this file)
├── PROGRESS_CHECKLIST.md
├── pyproject.toml
├── src/
│   └── dataknobs_kv/
│       ├── __init__.py
│       ├── store.py          # Main KV store implementation
│       ├── keys.py           # Key management and validation
│       ├── values.py         # Value types and serialization
│       ├── patterns.py       # Pattern matching engine
│       ├── metadata.py       # Metadata handling
│       ├── namespace.py      # Namespace isolation
│       ├── cache.py          # Caching layer
│       ├── expiration.py     # TTL and expiration
│       └── exceptions.py     # Custom exceptions
└── tests/
    ├── conftest.py
    ├── test_store.py
    ├── test_keys.py
    ├── test_patterns.py
    ├── test_metadata.py
    ├── test_namespace.py
    ├── test_expiration.py
    └── fixtures/
        ├── sample_data.json
        └── test_patterns.yaml
```

## Migration Path

### From Dictionary/JSON Files
```python
# Old code
import json
with open("config.json") as f:
    config = json.load(f)
value = config.get("database", {}).get("host")

# New code
store = KVStore(backend="file")
value = await store.get("database.host")
```

### From Redis
```python
# Old code
import redis
r = redis.Redis()
r.set("user:123:name", "John")
name = r.get("user:123:name")

# New code
store = KVStore(backend="redis")
await store.set("user.123.name", "John")
name = await store.get("user.123.name")
```

## Integration Examples

### With dataknobs-config
```python
from dataknobs_config import Config
from dataknobs_kv import KVStore

# Use KV store as config backend
store = KVStore(backend="postgres")
config = Config(backend=store)

# Load configuration from KV store
await config.load("app.config.**")
```

### With dataknobs-data
```python
from dataknobs_data import Database
from dataknobs_kv import KVStore

# Share the same backend
db = Database.create("postgres", config)
store = KVStore(database=db)

# Operations use the same underlying storage
await store.set("cache.user.123", user_data)
```

## Success Metrics
1. **Performance**: <10ms latency for basic operations
2. **Scalability**: Support for millions of keys
3. **Compatibility**: Works with all dataknobs-data backends
4. **Reliability**: 99.9% operation success rate
5. **Adoption**: Used by 2+ other packages