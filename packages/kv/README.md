# DataKnobs KV Package

A powerful key/value store abstraction with hierarchical keys, pattern matching, and rich metadata support, built on top of the dataknobs-data package.

## Overview

The `dataknobs-kv` package provides a simple yet powerful interface for managing key/value pairs across multiple storage backends. It features hierarchical key structures, advanced pattern matching, metadata management, and seamless integration with various storage technologies through the dataknobs-data abstraction layer.

## Features

- **Hierarchical Keys**: Dot-notation key paths for organized data structure
- **Pattern Matching**: Powerful glob-style patterns for key operations
- **Rich Metadata**: Associate custom metadata with every key/value pair
- **Multiple Backends**: Memory, File, PostgreSQL, Elasticsearch, S3, and more
- **TTL Support**: Automatic expiration for time-sensitive data
- **Atomic Operations**: Compare-and-swap, increment, append operations
- **Namespace Isolation**: Multi-tenant support with isolated namespaces
- **Type Safety**: Strong typing with automatic serialization/deserialization

## Installation

```bash
# Basic installation
pip install dataknobs-kv

# With caching support
pip install dataknobs-kv[cache]

# With all features
pip install dataknobs-kv[all]
```

## Quick Start

```python
from dataknobs_kv import KVStore

# Create a store with memory backend
store = KVStore(backend="memory")

# Basic operations
await store.set("app.name", "MyApplication")
await store.set("app.version", "1.0.0")
await store.set("app.config.debug", True)

# Get values
name = await store.get("app.name")
debug = await store.get("app.config.debug")

# Pattern matching
app_keys = await store.keys("app.*")
all_config = await store.get_pattern("app.config.*")

# Delete keys
await store.delete("app.config.debug")
await store.delete_pattern("app.temp.*")
```

## Hierarchical Key Structure

```python
# Organize data with dot-notation paths
await store.set("users.123.name", "John Doe")
await store.set("users.123.email", "john@example.com")
await store.set("users.123.settings.theme", "dark")
await store.set("users.123.settings.notifications", True)

# Get all user data
user_data = await store.get_pattern("users.123.*")

# Get all user settings
user_settings = await store.get_pattern("users.123.settings.*")

# Delete entire user
await store.delete_pattern("users.123.**")
```

## Pattern Matching

```python
# Wildcards
await store.keys("users.*.name")        # All user names
await store.keys("*.config.*")          # All config at any level
await store.keys("logs.2024-*")         # All 2024 logs

# Recursive wildcards
await store.keys("app.**")              # Everything under app
await store.keys("**.error")            # All error keys at any level

# Single character and sets
await store.keys("user.?.name")         # user.1.name, user.a.name, etc.
await store.keys("log.[0-9].txt")       # log.0.txt through log.9.txt

# Alternatives
await store.keys("env.{dev,prod}.config") # dev or prod config
```

## Metadata Management

```python
# Set with metadata
await store.set(
    "document.report.pdf",
    document_bytes,
    metadata={
        "content_type": "application/pdf",
        "author": "John Doe",
        "created": "2024-01-15",
        "tags": ["quarterly", "finance"],
        "ttl": 86400  # Expire in 24 hours
    }
)

# Get metadata
metadata = await store.get_metadata("document.report.pdf")
print(f"Author: {metadata['author']}")
print(f"Tags: {metadata['tags']}")

# Update metadata
await store.set_metadata("document.report.pdf", {
    "reviewed": True,
    "reviewer": "Jane Smith"
})
```

## TTL and Expiration

```python
# Set with TTL (time to live)
await store.set("session.abc123", session_data, ttl=3600)  # 1 hour
await store.set("cache.results", results, ttl=300)         # 5 minutes

# Check if key exists (returns False if expired)
if await store.exists("session.abc123"):
    data = await store.get("session.abc123")

# Manual cleanup of expired keys
expired_count = await store.cleanup_expired()
print(f"Removed {expired_count} expired keys")

# Automatic cleanup (runs in background)
store = KVStore(backend="memory", auto_cleanup=True, cleanup_interval=60)
```

## Atomic Operations

```python
# Compare and swap (CAS)
success = await store.compare_and_swap(
    "version",
    old_value="1.0.0",
    new_value="1.1.0"
)

# Increment/decrement counters
views = await store.increment("stats.page_views")
remaining = await store.increment("inventory.item_123", delta=-1)

# Append to strings
await store.append("logs.access", "\n2024-01-15 10:30 User login")
await store.append("notes", ", Remember to update docs")
```

## Batch Operations

```python
# Set multiple values
await store.set_many({
    "config.host": "localhost",
    "config.port": 8080,
    "config.ssl": True,
    "config.timeout": 30
})

# Get multiple values
values = await store.get_many([
    "config.host",
    "config.port",
    "config.ssl"
])

# Delete multiple keys
deleted = await store.delete_many([
    "temp.file1",
    "temp.file2",
    "cache.old"
])
```

## Namespace Isolation

```python
# Create isolated namespaces
user_store = KVStore(backend="postgres", namespace="user_data")
system_store = KVStore(backend="postgres", namespace="system")
cache_store = KVStore(backend="memory", namespace="cache")

# Same keys, different namespaces
await user_store.set("settings", {"theme": "dark"})
await system_store.set("settings", {"debug": True})

# Values are isolated
user_settings = await user_store.get("settings")   # {"theme": "dark"}
system_settings = await system_store.get("settings") # {"debug": True}
```

## Backend Configuration

### Memory Backend
```python
store = KVStore(backend="memory")
```

### File Backend
```python
store = KVStore(backend="file", config={
    "path": "/data/kv_store.json",
    "format": "json",
    "compression": "gzip"
})
```

### PostgreSQL Backend
```python
store = KVStore(backend="postgres", config={
    "host": "localhost",
    "database": "myapp",
    "table": "kv_store",
    "user": "dbuser",
    "password": "dbpass"
})
```

### Redis Backend
```python
store = KVStore(backend="redis", config={
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": "optional"
})
```

### S3 Backend
```python
store = KVStore(backend="s3", config={
    "bucket": "my-kv-store",
    "prefix": "data/",
    "region": "us-west-2"
})
```

## Advanced Usage

### Custom Serialization
```python
import pickle

# Custom serializer for complex objects
class CustomStore(KVStore):
    def serialize(self, value):
        if isinstance(value, MyComplexClass):
            return pickle.dumps(value)
        return super().serialize(value)
    
    def deserialize(self, data, value_type):
        if value_type == "custom":
            return pickle.loads(data)
        return super().deserialize(data, value_type)
```

### Caching Layer
```python
# Add caching for performance
store = KVStore(
    backend="postgres",
    cache="memory",
    cache_ttl=60,  # Cache for 60 seconds
    cache_size=1000  # Max 1000 items in cache
)
```

### Migration Between Backends
```python
from dataknobs_kv import migrate_store

# Migrate from file to PostgreSQL
source = KVStore(backend="file", config={"path": "data.json"})
target = KVStore(backend="postgres", config=postgres_config)

await migrate_store(source, target, batch_size=100)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=dataknobs_kv

# Type checking
mypy src/dataknobs_kv

# Linting
ruff check src/dataknobs_kv

# Format code
black src/dataknobs_kv
```

## Architecture

The package is built on top of dataknobs-data, providing:

- **Keys**: Hierarchical path management with validation
- **Values**: Type-safe serialization/deserialization
- **Patterns**: Advanced pattern matching engine
- **Metadata**: Rich metadata with system and user fields
- **Namespaces**: Isolation for multi-tenant applications
- **Backends**: Leverages dataknobs-data storage backends

## Performance

- Optimized pattern matching with compiled regex
- Efficient batch operations
- Optional caching layer
- Connection pooling for database backends
- Async/await support for concurrent operations

## Contributing

Contributions are welcome! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.