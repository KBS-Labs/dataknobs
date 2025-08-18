# Memory Backend

## Overview

The Memory Backend provides an in-memory storage solution ideal for development, testing, and small-scale applications. It offers full query capabilities without external dependencies.

## Features

- **Zero configuration** - No setup required
- **Fast operations** - All data in memory
- **Full query support** - All operators and features
- **Thread-safe** - Safe for concurrent access
- **Both sync and async** - `SyncMemoryDatabase` and `AsyncMemoryDatabase`

## Usage

### Basic Setup

```python
from dataknobs_data import SyncMemoryDatabase, Record

# Create database instance
db = SyncMemoryDatabase()

# Create and store records
record = Record({"name": "Alice", "age": 30})
record_id = db.create(record)

# Retrieve record
retrieved = db.read(record_id)
print(retrieved["name"])  # "Alice"
```

### Async Version

```python
from dataknobs_data import AsyncMemoryDatabase
import asyncio

async def main():
    db = AsyncMemoryDatabase()
    
    # Async operations
    record = Record({"name": "Bob", "score": 95})
    record_id = await db.create(record)
    
    retrieved = await db.read(record_id)
    print(retrieved.score)  # 95

asyncio.run(main())
```

## Use Cases

- **Unit testing** - Fast, isolated tests
- **Development** - No external dependencies
- **Prototyping** - Quick proof of concepts
- **Caching** - Temporary data storage
- **Small applications** - Under 10,000 records

## Limitations

- **Not persistent** - Data lost on restart
- **Memory constraints** - Limited by available RAM
- **Single instance** - No distribution/replication
- **No transactions** - Basic ACID only

## See Also

- [Backends Overview](backends.md)
- [File Backend](file-backend.md) - For persistence
- [Testing Guide](../../development/testing-guide.md)