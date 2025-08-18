# File Backend

## Overview

The File Backend provides JSON-based file storage with full query capabilities. It's ideal for small to medium datasets that need persistence without a database server.

## Features

- **JSON storage** - Human-readable format
- **Automatic persistence** - Writes to disk
- **Full query support** - All operators supported
- **Atomic operations** - File locking for safety
- **Both sync and async** - `SyncFileDatabase` and `AsyncFileDatabase`

## Configuration

```python
from dataknobs_data import SyncFileDatabase

# Configure with file path
config = {
    "path": "/path/to/data.json",
    "auto_save": True,  # Save after each write
    "pretty": True      # Pretty-print JSON
}

db = SyncFileDatabase(config)
```

## Usage

### Basic Operations

```python
from dataknobs_data import SyncFileDatabase, Record

# Setup database
db = SyncFileDatabase({"path": "data.json"})

# Store records
record = Record({
    "id": "user_001",
    "name": "Alice",
    "email": "alice@example.com"
})
db.create(record)

# Data is automatically saved to file
```

### Batch Operations

```python
# Batch insert for better performance
records = [
    Record({"name": f"User {i}", "score": i * 10})
    for i in range(100)
]

ids = db.create_batch(records)
print(f"Created {len(ids)} records")
```

## File Format

The JSON file structure:

```json
{
  "version": "1.0",
  "records": {
    "uuid-1": {
      "id": "uuid-1",
      "fields": {
        "name": {"type": "STRING", "value": "Alice"}
      },
      "metadata": {"created": "2024-01-01T00:00:00"}
    }
  },
  "indexes": {
    "name": ["uuid-1", "uuid-2"]
  }
}
```

## Performance Considerations

- **File size** - Performance degrades > 100MB
- **Query speed** - O(n) for most operations
- **Concurrent access** - File locking adds overhead
- **Batch operations** - Use batches for bulk inserts

## Use Cases

- **Configuration storage** - Application settings
- **Small datasets** - Under 10,000 records
- **Data export/import** - JSON interchange
- **Offline applications** - No server required
- **Development** - Easy debugging

## See Also

- [Backends Overview](backends.md)
- [Memory Backend](memory-backend.md) - For testing
- [PostgreSQL Backend](postgres-backend.md) - For production