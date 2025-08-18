# SQLite Backend

The SQLite backend provides a lightweight, embedded SQL database with full ACID compliance and zero configuration. It's perfect for applications that need SQL capabilities without the overhead of a database server.

## Features

- **Zero Configuration**: No server setup required
- **ACID Compliance**: Full transaction support
- **JSON Support**: Native JSON functions for document storage
- **Small Footprint**: ~1MB library size
- **Cross-Platform**: Works on all major operating systems
- **In-Memory Option**: Perfect for testing
- **WAL Mode**: Improved concurrency for multi-reader scenarios

## Installation

SQLite support is included in the base installation:

```bash
pip install dataknobs-data
```

For async support, you'll also need aiosqlite:

```bash
pip install aiosqlite
```

## Quick Start

### Synchronous Usage

```python
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.records import Record
from dataknobs_data.query import Query, Operator

# Create and connect to database
db = SyncSQLiteDatabase({"path": "app.db"})
db.connect()

# Create a record
record = Record(data={
    "name": "Alice",
    "age": 30,
    "city": "New York"
})
record_id = db.create(record)

# Read a record
retrieved = db.read(record_id)
print(f"Name: {retrieved['name']}")

# Update a record
retrieved.data["age"] = 31
db.update(record_id, retrieved)

# Search records
query = Query().filter("city", Operator.EQ, "New York")
results = db.search(query)

# Close when done
db.close()
```

### Asynchronous Usage

```python
import asyncio
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.records import Record
from dataknobs_data.query import Query, Operator

async def main():
    # Create and connect to database
    db = AsyncSQLiteDatabase({"path": "app.db"})
    await db.connect()
    
    # Create a record
    record = Record(data={
        "name": "Bob",
        "age": 25,
        "city": "San Francisco"
    })
    record_id = await db.create(record)
    
    # Read a record
    retrieved = await db.read(record_id)
    print(f"Name: {retrieved['name']}")
    
    # Search records
    query = Query().filter("age", Operator.GT, 20)
    results = await db.search(query)
    
    # Close when done
    await db.close()

asyncio.run(main())
```

## Configuration Options

```python
db = SyncSQLiteDatabase({
    # Database file path or ":memory:" for in-memory
    "path": "/path/to/database.db",
    
    # Table name for records (default: "records")
    "table": "my_records",
    
    # Connection timeout in seconds (default: 5.0)
    "timeout": 10.0,
    
    # Journal mode for concurrency (default: "WAL" for file-based)
    # Options: WAL, DELETE, TRUNCATE, PERSIST, MEMORY, OFF
    "journal_mode": "WAL",
    
    # Synchronous mode for durability (default: "NORMAL")
    # Options: FULL (safest), NORMAL (balanced), OFF (fastest)
    "synchronous": "NORMAL",
    
    # For async only - connection pool size (default: 5)
    "pool_size": 10
})
```

## Advanced Features

### Complex Queries

SQLite backend supports native SQL generation for complex boolean queries:

```python
from dataknobs_data.query_logic import ComplexQuery, LogicCondition, LogicOperator, FilterCondition
from dataknobs_data.query import Filter, Operator

# (city = "NYC" OR city = "LA") AND age > 25
query = ComplexQuery(
    condition=LogicCondition(
        operator=LogicOperator.AND,
        conditions=[
            LogicCondition(
                operator=LogicOperator.OR,
                conditions=[
                    FilterCondition(Filter("city", Operator.EQ, "NYC")),
                    FilterCondition(Filter("city", Operator.EQ, "LA"))
                ]
            ),
            FilterCondition(Filter("age", Operator.GT, 25))
        ]
    )
)

results = db.search(query)
```

### Batch Operations

All batch operations are wrapped in transactions for atomicity:

```python
# Batch create with transaction
records = [
    Record(data={"name": f"User{i}", "score": i * 10})
    for i in range(100)
]
ids = db.create_batch(records)

# Batch update
updates = [
    (ids[i], Record(data={"name": f"User{i}", "score": i * 20}))
    for i in range(50)
]
results = db.update_batch(updates)

# Batch delete
db.delete_batch(ids[:25])
```

### In-Memory Databases

Perfect for testing and temporary data:

```python
# Create in-memory database
test_db = SyncSQLiteDatabase({"path": ":memory:"})
test_db.connect()

# Use it like any other database
test_db.create(Record(data={"test": "data"}))

# Data is lost when connection closes
test_db.close()
```

### WAL Mode for Better Concurrency

Write-Ahead Logging (WAL) mode allows multiple readers with one writer:

```python
db = SyncSQLiteDatabase({
    "path": "app.db",
    "journal_mode": "WAL"
})
db.connect()

# Now multiple processes can read while one writes
```

## Performance Optimization

### PRAGMA Settings

```python
# Optimize for speed (less safe)
fast_db = SyncSQLiteDatabase({
    "path": "fast.db",
    "synchronous": "OFF",  # Don't wait for disk writes
    "journal_mode": "MEMORY"  # Keep journal in memory
})

# Optimize for safety (slower)
safe_db = SyncSQLiteDatabase({
    "path": "safe.db",
    "synchronous": "FULL",  # Wait for all disk writes
    "journal_mode": "DELETE"  # Traditional journaling
})

# Balanced (recommended)
balanced_db = SyncSQLiteDatabase({
    "path": "app.db",
    "synchronous": "NORMAL",
    "journal_mode": "WAL"
})
```

### Transaction Batching

```python
# Manual transaction control for maximum performance
db.conn.execute("BEGIN TRANSACTION")
try:
    for i in range(10000):
        record = Record(data={"id": i, "value": i * 2})
        db.create(record)
    db.conn.commit()
except Exception:
    db.conn.rollback()
    raise
```

## Use Cases

### 1. Desktop Applications

```python
# Desktop app with local data storage
app_db = SyncSQLiteDatabase({
    "path": "~/MyApp/data.db",
    "journal_mode": "WAL"
})
```

### 2. Testing

```python
import pytest

@pytest.fixture
def test_db():
    db = SyncSQLiteDatabase({"path": ":memory:"})
    db.connect()
    yield db
    db.close()

def test_user_creation(test_db):
    user = Record(data={"username": "testuser"})
    user_id = test_db.create(user)
    assert test_db.exists(user_id)
```

### 3. Prototyping

```python
# Quick prototype with persistence
prototype_db = SyncSQLiteDatabase({
    "path": "prototype.db"
})
prototype_db.connect()

# Iterate quickly with full SQL support
```

### 4. Data Migration

```python
# Use SQLite as intermediate format
from dataknobs_data.factory import create_database

# Import from CSV
csv_db = create_database("file", path="data.csv", format="csv")
sqlite_db = create_database("sqlite", path="migrated.db")

# Transfer data
for record in csv_db.search(Query()):
    sqlite_db.create(record)

# Later export to PostgreSQL
pg_db = create_database("postgres", **pg_config)
for record in sqlite_db.search(Query()):
    pg_db.create(record)
```

## Limitations

- **Concurrency**: Single writer, multiple readers
- **Scale**: Best for databases under 1TB
- **Network**: No built-in network access
- **Replication**: No native replication support

## Comparison with PostgreSQL

| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| Setup | Zero config | Server required |
| Concurrency | Limited | Full |
| Scale | Single machine | Distributed |
| Performance | Fast for small-medium | Fast for all sizes |
| Features | Basic SQL + JSON | Full SQL + Extensions |
| Use Case | Embedded/Desktop | Server applications |

## Best Practices

1. **Always use WAL mode** for better concurrency
2. **Use transactions** for batch operations
3. **Close connections** properly to avoid locks
4. **Use in-memory** for testing
5. **Consider file location** - use SSD for best performance
6. **Regular VACUUM** for long-running databases
7. **Monitor file size** - SQLite slows down with very large files

## Troubleshooting

### Database Locked Error

```python
# Increase timeout
db = SyncSQLiteDatabase({
    "path": "app.db",
    "timeout": 30.0  # Wait up to 30 seconds
})
```

### Slow Performance

```python
# Enable optimizations
db = SyncSQLiteDatabase({
    "path": "app.db",
    "journal_mode": "WAL",
    "synchronous": "NORMAL"
})
db.connect()

# Add indexes if needed
db.conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_data_name 
    ON records(json_extract(data, '$.name'))
""")
```

### Corruption Recovery

```python
import sqlite3
import shutil

# Backup before recovery
shutil.copy("app.db", "app.db.backup")

# Try to recover
conn = sqlite3.connect("app.db")
conn.execute("PRAGMA integrity_check")
conn.execute("VACUUM")
conn.close()
```

## See Also

- [Backend Comparison](backends.md)
- [Query System](query.md)
- [Migration Guide](migration.md)
- [Factory Pattern](factory-pattern.md)