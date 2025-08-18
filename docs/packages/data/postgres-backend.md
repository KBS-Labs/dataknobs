# PostgreSQL Backend

## Overview

The PostgreSQL Backend provides production-ready storage with full SQL capabilities, ACID compliance, and excellent performance for large datasets.

## Features

- **ACID compliance** - Full transaction support
- **SQL queries** - Native SQL optimization
- **Connection pooling** - Built-in pool management
- **JSON support** - JSONB for flexible schemas
- **Both sync and async** - Using psycopg2 and asyncpg

## Configuration

```python
from dataknobs_data import PostgresDatabase

config = {
    "host": "localhost",
    "port": 5432,
    "database": "myapp",
    "user": "postgres",
    "password": "secret",
    "pool_size": 10,
    "max_overflow": 20
}

db = PostgresDatabase(config)
```

## Schema Setup

```sql
-- Auto-created table structure
CREATE TABLE IF NOT EXISTS records (
    id UUID PRIMARY KEY,
    fields JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_fields ON records USING GIN (fields);
CREATE INDEX idx_metadata ON records USING GIN (metadata);
```

## Usage Examples

### Connection Pooling

```python
from dataknobs_data import PostgresDatabase
from dataknobs_data.pooling import PostgresConnectionPool

# With connection pooling
pool = PostgresConnectionPool(config)
db = PostgresDatabase(config, pool=pool)

# Automatic connection management
record = Record({"name": "Alice"})
db.create(record)  # Gets connection from pool
```

### Advanced Queries

```python
# Leverage PostgreSQL's JSONB operators
query = Query(filters=[
    Filter("fields->>'name'", Operator.EQ, "Alice"),
    Filter("metadata->>'version'", Operator.GT, "2.0")
])

results = db.search(query)
```

### Transactions

```python
# Transaction support
with db.transaction() as tx:
    tx.create(record1)
    tx.create(record2)
    tx.update(record3.id, updates)
    # Commits on success, rolls back on error
```

## Performance Optimization

- **Use indexes** - Create GIN indexes on JSONB fields
- **Connection pooling** - Reuse connections
- **Batch operations** - Use `create_batch()` for bulk inserts
- **Prepared statements** - Automatic query caching
- **Vacuum regularly** - Maintain database health

## Migration from Other Backends

```python
from dataknobs_data import SyncFileDatabase, PostgresDatabase

# Migrate from file to PostgreSQL
file_db = SyncFileDatabase({"path": "data.json"})
postgres_db = PostgresDatabase(postgres_config)

# Transfer all records
records = file_db.search(Query())
postgres_db.create_batch(records)
```

## Production Considerations

- **Backups** - Regular pg_dump backups
- **Monitoring** - Track connection pool metrics
- **Replication** - Set up read replicas
- **SSL** - Use SSL for connections
- **Upgrades** - Plan PostgreSQL version updates

## See Also

- [Backends Overview](backends.md)
- [Async Pooling](async-pooling.md)
- [Migration Guide](migration.md)