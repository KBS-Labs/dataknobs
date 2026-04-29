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

The postgres backends (sync and async) accept any input shape
supported by the shared
[Postgres connection config normalizer](../common/postgres-config.md):
a `connection_string`, individual keys as shown above, `DATABASE_URL`,
or `POSTGRES_*` env vars. Explicit config always wins over env vars.

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

## Schema Ownership

By default, the backend creates the records table on every `connect()` via
`CREATE TABLE IF NOT EXISTS …`. This is convenient for local development and
ad-hoc use but inappropriate for deployments where schema is managed by an
external migration tool (Alembic, Flyway, Sqitch, etc.) and the application
database role is DML-only.

Set `auto_create_table: False` to opt out:

```python
db = AsyncPostgresDatabase({
    "connection_string": "postgresql://app:secret@db/myapp",
    "auto_create_table": False,
})
await db.connect()
# → If the table does not exist, raises:
#   RuntimeError: Table public.records does not exist and
#   auto_create_table is disabled. Run your migrations before starting
#   the application.
```

When `auto_create_table` is `False`:

- `connect()` runs a single `SELECT EXISTS …` query to verify the table is present.
- If the table is missing, `connect()` raises `RuntimeError` with a clear message —
  the application fails fast instead of running against an empty schema.
- If the table is present, `connect()` is a no-op for DDL — no `CREATE TABLE`,
  no index DDL, no privileged operations.

The default is `True`, preserving backward compatibility with all existing consumers.

This is the same contract as `PgVectorStore.auto_create_table` — see
[pgvector backend](pgvector-backend.md) for the parallel pattern in the
vector-store layer. Together the two flags let Alembic (or any external tool)
own both the records table and the embeddings table.

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

## Identifier Quoting

Schema and table names supplied via configuration are internally quoted using
`quote_ident()` from `dataknobs_utils.sql_utils`. Any valid SQL identifier is
accepted — including mixed-case names, reserved words, and names with spaces
— without the consumer needing to pre-quote them. Existing consumers using
simple `[a-z_][a-z0-9_]*` names see no behavior change (quoting is idempotent
under correct SQL parsing).

## See Also

- [Backends Overview](backends.md)
- [Async Pooling](async-pooling.md)
- [Migration Guide](migration.md)