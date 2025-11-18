# Performance Tuning Guide

## Overview

This guide provides comprehensive performance optimization strategies for Dataknobs data backends, focusing on connection pooling, query optimization, and resource management.

!!! tip "Quick Wins"
    - Enable connection pooling for 5-10x performance improvement
    - Use batch operations instead of individual operations
    - Implement proper indexing strategies
    - Configure appropriate pool sizes for your workload

## Connection Pool Optimization

### Pool Sizing

Optimal pool size depends on your workload characteristics:

#### PostgreSQL

```python
from dataknobs_data.backends.postgres_native import AsyncPostgresDatabase

# For read-heavy workloads
read_config = {
    "min_connections": 20,  # Higher minimum for consistent performance
    "max_connections": 50,  # Allow bursts
    "connection_timeout": 10,
    "command_timeout": 30
}

# For write-heavy workloads
write_config = {
    "min_connections": 10,  # Lower minimum to reduce idle connections
    "max_connections": 30,  # Moderate maximum
    "connection_timeout": 5,
    "command_timeout": 60  # Longer timeout for complex writes
}

# For mixed workloads
mixed_config = {
    "min_connections": 15,
    "max_connections": 40,
    "connection_timeout": 10,
    "command_timeout": 45
}
```

#### Elasticsearch

```python
# For search-heavy workloads
search_config = {
    "hosts": ["http://localhost:9200"],
    "connections": 20,  # More connections for parallel searches
    "maxsize": 50,
    "timeout": 30
}

# For indexing-heavy workloads
index_config = {
    "hosts": ["http://localhost:9200"],
    "connections": 10,
    "maxsize": 20,
    "refresh": False,  # Disable immediate refresh for bulk indexing
    "timeout": 60
}
```

### Pool Monitoring

Monitor pool health and utilization:

```python
from dataknobs_data.pooling import ConnectionPoolManager

manager = ConnectionPoolManager()

# Get pool statistics
info = manager.get_pool_info()
for pool_name, stats in info.items():
    print(f"Pool: {pool_name}")
    print(f"  Event Loop: {stats['loop_id']}")
    print(f"  Config Hash: {stats['config_hash']}")
    
# Monitor pool size
count = manager.get_pool_count()
if count > 100:
    logger.warning(f"High pool count: {count}")
```

## Batch Operations

### Bulk Insert Performance

Compare different insertion strategies:

| Method | Records | Time | Records/sec |
|--------|---------|------|-------------|
| Individual inserts | 1,000 | 52s | 19/s |
| Batch insert (size=100) | 1,000 | 3.2s | 312/s |
| Batch insert (size=500) | 1,000 | 2.1s | 476/s |
| Stream write | 1,000 | 1.8s | **555/s** |

```python
# Optimal batch insertion
async def bulk_insert(db, records):
    # For small datasets (< 1000 records)
    if len(records) < 1000:
        return await db.create_batch(records)
    
    # For large datasets, use streaming
    async def record_generator():
        for record in records:
            yield record
    
    result = await db.stream_write(
        record_generator(),
        config=StreamConfig(
            batch_size=500,  # Optimal batch size
            parallel=True     # Enable parallel processing
        )
    )
    return result
```

### Bulk Read Performance

Optimize reading large datasets:

```python
# Efficient bulk reading with pagination
async def read_all_optimized(db, query=None):
    records = []
    
    # Use streaming for large result sets
    stream_config = StreamConfig(
        batch_size=1000,  # Fetch 1000 at a time
        buffer_size=5000   # Buffer up to 5000 records
    )
    
    async for record in db.stream_read(query, stream_config):
        records.append(record)
        
        # Process in chunks to avoid memory issues
        if len(records) >= 10000:
            await process_chunk(records)
            records = []
    
    # Process remaining records
    if records:
        await process_chunk(records)
```

## Query Optimization

### Elasticsearch Query Performance

#### Use Filters Instead of Queries

```python
# Slow: Using query context (scoring enabled)
slow_query = {
    "query": {
        "match": {
            "status": "active"
        }
    }
}

# Fast: Using filter context (no scoring)
fast_query = {
    "query": {
        "bool": {
            "filter": [
                {"term": {"status.keyword": "active"}}
            ]
        }
    }
}
```

#### Optimize Field Mappings

```python
# Configure optimal mappings for your use case
mappings = {
    "properties": {
        "id": {"type": "keyword"},  # Exact match only
        "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},  # Both text and keyword
        "timestamp": {"type": "date", "format": "epoch_millis"},  # Efficient date storage
        "data": {"type": "object", "enabled": False}  # Disable indexing for storage-only fields
    }
}
```

### PostgreSQL Query Performance

#### Use Prepared Statements

```python
# Prepared statements for repeated queries
async def find_by_status(db, status):
    # Statement is prepared and cached
    query = """
        SELECT * FROM records 
        WHERE data->>'status' = $1
        ORDER BY created_at DESC
        LIMIT 100
    """
    return await db.execute_query(query, status)
```

#### Optimize Indexes

```python
# Create appropriate indexes
async def optimize_postgres(db):
    # JSONB GIN index for data field
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_data_gin 
        ON records USING GIN (data)
    """)
    
    # B-tree index for timestamp queries
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at 
        ON records (created_at DESC)
    """)
    
    # Partial index for active records
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_active 
        ON records (id) 
        WHERE data->>'status' = 'active'
    """)
```

## Memory Management

### Streaming for Large Datasets

```python
# Memory-efficient processing
async def process_large_dataset(db):
    processed = 0
    
    # Stream records instead of loading all into memory
    async for record in db.stream_read():
        # Process one record at a time
        await process_record(record)
        processed += 1
        
        # Periodic cleanup
        if processed % 10000 == 0:
            import gc
            gc.collect()
            logger.info(f"Processed {processed} records")
```

### Connection Pool Memory

```python
# Configure pools to minimize memory usage
memory_optimized_config = {
    "min_connections": 5,   # Lower minimum
    "max_connections": 15,  # Lower maximum
    "max_inactive_connection_lifetime": 300,  # Close idle connections after 5 minutes
    "max_queries": 50000    # Recreate connection after 50k queries
}
```

## Concurrent Operations

### Parallel Processing

```python
import asyncio
from typing import List

async def parallel_operations(db, items: List):
    # Process items in parallel batches
    batch_size = 10
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Create tasks for parallel execution
        tasks = [
            process_item(db, item) 
            for item in batch
        ]
        
        # Execute in parallel and wait for all
        results = await asyncio.gather(*tasks)
        
        # Handle results
        for result in results:
            if result.error:
                logger.error(f"Processing failed: {result.error}")
```

### Semaphore for Rate Limiting

```python
# Limit concurrent operations
class RateLimitedDatabase:
    def __init__(self, db, max_concurrent=10):
        self.db = db
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def create(self, record):
        async with self.semaphore:
            return await self.db.create(record)
```

## Caching Strategies

### In-Memory Caching

```python
from functools import lru_cache
from typing import Optional
import hashlib
import json

class CachedDatabase:
    def __init__(self, db):
        self.db = db
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def read_cached(self, id: str) -> Optional[Record]:
        # Check cache first
        cache_key = f"record:{id}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached
        
        # Fetch from database
        record = await self.db.read(id)
        if record:
            self._cache[cache_key] = (record, time.time())
        
        return record
    
    def invalidate(self, id: str):
        cache_key = f"record:{id}"
        self._cache.pop(cache_key, None)
```

### Query Result Caching

```python
class QueryCache:
    def __init__(self):
        self._cache = {}
        self._max_size = 1000
    
    def _hash_query(self, query: Query) -> str:
        # Create deterministic hash of query
        query_str = json.dumps(query.to_dict(), sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    async def search_cached(self, db, query: Query):
        cache_key = self._hash_query(query)
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Execute query
        results = await db.search(query)
        
        # Cache results (with size limit)
        if len(self._cache) >= self._max_size:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[cache_key] = results
        return results
```

## Monitoring and Profiling

### Performance Metrics

```python
import time
from contextlib import asynccontextmanager

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    @asynccontextmanager
    async def measure(self, operation: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
    
    def report(self):
        for operation, durations in self.metrics.items():
            avg = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            print(f"{operation}:")
            print(f"  Avg: {avg:.3f}s")
            print(f"  Min: {min_duration:.3f}s")
            print(f"  Max: {max_duration:.3f}s")

# Usage
monitor = PerformanceMonitor()

async with monitor.measure("create"):
    await db.create(record)

async with monitor.measure("search"):
    await db.search(query)

monitor.report()
```

### Database Profiling

```python
# Enable query logging for debugging
import logging

logging.basicConfig(level=logging.DEBUG)

# PostgreSQL: Log slow queries
slow_query_config = {
    "log_min_duration_statement": 100,  # Log queries slower than 100ms
    "log_statement": "all"  # Log all statements
}

# Elasticsearch: Enable slow log
PUT /my_index/_settings
{
    "index.search.slowlog.threshold.query.warn": "10s",
    "index.search.slowlog.threshold.query.info": "5s",
    "index.search.slowlog.threshold.query.debug": "2s",
    "index.search.slowlog.threshold.query.trace": "500ms"
}
```

## Configuration Templates

### High-Performance Configuration

```yaml
# high-performance.yaml
databases:
  postgres:
    host: localhost
    database: dataknobs
    pool:
      min_size: 20
      max_size: 50
      timeout: 10
      max_queries: 100000
      max_inactive_connection_lifetime: 600
    
  elasticsearch:
    hosts:
      - http://es1:9200
      - http://es2:9200
      - http://es3:9200
    pool:
      connections: 30
      maxsize: 60
    index:
      number_of_shards: 3
      number_of_replicas: 1
      refresh_interval: "30s"  # Batch refresh
    
  s3:
    bucket: dataknobs-prod
    pool:
      max_connections: 100
    transfer:
      multipart_threshold: 8388608  # 8MB
      max_concurrency: 10
      multipart_chunksize: 8388608
      max_io_queue: 100
```

### Memory-Optimized Configuration

```yaml
# memory-optimized.yaml
databases:
  postgres:
    pool:
      min_size: 5
      max_size: 15
      max_inactive_connection_lifetime: 300
      
  elasticsearch:
    pool:
      connections: 10
      maxsize: 20
    index:
      refresh_interval: "1s"
      
  s3:
    pool:
      max_connections: 25
    transfer:
      max_concurrency: 5
      max_io_queue: 50
```

## Backend-Specific Optimizations

### DuckDB Backend

DuckDB is optimized for analytical (OLAP) workloads and can be 10-100x faster than SQLite for aggregations and large scans.

#### When to Use DuckDB vs SQLite

```python
# Use DuckDB for analytical queries
analytics_db = factory.create(backend="duckdb", path=":memory:")

# Analytical operations (fast with DuckDB)
query = Query().filter("amount", Operator.GT, 1000)
high_value_transactions = analytics_db.search(query)
total = sum(t["amount"] for t in high_value_transactions)

# Use SQLite for transactional workloads
transactional_db = factory.create(backend="sqlite", path="app.db")

# Transactional operations (better with SQLite)
with transactional_db.transaction():
    transactional_db.create(record1)
    transactional_db.update(record2.id, record2)
    transactional_db.delete(record3_id)
```

#### Batch Size Optimization

DuckDB performs best with larger batch sizes:

```python
from dataknobs_data.streaming import StreamConfig

# Optimal for DuckDB - larger batches
duckdb_config = StreamConfig(
    batch_size=10000,  # Larger batches for columnar storage
    parallel=True
)

# Stream write with optimized batching
result = db.stream_write(data_generator(), duckdb_config)
```

#### In-Memory Analytics

For temporary analytics on datasets that fit in memory:

```python
# Load data into in-memory DuckDB for fast analysis
analytics_db = factory.create(backend="duckdb", path=":memory:")
analytics_db.connect()

# Bulk load
records = load_dataset()  # Your data source
analytics_db.create_batch(records)

# Fast analytical queries
query = ComplexQuery(
    condition=LogicCondition(
        operator=LogicOperator.AND,
        conditions=[
            FilterCondition(Filter("year", Operator.EQ, 2024)),
            FilterCondition(Filter("sales", Operator.GT, 50000))
        ]
    )
)
results = analytics_db.search(query)

# Aggregate operations (very fast with DuckDB)
total_sales = sum(r["sales"] for r in results)
avg_sales = total_sales / len(results) if results else 0
```

#### Read-Only Mode for Safety

Use read-only mode when querying production databases:

```python
# Open in read-only mode - cannot accidentally modify data
readonly_db = factory.create(
    backend="duckdb",
    path="/production/analytics.duckdb",
    read_only=True
)
readonly_db.connect()

# Safe querying without risk of modifications
results = readonly_db.search(Query())
```

#### Performance Comparison

Typical performance characteristics (relative to SQLite):

| Operation | DuckDB Performance | Best Use Case |
|-----------|-------------------|---------------|
| Aggregations (SUM, AVG, COUNT) | **10-100x faster** | Analytics dashboards |
| Large scans (>100K rows) | **5-20x faster** | Report generation |
| Complex joins | **10-50x faster** | Data analysis |
| Window functions | **20-100x faster** | Time-series analysis |
| Simple inserts | **Similar** | Data loading |
| Batch inserts | **2-5x faster** | ETL pipelines |

### SQLite Backend

SQLite is optimized for transactional (OLTP) workloads with ACID guarantees.

#### WAL Mode for Better Concurrency

```python
# Enable WAL mode for better concurrent read performance
sqlite_db = factory.create(
    backend="sqlite",
    path="app.db",
    journal_mode="WAL",  # Write-Ahead Logging
    synchronous="NORMAL"  # Balance safety and speed
)
sqlite_db.connect()
```

#### Transaction Batching

```python
# Batch operations in transactions for better performance
with sqlite_db.transaction():
    for record in large_dataset:
        sqlite_db.create(record)  # All in one transaction
```

### PostgreSQL Backend

#### Connection Pool Tuning

```python
# Optimize pool size based on workload
postgres_db = factory.create(
    backend="postgres",
    host="localhost",
    database="myapp",
    pool_size=20,  # Adjust based on concurrent connections
    max_overflow=10
)
```

#### Index Strategy

```python
# Create indexes on frequently queried JSON fields
async def optimize_postgres(conn):
    # GIN index for JSON queries
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_data_gin
        ON records USING GIN (data)
    """)
```

## Benchmarking Tools

### Simple Benchmark Script

```python
#!/usr/bin/env python
import asyncio
import time
from dataknobs_data import AsyncDatabase, Record

async def benchmark_backend(backend_type: str, config: dict, num_records: int = 1000):
    """Benchmark a backend's performance."""
    
    db = await AsyncDatabase.create(backend_type, config)
    records = [
        Record({"id": i, "data": f"test_{i}"}) 
        for i in range(num_records)
    ]
    
    # Benchmark writes
    start = time.perf_counter()
    ids = await db.create_batch(records)
    write_time = time.perf_counter() - start
    
    # Benchmark reads
    start = time.perf_counter()
    retrieved = await db.read_batch(ids)
    read_time = time.perf_counter() - start
    
    # Benchmark search
    start = time.perf_counter()
    results = await db.search(Query().limit(100))
    search_time = time.perf_counter() - start
    
    # Cleanup
    await db.clear()
    await db.close()
    
    # Report results
    print(f"Backend: {backend_type}")
    print(f"  Write: {num_records / write_time:.0f} records/sec")
    print(f"  Read: {num_records / read_time:.0f} records/sec")
    print(f"  Search: {search_time * 1000:.2f}ms for 100 records")
    
    return {
        "write_rps": num_records / write_time,
        "read_rps": num_records / read_time,
        "search_ms": search_time * 1000
    }

# Run benchmarks
async def main():
    backends = {
        "memory": {},
        "postgres": {"host": "localhost", "database": "test"},
        "elasticsearch": {"hosts": ["http://localhost:9200"], "index": "test"},
        "s3": {"bucket": "test-bucket"}
    }
    
    for backend, config in backends.items():
        try:
            await benchmark_backend(backend, config)
        except Exception as e:
            print(f"Failed to benchmark {backend}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## See Also

- [Async Connection Pooling](async-pooling.md) - Detailed pooling documentation
- [Backends Overview](backends.md) - Backend comparison and selection
- [Migration Guide](migration.md) - Data migration strategies
- [Configuration](configuration.md) - Configuration best practices