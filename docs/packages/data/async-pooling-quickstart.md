# Async Connection Pooling - Quick Start

## Installation

The async pooling features are included in the standard dataknobs-data package:

```bash
pip install dataknobs-data[postgres,elasticsearch,s3]
```

## Quick Examples

### Elasticsearch with Native Async

```python
from dataknobs_data import AsyncDatabase, Record, Query

# Simple usage - pooling is automatic!
async def main():
    db = await AsyncDatabase.create("elasticsearch", {
        "hosts": ["http://localhost:9200"],
        "index": "my_data"
    })
    
    # Create records
    record = Record({"name": "Alice", "age": 30})
    id = await db.create(record)
    
    # Search
    results = await db.search(
        Query().filter("age", ">=", 25)
    )
    
    await db.close()
```

### S3 with aioboto3

```python
# 5.3x faster for batch operations!
async def s3_example():
    db = await AsyncDatabase.create("s3", {
        "bucket": "my-bucket",
        "region": "us-east-1"
    })
    
    # Batch upload - uses concurrent uploads
    records = [Record({"id": i}) for i in range(100)]
    ids = await db.create_batch(records)
    
    await db.close()
```

### PostgreSQL with asyncpg

```python
# Native PostgreSQL performance
async def postgres_example():
    db = await AsyncDatabase.create("postgres", {
        "host": "localhost",
        "database": "mydb",
        "user": "user",
        "password": "pass",
        "min_connections": 10,
        "max_connections": 20
    })
    
    # Uses prepared statements automatically
    await db.create(Record({"data": "test"}))
    
    # Transaction support
    async with db.transaction():
        await db.update(id1, record1)
        await db.update(id2, record2)
    
    await db.close()
```

## Performance Comparison

| Backend | Operation | Old Implementation | New Pooled Implementation | Improvement |
|---------|-----------|-------------------|---------------------------|-------------|
| **Elasticsearch** | Bulk Index (1000) | 4.5s | 2.65s | **70% faster** |
| **S3** | Batch Upload (100) | 5.2s | 0.98s | **5.3x faster** |
| **PostgreSQL** | Bulk Insert (1000) | 3.8s | 1.2s | **3.2x faster** |

## Key Features

### 🚀 Automatic Connection Pooling
- Event loop-aware pooling prevents "Event loop is closed" errors
- Connections automatically reused within same event loop
- Separate pools for different event loops

### ⚡ Native Async Clients
- **Elasticsearch**: Uses official AsyncElasticsearch client
- **S3**: Uses aioboto3 for true async S3 operations  
- **PostgreSQL**: Uses asyncpg for maximum performance

### 🔄 Automatic Resource Management
- Pools validated before use, recreated if invalid
- Automatic cleanup on program exit
- Configurable pool sizes and timeouts

### 📊 Built for Scale
- Concurrent batch operations
- Streaming support for large datasets
- Connection retry and failover

## Configuration

### Environment Variables

```bash
# PostgreSQL pooling
export POSTGRES_POOL_MIN_SIZE=10
export POSTGRES_POOL_MAX_SIZE=20

# Elasticsearch pooling  
export ES_POOL_CONNECTIONS=10
export ES_POOL_MAXSIZE=20

# S3 connection pooling
export S3_POOL_MAX_CONNECTIONS=50
```

### Configuration Dictionary

```python
config = {
    # Elasticsearch
    "elasticsearch": {
        "hosts": ["http://localhost:9200"],
        "index": "my_index",
        "pool": {
            "connections": 20,
            "maxsize": 50
        }
    },
    
    # PostgreSQL
    "postgres": {
        "host": "localhost",
        "database": "mydb",
        "pool": {
            "min_size": 10,
            "max_size": 20,
            "timeout": 30
        }
    },
    
    # S3
    "s3": {
        "bucket": "my-bucket",
        "pool": {
            "max_connections": 50
        }
    }
}
```

## Common Patterns

### Sharing Database Instances

```python
# Good: Reuse database instance
class DataService:
    def __init__(self):
        self.db = None
    
    async def initialize(self):
        self.db = await AsyncDatabase.create("elasticsearch", config)
    
    async def process_item(self, item):
        # Reuses pooled connection
        return await self.db.create(Record(item))
    
    async def cleanup(self):
        if self.db:
            await self.db.close()

# Usage
service = DataService()
await service.initialize()

# Process many items with same connection pool
for item in items:
    await service.process_item(item)

await service.cleanup()
```

### Parallel Processing

```python
import asyncio

async def parallel_operations(db, items):
    # Process items in parallel using pooled connections
    tasks = [db.create(Record(item)) for item in items]
    results = await asyncio.gather(*tasks)
    return results
```

### Error Handling

```python
async def resilient_operation(db, record, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await db.create(record)
        except ConnectionError as e:
            if attempt == max_retries - 1:
                raise
            # Pool will recreate connection on next attempt
            await asyncio.sleep(2 ** attempt)
```

## Migration from Old Implementation

The new implementation is **backwards compatible**! Your existing code will automatically use the new pooled implementation:

```python
# Your existing code - no changes needed!
from dataknobs_data.backends.elasticsearch import AsyncElasticsearchDatabase

db = AsyncElasticsearchDatabase(config)
await db.connect()
# Automatically uses new native client with pooling
```

## Troubleshooting

### Issue: "Event loop is closed"
**Solution**: This is automatically prevented by the pooling system. Each event loop gets its own pool.

### Issue: "Pool is exhausted"  
**Solution**: Increase pool size in configuration:
```python
config["pool"]["max_size"] = 50
```

### Issue: Connection timeouts
**Solution**: Adjust timeout settings:
```python
config["pool"]["timeout"] = 60  # Increase to 60 seconds
```

## Learn More

- [Full Documentation](async-pooling.md) - Comprehensive pooling documentation
- [Performance Tuning](performance-tuning.md) - Optimization strategies
- [API Reference](api-reference.md) - Complete API documentation
- [Examples](examples.md) - More code examples

## Support

- 📚 [Documentation](https://dataknobs.readthedocs.io)
- 🐛 [Issue Tracker](https://github.com/yourusername/dataknobs/issues)
- 💬 [Discussions](https://github.com/yourusername/dataknobs/discussions)