# DuckDB Backend

The DuckDB backend provides a high-performance, embedded analytical database optimized for OLAP (Online Analytical Processing) workloads. It's perfect for applications that need fast analytical queries on large datasets without the overhead of a database server.

## Features

- **High Performance**: 10-100x faster than SQLite for analytical queries
- **Columnar Storage**: Optimized for analytical workloads with efficient column-based storage
- **Zero Configuration**: No server setup required
- **SQL Support**: Full SQL with advanced analytical features (window functions, CTEs, etc.)
- **JSON Support**: Efficient JSON storage and querying via DuckDB's JSON functions
- **In-Memory Option**: Perfect for fast analytics and testing
- **Parallel Query Execution**: Automatic parallelization of queries
- **Compression**: Built-in compression for efficient storage

## Installation

DuckDB support requires the duckdb package:

```bash
pip install duckdb
```

Or install with dataknobs-data:

```bash
pip install dataknobs-data[duckdb]
```

For async support, the duckdb package works with ThreadPoolExecutor (included in Python standard library).

## Quick Start

### Synchronous Usage

```python
from dataknobs_data import DatabaseFactory
from dataknobs_data.records import Record
from dataknobs_data.query import Query, Operator

# Create factory
factory = DatabaseFactory()

# Create and connect to database
db = factory.create(backend="duckdb", path="analytics.duckdb")
db.connect()

# Create a record
record = Record(data={
    "product": "Widget",
    "sales": 150000,
    "region": "West",
    "quarter": "Q1"
})
record_id = db.create(record)

# Read a record
retrieved = db.read(record_id)
print(f"Product: {retrieved['product']}, Sales: {retrieved['sales']}")

# Update a record
retrieved.data["sales"] = 160000
db.update(record_id, retrieved)

# Search records
query = Query().filter("region", Operator.EQ, "West").filter("sales", Operator.GT, 100000)
results = db.search(query)

# Close when done
db.close()
```

### Asynchronous Usage

```python
import asyncio
from dataknobs_data import AsyncDatabaseFactory
from dataknobs_data.records import Record
from dataknobs_data.query import Query, Operator

async def main():
    # Create factory
    factory = AsyncDatabaseFactory()

    # Create and connect to database
    db = factory.create(backend="duckdb", path="analytics.duckdb")
    await db.connect()

    # Create a record
    record = Record(data={
        "product": "Gadget",
        "sales": 250000,
        "region": "East",
        "quarter": "Q2"
    })
    record_id = await db.create(record)

    # Read a record
    retrieved = await db.read(record_id)
    print(f"Product: {retrieved['product']}, Sales: {retrieved['sales']}")

    # Search records
    query = Query().filter("sales", Operator.GT, 200000)
    results = await db.search(query)

    # Close when done
    await db.close()

asyncio.run(main())
```

## Configuration Options

```python
from dataknobs_data import DatabaseFactory

factory = DatabaseFactory()

db = factory.create(
    backend="duckdb",

    # Database file path or ":memory:" for in-memory
    path="/path/to/analytics.duckdb",

    # Table name for records (default: "records")
    table="sales_data",

    # Connection timeout in seconds (default: 5.0)
    timeout=10.0,

    # Open in read-only mode (default: False)
    # Useful for querying production databases safely
    read_only=False
)

# For async - additional configuration
async_db = async_factory.create(
    backend="duckdb",
    path="/path/to/analytics.duckdb",

    # Number of worker threads for async operations (default: 4)
    max_workers=8
)
```

## Advanced Features

### In-Memory Analytics

Perfect for fast analytics on datasets that fit in memory:

```python
from dataknobs_data import DatabaseFactory

factory = DatabaseFactory()

# Create in-memory database for fast analytics
analytics_db = factory.create(backend="duckdb", path=":memory:")
analytics_db.connect()

# Load large dataset
records = [
    Record(data={"product": f"Product{i}", "sales": i * 1000, "quarter": f"Q{(i%4)+1}"})
    for i in range(10000)
]
analytics_db.create_batch(records)

# Fast analytical queries
from dataknobs_data.query import Query, Operator

# Aggregate by quarter
query = Query().filter("quarter", Operator.EQ, "Q1")
q1_records = analytics_db.search(query)
total_sales = sum(r["sales"] for r in q1_records)
print(f"Q1 Total Sales: ${total_sales:,}")

analytics_db.close()
```

### Complex Analytical Queries

DuckDB excels at complex queries with aggregations:

```python
from dataknobs_data.query_logic import ComplexQuery, LogicCondition, LogicOperator, FilterCondition
from dataknobs_data.query import Filter, Operator

# (region = "West" OR region = "East") AND sales > 100000
query = ComplexQuery(
    condition=LogicCondition(
        operator=LogicOperator.AND,
        conditions=[
            LogicCondition(
                operator=LogicOperator.OR,
                conditions=[
                    FilterCondition(Filter("region", Operator.EQ, "West")),
                    FilterCondition(Filter("region", Operator.EQ, "East"))
                ]
            ),
            FilterCondition(Filter("sales", Operator.GT, 100000))
        ]
    )
)

high_value_sales = db.search(query)
print(f"Found {len(high_value_sales)} high-value sales in West/East regions")
```

### Batch Operations

All batch operations are optimized for DuckDB's columnar storage:

```python
# Batch create with optimized inserts
records = [
    Record(data={
        "transaction_id": i,
        "amount": i * 10.5,
        "customer": f"Customer{i % 100}",
        "date": f"2024-01-{(i % 28) + 1:02d}"
    })
    for i in range(100000)
]
ids = db.create_batch(records)
print(f"Inserted {len(ids)} records efficiently")

# Batch update
updates = [
    (ids[i], Record(data={"transaction_id": i, "amount": i * 12.5}))
    for i in range(0, 10000, 100)
]
results = db.update_batch(updates)

# Batch delete
db.delete_batch(ids[:1000])
```

### Read-Only Mode for Safe Querying

Useful for querying production databases without risk of modifications:

```python
# Open production database in read-only mode
readonly_db = factory.create(
    backend="duckdb",
    path="/production/data.duckdb",
    read_only=True
)
readonly_db.connect()

# Can query but cannot modify
query = Query().filter("status", Operator.EQ, "active")
results = readonly_db.search(query)

# Any write operation will raise an exception
try:
    readonly_db.create(Record(data={"test": "data"}))
except Exception as e:
    print(f"Write blocked: {e}")

readonly_db.close()
```

### Streaming Large Datasets

Efficiently process large datasets without loading everything into memory:

```python
from dataknobs_data.streaming import StreamConfig

# Stream read with batching
config = StreamConfig(batch_size=1000)

total_processed = 0
for record in db.stream_read(config=config):
    # Process each record
    total_processed += 1
    if total_processed % 10000 == 0:
        print(f"Processed {total_processed} records...")

print(f"Total processed: {total_processed}")

# Stream write from generator
def data_generator():
    for i in range(50000):
        yield Record(data={"index": i, "value": i * 2})

config = StreamConfig(batch_size=5000)
result = db.stream_write(data_generator(), config)
print(f"Wrote {result.successful} records, {result.failed} failed")
```

## Performance Optimization

### When to Use DuckDB vs SQLite

**Use DuckDB when:**
- Performing analytical queries (aggregations, window functions, complex joins)
- Working with large datasets (millions of rows)
- Need fast read performance on columnar data
- Doing data analysis, reporting, or OLAP workloads
- Performance on aggregations is critical

**Use SQLite when:**
- Need ACID transactions with concurrent writes
- Performing transactional (OLTP) workloads
- Need vector similarity search
- Working with smaller datasets (<100K rows)
- Need maximum compatibility and stability

### Performance Tips

```python
# 1. Use in-memory for temporary analytics
temp_db = factory.create(backend="duckdb", path=":memory:")

# 2. Batch operations for better performance
# Instead of:
for record in records:
    db.create(record)  # Slow

# Do this:
db.create_batch(records)  # Much faster

# 3. Use appropriate batch sizes for streaming
config = StreamConfig(batch_size=10000)  # Larger batches for DuckDB

# 4. Use read-only mode when only querying
readonly = factory.create(backend="duckdb", path="data.duckdb", read_only=True)
```

### Benchmarks

Typical performance characteristics (compared to SQLite):

- **Aggregations**: 10-100x faster
- **Large scans**: 5-20x faster
- **Complex joins**: 10-50x faster
- **Analytical queries**: 20-100x faster
- **Simple inserts**: Similar performance
- **Batch inserts**: 2-5x faster

## Use Cases

### 1. Data Analytics

```python
# Load sales data for analysis
analytics_db = factory.create(backend="duckdb", path=":memory:")
analytics_db.connect()

# Load historical sales data
historical_sales = load_sales_from_csv()  # Your data source
analytics_db.create_batch(historical_sales)

# Perform analysis
query = Query().filter("date", Operator.BETWEEN, ["2024-01-01", "2024-03-31"])
q1_sales = analytics_db.search(query)

total_revenue = sum(sale["amount"] for sale in q1_sales)
print(f"Q1 Revenue: ${total_revenue:,.2f}")
```

### 2. Reporting and Business Intelligence

```python
# Generate quarterly reports
def generate_quarterly_report(db, quarter):
    query = Query().filter("quarter", Operator.EQ, quarter)
    results = db.search(query)

    # Aggregate metrics
    metrics = {
        "total_sales": sum(r["sales"] for r in results),
        "avg_sales": sum(r["sales"] for r in results) / len(results) if results else 0,
        "transaction_count": len(results)
    }

    return metrics

report_db = factory.create(backend="duckdb", path="reports.duckdb")
report_db.connect()

q1_report = generate_quarterly_report(report_db, "Q1")
print(f"Q1 Metrics: {q1_report}")
```

### 3. ETL and Data Transformation

```python
# Extract data from source
source_db = factory.create(backend="postgres", **pg_config)
source_db.connect()

# Load into DuckDB for transformation
etl_db = factory.create(backend="duckdb", path=":memory:")
etl_db.connect()

# Extract
source_data = source_db.search(Query())

# Transform (using DuckDB's fast analytics)
etl_db.create_batch(source_data)

# Perform transformations
query = Query().filter("status", Operator.EQ, "active")
transformed = etl_db.search(query)

# Load to destination
dest_db = factory.create(backend="elasticsearch", **es_config)
dest_db.connect()
dest_db.create_batch(transformed)
```

### 4. Testing with Production Data

```python
import pytest

@pytest.fixture
def analytics_fixture():
    """Create test database with sample analytics data."""
    db = factory.create(backend="duckdb", path=":memory:")
    db.connect()

    # Load test data
    test_data = [
        Record(data={"product": "A", "sales": 1000, "quarter": "Q1"}),
        Record(data={"product": "B", "sales": 2000, "quarter": "Q1"}),
        Record(data={"product": "C", "sales": 1500, "quarter": "Q2"}),
    ]
    db.create_batch(test_data)

    yield db
    db.close()

def test_quarterly_analysis(analytics_fixture):
    """Test quarterly sales analysis."""
    query = Query().filter("quarter", Operator.EQ, "Q1")
    results = analytics_fixture.search(query)

    total = sum(r["sales"] for r in results)
    assert total == 3000
```

## Limitations

- **No Native Async**: Uses ThreadPoolExecutor wrapper (still performant)
- **Single Writer**: One connection can write at a time (reads are parallel)
- **No Built-in Replication**: Not designed for distributed systems
- **Best for Analytics**: Optimized for OLAP, not OLTP workloads
- **No Vector Search**: Does not support vector embeddings (use SQLite or Postgres with pgvector)

## Comparison: DuckDB vs SQLite vs PostgreSQL

| Feature | DuckDB | SQLite | PostgreSQL |
|---------|--------|--------|------------|
| Setup | Zero config | Zero config | Server required |
| Analytical Performance | ⚡ Excellent | 🐌 Moderate | ⚡ Excellent |
| Transactional Performance | ✅ Good | ⚡ Excellent | ⚡ Excellent |
| Columnar Storage | ✅ Yes | ❌ No | ✅ Optional |
| Parallel Queries | ✅ Yes | ❌ No | ✅ Yes |
| Concurrent Writes | 🔄 Limited | 🔄 Limited | ✅ Full |
| Vector Search | ❌ No | ✅ Python-based | ✅ pgvector |
| Best For | Analytics, OLAP | Transactions, OLTP | Production, All |
| File Size | Compressed | Larger | Server |
| Use Case | Data analysis | Embedded apps | Production apps |

## Best Practices

1. **Use in-memory for temporary analytics** - Fast and efficient
2. **Use file-based for persistent analytics** - Store analytical results
3. **Batch operations** - Always prefer batch inserts/updates
4. **Read-only mode for production** - Safe querying without modification risk
5. **Choose DuckDB for analytics** - Use SQLite for transactional workloads
6. **Stream large datasets** - Don't load everything into memory
7. **Monitor file size** - DuckDB has excellent compression but still grows with data
8. **Use appropriate batch sizes** - Larger batches (5000-10000) work well with DuckDB

## Troubleshooting

### Performance Issues

```python
# Ensure you're using batch operations
records = [...]
db.create_batch(records)  # Fast

# Instead of:
for record in records:
    db.create(record)  # Slow
```

### Read-Only Errors

```python
# Make sure read_only is False for write operations
db = factory.create(
    backend="duckdb",
    path="data.duckdb",
    read_only=False  # Allow writes
)
```

### Memory Usage

```python
# Use streaming for large datasets
from dataknobs_data.streaming import StreamConfig

config = StreamConfig(batch_size=1000)
for record in db.stream_read(config=config):
    process(record)  # Process one at a time

# Instead of:
all_records = db.search(Query())  # Loads everything into memory
```

## See Also

- [Backend Comparison](backends.md)
- [SQLite Backend](sqlite-backend.md) - For transactional workloads
- [PostgreSQL Backend](postgres-backend.md) - For production applications
- [Query System](query.md)
- [Performance Tuning](performance-tuning.md)
- [Factory Pattern](factory-pattern.md)
