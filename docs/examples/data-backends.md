# Data Backends Example

This example demonstrates how to use different database backends with the dataknobs-data package.

## Complete Example

```python
#!/usr/bin/env python3
"""
Example showing how to use different database backends.
"""

from dataknobs_data import Record, Query, DatabaseFactory
from dataknobs_config import Config
import os
import tempfile

def demonstrate_memory_backend():
    """Demonstrate in-memory backend for caching."""
    print("\n=== Memory Backend (Caching) ===")
    
    factory = DatabaseFactory()
    cache = factory.create(backend="memory")
    
    # Store frequently accessed data
    hot_data = Record({
        "id": "hot-001",
        "type": "cache",
        "data": "frequently accessed",
        "hits": 0
    })
    
    cache_id = cache.create(hot_data)
    print(f"Cached data with ID: {cache_id}")
    
    # Simulate cache hits
    for _ in range(5):
        data = cache.read(cache_id)
        if data:
            data.fields["hits"] += 1
            cache.update(cache_id, data)
    
    final = cache.read(cache_id)
    print(f"Cache hits: {final.get_value('hits')}")
    
    return cache

def demonstrate_file_backend():
    """Demonstrate file backend for persistence."""
    print("\n=== File Backend (JSON/CSV/Parquet) ===")
    
    factory = DatabaseFactory()
    
    # JSON format
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = f.name
    
    json_db = factory.create(
        backend="file",
        path=json_path,
        format="json"
    )
    
    # Store structured data
    records = [
        Record({"name": "Alice", "age": 30, "city": "New York"}),
        Record({"name": "Bob", "age": 25, "city": "San Francisco"}),
        Record({"name": "Charlie", "age": 35, "city": "New York"})
    ]
    
    for record in records:
        json_db.create(record)
    
    # Query data
    ny_residents = json_db.search(
        Query().filter("city", "=", "New York")
    )
    print(f"New York residents: {len(ny_residents)}")
    
    # CSV format for tabular data
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    
    csv_db = factory.create(
        backend="file",
        path=csv_path,
        format="csv"
    )
    
    # Note: CSV backend works best with flat, tabular data
    csv_db.create(Record({"id": "1", "value": "100", "status": "active"}))
    csv_db.create(Record({"id": "2", "value": "200", "status": "inactive"}))
    
    print(f"CSV records: {csv_db.count()}")
    
    # Clean up
    os.unlink(json_path)
    os.unlink(csv_path)

    return json_db

def demonstrate_duckdb_backend():
    """Demonstrate DuckDB backend for fast analytics."""
    print("\n=== DuckDB Backend (Analytics) ===")

    factory = DatabaseFactory()

    # Create in-memory DuckDB for fast analytics
    duck_db = factory.create(backend="duckdb", path=":memory:")
    duck_db.connect()

    # Load analytical dataset
    sales_data = [
        Record({"product": "Widget", "sales": 15000, "region": "West", "quarter": "Q1"}),
        Record({"product": "Gadget", "sales": 25000, "region": "East", "quarter": "Q1"}),
        Record({"product": "Widget", "sales": 18000, "region": "West", "quarter": "Q2"}),
        Record({"product": "Gadget", "sales": 30000, "region": "East", "quarter": "Q2"}),
        Record({"product": "Doohickey", "sales": 12000, "region": "South", "quarter": "Q1"}),
        Record({"product": "Doohickey", "sales": 16000, "region": "South", "quarter": "Q2"}),
    ]

    # Batch load for optimal performance
    ids = duck_db.create_batch(sales_data)
    print(f"Loaded {len(ids)} sales records")

    # Fast analytical queries (much faster than SQLite)
    from dataknobs_data.query import Query, Operator

    # Aggregate by region
    west_sales = duck_db.search(Query().filter("region", Operator.EQ, "West"))
    west_total = sum(r["sales"] for r in west_sales)
    print(f"West region total sales: ${west_total:,}")

    # High-value products
    high_value = duck_db.search(Query().filter("sales", Operator.GT, 20000))
    print(f"High-value products (>$20K): {len(high_value)}")

    # Complex analytical query
    q2_analysis = duck_db.search(Query().filter("quarter", Operator.EQ, "Q2"))
    q2_total = sum(r["sales"] for r in q2_analysis)
    q2_avg = q2_total / len(q2_analysis) if q2_analysis else 0
    print(f"Q2 Analysis: Total=${q2_total:,}, Avg=${q2_avg:,.2f}")

    duck_db.close()
    return duck_db

def demonstrate_postgres_backend():
    """Demonstrate PostgreSQL backend (requires running instance)."""
    print("\n=== PostgreSQL Backend (Production) ===")
    
    # Check if PostgreSQL is available
    factory = DatabaseFactory()
    
    if not factory.is_backend_available("postgres"):
        print("PostgreSQL backend not available")
        print("Install with: pip install dataknobs-data[postgres]")
        return None
    
    try:
        # Create PostgreSQL database
        pg_db = factory.create(
            backend="postgres",
            host=os.environ.get("PG_HOST", "localhost"),
            port=int(os.environ.get("PG_PORT", 5432)),
            database=os.environ.get("PG_DATABASE", "test"),
            user=os.environ.get("PG_USER", "postgres"),
            password=os.environ.get("PG_PASSWORD", "postgres"),
            table="demo_records"
        )
        
        # Create some records
        user = Record({
            "username": "john_doe",
            "email": "john@example.com",
            "role": "admin",
            "active": True
        })
        
        user_id = pg_db.create(user)
        print(f"Created user in PostgreSQL: {user_id}")
        
        # Complex query
        admins = pg_db.search(
            Query()
            .filter("role", "=", "admin")
            .filter("active", "=", True)
            .sort("username", "ASC")
        )
        print(f"Active admins: {len(admins)}")
        
        return pg_db
        
    except Exception as e:
        print(f"Could not connect to PostgreSQL: {e}")
        return None

def demonstrate_elasticsearch_backend():
    """Demonstrate Elasticsearch backend (requires running instance)."""
    print("\n=== Elasticsearch Backend (Search) ===")
    
    factory = DatabaseFactory()
    
    if not factory.is_backend_available("elasticsearch"):
        print("Elasticsearch backend not available")
        print("Install with: pip install dataknobs-data[elasticsearch]")
        return None
    
    try:
        # Create Elasticsearch database
        es_db = factory.create(
            backend="elasticsearch",
            hosts=[os.environ.get("ES_HOST", "localhost:9200")],
            index="demo_index",
            username=os.environ.get("ES_USER"),
            password=os.environ.get("ES_PASSWORD")
        )
        
        # Index documents
        documents = [
            Record({
                "title": "Introduction to Python",
                "content": "Python is a versatile programming language...",
                "tags": ["python", "programming", "tutorial"],
                "views": 1000
            }),
            Record({
                "title": "Advanced Python Techniques",
                "content": "Learn advanced Python programming patterns...",
                "tags": ["python", "advanced", "patterns"],
                "views": 500
            }),
            Record({
                "title": "Web Development with Django",
                "content": "Build web applications using Django framework...",
                "tags": ["python", "django", "web"],
                "views": 750
            })
        ]
        
        for doc in documents:
            es_db.create(doc)
        
        # Full-text search
        results = es_db.search(
            Query().filter("content", "LIKE", "%Python%")
        )
        print(f"Documents mentioning Python: {len(results)}")
        
        # Range query
        popular = es_db.search(
            Query().filter("views", ">", 600)
        )
        print(f"Popular documents (>600 views): {len(popular)}")
        
        return es_db
        
    except Exception as e:
        print(f"Could not connect to Elasticsearch: {e}")
        return None

def demonstrate_s3_backend():
    """Demonstrate S3 backend (requires AWS credentials or LocalStack)."""
    print("\n=== S3 Backend (Archive) ===")
    
    factory = DatabaseFactory()
    
    if not factory.is_backend_available("s3"):
        print("S3 backend not available")
        print("Install with: pip install dataknobs-data[s3]")
        return None
    
    try:
        # For LocalStack testing
        if os.environ.get("USE_LOCALSTACK"):
            s3_db = factory.create(
                backend="s3",
                bucket="demo-bucket",
                prefix="archives/",
                region="us-east-1",
                endpoint_url="http://localhost:4566",
                access_key_id="test",
                secret_access_key="test"
            )
        else:
            # Production S3
            s3_db = factory.create(
                backend="s3",
                bucket=os.environ.get("S3_BUCKET", "my-archive"),
                prefix="demo/",
                region=os.environ.get("AWS_REGION", "us-east-1")
            )
        
        # Archive data
        archive = Record({
            "type": "backup",
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"important": "data", "to": "archive"},
            "size_mb": 42
        })
        
        archive_id = s3_db.create(archive)
        print(f"Archived to S3: {archive_id}")
        
        # List archives
        total = s3_db.count()
        print(f"Total archives in S3: {total}")
        
        return s3_db
        
    except Exception as e:
        print(f"Could not connect to S3: {e}")
        return None

def demonstrate_backend_migration():
    """Demonstrate migrating data between backends."""
    print("\n=== Backend Migration ===")
    
    factory = DatabaseFactory()
    
    # Source: Memory (simulate production data)
    source = factory.create(backend="memory")
    
    # Create sample data
    for i in range(10):
        source.create(Record({
            "id": f"record-{i}",
            "value": i * 100,
            "status": "active" if i % 2 == 0 else "inactive"
        }))
    
    print(f"Source has {source.count()} records")
    
    # Destination: File (for backup)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        backup_path = f.name
    
    dest = factory.create(
        backend="file",
        path=backup_path,
        format="json"
    )
    
    # Migrate all data
    all_records = source.search(Query())
    for record in all_records:
        dest.create(record)
    
    print(f"Migrated {dest.count()} records to {backup_path}")
    
    # Clean up
    os.unlink(backup_path)

def main():
    """Run all backend demonstrations."""
    print("DataKnobs Backend Examples")
    print("=" * 50)
    
    # Show available backends
    factory = DatabaseFactory()
    backends = factory.get_available_backends()
    print(f"Available backends: {', '.join(backends)}")
    
    # Demonstrate each backend
    memory_db = demonstrate_memory_backend()
    file_db = demonstrate_file_backend()
    duck_db = demonstrate_duckdb_backend()
    pg_db = demonstrate_postgres_backend()
    es_db = demonstrate_elasticsearch_backend()
    s3_db = demonstrate_s3_backend()

    # Demonstrate migration
    demonstrate_backend_migration()

    print("\n" + "=" * 50)
    print("✅ Backend examples completed!")
    print("\nKey Takeaways:")
    print("- Memory: Fast, temporary, good for caching")
    print("- File: Simple persistence, good for small datasets")
    print("- DuckDB: Fast analytics, 10-100x faster than SQLite for OLAP")
    print("- PostgreSQL: ACID compliance, complex queries")
    print("- Elasticsearch: Full-text search, analytics")
    print("- S3: Unlimited storage, cost-effective archival")

if __name__ == "__main__":
    main()
```

## Running the Example

### Basic Setup
```bash
# Install the package
pip install dataknobs-data

# Run with memory and file backends (no external dependencies)
python data_backends_example.py
```

### With PostgreSQL
```bash
# Install PostgreSQL support
pip install dataknobs-data[postgres]

# Set environment variables
export PG_HOST=localhost
export PG_DATABASE=test
export PG_USER=postgres
export PG_PASSWORD=postgres

# Run the example
python data_backends_example.py
```

### With Elasticsearch
```bash
# Install Elasticsearch support
pip install dataknobs-data[elasticsearch]

# Start Elasticsearch with Docker
docker run -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.11.0

# Set environment variables
export ES_HOST=localhost:9200

# Run the example
python data_backends_example.py
```

### With S3 (LocalStack)
```bash
# Install S3 support
pip install dataknobs-data[s3]

# Start LocalStack
docker run -p 4566:4566 localstack/localstack

# Set environment variables
export USE_LOCALSTACK=1

# Run the example
python data_backends_example.py
```

## Key Concepts Demonstrated

1. **Backend Selection**: Choose the right backend for your use case
2. **Configuration**: Use environment variables and config files
3. **CRUD Operations**: Create, Read, Update, Delete across all backends
4. **Querying**: Consistent query API across different storage types
5. **Migration**: Move data between different backends
6. **Error Handling**: Gracefully handle missing dependencies
7. **Performance**: Understand performance characteristics of each backend