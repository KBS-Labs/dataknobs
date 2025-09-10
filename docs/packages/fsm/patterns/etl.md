# ETL Pattern

The ETL (Extract, Transform, Load) pattern provides a robust framework for building data processing pipelines. It handles data extraction from various sources, applies transformations, and loads the results into target systems.

## Overview

The ETL pattern creates an FSM with three main stages:
1. **Extract**: Read data from source systems
2. **Transform**: Apply business logic and data transformations
3. **Load**: Write processed data to target systems

## Basic Usage

```python
from dataknobs_fsm.patterns import ETLPattern

# Create ETL pipeline
etl = ETLPattern(
    name="customer_etl",
    source_config={
        "type": "database",
        "provider": "postgresql",
        "connection": "postgresql://source_db/customers"
    },
    target_config={
        "type": "database",
        "provider": "postgresql",
        "connection": "postgresql://warehouse/customers_dim"
    }
)

# Add transformations
etl.add_transformation(lambda row: {
    **row,
    "full_name": f"{row['first_name']} {row['last_name']}",
    "processed_at": datetime.now()
})

# Execute pipeline
result = etl.run({
    "source_query": "SELECT * FROM customers WHERE created_at > '2024-01-01'",
    "target_table": "dim_customers"
})

print(f"Processed {result['records_processed']} records")
```

## Configuration

### YAML Configuration

```yaml
pattern: etl
name: sales_etl

source:
  type: database
  provider: mysql
  config:
    host: source.example.com
    port: 3306
    database: sales
    user: ${DB_USER}
    password: ${DB_PASSWORD}

target:
  type: database
  provider: postgresql
  config:
    host: warehouse.example.com
    database: analytics
    schema: sales

transformations:
  - type: filter
    condition: "status == 'completed'"
  
  - type: map
    fields:
      order_id: id
      customer_id: customer.id
      total_amount: amount * 1.1  # Add tax
      
  - type: aggregate
    group_by: [customer_id]
    aggregations:
      total_orders: count
      total_revenue: sum(total_amount)
      
  - type: enrich
    source: api
    endpoint: "/customers/{customer_id}"
    fields: [customer_name, customer_segment]

options:
  batch_size: 5000
  parallel: true
  workers: 4
  on_duplicate: update
  on_error: log_and_continue

monitoring:
  metrics: true
  logging: INFO
  checkpoint_interval: 1000
```

### Python Configuration

```python
from dataknobs_fsm.patterns import ETLPattern

etl = ETLPattern.from_config({
    "name": "product_etl",
    "source": {
        "type": "file",
        "format": "csv",
        "path": "/data/products.csv"
    },
    "target": {
        "type": "database",
        "provider": "sqlite",
        "database": "products.db"
    },
    "transformations": [
        {"type": "validate", "schema": {...}},
        {"type": "clean", "remove_nulls": True},
        {"type": "normalize", "fields": ["price", "quantity"]}
    ],
    "options": {
        "batch_size": 1000,
        "error_threshold": 0.05  # Allow 5% errors
    }
})
```

## Data Sources

### Database Sources

```python
# PostgreSQL
etl = ETLPattern(
    source_config={
        "type": "database",
        "provider": "postgresql",
        "config": {
            "host": "localhost",
            "database": "mydb",
            "user": "user",
            "password": "pass"
        }
    }
)

# MySQL
etl = ETLPattern(
    source_config={
        "type": "database",
        "provider": "mysql",
        "config": {...}
    }
)

# MongoDB
etl = ETLPattern(
    source_config={
        "type": "database",
        "provider": "mongodb",
        "config": {
            "connection_string": "mongodb://localhost:27017",
            "database": "mydb",
            "collection": "mycollection"
        }
    }
)
```

### File Sources

```python
# CSV
etl = ETLPattern(
    source_config={
        "type": "file",
        "format": "csv",
        "path": "/data/input.csv",
        "options": {
            "delimiter": ",",
            "header": True,
            "encoding": "utf-8"
        }
    }
)

# JSON
etl = ETLPattern(
    source_config={
        "type": "file",
        "format": "json",
        "path": "/data/input.json",
        "options": {
            "lines": True  # JSONL format
        }
    }
)

# Parquet
etl = ETLPattern(
    source_config={
        "type": "file",
        "format": "parquet",
        "path": "/data/input.parquet"
    }
)
```

### API Sources

```python
etl = ETLPattern(
    source_config={
        "type": "api",
        "base_url": "https://api.example.com",
        "endpoint": "/data",
        "auth": {
            "type": "bearer",
            "token": "${API_TOKEN}"
        },
        "pagination": {
            "type": "offset",
            "limit": 100
        }
    }
)
```

## Transformations

### Built-in Transformations

#### Filter
```python
etl.add_transformation({
    "type": "filter",
    "condition": lambda row: row["age"] >= 18
})
```

#### Map/Rename
```python
etl.add_transformation({
    "type": "map",
    "fields": {
        "customer_id": "id",
        "full_name": lambda r: f"{r['first']} {r['last']}"
    }
})
```

#### Aggregate
```python
etl.add_transformation({
    "type": "aggregate",
    "group_by": ["category"],
    "aggregations": {
        "total": "sum(amount)",
        "average": "avg(amount)",
        "count": "count(*)"
    }
})
```

#### Join/Enrich
```python
etl.add_transformation({
    "type": "enrich",
    "source": {
        "type": "database",
        "table": "lookup_table"
    },
    "join_on": "id",
    "fields": ["description", "category"]
})
```

#### Validate
```python
etl.add_transformation({
    "type": "validate",
    "schema": {
        "id": {"type": "integer", "required": True},
        "email": {"type": "email", "required": True},
        "amount": {"type": "float", "min": 0}
    }
})
```

### Custom Transformations

```python
def custom_transform(row):
    """Custom transformation logic."""
    # Complex business logic
    row["risk_score"] = calculate_risk(row)
    row["segment"] = determine_segment(row)
    return row

etl.add_transformation(custom_transform)

# Or use a class
class DataEnricher:
    def __init__(self, lookup_service):
        self.lookup = lookup_service
    
    def __call__(self, row):
        enriched = self.lookup.enrich(row["id"])
        return {**row, **enriched}

etl.add_transformation(DataEnricher(lookup_service))
```

## Data Targets

### Database Targets

```python
# Insert mode
etl = ETLPattern(
    target_config={
        "type": "database",
        "provider": "postgresql",
        "config": {...},
        "mode": "insert",
        "table": "target_table"
    }
)

# Upsert mode
etl = ETLPattern(
    target_config={
        "type": "database",
        "provider": "postgresql",
        "config": {...},
        "mode": "upsert",
        "table": "target_table",
        "key_columns": ["id"]
    }
)

# Replace mode
etl = ETLPattern(
    target_config={
        "type": "database",
        "provider": "postgresql",
        "config": {...},
        "mode": "replace",
        "table": "target_table"
    }
)
```

### File Targets

```python
# CSV output
etl = ETLPattern(
    target_config={
        "type": "file",
        "format": "csv",
        "path": "/output/data.csv",
        "options": {
            "header": True,
            "compression": "gzip"
        }
    }
)

# Parquet output with partitioning
etl = ETLPattern(
    target_config={
        "type": "file",
        "format": "parquet",
        "path": "/output/data",
        "partitions": ["year", "month"]
    }
)
```

## Advanced Features

### Batch Processing

```python
etl = ETLPattern(
    name="batch_etl",
    source_config={...},
    target_config={...},
    batch_config={
        "size": 10000,
        "parallel": True,
        "workers": 4,
        "memory_limit": "2GB"
    }
)

# Process with progress callback
def on_batch_complete(batch_num, records_processed):
    print(f"Batch {batch_num}: {records_processed} records")

result = etl.run(on_batch=on_batch_complete)
```

### Incremental Processing

```python
etl = ETLPattern(
    name="incremental_etl",
    source_config={
        "type": "database",
        "provider": "postgresql",
        "incremental": {
            "column": "updated_at",
            "start": "2024-01-01",
            "checkpoint": True
        }
    }
)

# Run incremental load
result = etl.run()  # Processes only new/changed records
```

### Error Handling

```python
etl = ETLPattern(
    name="resilient_etl",
    error_config={
        "strategy": "continue",  # continue, fail, compensate
        "max_errors": 100,
        "error_table": "etl_errors",
        "retry": {
            "max_attempts": 3,
            "backoff": "exponential"
        }
    }
)

# Handle errors in transformation
def safe_transform(row):
    try:
        return process(row)
    except Exception as e:
        # Log to error table
        return {"_error": str(e), **row}

etl.add_transformation(safe_transform)
```

### Monitoring and Logging

```python
etl = ETLPattern(
    name="monitored_etl",
    monitoring={
        "metrics": {
            "enabled": True,
            "export": "prometheus",
            "port": 9090
        },
        "logging": {
            "level": "INFO",
            "file": "/logs/etl.log",
            "format": "json"
        },
        "alerts": {
            "error_threshold": 0.01,
            "latency_threshold": 60,
            "webhook": "https://alerts.example.com"
        }
    }
)

# Access metrics during execution
metrics = etl.get_metrics()
print(f"Records/sec: {metrics['throughput']}")
print(f"Error rate: {metrics['error_rate']}")
```

## Performance Optimization

### Memory Management

```python
# Stream large datasets
etl = ETLPattern(
    streaming=True,
    buffer_size=1000,
    source_config={
        "type": "file",
        "format": "csv",
        "path": "/large/file.csv",
        "chunk_size": 10000
    }
)
```

### Parallel Processing

```python
# Parallel extraction and transformation
etl = ETLPattern(
    parallel_config={
        "extract_workers": 2,
        "transform_workers": 4,
        "load_workers": 2,
        "queue_size": 10000
    }
)
```

### Caching

```python
# Cache lookup data
etl = ETLPattern(
    cache_config={
        "enabled": True,
        "provider": "redis",
        "ttl": 3600,
        "size_limit": "100MB"
    }
)
```

## Complete Example

```python
from dataknobs_fsm.patterns import ETLPattern
from datetime import datetime, timedelta

# Create ETL pipeline for sales data
etl = ETLPattern(
    name="sales_analytics_etl",
    
    # Extract from transactional database
    source_config={
        "type": "database",
        "provider": "mysql",
        "config": {
            "host": "transactional.db.com",
            "database": "sales",
            "user": "etl_user",
            "password": "${ETL_PASSWORD}"
        },
        "query": """
            SELECT 
                o.id, o.customer_id, o.order_date, o.total,
                c.name, c.segment, c.region
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE o.order_date >= :start_date
        """,
        "parameters": {
            "start_date": datetime.now() - timedelta(days=1)
        }
    },
    
    # Load to data warehouse
    target_config={
        "type": "database",
        "provider": "postgresql",
        "config": {
            "host": "warehouse.db.com",
            "database": "analytics",
            "schema": "sales"
        },
        "table": "fact_sales",
        "mode": "upsert",
        "key_columns": ["order_id"]
    },
    
    # Processing options
    options={
        "batch_size": 5000,
        "parallel": True,
        "workers": 4,
        "checkpoint_interval": 10000,
        "error_threshold": 0.01
    }
)

# Add transformations
etl.add_transformation(lambda row: {
    **row,
    "order_id": row["id"],
    "revenue": row["total"] * 1.1,  # Add tax
    "quarter": f"Q{(row['order_date'].month - 1) // 3 + 1}",
    "year": row["order_date"].year,
    "etl_timestamp": datetime.now()
})

# Add data quality checks
etl.add_transformation({
    "type": "validate",
    "rules": [
        {"field": "revenue", "min": 0},
        {"field": "customer_id", "not_null": True},
        {"field": "region", "in": ["NA", "EU", "APAC"]}
    ]
})

# Execute with monitoring
def progress_callback(stats):
    print(f"Progress: {stats['processed']}/{stats['total']} records")
    print(f"Errors: {stats['errors']}, Rate: {stats['records_per_second']}/s")

result = etl.run(on_progress=progress_callback)

print(f"\nETL Complete:")
print(f"Records processed: {result['records_processed']}")
print(f"Records loaded: {result['records_loaded']}")
print(f"Errors: {result['errors']}")
print(f"Duration: {result['duration_seconds']} seconds")
```

## Testing

```python
import pytest
from dataknobs_fsm.patterns import ETLPattern

def test_etl_transformation():
    # Create test ETL
    etl = ETLPattern(
        source_config={"type": "memory", "data": [
            {"id": 1, "value": 10},
            {"id": 2, "value": 20}
        ]},
        target_config={"type": "memory"}
    )
    
    # Add transformation
    etl.add_transformation(lambda r: {**r, "doubled": r["value"] * 2})
    
    # Run and verify
    result = etl.run()
    assert result["records_processed"] == 2
    
    output = result["output_data"]
    assert output[0]["doubled"] == 20
    assert output[1]["doubled"] == 40
```

## Best Practices

1. **Use appropriate batch sizes** - Balance memory usage and performance
2. **Enable checkpointing** for long-running ETL jobs
3. **Implement data validation** early in the pipeline
4. **Use streaming** for large datasets
5. **Monitor metrics** and set up alerts
6. **Test transformations** with sample data
7. **Handle errors gracefully** with proper logging
8. **Use incremental loading** when possible
9. **Optimize queries** at the source
10. **Document data lineage** and transformations

## See Also

- [File Processing Pattern](file-processing.md) for file-specific operations
- [Error Recovery Pattern](error-recovery.md) for advanced error handling
- [Examples](../examples/database-etl.md) for real-world use cases
- [Performance Guide](../guides/performance.md) for optimization tips