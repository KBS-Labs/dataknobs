# ETL Pattern

**Important**: This pattern is implemented as `DatabaseETL` class, not `ETLPattern`. The pattern focuses on database-to-database ETL operations using the AsyncDatabase abstraction from dataknobs_data.

The ETL (Extract, Transform, Load) pattern provides a robust framework for building database-focused data processing pipelines. It handles data extraction from source databases, applies transformations, and loads the results into target databases.

## Overview

The ETL pattern creates an FSM with stages for:
1. **Extract**: Read data from source database
2. **Transform**: Apply business logic and data transformations
3. **Load**: Write processed data to target database

## Basic Usage

```python
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode

# Using the class directly
config = ETLConfig(
    source_db={
        "provider": "postgresql",
        "connection": "postgresql://source_db/customers"
    },
    target_db={
        "provider": "postgresql",
        "connection": "postgresql://warehouse/customers_dim"
    },
    mode=ETLMode.FULL_REFRESH,
    source_query="SELECT * FROM customers WHERE created_at > '2024-01-01'",
    target_table="dim_customers",
    transformations=[lambda row: {
        **row,
        "full_name": f"{row['first_name']} {row['last_name']}",
        "processed_at": datetime.now()
    }]
)

etl = DatabaseETL(config)

# Execute pipeline
import asyncio
result = asyncio.run(etl.run())
print(f"Processed {result['records_processed']} records")

# Or use the factory function
from dataknobs_fsm.patterns.etl import create_etl_pipeline

etl = create_etl_pipeline(
    source={"provider": "postgresql", "connection": "postgresql://source/db"},
    target={"provider": "postgresql", "connection": "postgresql://target/db"},
    mode=ETLMode.INCREMENTAL,
    transformations=[...]
)
result = asyncio.run(etl.run())
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
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode

config = ETLConfig(
    source_db={
        "provider": "sqlite",
        "database": "source.db"
    },
    target_db={
        "provider": "sqlite",
        "database": "target.db"
    },
    mode=ETLMode.UPSERT,
    source_query="SELECT * FROM products",
    target_table="products_transformed",
    key_columns=["product_id"],
    field_mappings={
        "id": "product_id",
        "name": "product_name",
        "price": "unit_price"
    },
    batch_size=1000,
    error_threshold=0.05  # Allow 5% errors
)

etl = DatabaseETL(config)
```

## ETL Modes

### Available Modes

```python
from dataknobs_fsm.patterns.etl import ETLMode

# Full refresh - replace all data
ETLMode.FULL_REFRESH

# Incremental - process only new/changed data
ETLMode.INCREMENTAL

# Upsert - update existing, insert new
ETLMode.UPSERT

# Append - always append, no updates
ETLMode.APPEND
```

## Factory Functions

### create_etl_pipeline

```python
from dataknobs_fsm.patterns.etl import create_etl_pipeline, ETLMode

etl = create_etl_pipeline(
    source={
        "provider": "postgresql",
        "host": "localhost",
        "database": "source_db",
        "user": "user",
        "password": "pass"
    },
    target={
        "provider": "postgresql",
        "host": "localhost",
        "database": "target_db"
    },
    mode=ETLMode.INCREMENTAL,
    transformations=[...]
)

# Run the ETL pipeline
import asyncio
result = asyncio.run(etl.run())
```

### create_database_sync

```python
from dataknobs_fsm.patterns.etl import create_database_sync

# Synchronize two databases
sync = create_database_sync(
    source={
        "provider": "mysql",
        "host": "source.example.com",
        "database": "production"
    },
    target={
        "provider": "postgresql",
        "host": "target.example.com",
        "database": "analytics"
    },
    tables=["users", "orders", "products"],
    sync_mode="incremental",
    timestamp_column="updated_at"
)

result = asyncio.run(sync.run())
```

### create_data_migration

```python
from dataknobs_fsm.patterns.etl import create_data_migration

# Migrate data with field mappings
migration = create_data_migration(
    source={
        "provider": "sqlite",
        "database": "old.db",
        "table": "customers"
    },
    target={
        "provider": "postgresql",
        "connection": "postgresql://new_db",
        "table": "customers_v2"
    },
    field_mappings={
        "customer_id": "id",
        "customer_name": "name",
        "customer_email": "email"
    },
    transformations=[...]
)

result = asyncio.run(migration.run())
```

## Transformations

Transformations are applied as a list of callable functions:

```python
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig

config = ETLConfig(
    source_db={...},
    target_db={...},
    transformations=[
        # Filter rows
        lambda row: row if row["age"] >= 18 else None,

        # Transform fields
        lambda row: {
            **row,
            "full_name": f"{row['first_name']} {row['last_name']}",
            "age_group": "adult" if row["age"] >= 18 else "minor"
        },

        # Clean data
        lambda row: {k: v for k, v in row.items() if v is not None}
    ]
)

etl = DatabaseETL(config)
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

# Or use a class
class DataEnricher:
    def __init__(self, lookup_service):
        self.lookup = lookup_service

    def __call__(self, row):
        enriched = self.lookup.enrich(row["id"])
        return {**row, **enriched}

# Add transformations to config
config = ETLConfig(
    source_db={...},
    target_db={...},
    transformations=[
        custom_transform,
        DataEnricher(lookup_service)
    ]
)
```

## Configuration Options

### ETLConfig Parameters

```python
from dataknobs_fsm.patterns.etl import ETLConfig, ETLMode

config = ETLConfig(
    # Required parameters
    source_db={"provider": "postgresql", "connection": "..."},
    target_db={"provider": "postgresql", "connection": "..."},

    # Mode selection
    mode=ETLMode.UPSERT,  # FULL_REFRESH, INCREMENTAL, UPSERT, APPEND

    # Query configuration
    source_query="SELECT * FROM source_table WHERE active = true",
    target_table="destination_table",
    key_columns=["id"],  # For upsert mode

    # Field mappings
    field_mappings={
        "src_id": "dest_id",
        "src_name": "dest_name"
    },

    # Performance options
    batch_size=5000,
    parallel_workers=4,

    # Error handling
    error_threshold=0.05,  # Max 5% errors
    checkpoint_interval=10000,  # Checkpoint every 10k records

    # Transformations
    transformations=[...],

    # Validation
    validation_schema={...},

    # Enrichment sources
    enrichment_sources=[...]
)
```

## Advanced Features

### Batch Processing

```python
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig

config = ETLConfig(
    source_db={...},
    target_db={...},
    batch_size=10000,
    parallel_workers=4,
    mode=ETLMode.FULL_REFRESH
)

etl = DatabaseETL(config)

# Run with async context
import asyncio
result = asyncio.run(etl.run())
print(f"Processed {result['records_processed']} records in batches")
```

### Incremental Processing

```python
config = ETLConfig(
    source_db={...},
    target_db={...},
    mode=ETLMode.INCREMENTAL,
    source_query="""
        SELECT * FROM source_table
        WHERE updated_at > :last_checkpoint
        ORDER BY updated_at
    """,
    checkpoint_interval=1000
)

etl = DatabaseETL(config)

# Resume from checkpoint if available
if etl.has_checkpoint():
    result = asyncio.run(etl.resume())
else:
    result = asyncio.run(etl.run())
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
- [CLI Guide](../guides/cli.md) for optimization tips

