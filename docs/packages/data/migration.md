# Data Migration Utilities

The DataKnobs data package provides comprehensive migration utilities to facilitate data movement between different backends, schema evolution, and data transformation.

## Overview

The migration utilities enable:

- **Backend-to-backend migration**: Move data between any supported backends (Memory, File, PostgreSQL, Elasticsearch, S3)
- **Schema evolution**: Manage schema versions and automatic migration generation
- **Data transformation**: Apply transformations during migration with pipeline support
- **Progress tracking**: Monitor migration progress with detailed statistics
- **Error handling**: Robust error recovery and retry mechanisms

## Core Components

### DataMigrator

The `DataMigrator` class handles the transfer of records between different database backends.

```python
from dataknobs_data.migration import DataMigrator
from dataknobs_data.backends.postgres import PostgresDatabase
from dataknobs_data.backends.s3 import S3Database

# Initialize source and target databases
source = PostgresDatabase.from_config(config.get_database("postgres"))
target = S3Database.from_config(config.get_database("s3"))

# Create migrator
migrator = DataMigrator(source, target)
```

#### Synchronous Migration

```python
# Simple migration
result = migrator.migrate_sync()
print(f"Migrated {result.successful_records} records")
print(f"Failed: {result.failed_records}")
print(f"Duration: {result.duration:.2f} seconds")

# Migration with options
result = migrator.migrate_sync(
    batch_size=1000,              # Process in batches
    transform=lambda r: r,         # Apply transformation
    on_error="skip",              # Skip errors or "stop"
    progress_callback=print_progress
)
```

#### Asynchronous Migration

```python
import asyncio

async def migrate_async():
    result = await migrator.migrate_async(
        batch_size=5000,
        parallel_batches=4,  # Process 4 batches concurrently
        transform=transform_record
    )
    return result

result = asyncio.run(migrate_async())
```

#### Progress Tracking

```python
def progress_callback(progress: MigrationProgress):
    pct = (progress.processed_records / progress.total_records) * 100
    print(f"Progress: {pct:.1f}% ({progress.processed_records}/{progress.total_records})")
    print(f"Rate: {progress.records_per_second:.0f} records/sec")
    if progress.estimated_time_remaining:
        print(f"ETA: {progress.estimated_time_remaining:.0f} seconds")

result = migrator.migrate_sync(
    progress_callback=progress_callback,
    progress_interval=1.0  # Update every second
)
```

### SchemaEvolution

The `SchemaEvolution` class manages schema versions and migrations between them.

```python
from dataknobs_data.migration import SchemaEvolution
from dataknobs_data.validation import Schema, FieldDefinition

# Define schema versions
evolution = SchemaEvolution("user_schema")

# Version 1: Basic user
v1_schema = Schema(
    name="UserV1",
    fields={
        "name": FieldDefinition(name="name", type=str, required=True),
        "email": FieldDefinition(name="email", type=str, required=True)
    }
)
evolution.add_version("1.0.0", v1_schema)

# Version 2: Add age field
v2_schema = Schema(
    name="UserV2",
    fields={
        "name": FieldDefinition(name="name", type=str, required=True),
        "email": FieldDefinition(name="email", type=str, required=True),
        "age": FieldDefinition(name="age", type=int, required=False, default=0)
    }
)
evolution.add_version("2.0.0", v2_schema)
```

#### Automatic Migration Generation

```python
# Generate migration from v1 to v2
migration = evolution.generate_migration("1.0.0", "2.0.0")

# The migration automatically detects:
# - Added fields (with defaults)
# - Removed fields
# - Type changes
# - Constraint changes

# Apply migration to records
migrated_records = migration.apply(v1_records)
```

#### Custom Migration Logic

```python
def custom_migration(record):
    """Custom migration from v1 to v2"""
    # Add computed field
    record.fields["age"] = Field(
        name="age",
        type=FieldType.INTEGER,
        value=calculate_age(record.fields["birthdate"].value)
    )
    # Remove deprecated field
    del record.fields["birthdate"]
    return record

evolution.add_migration("1.0.0", "2.0.0", custom_migration)
```

#### Migration History

```python
# Track applied migrations
evolution.apply_migration(database, "1.0.0", "2.0.0")

# Get migration history
history = evolution.get_history()
for entry in history:
    print(f"{entry.from_version} -> {entry.to_version}")
    print(f"Applied: {entry.timestamp}")
    print(f"Records: {entry.records_affected}")
```

### DataTransformer

The `DataTransformer` class provides field mapping and value transformation capabilities.

```python
from dataknobs_data.migration import DataTransformer

# Create transformer with field mapping
transformer = DataTransformer(
    field_mapping={
        "full_name": "name",           # Rename field
        "email_address": "email",       # Rename field
        "years": "age"                  # Rename field
    },
    value_transformers={
        "age": lambda v: int(v) if v else 0,  # Type conversion
        "email": lambda v: v.lower(),         # Normalize
        "name": lambda v: v.title()           # Format
    }
)

# Transform a record
transformed = transformer.transform(record)
```

#### Built-in Transformers

```python
from dataknobs_data.migration.transformers import (
    lowercase_transformer,
    uppercase_transformer,
    trim_transformer,
    default_value_transformer,
    type_cast_transformer,
    regex_replace_transformer
)

transformer = DataTransformer(
    value_transformers={
        "email": lowercase_transformer,
        "name": trim_transformer,
        "age": type_cast_transformer(int, default=0),
        "phone": regex_replace_transformer(r"[^0-9]", ""),
        "status": default_value_transformer("active")
    }
)
```

#### Transformation Pipelines

```python
from dataknobs_data.migration import TransformationPipeline

# Create a pipeline of transformations
pipeline = TransformationPipeline([
    # Step 1: Clean data
    DataTransformer(value_transformers={
        "email": lowercase_transformer,
        "name": trim_transformer
    }),
    
    # Step 2: Validate
    SchemaValidator(user_schema),
    
    # Step 3: Enrich
    DataTransformer(value_transformers={
        "created_at": lambda v: v or datetime.now(),
        "updated_at": lambda v: datetime.now()
    }),
    
    # Step 4: Custom logic
    custom_business_logic_transformer
])

# Apply pipeline during migration
migrator = DataMigrator(source, target)
result = migrator.migrate_sync(transform=pipeline.transform)
```

## Advanced Migration Patterns

### Conditional Migration

```python
def should_migrate(record):
    """Only migrate active records"""
    return record.fields.get("status", Field("", "", "")).value == "active"

result = migrator.migrate_sync(
    filter_fn=should_migrate,
    batch_size=1000
)
```

### Incremental Migration

```python
from datetime import datetime, timedelta

# Migrate only recent records
last_sync = datetime.now() - timedelta(days=1)

query = Query(
    filters=[
        Filter(field="updated_at", operator=">=", value=last_sync)
    ]
)

records = source.search(query)
migrator = DataMigrator(source, target)
result = migrator.migrate_records(records)
```

### Parallel Migration

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def migrate_partition(partition_key):
    """Migrate a single partition"""
    query = Query(filters=[
        Filter(field="partition", operator="=", value=partition_key)
    ])
    
    source = PostgresDatabase.from_config(config)
    target = S3Database.from_config(config)
    migrator = DataMigrator(source, target)
    
    return await migrator.migrate_async(query=query)

# Migrate multiple partitions in parallel
partitions = ["us-east", "us-west", "eu-west", "ap-south"]
tasks = [migrate_partition(p) for p in partitions]
results = await asyncio.gather(*tasks)
```

### Two-Phase Migration

```python
class TwoPhaseM igrator:
    """Migrate with validation phase"""
    
    def __init__(self, source, target, validator):
        self.source = source
        self.target = target
        self.validator = validator
    
    def migrate(self):
        # Phase 1: Validate all records
        print("Phase 1: Validation")
        invalid_records = []
        
        for record in self.source.all():
            result = self.validator.validate(record)
            if not result.is_valid:
                invalid_records.append((record, result.errors))
        
        if invalid_records:
            print(f"Found {len(invalid_records)} invalid records")
            # Handle invalid records (log, fix, or abort)
            return False
        
        # Phase 2: Migrate validated records
        print("Phase 2: Migration")
        migrator = DataMigrator(self.source, self.target)
        result = migrator.migrate_sync()
        
        return result
```

## Error Handling

### Error Recovery Strategies

```python
from dataknobs_data.migration import MigrationError, RetryPolicy

# Configure retry policy
retry_policy = RetryPolicy(
    max_retries=3,
    backoff_factor=2.0,  # Exponential backoff
    retry_on=[ConnectionError, TimeoutError]
)

# Migration with retry
result = migrator.migrate_sync(
    retry_policy=retry_policy,
    on_error="continue"  # Continue on error
)

# Check failed records
if result.failed_records > 0:
    for error in result.errors:
        print(f"Record {error.record_id}: {error.message}")
        # Optionally retry failed records
```

### Rollback Support

```python
class TransactionalMigration:
    """Migration with rollback capability"""
    
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.migrated_ids = []
    
    def migrate_with_rollback(self):
        try:
            # Track migrated records
            for record in self.source.all():
                self.target.create(record)
                self.migrated_ids.append(record.id)
            
            # Verify migration
            if not self.verify():
                raise MigrationError("Verification failed")
                
        except Exception as e:
            # Rollback on error
            self.rollback()
            raise
    
    def rollback(self):
        """Remove migrated records from target"""
        for record_id in self.migrated_ids:
            try:
                self.target.delete(record_id)
            except:
                pass  # Best effort rollback
```

## Performance Optimization

### Batch Size Tuning

```python
# Find optimal batch size
def find_optimal_batch_size(source, target, sample_size=1000):
    """Determine optimal batch size for migration"""
    
    test_sizes = [100, 500, 1000, 5000, 10000]
    results = {}
    
    # Test each batch size with sample
    sample_records = list(source.search(Query(limit=sample_size)))
    
    for size in test_sizes:
        start = time.time()
        migrator = DataMigrator(source, target)
        migrator.migrate_records(sample_records, batch_size=size)
        duration = time.time() - start
        
        results[size] = sample_size / duration  # Records per second
    
    # Return batch size with best throughput
    optimal = max(results, key=results.get)
    print(f"Optimal batch size: {optimal} ({results[optimal]:.0f} records/sec)")
    return optimal
```

### Memory-Efficient Migration

```python
def migrate_large_dataset(source, target, chunk_size=10000):
    """Migrate large datasets without loading all into memory"""
    
    migrator = DataMigrator(source, target)
    total = source.count()
    offset = 0
    
    while offset < total:
        # Process one chunk at a time
        query = Query(offset=offset, limit=chunk_size)
        chunk_result = migrator.migrate_sync(
            query=query,
            batch_size=1000
        )
        
        offset += chunk_size
        print(f"Processed {min(offset, total)}/{total} records")
        
        # Optional: Clear caches between chunks
        import gc
        gc.collect()
```

## Monitoring and Logging

```python
import logging
from dataknobs_data.migration import MigrationMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migration")

# Create monitor
monitor = MigrationMonitor(
    log_interval=10.0,  # Log stats every 10 seconds
    metrics_collector=prometheus_client  # Optional metrics
)

# Migration with monitoring
result = migrator.migrate_sync(
    monitor=monitor,
    progress_callback=monitor.update
)

# Get final statistics
stats = monitor.get_statistics()
print(f"Total time: {stats.duration:.2f}s")
print(f"Average speed: {stats.avg_records_per_second:.0f} records/sec")
print(f"Peak speed: {stats.peak_records_per_second:.0f} records/sec")
print(f"Memory used: {stats.peak_memory_mb:.0f} MB")
```

## Best Practices

1. **Test migrations thoroughly**: Always test with a subset of data first
2. **Monitor progress**: Use progress callbacks for long-running migrations
3. **Handle errors gracefully**: Implement retry logic and error recovery
4. **Optimize batch sizes**: Find the optimal batch size for your data and backends
5. **Validate data**: Ensure data integrity before and after migration
6. **Document schema changes**: Keep clear records of schema evolution
7. **Plan for rollback**: Have a rollback strategy for critical migrations
8. **Use appropriate backends**: Choose backends that match your performance needs
9. **Consider parallelization**: Use async/parallel migration for large datasets
10. **Monitor resource usage**: Track memory and CPU usage during migration

## See Also

- [Schema Validation](validation.md) - Data validation and schema management
- [Pandas Integration](pandas-integration.md) - Bulk operations with pandas
- [Backends Overview](backends.md) - Supported database backends
- [Migration Tutorial](tutorials/migration-tutorial.md) - Step-by-step migration guide