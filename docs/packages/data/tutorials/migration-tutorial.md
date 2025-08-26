# Data Migration Tutorial

This tutorial will guide you through the DataKnobs Data package migration features, from simple field transformations to complex database-to-database migrations.

## Prerequisites

```python
from dataknobs_data import (
    Migration, Migrator, Transformer,
    AddField, RemoveField, RenameField, TransformField, CompositeOperation,
    MemoryDatabase, Record, FieldType
)
from datetime import datetime
import json
```

## Part 1: Basic Field Operations

### Creating Your First Migration

Let's start with a simple scenario: your user records need to evolve from version 1 to version 2.

```python
# Create sample v1 records
v1_records = [
    Record(data={"username": "john_doe", "email": "john@example.com", "created": "2024-01-01"}),
    Record(data={"username": "jane_smith", "email": "jane@example.com", "created": "2024-01-02"}),
]

# Create a migration from v1 to v2
migration = Migration("v1", "v2", "Add user status and standardize fields")

# Add a status field with default value
migration.add(AddField("status", "active", FieldType.STRING))

# Rename username to user_name for consistency
migration.add(RenameField("username", "user_name"))

# Transform created date to timestamp
migration.add(TransformField("created", lambda x: datetime.fromisoformat(x).timestamp()))

# Apply the migration
v2_records = [migration.apply(record) for record in v1_records]

# Check the results
for record in v2_records:
    print(f"Migrated: {record.data}")
    # Output: {'user_name': 'john_doe', 'email': '...', 'created': 1704067200.0, 'status': 'active'}
```

### Reversing Migrations

All migrations in DataKnobs are reversible, allowing you to rollback if needed:

```python
# Rollback the migration
original_records = [migration.apply(record, reverse=True) for record in v2_records]

# Verify rollback
for record in original_records:
    print(f"Rolled back: {record.data}")
    # Should match original v1 structure
```

## Part 2: Complex Transformations

### Composite Operations

For more complex scenarios, use composite operations to group related changes:

```python
# Example: Splitting a full name field into first and last names
class SplitNameOperation(CompositeOperation):
    def __init__(self):
        super().__init__("Split full name into components")
    
    def apply(self, record: Record) -> Record:
        full_name = record.get_value("full_name", "")
        parts = full_name.split(" ", 1)
        
        data = record.data.copy()
        data["first_name"] = parts[0] if parts else ""
        data["last_name"] = parts[1] if len(parts) > 1 else ""
        del data["full_name"]
        
        return Record(data=data, metadata=record.metadata)
    
    def reverse(self, record: Record) -> Record:
        first = record.get_value("first_name", "")
        last = record.get_value("last_name", "")
        
        data = record.data.copy()
        data["full_name"] = f"{first} {last}".strip()
        del data["first_name"]
        del data["last_name"]
        
        return Record(data=data, metadata=record.metadata)

# Use the composite operation
migration = Migration("v2", "v3", "Split name fields")
migration.add(SplitNameOperation())

# Test data
test_record = Record(data={"full_name": "John Doe", "email": "john@example.com"})
migrated = migration.apply(test_record)
print(migrated.data)  # {'first_name': 'John', 'last_name': 'Doe', 'email': '...'}
```

## Part 3: Using the Transformer

The Transformer provides a fluent API for common data transformations:

```python
# Create a transformer for cleaning and standardizing data
transformer = (Transformer()
    # Map old field names to new ones
    .map("oldEmail", "email")
    .map("phoneNum", "phone_number")
    
    # Rename fields
    .rename("cost", "price")
    .rename("qty", "quantity")
    
    # Exclude sensitive fields
    .exclude("password", "ssn", "credit_card")
    
    # Add computed fields
    .add("processed_at", lambda r: datetime.now().isoformat())
    .add("full_name", lambda r: f"{r.get_value('first_name')} {r.get_value('last_name')}")
    .add("is_premium", lambda r: r.get_value("account_type") == "premium")
)

# Apply transformer to records
raw_records = [
    Record(data={
        "first_name": "John", 
        "last_name": "Doe",
        "oldEmail": "john@example.com",
        "phoneNum": "555-1234",
        "cost": 99.99,
        "qty": 2,
        "account_type": "premium",
        "password": "secret123"
    })
]

transformed = transformer.transform_many(raw_records)
for record in transformed:
    print(json.dumps(record.data, indent=2))
```

## Part 4: Database-to-Database Migration

### Simple Database Migration

Migrate data between databases with automatic progress tracking:

```python
# Setup source and target databases
source_db = MemoryDatabase()
target_db = MemoryDatabase()

# Populate source database
for i in range(100):
    source_db.insert(Record(data={
        "id": i,
        "name": f"Product {i}",
        "price": 10.0 + i,
        "category": "electronics" if i % 2 == 0 else "books"
    }))

# Create a migrator
migrator = Migrator()

# Define transformation rules
transformer = (Transformer()
    # Add 10% price increase
    .transform("price", lambda p: round(p * 1.1, 2))
    # Add migration timestamp
    .add("migrated_at", lambda r: datetime.now().isoformat())
    # Add computed discount field
    .add("discount", lambda r: 0.1 if r.get_value("category") == "books" else 0.05)
)

# Perform migration with progress tracking
def on_progress(progress):
    print(f"Migration: {progress.processed}/{progress.total} records "
          f"({progress.percent:.1f}%) - {progress.status}")

progress = migrator.migrate(
    source=source_db,
    target=target_db,
    transform=transformer,
    batch_size=10,
    on_progress=on_progress
)

print(f"\nMigration completed: {progress.processed} records in {progress.elapsed_time:.2f}s")
print(f"Rate: {progress.records_per_second:.1f} records/second")
```

### Streaming Migration for Large Datasets

For large datasets that don't fit in memory, use streaming migration:

```python
# Stream migration with memory-efficient processing
def create_large_source():
    """Simulate a large database"""
    db = MemoryDatabase()
    for i in range(10000):  # Simulate 10k records
        db.insert(Record(data={
            "id": i,
            "value": f"data_{i}",
            "timestamp": datetime.now().timestamp() - i
        }))
    return db

source_db = create_large_source()
target_db = MemoryDatabase()

# Stream with small batches to minimize memory usage
progress = migrator.migrate_stream(
    source=source_db,
    target=target_db,
    transform=transformer,
    batch_size=100,  # Process 100 records at a time
    on_progress=lambda p: print(f"Streamed: {p.processed} records", end="\r")
)

print(f"\nStreaming complete: {progress.processed} records")
```

## Part 5: Advanced Migration Patterns

### Conditional Transformations

Apply different transformations based on record content:

```python
class ConditionalTransformer:
    """Apply different transformations based on conditions"""
    
    def transform_many(self, records):
        transformed = []
        for record in records:
            if record.get_value("account_type") == "premium":
                # Premium account transformation
                t = Transformer().add("benefits", ["priority_support", "no_ads"])
            else:
                # Standard account transformation  
                t = Transformer().add("benefits", ["basic_support"])
            
            transformed.extend(t.transform_many([record]))
        return transformed

# Use conditional transformer in migration
migrator.migrate(
    source=source_db,
    target=target_db,
    transform=ConditionalTransformer()
)
```

### Migration with Validation

Combine migration with validation to ensure data quality:

```python
from dataknobs_data.validation import Schema, Range, Pattern

# Define target schema
target_schema = (Schema("TargetSchema")
    .field("email", "STRING", required=True, 
           constraints=[Pattern(r"^[\w\.-]+@[\w\.-]+\.\w+$")])
    .field("age", "INTEGER", constraints=[Range(min=0, max=150)])
    .field("status", "STRING", required=True)
)

class ValidatingTransformer:
    """Transformer that validates before migrating"""
    
    def __init__(self, schema, transformer):
        self.schema = schema
        self.transformer = transformer
    
    def transform_many(self, records):
        # First apply transformation
        transformed = self.transformer.transform_many(records)
        
        # Then validate
        valid_records = []
        for record in transformed:
            result = self.schema.validate(record)
            if result.valid:
                valid_records.append(result.value)
            else:
                print(f"Skipping invalid record: {result.errors}")
        
        return valid_records

# Create validating transformer
transformer = Transformer().add("status", "active")
validating_transformer = ValidatingTransformer(target_schema, transformer)

# Migrate with validation
progress = migrator.migrate(
    source=source_db,
    target=target_db,
    transform=validating_transformer
)
```

### Parallel Migration

For better performance with large datasets, use parallel processing:

```python
from concurrent.futures import ThreadPoolExecutor
from dataknobs_data import Query

def migrate_partition(partition_id, total_partitions):
    """Migrate a partition of data"""
    # Create query for this partition
    query = Query().filter("id", "%", lambda x: x % total_partitions == partition_id)
    
    # Get records for this partition
    records = source_db.search(query)
    
    # Transform and insert
    transformed = transformer.transform_many(records)
    for record in transformed:
        target_db.insert(record)
    
    return len(transformed)

# Parallel migration with 4 threads
num_partitions = 4
with ThreadPoolExecutor(max_workers=num_partitions) as executor:
    futures = [
        executor.submit(migrate_partition, i, num_partitions)
        for i in range(num_partitions)
    ]
    
    total_migrated = sum(f.result() for f in futures)
    print(f"Parallel migration complete: {total_migrated} records")
```

## Part 6: Migration Recipes

### Recipe 1: Schema Evolution

Evolving from a flat structure to nested structure:

```python
# v1: Flat structure
v1_record = Record(data={
    "user_id": 1,
    "user_name": "John",
    "user_email": "john@example.com",
    "address_street": "123 Main St",
    "address_city": "Boston",
    "address_zip": "02101"
})

# Transform to v2: Nested structure
class NestAddressTransformer:
    def transform_many(self, records):
        transformed = []
        for record in records:
            data = record.data.copy()
            
            # Extract address fields
            address = {
                "street": data.pop("address_street", ""),
                "city": data.pop("address_city", ""),
                "zip": data.pop("address_zip", "")
            }
            
            # Create nested structure
            user_data = {
                "id": data.pop("user_id"),
                "name": data.pop("user_name"),
                "email": data.pop("user_email"),
                "address": address,
                **data  # Include any remaining fields
            }
            
            transformed.append(Record(data=user_data))
        return transformed

transformer = NestAddressTransformer()
v2_records = transformer.transform_many([v1_record])
print(json.dumps(v2_records[0].data, indent=2))
```

### Recipe 2: Data Denormalization

Combine data from multiple record types:

```python
# Simulate related data
users_db = MemoryDatabase()
orders_db = MemoryDatabase()

# Add sample data
users_db.insert(Record(data={"id": 1, "name": "John", "email": "john@example.com"}))
orders_db.insert(Record(data={"order_id": 101, "user_id": 1, "total": 99.99}))
orders_db.insert(Record(data={"order_id": 102, "user_id": 1, "total": 149.99}))

class DenormalizingMigrator:
    """Combine user and order data"""
    
    def migrate(self, users_db, orders_db, target_db):
        for user_record in users_db.search(Query()):
            user_id = user_record.get_value("id")
            
            # Find all orders for this user
            user_orders = orders_db.search(
                Query().filter("user_id", "=", user_id)
            )
            
            # Create denormalized record
            order_totals = [o.get_value("total") for o in user_orders]
            denormalized = Record(data={
                **user_record.data,
                "order_count": len(user_orders),
                "total_spent": sum(order_totals),
                "average_order": sum(order_totals) / len(order_totals) if order_totals else 0,
                "order_ids": [o.get_value("order_id") for o in user_orders]
            })
            
            target_db.insert(denormalized)

# Perform denormalization
target_db = MemoryDatabase()
migrator = DenormalizingMigrator()
migrator.migrate(users_db, orders_db, target_db)

# Check results
for record in target_db.search(Query()):
    print(f"Denormalized: {json.dumps(record.data, indent=2)}")
```

## Best Practices

1. **Always Test Migrations**: Test on a copy of your data first
2. **Use Reversible Operations**: Ensure you can rollback if needed
3. **Validate After Migration**: Check data integrity post-migration
4. **Monitor Progress**: Use progress callbacks for long-running migrations
5. **Batch Processing**: Use appropriate batch sizes for your data volume
6. **Error Handling**: Implement proper error handling and logging
7. **Document Changes**: Keep a migration history with descriptions
8. **Backup Before Migration**: Always backup your data before migrating

## Summary

You've learned how to:

- Create and apply basic field operations
- Build complex transformations with composite operations
- Use the Transformer API for fluent data manipulation
- Perform database-to-database migrations
- Handle large datasets with streaming
- Implement validation during migration
- Apply advanced patterns like parallel processing

Next, explore the [Validation Tutorial](validation-tutorial.md) to learn about data validation and constraints.