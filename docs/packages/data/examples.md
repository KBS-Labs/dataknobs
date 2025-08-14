# Examples

This section provides comprehensive examples of using the dataknobs-data package in various scenarios.

## Basic Usage

### Simple CRUD Operations
```python
from dataknobs_data import Record, Query, DatabaseFactory

# Create a database
factory = DatabaseFactory()
db = factory.create(backend="memory")

# Create a record
user = Record({
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "age": 30,
    "active": True
})

# Store the record
user_id = db.create(user)
print(f"Created user with ID: {user_id}")

# Read the record
retrieved_user = db.read(user_id)
print(f"Retrieved: {retrieved_user.get_value('name')}")

# Update the record
user.fields["age"] = 31
db.update(user_id, user)

# Search for records
query = Query().filter("active", "=", True).filter("age", ">", 25)
results = db.search(query)
print(f"Found {len(results)} active users over 25")

# Delete the record
db.delete(user_id)
```

## Multi-Backend Application

### E-Commerce Platform Example
```python
from dataknobs_data import Record, Query, DatabaseFactory
from dataknobs_config import Config
from datetime import datetime, timedelta
import json

class ECommercePlatform:
    """Example e-commerce platform using multiple backends."""
    
    def __init__(self, config_path: str):
        # Load configuration
        self.config = Config(config_path)
        factory = DatabaseFactory()
        
        # Register factory for cleaner config
        self.config.register_factory("database", factory)
        
        # Initialize different backends for different purposes
        self.product_db = self.config.get_instance("databases", "products")
        self.order_db = self.config.get_instance("databases", "orders")
        self.cache_db = self.config.get_instance("databases", "cache")
        self.search_db = self.config.get_instance("databases", "search")
        self.archive_db = self.config.get_instance("databases", "archive")
    
    def add_product(self, product_data: dict) -> str:
        """Add a new product."""
        product = Record(product_data)
        product.metadata["created_at"] = datetime.utcnow().isoformat()
        
        # Store in primary database
        product_id = self.product_db.create(product)
        
        # Index for search
        self.search_db.create(product)
        
        # Cache popular products
        if product_data.get("featured", False):
            self.cache_db.create(product)
        
        return product_id
    
    def search_products(self, query_text: str, category: str = None):
        """Search for products."""
        # Build search query
        query = Query()
        
        if query_text:
            query = query.filter("name", "LIKE", f"%{query_text}%")
        
        if category:
            query = query.filter("category", "=", category)
        
        # Search in Elasticsearch for best performance
        return self.search_db.search(query)
    
    def get_product(self, product_id: str):
        """Get product with caching."""
        # Check cache first
        cached = self.cache_db.read(product_id)
        if cached:
            return cached
        
        # Get from primary database
        product = self.product_db.read(product_id)
        
        if product:
            # Update cache
            self.cache_db.create(product)
        
        return product
    
    def place_order(self, order_data: dict) -> str:
        """Place a new order."""
        order = Record(order_data)
        order.metadata["created_at"] = datetime.utcnow().isoformat()
        order.metadata["status"] = "pending"
        
        # Store order
        order_id = self.order_db.create(order)
        
        # Update product inventory
        for item in order_data.get("items", []):
            product = self.product_db.read(item["product_id"])
            if product:
                current_stock = product.get_value("stock", 0)
                product.fields["stock"] = current_stock - item["quantity"]
                self.product_db.update(item["product_id"], product)
        
        return order_id
    
    def archive_old_orders(self, days: int = 365):
        """Archive orders older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Find old orders
        old_orders = self.order_db.search(
            Query().filter("created_at", "<", cutoff.isoformat())
        )
        
        print(f"Archiving {len(old_orders)} old orders...")
        
        # Move to archive (S3)
        for order in old_orders:
            # Add archive metadata
            order.metadata["archived_at"] = datetime.utcnow().isoformat()
            order.metadata["archived_from"] = "orders"
            
            # Store in archive
            self.archive_db.create(order)
            
            # Remove from primary database
            self.order_db.delete(order.metadata["id"])
        
        return len(old_orders)

# Configuration file for the platform
config = {
    "databases": [
        {
            "name": "products",
            "factory": "database",
            "backend": "postgres",
            "host": "localhost",
            "database": "ecommerce",
            "table": "products"
        },
        {
            "name": "orders",
            "factory": "database",
            "backend": "postgres",
            "host": "localhost",
            "database": "ecommerce",
            "table": "orders"
        },
        {
            "name": "cache",
            "factory": "database",
            "backend": "memory"
        },
        {
            "name": "search",
            "factory": "database",
            "backend": "elasticsearch",
            "hosts": ["localhost:9200"],
            "index": "products"
        },
        {
            "name": "archive",
            "factory": "database",
            "backend": "s3",
            "bucket": "ecommerce-archive",
            "prefix": "orders/"
        }
    ]
}

# Save config and use platform
with open("ecommerce_config.yaml", "w") as f:
    import yaml
    yaml.dump(config, f)

platform = ECommercePlatform("ecommerce_config.yaml")
```

## Data Migration

### Migrating Between Backends
```python
from dataknobs_data import DatabaseFactory, Query, Record
from datetime import datetime
import json

def migrate_data(source_config: dict, dest_config: dict, 
                  transform_fn=None, batch_size: int = 100):
    """
    Migrate data between different backends.
    
    Args:
        source_config: Configuration for source database
        dest_config: Configuration for destination database
        transform_fn: Optional function to transform records
        batch_size: Number of records to process at once
    """
    factory = DatabaseFactory()
    
    # Create source and destination databases
    source_db = factory.create(**source_config)
    dest_db = factory.create(**dest_config)
    
    print(f"Starting migration from {source_config['backend']} "
          f"to {dest_config['backend']}...")
    
    # Get total count
    total = source_db.count()
    print(f"Total records to migrate: {total}")
    
    # Migrate in batches
    offset = 0
    migrated = 0
    
    while offset < total:
        # Get batch of records
        query = Query().limit(batch_size).offset(offset)
        batch = source_db.search(query)
        
        if not batch:
            break
        
        # Transform records if needed
        if transform_fn:
            batch = [transform_fn(record) for record in batch]
        
        # Add migration metadata
        for record in batch:
            record.metadata["migrated_at"] = datetime.utcnow().isoformat()
            record.metadata["migrated_from"] = source_config["backend"]
        
        # Batch create in destination
        dest_db.batch_create(batch)
        
        migrated += len(batch)
        offset += batch_size
        
        print(f"Progress: {migrated}/{total} records migrated")
    
    print(f"Migration complete! Migrated {migrated} records")
    
    # Verify migration
    dest_count = dest_db.count()
    if dest_count == total:
        print("✅ Verification passed: counts match")
    else:
        print(f"⚠️ Warning: source had {total} records, "
              f"destination has {dest_count}")
    
    return migrated

# Example: Migrate from JSON file to PostgreSQL
source = {
    "backend": "file",
    "path": "data.json",
    "format": "json"
}

destination = {
    "backend": "postgres",
    "host": "localhost",
    "database": "production",
    "user": "dbuser",
    "password": "dbpass"
}

# Define transformation function
def add_timestamps(record: Record) -> Record:
    """Add timestamps to records during migration."""
    if "created_at" not in record.fields:
        record.fields["created_at"] = datetime.utcnow().isoformat()
    record.fields["last_modified"] = datetime.utcnow().isoformat()
    return record

# Run migration
migrate_data(source, destination, transform_fn=add_timestamps)
```

## Testing with Different Backends

### Parameterized Testing
```python
import pytest
from dataknobs_data import DatabaseFactory, Record, Query

# Test with multiple backends
@pytest.fixture(params=["memory", "file"])
def database(request, tmp_path):
    """Create database for testing."""
    factory = DatabaseFactory()
    
    if request.param == "memory":
        return factory.create(backend="memory")
    elif request.param == "file":
        return factory.create(
            backend="file",
            path=str(tmp_path / "test.json")
        )

class TestDatabaseOperations:
    """Test database operations across different backends."""
    
    def test_create_and_read(self, database):
        """Test creating and reading records."""
        record = Record({"name": "Test", "value": 42})
        record_id = database.create(record)
        
        retrieved = database.read(record_id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "Test"
        assert retrieved.get_value("value") == 42
    
    def test_update(self, database):
        """Test updating records."""
        record = Record({"name": "Original"})
        record_id = database.create(record)
        
        record.fields["name"] = "Updated"
        success = database.update(record_id, record)
        assert success
        
        retrieved = database.read(record_id)
        assert retrieved.get_value("name") == "Updated"
    
    def test_search(self, database):
        """Test searching records."""
        # Create test data
        records = [
            Record({"type": "A", "value": 10}),
            Record({"type": "B", "value": 20}),
            Record({"type": "A", "value": 30}),
        ]
        
        for record in records:
            database.create(record)
        
        # Search for type A
        query = Query().filter("type", "=", "A")
        results = database.search(query)
        assert len(results) == 2
        
        # Search with value filter
        query = Query().filter("value", ">", 15)
        results = database.search(query)
        assert len(results) == 2
    
    def test_batch_operations(self, database):
        """Test batch operations."""
        records = [
            Record({"id": f"item-{i}", "value": i})
            for i in range(10)
        ]
        
        # Batch create
        record_ids = database.batch_create(records)
        assert len(record_ids) == 10
        
        # Batch read
        retrieved = database.batch_read(record_ids[:5])
        assert len(retrieved) == 5
        
        # Batch delete
        results = database.batch_delete(record_ids[5:])
        assert sum(results) == 5
        assert database.count() == 5
```

## Real-Time Data Processing

### Stream Processing Example
```python
import asyncio
from dataknobs_data import DatabaseFactory, Record
from datetime import datetime
import random

class DataStreamProcessor:
    """Process real-time data streams."""
    
    def __init__(self):
        factory = DatabaseFactory()
        
        # Hot storage for recent data
        self.hot_storage = factory.create(backend="memory")
        
        # Warm storage for processed data
        self.warm_storage = factory.create(
            backend="elasticsearch",
            hosts=["localhost:9200"],
            index="processed_data"
        )
        
        # Cold storage for archives
        self.cold_storage = factory.create(
            backend="s3",
            bucket="data-archive",
            prefix="streams/"
        )
    
    async def process_stream(self, stream_id: str):
        """Process incoming data stream."""
        buffer = []
        buffer_size = 100
        
        async for data in self.generate_stream():
            # Create record
            record = Record({
                "stream_id": stream_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
                "processed": False
            })
            
            # Store in hot storage immediately
            self.hot_storage.create(record)
            buffer.append(record)
            
            # Process in batches
            if len(buffer) >= buffer_size:
                await self.process_batch(buffer)
                buffer = []
        
        # Process remaining
        if buffer:
            await self.process_batch(buffer)
    
    async def process_batch(self, records: list):
        """Process a batch of records."""
        processed_records = []
        
        for record in records:
            # Simulate processing
            processed = await self.process_record(record)
            processed_records.append(processed)
        
        # Move to warm storage
        self.warm_storage.batch_create(processed_records)
        
        # Remove from hot storage
        for record in records:
            self.hot_storage.delete(record.metadata["id"])
        
        print(f"Processed batch of {len(records)} records")
    
    async def process_record(self, record: Record) -> Record:
        """Process individual record."""
        # Simulate processing time
        await asyncio.sleep(0.01)
        
        # Mark as processed
        record.fields["processed"] = True
        record.fields["processed_at"] = datetime.utcnow().isoformat()
        
        # Add computed fields
        if "value" in record.fields:
            record.fields["doubled"] = record.get_value("value") * 2
        
        return record
    
    async def generate_stream(self):
        """Generate simulated data stream."""
        for i in range(1000):
            yield {
                "value": random.randint(1, 100),
                "type": random.choice(["A", "B", "C"]),
                "sensor_id": f"sensor-{random.randint(1, 10)}"
            }
            await asyncio.sleep(0.1)  # Simulate real-time data
    
    def archive_old_data(self, days: int = 7):
        """Move old data to cold storage."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Find old records in warm storage
        old_records = self.warm_storage.search(
            Query().filter("timestamp", "<", cutoff.isoformat())
        )
        
        if old_records:
            # Move to cold storage
            self.cold_storage.batch_create(old_records)
            
            # Remove from warm storage
            for record in old_records:
                self.warm_storage.delete(record.metadata["id"])
            
            print(f"Archived {len(old_records)} old records")

# Run the stream processor
processor = DataStreamProcessor()
asyncio.run(processor.process_stream("stream-001"))
```

## Advanced Configuration Example

### Multi-Environment Setup
```python
import os
from dataknobs_config import Config
from dataknobs_data import database_factory

class ApplicationDatabase:
    """Application database with environment-specific configuration."""
    
    def __init__(self):
        self.env = os.environ.get("APP_ENV", "development")
        self.config = self._load_config()
        self.db = self._create_database()
    
    def _load_config(self):
        """Load environment-specific configuration."""
        config = Config()
        
        # Base configuration
        base_config = {
            "app_name": "MyApp",
            "version": "1.0.0"
        }
        
        # Environment-specific configs
        env_configs = {
            "development": {
                "database": {
                    "backend": "memory"
                }
            },
            "testing": {
                "database": {
                    "backend": "file",
                    "path": "/tmp/test_data.json"
                }
            },
            "staging": {
                "database": {
                    "backend": "postgres",
                    "host": "${DB_HOST:staging-db.example.com}",
                    "database": "${DB_NAME:staging}",
                    "user": "${DB_USER}",
                    "password": "${DB_PASSWORD}"
                }
            },
            "production": {
                "database": {
                    "backend": "postgres",
                    "host": "${DB_HOST}",
                    "database": "${DB_NAME}",
                    "user": "${DB_USER}",
                    "password": "${DB_PASSWORD}",
                    "pool_size": 50,
                    "ssl_mode": "require"
                }
            }
        }
        
        # Merge configurations
        config.load(base_config)
        config.merge(env_configs.get(self.env, {}))
        
        return config
    
    def _create_database(self):
        """Create database based on configuration."""
        db_config = self.config.get("database")
        return database_factory.create(**db_config)
    
    def get_database(self):
        """Get the configured database instance."""
        return self.db
    
    def health_check(self):
        """Check database health."""
        try:
            count = self.db.count()
            return {
                "status": "healthy",
                "environment": self.env,
                "backend": self.config.get("database.backend"),
                "record_count": count
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "environment": self.env,
                "error": str(e)
            }

# Usage
app_db = ApplicationDatabase()
db = app_db.get_database()
print(app_db.health_check())
```

These examples demonstrate the flexibility and power of the dataknobs-data package across various use cases and scenarios.