#!/usr/bin/env python3
"""
Complete example demonstrating all features of the DataKnobs data package.

This example shows:
1. Environment variable substitution
2. Factory registration
3. Multiple backend usage
4. Configuration-based instantiation
5. Data migration between backends
"""

import os
import tempfile
from datetime import datetime

# Set some environment variables for the example
os.environ["DB_BACKEND"] = "memory"  # Change to "postgres" for real DB
os.environ["S3_BUCKET"] = "example-bucket"
os.environ["ARCHIVE_PATH"] = "/tmp/archive.json"

from dataknobs_config import Config
from dataknobs_data import (
    Record,
    Query,
    database_factory,
    DatabaseFactory
)


def setup_configuration():
    """Set up configuration with multiple backends."""
    
    # Create config instance
    config = Config()
    
    # Register the database factory for cleaner configs
    config.register_factory("database", database_factory)
    
    # Load configuration with environment variable substitution
    config.load({
        "databases": [
            {
                "name": "primary",
                "factory": "database",  # Uses registered factory
                "backend": "${DB_BACKEND:postgres}",
                "host": "${DB_HOST:localhost}",
                "port": "${DB_PORT:5432}",
                "database": "${DB_NAME:myapp}",
                "user": "${DB_USER:user}",
                "password": "${DB_PASSWORD:pass}"
            },
            {
                "name": "cache",
                "factory": "database",
                "backend": "memory"
            },
            {
                "name": "archive",
                "factory": "database",
                "backend": "file",
                "path": "${ARCHIVE_PATH:/data/archive.json}",
                "format": "json"
            },
            {
                "name": "cloud",
                "factory": "database",
                "backend": "s3",
                "bucket": "${S3_BUCKET}",
                "prefix": "${S3_PREFIX:records/}",
                "region": "${AWS_REGION:us-east-1}",
                "endpoint_url": "${LOCALSTACK_ENDPOINT:}"  # Empty default for production
            }
        ]
    })
    
    return config


def demonstrate_crud_operations(db, db_name):
    """Demonstrate CRUD operations on a database."""
    print(f"\n=== {db_name} Database Operations ===")
    
    # Create records
    records = [
        Record({
            "name": "Alice Johnson",
            "age": 28,
            "department": "Engineering",
            "salary": 95000,
            "active": True
        }),
        Record({
            "name": "Bob Smith",
            "age": 35,
            "department": "Sales",
            "salary": 75000,
            "active": True
        }),
        Record({
            "name": "Charlie Brown",
            "age": 42,
            "department": "Engineering",
            "salary": 105000,
            "active": False
        })
    ]
    
    # Create records
    print(f"Creating {len(records)} records...")
    record_ids = []
    for record in records:
        record_id = db.create(record)
        record_ids.append(record_id)
        print(f"  Created: {record_id}")
    
    # Read a record
    print(f"\nReading record {record_ids[0]}...")
    retrieved = db.read(record_ids[0])
    if retrieved:
        print(f"  Name: {retrieved.get_value('name')}")
        print(f"  Age: {retrieved.get_value('age')}")
        print(f"  Department: {retrieved.get_value('department')}")
    
    # Update a record
    print(f"\nUpdating record {record_ids[1]}...")
    updated_record = Record({
        "name": "Bob Smith",
        "age": 36,  # Birthday!
        "department": "Sales",
        "salary": 80000,  # Raise!
        "active": True
    })
    success = db.update(record_ids[1], updated_record)
    print(f"  Update successful: {success}")
    
    # Search with queries
    print("\nSearching for Engineering employees...")
    query = Query().filter("department", "=", "Engineering").filter("active", "=", True)
    results = db.search(query)
    print(f"  Found {len(results)} active engineers")
    for result in results:
        print(f"    - {result.get_value('name')}: ${result.get_value('salary')}")
    
    # Complex query
    print("\nSearching for high earners...")
    query = (Query()
        .filter("salary", ">=", 80000)
        .sort("salary", "DESC")
        .limit(2))
    results = db.search(query)
    print(f"  Top {len(results)} earners:")
    for result in results:
        print(f"    - {result.get_value('name')}: ${result.get_value('salary')}")
    
    # Count records
    total = db.count()
    print(f"\nTotal records in {db_name}: {total}")
    
    return record_ids


def demonstrate_migration(source_db, dest_db, source_name, dest_name):
    """Demonstrate migrating data between backends."""
    print(f"\n=== Migrating from {source_name} to {dest_name} ===")
    
    # Get all records from source
    all_records = source_db.search(Query())
    print(f"Found {len(all_records)} records to migrate")
    
    # Migrate with transformation
    migrated = 0
    for record in all_records:
        # Add migration metadata
        record.metadata["migrated_from"] = source_name
        record.metadata["migrated_at"] = datetime.utcnow().isoformat()
        
        # Create in destination
        dest_db.create(record)
        migrated += 1
    
    print(f"Successfully migrated {migrated} records")
    
    # Verify migration
    dest_count = dest_db.count()
    print(f"Destination now has {dest_count} records")


def demonstrate_factory_info():
    """Show information about available backends."""
    print("\n=== Backend Information ===")
    
    factory = DatabaseFactory()
    
    backends = ["memory", "file", "postgres", "elasticsearch", "s3"]
    for backend in backends:
        info = factory.get_backend_info(backend)
        print(f"\n{backend.upper()}:")
        print(f"  Description: {info.get('description', 'N/A')}")
        print(f"  Persistent: {info.get('persistent', 'N/A')}")
        if info.get('requires_install'):
            print(f"  Install: {info['requires_install']}")


def main():
    """Run the complete example."""
    print("DataKnobs Data Package - Complete Example")
    print("=" * 50)
    
    # Set up configuration
    config = setup_configuration()
    
    # Show registered factories
    factories = config.get_registered_factories()
    print(f"\nRegistered factories: {list(factories.keys())}")
    
    # Get database instances
    # Note: For this example, we'll use memory/file backends
    # In production, you could use postgres/elasticsearch/s3
    primary_db = config.get_instance("databases", "primary")  # Will be memory due to env var
    cache_db = config.get_instance("databases", "cache")
    
    # For archive, create a real temp file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        os.environ["ARCHIVE_PATH"] = f.name
    config = setup_configuration()  # Reload with new path
    archive_db = config.get_instance("databases", "archive")
    
    # Note: S3 backend would require boto3 and either real AWS or LocalStack
    # cloud_db = config.get_instance("databases", "cloud")
    
    # Demonstrate operations on primary database
    record_ids = demonstrate_crud_operations(primary_db, "Primary")
    
    # Demonstrate caching
    print("\n=== Caching Strategy ===")
    print("Copying frequently accessed records to cache...")
    query = Query().filter("active", "=", True)
    active_records = primary_db.search(query)
    for record in active_records:
        cache_db.create(record)
    print(f"Cached {len(active_records)} active records")
    
    # Demonstrate archival
    demonstrate_migration(primary_db, archive_db, "Primary", "Archive")
    
    # Show backend information
    demonstrate_factory_info()
    
    # Clean up
    print("\n=== Cleanup ===")
    primary_db.clear()
    cache_db.clear()
    archive_db.clear()
    print("All databases cleared")
    
    # Clean up temp file
    if os.path.exists(os.environ.get("ARCHIVE_PATH", "")):
        os.unlink(os.environ["ARCHIVE_PATH"])
        print("Temporary archive file removed")
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()