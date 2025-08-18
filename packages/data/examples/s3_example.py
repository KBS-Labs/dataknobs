#!/usr/bin/env python3
"""
Example demonstrating S3 backend usage with DataKnobs.

This example shows:
1. S3 backend configuration
2. CRUD operations with S3
3. Batch operations for performance
4. Using LocalStack for testing
"""

import os
from datetime import datetime

from dataknobs_data import Record, Query, DatabaseFactory
from dataknobs_config import Config


def setup_s3_backend_direct():
    """Set up S3 backend directly using factory."""
    factory = DatabaseFactory()
    
    # For testing with LocalStack (docker run -p 4566:4566 localstack/localstack)
    if os.environ.get("USE_LOCALSTACK"):
        db = factory.create(
            backend="s3",
            bucket="test-bucket",
            prefix="records/",
            region="us-east-1",
            endpoint_url="http://localhost:4566",
            access_key_id="test",
            secret_access_key="test"
        )
    else:
        # Production S3 (uses IAM role or environment credentials)
        db = factory.create(
            backend="s3",
            bucket=os.environ.get("S3_BUCKET", "my-data-bucket"),
            prefix=os.environ.get("S3_PREFIX", "records/"),
            region=os.environ.get("AWS_REGION", "us-east-1")
        )
    
    return db


def setup_s3_backend_config():
    """Set up S3 backend using configuration."""
    config = Config()
    
    # Configuration with environment variables
    config.load({
        "databases": [{
            "name": "s3_storage",
            "class": "dataknobs_data.backends.s3.S3Database",
            "bucket": "${S3_BUCKET:my-data-bucket}",
            "prefix": "${S3_PREFIX:records/}",
            "region": "${AWS_REGION:us-east-1}",
            "endpoint_url": "${S3_ENDPOINT:}",  # Empty for production
            "max_workers": "${S3_MAX_WORKERS:10}"  # Parallel operations
        }]
    })
    
    return config.get_instance("databases", "s3_storage")


def demonstrate_s3_operations(db):
    """Demonstrate S3-specific operations."""
    print("=== S3 Backend Operations ===\n")
    
    # Create some test data
    records = []
    for i in range(10):
        records.append(Record({
            "id": f"user_{i:03d}",
            "name": f"User {i}",
            "email": f"user{i}@example.com",
            "score": 100 * (i + 1),
            "active": i % 2 == 0,
            "created_at": datetime.utcnow().isoformat()
        }))
    
    # Batch create for efficiency
    print("Batch creating 10 records...")
    start = datetime.utcnow()
    record_ids = db.batch_create(records)
    elapsed = (datetime.utcnow() - start).total_seconds()
    print(f"Created {len(record_ids)} records in {elapsed:.2f} seconds")
    print(f"Average: {elapsed/len(record_ids)*1000:.1f}ms per record\n")
    
    # Batch read
    print("Batch reading records...")
    start = datetime.utcnow()
    retrieved = db.batch_read(record_ids[:5])
    elapsed = (datetime.utcnow() - start).total_seconds()
    print(f"Read {len(retrieved)} records in {elapsed:.2f} seconds")
    
    # List all objects (uses S3 listing)
    print(f"\nTotal objects in S3: {db.count()}")
    
    # Search (note: requires downloading all objects)
    print("\nSearching for active users with score > 500...")
    query = (Query()
        .filter("active", "=", True)
        .filter("score", ">", 500)
        .sort("score", "DESC"))
    
    results = db.search(query)
    print(f"Found {len(results)} matching records:")
    for result in results:
        print(f"  - {result.get_value('name')}: score={result.get_value('score')}")
    
    # Update with metadata
    print("\nUpdating a record with metadata...")
    if record_ids:
        record = db.read(record_ids[0])
        record.metadata["last_modified_by"] = "s3_example.py"
        record.metadata["version"] = "2"
        success = db.update(record_ids[0], record)
        print(f"Update successful: {success}")
    
    # Batch delete
    print("\nBatch deleting records...")
    start = datetime.utcnow()
    results = db.batch_delete(record_ids[5:])
    elapsed = (datetime.utcnow() - start).total_seconds()
    deleted = sum(results)
    print(f"Deleted {deleted} records in {elapsed:.2f} seconds")
    
    return record_ids[:5]  # Return remaining IDs


def demonstrate_s3_cost_optimization(db):
    """Demonstrate cost optimization strategies."""
    print("\n=== S3 Cost Optimization ===\n")
    
    # 1. Use batch operations
    print("1. Batch Operations:")
    print("   - Batch create/read/delete reduces API calls")
    print("   - Parallel processing with ThreadPoolExecutor")
    print(f"   - Current max_workers: {db.max_workers}")
    
    # 2. Efficient searching
    print("\n2. Search Optimization:")
    print("   - S3 doesn't support native queries")
    print("   - Consider using S3 Select for large objects")
    print("   - Or use Elasticsearch for complex searches")
    print("   - Cache frequently accessed data locally")
    
    # 3. Object organization
    print("\n3. Object Organization:")
    print(f"   - Current prefix: {db.prefix}")
    print("   - Use prefixes for logical grouping")
    print("   - Enables efficient listing with prefix filters")
    
    # 4. Lifecycle policies
    print("\n4. Lifecycle Management:")
    print("   - Set up S3 lifecycle policies for old data")
    print("   - Move to Glacier for long-term storage")
    print("   - Auto-delete temporary data")


def main():
    """Run the S3 example."""
    print("DataKnobs S3 Backend Example")
    print("=" * 50 + "\n")
    
    # Check if we should use LocalStack
    if os.environ.get("USE_LOCALSTACK"):
        print("📦 Using LocalStack for S3 (http://localhost:4566)")
        print("   Run: docker run -p 4566:4566 localstack/localstack\n")
    else:
        print("☁️  Using AWS S3 (or compatible service)")
        print("   Ensure AWS credentials are configured\n")
    
    try:
        # Set up S3 backend
        print("Setting up S3 backend...")
        db = setup_s3_backend_direct()
        # Or use: db = setup_s3_backend_config()
        
        print(f"Connected to bucket: {db.bucket}")
        print(f"Using prefix: {db.prefix}\n")
        
        # Run demonstrations
        remaining_ids = demonstrate_s3_operations(db)
        demonstrate_s3_cost_optimization(db)
        
        # Cleanup
        print("\n=== Cleanup ===")
        if remaining_ids:
            print(f"Cleaning up {len(remaining_ids)} remaining records...")
            db.batch_delete(remaining_ids)
        
        print("\n✅ S3 example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. For LocalStack: export USE_LOCALSTACK=1")
        print("2. For AWS: Configure credentials (aws configure)")
        print("3. Install boto3: pip install dataknobs-data[s3]")


if __name__ == "__main__":
    main()