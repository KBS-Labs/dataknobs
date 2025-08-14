# S3 Storage Example

This example demonstrates how to use AWS S3 as a storage backend with the dataknobs-data package.

## Complete S3 Example

```python
#!/usr/bin/env python3
"""
Comprehensive S3 storage example with DataKnobs.

This example demonstrates:
- S3 configuration with environment variables
- Batch operations for performance
- Cost optimization strategies
- LocalStack for local testing
- Migration from other backends to S3
"""

import os
import json
import time
from datetime import datetime, timedelta
from dataknobs_data import Record, Query, DatabaseFactory
from dataknobs_config import Config

class S3StorageDemo:
    """Demonstrate S3 storage capabilities."""
    
    def __init__(self, use_localstack=False):
        """
        Initialize S3 storage demo.
        
        Args:
            use_localstack: Use LocalStack for local testing
        """
        self.factory = DatabaseFactory()
        self.use_localstack = use_localstack
        self.s3_db = self._setup_s3_backend()
    
    def _setup_s3_backend(self):
        """Set up S3 backend based on environment."""
        if self.use_localstack:
            print("📦 Using LocalStack for S3 testing")
            return self.factory.create(
                backend="s3",
                bucket="demo-bucket",
                prefix="dataknobs/",
                region="us-east-1",
                endpoint_url="http://localhost:4566",
                access_key_id="test",
                secret_access_key="test",
                max_workers=5  # Fewer workers for local testing
            )
        else:
            print("☁️  Using AWS S3")
            return self.factory.create(
                backend="s3",
                bucket=os.environ.get("S3_BUCKET", "dataknobs-demo"),
                prefix=os.environ.get("S3_PREFIX", "data/"),
                region=os.environ.get("AWS_REGION", "us-east-1"),
                max_workers=10  # More workers for production
            )
    
    def demonstrate_basic_operations(self):
        """Show basic CRUD operations with S3."""
        print("\n=== Basic S3 Operations ===")
        
        # Create a record
        document = Record({
            "id": "doc-001",
            "title": "Important Document",
            "content": "This is stored in S3",
            "size_kb": 42,
            "created_at": datetime.utcnow().isoformat()
        })
        
        # Add metadata (stored as S3 tags)
        document.metadata["author"] = "John Doe"
        document.metadata["department"] = "Engineering"
        document.metadata["version"] = "1.0"
        
        # Store in S3
        doc_id = self.s3_db.create(document)
        print(f"Stored document in S3: {doc_id}")
        
        # Read back
        retrieved = self.s3_db.read(doc_id)
        if retrieved:
            print(f"Retrieved: {retrieved.get_value('title')}")
            print(f"Metadata: {retrieved.metadata}")
        
        # Update
        retrieved.fields["version"] = "1.1"
        retrieved.fields["updated_at"] = datetime.utcnow().isoformat()
        success = self.s3_db.update(doc_id, retrieved)
        print(f"Update successful: {success}")
        
        # Delete
        deleted = self.s3_db.delete(doc_id)
        print(f"Deleted: {deleted}")
    
    def demonstrate_batch_operations(self):
        """Show efficient batch operations."""
        print("\n=== Batch Operations (Performance) ===")
        
        # Create test data
        batch_size = 50
        records = []
        for i in range(batch_size):
            records.append(Record({
                "id": f"batch-{i:04d}",
                "type": "test",
                "value": i * 100,
                "timestamp": datetime.utcnow().isoformat(),
                "category": f"cat-{i % 5}"  # 5 categories
            }))
        
        # Batch create with timing
        print(f"Creating {batch_size} records...")
        start = time.time()
        record_ids = self.s3_db.batch_create(records)
        elapsed = time.time() - start
        print(f"Created {len(record_ids)} records in {elapsed:.2f}s")
        print(f"Average: {elapsed/len(record_ids)*1000:.1f}ms per record")
        
        # Batch read
        print(f"\nReading {len(record_ids)//2} records...")
        start = time.time()
        retrieved = self.s3_db.batch_read(record_ids[:len(record_ids)//2])
        elapsed = time.time() - start
        print(f"Read {len(retrieved)} records in {elapsed:.2f}s")
        
        # Search (downloads all, filters locally)
        print("\nSearching for category 'cat-2'...")
        start = time.time()
        results = self.s3_db.search(
            Query().filter("category", "=", "cat-2")
        )
        elapsed = time.time() - start
        print(f"Found {len(results)} records in {elapsed:.2f}s")
        
        # Batch delete
        print(f"\nDeleting {len(record_ids)} records...")
        start = time.time()
        results = self.s3_db.batch_delete(record_ids)
        elapsed = time.time() - start
        deleted = sum(results)
        print(f"Deleted {deleted} records in {elapsed:.2f}s")
    
    def demonstrate_large_files(self):
        """Show handling of large files."""
        print("\n=== Large File Handling ===")
        
        # Create a large record (simulated)
        large_data = "x" * (1024 * 1024)  # 1MB of data
        
        large_record = Record({
            "id": "large-file-001",
            "type": "binary",
            "data": large_data,
            "size_mb": len(large_data) / (1024 * 1024),
            "checksum": hash(large_data)
        })
        
        print(f"Storing {large_record.get_value('size_mb'):.1f}MB file...")
        start = time.time()
        record_id = self.s3_db.create(large_record)
        elapsed = time.time() - start
        print(f"Stored in {elapsed:.2f}s")
        
        # Note: Files > 5MB automatically use multipart upload
        
        # Clean up
        self.s3_db.delete(record_id)
    
    def demonstrate_organization_strategies(self):
        """Show how to organize data with prefixes."""
        print("\n=== Data Organization with Prefixes ===")
        
        # Create S3 backends with different prefixes
        # This allows logical separation of data types
        
        configs = [
            ("logs/2024/01/", "Log entries"),
            ("users/active/", "Active user data"),
            ("backups/daily/", "Daily backups"),
            ("analytics/raw/", "Raw analytics data")
        ]
        
        for prefix, description in configs:
            db = self.factory.create(
                backend="s3",
                bucket=self.s3_db.bucket,
                prefix=prefix,
                region=self.s3_db.region,
                endpoint_url=self.s3_db.endpoint_url if self.use_localstack else None,
                access_key_id="test" if self.use_localstack else None,
                secret_access_key="test" if self.use_localstack else None
            )
            
            # Store sample data
            sample = Record({"type": description, "timestamp": datetime.utcnow().isoformat()})
            record_id = db.create(sample)
            
            print(f"Stored {description} at prefix: {prefix}")
            
            # Clean up
            db.delete(record_id)
    
    def demonstrate_cost_optimization(self):
        """Show cost optimization strategies."""
        print("\n=== Cost Optimization Strategies ===")
        
        print("1. Batch Operations:")
        print("   - Individual PUTs: $0.005 per 1,000 requests")
        print("   - Batch with 10 workers: 10x fewer requests")
        print(f"   - Current max_workers: {self.s3_db.max_workers}")
        
        print("\n2. Storage Classes:")
        print("   - Standard: $0.023/GB/month")
        print("   - Infrequent Access: $0.0125/GB/month")
        print("   - Glacier: $0.004/GB/month")
        
        print("\n3. Lifecycle Policies (set in AWS Console):")
        print("   - Move to IA after 30 days")
        print("   - Move to Glacier after 90 days")
        print("   - Delete after 365 days")
        
        print("\n4. Data Compression:")
        # Demonstrate compression
        import gzip
        import base64
        
        original = {"data": "x" * 1000}  # 1KB of repeated data
        compressed = base64.b64encode(
            gzip.compress(json.dumps(original).encode())
        ).decode()
        
        original_size = len(json.dumps(original))
        compressed_size = len(compressed)
        
        print(f"   - Original size: {original_size} bytes")
        print(f"   - Compressed size: {compressed_size} bytes")
        print(f"   - Compression ratio: {original_size/compressed_size:.1f}x")
    
    def demonstrate_migration_to_s3(self):
        """Show migrating data from other backends to S3."""
        print("\n=== Migration to S3 ===")
        
        # Create source database (memory for demo)
        source_db = self.factory.create(backend="memory")
        
        # Generate sample data
        print("Creating sample data in memory...")
        for i in range(20):
            source_db.create(Record({
                "id": f"migrate-{i:03d}",
                "data": f"Record {i}",
                "created": (datetime.utcnow() - timedelta(days=i)).isoformat()
            }))
        
        print(f"Source has {source_db.count()} records")
        
        # Migrate old records to S3
        cutoff = datetime.utcnow() - timedelta(days=7)
        old_records = source_db.search(
            Query().filter("created", "<", cutoff.isoformat())
        )
        
        print(f"Found {len(old_records)} records older than 7 days")
        
        if old_records:
            # Add migration metadata
            for record in old_records:
                record.metadata["migrated_at"] = datetime.utcnow().isoformat()
                record.metadata["source"] = "memory"
            
            # Batch upload to S3
            print("Migrating to S3...")
            migrated_ids = self.s3_db.batch_create(old_records)
            print(f"Migrated {len(migrated_ids)} records to S3")
            
            # Remove from source
            for record in old_records:
                source_db.delete(record.metadata["id"])
            
            print(f"Source now has {source_db.count()} records")
            print(f"S3 archive has {self.s3_db.count()} records")
            
            # Clean up S3
            self.s3_db.batch_delete(migrated_ids)

def main():
    """Run S3 storage demonstration."""
    print("DataKnobs S3 Storage Example")
    print("=" * 50)
    
    # Check environment
    use_localstack = os.environ.get("USE_LOCALSTACK", "").lower() == "true"
    
    if use_localstack:
        print("\n📝 Using LocalStack for testing")
        print("Start LocalStack with:")
        print("  docker run -p 4566:4566 localstack/localstack")
    else:
        print("\n☁️  Using AWS S3")
        print("Ensure AWS credentials are configured:")
        print("  aws configure")
    
    try:
        # Create demo instance
        demo = S3StorageDemo(use_localstack)
        
        # Run demonstrations
        demo.demonstrate_basic_operations()
        demo.demonstrate_batch_operations()
        demo.demonstrate_large_files()
        demo.demonstrate_organization_strategies()
        demo.demonstrate_cost_optimization()
        demo.demonstrate_migration_to_s3()
        
        print("\n" + "=" * 50)
        print("✅ S3 storage example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. For LocalStack: export USE_LOCALSTACK=true")
        print("2. For AWS: Ensure credentials are configured")
        print("3. Install S3 support: pip install dataknobs-data[s3]")

if __name__ == "__main__":
    main()
```

## Running the Example

### With LocalStack (Local Testing)

```bash
# Start LocalStack
docker run -d -p 4566:4566 --name localstack localstack/localstack

# Install S3 support
pip install dataknobs-data[s3]

# Run with LocalStack
export USE_LOCALSTACK=true
python s3_storage_example.py

# Stop LocalStack when done
docker stop localstack
docker rm localstack
```

### With AWS S3

```bash
# Configure AWS credentials
aws configure

# Set S3 bucket (must exist)
export S3_BUCKET=my-dataknobs-bucket
export S3_PREFIX=examples/
export AWS_REGION=us-east-1

# Install S3 support
pip install dataknobs-data[s3]

# Run with AWS
python s3_storage_example.py
```

### With MinIO (S3-Compatible)

```bash
# Start MinIO
docker run -d -p 9000:9000 -p 9001:9001 \
  --name minio \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"

# Configure for MinIO
export S3_ENDPOINT=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export S3_BUCKET=dataknobs

# Create bucket (using MinIO client or web console at http://localhost:9001)

# Run example
python s3_storage_example.py
```

## Key Concepts Demonstrated

1. **Configuration**: Environment variables and endpoint customization
2. **Batch Operations**: Parallel uploads/downloads for performance
3. **Metadata**: Using S3 tags for record metadata
4. **Large Files**: Automatic multipart upload for files >5MB
5. **Organization**: Using prefixes for logical data separation
6. **Cost Optimization**: Strategies to minimize S3 costs
7. **Migration**: Moving data from other backends to S3
8. **LocalStack**: Testing without AWS costs
9. **Error Handling**: Graceful handling of connection issues
10. **Performance Monitoring**: Timing operations for optimization

## Cost Considerations

### Storage Costs (AWS S3 Standard)
- First 50 TB: $0.023 per GB/month
- Next 450 TB: $0.022 per GB/month
- Over 500 TB: $0.021 per GB/month

### Request Costs
- PUT, COPY, POST, LIST: $0.005 per 1,000 requests
- GET, SELECT: $0.0004 per 1,000 requests
- DELETE: Free

### Data Transfer
- Transfer IN: Free
- Transfer OUT to Internet: $0.09 per GB (first 10 TB/month)
- Transfer OUT to same region: Free

### Optimization Tips
1. Use batch operations to reduce API calls
2. Enable S3 Intelligent-Tiering for automatic cost optimization
3. Set lifecycle policies for old data
4. Use S3 Select to reduce data transfer
5. Consider S3 Glacier for long-term archives
6. Monitor usage with AWS Cost Explorer