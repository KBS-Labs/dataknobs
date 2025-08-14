# S3 Backend

The S3 backend provides scalable object storage using AWS S3 or S3-compatible services. It's ideal for archival, backup, and storing large datasets with unlimited capacity.

## Features

- 🚀 **Parallel Operations**: Uses ThreadPoolExecutor for batch operations
- 🏷️ **Metadata Support**: Stores record metadata as S3 object tags
- 💾 **Index Caching**: Caches object listings for performance
- 🔧 **S3-Compatible**: Works with AWS S3, MinIO, LocalStack
- 📦 **Multipart Upload**: Automatic for large objects
- 💰 **Cost Optimized**: Efficient batching reduces API calls

## Installation

```bash
pip install dataknobs-data[s3]
```

## Configuration

### Basic Configuration
```python
from dataknobs_data.backends.s3 import S3Database

db = S3Database.from_config({
    "bucket": "my-data-bucket",
    "prefix": "records/",
    "region": "us-east-1"
})
```

### With Environment Variables
```yaml
# config.yaml
databases:
  - name: archive
    class: dataknobs_data.backends.s3.S3Database
    bucket: ${S3_BUCKET}
    prefix: ${S3_PREFIX:data/}
    region: ${AWS_REGION:us-east-1}
    endpoint_url: ${S3_ENDPOINT:}  # Empty for AWS
    max_workers: ${S3_MAX_WORKERS:10}
```

### Using LocalStack for Testing
```python
# For local development with LocalStack
db = S3Database.from_config({
    "bucket": "test-bucket",
    "prefix": "records/",
    "region": "us-east-1",
    "endpoint_url": "http://localhost:4566",
    "access_key_id": "test",
    "secret_access_key": "test"
})
```

Start LocalStack:
```bash
docker run -p 4566:4566 localstack/localstack
```

## Usage Examples

### Basic CRUD Operations
```python
from dataknobs_data import Record, Query
from dataknobs_data.backends.s3 import S3Database

# Initialize
db = S3Database.from_config({
    "bucket": "my-bucket",
    "prefix": "app/records/"
})

# Create
record = Record({
    "id": "user-123",
    "name": "Alice",
    "email": "alice@example.com",
    "created_at": "2024-01-01T00:00:00Z"
})
record_id = db.create(record)

# Read
retrieved = db.read(record_id)
print(f"Name: {retrieved.get_value('name')}")

# Update
record.fields["email"] = "alice.smith@example.com"
db.update(record_id, record)

# Delete
db.delete(record_id)
```

### Batch Operations
```python
# Batch operations are much more efficient
records = [
    Record({"id": f"user-{i}", "name": f"User {i}"})
    for i in range(100)
]

# Parallel upload (10 workers by default)
record_ids = db.batch_create(records)

# Parallel read
retrieved = db.batch_read(record_ids[:50])

# Parallel delete
results = db.batch_delete(record_ids[50:])
```

### Metadata and Tags
```python
# S3 backend stores metadata as object tags
record = Record({
    "name": "Important Document",
    "content": "..."
})

# Metadata is preserved
record.metadata["created_by"] = "admin"
record.metadata["version"] = "1.0"
record.metadata["department"] = "Engineering"

record_id = db.create(record)

# Tags are visible in S3 console
# Tags: id=xxx, created_at=xxx, created_by=admin, version=1.0, department=Engineering
```

### Searching and Filtering
```python
# Note: S3 doesn't support native queries
# All objects are downloaded and filtered locally
query = Query()\
    .filter("department", "=", "Engineering")\
    .filter("active", "=", True)\
    .sort("created_at", "DESC")\
    .limit(10)

results = db.search(query)
# This downloads ALL objects, then filters locally
# Consider using Elasticsearch for complex queries
```

## Performance Optimization

### 1. Batch Operations
```python
# Bad: Individual operations
for record in records:
    db.create(record)  # One API call each

# Good: Batch operations
db.batch_create(records)  # Parallel API calls
```

### 2. Prefix Organization
```python
# Organize data with prefixes for efficient listing
db = S3Database.from_config({
    "bucket": "my-bucket",
    "prefix": "data/2024/01/"  # Date-based organization
})

# Enables efficient listing of specific time periods
```

### 3. Index Caching
```python
# The S3 backend caches object listings
db.count()  # First call: lists all objects
db.count()  # Subsequent calls: uses cache

# Cache expires after 60 seconds by default
# Configurable via cache_ttl parameter
```

### 4. Multipart Upload
```python
# Large objects automatically use multipart upload
large_record = Record({
    "id": "large-file",
    "data": "x" * (10 * 1024 * 1024)  # 10MB
})

# Automatically uses multipart upload for objects > 5MB
db.create(large_record)
```

## Cost Optimization

### API Call Reduction
```python
# Use batch operations to minimize API calls
# Cost: $0.0004 per 1,000 PUT requests

# Bad: 1000 individual creates = 1000 PUT requests
for record in records:
    db.create(record)  # $0.40

# Good: Batch create = ~100 PUT requests (with 10 workers)
db.batch_create(records)  # $0.04
```

### Storage Classes
```python
# Configure storage class for cost optimization
db = S3Database.from_config({
    "bucket": "archive-bucket",
    "prefix": "archive/",
    "storage_class": "GLACIER"  # For infrequent access
})
```

### Lifecycle Policies
```python
# Set up lifecycle policies in AWS Console or via boto3
# Example: Move to Glacier after 30 days
lifecycle_config = {
    'Rules': [{
        'ID': 'ArchiveOldData',
        'Status': 'Enabled',
        'Transitions': [{
            'Days': 30,
            'StorageClass': 'GLACIER'
        }]
    }]
}
```

## Limitations and Considerations

### Query Performance
- S3 doesn't support native queries
- All filtering happens client-side
- Consider using S3 Select for large objects
- Use Elasticsearch for complex queries

### Consistency Model
- S3 provides strong read-after-write consistency
- List operations may have slight delay
- Use index caching carefully in high-write scenarios

### Size Limits
- Maximum object size: 5TB
- Multipart upload required for objects > 5GB
- Maximum 10,000 parts per multipart upload

## Advanced Features

### S3 Select Integration (Future)
```python
# Future enhancement: Use S3 Select for efficient queries
query = "SELECT * FROM S3Object WHERE department = 'Engineering'"
results = db.select_query(query)  # Coming soon
```

### Cross-Region Replication
```python
# Configure in AWS Console for disaster recovery
# Primary bucket: us-east-1
# Replica bucket: eu-west-1
```

### Versioning
```python
# Enable versioning in S3 bucket
# Useful for audit trails and recovery
```

## Testing with LocalStack

```python
import os
from dataknobs_data.backends.s3 import S3Database

# Set environment for LocalStack
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"

# Create test database
test_db = S3Database.from_config({
    "bucket": "test-bucket",
    "prefix": "test/",
    "region": "us-east-1",
    "endpoint_url": "http://localhost:4566"
})

# Run tests
record = Record({"test": "data"})
record_id = test_db.create(record)
assert test_db.read(record_id) is not None
test_db.delete(record_id)
```

## Migration from Other Backends

```python
from dataknobs_data import DatabaseFactory

factory = DatabaseFactory()

# Source database (e.g., PostgreSQL)
source = factory.create(
    backend="postgres",
    host="localhost",
    database="production"
)

# Destination S3 archive
archive = factory.create(
    backend="s3",
    bucket="archive-bucket",
    prefix="postgres-backup/"
)

# Migrate old records
from datetime import datetime, timedelta
cutoff = datetime.now() - timedelta(days=365)

old_records = source.search(
    Query().filter("created_at", "<", cutoff.isoformat())
)

# Batch upload to S3
archive.batch_create(old_records)

# Delete from source
for record in old_records:
    source.delete(record.metadata["id"])

print(f"Archived {len(old_records)} records to S3")
```

## Best Practices

1. **Use prefixes** for logical organization
2. **Batch operations** whenever possible
3. **Cache frequently accessed objects** locally
4. **Monitor costs** with AWS Cost Explorer
5. **Set up lifecycle policies** for old data
6. **Use versioning** for critical data
7. **Enable server-side encryption**
8. **Configure bucket policies** for security
9. **Use CloudFront** for global distribution
10. **Implement retry logic** for resilience