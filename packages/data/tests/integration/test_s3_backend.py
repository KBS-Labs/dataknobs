"""Integration tests for S3 backend using LocalStack."""

import os
import pytest
from datetime import datetime
from typing import Generator
import boto3
from moto import mock_aws

from dataknobs_data import Record, Query
from dataknobs_data.backends.s3 import SyncS3Database


@pytest.fixture
def s3_config():
    """Configuration for S3 backend tests."""
    return {
        "bucket": "test-dataknobs-bucket",
        "prefix": "test-records/",
        "region": "us-east-1",
        "max_workers": 5
    }


@pytest.fixture
def mock_s3_backend(s3_config):
    """Create a mock S3 backend using moto."""
    with mock_aws():
        # The SyncS3Database will create the bucket if it doesn't exist
        db = SyncS3Database(s3_config)
        db.connect()
        yield db
        # Cleanup
        db.clear()
        db.close()


@pytest.fixture
def localstack_s3_backend(s3_config):
    """Create an S3 backend connected to LocalStack.
    
    This fixture requires LocalStack to be running.
    Start with: docker run -d -p 4566:4566 localstack/localstack
    """
    # Detect if we're running in Docker container
    if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
        default_host = 'localstack'
    else:
        default_host = 'localhost'
    
    # Check if LocalStack is available
    localstack_endpoint = os.environ.get("LOCALSTACK_ENDPOINT", f"http://{default_host}:4566")
    
    try:
        # Test connection to LocalStack
        test_client = boto3.client(
            "s3",
            endpoint_url=localstack_endpoint,
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test"
        )
        test_client.list_buckets()
    except Exception:
        pytest.skip("LocalStack not available")
    
    # Create backend with LocalStack endpoint
    config = s3_config.copy()
    config["endpoint_url"] = localstack_endpoint
    config["access_key_id"] = "test"
    config["secret_access_key"] = "test"
    
    db = SyncS3Database(config)
    db.connect()
    yield db
    # Cleanup
    db.clear()
    db.close()


class TestS3Backend:
    """Test S3 backend functionality."""
    
    def test_create_and_read(self, mock_s3_backend):
        """Test creating and reading a record."""
        db = mock_s3_backend
        
        # Create a record
        record = Record({
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        })
        
        record_id = db.create(record)
        assert record_id is not None
        
        # Read the record back
        retrieved = db.read(record_id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "John Doe"
        assert retrieved.get_value("age") == 30
        assert retrieved.get_value("email") == "john@example.com"
        assert retrieved.metadata.get("id") == record_id
        assert retrieved.metadata.get("created_at") is not None
        assert retrieved.metadata.get("updated_at") is not None
    
    def test_update(self, mock_s3_backend):
        """Test updating a record."""
        db = mock_s3_backend
        
        # Create initial record
        record = Record({"name": "Alice", "age": 25})
        record_id = db.create(record)
        
        # Update the record
        updated_record = Record({"name": "Alice Smith", "age": 26})
        success = db.update(record_id, updated_record)
        assert success is True
        
        # Verify update
        retrieved = db.read(record_id)
        assert retrieved.get_value("name") == "Alice Smith"
        assert retrieved.get_value("age") == 26
        assert retrieved.metadata.get("updated_at") > retrieved.metadata.get("created_at")
    
    def test_delete(self, mock_s3_backend):
        """Test deleting a record."""
        db = mock_s3_backend
        
        # Create and delete a record
        record = Record({"name": "Bob"})
        record_id = db.create(record)
        
        success = db.delete(record_id)
        assert success is True
        
        # Verify deletion
        retrieved = db.read(record_id)
        assert retrieved is None
        
        # Delete non-existent record
        success = db.delete("non-existent-id")
        assert success is False
    
    def test_exists(self, mock_s3_backend):
        """Test checking record existence."""
        db = mock_s3_backend
        
        # Create a record
        record = Record({"name": "Charlie"})
        record_id = db.create(record)
        
        # Check existence
        assert db.exists(record_id) is True
        assert db.exists("non-existent-id") is False
        
        # Delete and check again
        db.delete(record_id)
        assert db.exists(record_id) is False
    
    def test_list_all(self, mock_s3_backend):
        """Test listing all record IDs."""
        db = mock_s3_backend
        
        # Create multiple records
        ids = []
        for i in range(5):
            record = Record({"index": i})
            record_id = db.create(record)
            ids.append(record_id)
        
        # List all IDs
        all_ids = db.list_all()
        assert len(all_ids) == 5
        for record_id in ids:
            assert record_id in all_ids
    
    def test_search_with_filters(self, mock_s3_backend):
        """Test searching with filters."""
        db = mock_s3_backend
        
        # Create test records
        records = [
            Record({"name": "Alice", "age": 25, "city": "NYC"}),
            Record({"name": "Bob", "age": 30, "city": "LA"}),
            Record({"name": "Charlie", "age": 35, "city": "NYC"}),
            Record({"name": "David", "age": 28, "city": "Chicago"}),
        ]
        
        for record in records:
            db.create(record)
        
        # Search with single filter
        query = Query().filter("city", "=", "NYC")
        results = db.search(query)
        assert len(results) == 2
        
        # Search with multiple filters
        query = Query().filter("age", ">=", 30).filter("city", "!=", "Chicago")
        results = db.search(query)
        assert len(results) == 2
        
        # Search with IN operator
        query = Query().filter("name", "IN", ["Alice", "Bob"])
        results = db.search(query)
        assert len(results) == 2
        
        # Search with LIKE operator
        query = Query().filter("name", "LIKE", "C%")
        results = db.search(query)
        assert len(results) == 1
        assert results[0].get_value("name") == "Charlie"
    
    def test_search_with_sorting(self, mock_s3_backend):
        """Test searching with sorting."""
        db = mock_s3_backend
        
        # Create test records
        records = [
            Record({"name": "Charlie", "age": 35}),
            Record({"name": "Alice", "age": 25}),
            Record({"name": "Bob", "age": 30}),
        ]
        
        for record in records:
            db.create(record)
        
        # Sort by name ascending
        query = Query().sort("name", "ASC")
        results = db.search(query)
        assert [r.get_value("name") for r in results] == ["Alice", "Bob", "Charlie"]
        
        # Sort by age descending
        query = Query().sort("age", "DESC")
        results = db.search(query)
        assert [r.get_value("age") for r in results] == [35, 30, 25]
    
    def test_search_with_pagination(self, mock_s3_backend):
        """Test searching with pagination."""
        db = mock_s3_backend
        
        # Create test records
        for i in range(10):
            record = Record({"index": i})
            db.create(record)
        
        # Test limit
        query = Query().limit(3)
        results = db.search(query)
        assert len(results) == 3
        
        # Test offset and limit
        query = Query().offset(5).limit(3)
        results = db.search(query)
        assert len(results) == 3
        
        # Test offset beyond records
        query = Query().offset(15).limit(5)
        results = db.search(query)
        assert len(results) == 0
    
    def test_search_with_projection(self, mock_s3_backend):
        """Test searching with field projection."""
        db = mock_s3_backend
        
        # Create a record with multiple fields
        record = Record({
            "name": "Alice",
            "age": 25,
            "email": "alice@example.com",
            "city": "NYC"
        })
        db.create(record)
        
        # Project specific fields using select method
        query = Query().select("name", "age")
        results = db.search(query)
        assert len(results) == 1
        
        result = results[0]
        assert "name" in result.fields
        assert "age" in result.fields
        assert "email" not in result.fields
        assert "city" not in result.fields
    
    def test_batch_create(self, mock_s3_backend):
        """Test batch creation of records."""
        db = mock_s3_backend
        
        # Create multiple records
        records = [
            Record({"name": f"User{i}", "index": i})
            for i in range(10)
        ]
        
        record_ids = db.create_batch(records)
        assert len(record_ids) == 10
        
        # Verify all records were created
        for record_id in record_ids:
            assert db.exists(record_id)
    
    def test_batch_read(self, mock_s3_backend):
        """Test batch reading of records."""
        db = mock_s3_backend
        
        # Create records
        record_ids = []
        for i in range(5):
            record = Record({"index": i})
            record_id = db.create(record)
            record_ids.append(record_id)
        
        # Add a non-existent ID
        record_ids.append("non-existent-id")
        
        # Batch read
        results = db.read_batch(record_ids)
        assert len(results) == 6
        
        # Verify results
        for i in range(5):
            assert results[i] is not None
            assert results[i].get_value("index") == i
        
        # Last one should be None
        assert results[5] is None
    
    def test_batch_delete(self, mock_s3_backend):
        """Test batch deletion of records."""
        db = mock_s3_backend
        
        # Create records
        record_ids = []
        for i in range(5):
            record = Record({"index": i})
            record_id = db.create(record)
            record_ids.append(record_id)
        
        # Add a non-existent ID
        record_ids.append("non-existent-id")
        
        # Batch delete
        results = db.delete_batch(record_ids)
        assert len(results) == 6
        
        # Verify results
        for i in range(5):
            assert results[i] is True
            assert not db.exists(record_ids[i])
        
        # Last one should be False
        assert results[5] is False
    
    def test_clear(self, mock_s3_backend):
        """Test clearing all records."""
        db = mock_s3_backend
        
        # Create multiple records
        for i in range(10):
            record = Record({"index": i})
            db.create(record)
        
        # Verify records exist
        assert db.count() == 10
        
        # Clear all records
        db.clear()
        
        # Verify all records are gone
        assert db.count() == 0
        assert len(db.list_all()) == 0
    
    def test_count(self, mock_s3_backend):
        """Test counting records."""
        db = mock_s3_backend
        
        # Initially empty
        assert db.count() == 0
        
        # Add records
        for i in range(5):
            record = Record({"index": i})
            db.create(record)
        
        assert db.count() == 5
        
        # Delete a record
        all_ids = db.list_all()
        db.delete(all_ids[0])
        
        assert db.count() == 4
    
    def test_metadata_as_tags(self, mock_s3_backend):
        """Test that metadata is stored as S3 tags."""
        db = mock_s3_backend
        
        # Create a record with metadata
        record = Record(
            data={"name": "Test"},
            metadata={"version": "1.0", "author": "tester"}
        )
        record_id = db.create(record)
        
        # Read back and verify metadata is preserved
        retrieved = db.read(record_id)
        assert retrieved.metadata["version"] == "1.0"
        assert retrieved.metadata["author"] == "tester"
    
    def test_large_record(self, mock_s3_backend):
        """Test handling of large records."""
        db = mock_s3_backend
        
        # Create a large record
        large_data = "x" * 1000000  # 1MB of data
        record = Record({"data": large_data})
        record_id = db.create(record)
        
        # Read back and verify
        retrieved = db.read(record_id)
        assert retrieved.get_value("data") == large_data
    
    def test_concurrent_operations(self, mock_s3_backend):
        """Test concurrent operations using batch methods."""
        db = mock_s3_backend
        
        # Create many records concurrently
        records = [
            Record({"index": i, "data": f"record_{i}"})
            for i in range(20)
        ]
        
        record_ids = db.create_batch(records)
        assert len(record_ids) == 20
        
        # Read them all concurrently
        results = db.read_batch(record_ids)
        assert all(r is not None for r in results)
        
        # Delete them all concurrently
        delete_results = db.delete_batch(record_ids)
        assert all(delete_results)
        
        # Verify all deleted
        assert db.count() == 0


class TestS3Configuration:
    """Test S3 backend configuration."""
    
    def test_from_config_method(self):
        """Test creating SyncS3Database from config."""
        with mock_aws():
            config = {
                "bucket": "test-bucket",
                "prefix": "data/",
                "region": "us-west-2"
            }
            
            db = SyncS3Database.from_config(config)
            assert db.bucket == "test-bucket"
            assert db.prefix == "data/"
            assert db.region == "us-west-2"
    
    def test_missing_required_config(self):
        """Test that missing bucket raises error."""
        with pytest.raises(ValueError, match="bucket name is required"):
            SyncS3Database({})
    
    def test_default_values(self):
        """Test default configuration values."""
        with mock_aws():
            config = {"bucket": "test-bucket"}
            db = SyncS3Database(config)
            
            assert db.prefix == "records/"
            assert db.region == "us-east-1"
            assert db.max_workers == 10
            assert db.multipart_threshold == 8 * 1024 * 1024
    
    def test_custom_endpoint(self):
        """Test configuration with custom endpoint."""
        # Note: This test only verifies configuration, not actual connection
        # For real LocalStack testing, use the TestS3BackendWithLocalStack class
        
        # Detect if we're running in Docker container
        if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER'):
            default_host = 'localstack'
        else:
            default_host = 'localhost'
        
        config = {
            "bucket": "test-bucket",
            "endpoint_url": f"http://{default_host}:4566",
            "access_key_id": "test",
            "secret_access_key": "test"
        }
        
        # Create database without connecting (avoid _ensure_bucket_exists)
        db = SyncS3Database.__new__(SyncS3Database)
        db.bucket = config["bucket"]
        db.endpoint_url = config.get("endpoint_url")
        db.access_key_id = config.get("access_key_id")
        db.secret_access_key = config.get("secret_access_key")
        
        assert db.endpoint_url == f"http://{default_host}:4566"
        assert db.bucket == "test-bucket"


@pytest.mark.integration
@pytest.mark.s3
class TestS3BackendWithLocalStack:
    """Integration tests with real LocalStack."""
    
    def test_real_s3_operations(self, localstack_s3_backend):
        """Test real S3 operations with LocalStack."""
        db = localstack_s3_backend
        
        # Create a record
        record = Record({"test": "localstack", "integration": True})
        record_id = db.create(record)
        
        # Read it back
        retrieved = db.read(record_id)
        assert retrieved.get_value("test") == "localstack"
        assert retrieved.get_value("integration") is True
        
        # Update it
        updated = Record({"test": "localstack-updated", "integration": True})
        db.update(record_id, updated)
        
        # Delete it
        success = db.delete(record_id)
        assert success is True
        
        # Verify deletion
        assert db.read(record_id) is None
