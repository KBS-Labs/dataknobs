"""Tests for ID field filtering in S3 backend."""

import os
import pytest
from dataknobs_data import Query, Record, SyncDatabase
from dataknobs_data.query import Operator


# Skip tests if S3 is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_S3", "").lower() == "true",
    reason="S3 tests require TEST_S3=true and a running LocalStack or S3 instance"
)


class TestS3IdFiltering:
    """Test ID field filtering for S3 backend."""
    
    @pytest.fixture
    def db(self):
        """Create database instance for S3 backend."""
        config = {
            "bucket": os.environ.get("S3_BUCKET", "test-dataknobs"),
            "prefix": "test_id_filtering/",
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", "test"),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
            "endpoint_url": os.environ.get("S3_ENDPOINT", "http://localhost:4566"),
            "region_name": os.environ.get("AWS_REGION", "us-east-1")
        }
        
        db = SyncDatabase.from_backend("s3", config=config)
        
        # Clear any existing data
        db.clear()
        
        yield db
        
        # Cleanup
        db.clear()
    
    def test_id_field_equality_filter(self, db):
        """Test filtering by ID with equality operator."""
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=f"obj_{i}", data={"value": i * 10})
            db.create(record)
        
        # Test EQ (equal)
        query = Query().filter("id", Operator.EQ, "obj_3")
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "obj_3"
        assert results[0].get_value("value") == 30
    
    def test_id_field_comparison_filters(self, db):
        """Test filtering by ID with comparison operators."""
        # Create records with specific IDs (use consistent format for string comparison)
        for i in range(5):
            record = Record(id=f"obj_{i:03d}", data={"value": i * 10})
            db.create(record)
        
        # Test GT (greater than)
        query = Query().filter("id", Operator.GT, "obj_002").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "obj_003"
        assert results[1].id == "obj_004"
        
        # Test LT (less than)
        query = Query().filter("id", Operator.LT, "obj_002").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "obj_000"
        assert results[1].id == "obj_001"
    
    def test_id_field_in_filter(self, db):
        """Test filtering by ID with IN operator."""
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=f"obj_{i}", data={"value": i * 10})
            db.create(record)
        
        # Test IN
        query = Query().filter("id", Operator.IN, ["obj_0", "obj_2", "obj_4"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert set(r.id for r in results) == {"obj_0", "obj_2", "obj_4"}
        
        # Test NOT_IN
        query = Query().filter("id", Operator.NOT_IN, ["obj_0", "obj_2", "obj_4"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"obj_1", "obj_3"}
    
    def test_id_field_sorting(self, db):
        """Test sorting by ID field."""
        # Create records with specific IDs (not in order)
        ids = ["obj_3", "obj_1", "obj_4", "obj_0", "obj_2"]
        for id_val in ids:
            num = int(id_val.split("_")[1])
            record = Record(id=id_val, data={"value": num * 10})
            db.create(record)
        
        # Test ascending sort
        query = Query().sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 5
        assert [r.id for r in results] == ["obj_0", "obj_1", "obj_2", "obj_3", "obj_4"]
        
        # Test descending sort
        query = Query().sort_by("id", "desc")
        results = db.search(query)
        assert len(results) == 5
        assert [r.id for r in results] == ["obj_4", "obj_3", "obj_2", "obj_1", "obj_0"]
    
    def test_id_field_combined_with_data_filters(self, db):
        """Test combining ID field filters with data field filters."""
        # Create records with specific IDs
        for i in range(10):
            record = Record(
                id=f"obj_{i:03d}", 
                data={"value": i * 10, "category": "even" if i % 2 == 0 else "odd"}
            )
            db.create(record)
        
        # Test ID filter combined with data filter
        query = Query().filter("id", Operator.GT, "obj_003").filter("category", Operator.EQ, "even").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert [r.id for r in results] == ["obj_004", "obj_006", "obj_008"]