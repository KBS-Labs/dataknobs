"""Tests for ID field filtering in Elasticsearch backend."""

import os
import pytest
from dataknobs_data import Query, Record, SyncDatabase
from dataknobs_data.query import Operator


# Skip tests if Elasticsearch is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_ELASTICSEARCH", "").lower() == "true",
    reason="Elasticsearch tests require TEST_ELASTICSEARCH=true and a running Elasticsearch instance"
)


class TestElasticsearchIdFiltering:
    """Test ID field filtering for Elasticsearch backend."""
    
    @pytest.fixture
    def db(self):
        """Create database instance for Elasticsearch backend."""
        config = {
            "hosts": [os.environ.get("ELASTICSEARCH_HOST", "http://localhost:9200")],
            "index": "test_id_filtering"
        }
        
        db = SyncDatabase.from_backend("elasticsearch", config=config)
        
        # Clear any existing data
        db.clear()
        
        yield db
        
        # Cleanup
        db.clear()
    
    def test_id_field_equality_filter(self, db):
        """Test filtering by ID with equality operator."""
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=f"doc_{i}", data={"value": i * 10})
            db.create(record)
        
        # Test EQ (equal)
        query = Query().filter("id", Operator.EQ, "doc_3")
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "doc_3"
        assert results[0].get_value("value") == 30
    
    def test_id_field_in_filter(self, db):
        """Test filtering by ID with IN operator."""
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=f"doc_{i}", data={"value": i * 10})
            db.create(record)
        
        # Test IN
        query = Query().filter("id", Operator.IN, ["doc_0", "doc_2", "doc_4"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert set(r.id for r in results) == {"doc_0", "doc_2", "doc_4"}
        
        # Test NOT_IN
        query = Query().filter("id", Operator.NOT_IN, ["doc_0", "doc_2", "doc_4"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"doc_1", "doc_3"}
    
    def test_id_field_sorting(self, db):
        """Test sorting by ID field."""
        # Create records with specific IDs (not in order)
        ids = ["doc_3", "doc_1", "doc_4", "doc_0", "doc_2"]
        for id_val in ids:
            num = int(id_val.split("_")[1])
            record = Record(id=id_val, data={"value": num * 10})
            db.create(record)
        
        # Test ascending sort
        query = Query().sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 5
        assert [r.id for r in results] == ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]
        
        # Test descending sort
        query = Query().sort_by("id", "desc")
        results = db.search(query)
        assert len(results) == 5
        assert [r.id for r in results] == ["doc_4", "doc_3", "doc_2", "doc_1", "doc_0"]
    
    def test_id_field_combined_with_data_filters(self, db):
        """Test combining ID field filters with data field filters."""
        # Create records with specific IDs
        for i in range(10):
            record = Record(
                id=f"doc_{i:03d}", 
                data={"value": i * 10, "category": "even" if i % 2 == 0 else "odd"}
            )
            db.create(record)
        
        # Test ID filter combined with data filter
        query = Query().filter("id", Operator.IN, ["doc_004", "doc_006", "doc_008"]).filter("category", Operator.EQ, "even").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert [r.id for r in results] == ["doc_004", "doc_006", "doc_008"]