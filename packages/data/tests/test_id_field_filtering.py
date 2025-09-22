"""Tests for ID field filtering across all database backends."""

import pytest
from dataknobs_data import Query, Record, SyncDatabase
from dataknobs_data.query import Operator


class TestIdFieldFilteringAllBackends:
    """Test ID field filtering works correctly across all backends."""
    
    @pytest.fixture(params=["memory", "sqlite", "file"])
    def db(self, request, tmp_path):
        """Create database instance for each backend."""
        backend = request.param
        
        if backend == "sqlite":
            # Use temporary SQLite database
            db_path = tmp_path / "test.db"
            db = SyncDatabase.from_backend(backend, config={"db_path": str(db_path)})
            db.connect()
        elif backend == "file":
            # Use temporary file database
            file_path = tmp_path / "test.json"
            db = SyncDatabase.from_backend(backend, config={"path": str(file_path)})
        else:
            # Memory backend
            db = SyncDatabase.from_backend(backend)
        
        yield db
        
        # Cleanup
        if hasattr(db, 'disconnect'):
            db.disconnect()
    
    def test_id_field_equality_filter(self, db):
        """Test filtering by ID with equality operator."""
        # Clear any existing data
        if hasattr(db, 'clear'):
            db.clear()
        
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=str(i), data={"value": i * 10})
            db.create(record)
        
        # Test EQ (equal)
        query = Query().filter("id", Operator.EQ, "3")
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "3"
        assert results[0].get_value("value") == 30
    
    def test_id_field_comparison_filters(self, db):
        """Test filtering by ID with comparison operators."""
        # Clear any existing data
        if hasattr(db, 'clear'):
            db.clear()
        
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=str(i), data={"value": i * 10})
            db.create(record)
        
        # Test GT (greater than)
        query = Query().filter("id", Operator.GT, "2").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "3"
        assert results[1].id == "4"
        
        # Test LT (less than)
        query = Query().filter("id", Operator.LT, "2").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "0"
        assert results[1].id == "1"
        
        # Test GTE (greater than or equal)
        query = Query().filter("id", Operator.GTE, "3").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "3"
        assert results[1].id == "4"
        
        # Test LTE (less than or equal)
        query = Query().filter("id", Operator.LTE, "1").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "0"
        assert results[1].id == "1"
    
    def test_id_field_in_filter(self, db):
        """Test filtering by ID with IN operator."""
        # Clear any existing data
        if hasattr(db, 'clear'):
            db.clear()
        
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=str(i), data={"value": i * 10})
            db.create(record)
        
        # Test IN
        query = Query().filter("id", Operator.IN, ["0", "2", "4"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert [r.id for r in results] == ["0", "2", "4"]
        
        # Test NOT_IN
        query = Query().filter("id", Operator.NOT_IN, ["0", "2", "4"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert [r.id for r in results] == ["1", "3"]
    
    def test_id_field_between_filter(self, db):
        """Test filtering by ID with BETWEEN operator."""
        # Clear any existing data
        if hasattr(db, 'clear'):
            db.clear()
        
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=str(i), data={"value": i * 10})
            db.create(record)
        
        # Test BETWEEN
        query = Query().filter("id", Operator.BETWEEN, ["1", "3"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert [r.id for r in results] == ["1", "2", "3"]
        
        # Test NOT_BETWEEN
        query = Query().filter("id", Operator.NOT_BETWEEN, ["1", "3"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert [r.id for r in results] == ["0", "4"]
    
    def test_id_field_sorting(self, db):
        """Test sorting by ID field."""
        # Clear any existing data
        if hasattr(db, 'clear'):
            db.clear()
        
        # Create records with specific IDs (not in order)
        ids = ["3", "1", "4", "0", "2"]
        for id_val in ids:
            record = Record(id=id_val, data={"value": int(id_val) * 10})
            db.create(record)
        
        # Test ascending sort
        query = Query().sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 5
        assert [r.id for r in results] == ["0", "1", "2", "3", "4"]
        
        # Test descending sort
        query = Query().sort_by("id", "desc")
        results = db.search(query)
        assert len(results) == 5
        assert [r.id for r in results] == ["4", "3", "2", "1", "0"]
    
    def test_id_field_combined_with_data_filters(self, db):
        """Test combining ID field filters with data field filters."""
        # Clear any existing data
        if hasattr(db, 'clear'):
            db.clear()
        
        # Create records with specific IDs
        for i in range(10):
            record = Record(id=str(i), data={"value": i * 10, "category": "even" if i % 2 == 0 else "odd"})
            db.create(record)
        
        # Test ID filter combined with data filter
        query = Query().filter("id", Operator.GT, "3").filter("category", Operator.EQ, "even").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert [r.id for r in results] == ["4", "6", "8"]
        
        # Test ID range with value filter
        query = Query().filter("id", Operator.BETWEEN, ["2", "7"]).filter("value", Operator.GTE, 40).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 4
        assert [r.id for r in results] == ["4", "5", "6", "7"]
    
    def test_id_field_with_pagination(self, db):
        """Test ID field filtering with pagination."""
        # Clear any existing data
        if hasattr(db, 'clear'):
            db.clear()
        
        # Create records with specific IDs
        for i in range(10):
            record = Record(id=str(i).zfill(2), data={"value": i})  # Use zero-padded IDs for consistent string sorting
            db.create(record)
        
        # Test with limit and offset
        query = Query().filter("id", Operator.GTE, "03").sort_by("id", "asc").limit(3).offset(2)
        results = db.search(query)
        assert len(results) == 3
        assert [r.id for r in results] == ["05", "06", "07"]
    
    def test_numeric_id_field_comparison(self, db):
        """Test that numeric IDs are compared correctly as strings."""
        # Clear any existing data
        if hasattr(db, 'clear'):
            db.clear()
        
        # Create records with numeric-looking string IDs
        ids = ["1", "2", "10", "20", "100"]
        for id_val in ids:
            record = Record(id=id_val, data={"value": int(id_val)})
            db.create(record)
        
        # String comparison: "2" > "10" is True, "2" > "100" is True
        query = Query().filter("id", Operator.GT, "2").sort_by("id", "asc")
        results = db.search(query)
        # In string comparison, "20" is the only one > "2"
        assert len(results) == 1
        assert results[0].id == "20"
        
        # Test with zero-padded IDs for consistent string comparison
        db.clear()
        for i in [1, 2, 10, 20, 100]:
            record = Record(id=str(i).zfill(3), data={"value": i})  # "001", "002", "010", "020", "100"
            db.create(record)
        
        query = Query().filter("id", Operator.GT, "002").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert [r.id for r in results] == ["010", "020", "100"]