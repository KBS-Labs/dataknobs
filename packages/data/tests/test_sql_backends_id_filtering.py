"""Tests for ID field filtering in SQL backends (PostgreSQL and MySQL)."""

import os
import pytest
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dataknobs_data import Query, Record, SyncDatabase
from dataknobs_data.query import Operator


# Skip tests if database environment variables are not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES", "").lower() == "true",
    reason="PostgreSQL tests require TEST_POSTGRES=true and a running PostgreSQL instance"
)


class TestSqlBackendsIdFiltering:
    """Test ID field filtering for PostgreSQL backend."""
    
    @pytest.fixture
    def db(self):
        """Create database instance for PostgreSQL backend."""
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = int(os.environ.get("POSTGRES_PORT", "5432"))
        database = os.environ.get("POSTGRES_DB", "test_dataknobs")
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD", "postgres")

        # Ensure database exists by connecting to 'postgres' first
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{database}'")
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {database}")
        except psycopg2.errors.DuplicateDatabase:
            pass
        finally:
            cursor.close()
            conn.close()

        config = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
            "table_name": "test_id_filtering"
        }

        db = SyncDatabase.from_backend("postgres", config=config)
        db.connect()

        # Clear any existing data
        db.clear()

        yield db

        # Cleanup
        db.clear()
        db.disconnect()
    
    def test_id_field_equality_filter(self, db):
        """Test filtering by ID with equality operator."""
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=f"rec_{i}", data={"value": i * 10})
            db.create(record)
        
        # Test EQ (equal)
        query = Query().filter("id", Operator.EQ, "rec_3")
        results = db.search(query)
        assert len(results) == 1
        assert results[0].id == "rec_3"
        assert results[0].get_value("value") == 30
    
    def test_id_field_comparison_filters(self, db):
        """Test filtering by ID with comparison operators."""
        # Create records with specific IDs (use consistent format for string comparison)
        for i in range(5):
            record = Record(id=f"rec_{i:03d}", data={"value": i * 10})
            db.create(record)
        
        # Test GT (greater than)
        query = Query().filter("id", Operator.GT, "rec_002").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "rec_003"
        assert results[1].id == "rec_004"
        
        # Test LT (less than)
        query = Query().filter("id", Operator.LT, "rec_002").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "rec_000"
        assert results[1].id == "rec_001"
        
        # Test GTE (greater than or equal)
        query = Query().filter("id", Operator.GTE, "rec_003").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "rec_003"
        assert results[1].id == "rec_004"
        
        # Test LTE (less than or equal)
        query = Query().filter("id", Operator.LTE, "rec_001").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert results[0].id == "rec_000"
        assert results[1].id == "rec_001"
    
    def test_id_field_in_filter(self, db):
        """Test filtering by ID with IN operator."""
        # Create records with specific IDs
        for i in range(5):
            record = Record(id=f"rec_{i}", data={"value": i * 10})
            db.create(record)
        
        # Test IN
        query = Query().filter("id", Operator.IN, ["rec_0", "rec_2", "rec_4"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert set(r.id for r in results) == {"rec_0", "rec_2", "rec_4"}
        
        # Test NOT_IN
        query = Query().filter("id", Operator.NOT_IN, ["rec_0", "rec_2", "rec_4"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert set(r.id for r in results) == {"rec_1", "rec_3"}
    
    def test_id_field_between_filter(self, db):
        """Test filtering by ID with BETWEEN operator."""
        # Create records with specific IDs (use consistent format)
        for i in range(5):
            record = Record(id=f"rec_{i:03d}", data={"value": i * 10})
            db.create(record)
        
        # Test BETWEEN
        query = Query().filter("id", Operator.BETWEEN, ["rec_001", "rec_003"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert [r.id for r in results] == ["rec_001", "rec_002", "rec_003"]
        
        # Test NOT_BETWEEN
        query = Query().filter("id", Operator.NOT_BETWEEN, ["rec_001", "rec_003"]).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert [r.id for r in results] == ["rec_000", "rec_004"]
    
    def test_id_field_sorting(self, db):
        """Test sorting by ID field."""
        # Create records with specific IDs (not in order)
        ids = ["rec_003", "rec_001", "rec_004", "rec_000", "rec_002"]
        for id_val in ids:
            num = int(id_val.split("_")[1])
            record = Record(id=id_val, data={"value": num * 10})
            db.create(record)
        
        # Test ascending sort
        query = Query().sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 5
        assert [r.id for r in results] == ["rec_000", "rec_001", "rec_002", "rec_003", "rec_004"]
        
        # Test descending sort
        query = Query().sort_by("id", "desc")
        results = db.search(query)
        assert len(results) == 5
        assert [r.id for r in results] == ["rec_004", "rec_003", "rec_002", "rec_001", "rec_000"]
    
    def test_id_field_combined_with_data_filters(self, db):
        """Test combining ID field filters with data field filters."""
        # Create records with specific IDs
        for i in range(10):
            record = Record(
                id=f"rec_{i:03d}", 
                data={"value": i * 10, "category": "even" if i % 2 == 0 else "odd"}
            )
            db.create(record)
        
        # Test ID filter combined with data filter
        query = Query().filter("id", Operator.GT, "rec_003").filter("category", Operator.EQ, "even").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert [r.id for r in results] == ["rec_004", "rec_006", "rec_008"]
        
        # Test ID range with value filter
        query = Query().filter("id", Operator.BETWEEN, ["rec_002", "rec_007"]).filter("value", Operator.GTE, 40).sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 4
        assert [r.id for r in results] == ["rec_004", "rec_005", "rec_006", "rec_007"]
    
    def test_id_field_with_pagination(self, db):
        """Test ID field filtering with pagination."""
        # Create records with specific IDs
        for i in range(10):
            record = Record(id=f"rec_{i:03d}", data={"value": i})
            db.create(record)
        
        # Test with limit and offset
        query = Query().filter("id", Operator.GTE, "rec_003").sort_by("id", "asc").limit(3).offset(2)
        results = db.search(query)
        assert len(results) == 3
        assert [r.id for r in results] == ["rec_005", "rec_006", "rec_007"]
    
    def test_id_field_like_operator(self, db):
        """Test LIKE operator on ID field."""
        # Create records with various ID patterns
        ids = ["user_001", "user_002", "admin_001", "admin_002", "guest_001"]
        for id_val in ids:
            record = Record(id=id_val, data={"type": id_val.split("_")[0]})
            db.create(record)
        
        # Test LIKE with prefix pattern
        query = Query().filter("id", Operator.LIKE, "user%").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 2
        assert [r.id for r in results] == ["user_001", "user_002"]
        
        # Test LIKE with suffix pattern
        query = Query().filter("id", Operator.LIKE, "%001").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert set(r.id for r in results) == {"user_001", "admin_001", "guest_001"}
        
        # Test NOT_LIKE
        query = Query().filter("id", Operator.NOT_LIKE, "admin%").sort_by("id", "asc")
        results = db.search(query)
        assert len(results) == 3
        assert set(r.id for r in results) == {"user_001", "user_002", "guest_001"}
    
    def test_numeric_id_comparison(self, db):
        """Test that numeric IDs are compared as strings in SQL."""
        # Create records with numeric-looking string IDs
        ids = ["1", "2", "10", "20", "100"]
        for id_val in ids:
            record = Record(id=id_val, data={"value": int(id_val)})
            db.create(record)
        
        # String comparison in SQL: "2" > "10" is True, "2" > "100" is True
        query = Query().filter("id", Operator.GT, "2").sort_by("id", "asc")
        results = db.search(query)
        # In string comparison, only "20" is > "2"
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