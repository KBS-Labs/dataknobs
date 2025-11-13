"""Tests for synchronous DuckDB backend implementation."""

import tempfile
from pathlib import Path

import pytest

# Try to import duckdb and backend - skip tests if not available
try:
    import duckdb
    from dataknobs_data.backends.duckdb import SyncDuckDBDatabase
    from dataknobs_data.query import Filter, Operator, Query, SortOrder
    from dataknobs_data.query_logic import ComplexQuery, FilterCondition, LogicCondition, LogicOperator
    from dataknobs_data.records import Record
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    # Create dummy classes to prevent NameError in test class definitions
    SyncDuckDBDatabase = None
    Filter = Operator = Query = SortOrder = None
    ComplexQuery = FilterCondition = LogicCondition = LogicOperator = None
    Record = None

# Skip all tests if DuckDB is not installed
pytestmark = pytest.mark.skipif(
    not DUCKDB_AVAILABLE,
    reason="DuckDB tests require duckdb package (pip install duckdb)"
)


class TestSyncDuckDBDatabase:
    """Test synchronous DuckDB database backend."""

    @pytest.fixture
    def memory_db(self):
        """Create an in-memory DuckDB database."""
        db = SyncDuckDBDatabase({"path": ":memory:"})
        db.connect()
        yield db
        db.close()

    @pytest.fixture
    def file_db(self, tmp_path):
        """Create a file-based DuckDB database."""
        db_path = tmp_path / "test.duckdb"
        db = SyncDuckDBDatabase({"path": str(db_path)})
        db.connect()
        yield db
        db.close()

    def test_connect_memory(self):
        """Test connecting to in-memory database."""
        db = SyncDuckDBDatabase({"path": ":memory:"})
        db.connect()
        assert db._connected
        db.close()
        assert not db._connected

    def test_connect_file(self, tmp_path):
        """Test connecting to file-based database."""
        db_path = tmp_path / "test.duckdb"
        db = SyncDuckDBDatabase({"path": str(db_path)})
        db.connect()
        assert db._connected
        assert db_path.exists()
        db.close()

    def test_create_read(self, memory_db):
        """Test creating and reading records."""
        record = Record(data={"name": "Alice", "age": 30})

        # Create record
        record_id = memory_db.create(record)
        assert record_id is not None

        # Read record
        retrieved = memory_db.read(record_id)
        assert retrieved is not None
        assert retrieved["name"] == "Alice"
        assert retrieved["age"] == 30

    def test_update(self, memory_db):
        """Test updating records."""
        record = Record(data={"name": "Bob", "age": 25})
        record_id = memory_db.create(record)

        # Update record
        updated_record = Record(data={"name": "Bob", "age": 26})
        success = memory_db.update(record_id, updated_record)
        assert success

        # Verify update
        retrieved = memory_db.read(record_id)
        assert retrieved["age"] == 26

    def test_delete(self, memory_db):
        """Test deleting records."""
        record = Record(data={"name": "Charlie"})
        record_id = memory_db.create(record)

        # Delete record
        success = memory_db.delete(record_id)
        assert success

        # Verify deletion
        retrieved = memory_db.read(record_id)
        assert retrieved is None

    def test_exists(self, memory_db):
        """Test checking record existence."""
        record = Record(data={"name": "David"})
        record_id = memory_db.create(record)

        assert memory_db.exists(record_id)
        assert not memory_db.exists("nonexistent")

    def test_search_basic(self, memory_db):
        """Test basic search functionality."""
        # Create test records
        memory_db.create(Record(data={"name": "Alice", "age": 30, "city": "NYC"}))
        memory_db.create(Record(data={"name": "Bob", "age": 25, "city": "LA"}))
        memory_db.create(Record(data={"name": "Charlie", "age": 35, "city": "NYC"}))

        # Search by city
        query = Query().filter("city", Operator.EQ, "NYC")
        results = memory_db.search(query)
        assert len(results) == 2

        # Search with multiple filters
        query = Query().filter("city", Operator.EQ, "NYC").filter("age", Operator.GT, 30)
        results = memory_db.search(query)
        assert len(results) == 1
        assert results[0]["name"] == "Charlie"

    def test_search_operators(self, memory_db):
        """Test various search operators."""
        # Create test records
        memory_db.create(Record(data={"value": 10}))
        memory_db.create(Record(data={"value": 20}))
        memory_db.create(Record(data={"value": 30}))
        memory_db.create(Record(data={"value": 40}))

        # Test GT
        query = Query().filter("value", Operator.GT, 25)
        results = memory_db.search(query)
        assert len(results) == 2

        # Test LTE
        query = Query().filter("value", Operator.LTE, 20)
        results = memory_db.search(query)
        assert len(results) == 2

        # Test BETWEEN
        query = Query().filter("value", Operator.BETWEEN, [15, 35])
        results = memory_db.search(query)
        assert len(results) == 2

        # Test IN
        query = Query().filter("value", Operator.IN, [10, 30, 50])
        results = memory_db.search(query)
        assert len(results) == 2

    def test_search_complex_query(self, memory_db):
        """Test complex query with boolean logic."""
        # Create test records
        memory_db.create(Record(data={"name": "Alice", "age": 30, "city": "NYC"}))
        memory_db.create(Record(data={"name": "Bob", "age": 25, "city": "LA"}))
        memory_db.create(Record(data={"name": "Charlie", "age": 35, "city": "SF"}))
        memory_db.create(Record(data={"name": "David", "age": 28, "city": "NYC"}))

        # Test OR condition: city = NYC OR age > 30
        query = ComplexQuery(
            condition=LogicCondition(
                operator=LogicOperator.OR,
                conditions=[
                    FilterCondition(Filter("city", Operator.EQ, "NYC")),
                    FilterCondition(Filter("age", Operator.GT, 30))
                ]
            )
        )
        results = memory_db.search(query)
        assert len(results) == 3  # Alice, Charlie, David

        # Test AND with OR: (city = NYC OR city = LA) AND age < 30
        query = ComplexQuery(
            condition=LogicCondition(
                operator=LogicOperator.AND,
                conditions=[
                    LogicCondition(
                        operator=LogicOperator.OR,
                        conditions=[
                            FilterCondition(Filter("city", Operator.EQ, "NYC")),
                            FilterCondition(Filter("city", Operator.EQ, "LA"))
                        ]
                    ),
                    FilterCondition(Filter("age", Operator.LT, 30))
                ]
            )
        )
        results = memory_db.search(query)
        assert len(results) == 2  # Bob, David

    def test_sorting(self, memory_db):
        """Test sorting functionality."""
        # Create test records
        memory_db.create(Record(data={"name": "Charlie", "age": 35}))
        memory_db.create(Record(data={"name": "Alice", "age": 30}))
        memory_db.create(Record(data={"name": "Bob", "age": 25}))

        # Sort by age ascending
        query = Query().sort("age", SortOrder.ASC)
        results = memory_db.search(query)
        assert results[0]["name"] == "Bob"
        assert results[1]["name"] == "Alice"
        assert results[2]["name"] == "Charlie"

        # Sort by age descending
        query = Query().sort("age", SortOrder.DESC)
        results = memory_db.search(query)
        assert results[0]["name"] == "Charlie"
        assert results[1]["name"] == "Alice"
        assert results[2]["name"] == "Bob"

    def test_limit_offset(self, memory_db):
        """Test limit and offset functionality."""
        # Create test records
        for i in range(10):
            memory_db.create(Record(data={"value": i}))

        # Test limit
        query = Query().limit(5)
        results = memory_db.search(query)
        assert len(results) == 5

        # Test offset
        query = Query().offset(5)
        results = memory_db.search(query)
        assert len(results) == 5

        # Test limit and offset together
        query = Query().limit(3).offset(2)
        results = memory_db.search(query)
        assert len(results) == 3

    def test_count(self, memory_db):
        """Test count functionality."""
        # Create test records
        memory_db.create(Record(data={"category": "A", "value": 10}))
        memory_db.create(Record(data={"category": "A", "value": 20}))
        memory_db.create(Record(data={"category": "B", "value": 30}))

        # Count all
        count = memory_db.count()
        assert count == 3

        # Count with filter
        query = Query().filter("category", Operator.EQ, "A")
        count = memory_db.count(query)
        assert count == 2

    def test_batch_create(self, memory_db):
        """Test batch create functionality."""
        records = [
            Record(data={"name": f"User{i}", "value": i})
            for i in range(100)
        ]

        ids = memory_db.create_batch(records)
        assert len(ids) == 100

        # Verify records were created
        count = memory_db.count()
        assert count == 100

    def test_batch_update(self, memory_db):
        """Test batch update functionality."""
        # Create records
        records = [Record(data={"value": i, "index": i}) for i in range(10)]
        ids = memory_db.create_batch(records)

        # Update records
        updates = [
            (record_id, Record(data={"value": i * 2, "index": i}))
            for i, record_id in enumerate(ids)
        ]
        results = memory_db.update_batch(updates)
        assert all(results)

        # Verify updates
        for i, record_id in enumerate(ids):
            record = memory_db.read(record_id)
            assert record is not None
            assert "value" in record.fields

    def test_batch_delete(self, memory_db):
        """Test batch delete functionality."""
        # Create records
        records = [Record(data={"value": i}) for i in range(10)]
        ids = memory_db.create_batch(records)

        # Delete some records
        delete_ids = ids[:5]
        results = memory_db.delete_batch(delete_ids)
        assert all(results)

        # Verify deletions
        count = memory_db.count()
        assert count == 5

    def test_stream_read(self, memory_db):
        """Test stream reading."""
        # Create test records
        records = [Record(data={"value": i}) for i in range(50)]
        memory_db.create_batch(records)

        # Stream read
        from dataknobs_data.streaming import StreamConfig
        config = StreamConfig(batch_size=10)

        count = 0
        for record in memory_db.stream_read(config=config):
            count += 1

        assert count == 50

    def test_stream_write(self, memory_db):
        """Test stream writing."""
        from dataknobs_data.streaming import StreamConfig

        # Create a generator
        def record_generator():
            for i in range(50):
                yield Record(data={"value": i})

        # Stream write
        config = StreamConfig(batch_size=10)
        result = memory_db.stream_write(record_generator(), config)

        assert result.total_processed == 50
        assert result.successful == 50
        assert result.failed == 0

        # Verify records were written
        count = memory_db.count()
        assert count == 50

    def test_persistence(self, tmp_path):
        """Test data persistence in file-based database."""
        db_path = tmp_path / "persist.duckdb"

        # Create database and add records
        db1 = SyncDuckDBDatabase({"path": str(db_path)})
        db1.connect()

        record = Record(data={"name": "Alice", "age": 30})
        record_id = db1.create(record)

        db1.close()

        # Reopen database and verify data persists
        db2 = SyncDuckDBDatabase({"path": str(db_path)})
        db2.connect()

        retrieved = db2.read(record_id)
        assert retrieved is not None
        assert retrieved["name"] == "Alice"
        assert retrieved["age"] == 30

        db2.close()

    def test_field_projection(self, memory_db):
        """Test field projection in queries."""
        # Create record with multiple fields
        memory_db.create(Record(data={
            "name": "Alice",
            "age": 30,
            "city": "NYC",
            "country": "USA"
        }))

        # Query with field projection
        query = Query()
        query.fields = ["name", "age"]
        results = memory_db.search(query)

        assert len(results) == 1
        assert "name" in results[0].fields
        assert "age" in results[0].fields
        assert "city" not in results[0].fields
        assert "country" not in results[0].fields

    def test_metadata_storage(self, memory_db):
        """Test metadata storage and retrieval."""
        record = Record(
            data={"name": "Alice"},
            metadata={"source": "test", "version": 1}
        )

        record_id = memory_db.create(record)
        retrieved = memory_db.read(record_id)

        assert retrieved.metadata["source"] == "test"
        assert retrieved.metadata["version"] == 1

    def test_large_dataset_performance(self, memory_db):
        """Test performance with larger datasets."""
        # Create a larger dataset
        records = [
            Record(data={
                "id": i,
                "name": f"User{i}",
                "age": 20 + (i % 50),
                "category": f"Cat{i % 10}"
            })
            for i in range(1000)
        ]

        # Batch create
        ids = memory_db.create_batch(records)
        assert len(ids) == 1000

        # Complex query on large dataset
        query = Query().filter("age", Operator.GT, 40).limit(50)
        results = memory_db.search(query)
        assert len(results) == 50

        # Count with filter
        count_query = Query().filter("category", Operator.EQ, "Cat5")
        count = memory_db.count(count_query)
        assert count == 100  # 1000 / 10 categories

    def test_null_and_empty_values(self, memory_db):
        """Test handling of null and empty values."""
        # Create records with null/empty values
        memory_db.create(Record(data={"name": "Alice", "email": None}))
        memory_db.create(Record(data={"name": "Bob", "email": ""}))
        memory_db.create(Record(data={"name": "Charlie", "email": "charlie@example.com"}))

        # Test EXISTS operator
        query = Query().filter("email", Operator.EXISTS, None)
        results = memory_db.search(query)
        # All records should have the email field (even if null or empty)
        assert len(results) >= 1

        # Test NOT_EXISTS operator
        query = Query().filter("phone", Operator.NOT_EXISTS, None)
        results = memory_db.search(query)
        assert len(results) == 3

    def test_from_config(self):
        """Test creating database from config."""
        config = {"path": ":memory:", "table": "test_table"}
        db = SyncDuckDBDatabase.from_config(config)
        db.connect()

        assert db.table_name == "test_table"
        assert db._connected

        db.close()

    def test_error_handling(self, memory_db):
        """Test error handling for invalid operations."""
        # Test reading non-existent record
        result = memory_db.read("nonexistent")
        assert result is None

        # Test deleting non-existent record
        success = memory_db.delete("nonexistent")
        assert not success

        # Test updating non-existent record
        success = memory_db.update("nonexistent", Record(data={"test": "data"}))
        assert not success

    def test_custom_table_name(self, tmp_path):
        """Test using a custom table name."""
        db_path = tmp_path / "custom_table.duckdb"
        db = SyncDuckDBDatabase({"path": str(db_path), "table": "my_records"})
        db.connect()

        assert db.table_name == "my_records"

        # Verify operations work with custom table
        record = Record(data={"test": "value"})
        record_id = db.create(record)

        retrieved = db.read(record_id)
        assert retrieved is not None
        assert retrieved["test"] == "value"

        db.close()

    def test_read_only_mode(self, tmp_path):
        """Test read-only mode."""
        db_path = tmp_path / "readonly.duckdb"

        # Create database with some data
        db_write = SyncDuckDBDatabase({"path": str(db_path)})
        db_write.connect()
        record = Record(data={"name": "Test"})
        record_id = db_write.create(record)
        db_write.close()

        # Open in read-only mode
        db_read = SyncDuckDBDatabase({"path": str(db_path), "read_only": True})
        db_read.connect()

        # Reading should work
        retrieved = db_read.read(record_id)
        assert retrieved is not None
        assert retrieved["name"] == "Test"

        # Writing should fail
        with pytest.raises(Exception):  # DuckDB will raise an exception for writes in read-only mode
            db_read.create(Record(data={"name": "New"}))

        db_read.close()

    def test_timeout_configuration(self):
        """Test timeout configuration."""
        db = SyncDuckDBDatabase({"path": ":memory:", "timeout": 10.0})
        db.connect()

        assert db.timeout == 10.0

        # Basic operation should work
        record = Record(data={"test": "value"})
        record_id = db.create(record)
        assert record_id is not None

        db.close()
