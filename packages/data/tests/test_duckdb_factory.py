"""Tests for DuckDB backend factory integration."""

import pytest

# Try to import duckdb and dependencies - skip tests if not available
try:
    import duckdb
    from dataknobs_data.factory import AsyncDatabaseFactory, DatabaseFactory
    from dataknobs_data.records import Record
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    # Create dummy classes to prevent NameError
    AsyncDatabaseFactory = DatabaseFactory = Record = None

# Skip all tests if DuckDB is not installed
pytestmark = pytest.mark.skipif(
    not DUCKDB_AVAILABLE,
    reason="DuckDB tests require duckdb package (pip install duckdb)"
)


class TestDuckDBFactoryIntegration:
    """Test DuckDB backend creation through factories."""

    def test_sync_factory_creates_duckdb(self):
        """Test sync factory can create DuckDB backend."""
        factory = DatabaseFactory()

        # Create DuckDB backend through factory
        db = factory.create(backend="duckdb", path=":memory:")

        # Verify it's the correct type
        from dataknobs_data.backends.duckdb import SyncDuckDBDatabase
        assert isinstance(db, SyncDuckDBDatabase)

        # Test basic operations
        db.connect()

        record = Record(data={"name": "Alice", "age": 30})
        record_id = db.create(record)

        retrieved = db.read(record_id)
        assert retrieved is not None
        assert retrieved["name"] == "Alice"
        assert retrieved["age"] == 30

        db.close()

    @pytest.mark.asyncio
    async def test_async_factory_creates_duckdb(self):
        """Test async factory can create DuckDB backend."""
        factory = AsyncDatabaseFactory()

        # Create async DuckDB backend through factory
        db = factory.create(backend="duckdb", path=":memory:")

        # Verify it's the correct type
        from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase
        assert isinstance(db, AsyncDuckDBDatabase)

        # Test basic operations
        await db.connect()

        record = Record(data={"name": "Bob", "age": 25})
        record_id = await db.create(record)

        retrieved = await db.read(record_id)
        assert retrieved is not None
        assert retrieved["name"] == "Bob"
        assert retrieved["age"] == 25

        await db.close()

    def test_sync_factory_with_custom_config(self):
        """Test sync factory with custom configuration."""
        factory = DatabaseFactory()

        db = factory.create(
            backend="duckdb",
            path=":memory:",
            table="custom_table"
        )

        db.connect()
        assert db.table_name == "custom_table"

        # Verify operations work with custom table
        record = Record(data={"value": 42})
        record_id = db.create(record)

        count = db.count()
        assert count == 1

        db.close()

    @pytest.mark.asyncio
    async def test_async_factory_with_custom_config(self):
        """Test async factory with custom configuration."""
        factory = AsyncDatabaseFactory()

        db = factory.create(
            backend="duckdb",
            path=":memory:",
            table="async_custom_table"
        )

        await db.connect()
        assert db.table_name == "async_custom_table"

        # Verify operations work with custom table
        record = Record(data={"value": 100})
        record_id = await db.create(record)

        count = await db.count()
        assert count == 1

        await db.close()

    def test_factory_backend_info(self):
        """Test factory can provide backend information."""
        factory = DatabaseFactory()

        info = factory.get_backend_info("duckdb")

        assert info["description"] == "DuckDB database backend for analytical workloads with columnar storage"
        assert info["persistent"] is True
        assert info["vector_support"] is False
        assert "path" in info["config_options"]
        assert "table" in info["config_options"]

    def test_sync_factory_file_based(self, tmp_path):
        """Test sync factory with file-based database."""
        factory = DatabaseFactory()
        db_path = tmp_path / "test.duckdb"

        db = factory.create(backend="duckdb", path=str(db_path))
        db.connect()

        # Create some data
        record = Record(data={"test": "data"})
        record_id = db.create(record)

        db.close()

        # Verify file was created
        assert db_path.exists()

        # Reopen and verify data persists
        db2 = factory.create(backend="duckdb", path=str(db_path))
        db2.connect()

        retrieved = db2.read(record_id)
        assert retrieved is not None
        assert retrieved["test"] == "data"

        db2.close()

    @pytest.mark.asyncio
    async def test_async_factory_file_based(self, tmp_path):
        """Test async factory with file-based database."""
        factory = AsyncDatabaseFactory()
        db_path = tmp_path / "test_async.duckdb"

        db = factory.create(backend="duckdb", path=str(db_path))
        await db.connect()

        # Create some data
        record = Record(data={"async_test": "data"})
        record_id = await db.create(record)

        await db.close()

        # Verify file was created
        assert db_path.exists()

        # Reopen and verify data persists
        db2 = factory.create(backend="duckdb", path=str(db_path))
        await db2.connect()

        retrieved = await db2.read(record_id)
        assert retrieved is not None
        assert retrieved["async_test"] == "data"

        await db2.close()
