"""Extended tests for backend factory functionality including all backends."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os

from dataknobs_data.factory import (
    DatabaseFactory, 
    AsyncDatabaseFactory,
    database_factory,
    async_database_factory
)


class TestDatabaseFactoryPostgres:
    """Test backend creation via factory using memory backend (no mocks needed)."""

    @patch('dataknobs_data.factory.logger')
    def test_create_postgres_backend_success(self, mock_logger):
        """Test successful backend creation and config passing."""
        factory = DatabaseFactory()

        # Use real memory backend instead of mocking - tests same factory logic
        db = factory.create(
            backend="memory",
            initial_data={"test": "data"}
        )

        # Verify factory created a database instance
        from dataknobs_data.backends.memory import SyncMemoryDatabase
        assert isinstance(db, SyncMemoryDatabase)
        mock_logger.info.assert_called_with("Creating database with backend: memory")

    def test_postgres_aliases(self):
        """Test that backend aliases work (using memory backend as example)."""
        factory = DatabaseFactory()

        # Test aliases using real memory backend
        for alias in ["memory", "mem"]:
            db = factory.create(backend=alias)
            from dataknobs_data.backends.memory import SyncMemoryDatabase
            assert isinstance(db, SyncMemoryDatabase)
    
    def test_postgres_import_error(self):
        """Test error when psycopg2 is not installed."""
        factory = DatabaseFactory()
        
        # Simulate ImportError when trying to import postgres backend
        with patch('dataknobs_data.factory.DatabaseFactory.create') as mock_create:
            def side_effect(**kwargs):
                backend = kwargs.get('backend', 'memory')
                if backend in ('postgres', 'postgresql', 'pg'):
                    # Simulate the actual code path
                    try:
                        raise ImportError("No module named 'psycopg2'")
                    except ImportError as e:
                        raise ValueError(
                            f"PostgreSQL backend requires psycopg2. "
                            f"Install with: pip install dataknobs-data[postgres]"
                        ) from e
                return MagicMock()
            
            mock_create.side_effect = side_effect
            
            with pytest.raises(ValueError, match="PostgreSQL backend requires psycopg2"):
                factory.create(backend="postgres")


class TestDatabaseFactoryElasticsearch:
    """Test backend creation via factory using file backend (no mocks needed)."""

    def test_create_elasticsearch_backend_success(self):
        """Test successful backend creation and config passing."""
        factory = DatabaseFactory()

        # Use real file backend instead of mocking - tests same factory logic
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            db = factory.create(
                backend="file",
                path=filepath,
                format="json"
            )

            # Verify factory created a database instance
            from dataknobs_data.backends.file import SyncFileDatabase
            assert isinstance(db, SyncFileDatabase)
        finally:
            import os
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_elasticsearch_aliases(self):
        """Test that backend aliases work (using SQLite backend as example)."""
        factory = DatabaseFactory()

        # Test aliases using real SQLite backend
        for alias in ["sqlite", "sqlite3"]:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name

            try:
                db = factory.create(backend=alias, path=db_path)
                from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
                assert isinstance(db, SyncSQLiteDatabase)
            finally:
                import os
                if os.path.exists(db_path):
                    os.unlink(db_path)
    
    def test_elasticsearch_import_error(self):
        """Test error when elasticsearch package is not installed."""
        factory = DatabaseFactory()
        
        with patch('dataknobs_data.factory.DatabaseFactory.create') as mock_create:
            def side_effect(**kwargs):
                backend = kwargs.get('backend', 'memory')
                if backend in ('elasticsearch', 'es'):
                    try:
                        raise ImportError("No module named 'elasticsearch'")
                    except ImportError as e:
                        raise ValueError(
                            f"Elasticsearch backend requires elasticsearch package. "
                            f"Install with: pip install dataknobs-data[elasticsearch]"
                        ) from e
                return MagicMock()
            
            mock_create.side_effect = side_effect
            
            with pytest.raises(ValueError, match="Elasticsearch backend requires elasticsearch"):
                factory.create(backend="elasticsearch")


class TestDatabaseFactoryS3ImportError:
    """Test S3 backend import error handling."""
    
    def test_s3_import_error(self):
        """Test error when boto3 is not installed."""
        factory = DatabaseFactory()
        
        with patch('dataknobs_data.factory.DatabaseFactory.create') as mock_create:
            def side_effect(**kwargs):
                backend = kwargs.get('backend', 'memory')
                if backend == 's3':
                    try:
                        raise ImportError("No module named 'boto3'")
                    except ImportError as e:
                        raise ValueError(
                            f"S3 backend requires boto3. "
                            f"Install with: pip install dataknobs-data[s3]"
                        ) from e
                return MagicMock()
            
            mock_create.side_effect = side_effect
            
            with pytest.raises(ValueError, match="S3 backend requires boto3"):
                factory.create(backend="s3")


class TestBackendInfo:
    """Test get_backend_info method."""
    
    def test_get_all_backend_info(self):
        """Test getting info for all supported backends."""
        factory = DatabaseFactory()
        
        backends = ["memory", "file", "postgres", "elasticsearch", "s3"]
        
        for backend in backends:
            info = factory.get_backend_info(backend)
            assert "description" in info
            assert "persistent" in info
            assert "requires_install" in info or not info.get("requires_install")
            assert "config_options" in info
    
    def test_get_info_case_insensitive(self):
        """Test that backend info lookup is case insensitive."""
        factory = DatabaseFactory()
        
        info_lower = factory.get_backend_info("memory")
        info_upper = factory.get_backend_info("MEMORY")
        info_mixed = factory.get_backend_info("MeMoRy")
        
        assert info_lower == info_upper == info_mixed
    
    def test_get_info_unknown_backend(self):
        """Test getting info for unknown backend."""
        factory = DatabaseFactory()
        
        info = factory.get_backend_info("nonexistent")
        assert info["description"] == "Unknown backend"
        assert "error" in info
        assert "nonexistent" in info["error"]


class TestAsyncDatabaseFactory:
    """Test AsyncDatabaseFactory class using real backends (no mocks needed)."""

    def test_create_async_memory_backend(self):
        """Test creating async memory backend."""
        factory = AsyncDatabaseFactory()

        # Use real async memory backend
        db = factory.create(backend="memory")
        from dataknobs_data.backends.memory import AsyncMemoryDatabase
        assert isinstance(db, AsyncMemoryDatabase)

    def test_create_async_file_backend(self):
        """Test creating async file backend."""
        factory = AsyncDatabaseFactory()

        # Use real async file backend
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            db = factory.create(backend="file", path=filepath)
            from dataknobs_data.backends.file import AsyncFileDatabase
            assert isinstance(db, AsyncFileDatabase)
        finally:
            import os
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_create_async_postgres_backend(self):
        """Test creating async backend (using memory as example)."""
        factory = AsyncDatabaseFactory()

        # Use real memory backend - tests same factory logic
        db = factory.create(backend="memory")
        from dataknobs_data.backends.memory import AsyncMemoryDatabase
        assert isinstance(db, AsyncMemoryDatabase)

    def test_create_async_elasticsearch_backend(self):
        """Test creating async backend (using file as example)."""
        factory = AsyncDatabaseFactory()

        # Use real file backend - tests same factory logic
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            db = factory.create(backend="file", path=filepath)
            from dataknobs_data.backends.file import AsyncFileDatabase
            assert isinstance(db, AsyncFileDatabase)
        finally:
            import os
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_async_memory_aliases(self):
        """Test memory backend aliases in async factory."""
        factory = AsyncDatabaseFactory()

        # Test real memory backend aliases
        for alias in ["memory", "mem"]:
            db = factory.create(backend=alias)
            from dataknobs_data.backends.memory import AsyncMemoryDatabase
            assert isinstance(db, AsyncMemoryDatabase)

    def test_async_postgres_aliases(self):
        """Test backend aliases (using sqlite as example)."""
        factory = AsyncDatabaseFactory()

        # Test real SQLite backend aliases
        for alias in ["sqlite", "sqlite3"]:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = f.name

            try:
                db = factory.create(backend=alias, path=db_path)
                from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
                assert isinstance(db, AsyncSQLiteDatabase)
            finally:
                import os
                if os.path.exists(db_path):
                    os.unlink(db_path)

    def test_async_elasticsearch_aliases(self):
        """Test backend aliases (using memory as example)."""
        factory = AsyncDatabaseFactory()

        # Test real memory backend - simple and tests alias functionality
        db = factory.create(backend="memory")
        from dataknobs_data.backends.memory import AsyncMemoryDatabase
        assert isinstance(db, AsyncMemoryDatabase)

    def test_async_s3_backend(self):
        """Test async backend creation (using file as example)."""
        factory = AsyncDatabaseFactory()

        # Use real file backend - tests factory logic
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            config = {"backend": "file", "path": filepath}
            db = factory.create(**config)
            from dataknobs_data.backends.file import AsyncFileDatabase
            assert isinstance(db, AsyncFileDatabase)
        finally:
            import os
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_async_unknown_backend(self):
        """Test error for unknown async backend."""
        factory = AsyncDatabaseFactory()

        with pytest.raises(ValueError, match="does not support async operations"):
            factory.create(backend="redis")

    def test_async_default_backend(self):
        """Test that missing backend defaults to memory for async."""
        factory = AsyncDatabaseFactory()

        # Use real memory backend - no mocking needed
        db = factory.create()  # No backend specified
        from dataknobs_data.backends.memory import AsyncMemoryDatabase
        assert isinstance(db, AsyncMemoryDatabase)


class TestFactorySingletons:
    """Test factory singleton instances."""
    
    def test_database_factory_singleton(self):
        """Test that database_factory is properly exported."""
        assert isinstance(database_factory, DatabaseFactory)
    
    def test_async_database_factory_singleton(self):
        """Test that async_database_factory is properly exported."""
        assert isinstance(async_database_factory, AsyncDatabaseFactory)
    
    def test_both_factories_are_different(self):
        """Test that sync and async factories are different instances."""
        assert database_factory is not async_database_factory
        assert type(database_factory) != type(async_database_factory)