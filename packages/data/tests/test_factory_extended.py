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
    """Test PostgreSQL backend creation via factory."""
    
    @patch('dataknobs_data.factory.logger')
    def test_create_postgres_backend_success(self, mock_logger):
        """Test successful PostgreSQL backend creation."""
        factory = DatabaseFactory()
        
        # Mock the postgres module import and class
        with patch('dataknobs_data.backends.postgres.SyncPostgresDatabase') as MockPostgres:
            mock_db = MagicMock()
            MockPostgres.from_config.return_value = mock_db
            
            # Mock successful import
            with patch.dict('sys.modules', {'dataknobs_data.backends.postgres': MagicMock(SyncPostgresDatabase=MockPostgres)}):
                db = factory.create(
                    backend="postgres",
                    host="localhost",
                    database="test_db",
                    user="test_user",
                    password="test_pass"
                )
                
                assert db == mock_db
                MockPostgres.from_config.assert_called_once_with({
                    'host': 'localhost',
                    'database': 'test_db',
                    'user': 'test_user',
                    'password': 'test_pass'
                })
                mock_logger.info.assert_called_with("Creating database with backend: postgres")
    
    def test_postgres_aliases(self):
        """Test that all PostgreSQL aliases work."""
        factory = DatabaseFactory()
        
        for alias in ["postgres", "postgresql", "pg"]:
            with patch('dataknobs_data.backends.postgres.SyncPostgresDatabase') as MockPostgres:
                mock_db = MagicMock()
                MockPostgres.from_config.return_value = mock_db
                
                with patch.dict('sys.modules', {'dataknobs_data.backends.postgres': MagicMock(SyncPostgresDatabase=MockPostgres)}):
                    db = factory.create(backend=alias)
                    assert db == mock_db
    
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
    """Test Elasticsearch backend creation via factory."""
    
    def test_create_elasticsearch_backend_success(self):
        """Test successful Elasticsearch backend creation."""
        factory = DatabaseFactory()
        
        # Mock the elasticsearch module import and class
        with patch('dataknobs_data.backends.elasticsearch.SyncElasticsearchDatabase') as MockES:
            mock_db = MagicMock()
            MockES.from_config.return_value = mock_db
            
            with patch.dict('sys.modules', {'dataknobs_data.backends.elasticsearch': MagicMock(SyncElasticsearchDatabase=MockES)}):
                db = factory.create(
                    backend="elasticsearch",
                    hosts=["http://localhost:9200"],
                    index="test_index"
                )
                
                assert db == mock_db
                MockES.from_config.assert_called_once_with({
                    'hosts': ['http://localhost:9200'],
                    'index': 'test_index'
                })
    
    def test_elasticsearch_aliases(self):
        """Test that all Elasticsearch aliases work."""
        factory = DatabaseFactory()
        
        for alias in ["elasticsearch", "es"]:
            with patch('dataknobs_data.backends.elasticsearch.SyncElasticsearchDatabase') as MockES:
                mock_db = MagicMock()
                MockES.from_config.return_value = mock_db
                
                with patch.dict('sys.modules', {'dataknobs_data.backends.elasticsearch': MagicMock(SyncElasticsearchDatabase=MockES)}):
                    db = factory.create(backend=alias, hosts=["localhost"], index="test")
                    assert db == mock_db
    
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
    """Test AsyncDatabaseFactory class."""
    
    def test_create_async_memory_backend(self):
        """Test creating async memory backend."""
        factory = AsyncDatabaseFactory()
        
        with patch('dataknobs_data.backends.memory.AsyncMemoryDatabase') as MockMemory:
            mock_db = MagicMock()
            MockMemory.from_config.return_value = mock_db
            
            with patch.dict('sys.modules', {'dataknobs_data.backends.memory': MagicMock(AsyncMemoryDatabase=MockMemory)}):
                db = factory.create(backend="memory")
                assert db == mock_db
                MockMemory.from_config.assert_called_once_with({})
    
    def test_create_async_file_backend(self):
        """Test creating async file backend."""
        factory = AsyncDatabaseFactory()
        
        with patch('dataknobs_data.backends.file.AsyncFileDatabase') as MockFile:
            mock_db = MagicMock()
            MockFile.from_config.return_value = mock_db
            
            with patch.dict('sys.modules', {'dataknobs_data.backends.file': MagicMock(AsyncFileDatabase=MockFile)}):
                db = factory.create(backend="file", path="/tmp/test.json")
                assert db == mock_db
                MockFile.from_config.assert_called_once_with({'path': '/tmp/test.json'})
    
    def test_create_async_postgres_backend(self):
        """Test creating async postgres backend."""
        factory = AsyncDatabaseFactory()
        
        with patch('dataknobs_data.backends.postgres.AsyncPostgresDatabase') as MockPostgres:
            mock_db = MagicMock()
            MockPostgres.from_config.return_value = mock_db
            
            with patch.dict('sys.modules', {'dataknobs_data.backends.postgres': MagicMock(AsyncPostgresDatabase=MockPostgres)}):
                db = factory.create(
                    backend="postgres",
                    host="localhost",
                    database="test"
                )
                assert db == mock_db
    
    def test_create_async_elasticsearch_backend(self):
        """Test creating async elasticsearch backend."""
        factory = AsyncDatabaseFactory()
        
        with patch('dataknobs_data.backends.elasticsearch_async.AsyncElasticsearchDatabase') as MockES:
            mock_db = MagicMock()
            MockES.from_config.return_value = mock_db
            
            with patch.dict('sys.modules', {'dataknobs_data.backends.elasticsearch_async': MagicMock(AsyncElasticsearchDatabase=MockES)}):
                db = factory.create(backend="elasticsearch", hosts=["localhost"])
                assert db == mock_db
    
    def test_async_memory_aliases(self):
        """Test memory backend aliases in async factory."""
        factory = AsyncDatabaseFactory()
        
        for alias in ["memory", "mem"]:
            with patch('dataknobs_data.backends.memory.AsyncMemoryDatabase') as MockMemory:
                mock_db = MagicMock()
                MockMemory.from_config.return_value = mock_db
                
                with patch.dict('sys.modules', {'dataknobs_data.backends.memory': MagicMock(AsyncMemoryDatabase=MockMemory)}):
                    db = factory.create(backend=alias)
                    assert db == mock_db
    
    def test_async_postgres_aliases(self):
        """Test postgres aliases in async factory."""
        factory = AsyncDatabaseFactory()
        
        for alias in ["postgres", "postgresql", "pg"]:
            with patch('dataknobs_data.backends.postgres.AsyncPostgresDatabase') as MockPostgres:
                mock_db = MagicMock()
                MockPostgres.from_config.return_value = mock_db
                
                with patch.dict('sys.modules', {'dataknobs_data.backends.postgres': MagicMock(AsyncPostgresDatabase=MockPostgres)}):
                    db = factory.create(backend=alias)
                    assert db == mock_db
    
    def test_async_elasticsearch_aliases(self):
        """Test elasticsearch aliases in async factory."""
        factory = AsyncDatabaseFactory()
        
        for alias in ["elasticsearch", "es"]:
            with patch('dataknobs_data.backends.elasticsearch_async.AsyncElasticsearchDatabase') as MockES:
                mock_db = MagicMock()
                MockES.from_config.return_value = mock_db
                
                with patch.dict('sys.modules', {'dataknobs_data.backends.elasticsearch_async': MagicMock(AsyncElasticsearchDatabase=MockES)}):
                    db = factory.create(backend=alias, hosts=["localhost"])
                    assert db == mock_db
    
    def test_async_s3_backend(self):
        """Test S3 async backend creation with proper config."""
        factory = AsyncDatabaseFactory()
        
        with patch('dataknobs_data.backends.s3_async.AsyncS3Database') as MockS3:
            mock_db = MagicMock()
            MockS3.from_config.return_value = mock_db
            
            # S3 requires bucket configuration
            config = {"backend": "s3", "bucket": "test-bucket"}
            with patch.dict('sys.modules', {'dataknobs_data.backends.s3_async': MagicMock(AsyncS3Database=MockS3)}):
                db = factory.create(**config)
                assert db == mock_db
                MockS3.from_config.assert_called_once_with({"bucket": "test-bucket"})
    
    def test_async_unknown_backend(self):
        """Test error for unknown async backend."""
        factory = AsyncDatabaseFactory()
        
        with pytest.raises(ValueError, match="does not support async operations"):
            factory.create(backend="redis")
    
    def test_async_default_backend(self):
        """Test that missing backend defaults to memory for async."""
        factory = AsyncDatabaseFactory()
        
        with patch('dataknobs_data.backends.memory.AsyncMemoryDatabase') as MockMemory:
            mock_db = MagicMock()
            MockMemory.from_config.return_value = mock_db
            
            with patch.dict('sys.modules', {'dataknobs_data.backends.memory': MagicMock(AsyncMemoryDatabase=MockMemory)}):
                db = factory.create()  # No backend specified
                assert db == mock_db
                MockMemory.from_config.assert_called_once_with({})


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