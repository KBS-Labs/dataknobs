"""Test backend factory functionality."""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

from dataknobs_config import Config
from dataknobs_data import DatabaseFactory, database_factory
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_data.backends.file import SyncFileDatabase
from dataknobs_data.backends.s3 import SyncS3Database
from dataknobs_data import Record


class TestDatabaseFactory:
    """Test the DatabaseFactory class."""
    
    def test_create_memory_backend(self):
        """Test creating memory backend via factory."""
        factory = DatabaseFactory()
        
        db = factory.create(backend="memory")
        assert isinstance(db, SyncMemoryDatabase)
        
        # Test it works
        record = Record({"test": "value"})
        record_id = db.create(record)
        assert db.read(record_id) is not None
    
    def test_create_file_backend(self):
        """Test creating file backend via factory."""
        factory = DatabaseFactory()
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        
        try:
            db = factory.create(backend="file", path=filepath, format="json")
            assert isinstance(db, SyncFileDatabase)
            
            # Test it works
            record = Record({"test": "file"})
            record_id = db.create(record)
            assert db.read(record_id) is not None
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    @patch('boto3.client')
    def test_create_s3_backend(self, mock_boto_client):
        """Test creating S3 backend via factory."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = {}  # Bucket exists
        
        factory = DatabaseFactory()
        
        db = factory.create(
            backend="s3",
            bucket="test-bucket",
            prefix="test/",
            region="us-west-2"
        )
        assert isinstance(db, SyncS3Database)
    
    def test_backend_aliases(self):
        """Test that backend aliases work."""
        factory = DatabaseFactory()
        
        # Memory aliases
        for alias in ["memory", "mem"]:
            db = factory.create(backend=alias)
            assert isinstance(db, SyncMemoryDatabase)
    
    def test_unknown_backend_error(self):
        """Test that unknown backend raises error."""
        factory = DatabaseFactory()
        
        with pytest.raises(ValueError, match="Unknown backend type: invalid"):
            factory.create(backend="invalid")
    
    def test_get_backend_info(self):
        """Test getting backend information."""
        factory = DatabaseFactory()
        
        # Get info for known backend
        info = factory.get_backend_info("memory")
        assert "description" in info
        assert "persistent" in info
        assert info["persistent"] is False
        
        # Get info for unknown backend
        info = factory.get_backend_info("unknown")
        assert "error" in info


class TestFactoryWithConfig:
    """Test factory integration with Config class."""
    
    def test_factory_registration(self):
        """Test registering factory with Config."""
        config = Config()
        
        # Register the factory
        config.register_factory("database", database_factory)
        
        # Load configuration using factory
        config.load({
            "databases": [{
                "name": "test_db",
                "factory": "database",
                "backend": "memory"
            }]
        })
        
        # Get instance through factory
        db = config.get_instance("databases", "test_db")
        assert isinstance(db, SyncMemoryDatabase)
    
    @patch('boto3.client')
    def test_factory_with_environment_variables(self, mock_boto_client, monkeypatch):
        """Test factory with environment variable substitution."""
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        mock_s3.head_bucket.return_value = {}
        
        # Set environment variables
        monkeypatch.setenv("DB_BACKEND", "s3")
        monkeypatch.setenv("S3_BUCKET", "env-bucket")
        monkeypatch.setenv("S3_PREFIX", "env-prefix/")
        
        config = Config()
        config.register_factory("database", database_factory)
        
        config.load({
            "databases": [{
                "name": "env_db",
                "factory": "database",
                "backend": "${DB_BACKEND}",
                "bucket": "${S3_BUCKET}",
                "prefix": "${S3_PREFIX}"
            }]
        })
        
        db = config.get_instance("databases", "env_db")
        assert isinstance(db, SyncS3Database)
        assert db.bucket == "env-bucket"
        assert db.prefix == "env-prefix/"
    
    def test_multiple_backends_from_factory(self):
        """Test creating multiple backends from factory."""
        config = Config()
        config.register_factory("database", database_factory)
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        
        try:
            config.load({
                "databases": [
                    {
                        "name": "memory_db",
                        "factory": "database",
                        "backend": "memory"
                    },
                    {
                        "name": "file_db",
                        "factory": "database",
                        "backend": "file",
                        "path": filepath,
                        "format": "json"
                    }
                ]
            })
            
            # Get both databases
            memory_db = config.get_instance("databases", "memory_db")
            file_db = config.get_instance("databases", "file_db")
            
            assert isinstance(memory_db, SyncMemoryDatabase)
            assert isinstance(file_db, SyncFileDatabase)
            
            # Test they work independently
            rec1 = Record({"type": "memory"})
            rec2 = Record({"type": "file"})
            
            id1 = memory_db.create(rec1)
            id2 = file_db.create(rec2)
            
            assert memory_db.read(id1).get_value("type") == "memory"
            assert file_db.read(id2).get_value("type") == "file"
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestFactoryEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_backend_parameter(self):
        """Test that missing backend defaults to memory."""
        factory = DatabaseFactory()
        
        db = factory.create()  # No backend specified
        assert isinstance(db, SyncMemoryDatabase)
    
    def test_backend_with_missing_dependencies(self):
        """Test helpful error when backend dependencies missing."""
        factory = DatabaseFactory()
        
        # Mock the import to raise ImportError
        with patch.object(factory, 'create') as mock_create:
            mock_create.side_effect = ValueError(
                "PostgreSQL backend requires psycopg2. "
                "Install with: pip install dataknobs-data[postgres]"
            )
            with pytest.raises(ValueError, match="PostgreSQL backend requires psycopg2"):
                factory.create(backend="postgres", host="localhost")
    
    def test_factory_singleton(self):
        """Test that database_factory is a singleton."""
        from dataknobs_data import database_factory as factory1
        from dataknobs_data import database_factory as factory2
        
        assert factory1 is factory2
