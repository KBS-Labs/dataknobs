"""Test config package integration with data backends."""

import pytest

from dataknobs_config import Config
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_data.backends.postgres import SyncPostgresDatabase
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.file import SyncFileDatabase
from dataknobs_data.records import Record


class TestConfigIntegration:
    """Test that backends integrate properly with config package."""
    
    def test_memory_backend_from_config(self):
        """Test MemoryDatabase can be created via from_config."""
        db = SyncMemoryDatabase.from_config({})
        assert db is not None
        
        # Verify it works
        record = Record({"test": "value"})
        id = db.create(record)
        retrieved = db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("test") == "value"
    
    def test_file_backend_from_config(self, tmp_path):
        """Test FileDatabase can be created via from_config."""
        config = {
            "path": str(tmp_path / "test.json"),
            "format": "json"
        }
        db = SyncFileDatabase.from_config(config)
        assert db is not None
        assert db.filepath == str(tmp_path / "test.json")
        
        # Verify it works
        record = Record({"test": "value"})
        id = db.create(record)
        retrieved = db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("test") == "value"
    
    def test_config_build_memory(self):
        """Test building MemoryDatabase via Config.build_object()."""
        config = Config()
        config.load({
            "databases": [{
                "name": "test_db",
                "class": "dataknobs_data.backends.memory.SyncMemoryDatabase"
            }]
        })
        
        db = config.get_instance("databases", "test_db")
        assert isinstance(db, SyncMemoryDatabase)
        
        # Verify it works
        record = Record({"name": "test", "value": 42})
        id = db.create(record)
        retrieved = db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("name") == "test"
        assert retrieved.get_value("value") == 42
    
    def test_config_build_file_with_params(self, tmp_path):
        """Test building FileDatabase with parameters via Config.get_instance()."""
        config = Config()
        config.load({
            "databases": [{
                "name": "file_db",
                "class": "dataknobs_data.backends.file.SyncFileDatabase",
                "path": str(tmp_path / "config_test.json"),
                "format": "json"
            }]
        })
        
        db = config.get_instance("databases", "file_db")
        assert isinstance(db, SyncFileDatabase)
        assert db.filepath == str(tmp_path / "config_test.json")
        
        # Verify it works
        record = Record({"config": "test"})
        id = db.create(record)
        retrieved = db.read(id)
        assert retrieved is not None
        assert retrieved.get_value("config") == "test"
    
    def test_multiple_backend_configs(self, tmp_path):
        """Test configuring multiple backends in a single config."""
        config = Config()
        config.load({
            "databases": [
                {
                    "name": "memory",
                    "class": "dataknobs_data.backends.memory.SyncMemoryDatabase"
                },
                {
                    "name": "file",
                    "class": "dataknobs_data.backends.file.SyncFileDatabase",
                    "path": str(tmp_path / "multi.csv"),
                    "format": "csv"
                }
            ]
        })
        
        # Build both databases
        memory_db = config.get_instance("databases", "memory")
        file_db = config.get_instance("databases", "file")
        
        assert isinstance(memory_db, SyncMemoryDatabase)
        assert isinstance(file_db, SyncFileDatabase)
        assert file_db.format == "csv"
        
        # Test they work independently
        record1 = Record({"db": "memory"})
        record2 = Record({"db": "file"})
        
        id1 = memory_db.create(record1)
        id2 = file_db.create(record2)
        
        assert memory_db.read(id1).get_value("db") == "memory"
        assert file_db.read(id2).get_value("db") == "file"
    
    @pytest.mark.parametrize("backend_class,config_params", [
        ("dataknobs_data.backends.memory.SyncMemoryDatabase", {}),
        ("dataknobs_data.backends.memory.AsyncMemoryDatabase", {}),
        ("dataknobs_data.backends.file.SyncFileDatabase", {"path": "/tmp/test.json"}),
        ("dataknobs_data.backends.file.AsyncFileDatabase", {"path": "/tmp/test.json"}),
    ])
    def test_all_backends_configurable(self, backend_class, config_params):
        """Test that all backend classes can be instantiated via config."""
        config = Config()
        config.load({
            "databases": [{
                "name": "test",
                "class": backend_class,
                **config_params
            }]
        })
        
        # Should not raise an exception
        db = config.get_instance("databases", "test")
        assert db is not None