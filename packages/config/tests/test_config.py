"""Tests for the core Config class."""

import json
from pathlib import Path

import pytest
import yaml

from dataknobs_config import Config, ConfigNotFoundError, ValidationError


class TestConfigBasics:
    """Test basic Config functionality."""

    def test_empty_config(self):
        """Test creating an empty config."""
        config = Config()
        assert config.get_types() == []
        assert config.to_dict() == {}

    def test_from_dict(self, sample_config_dict):
        """Test creating config from dictionary."""
        config = Config.from_dict(sample_config_dict)

        assert "database" in config.get_types()
        assert "cache" in config.get_types()
        assert config.get_count("database") == 2
        assert config.get_count("cache") == 1

    def test_get_by_index(self, sample_config_dict):
        """Test getting configuration by index."""
        config = Config(sample_config_dict)

        db0 = config.get("database", 0)
        assert db0["name"] == "primary"
        assert db0["host"] == "localhost"

        db1 = config.get("database", 1)
        assert db1["name"] == "secondary"
        assert db1["host"] == "backup.example.com"

    def test_get_by_name(self, sample_config_dict):
        """Test getting configuration by name."""
        config = Config(sample_config_dict)

        db = config.get("database", "primary")
        assert db["host"] == "localhost"
        assert db["port"] == 5432

        cache = config.get("cache", "redis")
        assert cache["host"] == "localhost"
        assert cache["port"] == 6379

    def test_get_negative_index(self, sample_config_dict):
        """Test getting configuration by negative index."""
        config = Config(sample_config_dict)

        # Get last database
        db = config.get("database", -1)
        assert db["name"] == "secondary"

    def test_get_names(self, sample_config_dict):
        """Test getting configuration names."""
        config = Config(sample_config_dict)

        names = config.get_names("database")
        assert names == ["primary", "secondary"]

    def test_set_config(self):
        """Test setting a configuration."""
        config = Config()

        # Set by index
        config.set("database", 0, {"name": "test", "host": "localhost"})
        assert config.get_count("database") == 1

        db = config.get("database", 0)
        assert db["name"] == "test"
        assert db["host"] == "localhost"

        # Set by name
        config.set("database", "prod", {"host": "prod.example.com"})
        assert config.get_count("database") == 2

        db = config.get("database", "prod")
        assert db["name"] == "prod"
        assert db["host"] == "prod.example.com"

    def test_config_not_found(self):
        """Test ConfigNotFoundError."""
        config = Config()

        with pytest.raises(ConfigNotFoundError):
            config.get("nonexistent", 0)

        config.set("database", 0, {"name": "test"})

        with pytest.raises(ConfigNotFoundError):
            config.get("database", "nonexistent")

        with pytest.raises(ConfigNotFoundError):
            config.get("database", 10)


class TestFileLoading:
    """Test configuration file loading."""

    def test_load_yaml(self):
        """Test loading YAML configuration."""
        yaml_path = Path(__file__).parent / "fixtures" / "test_config.yaml"
        config = Config.from_file(yaml_path)

        assert "database" in config.get_types()
        assert "cache" in config.get_types()
        assert "api" in config.get_types()

        db = config.get("database", "primary")
        assert db["host"] == "localhost"
        assert db["database"] == "testdb"

    def test_load_json(self):
        """Test loading JSON configuration."""
        json_path = Path(__file__).parent / "fixtures" / "test_config.json"
        config = Config.from_file(json_path)

        assert "server" in config.get_types()
        assert "logging" in config.get_types()

        server = config.get("server", "web")
        assert server["port"] == 8000
        assert server["debug"] is False

    def test_load_multiple_sources(self):
        """Test loading from multiple sources."""
        yaml_path = Path(__file__).parent / "fixtures" / "test_config.yaml"
        json_path = Path(__file__).parent / "fixtures" / "test_config.json"

        config = Config(yaml_path, json_path)

        # Should have types from both files
        assert "database" in config.get_types()  # from YAML
        assert "server" in config.get_types()  # from JSON

    def test_file_reference(self, temp_dir):
        """Test @-prefixed file references."""
        # Create atomic config file
        atomic_file = temp_dir / "atomic.yaml"
        atomic_file.write_text(yaml.safe_dump({"name": "referenced", "value": 42}))

        # Create main config with reference
        main_config = {"mytype": ["@atomic.yaml"], "settings": {"config_root": str(temp_dir)}}

        config = Config(main_config)

        obj = config.get("mytype", "referenced")
        assert obj["value"] == 42

    def test_nonexistent_file(self):
        """Test loading nonexistent file."""
        with pytest.raises(Exception):  # FileNotFoundError or ConfigFileNotFoundError
            Config.from_file("/nonexistent/file.yaml")


class TestDefaults:
    """Test default value handling."""

    def test_global_defaults(self):
        """Test global default values."""
        config_dict = {
            "server": [{"name": "web"}],
            "settings": {"default_port": 8080, "default_host": "0.0.0.0"},
        }

        config = Config(config_dict)
        server = config.get("server", "web")

        # Global defaults should be applied
        assert server.get("default_port") == 8080
        assert server.get("default_host") == "0.0.0.0"

    def test_type_specific_defaults(self):
        """Test type-specific default values."""
        config_dict = {
            "database": [{"name": "db1"}],
            "cache": [{"name": "cache1"}],
            "settings": {"database.default_pool_size": 10, "cache.default_ttl": 3600},
        }

        config = Config(config_dict)

        db = config.get("database", "db1")
        assert db.get("default_pool_size") == 10

        cache = config.get("cache", "cache1")
        assert cache.get("default_ttl") == 3600

    def test_default_precedence(self):
        """Test that explicit values override defaults."""
        config_dict = {
            "server": [{"name": "web", "port": 9000}],
            "settings": {"default_port": 8080},
        }

        config = Config(config_dict)
        server = config.get("server", "web")

        # Explicit value should not be overridden
        assert server["port"] == 9000


class TestMerging:
    """Test configuration merging."""

    def test_merge_configs(self):
        """Test merging two configurations."""
        config1 = Config({"database": [{"name": "db1", "host": "host1"}]})
        config2 = Config({"database": [{"name": "db2", "host": "host2"}]})

        config1.merge(config2)

        assert config1.get_count("database") == 2
        assert config1.get("database", "db1")["host"] == "host1"
        assert config1.get("database", "db2")["host"] == "host2"

    def test_merge_precedence_first(self):
        """Test merge with 'first' precedence."""
        config1 = Config(
            {"database": [{"name": "db", "host": "host1"}], "settings": {"timeout": 30}}
        )
        config2 = Config(
            {"database": [{"name": "db", "host": "host2"}], "settings": {"timeout": 60}}
        )

        config1.merge(config2, precedence="first")

        # First config's values should be kept
        assert config1.get("database", "db")["host"] == "host1"
        # Settings are handled differently in the implementation

    def test_merge_precedence_last(self):
        """Test merge with 'last' precedence."""
        config1 = Config({"database": [{"name": "db", "host": "host1"}]})
        config2 = Config({"database": [{"name": "db", "host": "host2"}]})

        config1.merge(config2, precedence="last")

        # Last config's values should be used
        assert config1.get("database", "db")["host"] == "host2"


class TestExport:
    """Test configuration export."""

    def test_to_dict(self, sample_config_dict):
        """Test exporting to dictionary."""
        config = Config(sample_config_dict)
        exported = config.to_dict()

        assert "database" in exported
        assert "cache" in exported
        assert "settings" in exported
        assert len(exported["database"]) == 2

    def test_to_yaml_file(self, temp_dir, sample_config_dict):
        """Test saving to YAML file."""
        config = Config(sample_config_dict)

        output_path = temp_dir / "output.yaml"
        config.to_file(output_path)

        assert output_path.exists()

        with open(output_path) as f:
            loaded = yaml.safe_load(f)

        assert "database" in loaded
        assert len(loaded["database"]) == 2

    def test_to_json_file(self, temp_dir, sample_config_dict):
        """Test saving to JSON file."""
        config = Config(sample_config_dict)

        output_path = temp_dir / "output.json"
        config.to_file(output_path)

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert "database" in loaded
        assert len(loaded["database"]) == 2


class TestAtomicConfig:
    """Test atomic configuration handling."""

    def test_auto_name_single(self):
        """Test automatic naming for single config."""
        config = Config({"database": {"host": "localhost"}})

        db = config.get("database", 0)
        assert db["name"] == "0"

    def test_auto_name_multiple(self):
        """Test automatic naming for multiple configs."""
        config = Config({"database": [{"host": "host1"}, {"host": "host2"}, {"host": "host3"}]})

        assert config.get("database", 0)["name"] == "0"
        assert config.get("database", 1)["name"] == "1"
        assert config.get("database", 2)["name"] == "2"

    def test_type_attribute(self):
        """Test that type attribute is set correctly."""
        config = Config({"database": [{"name": "db1"}]})

        db = config.get("database", "db1")
        assert db["type"] == "database"

    def test_type_mismatch(self):
        """Test that type mismatch raises error."""
        with pytest.raises(ValidationError):
            Config({"database": [{"type": "cache", "name": "wrong"}]})
