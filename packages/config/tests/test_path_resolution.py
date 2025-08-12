"""Tests for path resolution functionality."""

from pathlib import Path

import pytest

from dataknobs_config import Config


class TestPathResolution:
    """Test path resolution in configurations."""

    def test_resolve_relative_paths(self, temp_dir):
        """Test resolving relative paths to absolute."""
        config = Config(
            {
                "database": [
                    {"name": "db1", "data_dir": "./data", "config_path": "../configs/db.conf"}
                ],
                "settings": {
                    "global_root": str(temp_dir),
                    "path_resolution_attributes": ["data_dir", "config_path"],
                },
            }
        )

        db = config.get("database", "db1")

        # Paths should be resolved to absolute
        assert Path(db["data_dir"]).is_absolute()
        assert db["data_dir"] == str(temp_dir / "data")

        assert Path(db["config_path"]).is_absolute()
        assert db["config_path"] == str(temp_dir.parent / "configs" / "db.conf")

    def test_type_specific_path_attributes(self, temp_dir):
        """Test type-specific path resolution attributes."""
        config = Config(
            {
                "database": [{"name": "db1", "data_dir": "./db_data", "log_file": "./logs/db.log"}],
                "cache": [
                    {"name": "redis", "data_dir": "./cache_data", "log_file": "./logs/cache.log"}
                ],
                "settings": {
                    "global_root": str(temp_dir),
                    "path_resolution_attributes": [
                        "database.data_dir",  # Only for database type
                        "log_file",  # For all types
                    ],
                },
            }
        )

        db = config.get("database", "db1")
        cache = config.get("cache", "redis")

        # Database data_dir should be resolved
        assert Path(db["data_dir"]).is_absolute()

        # Cache data_dir should NOT be resolved (not in attributes)
        assert db["data_dir"] == str(temp_dir / "db_data")
        assert cache["data_dir"] == "./cache_data"

        # Log files should be resolved for both
        assert Path(db["log_file"]).is_absolute()
        assert Path(cache["log_file"]).is_absolute()

    def test_global_root(self, temp_dir):
        """Test using global_root for path resolution."""
        global_root = temp_dir / "app"
        global_root.mkdir()

        config = Config(
            {
                "server": [
                    {"name": "web", "static_dir": "./static", "template_dir": "./templates"}
                ],
                "settings": {
                    "global_root": str(global_root),
                    "path_resolution_attributes": ["static_dir", "template_dir"],
                },
            }
        )

        server = config.get("server", "web")

        # Should use global_root for resolution
        assert server["static_dir"] == str(global_root / "static")
        assert server["template_dir"] == str(global_root / "templates")

    def test_type_specific_root(self, temp_dir):
        """Test type-specific root directories."""
        db_root = temp_dir / "database"
        cache_root = temp_dir / "cache"
        db_root.mkdir()
        cache_root.mkdir()

        config = Config(
            {
                "database": [{"name": "db1", "data_dir": "./data"}],
                "cache": [{"name": "redis", "data_dir": "./data"}],
                "settings": {
                    "database.global_root": str(db_root),
                    "cache.global_root": str(cache_root),
                    "path_resolution_attributes": ["data_dir"],
                },
            }
        )

        db = config.get("database", "db1")
        cache = config.get("cache", "redis")

        # Each should use its type-specific root
        assert db["data_dir"] == str(db_root / "data")
        assert cache["data_dir"] == str(cache_root / "data")

    def test_root_precedence(self, temp_dir):
        """Test precedence of root directories."""
        config_root = temp_dir / "config"
        global_root = temp_dir / "global"
        type_root = temp_dir / "type_specific"

        for d in [config_root, global_root, type_root]:
            d.mkdir()

        config = Config(
            {
                "database": [{"name": "db1", "path": "./file"}],
                "cache": [{"name": "c1", "path": "./file"}],
                "server": [{"name": "s1", "path": "./file"}],
                "settings": {
                    "config_root": str(config_root),
                    "global_root": str(global_root),
                    "database.global_root": str(type_root),
                    "path_resolution_attributes": ["path"],
                },
            }
        )

        # Database should use type-specific root (highest precedence)
        db = config.get("database", "db1")
        assert db["path"] == str(type_root / "file")

        # Cache should use global_root (no type-specific)
        cache = config.get("cache", "c1")
        assert cache["path"] == str(global_root / "file")

        # Server should also use global_root
        server = config.get("server", "s1")
        assert server["path"] == str(global_root / "file")

    def test_absolute_paths_unchanged(self, temp_dir):
        """Test that absolute paths are not modified."""
        abs_path = "/absolute/path/to/file"

        config = Config(
            {
                "database": [{"name": "db1", "data_dir": abs_path, "relative_dir": "./relative"}],
                "settings": {
                    "global_root": str(temp_dir),
                    "path_resolution_attributes": ["data_dir", "relative_dir"],
                },
            }
        )

        db = config.get("database", "db1")

        # Absolute path should remain unchanged
        assert db["data_dir"] == abs_path

        # Relative path should be resolved
        assert db["relative_dir"] == str(temp_dir / "relative")

    def test_non_string_values_unchanged(self, temp_dir):
        """Test that non-string values are not affected by path resolution."""
        config = Config(
            {
                "database": [
                    {
                        "name": "db1",
                        "port": 5432,  # Integer
                        "enabled": True,  # Boolean
                        "data_dir": "./data",  # String path
                        "options": {"key": "value"},  # Dict
                    }
                ],
                "settings": {
                    "global_root": str(temp_dir),
                    "path_resolution_attributes": ["port", "enabled", "data_dir", "options"],
                },
            }
        )

        db = config.get("database", "db1")

        # Non-string values should be unchanged
        assert db["port"] == 5432
        assert isinstance(db["port"], int)

        assert db["enabled"] is True
        assert isinstance(db["enabled"], bool)

        assert db["options"] == {"key": "value"}
        assert isinstance(db["options"], dict)

        # Only string path should be resolved
        assert db["data_dir"] == str(temp_dir / "data")

    def test_no_global_root_raises_for_relative_paths(self):
        """Test that relative paths raise an error when no global_root is set."""
        from dataknobs_config.exceptions import ConfigError

        with pytest.raises(ConfigError, match="Cannot resolve relative path.*no global_root"):
            config = Config(
                {
                    "database": [{"name": "db1", "data_dir": "./data"}],
                    "settings": {
                        "path_resolution_attributes": ["data_dir"]
                        # Note: no global_root or config_root set
                    },
                }
            )

    def test_no_global_root_ok_for_absolute_paths(self):
        """Test that absolute paths work fine without global_root."""
        config = Config(
            {
                "database": [{"name": "db1", "data_dir": "/absolute/path/data"}],
                "settings": {
                    "path_resolution_attributes": ["data_dir"]
                    # Note: no global_root set
                },
            }
        )

        db = config.get("database", "db1")

        # Absolute paths should work without global_root
        assert db["data_dir"] == "/absolute/path/data"

    def test_path_resolution_with_file_loading(self, temp_dir):
        """Test path resolution when loading from files."""
        config_file = temp_dir / "config.yaml"
        # Write YAML with explicit global_root
        config_file.write_text(f"""
database:
  - name: db1
    data_dir: ./data
    backup_dir: ../backups
settings:
  global_root: {temp_dir}
  path_resolution_attributes:
    - data_dir
    - backup_dir
""")

        config = Config.from_file(config_file)

        db = config.get("database", "db1")

        # Paths should be resolved relative to global_root
        assert Path(db["data_dir"]).is_absolute()
        assert db["data_dir"] == str(temp_dir / "data")
        assert db["backup_dir"] == str(temp_dir.parent / "backups")
