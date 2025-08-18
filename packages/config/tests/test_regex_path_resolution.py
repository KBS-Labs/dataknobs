"""Tests for regex pattern support in path resolution."""

from pathlib import Path

from dataknobs_config import Config


def assert_paths_equal(path1, path2):
    """Assert two paths are equal, resolving symlinks."""
    # Resolve symlinks to handle macOS /var -> /private/var
    p1 = Path(path1).resolve()
    p2 = Path(path2).resolve()
    assert p1 == p2, f"{p1} != {p2}"


class TestRegexPathResolution:
    """Test regex pattern support in path_resolution_attributes."""

    def test_regex_pattern_matches_multiple_attributes(self, temp_dir):
        """Test that regex patterns can match multiple attributes."""
        config = Config(
            {
                "database": [
                    {
                        "name": "db1",
                        "data_path": "./data",
                        "log_path": "./logs",
                        "backup_path": "./backups",
                        "port": 5432,  # Not a path
                    }
                ],
                "settings": {
                    "global_root": str(temp_dir),
                    "path_resolution_attributes": [
                        "/.*_path$/"  # Regex to match all attributes ending with "_path"
                    ],
                },
            }
        )

        db = config.get("database", "db1")

        # All *_path attributes should be resolved
        assert_paths_equal(db["data_path"], temp_dir / "data")
        assert_paths_equal(db["log_path"], temp_dir / "logs")
        assert_paths_equal(db["backup_path"], temp_dir / "backups")

        # Non-path attribute should remain unchanged
        assert db["port"] == 5432

    def test_regex_pattern_with_type_specific(self, temp_dir):
        """Test regex patterns with type-specific prefixes."""
        db_root = temp_dir / "database"
        cache_root = temp_dir / "cache"
        db_root.mkdir()
        cache_root.mkdir()

        config = Config(
            {
                "database": [
                    {
                        "name": "db1",
                        "data_dir": "./data",
                        "log_dir": "./logs",
                        "config_file": "./config.yaml",
                    }
                ],
                "cache": [
                    {
                        "name": "redis",
                        "data_dir": "./data",
                        "log_dir": "./logs",
                        "pid_file": "./redis.pid",
                    }
                ],
                "settings": {
                    "database.global_root": str(db_root),
                    "cache.global_root": str(cache_root),
                    "path_resolution_attributes": [
                        "database./.*_dir$/",  # Only resolve *_dir for database type
                        "cache./.*_file$/",  # Only resolve *_file for cache type
                    ],
                },
            }
        )

        db = config.get("database", "db1")
        cache = config.get("cache", "redis")

        # Database: only *_dir should be resolved
        assert_paths_equal(db["data_dir"], db_root / "data")
        assert_paths_equal(db["log_dir"], db_root / "logs")
        assert db["config_file"] == "./config.yaml"  # Not resolved

        # Cache: only *_file should be resolved
        assert cache["data_dir"] == "./data"  # Not resolved
        assert cache["log_dir"] == "./logs"  # Not resolved
        assert_paths_equal(cache["pid_file"], cache_root / "redis.pid")

    def test_mixed_exact_and_regex_patterns(self, temp_dir):
        """Test mixing exact attribute names with regex patterns."""
        config = Config(
            {
                "server": [
                    {
                        "name": "web",
                        "static_dir": "./static",
                        "upload_dir": "./uploads",
                        "template_path": "./templates",
                        "config_path": "./config",
                        "port": 8080,
                    }
                ],
                "settings": {
                    "global_root": str(temp_dir),
                    "path_resolution_attributes": [
                        "config_path",  # Exact match
                        "/.*_dir$/",  # Regex pattern
                    ],
                },
            }
        )

        server = config.get("server", "web")

        # Exact match
        assert_paths_equal(server["config_path"], temp_dir / "config")

        # Regex matches
        assert_paths_equal(server["static_dir"], temp_dir / "static")
        assert_paths_equal(server["upload_dir"], temp_dir / "uploads")

        # Not matched by either
        assert server["template_path"] == "./templates"
        assert server["port"] == 8080

    def test_regex_pattern_with_groups(self, temp_dir):
        """Test regex patterns with capture groups."""
        config = Config(
            {
                "app": [
                    {
                        "name": "myapp",
                        "path_to_data": "./data",
                        "path_to_logs": "./logs",
                        "path_to_cache": "./cache",
                        "url_to_api": "http://api.example.com",  # Not a file path
                    }
                ],
                "settings": {
                    "global_root": str(temp_dir),
                    "path_resolution_attributes": [
                        "/^path_to_.*/"  # Match attributes starting with "path_to_"
                    ],
                },
            }
        )

        app = config.get("app", "myapp")

        # Matched by regex
        assert_paths_equal(app["path_to_data"], temp_dir / "data")
        assert_paths_equal(app["path_to_logs"], temp_dir / "logs")
        assert_paths_equal(app["path_to_cache"], temp_dir / "cache")

        # Not matched (starts with url_to_)
        assert app["url_to_api"] == "http://api.example.com"

    def test_invalid_regex_pattern_ignored(self, temp_dir):
        """Test that invalid regex patterns are silently ignored."""
        config = Config(
            {
                "database": [{"name": "db1", "data_dir": "./data", "log_file": "./app.log"}],
                "settings": {
                    "global_root": str(temp_dir),
                    "path_resolution_attributes": [
                        "/[invalid(regex/",  # Invalid regex
                        "data_dir",  # Valid exact match
                    ],
                },
            }
        )

        db = config.get("database", "db1")

        # Invalid regex is ignored, but valid exact match works
        assert_paths_equal(db["data_dir"], temp_dir / "data")
        assert db["log_file"] == "./app.log"

    def test_regex_case_sensitive(self, temp_dir):
        """Test that regex patterns are case-sensitive by default."""
        config = Config(
            {
                "server": [
                    {
                        "name": "web",
                        "data_Path": "./data1",  # Capital P
                        "data_path": "./data2",  # Lowercase p
                        "DATA_PATH": "./data3",  # All caps
                    }
                ],
                "settings": {
                    "global_root": str(temp_dir),
                    "path_resolution_attributes": [
                        "/data_path/"  # Should only match lowercase
                    ],
                },
            }
        )

        server = config.get("server", "web")

        # Only exact case match
        assert server["data_Path"] == "./data1"  # Not matched
        assert_paths_equal(server["data_path"], temp_dir / "data2")  # Matched
        assert server["DATA_PATH"] == "./data3"  # Not matched

    def test_regex_with_special_characters(self, temp_dir):
        """Test regex patterns with special characters in attribute names."""
        config = Config(
            {
                "service": [
                    {
                        "name": "api",
                        "base.path": "./base",
                        "config-dir": "./config",
                        "data_dir": "./data",
                        "log@path": "./logs",
                    }
                ],
                "settings": {
                    "global_root": str(temp_dir),
                    "path_resolution_attributes": [
                        "/.*[-.].*/"  # Match attributes containing - or .
                    ],
                },
            }
        )

        service = config.get("service", "api")

        # Matched by regex (contain - or .)
        assert_paths_equal(service["base.path"], temp_dir / "base")
        assert_paths_equal(service["config-dir"], temp_dir / "config")

        # Not matched
        assert service["data_dir"] == "./data"
        assert service["log@path"] == "./logs"
