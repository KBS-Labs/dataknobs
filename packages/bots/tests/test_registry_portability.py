"""Tests for registry portability validation."""

import pytest

from dataknobs_bots.registry import (
    PortabilityError,
    has_resource_references,
    is_portable,
    validate_portability,
)


class TestValidatePortability:
    """Tests for validate_portability function."""

    def test_portable_with_resource_refs(self):
        """Test portable config with $resource references."""
        config = {
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers"},
                "storage": {"$resource": "db", "type": "databases"},
            }
        }

        # Should not raise
        issues = validate_portability(config)
        assert issues == []

    def test_portable_with_env_vars(self):
        """Test portable config with environment variables."""
        config = {
            "database": {
                "connection_string": "${DATABASE_URL}",
                "pool_size": 10,
            },
            "storage": {
                "path": "${DATA_PATH:/default/path}",
            },
        }

        issues = validate_portability(config)
        assert issues == []

    def test_non_portable_macos_path(self):
        """Test detection of macOS home directory paths."""
        config = {
            "storage": {"path": "/Users/developer/data/vectors"},
        }

        with pytest.raises(PortabilityError) as exc_info:
            validate_portability(config)

        assert "macOS home directory" in str(exc_info.value)
        assert "/Users/developer" in str(exc_info.value)

    def test_non_portable_linux_path(self):
        """Test detection of Linux home directory paths."""
        config = {
            "storage": {"path": "/home/developer/data"},
        }

        with pytest.raises(PortabilityError) as exc_info:
            validate_portability(config)

        assert "Linux home directory" in str(exc_info.value)

    def test_non_portable_windows_path(self):
        """Test detection of Windows home directory paths."""
        config = {
            "storage": {"path": "C:\\Users\\developer\\data"},
        }

        with pytest.raises(PortabilityError) as exc_info:
            validate_portability(config)

        assert "Windows home directory" in str(exc_info.value)

    def test_non_portable_localhost(self):
        """Test detection of localhost URLs."""
        config = {
            "api": {"url": "http://localhost:8080/api"},
        }

        with pytest.raises(PortabilityError) as exc_info:
            validate_portability(config)

        assert "localhost" in str(exc_info.value)

    def test_non_portable_localhost_ip(self):
        """Test detection of localhost IP addresses."""
        config = {
            "database": {"host": "127.0.0.1"},
        }

        with pytest.raises(PortabilityError) as exc_info:
            validate_portability(config)

        assert "localhost IP" in str(exc_info.value)

    def test_non_portable_all_interfaces(self):
        """Test detection of 0.0.0.0 binding."""
        config = {
            "server": {"host": "0.0.0.0"},
        }

        with pytest.raises(PortabilityError) as exc_info:
            validate_portability(config)

        assert "all interfaces" in str(exc_info.value)

    def test_multiple_issues(self):
        """Test detection of multiple portability issues."""
        config = {
            "storage": {"path": "/Users/dev/data"},
            "api": {"url": "http://localhost:3000"},
            "db": {"host": "127.0.0.1"},
        }

        with pytest.raises(PortabilityError) as exc_info:
            validate_portability(config)

        error_msg = str(exc_info.value)
        assert "macOS home directory" in error_msg
        assert "localhost" in error_msg
        assert "127.0.0.1" in error_msg

    def test_raise_on_error_false(self):
        """Test returning issues instead of raising."""
        config = {
            "storage": {"path": "/Users/dev/data"},
            "api": {"url": "http://localhost:3000"},
        }

        issues = validate_portability(config, raise_on_error=False)

        assert len(issues) == 2
        assert any("macOS" in issue for issue in issues)
        assert any("localhost" in issue for issue in issues)

    def test_nested_config(self):
        """Test validation works with deeply nested configs."""
        config = {
            "level1": {
                "level2": {
                    "level3": {
                        "path": "/home/user/data",
                    }
                }
            }
        }

        with pytest.raises(PortabilityError):
            validate_portability(config)

    def test_list_values(self):
        """Test validation works with list values."""
        config = {
            "paths": ["/home/user/path1", "/home/user/path2"],
        }

        with pytest.raises(PortabilityError):
            validate_portability(config)

    def test_safe_relative_paths(self):
        """Test relative paths are OK."""
        config = {
            "storage": {"path": "./data/vectors"},
            "config": {"file": "config/settings.yaml"},
        }

        issues = validate_portability(config)
        assert issues == []

    def test_safe_absolute_non_home_paths(self):
        """Test absolute paths outside home directories are OK."""
        config = {
            "storage": {"path": "/var/data/app"},
            "logs": {"path": "/var/log/app"},
        }

        issues = validate_portability(config)
        assert issues == []


class TestHasResourceReferences:
    """Tests for has_resource_references function."""

    def test_config_with_resource_refs(self):
        """Test detecting $resource references."""
        config = {
            "bot": {
                "llm": {"$resource": "default", "type": "llm_providers"},
            }
        }

        assert has_resource_references(config) is True

    def test_config_without_resource_refs(self):
        """Test config without $resource references."""
        config = {
            "bot": {
                "llm": {"provider": "openai", "model": "gpt-4"},
            }
        }

        assert has_resource_references(config) is False

    def test_nested_resource_refs(self):
        """Test deeply nested $resource references."""
        config = {
            "level1": {
                "level2": {
                    "item": {"$resource": "nested", "type": "things"},
                }
            }
        }

        assert has_resource_references(config) is True

    def test_resource_in_string(self):
        """Test $resource as part of a string (false positive check)."""
        config = {
            "message": "Use $resource to reference things",
        }

        # This is technically a false positive, but acceptable
        # since configs shouldn't have this string naturally
        assert has_resource_references(config) is True


class TestIsPortable:
    """Tests for is_portable function."""

    def test_portable_with_resource_refs(self):
        """Test config with $resource is portable."""
        config = {
            "bot": {"llm": {"$resource": "default", "type": "llm_providers"}},
        }

        assert is_portable(config) is True

    def test_portable_resolved_no_local(self):
        """Test resolved config without local paths is portable."""
        config = {
            "bot": {"llm": {"provider": "openai", "model": "gpt-4"}},
        }

        assert is_portable(config) is True

    def test_not_portable_with_local_path(self):
        """Test config with local path is not portable."""
        config = {
            "storage": {"path": "/Users/dev/data"},
        }

        assert is_portable(config) is False

    def test_not_portable_with_localhost(self):
        """Test config with localhost is not portable."""
        config = {
            "api": {"url": "http://localhost:8080"},
        }

        assert is_portable(config) is False

    def test_portable_with_env_vars(self):
        """Test config with env vars is portable."""
        config = {
            "database": {"url": "${DATABASE_URL}"},
        }

        assert is_portable(config) is True


class TestPortabilityError:
    """Tests for PortabilityError exception."""

    def test_exception_message(self):
        """Test exception contains helpful message."""
        config = {"path": "/Users/dev/data"}

        try:
            validate_portability(config)
            pytest.fail("Should have raised PortabilityError")
        except PortabilityError as e:
            msg = str(e)
            assert "portable" in msg.lower()
            assert "$resource" in msg
            assert "Issues found" in msg

    def test_exception_is_catchable(self):
        """Test exception can be caught."""
        config = {"path": "/home/user/data"}

        caught = False
        try:
            validate_portability(config)
        except PortabilityError:
            caught = True

        assert caught is True
