"""Tests for EnvironmentConfig class."""

import os
from pathlib import Path

import pytest
import yaml

from dataknobs_config.environment_config import (
    EnvironmentConfig,
    EnvironmentConfigError,
    ResourceBinding,
    ResourceNotFoundError,
)


class TestEnvironmentDetection:
    """Test environment auto-detection."""

    def test_detect_from_dataknobs_env(self, monkeypatch):
        """Test detection from DATAKNOBS_ENVIRONMENT."""
        monkeypatch.setenv("DATAKNOBS_ENVIRONMENT", "staging")
        assert EnvironmentConfig.detect_environment() == "staging"

    def test_detect_case_insensitive(self, monkeypatch):
        """Test that environment name is lowercased."""
        monkeypatch.setenv("DATAKNOBS_ENVIRONMENT", "PRODUCTION")
        assert EnvironmentConfig.detect_environment() == "production"

    def test_detect_aws_lambda(self, monkeypatch):
        """Test detection in AWS Lambda."""
        monkeypatch.delenv("DATAKNOBS_ENVIRONMENT", raising=False)
        monkeypatch.setenv("AWS_EXECUTION_ENV", "AWS_Lambda_python3.9")
        assert EnvironmentConfig.detect_environment() == "production"

    def test_detect_aws_lambda_with_environment(self, monkeypatch):
        """Test detection in AWS Lambda with ENVIRONMENT var."""
        monkeypatch.delenv("DATAKNOBS_ENVIRONMENT", raising=False)
        monkeypatch.setenv("AWS_EXECUTION_ENV", "AWS_Lambda_python3.9")
        monkeypatch.setenv("ENVIRONMENT", "staging")
        assert EnvironmentConfig.detect_environment() == "staging"

    def test_detect_kubernetes(self, monkeypatch):
        """Test detection in Kubernetes."""
        monkeypatch.delenv("DATAKNOBS_ENVIRONMENT", raising=False)
        monkeypatch.delenv("AWS_EXECUTION_ENV", raising=False)
        monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
        assert EnvironmentConfig.detect_environment() == "production"

    def test_detect_default_development(self, monkeypatch):
        """Test default to development."""
        for var in ["DATAKNOBS_ENVIRONMENT", "AWS_EXECUTION_ENV",
                    "KUBERNETES_SERVICE_HOST", "K_SERVICE", "FUNCTIONS_WORKER_RUNTIME",
                    "ECS_CONTAINER_METADATA_URI"]:
            monkeypatch.delenv(var, raising=False)
        assert EnvironmentConfig.detect_environment() == "development"


class TestEnvironmentConfigLoading:
    """Test EnvironmentConfig loading from files."""

    @pytest.fixture
    def env_dir(self, tmp_path):
        """Create temporary environment config directory."""
        env_dir = tmp_path / "environments"
        env_dir.mkdir()
        return env_dir

    def test_load_yaml_config(self, env_dir):
        """Test loading YAML environment config."""
        config_file = env_dir / "production.yaml"
        config_file.write_text(yaml.dump({
            "name": "production",
            "description": "Production environment",
            "settings": {"log_level": "INFO"},
            "resources": {
                "databases": {
                    "default": {
                        "backend": "postgres",
                        "connection_string": "${DATABASE_URL}"
                    }
                }
            }
        }))

        env = EnvironmentConfig.load("production", config_dir=env_dir)

        assert env.name == "production"
        assert env.description == "Production environment"
        assert env.settings["log_level"] == "INFO"
        assert "default" in env.resources["databases"]

    def test_load_yml_extension(self, env_dir):
        """Test loading .yml extension."""
        config_file = env_dir / "staging.yml"
        config_file.write_text("name: staging\nsettings:\n  debug: true")

        env = EnvironmentConfig.load("staging", config_dir=env_dir)
        assert env.name == "staging"
        assert env.settings["debug"] is True

    def test_load_json_config(self, env_dir):
        """Test loading JSON environment config."""
        config_file = env_dir / "development.json"
        config_file.write_text('{"name": "development", "settings": {"debug": true}}')

        env = EnvironmentConfig.load("development", config_dir=env_dir)
        assert env.name == "development"
        assert env.settings["debug"] is True

    def test_load_missing_config(self, env_dir):
        """Test loading non-existent config returns empty."""
        env = EnvironmentConfig.load("nonexistent", config_dir=env_dir)
        assert env.name == "nonexistent"
        assert env.resources == {}
        assert env.settings == {}

    def test_load_auto_detect(self, env_dir, monkeypatch):
        """Test loading with auto-detected environment."""
        monkeypatch.setenv("DATAKNOBS_ENVIRONMENT", "production")

        config_file = env_dir / "production.yaml"
        config_file.write_text("name: production")

        env = EnvironmentConfig.load(config_dir=env_dir)
        assert env.name == "production"

    def test_load_invalid_yaml(self, env_dir):
        """Test error on invalid YAML."""
        config_file = env_dir / "invalid.yaml"
        config_file.write_text("key: [unclosed")

        with pytest.raises(EnvironmentConfigError, match="Failed to parse"):
            EnvironmentConfig.load("invalid", config_dir=env_dir)

    def test_load_non_dict_config(self, env_dir):
        """Test error when config is not a dict."""
        config_file = env_dir / "list.yaml"
        config_file.write_text("- item1\n- item2")

        with pytest.raises(EnvironmentConfigError, match="must be a dictionary"):
            EnvironmentConfig.load("list", config_dir=env_dir)


class TestResourceAccess:
    """Test resource access methods."""

    @pytest.fixture
    def env_config(self):
        """Create sample environment config."""
        return EnvironmentConfig(
            name="test",
            resources={
                "databases": {
                    "default": {"backend": "postgres", "host": "localhost"},
                    "analytics": {"backend": "clickhouse", "host": "analytics.local"},
                },
                "vector_stores": {
                    "default": {"backend": "pgvector", "dimensions": 1536},
                },
            },
            settings={"log_level": "DEBUG", "enable_metrics": True},
        )

    def test_get_resource(self, env_config):
        """Test getting a resource by type and name."""
        db = env_config.get_resource("databases", "default")
        assert db["backend"] == "postgres"
        assert db["host"] == "localhost"

    def test_get_resource_returns_copy(self, env_config):
        """Test that get_resource returns a copy."""
        db = env_config.get_resource("databases", "default")
        db["backend"] = "mysql"

        db2 = env_config.get_resource("databases", "default")
        assert db2["backend"] == "postgres"

    def test_get_resource_with_defaults(self, env_config):
        """Test getting resource with defaults for missing keys."""
        db = env_config.get_resource(
            "databases", "default",
            defaults={"port": 5432, "host": "ignored"}
        )
        assert db["port"] == 5432
        assert db["host"] == "localhost"  # Existing value not overwritten

    def test_get_resource_not_found_with_defaults(self, env_config):
        """Test getting missing resource with defaults."""
        cache = env_config.get_resource(
            "caches", "redis",
            defaults={"backend": "redis", "host": "localhost"}
        )
        assert cache["backend"] == "redis"
        assert cache["host"] == "localhost"

    def test_get_resource_not_found_raises(self, env_config):
        """Test error when resource not found and no defaults."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            env_config.get_resource("databases", "missing")

    def test_has_resource(self, env_config):
        """Test checking resource existence."""
        assert env_config.has_resource("databases", "default") is True
        assert env_config.has_resource("databases", "missing") is False
        assert env_config.has_resource("caches", "default") is False

    def test_get_setting(self, env_config):
        """Test getting settings."""
        assert env_config.get_setting("log_level") == "DEBUG"
        assert env_config.get_setting("enable_metrics") is True
        assert env_config.get_setting("missing") is None
        assert env_config.get_setting("missing", "default") == "default"

    def test_get_resource_types(self, env_config):
        """Test getting all resource types."""
        types = env_config.get_resource_types()
        assert "databases" in types
        assert "vector_stores" in types

    def test_get_resource_names(self, env_config):
        """Test getting resource names for a type."""
        names = env_config.get_resource_names("databases")
        assert "default" in names
        assert "analytics" in names


class TestEnvironmentConfigMerge:
    """Test environment config merging."""

    def test_merge_resources(self):
        """Test merging resources from two configs."""
        base = EnvironmentConfig(
            name="base",
            resources={
                "databases": {
                    "default": {"backend": "sqlite"}
                }
            }
        )
        override = EnvironmentConfig(
            name="override",
            resources={
                "databases": {
                    "analytics": {"backend": "postgres"}
                }
            }
        )

        merged = base.merge(override)
        assert merged.name == "override"
        assert "default" in merged.resources["databases"]
        assert "analytics" in merged.resources["databases"]

    def test_merge_overwrites_existing(self):
        """Test that merge overwrites existing resource configs."""
        base = EnvironmentConfig(
            name="base",
            resources={
                "databases": {
                    "default": {"backend": "sqlite", "path": "/data/db"}
                }
            }
        )
        override = EnvironmentConfig(
            name="override",
            resources={
                "databases": {
                    "default": {"backend": "postgres"}
                }
            }
        )

        merged = base.merge(override)
        db = merged.resources["databases"]["default"]
        assert db["backend"] == "postgres"
        assert db["path"] == "/data/db"  # Merged from base

    def test_merge_settings(self):
        """Test merging settings."""
        base = EnvironmentConfig(
            name="base",
            settings={"a": 1, "b": 2}
        )
        override = EnvironmentConfig(
            name="override",
            settings={"b": 20, "c": 30}
        )

        merged = base.merge(override)
        assert merged.settings == {"a": 1, "b": 20, "c": 30}


class TestEnvironmentConfigSerialization:
    """Test serialization methods."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        env = EnvironmentConfig(
            name="test",
            description="Test environment",
            resources={"databases": {"default": {"backend": "sqlite"}}},
            settings={"debug": True}
        )

        data = env.to_dict()
        assert data["name"] == "test"
        assert data["description"] == "Test environment"
        assert data["resources"]["databases"]["default"]["backend"] == "sqlite"
        assert data["settings"]["debug"] is True

    def test_to_dict_returns_copy(self):
        """Test that to_dict returns a copy."""
        env = EnvironmentConfig(
            name="test",
            resources={"databases": {"default": {"backend": "sqlite"}}}
        )

        data = env.to_dict()
        data["resources"]["databases"]["default"]["backend"] = "postgres"

        assert env.resources["databases"]["default"]["backend"] == "sqlite"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "name": "production",
            "description": "Prod",
            "resources": {"databases": {"default": {"backend": "postgres"}}},
            "settings": {"log_level": "INFO"}
        }

        env = EnvironmentConfig.from_dict(data)
        assert env.name == "production"
        assert env.description == "Prod"
        assert env.resources["databases"]["default"]["backend"] == "postgres"
        assert env.settings["log_level"] == "INFO"


class TestResourceBinding:
    """Test ResourceBinding dataclass."""

    def test_resource_binding(self):
        """Test ResourceBinding creation."""
        binding = ResourceBinding(
            name="default",
            resource_type="databases",
            config={"backend": "postgres"}
        )
        assert binding.name == "default"
        assert binding.resource_type == "databases"
        assert binding.config["backend"] == "postgres"
