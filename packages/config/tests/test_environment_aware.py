"""Tests for EnvironmentAwareConfig class."""

import os
from pathlib import Path

import pytest
import yaml

from dataknobs_config.environment_aware import (
    EnvironmentAwareConfig,
    EnvironmentAwareConfigError,
)
from dataknobs_config.environment_config import EnvironmentConfig


class TestEnvironmentAwareConfigBasics:
    """Test basic EnvironmentAwareConfig functionality."""

    @pytest.fixture
    def sample_config(self):
        """Sample application configuration."""
        return {
            "name": "test-app",
            "version": "1.0.0",
            "bot": {
                "llm": {
                    "$resource": "default",
                    "type": "llm_providers",
                    "temperature": 0.7,
                },
                "database": {
                    "$resource": "conversations",
                    "type": "databases",
                },
            },
            "settings": {
                "debug": True,
            },
        }

    @pytest.fixture
    def sample_env(self):
        """Sample environment configuration."""
        return EnvironmentConfig(
            name="development",
            resources={
                "llm_providers": {
                    "default": {
                        "provider": "openai",
                        "model": "gpt-4",
                        "api_key": "${OPENAI_API_KEY}",
                    },
                },
                "databases": {
                    "conversations": {
                        "backend": "sqlite",
                        "path": "~/data/conversations.db",
                    },
                },
            },
            settings={"log_level": "DEBUG"},
        )

    def test_init_with_config(self, sample_config, sample_env):
        """Test initialization with config and environment."""
        config = EnvironmentAwareConfig(
            config=sample_config,
            environment=sample_env,
        )

        assert config.app_name == "test-app"
        assert config.environment_name == "development"

    def test_init_auto_detect_env(self, sample_config, monkeypatch):
        """Test initialization with auto-detected environment."""
        monkeypatch.setenv("DATAKNOBS_ENVIRONMENT", "production")

        config = EnvironmentAwareConfig(config=sample_config)
        assert config.environment_name == "production"

    def test_get_simple_value(self, sample_config, sample_env):
        """Test getting simple config values."""
        config = EnvironmentAwareConfig(config=sample_config, environment=sample_env)

        assert config.get("name") == "test-app"
        assert config.get("version") == "1.0.0"

    def test_get_nested_value(self, sample_config, sample_env):
        """Test getting nested config values with dot notation."""
        config = EnvironmentAwareConfig(config=sample_config, environment=sample_env)

        assert config.get("bot.llm.temperature") == 0.7
        assert config.get("settings.debug") is True

    def test_get_with_default(self, sample_config, sample_env):
        """Test getting missing values with default."""
        config = EnvironmentAwareConfig(config=sample_config, environment=sample_env)

        assert config.get("missing") is None
        assert config.get("missing", "default") == "default"
        assert config.get("bot.missing", "default") == "default"

    def test_get_returns_copy(self, sample_config, sample_env):
        """Test that get returns a copy for dicts."""
        config = EnvironmentAwareConfig(config=sample_config, environment=sample_env)

        bot = config.get("bot")
        bot["llm"]["temperature"] = 0.9

        assert config.get("bot.llm.temperature") == 0.7


class TestResourceResolution:
    """Test resource reference resolution."""

    @pytest.fixture
    def config_with_resources(self):
        """Config with resource references."""
        return {
            "name": "test-app",
            "database": {
                "$resource": "primary",
                "type": "databases",
                "extra_param": "value",
            },
            "vector_store": {
                "$resource": "knowledge",
                "type": "vector_stores",
            },
        }

    @pytest.fixture
    def env_with_resources(self):
        """Environment with resource bindings."""
        return EnvironmentConfig(
            name="test",
            resources={
                "databases": {
                    "primary": {
                        "backend": "postgres",
                        "host": "localhost",
                        "port": 5432,
                    },
                },
                "vector_stores": {
                    "knowledge": {
                        "backend": "pgvector",
                        "dimensions": 1536,
                    },
                },
            },
        )

    def test_resolve_resource_reference(self, config_with_resources, env_with_resources):
        """Test resolving $resource references."""
        config = EnvironmentAwareConfig(
            config=config_with_resources,
            environment=env_with_resources,
        )

        resolved = config.resolve_for_build(resolve_env_vars=False)

        # Resource should be resolved
        assert resolved["database"]["backend"] == "postgres"
        assert resolved["database"]["host"] == "localhost"
        # Extra params should be merged
        assert resolved["database"]["extra_param"] == "value"

    def test_resolve_preserves_defaults(self, env_with_resources):
        """Test that resource defaults are preserved."""
        config_dict = {
            "database": {
                "$resource": "primary",
                "type": "databases",
                "timeout": 30,  # Default not in environment
            },
        }
        config = EnvironmentAwareConfig(
            config=config_dict,
            environment=env_with_resources,
        )

        resolved = config.resolve_for_build(resolve_env_vars=False)
        assert resolved["database"]["timeout"] == 30
        assert resolved["database"]["backend"] == "postgres"

    def test_resolve_nested_resources(self, env_with_resources):
        """Test resolving nested resource references."""
        config_dict = {
            "bot": {
                "storage": {
                    "$resource": "primary",
                    "type": "databases",
                },
            },
        }
        config = EnvironmentAwareConfig(
            config=config_dict,
            environment=env_with_resources,
        )

        resolved = config.resolve_for_build(resolve_env_vars=False)
        assert resolved["bot"]["storage"]["backend"] == "postgres"

    def test_resolve_missing_resource_uses_defaults(self, env_with_resources):
        """Test that missing resource falls back to defaults."""
        config_dict = {
            "database": {
                "$resource": "missing",
                "type": "databases",
                "backend": "default_backend",
            },
        }
        config = EnvironmentAwareConfig(
            config=config_dict,
            environment=env_with_resources,
        )

        resolved = config.resolve_for_build(resolve_env_vars=False)
        assert resolved["database"]["backend"] == "default_backend"

    def test_resolve_without_resource_resolution(self, config_with_resources, env_with_resources):
        """Test skipping resource resolution."""
        config = EnvironmentAwareConfig(
            config=config_with_resources,
            environment=env_with_resources,
        )

        resolved = config.resolve_for_build(resolve_resources=False, resolve_env_vars=False)
        assert "$resource" in resolved["database"]

    def test_resolve_specific_key(self, config_with_resources, env_with_resources):
        """Test resolving a specific config key."""
        config = EnvironmentAwareConfig(
            config=config_with_resources,
            environment=env_with_resources,
        )

        resolved = config.resolve_for_build("database", resolve_env_vars=False)
        assert resolved["backend"] == "postgres"

    def test_resolve_missing_key_raises(self, config_with_resources, env_with_resources):
        """Test error when resolving missing key."""
        config = EnvironmentAwareConfig(
            config=config_with_resources,
            environment=env_with_resources,
        )

        with pytest.raises(EnvironmentAwareConfigError, match="not found"):
            config.resolve_for_build("nonexistent")


class TestEnvVarResolution:
    """Test environment variable resolution."""

    @pytest.fixture
    def config_with_env_vars(self):
        """Config with env var placeholders."""
        return {
            "api_key": "${API_KEY}",
            "database_url": "${DATABASE_URL:sqlite:///default.db}",
        }

    def test_resolve_env_vars(self, config_with_env_vars, monkeypatch):
        """Test resolving environment variables."""
        monkeypatch.setenv("API_KEY", "secret123")
        monkeypatch.delenv("DATABASE_URL", raising=False)

        config = EnvironmentAwareConfig(config=config_with_env_vars)
        resolved = config.resolve_for_build()

        assert resolved["api_key"] == "secret123"
        # Path normalization occurs during substitution
        assert "default.db" in resolved["database_url"]

    def test_env_vars_not_resolved_in_portable(self, config_with_env_vars, monkeypatch):
        """Test that portable config keeps env var placeholders."""
        monkeypatch.setenv("API_KEY", "secret123")

        config = EnvironmentAwareConfig(config=config_with_env_vars)
        portable = config.get_portable_config()

        assert portable["api_key"] == "${API_KEY}"

    def test_skip_env_var_resolution(self, config_with_env_vars):
        """Test skipping env var resolution."""
        config = EnvironmentAwareConfig(config=config_with_env_vars)
        resolved = config.resolve_for_build(resolve_env_vars=False)

        assert resolved["api_key"] == "${API_KEY}"


class TestLoadApp:
    """Test loading application configurations."""

    @pytest.fixture
    def config_dirs(self, tmp_path):
        """Create temporary app and environment config directories."""
        app_dir = tmp_path / "apps"
        env_dir = tmp_path / "environments"
        app_dir.mkdir()
        env_dir.mkdir()
        return app_dir, env_dir

    def test_load_app_yaml(self, config_dirs, monkeypatch):
        """Test loading YAML app config."""
        app_dir, env_dir = config_dirs
        monkeypatch.setenv("DATAKNOBS_ENVIRONMENT", "development")

        # Create app config
        (app_dir / "my-app.yaml").write_text(yaml.dump({
            "name": "my-app",
            "bot": {"setting": "value"},
        }))

        # Create environment config
        (env_dir / "development.yaml").write_text(yaml.dump({
            "name": "development",
            "settings": {"debug": True},
        }))

        config = EnvironmentAwareConfig.load_app(
            "my-app",
            app_dir=app_dir,
            env_dir=env_dir,
        )

        assert config.app_name == "my-app"
        assert config.environment_name == "development"
        assert config.get("bot.setting") == "value"

    def test_load_app_explicit_environment(self, config_dirs):
        """Test loading with explicit environment."""
        app_dir, env_dir = config_dirs

        (app_dir / "my-app.yaml").write_text("name: my-app")
        (env_dir / "production.yaml").write_text("name: production")

        config = EnvironmentAwareConfig.load_app(
            "my-app",
            app_dir=app_dir,
            env_dir=env_dir,
            environment="production",
        )

        assert config.environment_name == "production"

    def test_load_app_not_found(self, config_dirs):
        """Test error when app config not found."""
        app_dir, env_dir = config_dirs

        with pytest.raises(EnvironmentAwareConfigError, match="not found"):
            EnvironmentAwareConfig.load_app(
                "missing-app",
                app_dir=app_dir,
                env_dir=env_dir,
            )

    def test_load_app_missing_env_config_ok(self, config_dirs):
        """Test loading works when environment config is missing."""
        app_dir, env_dir = config_dirs

        (app_dir / "my-app.yaml").write_text("name: my-app")
        # No environment config created

        config = EnvironmentAwareConfig.load_app(
            "my-app",
            app_dir=app_dir,
            env_dir=env_dir,
            environment="production",
        )

        assert config.app_name == "my-app"
        assert config.environment_name == "production"

    def test_load_app_json(self, config_dirs, monkeypatch):
        """Test loading JSON app config."""
        app_dir, env_dir = config_dirs
        monkeypatch.setenv("DATAKNOBS_ENVIRONMENT", "development")

        (app_dir / "my-app.json").write_text('{"name": "my-app"}')
        (env_dir / "development.yaml").write_text("name: development")

        config = EnvironmentAwareConfig.load_app(
            "my-app",
            app_dir=app_dir,
            env_dir=env_dir,
        )

        assert config.app_name == "my-app"


class TestWithEnvironment:
    """Test switching environments."""

    @pytest.fixture
    def base_config(self):
        """Base configuration."""
        return {
            "name": "app",
            "database": {
                "$resource": "default",
                "type": "databases",
            },
        }

    def test_with_environment_string(self, base_config, tmp_path):
        """Test creating new config with different environment name."""
        env_dir = tmp_path / "environments"
        env_dir.mkdir()
        (env_dir / "production.yaml").write_text(yaml.dump({
            "name": "production",
            "resources": {
                "databases": {
                    "default": {"backend": "postgres"},
                },
            },
        }))

        original = EnvironmentAwareConfig(
            config=base_config,
            environment=EnvironmentConfig(name="development"),
        )

        new_config = original.with_environment("production", env_dir=env_dir)

        assert original.environment_name == "development"
        assert new_config.environment_name == "production"

    def test_with_environment_object(self, base_config):
        """Test creating new config with EnvironmentConfig object."""
        dev_env = EnvironmentConfig(name="development")
        prod_env = EnvironmentConfig(
            name="production",
            resources={
                "databases": {
                    "default": {"backend": "postgres"},
                },
            },
        )

        original = EnvironmentAwareConfig(config=base_config, environment=dev_env)
        new_config = original.with_environment(prod_env)

        assert new_config.environment_name == "production"
        resolved = new_config.resolve_for_build(resolve_env_vars=False)
        assert resolved["database"]["backend"] == "postgres"


class TestConvenienceMethods:
    """Test convenience methods."""

    @pytest.fixture
    def full_config(self):
        """Full configuration with environment."""
        env = EnvironmentConfig(
            name="test",
            resources={
                "databases": {
                    "default": {"backend": "sqlite"},
                },
            },
            settings={"log_level": "DEBUG"},
        )
        return EnvironmentAwareConfig(
            config={"name": "app"},
            environment=env,
        )

    def test_get_resource(self, full_config):
        """Test direct resource access."""
        db = full_config.get_resource("databases", "default")
        assert db["backend"] == "sqlite"

    def test_get_setting(self, full_config):
        """Test direct setting access."""
        assert full_config.get_setting("log_level") == "DEBUG"
        assert full_config.get_setting("missing", "default") == "default"

    def test_to_dict(self, full_config):
        """Test to_dict returns portable config."""
        data = full_config.to_dict()
        assert data == {"name": "app"}

    def test_repr(self, full_config):
        """Test string representation."""
        repr_str = repr(full_config)
        assert "app" in repr_str
        assert "test" in repr_str


class TestFromDict:
    """Test from_dict class method."""

    def test_from_dict_basic(self, monkeypatch):
        """Test creating from dict."""
        monkeypatch.setenv("DATAKNOBS_ENVIRONMENT", "development")

        config = EnvironmentAwareConfig.from_dict({"name": "app"})
        assert config.app_name == "app"
        assert config.environment_name == "development"

    def test_from_dict_with_environment(self, tmp_path, monkeypatch):
        """Test from_dict with explicit environment."""
        env_dir = tmp_path / "environments"
        env_dir.mkdir()
        (env_dir / "staging.yaml").write_text("name: staging")

        config = EnvironmentAwareConfig.from_dict(
            {"name": "app"},
            environment="staging",
            env_dir=env_dir,
        )
        assert config.environment_name == "staging"


class TestResourcesInLists:
    """Test resource references in list structures."""

    def test_resources_in_list(self):
        """Test resolving resources inside lists."""
        config_dict = {
            "connectors": [
                {
                    "name": "primary",
                    "database": {
                        "$resource": "main",
                        "type": "databases",
                    },
                },
                {
                    "name": "backup",
                    "database": {
                        "$resource": "backup",
                        "type": "databases",
                    },
                },
            ],
        }
        env = EnvironmentConfig(
            name="test",
            resources={
                "databases": {
                    "main": {"backend": "postgres", "host": "main.db"},
                    "backup": {"backend": "postgres", "host": "backup.db"},
                },
            },
        )

        config = EnvironmentAwareConfig(config=config_dict, environment=env)
        resolved = config.resolve_for_build(resolve_env_vars=False)

        assert resolved["connectors"][0]["database"]["host"] == "main.db"
        assert resolved["connectors"][1]["database"]["host"] == "backup.db"
