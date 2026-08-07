"""Tests for EnvironmentAwareConfig class."""

import logging
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


class TestMissingResourceIsObservable:
    """A ``$resource`` name that matches nothing must say so.

    ``_resolve_resource_refs`` degrades to the reference's inline defaults
    when a resource is missing, which is deliberate. What was not deliberate
    is that the degrade was **silent**: the warning lived in an
    ``except KeyError`` branch that could never run, because
    ``_resolve_resource_refs`` always passes a (possibly empty) defaults dict
    and ``get_resource`` returns those defaults rather than raising whenever
    one is supplied. A typo'd binding name therefore produced an empty config
    and no log line anywhere.
    """

    @pytest.fixture
    def env(self):
        return EnvironmentConfig.from_dict(
            {
                "name": "test",
                "resources": {"databases": {"main": {"backend": "postgres"}}},
            }
        )

    def test_missing_resource_warns_with_no_inline_defaults(self, env, caplog):
        app = EnvironmentAwareConfig(
            config={"db": {"$resource": "typo", "type": "databases"}},
            environment=env,
        )

        with caplog.at_level(logging.WARNING):
            resolved = app.resolve_for_build()

        assert resolved["db"] == {}
        assert "typo" in caplog.text
        assert "not found" in caplog.text

    def test_missing_resource_warns_with_inline_defaults(self, env, caplog):
        app = EnvironmentAwareConfig(
            config={
                "db": {"$resource": "typo", "type": "databases", "timeout": 5}
            },
            environment=env,
        )

        with caplog.at_level(logging.WARNING):
            resolved = app.resolve_for_build()

        assert resolved["db"] == {"timeout": 5}
        assert "typo" in caplog.text

    def test_found_resource_does_not_warn(self, env, caplog):
        app = EnvironmentAwareConfig(
            config={"db": {"$resource": "main", "type": "databases"}},
            environment=env,
        )

        with caplog.at_level(logging.WARNING):
            resolved = app.resolve_for_build()

        assert resolved["db"] == {"backend": "postgres"}
        assert "not found" not in caplog.text


class TestResolveForBuildSubstitutionOrder:
    """``resolve_for_build`` substitutes each source exactly once.

    The app config is loaded *without* substitution (late binding), so its
    ``${VAR}`` refs must still be expanded here. Environment values were
    expanded at load. Both are true simultaneously only if the app config is
    substituted **before** resource refs are spliced in — afterwards the two
    are merged beyond distinguishing.
    """

    @pytest.fixture(autouse=True)
    def _env_vars(self, monkeypatch):
        monkeypatch.setenv("ORDER_PW", "p${ORDER_INNER}ss")
        monkeypatch.setenv("ORDER_INNER", "INJECTED")
        monkeypatch.setenv("ORDER_TEMP", "0.9")

    @pytest.fixture
    def env(self):
        return EnvironmentConfig.from_dict(
            {
                "name": "test",
                "resources": {
                    "databases": {"main": {"password": "${ORDER_PW}"}}
                },
            }
        )

    @pytest.fixture
    def app(self, env):
        return EnvironmentAwareConfig(
            config={
                "db": {"$resource": "main", "type": "databases"},
                "temperature": "${ORDER_TEMP}",
            },
            environment=env,
        )

    def test_late_binding_still_works(self, app):
        """The headline feature: app-authored refs resolve at build time."""
        assert app.resolve_for_build()["temperature"] == "0.9"

    def test_environment_value_is_not_re_expanded(self, app):
        assert app.resolve_for_build()["db"]["password"] == "p${ORDER_INNER}ss"

    def test_both_at_once(self, app):
        """Neither provenance is served at the other's expense."""
        resolved = app.resolve_for_build()

        assert resolved["temperature"] == "0.9"
        assert resolved["db"]["password"] == "p${ORDER_INNER}ss"

    def test_resolve_env_vars_false_expands_nothing_at_this_layer(self, app):
        resolved = app.resolve_for_build(resolve_env_vars=False)

        assert resolved["temperature"] == "${ORDER_TEMP}"
        # The environment was substituted at load, so its values arrive
        # expanded once regardless -- this flag governs *this* layer only.
        assert resolved["db"]["password"] == "p${ORDER_INNER}ss"

    def test_resolve_resources_false_is_unchanged_by_the_reorder(self, app):
        """With no splice, ordering is unobservable."""
        resolved = app.resolve_for_build(resolve_resources=False)

        assert resolved["temperature"] == "0.9"
        assert resolved["db"] == {"$resource": "main", "type": "databases"}

    def test_nested_refs_are_expanded_exactly_once_at_any_depth(self):
        """A resource reached through another resource is still one source.

        Each is expanded as it is spliced, so the inner one is expanded by
        its own splice rather than by a pass over the outer's already-spliced
        result — which is what a value reached at depth being expanded twice
        would look like.
        """
        env = EnvironmentConfig(
            name="test",
            resources={
                "databases": {
                    "outer": {"inner": {"$resource": "leaf", "type": "secrets"}}
                },
                "secrets": {"leaf": {"password": "${ORDER_PW}"}},
            },
        )
        app = EnvironmentAwareConfig(
            config={"db": {"$resource": "outer", "type": "databases"}},
            environment=env,
        )

        resolved = app.resolve_for_build()

        assert resolved["db"]["inner"]["password"] == "p${ORDER_INNER}ss"


class TestResourceNameFromEnvVar:
    """``$resource: ${VAR}`` resolves (behaviour change, D-199-2).

    Substituting the app config before the splice means the ``$resource``
    and ``type`` values are themselves expanded, so binding resource
    *selection* to an environment variable now works. Previously the literal
    ``${VAR}`` text was looked up, matched nothing, and degraded to the
    reference's inline defaults.
    """

    @pytest.fixture(autouse=True)
    def _env_vars(self, monkeypatch):
        monkeypatch.setenv("BINDING_NAME", "primary")
        monkeypatch.setenv("BINDING_TYPE", "databases")

    @pytest.fixture
    def env(self):
        return EnvironmentConfig.from_dict(
            {
                "name": "test",
                "resources": {
                    "databases": {"primary": {"backend": "postgres"}}
                },
            }
        )

    def test_resource_name_from_env_var(self, env):
        app = EnvironmentAwareConfig(
            config={"db": {"$resource": "${BINDING_NAME}", "type": "databases"}},
            environment=env,
        )

        assert app.resolve_for_build()["db"]["backend"] == "postgres"

    def test_resource_type_from_env_var(self, env):
        app = EnvironmentAwareConfig(
            config={"db": {"$resource": "primary", "type": "${BINDING_TYPE}"}},
            environment=env,
        )

        assert app.resolve_for_build()["db"]["backend"] == "postgres"

    def test_still_degrades_when_the_expansion_names_nothing(self, env, caplog):
        """Expanding first does not remove the not-found path, only moves it.

        The name is now resolved before lookup, so the warning names the
        expanded value -- which is the one an operator needs to see.
        """
        app = EnvironmentAwareConfig(
            config={"db": {"$resource": "${BINDING_TYPE}", "type": "databases"}},
            environment=env,
        )

        with caplog.at_level(logging.WARNING):
            resolved = app.resolve_for_build()

        assert resolved["db"] == {}
        assert "databases" in caplog.text
        assert "${BINDING_TYPE}" not in caplog.text

    def test_unset_var_in_resource_name_is_unchanged(self, env, monkeypatch):
        """An unset ref without a default still raises, as it does anywhere."""
        monkeypatch.delenv("MISSING_BINDING", raising=False)
        app = EnvironmentAwareConfig(
            config={
                "db": {"$resource": "${MISSING_BINDING}", "type": "databases"}
            },
            environment=env,
        )

        with pytest.raises(ValueError, match="MISSING_BINDING"):
            app.resolve_for_build()


class TestMissingResourceStillResolvesItsDefaults:
    """Degrading to inline defaults must not stop treating them as config.

    The missing-resource branch used to reach the shared tail — ``$requires``
    validation and the recursive walk — because ``get_resource`` returned the
    supplied defaults instead of raising, so ``resolved`` was simply the
    defaults and execution continued. Testing membership explicitly made the
    branch reachable but also made it return early, dropping both.

    The recursive walk is what resolves a nested ``$resource`` *inside* the
    defaults and what rebuilds the structure so the returned config does not
    alias the environment. Returning the reference's own marker keys to a
    factory is the failure the branch's comment says it was avoiding; without
    the walk it happens one level down instead of at the top.
    """

    @pytest.fixture
    def env(self):
        return EnvironmentConfig.from_dict(
            {
                "name": "test",
                "resources": {
                    "databases": {
                        "fallback": {"backend": "sqlite", "path": ":memory:"}
                    }
                },
            }
        )

    def test_nested_reference_in_defaults_is_resolved(self, env):
        """A ``$resource`` inside the inline defaults must still resolve."""
        app = EnvironmentAwareConfig(
            config={
                "db": {
                    "$resource": "typo",
                    "type": "databases",
                    "spare": {"$resource": "fallback", "type": "databases"},
                }
            },
            environment=env,
        )

        resolved = app.resolve_for_build()

        assert resolved["db"]["spare"] == {
            "backend": "sqlite",
            "path": ":memory:",
        }

    def test_nested_marker_keys_never_reach_the_result(self, env):
        """The markers are the reference's syntax, never a factory kwarg."""
        app = EnvironmentAwareConfig(
            config={
                "db": {
                    "$resource": "typo",
                    "type": "databases",
                    "spare": {"$resource": "fallback", "type": "databases"},
                }
            },
            environment=env,
        )

        resolved = app.resolve_for_build()

        assert "$resource" not in resolved["db"]["spare"]
        assert "type" not in resolved["db"]["spare"]

    def test_requires_is_validated_against_inline_defaults(self, env):
        """``$requires`` was checked on the degraded config before, too."""
        from dataknobs_config.exceptions import ConfigError

        app = EnvironmentAwareConfig(
            config={
                "db": {
                    "$resource": "typo",
                    "type": "databases",
                    "$requires": ["transactions"],
                    "capabilities": ["reads"],
                }
            },
            environment=env,
        )

        with pytest.raises(ConfigError, match="transactions"):
            app.resolve_for_build()

    def test_degraded_result_does_not_alias_the_environment(self, env):
        """The walk rebuilds the structure it returns.

        ``resolve_for_build`` deep-copies the app config on the way in, so
        the defaults themselves are already safe; what the walk protects is
        anything spliced in from the environment during the recursion.
        """
        app = EnvironmentAwareConfig(
            config={
                "db": {
                    "$resource": "typo",
                    "type": "databases",
                    "spare": {"$resource": "fallback", "type": "databases"},
                }
            },
            environment=env,
        )

        resolved = app.resolve_for_build()
        resolved["db"]["spare"]["path"] = "/tmp/clobbered"

        assert env.resources["databases"]["fallback"]["path"] == ":memory:"


class TestOnlyReferencedResourcesAreExpanded:
    """An environment holds resources this app never asked for.

    Substituting the environment as a whole before the splice reads every
    value in it, so an unset required ``${VAR}`` in a resource no reference
    names aborts a build that would never have looked at it. The pre-change
    pass ran *after* the splice and so covered exactly the values spliced in.

    The invariant is "each source exactly once", not "every source eagerly";
    a resource is still separable at the point it is spliced, which is the
    latest point it can be substituted.
    """

    @pytest.fixture
    def env(self):
        # Loaded unsubstituted: the deliberate late-binding path, and the
        # only one where the downstream pass is load-bearing.
        return EnvironmentConfig.from_dict(
            {
                "name": "test",
                "resources": {
                    "databases": {"main": {"dsn": "${WANTED_DSN}"}},
                    "warehouses": {"analytics": {"dsn": "${NEVER_REFERENCED}"}},
                },
                "settings": {"unused": "${ALSO_NEVER_REFERENCED}"},
            },
            substitute_vars=False,
        )

    def test_an_unreferenced_resource_does_not_abort_the_build(
        self, env, monkeypatch
    ):
        monkeypatch.setenv("WANTED_DSN", "postgres://real")
        monkeypatch.delenv("NEVER_REFERENCED", raising=False)
        monkeypatch.delenv("ALSO_NEVER_REFERENCED", raising=False)

        app = EnvironmentAwareConfig(
            config={"db": {"$resource": "main", "type": "databases"}},
            environment=env,
        )

        resolved = app.resolve_for_build()

        assert resolved["db"]["dsn"] == "postgres://real"

    def test_the_referenced_resource_is_still_expanded_exactly_once(
        self, env, monkeypatch
    ):
        """The value's own ``${...}`` text stays literal — the whole point."""
        monkeypatch.setenv("WANTED_DSN", "p${x}ss")
        monkeypatch.setenv("x", "INJECTED")
        monkeypatch.delenv("NEVER_REFERENCED", raising=False)
        monkeypatch.delenv("ALSO_NEVER_REFERENCED", raising=False)

        app = EnvironmentAwareConfig(
            config={"db": {"$resource": "main", "type": "databases"}},
            environment=env,
        )

        assert app.resolve_for_build()["db"]["dsn"] == "p${x}ss"

    def test_the_environment_itself_is_never_mutated(self, env, monkeypatch):
        monkeypatch.setenv("WANTED_DSN", "postgres://real")
        monkeypatch.delenv("NEVER_REFERENCED", raising=False)
        monkeypatch.delenv("ALSO_NEVER_REFERENCED", raising=False)

        app = EnvironmentAwareConfig(
            config={"db": {"$resource": "main", "type": "databases"}},
            environment=env,
        )
        app.resolve_for_build()

        assert env.resources["databases"]["main"]["dsn"] == "${WANTED_DSN}"
        assert env.substituted is False


class TestInlineDefaultsAreExpandedOnlyWhenTheySurvive:
    """The app-config mirror of :class:`TestOnlyReferencedResourcesAreExpanded`.

    A ``$resource`` reference's inline defaults are app config, and the splice
    discards every one the environment supplies. Substituting the whole app
    config on entry reads them all, so a dev-time fallback that production
    overrides still has to resolve in production — the build aborts on a value
    it was about to throw away.

    Inline defaults stay separable until ``setdefault`` decides which survive,
    so that is the latest point they can be expanded, and therefore where they
    are. Same rule as the environment side, applied to the other source.
    """

    @pytest.fixture
    def env(self):
        return EnvironmentConfig.from_dict(
            {
                "name": "prod",
                "resources": {
                    "databases": {
                        "main": {"host": "db.prod", "password": "realsecret"}
                    }
                },
            }
        )

    def test_an_overridden_default_does_not_have_to_resolve(
        self, env, monkeypatch
    ):
        monkeypatch.delenv("LOCAL_DB_PASSWORD", raising=False)

        app = EnvironmentAwareConfig(
            config={
                "db": {
                    "$resource": "main",
                    "type": "databases",
                    "password": "${LOCAL_DB_PASSWORD}",
                }
            },
            environment=env,
        )

        assert app.resolve_for_build()["db"]["password"] == "realsecret"

    def test_a_surviving_default_is_still_expanded(self, env, monkeypatch):
        """Deferring the pass must not become skipping it."""
        monkeypatch.setenv("LOCAL_POOL_SIZE", "7")

        app = EnvironmentAwareConfig(
            config={
                "db": {
                    "$resource": "main",
                    "type": "databases",
                    "pool_size": "${LOCAL_POOL_SIZE}",
                }
            },
            environment=env,
        )

        assert app.resolve_for_build()["db"]["pool_size"] == "7"

    def test_a_surviving_default_is_expanded_exactly_once(
        self, env, monkeypatch
    ):
        monkeypatch.setenv("LOCAL_POOL_SIZE", "p${x}ss")
        monkeypatch.setenv("x", "INJECTED")

        app = EnvironmentAwareConfig(
            config={
                "db": {
                    "$resource": "main",
                    "type": "databases",
                    "pool_size": "${LOCAL_POOL_SIZE}",
                }
            },
            environment=env,
        )

        assert app.resolve_for_build()["db"]["pool_size"] == "p${x}ss"

    def test_an_unset_var_in_a_surviving_default_still_raises(
        self, env, monkeypatch
    ):
        """The deferral moves the pass; it does not suppress its errors."""
        monkeypatch.delenv("LOCAL_POOL_SIZE", raising=False)

        app = EnvironmentAwareConfig(
            config={
                "db": {
                    "$resource": "main",
                    "type": "databases",
                    "pool_size": "${LOCAL_POOL_SIZE}",
                }
            },
            environment=env,
        )

        with pytest.raises(ValueError, match="LOCAL_POOL_SIZE"):
            app.resolve_for_build()

    def test_a_degraded_reference_expands_the_defaults_it_falls_back_to(
        self, env, monkeypatch
    ):
        """Nothing overrides them, so every one survives — and must resolve."""
        monkeypatch.setenv("LOCAL_DB_PASSWORD", "devsecret")

        app = EnvironmentAwareConfig(
            config={
                "db": {
                    "$resource": "typo",
                    "type": "databases",
                    "password": "${LOCAL_DB_PASSWORD}",
                }
            },
            environment=env,
        )

        assert app.resolve_for_build()["db"]["password"] == "devsecret"

    def test_ordinary_app_values_are_untouched_by_the_deferral(
        self, env, monkeypatch
    ):
        """Only inline defaults defer; the rest of the app config does not."""
        monkeypatch.setenv("APP_NAME", "billing")

        app = EnvironmentAwareConfig(
            config={
                "name": "${APP_NAME}",
                "db": {"$resource": "main", "type": "databases"},
            },
            environment=env,
        )

        assert app.resolve_for_build()["name"] == "billing"
