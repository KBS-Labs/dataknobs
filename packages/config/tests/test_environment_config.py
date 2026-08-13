"""Tests for EnvironmentConfig class."""

import os
import threading
from pathlib import Path
from typing import Any

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
        for var in [
            "DATAKNOBS_ENVIRONMENT",
            "AWS_EXECUTION_ENV",
            "KUBERNETES_SERVICE_HOST",
            "K_SERVICE",
            "FUNCTIONS_WORKER_RUNTIME",
            "ECS_CONTAINER_METADATA_URI",
        ]:
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

    def test_load_yaml_config(self, env_dir, monkeypatch):
        """Test loading YAML environment config."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
        config_file = env_dir / "production.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "name": "production",
                    "description": "Production environment",
                    "settings": {"log_level": "INFO"},
                    "resources": {
                        "databases": {
                            "default": {
                                "backend": "postgres",
                                "connection_string": "${DATABASE_URL}",
                            }
                        }
                    },
                }
            )
        )

        env = EnvironmentConfig.load("production", config_dir=env_dir)

        assert env.name == "production"
        assert env.description == "Production environment"
        assert env.settings["log_level"] == "INFO"
        assert "default" in env.resources["databases"]
        assert (
            env.resources["databases"]["default"]["connection_string"]
            == "postgresql://localhost/test"
        )

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

        with pytest.raises(EnvironmentConfigError, match="Expected a dict at the root"):
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
            "databases", "default", defaults={"port": 5432, "host": "ignored"}
        )
        assert db["port"] == 5432
        assert db["host"] == "localhost"  # Existing value not overwritten

    def test_get_resource_not_found_with_defaults(self, env_config):
        """Test getting missing resource with defaults."""
        cache = env_config.get_resource(
            "caches", "redis", defaults={"backend": "redis", "host": "localhost"}
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
            name="base", resources={"databases": {"default": {"backend": "sqlite"}}}
        )
        override = EnvironmentConfig(
            name="override", resources={"databases": {"analytics": {"backend": "postgres"}}}
        )

        merged = base.merge(override)
        assert merged.name == "override"
        assert "default" in merged.resources["databases"]
        assert "analytics" in merged.resources["databases"]

    def test_merge_overwrites_existing(self):
        """Test that merge overwrites existing resource configs."""
        base = EnvironmentConfig(
            name="base",
            resources={"databases": {"default": {"backend": "sqlite", "path": "/data/db"}}},
        )
        override = EnvironmentConfig(
            name="override", resources={"databases": {"default": {"backend": "postgres"}}}
        )

        merged = base.merge(override)
        db = merged.resources["databases"]["default"]
        assert db["backend"] == "postgres"
        assert db["path"] == "/data/db"  # Merged from base

    def test_merge_settings(self):
        """Test merging settings."""
        base = EnvironmentConfig(name="base", settings={"a": 1, "b": 2})
        override = EnvironmentConfig(name="override", settings={"b": 20, "c": 30})

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
            settings={"debug": True},
        )

        data = env.to_dict()
        assert data["name"] == "test"
        assert data["description"] == "Test environment"
        assert data["resources"]["databases"]["default"]["backend"] == "sqlite"
        assert data["settings"]["debug"] is True

    def test_to_dict_returns_copy(self):
        """Test that to_dict returns a copy."""
        env = EnvironmentConfig(
            name="test", resources={"databases": {"default": {"backend": "sqlite"}}}
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
            "settings": {"log_level": "INFO"},
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
            name="default", resource_type="databases", config={"backend": "postgres"}
        )
        assert binding.name == "default"
        assert binding.resource_type == "databases"
        assert binding.config["backend"] == "postgres"


class TestEnvironmentConfigEnvVarSubstitution:
    """Test ${VAR} substitution in EnvironmentConfig.load()/from_dict().

    Mirrors InheritableConfigLoader.load() substitution behaviour, applied
    by default to env-config YAML so resource blocks containing ${VAR}
    refs are resolved at load time rather than surviving into consumer
    code as literal strings.
    """

    def test_load_substitutes_env_vars(self, tmp_path, monkeypatch):
        """Default load() applies ${VAR} substitution to resource values."""
        monkeypatch.setenv("DB_HOST", "rds.example.com")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.delenv("MISSING", raising=False)
        config_path = tmp_path / "test.yaml"
        config_path.write_text(
            "name: test\n"
            "resources:\n"
            "  databases:\n"
            "    primary:\n"
            "      host: ${DB_HOST}\n"
            "      port: ${DB_PORT}\n"
            "      extra: ${MISSING:fallback}\n"
        )
        cfg = EnvironmentConfig.load("test", tmp_path)
        primary = cfg.get_resource("databases", "primary")
        assert primary == {
            "host": "rds.example.com",
            "port": "5432",
            "extra": "fallback",
        }

    def test_load_required_var_missing_raises(self, tmp_path, monkeypatch):
        """Required ${VAR} without a default raises ValueError, like domain configs."""
        monkeypatch.delenv("REQUIRED_VAR", raising=False)
        config_path = tmp_path / "test.yaml"
        config_path.write_text(
            "name: test\nresources:\n  databases:\n    primary:\n      url: ${REQUIRED_VAR}\n"
        )
        with pytest.raises(ValueError, match="REQUIRED_VAR"):
            EnvironmentConfig.load("test", tmp_path)

    def test_load_substitute_vars_false_preserves_literals(self, tmp_path):
        """Opt-out for consumers that want to inspect raw refs."""
        config_path = tmp_path / "test.yaml"
        config_path.write_text(
            "name: test\nresources:\n  databases:\n    primary:\n      url: ${DB_URL}\n"
        )
        cfg = EnvironmentConfig.load("test", tmp_path, substitute_vars=False)
        assert cfg.get_resource("databases", "primary") == {"url": "${DB_URL}"}

    def test_from_dict_substitutes_env_vars(self, monkeypatch):
        """from_dict applies the same substitution path."""
        monkeypatch.setenv("DB_HOST", "rds.example.com")
        cfg = EnvironmentConfig.from_dict(
            {
                "name": "test",
                "resources": {"databases": {"primary": {"host": "${DB_HOST}"}}},
            }
        )
        assert cfg.get_resource("databases", "primary") == {"host": "rds.example.com"}

    def test_from_dict_substitute_vars_false_preserves_literals(self):
        """from_dict opt-out matches load() opt-out."""
        cfg = EnvironmentConfig.from_dict(
            {
                "name": "test",
                "resources": {"databases": {"primary": {"url": "${DB_URL}"}}},
            },
            substitute_vars=False,
        )
        assert cfg.get_resource("databases", "primary") == {"url": "${DB_URL}"}

    def test_binding_resolver_with_env_substituted_config(self, tmp_path, monkeypatch):
        """End-to-end: ConfigBindingResolver sees substituted values from load-time substitution."""
        from dataknobs_config.binding_resolver import (
            ConfigBindingResolver,
            SimpleFactory,
        )

        class _Resource:
            def __init__(self, host: str) -> None:
                self.host = host

        monkeypatch.setenv("TEST_HOST", "rds.example.com")
        config_path = tmp_path / "test.yaml"
        config_path.write_text(
            "name: test\nresources:\n  databases:\n    primary:\n      host: ${TEST_HOST}\n"
        )
        env = EnvironmentConfig.load("test", tmp_path)
        resolver = ConfigBindingResolver(env)
        resolver.register_factory("databases", SimpleFactory(_Resource))

        instance = resolver.resolve("databases", "primary")
        assert instance.host == "rds.example.com"


class TestSubstitutionProvenance:
    """``substituted`` records whether values have already been expanded.

    Downstream resolution layers read this so each source is substituted
    exactly once. Without it they can only guess, and the guess that shipped
    ("run it again, it's idempotent") is wrong for any value whose own text
    contains ``${...}``.
    """

    PAYLOAD = {
        "name": "test",
        "resources": {"databases": {"main": {"password": "${PROV_PW}"}}},
        "settings": {"note": "${PROV_PW}"},
    }

    @pytest.fixture(autouse=True)
    def _prov_env(self, monkeypatch):
        monkeypatch.setenv("PROV_PW", "s3cret")

    def test_from_dict_records_substitution(self):
        assert EnvironmentConfig.from_dict(self.PAYLOAD).substituted is True

    def test_from_dict_records_opt_out(self):
        env = EnvironmentConfig.from_dict(self.PAYLOAD, substitute_vars=False)
        assert env.substituted is False

    def test_load_records_substitution(self, tmp_path):
        (tmp_path / "test.yaml").write_text(yaml.dump(self.PAYLOAD))

        assert EnvironmentConfig.load("test", tmp_path).substituted is True
        assert EnvironmentConfig.load("test", tmp_path, substitute_vars=False).substituted is False

    def test_load_of_absent_file_reports_what_was_asked_of_it(self, tmp_path):
        """An empty config holds no values, so either answer is vacuous.

        What settles it is the sibling path: ``from_dict({})`` has always
        reported ``True``, so reporting ``False`` here made two empty configs
        built from the same request disagree — and made the documented truth
        table ("``load(...)`` default → ``True``") wrong on a path it does
        not carve out.
        """
        assert EnvironmentConfig.load("absent", tmp_path).substituted is True
        assert EnvironmentConfig.from_dict({}).substituted is True

    def test_absent_and_raw_disagree_the_way_mixed_provenance_should(self, tmp_path, monkeypatch):
        """The consequence of the above, stated rather than discovered.

        An absent-file config now reports ``True``, so merging it with a
        directly-constructed (raw) one is a *mixed*-provenance merge, and
        normalization expands the raw side — which can raise, exactly as
        :meth:`merge` documents. Previously both sides read ``False`` and the
        merge left the raw refs alone.
        """
        monkeypatch.delenv("ABSENT_MERGE_VAR", raising=False)
        absent = EnvironmentConfig.load("absent", tmp_path)
        raw = EnvironmentConfig(
            name="raw",
            resources={"databases": {"main": {"dsn": "${ABSENT_MERGE_VAR}"}}},
        )

        with pytest.raises(ValueError, match="ABSENT_MERGE_VAR"):
            absent.merge(raw)

    def test_direct_construction_is_unsubstituted(self):
        """The path that keeps the downstream passes load-bearing."""
        env = EnvironmentConfig(
            name="test",
            resources={"databases": {"main": {"password": "${PROV_PW}"}}},
        )

        assert env.substituted is False

    def test_excluded_from_equality(self):
        """Two configs with the same values are the same environment.

        Including provenance would break the natural assertion
        ``assert loaded == EnvironmentConfig.from_dict(expected)``, which is
        about values and never about which layer expanded them.
        """
        substituted = EnvironmentConfig.from_dict({"name": "test", "settings": {"note": "s3cret"}})
        constructed = EnvironmentConfig(name="test", settings={"note": "s3cret"})

        assert substituted.substituted is not constructed.substituted
        assert substituted == constructed


class TestSubstitutedView:
    """``substituted_view()`` is idempotent and never mutates."""

    @pytest.fixture(autouse=True)
    def _prov_env(self, monkeypatch):
        monkeypatch.setenv("PROV_PW", "s3cret")

    @pytest.fixture
    def raw(self):
        return EnvironmentConfig(
            name="test",
            resources={"databases": {"main": {"password": "${PROV_PW}"}}},
            settings={"note": "${PROV_PW}"},
        )

    def test_returns_self_when_already_substituted(self):
        env = EnvironmentConfig.from_dict({"name": "test", "settings": {"note": "${PROV_PW}"}})

        assert env.substituted_view() is env

    def test_returns_substituted_copy_when_not(self, raw):
        view = raw.substituted_view()

        assert view is not raw
        assert view.substituted is True
        assert view.get_resource("databases", "main")["password"] == "s3cret"
        assert view.get_setting("note") == "s3cret"

    def test_never_mutates_the_receiver(self, raw):
        raw.substituted_view()

        assert raw.substituted is False
        assert raw.get_resource("databases", "main")["password"] == "${PROV_PW}"
        assert raw.get_setting("note") == "${PROV_PW}"

    def test_view_does_not_alias_the_receiver(self, raw):
        view = raw.substituted_view()
        view.resources["databases"]["main"]["password"] = "mutated"

        assert raw.get_resource("databases", "main")["password"] == "${PROV_PW}"

    def test_is_idempotent(self, raw):
        once = raw.substituted_view()

        assert once.substituted_view() is once


class TestMergeProvenance:
    """``merge()`` keeps provenance uniform across the result.

    ``substituted`` describes the whole config, so a merge of two configs
    that disagree must resolve the disagreement rather than pick a side.
    Degrading to ``False`` would leave the already-substituted half exposed
    to a second pass downstream — the very defect the flag prevents.
    """

    @pytest.fixture(autouse=True)
    def _prov_env(self, monkeypatch):
        # A value whose own text contains ${...}: re-expanding it is visible.
        monkeypatch.setenv("MERGE_PW", "p${MERGE_INNER}ss")
        monkeypatch.setenv("MERGE_INNER", "INJECTED")

    @staticmethod
    def _config(name, key, *, substitute):
        return EnvironmentConfig.from_dict(
            {
                "name": name,
                "resources": {"databases": {key: {"password": "${MERGE_PW}"}}},
            },
            substitute_vars=substitute,
        )

    @pytest.mark.parametrize("left", [True, False])
    @pytest.mark.parametrize("right", [True, False])
    def test_values_are_substituted_exactly_once(self, left, right):
        """One expansion in total, wherever it happens to land.

        For a mixed merge that is at merge time; when neither side was
        substituted the merged config is still raw and the single expansion
        is owed downstream. ``substituted_view()`` is what a resolution layer
        applies, and it is a no-op on an already-substituted config — so
        applying it here asks the question uniformly across all four cells:
        after the one pass anyone is entitled to, is the value correct?
        """
        merged = self._config("a", "one", substitute=left).merge(
            self._config("b", "two", substitute=right)
        )
        resolved = merged.substituted_view()

        for key in ("one", "two"):
            password = resolved.get_resource("databases", key)["password"]
            assert password == "p${MERGE_INNER}ss", (
                f"'{key}' (from the {'substituted' if left else 'raw'}/"
                f"{'substituted' if right else 'raw'} merge) was expanded "
                f"the wrong number of times: {password!r}"
            )
            assert "INJECTED" not in password

    @pytest.mark.parametrize("left", [True, False])
    @pytest.mark.parametrize("right", [True, False])
    def test_result_provenance_is_uniform(self, left, right):
        merged = self._config("a", "one", substitute=left).merge(
            self._config("b", "two", substitute=right)
        )

        # Substituted unless *neither* side was: only then is the result
        # still carrying raw refs that a downstream pass must expand.
        assert merged.substituted is (left or right)

    def test_normalizing_merge_does_not_mutate_either_side(self):
        left = self._config("a", "one", substitute=False)
        right = self._config("b", "two", substitute=True)

        left.merge(right)

        assert left.substituted is False
        assert left.get_resource("databases", "one")["password"] == "${MERGE_PW}"
        assert right.substituted is True


class TestGetResourceHandsOutIsolatedConfig:
    """ "Copy to avoid mutation" has to mean the whole config.

    ``get_resource`` copies the resource dict, but shallowly, so every
    nested container in the returned config is the environment's own object.
    A consumer that adjusts a nested section writes through into the
    environment, and the environment outlives the resolution — later reads
    see the adjusted value.

    This was masked wherever a substitution pass happened to run afterwards:
    ``substitute_env_vars`` rebuilds the structure through comprehensions, so
    it incidentally isolated the result. Skipping that pass for an
    already-substituted environment — correct on its own terms — removed the
    accident and left the shallow copy on its own.
    """

    @pytest.fixture
    def env(self):
        return EnvironmentConfig.from_dict(
            {
                "name": "test",
                "resources": {
                    "databases": {
                        "main": {
                            "backend": "postgres",
                            "pool": {"min": 1, "max": 10},
                            "tags": ["primary"],
                        }
                    }
                },
            }
        )

    def test_mutating_a_nested_dict_does_not_reach_the_environment(self, env):
        config = env.get_resource("databases", "main")
        config["pool"]["max"] = 999

        assert env.get_resource("databases", "main")["pool"]["max"] == 10

    def test_mutating_a_nested_list_does_not_reach_the_environment(self, env):
        config = env.get_resource("databases", "main")
        config["tags"].append("clobbered")

        assert env.get_resource("databases", "main")["tags"] == ["primary"]

    def test_defaults_are_isolated_too(self, env):
        """The not-found path hands back the caller's own dict otherwise."""
        defaults = {"pool": {"min": 0}}

        config = env.get_resource("databases", "absent", defaults)
        config["pool"]["min"] = 42

        assert defaults["pool"]["min"] == 0


class TestSubstitutedViewCoversEveryField:
    """The flag describes the config, so the view has to cover the config.

    ``load`` and ``from_dict`` substitute the entire raw document — ``name``
    and ``description`` included — and then set the flag. ``substituted_view``
    sets the same flag having covered only ``resources`` and ``settings``, so
    the two constructors of a ``substituted=True`` config disagree about what
    the flag is claiming. ``merge`` carries ``description`` across, which is
    how an unsubstituted value ends up inside a config marked substituted.
    """

    def test_the_view_substitutes_description(self, monkeypatch):
        monkeypatch.setenv("ENV_BLURB", "production west")
        env = EnvironmentConfig(name="test", description="${ENV_BLURB}", substituted=False)

        assert env.substituted_view().description == "production west"

    def test_the_view_substitutes_name(self, monkeypatch):
        monkeypatch.setenv("ENV_LABEL", "prod-west")
        env = EnvironmentConfig(name="${ENV_LABEL}", substituted=False)

        assert env.substituted_view().name == "prod-west"

    def test_the_view_matches_what_load_would_have_produced(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENV_BLURB", "production west")
        path = tmp_path / "prod.yaml"
        path.write_text(
            yaml.dump(
                {
                    "name": "prod",
                    "description": "${ENV_BLURB}",
                    "resources": {"databases": {"main": {"dsn": "fixed"}}},
                }
            )
        )

        loaded = EnvironmentConfig.load("prod", config_dir=tmp_path)
        raw = EnvironmentConfig.load("prod", config_dir=tmp_path, substitute_vars=False)

        assert raw.substituted_view() == loaded
        assert raw.substituted_view().description == loaded.description


class TestMergeNormalizationCanRaise:
    """Normalizing mixed provenance runs a substitution pass.

    ``merge`` used to be pure data manipulation with no dependency on the
    process environment at all. Normalizing the unsubstituted side means a
    merge can now fail on an unset variable, which is a widened exception
    contract on a public method and belongs in its docstring.
    """

    def test_merging_a_raw_side_with_an_unset_var_raises(self, monkeypatch):
        monkeypatch.delenv("UNSET_FOR_MERGE", raising=False)
        substituted = EnvironmentConfig.from_dict(
            {"name": "base", "resources": {"databases": {"a": {"dsn": "x"}}}}
        )
        raw = EnvironmentConfig.from_dict(
            {
                "name": "overlay",
                "resources": {"databases": {"b": {"dsn": "${UNSET_FOR_MERGE}"}}},
            },
            substitute_vars=False,
        )

        with pytest.raises(ValueError, match="UNSET_FOR_MERGE"):
            substituted.merge(raw)

    def test_the_raise_is_documented(self):
        """Named, not just present: any ``Raises:`` block satisfied the latter."""
        doc = EnvironmentConfig.merge.__doc__ or ""
        raises_section = doc.partition("Raises:")[2]

        assert "ValueError" in raises_section
        assert "substituted" in raises_section


class TestSubstitutedIsProvenanceNotAnAssertion:
    """The flag records how a config was built, not what it currently holds.

    Amending a constructed config is out of contract; these pin what happens
    when you do, so the boundary is explicit rather than folklore. The
    dataclass stays mutable on purpose — it is public, and freezing it would
    break consumers that assemble an environment field by field — which makes
    the contract worth stating in a test.
    """

    def test_amending_a_substituted_config_leaves_the_flag_stale(self):
        env = EnvironmentConfig.from_dict(
            {"name": "test", "resources": {"databases": {"a": {"dsn": "x"}}}}
        )
        env.resources["databases"]["b"] = {"dsn": "${LATE_ADDITION}"}

        # Still True: the flag was not re-derived, and cannot be.
        assert env.substituted is True

    def test_replace_is_the_supported_way_to_re_mark(self, monkeypatch):
        import dataclasses

        monkeypatch.setenv("LATE_ADDITION", "postgres://late")
        env = EnvironmentConfig.from_dict(
            {"name": "test", "resources": {"databases": {"a": {"dsn": "x"}}}}
        )
        env.resources["databases"]["b"] = {"dsn": "${LATE_ADDITION}"}

        remarked = dataclasses.replace(env, substituted=False)

        assert (
            remarked.substituted_view().get_resource("databases", "b")["dsn"] == "postgres://late"
        )


class TestCopyingPreservesLeafIdentity:
    """Isolating the caller means copying the *structure*, not the values.

    ``get_resource`` had to stop handing out aliased nested containers, but a
    deep copy overshoots: it duplicates the leaves too. A resource assembled
    in Python — the supported path, and the one where downstream substitution
    stays load-bearing — can legitimately hold a live object, and duplicating
    a connection pool or raising on a lock is a worse failure than the
    aliasing that motivated the copy.

    The bound is set by what the substitution pass did when it was
    incidentally providing this isolation: it rebuilds dicts and lists and
    returns every other value by identity.
    """

    def test_a_live_object_is_handed_back_not_duplicated(self):
        sentinel = object()
        env = EnvironmentConfig(
            name="test",
            resources={"databases": {"main": {"client": sentinel}}},
        )

        assert env.get_resource("databases", "main")["client"] is sentinel

    def test_an_uncopyable_value_does_not_raise(self):
        """A lock is not deep-copyable; reading a resource must not care."""
        lock = threading.Lock()
        env = EnvironmentConfig(
            name="test",
            resources={"databases": {"main": {"handle": lock}}},
        )

        assert env.get_resource("databases", "main")["handle"] is lock

    def test_defaults_are_copied_the_same_way(self):
        sentinel = object()
        env = EnvironmentConfig(name="test", resources={})

        got = env.get_resource("databases", "absent", {"client": sentinel})

        assert got["client"] is sentinel

    def test_nested_containers_are_still_isolated(self):
        """The structural half of the copy is what the aliasing fix needs."""
        env = EnvironmentConfig(
            name="test",
            resources={"databases": {"main": {"pool": {"max": 5}}}},
        )

        got = env.get_resource("databases", "main")
        got["pool"]["max"] = 999

        assert env.resources["databases"]["main"]["pool"]["max"] == 5


class TestEveryHandOutIsolatesNestedStructure:
    """``get_resource`` was not the only method promising a copy.

    ``merge`` and ``to_dict`` copy one level, so every nested container in
    what they return is still the receiver's own object — the same defect,
    one and two methods away. Fixing only the one it was discovered in is how
    it gets rediscovered from a different caller.
    """

    @pytest.fixture
    def env(self):
        return EnvironmentConfig(
            name="a",
            resources={"databases": {"main": {"pool": {"max": 5}}}},
            settings={"limits": {"rps": 10}},
        )

    def test_merge_does_not_alias_nested_structure(self, env):
        other = EnvironmentConfig(name="b", resources={"caches": {"r": {}}})

        merged = env.merge(other)
        merged.resources["databases"]["main"]["pool"]["max"] = 999

        assert env.resources["databases"]["main"]["pool"]["max"] == 5

    def test_merge_does_not_alias_the_argument_either(self, env):
        other = EnvironmentConfig(name="b", resources={"caches": {"r": {"opts": {"ttl": 1}}}})

        merged = env.merge(other)
        merged.resources["caches"]["r"]["opts"]["ttl"] = 999

        assert other.resources["caches"]["r"]["opts"]["ttl"] == 1

    def test_to_dict_does_not_alias_nested_structure(self, env):
        emitted = env.to_dict()

        emitted["resources"]["databases"]["main"]["pool"]["max"] = 999
        emitted["settings"]["limits"]["rps"] = 999

        assert env.resources["databases"]["main"]["pool"]["max"] == 5
        assert env.settings["limits"]["rps"] == 10

    def test_merge_preserves_leaf_identity(self, env):
        """Same bound as ``get_resource``: structure copied, values are not."""
        sentinel = object()
        other = EnvironmentConfig(name="b", resources={"caches": {"r": {"client": sentinel}}})

        merged = env.merge(other)

        assert merged.resources["caches"]["r"]["client"] is sentinel

    def test_a_self_referential_container_terminates(self):
        """A cyclic structure survives the copy.

        The copy this replaced carried a memo; a config read must not become
        a ``RecursionError`` for having dropped it.
        """
        cyclic: dict[str, Any] = {"host": "db.prod"}
        cyclic["self"] = cyclic
        env = EnvironmentConfig(name="a", resources={"databases": {"main": cyclic}})

        config = env.get_resource("databases", "main")

        assert config["host"] == "db.prod"
        assert config["self"] is config
        assert config is not cyclic

    def test_a_subtree_shared_between_two_keys_stays_shared(self):
        """A memo also preserves sharing, rather than silently forking it."""
        shared = {"timeout": 30}
        env = EnvironmentConfig(
            name="a",
            resources={"databases": {"main": {"read": shared, "write": shared}}},
        )

        config = env.get_resource("databases", "main")

        assert config["read"] is config["write"]
        assert config["read"] is not shared

    def test_sharing_among_defaults_does_not_depend_on_the_branch_taken(self):
        """Both paths out of ``get_resource`` assemble one hand-out.

        The absent path copies the defaults dict whole, so two of its keys
        sharing a subtree keep sharing it. Copying each surviving default
        under its own memo would fork them on the found path only, making an
        observable property of the result depend on whether the resource
        happened to exist.
        """
        shared = {"timeout": 30}
        defaults = {"read": shared, "write": shared}

        env = EnvironmentConfig(name="a", resources={"databases": {"main": {"host": "db"}}})
        found = env.get_resource("databases", "main", defaults=defaults)
        absent = env.get_resource("databases", "typo", defaults=defaults)

        assert found["read"] is found["write"]
        assert absent["read"] is absent["write"]
        assert found["read"] is not shared

    def test_get_resource_does_not_alias_the_defaults_it_falls_back_on(self, env):
        """The found path fills gaps from the caller's own dict.

        Its sibling — the path taken when the resource is absent — copies
        what it returns. Both hand back a config someone will adjust, so
        whether adjusting it reaches back into the caller's dict should not
        depend on which branch ran.
        """
        defaults = {"pool_defaults": {"min": 0}}

        # "main" exists but declares no `pool_defaults`, so the default fills
        # the gap and the found path is the one exercised.
        config = env.get_resource("databases", "main", defaults)
        config["pool_defaults"]["min"] = 42

        assert defaults["pool_defaults"]["min"] == 0


class TestProvenanceIsRecordedOnEveryConstructionPath:
    """A missing environment file is still a config that was built somehow.

    ``load`` short-circuits to an empty config before it reaches the
    construction that records provenance, so it reported ``substituted=False``
    however it was called. Harmless while the config is empty, and a flag that
    lies about how it was built the moment one is merged or amended — the
    exact failure the flag exists to prevent.
    """

    def test_a_missing_file_records_that_substitution_was_requested(self, tmp_path):
        cfg = EnvironmentConfig.load("absent", config_dir=tmp_path)

        assert cfg.substituted is True

    def test_a_missing_file_records_when_it_was_not(self, tmp_path):
        cfg = EnvironmentConfig.load("absent", config_dir=tmp_path, substitute_vars=False)

        assert cfg.substituted is False

    def test_the_flag_survives_a_merge_with_a_raw_config(self, tmp_path):
        """What the lie would cost: the merge skips the pass it needed."""
        absent = EnvironmentConfig.load("absent", config_dir=tmp_path)
        raw = EnvironmentConfig(
            name="raw", resources={"databases": {"main": {"dsn": "${MERGE_DSN}"}}}
        )
        os.environ["MERGE_DSN"] = "postgres://real"
        try:
            merged = absent.merge(raw)
        finally:
            del os.environ["MERGE_DSN"]

        assert merged.substituted is True
        assert merged.resources["databases"]["main"]["dsn"] == "postgres://real"


class TestEnvironmentNamesCannotEscapeTheConfigDir:
    """The environment name can come from the process environment.

    ``detect_environment`` reads ``DATAKNOBS_ENVIRONMENT`` (or
    ``ENVIRONMENT`` under the cloud indicators) and hands the value
    straight to the file lookup, so the name is not always a literal the
    caller wrote.
    """

    @pytest.fixture
    def tree(self, tmp_path):
        """``<root>/environments`` beside a readable ``<root>/outside``."""
        environments = tmp_path / "environments"
        environments.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.yaml").write_text(yaml.dump({"name": "secret", "settings": {"k": 1}}))
        return tmp_path, environments

    def test_a_parent_segment_is_rejected(self, tree):
        _, environments = tree
        with pytest.raises(EnvironmentConfigError, match="outside the configuration directory"):
            EnvironmentConfig.load("../outside/secret", environments)

    def test_a_detected_environment_is_rejected_too(self, tree, monkeypatch):
        """Same guard on the path where the name came from the environment."""
        _, environments = tree
        monkeypatch.setenv("DATAKNOBS_ENVIRONMENT", "../outside/secret")

        with pytest.raises(EnvironmentConfigError, match="outside the configuration directory"):
            EnvironmentConfig.load(None, environments)

    def test_an_absolute_name_is_rejected(self, tree):
        root, environments = tree
        with pytest.raises(EnvironmentConfigError, match="outside the configuration directory"):
            EnvironmentConfig.load(str(root / "outside" / "secret"), environments)

    def test_allow_outside_reaches_a_sibling_tree(self, tree):
        """The opt-out, off by default -- every other test here proves that."""
        _, environments = tree
        config = EnvironmentConfig.load("../outside/secret", environments, allow_outside=True)
        assert config.name == "secret"

    def test_a_missing_environment_is_still_an_empty_config(self, tree):
        """Fail-closed applies to escapes, not to the documented empty case."""
        _, environments = tree
        config = EnvironmentConfig.load("nonexistent", environments)

        assert config.name == "nonexistent"
        assert config.settings == {}

    def test_a_subdirectory_name_still_loads(self, tree):
        _, environments = tree
        (environments / "tier").mkdir()
        (environments / "tier" / "prod.yaml").write_text(
            yaml.dump({"name": "prod", "settings": {"k": 1}})
        )

        config = EnvironmentConfig.load("tier/prod", environments)

        assert config.settings == {"k": 1}
