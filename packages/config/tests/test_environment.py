"""Tests for environment variable overrides."""

import logging

import pytest

from dataknobs_config import Config
from dataknobs_config.environment import EnvironmentOverrides


class TestEnvironmentVariableNaming:
    """Test environment variable naming conventions."""

    def test_reference_to_env_var(self):
        """Test converting references to environment variable names."""
        env = EnvironmentOverrides()

        # Named reference
        env_var = env.reference_to_env_var("xref:database[primary]", "host")
        assert env_var == "DATAKNOBS_DATABASE__PRIMARY__HOST"

        # Index reference
        env_var = env.reference_to_env_var("xref:cache[0]", "port")
        assert env_var == "DATAKNOBS_CACHE__0__PORT"

        # Negative index
        env_var = env.reference_to_env_var("xref:server[-1]", "workers")
        assert env_var == "DATAKNOBS_SERVER__-1__WORKERS"

    def test_env_var_to_reference(self):
        """Test converting environment variables to references."""
        env = EnvironmentOverrides()

        ref = env._env_var_to_reference("DATAKNOBS_DATABASE__PRIMARY__HOST")
        assert ref == "xref:database[primary].host"

        ref = env._env_var_to_reference("DATAKNOBS_CACHE__0__PORT")
        assert ref == "xref:cache.port"  # Index 0 is default

        ref = env._env_var_to_reference("DATAKNOBS_SERVER__1__WORKERS")
        assert ref == "xref:server[1].workers"

    def test_nested_attribute_naming(self):
        """Test environment variable naming for nested attributes."""
        env = EnvironmentOverrides()

        env_var = env.reference_to_env_var("xref:database[0]", "connection.pool.size")
        assert env_var == "DATAKNOBS_DATABASE__0__CONNECTION__POOL__SIZE"

    def test_custom_prefix(self):
        """Test custom environment variable prefix."""
        env = EnvironmentOverrides(prefix="MYAPP_")

        env_var = env.reference_to_env_var("xref:database[0]", "host")
        assert env_var == "MYAPP_DATABASE__0__HOST"


class TestEnvironmentOverrides:
    """Test environment variable override functionality."""

    def test_simple_override(self, env_vars):
        """Test simple environment variable override."""
        # Set environment variable
        env_vars(DATAKNOBS_DATABASE__PRIMARY__HOST="overridden.host")

        config = Config({"database": [{"name": "primary", "host": "original.host"}]})

        db = config.get("database", "primary")
        assert db["host"] == "overridden.host"

    def test_multiple_overrides(self, env_vars):
        """Test multiple environment variable overrides."""
        env_vars(DATAKNOBS_DATABASE__0__PORT="5433", DATAKNOBS_CACHE__REDIS__TTL="7200")

        config = Config(
            {
                "database": [{"name": "db1", "host": "localhost", "port": 5432}],
                "cache": [{"name": "redis", "ttl": 3600}],
            }
        )

        db = config.get("database", 0)
        assert db["port"] == 5433  # Should be converted to int

        cache = config.get("cache", "redis")
        assert cache["ttl"] == 7200  # Should be converted to int

    def test_type_conversion(self, env_vars):
        """Test automatic type conversion of environment values."""
        env_vars(
            DATAKNOBS_SERVER__WEB__PORT="8080",  # Integer
            DATAKNOBS_SERVER__WEB__DEBUG="true",  # Boolean
            DATAKNOBS_SERVER__WEB__TIMEOUT="30.5",  # Float
            DATAKNOBS_SERVER__WEB__HOST="webserver.example.com",  # String
        )

        config = Config(
            {
                "server": [
                    {
                        "name": "web",
                        "port": 3000,
                        "debug": False,
                        "timeout": 10.0,
                        "host": "localhost",
                    }
                ]
            }
        )

        server = config.get("server", "web")
        assert server["port"] == 8080
        assert isinstance(server["port"], int)

        assert server["debug"] is True
        assert isinstance(server["debug"], bool)

        assert server["timeout"] == 30.5
        assert isinstance(server["timeout"], float)

        assert server["host"] == "webserver.example.com"
        assert isinstance(server["host"], str)

    def test_boolean_conversion(self, env_vars):
        """Test various boolean value formats."""
        env = EnvironmentOverrides()

        # Test various true values
        assert env._parse_value("true") is True
        assert env._parse_value("True") is True
        assert env._parse_value("TRUE") is True
        assert env._parse_value("yes") is True
        assert env._parse_value("1") is True

        # Test various false values
        assert env._parse_value("false") is False
        assert env._parse_value("False") is False
        assert env._parse_value("FALSE") is False
        assert env._parse_value("no") is False
        assert env._parse_value("0") is False

    def test_disable_env_overrides(self, env_vars):
        """Test disabling environment variable overrides."""
        env_vars(DATAKNOBS_DATABASE__0__HOST="overridden")

        # Create config with env overrides disabled
        config = Config({"database": [{"name": "db", "host": "original"}]}, use_env=False)

        db = config.get("database", 0)
        assert db["host"] == "original"  # Should not be overridden

    def test_the_type_segment_is_matched_verbatim(self, env_vars):
        """Test that TYPE is the config's own key, with no plural folding.

        The documentation carried forty-odd examples spelling the type
        singular against a config declaring it plural. Every one of them was
        inert: `_apply_environment_overrides` hands the parsed name straight
        to `get`, which raises `ConfigNotFoundError`, and the override is
        logged at WARNING and dropped. Nothing about the variable's shape says
        it missed.
        """
        env_vars(
            DATAKNOBS_DATABASE__PRIMARY__HOST="singular",
            DATAKNOBS_DATABASES__PRIMARY__PORT="6000",
        )
        declared = {"databases": [{"name": "primary", "host": "original", "port": 5432}]}

        db = Config(declared).get("databases", "primary")

        assert db["host"] == "original", "a type that does not exist is skipped"
        assert db["port"] == 6000, "the type spelled as declared is applied"

    def test_a_list_element_is_addressable_by_index(self, env_vars):
        """Test the other direction of the descent: an index into a list.

        `A__B__C__0` used to read as an attribute literally named `c__0`, so
        the list gained a flat neighbour and kept its elements. A trailing
        segment that is all digits now indexes the list it descends into, which
        is the shape readers reach for second.
        """
        env_vars(DATAKNOBS_SERVICES__API__ALLOWED_ORIGINS__0="https://app.example.com")

        declared = {"services": [{"name": "api", "allowed_origins": ["http://localhost:3000"]}]}
        api = Config(declared).get("services", "api")

        assert api["allowed_origins"] == ["https://app.example.com"]
        assert "allowed_origins__0" not in api

    def test_an_index_past_the_end_of_a_list_is_refused(self, env_vars, caplog):
        """Test that a list is not extended to make an override fit.

        An absent dict key is created, because that is what the single-segment
        case has always done. An absent list index cannot be: there is no
        position to create, and silently appending would put the value
        somewhere the operator did not name.
        """
        env_vars(DATAKNOBS_SERVICES__API__ALLOWED_ORIGINS__7="https://app.example.com")

        declared = {"services": [{"name": "api", "allowed_origins": ["http://localhost:3000"]}]}
        with caplog.at_level(logging.WARNING, logger="dataknobs_config.config"):
            api = Config(declared).get("services", "api")

        assert api["allowed_origins"] == ["http://localhost:3000"]
        assert "allowed_origins__7" not in api
        assert "allowed_origins__7" in caplog.text

    def test_from_file_can_decline_the_environment(self, temp_dir, env_vars):
        """Test that `from_file` forwards the constructor's opt-out."""
        config_file = temp_dir / "config.yaml"
        config_file.write_text("database:\n  - name: db\n    host: original\n")
        env_vars(DATAKNOBS_DATABASE__DB__HOST="overridden")

        # The switch was read out of `**kwargs` by `__init__` alone, so the
        # documented door was the one that could not decline: `from_file` had
        # no `use_env` to forward and every classmethod-built Config took the
        # environment whether or not the caller wanted it.
        assert Config.from_file(config_file, use_env=False).get("database", "db")["host"] == (
            "original"
        )
        assert Config.from_file(config_file).get("database", "db")["host"] == "overridden"

    def test_from_dict_can_decline_the_environment(self, env_vars):
        """Test that `from_dict` forwards the constructor's opt-out."""
        env_vars(DATAKNOBS_DATABASE__DB__HOST="overridden")
        declared = {"database": [{"name": "db", "host": "original"}]}

        assert Config.from_dict(declared, use_env=False).get("database", "db")["host"] == (
            "original"
        )
        assert Config.from_dict(declared).get("database", "db")["host"] == "overridden"

    def test_a_misspelled_opt_out_is_refused_rather_than_ignored(self, env_vars):
        """Test that a typo in the switch raises instead of failing open."""
        env_vars(DATAKNOBS_DATABASE__DB__HOST="overridden")

        # `use_env` used to be read as `kwargs.get("use_env", True)`, so every
        # misspelling was absorbed by `**kwargs` and left the overrides on --
        # silently, and for the one switch that decides whether environment
        # values reach configuration at all. Declaring the parameter makes
        # Python refuse the call.
        with pytest.raises(TypeError, match="use_emv"):
            Config({"database": [{"name": "db", "host": "original"}]}, use_emv=False)

    def test_override_nonexistent_config(self, env_vars):
        """Test that overrides for nonexistent configs don't cause errors."""
        env_vars(DATAKNOBS_NONEXISTENT__0__VALUE="test", DATAKNOBS_DATABASE__NOTFOUND__HOST="test")

        # Should not raise an error, just skip invalid overrides
        config = Config({"database": [{"name": "db1"}]})

        assert "nonexistent" not in config.get_types()

    def test_a_nested_attribute_reaches_the_nested_value(self, env_vars):
        """Test that a `__` inside the attribute field descends into the value.

        The grammar is three fields — TYPE__NAME_OR_INDEX__ATTRIBUTE — and
        `_env_var_to_reference` rejoins everything past the second separator
        into one attribute name. That name used to be assigned flat, so an
        override aimed at a nested value wrote a junk key beside its target and
        left the target alone, with nothing logged: the operator saw a variable
        that looked applied and a value that had not moved.
        """
        env_vars(DATAKNOBS_DATABASE__0__CONNECTION__TIMEOUT="60")

        declared = {"name": "db", "connection": {"timeout": 30, "retry": 3}}
        config = Config({"database": [dict(declared)]})

        db = config.get("database", 0)

        assert db["connection"] == {"timeout": 60, "retry": 3}
        assert "connection__timeout" not in db

    def test_a_path_that_does_not_resolve_is_refused_rather_than_written(self, env_vars, caplog):
        """Test that a descent which cannot be made writes nothing at all.

        This is the disposition the flat assignment never took. An unknown
        *type* has always been logged and skipped, because `get` raises — see
        test_override_nonexistent_config. An unknown *path* was created
        instead, as a key nobody reads. It is now treated the same way as the
        unknown type.
        """
        env_vars(DATAKNOBS_DATABASE__0__NOPE__TIMEOUT="60")

        declared = {"name": "db", "connection": {"timeout": 30}}
        with caplog.at_level(logging.WARNING, logger="dataknobs_config.config"):
            db = Config({"database": [dict(declared)]}).get("database", 0)

        assert db["connection"] == {"timeout": 30}
        assert "nope__timeout" not in db
        assert "nope" not in db
        assert "nope__timeout" in caplog.text

    def test_descending_through_a_scalar_is_refused(self, env_vars, caplog):
        """Test the other way a descent fails: the path runs into a leaf."""
        env_vars(DATAKNOBS_DATABASE__0__HOST__PORT="6000")

        declared = {"name": "db", "host": "localhost"}
        with caplog.at_level(logging.WARNING, logger="dataknobs_config.config"):
            db = Config({"database": [dict(declared)]}).get("database", 0)

        assert db["host"] == "localhost"
        assert "host__port" not in db
        assert "host__port" in caplog.text

    def test_a_single_segment_attribute_is_still_created(self, env_vars):
        """Test that the change is scoped to the shape that was broken.

        An attribute with no separator in it has always been created when
        absent, and that is how a config gains a value it did not declare.
        Descent applies to multi-segment names only, so this is untouched.
        """
        env_vars(DATAKNOBS_DATABASE__0__WORKERS="8")

        db = Config({"database": [{"name": "db"}]}).get("database", 0)

        assert db["workers"] == 8

    def test_a_reference_naming_no_attribute_parses_to_none(self):
        """Test the contract that makes the attr-less branch necessary.

        `parse_env_reference` is public and its attribute is declared
        optional: a reference that names a whole configuration rather than one
        of its values legitimately has none. `reference_to_env_var` is the
        other consumer of that state and raises on it; the override loop skips
        it. Neither is dead code, and this is why.
        """
        env = EnvironmentOverrides()

        assert env.parse_env_reference("xref:databases[primary]") == (
            "databases",
            "primary",
            None,
        )
        assert env.parse_env_reference("xref:databases[primary].host") == (
            "databases",
            "primary",
            "host",
        )

    def test_a_reference_naming_no_attribute_is_skipped(self, caplog):
        """Test that an attr-less reference is logged and dropped.

        No `DATAKNOBS_` variable produces one: `_env_var_to_reference` requires
        three fields and always joins the third onto the reference with a `.`,
        so every key `get_overrides` builds carries an attribute. The state is
        reachable through `parse_env_reference` itself, which is public, so the
        loop is written against its declared return rather than against its
        one current producer. Driving the loop with a real overrides source
        that returns such a reference is what pins the skip.
        """

        class WholeConfigOverrides(EnvironmentOverrides):
            """An overrides source naming a configuration, not a value."""

            def get_overrides(self) -> dict:
                return {"xref:database[primary]": 6000}

        config = Config({"database": [{"name": "primary", "port": 5432}]}, use_env=False)
        config._environment_overrides = WholeConfigOverrides()

        with caplog.at_level(logging.WARNING, logger="dataknobs_config.config"):
            config._apply_environment_overrides()

        assert config.get("database", "primary")["port"] == 5432, "nothing was written"
        assert "names no attribute" in caplog.text


class TestInjectedOverridesSource:
    """Test that the environment source `Config` reads is the caller's to supply."""

    def test_an_injected_source_supplies_the_prefix(self, env_vars):
        """Test that a custom prefix is reachable through `Config`.

        `EnvironmentOverrides` has declared a `prefix` since it was written and
        honours it when driven directly, but `Config` constructed its own with
        the default and offered no way to pass one, so no `Config` caller could
        reach the option at all.
        """
        env_vars(MYAPP_DATABASE__0__HOST="from.myapp")

        config = Config(
            {"database": [{"name": "db", "host": "original"}]},
            env_overrides=EnvironmentOverrides(prefix="MYAPP_"),
        )

        assert config.get("database", 0)["host"] == "from.myapp"

    def test_an_injected_source_is_applied_by_the_one_loop(self, env_vars):
        """Test that injecting a source does not fork the assignment logic.

        The guide used to answer "custom prefix" with a recipe that drove
        `EnvironmentOverrides` itself and assigned `item[attr] = value`, which
        is `_apply_environment_overrides` rewritten by hand -- and rewritten as
        it behaved before the attribute path was walked. A caller who took that
        recipe got a junk key beside the target and the target untouched, which
        is what this asserts has no second implementation to drift from.
        """
        env_vars(MYAPP_DATABASE__0__CONNECTION__TIMEOUT="60")

        config = Config(
            {"database": [{"name": "db", "connection": {"timeout": 30, "retry": 3}}]},
            env_overrides=EnvironmentOverrides(prefix="MYAPP_"),
        )

        db = config.get("database", 0)
        assert db["connection"] == {"timeout": 60, "retry": 3}
        assert "connection__timeout" not in db

    def test_a_subclass_can_filter_what_reaches_configuration(self, env_vars):
        """Test the other half of what the recipe was reached for.

        Filtering is a property of the source, so it belongs to a subclass of
        the source rather than to a reimplementation of the loop that reads it.
        """
        env_vars(
            DATAKNOBS_DATABASE__0__HOST="applied",
            DATAKNOBS_DATABASE__0__PASSWORD="should-not-land",
        )

        class SkipSecrets(EnvironmentOverrides):
            """An overrides source that declines to hand over secrets."""

            def get_overrides(self) -> dict:
                return {
                    ref: value
                    for ref, value in super().get_overrides().items()
                    if self.parse_env_reference(ref)[2] not in ("password",)
                }

        config = Config(
            {"database": [{"name": "db", "host": "original", "password": "declared"}]},
            env_overrides=SkipSecrets(),
        )

        db = config.get("database", 0)
        assert db["host"] == "applied"
        assert db["password"] == "declared"

    def test_from_file_forwards_the_source(self, temp_dir, env_vars):
        """Test that the classmethods reach the parameter too.

        `use_env` was a constructor-only switch until it was declared, and the
        classmethods could not decline the environment at all. A source the
        classmethods cannot take is the same gap one parameter along.
        """
        env_vars(MYAPP_DATABASE__PRIMARY__HOST="from.myapp")
        config_file = temp_dir / "config.yaml"
        config_file.write_text("database:\n  - name: primary\n    host: file.host\n")

        config = Config.from_file(config_file, env_overrides=EnvironmentOverrides(prefix="MYAPP_"))

        assert config.get("database", "primary")["host"] == "from.myapp"

    def test_from_dict_forwards_the_source(self, env_vars):
        """Test the other classmethod, for the same reason."""
        env_vars(MYAPP_DATABASE__0__HOST="from.myapp")

        config = Config.from_dict(
            {"database": [{"name": "db", "host": "original"}]},
            env_overrides=EnvironmentOverrides(prefix="MYAPP_"),
        )

        assert config.get("database", 0)["host"] == "from.myapp"

    def test_a_source_the_switch_would_discard_is_refused(self):
        """Test that the two environment parameters cannot contradict silently.

        `use_env=False` means no environment value reaches configuration, so a
        source passed alongside it would be built, never read, and never
        mentioned. Refusing says which of the two the caller has to change.
        """
        with pytest.raises(ValueError, match="use_env=False"):
            Config(
                {"database": [{"name": "db"}]},
                use_env=False,
                env_overrides=EnvironmentOverrides(prefix="MYAPP_"),
            )


class TestEnvironmentIntegration:
    """Test environment override integration with Config."""

    def test_env_override_with_file_loading(self, temp_dir, env_vars):
        """Test environment overrides work with file-loaded configs."""
        # Create config file
        config_file = temp_dir / "config.yaml"
        config_file.write_text("""
database:
  - name: primary
    host: file.host
    port: 5432
""")

        # Set environment override
        env_vars(DATAKNOBS_DATABASE__PRIMARY__HOST="env.host")

        # Load config
        config = Config.from_file(config_file)

        db = config.get("database", "primary")
        assert db["host"] == "env.host"
        assert db["port"] == 5432  # Not overridden

    def test_env_override_precedence(self, env_vars):
        """Test that environment overrides take precedence."""
        env_vars(DATAKNOBS_SERVER__WEB__PORT="9000")

        config = Config(
            {"server": [{"name": "web", "port": 8000}]},
            {"server": [{"name": "web", "port": 8080}]},  # Second source
        )

        server = config.get("server", "web")
        # Environment should override all sources
        assert server["port"] == 9000
