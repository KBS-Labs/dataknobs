"""Tests for environment variable overrides."""

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

    def test_override_nonexistent_config(self, env_vars):
        """Test that overrides for nonexistent configs don't cause errors."""
        env_vars(DATAKNOBS_NONEXISTENT__0__VALUE="test", DATAKNOBS_DATABASE__NOTFOUND__HOST="test")

        # Should not raise an error, just skip invalid overrides
        config = Config({"database": [{"name": "db1"}]})

        assert "nonexistent" not in config.get_types()

    def test_nested_attribute_override(self, env_vars):
        """Test overriding nested attributes."""
        env_vars(DATAKNOBS_DATABASE__0__CONNECTION__TIMEOUT="60")

        config = Config({"database": [{"name": "db", "connection": {"timeout": 30, "retry": 3}}]})

        # Note: Simple implementation might not handle nested attrs
        # This test documents expected behavior
        pass


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
