"""Unit tests for PostgreSQL ensure_database config and database name validation.

These tests do NOT require a running PostgreSQL instance.
"""

import pytest

from dataknobs_data.backends.postgres_mixins import (
    PostgresBaseConfig,
    validate_database_name,
)
from dataknobs_data.pooling.postgres import PostgresPoolConfig


class TestPoolConfigNoEnsureDatabase:
    """ensure_database is a setup flag, not a pool parameter."""

    def test_pool_config_has_no_ensure_database_field(self) -> None:
        config = PostgresPoolConfig()
        assert not hasattr(config, "ensure_database")

    def test_from_dict_ignores_ensure_database(self) -> None:
        config = PostgresPoolConfig.from_dict({"ensure_database": False})
        assert not hasattr(config, "ensure_database")

    def test_from_dict_with_connection_string(self) -> None:
        config = PostgresPoolConfig.from_dict({
            "connection_string": "postgresql://user:pass@host:5432/mydb",
        })
        assert config.database == "mydb"
        assert config.host == "host"
        assert config.port == 5432

    def test_from_dict_preserves_fields(self) -> None:
        config = PostgresPoolConfig.from_dict({
            "host": "dbhost",
            "port": 5433,
            "database": "mydb",
            "user": "admin",
            "password": "secret",
        })
        assert config.host == "dbhost"
        assert config.port == 5433
        assert config.database == "mydb"
        assert config.user == "admin"
        assert config.password == "secret"


class TestValidateDatabaseName:
    """Tests for database name validation."""

    @pytest.mark.parametrize("name", [
        "mydb",
        "my_database",
        "DB123",
        "_private",
        "a",
        "test_db_001",
    ])
    def test_valid_names(self, name: str) -> None:
        validate_database_name(name)  # Should not raise

    @pytest.mark.parametrize("name,reason", [
        ("my-db", "hyphens"),
        ("my db", "spaces"),
        ("123db", "starts with digit"),
        ('"; DROP TABLE users; --', "SQL injection"),
        ("my.db", "dots"),
        ("", "empty string — fails ^[a-zA-Z_] anchor"),
        ("my/db", "slashes"),
    ])
    def test_invalid_names(self, name: str, reason: str) -> None:
        with pytest.raises(ValueError, match="Invalid database name"):
            validate_database_name(name)


class TestParsePostgresConfigEnsureDatabase:
    """Tests that _parse_postgres_config extracts ensure_database correctly."""

    def test_default_true(self) -> None:
        mixin = PostgresBaseConfig()
        _, _, conn_config, ensure_db = mixin._parse_postgres_config({
            "host": "localhost",
            "database": "mydb",
        })
        assert ensure_db is True
        assert "ensure_database" not in conn_config

    def test_explicit_false(self) -> None:
        mixin = PostgresBaseConfig()
        _, _, conn_config, ensure_db = mixin._parse_postgres_config({
            "host": "localhost",
            "database": "mydb",
            "ensure_database": False,
        })
        assert ensure_db is False
        assert "ensure_database" not in conn_config


class TestParsePostgresConfigBoolCoercion:
    """Tests that ensure_database string values are coerced correctly (A1)."""

    @pytest.mark.parametrize("value,expected", [
        (True, True),
        (False, False),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("", False),
        ("anything_else", False),
    ])
    def test_bool_coercion(self, value: bool | str, expected: bool) -> None:
        mixin = PostgresBaseConfig()
        _, _, _, ensure_db = mixin._parse_postgres_config({
            "ensure_database": value,
        })
        assert ensure_db is expected


class TestParsePostgresConfigConnectionString:
    """Tests that _parse_postgres_config normalizes connection_string (P1)."""

    def test_normalizes_connection_string_into_individual_keys(self) -> None:
        mixin = PostgresBaseConfig()
        _, _, conn_config, _ = mixin._parse_postgres_config({
            "connection_string": "postgresql://admin:secret@dbhost:5433/mydb",
        })
        assert conn_config["host"] == "dbhost"
        assert conn_config["port"] == 5433
        assert conn_config["database"] == "mydb"
        assert conn_config["user"] == "admin"
        assert conn_config["password"] == "secret"
        # connection_string still present for PostgresPoolConfig.from_dict
        assert "connection_string" in conn_config

    def test_individual_keys_win_over_connection_string(self) -> None:
        mixin = PostgresBaseConfig()
        _, _, conn_config, _ = mixin._parse_postgres_config({
            "connection_string": "postgresql://admin:secret@dbhost:5433/mydb",
            "database": "override_db",
        })
        assert conn_config["database"] == "override_db"
        assert conn_config["host"] == "dbhost"  # from connection_string

    def test_connection_string_with_ensure_database(self) -> None:
        mixin = PostgresBaseConfig()
        _, _, conn_config, ensure_db = mixin._parse_postgres_config({
            "connection_string": "postgresql://admin:secret@dbhost:5433/mydb",
            "ensure_database": False,
        })
        assert ensure_db is False
        assert conn_config["database"] == "mydb"

    def test_connection_string_default_ensure_database_true(self) -> None:
        mixin = PostgresBaseConfig()
        _, _, conn_config, ensure_db = mixin._parse_postgres_config({
            "connection_string": "postgresql://admin:secret@dbhost:5433/mydb",
        })
        assert ensure_db is True
        assert conn_config["database"] == "mydb"


class TestIsInvalidCatalogError:
    """Tests for SyncPostgresDatabase._is_invalid_catalog_error.

    This static method gates whether the catch-and-create path fires.
    psycopg2 connection-level OperationalError has pgcode=None (read-only,
    set only by the C layer from server responses), so the method falls
    back to message matching for the common case.
    """

    def test_matches_operational_error_with_does_not_exist_message(self) -> None:
        """The real-world case: FATAL: database "x" does not exist."""
        import psycopg2
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        exc = psycopg2.OperationalError(
            'FATAL:  database "dk_nonexistent" does not exist\n'
        )
        # pgcode is None for manually-constructed OperationalError (matches
        # real behavior — psycopg2 doesn't set pgcode on connection failures)
        assert getattr(exc, "pgcode", None) is None
        assert SyncPostgresDatabase._is_invalid_catalog_error(exc) is True

    def test_rejects_operational_error_connection_refused(self) -> None:
        """Connection refused should NOT trigger database creation."""
        import psycopg2
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        exc = psycopg2.OperationalError(
            "connection to server at \"localhost\" (127.0.0.1), port 5432 "
            "failed: Connection refused"
        )
        assert SyncPostgresDatabase._is_invalid_catalog_error(exc) is False

    def test_rejects_operational_error_auth_failure(self) -> None:
        """Auth failures should NOT trigger database creation."""
        import psycopg2
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        exc = psycopg2.OperationalError(
            'FATAL:  password authentication failed for user "baduser"'
        )
        assert SyncPostgresDatabase._is_invalid_catalog_error(exc) is False

    def test_rejects_non_operational_error(self) -> None:
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        assert SyncPostgresDatabase._is_invalid_catalog_error(ValueError("bad")) is False
        assert SyncPostgresDatabase._is_invalid_catalog_error(RuntimeError("fail")) is False
