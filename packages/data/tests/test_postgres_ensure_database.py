"""Unit tests for PostgreSQL ensure_database config and database name validation.

These tests do NOT require a running PostgreSQL instance.
"""

import pytest

from dataknobs_data.backends.postgres_mixins import (
    _SYSTEM_DATABASES,
    validate_database_name,
)
from dataknobs_data.pooling.postgres import PostgresPoolConfig


class TestPostgresPoolConfigEnsureDatabase:
    """Tests for ensure_database field on PostgresPoolConfig."""

    def test_default_is_true(self) -> None:
        config = PostgresPoolConfig()
        assert config.ensure_database is True

    def test_from_dict_default_is_true(self) -> None:
        config = PostgresPoolConfig.from_dict({})
        assert config.ensure_database is True

    def test_from_dict_explicit_true(self) -> None:
        config = PostgresPoolConfig.from_dict({"ensure_database": True})
        assert config.ensure_database is True

    def test_from_dict_explicit_false(self) -> None:
        config = PostgresPoolConfig.from_dict({"ensure_database": False})
        assert config.ensure_database is False

    def test_from_dict_with_connection_string_explicit_false(self) -> None:
        config = PostgresPoolConfig.from_dict({
            "connection_string": "postgresql://user:pass@host:5432/mydb",
            "ensure_database": False,
        })
        assert config.ensure_database is False
        assert config.database == "mydb"

    def test_from_dict_with_connection_string_default_true(self) -> None:
        config = PostgresPoolConfig.from_dict({
            "connection_string": "postgresql://user:pass@host:5432/mydb",
        })
        assert config.ensure_database is True
        assert config.database == "mydb"

    def test_from_dict_preserves_other_fields(self) -> None:
        config = PostgresPoolConfig.from_dict({
            "host": "dbhost",
            "port": 5433,
            "database": "mydb",
            "user": "admin",
            "password": "secret",
            "ensure_database": False,
        })
        assert config.host == "dbhost"
        assert config.port == 5433
        assert config.database == "mydb"
        assert config.user == "admin"
        assert config.password == "secret"
        assert config.ensure_database is False


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
        from dataknobs_data.backends.postgres_mixins import PostgresBaseConfig

        mixin = PostgresBaseConfig()
        _, _, conn_config, ensure_db = mixin._parse_postgres_config({
            "host": "localhost",
            "database": "mydb",
        })
        assert ensure_db is True
        # ensure_database should NOT be in the remaining connection config
        assert "ensure_database" not in conn_config

    def test_explicit_false(self) -> None:
        from dataknobs_data.backends.postgres_mixins import PostgresBaseConfig

        mixin = PostgresBaseConfig()
        _, _, conn_config, ensure_db = mixin._parse_postgres_config({
            "host": "localhost",
            "database": "mydb",
            "ensure_database": False,
        })
        assert ensure_db is False
        assert "ensure_database" not in conn_config


class TestSystemDatabases:
    """Tests for system database protection."""

    def test_system_databases_set(self) -> None:
        assert "postgres" in _SYSTEM_DATABASES
        assert "template0" in _SYSTEM_DATABASES
        assert "template1" in _SYSTEM_DATABASES

    def test_system_databases_is_frozenset(self) -> None:
        assert isinstance(_SYSTEM_DATABASES, frozenset)
