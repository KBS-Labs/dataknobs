"""Unit tests for PostgreSQL ensure_database config and database name validation.

These tests do NOT require a running PostgreSQL instance.
"""

import pytest

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_data.backends.config import PostgresDatabaseConfig
from dataknobs_data.backends.postgres_mixins import validate_database_name
from dataknobs_data.pooling.postgres import PostgresPoolConfig


class TestPoolConfigNoEnsureDatabase:
    """ensure_database is a setup flag, not a pool parameter."""

    @pytest.fixture(autouse=True)
    def _clear_postgres_env(self, monkeypatch):
        """Isolate from ambient POSTGRES_*/DATABASE_URL env vars and
        ``.env`` / ``.project_vars`` files in the repo tree.

        These tests assert dataclass-level defaults and explicit-value
        wiring; env or dotenv fallbacks in the normalizer would
        otherwise bleed shell/workspace config into the assertions.
        """
        for key in (
            "DATABASE_URL",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            "dataknobs_common.postgres_config._load_dotenv_fallbacks",
            lambda start_path=None: {},
        )

    def test_pool_config_has_no_ensure_database_field(self) -> None:
        config = PostgresPoolConfig()
        assert not hasattr(config, "ensure_database")

    def test_from_dict_ignores_ensure_database(self) -> None:
        config = PostgresPoolConfig.from_dict({"ensure_database": False})
        assert not hasattr(config, "ensure_database")

    def test_from_dict_with_connection_string(self) -> None:
        config = PostgresPoolConfig.from_dict(
            {
                "connection_string": "postgresql://user:pass@host:5432/mydb",
            }
        )
        assert config.database == "mydb"
        assert config.host == "host"
        assert config.port == 5432

    def test_from_dict_preserves_fields(self) -> None:
        config = PostgresPoolConfig.from_dict(
            {
                "host": "dbhost",
                "port": 5433,
                "database": "mydb",
                "user": "admin",
                "password": "secret",
            }
        )
        assert config.host == "dbhost"
        assert config.port == 5433
        assert config.database == "mydb"
        assert config.user == "admin"
        assert config.password == "secret"


class TestValidateDatabaseName:
    """Tests for database name validation."""

    @pytest.mark.parametrize(
        "name",
        [
            "mydb",
            "my_database",
            "DB123",
            "_private",
            "a",
            "test_db_001",
        ],
    )
    def test_valid_names(self, name: str) -> None:
        validate_database_name(name)  # Should not raise

    @pytest.mark.parametrize(
        "name,reason",
        [
            ("my-db", "hyphens"),
            ("my db", "spaces"),
            ("123db", "starts with digit"),
            ('"; DROP TABLE users; --', "SQL injection"),
            ("my.db", "dots"),
            ("", "empty string — fails ^[a-zA-Z_] anchor"),
            ("my/db", "slashes"),
        ],
    )
    def test_invalid_names(self, name: str, reason: str) -> None:
        with pytest.raises(ConfigurationError, match="Invalid database name"):
            validate_database_name(name)


class TestPostgresConfigEnsureDatabase:
    """The typed config reads ``ensure_database`` (both backends construct from it)."""

    def test_default_true(self) -> None:
        cfg = PostgresDatabaseConfig.from_dict({"host": "localhost", "database": "mydb"})
        assert cfg.ensure_database is True

    def test_explicit_false(self) -> None:
        cfg = PostgresDatabaseConfig.from_dict(
            {"host": "localhost", "database": "mydb", "ensure_database": False}
        )
        assert cfg.ensure_database is False


class TestPostgresConfigBoolCoercion:
    """Tests that ensure_database string values are coerced correctly (A1)."""

    @pytest.mark.parametrize(
        "value,expected",
        [
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
            ("anything_else", True),  # blocklist: only explicit falsy strings return False
        ],
    )
    def test_bool_coercion(self, value: bool | str, expected: bool) -> None:
        cfg = PostgresDatabaseConfig.from_dict({"ensure_database": value})
        assert cfg.ensure_database is expected


class TestPostgresConfigConnectionString:
    """Tests that the typed config normalizes connection_string (P1)."""

    def test_normalizes_connection_string_into_individual_keys(self) -> None:
        cfg = PostgresDatabaseConfig.from_dict(
            {"connection_string": "postgresql://admin:secret@dbhost:5433/mydb"}
        )
        assert cfg.host == "dbhost"
        assert cfg.port == 5433
        assert cfg.database == "mydb"
        assert cfg.user == "admin"
        assert cfg.password == "secret"

    def test_individual_keys_win_over_connection_string(self) -> None:
        """When both are present, individual keys override URL fields.

        The normalizer documents this precedence: individual keys >
        ``connection_string`` > env. This preserves the historical
        "use this URL, but override the database" pattern — a common
        way to aim a test suite at a non-default database while
        reusing a shared URL for the other fields.
        """
        cfg = PostgresDatabaseConfig.from_dict(
            {
                "connection_string": "postgresql://admin:secret@dbhost:5433/mydb",
                "database": "override_db",
            }
        )
        assert cfg.database == "override_db"  # individual key wins
        assert cfg.host == "dbhost"  # from connection_string

    def test_connection_string_with_ensure_database(self) -> None:
        cfg = PostgresDatabaseConfig.from_dict(
            {
                "connection_string": "postgresql://admin:secret@dbhost:5433/mydb",
                "ensure_database": False,
            }
        )
        assert cfg.ensure_database is False
        assert cfg.database == "mydb"

    def test_connection_string_default_ensure_database_true(self) -> None:
        cfg = PostgresDatabaseConfig.from_dict(
            {"connection_string": "postgresql://admin:secret@dbhost:5433/mydb"}
        )
        assert cfg.ensure_database is True
        assert cfg.database == "mydb"


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

        exc = psycopg2.OperationalError('FATAL:  database "dk_nonexistent" does not exist\n')
        # pgcode is None for manually-constructed OperationalError (matches
        # real behavior — psycopg2 doesn't set pgcode on connection failures)
        assert getattr(exc, "pgcode", None) is None
        assert SyncPostgresDatabase._is_invalid_catalog_error(exc) is True

    def test_rejects_operational_error_connection_refused(self) -> None:
        """Connection refused should NOT trigger database creation."""
        import psycopg2
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        exc = psycopg2.OperationalError(
            'connection to server at "localhost" (127.0.0.1), port 5432 failed: Connection refused'
        )
        assert SyncPostgresDatabase._is_invalid_catalog_error(exc) is False

    def test_rejects_operational_error_auth_failure(self) -> None:
        """Auth failures should NOT trigger database creation."""
        import psycopg2
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        exc = psycopg2.OperationalError('FATAL:  password authentication failed for user "baduser"')
        assert SyncPostgresDatabase._is_invalid_catalog_error(exc) is False

    def test_rejects_non_operational_error(self) -> None:
        from dataknobs_data.backends.postgres import SyncPostgresDatabase

        assert SyncPostgresDatabase._is_invalid_catalog_error(ValueError("bad")) is False
        assert SyncPostgresDatabase._is_invalid_catalog_error(RuntimeError("fail")) is False
