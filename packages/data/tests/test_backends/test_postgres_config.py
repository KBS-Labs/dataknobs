"""Config-parsing tests for the PostgreSQL backends.

These tests exercise ``PostgresBaseConfig._parse_postgres_config`` and
``SyncPostgresDatabase`` / ``AsyncPostgresDatabase`` config plumbing
without requiring a running postgres server. They verify that the
shared normalizer is wired in correctly and that both input shapes
(``connection_string`` and individual keys) produce the same
canonical ``_conn_config`` dict.
"""

from __future__ import annotations

import pytest

from dataknobs_data.backends.postgres import (
    AsyncPostgresDatabase,
    SyncPostgresDatabase,
)


_POSTGRES_ENV_KEYS = (
    "DATABASE_URL",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)


@pytest.fixture(autouse=True)
def _clear_postgres_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from ambient postgres env vars."""
    for key in _POSTGRES_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_connection_string_and_individual_keys_equivalent() -> None:
    """Both input shapes produce the same canonical _conn_config."""
    via_url = SyncPostgresDatabase(
        {"connection_string": "postgresql://u:p@h:5433/db"}
    )
    via_keys = SyncPostgresDatabase(
        {
            "host": "h",
            "port": 5433,
            "database": "db",
            "user": "u",
            "password": "p",
        }
    )
    for key in ("host", "port", "database", "user", "password"):
        assert via_url._conn_config[key] == via_keys._conn_config[key]
    assert (
        via_url._conn_config["connection_string"]
        == via_keys._conn_config["connection_string"]
    )


def test_dialect_prefix_stripped() -> None:
    db = SyncPostgresDatabase(
        {"connection_string": "postgresql+asyncpg://u:p@h/db"}
    )
    assert db._conn_config["connection_string"].startswith("postgresql://")
    assert "asyncpg" not in db._conn_config["connection_string"]
    assert db._conn_config["host"] == "h"


def test_sync_postgres_reads_postgres_env_vars_via_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With empty config, POSTGRES_* env vars populate _conn_config."""
    monkeypatch.setenv("POSTGRES_HOST", "env-h")
    monkeypatch.setenv("POSTGRES_PORT", "5678")
    monkeypatch.setenv("POSTGRES_DB", "env-db")
    monkeypatch.setenv("POSTGRES_USER", "env-u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "env-p")

    db = SyncPostgresDatabase({})
    assert db._conn_config["host"] == "env-h"
    assert db._conn_config["port"] == 5678
    assert db._conn_config["database"] == "env-db"
    assert db._conn_config["user"] == "env-u"
    assert db._conn_config["password"] == "env-p"


def test_empty_config_no_env_preserves_pre_connect_behavior() -> None:
    """Before connect(), _conn_config may be empty when nothing configured.

    The require=False path in the normalizer means no exception at init —
    failures surface only when the connection is actually attempted.
    """
    db = SyncPostgresDatabase({})
    # Nothing to populate; _conn_config is the fallback empty dict minus
    # vector keys. Constructing must not raise.
    assert isinstance(db._conn_config, dict)


def test_async_postgres_connection_string_parsed_into_keys() -> None:
    """AsyncPostgresDatabase uses the same normalizer through the mixin."""
    db = AsyncPostgresDatabase(
        {"connection_string": "postgresql://u:p@h:5433/db"}
    )
    # _pool_config is built from _conn_config in the constructor path,
    # so the individual keys must be populated.
    assert db._pool_config.host == "h"
    assert db._pool_config.port == 5433
    assert db._pool_config.database == "db"
    assert db._pool_config.user == "u"
    assert db._pool_config.password == "p"


def test_async_postgres_individual_keys_match_connection_string() -> None:
    """Site 1 + Site 6 parity: same observable _pool_config for both shapes."""
    via_url = AsyncPostgresDatabase(
        {"connection_string": "postgresql://u:p@h:5433/db"}
    )
    via_keys = AsyncPostgresDatabase(
        {
            "host": "h",
            "port": 5433,
            "database": "db",
            "user": "u",
            "password": "p",
        }
    )
    assert via_url._pool_config.host == via_keys._pool_config.host
    assert via_url._pool_config.port == via_keys._pool_config.port
    assert via_url._pool_config.database == via_keys._pool_config.database
    assert via_url._pool_config.user == via_keys._pool_config.user
    assert via_url._pool_config.password == via_keys._pool_config.password
