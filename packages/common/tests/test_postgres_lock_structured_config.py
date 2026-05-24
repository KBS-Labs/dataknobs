"""Construction-path tests for the structured-config lock refactor.

``PostgresAdvisoryLock`` builds through a typed
:class:`PostgresLockConfig` (a
:class:`~dataknobs_common.structured_config.StructuredConfigConsumer`),
mirroring the post-refactor ``PostgresEventBus`` / ``PostgresEventBusConfig``.
These tests pin the new contract:

- ``PostgresLockConfig`` resolves every accepted input shape
  (``connection_string``, individual host/port/... keys, ``DATABASE_URL``,
  ``POSTGRES_*`` env-var fallbacks) to the same canonical DSN, and
  round-trips through ``from_dict``/``to_dict``.
- The three construction shapes (positional ``connection_string``, loose
  dict, typed config) plus ``from_config`` all agree; mixing the typed
  config with the positional raises ``TypeError``; a missing DSN raises
  ``ConfigurationError``.

No service is required — they exercise construction only.
``PostgresLockConfig`` and ``PostgresAdvisoryLock`` construction never
import ``asyncpg`` (the driver is imported lazily inside ``acquire``),
so these run unconditionally. The behavioural arm (real
``pg_advisory_lock`` mutual exclusion) is covered by
``test_locks.py::TestPostgresAdvisoryLock`` against real Postgres.
"""

from __future__ import annotations

import dataclasses

import pytest

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.locks import PostgresAdvisoryLock, PostgresLockConfig
from dataknobs_common.testing import assert_structured_config_roundtrip

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
    """Isolate tests from ambient postgres env vars and dotenv files.

    Without this, a developer's ``DATABASE_URL`` / ``POSTGRES_*`` env or
    a project ``.env`` would mask the missing-DSN test (the normalizer
    would resolve a real connection and ``from_dict({})`` would not
    raise).
    """
    for key in _POSTGRES_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "dataknobs_common.postgres_config._load_dotenv_fallbacks",
        lambda _start_path=None: {},
    )


# ---------------------------------------------------------------------------
# PostgresLockConfig — input-shape parity + round-trip
# ---------------------------------------------------------------------------


class TestPostgresLockConfig:
    """``PostgresLockConfig`` resolves every input shape to one DSN."""

    DSN = "postgresql://u:p@h:5432/db"

    def test_connection_string_passthrough(self) -> None:
        cfg = PostgresLockConfig.from_dict({"connection_string": self.DSN})
        assert cfg.connection_string == self.DSN

    def test_individual_keys_synthesize_connection_string(self) -> None:
        cfg = PostgresLockConfig.from_dict(
            {
                "host": "h",
                "port": 5433,
                "database": "db",
                "user": "u",
                "password": "p",
            }
        )
        assert cfg.connection_string == "postgresql://u:p@h:5433/db"

    def test_database_url_env_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql://u:p@env-h:6000/env-db"
        )
        cfg = PostgresLockConfig.from_dict({})
        assert cfg.connection_string == "postgresql://u:p@env-h:6000/env-db"

    def test_postgres_env_vars_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("POSTGRES_HOST", "env-h")
        monkeypatch.setenv("POSTGRES_PORT", "5678")
        monkeypatch.setenv("POSTGRES_DB", "env-db")
        monkeypatch.setenv("POSTGRES_USER", "env-u")
        monkeypatch.setenv("POSTGRES_PASSWORD", "env-p")
        cfg = PostgresLockConfig.from_dict({})
        assert cfg.connection_string == (
            "postgresql://env-u:env-p@env-h:5678/env-db"
        )

    def test_asyncpg_dialect_prefix_normalized(self) -> None:
        cfg = PostgresLockConfig.from_dict(
            {"connection_string": "postgresql+asyncpg://u:p@h:5433/db"}
        )
        assert cfg.connection_string.startswith("postgresql://")
        assert "asyncpg" not in cfg.connection_string

    def test_missing_all_forms_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            PostgresLockConfig.from_dict({})

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Backend-routing keys (``"backend"``) in the same dict are tolerated."""
        cfg = PostgresLockConfig.from_dict(
            {"backend": "postgres", "connection_string": self.DSN}
        )
        assert cfg.connection_string == self.DSN

    def test_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            PostgresLockConfig(connection_string=self.DSN)
        )

    def test_connection_string_redacted_from_repr(self) -> None:
        """The DSN embeds the password; ``repr`` masks it, ``to_dict`` keeps it."""
        dsn = "postgresql://user:pa55w0rd@host:5432/db"
        cfg = PostgresLockConfig(connection_string=dsn)
        rendered = repr(cfg)
        assert "pa55w0rd" not in rendered
        assert "connection_string='***'" in rendered
        assert cfg.to_dict()["connection_string"] == dsn

    def test_frozen_dataclass_rejects_mutation(self) -> None:
        cfg = PostgresLockConfig(connection_string=self.DSN)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.connection_string = "postgresql://other/db"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PostgresAdvisoryLock — construction shapes (no asyncpg needed)
# ---------------------------------------------------------------------------


class TestPostgresAdvisoryLockConstructionShapes:
    """Three construction shapes, all routed through the dataclass."""

    DSN = "postgresql://u:p@h:5432/db"

    def test_positional_connection_string(self) -> None:
        lock = PostgresAdvisoryLock(self.DSN)
        assert lock.config.connection_string == self.DSN

    def test_keyword_connection_string(self) -> None:
        lock = PostgresAdvisoryLock(connection_string=self.DSN)
        assert lock.config.connection_string == self.DSN

    def test_dict_config(self) -> None:
        lock = PostgresAdvisoryLock(
            config={"connection_string": self.DSN}
        )
        assert lock.config.connection_string == self.DSN

    def test_dict_config_individual_keys(self) -> None:
        lock = PostgresAdvisoryLock(
            config={
                "host": "h",
                "port": 5433,
                "database": "db",
                "user": "u",
                "password": "p",
            }
        )
        assert lock.config.connection_string == "postgresql://u:p@h:5433/db"

    def test_typed_config(self) -> None:
        cfg = PostgresLockConfig(connection_string=self.DSN)
        lock = PostgresAdvisoryLock(config=cfg)
        assert lock.config is cfg

    def test_from_config_dict(self) -> None:
        lock = PostgresAdvisoryLock.from_config(
            {"connection_string": self.DSN}
        )
        assert lock.config.connection_string == self.DSN

    def test_from_config_typed(self) -> None:
        cfg = PostgresLockConfig(connection_string=self.DSN)
        lock = PostgresAdvisoryLock.from_config(cfg)
        assert lock.config is cfg

    def test_mixing_typed_config_with_positional_raises(self) -> None:
        cfg = PostgresLockConfig(connection_string=self.DSN)
        with pytest.raises(TypeError, match="cannot mix"):
            PostgresAdvisoryLock(self.DSN, config=cfg)

    def test_missing_dsn_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            PostgresAdvisoryLock(config={})

    def test_from_config_missing_dsn_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            PostgresAdvisoryLock.from_config({})
