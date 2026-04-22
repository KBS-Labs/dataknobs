"""Config-parsing tests for PgVectorStore.

These tests exercise ``_parse_backend_config`` and the Defect A / C
behavior at the layer that does not require a running postgres:

- Config-shape parity: ``connection_string``, individual keys,
  ``DATABASE_URL`` env var, ``POSTGRES_*`` env vars all reach the
  same canonical normalized form.
- Default ``id_type`` (Defect A).
- ``id_type`` validation.

Behavioral/integration tests (add_vectors text IDs, guided errors on
UUID mismatch, etc.) live alongside the existing pgvector integration
tests and require ``TEST_POSTGRES=true``.
"""

from __future__ import annotations

import pytest

try:
    import asyncpg  # noqa: F401

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ASYNCPG_AVAILABLE, reason="asyncpg not installed"
)

if ASYNCPG_AVAILABLE:
    from dataknobs_data.vector.stores.pgvector import PgVectorStore


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
    """Isolate tests from ambient postgres env vars and dotenv files."""
    for key in _POSTGRES_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "dataknobs_common.postgres_config._load_dotenv_fallbacks",
        lambda start_path=None: {},
    )


# ---------------------------------------------------------------------------
# Config-shape parity (Defect B)
# ---------------------------------------------------------------------------


def test_connection_string_still_works() -> None:
    store = PgVectorStore(
        {
            "connection_string": "postgresql://u:p@h:5433/db",
            "dimensions": 4,
        }
    )
    assert store.connection_string == "postgresql://u:p@h:5433/db"


def test_individual_keys_synthesize_connection_string() -> None:
    store = PgVectorStore(
        {
            "host": "h",
            "port": 5433,
            "database": "db",
            "user": "u",
            "password": "p",
            "dimensions": 4,
        }
    )
    assert store.connection_string == "postgresql://u:p@h:5433/db"


def test_database_url_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://u:p@env-h:6000/env-db"
    )
    store = PgVectorStore({"dimensions": 4})
    assert store.connection_string == "postgresql://u:p@env-h:6000/env-db"


def test_postgres_env_vars_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "env-h")
    monkeypatch.setenv("POSTGRES_PORT", "5678")
    monkeypatch.setenv("POSTGRES_DB", "env-db")
    monkeypatch.setenv("POSTGRES_USER", "env-u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "env-p")
    store = PgVectorStore({"dimensions": 4})
    assert store.connection_string == (
        "postgresql://env-u:env-p@env-h:5678/env-db"
    )


def test_asyncpg_dialect_prefix_normalized() -> None:
    store = PgVectorStore(
        {
            "connection_string": "postgresql+asyncpg://u:p@h:5433/db",
            "dimensions": 4,
        }
    )
    assert store.connection_string.startswith("postgresql://")
    assert "asyncpg" not in store.connection_string


def test_missing_all_forms_raises() -> None:
    with pytest.raises(ValueError) as excinfo:
        PgVectorStore({"dimensions": 4})
    msg = str(excinfo.value)
    assert "postgres" in msg.lower()


# ---------------------------------------------------------------------------
# id_type configuration (Defect A)
# ---------------------------------------------------------------------------


def test_id_type_default_is_text() -> None:
    """Defect A: default flipped from "uuid" to "text".

    Consumers that do not specify ``id_type`` get text ids — matching
    the RAG chunk-id use case without requiring any config change.
    """
    store = PgVectorStore(
        {
            "connection_string": "postgresql://u:p@h/db",
            "dimensions": 4,
        }
    )
    assert store.id_type == "text"


def test_id_type_explicit_uuid() -> None:
    store = PgVectorStore(
        {
            "connection_string": "postgresql://u:p@h/db",
            "dimensions": 4,
            "id_type": "uuid",
        }
    )
    assert store.id_type == "uuid"


def test_id_type_explicit_text() -> None:
    store = PgVectorStore(
        {
            "connection_string": "postgresql://u:p@h/db",
            "dimensions": 4,
            "id_type": "text",
        }
    )
    assert store.id_type == "text"


def test_id_type_invalid_raises() -> None:
    with pytest.raises(ValueError, match="id_type must be"):
        PgVectorStore(
            {
                "connection_string": "postgresql://u:p@h/db",
                "dimensions": 4,
                "id_type": "bigint",
            }
        )
