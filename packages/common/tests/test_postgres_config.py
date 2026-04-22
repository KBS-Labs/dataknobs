"""Tests for dataknobs_common.postgres_config."""

from __future__ import annotations

import pytest

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.postgres_config import (
    normalize_postgres_connection_config,
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
    """Ensure env-var-derived inputs are isolated per test.

    Also disables the ``.env`` / ``.project_vars`` fallback so
    workspace dotenv files cannot shadow the assertions.
    """
    for key in _POSTGRES_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "dataknobs_common.postgres_config._load_dotenv_fallbacks",
        lambda start_path=None: {},
    )


def test_connection_string_parses_into_keys() -> None:
    result = normalize_postgres_connection_config(
        {"connection_string": "postgresql://u:p@h:5433/db"}
    )
    assert result is not None
    assert result["host"] == "h"
    assert result["port"] == 5433
    assert result["user"] == "u"
    assert result["password"] == "p"
    assert result["database"] == "db"
    assert result["connection_string"] == "postgresql://u:p@h:5433/db"


def test_asyncpg_dialect_prefix_normalized() -> None:
    result = normalize_postgres_connection_config(
        {"connection_string": "postgresql+asyncpg://u:p@h/db"}
    )
    assert result is not None
    assert result["connection_string"].startswith("postgresql://")
    assert "asyncpg" not in result["connection_string"]
    assert result["host"] == "h"
    assert result["database"] == "db"


def test_database_url_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://u:p@env-host:5999/env-db"
    )
    result = normalize_postgres_connection_config({})
    assert result is not None
    assert result["host"] == "env-host"
    assert result["port"] == 5999
    assert result["database"] == "env-db"
    assert (
        result["connection_string"]
        == "postgresql://u:p@env-host:5999/env-db"
    )


def test_individual_keys_win_over_connection_string() -> None:
    """Individual keys override same-field values parsed from a URL.

    Restoring the historical precedence: ``{"connection_string":
    "...", "database": "override"}`` connects to ``override``, not to
    the URL's database. Keys the caller does NOT override still come
    from the URL.
    """
    result = normalize_postgres_connection_config(
        {
            "connection_string": "postgresql://u:p@url-host:5433/url-db",
            "database": "override-db",
            "user": "override-user",
        }
    )
    assert result is not None
    # Overridden fields come from the individual keys.
    assert result["database"] == "override-db"
    assert result["user"] == "override-user"
    # Non-overridden fields come from the URL.
    assert result["host"] == "url-host"
    assert result["port"] == 5433
    assert result["password"] == "p"
    # The synthesized connection_string reflects the merged values.
    assert (
        result["connection_string"]
        == "postgresql://override-user:p@url-host:5433/override-db"
    )


def test_individual_keys_fill_gaps_in_connection_string() -> None:
    """Individual keys supply values the URL omits (e.g. no port)."""
    result = normalize_postgres_connection_config(
        {
            "connection_string": "postgresql://u:p@url-host/url-db",
            "port": 5433,
        }
    )
    assert result is not None
    assert result["host"] == "url-host"
    assert result["port"] == 5433
    assert result["database"] == "url-db"


def test_individual_keys_synthesize_connection_string() -> None:
    result = normalize_postgres_connection_config(
        {
            "host": "h",
            "port": 5433,
            "database": "db",
            "user": "u",
            "password": "p",
        }
    )
    assert result is not None
    assert (
        result["connection_string"] == "postgresql://u:p@h:5433/db"
    )
    assert result["port"] == 5433


def test_postgres_env_vars_fallback_for_absent_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "env-h")
    monkeypatch.setenv("POSTGRES_DB", "env-db")
    monkeypatch.setenv("POSTGRES_USER", "env-u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "env-p")
    monkeypatch.setenv("POSTGRES_PORT", "5678")
    result = normalize_postgres_connection_config({})
    assert result is not None
    assert result["host"] == "env-h"
    assert result["port"] == 5678
    assert result["database"] == "env-db"
    assert result["user"] == "env-u"
    assert result["password"] == "env-p"


def test_individual_config_key_wins_over_postgres_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "env-h")
    result = normalize_postgres_connection_config(
        {
            "host": "config-h",
            "database": "db",
            "user": "u",
            "password": "p",
        }
    )
    assert result is not None
    assert result["host"] == "config-h"


def test_require_true_raises_when_nothing_configured() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        normalize_postgres_connection_config({})
    msg = str(excinfo.value)
    assert "connection_string" in msg
    assert "DATABASE_URL" in msg
    assert "POSTGRES_HOST" in msg


def test_require_false_returns_none_when_nothing_configured() -> None:
    result = normalize_postgres_connection_config({}, require=False)
    assert result is None


def test_password_with_special_chars_url_encoded() -> None:
    result = normalize_postgres_connection_config(
        {
            "host": "h",
            "port": 5432,
            "database": "db",
            "user": "u@ser",
            "password": "p@ss/word:1",
        }
    )
    assert result is not None
    conn = result["connection_string"]
    # Characters must be percent-encoded so the URL is valid.
    assert "@ser" not in conn.split("@")[0]
    assert "p@ss" not in conn
    assert "u%40ser" in conn
    assert "p%40ss%2Fword%3A1" in conn
    # Raw individual keys preserved.
    assert result["password"] == "p@ss/word:1"
    assert result["user"] == "u@ser"


def test_port_coerced_to_int() -> None:
    result = normalize_postgres_connection_config(
        {
            "host": "h",
            "port": "5433",
            "database": "db",
            "user": "u",
            "password": "p",
        }
    )
    assert result is not None
    assert result["port"] == 5433
    assert isinstance(result["port"], int)


def test_config_dict_not_mutated() -> None:
    source = {
        "connection_string": "postgresql://u:p@h:5432/db",
        "extra": "keep-me",
    }
    snapshot = dict(source)
    normalize_postgres_connection_config(source)
    assert source == snapshot


def test_extra_keys_preserved() -> None:
    result = normalize_postgres_connection_config(
        {
            "connection_string": "postgresql://u:p@h/db",
            "sslmode": "require",
            "application_name": "dataknobs-test",
        }
    )
    assert result is not None
    assert result["sslmode"] == "require"
    assert result["application_name"] == "dataknobs-test"


def test_none_input_equivalent_to_empty_with_env_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "env-h")
    monkeypatch.setenv("POSTGRES_DB", "env-db")
    monkeypatch.setenv("POSTGRES_USER", "env-u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "env-p")
    result = normalize_postgres_connection_config(None)
    assert result is not None
    assert result["host"] == "env-h"
    assert result["database"] == "env-db"


def test_dotenv_loaded_as_env_fallback_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dotenv values fill ``POSTGRES_*`` gaps when os.environ is empty.

    Restores the behavior of the retired ``DotenvPostgresConnector`` —
    developers who keep postgres credentials in ``.env`` /
    ``.project_vars`` can still rely on them being picked up.
    """
    monkeypatch.setattr(
        "dataknobs_common.postgres_config._load_dotenv_fallbacks",
        lambda start_path=None: {
            "POSTGRES_HOST": "dotenv-h",
            "POSTGRES_DB": "dotenv-db",
            "POSTGRES_USER": "dotenv-u",
            "POSTGRES_PASSWORD": "dotenv-p",
        },
    )
    result = normalize_postgres_connection_config({})
    assert result is not None
    assert result["host"] == "dotenv-h"
    assert result["database"] == "dotenv-db"
    assert result["user"] == "dotenv-u"
    assert result["password"] == "dotenv-p"


def test_os_environ_overrides_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``os.environ`` wins over ``.env`` file values.

    The dotenv layer only fills gaps left by ``os.environ`` — it must
    not shadow a value the shell has already set.
    """
    monkeypatch.setenv("POSTGRES_HOST", "env-h")
    monkeypatch.setattr(
        "dataknobs_common.postgres_config._load_dotenv_fallbacks",
        lambda start_path=None: {
            "POSTGRES_HOST": "dotenv-h",
            "POSTGRES_DB": "dotenv-db",
        },
    )
    result = normalize_postgres_connection_config({})
    assert result is not None
    # os.environ wins over dotenv for overlapping keys
    assert result["host"] == "env-h"
    # dotenv fills the POSTGRES_DB gap that os.environ did not set
    assert result["database"] == "dotenv-db"


def test_load_dotenv_false_disables_dotenv_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``load_dotenv=False`` skips the .env fallback.

    Needed for tests that require strict env isolation even in the
    presence of workspace dotenv files.
    """
    # Even if dotenv files exist, the flag suppresses the layer.
    monkeypatch.setattr(
        "dataknobs_common.postgres_config._load_dotenv_fallbacks",
        lambda start_path=None: {"POSTGRES_HOST": "dotenv-h"},
    )
    with pytest.raises(ConfigurationError):
        normalize_postgres_connection_config({}, load_dotenv=False)


def test_partial_config_warns_on_defaulted_keys(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit partial config + defaults should log a warning.

    A caller passing only ``{"host": "foo"}`` gets defaults for
    user/database/password — and the normalizer names those fields
    in a WARNING so the defaulting does not slip through silently.
    """
    import logging

    with caplog.at_level(
        logging.WARNING, logger="dataknobs_common.postgres_config"
    ):
        result = normalize_postgres_connection_config({"host": "foo"})
    assert result is not None
    assert "user" in caplog.text
    assert "database" in caplog.text
    assert "password" in caplog.text


def test_host_with_at_sign_rejected() -> None:
    """``@`` in host would produce a malformed URI — reject it."""
    with pytest.raises(ValueError, match="host"):
        normalize_postgres_connection_config(
            {
                "host": "bad@host",
                "database": "db",
                "user": "u",
                "password": "p",
            }
        )


def test_database_with_slash_rejected() -> None:
    """``/`` in database name breaks URI parsing — reject it."""
    with pytest.raises(ValueError, match="database"):
        normalize_postgres_connection_config(
            {
                "host": "h",
                "database": "bad/db",
                "user": "u",
                "password": "p",
            }
        )
