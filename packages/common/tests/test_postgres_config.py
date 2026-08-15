"""Tests for dataknobs_common.postgres_config."""

from __future__ import annotations

from urllib.parse import unquote, urlparse

import pytest

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.postgres_config import (
    build_postgres_dsn,
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
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@env-host:5999/env-db")
    result = normalize_postgres_connection_config({})
    assert result is not None
    assert result["host"] == "env-host"
    assert result["port"] == 5999
    assert result["database"] == "env-db"
    assert result["connection_string"] == "postgresql://u:p@env-host:5999/env-db"


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
    assert result["connection_string"] == "postgresql://override-user:p@url-host:5433/override-db"


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
    assert result["connection_string"] == "postgresql://u:p@h:5433/db"
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

    with caplog.at_level(logging.WARNING, logger="dataknobs_common.postgres_config"):
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


# -- percent-decoding on the way in -----------------------------------------
#
# ``urlparse`` decodes nothing: ``.username``/``.password`` come back
# exactly as they appeared in the URI, still percent-encoded. So the
# canonical dict is only uniform — raw values whichever input shape
# produced them — if the parser decodes. Everything downstream depends
# on that: ``build_postgres_dsn`` encodes what it is given, and the
# backends' ``_create_database`` passes the same field to
# ``psycopg2.connect``/``asyncpg.connect`` as a kwarg, which wants it
# raw. A field that is sometimes encoded breaks one of the two.


def test_dsn_userinfo_is_decoded_into_the_canonical_keys() -> None:
    """A percent-encoded DSN must yield raw ``user``/``password``.

    ``postgresql://svc:p%40ss@h/db`` is the *only* correct way to write
    the password ``p@ss`` as a URI, so this is the ordinary shape of a
    working config — not an edge case.
    """
    result = normalize_postgres_connection_config(
        {"connection_string": "postgresql://sv%40c:p%40ss%2Fw0rd@db.internal:5432/prod"}
    )

    assert result is not None
    assert result["user"] == "sv@c"
    assert result["password"] == "p@ss/w0rd"


def test_dsn_password_survives_a_decode_and_rebuild_round_trip() -> None:
    """Parsing a DSN then re-synthesizing one must not change the password.

    The rebuild is forced by an individual key the URI did not carry, so
    the normalizer re-synthesizes rather than preserving the original
    string. Decode-then-encode is the identity; decode-less-then-encode
    doubles the escaping and the credential silently changes.
    """
    result = normalize_postgres_connection_config(
        {
            "connection_string": "postgresql://svc:p%40ss@db.internal:5432/prod",
            "port": 5433,
        }
    )

    assert result is not None
    parsed = urlparse(result["connection_string"])
    assert unquote(parsed.password or "") == "p@ss"
    assert parsed.hostname == "db.internal"
    assert parsed.port == 5433


def test_dsn_database_encoding_cannot_smuggle_a_path_separator() -> None:
    """``%2F`` in the database must be rejected, not passed through.

    The validator rejects a literal ``/`` in a database name. Left
    encoded, ``db%2Fetc`` walks straight past that check and the driver
    unquotes it back to ``db/etc`` at connect time — the rejection
    defeated by spelling.
    """
    with pytest.raises(ValueError, match="database"):
        normalize_postgres_connection_config(
            {"connection_string": "postgresql://u:p@h:5432/db%2Fetc"}
        )


# -- build_postgres_dsn -----------------------------------------------------
#
# The synthesis step above, reachable on its own. The tests above assert
# encoding by substring; these assert it by parsing the result back,
# which is the property that actually matters — a DSN can contain the
# expected substring and still resolve to the wrong server.


def test_build_dsn_round_trips_every_component() -> None:
    """Each field must survive a parse back out of the built URI."""
    dsn = build_postgres_dsn(
        host="db.internal",
        port=5432,
        database="prod",
        user="svc",
        password="p@ss/w0rd",
    )

    parsed = urlparse(dsn)

    assert parsed.scheme == "postgresql"
    assert parsed.hostname == "db.internal"
    assert parsed.port == 5432
    assert parsed.path == "/prod"
    assert unquote(parsed.username or "") == "svc"
    assert unquote(parsed.password or "") == "p@ss/w0rd"


def test_build_dsn_would_misroute_if_interpolated_raw() -> None:
    """Pin why the encoding exists, not merely that it happens.

    The naive f-string this function replaces does not yield a URI that
    fails to parse — it yields one that parses cleanly against the wrong
    authority, because the last ``@`` delimits userinfo. A caller gets a
    working connection to a host it never configured.
    """
    fields = {
        "host": "db.internal",
        "port": 5432,
        "database": "prod",
        "user": "svc",
        "password": "p@ss/w0rd",
    }
    naive = (
        f"postgresql://{fields['user']}:{fields['password']}"
        f"@{fields['host']}:{fields['port']}/{fields['database']}"
    )

    assert urlparse(naive).hostname == "ss"
    assert urlparse(build_postgres_dsn(**fields)).hostname == "db.internal"  # type: ignore[arg-type]


def test_build_dsn_resolves_nothing_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder takes what it is given; only the normalizer resolves.

    This is the whole distinction between the two entry points, and a
    caller picking the wrong one would otherwise find out through a
    connection aimed at whatever ``POSTGRES_*`` happened to be exported.
    """
    monkeypatch.setenv("POSTGRES_HOST", "env-host")
    monkeypatch.setenv("POSTGRES_DB", "env-db")
    monkeypatch.setenv("DATABASE_URL", "postgresql://env:env@env-host:5432/env-db")

    dsn = build_postgres_dsn(
        host="explicit-host",
        port=5432,
        database="explicit-db",
        user="u",
        password="p",
    )

    assert dsn == "postgresql://u:p@explicit-host:5432/explicit-db"


def test_build_dsn_empty_password_emits_an_empty_userinfo_field() -> None:
    """An empty password must not be dropped from the URI's shape.

    Named for what it checks: the *string* keeps the ``user:@host`` form
    rather than collapsing to ``user@host``. It does not claim the
    connection authenticates with an empty password — asyncpg reads an
    empty DSN password as absent and falls through to ``PGPASSWORD`` /
    ``.pgpass``, which the builder's docstring records.
    """
    dsn = build_postgres_dsn(host="h", port=5432, database="db", user="u", password="")

    assert dsn == "postgresql://u:@h:5432/db"
    assert urlparse(dsn).hostname == "h"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host", "bad@host"),
        ("host", "bad/host"),
        ("host", ""),
        ("database", "bad/db"),
        ("database", "bad@db"),
        ("database", ""),
        # ``?`` and ``#`` end the path just as surely as ``/`` splits it.
        ("database", "tenant?sslmode=disable"),
        ("database", "tenant#frag"),
        ("host", "h?host=elsewhere"),
        ("host", "h#frag"),
        # ``%`` would be read back as the start of an escape sequence,
        # which is ambiguous in a component this builder does not encode.
        ("database", "db%2Fetc"),
        ("host", "h%2Fx"),
        # Whitespace beyond the four ASCII ones originally listed. Written
        # as escapes, not literals: a raw NO-BREAK SPACE in the source is
        # invisible to a reader and indistinguishable from the ordinary
        # space one line up, which is the same ambiguity ``RUF001`` exists
        # to catch.
        ("database", "db\vetc"),
        ("host", "h\x00x"),
        ("database", "db\xa0etc"),
    ],
)
def test_build_dsn_rejects_unencodable_components(field: str, value: str) -> None:
    """Validation must reach the direct entry point, not only the normalizer.

    ``host`` and ``database`` are validated rather than encoded, so the
    check is the only thing standing between a bad value and a malformed
    URI — and callers reaching this function skip the normalizer that
    would otherwise have run it.
    """
    fields: dict[str, object] = {
        "host": "h",
        "port": 5432,
        "database": "db",
        "user": "u",
        "password": "p",
    }
    fields[field] = value

    with pytest.raises(ValueError, match=field):
        build_postgres_dsn(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "port",
    [
        "5432/otherdb?sslmode=disable",
        "not-a-port",
        "",
        -1,
        0,
        65536,
    ],
)
def test_build_dsn_rejects_a_port_that_is_not_a_port(port: object) -> None:
    """``port`` is interpolated, so it is an injection point like any other.

    Unvalidated, ``5432/otherdb?sslmode=disable`` yields a URI whose path
    is ``/otherdb`` and whose query disables TLS — the requested database
    discarded into the query string, on a connection no longer encrypted.
    """
    with pytest.raises(ValueError, match="port"):
        build_postgres_dsn(
            host="h",
            port=port,  # type: ignore[arg-type]
            database="db",
            user="u",
            password="p",
        )


def test_build_dsn_accepts_a_port_given_as_a_string() -> None:
    """``POSTGRES_PORT`` arrives as a string; coercion is the contract."""
    assert (
        build_postgres_dsn(host="h", port="5433", database="db", user="u", password="p")
        == "postgresql://u:p@h:5433/db"
    )


def test_build_dsn_brackets_an_ipv6_host() -> None:
    """A bare IPv6 literal makes the authority unreadable — bracket it.

    Unbracketed, ``::1`` gives ``postgresql://u:p@::1:5432/db``, whose
    ``hostname`` parses as ``None`` and whose ``.port`` raises. IPv6 is a
    legitimate host, so the fix is to bracket rather than to reject.
    """
    dsn = build_postgres_dsn(host="::1", port=5432, database="db", user="u", password="p")

    parsed = urlparse(dsn)

    assert dsn == "postgresql://u:p@[::1]:5432/db"
    assert parsed.hostname == "::1"
    assert parsed.port == 5432


def test_build_dsn_rejects_a_host_that_only_looks_like_ipv6() -> None:
    """``:`` in a host is an IPv6 literal or it is a smuggled port."""
    with pytest.raises(ValueError, match="host"):
        build_postgres_dsn(
            host="evil.com:9999",
            port=5432,
            database="db",
            user="u",
            password="p",
        )
