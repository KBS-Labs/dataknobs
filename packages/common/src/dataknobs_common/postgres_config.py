"""Postgres connection-config normalization.

Single source of truth for dataknobs's understanding of how a postgres
connection is specified. Accepts every input shape the framework has
historically supported — ``connection_string``, ``DATABASE_URL`` env var,
individual ``host``/``port``/... keys, ``POSTGRES_*`` env-var
fallbacks, and values read from ``.env`` / ``.project_vars`` files —
and emits a canonical dict that downstream consumers read uniformly.

The function does not mutate its input and does not open any
connections. It is safe to call from sync or async contexts. When
``load_dotenv=True`` (default), it reads ``.env`` and ``.project_vars``
files walking up from the current working directory; those values are
used as an additional env fallback layer (they do NOT override
``os.environ``), preserving the behavior of the retired
``DotenvPostgresConnector``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from dataknobs_common.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

_INDIVIDUAL_KEYS = ("host", "port", "database", "user", "password")
_DOTENV_KEYS = (
    "DATABASE_URL",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)


def _load_dotenv_fallbacks(
    start_path: Path | None = None,
) -> dict[str, str]:
    """Read postgres-related values from ``.env`` and ``.project_vars``.

    Walks up from ``start_path`` (or cwd) to find the closest
    ``.project_vars`` and ``.env`` files. Values from ``.env`` take
    precedence over ``.project_vars`` — matching the historical
    ``load_project_vars`` semantics. Values from these files do NOT
    override ``os.environ``; callers should consult them only after
    the environment has been checked.

    Silently returns an empty dict if ``python-dotenv`` is not
    installed or no files are found, so the function stays
    dependency-optional.
    """
    try:
        from dotenv import dotenv_values  # type: ignore[import-not-found]
    except ImportError:
        return {}

    merged: dict[str, str] = {}
    base = start_path if start_path is not None else Path.cwd()

    # Order matters: .project_vars first, then .env overrides, matching
    # load_project_vars(include_dot_env=True) semantics.
    for filename in (".project_vars", ".env"):
        cur = base if base.is_dir() else base.parent
        while True:
            candidate = cur / filename
            if candidate.exists():
                try:
                    values = dotenv_values(candidate)
                except OSError as e:
                    logger.debug(
                        "Could not read %s: %s", candidate, e,
                    )
                    break
                for key, value in values.items():
                    if key in _DOTENV_KEYS and value is not None:
                        merged[key] = value
                break
            if cur.parent == cur:
                break
            cur = cur.parent
    return merged


def normalize_postgres_connection_config(
    config: dict[str, Any] | None = None,
    *,
    require: bool = True,
    load_dotenv: bool = True,
) -> dict[str, Any] | None:
    """Normalize a Postgres connection config into a canonical shape.

    Input shapes accepted (in precedence order — most specific first):

    1. Individual ``host``/``port``/``database``/``user``/``password``
       keys in ``config`` — each explicit per-field value overrides
       any parallel value from a ``connection_string`` or from env.
       This preserves the historical "use this URL, but override the
       database" pattern.
    2. ``config["connection_string"]`` — supplies the base values for
       any canonical key not set explicitly at level 1.
    3. ``DATABASE_URL`` env var — used only when no ``connection_string``
       is in config and no individual keys in config provide any value.
    4. ``POSTGRES_HOST`` / ``POSTGRES_PORT`` / ``POSTGRES_DB`` /
       ``POSTGRES_USER`` / ``POSTGRES_PASSWORD`` env vars — used to
       fill missing individual-key gaps (or as the sole source when
       no config individual keys are present and ``DATABASE_URL`` is
       absent).
    5. Values from ``.env`` / ``.project_vars`` files walking up from
       the current working directory — only filled in when neither
       ``os.environ`` nor explicit config has already resolved them.
       Requires ``python-dotenv`` to be installed; absent that package
       this layer is silently skipped. Disable with ``load_dotenv=False``
       (useful in tests that need strict env isolation).

    Key design principle: **explicit config always beats env, and
    individual keys always beat the same field from a
    ``connection_string``.** A caller passing ``{"host": "myhost"}``
    expects ``"myhost"`` — ``DATABASE_URL`` in the shell must not
    silently override it. A caller passing
    ``{"connection_string": "postgresql://u:p@host:5432/db",
    "database": "override"}`` expects the final connection to target
    ``"override"``, not ``"db"``.

    After resolving the canonical keys, the output
    ``connection_string`` is either the original URL (when the caller
    supplied one and nothing overrode it) or a freshly synthesized
    URL built from the merged canonical keys. Either way the two
    forms — the URL and the individual keys — always agree on every
    field, so downstream consumers can read whichever they prefer.

    The ``postgresql+asyncpg://`` dialect prefix is stripped to
    ``postgresql://`` so asyncpg, psycopg2, and sqlalchemy-style URIs
    are all accepted.

    When synthesizing a ``connection_string`` from individual keys,
    ``host`` and ``database`` are validated for shell-unsafe
    characters (``@``, ``/``, whitespace) that could produce a
    malformed or misrouted URI; ``user`` and ``password`` are
    URL-encoded so values containing those characters (common in
    secrets-manager output) produce a valid URI.

    If the caller supplies a partial individual-keys config and we
    have to synthesize missing values from defaults (e.g. ``user`` or
    ``database`` fell through to ``"postgres"``), a WARNING is logged
    naming the defaulted fields so misconfiguration does not slip
    through silently.

    Args:
        config: User config dict (may contain extra backend-specific
            keys, which are preserved verbatim). ``None`` is treated
            as an empty dict.
        require: When ``True`` (default), raise ``ConfigurationError``
            if no postgres connection can be resolved. When ``False``,
            return ``None`` instead — lets optional-backend consumers
            treat "no postgres configured" as a soft signal.
        load_dotenv: When ``True`` (default), read ``.env`` /
            ``.project_vars`` files as an additional env fallback
            layer (see input shape #5 above).

    Returns:
        A new dict containing every key from the input plus canonical
        ``connection_string``, ``host``, ``port`` (int), ``database``,
        ``user``, ``password``. Returns ``None`` only when
        ``require=False`` and no connection is resolvable.

    Raises:
        ConfigurationError: If ``require=True`` and no connection can
            be resolved from any of the accepted input shapes.
        ValueError: If ``host`` or ``database`` contain shell-unsafe
            characters (``@``, ``/``, whitespace).
    """
    out: dict[str, Any] = dict(config or {})

    conn_str_from_config = out.get("connection_string")

    # Detect explicit individual keys in config — they win over any
    # parallel value from a connection_string (in config or env).
    explicit_individual = {
        k: out[k] for k in _INDIVIDUAL_KEYS if out.get(k) is not None
    }
    has_explicit_individual = bool(explicit_individual)

    # Load dotenv fallbacks lazily — only if we actually need them
    # (i.e., env + config didn't already resolve the connection).
    dotenv_fallbacks: dict[str, str] | None = None

    def _dotenv() -> dict[str, str]:
        nonlocal dotenv_fallbacks
        if dotenv_fallbacks is None:
            dotenv_fallbacks = (
                _load_dotenv_fallbacks() if load_dotenv else {}
            )
        return dotenv_fallbacks

    def _env_or_dotenv(env_key: str) -> str | None:
        """Read an env key, falling back to .env/.project_vars."""
        value = os.environ.get(env_key)
        if value is not None:
            return value
        return _dotenv().get(env_key)

    # Step 1: start with whatever base values the config's
    # connection_string provides.
    base_keys: dict[str, Any] = {}
    if conn_str_from_config:
        base_keys = _parse_connection_string(conn_str_from_config)
    else:
        # Step 2: fall back to DATABASE_URL env only when there is no
        # connection_string and no explicit individual keys in config
        # (individual keys in config suppress DATABASE_URL so the
        # shell env cannot silently override caller intent).
        db_url = _env_or_dotenv("DATABASE_URL")
        if db_url and not has_explicit_individual:
            base_keys = _parse_connection_string(db_url)
            # Preserve the original DATABASE_URL in out for callers
            # who inspect raw values before synthesis.
            conn_str_from_config = db_url

    # Step 3: env fallbacks (POSTGRES_*) — only consulted for keys
    # that aren't already resolved by config or connection_string.
    env_fallbacks = {
        "host": _env_or_dotenv("POSTGRES_HOST"),
        "port": _env_or_dotenv("POSTGRES_PORT"),
        "database": _env_or_dotenv("POSTGRES_DB"),
        "user": _env_or_dotenv("POSTGRES_USER"),
        "password": _env_or_dotenv("POSTGRES_PASSWORD"),
    }

    have_any = (
        bool(base_keys)
        or has_explicit_individual
        or any(v is not None for v in env_fallbacks.values())
    )
    if not have_any:
        if require:
            raise ConfigurationError(
                "Postgres connection requires one of: "
                "'connection_string' (or DATABASE_URL env), "
                "individual host/port/database/user/password keys, "
                "or POSTGRES_HOST/POSTGRES_PORT/POSTGRES_DB/"
                "POSTGRES_USER/POSTGRES_PASSWORD env vars."
            )
        return None

    # Track which fields fell through to defaults so we can warn.
    defaulted: list[str] = []

    def _resolve(key: str, default: Any) -> Any:
        """Resolve a canonical key using the full precedence ladder."""
        # 1. Explicit individual key in config always wins.
        if key in explicit_individual:
            return explicit_individual[key]
        # 2. Base keys from a connection_string (config or DATABASE_URL).
        if key in base_keys and base_keys[key] is not None:
            return base_keys[key]
        # 3. POSTGRES_* env fallback.
        env_value = env_fallbacks.get(key)
        if env_value is not None:
            return env_value
        # 4. Built-in default (tracked so we can warn).
        defaulted.append(key)
        return default

    host = _resolve("host", "localhost")
    port = int(_resolve("port", 5432))
    database = _resolve("database", "postgres")
    user = _resolve("user", "postgres")
    # Password has a special rule: an empty string is a valid explicit
    # choice (distinct from "unset"), so we mirror _resolve manually
    # to preserve that signal.
    if "password" in explicit_individual:
        password = explicit_individual["password"]
    elif "password" in base_keys and base_keys["password"] is not None:
        password = base_keys["password"]
    elif env_fallbacks["password"] is not None:
        password = env_fallbacks["password"]
    else:
        defaulted.append("password")
        password = ""

    # Validate host/database — URL-encoding them would distort values
    # that actually need to parse cleanly as URI components. Reject
    # shell-unsafe characters instead so misconfiguration surfaces
    # loudly rather than producing a malformed URI.
    _validate_url_component("host", str(host))
    _validate_url_component("database", str(database))

    # Warn only when the caller mixed explicit config with defaults —
    # pure env-driven or fully-implicit configs are a separate
    # "nothing configured" signal handled by the ``require`` check.
    if defaulted and (has_explicit_individual or bool(base_keys)):
        logger.warning(
            "Postgres connection synthesized default values for "
            "%s — verify this is intended. Explicit config was "
            "provided for other fields; these fields fell through "
            "to the built-in defaults.",
            ", ".join(sorted(set(defaulted))),
        )

    out["host"] = host
    out["port"] = port
    out["database"] = database
    out["user"] = user
    out["password"] = password

    # Preserve the original ``connection_string`` (with the
    # ``postgresql+asyncpg://`` dialect prefix stripped) when the
    # caller supplied one and no individual key overrode it. This
    # keeps callers who rely on the exact URL shape (dialect-specific
    # query params, custom ordering, omitted default ports) seeing
    # what they passed in. Canonical individual keys (``host``,
    # ``port``, etc.) are still populated from the URL plus defaults
    # so consumers reading those keys get the same resolved values.
    # We only re-synthesize when an individual key in config or an
    # env fallback contributed something the URL did not already
    # carry — otherwise the original URL is authoritative.
    needs_synthesis = has_explicit_individual or (
        not conn_str_from_config
    )
    # Env fallbacks also force synthesis — if POSTGRES_* contributed
    # a value the URL did not have, we must record that in the
    # canonical connection_string.
    if not needs_synthesis:
        env_contributed = any(
            env_fallbacks[k] is not None
            and (base_keys.get(k) is None)
            for k in _INDIVIDUAL_KEYS
        )
        needs_synthesis = env_contributed
    if not needs_synthesis and conn_str_from_config is not None:
        preserved = conn_str_from_config
        if preserved.startswith("postgresql+asyncpg://"):
            preserved = preserved.replace(
                "postgresql+asyncpg://", "postgresql://", 1,
            )
        out["connection_string"] = preserved
    else:
        userinfo = (
            f"{quote(str(user), safe='')}:{quote(str(password), safe='')}@"
        )
        out["connection_string"] = (
            f"postgresql://{userinfo}{host}:{port}/{database}"
        )
    return out


def _validate_url_component(field: str, value: str) -> None:
    """Reject shell-unsafe characters in a URI host or database name.

    URL-encoding ``host``/``database`` would mangle legitimate values;
    asyncpg/psycopg2 expect them in bare form. So we instead reject
    characters that would produce a malformed or misrouted URI if
    interpolated directly: ``@`` (userinfo delimiter), ``/`` (path
    separator), and whitespace.
    """
    if not value:
        raise ValueError(f"Postgres {field} must be non-empty")
    for bad in ("@", "/", " ", "\t", "\n", "\r"):
        if bad in value:
            raise ValueError(
                f"Postgres {field}={value!r} contains a disallowed "
                f"character {bad!r}. Use ``connection_string`` for "
                f"values that require URL-escaping."
            )


def _parse_connection_string(conn_str: str) -> dict[str, Any]:
    """Parse a postgres URI into canonical keys.

    Pure: does not mutate anything. Returns a dict with the canonical
    keys populated from the URI; keys absent from the URI map to
    ``None`` (except ``host``/``port`` which have URL-level defaults).
    """
    if conn_str.startswith("postgresql+asyncpg://"):
        conn_str = conn_str.replace(
            "postgresql+asyncpg://", "postgresql://", 1
        )
    parsed = urlparse(conn_str)
    database: str | None = None
    if parsed.path and len(parsed.path) > 1:
        database = parsed.path[1:]
    return {
        "host": parsed.hostname,
        "port": parsed.port,
        "database": database,
        "user": parsed.username,
        "password": parsed.password,
    }
