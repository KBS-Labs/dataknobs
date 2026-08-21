"""A config key the backend does not accept must be reported, not discarded.

``StructuredConfig.from_dict`` projects a dict onto the declared fields and
drops whatever is left over, so a misspelled key and an absent key produced
the same object. The failure that makes this worth a guard is not the typo
itself but where it lands: a Postgres config carrying ``hosst`` connects to
``localhost`` and reports nothing, because every field it meant to set fell
through to a built-in default.

The existing "synthesized default values" WARNING cannot cover it. That
warning fires when a caller mixes *recognized* explicit keys with defaults;
an unrecognized key enters neither bucket, so a config made entirely of
misspelled keys reads as "nothing was configured" — the one case most in
need of the warning is the one case it structurally cannot see.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_common.structured_config import _ROUTING_KEYS, StructuredConfig
from dataknobs_data import AsyncDatabase, SyncDatabase
from dataknobs_data.backends import async_backends, sync_backends
from dataknobs_data.factory import async_database_factory, database_factory


#: Enough config to build each backend, so a rejection in these tests is
#: caused by the key under test rather than by an absent required one.
#:
#: The file-backed entries take their path from ``tmp_path`` rather than a
#: fixed name: a literal under ``/tmp`` collides when the suite is run in
#: parallel, and outlives the session either way.
MINIMAL: dict[str, dict[str, object]] = {
    "memory": {},
    "file": {"path": "{tmp}/store.json"},
    "sqlite": {"path": "{tmp}/store.db"},
    "duckdb": {"path": "{tmp}/store.duckdb"},
    "postgres": {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p"},
    "s3": {"bucket": "b"},
    "elasticsearch": {"host": "localhost", "port": 9200, "index": "i"},
}


def _minimal(backend: str, tmp_path: Path) -> dict[str, object]:
    """``MINIMAL[backend]`` with its paths bound, or a skip.

    ``MINIMAL`` is a fixed literal while ``_registered`` reads the live
    registry, so a consumer that registers a backend of its own would
    otherwise turn this suite into a ``KeyError`` rather than a skip.
    """
    if backend not in MINIMAL:
        pytest.skip(f"no minimal config recorded for the {backend!r} backend")
    return {
        key: value.format(tmp=tmp_path) if isinstance(value, str) and "{tmp}" in value else value
        for key, value in MINIMAL[backend].items()
    }


def _registered(registry: Any) -> list[str]:
    """Backend names actually creatable here, so an absent driver skips."""
    return sorted(registry.list_canonical_keys())


def test_a_misspelled_host_is_rejected_rather_than_silently_defaulted() -> None:
    """The measured failure: ``hosst`` used to yield ``host='localhost'``."""
    with pytest.raises(ValueError) as excinfo:
        database_factory.create(
            backend="postgres",
            hosst="typohost",
            database="d",
            user="u",
            password="p",
        )

    message = str(excinfo.value)
    assert "hosst" in message, f"the error must name the offending key: {message}"
    assert "host" in message, f"the error must offer the accepted spelling: {message}"


def test_the_legacy_connection_key_is_rejected() -> None:
    """``connection`` is not the key; ``connection_string`` is.

    Worth its own case because this spelling appears in configs that look
    entirely reasonable, and dropping it silently produces a database
    pointed at ``localhost/postgres`` rather than a failure.
    """
    with pytest.raises(ValueError) as excinfo:
        database_factory.create(
            backend="postgres",
            connection="postgresql://u:p@db.internal:5432/appdb",
        )

    message = str(excinfo.value)
    assert "connection" in message
    assert "connection_string" in message, f"name the real key: {message}"


@pytest.mark.parametrize("backend", _registered(sync_backends))
def test_every_sync_backend_rejects_an_unknown_key(backend: str, tmp_path: Path) -> None:
    """The guard belongs to the shared base, so no backend is exempt."""
    config = _minimal(backend, tmp_path)
    config["definitely_not_a_key"] = "x"

    with pytest.raises(ValueError, match="definitely_not_a_key"):
        database_factory.create(backend=backend, **config)


@pytest.mark.parametrize("backend", _registered(async_backends))
def test_every_async_backend_rejects_an_unknown_key(backend: str, tmp_path: Path) -> None:
    """The async factory reaches the same config classes by the same route."""
    config = _minimal(backend, tmp_path)
    config["definitely_not_a_key"] = "x"

    with pytest.raises(ValueError, match="definitely_not_a_key"):
        async_database_factory.create(backend=backend, **config)


def test_from_backend_rejects_an_unknown_key() -> None:
    """The other public construction entry, which does not go through a factory."""
    with pytest.raises(ValueError, match="definitely_not_a_key"):
        SyncDatabase.from_backend("memory", {"definitely_not_a_key": "x"})


async def test_async_from_backend_rejects_an_unknown_key() -> None:
    """Same entry on the async side, which also connects on success."""
    with pytest.raises(ValueError, match="definitely_not_a_key"):
        await AsyncDatabase.from_backend("memory", {"definitely_not_a_key": "x"})


@pytest.mark.parametrize("backend", _registered(sync_backends))
def test_a_legitimate_config_still_builds(backend: str, tmp_path: Path) -> None:
    """The guard must not reject the configs it is meant to protect."""
    db = database_factory.create(backend=backend, **_minimal(backend, tmp_path))
    assert db is not None


def test_the_backend_key_itself_is_not_reported_as_unknown() -> None:
    """``backend`` is the factory's discriminator, stripped before the config."""
    assert database_factory.create(backend="memory") is not None


# --- The alias declarations must not drift from the normalizers ----------
#
# ``_INPUT_KEYS`` widens what the guard accepts and what its error offers.
# A declaration that outlives the normalizing code it describes therefore
# fails open: the key is accepted, the normalizer no longer maps it, and the
# value is discarded again -- the exact behaviour the guard exists to stop,
# now with a declaration asserting it is fine.


def _declared_aliases(config_cls: type) -> list[str]:
    """Every ``_INPUT_KEYS`` entry reachable from ``config_cls``."""
    declared: set[str] = set()
    for base in config_cls.__mro__:
        declared |= set(base.__dict__.get("_INPUT_KEYS", ()))
    return sorted(declared)


def _backend_config_classes() -> list[type[StructuredConfig]]:
    """The config class of every registered backend, deduplicated.

    ``get_factory`` returns ``None`` for a backend whose driver is absent
    rather than raising, so the declared type is the fallback -- reading a
    schema off a plugin that cannot be built is what it is for. Without it
    this helper raised ``AttributeError`` in any environment thinner than
    a full dev install, and dropped that backend's coverage entirely.
    """
    classes: dict[str, type[StructuredConfig]] = {}
    for registry in (sync_backends, async_backends):
        for name in registry.list_canonical_keys():
            config_cls = getattr(registry.get_factory(name), "CONFIG_CLS", None)
            if config_cls is None:
                config_cls = getattr(registry.load_declared_type(name), "CONFIG_CLS", None)
            if isinstance(config_cls, type) and issubclass(config_cls, StructuredConfig):
                classes[config_cls.__name__] = config_cls
    return [classes[name] for name in sorted(classes)]


#: Values plausible enough for the normalizers that parse rather than copy.
ALIAS_PROBES: dict[str, object] = {
    "connection_string": "postgresql://u:p@probehost:5432/probedb",
    "max_retries": 7,
    "max_workers": 3,
}


@pytest.mark.parametrize(
    ("config_cls", "alias"),
    [
        (config_cls, alias)
        for config_cls in _backend_config_classes()
        for alias in _declared_aliases(config_cls)
    ],
    ids=lambda p: p.__name__ if isinstance(p, type) else str(p),
)
def test_every_declared_alias_is_consumed_by_the_normalizer(
    config_cls: type[StructuredConfig], alias: str
) -> None:
    """A declared alias must not survive normalization as itself."""
    probe = ALIAS_PROBES.get(alias, "probe-value")
    normalized = config_cls._normalize_dict({alias: probe})

    assert alias not in normalized, (
        f"{config_cls.__name__} declares {alias!r} in _INPUT_KEYS but its "
        "_normalize_dict leaves the key in place, so the value is discarded "
        "by field projection while the declaration says it is accepted"
    )


#: ``(config class name, alias, canonical field, value, expected)``. Explicit
#: rather than derived: the alias-to-field mapping is the contract itself, and
#: deriving it from the code under test would assert only self-consistency.
ALIAS_MAPPINGS = [
    ("PostgresDatabaseConfig", "table_name", "table", "bank_notes", "bank_notes"),
    (
        "PostgresDatabaseConfig",
        "connection_string",
        "host",
        ALIAS_PROBES["connection_string"],
        "probehost",
    ),
    ("SyncS3DatabaseConfig", "region", "region_name", "eu-west-1", "eu-west-1"),
    ("SyncS3DatabaseConfig", "access_key_id", "aws_access_key_id", "AKIA", "AKIA"),
    ("SyncS3DatabaseConfig", "secret_access_key", "aws_secret_access_key", "s3cr3t", "s3cr3t"),
    ("SyncS3DatabaseConfig", "session_token", "aws_session_token", "tok", "tok"),
    ("SyncS3DatabaseConfig", "max_workers", "max_pool_connections", 3, 3),
    ("SyncS3DatabaseConfig", "max_retries", "max_attempts", 7, 7),
    ("AsyncS3DatabaseConfig", "region", "region_name", "eu-west-1", "eu-west-1"),
]

#: Keys each class needs before it will construct at all.
REQUIRED: dict[str, dict[str, object]] = {
    "SyncS3DatabaseConfig": {"bucket": "b"},
    "AsyncS3DatabaseConfig": {"bucket": "b"},
}


@pytest.mark.parametrize(
    ("class_name", "alias", "field_name", "value", "expected"),
    ALIAS_MAPPINGS,
    ids=[f"{c}.{a}" for c, a, *_ in ALIAS_MAPPINGS],
)
def test_a_declared_alias_reaches_its_field(
    class_name: str, alias: str, field_name: str, value: object, expected: object
) -> None:
    """Consumed is necessary but not sufficient -- it must also land."""
    config_cls = {c.__name__: c for c in _backend_config_classes()}[class_name]
    config = config_cls.from_dict({**REQUIRED.get(class_name, {}), alias: value})

    assert getattr(config, field_name) == expected


@pytest.mark.parametrize("config_cls", _backend_config_classes(), ids=lambda c: c.__name__)
def test_every_backend_config_opts_into_the_guard(config_cls: type[StructuredConfig]) -> None:
    """Inherited from ``DatabaseConfig``, so a new backend gets it for free."""
    assert config_cls._UNKNOWN_KEYS == "raise"


def _advertised(registry: Any, name: str) -> tuple[type[StructuredConfig] | None, set[str]]:
    """The backend's config class and the keys its registry entry advertises."""
    factory = registry.get_factory(name)
    cls = getattr(factory, "CONFIG_CLS", None)
    options = set((registry.get_metadata(name) or {}).get("config_options", {}))
    return (cls if isinstance(cls, type) and issubclass(cls, StructuredConfig) else None), options


@pytest.mark.parametrize(
    ("flavor", "backend"),
    [
        (f, b)
        for f, r in (("sync", sync_backends), ("async", async_backends))
        for b in _registered(r)
    ],
)
def test_the_registry_advertises_only_keys_the_backend_accepts(flavor: str, backend: str) -> None:
    """``config_options`` is documentation, and it is checkable like any other.

    ``DatabaseFactory.get_backend_info()`` returns this metadata, so it is
    the *programmatic* equivalent of the markdown samples the sibling guard
    in ``tests/test_documented_backend_config_keys.py`` checks -- read by a
    consumer building a config form or validating input, and until now read
    by nothing that could tell it was wrong.

    It carried the same three defects that guard was written for: a field
    belonging to the sibling backend (``hosts``, advertised on the *sync*
    Elasticsearch entry), another library's vocabulary (``username`` /
    ``password`` where the field is ``basic_auth``), and a field that never
    existed (``initial_data`` on the memory backend).
    """
    registry = sync_backends if flavor == "sync" else async_backends
    config_cls, advertised = _advertised(registry, backend)
    if config_cls is None:
        pytest.skip(f"{flavor}/{backend} registers no CONFIG_CLS")
    rejected = sorted(k for k in advertised - _ROUTING_KEYS if not config_cls.accepts(k))
    assert not rejected, (
        f"{flavor}/{backend} advertises {rejected} in config_options, but "
        f"{config_cls.__name__}.from_dict would raise on them"
    )
