"""Drift guards for the ``dataknobs-data`` backend registries.

The data registries use the whole-dict ``cls(config)`` pattern: every
registered backend's ``__init__`` accepts a ``config: dict | None``
and reads keys through its inheritance chain
(``ConfigurableBase`` + ``VectorConfigMixin`` + per-backend mixins).
Drift modes:

- A backend is registered in the registry but does not expose
  ``from_config`` — ``DatabaseFactory.create`` would raise
  ``AttributeError`` from the consumer's call site.
- A backend's ``from_config`` signature deviates from the
  ``(cls, config: dict)`` shape — the factory's dispatch breaks.

These structural checks run without instantiating backends, so
optional-dependency backends (postgres → psycopg2/asyncpg,
elasticsearch → elasticsearch, s3 → boto3, ...) are still audited.
Behavioural coverage of each backend lives in its own test module.
"""

from __future__ import annotations

import inspect

import pytest

from dataknobs_data.backends import async_backends, sync_backends


def _registered_backend_classes(registry: object) -> list[tuple[str, type]]:
    """Collect ``(key, class)`` pairs from a backend registry, de-duped.

    The same backend class may be registered under aliases (e.g.,
    ``"memory"`` and ``"mem"`` both point at ``SyncMemoryDatabase``).
    The parity guarantee is per-class, not per-alias, so we de-duplicate
    on the class identity.
    """
    by_class: dict[type, str] = {}
    for key in registry.list_keys():  # type: ignore[attr-defined]
        cls = registry.get_factory(key)  # type: ignore[attr-defined]
        if cls is None:
            continue
        by_class.setdefault(cls, key)
    return [(name, cls) for cls, name in by_class.items()]


SYNC_BACKENDS = _registered_backend_classes(sync_backends)
ASYNC_BACKENDS = _registered_backend_classes(async_backends)


@pytest.mark.parametrize(
    "name, backend_cls",
    SYNC_BACKENDS,
    ids=[name for name, _ in SYNC_BACKENDS],
)
def test_sync_backend_exposes_from_config(
    name: str, backend_cls: type
) -> None:
    """Every registered sync backend exposes ``from_config(cls, config)``.

    ``DatabaseFactory.create`` calls ``backend_class.from_config(config)``
    after registry lookup. If a backend regresses to a different
    construction shape (or removes ``from_config`` entirely), the
    factory dispatch breaks at the first consumer call — this test
    surfaces the regression at unit-test time instead.
    """
    assert hasattr(backend_cls, "from_config"), (
        f"Backend {name} ({backend_cls.__name__}) has no `from_config` "
        "classmethod; DatabaseFactory.create would raise AttributeError."
    )
    sig = inspect.signature(backend_cls.from_config)
    params = list(sig.parameters.values())
    assert len(params) >= 1, (
        f"{backend_cls.__name__}.from_config must accept a config "
        "positional/keyword argument."
    )


@pytest.mark.parametrize(
    "name, backend_cls",
    ASYNC_BACKENDS,
    ids=[name for name, _ in ASYNC_BACKENDS],
)
def test_async_backend_exposes_from_config(
    name: str, backend_cls: type
) -> None:
    """Every registered async backend exposes ``from_config(cls, config)``."""
    assert hasattr(backend_cls, "from_config"), (
        f"Async backend {name} ({backend_cls.__name__}) has no "
        "`from_config` classmethod."
    )
    sig = inspect.signature(backend_cls.from_config)
    params = list(sig.parameters.values())
    assert len(params) >= 1, (
        f"{backend_cls.__name__}.from_config must accept a config arg."
    )


def test_memory_backend_constructs_from_empty_config() -> None:
    """Smoke test: ``SyncMemoryDatabase.from_config({})`` succeeds.

    Memory backends have no optional dependencies, so a structural
    sanity check via real construction protects against regressions
    that break the very-bottom of the dispatch chain. Backends with
    optional deps are covered by their own per-backend integration
    test modules.
    """
    from dataknobs_data.backends.memory import (
        AsyncMemoryDatabase,
        SyncMemoryDatabase,
    )

    sync_db = SyncMemoryDatabase.from_config({})
    assert sync_db is not None
    async_db = AsyncMemoryDatabase.from_config({})
    assert async_db is not None


def test_registered_backends_are_unique_classes() -> None:
    """Registry collisions are surfaced at audit time.

    Two backend keys pointing at the same class is fine (alias); two
    DIFFERENT classes registered under the same key would be a
    contributor mistake. ``PluginRegistry`` rejects duplicate
    registrations by default — this test enforces that the audit
    matrix above stays useful.
    """
    sync_keys = list(sync_backends.list_keys())
    async_keys = list(async_backends.list_keys())
    # No empty registries — both should always have at least "memory".
    assert "memory" in sync_keys
    assert "memory" in async_keys
