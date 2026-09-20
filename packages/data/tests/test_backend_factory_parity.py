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
elasticsearch → elasticsearch, s3 → boto3, ...) are still audited --- the
population comes from every backend the registry *knows of* rather than
every one it can build, so a missing driver costs a named skip rather than
a case that quietly stops being collected. Behavioural coverage of each
backend lives in its own test module.
"""

from __future__ import annotations

import inspect

import pytest

from dataknobs_data.backend_selection import KnownBackend, known_backend_classes
from dataknobs_data.backends import async_backends, sync_backends


#: One entry per backend, aliases collapsed --- the parity guarantee is a
#: property of the class, and ``mem`` and ``memory`` are one class.
#:
#: Derived through the shared accessor rather than by walking ``list_keys``
#: here, because that walk answers "what can this installation build?" and the
#: claim in this module's docstring is the wider one: that optional-dependency
#: backends are audited too. They were not. A backend whose driver is absent is
#: declared unavailable, drops out of ``list_keys``, and takes its parametrized
#: case with it --- so the audit went on reporting green over whichever
#: backends the environment happened to have. ``known_backend_classes`` keeps
#: it in the population and reports why its class could not be reached, which
#: :func:`_reachable` turns into a named skip.
SYNC_BACKENDS = known_backend_classes(sync_backends)
ASYNC_BACKENDS = known_backend_classes(async_backends)


def _reachable(entry: KnownBackend) -> type:
    """The backend's class, or a skip naming the one that could not be reached."""
    if entry.cls is None:
        pytest.skip(f"{entry.key}: {entry.unavailable_reason}")
    return entry.cls


@pytest.mark.parametrize("entry", SYNC_BACKENDS, ids=lambda e: e.key)
def test_sync_backend_exposes_from_config(entry: KnownBackend) -> None:
    """Every registered sync backend exposes ``from_config(cls, config)``.

    ``DatabaseFactory.create`` calls ``backend_class.from_config(config)``
    after registry lookup. If a backend regresses to a different
    construction shape (or removes ``from_config`` entirely), the
    factory dispatch breaks at the first consumer call — this test
    surfaces the regression at unit-test time instead.
    """
    name, backend_cls = entry.key, _reachable(entry)
    assert hasattr(backend_cls, "from_config"), (
        f"Backend {name} ({backend_cls.__name__}) has no `from_config` "
        "classmethod; DatabaseFactory.create would raise AttributeError."
    )
    sig = inspect.signature(backend_cls.from_config)
    params = list(sig.parameters.values())
    assert len(params) >= 1, (
        f"{backend_cls.__name__}.from_config must accept a config positional/keyword argument."
    )


@pytest.mark.parametrize("entry", ASYNC_BACKENDS, ids=lambda e: e.key)
def test_async_backend_exposes_from_config(entry: KnownBackend) -> None:
    """Every known async backend exposes ``from_config(cls, config)``."""
    name, backend_cls = entry.key, _reachable(entry)
    assert hasattr(backend_cls, "from_config"), (
        f"Async backend {name} ({backend_cls.__name__}) has no `from_config` classmethod."
    )
    sig = inspect.signature(backend_cls.from_config)
    params = list(sig.parameters.values())
    assert len(params) >= 1, f"{backend_cls.__name__}.from_config must accept a config arg."


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
