"""Tests for the ``vector_store`` config resolver.

The resolver registered by ``dataknobs-data`` into the shared
``config_registries`` lets ``StructuredConfig.validate`` check a raw
``vector_store`` section without constructing the store. These tests pin:

- The resolver is registered eagerly on import.
- It returns the same ``CONFIG_CLS`` the construction registry would use,
  for *every* registered backend (the no-drift guarantee).
- It returns ``None`` for an unknown backend.
- It defaults the discriminator to ``"memory"`` (the factory's own default).
"""

from __future__ import annotations

import dataknobs_data.vector.stores  # noqa: F401 — eager resolver registration
from dataknobs_data.vector.stores import vector_backends
from dataknobs_data.vector.stores.config import MemoryVectorStoreConfig

from dataknobs_common.structured_config import (
    SKIP_VALIDATION,
    StructuredConfig,
    config_registries,
)


def _resolver():
    return config_registries.get("vector_store")


def test_resolver_registered_on_import() -> None:
    assert config_registries.has("vector_store")


def test_resolver_agrees_with_construction_registry_for_all_backends() -> None:
    # Drift guard: for every registered backend, the resolver must return
    # exactly the CONFIG_CLS the construction path reads off the store class,
    # so validation and construction can never resolve to different configs.
    resolver = _resolver()
    keys = vector_backends.list_keys()
    assert keys, "expected at least the always-available memory backend"
    for key in keys:
        expected = getattr(vector_backends.get_factory(key), "CONFIG_CLS", None)
        assert resolver({"backend": key}) is expected, f"drift for backend {key!r}"


def test_resolver_defaults_to_memory() -> None:
    # No "backend" key => memory (the VectorStoreFactory default).
    assert _resolver()({}) is MemoryVectorStoreConfig


def test_resolver_returns_none_for_unknown_backend() -> None:
    result = _resolver()({"backend": "pgvektor"})
    assert result is None


def test_registered_backend_without_config_cls_is_skipped() -> None:
    # A backend registered as a bare callable (no CONFIG_CLS) is recognized
    # but has no typed schema to validate against. The resolver returns
    # SKIP_VALIDATION (not None), so validate() skips it rather than
    # false-positive-raising on a valid, constructible backend.
    def _untyped_factory(config: object = None, **_: object) -> object:
        raise NotImplementedError  # never built — resolver only reads the type

    vector_backends.register(
        "untyped_test_backend", _untyped_factory, override=True
    )
    try:
        assert _resolver()({"backend": "untyped_test_backend"}) is SKIP_VALIDATION
    finally:
        vector_backends.unregister("untyped_test_backend")


def test_resolved_config_is_structured_config_subclass() -> None:
    cls = _resolver()({"backend": "memory"})
    assert isinstance(cls, type) and issubclass(cls, StructuredConfig)
