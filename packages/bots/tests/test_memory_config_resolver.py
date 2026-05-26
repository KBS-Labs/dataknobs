"""Tests for the ``memory`` config resolver.

The resolver registered by ``dataknobs-bots`` into the shared
``config_registries`` lets ``StructuredConfig.validate`` check a raw
``memory`` section (and each element of a composite's ``strategies`` list)
without constructing the backend. These tests pin:

- The resolver is registered eagerly on import.
- It returns the same ``CONFIG_CLS`` the construction registry would use,
  for *every* registered backend (the no-drift guarantee).
- It returns ``None`` for an unknown type.
- It defaults the discriminator to ``"buffer"`` (the factory's own default).
- A registered bare-callable backend (no ``CONFIG_CLS``) resolves to
  ``SKIP_VALIDATION`` rather than ``None``.

These construct config internals directly (not bot flows), so no
``BotTestHarness`` is needed.
"""

from __future__ import annotations

# Required side-effect import: importing this module registers the "memory"
# resolver in config_registries. Do NOT remove as "unused".
import dataknobs_bots.memory.registry  # noqa: F401
from dataknobs_bots.memory.config import BufferMemoryConfig
from dataknobs_bots.memory.registry import memory_backends

from dataknobs_common.structured_config import (
    SKIP_VALIDATION,
    StructuredConfig,
    config_registries,
)


def _resolver():
    return config_registries.get("memory")


def test_resolver_registered_on_import() -> None:
    assert config_registries.has("memory")


def test_resolver_agrees_with_construction_registry_for_all_backends() -> None:
    # Drift guard: for every registered backend, the resolver must return
    # exactly the CONFIG_CLS the construction path reads off the backend class,
    # so validation and construction can never resolve to different configs.
    resolver = _resolver()
    keys = memory_backends.list_keys()
    assert keys, "expected the built-in memory backends to be registered"
    for key in keys:
        expected = getattr(memory_backends.get_factory(key), "CONFIG_CLS", None)
        assert resolver({"type": key}) is expected, f"drift for backend {key!r}"


def test_resolver_defaults_to_buffer() -> None:
    # No "type" key => buffer (the memory_backends default).
    assert _resolver()({}) is BufferMemoryConfig


def test_resolver_returns_none_for_unknown_type() -> None:
    assert _resolver()({"type": "buffferr"}) is None


def test_registered_backend_without_config_cls_is_skipped() -> None:
    # A backend registered as a bare callable (no CONFIG_CLS) is recognized
    # but has no typed schema to validate against. The resolver returns
    # SKIP_VALIDATION (not None), so validate() skips it rather than
    # false-positive-raising on a valid, constructible backend.
    def _untyped_factory(config: object = None, **_: object) -> object:
        raise NotImplementedError  # never built — resolver only reads the type

    memory_backends.register("untyped_test_backend", _untyped_factory, override=True)
    try:
        assert _resolver()({"type": "untyped_test_backend"}) is SKIP_VALIDATION
    finally:
        memory_backends.unregister("untyped_test_backend")


def test_resolved_config_is_structured_config_subclass() -> None:
    cls = _resolver()({"type": "buffer"})
    assert isinstance(cls, type) and issubclass(cls, StructuredConfig)
