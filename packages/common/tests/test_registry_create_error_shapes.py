"""Tests pinning :meth:`PluginRegistry.create` / ``create_async`` error
shapes and the :class:`BackendRegistry` Protocol surface.

Covers:

- Default not-found error shape (``NotFoundError`` + the
  ``"Plugin '<key>' not registered"`` text + structured ``context``).
- Opt-in kind-shaped error (``not_found_kind`` /
  ``not_found_exception`` ctor kwargs) for consolidating shims.
- ``OperationError`` wrapping of factory-internal errors (regression
  pin — unchanged by the new ctor kwargs).
- ``validate_type`` failure wrapping (regression pin).
- ``BackendRegistry`` Protocol conformance for :class:`Registry`,
  :class:`PluginRegistry`, and a class deliberately missing the
  surface.
- ``allow_overwrite=`` keyword alias on :meth:`PluginRegistry.register`.
"""

from __future__ import annotations

from typing import Any

import pytest

from dataknobs_common import (
    BackendRegistry,
    NotFoundError,
    OperationError,
    PluginRegistry,
    Registry,
)


# ---------------------------------------------------------------------------
# Default not-found shape (unchanged contract)
# ---------------------------------------------------------------------------


def _make_registry_default() -> PluginRegistry[Any]:
    registry: PluginRegistry[Any] = PluginRegistry("test_default")
    registry.register("known", lambda config: {"config": config})
    return registry


def test_create_unknown_key_default_shape() -> None:
    registry = _make_registry_default()

    with pytest.raises(NotFoundError) as excinfo:
        registry.create(key="missing")

    assert "Plugin 'missing' not registered" in str(excinfo.value)
    assert excinfo.value.context["key"] == "missing"
    assert excinfo.value.context["registry"] == "test_default"
    assert "known" in excinfo.value.context["available"]


async def test_create_async_unknown_key_default_shape() -> None:
    registry = _make_registry_default()

    with pytest.raises(NotFoundError) as excinfo:
        await registry.create_async(key="missing")

    assert "Plugin 'missing' not registered" in str(excinfo.value)
    assert excinfo.value.context["registry"] == "test_default"


# ---------------------------------------------------------------------------
# Opt-in kind-shaped error
# ---------------------------------------------------------------------------


def _make_registry_kind_shaped() -> PluginRegistry[Any]:
    registry: PluginRegistry[Any] = PluginRegistry(
        "event_bus_backends",
        config_key="backend",
        config_key_default="memory",
        not_found_kind="event bus backend",
        not_found_exception=ValueError,
    )
    registry.register("memory", lambda config: {"config": config})
    registry.register("postgres", lambda config: {"config": config})
    registry.register("redis", lambda config: {"config": config})
    return registry


def test_create_unknown_key_kind_shape() -> None:
    registry = _make_registry_kind_shaped()

    with pytest.raises(ValueError) as excinfo:
        registry.create(config={"backend": "kafka"})

    message = str(excinfo.value)
    assert "Unknown event bus backend: kafka" in message
    assert "Available backends: memory, postgres, redis" in message


async def test_create_async_unknown_key_kind_shape() -> None:
    registry = _make_registry_kind_shaped()

    with pytest.raises(ValueError) as excinfo:
        await registry.create_async(config={"backend": "kafka"})

    message = str(excinfo.value)
    assert "Unknown event bus backend: kafka" in message
    assert "Available backends: memory, postgres, redis" in message


def test_create_unknown_key_lists_available_sorted() -> None:
    registry = _make_registry_kind_shaped()
    # Register out-of-alphabetical-order entry to confirm sort
    registry.register("aardvark", lambda config: {"config": config})

    with pytest.raises(ValueError) as excinfo:
        registry.create(config={"backend": "kafka"})

    # Sorted, with aardvark first
    assert (
        "Available backends: aardvark, memory, postgres, redis"
        in str(excinfo.value)
    )


def test_unknown_key_message_handles_value_error_no_context() -> None:
    """``ValueError`` (and other non-DataknobsError classes) must NOT
    receive a ``context=`` kwarg — they would crash with ``TypeError``.
    """
    registry = _make_registry_kind_shaped()

    with pytest.raises(ValueError) as excinfo:
        registry.create(config={"backend": "kafka"})

    # Should be a clean ValueError; the message stands alone.
    assert not hasattr(excinfo.value, "context")
    # And the message itself is intact.
    assert "Unknown event bus backend: kafka" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Existing-behavior regression pins (must NOT be disturbed by the new
# ctor kwargs)
# ---------------------------------------------------------------------------


def test_create_factory_raises_wraps_in_operation_error() -> None:
    registry: PluginRegistry[Any] = PluginRegistry("test_wrap")

    def boom(config: dict[str, Any]) -> Any:
        raise RuntimeError("factory exploded")

    registry.register("explosive", boom)

    with pytest.raises(OperationError) as excinfo:
        registry.create(key="explosive")

    assert "Failed to create plugin 'explosive'" in str(excinfo.value)
    # Original cause is preserved via __cause__
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_create_validate_type_failure() -> None:
    class Base:
        pass

    registry: PluginRegistry[Base] = PluginRegistry(
        "test_validate", validate_type=Base
    )
    # Factory returns the wrong type
    registry.register("wrong", lambda config: object())

    with pytest.raises(OperationError) as excinfo:
        registry.create(key="wrong")

    assert "Failed to create plugin 'wrong'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# BackendRegistry Protocol conformance
# ---------------------------------------------------------------------------


def test_registry_conforms_to_backend_registry_protocol() -> None:
    registry: Registry[str] = Registry("items")
    assert isinstance(registry, BackendRegistry)


def test_plugin_registry_conforms_to_backend_registry_protocol() -> None:
    registry: PluginRegistry[Any] = PluginRegistry("plugins")
    assert isinstance(registry, BackendRegistry)


def test_backend_registry_protocol_rejects_non_conforming() -> None:
    class NotARegistry:
        """Deliberately missing ``has`` / ``list_keys`` / ``unregister``."""

        @property
        def name(self) -> str:
            return "fake"

    assert not isinstance(NotARegistry(), BackendRegistry)


# ---------------------------------------------------------------------------
# allow_overwrite= alias on PluginRegistry.register
# ---------------------------------------------------------------------------


def test_allow_overwrite_alias_replaces_override() -> None:
    registry: PluginRegistry[Any] = PluginRegistry("test_overwrite")
    registry.register("k", lambda config: {"v": 1})
    # Should succeed via the alias
    registry.register("k", lambda config: {"v": 2}, allow_overwrite=True)

    instance = registry.create(key="k")
    assert instance == {"v": 2}


def test_allow_overwrite_alias_no_arg_unchanged() -> None:
    registry: PluginRegistry[Any] = PluginRegistry("test_overwrite_default")
    registry.register("k", lambda config: {"v": 1})

    with pytest.raises(OperationError):
        registry.register("k", lambda config: {"v": 2})


def test_allow_overwrite_alias_explicit_false_no_effect() -> None:
    """Passing ``allow_overwrite=False`` explicitly is the opt-in
    "I want NO overwrite" spelling — equivalent to omitting the kwarg.
    """
    registry: PluginRegistry[Any] = PluginRegistry("test_overwrite_false")
    registry.register("k", lambda config: {"v": 1})

    with pytest.raises(OperationError):
        registry.register("k", lambda config: {"v": 2}, allow_overwrite=False)


def test_override_kwarg_unchanged() -> None:
    """The pre-existing ``override`` kwarg continues to work."""
    registry: PluginRegistry[Any] = PluginRegistry("test_override_kwarg")
    registry.register("k", lambda config: {"v": 1})
    # Positional + keyword both work
    registry.register("k", lambda config: {"v": 2}, override=True)
    instance = registry.create(key="k")
    assert instance == {"v": 2}
