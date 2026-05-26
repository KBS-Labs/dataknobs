"""Tests for ``StructuredConfig.validate`` and the config-resolver registry.

Pins the registry-of-registries config-resolver contract:

- ``validate`` is a no-op for a config that declares no
  ``_polymorphic_fields``.
- An empty / ``None`` section is skipped.
- A binding with no registered resolver is skipped with a debug log (the
  fail-soft behavior that keeps validation un-brittle to import order).
- A resolver that returns ``None`` (unknown discriminator) raises
  ``ConfigurationError``.
- A resolver that returns a config class dry-run-builds it, surfacing the
  child's own field-level errors, and recurses so one ``parent.validate()``
  validates the whole polymorphic tree.
- ``list``-valued sections validate every element.
- ``assert_polymorphic_bindings_resolve`` passes when wired and fails when
  a binding is unregistered.

All tests use real ``StructuredConfig`` subclasses and a test-registered
resolver — no mocks. The shared ``config_registries`` is restored after each
test so registration is isolated.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.structured_config import (
    SKIP_VALIDATION,
    StructuredConfig,
    config_registries,
)
from dataknobs_common.testing import assert_polymorphic_bindings_resolve


# --- Test config classes -------------------------------------------------


@dataclass(frozen=True)
class _LeafConfig(StructuredConfig):
    """A resolvable section config with a ``__post_init__`` invariant."""

    size: int = 1

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError(f"size must be >= 0, got {self.size}")


@dataclass(frozen=True)
class _NonAdopter(StructuredConfig):
    name: str = "x"


@dataclass(frozen=True)
class _Adopter(StructuredConfig):
    """Holds a single polymorphic raw-dict section bound to ``leaf``."""

    _polymorphic_fields: ClassVar[Mapping[str, str]] = {"section": "leaf"}
    section: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ListAdopter(StructuredConfig):
    """Holds a list-valued polymorphic section."""

    _polymorphic_fields: ClassVar[Mapping[str, str]] = {"sections": "leaf"}
    sections: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _ParentAdopter(StructuredConfig):
    """Holds a section that resolves to ``_Adopter`` — a two-level tree."""

    _polymorphic_fields: ClassVar[Mapping[str, str]] = {"child": "adopter"}
    child: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Node(StructuredConfig):
    """Self-referential statically-typed node for the cycle-guard test."""

    label: str = "n"
    next: "_Node | None" = None


def _resolve_leaf(raw: Mapping[str, Any]) -> type[StructuredConfig] | None:
    # Discriminator is ``kind``; only ``leaf`` is known.
    return _LeafConfig if raw.get("kind") == "leaf" else None


def _resolve_adopter(raw: Mapping[str, Any]) -> type[StructuredConfig] | None:
    return _Adopter


# Return annotation omitted deliberately: the resolver's return type union
# includes the private ``_SkipValidation`` sentinel type, which tests should
# not import. The contract is enforced where this is passed to
# ``config_registries.register`` (typed ``ConfigClassResolver``).
def _resolve_skip(raw: Mapping[str, Any]):
    # Recognizes the discriminator but has no typed config to validate
    # against (the bare-callable-backend case) — skip, don't raise.
    return SKIP_VALIDATION if raw.get("kind") == "leaf" else None


@pytest.fixture
def leaf_registered() -> Any:
    """Register the ``leaf`` resolver, restoring the registry afterward."""
    config_registries.register("leaf", _resolve_leaf, allow_overwrite=True)
    try:
        yield
    finally:
        if config_registries.has("leaf"):
            config_registries.unregister("leaf")


# --- No-op / skip cases ---------------------------------------------------


def test_non_adopter_validate_is_noop() -> None:
    # No _polymorphic_fields => nothing to resolve, no registry access.
    _NonAdopter.from_dict({"name": "y"}).validate()


def test_empty_section_skipped(leaf_registered: Any) -> None:
    # Default empty dict section is skipped (the from_components path).
    _Adopter.from_dict({}).validate()
    _Adopter.from_dict({"section": {}}).validate()


def test_unregistered_binding_skips_with_debug(caplog: Any) -> None:
    # No resolver registered for "leaf" here => skip + debug, no raise.
    assert not config_registries.has("leaf")
    with caplog.at_level(logging.DEBUG, logger="dataknobs_common.structured_config"):
        _Adopter.from_dict({"section": {"kind": "leaf"}}).validate()
    assert any("No config resolver registered" in r.message for r in caplog.records)


# --- Resolve + build + recurse cases -------------------------------------


def test_good_section_validates(leaf_registered: Any) -> None:
    _Adopter.from_dict({"section": {"kind": "leaf", "size": 3}}).validate()


def test_unknown_discriminator_raises(leaf_registered: Any) -> None:
    with pytest.raises(ConfigurationError) as exc:
        _Adopter.from_dict({"section": {"kind": "bogus"}}).validate()
    msg = str(exc.value)
    assert "_Adopter" in msg
    assert "section" in msg
    assert "leaf" in msg


def test_skip_sentinel_skips_without_raising(caplog: Any) -> None:
    # A resolver that returns SKIP_VALIDATION means "recognized but no typed
    # config" — validate() skips with a debug log, distinct from the None
    # (unknown discriminator) path which raises.
    config_registries.register("leaf", _resolve_skip, allow_overwrite=True)
    try:
        with caplog.at_level(
            logging.DEBUG, logger="dataknobs_common.structured_config"
        ):
            _Adopter.from_dict({"section": {"kind": "leaf", "size": -1}}).validate()
        # -1 would fail _LeafConfig.__post_init__ if it were built; the skip
        # means the section is never dry-run-built, so no error surfaces.
        assert any(
            "exposes no typed config" in r.message for r in caplog.records
        )
    finally:
        config_registries.unregister("leaf")


def test_skip_sentinel_does_not_suppress_unknown_discriminator() -> None:
    # The same resolver returns None (not SKIP) for an unrecognized
    # discriminator, which must still raise — SKIP is not a blanket mute.
    config_registries.register("leaf", _resolve_skip, allow_overwrite=True)
    try:
        with pytest.raises(ConfigurationError):
            _Adopter.from_dict({"section": {"kind": "bogus"}}).validate()
    finally:
        config_registries.unregister("leaf")


def test_child_field_error_propagates(leaf_registered: Any) -> None:
    # The child's __post_init__ raises ValueError, surfaced at validate().
    with pytest.raises(ValueError, match="size must be >= 0"):
        _Adopter.from_dict({"section": {"kind": "leaf", "size": -1}}).validate()


def test_list_valued_section_validates_each(leaf_registered: Any) -> None:
    _ListAdopter.from_dict(
        {"sections": [{"kind": "leaf", "size": 1}, {"kind": "leaf", "size": 2}]}
    ).validate()
    # A bad element anywhere in the list surfaces.
    with pytest.raises(ValueError, match="size must be >= 0"):
        _ListAdopter.from_dict(
            {"sections": [{"kind": "leaf", "size": 1}, {"kind": "leaf", "size": -5}]}
        ).validate()


def test_recursion_validates_whole_tree() -> None:
    # _ParentAdopter.child -> _Adopter; _Adopter.section -> _LeafConfig.
    # One parent.validate() must reach the leaf's __post_init__.
    config_registries.register("adopter", _resolve_adopter, allow_overwrite=True)
    config_registries.register("leaf", _resolve_leaf, allow_overwrite=True)
    try:
        _ParentAdopter.from_dict(
            {"child": {"section": {"kind": "leaf", "size": 0}}}
        ).validate()
        with pytest.raises(ValueError, match="size must be >= 0"):
            _ParentAdopter.from_dict(
                {"child": {"section": {"kind": "leaf", "size": -1}}}
            ).validate()
    finally:
        config_registries.unregister("adopter")
        config_registries.unregister("leaf")


# --- Cycle guard ----------------------------------------------------------


def test_validate_cycle_guard_terminates() -> None:
    # A statically-typed nested graph with a cycle must terminate via the
    # visited-set guard rather than recursing without bound. ``from_dict``
    # cannot build a cycle (frozen + acyclic construction), so force one with
    # ``object.__setattr__`` to exercise the guard directly. Without it this
    # would raise ``RecursionError``.
    a = _Node(label="a")
    b = _Node(label="b", next=a)
    object.__setattr__(a, "next", b)  # a -> b -> a
    a.validate()  # returns; no RecursionError


# --- Parity guard ---------------------------------------------------------


def test_bindings_resolve_guard_passes_when_registered(leaf_registered: Any) -> None:
    assert_polymorphic_bindings_resolve(_Adopter)


def test_bindings_resolve_guard_fails_when_unregistered() -> None:
    assert not config_registries.has("leaf")
    with pytest.raises(AssertionError, match="not registered in config_registries"):
        assert_polymorphic_bindings_resolve(_Adopter)


def test_bindings_resolve_guard_trivial_for_non_adopter() -> None:
    assert_polymorphic_bindings_resolve(_NonAdopter)
