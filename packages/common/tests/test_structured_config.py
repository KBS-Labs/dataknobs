"""Property tests for the ``StructuredConfig`` base class.

Pin the contract that downstream consumers (data backends, vector
stores, bots subsystems, FSM patterns, ...) build on:

- ``from_dict`` projects only declared fields.
- Unknown keys (registry-routing keys like ``"backend"``) pass through.
- Defaults / ``default_factory`` are honoured.
- ``_normalize_dict`` override runs before field projection.
- ``__post_init__`` validation surfaces through ``from_dict``.
- ``to_dict`` round-trips: ``cls.from_dict(cfg.to_dict()) == cfg``.
- Frozen-dataclass invariant blocks runtime mutation.
- ``StructuredConfig`` is structurally a ``Serializable``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import pytest

from dataknobs_common.serialization import Serializable
from dataknobs_common.structured_config import StructuredConfig
from dataknobs_common.testing import assert_structured_config_roundtrip


@dataclass(frozen=True)
class _Empty(StructuredConfig):
    pass


@dataclass(frozen=True)
class _Simple(StructuredConfig):
    required: str
    optional_default: int = 5
    optional_none: str | None = None


@dataclass(frozen=True)
class _WithFactory(StructuredConfig):
    items: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Validating(StructuredConfig):
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")


@dataclass(frozen=True)
class _Renamed(StructuredConfig):
    new_field: str = "default"

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        if "legacy_key" in raw and "new_field" not in raw:
            raw["new_field"] = raw.pop("legacy_key")
        return raw


class TestStructuredConfigFromDict:
    """``from_dict`` projects declared fields and tolerates extras."""

    def test_empty_subclass_from_dict_with_no_keys(self) -> None:
        assert _Empty.from_dict({}) == _Empty()

    def test_empty_subclass_from_dict_ignores_unknown_keys(self) -> None:
        """Routing keys like ``backend`` pass through unharmed."""
        assert _Empty.from_dict({"backend": "x"}) == _Empty()

    def test_required_field_must_be_supplied(self) -> None:
        with pytest.raises(TypeError, match="required"):
            _Simple.from_dict({})

    def test_field_projection_with_defaults(self) -> None:
        cfg = _Simple.from_dict({"required": "hello"})
        assert cfg.required == "hello"
        assert cfg.optional_default == 5
        assert cfg.optional_none is None

    def test_field_projection_preserves_explicit_falsy(self) -> None:
        """``{optional_default: 0}`` must produce ``0`` — not the default 5."""
        cfg = _Simple.from_dict({"required": "x", "optional_default": 0})
        assert cfg.optional_default == 0

    def test_field_projection_preserves_explicit_none(self) -> None:
        cfg = _Simple.from_dict(
            {"required": "x", "optional_none": None}
        )
        assert cfg.optional_none is None

    def test_default_factory_runs(self) -> None:
        cfg = _WithFactory.from_dict({})
        assert cfg.items == []

    def test_default_factory_overridden_by_dict(self) -> None:
        cfg = _WithFactory.from_dict({"items": ["a", "b"]})
        assert cfg.items == ["a", "b"]

    def test_unknown_keys_ignored(self) -> None:
        cfg = _Simple.from_dict({"required": "x", "garbage": "ignored"})
        assert cfg.required == "x"

    def test_caller_dict_not_mutated(self) -> None:
        """``from_dict`` shallow-copies; caller's dict is untouched."""
        raw = {"legacy_key": "value"}
        _Renamed.from_dict(raw)
        assert raw == {"legacy_key": "value"}


class TestStructuredConfigNormalizeDict:
    """``_normalize_dict`` runs before field projection."""

    def test_override_renames_legacy_key(self) -> None:
        cfg = _Renamed.from_dict({"legacy_key": "v"})
        assert cfg.new_field == "v"

    def test_override_no_op_for_canonical_key(self) -> None:
        cfg = _Renamed.from_dict({"new_field": "v"})
        assert cfg.new_field == "v"


class TestStructuredConfigPostInit:
    """``__post_init__`` validation surfaces through ``from_dict``."""

    def test_validation_failure_propagates(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _Validating.from_dict({})

    def test_validation_passes_when_field_supplied(self) -> None:
        cfg = _Validating.from_dict({"name": "alice"})
        assert cfg.name == "alice"


class TestStructuredConfigToDict:
    """``to_dict`` round-trips through ``from_dict``."""

    def test_simple_roundtrip(self) -> None:
        original = _Simple(required="hello", optional_default=7)
        recovered = _Simple.from_dict(original.to_dict())
        assert recovered == original

    def test_factory_field_roundtrip(self) -> None:
        original = _WithFactory(items=["a", "b"])
        assert _WithFactory.from_dict(original.to_dict()) == original

    def test_roundtrip_helper_passes(self) -> None:
        assert_structured_config_roundtrip(_Simple(required="x"))

    def test_roundtrip_helper_for_factory_field(self) -> None:
        assert_structured_config_roundtrip(_WithFactory(items=["a"]))


class TestStructuredConfigFrozen:
    """Frozen-dataclass invariant blocks runtime mutation."""

    def test_assignment_raises_frozen_error(self) -> None:
        cfg = _Simple(required="x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.required = "y"  # type: ignore[misc]


class TestStructuredConfigSerializable:
    """``StructuredConfig`` instances structurally satisfy ``Serializable``.

    No nominal inheritance from ``Serializable`` — the relationship is
    purely structural (``to_dict`` / ``from_dict`` are both present).
    """

    def test_instance_is_serializable_protocol(self) -> None:
        assert isinstance(_Simple(required="x"), Serializable)

    def test_empty_subclass_is_serializable_protocol(self) -> None:
        assert isinstance(_Empty(), Serializable)
