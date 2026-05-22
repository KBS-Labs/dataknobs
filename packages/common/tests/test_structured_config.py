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


@dataclass(frozen=True)
class _Leaf(StructuredConfig):
    value: int = 0
    label: str = "leaf"


@dataclass(frozen=True)
class _Nested(StructuredConfig):
    name: str = ""
    leaf: _Leaf = field(default_factory=_Leaf)


@dataclass(frozen=True)
class _OptionalNested(StructuredConfig):
    leaf: _Leaf | None = None


@dataclass(frozen=True)
class _ListNested(StructuredConfig):
    leaves: list[_Leaf] = field(default_factory=list)


@dataclass(frozen=True)
class _TupleNested(StructuredConfig):
    leaves: tuple[_Leaf, ...] = ()


@dataclass(frozen=True)
class _DictNested(StructuredConfig):
    leaves: dict[str, _Leaf] = field(default_factory=dict)


@dataclass(frozen=True)
class _DictListNested(StructuredConfig):
    groups: dict[str, list[_Leaf]] = field(default_factory=dict)


class TestNestedComposition:
    """``from_dict`` recurses into ``StructuredConfig`` field types (146b C)."""

    def test_single_nested_dict_becomes_typed(self) -> None:
        cfg = _Nested.from_dict({"name": "n", "leaf": {"value": 3}})
        assert isinstance(cfg.leaf, _Leaf)
        assert cfg.leaf.value == 3
        assert cfg.leaf.label == "leaf"

    def test_nested_default_factory_when_absent(self) -> None:
        cfg = _Nested.from_dict({"name": "n"})
        assert cfg.leaf == _Leaf()

    def test_optional_none_stays_none(self) -> None:
        cfg = _OptionalNested.from_dict({"leaf": None})
        assert cfg.leaf is None

    def test_optional_dict_becomes_typed(self) -> None:
        cfg = _OptionalNested.from_dict({"leaf": {"value": 7}})
        assert isinstance(cfg.leaf, _Leaf)
        assert cfg.leaf.value == 7

    def test_list_of_dicts_becomes_list_of_typed(self) -> None:
        cfg = _ListNested.from_dict(
            {"leaves": [{"value": 1}, {"value": 2}]}
        )
        assert cfg.leaves == [_Leaf(value=1), _Leaf(value=2)]
        assert all(isinstance(item, _Leaf) for item in cfg.leaves)

    def test_tuple_of_dicts_becomes_tuple_of_typed(self) -> None:
        cfg = _TupleNested.from_dict({"leaves": [{"value": 5}]})
        assert isinstance(cfg.leaves, tuple)
        assert cfg.leaves == (_Leaf(value=5),)

    def test_dict_of_dicts_becomes_dict_of_typed(self) -> None:
        cfg = _DictNested.from_dict(
            {"leaves": {"a": {"value": 1}, "b": {"value": 2}}}
        )
        assert cfg.leaves == {"a": _Leaf(value=1), "b": _Leaf(value=2)}

    def test_dict_of_lists_recurses_both_levels(self) -> None:
        cfg = _DictListNested.from_dict(
            {"groups": {"g": [{"value": 1}, {"value": 2}]}}
        )
        assert cfg.groups == {"g": [_Leaf(value=1), _Leaf(value=2)]}
        assert all(
            isinstance(item, _Leaf) for item in cfg.groups["g"]
        )

    def test_pretyped_value_passes_through(self) -> None:
        """A field already holding a typed instance is left untouched."""
        leaf = _Leaf(value=9)
        cfg = _Nested.from_dict({"name": "n", "leaf": leaf})
        assert cfg.leaf is leaf

    def test_pretyped_list_elements_pass_through(self) -> None:
        leaf = _Leaf(value=9)
        cfg = _ListNested.from_dict({"leaves": [leaf]})
        assert cfg.leaves[0] is leaf


class TestNestedRoundTrip:
    """Round-trip now holds for nested configs without ``_normalize_dict``."""

    def test_single_nested_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            _Nested(name="n", leaf=_Leaf(value=3))
        )

    def test_optional_nested_roundtrip(self) -> None:
        assert_structured_config_roundtrip(_OptionalNested(leaf=_Leaf(value=1)))
        assert_structured_config_roundtrip(_OptionalNested(leaf=None))

    def test_list_nested_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            _ListNested(leaves=[_Leaf(value=1), _Leaf(value=2)])
        )

    def test_dict_list_nested_roundtrip(self) -> None:
        assert_structured_config_roundtrip(
            _DictListNested(groups={"g": [_Leaf(value=1)]})
        )


class TestFlatConfigUnchanged:
    """Flat / non-config fields project verbatim, without recursion (regression)."""

    def test_flat_scalar_projection_unchanged(self) -> None:
        cfg = _Simple.from_dict({"required": "x", "optional_default": 9})
        assert cfg == _Simple(required="x", optional_default=9)

    def test_plain_list_field_not_transformed(self) -> None:
        """A ``list[str]`` field (no nested config) is assigned verbatim."""
        original = ["a", "b"]
        cfg = _WithFactory.from_dict({"items": original})
        assert cfg.items == ["a", "b"]
        # The non-config gate avoids the coercion path entirely, so the
        # value is the same object the caller supplied (no rebuild).
        assert cfg.items is original
