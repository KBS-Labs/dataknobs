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
from typing import Any, ClassVar

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
class _LeafB(StructuredConfig):
    other: int = 0


@dataclass(frozen=True)
class _DictListNested(StructuredConfig):
    groups: dict[str, list[_Leaf]] = field(default_factory=dict)


@dataclass(frozen=True)
class _MultiUnionNested(StructuredConfig):
    """A union of *two* config subclasses — a polymorphic shape."""

    leaf: _Leaf | _LeafB = field(default_factory=_Leaf)


class TestNestedComposition:
    """``from_dict`` recurses into ``StructuredConfig`` field types."""

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

    def test_multi_config_union_dict_passes_through(self) -> None:
        """A union of several config arms is NOT auto-coerced.

        Selecting among ``_Leaf`` / ``_LeafB`` from the data is a
        discriminated/polymorphic decision that stays in the object-graph
        layer, so ``from_dict`` leaves the raw dict for the consumer's
        factory to dispatch — it must not silently bind the first arm.
        """
        raw = {"value": 3}
        cfg = _MultiUnionNested.from_dict({"leaf": raw})
        assert cfg.leaf is raw
        assert not isinstance(cfg.leaf, (_Leaf, _LeafB))

    def test_multi_config_union_pretyped_passes_through(self) -> None:
        """A value already typed as one arm is preserved as-is."""
        leaf = _LeafB(other=7)
        cfg = _MultiUnionNested.from_dict({"leaf": leaf})
        assert cfg.leaf is leaf


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


@dataclass(frozen=True)
class _WithSecret(StructuredConfig):
    """Opts into redaction with ONLY ``_SENSITIVE_FIELDS`` — no ``__repr__``.

    Proves the base installs the redacting repr automatically: a leaf
    needs no boilerplate beyond naming its secret fields.
    """

    host: str = "db"
    api_key: str | None = None

    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key"})


@dataclass(frozen=True)
class _WithEmptyDefaultSecret(StructuredConfig):
    """A non-optional credential that defaults to ``""`` (e.g. a password)."""

    password: str = ""

    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"password"})


@dataclass(frozen=True)
class _SecretBase(StructuredConfig):
    """An intermediate ``@dataclass`` base (generates its own repr)."""

    shared: str = "s"


@dataclass(frozen=True)
class _SecretLeaf(_SecretBase):
    """Multi-level leaf — also needs no hand-written ``__repr__``.

    The intermediate ``_SecretBase`` is a ``@dataclass`` whose generated
    repr would shadow an *inherited* redacting repr; the base installs the
    redacting repr into every subclass dict (this leaf included), so
    redaction survives the extra inheritance level automatically.
    """

    token: str | None = None

    _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"token"})


@dataclass(frozen=True)
class _WithRawMapping(StructuredConfig):
    """A config holding an intentionally-untyped polymorphic section.

    Mirrors the real shape (``RAGKnowledgeBaseConfig.vector_store``,
    ``SummaryMemoryConfig.llm``): a raw ``dict`` whose schema is owned by
    another package and dispatched by a discriminator key. No
    ``_SENSITIVE_FIELDS`` is declared — the default interior-key set must
    mask credentials nested inside the mapping with zero per-class config.
    """

    label: str = "x"
    section: dict[str, Any] = field(default_factory=dict)
    items: list[dict[str, Any]] = field(default_factory=list)


class TestSensitiveFieldRedaction:
    """``__repr__`` masks ``_SENSITIVE_FIELDS``; ``to_dict`` does not.

    No leaf defines its own ``__repr__`` — the ``StructuredConfig`` base
    installs the redacting repr on every subclass via ``__init_subclass__``.
    """

    def test_repr_auto_installed_from_base(self) -> None:
        """``__init_subclass__`` writes the redacting repr into each subclass
        dict (before ``@dataclass`` runs), so no per-leaf boilerplate is
        needed — even on a no-secret config and a multi-level leaf."""
        assert _WithSecret.__dict__["__repr__"] is StructuredConfig._redacted_repr
        assert _SecretLeaf.__dict__["__repr__"] is StructuredConfig._redacted_repr
        # Inert (empty _SENSITIVE_FIELDS) but still installed.
        assert _Simple.__dict__["__repr__"] is StructuredConfig._redacted_repr

    def test_repr_masks_set_secret(self) -> None:
        rendered = repr(_WithSecret(host="db", api_key="sk-super-secret"))
        assert "sk-super-secret" not in rendered
        assert "api_key='***'" in rendered
        # Non-sensitive fields render normally.
        assert "host='db'" in rendered

    def test_repr_shows_none_secret_verbatim(self) -> None:
        """An unset secret is not masked — absence is not a secret."""
        assert "api_key=None" in repr(_WithSecret(host="db", api_key=None))

    def test_repr_shows_empty_secret_verbatim(self) -> None:
        """An empty-string credential renders verbatim — masking ``""`` would
        falsely imply a secret is configured."""
        assert "password=''" in repr(_WithEmptyDefaultSecret())
        # A real value is still masked.
        rendered = repr(_WithEmptyDefaultSecret(password="hunter2"))
        assert "hunter2" not in rendered
        assert "password='***'" in rendered

    def test_to_dict_is_not_redacted(self) -> None:
        """Redaction is display-only; serialization keeps the real value."""
        cfg = _WithSecret(host="db", api_key="sk-super-secret")
        assert cfg.to_dict()["api_key"] == "sk-super-secret"

    def test_roundtrip_preserved_with_secret(self) -> None:
        """``from_dict(to_dict())`` reconstructs the real secret."""
        cfg = _WithSecret(host="db", api_key="sk-super-secret")
        assert_structured_config_roundtrip(cfg)
        assert _WithSecret.from_dict(cfg.to_dict()).api_key == "sk-super-secret"

    def test_sensitive_fields_is_not_a_dataclass_field(self) -> None:
        """The ``ClassVar`` marker stays off the field set (and the ctor)."""
        names = {f.name for f in dataclasses.fields(_WithSecret)}
        assert "_SENSITIVE_FIELDS" not in names

    def test_default_config_repr_unchanged(self) -> None:
        """A config with no sensitive fields renders like a plain dataclass."""
        # ``_Simple`` renders through ``_redacted_repr`` (auto-installed on
        # every subclass) with an empty ``_SENSITIVE_FIELDS``, which produces
        # output byte-identical to the standard dataclass repr -- proving the
        # mechanism is *safe* (not inert) for configs with no secrets.
        assert repr(_Simple(required="x")) == (
            "_Simple(required='x', optional_default=5, optional_none=None)"
        )

    def test_multilevel_leaf_redacts_and_shows_inherited_fields(self) -> None:
        """A leaf under an intermediate dataclass base masks its secret and
        still renders the inherited field (the generated base repr would
        have dropped the leaf's own fields)."""
        rendered = repr(_SecretLeaf(shared="abc", token="t-secret"))
        assert "t-secret" not in rendered
        assert "token='***'" in rendered
        assert "shared='abc'" in rendered  # inherited field still present


class TestNestedMappingRedaction:
    """``repr`` descends into raw ``Mapping``/``list`` fields, masking values
    under the default sensitive-key set unioned with ``_SENSITIVE_FIELDS``.

    These cover the leak that field-name redaction alone could not reach: a
    credential nested inside an intentionally-untyped polymorphic section
    (``vector_store``/``embedding``/``llm``) printed in cleartext via
    ``repr`` before the typed child was constructed. Display-only — the
    real nested value and ``to_dict()`` are untouched.
    """

    def test_repr_masks_connection_string_in_raw_mapping(self) -> None:
        """The default set covers ``connection_string`` with zero per-class
        config (the pgvector ``vector_store`` leak)."""
        cfg = _WithRawMapping(
            section={
                "backend": "pgvector",
                "connection_string": "postgresql://u:pw@h/db",
            }
        )
        rendered = repr(cfg)
        assert "pw" not in rendered
        assert "postgresql://" not in rendered
        assert "'connection_string': '***'" in rendered
        # The discriminator and non-secret keys still render.
        assert "'backend': 'pgvector'" in rendered

    def test_repr_masks_api_key_in_raw_mapping(self) -> None:
        """The embedder ``api_key`` leak (default set, no ``_SENSITIVE_FIELDS``)."""
        cfg = _WithRawMapping(section={"provider": "openai", "api_key": "sk-secret"})
        rendered = repr(cfg)
        assert "sk-secret" not in rendered
        assert "'api_key': '***'" in rendered

    def test_repr_masks_secret_nested_two_levels_deep(self) -> None:
        """Descent reaches a mapping nested inside another mapping."""
        cfg = _WithRawMapping(section={"outer": {"password": "hunter2"}})
        rendered = repr(cfg)
        assert "hunter2" not in rendered
        assert "'password': '***'" in rendered

    def test_repr_masks_secret_in_mapping_inside_list(self) -> None:
        """A mapping inside a ``list`` field is reached (e.g. a strategies
        list of raw dicts)."""
        cfg = _WithRawMapping(
            items=[{"connection_string": "postgresql://u:pw@h/db"}]
        )
        rendered = repr(cfg)
        assert "pw" not in rendered
        assert "'connection_string': '***'" in rendered

    def test_exact_key_match_does_not_over_redact(self) -> None:
        """A benign key that merely *contains* a sensitive substring
        (``token_count`` contains ``token``) is NOT masked — match is exact,
        not substring."""
        cfg = _WithRawMapping(section={"token_count": 5, "model": "llama3.2"})
        rendered = repr(cfg)
        assert "'token_count': 5" in rendered
        assert "'model': 'llama3.2'" in rendered
        assert "***" not in rendered

    def test_case_insensitive_key_match(self) -> None:
        """Interior keys match case-insensitively (``API_KEY`` == ``api_key``)."""
        cfg = _WithRawMapping(section={"API_KEY": "sk-secret"})
        rendered = repr(cfg)
        assert "sk-secret" not in rendered
        assert "'API_KEY': '***'" in rendered

    def test_falsy_secret_value_not_masked(self) -> None:
        """An empty/absent nested credential renders verbatim — masking it
        would falsely imply one is configured (mirrors the scalar rule)."""
        cfg = _WithRawMapping(section={"api_key": "", "connection_string": None})
        rendered = repr(cfg)
        assert "'api_key': ''" in rendered
        assert "'connection_string': None" in rendered
        assert "***" not in rendered

    def test_class_sensitive_fields_extend_interior_keys(self) -> None:
        """A class's ``_SENSITIVE_FIELDS`` names also mask matching interior
        keys, beyond the module default set."""

        @dataclass(frozen=True)
        class _CustomInterior(StructuredConfig):
            section: dict[str, Any] = field(default_factory=dict)
            _SENSITIVE_FIELDS: ClassVar[frozenset[str]] = frozenset({"custom_key"})

        cfg = _CustomInterior(section={"custom_key": "v", "api_key": "sk"})
        rendered = repr(cfg)
        assert "'custom_key': '***'" in rendered  # class-declared interior key
        assert "'api_key': '***'" in rendered  # default-set interior key

    def test_depth_bound_terminates_without_error(self) -> None:
        """A structure nested deeper than the bound does not raise and
        terminates (the bound is a safety stop, not an expected path)."""
        nested: dict[str, Any] = {"password": "deep-secret"}
        for _ in range(12):
            nested = {"level": nested}
        cfg = _WithRawMapping(section=nested)
        rendered = repr(cfg)  # must not raise / recurse unbounded
        assert isinstance(rendered, str)

    def test_to_dict_keeps_real_nested_secret(self) -> None:
        """Display-only: ``to_dict`` returns the real nested value verbatim."""
        cfg = _WithRawMapping(
            section={"connection_string": "postgresql://u:pw@h/db"}
        )
        assert (
            cfg.to_dict()["section"]["connection_string"]
            == "postgresql://u:pw@h/db"
        )

    def test_roundtrip_preserved_with_nested_secret(self) -> None:
        """The nested-mapping round-trip still holds — repr descent is not
        serialization."""
        cfg = _WithRawMapping(
            section={"connection_string": "postgresql://u:pw@h/db"},
            items=[{"api_key": "sk-secret"}],
        )
        assert_structured_config_roundtrip(cfg)

    def test_no_secret_mapping_renders_verbatim(self) -> None:
        """A config whose mappings hold no sensitive keys renders byte-for-byte
        as the plain-dataclass repr would (descent is safe, not lossy)."""
        cfg = _WithRawMapping(
            label="y",
            section={"backend": "memory", "dimension": 384},
            items=[{"weight": 0.5}],
        )
        assert repr(cfg) == (
            "_WithRawMapping(label='y', "
            "section={'backend': 'memory', 'dimension': 384}, "
            "items=[{'weight': 0.5}])"
        )
