"""Tests for the vocabulary-free pack composition core.

The subject is pure data (frozen spec records and a fold over them), so no
test doubles are involved at all — every spec below is a real ``PackSpec``
subclass and every registry is a real ``PackRegistry``.

``DemoPack`` exercises all six built-in merge kinds in one spec so the
ordering, warning, and conflict paths are covered against a single shape.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import gc
import pathlib
import pickle
import re
import weakref
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import pytest

from dataknobs_common.exceptions import ConfigurationError, OperationError
from dataknobs_common.packs import (
    UNSET,
    MergeKind,
    PackRegistry,
    PackResolution,
    PackResolutionError,
    PackSpec,
    PackWarning,
    compose_packs,
    merge_bindings,
)
from dataknobs_common.registry import BackendRegistry
from dataknobs_common.testing import assert_structured_config_roundtrip


@dataclass(frozen=True)
class DemoPack(PackSpec):
    """A spec covering every built-in merge kind."""

    tier: str | None = None
    pinned: str | None = None
    mode: str | None = None
    checks: tuple[str, ...] = ()
    steps: tuple[dict[str, Any], ...] = ()
    limits: dict[str, int] = field(default_factory=dict)

    _COMPOSITION = MappingProxyType(
        {
            "tier": MergeKind.UNANIMOUS,
            "pinned": MergeKind.FIRST_WINS,
            "mode": MergeKind.LAST_WINS,
            "checks": MergeKind.CONCAT_UNIQUE,
            "steps": MergeKind.CONCAT,
            "limits": MergeKind.MERGE,
        }
    )


@pytest.fixture
def registry() -> PackRegistry[DemoPack]:
    """Three packs at distinct priorities, registered out of priority order."""
    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)
    reg.register_pack(
        DemoPack(
            name="base",
            priority=0,
            tier="std",
            checks=("pii",),
            limits={"rps": 10, "burst": 5},
        )
    )
    reg.register_pack(
        DemoPack(
            name="mid",
            priority=5,
            checks=("pii", "toxicity"),
            steps=({"class": "M"},),
        )
    )
    reg.register_pack(
        DemoPack(
            name="top",
            priority=10,
            tier="std",
            limits={"rps": 2},
            steps=({"class": "T"},),
        )
    )
    return reg


def _codes(resolution: PackResolution[Any]) -> list[str]:
    return [warning.code for warning in resolution.warnings]


# --------------------------------------------------------------------------
# 1. ordering and the collection merge kinds
# --------------------------------------------------------------------------


def test_distinct_priorities_fold_in_ascending_order(registry: PackRegistry[DemoPack]) -> None:
    """Binding order is irrelevant: the fold runs low priority -> high."""
    resolution = registry.resolve({"top": {}, "base": {}, "mid": {}})

    assert resolution.packs == ("base", "mid", "top")
    # CONCAT keeps every contribution, flattened in fold order.
    assert resolution.spec.steps == ({"class": "M"}, {"class": "T"})
    # CONCAT_UNIQUE keeps first-seen order and drops the repeat of "pii".
    assert resolution.spec.checks == ("pii", "toxicity")
    # MERGE lets the highest-priority pack win per key, keeping the rest.
    assert resolution.spec.limits == {"rps": 2, "burst": 5}


def test_composed_spec_carries_the_composed_name(registry: PackRegistry[DemoPack]) -> None:
    resolution = registry.resolve({"base": {}}, composed_name="effective")

    assert resolution.spec.name == "effective"
    assert resolution.spec.priority == 0


# --------------------------------------------------------------------------
# 2. equal priority -> FIFO plus a diagnostic
# --------------------------------------------------------------------------


def test_priority_tie_falls_back_to_registration_order_and_warns() -> None:
    reg: PackRegistry[DemoPack] = PackRegistry("tie", DemoPack)
    reg.register_pack(DemoPack(name="first", priority=3, checks=("a",)))
    reg.register_pack(DemoPack(name="second", priority=3, checks=("b",)))

    # Bound in the opposite order to prove binding order does not decide it.
    resolution = reg.resolve({"second": {}, "first": {}})

    assert resolution.packs == ("first", "second")
    assert resolution.spec.checks == ("a", "b")
    assert _codes(resolution) == ["priority_tie"]
    assert resolution.warnings[0].packs == ("first", "second")


# --------------------------------------------------------------------------
# 3 + 4. the locked / enabled safety contract
# --------------------------------------------------------------------------


def test_disabling_a_locked_pack_raises(registry: PackRegistry[DemoPack]) -> None:
    with pytest.raises(PackResolutionError) as excinfo:
        registry.resolve({"base": {"locked": True, "enabled": False}})

    assert excinfo.value.reason == "locked_pack_disabled"
    assert excinfo.value.context["reason"] == "locked_pack_disabled"


def test_locked_alone_resolves_and_disabled_alone_excludes(
    registry: PackRegistry[DemoPack],
) -> None:
    """Over-trigger guard: neither flag alone may raise."""
    assert registry.resolve({"base": {"locked": True}}).packs == ("base",)
    assert registry.resolve({"base": {"enabled": False}}).packs == ()
    assert registry.resolve({"base": {"locked": False, "enabled": False}}).packs == ()


# --------------------------------------------------------------------------
# 5. UNANIMOUS
# --------------------------------------------------------------------------


def test_unanimous_conflict_across_enabled_packs_raises() -> None:
    reg: PackRegistry[DemoPack] = PackRegistry("conflict", DemoPack)
    reg.register_pack(DemoPack(name="a", tier="std"))
    reg.register_pack(DemoPack(name="b", priority=1, tier="pro"))

    with pytest.raises(PackResolutionError) as excinfo:
        reg.resolve({"a": {}, "b": {}})

    assert excinfo.value.reason == "field_conflict"


def test_unanimous_agreeing_values_reconcile(registry: PackRegistry[DemoPack]) -> None:
    """"base" and "top" both declare tier="std"; "mid" leaves it at default."""
    assert registry.resolve({"base": {}, "mid": {}, "top": {}}).spec.tier == "std"


def test_unanimous_reconciles_when_only_one_pack_participates() -> None:
    reg: PackRegistry[DemoPack] = PackRegistry("conflict", DemoPack)
    reg.register_pack(DemoPack(name="a", tier="std"))
    reg.register_pack(DemoPack(name="b", priority=1, tier="pro"))

    # A field left at its declared default does not participate, so the single
    # non-default contribution wins outright.
    assert reg.resolve({"b": {}}).spec.tier == "pro"


# --------------------------------------------------------------------------
# 6 + 10 + 11. fail-closed binding validation
# --------------------------------------------------------------------------


def test_unknown_pack_name_raises(registry: PackRegistry[DemoPack]) -> None:
    with pytest.raises(PackResolutionError) as excinfo:
        registry.resolve({"nope": {}})

    assert excinfo.value.reason == "unknown_pack"


@pytest.mark.parametrize(
    ("binding", "reason"),
    [
        ({"lockd": True}, "unknown_binding_key"),
        ({"name": "other"}, "unknown_binding_key"),
        ({"enabled": "yes"}, "invalid_binding"),
        ({"locked": 1}, "invalid_binding"),
        ({"priority": "high"}, "invalid_binding"),
    ],
)
def test_malformed_binding_bodies_raise(
    registry: PackRegistry[DemoPack], binding: dict[str, Any], reason: str
) -> None:
    with pytest.raises(PackResolutionError) as excinfo:
        registry.resolve({"base": binding})

    assert excinfo.value.reason == reason


def test_non_mapping_binding_body_raises(registry: PackRegistry[DemoPack]) -> None:
    with pytest.raises(PackResolutionError) as excinfo:
        registry.resolve({"base": None})  # type: ignore[dict-item]

    assert excinfo.value.reason == "invalid_binding"


# --------------------------------------------------------------------------
# 7. binding overrides
# --------------------------------------------------------------------------


def test_binding_priority_beats_spec_priority(registry: PackRegistry[DemoPack]) -> None:
    resolution = registry.resolve({"base": {"priority": 99}, "top": {}})

    assert resolution.packs == ("top", "base")


def test_binding_field_override_composes_over_its_own_spec(
    registry: PackRegistry[DemoPack],
) -> None:
    """A binding is a partial spec folded over its pack under the same rules."""
    resolution = registry.resolve({"base": {"limits": {"rps": 1}}})

    assert resolution.spec.limits == {"rps": 1, "burst": 5}


def test_binding_may_reassert_but_not_change_a_unanimous_field(
    registry: PackRegistry[DemoPack],
) -> None:
    assert registry.resolve({"base": {"tier": "std"}}).spec.tier == "std"

    with pytest.raises(PackResolutionError) as excinfo:
        registry.resolve({"base": {"tier": "other"}})

    assert excinfo.value.reason == "field_conflict"


def test_binding_overrides_do_not_emit_warnings(registry: PackRegistry[DemoPack]) -> None:
    """A deployment overriding its own pack is intentional, not a diagnostic."""
    assert registry.resolve({"base": {"limits": {"rps": 1}}}).warnings == ()


def test_binding_can_reset_a_field_to_its_default() -> None:
    """A binding names fields explicitly, so naming one IS the contribution.

    A spec is a frozen dataclass — every field is always present, so "unset"
    can only be inferred from the value. A binding body is a partial mapping
    where presence is unambiguous, and discarding that in favour of the same
    default-comparison silently ignored an explicit operator instruction.
    """
    reg: PackRegistry[DemoPack] = PackRegistry("reset", DemoPack)
    reg.register_pack(DemoPack(name="p", mode="strict"))

    assert reg.resolve({"p": {}}).spec.mode == "strict"
    assert reg.resolve({"p": {"mode": None}}).spec.mode is None


def test_binding_reset_still_obeys_the_fields_declared_rule() -> None:
    """Presence decides *participation*, not the outcome — the rule decides that.

    ``FIRST_WINS`` and ``UNANIMOUS`` exist precisely to stop a later
    contribution from changing a value, and a binding is a later
    contribution like any other.
    """
    reg: PackRegistry[DemoPack] = PackRegistry("reset_rules", DemoPack)
    reg.register_pack(DemoPack(name="p", pinned="held", tier="std"))

    # FIRST_WINS: the pack's value came first and keeps winning.
    assert reg.resolve({"p": {"pinned": None}}).spec.pinned == "held"

    # UNANIMOUS: clearing is a disagreement, which is unsatisfiable.
    with pytest.raises(PackResolutionError) as excinfo:
        reg.resolve({"p": {"tier": None}})
    assert excinfo.value.reason == "field_conflict"


def test_a_pack_leaving_a_field_at_its_default_still_does_not_participate() -> None:
    """The spec-side rule is unchanged: a pack cannot clobber by omission."""
    reg: PackRegistry[DemoPack] = PackRegistry("omission", DemoPack)
    reg.register_pack(DemoPack(name="low", priority=0, mode="strict"))
    reg.register_pack(DemoPack(name="high", priority=10))  # mode left at default

    assert reg.resolve({"low": {}, "high": {}}).spec.mode == "strict"


# --------------------------------------------------------------------------
# 8. purity
# --------------------------------------------------------------------------


def test_resolve_is_pure_and_deterministic(registry: PackRegistry[DemoPack]) -> None:
    bindings = {"top": {"limits": {"rps": 3}}, "base": {}, "mid": {}}
    snapshot = {"top": {"limits": {"rps": 3}}, "base": {}, "mid": {}}
    keys_before = registry.list_keys()
    base_before = dataclasses.replace(registry.get("base"))

    first = registry.resolve(bindings)
    second = registry.resolve(bindings)

    assert first == second
    assert bindings == snapshot, "resolve() mutated the caller's bindings"
    assert registry.list_keys() == keys_before, "resolve() mutated the registry"
    assert registry.get("base") == base_before, "resolve() mutated a registered spec"


def test_resolve_does_not_alias_the_binding_override_mapping(
    registry: PackRegistry[DemoPack],
) -> None:
    override = {"rps": 1}
    resolution = registry.resolve({"base": {"limits": override}})

    override["rps"] = 999

    assert resolution.spec.limits["rps"] == 1


# --------------------------------------------------------------------------
# 9. empty bindings
# --------------------------------------------------------------------------


def test_empty_bindings_produce_an_all_default_spec(registry: PackRegistry[DemoPack]) -> None:
    resolution = registry.resolve({})

    assert resolution.applied == ()
    assert resolution.packs == ()
    assert resolution.warnings == ()
    assert resolution.spec.checks == ()
    assert resolution.spec.limits == {}
    assert resolution.spec.tier is None


# --------------------------------------------------------------------------
# 12. composition-plan validation (declaration errors, not resolution errors)
# --------------------------------------------------------------------------


def _assert_declaration_error(exc: ConfigurationError) -> None:
    """Declaration problems use the plain error, not the resolution family."""
    assert not isinstance(exc, PackResolutionError)


def test_field_without_a_composition_rule_is_rejected_at_construction() -> None:
    @dataclass(frozen=True)
    class NoRule(PackSpec):
        orphan: str | None = None

    with pytest.raises(ConfigurationError) as excinfo:
        PackRegistry("bad", NoRule)

    _assert_declaration_error(excinfo.value)
    assert "orphan" in str(excinfo.value)


def test_composition_key_that_is_not_a_field_is_rejected() -> None:
    @dataclass(frozen=True)
    class GhostRule(PackSpec):
        real: str | None = None

        _COMPOSITION = MappingProxyType(
            {"real": MergeKind.LAST_WINS, "ghost": MergeKind.LAST_WINS}
        )

    with pytest.raises(ConfigurationError) as excinfo:
        PackRegistry("bad", GhostRule)

    _assert_declaration_error(excinfo.value)
    assert "ghost" in str(excinfo.value)


def test_field_without_a_default_is_rejected() -> None:
    # A defaultless field after a defaulted one is a TypeError from dataclasses
    # itself, so kw_only is the only way to declare one -- and therefore the
    # only way this check is reachable.
    @dataclass(frozen=True, kw_only=True)
    class Undefaulted(PackSpec):
        needy: str

        _COMPOSITION = MappingProxyType({"needy": MergeKind.LAST_WINS})

    with pytest.raises(ConfigurationError) as excinfo:
        PackRegistry("bad", Undefaulted)

    _assert_declaration_error(excinfo.value)
    assert "needy" in str(excinfo.value)


def test_rule_that_is_neither_a_merge_kind_nor_callable_is_rejected() -> None:
    @dataclass(frozen=True)
    class BadRule(PackSpec):
        thing: str | None = None

        _COMPOSITION = MappingProxyType({"thing": "not-a-rule"})

    with pytest.raises(ConfigurationError) as excinfo:
        PackRegistry("bad", BadRule)

    _assert_declaration_error(excinfo.value)
    assert "thing" in str(excinfo.value)


def test_plan_validation_reports_every_problem_at_once() -> None:
    @dataclass(frozen=True)
    class ManyProblems(PackSpec):
        orphan: str | None = None
        bad: str | None = None

        _COMPOSITION = MappingProxyType({"bad": "nope", "ghost": MergeKind.LAST_WINS})

    with pytest.raises(ConfigurationError) as excinfo:
        PackRegistry("bad", ManyProblems)

    message = str(excinfo.value)
    assert "orphan" in message
    assert "ghost" in message
    assert "bad" in message


def test_value_whose_shape_contradicts_its_rule_is_rejected() -> None:
    @dataclass(frozen=True)
    class ShapePack(PackSpec):
        items: tuple[str, ...] = ()

        _COMPOSITION = MappingProxyType({"items": MergeKind.CONCAT})

    reg: PackRegistry[ShapePack] = PackRegistry("shape", ShapePack)
    # A bare string is a Sequence; concatenating it would silently explode it
    # into characters, so it is rejected instead.
    reg.register_pack(ShapePack(name="s", items="oops"))  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError) as excinfo:
        reg.resolve({"s": {}})

    _assert_declaration_error(excinfo.value)


# --------------------------------------------------------------------------
# 13. FIRST_WINS
# --------------------------------------------------------------------------


def test_first_wins_lets_a_lower_priority_pack_pin_a_value() -> None:
    reg: PackRegistry[DemoPack] = PackRegistry("pin", DemoPack)
    reg.register_pack(DemoPack(name="low", priority=0, pinned="locked-in"))
    reg.register_pack(DemoPack(name="high", priority=10, pinned="ignored"))

    resolution = reg.resolve({"low": {}, "high": {}})

    assert resolution.spec.pinned == "locked-in"
    assert _codes(resolution) == ["value_override"]


# --------------------------------------------------------------------------
# 14. the callable escape hatch
# --------------------------------------------------------------------------


def test_callable_reducer_is_honored() -> None:
    """Any ``(base, override) -> value`` callable may stand in for a kind."""

    def sum_ints(base: Any, override: Any) -> Any:
        return base + override

    @dataclass(frozen=True)
    class TallyPack(PackSpec):
        total: int = 0

        _COMPOSITION = MappingProxyType({"total": sum_ints})

    reg: PackRegistry[TallyPack] = PackRegistry("tally", TallyPack)
    reg.register_pack(TallyPack(name="a", total=2))
    reg.register_pack(TallyPack(name="b", priority=1, total=3))

    assert reg.resolve({"a": {}, "b": {}}).spec.total == 5


def test_callable_hatch_accepts_the_config_packages_deep_merge() -> None:
    """The documented recursive-merge recipe, pinned against the real function.

    ``dataknobs-common`` has no dependencies, so this import is guarded --
    the core must never require the sibling package, only accept its callable.
    """
    pytest.importorskip("dataknobs_config")
    from dataknobs_config import deep_merge

    @dataclass(frozen=True)
    class DeepPack(PackSpec):
        settings: dict[str, Any] = field(default_factory=dict)

        _COMPOSITION = MappingProxyType({"settings": deep_merge})

    reg: PackRegistry[DeepPack] = PackRegistry("deep", DeepPack)
    reg.register_pack(DeepPack(name="a", settings={"llm": {"model": "x", "temp": 0.1}}))
    reg.register_pack(DeepPack(name="b", priority=1, settings={"llm": {"temp": 0.9}}))

    resolution = reg.resolve({"a": {}, "b": {}})

    # Nested, unlike MergeKind.MERGE which replaces the whole "llm" value.
    assert resolution.spec.settings == {"llm": {"model": "x", "temp": 0.9}}


# --------------------------------------------------------------------------
# 15. override diagnostics
# --------------------------------------------------------------------------


def test_differing_values_emit_override_warnings() -> None:
    reg: PackRegistry[DemoPack] = PackRegistry("warn", DemoPack)
    reg.register_pack(DemoPack(name="a", mode="fast", limits={"rps": 10}))
    reg.register_pack(DemoPack(name="b", priority=1, mode="slow", limits={"rps": 2}))

    resolution = reg.resolve({"a": {}, "b": {}})

    assert resolution.spec.mode == "slow"
    assert sorted(_codes(resolution)) == ["key_override", "value_override"]
    by_code = {warning.code: warning for warning in resolution.warnings}
    assert by_code["value_override"].field == "mode"
    assert by_code["key_override"].field == "limits"
    assert by_code["value_override"].packs == ("a", "b")


def test_equal_values_emit_no_warning() -> None:
    reg: PackRegistry[DemoPack] = PackRegistry("quiet", DemoPack)
    reg.register_pack(DemoPack(name="a", mode="fast", limits={"rps": 1}))
    reg.register_pack(DemoPack(name="b", priority=1, mode="fast", limits={"rps": 1}))

    assert reg.resolve({"a": {}, "b": {}}).warnings == ()


def test_pack_warning_str_is_the_message() -> None:
    assert str(PackWarning(code="c", message="hello")) == "hello"


# --------------------------------------------------------------------------
# 16 + 17. spec normalization and round-trip
# --------------------------------------------------------------------------


def test_from_dict_normalizes_lists_to_tuples() -> None:
    spec = DemoPack.from_dict(
        {"name": "loaded", "checks": ["a", "b"], "steps": [{"class": "X"}]}
    )

    assert spec.checks == ("a", "b")
    assert isinstance(spec.checks, tuple)
    assert isinstance(spec.steps, tuple)


def test_spec_does_not_alias_a_caller_supplied_mapping() -> None:
    raw = {"rps": 10}
    spec = DemoPack.from_dict({"name": "loaded", "limits": raw})

    raw["rps"] = 999

    assert spec.limits == {"rps": 10}


def test_spec_round_trips_through_dict() -> None:
    spec = DemoPack.from_dict(
        {
            "name": "loaded",
            "priority": 4,
            "tier": "std",
            "checks": ["a", "b"],
            "steps": [{"class": "X"}],
            "limits": {"rps": 10},
        }
    )

    assert_structured_config_roundtrip(spec)


# --------------------------------------------------------------------------
# 18 + 19. registry contract
# --------------------------------------------------------------------------


def test_register_pack_keys_off_the_spec_name(registry: PackRegistry[DemoPack]) -> None:
    assert set(registry.list_keys()) == {"base", "mid", "top"}
    assert registry.get("base").name == "base"


def test_register_rejects_a_key_that_disagrees_with_the_spec_name(
    registry: PackRegistry[DemoPack],
) -> None:
    with pytest.raises(ConfigurationError):
        registry.register("x", DemoPack(name="y"))


def test_register_rejects_a_foreign_spec_type(registry: PackRegistry[DemoPack]) -> None:
    @dataclass(frozen=True)
    class OtherPack(PackSpec):
        pass

    with pytest.raises(ConfigurationError):
        registry.register("s", OtherPack(name="s"))  # type: ignore[arg-type]


def test_register_pack_rejects_a_duplicate_name(registry: PackRegistry[DemoPack]) -> None:
    with pytest.raises(OperationError):
        registry.register_pack(DemoPack(name="base"))

    registry.register_pack(DemoPack(name="base", tier="std"), allow_overwrite=True)
    assert registry.get("base").checks == ()


def test_registry_satisfies_the_backend_registry_protocol(
    registry: PackRegistry[DemoPack],
) -> None:
    assert isinstance(registry, BackendRegistry)
    assert registry.spec_cls is DemoPack


# --------------------------------------------------------------------------
# compose_packs used directly, without a registry
# --------------------------------------------------------------------------


def test_compose_packs_folds_in_the_given_order() -> None:
    composed, warnings = compose_packs(
        [DemoPack(name="a", checks=("x",)), DemoPack(name="b", checks=("y",))],
        DemoPack,
        composed_name="direct",
    )

    assert composed.name == "direct"
    assert composed.checks == ("x", "y")
    assert warnings == ()


def test_compose_packs_does_not_sort_by_priority() -> None:
    """Ordering is the caller's job; the registry is what sorts."""
    composed, _ = compose_packs(
        [DemoPack(name="a", priority=9, checks=("x",)), DemoPack(name="b", checks=("y",))],
        DemoPack,
    )

    assert composed.checks == ("x", "y")


def test_compose_packs_rejects_a_foreign_spec() -> None:
    @dataclass(frozen=True)
    class OtherPack(PackSpec):
        pass

    with pytest.raises(ConfigurationError):
        compose_packs([OtherPack(name="s")], DemoPack)  # type: ignore[list-item]


def test_compose_packs_of_nothing_is_the_default_spec() -> None:
    composed, warnings = compose_packs([], DemoPack)

    assert composed == DemoPack(name="composed")
    assert warnings == ()


# --------------------------------------------------------------------------
# 21. the UNSET sentinel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SentinelPack(PackSpec):
    """A spec whose scalar fields use ``UNSET`` rather than ``None``.

    Declaring the sentinel as the default is what makes ``None`` an ordinary
    participating value: participation compares against *the declared
    default*, so with ``UNSET`` there the domain's own values — ``None``
    included — all differ from it.
    """

    mode: str | None = UNSET
    tier: str | None = UNSET

    _COMPOSITION = MappingProxyType(
        {"mode": MergeKind.LAST_WINS, "tier": MergeKind.UNANIMOUS}
    )


def test_unset_default_lets_a_higher_pack_contribute_none() -> None:
    """The recipe the guide documents, exercised end to end."""
    reg: PackRegistry[SentinelPack] = PackRegistry("sentinel", SentinelPack)
    reg.register_pack(SentinelPack(name="base", priority=0, mode="strict"))
    reg.register_pack(SentinelPack(name="off", priority=10, mode=None))
    reg.register_pack(SentinelPack(name="quiet", priority=10))

    # An explicit None now outranks a lower pack's value...
    assert reg.resolve({"base": {}, "off": {}}).spec.mode is None
    # ...while genuine silence still does not.
    assert reg.resolve({"base": {}, "quiet": {}}).spec.mode == "strict"


def test_unset_survives_a_field_nothing_ever_set() -> None:
    """The cost of the recipe: untouched fields read back as ``UNSET``."""
    reg: PackRegistry[SentinelPack] = PackRegistry("sentinel_gap", SentinelPack)
    reg.register_pack(SentinelPack(name="only", mode="strict"))

    assert reg.resolve({"only": {}}).spec.tier is UNSET


def test_unset_is_falsy_so_or_default_idioms_work() -> None:
    assert not UNSET
    assert (UNSET or "fallback") == "fallback"
    assert repr(UNSET) == "UNSET"


def test_unset_is_identity_stable_across_copy_and_pickle() -> None:
    """Identity is the whole contract — ``is UNSET`` must survive round-trips."""
    assert copy.copy(UNSET) is UNSET
    assert copy.deepcopy(UNSET) is UNSET
    assert pickle.loads(pickle.dumps(UNSET)) is UNSET
    assert dataclasses.replace(SentinelPack(name="s")).tier is UNSET


def test_unset_spec_round_trips_through_dict() -> None:
    assert_structured_config_roundtrip(SentinelPack(name="s", mode="strict"))


# --------------------------------------------------------------------------
# Reconstruction fidelity — a spec must survive resolution intact
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TenantDemoPack(DemoPack):
    """A registrable subclass carrying a field the base plan cannot see."""

    tenant: str | None = None

    _COMPOSITION = MappingProxyType(
        {**DemoPack._COMPOSITION, "tenant": MergeKind.LAST_WINS}
    )


@dataclass(frozen=True)
class RegionPack(PackSpec):
    """A spec that extends ``_META_FIELDS`` with a third descriptor."""

    region: str = "us"
    mode: str | None = None

    _META_FIELDS = frozenset({"name", "priority", "region"})
    _COMPOSITION = MappingProxyType({"mode": MergeKind.LAST_WINS})


def test_subclass_adding_no_fields_is_still_registrable() -> None:
    """The guard is about uncomposable *fields*, not about subclassing.

    A subclass that only adds behaviour composes identically to its base,
    so rejecting it would be a blanket ban where a precise one is possible.
    """

    @dataclass(frozen=True)
    class BehaviourOnly(DemoPack):
        def describe(self) -> str:
            return f"{self.name}:{self.mode}"

    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)
    reg.register_pack(BehaviourOnly(name="ext", mode="strict"))

    resolution = reg.resolve({"ext": {"mode": "lax"}})

    assert resolution.applied[0].mode == "lax"


def test_binding_application_preserves_extra_meta_fields() -> None:
    """A third ``_META_FIELDS`` entry must survive binding application."""
    reg: PackRegistry[RegionPack] = PackRegistry("regions", RegionPack)
    reg.register_pack(RegionPack(name="eu", region="eu", mode="strict"))

    resolution = reg.resolve({"eu": {"mode": "lax"}})

    assert resolution.applied[0].region == "eu"


def test_binding_naming_an_unhandled_meta_field_is_rejected() -> None:
    """``binding_keys`` must not advertise a meta field the fold ignores."""
    reg: PackRegistry[RegionPack] = PackRegistry("regions", RegionPack)
    reg.register_pack(RegionPack(name="eu", region="eu"))

    with pytest.raises(PackResolutionError) as excinfo:
        reg.resolve({"eu": {"region": "apac"}})

    assert excinfo.value.reason == "unknown_binding_key"


def test_registering_a_subclass_with_uncomposable_fields_is_rejected() -> None:
    """The composed spec cannot carry a field the plan has no rule for.

    Registration is the earliest point this is knowable, so it fails there
    rather than silently dropping the field from ``resolution.spec``.
    """
    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)

    with pytest.raises(ConfigurationError) as excinfo:
        reg.register_pack(TenantDemoPack(name="ext", tenant="acme"))

    assert "tenant" in str(excinfo.value)


def test_compose_packs_rejects_a_subclass_with_uncomposable_fields() -> None:
    """The same guard on the lower-level primitive."""
    with pytest.raises(ConfigurationError) as excinfo:
        compose_packs([TenantDemoPack(name="ext", tenant="acme")], DemoPack)

    assert "tenant" in str(excinfo.value)


# --------------------------------------------------------------------------
# Binding overrides that the declared rule discards
# --------------------------------------------------------------------------


def test_discarded_binding_override_is_surfaced_as_a_warning() -> None:
    """A FIRST_WINS field pinned by the pack silently ignores its binding.

    The binding key was typed by an operator and had no effect; without a
    diagnostic that is indistinguishable from it having been honoured.
    """
    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)
    reg.register_pack(DemoPack(name="base", pinned="a"))

    resolution = reg.resolve({"base": {"pinned": "b"}})

    assert resolution.applied[0].pinned == "a"
    codes = {w.code for w in resolution.warnings}
    assert "binding_override_ignored" in codes
    ignored = next(w for w in resolution.warnings if w.code == "binding_override_ignored")
    assert ignored.field == "pinned"


def test_honored_binding_override_emits_no_warning() -> None:
    """A binding that wins is an intentional deployment act, not a diagnostic."""
    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)
    reg.register_pack(DemoPack(name="base", mode="strict"))

    resolution = reg.resolve({"base": {"mode": "lax"}})

    assert resolution.applied[0].mode == "lax"
    assert resolution.warnings == ()


# --------------------------------------------------------------------------
# Error family for operator-supplied values
# --------------------------------------------------------------------------


def test_binding_supplied_bad_shape_raises_the_resolution_family() -> None:
    """A bad value in a binding is operator input, not an authoring bug.

    It must be catchable through the documented
    ``except PackResolutionError as exc: exc.reason`` recipe.
    """
    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)
    reg.register_pack(DemoPack(name="base", checks=("pii",)))

    with pytest.raises(PackResolutionError) as excinfo:
        reg.resolve({"base": {"checks": "pii"}})

    assert excinfo.value.reason == "invalid_binding"
    assert excinfo.value.context["field"] == "checks"


def test_spec_supplied_bad_shape_still_raises_the_authoring_family() -> None:
    """The spec-sourced half of the split is unchanged."""
    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)
    reg.register_pack(DemoPack(name="base", checks="pii"))  # type: ignore[arg-type]

    with pytest.raises(ConfigurationError) as excinfo:
        reg.resolve({"base": {}})

    assert not isinstance(excinfo.value, PackResolutionError)


# --------------------------------------------------------------------------
# Spec-class validation reachability
# --------------------------------------------------------------------------


def test_undecorated_subclass_is_named_as_the_problem() -> None:
    """``is_dataclass`` is true for any subclass, so the guard needs the own-dict.

    Without it the failure surfaces as "keys that are not non-meta fields",
    pointing the author at ``_COMPOSITION`` rather than the missing
    decorator.
    """

    class Undecorated(PackSpec):
        mode: str | None = None

        _COMPOSITION = MappingProxyType({"mode": MergeKind.LAST_WINS})

    with pytest.raises(ConfigurationError) as excinfo:
        PackRegistry("undecorated", Undecorated)

    assert "@dataclass(frozen=True)" in str(excinfo.value)


# --------------------------------------------------------------------------
# Custom reducer failures
# --------------------------------------------------------------------------


def test_failing_custom_reducer_is_wrapped_with_field_context() -> None:
    """A reducer given a shape it cannot handle must not surface raw.

    ``_check_value_shape`` exists so the built-in kinds never raise an
    opaque ``TypeError`` from inside the fold; a custom reducer has no
    such check, so the wrap is what supplies the same context.
    """

    def add(acc: Any, nxt: Any) -> Any:
        return acc + nxt

    @dataclass(frozen=True)
    class ReducerPack(PackSpec):
        total: Any = 0

        _COMPOSITION = MappingProxyType({"total": add})

    reg: PackRegistry[ReducerPack] = PackRegistry("reducers", ReducerPack)
    reg.register_pack(ReducerPack(name="a", priority=0, total=1))
    reg.register_pack(ReducerPack(name="b", priority=1, total="oops"))

    with pytest.raises(ConfigurationError) as excinfo:
        reg.resolve({"a": {}, "b": {}})

    message = str(excinfo.value)
    assert "total" in message
    assert "b" in message


# --------------------------------------------------------------------------
# Priority ties that cannot matter
# --------------------------------------------------------------------------


def test_priority_tie_on_disjoint_fields_does_not_warn() -> None:
    """Two packs at the default priority touching different fields.

    ``priority`` defaults to 0, so warning on every tie fires in the
    ordinary case and trains consumers to filter the code out.
    """
    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)
    reg.register_pack(DemoPack(name="a", mode="strict"))
    reg.register_pack(DemoPack(name="b", pinned="x"))

    resolution = reg.resolve({"a": {}, "b": {}})

    assert [w.code for w in resolution.warnings] == []


def test_priority_tie_on_a_shared_field_still_warns() -> None:
    """The case the diagnostic exists for: order decides the outcome."""
    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)
    reg.register_pack(DemoPack(name="a", mode="strict"))
    reg.register_pack(DemoPack(name="b", mode="lax"))

    resolution = reg.resolve({"a": {}, "b": {}})

    assert "priority_tie" in {w.code for w in resolution.warnings}


# --------------------------------------------------------------------------
# Immutability of registered specs
# --------------------------------------------------------------------------


def test_set_field_is_immutable_by_construction() -> None:
    """``set`` normalizes to ``frozenset``, matching ``list`` -> ``tuple``.

    Mappings deliberately stay writable — ``MappingProxyType`` cannot be
    deep-copied, and ``StructuredConfig.to_dict`` delegates to
    ``dataclasses.asdict``, which deep-copies every field value. The
    read-only contract for dict fields is therefore documented rather than
    enforced; see ``_normalize_value``.
    """

    @dataclass(frozen=True)
    class SetPack(PackSpec):
        tags: frozenset[str] = frozenset()

        _COMPOSITION = MappingProxyType({"tags": MergeKind.LAST_WINS})

    spec = SetPack(name="a", tags={"x", "y"})  # type: ignore[arg-type]

    assert spec.tags == frozenset({"x", "y"})
    with pytest.raises(AttributeError):
        spec.tags.add("z")  # type: ignore[attr-defined]


def test_specs_with_container_fields_survive_serialization(
    registry: PackRegistry[DemoPack],
) -> None:
    """The contract that rules out freezing mapping fields.

    ``to_dict`` deep-copies through ``dataclasses.asdict``, and a consumer
    may reasonably call ``asdict`` directly, so every field value has to
    remain deep-copyable.
    """
    spec = registry.get("base")
    assert spec is not None

    assert copy.deepcopy(spec) == spec
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert dataclasses.asdict(spec)["limits"] == dict(spec.limits)


# --------------------------------------------------------------------------
# Layered bindings
# --------------------------------------------------------------------------


def test_merge_bindings_lets_a_later_layer_override_one_key() -> None:
    merged = merge_bindings(
        {"base": {"mode": "strict", "priority": 5}},
        {"base": {"mode": "lax"}},
    )

    assert merged == {"base": {"mode": "lax", "priority": 5}}


def test_merge_bindings_unions_packs_across_layers() -> None:
    merged = merge_bindings({"a": {}}, {"b": {"enabled": False}})

    assert sorted(merged) == ["a", "b"]


def test_merge_bindings_does_not_mutate_or_alias_its_layers() -> None:
    platform = {"base": {"mode": "strict"}}
    tenant: dict[str, Any] = {"base": {}}

    merged = merge_bindings(platform, tenant)
    merged["base"]["mode"] = "clobbered"

    assert platform == {"base": {"mode": "strict"}}
    assert tenant == {"base": {}}


def test_locked_becomes_load_bearing_across_layers(
    registry: PackRegistry[DemoPack],
) -> None:
    """The workflow ``locked`` exists for.

    Within one body the contradiction is one author's typo; across layers a
    platform baseline is asserting a pack a tenant must not switch off.
    """
    platform = {"base": {"locked": True}}
    tenant = {"base": {"enabled": False}}

    with pytest.raises(PackResolutionError) as excinfo:
        registry.resolve(merge_bindings(platform, tenant))

    assert excinfo.value.reason == "locked_pack_disabled"


def test_an_unlocked_pack_can_still_be_disabled_by_a_later_layer(
    registry: PackRegistry[DemoPack],
) -> None:
    resolution = registry.resolve(
        merge_bindings({"base": {}}, {"base": {"enabled": False}})
    )

    assert resolution.packs == ()


def test_merge_bindings_of_nothing_is_empty() -> None:
    assert merge_bindings() == {}


def test_merge_bindings_rejects_a_non_mapping_layer() -> None:
    with pytest.raises(PackResolutionError) as excinfo:
        merge_bindings({"a": {}}, ["not-a-layer"])  # type: ignore[arg-type]

    assert excinfo.value.reason == "invalid_binding"


# --------------------------------------------------------------------------
# Hygiene: cache lifetime, shape uniformity, provenance accuracy
# --------------------------------------------------------------------------


def test_composition_plan_cache_does_not_pin_spec_classes() -> None:
    """The plan cache must not keep every spec class alive forever.

    Keying a process-lifetime cache on the class itself makes the class
    permanently reachable. Harmless for module-scope specs, but a process
    that builds spec classes dynamically — or a test suite that defines them
    per-test — leaks one class plus its module globals each time.
    """

    @dataclass(frozen=True)
    class Ephemeral(PackSpec):
        mode: str | None = None

        _COMPOSITION = MappingProxyType({"mode": MergeKind.LAST_WINS})

    PackRegistry("ephemeral", Ephemeral)  # populates the plan cache
    ref = weakref.ref(Ephemeral)

    del Ephemeral
    gc.collect()

    assert ref() is None


def test_concat_yields_a_tuple_even_with_one_participating_pack() -> None:
    """The composed shape must not depend on how many packs contributed.

    Two or more contributions go through ``tuple(acc) + tuple(nxt)``; a lone
    contribution used to pass through with whatever sequence type it had, so
    a consumer's ``spec.steps[0]`` worked but ``spec.steps + (...)`` did not.
    """
    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)
    reg.register_pack(DemoPack(name="solo", steps=range(3)))  # type: ignore[arg-type]

    composed = reg.resolve({"solo": {}}).spec

    assert isinstance(composed.steps, tuple)
    assert composed.steps == (0, 1, 2)


def test_first_wins_warning_omits_an_already_discarded_pack() -> None:
    """Provenance must list packs that contributed, not packs that tried.

    Under FIRST_WINS the second pack's value is discarded, so naming it
    among the sources of the third pack's warning points the reader at a
    pack that never influenced the value.
    """
    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)
    reg.register_pack(DemoPack(name="a", priority=0, pinned="x"))
    reg.register_pack(DemoPack(name="b", priority=1, pinned="y"))
    reg.register_pack(DemoPack(name="c", priority=2, pinned="z"))

    resolution = reg.resolve({"a": {}, "b": {}, "c": {}})

    assert resolution.spec.pinned == "x"
    pinned_warnings = [w for w in resolution.warnings if w.field == "pinned"]
    assert pinned_warnings[-1].packs == ("a", "c")


def test_unanimous_agreement_still_records_every_asserting_pack() -> None:
    """Over-trigger guard: agreeing packs all contributed the value."""
    reg: PackRegistry[DemoPack] = PackRegistry("demo", DemoPack)
    reg.register_pack(DemoPack(name="a", priority=0, tier="std", mode="m1"))
    reg.register_pack(DemoPack(name="b", priority=1, tier="std", mode="m2"))
    reg.register_pack(DemoPack(name="c", priority=2, tier="std", mode="m3"))

    resolution = reg.resolve({"a": {}, "b": {}, "c": {}})

    # LAST_WINS on `mode` keeps full history: every pack did set it.
    mode_warnings = [w for w in resolution.warnings if w.field == "mode"]
    assert mode_warnings[-1].packs == ("a", "b", "c")


# --------------------------------------------------------------------------
# Contract surfaces: the public string vocabularies and the export list
# --------------------------------------------------------------------------
#
# ``PackWarning.code`` and ``PackResolutionError.reason`` are documented as
# machine-readable discriminators — the whole reason they are structured
# rather than prose is so a deployment can branch on one. That makes each
# emitted literal part of the public contract, and its docstring the place
# consumers read it from. Both lists live in more than one place (the
# emission sites, the class docstring, the guide's table), with nothing
# tying them together, so the drift these guards catch has already happened
# once: ``binding_override_ignored`` shipped emitted, tabled in the guide,
# and absent from the docstring a consumer would build an escalation table
# from.


def _module_source() -> str:
    import dataknobs_common.packs as packs_module

    return pathlib.Path(packs_module.__file__).read_text()


def _emitted_literals(keyword: str) -> set[str]:
    """Every constant string passed as ``keyword=`` anywhere in the module."""
    tree = ast.parse(_module_source())
    return {
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == keyword
        and isinstance(kw.value, ast.Constant)
        and isinstance(kw.value.value, str)
    }


@pytest.mark.parametrize(
    ("keyword", "documented_by"),
    [("code", PackWarning), ("reason", PackResolutionError)],
)
def test_every_emitted_discriminator_is_documented(keyword: str, documented_by: type) -> None:
    """Each emitted ``code``/``reason`` appears in its class docstring.

    A consumer escalating a specific value to a hard failure reads that
    docstring. One emitted but undocumented is a value they will never
    handle, and they get no signal that it exists.
    """
    emitted = _emitted_literals(keyword)
    assert emitted, f"no {keyword}= literals found — the scan is broken"

    doc = documented_by.__doc__ or ""
    undocumented = sorted(value for value in emitted if f'"{value}"' not in doc)
    assert not undocumented, (
        f"{documented_by.__name__} emits {undocumented} but its docstring does not list them"
    )


@pytest.mark.parametrize(
    ("keyword", "documented_by"),
    [("code", PackWarning), ("reason", PackResolutionError)],
)
def test_no_documented_discriminator_is_stale(keyword: str, documented_by: type) -> None:
    """The reverse drift: documented but no longer emitted.

    Checked against the docstring's quoted values only, so surrounding
    prose can be rewritten freely without tripping this.
    """
    emitted = _emitted_literals(keyword)
    doc = documented_by.__doc__ or ""
    quoted = set(re.findall(r'``"([a-z_]+)"``', doc))
    assert quoted, f"no quoted values found in {documented_by.__name__} docstring"

    stale = sorted(quoted - emitted)
    assert not stale, f"{documented_by.__name__} documents {stale}, which nothing emits"


def test_every_emitted_warning_code_is_in_the_guide() -> None:
    """The third copy: the guide's warning-code table.

    Containment only, not table parsing — the guide can be restructured
    freely, but a new code must be described somewhere in it.
    """
    guide = pathlib.Path(__file__).parents[1] / "docs" / "guides" / "packs.md"
    if not guide.is_file():  # pragma: no cover - packaging layouts without docs
        pytest.skip("package docs not present in this layout")

    text = guide.read_text()
    missing = sorted(code for code in _emitted_literals("code") if code not in text)
    assert not missing, f"warning codes absent from the guide: {missing}"


def test_all_lists_every_public_name_the_module_defines() -> None:
    """``__all__`` is the module's own export list, not a partial one.

    ``UNSET`` shipped importable, re-exported from ``dataknobs_common``, and
    missing here — invisible to ``import *`` and to any tooling that reads
    ``__all__`` to enumerate a module's surface.
    """
    import dataknobs_common.packs as packs_module

    tree = ast.parse(_module_source())
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, ast.Assign):
            # A TypeVar is a typing artifact, not an exported name.
            if isinstance(node.value, ast.Call) and (
                getattr(node.value.func, "id", "") == "TypeVar"
            ):
                continue
            defined.update(target.id for target in node.targets if isinstance(target, ast.Name))

    public = {name for name in defined if not name.startswith("_")}
    assert not (public - set(packs_module.__all__)), (
        f"public names missing from __all__: {sorted(public - set(packs_module.__all__))}"
    )
