"""Tests for the vocabulary-free pack composition core.

The subject is pure data (frozen spec records and a fold over them), so no
test doubles are involved at all — every spec below is a real ``PackSpec``
subclass and every registry is a real ``PackRegistry``.

``DemoPack`` exercises all six built-in merge kinds in one spec so the
ordering, warning, and conflict paths are covered against a single shape.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import pytest

from dataknobs_common.exceptions import ConfigurationError, OperationError
from dataknobs_common.packs import (
    MergeKind,
    PackRegistry,
    PackResolution,
    PackResolutionError,
    PackSpec,
    PackWarning,
    compose_packs,
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
