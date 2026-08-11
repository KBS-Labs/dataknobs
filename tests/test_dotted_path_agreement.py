"""Every dotted-path entry point in the workspace resolves the same way.

Nine functions in ``bots`` turned a dotted string from config into a live
Python object, and they disagreed on four axes: which separator they accept,
what they raise, whether they check the target's shape before or after
constructing it, and whether a typo is fatal at all. They are one
implementation now (``dataknobs_common.imports``), and this is what keeps them
that way.

The table has since grown past ``bots``: ``config``'s ``class:``/``factory:``
key, ``fsm``'s custom-resource ``class:``, and ``xization``'s ``chunker:`` key
were the same operation written three more times, and are rows here now.

The guard drives each entry point through its **public** API wherever there is
one, rather than through the shared helper. That distinction is the whole
point: a site that stops delegating and re-inlines a copy would still pass any
test written against ``dataknobs_common.imports``, and would be caught here
only because its observable behavior changed. ``FSMBuilder._create_resource``
is the one row driven through a private method — its public route is a full
``FSMConfig`` with networks and a main network, which is a great deal of
unrelated construction between the caller and the one line under test.

Filed at the workspace root rather than in ``common`` or ``bots`` because the
subject is agreement *between* packages — a copy of it inside any one is a
copy that can be deleted by a refactor of that package alone.

**Two of these assertions are deliberately written against a computed value
rather than a literal.** ``test_single_reference_resolvers_agree_on_the_failure_type``
collects the exception type every single-reference entry point actually raises
and asserts the set has one element. A test naming ``DottedPathError`` outright would pass the
day the last site was converted and say nothing before then; this one fails
today, names the disagreement, and keeps failing for as long as any site
disagrees — including a site added later that nobody thought to convert.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

FIXTURES = "tests._dotted_path_fixtures"

# A path whose module exists but whose attribute does not, and one whose
# module does not exist at all. Both are what a typo in config looks like.
MISSING_ATTRIBUTE = f"{FIXTURES}:no_such_name"
MISSING_MODULE = "tests._no_such_fixture_module:anything"


# ── Entry-point adapters ──────────────────────────────────────────────
#
# Each adapter takes a dotted path and drives ONE entry point through its
# public surface, returning whatever that surface produces. Adapters never
# catch: a failure must reach the test as the exception the entry point
# actually raises, because the exception type is one of the things under test.
#
# `target` names which fixture symbol a valid path for this entry point must
# point at — a class-resolving site needs a class of the right shape, a
# callable-resolving site needs a function.


def _via_common_resolve_dotted(ref: str) -> Any:
    from dataknobs_common.imports import resolve_dotted

    return resolve_dotted(ref)


def _via_common_resolve_callable(ref: str) -> Any:
    from dataknobs_common.imports import resolve_callable

    return resolve_callable(ref)


def _via_common_resolve_class(ref: str) -> Any:
    from dataknobs_bots.reasoning.wizard_types import MergeFilter
    from dataknobs_common.imports import resolve_class

    return resolve_class(ref, MergeFilter)


def _via_common_resolve_optional_callable(ref: str) -> Any:
    from dataknobs_common.imports import resolve_optional_callable

    return resolve_optional_callable(ref, field_name="fixture", owner="guard")


def _via_bots_tools_resolve(ref: str) -> Any:
    """The re-export at ``bots.tools.resolve`` — the deep-import path."""
    from dataknobs_bots.tools.resolve import resolve_callable

    return resolve_callable(ref)


def _via_resolve_function(ref: str) -> Any:
    from dataknobs_bots.reasoning.function_resolver import resolve_function

    return resolve_function(ref)


def _via_task_injection(ref: str) -> Any:
    """``TaskInjector.from_config`` — the public surface of former site #5.

    Returns the registered hook count so a silent skip is observable as a
    value rather than only as a missing side effect.
    """
    from dataknobs_bots.reasoning.task_injection import TaskInjector

    injector = TaskInjector.from_config({"hooks": {"artifact_created": [{"function": ref}]}})
    return len(injector._hooks["artifact_created"]) or None


def _via_rubric_registry(ref: str) -> Any:
    from dataknobs_bots.rubrics.executor import FunctionRegistry

    return FunctionRegistry().get(ref)


def _via_lifecycle_hooks(ref: str) -> Any:
    """``LifecycleHooks.from_config`` — site #7, through its public loader.

    Returns the registered callback so a *silent skip* is observable as a
    return value rather than only as a missing side effect.
    """
    from dataknobs_bots.reasoning.lifecycle import LifecycleHooks

    hooks = LifecycleHooks.from_config({"on_turn_start": [{"function": ref}]})
    return hooks.turn_start_count or None


def _via_load_merge_filter(ref: str) -> Any:
    from dataknobs_bots.reasoning.wizard_types import load_merge_filter

    return load_merge_filter(ref)


def _via_derivation_rules(ref: str) -> Any:
    """``parse_derivation_rules`` — site #9, through the function that calls it.

    Returns the parsed rule so a skipped rule is observable: before this item
    an unresolvable ``custom_class`` produced an empty list and no error.
    """
    from dataknobs_bots.reasoning.wizard_derivations import parse_derivation_rules

    rules = parse_derivation_rules(
        [
            {
                "source": "a",
                "target": "b",
                "transform": "custom",
                "custom_class": ref,
            }
        ]
    )
    return rules[0] if rules else None


def _via_middleware_factory(ref: str) -> Any:
    from dataknobs_bots.middleware.base import Middleware
    from dataknobs_bots.middleware.factory import resolve_middleware_from_spec

    return resolve_middleware_from_spec({"class": ref}, Middleware, label="middleware")


def _via_resolve_tool(ref: str) -> Any:
    from dataknobs_bots.bot.base import DynaBot

    return DynaBot._resolve_tool({"class": ref}, {})


def _via_config_build_object(ref: str) -> Any:
    """``Config.build_object`` — the ``class:`` key of an object-graph config.

    Resolves unchecked (``resolve_dotted``), because the same method serves
    ``factory:``, which deliberately accepts a module-level function. So this
    row is a callable-target row and never joins the wrong-shape check below.
    """
    from dataknobs_config import Config

    config = Config({"widget": [{"name": "w", "class": ref}]})
    return config.build_object("xref:widget[w]")


def _via_fsm_custom_resource(ref: str) -> Any:
    """``FSMBuilder._create_resource`` — a ``custom`` resource's ``class:``."""
    from dataknobs_fsm.config.builder import FSMBuilder
    from dataknobs_fsm.config.schema import ResourceConfig

    return FSMBuilder()._create_resource(
        ResourceConfig(name="fixture", type="custom", config={"class": ref})
    )


def _via_create_chunker(ref: str) -> Any:
    """``create_chunker`` — the ``chunker:`` key of a chunking config."""
    from dataknobs_xization.chunking.registry import chunker_registry, create_chunker

    try:
        return create_chunker({"chunker": ref})
    finally:
        # The factory registers a resolved dotted path under its own key, so
        # a successful call leaves state behind that the next parametrization
        # would hit instead of re-resolving.
        if chunker_registry.is_registered(ref):
            chunker_registry.unregister(ref)


#: ``(name, adapter, fixture attribute a valid path must name)``.
#: The third element is what makes one table cover both the callable-resolving
#: and class-resolving halves of the family.
ENTRY_POINTS: list[tuple[str, Callable[[str], Any], str]] = [
    ("common.resolve_dotted", _via_common_resolve_dotted, "resolvable_function"),
    ("common.resolve_callable", _via_common_resolve_callable, "resolvable_function"),
    ("common.resolve_class", _via_common_resolve_class, "ConformingMergeFilter"),
    (
        "common.resolve_optional_callable",
        _via_common_resolve_optional_callable,
        "resolvable_function",
    ),
    ("bots.tools.resolve", _via_bots_tools_resolve, "resolvable_function"),
    ("resolve_function", _via_resolve_function, "resolvable_function"),
    ("TaskInjector.from_config", _via_task_injection, "resolvable_function"),
    ("rubrics FunctionRegistry.get", _via_rubric_registry, "resolvable_function"),
    ("LifecycleHooks.from_config", _via_lifecycle_hooks, "resolvable_function"),
    ("load_merge_filter", _via_load_merge_filter, "ConformingMergeFilter"),
    ("parse_derivation_rules", _via_derivation_rules, "ConformingFieldTransform"),
    ("resolve_middleware_from_spec", _via_middleware_factory, "ConformingMiddleware"),
    ("DynaBot._resolve_tool", _via_resolve_tool, "ConformingTool"),
    ("Config.build_object", _via_config_build_object, "resolvable_function"),
    (
        "FSMBuilder._create_resource",
        _via_fsm_custom_resource,
        "ConformingResourceProvider",
    ),
    ("create_chunker", _via_create_chunker, "ConformingChunker"),
]

IDS = [name for name, _, _ in ENTRY_POINTS]

#: Entry points that load a whole config *block* rather than one reference.
#:
#: These deliberately do NOT raise ``DottedPathError``. They collect every
#: fault in the block and raise one ``ConfigurationError`` naming all of
#: them, because most of what they can reject is not a dotted-path failure at
#: all — a derivation rule missing its ``target``, a hook naming an unknown
#: event — and ``DottedPathError`` would misdescribe those. Aggregating is
#: the point: an author with three bad rules should learn about three.
#:
#: So the agreement they hold to is the weaker, true one — the same base
#: type, and a message that names the offending reference. Asserting
#: ``DottedPathError`` across the whole table would have forced either a
#: misdescribed error type or fail-fast loading, and both are worse than the
#: split.
BLOCK_LOADERS = frozenset(
    {
        "TaskInjector.from_config",
        "LifecycleHooks.from_config",
        "parse_derivation_rules",
    }
)


@pytest.fixture(autouse=True)
def _reset_instantiation_counter():
    """Zero the fixture module's construction counter around every test."""
    from tests import _dotted_path_fixtures as fixtures

    fixtures.reset_instantiations()
    yield


@pytest.mark.parametrize(("name", "resolve", "target"), ENTRY_POINTS, ids=IDS)
@pytest.mark.parametrize("separator", [":", "."], ids=["colon", "dot"])
def test_entry_points_accept_both_separators(
    name: str, resolve: Callable[[str], Any], target: str, separator: str
) -> None:
    """``module:name`` and ``module.name`` resolve identically, everywhere.

    Before this item, three sites accepted only ``:``, four accepted only
    ``.``, and two accepted either — so the same config value was valid or
    invalid depending on which key it was written under. Nothing about a
    dotted path depends on which subsystem is reading it.
    """
    result = resolve(f"{FIXTURES}{separator}{target}")

    assert result is not None, (
        f"{name} resolved a valid {separator!r}-separated path to None — "
        "a silent skip, not a resolution"
    )


@pytest.mark.parametrize(("name", "resolve", "target"), ENTRY_POINTS, ids=IDS)
def test_no_entry_point_silently_skips_a_bad_path(
    name: str, resolve: Callable[[str], Any], target: str
) -> None:
    """A typo raises. It never resolves to nothing and carries on.

    This is the item's core symptom: a bot whose config named a hook, a
    transform, or a callback that could not be imported would start
    successfully and quietly do less than its config said. Four callers
    behaved that way, and three of them were pinned by tests asserting the
    silence.
    """
    with pytest.raises(Exception) as excinfo:
        resolve(MISSING_ATTRIBUTE)

    assert not isinstance(excinfo.value, AssertionError), (
        f"{name} did not raise for an unresolvable path"
    )


def _failure_types(ref: str, names: frozenset[str] | None = None) -> dict[str, str]:
    """What each entry point raises for *ref*, by name. Never raises itself."""
    observed: dict[str, str] = {}
    for name, resolve, _ in ENTRY_POINTS:
        if names is not None and name not in names:
            continue
        try:
            resolve(ref)
        except Exception as exc:
            observed[name] = type(exc).__name__
        else:
            observed[name] = "<did not raise>"
    return observed


@pytest.mark.parametrize(
    ("case", "ref"),
    [("missing module", MISSING_MODULE), ("missing attribute", MISSING_ATTRIBUTE)],
)
def test_single_reference_resolvers_agree_on_the_failure_type(case: str, ref: str) -> None:
    """One dotted-path failure has one type, whichever key it was written under.

    Asserted against the observed set rather than against ``DottedPathError``
    by name, so this failed *before* the consolidation — naming the
    divergence, which ran to five distinct behaviours including two that did
    not raise at all — instead of only starting to mean something once the
    last site was converted.
    """
    single = frozenset(IDS) - BLOCK_LOADERS
    observed = _failure_types(ref, single)
    distinct = sorted(set(observed.values()))

    assert len(distinct) == 1, (
        f"{case}: entry points disagree on the failure type: {distinct}\n"
        + "\n".join(f"  {n:34} {t}" for n, t in sorted(observed.items()))
    )
    assert distinct[0] == "DottedPathError", (
        f"{case}: entry points agree on {distinct[0]!r}, which is not the family's own type"
    )


@pytest.mark.parametrize(
    ("case", "ref"),
    [("missing module", MISSING_MODULE), ("missing attribute", MISSING_ATTRIBUTE)],
)
def test_every_entry_point_raises_a_configuration_error(case: str, ref: str) -> None:
    """The floor that does hold across the whole table.

    A caller writing ``except ConfigurationError`` around bot construction
    catches every dotted-path fault from every one of these surfaces —
    including the block loaders, whose aggregate error is deliberately not a
    ``DottedPathError`` (see ``BLOCK_LOADERS``). That is the property
    consumers actually depend on, so it is asserted separately rather than
    left implied by the narrower one above.
    """
    from dataknobs_common.exceptions import ConfigurationError

    # Collected rather than asserted per iteration, so a regression names
    # every entry point that broke instead of aborting on the first — the
    # same reason `_failure_types` above reports a table. `_failure_types`
    # itself cannot serve here: it records the exception's type *name*, and
    # this claim is about a subclass relationship.
    escaped: dict[str, str] = {}
    for name, resolve, _ in ENTRY_POINTS:
        try:
            resolve(ref)
        except ConfigurationError:
            continue
        except Exception as exc:
            escaped[name] = type(exc).__name__
        else:
            escaped[name] = "<did not raise>"

    assert not escaped, (
        f"{case}: entry points that did not raise a ConfigurationError:\n"
        + "\n".join(f"  {n:34} {t}" for n, t in sorted(escaped.items()))
    )


@pytest.mark.parametrize("name", sorted(BLOCK_LOADERS))
def test_a_block_loader_names_the_offending_reference(name: str) -> None:
    """Aggregating must not cost the diagnostic.

    Collecting faults into one error is only an improvement if each fault is
    still identifiable — an error that says "3 problems" and nothing else is
    worse than three that each say what is wrong.
    """
    resolve = next(fn for n, fn, _ in ENTRY_POINTS if n == name)

    with pytest.raises(Exception) as excinfo:
        resolve(MISSING_ATTRIBUTE)

    assert "no_such_name" in str(excinfo.value)


@pytest.mark.parametrize(
    ("name", "resolve", "target"),
    [row for row in ENTRY_POINTS if row[2][0].isupper()],
    ids=[name for name, _, target in ENTRY_POINTS if target[0].isupper()],
)
def test_a_wrong_shape_target_is_rejected_before_it_is_constructed(
    name: str, resolve: Callable[[str], Any], target: str
) -> None:
    """A class of the wrong shape must not run its ``__init__``.

    ``resolve_class`` returns the class and leaves construction to the caller,
    which is what makes validate-before-instantiate the only expressible
    order. Two sites used to instantiate first and check after, so a mistyped
    path ran an unrelated class's constructor — arbitrary code, with whatever
    side effects it has — before being rejected.

    The counter is what gives this teeth. Asserting only that the call raised
    would pass against an implementation that constructed the object, checked
    it, threw it away and raised.
    """
    from dataknobs_common.exceptions import ConfigurationError
    from tests import _dotted_path_fixtures as fixtures

    # `ConfigurationError` rather than bare `Exception`: a wrong-shape target
    # is a config fault like any other, and the floor asserted by
    # `test_every_entry_point_raises_a_configuration_error` is the floor here
    # too. Catching anything would also pass on an `AttributeError` raised by
    # a typo in the adapter above, which is not what this is testing.
    with pytest.raises(ConfigurationError):
        resolve(f"{FIXTURES}:BareClass")

    assert fixtures.instantiations.count == 0, (
        f"{name} constructed the wrong-shape class before rejecting it "
        f"({fixtures.instantiations.count} construction(s))"
    )


def test_the_bots_re_export_is_the_same_object_not_a_copy() -> None:
    """``bots.tools.resolve`` must *be* the common function, by identity.

    Every other check in this file is behavioural, and behaviour cannot see
    the difference that matters here: a re-inlined copy of the resolver would
    satisfy all of them and still be a fifth implementation free to drift.
    Identity is the only assertion that distinguishes "delegates" from
    "happens to agree today".

    The source guard catches a copy that reaches for ``import_module``, but
    not one written against the shared helper — a wrapper that adds a default,
    narrows the separator, or swallows a reason would pass the scan and pass
    the behavioural table for the fixtures it is given.
    """
    from dataknobs_bots.tools import resolve as bots_resolve
    from dataknobs_common import imports as common_imports

    exported = list(bots_resolve.__all__)
    assert exported, "the re-export exports nothing"

    for name in exported:
        assert getattr(bots_resolve, name) is getattr(common_imports, name), (
            f"bots.tools.resolve.{name} is not "
            f"dataknobs_common.imports.{name} — a wrapper or a re-inlined "
            f"copy would pass every behavioural check in this file"
        )
