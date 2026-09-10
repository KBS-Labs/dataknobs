"""A type that answers ``Hashable`` and raises at the call is worse than one
that never claimed the capability, because the check is how a caller is
supposed to ask.

``@dataclass(frozen=True)`` with equality left on gets a generated ``__hash__``
over the field tuple, so the TYPE satisfies :class:`collections.abc.Hashable`
whatever its fields hold. Give it a ``Mapping``, a ``dict``, or a field whose
own type is unhashable, and ``isinstance(value, Hashable)`` answers True while
``hash(value)`` raises ``TypeError``. A caller guarding with the check is
guarded against nothing and finds out at the first ``set.add``.

Two answers are already in this package and both are honest:

* ``eq=True`` and not frozen -- ``__hash__`` is None, the check answers False,
  and a caller learns it from the check.
* ``frozen=True, eq=False`` -- hashes by identity, the check answers True, and
  the call cannot raise.

Only the third combination is wrong.

**What this module does NOT assert, and why.** There is no test here saying
*no type is in that state*, because twenty-three are, and the way out is
per type: identity for a built value nobody compares field-wise, unfrozen
equality for a record two of which really can be equal. Both are behaviour
changes to published types and neither is a default, so the choice is a ruling
rather than a cleanup, and a guard cannot make it. What this module does
instead is **fix the population**: it measures the whole package by
construction and fails when the measurement and :data:`OPEN` disagree in
either direction. That makes a new instance of the shape distinguishable from
the ones already known, which is the part that does not need the ruling and
should not wait for it.

The sweep itself lives in ``_dataclass_sweep``, beside the reasons it builds a
value rather than reading an annotation.
"""

from __future__ import annotations

import pytest

from _dataclass_sweep import every_dataclass, import_failures, probe_hashability

#: Types that claim ``Hashable`` and raise on a constructed instance.
#:
#: Recorded so a NEW instance of the shape is distinguishable from the ones
#: already here. Every entry is open: the fix is per type -- identity for a
#: built value nobody compares field-wise, unfrozen equality for a record two
#: of which really can be equal -- and there is no third correct answer.
#: Entries leave this list as they are decided; nothing is added without one.
OPEN: frozenset[str] = frozenset(
    {
        "_nested_core._MintedNode",
        "aws.AwsSessionConfig",
        "discriminator.AsyncChainedDiscriminator",
        "discriminator.ChainedDiscriminator",
        "discriminator.MultiFieldDiscriminator",
        "entity_resolution.cascade.CascadeState",
        "entity_resolution.values.ResolutionRef",
        "ontology.config.OntologyConfig",
        "ontology.model.Literal",
        "ontology.model.ProjectionContext",
        "ontology.model.Provenance",
        "ontology.model.SourceRef",
        "ontology.sources.SourceDescription",
        "ontology.values.AsyncOntology",
        "ontology.values.Ontology",
        "ontology.values.OntologyParts",
        "packs._CompositionPlan",
        "packs._Contribution",
        "ratelimit.types.RateLimiterConfig",
        "resolver.CompositeResolver",
        "resolver.JoiningPartitionResolver",
        "resolver.MappingResolver",
        "retry.RetryConfig",
    }
)

#: Types the builder cannot construct, so the contract is unmeasured for them.
#:
#: A hole in the sweep rather than a verdict, declared so it cannot grow
#: quietly. This one rejects the generic witness in ``__post_init__`` because
#: it validates a value out of a vocabulary the annotation does not carry:
#: the field is a plain ``str`` and only certain strings are accepted.
UNCONSTRUCTIBLE: frozenset[str] = frozenset(
    {
        "resolver.TemporalPartitionResolver",
    }
)


@pytest.fixture(scope="module")
def swept() -> dict[str, tuple[str, str]]:
    """Every dataclass this package defines whose type claims to be hashable."""
    return {
        name: probe_hashability(cls)
        for name, cls in sorted(every_dataclass().items())
        if cls.__hash__ is not None
    }


def test_the_open_set_is_exactly_what_was_measured(
    swept: dict[str, tuple[str, str]],
) -> None:
    """:data:`OPEN` names the shape's instances, so a new one is distinguishable.

    Fails in both directions on purpose. A name that starts raising is a fresh
    instance of the shape and needs its own answer; a name still listed that
    now hashes has been fixed, and leaving it recorded would let the next
    regression hide behind it.

    This is the whole of what can be asserted before the per-type answers are
    decided, and it is not nothing: it turns twenty-three known defects into a
    closed set, which is what makes the twenty-fourth visible.
    """
    raising = {name for name, (verdict, _) in swept.items() if verdict == "raises"}

    # Both assertions below compare two sets for an empty difference, and two
    # empty sets satisfy both -- so a sweep that measured nothing would report
    # clean. The control is here rather than in a test of its own because it is
    # a property of the measurement, not of either direction asked of it.
    assert swept, "the sweep found no hashable dataclass at all; the comparison below is vacuous"

    assert raising - OPEN == set(), (
        f"new type(s) claim Hashable and raise at the call: {sorted(raising - OPEN)}. "
        "Each needs one of the two honest answers: identity (frozen=True, "
        "eq=False) for a built value nobody compares field-wise, or unfrozen "
        "equality (eq=True, frozen=False) for a record two of which really can "
        "be equal."
    )
    assert OPEN - raising == set(), (
        f"type(s) recorded as raising now hash; remove them from OPEN: {sorted(OPEN - raising)}"
    )


def test_the_sweep_reaches_every_type_it_claims_to(
    swept: dict[str, tuple[str, str]],
) -> None:
    """A type the builder cannot construct is unmeasured, and may not multiply.

    The sweep's worth is that it covers the whole package rather than the
    instances somebody thought to build by hand, so the set it cannot reach is
    declared and checked rather than left implicit. Without this, the census
    above could shrink to nothing one unconstructible type at a time and stay
    green the whole way down.
    """
    assert import_failures() == [], f"modules the sweep could not import: {import_failures()}"
    unreached = {name for name, (verdict, _) in swept.items() if verdict == "unconstructible"}
    assert unreached - UNCONSTRUCTIBLE == set(), (
        "type(s) the builder can no longer construct, so the contract is "
        f"unmeasured for them: {sorted(unreached - UNCONSTRUCTIBLE)}"
    )
    assert UNCONSTRUCTIBLE - unreached == set(), (
        "type(s) recorded as unconstructible are now reachable; remove them "
        f"from UNCONSTRUCTIBLE: {sorted(UNCONSTRUCTIBLE - unreached)}"
    )
