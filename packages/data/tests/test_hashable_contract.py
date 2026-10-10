# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A type that answers ``Hashable`` and raises at the call is worse than one
that never claimed the capability, because the check is how a caller is
supposed to ask.

``@dataclass(frozen=True)`` with equality left on gets a generated ``__hash__``
over the field tuple, so the TYPE satisfies :class:`collections.abc.Hashable`
whatever its fields hold. Give it a ``Mapping``, a ``dict``, or a field whose
own type is unhashable, and ``isinstance(value, Hashable)`` answers True while
``hash(value)`` raises ``TypeError``. A caller guarding with the check is
guarded against nothing and finds out at the first ``set.add``.

Two answers are honest:

* ``eq=True`` and not frozen -- ``__hash__`` is None, the check answers False,
  and a caller learns it from the check.
* ``frozen=True, eq=False`` -- hashes by identity, the check answers True, and
  the call cannot raise.

A frozen type with equality on can still be honest by declaring ``__hash__``
None, which is how every ``StructuredConfig`` answers: its base declares it for
each subclass, because a frozen base cannot be unfrozen. Left to the generated
hash, that combination is the wrong one.

**This package had no census at all, and that is how it got one.** The
equivalent suite in ``dataknobs-common`` had measured that package for some
time, and its sweep named ``dataknobs_common`` in four places -- so the
question was asked of one package because that was the only package it could
be asked of, not because the others had been cleared. A deep review found
``RecordFieldSource`` and ``MultiFieldSource`` in exactly the shape this
module is about, in a family whose three other members had already been ruled
the other way one package over. The guard that would have caught it swept a
tree it could not see them in.

**What this module does NOT assert, and why.** There is no test here saying
*no type is in that state*, because two are, and the way out is per type:
identity for a built value nobody compares field-wise, unfrozen equality for a
record two of which really can be equal. Both are behaviour changes to
published types and neither is a default, so the choice is a ruling rather
than a cleanup, and a guard cannot make it. What this module does instead is
**fix the population**: it measures the whole package by construction and
fails when the measurement and :data:`OPEN` disagree in either direction. That
makes a new instance of the shape distinguishable from the ones already known,
which is the part that does not need the ruling and should not wait for it.

The sweep is :class:`~dataknobs_common.testing.DataclassSweep`, beside the
reasons it builds a value rather than reading an annotation.
"""

from __future__ import annotations

import pytest

import dataknobs_data
from dataknobs_common.testing import DataclassSweep

#: Types that claim ``Hashable`` and raise on a constructed instance.
#:
#: Recorded so a NEW instance of the shape is distinguishable from the ones
#: already here. Every entry is open: the fix is per type -- identity for a
#: built value nobody compares field-wise, unfrozen equality for a record two
#: of which really can be equal -- and there is no third correct answer.
#: Entries leave this list as they are decided; nothing is added without one.
#:
#: It held twenty-one. Nineteen were configs -- the eighteen backend configs,
#: which inherit a ``schema: DatabaseSchema`` field from ``DatabaseConfig``, and
#: ``UserStateSectionSpec`` -- and left when every ``StructuredConfig`` was
#: declared unhashable. The two that remain each hold a ``dict`` directly.
OPEN: frozenset[str] = frozenset(
    {
        "sources.cluster_index.ClusterTopicConfig",
        "user.migration.SectionMigrator",
    }
)

#: Types the builder cannot construct, so the contract is unmeasured for them.
#:
#: A hole in the sweep rather than a verdict, declared so it cannot grow
#: quietly. Empty: the seven it held were configs, which no longer claim to be
#: hashable and so are not asked. A type whose ``__post_init__`` rejects the
#: generic witness -- a plain ``str`` validated against a vocabulary the
#: annotation does not carry -- belongs here, with that reason.
UNCONSTRUCTIBLE: frozenset[str] = frozenset()


@pytest.fixture(scope="module")
def sweep() -> DataclassSweep:
    """One walk for this module, since :meth:`every_dataclass` caches per instance."""
    return DataclassSweep(dataknobs_data)


@pytest.fixture(scope="module")
def swept(sweep: DataclassSweep) -> dict[str, tuple[str, str]]:
    """Every dataclass this package defines whose type claims to be hashable."""
    return {
        name: sweep.probe_hashability(cls)
        for name, cls in sorted(sweep.every_dataclass().items())
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
    sweep: DataclassSweep,
    swept: dict[str, tuple[str, str]],
) -> None:
    """A type the builder cannot construct is unmeasured, and may not multiply.

    The sweep's worth is that it covers the whole package rather than the
    instances somebody thought to build by hand, so the set it cannot reach is
    declared and checked rather than left implicit. Without this, the census
    above could shrink to nothing one unconstructible type at a time and stay
    green the whole way down.
    """
    failures = sweep.import_failures()
    assert failures == [], f"modules the sweep could not import: {failures}"
    unreached = {name for name, (verdict, _) in swept.items() if verdict == "unconstructible"}
    assert unreached - UNCONSTRUCTIBLE == set(), (
        "type(s) the builder can no longer construct, so the contract is "
        f"unmeasured for them: {sorted(unreached - UNCONSTRUCTIBLE)}"
    )
    assert UNCONSTRUCTIBLE - unreached == set(), (
        "type(s) recorded as unconstructible are now reachable; remove them "
        f"from UNCONSTRUCTIBLE: {sorted(UNCONSTRUCTIBLE - unreached)}"
    )


def test_the_index_source_family_hashes_by_identity(sweep: DataclassSweep) -> None:
    """The five members of one family, given one answer, asserted together.

    Three of these are declared in ``dataknobs-common`` and two here, and the
    split is why the two were missed: the ruling was taken on the three, in a
    commit whose message states the hazard, and the census that would have
    caught the other two could not see this package. Asserted across the split
    rather than per package, so the family is measured as a family.
    """
    from dataknobs_common.index import AliasSource, CallableSource, MappingSource

    from dataknobs_data.vector.index_sources import MultiFieldSource, RecordFieldSource

    for cls in (MappingSource, CallableSource, AliasSource, RecordFieldSource, MultiFieldSource):
        assert cls.__hash__ is object.__hash__, (
            f"{cls.__name__} does not hash by identity; a source is a configured "
            f"behaviour rather than a value, and frozen with equality on it would "
            f"claim Hashable and raise the moment a caller passed a list"
        )

    swept = sweep.every_dataclass()
    assert sweep.probe_hashability(swept["vector.index_sources.MultiFieldSource"])[0] == "hashes"
    assert sweep.probe_hashability(swept["vector.index_sources.RecordFieldSource"])[0] == "hashes"
