"""Placing a term against an authored vocabulary, with no loop and no store.

The worked call site's last line, which until now could be written and not
run. Everything here goes through ``build_resolver`` over the same file
``load_ontology`` was given, because that is the call a consumer makes.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
import yaml

from dataknobs_common.entity_resolution import (
    AliasSignal,
    AsyncAliasSignal,
    AsyncExactNormalizedSignal,
    CascadingResolver,
    EvidenceKind,
    ExactNormalizedSignal,
    Scoring,
)
from dataknobs_common.ontology import (
    async_build_resolver,
    async_load_ontology,
    build_resolver,
    load_ontology,
)

if TYPE_CHECKING:
    from pathlib import Path


def with_resolver(source: Path, destination: Path, section: dict) -> Path:
    """The fixture's document, carrying a ``resolver:`` section.

    Derived from the fixture rather than written out again: the vocabulary is
    not what any of these tests is about, and a second copy of it would let
    the two drift on everything except the section under test.
    """
    document = yaml.safe_load(source.read_text())
    document["ontology"]["resolver"] = section
    destination.write_text(yaml.safe_dump(document))
    return destination


def test_a_placement_completes_with_no_event_loop_running(mammals_path: Path) -> None:
    """Criterion 11: the whole of it, in the order the criterion states it.

    ``asyncio.get_running_loop()`` raising is asserted rather than assumed. A
    cascade that merely *happens* not to await is not the claim: the claim is
    that the synchronous flavour is native here, so the test says so in the
    one way that cannot pass by accident.
    """
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()

    ontology = load_ontology(mammals_path)
    resolver = build_resolver(mammals_path, ontology)

    result = resolver.resolve("beagles", k=5)

    assert result.candidates[0].entity_id == "beagle"
    assert result.query == "beagles"

    evidence = result.candidates[0].evidence[0]
    assert evidence.kind is EvidenceKind.DECLARED
    assert evidence.signal == "exact"
    assert evidence.scoring is Scoring.DECLARED


def test_every_candidate_reports_the_rung_that_produced_it(mammals_path: Path) -> None:
    """Per candidate, not per result.

    A cascade's candidates are heterogeneous by construction, so a single
    field on the result could not describe them -- which is why ``scoring``
    sits on the evidence and why this asserts over every candidate rather
    than over the first.
    """
    resolver = build_resolver(mammals_path, load_ontology(mammals_path))

    result = resolver.resolve("dog", k=5)

    assert result.candidates
    for candidate in result.candidates:
        assert candidate.evidence
        for evidence in candidate.evidence:
            assert evidence.signal
            assert evidence.scoring is Scoring.DECLARED


def test_a_miss_is_an_empty_candidate_tuple(mammals_path: Path) -> None:
    """There is no outcome enum: ``RESOLVED`` and ``UNRESOLVED`` said this.

    The hit is asserted first, in this test rather than in a sibling. A
    resolver built with no rungs misses everything, so the miss alone is true
    of a cascade that works and of one that was never assembled -- and a
    control living in another test stops covering this one the moment that
    test is skipped, renamed or deleted.
    """
    resolver = build_resolver(mammals_path, load_ontology(mammals_path))

    assert resolver.resolve("beagles", k=5).candidates
    assert not resolver.resolve("nothing in this vocabulary", k=5).candidates


def test_an_absent_resolver_section_builds_the_declared_order(mammals_path: Path) -> None:
    """Silence builds something, and what it builds is exact then alias.

    The fixture declares no ``resolver:`` at all, so if silence meant *no
    rungs* the criterion above would have nothing to be true of. Asserted on
    the composition rather than only on the answer, because a cascade with one
    rung would also place ``beagles``.
    """
    resolver = build_resolver(mammals_path, load_ontology(mammals_path))

    assert isinstance(resolver, CascadingResolver)
    assert [type(rung) for rung in resolver.rungs] == [ExactNormalizedSignal, AliasSignal]


def test_an_empty_rung_list_builds_a_cascade_that_misses_everything(
    mammals_path: Path, tmp_path: Path
) -> None:
    """An absent section and an empty one are different, and must stay so.

    ``rungs: []`` is a composition somebody wrote. The composition is the
    policy, so a consumer who wants no rungs can say so -- and this is the
    assertion that stops *absent* and *empty* collapsing into each other the
    next time the default is touched.
    """
    path = with_resolver(mammals_path, tmp_path / "no-rungs.yaml", {"rungs": []})

    resolver = build_resolver(path, load_ontology(path))

    assert resolver.rungs == ()
    assert not resolver.resolve("beagles", k=5).candidates


def test_the_moved_value_types_are_one_object_at_both_depths() -> None:
    """The move is a move, not a copy.

    ``Scoring``, ``CompatibilityVerdict`` and ``ResolutionRef`` were declared
    in the ontology model and now live with the family that constructs them,
    re-exported. Identity rather than equality: two enum classes with the same
    members compare unequal member-for-member, so an accidental second
    definition would fail at a consumer's ``is`` check and pass a shallow one.
    """
    from dataknobs_common import entity_resolution, ontology

    assert ontology.Scoring is entity_resolution.Scoring
    assert ontology.CompatibilityVerdict is entity_resolution.CompatibilityVerdict
    assert ontology.ResolutionRef is entity_resolution.ResolutionRef


def test_the_normalizer_is_one_object_wherever_it_is_reached() -> None:
    """``default_normalizer`` moved too, for the same reason and to its own module."""
    from dataknobs_common import text
    from dataknobs_common.ontology import default_normalizer

    assert default_normalizer is text.default_normalizer


def test_a_vocabularys_own_fold_is_the_only_fold(mammals_path: Path) -> None:
    """A rung handed no normalizer does not fold; the source it reads does.

    Two folds ran, and the second was invisible because it agreed with the
    first on every form this vocabulary happens to spell in lower case. Loading
    with a normalizer that does **not** case-fold is what separates them: the
    rung pre-folded the query to lower case, the index had folded ``Beagles``
    to ``Beagles``, and the two could never meet.

    **Both spellings, because one of them alone reads as an ordinary case
    bug.** Under the defect neither works -- not the authored spelling and not
    the folded one -- which is the tell that a *second* fold is running rather
    than that the wrong case was asked for. The ``beagles`` line is therefore a
    positive control, and it stays ``[]`` after the fix: this vocabulary was
    loaded case-sensitively, so the lower-case spelling missing is the caller's
    own instruction being obeyed.
    """
    onto = load_ontology(mammals_path, normalizer=str.strip)
    resolver = build_resolver(mammals_path, onto)

    def placed(query: str) -> list[str]:
        return [candidate.entity_id for candidate in resolver.resolve(query).candidates]

    assert placed("Beagles") == ["beagle"]
    assert placed("beagles") == []


def test_the_default_fold_still_matches_the_way_it_always_did(mammals_path: Path) -> None:
    """The other half: dropping the rung's fold changes nothing by default.

    ``default_normalizer`` is what an unconfigured ``load_ontology`` folds its
    index with, so removing the rung's copy of it leaves every query in this
    file answering exactly as before. Without this, the assertion above is
    satisfied by a rung that stopped matching anything at all.
    """
    resolver = build_resolver(mammals_path, load_ontology(mammals_path))

    def placed(query: str) -> list[str]:
        return [candidate.entity_id for candidate in resolver.resolve(query).candidates]

    assert placed("Beagles") == ["beagle"]
    assert placed("beagles") == ["beagle"]
    assert placed("  Domestic dog  ") == ["dog"]


def test_the_async_door_builds_the_same_declared_order(mammals_path: Path) -> None:
    """``async_build_resolver`` over a document that configures nothing.

    Its synchronous twin has had this assertion since the door shipped and
    this one had none at all -- the name appeared nowhere in the suite. What
    it guards is not the ordering, which is one shared function, but that the
    asynchronous door builds *anything*: it is the remedy the synchronous
    door's refusal names, and a remedy that raises is worse than no remedy.
    """

    async def exercise() -> None:
        onto = await async_load_ontology(mammals_path)
        resolver = await async_build_resolver(mammals_path, onto)

        assert [type(rung) for rung in resolver.rungs] == [
            AsyncExactNormalizedSignal,
            AsyncAliasSignal,
        ]

        result = await resolver.resolve("beagles", k=5)
        assert [c.entity_id for c in result.candidates] == ["beagle"]

    asyncio.run(exercise())
