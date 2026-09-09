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
    CascadingResolver,
    EvidenceKind,
    ExactNormalizedSignal,
    Scoring,
)
from dataknobs_common.ontology import load_ontology
from dataknobs_common.ontology.loader import build_resolver

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
