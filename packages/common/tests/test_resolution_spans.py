"""Where a match sat, and what a rung has to implement to say so.

Three claims, and the third is the one the other two exist for.

The **fold** and the **boundary policy** are different questions, answered by
different functions, and a scan needs both. Coverage is a **positional** report
derived from the evidence rather than a second computation kept beside it. And
``DeclaredSignal``'s span hook is sufficient for a rung that locates forms
*inside* a query -- proven the only way that claim can be proven, by writing
one from outside the package and asking it to do the job.

**The scanning rung below is a consumer's, deliberately.** It was written from
outside the package, against the published hook and with nothing to copy, which
is the only way the seam can be shown to be one. The library now ships
``ScanningSignal`` doing the same job -- so this class is no longer the only
implementation, and is still the only *independent* one. Rewriting it against
the shipped class would retire the evidence rather than tidy it; what the leg
that shipped the rung did instead was take its name and rename this one.

``test_the_shipped_rung_and_an_independent_one_agree`` is what that
independence is now spent on: two implementations of one description, compared
against each other rather than each against a literal.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest

from dataknobs_common.entity_resolution import (
    AliasSignal,
    AsyncCascadingResolver,
    AsyncDeclaredSignal,
    AsyncScanningSignal,
    CascadingResolver,
    CompatibilityVerdict,
    DeclaredSignal,
    EvidenceKind,
    ExactNormalizedSignal,
    FormHit,
    MatchEvidence,
    ResolutionRef,
    RunnerUp,
    ScanningSignal,
    Scoring,
    content_span,
    token_spans,
)
from dataknobs_common.ontology import (
    Entity,
    MappingEntitySource,
    async_load_ontology,
    load_ontology,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class ConsumerScanningSignal(DeclaredSignal):
    """An n-gram probe over the folded form index, written as a consumer would.

    Every span of consecutive tokens is looked up as the slice it covers --
    21 lookups for a six-token utterance -- so the form reaches the index
    carrying whatever separated its tokens, and matches whichever spelling the
    vocabulary declared.

    **Longest first**, which is the order a declared rung has to publish
    itself: its score is ``1.0`` by fiat and carries none, so nothing
    downstream could recover it. The cascade positions by arrival, so the
    containing form is proposed before the form it contains.
    """

    key = "scan"

    def _located(self, query: str) -> Sequence[FormHit]:
        tokens = token_spans(query)
        found: list[FormHit] = []
        for length in range(len(tokens), 0, -1):
            for first in range(len(tokens) - length + 1):
                start, end = tokens[first][0], tokens[first + length - 1][1]
                hits = self._entities.by_surface_form(self._fold(query[start:end]))
                found += [
                    FormHit(entity_id=entity_id, span=(start, end)) for entity_id in sorted(hits)
                ]
        return found


class AsyncConsumerScanningSignal(AsyncDeclaredSignal):
    """:class:`ConsumerScanningSignal` over an asynchronous source.

    The same hook, one ``await`` further in -- which is the point of asserting
    it separately: the span seam is on both halves of the twin, so a consumer
    does not have to drop to the bare protocol to scan asynchronously.
    """

    key = "scan"

    async def _located(self, query: str) -> Sequence[FormHit]:
        tokens = token_spans(query)
        found: list[FormHit] = []
        for length in range(len(tokens), 0, -1):
            for first in range(len(tokens) - length + 1):
                start, end = tokens[first][0], tokens[first + length - 1][1]
                hits = await self._entities.by_surface_form(self._fold(query[start:end]))
                found += [
                    FormHit(entity_id=entity_id, span=(start, end)) for entity_id in sorted(hits)
                ]
        return found


def test_the_fold_and_the_boundary_policy_answer_different_questions() -> None:
    """``default_normalizer`` strips; it does not know what a token is.

    Both spans below are into the same string and neither is derivable from
    the other. ``content_span`` says which characters survived a fold applied
    to the whole string, which is what a rung comparing the whole query has to
    report. ``token_spans`` says where the words are, which is what stops a
    scan finding ``beagle`` inside ``unbeagleable``.
    """
    assert content_span("  Beagle!  ") == (2, 9)
    assert token_spans("  Beagle!  ") == ((2, 8),)

    assert content_span("") == (0, 0)
    assert content_span("   ") == (0, 0)
    assert token_spans("   ") == ()

    # Unicode-aware, by `str.isalnum` rather than by an ASCII class.
    assert token_spans("straße 7") == ((0, 6), (7, 8))


def test_token_spans_are_the_n_grams_a_probe_enumerates() -> None:
    """Six tokens, 21 candidate forms, and the slice carries the separator.

    The count is the whole reason a scan over an authored vocabulary needs no
    index of its own: it is a bounded number of dictionary lookups, not a
    search. And each probe is a *slice* of the query rather than a join of its
    tokens, so no offset ever points at a string the text does not contain --
    which is also why a form separated differently from the way it was
    declared does not match, and must not.
    """
    query = "my golden retriever has been limping"
    tokens = token_spans(query)
    assert len(tokens) == 6

    probes = [
        query[tokens[first][0] : tokens[first + length - 1][1]]
        for length in range(1, len(tokens) + 1)
        for first in range(len(tokens) - length + 1)
    ]
    assert len(probes) == 21
    assert "golden retriever" in probes
    assert query[3:19] == "golden retriever"


def test_a_scanning_rung_reports_where_each_declared_form_sat(
    mammals_v11_path: Path,
) -> None:
    """The hook is sufficient, and overlapping forms are both returned.

    ``"my golden retriever has been limping"`` carries two declared forms at
    overlapping spans. Both are candidates: choosing the longer one and
    dropping the other is a verdict, and this family refuses verdicts -- the
    offsets make the containment visible and the consumer decides.

    Coverage then merges them, because it is the union of the spans rather
    than a list of what each rung reported, and the residue is trimmed: the
    gap between two matches is bounded by them and so begins and ends on
    whatever separated them.
    """
    onto = load_ontology(mammals_v11_path)
    resolver = CascadingResolver([ConsumerScanningSignal(onto.entities)], onto.entities)

    result = resolver.resolve("my golden retriever has been limping", k=5)

    assert [c.entity_id for c in result.candidates] == ["golden_retriever", "retriever"]
    assert result.explain("golden_retriever")[0].span == (3, 19)
    assert result.explain("retriever")[0].span == (10, 19)
    assert result.explain("golden_retriever")[0].matched_text == "golden retriever"

    # The union, not the two reported spans: (10, 19) is inside (3, 19).
    assert result.coverage.matched == ((3, 19),)
    assert result.matched_text() == ("golden retriever",)
    assert result.coverage.unmatched == ((0, 2), (20, 36))
    assert result.unmatched_text() == ("my", "has been limping")


async def test_the_span_hook_is_on_both_halves_of_the_twin(
    mammals_v11_path: Path,
) -> None:
    """The asynchronous rung locates the same forms in the same places.

    Asserted against the synchronous answer rather than against a literal, so
    the claim is *the twins agree* rather than two copies of one expectation
    that can be edited apart.
    """
    query = "my golden retriever has been limping"
    onto = load_ontology(mammals_v11_path)
    expected = CascadingResolver([ConsumerScanningSignal(onto.entities)], onto.entities).resolve(
        query, k=5
    )

    async_onto = await async_load_ontology(mammals_v11_path)
    resolver = AsyncCascadingResolver(
        [AsyncConsumerScanningSignal(async_onto.entities)], async_onto.entities
    )
    result = await resolver.resolve(query, k=5)

    assert [c.entity_id for c in result.candidates] == [c.entity_id for c in expected.candidates]
    assert result.coverage == expected.coverage
    assert [e.span for e in result.explain("golden_retriever")] == [(3, 19)]


def test_k_counts_entities_rather_than_places(mammals_v11_path: Path) -> None:
    """One entity named twice is one candidate carrying two spans.

    A rung that spent two of the caller's ``k`` on one entity would return
    fewer entities than asked for while the vocabulary could have filled the
    list -- and it would also disagree with what the *cascade* does when two
    rungs produce the same id, which is to append evidence and move nothing.
    """
    onto = load_ontology(mammals_v11_path)
    resolver = CascadingResolver([ConsumerScanningSignal(onto.entities)], onto.entities)

    result = resolver.resolve("a beagle met a beagle", k=1)

    assert [c.entity_id for c in result.candidates] == ["beagle"]
    assert [e.span for e in result.explain("beagle")] == [(2, 8), (15, 21)]
    assert result.coverage.matched == ((2, 8), (15, 21))
    assert result.matched_text() == ("beagle", "beagle")
    assert result.unmatched_text() == ("a", "met a")


def test_the_order_a_declared_rung_publishes_is_its_own(mammals_path: Path) -> None:
    """``_order`` is a hook because a declared score carries no order.

    Alphabetical is stable and means nothing beyond stability, which is the
    right default for a rung whose hits are all one form. A rung with
    something to say about the order is the only layer that *can* say it, so
    it says it here rather than by a number the cascade would have to trust.
    """

    class ReversedSignal(ExactNormalizedSignal):
        key = "reversed"

        def _order(self, hits: frozenset[str]) -> Sequence[str]:
            return sorted(hits, reverse=True)

    onto = load_ontology(mammals_path)
    shared = {"dog", "beagle"}

    plain = ExactNormalizedSignal(onto.entities)
    assert list(plain._order(frozenset(shared))) == ["beagle", "dog"]
    assert list(ReversedSignal(onto.entities)._order(frozenset(shared))) == ["dog", "beagle"]


def test_a_whole_string_rung_still_goes_through_the_span_hook(
    mammals_path: Path,
) -> None:
    """``_hits`` stays, and the base class locates what it returns.

    The slot-filling path -- where the caller supplies the value and the
    string *is* the phrase -- needs no scan and gets an offset anyway, so a
    caller reading ``span`` never has to know which kind of rung answered.
    Two rungs, two pieces of evidence, one span.

    That is the whole of the claim. It is **not** that a scan subsumes this
    path: ``test_neither_rung_subsumes_the_other`` measures a declared form
    each one reaches and the other cannot, in both directions.
    """
    onto = load_ontology(mammals_path)
    resolver = CascadingResolver(
        [ExactNormalizedSignal(onto.entities), AliasSignal(onto.entities)], onto.entities
    )

    result = resolver.resolve("Beagles", k=5)

    assert [e.signal for e in result.explain("beagle")] == ["exact", "alias"]
    assert {e.span for e in result.explain("beagle")} == {(0, 7)}
    assert result.coverage.matched == ((0, 7),)


def test_the_shipped_rung_and_an_independent_one_agree(mammals_v11_path: Path) -> None:
    """Two implementations of one description, compared against each other.

    ``ConsumerScanningSignal`` was written from outside the package against the
    published hook, before ``ScanningSignal`` existed. Asserting the shipped
    class against a literal would only say it matches what somebody typed
    beside it; asserting it against the independent one says the description
    was sufficient -- which is the claim the fixture was written to support and
    the only thing it is still uniquely good for.

    Candidates, spans and coverage, because a rung that agreed on *which*
    entities and not on *where* would be the interesting way for this to fail.
    """
    onto = load_ontology(mammals_v11_path)
    query = "my golden retriever has been limping"

    independent = CascadingResolver([ConsumerScanningSignal(onto.entities)], onto.entities).resolve(
        query, k=5
    )
    shipped = CascadingResolver([ScanningSignal(onto.entities)], onto.entities).resolve(query, k=5)

    assert [c.entity_id for c in shipped.candidates] == [
        c.entity_id for c in independent.candidates
    ]
    assert shipped.coverage == independent.coverage
    assert [(e.span, e.matched_text) for e in shipped.explain("golden_retriever")] == [
        (e.span, e.matched_text) for e in independent.explain("golden_retriever")
    ]
    assert [e.signal for e in shipped.explain("retriever")] == ["scan"]


async def test_the_shipped_twins_locate_the_same_forms(mammals_v11_path: Path) -> None:
    """The asynchronous half of the shipped pair, against the synchronous one.

    The same shape as the hook's own twin test above, one layer up: the claim
    is *the twins agree*, so neither half can be edited to a new answer alone.
    """
    query = "my golden retriever has been limping"
    onto = load_ontology(mammals_v11_path)
    expected = CascadingResolver([ScanningSignal(onto.entities)], onto.entities).resolve(query, k=5)

    async_onto = await async_load_ontology(mammals_v11_path)
    result = await AsyncCascadingResolver(
        [AsyncScanningSignal(async_onto.entities)], async_onto.entities
    ).resolve(query, k=5)

    assert [c.entity_id for c in result.candidates] == [c.entity_id for c in expected.candidates]
    assert result.coverage == expected.coverage
    assert [e.span for e in result.explain("golden_retriever")] == [(3, 19)]


def test_neither_rung_subsumes_the_other() -> None:
    """The scan and the whole-string rung each reach what the other cannot.

    This docstring's claim used to be the opposite, in
    ``signals._candidate`` and in the design it came from: that a scan
    *subsumes* whole-string matching. It does not, and the reason is a property
    of the boundary policy rather than a bug in either rung -- every probe is a
    slice **between** token boundaries, so a declared form whose first or last
    character is not alphanumeric is never probed at all.

    It survived because the sentence lived in a private function's docstring,
    which nothing reads, and because no fixture in this repository declared a
    form with a punctuated edge. That is the shape of defect this file now
    carries a measurement for rather than a sentence.
    """
    punctuated = MappingEntitySource(
        {"k9": Entity(id="k9", type="Thing", name="K-9", aliases=("(beagle)", "C.D.C."))}
    )
    scan = CascadingResolver([ScanningSignal(punctuated)], punctuated)
    whole = CascadingResolver([ExactNormalizedSignal(punctuated)], punctuated)

    # The whole-string rung reaches a form the scan never probes.
    for form, span in (("(beagle)", (0, 8)), ("C.D.C.", (0, 6))):
        assert [c.entity_id for c in whole.resolve(form, k=5).candidates] == ["k9"]
        assert whole.resolve(form, k=5).explain("k9")[0].span == span
        assert scan.resolve(form, k=5).candidates == ()

    # An interior boundary character is fine: the edges are what matter.
    assert token_spans("(beagle)") == ((1, 7),)
    assert token_spans("K-9") == ((0, 1), (2, 3))
    assert [c.entity_id for c in scan.resolve("K-9", k=5).candidates] == ["k9"]

    # And the other direction, which is why neither is the stronger rung.
    plain = MappingEntitySource(
        {"beagle": Entity(id="beagle", type="Breed", name="Beagle", aliases=("Beagles",))}
    )
    inside = CascadingResolver([ScanningSignal(plain)], plain).resolve("Beagles!", k=5)

    assert [c.entity_id for c in inside.candidates] == ["beagle"]
    assert inside.explain("beagle")[0].span == (0, 7)
    assert (
        CascadingResolver([ExactNormalizedSignal(plain)], plain).resolve("Beagles!", k=5).candidates
        == ()
    )


def test_a_reference_records_which_rung_and_what_kind() -> None:
    """A stored resolution is judgeable, and so is every alternative it ranked.

    ``signals`` was a mapping of rung name to score: it recorded which rungs
    fired and could not record what any of them meant. ``runners_up`` was a
    tuple of id and bare float, with the same hole one field over. Both are
    replaced by something carrying ``kind``, which is the half that separates
    a declared alias from a vector guess -- the distinction the whole family
    exists to keep, and the one a row in a database loses first.
    """
    runner = RunnerUp(
        entity_id="retriever",
        evidence=MatchEvidence(
            signal="semantic",
            kind=EvidenceKind.INFERRED,
            score=0.83,
            scoring=Scoring.NORMALIZED,
            matched_text="golden retriever",
            span=None,
        ),
    )
    ref = ResolutionRef(
        query="my golden retriever has been limping",
        entity_id="golden_retriever",
        score=1.0,
        scoring=Scoring.DECLARED,
        signal="scan",
        kind=EvidenceKind.DECLARED,
        compatibility=CompatibilityVerdict.UNKNOWN,
        corpus={},
        span=(3, 19),
        runners_up=(runner,),
    )

    assert ref.kind is EvidenceKind.DECLARED
    assert ref.query[ref.span[0] : ref.span[1]] == "golden retriever"
    assert ref.runners_up[0].evidence.kind is EvidenceKind.INFERRED
    assert not hasattr(ref, "signals")

    # Compared field-wise, and therefore unhashable: two references recording
    # one resolution are one reference.
    assert dataclasses.replace(ref) == ref
    with pytest.raises(TypeError):
        hash(ref)
