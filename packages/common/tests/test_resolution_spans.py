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
import re
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.entity_resolution import (
    AliasSignal,
    AsyncCascadingResolver,
    AsyncDeclaredSignal,
    AsyncScanningSignal,
    CascadingResolver,
    CompatibilityVerdict,
    DeclaredSignal,
    EntityCandidate,
    EvidenceKind,
    ExactNormalizedSignal,
    FormHit,
    MatchEvidence,
    ResolutionRef,
    ResolutionResult,
    RunnerUp,
    ScanningSignal,
    Scoring,
    content_span,
    token_spans,
)
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import (
    AsyncMappingEntitySource,
    Entity,
    MappingEntitySource,
    async_load_ontology,
    load_ontology,
)
from dataknobs_common.text import default_normalizer

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
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


class _CountingSource:
    """The flavourless half of a source that tallies the lookups spent on it.

    Delegation to a real :class:`MappingEntitySource` plus a counter, rather
    than a double standing in for one: every answer below is the real index's,
    so the rung under test runs its real path and the tally describes that
    path rather than an approximation of it.

    What it measures is the half of a rung's behaviour no assertion on a
    *result* can reach. Two rungs probing different numbers of windows return
    identical candidates, so a cost is only visible by counting -- which is
    why an unbounded enumeration survived a suite that asserted thoroughly on
    what came back, on both flavours and for two different reasons.

    **The bound accessor is here rather than on each twin**, because it is the
    number the tests below assert against and a counting source that reported
    it differently from its own index would make every one of them measure
    itself. The flavoured members are the ones the protocols spell
    differently, and they are the only ones the subclasses carry.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.probes = 0

    def describe(self) -> Any:
        return self._inner.describe()

    def longest_form_tokens(self) -> int | None:
        return self._inner.longest_form_tokens()


class CountingEntitySource(_CountingSource):
    """The synchronous flavour."""

    def __init__(
        self,
        entities: Mapping[str, Entity],
        *,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(MappingEntitySource(entities, normalizer=normalizer))

    def get(self, entity_id: str) -> Entity | None:
        return self._inner.get(entity_id)

    def get_many(self, entity_ids: Sequence[str]) -> dict[str, Entity]:
        return self._inner.get_many(entity_ids)

    def fetch_origin(self, ref: Any) -> Any:
        return self._inner.fetch_origin(ref)

    def fetch_origins(self, refs: Sequence[Any]) -> Any:
        return self._inner.fetch_origins(refs)

    def by_surface_form(self, form: str) -> frozenset[str]:
        self.probes += 1
        return self._inner.by_surface_form(form)

    def by_type(self, type_id: str) -> frozenset[str]:
        return self._inner.by_type(type_id)


class AsyncCountingEntitySource(_CountingSource):
    """The asynchronous flavour, counting the round trips rather than the calls.

    The distinction is the reason this exists. A probe on the synchronous side
    is a dictionary lookup; here it is whatever the backing does when awaited,
    so an enumeration nobody bounded is the more expensive of the two mistakes
    and was the less measured.
    """

    def __init__(
        self,
        entities: Mapping[str, Entity],
        *,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(AsyncMappingEntitySource(entities, normalizer=normalizer))

    async def get(self, entity_id: str) -> Entity | None:
        return await self._inner.get(entity_id)

    async def get_many(self, entity_ids: Sequence[str]) -> dict[str, Entity]:
        return await self._inner.get_many(entity_ids)

    async def fetch_origin(self, ref: Any) -> Any:
        return await self._inner.fetch_origin(ref)

    async def fetch_origins(self, refs: Sequence[Any]) -> Any:
        return await self._inner.fetch_origins(refs)

    async def by_surface_form(self, form: str) -> frozenset[str]:
        self.probes += 1
        return await self._inner.by_surface_form(form)

    async def by_type(self, type_id: str) -> frozenset[str]:
        return await self._inner.by_type(type_id)


def test_a_scan_probes_no_window_longer_than_a_declared_form() -> None:
    """The enumeration is bounded by the vocabulary, not by the query.

    Every contiguous window of a query is *n(n+1)/2* probes -- quadratic in the
    token count, and the characters copied grow with the cube of it, since each
    probe slices a span that may be the whole query. On text a caller hands
    over, through a rung a document reaches by writing ``kind: scan``. Measured
    before this assertion existed: 20,100 probes for 200 tokens, 500,500 for
    1,000.

    **A window longer than the longest form the vocabulary declares cannot
    match it**, while the fold keeps token boundaries where it finds them --
    which ``default_normalizer``, used here, does.
    ``by_surface_form`` answers the empty set for every such window, so cutting
    the enumeration there removes only probes that could not have contributed:
    an exact bound rather than a budget, which is why the count and the answer
    are asserted together below. A cap that lost a candidate would be a
    different change needing a different argument, and
    ``test_a_fold_that_merges_tokens_leaves_the_scan_with_no_bound`` is the
    vocabulary where that implication fails and the bound is therefore not
    derived at all.
    """
    source = CountingEntitySource(
        {
            "beagle": Entity(id="beagle", type="Breed", name="Beagle"),
            "golden_retriever": Entity(
                id="golden_retriever", type="Breed", name="Golden Retriever"
            ),
        }
    )
    query = " ".join(["word"] * 38 + ["golden", "retriever"])
    tokens = len(token_spans(query))
    assert tokens == 40, "the arithmetic below is written for a forty-token query"

    found = ScanningSignal(source).candidates(query, k=5)

    assert [c.entity_id for c in found] == ["golden_retriever"]
    assert source.longest_form_tokens() == 2
    assert source.probes <= tokens * source.longest_form_tokens(), (
        f"the scan spent {source.probes} lookups on a {tokens}-token query "
        f"over a vocabulary whose longest declared form is "
        f"{source.longest_form_tokens()} tokens. Unbounded, that is "
        f"{tokens * (tokens + 1) // 2}: every window is probed, including the "
        f"ones no declared form could fill."
    )


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
        found = whole.resolve(form, k=5)
        assert [c.entity_id for c in found.candidates] == ["k9"]
        assert found.explain("k9")[0].span == span
        assert scan.resolve(form, k=5).candidates == ()

    # An interior boundary character is fine: the edges are what matter, so
    # this is the one form in the vocabulary both rungs reach -- asserted in
    # both directions, because "by both" is the claim and one half of it is
    # what the two cases above are contrasted against.
    assert token_spans("(beagle)") == ((1, 7),)
    assert token_spans("K-9") == ((0, 1), (2, 3))
    for rung in (scan, whole):
        found = rung.resolve("K-9", k=5)
        assert [c.entity_id for c in found.candidates] == ["k9"]
        assert found.explain("k9")[0].span == (0, 3)

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


def test_the_scanning_rung_proposes_ids_in_its_own_order() -> None:
    """``_order`` is live on the scanning path, not only the whole-string one.

    Both were published as a rung's extension surface, and only one of them
    was reachable: the scan spelled ``sorted(hits)`` inline, so a subclass
    overriding the hook had it read and discarded. Silently -- the default
    ``_order`` *is* ``sorted``, so every test written against the shipped rung
    agreed with the bypass.

    It is the scanning rung the hook's own docstring describes, which is what
    makes this the wrong place to have hard-coded the answer: "a rung with a
    longer form and a shorter one inside it has something to say and says it
    here". Two entities declaring one form is the same ambiguity one span
    down, and the rung is the only layer that can rank them -- a declared
    score is ``1.0`` by fiat, so nothing downstream could recover an order the
    rung did not publish.
    """

    class ReverseOrderScanningSignal(ScanningSignal):
        """The shipped rung with the one hook a consumer is invited to override."""

        def _order(self, hits: frozenset[str]) -> Sequence[str]:
            return sorted(hits, reverse=True)

    shared = MappingEntitySource(
        {
            "a_beagle": Entity(id="a_beagle", type="Breed", name="Beagle"),
            "z_beagle": Entity(id="z_beagle", type="Breed", name="Beagle"),
        }
    )

    default = ScanningSignal(shared).candidates("a beagle here", k=5)
    reversed_ = ReverseOrderScanningSignal(shared).candidates("a beagle here", k=5)

    assert [c.entity_id for c in default] == ["a_beagle", "z_beagle"]
    assert [c.entity_id for c in reversed_] == ["z_beagle", "a_beagle"], (
        "the override was read and discarded: the scan is ordering its own "
        "hits instead of asking _order, so the published hook does nothing "
        "for the one rung whose docstring describes it"
    )


async def test_the_async_scan_is_ordered_the_same_way() -> None:
    """``_order`` reaches the asynchronous twin, measured rather than assumed.

    The twin-parity guard compares *signatures*, so it would have agreed
    happily while one flavour honoured its hook and the other did not. This is
    the behaviour the sync assertion above pins, asked one ``await`` further
    in.

    **The bound is not asserted here, and used to be.** The assertion read
    ``len(_probe_spans(query, shared.longest_form_tokens())) == tokens``, which
    is arithmetic on the helper rather than a fact about this rung: it held
    whatever the twin passed, and held if the twin stopped calling it at all.
    A cost is only visible by counting what the *source* was asked, which is
    what ``test_the_async_scan_spends_the_bound_it_was_given`` does.
    """

    class ReverseOrderAsyncScanningSignal(AsyncScanningSignal):
        def _order(self, hits: frozenset[str]) -> Sequence[str]:
            return sorted(hits, reverse=True)

    # Single-token ids as well as names, so the two entities differ only in
    # the id that orders them and one probe reaches both.
    shared = AsyncMappingEntitySource(
        {
            "abeagle": Entity(id="abeagle", type="Breed", name="Beagle"),
            "zbeagle": Entity(id="zbeagle", type="Breed", name="Beagle"),
        }
    )
    query = " ".join(["word"] * 39 + ["beagle"])
    tokens = len(token_spans(query))

    default = await AsyncScanningSignal(shared).candidates(query, k=5)
    reversed_ = await ReverseOrderAsyncScanningSignal(shared).candidates(query, k=5)

    assert [c.entity_id for c in default] == ["abeagle", "zbeagle"]
    assert [c.entity_id for c in reversed_] == ["zbeagle", "abeagle"], (
        "the override was read and discarded on the asynchronous twin, and "
        "honoured on the synchronous one"
    )
    assert tokens == 40


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


# ---------------------------------------------------------------------------
# What the window bound survives, and what it does not.
# ---------------------------------------------------------------------------


def _squash(form: str) -> str:
    """A fold that deletes the characters :func:`token_spans` reads as boundaries.

    Unremarkable and legitimate -- it is how a vocabulary matches ``C.D.C.``
    against ``cdc``, or ``K-9`` against ``k9`` -- and it is the one property a
    window bound measured in *tokens* cannot survive. Every assertion in this
    section is about a fold of this shape rather than about this function.
    """
    return re.sub(r"[^0-9a-z]", "", form.casefold())


#: Two forms, one containing the other, so a scan that reaches them reports
#: both and a scan that does not reports the shorter alone. The difference
#: between the two answers is what makes the loss visible rather than a count.
RETRIEVERS = {
    "golden_retriever": Entity(id="golden_retriever", type="Breed", name="Golden Retriever"),
    "retriever": Entity(id="retriever", type="Breed", name="Retriever"),
}

#: The guide's own sentence: neither form is the whole string, so the scan is
#: the only rung that can place either of them.
SENTENCE = "my golden retriever has been limping"


async def _agreeing_scan(
    entities: Mapping[str, Entity],
    query: str,
    *,
    source_normalizer: Callable[[str], str] | None = None,
    **rung: Any,
) -> list[str]:
    """What **both** flavours answer, having asserted that it is one answer.

    The twins are one description with two drivers, so every claim in this
    section is a claim about both. Asserting them separately would state it
    twice and hold it nowhere: a fix applied to one ``_located`` and not the
    other passes the flavour it was written against, which is the shape the
    scan's own review found twice. Here the comparison *is* the assertion, so
    a divergence fails whichever side caused it.

    Spans as well as ids, because the two rungs enumerate windows in an order
    that decides where a candidate is reported to sit.
    """
    found = ScanningSignal(
        MappingEntitySource(entities, normalizer=source_normalizer), **rung
    ).candidates(query, k=5)
    awaited = await AsyncScanningSignal(
        AsyncMappingEntitySource(entities, normalizer=source_normalizer), **rung
    ).candidates(query, k=5)

    assert [c.entity_id for c in found] == [c.entity_id for c in awaited], (
        "the twins disagree on which entities a query reaches. They read one "
        "index through one enumeration and differ only in awaiting it, so a "
        "difference here is a fix that reached one _located and not the other"
    )
    assert [e.span for c in found for e in c.evidence] == [
        e.span for c in awaited for e in c.evidence
    ], "the twins agree on what a query reaches and disagree on where it sat"
    return [c.entity_id for c in found]


async def test_a_fold_that_merges_tokens_leaves_the_scan_with_no_bound() -> None:
    """A source whose own fold merges token boundaries cannot bound a window.

    The bound's derivation is that a window of *L* tokens can only match a key
    of at least *L* tokens, so a key count bounds the enumeration. That holds
    while the fold preserves boundaries, which ``default_normalizer`` does.
    ``_squash`` does not: ``Golden Retriever`` is two tokens and folds to the
    one-token key ``goldenretriever``, so the two-token window the sentence
    carries is the *only* way to reach it -- and a bound measured over the keys
    says one.

    Measured before this assertion existed: ``golden_retriever`` was not among
    the candidates, and ``coverage.matched`` reported ``(10, 19)`` -- the
    shorter form alone, with ``golden`` reported as text the vocabulary did not
    account for. A wrong answer rather than a slow one, which is the failure
    the bound was introduced to prevent in the other direction.
    """
    source = MappingEntitySource(RETRIEVERS, normalizer=_squash)

    assert source.longest_form_tokens() is None, (
        "the source reported a number for a fold that merges token "
        "boundaries. No number is correct here -- see the test below -- so "
        "reporting one is how the scan silently stops finding a declared form"
    )
    assert await _agreeing_scan(RETRIEVERS, SENTENCE, source_normalizer=_squash) == [
        "golden_retriever",
        "retriever",
    ]


async def test_no_number_bounds_a_window_once_the_fold_merges_tokens() -> None:
    """Why the answer is *stop bounding* rather than *bound wider*.

    The obvious repair is to measure the bound over the declared form as
    written as well as over the folded key -- ``Golden Retriever`` is two
    tokens, so bound two and the sentence is reachable again. It is not
    enough, and this is the measurement that says so: under a fold that
    deletes boundaries the *query* decides how many tokens a window needs, and
    it can spell the form with as many as it likes.

    Fifteen here, and fifteen is not the limit -- it is the longest spelling
    anybody bothered to write down. A bound that is finite is therefore a
    bound that is wrong for some query, which leaves declining to bound as the
    only answer that loses nothing.
    """
    source = MappingEntitySource(RETRIEVERS, normalizer=_squash)
    letter_by_letter = " ".join("goldenretriever")

    for spelling in ("golden retriever", "gold en retriever", letter_by_letter):
        assert source.by_surface_form(spelling) == frozenset({"golden_retriever"}), (
            f"{spelling!r} folds onto the declared key and must reach it"
        )

    assert len(token_spans(letter_by_letter)) == 15
    assert len(token_spans("Golden Retriever")) == 2, (
        "the declared form is two tokens, and a fifteen-token window reaches "
        "it -- so measuring the bound over the declaration would still lose"
    )


async def test_the_default_fold_merges_tokens_too_and_is_caught() -> None:
    """The check is about the fold's behaviour, not about custom normalizers.

    ``default_normalizer`` strips and case-folds, and case-folding is not
    boundary-preserving in general: ``U+0345 COMBINING GREEK YPOGEGRAMMENI``
    is a non-spacing mark, so :func:`token_spans` reads it as a boundary and
    ``a<U+0345>b`` is two tokens -- but it case-folds to ``iota``, which is
    alphanumeric, so the folded key is one. A vocabulary declaring the
    combining spelling therefore trips the same implication a squashing
    normalizer does, under the fold every caller gets by default.

    Exotic, and that is the point: the condition detected is a property of the
    fold rather than of who supplied it, so it is caught here without
    ``_squash`` anywhere in sight. Measured before the check existed: the
    query below returned no candidates.
    """
    combining = {"x": Entity(id="x", type="Thing", name="a\u0345b")}
    source = MappingEntitySource(combining)

    assert len(token_spans("a\u0345b")) == 2, "the combining mark is a token boundary"
    assert len(token_spans(default_normalizer("a\u0345b"))) == 1, "and folds away to one"
    assert source.longest_form_tokens() is None
    assert await _agreeing_scan(combining, "a\u0345b") == ["x"]


async def test_a_rung_that_folds_does_not_trust_a_bound_measured_without_it() -> None:
    """The source's number describes the source's fold, and the rung has another.

    ``normalizer=`` on a rung is the documented way to match differently from
    the way the index was built, and the registry's ``kind: scan`` factory
    hands one over. The source cannot see it: it measured its keys before the
    rung existed, so a rung that folds is asking a question the number was not
    an answer to.

    Here the index is built with the default fold and holds the single-token
    key ``goldenretriever``, so the source says one -- correctly, for itself.
    The rung folds with ``_squash``, which makes the two-token window
    ``golden retriever`` reach that key, and one is the wrong number for that
    enumeration. No source-side measurement can ever be the right one, so a
    rung that folds declines the bound instead.
    """
    entities = {
        "goldenretriever": Entity(id="goldenretriever", type="Breed", name="GoldenRetriever"),
    }
    plain = MappingEntitySource(entities)

    assert plain.longest_form_tokens() == 1, "the source's own fold keeps one token"
    assert await _agreeing_scan(entities, SENTENCE, normalizer=_squash) == ["goldenretriever"]


async def test_max_window_is_the_bound_a_caller_reasons_about_themselves() -> None:
    """The escape hatch for a vocabulary the source cannot bound.

    Declining the bound is correct and it is not free: a fold that merges
    boundaries puts the enumeration back to *n(n+1)/2*, which is where the
    scan started. A caller who knows their own queries can say how wide a
    window is worth probing, and that is a number they chose rather than one
    derived from a vocabulary that cannot support it.

    So it caps, and it is honest about what capping costs: at two, the
    two-token spelling is reached and the fifteen-token one is not. That is a
    *budget*, unlike the derived bound, and the assertion below says so by
    measuring both what it keeps and what it gives up.
    """
    source = CountingEntitySource(RETRIEVERS, normalizer=_squash)
    query = " ".join(["word"] * 34 + SENTENCE.split())
    tokens = len(token_spans(query))
    assert tokens == 40, "the arithmetic below is written for a forty-token query"

    found = ScanningSignal(source, max_window=2).candidates(query, k=5)

    assert [c.entity_id for c in found] == ["golden_retriever", "retriever"]
    assert source.probes <= tokens * 2, (
        f"a max_window of 2 spent {source.probes} lookups on {tokens} tokens; "
        f"unbounded that is {tokens * (tokens + 1) // 2}"
    )
    assert (
        await _agreeing_scan(
            RETRIEVERS, " ".join("goldenretriever"), source_normalizer=_squash, max_window=2
        )
        == []
    ), "a cap is a budget: the spelling wider than it is given up, in both flavours"


async def test_both_scans_refuse_a_window_narrower_than_one_token() -> None:
    """Zero and negatives, refused by both flavours in the same words.

    ``_checked_max_window`` is shared *so that the twins refuse identically*,
    and that claim is the half no other guard reaches. The parity check over
    the two ``__init__`` compares signatures, so a flavour that stopped
    calling it would keep its keyword, keep its annotation, and pass. What it
    would do instead is accept ``max_window=0`` and report an empty result for
    every query -- indistinguishable from a vocabulary that matches nothing,
    which is the silence the refusal exists to prevent. So the messages are
    compared and not only the type: a flavour refusing in its own words is a
    flavour refusing on its own, one edit away from not refusing at all.

    ``1`` is asserted accepted beside them because that is where an off-by-one
    would hide, and because a scan capped at a single token is a legitimate
    thing for a caller to ask for rather than the degenerate case above.
    """
    declared = {"beagle": Entity(id="beagle", type="Breed", name="Beagle")}
    synchronous = MappingEntitySource(declared)
    asynchronous = AsyncMappingEntitySource(declared)

    for refused in (0, -1, -7):
        with pytest.raises(ValidationError) as from_sync:
            ScanningSignal(synchronous, max_window=refused)
        with pytest.raises(ValidationError) as from_async:
            AsyncScanningSignal(asynchronous, max_window=refused)

        assert str(refused) in str(from_sync.value), (
            f"the refusal of max_window={refused} does not say which value was refused"
        )
        assert str(from_sync.value) == str(from_async.value), (
            f"the twins refuse max_window={refused} in different words, so one of "
            f"them has stopped reaching the shared check they are meant to share"
        )

    assert await _agreeing_scan(declared, "my beagle", max_window=1) == ["beagle"], (
        "a window of one token is the narrowest scan that probes anything, and "
        "both flavours have to build it"
    )


def test_the_bound_costs_nothing_an_unbounded_implementation_finds() -> None:
    """The losslessness claim, against an implementation that does not bound.

    Every other assertion about the bound is against an expectation somebody
    typed. This one is against ``ConsumerScanningSignal``, which was written
    from outside the package before the bound existed and still enumerates
    every window -- so *the shipped rung loses no answer* is compared with a
    rung that cannot lose one, rather than restated.

    **Parametrized over the fold, because that is what the claim turns on.**
    Under ``default_normalizer`` the bound applies and the two agree; under a
    fold that merges token boundaries the source declines to bound and the two
    agree again, which is the case that decides whether declining was the
    right answer. A bound kept over the second vocabulary would show up here
    as the shipped rung finding strictly less.
    """
    for label, fold in (("default", None), ("merging", _squash)):
        source = MappingEntitySource(RETRIEVERS, normalizer=fold)
        unbounded = CascadingResolver([ConsumerScanningSignal(source)], source).resolve(
            SENTENCE, k=5
        )
        shipped = CascadingResolver([ScanningSignal(source)], source).resolve(SENTENCE, k=5)

        assert [c.entity_id for c in shipped.candidates] == [
            c.entity_id for c in unbounded.candidates
        ], f"the bounded and unbounded rungs disagree under the {label} fold"
        assert shipped.coverage == unbounded.coverage, (
            f"the two rungs cover different text under the {label} fold"
        )

    # The control: the two folds are not the same vocabulary, so the loop
    # above is comparing two situations rather than one twice.
    assert MappingEntitySource(RETRIEVERS).longest_form_tokens() == 2
    assert MappingEntitySource(RETRIEVERS, normalizer=_squash).longest_form_tokens() is None


async def test_the_async_scan_spends_the_bound_it_was_given() -> None:
    """Finding the twin's probe count, rather than the helper's arithmetic.

    The assertion this replaces read
    ``len(_probe_spans(query, source.longest_form_tokens())) == tokens``, which
    is a property of ``_probe_spans`` alone -- true whatever the rung passes
    it, and true if the rung stopped calling it. The twin could have gone back
    to enumerating every window with the whole file still green, which matters
    here more than on the sync side: each probe on this flavour is a round trip
    for a source that is not a mapping.

    So the count comes from the source, the way the sync test takes it.
    """
    source = AsyncCountingEntitySource({"beagle": Entity(id="beagle", type="Breed", name="Beagle")})
    query = " ".join(["word"] * 39 + ["beagle"])
    tokens = len(token_spans(query))

    found = await AsyncScanningSignal(source).candidates(query, k=5)

    assert [c.entity_id for c in found] == ["beagle"]
    assert source.longest_form_tokens() == 1
    assert source.probes <= tokens, (
        f"the asynchronous scan spent {source.probes} lookups on a "
        f"{tokens}-token query over a one-token vocabulary. Unbounded that is "
        f"{tokens * (tokens + 1) // 2}, and every one of them is a round trip"
    )


def test_a_result_hands_back_the_reference_a_consumer_would_have_assembled() -> None:
    """The stored form of a resolution comes off the result, not off ten literals.

    ``ResolutionRef`` is what four of v1's use cases mean by *a hit*, and
    until now nothing in ``packages/*/src`` returned one: the only
    construction anywhere was a test assembling one field at a time, and a
    consumer driving the use cases had to do the same. The class's own
    docstring names ``SourceRef`` as its model, and a consumer never
    assembles a ``SourceRef`` -- the loader builds one and every ``Entity``
    carries it.

    ``corpus`` is the field that made the door conditional on itself: it is
    required, has no default, and nothing determined its value. It takes an
    empty mapping, on the published argument its neighbour ``compatibility``
    already makes -- ``UNKNOWN`` with ``{}`` means nobody looked, and a
    searched-and-silent corpus is a verdict beside an empty mapping. No third
    sentinel is needed because the thing that knows the difference already
    travels beside the value.
    """
    winner = EntityCandidate(
        entity_id="golden_retriever",
        score=1.0,
        evidence=(
            MatchEvidence(
                signal="scan",
                kind=EvidenceKind.DECLARED,
                score=1.0,
                scoring=Scoring.DECLARED,
                matched_text="golden retriever",
                span=(3, 19),
            ),
        ),
    )
    loser = EntityCandidate(
        entity_id="retriever",
        score=0.83,
        evidence=(
            MatchEvidence(
                signal="semantic",
                kind=EvidenceKind.INFERRED,
                score=0.83,
                scoring=Scoring.NORMALIZED,
                matched_text="golden retriever",
                span=None,
            ),
        ),
    )
    result = ResolutionResult(
        candidates=(winner, loser),
        query="my golden retriever has been limping",
        compatibility=CompatibilityVerdict.UNKNOWN,
    )

    ref = result.ref()

    assert isinstance(ref, ResolutionRef)
    assert ref.entity_id == "golden_retriever"
    assert ref.query == result.query
    assert ref.score == 1.0
    assert ref.scoring is Scoring.DECLARED
    assert ref.signal == "scan"
    assert ref.kind is EvidenceKind.DECLARED
    assert ref.span == (3, 19)
    assert ref.compatibility is CompatibilityVerdict.UNKNOWN
    assert ref.corpus == {}

    # The losing candidates, reshaped -- each carrying its rung of record's
    # evidence whole, which is what ``RunnerUp`` holds rather than a float.
    assert len(ref.runners_up) == 1
    assert ref.runners_up[0].entity_id == "retriever"
    assert ref.runners_up[0].evidence.kind is EvidenceKind.INFERRED
    assert ref.runners_up[0].evidence.score == 0.83


def test_a_reference_can_be_taken_for_a_candidate_that_did_not_win() -> None:
    """``entity_id=`` names the subject; everything else becomes a runner-up.

    A consumer storing *the one a person chose* rather than *the one that
    ranked first* is the case this argument exists for, and it is the reason
    the parameter is not simply ``ranked()[0]`` spelled shorter. Order among
    the runners-up is the result's own, with the named subject removed.
    """
    candidates = tuple(
        EntityCandidate(
            entity_id=key,
            score=score,
            evidence=(
                MatchEvidence(
                    signal="scan",
                    kind=EvidenceKind.DECLARED,
                    score=score,
                    scoring=Scoring.DECLARED,
                    matched_text=key,
                    span=None,
                ),
            ),
        )
        for key, score in (("a", 1.0), ("b", 0.7), ("c", 0.4))
    )
    result = ResolutionResult(candidates=candidates, query="a b c")

    ref = result.ref(entity_id="b")

    assert ref.entity_id == "b"
    assert ref.score == 0.7
    assert tuple(runner.entity_id for runner in ref.runners_up) == ("a", "c")


def test_a_reference_refuses_the_two_things_it_cannot_identify() -> None:
    """An id no candidate carries, and a result with no candidates at all.

    ``KeyError`` for the first, on ``explain()``'s own precedent and for its
    reason: *not a candidate* and *a candidate with nothing to say* must not
    be the same answer. ``ValueError`` for the second, because a miss has no
    subject to name and a reference identifies one -- returning something for
    a resolution that resolved nothing is the failure the whole family is
    built to refuse.
    """
    resolved = ResolutionResult(
        candidates=(
            EntityCandidate(
                entity_id="a",
                score=1.0,
                evidence=(
                    MatchEvidence(
                        signal="scan",
                        kind=EvidenceKind.DECLARED,
                        score=1.0,
                        scoring=Scoring.DECLARED,
                        matched_text="a",
                    ),
                ),
            ),
        ),
        query="a",
    )
    with pytest.raises(KeyError):
        resolved.ref(entity_id="nobody")

    missed = ResolutionResult(candidates=(), query="a")
    with pytest.raises(ValueError, match="no candidates"):
        missed.ref()


def test_a_reference_carries_the_rung_of_record_when_a_candidate_has_none() -> None:
    """An inherited match has no evidence, and the reference says so rather than lying.

    ``EntityCandidate.evidence`` is empty *"only where a match was inherited
    rather than made"*. The three fields taken off the rung of record have no
    source in that case, and inventing a ``signal`` string or claiming
    ``DECLARED`` would put a rung's name on a match no rung made. The
    reference reports the empty signal and ``INFERRED`` -- the kind that
    means *this was not found in the surface* -- and takes its ``scoring``
    from the candidate rather than from a rung.
    """
    inherited = EntityCandidate(entity_id="mammal", score=0.5, evidence=())
    result = ResolutionResult(candidates=(inherited,), query="cat")

    ref = result.ref()

    assert ref.signal == ""
    assert ref.kind is EvidenceKind.INFERRED
    assert ref.scoring is Scoring.NATIVE
    assert ref.span is None


def test_an_evidence_free_loser_is_left_out_rather_than_given_an_invented_rung() -> None:
    """The one lossy edge of the door, asserted so it is a limit and not a surprise.

    ``RunnerUp.evidence`` is one required ``MatchEvidence``, so a losing
    candidate that carries none cannot be represented. Inventing a rung name
    for it is what the subject path refuses; widening the field is a change
    to a shipped value type and outside a convenience door's remit. So it is
    dropped, and this pins that -- the winner still resolves, the
    evidence-bearing loser still appears, and the count says which.
    """
    winner = EntityCandidate(
        entity_id="a",
        score=1.0,
        evidence=(
            MatchEvidence(
                signal="scan",
                kind=EvidenceKind.DECLARED,
                score=1.0,
                scoring=Scoring.DECLARED,
                matched_text="a",
            ),
        ),
    )
    spoken_for = EntityCandidate(
        entity_id="b",
        score=0.6,
        evidence=(
            MatchEvidence(
                signal="semantic",
                kind=EvidenceKind.INFERRED,
                score=0.6,
                scoring=Scoring.NORMALIZED,
                matched_text="a",
            ),
        ),
    )
    inherited = EntityCandidate(entity_id="c", score=0.3, evidence=())
    result = ResolutionResult(candidates=(winner, spoken_for, inherited), query="a")

    ref = result.ref()

    assert tuple(runner.entity_id for runner in ref.runners_up) == ("b",)
