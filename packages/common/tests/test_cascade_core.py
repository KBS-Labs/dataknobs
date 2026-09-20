"""The control flow itself: sharing, ordering, saturation and provenance.

None of these is named by one of the two criteria this leg moves, and all of
them are required. What they guard is the half a signature cannot show -- that
both flavours run *the same* merge rather than two that agree today, and that
a candidate's position and evidence follow the rules the cascade was chosen
for.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import pkgutil
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common import SyncLoopBridge, entity_resolution
from dataknobs_common.entity_resolution import (
    AliasSignal,
    AsyncAliasSignal,
    AsyncCascadingResolver,
    AsyncDeclaredSignal,
    AsyncEntityResolver,
    AsyncExactNormalizedSignal,
    AsyncMatchSignal,
    AsyncScanningSignal,
    BridgedEntityResolver,
    CascadeState,
    CascadingResolver,
    DeclaredSignal,
    EntityCandidate,
    EntityResolver,
    EvidenceKind,
    ExactNormalizedSignal,
    MatchEvidence,
    MatchSignal,
    MembershipOracle,
    ENTITY_TYPE_KEY,
    ScanningSignal,
    Scoring,
    cascade as cascade_module,
    signals as signals_module,
)
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import (
    AsyncMappingEntitySource,
    Entity,
    MappingEntitySource,
    async_load_ontology,
    load_ontology,
)
from dataknobs_common.testing import (
    assert_no_leaked_bridge_threads,
    live_dk_daemon_threads,
    assert_twin_types_agree,
    assert_twins_agree,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


#: A source holding nothing, for the tests whose rungs are stubs.
#:
#: Sound because an unscoped ``resolve`` never consults the cascade's source at
#: all: the authority is asked only when ``within`` names an axis, and these
#: tests are about ordering, saturation and provenance. A stub's entity ids are
#: chosen for those questions and are in no vocabulary.
NO_ENTITIES = MappingEntitySource({})
NO_ENTITIES_ASYNC = AsyncMappingEntitySource({})


class StubSignal:
    """A rung returning what a test hands it.

    A real implementation of the protocol, in the test that needs a rung whose
    answers are chosen rather than derived from a vocabulary. Nothing in
    ``dataknobs`` ships a configurable rung and building one in core code
    would be shipping a construct for one test's benefit.
    """

    def __init__(self, name: str, hits: Sequence[tuple[str, float]]) -> None:
        self._name = name
        self._hits = tuple(hits)
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def narrows(self) -> bool:
        return False

    def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate]:
        self.calls += 1
        return [
            EntityCandidate(
                entity_id=entity_id,
                score=score,
                evidence=(
                    MatchEvidence(
                        signal=self._name,
                        kind=EvidenceKind.DECLARED,
                        score=score,
                        scoring=Scoring.DECLARED,
                        matched_text=query,
                    ),
                ),
            )
            for entity_id, score in self._hits
        ][:k]

    def candidates_many(
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate]]:
        return [self.candidates(query, k, filter=filter) for query in queries]


class AsyncStubSignal(StubSignal):
    """:class:`StubSignal` with the two reading members awaited."""

    async def candidates(  # type: ignore[override]
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate]:
        return StubSignal.candidates(self, query, k, filter=filter)

    async def candidates_many(  # type: ignore[override]
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate]]:
        return [await self.candidates(query, k, filter=filter) for query in queries]


# --------------------------------------------------------------------------
# One core under both flavours
# --------------------------------------------------------------------------


def test_patching_the_core_changes_both_flavours(monkeypatch: pytest.MonkeyPatch) -> None:
    """The delegation test, and it is the reason the core is a module function.

    Surface equality cannot tell *calls the core* from *agrees with the core
    today*. This can: one patch, and if either resolver had its own copy of
    the merge, that half would go on returning the unpatched answer.

    De-duplication is where it would bite first, and invisibly -- two rungs
    returning one id is the only input that distinguishes the two
    implementations, and nothing else in the suite would notice.
    """

    def refuse_everything(
        state: CascadeState,
        produced: Sequence[EntityCandidate],
        *,
        signal: str,
        kind: EvidenceKind,
    ) -> CascadeState:
        return state

    def sync_result() -> tuple[EntityCandidate, ...]:
        return (
            CascadingResolver([StubSignal("a", [("x", 1.0)])], NO_ENTITIES).resolve("q").candidates
        )

    def async_result() -> tuple[EntityCandidate, ...]:
        return asyncio.run(
            AsyncCascadingResolver([AsyncStubSignal("a", [("x", 1.0)])], NO_ENTITIES_ASYNC).resolve(
                "q"
            )
        ).candidates

    # Unpatched first, and this half is the test. "Both went empty" is also
    # true of two resolvers that never produced anything -- so without showing
    # that each yields a candidate with the core intact, the assertions below
    # would hold against a cascade that was broken rather than delegating.
    assert sync_result()
    assert async_result()

    monkeypatch.setattr(cascade_module, "merge_rung", refuse_everything)

    assert sync_result() == ()
    assert async_result() == ()


#: The twin pairs this family declares one surface for. Named rather than
#: written inline at the ``parametrize`` because
#: ``test_the_bridge_is_not_one_of_the_twins`` has to read it: the claim it
#: makes is about what is *absent* here, and a list only ``parametrize`` can
#: see is one no test can make that claim about.
_TWIN_PAIRS = [
    (EntityResolver, AsyncEntityResolver, ["resolve", "resolve_many"], (), ()),
    (MatchSignal, AsyncMatchSignal, ["candidates", "candidates_many"], (), ()),
    (CascadingResolver, AsyncCascadingResolver, ["resolve", "resolve_many"], (), ()),
    (
        ExactNormalizedSignal,
        AsyncExactNormalizedSignal,
        ["candidates", "candidates_many", "_hits"],
        (),
        (),
    ),
    (AliasSignal, AsyncAliasSignal, ["candidates", "candidates_many", "_hits"], (), ()),
    (
        ScanningSignal,
        AsyncScanningSignal,
        # ``__init__`` is checked by
        # ``test_the_scanning_twins_constructors_agree`` rather than here:
        # ``flavour_typed`` is declared once for the whole pair and
        # compared against *every* member, so a parameter flavoured on one
        # member only cannot be expressed in this table.
        ["candidates", "candidates_many", "_located"],
        (),
        (),
    ),
    (
        DeclaredSignal,
        AsyncDeclaredSignal,
        ["candidates", "candidates_many", "_hits", "_located", "_fold", "_order"],
        ("_fold", "_order"),
        (),
    ),
]


@pytest.mark.parametrize(
    ("sync_type", "async_type", "members", "unflavoured", "flavour_typed"),
    _TWIN_PAIRS,
)
def test_the_twins_expose_one_surface(
    sync_type: type,
    async_type: type,
    members: list[str],
    unflavoured: tuple[str, ...],
    flavour_typed: tuple[str, ...],
) -> None:
    """Parity over **annotations**, not only names and defaults.

    The weaker form passes a twin whose ``within`` still reads
    ``str | Collection[str] | None`` while the other half has widened -- which
    is precisely the drift the alias was published to prevent, arriving
    through the test written to catch it.

    **The hooks are listed, not only the surface.** ``_hits``, ``_located``,
    ``_fold`` and ``_order`` are what a consumer overrides to write a rung, so
    they are the pair's extension surface and drift there reaches consumer
    code directly -- a keyword added to the asynchronous ``_located`` and
    forgotten on the synchronous one is wrong against whichever half its
    author did not reach for. ``_fold`` and ``_order`` are declared
    ``unflavoured`` because they stay synchronous on both halves for the
    reason their docstrings give: ordering a set that has already arrived
    reaches for nothing. That declaration relaxes the flavour assertion and
    nothing else, so the parameters are still compared.
    """
    assert_twin_types_agree(
        sync_type,
        async_type,
        members,
        unflavoured_members=unflavoured,
        flavour_typed=flavour_typed,
        compare_return=True,
    )


def test_the_scanning_twins_constructors_agree() -> None:
    """``max_window`` and ``normalizer`` are the same parameters on both rungs.

    These two are the only rungs that override ``__init__`` -- every other
    rung inherits the base's whole -- so they are the only pair whose
    constructors can drift. A keyword added to one flavour and forgotten on
    the other is wrong against whichever half a consumer reached for, and no
    assertion on a *result* would reach it: the twin that lacks the keyword
    raises ``TypeError`` at construction, in the consumer's code rather than
    in ours.

    Checked here rather than in the table above because ``entities`` is
    flavoured by definition -- ``EntitySource`` against ``AsyncEntitySource``
    -- while ``candidates`` and ``_located`` carry no flavoured parameter, and
    ``flavour_typed`` is declared once per pair and compared against every
    member of it. Declaring it there would fail the members that do not have
    it.
    """
    assert_twins_agree(
        ScanningSignal.__init__,
        AsyncScanningSignal.__init__,
        flavour_typed=("entities",),
        unflavoured=True,
        compare_return=True,
    )


def test_the_bridge_is_not_one_of_the_twins() -> None:
    """It satisfies the synchronous protocol by forwarding, not by twinning.

    Stated as a test because the parity list above is hand-written, and the
    obvious maintenance move on it is to add every class that has both
    flavours' members. This one does not have a twin to compare against: the
    thing it would be twinned with is the resolver it holds.

    Two halves, because the docstring makes two claims. That the bridge is
    **not in the table** is the maintenance move this test is named for, and
    only a reading of the table itself can catch it --- which is why
    ``_TWIN_PAIRS`` is a name rather than a literal inside the
    ``parametrize``. That it *should* not be is settled by the flavour of the
    resolution members, checked against the async cascade as a control. This asserted ``__aenter__``'s
    absence when it was written, which was the same claim only for as long as
    the class had no async anything: it has since gained ``aclose()`` and the
    ``async with`` that pairs with it, and a second way to *tear down* is not
    a second flavour of resolving. Both context-manager protocols on one
    object is the shape ``SimpleFSM`` already ships.
    """
    declared = {pair[0] for pair in _TWIN_PAIRS} | {pair[1] for pair in _TWIN_PAIRS}
    assert BridgedEntityResolver not in declared, (
        "the bridge was added to the twin table; it forwards to a resolver rather "
        "than twinning one, so there is no second half to hold it to"
    )

    assert hasattr(BridgedEntityResolver, "__enter__")
    assert hasattr(BridgedEntityResolver, "__exit__")
    assert not inspect.iscoroutinefunction(BridgedEntityResolver.resolve)
    assert not inspect.iscoroutinefunction(BridgedEntityResolver.resolve_many)
    assert inspect.iscoroutinefunction(AsyncCascadingResolver.resolve)


# --------------------------------------------------------------------------
# Saturation, position and provenance
# --------------------------------------------------------------------------


def test_one_candidate_two_rungs_two_evidences(mammals_path: Path) -> None:
    """The provenance guarantee, under a control flow with no merge step.

    Runnable with the two declared rungs and nothing else: ``beagles`` is
    ``beagle``'s alias, so the exact rung finds it through the folded map and
    the alias rung finds it through the alias map. The candidate appears
    **once**, positioned by the earlier rung, carrying **two** pieces of
    evidence -- and the two are genuinely different witnesses rather than one
    answer under two names, because they came from different indexes.
    """
    ontology = load_ontology(mammals_path)
    resolver = CascadingResolver(
        [ExactNormalizedSignal(ontology.entities), AliasSignal(ontology.entities)],
        ontology.entities,
    )

    result = resolver.resolve("beagles", k=5)

    assert [c.entity_id for c in result.candidates] == ["beagle"]
    assert [e.signal for e in result.candidates[0].evidence] == ["exact", "alias"]
    assert [e.signal for e in result.explain("beagle")] == ["exact", "alias"]


def test_explain_refuses_an_id_that_is_not_a_candidate(mammals_path: Path) -> None:
    """``()`` is a real answer for a real candidate, so it cannot mean *absent*."""
    ontology = load_ontology(mammals_path)
    resolver = CascadingResolver([ExactNormalizedSignal(ontology.entities)], ontology.entities)

    result = resolver.resolve("beagles", k=5)

    with pytest.raises(KeyError):
        result.explain("mammal")


def test_position_is_fixed_by_the_first_rung_that_produced_an_id() -> None:
    """A later rung appends evidence and moves nothing, whatever it scores.

    The regression guard for the ruling the cascade exists to carry: a
    declared hit outranks a higher-scoring one from further down, and it does
    so because of *where the rung sits*, not because of a weight anywhere.
    """
    first = StubSignal("first", [("a", 0.1)])
    second = StubSignal("second", [("b", 0.9), ("a", 0.9)])

    result = CascadingResolver([first, second], NO_ENTITIES).resolve("q", k=5)

    assert [c.entity_id for c in result.candidates] == ["a", "b"]
    assert result.candidates[0].score == pytest.approx(0.1)
    assert [e.signal for e in result.explain("a")] == ["first", "second"]


def test_a_rung_is_not_consulted_once_k_is_filled() -> None:
    """Asserted by call count, not by output.

    An unconsulted rung and a rung whose hits lost look identical downstream,
    so the only place the difference is visible is at the rung itself.
    """
    first = StubSignal("first", [("a", 1.0), ("b", 1.0)])
    second = StubSignal("second", [("c", 1.0)])

    result = CascadingResolver([first, second], NO_ENTITIES).resolve("q", k=2)

    assert [c.entity_id for c in result.candidates] == ["a", "b"]

    # The counter is shown to count before its zero is believed. This whole
    # criterion rests on a call count rather than on output, so a counter that
    # had stopped incrementing -- or a cascade that had stopped consulting
    # rungs at all -- would satisfy the assertion below while proving the
    # opposite of what it claims.
    assert first.calls == 1
    assert second.calls == 0


def test_a_rung_is_asked_for_k_rather_than_for_the_remainder() -> None:
    """Saturation, not short-circuit: a rung never learns what came before it."""
    seen: list[int] = []

    class RecordingSignal(StubSignal):
        def candidates(
            self, query: str, k: int, *, filter: dict[str, Any] | None = None
        ) -> list[EntityCandidate]:
            seen.append(k)
            return StubSignal.candidates(self, query, k, filter=filter)

    first = RecordingSignal("first", [("a", 1.0)])
    second = RecordingSignal("second", [("b", 1.0)])

    CascadingResolver([first, second], NO_ENTITIES).resolve("q", k=5)

    assert seen == [5, 5]


def test_resolve_many_answers_each_query_independently(mammals_path: Path) -> None:
    """The bulk form is the same cascade, once per query."""
    ontology = load_ontology(mammals_path)
    resolver = CascadingResolver(
        [ExactNormalizedSignal(ontology.entities), AliasSignal(ontology.entities)],
        ontology.entities,
    )

    results = resolver.resolve_many(["beagles", "canine", "nothing"], k=5)

    assert [r.query for r in results] == ["beagles", "canine", "nothing"]
    assert [c.entity_id for c in results[0].candidates] == ["beagle"]
    assert [c.entity_id for c in results[1].candidates] == ["dog"]
    assert results[2].candidates == ()


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


def test_within_keeps_only_what_the_named_type_admits(mammals_path: Path) -> None:
    """A scope names a set the vocabulary declares, and the rest is dropped."""
    ontology = load_ontology(mammals_path)
    resolver = CascadingResolver([ExactNormalizedSignal(ontology.entities)], ontology.entities)

    assert [c.entity_id for c in resolver.resolve("beagle", k=5, within="Breed").candidates] == [
        "beagle"
    ]
    assert resolver.resolve("beagle", k=5, within="Species").candidates == ()


def test_within_unions_within_an_axis_and_conjoins_across_them() -> None:
    """Both halves of the rule, on a source that publishes both axes.

    An earlier version used two invented axis names, ``a`` and ``b``, and
    passed -- because the rung-side implementation it ran against intersected
    ``by_type`` per key and never looked at the key. Under the published
    predicate the axis name is load-bearing, so a scope must name an axis the
    source actually publishes; one that does not is now refused rather than
    answered with nothing.

    That refusal is why this runs against a
    :class:`~dataknobs_common.entity_resolution.MembershipOracle`. The
    conjunctive half could previously be asserted only *negatively* -- a
    second axis excluded because no source could publish one -- and a negative
    assertion is satisfied by a scope that admits nothing for any reason at
    all. Here both readings are asserted in the direction that fails if the
    rule is wrong.
    """
    source = TwoAxisSource(
        {
            "beagle": Entity(
                id="beagle", type="Breed", name="Beagle", metadata={"habitat": "forest"}
            ),
            "dog": Entity(id="dog", type="Species", name="Dog"),
        }
    )
    resolver = CascadingResolver([ExactNormalizedSignal(source)], source)

    def placed(query: str, within: Any) -> list[str]:
        return [c.entity_id for c in resolver.resolve(query, k=5, within=within).candidates]

    # Union within one axis: either id admits.
    assert placed("beagle", {ENTITY_TYPE_KEY: ["Breed", "Species"]}) == ["beagle"]
    assert placed("dog", {ENTITY_TYPE_KEY: ["Breed", "Species"]}) == ["dog"]
    assert placed("beagle", {ENTITY_TYPE_KEY: "Breed"}) == ["beagle"]
    assert placed("beagle", {ENTITY_TYPE_KEY: "Species"}) == []

    # Conjunction across axes: admitted only where both hold.
    assert placed("beagle", {ENTITY_TYPE_KEY: "Breed", "habitat": "forest"}) == ["beagle"]
    assert placed("beagle", {ENTITY_TYPE_KEY: "Species", "habitat": "forest"}) == []
    assert placed("beagle", {ENTITY_TYPE_KEY: "Breed", "habitat": "desert"}) == []

    # An entity silent on a named axis is excluded, rather than passing every
    # filter on it. ``dog`` declares no habitat, so a scope naming one leaves
    # it out even though its type admits.
    assert placed("dog", {ENTITY_TYPE_KEY: "Species", "habitat": "forest"}) == []


def test_the_cascade_overrules_a_rung_that_answers_outside_the_scope(
    mammals_path: Path,
) -> None:
    """The reason the drop is the cascade's, asserted rather than stated.

    ``MatchSignal`` does not require two rungs to share a backing, so a
    rung-side drop lets a rung's own idea of what is in a type stand
    unchallenged. Here the rung declares ``narrows()`` false and answers with a
    ``Species`` under a ``Breed`` scope; the cascade decides against its own
    source and overrules it.
    """
    ontology = load_ontology(mammals_path)
    defiant = StubSignal("defiant", [("dog", 1.0)])
    resolver = CascadingResolver([defiant], ontology.entities)

    # Unscoped first, and this half is the test: "the scope dropped it" is also
    # true of a rung that never answered.
    assert [c.entity_id for c in resolver.resolve("anything", k=5).candidates] == ["dog"]

    assert resolver.resolve("anything", k=5, within="Breed").candidates == ()


def test_an_id_the_authority_cannot_check_is_named_rather_than_silently_dropped(
    mammals_path: Path,
) -> None:
    """The cross-source drop stays, and stops looking like a vocabulary miss.

    A candidate whose id the cascade's own source does not carry is still
    dropped under a scope -- it cannot be shown to be inside one, and admitting
    what cannot be checked is what this path exists to refuse. What the drop
    must not do is *report* like a miss. Without this field the caller gets an
    empty result whose ``unmatched`` names the query, which is exactly what a
    correctly spelled scope over a vocabulary lacking the phrase returns; the
    two are then the same answer, and only one of them is the caller's fault.

    Unscoped first, for the reason the neighbouring overrule test gives for its
    own pair: "the scope dropped it" is also true of a rung that never
    answered. That half is also the assertion that nothing is reported where
    nothing was checked -- the source lacks ``wombat`` there too.
    """
    ontology = load_ontology(mammals_path)
    stray = StubSignal("stray", [("wombat", 1.0)])
    resolver = CascadingResolver([stray], ontology.entities)

    unscoped = resolver.resolve("anything", k=5)
    assert [c.entity_id for c in unscoped.candidates] == ["wombat"]
    assert unscoped.coverage.beyond_authority == ()

    scoped = resolver.resolve("anything", k=5, within="Breed")
    assert scoped.candidates == ()
    assert scoped.coverage.beyond_authority == ("wombat",)
    # The half a caller previously got on its own, and could not read. It is
    # spans now, and the text reader is the other half of the same claim.
    assert scoped.coverage.unmatched == ((0, 8),)
    assert scoped.unmatched_text() == ("anything",)


def test_a_candidate_the_scope_merely_excludes_is_not_reported_as_uncheckable(
    mammals_path: Path,
) -> None:
    """The two drops are different, and only one of them is a report.

    ``dog`` is in the vocabulary and is a ``Species``; under a ``Breed`` scope
    it is dropped because the scope was applied and said no. That is a correct,
    silent answer. Naming it here would turn the field into *dropped* rather
    than *could not be checked*, at which point it says nothing a caller can
    act on -- every scoped query would report its own exclusions back.
    """
    ontology = load_ontology(mammals_path)
    resolver = CascadingResolver([StubSignal("defiant", [("dog", 1.0)])], ontology.entities)

    result = resolver.resolve("anything", k=5, within="Breed")

    assert result.candidates == ()
    assert result.coverage.beyond_authority == ()


def test_two_rungs_naming_the_same_unchecked_id_report_it_once(
    mammals_path: Path,
) -> None:
    """Accumulated across rungs, in first-seen order, without duplicates.

    ``matched`` is built the same way and for the same reason: a field read to
    decide what a vocabulary is missing is unusable if an id appears once per
    rung that happened to produce it. ``quokka`` is here so the order being
    asserted is a real one rather than a single element.
    """
    ontology = load_ontology(mammals_path)
    resolver = CascadingResolver(
        [
            StubSignal("first", [("wombat", 1.0), ("quokka", 0.9)]),
            StubSignal("second", [("wombat", 0.8)]),
        ],
        ontology.entities,
    )

    result = resolver.resolve("anything", k=5, within="Breed")

    assert result.coverage.beyond_authority == ("wombat", "quokka")


def test_a_filled_k_does_not_truncate_what_went_unchecked(
    mammals_path: Path,
) -> None:
    """``k`` caps the candidate list and reaches nothing else.

    An unchecked id never enters the order, so the cap that thins the order
    cannot thin it. Here one rung fills ``k=1`` with an admitted candidate and
    names two ids the authority does not carry in the same breath; both are
    reported, and the candidate list is still one long.
    """

    class Ungenerous(StubSignal):
        """A rung answering with everything it has, whatever ``k`` says.

        Permitted, and the reason the cascade caps rather than trusting: a
        rung asked for ``k`` may return more than the state has room for. A
        rung that truncates to ``k`` itself -- which :class:`StubSignal` does
        -- cannot pose this question at all, since the ids past the cap are
        never produced.
        """

        def candidates(
            self, query: str, k: int, *, filter: dict[str, Any] | None = None
        ) -> list[EntityCandidate]:
            return StubSignal.candidates(self, query, len(self._hits), filter=filter)

    ontology = load_ontology(mammals_path)
    resolver = CascadingResolver(
        [Ungenerous("mixed", [("beagle", 1.0), ("wombat", 0.9), ("quokka", 0.8)])],
        ontology.entities,
    )

    result = resolver.resolve("anything", k=1, within="Breed")

    assert [c.entity_id for c in result.candidates] == ["beagle"]
    assert result.coverage.beyond_authority == ("wombat", "quokka")


def test_the_bulk_form_reports_an_unchecked_id_against_its_own_query(
    mammals_path: Path,
) -> None:
    """One ``get_many`` covers the whole batch, and the report is still per query.

    The batch path asks the authority once for every id across every query, so
    the pairing back to a query is the thing that can be got wrong here and
    nowhere else. The rung is backed by a *different* source than the cascade
    holds, which is the shape the ruling is about rather than a stub standing
    in for it.
    """
    ontology = load_ontology(mammals_path)
    elsewhere = MappingEntitySource(
        {
            "wombat": Entity(id="wombat", type="Breed", name="Wombat"),
            "beagle": Entity(id="beagle", type="Breed", name="Beagle"),
        }
    )
    resolver = CascadingResolver([ExactNormalizedSignal(elsewhere)], ontology.entities)

    wombat, beagle = resolver.resolve_many(["wombat", "beagle"], k=5, within="Breed")

    assert wombat.candidates == ()
    assert wombat.coverage.beyond_authority == ("wombat",)
    assert [c.entity_id for c in beagle.candidates] == ["beagle"]
    assert beagle.coverage.beyond_authority == ()


def test_the_async_flavour_reports_an_unchecked_id_the_same_way(
    mammals_path: Path,
) -> None:
    """Both flavours, both forms -- the twin has its own ``get_many`` branch.

    The ``await self._entities.get_many(...)`` lines are written once per
    flavour, so an accumulator threaded through one of them is not thereby
    threaded through the other.
    """

    async def exercise() -> None:
        source = AsyncMappingEntitySource(
            {"beagle": Entity(id="beagle", type="Breed", name="Beagle")}
        )
        resolver = AsyncCascadingResolver([AsyncStubSignal("stray", [("wombat", 1.0)])], source)

        single = await resolver.resolve("anything", k=5, within="Breed")
        assert single.candidates == ()
        assert single.coverage.beyond_authority == ("wombat",)

        unscoped = await resolver.resolve("anything", k=5)
        assert unscoped.coverage.beyond_authority == ()

        batch = await resolver.resolve_many(["anything", "other"], k=5, within="Breed")
        assert [result.coverage.beyond_authority for result in batch] == [
            ("wombat",),
            ("wombat",),
        ]

    asyncio.run(exercise())


def test_a_filter_is_offered_only_to_a_rung_that_declares_it_narrows(
    mammals_path: Path,
) -> None:
    """``narrows()`` is a rung's statement and the cascade honours it.

    Handing a scope to a rung that answered ``False`` would be the caller
    ignoring the declaration. It is safe either way now -- the cascade rules on
    the scope regardless -- which is precisely why it has to be asserted: a
    correctness bug here would no longer show up as a wrong answer.
    """
    ontology = load_ontology(mammals_path)
    offered: dict[str, Any] = {}

    class Declaring(StubSignal):
        def __init__(self, name: str, narrowing: bool) -> None:
            super().__init__(name, [("beagle", 1.0)])
            self._narrowing = narrowing

        def narrows(self) -> bool:
            return self._narrowing

        def candidates(
            self, query: str, k: int, *, filter: dict[str, Any] | None = None
        ) -> list[EntityCandidate]:
            offered[self.name] = filter
            return StubSignal.candidates(self, query, k, filter=filter)

    resolver = CascadingResolver(
        [Declaring("narrowing", True), Declaring("blunt", False)], ontology.entities
    )
    resolver.resolve("beagle", k=5, within="Breed")

    assert offered["narrowing"] == {ENTITY_TYPE_KEY: ["Breed"]}
    assert offered["blunt"] is None


# --------------------------------------------------------------------------
# The bridge
# --------------------------------------------------------------------------


def test_the_bridge_resolves_from_inside_a_running_loop(mammals_path: Path) -> None:
    """Callable from inside a loop without deadlocking, and it leaks no thread."""

    async def inside_a_loop() -> list[str]:
        ontology = await async_load_ontology(mammals_path)
        inner = AsyncCascadingResolver(
            [
                AsyncExactNormalizedSignal(ontology.entities),
                AsyncAliasSignal(ontology.entities),
            ],
            ontology.entities,
        )
        with BridgedEntityResolver(inner) as bridged:
            result = bridged.resolve("beagles", k=5)
        return [c.entity_id for c in result.candidates]

    with assert_no_leaked_bridge_threads():
        assert asyncio.run(inside_a_loop()) == ["beagle"]


def test_two_bridges_can_share_one_loop_thread(mammals_path: Path) -> None:
    """``bridge=``, inherited from :class:`SyncBridgeAdapter`.

    Before the shape was declared once, this class had no way to say it: two
    resolvers meant two daemon threads, and a consumer holding one of these
    beside a ``SyncTextEmbedder`` could not ask for one thread between them.
    """
    ontology = asyncio.run(async_load_ontology(mammals_path))

    def inner() -> AsyncCascadingResolver:
        return AsyncCascadingResolver(
            [AsyncExactNormalizedSignal(ontology.entities)], ontology.entities
        )

    with assert_no_leaked_bridge_threads():
        with SyncLoopBridge(thread_name="dk-sync-resolver") as shared:
            first = BridgedEntityResolver(inner(), bridge=shared)
            second = BridgedEntityResolver(inner(), bridge=shared)
            assert first.resolve("beagles", k=5).candidates
            first.close()
            assert second.resolve("beagles", k=5).candidates, (
                "closing one holder must not end a bridge it only borrowed"
            )
            second.close()


def test_constructing_a_bridge_allocates_no_thread(mammals_path: Path) -> None:
    """Lazy, so building one to hand around costs nothing until it is used."""
    ontology = asyncio.run(async_load_ontology(mammals_path))
    inner = AsyncCascadingResolver(
        [AsyncExactNormalizedSignal(ontology.entities)], ontology.entities
    )
    before = set(live_dk_daemon_threads({"dk-sync-resolver"}))
    bridged = BridgedEntityResolver(inner)
    try:
        assert set(live_dk_daemon_threads({"dk-sync-resolver"})) == before
    finally:
        bridged.close()


def test_the_bridge_has_an_async_teardown(mammals_path: Path) -> None:
    """``aclose()``, for a holder that is itself on a loop.

    ``close()`` is synchronous and joins the loop thread; from async code
    ``aclose()`` is the form that says so at the call site. The resolver
    inside is not this object's to close either way.
    """

    async def inside_a_loop() -> None:
        ontology = await async_load_ontology(mammals_path)
        inner = AsyncCascadingResolver(
            [AsyncExactNormalizedSignal(ontology.entities)], ontology.entities
        )
        bridged = BridgedEntityResolver(inner)
        assert bridged.resolve("beagles", k=5).candidates
        await bridged.aclose()

    with assert_no_leaked_bridge_threads():
        asyncio.run(inside_a_loop())


# --------------------------------------------------------------------------
# The word this family must not spend
# --------------------------------------------------------------------------


def test_nothing_in_the_family_is_named_origin() -> None:
    """The collision guard, asserted by introspection over the built package.

    Two orthogonal axes want this word -- *which rung matched* and *did this
    entity match or inherit* -- and the second already owns it one phase on.
    The failure it guards is fatal at **construction** and passes an import
    check, so only introspection over what was actually built catches a
    reintroduction.
    """
    offenders: list[str] = []
    walked: list[str] = []

    for info in pkgutil.iter_modules(entity_resolution.__path__):
        walked.append(info.name)
        module = __import__(f"dataknobs_common.entity_resolution.{info.name}", fromlist=["_"])
        for attribute_name in dir(module):
            if attribute_name.startswith("_"):
                continue
            attribute = getattr(module, attribute_name)
            if attribute_name == "origin":
                offenders.append(f"{info.name}.{attribute_name}")
            if dataclasses.is_dataclass(attribute) and isinstance(attribute, type):
                offenders += [
                    f"{info.name}.{attribute_name}.{f.name}"
                    for f in dataclasses.fields(attribute)
                    if f.name == "origin"
                ]
            if inspect.isclass(attribute) or inspect.isfunction(attribute):
                offenders += _origin_parameters(f"{info.name}.{attribute_name}", attribute)

    # The positive control, and it is not ceremony: every assertion above is
    # that something is ABSENT, and an empty walk satisfies all of them. A
    # guard that reports clean because it read nothing is the failure this
    # whole family of checks exists to avoid, so the walk states what it
    # covered before the absence is believed.
    assert sorted(walked) == ["cascade", "protocols", "registry", "signals", "values"]
    assert offenders == []


def _origin_parameters(where: str, attribute: Any) -> list[str]:
    """Any parameter named ``origin`` on this callable or its public methods."""
    found: list[str] = []
    candidates = [(where, attribute)]
    if inspect.isclass(attribute):
        candidates += [
            (f"{where}.{name}", member)
            for name, member in vars(attribute).items()
            if inspect.isfunction(member)
        ]
    for label, member in candidates:
        try:
            signature = inspect.signature(member)
        except (TypeError, ValueError):  # pragma: no cover - not introspectable
            continue
        found += [f"{label}({name})" for name in signature.parameters if name == "origin"]
    return found


def test_a_miss_reports_what_it_could_not_place(mammals_path: Path) -> None:
    """Coverage is the maintenance signal, and a miss is when it matters.

    A resolution that placed nothing and *also* reported nothing unaccounted
    for is the quiet failure this field exists to make loud: the phrases users
    ask about and the vocabulary does not carry are the next entries somebody
    should add, and they are only visible here.

    **Spans, with the text as a reader over them.** Both halves are asserted
    at every point below, because the pair is the claim: a position nobody can
    turn back into a phrase is no more use than a phrase nobody can locate.
    """
    ontology = load_ontology(mammals_path)
    resolver = CascadingResolver([ExactNormalizedSignal(ontology.entities)], ontology.entities)

    hit = resolver.resolve("beagle", k=5)
    assert hit.coverage.matched == ((0, 6),)
    assert hit.matched_text() == ("beagle",)
    assert hit.coverage.unmatched == ()
    assert hit.unmatched_text() == ()

    miss = resolver.resolve("wombat", k=5)
    assert miss.coverage.matched == ()
    assert miss.matched_text() == ()
    assert miss.coverage.unmatched == ((0, 6),)
    assert miss.unmatched_text() == ("wombat",)


def test_a_whole_string_match_reports_the_extent_the_fold_kept(
    mammals_path: Path,
) -> None:
    """``(0, len(query))`` is not the honest span when the fold strips.

    The rung compared the whole query, so it has one span to report -- but the
    fold that made the match strips, so two of the characters in the query
    took no part in it. Reporting them as evidence is the same class of claim
    as reporting a corpus compatible because nobody looked.

    The slice is also what ``matched_text`` carries now, in the caller's own
    casing rather than the index's: the field is what the span points at, so
    the two agree by construction instead of by trust.
    """
    ontology = load_ontology(mammals_path)
    resolver = CascadingResolver([ExactNormalizedSignal(ontology.entities)], ontology.entities)

    result = resolver.resolve("  Beagle  ", k=5)

    assert [c.entity_id for c in result.candidates] == ["beagle"]
    assert result.coverage.matched == ((2, 8),)
    assert result.matched_text() == ("Beagle",)
    assert result.candidates[0].evidence[0].span == (2, 8)
    assert result.candidates[0].evidence[0].matched_text == "Beagle"
    assert result.coverage.unmatched == ()


class TwoAxisSource(MappingEntitySource):
    """A source that answers for its own entities, on more than one axis.

    A real implementation of
    :class:`~dataknobs_common.entity_resolution.MembershipOracle` rather than a
    stub, because the question the test below asks is *what does a source with
    a second axis make possible*, and a source that answers it by existing is
    also the cheapest proof that the seam is reachable from outside.

    The second axis is read off ``metadata``, which is where a hand-edited
    vocabulary would put it.
    """

    def memberships(self, entity: Entity) -> dict[str, str]:
        placed = {ENTITY_TYPE_KEY: entity.type}
        habitat = entity.metadata.get("habitat")
        if habitat is not None:
            placed["habitat"] = str(habitat)
        return placed

    def axes(self) -> frozenset[str]:
        return frozenset({ENTITY_TYPE_KEY, "habitat"})


def two_axis_resolver() -> CascadingResolver:
    """One entity on two axes, and a cascade over it."""
    source = TwoAxisSource(
        {"beagle": Entity(id="beagle", type="Breed", name="Beagle", metadata={"habitat": "forest"})}
    )
    return CascadingResolver([ExactNormalizedSignal(source)], source)


def test_a_two_axis_scope_can_still_admit() -> None:
    """A POSITIVE result under a TWO-AXIS scope -- the shape the suite lacked.

    Every other scoped assertion here is either one-axis or negative, and that
    combination cannot see the defect this guards. The three copies of the
    membership projection agreed on ``taxonomy_id`` **by construction**, since
    each read ``Entity.type``; a divergence between them could therefore only
    appear on a *second* axis, where the only assertion was ``== []``. A
    negative assertion is satisfied by returning less, so the rung-side
    narrowing could silently start dropping everything and nothing here would
    have failed.

    This is the one shape that fails on that divergence: it needs both axes to
    survive **both** filters -- the rung's narrowing and the cascade's ruling
    -- so a projection that loses an axis at either site turns this ``==
    ["beagle"]`` into ``== []``.
    """
    resolver = two_axis_resolver()

    def placed(within: Any) -> list[str]:
        return [c.entity_id for c in resolver.resolve("beagle", k=5, within=within).candidates]

    # The positive the suite did not have: two axes, both satisfied.
    assert placed({ENTITY_TYPE_KEY: "Breed", "habitat": "forest"}) == ["beagle"]

    # And the negatives that make it a two-axis claim rather than a one-axis
    # one that happens to carry a second key: either axis alone refuses it.
    assert placed({ENTITY_TYPE_KEY: "Breed", "habitat": "tundra"}) == []
    assert placed({ENTITY_TYPE_KEY: "Species", "habitat": "forest"}) == []


def test_the_rung_may_over_admit_but_never_under_admit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The superset-filter discipline, in the direction that cannot be recovered.

    ``_admits`` filters what a rung *produced*, so it can only remove. A rung
    that admits too much is overruled; a rung that admits too little never
    hands the candidate over and nothing downstream recovers it. Both halves
    are asserted, because the claim is an asymmetry and stating only the
    forgiving half is how it got written down wrong the first time.
    """
    resolver = two_axis_resolver()
    scope = {ENTITY_TYPE_KEY: "Breed", "habitat": "forest"}

    def placed() -> list[str]:
        return [c.entity_id for c in resolver.resolve("beagle", k=5, within=scope).candidates]

    assert placed() == ["beagle"], "the unpatched control: this scope admits"

    # Over-admitting: the rung ignores the scope entirely. The cascade rules.
    monkeypatch.setattr(signals_module, "_admitted", lambda entities, hits, filter: hits)
    assert placed() == ["beagle"]

    # Under-admitting: the rung drops what the scope admits. Unrecoverable.
    monkeypatch.setattr(signals_module, "_admitted", lambda entities, hits, filter: frozenset())
    assert placed() == []


def test_the_membership_oracle_has_no_twin_on_purpose() -> None:
    """One protocol for both flavours, declared rather than left to be noticed.

    Every other member of this family that reaches for data is twinned, so a
    single-flavour protocol is the kind of asymmetry the parity list above
    exists to make somebody state out loud. The reason is ``describe()``'s: an
    oracle reads an entity the **caller already holds** and touches no backing,
    so an awaitable form would cost every caller an ``await`` and buy nothing.
    The asynchronous cascade consults this same synchronous member.

    Asserted rather than written in a docstring alone, because the obvious
    maintenance move -- adding ``AsyncMembershipOracle`` for symmetry -- would
    otherwise land without anyone revisiting why there is only one.
    """
    assert not inspect.iscoroutinefunction(MembershipOracle.memberships)
    assert not hasattr(entity_resolution, "AsyncMembershipOracle")


class AsyncTwoAxisSource(AsyncMappingEntitySource):
    """:class:`TwoAxisSource` over an asynchronous source.

    ``memberships`` and ``axes`` stay synchronous on this twin as well: the
    oracle reads an entity the caller already holds, so there is nothing to
    await and the asynchronous cascade consults the same member.
    """

    def memberships(self, entity: Entity) -> dict[str, str]:
        placed = {ENTITY_TYPE_KEY: entity.type}
        habitat = entity.metadata.get("habitat")
        if habitat is not None:
            placed["habitat"] = str(habitat)
        return placed

    def axes(self) -> frozenset[str]:
        return frozenset({ENTITY_TYPE_KEY, "habitat"})


def async_two_axis_resolver() -> AsyncCascadingResolver:
    """:func:`two_axis_resolver` over an asynchronous source."""
    source = AsyncTwoAxisSource(
        {"beagle": Entity(id="beagle", type="Breed", name="Beagle", metadata={"habitat": "forest"})}
    )
    return AsyncCascadingResolver([AsyncExactNormalizedSignal(source)], source)


def test_a_scope_naming_an_axis_the_source_does_not_publish_is_refused(
    mammals_path: Path,
) -> None:
    """A mis-keyed axis is refused rather than answered with nothing.

    ``within_admits`` reads an axis a candidate declares nothing on as
    *excluded*, which is the right reading for an entity and the wrong one for
    a typo: ``{"taxonomy": "Breed"}`` matched no candidate and returned ``()``,
    which is exactly what a correct scope over an empty vocabulary returns. The
    caller cannot tell those apart, and the one they will assume is the one
    that is not their fault.

    The refusal names both halves -- what was asked for and what this source
    publishes -- because a caller who mistyped an axis needs the spelling, not
    the fact that they were wrong.
    """
    onto = load_ontology(mammals_path)
    resolver = CascadingResolver([ExactNormalizedSignal(onto.entities)], onto.entities)

    with pytest.raises(ValidationError) as raised:
        resolver.resolve("beagle", within={"taxonomy": "Breed"})

    message = str(raised.value)
    assert "taxonomy" in message
    assert ENTITY_TYPE_KEY in message


def test_the_bare_scope_forms_are_never_refused(mammals_path: Path) -> None:
    """The sugar names the one axis every source publishes, so it always passes.

    Without this the refusal above is satisfied by refusing everything, and the
    two spellings a caller most often writes are the ones that would break.
    """
    onto = load_ontology(mammals_path)
    resolver = CascadingResolver([ExactNormalizedSignal(onto.entities)], onto.entities)

    assert [c.entity_id for c in resolver.resolve("beagle", within="Breed").candidates] == [
        "beagle"
    ]
    assert [
        c.entity_id for c in resolver.resolve("beagle", within=["Breed", "Species"]).candidates
    ] == ["beagle"]
    assert [
        c.entity_id
        for c in resolver.resolve("beagle", within={ENTITY_TYPE_KEY: "Breed"}).candidates
    ] == ["beagle"]


def test_a_source_that_publishes_an_axis_makes_it_askable() -> None:
    """What the legal set is *for*: an oracle widens it, and only it can.

    The same scope refused above is answered here, by a source that says it
    knows the axis. That is the whole seam -- the legal set is the source's
    answer rather than a constant -- and it is why the refusal cannot simply
    be a check against ``taxonomy_id``.
    """
    resolver = two_axis_resolver()

    admitted = resolver.resolve("beagle", within={ENTITY_TYPE_KEY: "Breed", "habitat": "forest"})
    assert [c.entity_id for c in admitted.candidates] == ["beagle"]

    with pytest.raises(ValidationError):
        resolver.resolve("beagle", within={"habitatt": "forest"})


def test_resolve_many_refuses_the_same_scope(mammals_path: Path) -> None:
    """The bulk form is a boundary too, and it was the one with no assertion."""
    onto = load_ontology(mammals_path)
    resolver = CascadingResolver([ExactNormalizedSignal(onto.entities)], onto.entities)

    with pytest.raises(ValidationError):
        resolver.resolve_many(["beagle", "dog"], within={"taxonomy": "Breed"})


def test_the_async_flavour_refuses_and_admits_the_same_scopes() -> None:
    """The asynchronous half of the scope path, which had no test at all.

    ``AsyncCascadingResolver.resolve_many``, ``_async_admitted`` and the
    ``await self._entities.get_many(...)`` branch were never executed by this
    suite: the async resolver appeared only in a list of types and in the
    bridge test, which resolves unscoped. ``assert_twin_types_agree`` compares
    signatures, so nothing here was checking that the twin *behaves* the same
    -- and the scope ruling is the most delicate thing both flavours share.
    """

    async def exercise() -> None:
        resolver = async_two_axis_resolver()

        admitted = await resolver.resolve(
            "beagle", within={ENTITY_TYPE_KEY: "Breed", "habitat": "forest"}
        )
        assert [c.entity_id for c in admitted.candidates] == ["beagle"]

        outside = await resolver.resolve("beagle", within={"habitat": "desert"})
        assert [c.entity_id for c in outside.candidates] == []

        batch = await resolver.resolve_many(["beagle", "wombat"], within={"habitat": "forest"})
        assert [[c.entity_id for c in result.candidates] for result in batch] == [["beagle"], []]

        with pytest.raises(ValidationError):
            await resolver.resolve("beagle", within={"habitatt": "forest"})

        with pytest.raises(ValidationError):
            await resolver.resolve_many(["beagle"], within={"habitatt": "forest"})

    asyncio.run(exercise())


class DecayRung:
    """A rung whose candidate score is not its evidence's score.

    The case :attr:`EntityCandidate.score`'s own docstring is written about --
    *"a subclass one phase on carries a score no rung produced, a fused-or-
    native number multiplied by a hop decay"* -- and therefore the one shape
    that can tell whether the cascade carries a candidate through or rebuilds
    it from the evidence.
    """

    @property
    def name(self) -> str:
        return "decay"

    def narrows(self) -> bool:
        return False

    def candidates(
        self, query: str, k: int, *, filter: dict[str, Any] | None = None
    ) -> list[EntityCandidate]:
        return [
            EntityCandidate(
                entity_id="beagle",
                score=0.42,
                evidence=(
                    MatchEvidence(
                        signal="decay",
                        kind=EvidenceKind.INFERRED,
                        score=0.83,
                        scoring=Scoring.DECAYED,
                        matched_text="beagle",
                    ),
                ),
            )
        ]

    def candidates_many(
        self, queries: Sequence[str], k: int, *, filter: dict[str, Any] | None = None
    ) -> list[list[EntityCandidate]]:
        return [self.candidates(query, k, filter=filter) for query in queries]


def test_the_cascade_returns_the_score_the_rung_gave(mammals_path: Path) -> None:
    """A candidate's own score survives the cascade, rather than its evidence's.

    ``finish`` rebuilt every candidate as ``score=evidence[0].score``, which is
    precisely the move :attr:`EntityCandidate.score` exists to refuse -- one
    layer up, where the dataclass cannot defend itself. A decay rung's 0.42
    came back as 0.83, so ``ranked()`` and ``as_distribution()`` both read the
    undecayed number and ``Scoring.DECAYED`` could never describe a candidate
    the cascade returned.
    """
    onto = load_ontology(mammals_path)
    resolver = CascadingResolver([DecayRung()], onto.entities)

    candidate = resolver.resolve("beagle").candidates[0]

    assert candidate.score == 0.42
    assert candidate.evidence[0].score == 0.83
    assert candidate.evidence[0].scoring is Scoring.DECAYED


def test_a_candidate_subclass_survives_the_cascade(mammals_path: Path) -> None:
    """And the type comes through, which rebuilding also lost.

    ``EntityCandidate`` is documented as a base a later phase subclasses. A
    cascade that reconstructs returns the base and drops whatever the subclass
    added, so a phase downstream of the cascade could not receive its own type
    back from it.
    """

    @dataclasses.dataclass(frozen=True)
    class FusedCandidate(EntityCandidate):
        fused_from: tuple[str, ...] = ()

    class FusingRung(DecayRung):
        @property
        def name(self) -> str:
            return "fusing"

        def candidates(
            self, query: str, k: int, *, filter: dict[str, Any] | None = None
        ) -> list[EntityCandidate]:
            return [FusedCandidate(entity_id="beagle", score=0.5, fused_from=("a", "b"))]

    onto = load_ontology(mammals_path)
    resolver = CascadingResolver([FusingRung()], onto.entities)

    candidate = resolver.resolve("beagle").candidates[0]

    assert isinstance(candidate, FusedCandidate)
    assert candidate.fused_from == ("a", "b")


def test_a_candidate_that_located_nothing_leaves_the_query_unmatched(
    mammals_path: Path,
) -> None:
    """Coverage is positional, and the reading follows from that.

    ``unmatched`` was *"what reached no candidate"*, all-or-nothing, so a
    resolution that produced anything at all reported the whole query placed.
    It is *"what no evidence located"* now, and the two differ exactly where a
    candidate carries no position -- a bare candidate from a fusing or
    decaying rung, a cosine neighbour over an embedded utterance.

    **That reads correctly rather than being the cost of the change.** No
    declared form was found in this text and a match is being offered anyway,
    which is precisely the line a consumer maintaining a vocabulary acts on;
    the older rule could not state it, because a vector-only hit and an exact
    one produced the same empty report.

    The positive control is the neighbouring test: the same query through a
    rung that *does* locate its match comes back with nothing unmatched, so
    this is a property of the evidence rather than of the query.
    """

    class BareRung(DecayRung):
        @property
        def name(self) -> str:
            return "bare"

        def candidates(
            self, query: str, k: int, *, filter: dict[str, Any] | None = None
        ) -> list[EntityCandidate]:
            return [EntityCandidate(entity_id="beagle", score=1.0)]

    onto = load_ontology(mammals_path)
    result = CascadingResolver([BareRung()], onto.entities).resolve("beagle")

    assert [c.entity_id for c in result.candidates] == ["beagle"]
    assert result.candidates[0].evidence[0].span is None
    assert result.coverage.matched == ()
    assert result.coverage.unmatched == ((0, 6),)
    assert result.unmatched_text() == ("beagle",)

    located = CascadingResolver([ExactNormalizedSignal(onto.entities)], onto.entities)
    assert located.resolve("beagle").coverage.unmatched == ()


class AliaslessSource:
    """A source that conforms to :class:`EntitySource` and reports no aliases.

    A real implementation delegating to the shipped
    :class:`MappingEntitySource` for every member it has, and simply not
    having the optional one -- which is what an out-of-tree source written
    against the protocol looks like. Hiding a member from a subclass would not
    do: the question is what happens to a class that never had it.
    """

    def __init__(self, entities: dict[str, Entity]) -> None:
        self._inner = MappingEntitySource(entities)

    def get(self, entity_id: str) -> Entity | None:
        return self._inner.get(entity_id)

    def get_many(self, entity_ids: Sequence[str]) -> dict[str, Entity]:
        return self._inner.get_many(entity_ids)

    def fetch_origin(self, ref: Any) -> Any:
        return self._inner.fetch_origin(ref)

    def fetch_origins(self, refs: Sequence[Any]) -> Any:
        return self._inner.fetch_origins(refs)

    def describe(self) -> Any:
        return self._inner.describe()

    def by_surface_form(self, form: str) -> frozenset[str]:
        return self._inner.by_surface_form(form)

    def by_type(self, type_id: str) -> frozenset[str]:
        return self._inner.by_type(type_id)

    def longest_form_tokens(self) -> int:
        return self._inner.longest_form_tokens()


def test_a_source_without_alias_forms_still_conforms_and_still_resolves() -> None:
    """Reporting alias forms is optional, and a source lacking it is not broken.

    Both halves matter and they fail differently. Conformance is what the
    member being on a *separate* protocol buys -- as a member of
    ``EntitySource`` this class would have stopped conforming the day it was
    added, without changing. And the rung has to degrade rather than raise:
    ``AliasSignal`` is in the default composition every unconfigured document
    gets, so a source that cannot answer it would take down a cascade that
    never asked for aliases at all.
    """
    from dataknobs_common.ontology import AliasFormSource, EntitySource

    source = AliaslessSource({"beagle": Entity(id="beagle", type="Breed", name="Beagle")})

    assert isinstance(source, EntitySource)
    assert not isinstance(source, AliasFormSource)

    resolver = CascadingResolver([ExactNormalizedSignal(source), AliasSignal(source)], source)
    result = resolver.resolve("beagle")

    assert [c.entity_id for c in result.candidates] == ["beagle"]
    assert [e.signal for e in result.explain("beagle")] == ["exact"]
    assert AliasSignal(source).candidates("beagle", 5) == []


def test_an_alias_source_of_the_wrong_flavour_is_refused_rather_than_awaited():
    """The other way a capability check can be satisfied without being right.

    ``AliasSignal`` asks ``isinstance(..., AliasFormSource)`` and falls back to
    *this vocabulary declares no aliases* when the answer is no. A
    runtime-checkable protocol compares member **names**, and both flavours
    spell it ``by_alias_form`` -- so the answer was *yes* for an asynchronous
    source too, and the rung went on to call it synchronously. What a cascade
    then saw was a ``'coroutine' object is not iterable`` from inside a rung
    that had reported itself satisfied, plus a *coroutine was never awaited*
    warning from somewhere else entirely.

    The flavour is now settled at construction, which is where the source was
    chosen. The **absent** case is untouched and is asserted beside this one:
    those two are different facts, and only one of them is a mistake.
    """
    from dataknobs_common.entity_resolution.signals import AsyncAliasSignal
    from dataknobs_common.exceptions import ValidationError
    from dataknobs_common.ontology import AsyncMappingEntitySource

    vocabulary = {"beagle": Entity(id="beagle", type="Breed", name="Beagle", aliases=("Beagles",))}

    with pytest.raises(ValidationError, match="synchronous"):
        AliasSignal(AsyncMappingEntitySource(vocabulary))
    with pytest.raises(ValidationError, match="asynchronous"):
        AsyncAliasSignal(MappingEntitySource(vocabulary))

    assert AliasSignal(AliaslessSource(vocabulary)).candidates("beagles", 5) == [], (
        "a source that simply lacks the member still declares no aliases, "
        "which is a fact rather than a misconfiguration"
    )
