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

from dataknobs_common import entity_resolution
from dataknobs_common.entity_resolution import (
    AliasSignal,
    AsyncAliasSignal,
    AsyncCascadingResolver,
    AsyncDeclaredSignal,
    AsyncEntityResolver,
    AsyncExactNormalizedSignal,
    AsyncMatchSignal,
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
    TAXONOMY_ID_KEY,
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
    assert_twin_types_agree,
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


@pytest.mark.parametrize(
    ("sync_type", "async_type", "members"),
    [
        (EntityResolver, AsyncEntityResolver, ["resolve", "resolve_many"]),
        (MatchSignal, AsyncMatchSignal, ["candidates", "candidates_many"]),
        (CascadingResolver, AsyncCascadingResolver, ["resolve", "resolve_many"]),
        (
            ExactNormalizedSignal,
            AsyncExactNormalizedSignal,
            ["candidates", "candidates_many"],
        ),
        (AliasSignal, AsyncAliasSignal, ["candidates", "candidates_many"]),
        (
            DeclaredSignal,
            AsyncDeclaredSignal,
            ["candidates", "candidates_many"],
        ),
    ],
)
def test_the_twins_expose_one_surface(
    sync_type: type, async_type: type, members: list[str]
) -> None:
    """Parity over **annotations**, not only names and defaults.

    The weaker form passes a twin whose ``within`` still reads
    ``str | Collection[str] | None`` while the other half has widened -- which
    is precisely the drift the alias was published to prevent, arriving
    through the test written to catch it.
    """
    assert_twin_types_agree(sync_type, async_type, members, compare_return=True)


def test_the_bridge_is_not_one_of_the_twins() -> None:
    """It satisfies the synchronous protocol by forwarding, not by twinning.

    Stated as a test because the parity list above is hand-written, and the
    obvious maintenance move on it is to add every class that has both
    flavours' members. This one does not have a twin to compare against: the
    thing it would be twinned with is the resolver it holds.
    """
    assert hasattr(BridgedEntityResolver, "__enter__")
    assert hasattr(BridgedEntityResolver, "__exit__")
    assert not hasattr(BridgedEntityResolver, "__aenter__")


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
    assert placed("beagle", {TAXONOMY_ID_KEY: ["Breed", "Species"]}) == ["beagle"]
    assert placed("dog", {TAXONOMY_ID_KEY: ["Breed", "Species"]}) == ["dog"]
    assert placed("beagle", {TAXONOMY_ID_KEY: "Breed"}) == ["beagle"]
    assert placed("beagle", {TAXONOMY_ID_KEY: "Species"}) == []

    # Conjunction across axes: admitted only where both hold.
    assert placed("beagle", {TAXONOMY_ID_KEY: "Breed", "habitat": "forest"}) == ["beagle"]
    assert placed("beagle", {TAXONOMY_ID_KEY: "Species", "habitat": "forest"}) == []
    assert placed("beagle", {TAXONOMY_ID_KEY: "Breed", "habitat": "desert"}) == []

    # An entity silent on a named axis is excluded, rather than passing every
    # filter on it. ``dog`` declares no habitat, so a scope naming one leaves
    # it out even though its type admits.
    assert placed("dog", {TAXONOMY_ID_KEY: "Species", "habitat": "forest"}) == []


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

    assert offered["narrowing"] == {TAXONOMY_ID_KEY: ["Breed"]}
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
    """
    ontology = load_ontology(mammals_path)
    resolver = CascadingResolver([ExactNormalizedSignal(ontology.entities)], ontology.entities)

    hit = resolver.resolve("beagle", k=5)
    assert hit.coverage.matched == ("beagle",)
    assert hit.coverage.unmatched == ()

    miss = resolver.resolve("wombat", k=5)
    assert miss.coverage.matched == ()
    assert miss.coverage.unmatched == ("wombat",)


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
        placed = {TAXONOMY_ID_KEY: entity.type}
        habitat = entity.metadata.get("habitat")
        if habitat is not None:
            placed["habitat"] = str(habitat)
        return placed

    def axes(self) -> frozenset[str]:
        return frozenset({TAXONOMY_ID_KEY, "habitat"})


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
    assert placed({TAXONOMY_ID_KEY: "Breed", "habitat": "forest"}) == ["beagle"]

    # And the negatives that make it a two-axis claim rather than a one-axis
    # one that happens to carry a second key: either axis alone refuses it.
    assert placed({TAXONOMY_ID_KEY: "Breed", "habitat": "tundra"}) == []
    assert placed({TAXONOMY_ID_KEY: "Species", "habitat": "forest"}) == []


def test_the_rung_may_over_admit_but_never_under_admit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The superset-filter discipline, in the direction that cannot be recovered.

    ``_admits`` filters what a rung *produced*, so it can only remove. A rung
    that admits too much is overruled; a rung that admits too little never
    hands the candidate over and nothing downstream recovers it. Both halves
    are asserted, because the claim is an asymmetry and stating only the
    forgiving half is how it got written down wrong the first time.
    """
    resolver = two_axis_resolver()
    scope = {TAXONOMY_ID_KEY: "Breed", "habitat": "forest"}

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
        placed = {TAXONOMY_ID_KEY: entity.type}
        habitat = entity.metadata.get("habitat")
        if habitat is not None:
            placed["habitat"] = str(habitat)
        return placed

    def axes(self) -> frozenset[str]:
        return frozenset({TAXONOMY_ID_KEY, "habitat"})


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
    assert TAXONOMY_ID_KEY in message


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
        for c in resolver.resolve("beagle", within={TAXONOMY_ID_KEY: "Breed"}).candidates
    ] == ["beagle"]


def test_a_source_that_publishes_an_axis_makes_it_askable() -> None:
    """What the legal set is *for*: an oracle widens it, and only it can.

    The same scope refused above is answered here, by a source that says it
    knows the axis. That is the whole seam -- the legal set is the source's
    answer rather than a constant -- and it is why the refusal cannot simply
    be a check against ``taxonomy_id``.
    """
    resolver = two_axis_resolver()

    admitted = resolver.resolve("beagle", within={TAXONOMY_ID_KEY: "Breed", "habitat": "forest"})
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
            "beagle", within={TAXONOMY_ID_KEY: "Breed", "habitat": "forest"}
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


def test_a_query_that_placed_something_is_not_reported_unmatched(mammals_path: Path) -> None:
    """``unmatched`` keys off the candidates, not off a projection of them.

    It was derived from ``matched_text``, which ``_stamp`` leaves empty for a
    candidate that carried no evidence of its own -- the case ``merge_rung``'s
    ``kind`` parameter exists to serve. So a rung returning a bare candidate
    produced a result that reported the entity **and** reported the query as
    placeable nowhere, and the guide teaches ``coverage.unmatched`` as the
    signal for what to add to a vocabulary next.
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
    assert result.coverage.unmatched == ()


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
