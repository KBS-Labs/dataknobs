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
    AsyncEntityResolver,
    AsyncExactNormalizedSignal,
    AsyncMatchSignal,
    BridgedEntityResolver,
    CascadeState,
    CascadingResolver,
    EntityCandidate,
    EntityResolver,
    EvidenceKind,
    ExactNormalizedSignal,
    MatchEvidence,
    MatchSignal,
    TAXONOMY_ID_KEY,
    Scoring,
    cascade as cascade_module,
)
from dataknobs_common.ontology import (
    AsyncMappingEntitySource,
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


def test_within_unions_within_an_axis_and_conjoins_across_them(mammals_path: Path) -> None:
    """Both halves of the rule, against the axis this source actually declares.

    An earlier version of this test used two invented axis names, ``a`` and
    ``b``, and passed -- because the rung-side implementation it ran against
    intersected ``by_type`` per key and never looked at the key. Under the
    published predicate the axis name is load-bearing: what an entity declares
    nothing on cannot admit it. So a scope must name an axis the source has,
    and ``describe().declares`` publishes exactly one.
    """
    ontology = load_ontology(mammals_path)
    resolver = CascadingResolver([ExactNormalizedSignal(ontology.entities)], ontology.entities)

    def placed(within: Any) -> list[str]:
        return [c.entity_id for c in resolver.resolve("beagle", k=5, within=within).candidates]

    # Union within one axis: either id admits.
    assert placed({TAXONOMY_ID_KEY: ["Breed", "Species"]}) == ["beagle"]
    assert placed({TAXONOMY_ID_KEY: "Breed"}) == ["beagle"]

    # Conjunction across axes: a second axis this source declares nothing on
    # excludes, rather than being ignored. The forgiving reading -- a candidate
    # silent on an axis passing every filter on it -- is the one refused.
    assert placed({TAXONOMY_ID_KEY: "Breed", "habitat": "forest"}) == []
    assert placed({TAXONOMY_ID_KEY: "Species"}) == []


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
