# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The vector rung, alone and inside the cascade that ranks it.

The store is real and in memory, and the embedder is a real ``TextEmbedder``
with no model behind it --- so a ranking asserted here is reproducible in
another process rather than a property of a fixture.

**What that embedder is and is not.** ``DeterministicEmbedder`` seeds from a
full-text digest and draws from a symmetric distribution, which is what makes
a *ranking* assertable at all; it is not a model, so *near* here means
*deterministically near* rather than *semantically near*. Every ordering this
file asserts was therefore **measured first** and is asserted beside the
measurement that makes it non-vacuous --- an assertion that the embedding
prefers the wrong entity is worth nothing if the embedding happened to prefer
the right one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.entity_resolution import (
    AsyncCascadingResolver,
    AsyncExactNormalizedSignal,
    EvidenceKind,
    Scoring,
)
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.index import MappingSource
from dataknobs_common.ontology import EntitySourceIndexSource, OntologyConfig
from dataknobs_common.records import Record

from dataknobs_data.entity_resolution import SemanticSignal
from dataknobs_data.ontology import OntologyRegistry
from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector import SemanticIndex
from dataknobs_data.vector.stores.memory import MemoryVectorStore
from dataknobs_data.vector.types import VectorSearchResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from dataknobs_common.ontology import AsyncOntology

#: Thirty-two, rather than the sixteen the index suite uses: a ranking over
#: five entities needs the directions to separate, and a ranking is what this
#: file asserts.
DIMENSIONS = 32

#: Five entities, of which the declared rungs reach **one** for the queries
#: below. That gap is the fixture: what these tests are about is a candidate
#: **no declared rung produced**, so a vocabulary whose every entity is
#: reachable by name cannot express it.
MAMMALS: dict[str, Any] = {
    "id": "mammals",
    "version": "1.0",
    "entity_types": [{"id": "Species"}, {"id": "Breed"}],
    "entities": [
        {
            "id": "beagle",
            "type": "Breed",
            "name": "Beagle",
            "aliases": ["Beagles"],
            "description": "a small scent hound bred for rabbit hunting",
        },
        {
            "id": "dog",
            "type": "Species",
            "name": "Dog",
            "aliases": ["Canine"],
            "description": "a domesticated descendant of the wolf",
        },
        {
            "id": "cat",
            "type": "Species",
            "name": "Cat",
            "description": "a small carnivore kept for companionship",
        },
        {
            "id": "ferret",
            "type": "Species",
            "name": "Ferret",
            "description": "a domesticated polecat with a long body",
        },
        {
            "id": "hamster",
            "type": "Species",
            "name": "Hamster",
            "description": "a burrowing rodent with cheek pouches",
        },
    ],
}

#: The document as a deployment writes it: a store to put the vectors in, and
#: the composition the cascade is assembled from.
INDEXED_AND_RESOLVED: dict[str, Any] = {
    **MAMMALS,
    "index": {"store": {"backend": "memory", "dimensions": DIMENSIONS}},
    "resolver": {"rungs": [{"kind": "exact"}, {"kind": "lexical"}, {"kind": "semantic"}]},
}


@pytest.fixture
async def registry() -> AsyncIterator[OntologyRegistry]:
    """A loaded registry whose index has been **built**, which is the caller's line.

    ``load()`` assembles an index and returns; filling it is a separate call,
    for the reason :meth:`OntologyRegistry.index` gives --- so a fixture that
    skipped it would be testing the empty-index path in every test that used
    it. That path has a test of its own below.
    """
    live = OntologyRegistry.from_components(
        config=OntologyConfig(**INDEXED_AND_RESOLVED),
        embedder=DeterministicEmbedder(dimensions=DIMENSIONS),
    )
    await live.load()
    index = live.index("mammals")
    assert index is not None
    await index.build()
    yield live
    await live.close()


async def _standalone(
    ontology: AsyncOntology[str], *, build: bool = True
) -> tuple[SemanticIndex, MemoryVectorStore]:
    """An index over a vocabulary, outside any registry."""
    store = MemoryVectorStore({"dimensions": DIMENSIONS})
    await store.initialize()
    index = SemanticIndex(
        EntitySourceIndexSource(ontology), DeterministicEmbedder(dimensions=DIMENSIONS), store
    )
    if build:
        await index.build()
    return index, store


# --------------------------------------------------------------------------
# What a candidate from this rung is
# --------------------------------------------------------------------------


async def test_a_candidate_carries_no_span_and_is_inferred(
    registry: OntologyRegistry,
) -> None:
    """A cosine neighbour has no position in the query, and says so.

    Two assertions and no cascade, because this is a property of the rung
    rather than of anything ranking it. ``span is None`` is what distinguishes
    this rung's evidence from every located one: a near-spelling rung is
    *also* ``INFERRED`` and does carry a span, so the kind alone would not
    have said it.

    Nothing raises and nothing invents a span --- which is the half worth
    asserting explicitly, because the shape this rung declined to inherit
    (``FormHit``) makes a span **required**, and a rung squeezed into it would
    have had to make one up.
    """
    index = registry.index("mammals")
    assert index is not None
    rung = SemanticSignal(index, registry.get("mammals"))

    [candidate, *_] = await rung.candidates("a burrowing rodent", k=3)

    [evidence] = candidate.evidence
    assert evidence.span is None
    assert evidence.kind is EvidenceKind.INFERRED
    assert evidence.scoring is Scoring.NATIVE
    assert evidence.signal == "semantic"
    assert evidence.matched_text == "a burrowing rodent", (
        "the whole query proposed it, so the whole query is what matched"
    )


async def test_one_entity_is_one_candidate_because_the_rung_answers_in_the_cascades_space(
    registry: OntologyRegistry,
) -> None:
    """The rung localizes on the way out, so two rungs finding one entity agree.

    **The failure this prevents is invisible from the result's shape.** Every
    row the index holds is a *qualified* id and the cascade's entity source
    speaks local ones, so a rung answering the store's ids puts two spaces in
    one candidate list --- where nothing de-duplicates between them, every
    assertion about labels and positions still passes, and one entity comes
    back twice under two keys.

    Asserted as the conjunction, because either half alone is satisfiable by
    the broken version: the list holds **one** entry for the entity, and
    ``explain`` on it names **both** rungs.
    """
    resolver = registry.resolver("mammals")
    assert resolver is not None

    result = await resolver.resolve("Beagle", k=5)

    ids = [candidate.entity_id for candidate in result.candidates]
    assert ids.count("beagle") == 1
    assert "mammals:beagle" not in ids, "a qualified id here is the two-space failure"
    assert sorted(evidence.signal for evidence in result.explain("beagle")) == [
        "exact",
        "lexical",
        "semantic",
    ]


# --------------------------------------------------------------------------
# The cascade, reached the way a consumer reaches it
# --------------------------------------------------------------------------


async def test_a_semantic_only_candidate_is_last_undeclared_and_unreached_by_any_declared_rung(
    registry: OntologyRegistry,
) -> None:
    """The three clauses together, and the third is what makes it a criterion.

    ``declared`` is ``any(kind is DECLARED ...)`` over a candidate's evidence,
    and ``merge_rung`` folds a later rung's evidence onto an id an earlier one
    already produced --- so a candidate a declared rung *also* reached answers
    ``True`` however far down the list it sits. Only an entity **no** declared
    rung reached at all can carry the clause.

    **And a fourth, implied by the cascade rather than by the criterion:**
    ``CascadeState.saturated`` stops the loop once ``k`` ids are held, so a
    test whose declared rungs fill ``k`` never asks this rung anything. ``k``
    leaves room here, and the assertion below says so rather than leaving it
    to be inferred from a passing run.
    """
    resolver = registry.resolver("mammals")
    assert resolver is not None

    result = await resolver.resolve("Beagle", k=5)

    declared_reached = {
        candidate.entity_id for candidate in result.candidates if candidate.declared
    }
    assert declared_reached == {"beagle"}, "the declared rungs must not fill k, or nothing asks"

    last = result.candidates[-1]
    assert last.declared is False
    assert last.entity_id not in declared_reached
    assert [evidence.signal for evidence in last.evidence] == ["semantic"]


async def test_a_near_spelling_outranks_the_neighbour_the_embedding_prefers(
    registry: OntologyRegistry,
) -> None:
    """The order two rungs put the entities in, not either rung's output alone.

    A misspelling is where the two disagree usefully: a near-spelling rung
    measures the typo against a declared form and is right, and an embedding
    of the typo is a different text from an embedding of the form and lands
    wherever it lands.

    **The measurement that makes the assertion non-vacuous is asserted too.**
    Over this vocabulary and this embedder, the semantic rung alone ranks
    ``dog`` above ``beagle`` for ``"Beaagle"``; if it ever ranked ``beagle``
    first, the interesting half of this test would pass without the near
    spelling having done anything, so that is checked rather than assumed.

    No cosine value is pinned. What is asserted is a *relative* order between
    two rungs, which is what the signal being multi- rather than cosine-only
    is for.
    """
    index = registry.index("mammals")
    assert index is not None
    mammals = registry.get("mammals")
    typo = "Beaagle"

    embedding_only = AsyncCascadingResolver([SemanticSignal(index, mammals)], mammals.entities)
    unaided = [
        candidate.entity_id for candidate in (await embedding_only.resolve(typo, k=5)).candidates
    ]
    assert unaided.index("dog") < unaided.index("beagle"), (
        "the embedding already prefers the right entity, so this fixture asserts nothing"
    )

    resolver = registry.resolver("mammals")
    assert resolver is not None
    aided = await resolver.resolve(typo, k=5)

    assert aided.candidates[0].entity_id == "beagle"
    assert "lexical" in [evidence.signal for evidence in aided.explain("beagle")]
    ranked = [candidate.entity_id for candidate in aided.candidates]
    assert ranked.index("beagle") < ranked.index("dog")


async def test_a_scoped_resolve_keeps_this_rung_and_the_rung_is_offered_no_filter(
    registry: OntologyRegistry,
) -> None:
    """Declining a filter is what keeps a scoped resolve from emptying.

    A scope is rendered as ``{entity_type: [...]}`` and a store's metadata
    filter fails a row that carries no such key --- and no row this index
    holds carries one. A rung that forwarded it would answer nothing under
    every scope, which reads downstream as *not in the corpus*.

    **Three assertions, because two of them are satisfiable by the wrong
    fix.** A rung that declines the filter *and* answers in the store's id
    space has every candidate set aside as ``beyond_authority`` --- three
    entities that are in the vocabulary, inside the scope and correctly
    retrieved, announced as ids the scope could not be applied to. That is a
    false report where the unfixed behaviour was merely an empty one, so the
    coverage is asserted beside the candidates.
    """
    index = registry.index("mammals")
    assert index is not None
    mammals = registry.get("mammals")
    rung = SemanticSignal(index, mammals)
    seen: list[dict[str, Any] | None] = []

    original = rung.candidates

    async def recording(query: str, k: int, **kwargs: Any) -> Any:
        seen.append(kwargs.get("filter"))
        return await original(query, k, **kwargs)

    rung.candidates = recording  # type: ignore[method-assign]
    cascade = AsyncCascadingResolver(
        [AsyncExactNormalizedSignal(mammals.entities), rung], mammals.entities
    )

    scoped = await cascade.resolve("Beagle", k=5, within=["Breed"])

    assert [candidate.entity_id for candidate in scoped.candidates] == ["beagle"]
    assert sorted(evidence.signal for evidence in scoped.explain("beagle")) == [
        "exact",
        "semantic",
    ]
    assert scoped.coverage.beyond_authority == ()
    assert seen == [None], "narrows() is False, so the cascade offers this rung no filter"
    assert rung.narrows() is False


# --------------------------------------------------------------------------
# What the rung does that no criterion reads
# --------------------------------------------------------------------------


async def test_a_batch_embeds_every_query_in_one_ask_rather_than_one_each(
    registry: OntologyRegistry,
) -> None:
    """The batch path is real, and the embedder is where that is visible.

    ``candidates_many`` has a looping default, so a rung that never overrode
    it would answer correctly and ask its backing *n* times. Counted at the
    embedder rather than at ``search_batch``, because the round trip a batch
    saves is the embedding call --- asserting that a method was called would
    say nothing about whether the work was batched.
    """
    index = registry.index("mammals")
    assert index is not None
    asked: list[int] = []
    inner = index.embedder

    class _Counting:
        model_id = inner.model_id

        async def embed(self, texts: list[str]) -> Any:
            asked.append(len(texts))
            return await inner.embed(texts)

    index.embedder = _Counting()  # type: ignore[assignment]
    try:
        rung = SemanticSignal(index, registry.get("mammals"))
        found = await rung.candidates_many(["Beagle", "Ferret", "a burrowing rodent"], k=2)
    finally:
        index.embedder = inner

    assert [len(one) for one in found] == [2, 2, 2]
    assert asked == [3], "three queries reached the embedder as three separate asks"


async def test_an_index_with_no_rows_is_reported_once_rather_than_read_as_an_empty_vocabulary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The one state a caller cannot tell from *nothing was near enough*.

    ``load()`` assembles an index and does not fill it --- building is the
    caller's line --- so a cascade resolving before that call answers from its
    declared rungs alone and reports ``declared == True`` where the criterion
    one section up asserts ``False``. Nothing is wrong with any of those
    answers; what is wrong is that nobody said the corpus was empty.

    **The count is what separates the two cases and nothing else does.** An
    unthresholded search answers the *k* nearest rows whatever the query, so a
    rung returning nothing over a populated store is anomalous --- but a rung
    carrying a ``threshold`` may legitimately answer nothing over a full one.

    Asserted with the built case beside it, because a warning that fires
    either way is not a report.
    """
    live = OntologyRegistry.from_components(
        config=OntologyConfig(**INDEXED_AND_RESOLVED),
        embedder=DeterministicEmbedder(dimensions=DIMENSIONS),
    )
    await live.load()
    try:
        index = live.index("mammals")
        resolver = live.resolver("mammals")
        assert index is not None and resolver is not None
        assert await index.store.count() == 0

        with caplog.at_level(logging.WARNING, logger="dataknobs_data.entity_resolution"):
            first = await resolver.resolve("Beagle", k=5)
            await resolver.resolve("Ferret", k=5)

        assert [candidate.entity_id for candidate in first.candidates] == ["beagle"]
        assert first.candidates[-1].declared is True, (
            "the state this warning exists for: the criterion's own assertion inverted"
        )
        [reported] = [record for record in caplog.records if "holds no rows" in record.message]
        assert "mammals" in reported.getMessage()
        assert "build()" in reported.getMessage()

        caplog.clear()
        await index.build()
        with caplog.at_level(logging.WARNING, logger="dataknobs_data.entity_resolution"):
            await resolver.resolve("Beagle", k=5)
        assert [record for record in caplog.records if "holds no rows" in record.message] == []
    finally:
        await live.close()


async def test_an_index_whose_ids_are_not_this_vocabularys_is_refused_at_construction(
    registry: OntologyRegistry,
) -> None:
    """Named where the caller is still holding the mistake.

    The alternative is a rung that constructs cleanly and fails on its first
    hit, from inside the ontology's id parser --- several frames from anything
    naming a rung. The member read is the one written for this question: a
    source says which named sets its ids fall in, and a source over a bare
    table says ``frozenset()``, whose ids are local and belong to no
    vocabulary.

    Both shapes, because they fail for different reasons and a guard catching
    one would look like a guard catching both.
    """
    store = MemoryVectorStore({"dimensions": DIMENSIONS})
    await store.initialize()
    try:
        bare = SemanticIndex(
            MappingSource({"beagle": "Beagle"}),
            DeterministicEmbedder(dimensions=DIMENSIONS),
            store,
        )
        with pytest.raises(ValidationError, match="no named set at all"):
            SemanticSignal(bare, registry.get("mammals"))
    finally:
        await store.close()

    other = OntologyRegistry.from_components(
        config=OntologyConfig(**{**MAMMALS, "id": "elsewhere"}),
        embedder=DeterministicEmbedder(dimensions=DIMENSIONS),
    )
    await other.load()
    try:
        index = registry.index("mammals")
        assert index is not None
        with pytest.raises(ValidationError, match="mammals"):
            SemanticSignal(index, other.get("elsewhere"))
    finally:
        await other.close()


async def test_a_hit_with_no_record_id_is_refused_rather_than_placed(
    registry: OntologyRegistry,
) -> None:
    """The published read door types a record's id as optional, so this is reachable.

    Skipping the hit loses a candidate silently and coercing the absence
    invents an id, which are the two readings this family refuses everywhere
    else. Measured before the refusal existed, the failure was an
    ``AttributeError`` out of the id parser.
    """
    mammals = registry.get("mammals")
    store = MemoryVectorStore({"dimensions": DIMENSIONS})
    await store.initialize()

    class _Idless(MemoryVectorStore):
        async def search_similar_records(
            self, *args: Any, **kwargs: Any
        ) -> list[VectorSearchResult]:
            return [VectorSearchResult(record=Record({"name": "Beagle"}), score=0.9, metadata={})]

    idless = _Idless({"dimensions": DIMENSIONS})
    await idless.initialize()
    try:
        index = SemanticIndex(
            EntitySourceIndexSource(mammals),
            DeterministicEmbedder(dimensions=DIMENSIONS),
            idless,
        )
        rung = SemanticSignal(index, mammals)
        with pytest.raises(ValidationError, match="no record id"):
            await rung.candidates("Beagle", 5)
    finally:
        await idless.close()
        await store.close()


async def test_a_threshold_a_document_wrote_is_the_rungs_own(
    registry: OntologyRegistry,
) -> None:
    """Forwarded to the index's search, which is the only place it can act.

    The constructor is where a document writing ``kind: semantic`` supplies
    one --- the reason the near-spelling rung's factory forwards its own
    threshold one rung over --- and without it a cascade could only take what
    the index defaults to, which is *keep everything*.
    """
    index = registry.index("mammals")
    assert index is not None
    mammals = registry.get("mammals")

    everything = await SemanticSignal(index, mammals).candidates("Beagle", 5)
    assert any(candidate.score < 0.2 for candidate in everything), (
        "nothing here scores under the cut, so the thresholded call would pass vacuously"
    )

    cut = await SemanticSignal(index, mammals, threshold=0.2).candidates("Beagle", 5)
    assert cut, "a threshold this vocabulary can meet, or the assertion below is trivial"
    assert all(candidate.score >= 0.2 for candidate in cut)
    assert len(cut) < len(everything)


def test_the_two_registries_answer_the_same_facts_before_and_after_the_import() -> None:
    """The mark and the registration declare one thing, so both must say it.

    ``common``'s mark writes ``reads_surface_forms`` and
    ``bounded_by_longest_form`` by hand because the class is not importable
    there; ``data``'s registration derives them from the class. Two spellings
    of one fact drift, and the drift would be silent in the worst place: two
    load-time refusals read exactly these keys, so a mark that disagreed would
    make them answer differently over one document depending on what had been
    imported.

    Asserted as equality of the whole mapping rather than key by key, so a
    fact added to one side and not the other fails here rather than later.
    """
    from dataknobs_common.entity_resolution.registry import (
        async_signal_backends,
        signal_backends,
    )

    assert async_signal_backends.get_metadata("semantic") == {
        "flavour": "async",
        "needs_io": True,
        "requires_install": "pip install dataknobs-data",
        "reads_surface_forms": False,
        "bounded_by_longest_form": False,
    }
    assert signal_backends.unavailable_reason("semantic") == (
        "SemanticSignal has no synchronous form"
    )
    assert not signal_backends.is_registered("semantic"), (
        "there is no synchronous form of this rung to register, here or anywhere"
    )
