# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The roll-up: what a set of rows is about, and what it refuses to guess.

The operation only exists at three rows or more --- over one row it is an
accessor --- so every test here works from the same five-row corpus, and the
five are deliberately unlike each other. One names two nodes, one touched no
vocabulary at all, one names a node this vocabulary does not carry, and two are
ordinary. A roll-up whose worked case contains neither of the middle two is a
roll-up whose residue was never designed.

**The corpus is the caller's and the vocabulary is the vocabulary's**, which is
the split most of this file is about. A name off a row is not a name anybody
typed, so it is filtered; a name the caller passed is one somebody typed, so it
is refused.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.exceptions import NotFoundError, ValidationError
from dataknobs_common.hierarchy import deepest_common_ancestor
from dataknobs_common.ontology import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    AsyncMappingAssertionSource,
    AsyncMappingEntitySource,
    AsyncOntology,
    Granularity,
    MappingAssertionSource,
    MappingEntitySource,
    NodeSupport,
    NodeTag,
    Ontology,
    OntologySupport,
    StrCodec,
    SupportSet,
    TaxonomyDefinition,
    async_roll_up,
    load_ontology,
    ontology_support,
    read_node_tags,
    roll_up,
)
from dataknobs_common.ontology import ascent
from dataknobs_common.testing import assert_twins_agree

from _vocabularies import Sku, SkuCodec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dataknobs_common.ontology.taxonomy import Taxonomy


def _row(*node_ids: str, ontology: str = "mammals", axis: str = "species") -> dict[str, Any]:
    """One row's metadata, as a consumer's own retrieval would hand it back."""
    return {
        ONTOLOGY_ID_KEY: ontology,
        TAXONOMY_ID_KEY: axis,
        NODE_ID_KEY: list(node_ids),
    }


#: The five rows every criterion here is measured over --- the corpus the design
#: section publishes, in the order it publishes them. Row 1 touched no
#: vocabulary; row 4 names ``wolfhound``, which this vocabulary does not carry.
CORPUS = [
    _row("golden_retriever", "beagle"),
    {"invoice_id": "2291"},
    _row("dog"),
    _row("golden_retriever"),
    _row("wolfhound"),
]


@dataclass
class _Edges:
    """A structure axis over an explicit parent map, recording what it was asked.

    Hand-built rather than loaded, because the properties below are about
    vocabularies a document cannot conveniently express: two axes over one node
    set, a cycle, and a key space that is not ``str``. It is a real
    ``Hierarchy`` --- the four members, answered from a mapping --- so what runs
    over it is the walk rather than a stand-in for one, and ``asked`` is the
    structure's own record of the question that has no cache.
    """

    up: Mapping[Any, tuple[Any, ...]]
    asked: list[Any] = field(default_factory=list)

    def roots(self) -> Sequence[Any]:
        return tuple(node for node, above in self.up.items() if not above)

    def parents(self, node_id: Any) -> Sequence[Any]:
        return self.up.get(node_id, ())

    def children(self, node_id: Any) -> Sequence[Any]:
        return tuple(node for node, above in self.up.items() if node_id in above)

    def contains(self, node_id: Any) -> bool:
        self.asked.append(node_id)
        return node_id in self.up


@dataclass
class _AsyncEdges:
    """:class:`_Edges` with its four members awaited --- what a live backing is.

    Async on every member rather than only the ones a test happens to reach: a
    synchronous stand-in for an asynchronous backing is how a missing ``await``
    survives a suite, because calling a sync function without one works.
    """

    up: Mapping[Any, tuple[Any, ...]]
    asked: list[Any] = field(default_factory=list)

    async def roots(self) -> Sequence[Any]:
        return tuple(node for node, above in self.up.items() if not above)

    async def parents(self, node_id: Any) -> Sequence[Any]:
        return self.up.get(node_id, ())

    async def children(self, node_id: Any) -> Sequence[Any]:
        return tuple(node for node, above in self.up.items() if node_id in above)

    async def contains(self, node_id: Any) -> bool:
        self.asked.append(node_id)
        return node_id in self.up


def _built(
    axes: Mapping[str, _Edges],
    *,
    ontology_id: str = "mammals",
    codec: Any = None,
) -> Ontology[Any]:
    """A vocabulary whose axes are the given structures, and which loads nothing.

    ``structures`` is the field a door fills and a caller building directly may
    fill with any ``Hierarchy``, so this is the documented seam rather than a
    way around one. The two sources are empty because nothing here reads them:
    the roll-up walks the structure and counts, and asks what a node *is*
    nowhere.
    """
    return Ontology(
        id=ontology_id,
        version="1.0",
        entity_types={},
        relation_types={},
        entities=MappingEntitySource({}),
        assertions=MappingAssertionSource([]),
        taxonomies={name: TaxonomyDefinition(id=name, relation=name) for name in axes},
        describes=(),
        codec=StrCodec() if codec is None else codec,
        structures=dict(axes),
    )


def _built_async(
    axes: Mapping[str, _AsyncEdges], *, ontology_id: str = "mammals"
) -> AsyncOntology[Any]:
    """:func:`_built`'s twin, over asynchronous backings."""
    return AsyncOntology(
        id=ontology_id,
        version="1.0",
        entity_types={},
        relation_types={},
        entities=AsyncMappingEntitySource({}),
        assertions=AsyncMappingAssertionSource([]),
        taxonomies={name: TaxonomyDefinition(id=name, relation=name) for name in axes},
        describes=(),
        codec=StrCodec(),
        structures=dict(axes),
    )


@pytest.fixture
def onto(mammals_v11_path: Path) -> Ontology[str]:
    """The vocabulary the whole file rolls up for."""
    return load_ontology(mammals_v11_path)


@pytest.fixture
def tagged() -> list[tuple[NodeTag, ...]]:
    """:data:`CORPUS` read into tags --- one entry per position, in order."""
    return [read_node_tags(row) for row in CORPUS]


@pytest.fixture
def answer(onto: Ontology[str], tagged: list[tuple[NodeTag, ...]]) -> SupportSet[str]:
    """The roll-up under test."""
    return roll_up(onto, "species", tagged)


def test_the_rows_are_rolled_up_in_the_order_they_named(answer: SupportSet[str]) -> None:
    """The record: every node the rows named, with the rows that named it.

    **First-seen order, and the order is the assertion.** ``supported`` is the
    record rather than the presentation --- what to put in front of a person is
    ``prune()``'s answer --- so this asserts the order the rows arrived in and
    not a ranking. ``golden_retriever`` is first because row 0 named it first,
    not because it has the most rows; the two happen to agree here, which is why
    ``beagle`` and ``dog`` are the pair that carries the claim: equal support,
    and ``beagle`` first because row 0 came before row 2.

    The rows are positions in the sequence the caller passed, ascending and
    without duplicates, which is what makes ``[corpus[i] for i in support.rows]``
    the evidence rather than an approximation of it.
    """
    assert [(s.node_id, s.rows) for s in answer.supported] == [
        ("golden_retriever", (0, 3)),
        ("beagle", (0,)),
        ("dog", (2,)),
    ]
    assert [s.above for s in answer.supported] == [("dog",), ("dog",), ()]
    assert answer.ontology_id == "mammals"
    assert answer.taxonomy_id == "species"


def test_the_residue_is_reported_whole(answer: SupportSet[str]) -> None:
    """A node the axis does not carry is reported, and the call does not raise.

    Three claims in one test because either half alone reads as a gap and the
    two together are the residue.

    **Reported rather than refused**: a corpus tagged before a vocabulary was
    reorganised is the ordinary case, and the person who could fix it is on the
    other side of a boundary this package does not cross. So ``wolfhound``
    arrives in ``unplaced`` carrying the row that named it, and the maintainer
    who has to find the tagger has the row rather than just the id.

    **And its row is unsupported too.** Row 4 contributed to no entry in
    ``supported``, so it is in ``unsupported_rows`` alongside row 1, which
    carried no tag at all. The two causes are deliberately not distinguished
    here --- the caller holds the tags and is one lookup from telling them
    apart --- and the field exists because an answer drawn from five rows of
    which two touched nothing is exactly the case a consumer must be able to
    see.
    """
    assert [(s.node_id, s.rows, s.above) for s in answer.unplaced] == [("wolfhound", (4,), ())]
    assert "wolfhound" not in {s.node_id for s in answer.supported}
    assert answer.unsupported_rows == (1, 4)


def test_the_projection_is_one_vocabulary(answer: SupportSet[str]) -> None:
    """Three members of one enum, and the fourth's refusal, in one test.

    An enum is a vocabulary, and a test per member asserts nothing about the
    relation between them. What matters is that the three answers are different
    projections of one record: ``dog`` is gone from ``MOST_SPECIFIC`` because two
    nodes below it were named, it is the whole of ``MOST_GENERAL`` because
    nothing in the set stands above it, and ``ALL`` is the identity. In every
    case it is still in ``supported``, so row 2 stays reachable --- pruning
    changes what is presented, never what is reachable.

    **The fourth is asserted here rather than beside it**, so that the domain and
    its edge are one behaviour. ``AT_TYPE`` is a published member whose only
    behaviour is a refusal, and the refusal names what it would need: reading
    what a node *is* is a source read, and this value holds no source.
    """
    assert [s.node_id for s in answer.prune()] == ["golden_retriever", "beagle"]
    assert [s.node_id for s in answer.prune(Granularity.MOST_GENERAL)] == ["dog"]
    assert [s.node_id for s in answer.prune(Granularity.ALL)] == [
        "golden_retriever",
        "beagle",
        "dog",
    ]
    assert "dog" in {s.node_id for s in answer.supported}

    with pytest.raises(ValidationError) as refusal:
        answer.prune(Granularity.AT_TYPE)
    assert "entity source" in str(refusal.value)
    assert "mammals" in str(refusal.value)


def test_the_projection_cannot_walk() -> None:
    """``prune`` performs no I/O, asserted on the shape that makes it true.

    The guarantee is **structural**: ``MOST_SPECIFIC`` and ``MOST_GENERAL`` are
    computed from ``NodeSupport.above`` alone and ``ALL`` from nothing, and
    ``above`` is carried rather than derived precisely so that deriving it --- a
    hierarchy walk --- happens once, in the roll-up, and never in a member on a
    value.

    **So what is guarded is the field list, not a call.** Neither value holds a
    reference to an ontology, an axis or a structure, so there is nowhere to
    install a raising implementation and watch it not be reached: a test written
    that way would pass for the reason it was trying to falsify. What can fail is
    the edit this criterion is actually about --- a field added later that the
    walk could arrive through --- and that is what this asserts.

    **The dotted spelling counts**, which is the difference between a guard and
    the appearance of one. An annotation is a string under
    ``from __future__ import annotations``, so a field written
    ``dataknobs_common.ontology.taxonomy.Taxonomy[K]`` is exactly as reachable as
    one written ``Taxonomy[K]`` and reads as neither under a split that does not
    separate on the dot. The extraction is a named function here and is asserted
    against both spellings, so the test that catches a later edit is itself
    caught when it stops being able to.

    The answer over a hand-built set is asserted with it, so the pair is *the
    projection is right* and *the projection cannot become I/O*.
    """
    reachable = {"Ontology", "Taxonomy", "TaxonomyView", "Hierarchy", "HierarchyView"}

    def _named(annotation: object) -> set[str]:
        """Every bare name an annotation mentions, however it is qualified."""
        spelled = str(annotation)
        for separator in ("[", "]", ",", "|", "."):
            spelled = spelled.replace(separator, " ")
        return {part for part in spelled.split() if part}

    assert _named("dataknobs_common.ontology.taxonomy.Taxonomy[K]") & reachable, (
        "a dotted annotation no longer reads as naming a vocabulary object, so the "
        "assertion below would pass for a field this test exists to catch"
    )
    assert _named("Taxonomy[K]") & reachable
    assert not (_named("tuple[int, ...]") & reachable)

    for value in (SupportSet, NodeSupport):
        for declared in fields(value):
            named = _named(declared.type)
            assert not (named & reachable), (
                f"{value.__name__}.{declared.name} is annotated {declared.type!r}, which names a "
                f"vocabulary object. prune() is a member on a value and its no-I/O guarantee "
                f"is that no field can reach one -- a field that can is the edit this test "
                f"exists to catch"
            )

    built = SupportSet(
        ontology_id="mammals",
        taxonomy_id="species",
        supported=(
            NodeSupport(node_id="golden_retriever", rows=(0, 3), above=("dog",)),
            NodeSupport(node_id="beagle", rows=(0,), above=("dog",)),
            NodeSupport(node_id="dog", rows=(2,), above=()),
        ),
    )
    assert [s.node_id for s in built.prune()] == ["golden_retriever", "beagle"]
    assert [s.node_id for s in built.prune(Granularity.MOST_GENERAL)] == ["dog"]


def test_the_ranking_is_by_evidence_not_by_name(answer: SupportSet[str]) -> None:
    """Row count descending, ties first-seen --- over a set where that differs from id order.

    The requirement the corpus was chosen for: alphabetically ``beagle``
    precedes ``golden_retriever``, and by row support ``golden_retriever``
    precedes ``beagle``. The two orders **differ**, so a ranking that had
    quietly become alphabetical would fail here. A carelessly chosen input
    would let both readings pass.

    The tie is carried by ``ALL``, where ``beagle`` and ``dog`` both hold one
    row and ``beagle`` is first because row 0 came before row 2.
    """
    ranked = [s.node_id for s in answer.prune(Granularity.ALL)]

    assert ranked == ["golden_retriever", "beagle", "dog"]
    assert sorted(ranked) != ranked, (
        "the corpus no longer distinguishes row-support order from id order, so this "
        "test can no longer fail for the reason it exists"
    )
    assert [len(s.rows) for s in answer.prune(Granularity.ALL)] == [2, 1, 1]


def test_a_cache_changes_cost_and_never_the_answer(
    onto: Ontology[str], tagged: list[tuple[NodeTag, ...]]
) -> None:
    """A supplied cache gives an equal answer and is non-empty afterwards.

    One is used within a call whether or not one is given, because the several
    ascents share a frontier, and supplying one is how those replies outlive the
    call. What may be done with them afterwards is the neighbouring test's
    subject: a cache carries no hierarchy in its key, so spending one across two
    axes is safe here only because this call scopes what it forwards.

    A bare ``dict`` satisfies the cache protocol, which is what makes this
    assertable without building anything: the answer is compared for equality and
    the dict is asserted non-empty, so a forwarding that had been dropped shows
    up as an empty cache rather than as a wrong answer.
    """
    cache: dict[Any, Any] = {}

    with_cache = roll_up(onto, "species", tagged, cache=cache)
    without = roll_up(onto, "species", tagged)

    assert with_cache == without
    assert cache, "the cache was not reached, so the parameter is being dropped"


def test_the_filter_runs_before_the_door(
    onto: Ontology[str], tagged: list[tuple[NodeTag, ...]]
) -> None:
    """A tag naming another vocabulary contributes nothing, and ``localize`` is never asked.

    The order is the subject. ``localize`` refuses an id another vocabulary
    qualified, and that refusal is the right answer for a caller who handed it
    one directly --- but a name that came off a *row* is not a name anybody
    typed. So the filter runs first and the door is never reached, which is what
    makes a drifted corpus an ordinary answer instead of an exception.

    Asserted with a real subclass that records its calls rather than with a mock:
    an ontology is a frozen dataclass compared by identity, so a subclass
    overriding one member is an ordinary object that still answers everything
    else.
    """
    calls: list[str] = []

    class Recording(Ontology[str]):
        def localize(self, qualified_id: str) -> str:
            calls.append(qualified_id)
            return super().localize(qualified_id)

    recording = Recording(**{f.name: getattr(onto, f.name) for f in fields(onto)})
    foreign = [*tagged, read_node_tags(_row("spay", ontology="procedures"))]

    answer = roll_up(recording, "species", foreign)

    assert calls, "localize was never called at all, so this asserts nothing about order"
    assert not any(call.startswith("procedures:") for call in calls), (
        "a tag naming another vocabulary reached localize: the filter must run first, or "
        "a drifted corpus becomes a refusal the caller cannot act on"
    )
    assert "spay" not in {s.node_id for s in (*answer.supported, *answer.unplaced)}
    assert 5 in answer.unsupported_rows


def test_the_caller_s_axis_is_refused_where_a_tag_s_is_filtered(
    onto: Ontology[str], tagged: list[tuple[NodeTag, ...]]
) -> None:
    """The asymmetry the signature encodes, asserted in both directions at once.

    ``taxonomy_id`` is the caller's value, so a name this vocabulary does not
    declare is a name somebody typed and the accessor refuses it, naming the
    axes that do exist. A *tag*'s axis id is a third party's, so a row naming an
    axis this call was not asked about lands in ``unsupported_rows`` and raises
    nothing --- indistinguishably from a row naming an axis the vocabulary has
    never heard of, because the filter compares against the caller's value and
    asks the vocabulary nothing.
    """
    with pytest.raises(NotFoundError) as refusal:
        roll_up(onto, "coat", tagged)
    assert "species" in str(refusal.value)

    drifted = [*tagged, read_node_tags(_row("double_coat", axis="coat"))]
    assert roll_up(onto, "species", drifted).unsupported_rows == (1, 4, 5)


def test_the_module_reaches_no_optional_dependency() -> None:
    """Nothing outside this package and the standard library, written as a test.

    The roll-up is in ``common`` because a count over tags and a walk over a
    vocabulary's own fields have no third-party dependency --- which is what
    makes every question this module answers askable with no store, no embedder
    and no event loop. An import arriving here by a later edit would make that
    false and nothing else would report it.

    Read out of the module's source rather than off ``sys.modules``: what is
    asserted is what the file says, and a transitively imported module is
    somebody else's business.
    """
    reached: set[str] = set()
    for node in ast.walk(ast.parse(Path(ascent.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            reached.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            reached.update(alias.name.split(".")[0] for alias in node.names)

    assert reached
    outside = {
        name
        for name in reached
        if name != "dataknobs_common" and name not in sys.stdlib_module_names
    }
    assert not outside, f"{sorted(outside)} is neither this package nor the standard library"


def test_the_walk_one_word_away_returns_the_opposite_answer(
    onto: Ontology[str], answer: SupportSet[str]
) -> None:
    """``deepest_common_ancestor`` over the entities ``prune`` keeps returns ``dog``.

    The two operations are one word apart and go in opposite directions, and
    this is the assertion that keeps the docstring's disclaimer honest. That walk
    is pairwise and **generalises**: over the two entities the projection keeps it
    returns ``dog`` --- this vocabulary's answer to ``MOST_GENERAL`` --- where the
    projection returns ``golden_retriever`` and ``beagle``, which are what the
    rows actually said. Two opposite members of one enum, over one corpus.

    **The sharp part is the evidence, and it is asserted rather than described.**
    A caller who pruned holds entries for ``golden_retriever`` and ``beagle``,
    whose rows are 0 and 3. ``dog`` is not one of those entries, and neither of
    those rows named it --- so the answer that walk hands back is one the caller
    has no evidence for, in an operation whose whole contract is each entry
    carrying the rows that are its evidence. That ``dog`` is in ``supported`` on
    the strength of a *different* row is what makes the point precise: the
    evidence exists and is not the evidence for what was asked.

    Folded over all three supported entities it returns ``dog`` as well, so the
    disagreement is not an artefact of taking two.
    """
    axis: Taxonomy[str] = onto.taxonomy("species")
    kept = answer.prune()

    assert [s.node_id for s in kept] == ["golden_retriever", "beagle"]
    assert deepest_common_ancestor(axis.structure, kept[0].node_id, kept[1].node_id) == "dog"

    assert "dog" not in {s.node_id for s in kept}
    evidence = sorted({position for entry in kept for position in entry.rows})
    assert evidence == [0, 3]
    assert "dog" not in {
        tag.node_id for position in evidence for tag in read_node_tags(CORPUS[position])
    }

    folded = answer.supported[0].node_id
    for entry in answer.supported[1:]:
        folded = deepest_common_ancestor(axis.structure, folded, entry.node_id)
    assert folded == "dog"
    assert [s.node_id for s in answer.prune(Granularity.MOST_GENERAL)] == [folded]


def test_the_vocabularies_in_play_are_ranked_by_the_same_measure(
    tagged: list[tuple[NodeTag, ...]],
) -> None:
    """The same corpus grouped by vocabulary rather than by node.

    One measure at two heights: ``roll_up`` counts rows per node within one
    vocabulary, this counts rows per vocabulary, and both rank by row count
    descending with ties in first-seen order. A caller holding a page of results
    and no idea what is on it asks this one first.

    **It loads nothing.** The tags carry their own ``ontology_id``, so this
    answers over vocabularies nobody holds --- which is why the tie-break is
    exercised here over three, one of which no ontology in this test exists for.

    The rows are every position that named the vocabulary, including the one
    whose node the vocabulary does not carry: the row *named* ``mammals``, and
    whether ``mammals`` carries the node is the next question rather than this
    one.
    """
    assert ontology_support(tagged) == (OntologySupport(ontology_id="mammals", rows=(0, 2, 3, 4)),)

    mixed = [
        read_node_tags(_row("beagle")),
        read_node_tags(_row("spay", ontology="procedures")),
        read_node_tags(_row("dog")),
        read_node_tags(_row("invoice", ontology="billing")),
        read_node_tags(_row("suture", ontology="procedures")),
        {},
    ]
    assert [(s.ontology_id, s.rows) for s in ontology_support(mixed)] == [
        ("mammals", (0, 2)),
        ("procedures", (1, 4)),
        ("billing", (3,)),
    ]
    assert ontology_support([]) == ()


def test_the_vocabulary_level_entry_travels_on_its_own(
    tagged: list[tuple[NodeTag, ...]],
) -> None:
    """``OntologySupport`` is constructible, hashable, positional, and on the door.

    **Reachability is what this asserts**, and the distinction is worth keeping:
    the type is published by this module and nothing outside the package
    produces one today. It is on the door so that a registry narrowing this
    answer to the vocabularies it holds could be a filter over this module's
    answer rather than a second count --- and being importable is the part of
    that which is true now and testable now.

    Positional for the reason the node-level rows are: the row is recoverable
    from its position and the position is never recoverable from the row.
    Ascending and without duplicates, so a row naming one vocabulary twice
    contributes one entry.
    """
    from dataknobs_common import ontology as door

    assert "OntologySupport" in door.__all__
    assert door.OntologySupport is OntologySupport

    entry = OntologySupport(ontology_id="mammals", rows=(0, 2, 3, 4))
    assert hash(entry) == hash(OntologySupport(ontology_id="mammals", rows=(0, 2, 3, 4)))
    assert entry == ontology_support(tagged)[0]

    twice = [read_node_tags(_row("beagle", "dog"))]
    assert ontology_support(twice) == (OntologySupport(ontology_id="mammals", rows=(0,)),)

    for support in ontology_support(tagged):
        assert list(support.rows) == sorted(set(support.rows))


def test_a_cache_is_scoped_to_the_axis_it_was_filled_from() -> None:
    """One cache handed to two axes answers the second from the first's edges.

    **The hazard is silent, which is why it is asserted rather than documented.**
    A walk cache's key is ``(member, node)`` and deliberately carries no
    hierarchy --- a backing is arbitrary consumer code that need not be hashable
    --- so a cache filled over one axis and read over another returns the first
    axis's parents with no exception anywhere. What comes back is a *wrong
    answer*, not a failure: ``beagle`` stands under ``dog`` on one axis and under
    ``small`` on the other, and the second roll-up reading the first's replies
    finds ``dog``, which is not in the second answer's own node set, and reports
    ``beagle`` as standing under nothing.

    ``roll_up`` is the one member of this family that takes the axis **by name**
    rather than taking the hierarchy, so it is also the one that can scope the
    cache rather than asking the caller to. A caller holding one vocabulary and
    two axis names holds one ontology, and reusing the cache across them is the
    natural call shape rather than an exotic one.

    The equality against a fresh cache is the whole assertion: a scoping that
    stopped working shows up as a different answer, which is exactly the symptom
    a consumer would otherwise have to diagnose from a UI.
    """
    onto = _built(
        {
            "species": _Edges({"beagle": ("dog",), "dog": ()}),
            "size": _Edges({"beagle": ("small",), "small": ()}),
        }
    )
    species_rows = [read_node_tags(_row("beagle")), read_node_tags(_row("dog"))]
    size_rows = [
        read_node_tags(_row("beagle", axis="size")),
        read_node_tags(_row("small", axis="size")),
    ]

    shared: dict[Any, Any] = {}
    first = roll_up(onto, "species", species_rows, cache=shared)
    second = roll_up(onto, "size", size_rows, cache=shared)

    assert second == roll_up(onto, "size", size_rows), (
        "the second axis was answered from the first axis's cached edges: a cache "
        "carries no hierarchy in its key, so roll_up must scope what it forwards"
    )
    assert [s.above for s in second.supported] == [("small",), ()]
    assert [s.above for s in first.supported] == [("dog",), ()]
    assert shared, "the cache was not reached, so the parameter is being dropped"


def test_an_unreadable_local_id_is_residue_rather_than_a_refusal() -> None:
    """A node id this vocabulary's key space cannot parse costs its row, not the answer.

    **Only reachable off ``str``.** ``localize`` hands the local segment to the
    vocabulary's own codec and lets the codec's refusal through unwrapped, which
    is that door's documented contract. Over ``StrCodec`` nothing can fail, so a
    suite written entirely over string keys cannot see this at all --- and the
    operation is generic in the key precisely so that a consumer binds their own.

    The failure it replaces is the disproportionate one: a page of rows, one of
    them tagged before the key space changed, and the *whole* roll-up raises ---
    no supported set, no unplaced, no residue --- over one row in a corpus the
    caller did not write and cannot fix.

    So it is the same operational class as ``unplaced``, disposed of the way
    ``unsupported_rows`` disposes of the other three: the row contributed to no
    entry, and the caller holds the tags, so which of the four causes applies is
    one lookup away rather than a field.
    """
    line = Sku(plant="ashland", line=3)
    plant = Sku(plant="ashland", line=0)
    onto = _built(
        {"lines": _Edges({line: (plant,), plant: ()})},
        ontology_id="acme",
        codec=SkuCodec(),
    )

    def _row_of(*ids: str) -> dict[str, Any]:
        return {
            ONTOLOGY_ID_KEY: "acme",
            TAXONOMY_ID_KEY: "lines",
            NODE_ID_KEY: list(ids),
        }

    drifted = [
        read_node_tags(_row_of("ashland/3")),
        read_node_tags(_row_of("dog")),
        read_node_tags(_row_of("ashland/0")),
    ]

    answer = roll_up(onto, "lines", drifted)

    assert [(s.node_id, s.rows) for s in answer.supported] == [(line, (0,)), (plant, (2,))]
    assert answer.unsupported_rows == (1,)
    assert answer.unplaced == ()


def test_the_axis_is_asked_once_per_node_not_once_per_tag() -> None:
    """``contains`` is a question about a node, and the corpus has fewer nodes than tags.

    **The cache does not cover this one.** ``ancestors`` takes a cache and the
    *n* ascents share a frontier, but ``exists()`` takes none --- so a
    membership test written inside the per-tag loop is one backing round trip
    per tag occurrence, not per node. Over a page of results that is the whole
    page's worth of identical queries to answer one question.

    The protocol invites exactly the backing that makes it expensive: a
    hierarchy over rows answers ``contains`` with a query, and the bulk variants
    exist because per-node querying is the cost the singular members force.

    **A second reading comes free and is asserted with it.** Asking once per
    node also makes the partition a function of the axis at one instant: asked
    per occurrence, a backing that changes mid-call can put one node in both
    ``supported`` and ``unplaced``, which is two answers to one question.
    """
    axis = _Edges({"dog": (), "beagle": ("dog",)})
    onto = _built({"species": axis})
    page = [read_node_tags(_row("dog")) for _ in range(5)]
    page.append(read_node_tags(_row("beagle", "dog")))

    answer = roll_up(onto, "species", page)

    assert sorted(axis.asked) == ["beagle", "dog"], (
        f"the axis was asked {len(axis.asked)} times about 2 nodes: membership is a "
        f"question about a node, so it belongs outside the per-tag loop"
    )
    assert [(s.node_id, s.rows) for s in answer.supported] == [
        ("dog", (0, 1, 2, 3, 4, 5)),
        ("beagle", (5,)),
    ]


def test_a_row_naming_one_node_twice_contributes_one_row() -> None:
    """Ascending and without duplicates, over a row that names the same node twice.

    The tag reader does not deduplicate --- two entries under the node-id key
    are two tags --- so the guarantee is this operation's to keep, and it is
    kept at both heights. The vocabulary level already asserts it; this is the
    node level, which had the same guard and no test over it.
    """
    onto = _built({"species": _Edges({"dog": ()})})

    answer = roll_up(onto, "species", [read_node_tags(_row("dog", "dog"))])

    assert [(s.node_id, s.rows) for s in answer.supported] == [("dog", (0,))]
    assert answer.unsupported_rows == ()


def test_a_cycle_is_presented_rather_than_emptied() -> None:
    """Two nodes each above the other are equally specific, so neither excludes the other.

    **A cyclic document loads**, which is what makes this reachable rather than
    hypothetical: nothing refuses ``A isa B`` alongside ``B isa A``, and the
    walks over such an axis terminate. So a consumer can hold a support set in
    which ``a`` stands above ``b`` and ``b`` above ``a``, both with real rows
    behind them.

    Read literally, every projection then empties: ``MOST_SPECIFIC`` drops both
    because each is an ancestor of something in the set, and ``MOST_GENERAL``
    drops both because neither has an empty ``above``. A page with evidence for
    two entities presents as a page about nothing --- the one outcome that is
    certainly wrong, because ``supported`` plainly holds them.

    Mutual ancestry is not a specificity relation. It is the statement that the
    two are at the same height, so the projection keeps both and the ranking ---
    by rows, which always exists --- decides the order. ``above`` itself is
    unchanged: it is the record of what the axis says, and the record is that
    each stands above the other.
    """
    onto = _built({"species": _Edges({"a": ("b",), "b": ("a",)})})
    rows = [read_node_tags(_row(node)) for node in ("a", "a", "a", "b", "b")]

    answer = roll_up(onto, "species", rows)

    assert [(s.node_id, s.rows, s.above) for s in answer.supported] == [
        ("a", (0, 1, 2), ("b",)),
        ("b", (3, 4), ("a",)),
    ]
    assert [s.node_id for s in answer.prune()] == ["a", "b"]
    assert [s.node_id for s in answer.prune(Granularity.MOST_GENERAL)] == ["a", "b"]
    assert [s.node_id for s in answer.prune(Granularity.ALL)] == ["a", "b"]


async def test_the_two_flavours_answer_the_same_roll_up() -> None:
    """The asynchronous twin, over the backing that is the reason it exists.

    **The flavour a live vocabulary has.** An axis over rows is asynchronous ---
    that is what a by-reference backing needs --- so a consumer whose vocabulary
    came from a registry holds an ``AsyncOntology`` and cannot call the
    synchronous roll-up at all. Without the twin, the operation is unreachable
    from the flavour built for the backing it was designed around.

    Asserted as an *equality between the two answers over one corpus* rather
    than as a second expectation, so the twin cannot drift into being right
    about something else: the structures carry the same edges, the tags are the
    same tags, and the two support sets must be the same value.
    """
    edges = {"beagle": ("dog",), "golden_retriever": ("dog",), "dog": ()}
    rows = [
        read_node_tags(_row("golden_retriever", "beagle")),
        {},
        read_node_tags(_row("dog")),
        read_node_tags(_row("golden_retriever")),
        read_node_tags(_row("wolfhound")),
    ]

    synchronous = roll_up(_built({"species": _Edges(edges)}), "species", rows)
    asynchronous = await async_roll_up(
        _built_async({"species": _AsyncEdges(edges)}), "species", rows
    )

    assert asynchronous == synchronous
    assert [s.node_id for s in asynchronous.prune()] == ["golden_retriever", "beagle"]
    assert asynchronous.unplaced == (NodeSupport(node_id="wolfhound", rows=(4,)),)
    assert asynchronous.unsupported_rows == (1, 4)


def test_the_two_flavours_expose_one_surface() -> None:
    """The parity guard every other twin in this family carries.

    A twin is not two functions that happen to agree on one corpus; it is one
    surface with one flavoured member. ``max_concurrency`` is the asynchronous
    half's alone --- a frontier fetched concurrently is the thing the flavour
    adds --- and ``ontology`` is flavoured by definition, since the whole point
    is which kind of vocabulary it takes. Everything else must match, and an
    argument added to one half and not the other fails here rather than being
    discovered by the consumer who needed it.
    """
    assert_twins_agree(
        roll_up,
        async_roll_up,
        async_only=("max_concurrency",),
        flavour_typed=("ontology",),
        compare_return=False,
        label="roll_up/async_roll_up",
    )
