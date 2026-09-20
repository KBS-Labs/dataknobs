# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Which vocabularies a page of results is about, narrowed to the ones held.

The step between reading a corpus's tags and descending into one vocabulary.
The counting is ``dataknobs_common``'s and is tested there; what is tested here
is the half that needs a registry --- that the narrowing is a *filter* over that
answer rather than a second count, that it agrees with the roll-up one level
down, and that what it drops stays reachable.

**Three vocabularies, two of them tied deliberately, and three orderings held
apart.** A ranking asserted over a corpus whose row-count order and id order
agree asserts nothing: either implementation passes. So the tied pair is named
in the corpus in the reverse of its id order, which separates a first-seen
tie-break from an alphabetical one --- and the registry loads the pair in the
reverse of *that*, which separates it from the registry's own ``list_ids``
order. The third is the one a member of a *registry* is likeliest to reach for
by accident, and it is the one a fixture loading in corpus order cannot see.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.ontology import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    OntologyConfig,
    async_roll_up,
    ontology_support,
    read_node_tags_many,
)

from dataknobs_data.ontology import OntologyRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from dataknobs_common.ontology import NodeTag, SupportSet

#: The vocabulary with structure in it, and it declares **two** axes.
#:
#: The second is what makes the agreement below a containment rather than an
#: equality: a roll-up is scoped to one axis and a selection is scoped to a
#: vocabulary, so a corpus naming one vocabulary on two axes is one the two
#: honestly disagree about.
MAMMALS: dict[str, Any] = {
    "id": "mammals",
    "version": "1.0",
    "entity_types": [{"id": "Species"}, {"id": "Breed", "isa": "Species"}, {"id": "Habitat"}],
    "relation_types": [{"id": "isa", "transitive": True}, {"id": "lives_in", "transitive": True}],
    "entities": [
        {"id": "dog", "type": "Species", "name": "Dog"},
        {"id": "golden_retriever", "type": "Breed", "name": "Golden Retriever"},
        {"id": "beagle", "type": "Breed", "name": "Beagle"},
        {"id": "land", "type": "Habitat", "name": "Land"},
        {"id": "domestic", "type": "Habitat", "name": "Domestic"},
    ],
    "assertions": [
        {"subject": "golden_retriever", "relation": "isa", "object": "dog"},
        {"subject": "beagle", "relation": "isa", "object": "dog"},
        {"subject": "domestic", "relation": "lives_in", "object": "land"},
    ],
    "taxonomies": [
        {"id": "species", "name": "Species", "relation": "isa"},
        {"id": "habitat", "name": "Habitat", "relation": "lives_in"},
    ],
}

#: A second vocabulary, structureless. Nothing here descends into it --- it is
#: in the corpus to be counted and in the registry to be held.
PROCEDURES: dict[str, Any] = {
    "id": "procedures",
    "version": "1.0",
    "entity_types": [{"id": "Procedure"}],
    "entities": [{"id": "spay", "type": "Procedure", "name": "Spay"}],
}

#: The third, and the one the drop test leaves unloaded.
BILLING: dict[str, Any] = {
    "id": "billing",
    "version": "1.0",
    "entity_types": [{"id": "Code"}],
    "entities": [{"id": "inv", "type": "Code", "name": "Invoice"}],
}


def _row(*node_ids: str, ontology: str = "mammals", axis: str = "species") -> dict[str, Any]:
    """One row's metadata, as a consumer's own retrieval would hand it back."""
    return {
        ONTOLOGY_ID_KEY: ontology,
        TAXONOMY_ID_KEY: axis,
        NODE_ID_KEY: list(node_ids),
    }


#: Seven rows naming three vocabularies, with ``procedures`` and ``billing``
#: tied at two rows each.
#:
#: ``procedures`` is named first and sorts second, so *descending by row count*
#: and *ties in first-seen order* are both exercised rather than both satisfied
#: by accident. Row 4 names a node ``mammals`` does not carry, which is what
#: puts a row in the roll-up's ``unplaced`` half and keeps the agreement below
#: from being an agreement about ``supported`` alone.
CORPUS = [
    _row("golden_retriever", "beagle"),
    _row("spay", ontology="procedures", axis="kind"),
    _row("dog"),
    _row("inv", ontology="billing", axis="code"),
    _row("wolfhound"),
    _row("spay", ontology="procedures", axis="kind"),
    _row("inv", ontology="billing", axis="code"),
]

#: One vocabulary on two axes --- the corpus the two heights disagree over.
TWO_AXIS = [
    _row("golden_retriever"),
    _row("domestic", axis="habitat"),
    _row("dog"),
]


async def _registry(*documents: dict[str, Any]) -> OntologyRegistry:
    """A registry holding what it was handed, in the order it was handed them.

    The first document is the one the registry is constructed with and the rest
    arrive through ``load()``, because ``from_components`` takes a config rather
    than nothing --- a registry has an id space before it has anything in it.
    """
    first, rest = documents[0], documents[1:]
    registry = OntologyRegistry.from_components(config=OntologyConfig.from_dict(first))
    await registry.load()
    for document in rest:
        await registry.load(document)
    return registry


@pytest.fixture
async def holding_all_three() -> AsyncIterator[OntologyRegistry]:
    """The registry the ranking is measured against, loaded out of corpus order.

    It holds every id the corpus names, and the order it holds them in is
    ``mammals``, ``billing``, ``procedures`` --- deliberately neither the
    corpus's first-seen order nor the alphabet, so that ranking by
    :meth:`~dataknobs_data.ontology.OntologyRegistry.list_ids` order is a third
    wrong answer the assertion can see rather than one it agrees with.
    """
    registry = await _registry(MAMMALS, BILLING, PROCEDURES)
    try:
        yield registry
    finally:
        await registry.close()


@pytest.fixture
async def holding_mammals_alone() -> AsyncIterator[OntologyRegistry]:
    """The registry the drop is measured against: ``billing`` is never loaded."""
    registry = await _registry(MAMMALS)
    try:
        yield registry
    finally:
        await registry.close()


def _touched(answer: SupportSet[str]) -> tuple[int, ...]:
    """Every row a roll-up's support set reaches, placed or not.

    ``unplaced`` is included because a tag this vocabulary does not carry is
    still a row that named the vocabulary, and the selection counts it. Reading
    ``supported`` alone would make the agreement below fail against correct code
    over any corpus with a residue in it --- which every real one has.

    ``unsupported_rows`` is the third collection and is deliberately **not**
    read: it holds positions the roll-up reached for and did not place at all,
    and one of its four causes --- a local id this vocabulary's ``KeyCodec``
    could not read --- is a row the selection *does* count. That is the second
    condition on the equality clause below, and it is why the clause names the
    codec as well as the axis.
    """
    return tuple(
        sorted({row for entry in (*answer.supported, *answer.unplaced) for row in entry.rows})
    )


async def test_the_selection_ranks_by_row_count_with_ties_in_first_seen_order(
    holding_all_three: OntologyRegistry,
) -> None:
    """Descending by rows, and the tie broken by the corpus rather than by the id.

    The tied pair is the assertion. ``procedures`` and ``billing`` name two rows
    each; ``procedures`` is named first and sorts second, so an implementation
    that ranked alphabetically would put ``billing`` ahead of it and fail here
    while passing over any corpus whose two orders agree. The fixture then loads
    the pair the other way round again, so ranking by the registry's own load
    order is a third arrangement this refuses rather than one it agrees with ---
    and that is the tie-break a member *of a registry* would most plausibly get
    wrong, since :meth:`~dataknobs_data.ontology.OntologyRegistry.list_ids` is
    right there and already ordered.

    The rows travel with each entry and are the caller's positions, which is
    what makes the answer usable without a second pass over the corpus.
    """
    assert holding_all_three.list_ids() == ["mammals", "billing", "procedures"], (
        "the fixture no longer holds all three, or no longer loads them out of "
        "corpus order. Either way the ranking below has stopped measuring how "
        "many rows named each: it is reading which ids are loaded, or agreeing "
        "with the load order rather than refusing it"
    )

    tags = read_node_tags_many(CORPUS).tags

    assert [
        (entry.ontology_id, entry.rows) for entry in holding_all_three.ontologies_in_play(tags)
    ] == [
        ("mammals", (0, 2, 4)),
        ("procedures", (1, 5)),
        ("billing", (3, 6)),
    ]


async def test_the_selection_and_the_roll_up_agree_over_one_corpus(
    holding_all_three: OntologyRegistry,
) -> None:
    """One measure at two heights, and the agreement has two strengths.

    **Containment holds over any corpus**: every row the roll-up's support set
    reaches is a row the selection attributed to that vocabulary. That is the
    invariant, and it is what fails the moment the selection counts rows a
    second way rather than filtering the count that already exists.

    **Equality holds over a corpus stated to be single-axis**, and only there. A
    roll-up is scoped to one axis and a selection is scoped to a vocabulary, so
    a row tagged on a second axis of the same vocabulary is counted by one and
    filtered out by the other --- correct in both, and the reason the strong
    clause carries its qualifier instead of being asserted everywhere.

    **Single-axis is necessary and is not sufficient**; the second condition is
    the codec. A tag whose local id this vocabulary's ``KeyCodec`` cannot read
    localizes to nothing, so no grouping ever sees it and it lands in
    ``unsupported_rows`` --- which :func:`_touched` does not read. The selection
    counts that same row, because ``ontology_support`` reads the tag's
    ``ontology_id`` and localizes nothing. Equality would then fail against
    entirely correct code over a single-axis corpus. It cannot happen here,
    where the key space is ``str`` and the codec is the identity; a consumer who
    binds another must read the qualifier as *single-axis and every local id
    readable*, which is the form this clause is asserted in rather than the
    shorter one.
    """
    tags = read_node_tags_many(CORPUS).tags
    selection = {
        entry.ontology_id: entry.rows for entry in holding_all_three.ontologies_in_play(tags)
    }
    ontology = holding_all_three.get("mammals")
    assert ontology is not None

    touched = _touched(await async_roll_up(ontology, "species", tags))
    assert set(touched) <= set(selection["mammals"])
    assert touched == selection["mammals"]

    two_axis = read_node_tags_many(TWO_AXIS).tags
    across = {
        entry.ontology_id: entry.rows for entry in holding_all_three.ontologies_in_play(two_axis)
    }
    reached = _touched(await async_roll_up(ontology, "species", two_axis))

    assert set(reached) <= set(across["mammals"])
    assert reached != across["mammals"], (
        "the two-axis corpus no longer separates the two clauses, so the "
        "equality above is being asserted over a case that cannot refute it"
    )


async def test_a_vocabulary_this_registry_does_not_hold_is_dropped_and_recoverable(
    holding_mammals_alone: OntologyRegistry,
) -> None:
    """The drop is the answer, and the door out of it is one import away.

    A registry narrows to what it holds, so a tag naming a vocabulary nobody
    loaded does not appear --- and nothing reports that it was there. What makes
    that a design rather than a silence is that the unnarrowed count is public:
    the same corpus through ``ontology_support`` answers both, so a caller who
    wants the residue takes one set difference.

    **And the empty answer is a conflation, asserted rather than argued away.**
    A corpus naming only vocabularies this registry has never seen answers the
    same ``()`` an empty corpus answers, and a registry that has loaded nothing
    answers it for every corpus at all. All three are asserted below. One level
    down the first of them is kept visible by ``unsupported_rows``; here none of
    them is, and a test says so instead of leaving a consumer to find it.

    **The third is the one reached by accident**, because the answer is over
    what a registry has *loaded* rather than over what its configuration names,
    and the synchronous door loads nothing --- ``_ainit`` is the only auto-load
    and it runs on the asynchronous door alone. Every other member worth calling
    on an unloaded registry is a coroutine, so the await is forced; this one is
    not, so a caller can reach it one line too early and be told ``()``.
    """
    mixed: Sequence[Sequence[NodeTag]] = read_node_tags_many(
        [_row("dog"), _row("inv", ontology="billing", axis="code"), _row("beagle")]
    ).tags

    assert [
        (entry.ontology_id, entry.rows) for entry in holding_mammals_alone.ontologies_in_play(mixed)
    ] == [("mammals", (0, 2))]
    assert [(entry.ontology_id, entry.rows) for entry in ontology_support(mixed)] == [
        ("mammals", (0, 2)),
        ("billing", (1,)),
    ]

    only_unheld = read_node_tags_many([_row("inv", ontology="billing", axis="code")]).tags
    assert holding_mammals_alone.ontologies_in_play(only_unheld) == ()
    assert holding_mammals_alone.ontologies_in_play([]) == ()
    assert (
        holding_mammals_alone.ontologies_in_play(read_node_tags_many([{"invoice_id": "2291"}]).tags)
        == ()
    )

    # The third state, and the one no line above reaches: a registry holding
    # nothing answers `()` for the very corpus the first assertion here gets a
    # ranking out of. `from_config` is the synchronous door, so it resolves the
    # document and loads none of it -- which is the whole hazard, because the
    # config names `mammals` and the answer does not.
    unloaded = OntologyRegistry.from_config(OntologyConfig.from_dict(MAMMALS))
    try:
        assert unloaded.list_ids() == [], (
            "the synchronous door has started loading, so the assertion below "
            "is no longer about a registry that holds nothing"
        )
        assert unloaded.ontologies_in_play(mixed) == ()
    finally:
        await unloaded.close()
