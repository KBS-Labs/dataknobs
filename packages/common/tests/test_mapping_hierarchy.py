"""The structure axis a consumer holds rather than reads: a parent mapping.

Three ways in, and each answers a different question -- carried inline, minted
from a nested tree, or taken as a snapshot of a live axis. All three produce the
same object, which is why they are constructors rather than three classes.

The tests here are grouped by what could go silently wrong, not by member.
``contains`` gets a section of its own because it stopped being informational:
``Taxonomy.walk`` refuses an unknown anchor by asking it, so an answer narrower
than the walk-reachable set refuses nodes that are genuinely there and blames
the anchor.
"""

from __future__ import annotations

import asyncio

import pytest

from dataknobs_common.exceptions import NotFoundError, ValidationError
from dataknobs_common.hierarchy import (
    AsyncBulkHierarchy,
    AsyncHierarchy,
    AsyncMappingHierarchy,
    BulkHierarchy,
    Hierarchy,
    MappingHierarchy,
    ancestors,
    async_ancestors,
)
from dataknobs_common.ontology import load_ontology
from dataknobs_common.ontology.model import TaxonomyDefinition
from dataknobs_common.ontology.sources import MappingEntitySource
from dataknobs_common.taxonomy import Taxonomy

#: A parent mapping whose top node appears **only as a parent**.
#:
#: One fixture node carries two tests. ``mammal`` is never a key here, so it is
#: the node a ``contains`` answering on keys alone gets wrong -- and it is
#: therefore also the cheapest anchor for the refusal test, since that refusal
#: is ``contains`` asked by another name.
SPECIES = {"dog": ("mammal",), "beagle": ("dog",), "retriever": ("dog",)}

#: A DAG node with two parents. The shape the structure axis exists to carry.
DIAMOND = {"b": ("a",), "c": ("a",), "d": ("b", "c")}

PRODUCT_AREAS_TREE = {
    "name": "Billing",
    "children": [
        {"name": "Invoices", "children": [{"name": "Late Fees"}]},
        {"name": "Refunds"},
    ],
}


def _axis(parent_map: dict[str, tuple[str, ...]]) -> Taxonomy:
    """A taxonomy whose structure is a mapping and whose content is empty.

    ``walk`` reads the structure axis and nothing else, so an empty entity
    source is the honest backing rather than a stub: there is no content in this
    test and pretending otherwise would be the thing to explain.
    """
    return Taxonomy(
        definition=TaxonomyDefinition(id="species", relation="isa"),
        structure=MappingHierarchy(parent_map),
        entities=MappingEntitySource({}),
    )


# --------------------------------------------------------------------------
# Carried inline -- the four members, and the node that is only a parent
# --------------------------------------------------------------------------


def test_the_members_answer_off_the_mapping_and_its_inversion() -> None:
    axis = MappingHierarchy(SPECIES)

    assert axis.parents("beagle") == ("dog",)
    assert axis.children("dog") == ("beagle", "retriever")
    assert axis.parents("mammal") == ()
    assert axis.children("beagle") == ()


def test_a_node_that_is_only_a_parent_is_contained() -> None:
    """The half a naive ``node_id in parent_map`` misses.

    ``mammal`` is a value under ``dog`` and a key of nothing. It is as much a
    term of this axis as any other node, and the walks reach it -- so a
    ``contains`` that answered on keys alone would call it absent while
    ``ancestors`` returns it.
    """
    axis = MappingHierarchy(SPECIES)

    assert axis.contains("mammal")
    assert axis.contains("beagle")
    assert not axis.contains("wolf")
    assert "mammal" in ancestors(axis, "beagle")


def test_roots_are_the_nodes_with_no_parents_including_the_unkeyed_one() -> None:
    """A root need not be a key: having no parents is what makes one."""
    assert MappingHierarchy(SPECIES).roots() == ("mammal",)
    assert MappingHierarchy(DIAMOND).roots() == ("a",)
    assert MappingHierarchy({"lone": ()}).roots() == ("lone",)
    assert MappingHierarchy({}).roots() == ()


def test_an_anchor_that_is_only_a_parent_is_not_refused() -> None:
    """``contains`` is a refusal's evidence, so this is the same test again.

    ``Taxonomy.walk(from_id=...)`` refuses an anchor its axis does not contain.
    A ``contains`` narrower than the reachable set therefore surfaces as a
    refusal naming the *anchor* -- a message pointing at the caller for a defect
    in the backing, which is the hardest kind to trace back.
    """
    assert tuple(_axis(SPECIES).walk(from_id="mammal")) == (
        "mammal",
        "dog",
        "beagle",
        "retriever",
    )

    with pytest.raises(NotFoundError):
        tuple(_axis(SPECIES).walk(from_id="wolf"))


# --------------------------------------------------------------------------
# The bulk members -- the first non-assertion adopter
# --------------------------------------------------------------------------


def test_the_bulk_members_reply_positionally_one_sequence_per_node() -> None:
    """One reply per node asked about, in the order asked, empties included."""
    axis = MappingHierarchy(SPECIES)

    assert axis.parents_many(("beagle", "mammal", "wolf")) == (("dog",), (), ())
    assert axis.children_many(("dog", "beagle")) == (("beagle", "retriever"), ())


def test_both_flavours_satisfy_the_bulk_protocol_at_runtime() -> None:
    """Opting in is a structural claim, so it is checked structurally.

    The drivers dispatch on **presence** rather than on ``isinstance``, which is
    what keeps the capability optional -- but a consumer typing against
    ``BulkHierarchy`` is making the claim these assertions are about.
    """
    assert isinstance(MappingHierarchy(SPECIES), BulkHierarchy)
    assert isinstance(AsyncMappingHierarchy(SPECIES), AsyncBulkHierarchy)


def test_the_bulk_members_agree_with_the_singular_ones() -> None:
    """The driver prefers bulk without asking, so disagreement is invisible."""
    axis = MappingHierarchy(DIAMOND)
    nodes = ("a", "b", "c", "d", "absent")

    assert axis.parents_many(nodes) == tuple(axis.parents(n) for n in nodes)
    assert axis.children_many(nodes) == tuple(axis.children(n) for n in nodes)


def test_the_bound_is_validated_even_where_the_bulk_path_skips_it() -> None:
    """A bulk backing never builds the semaphore -- and still checks the width.

    ``_async_reply`` returns on the bulk branch *before* the semaphore is
    constructed, which is correct: there is nothing to bound in a dict lookup.
    The validation is deliberately not on the far side of that branch, so a
    caller cannot pass a deadlocking width and have it silently accepted because
    their backing happened to offer bulk members.
    """
    axis = AsyncMappingHierarchy(SPECIES)

    with pytest.raises(ValueError, match="max_concurrency must be at least 1"):
        asyncio.run(AsyncMappingHierarchy.snapshot(axis, max_concurrency=0))


# --------------------------------------------------------------------------
# from_nested -- one document, one set of keys, whichever door read it
# --------------------------------------------------------------------------


def test_from_nested_mints_the_ids_a_nested_ontology_source_mints() -> None:
    """The whole reason the minting is a shared core rather than two copies.

    A consumer who loads a document through ``kind: nested`` and holds the same
    tree in memory must get **one** vocabulary. Two implementations of the slug
    rule would be two sets of keys for one tree, and nothing would say so.
    """
    onto = load_ontology(
        {
            "id": "areas",
            "sources": [{"id": "product_areas", "kind": "nested", "tree": PRODUCT_AREAS_TREE}],
        }
    )
    held = MappingHierarchy.from_nested(PRODUCT_AREAS_TREE)

    assert set(held.parent_map) == set(onto.entities.by_type("product_areas"))
    for entity_id in held.parent_map:
        edges = onto.assertions.find(subject=entity_id, relation="isa")
        assert held.parents(entity_id) == tuple(edge.object.entity_id for edge in edges)


def test_from_nested_builds_a_walkable_axis() -> None:
    axis = MappingHierarchy.from_nested(PRODUCT_AREAS_TREE)

    assert axis.roots() == ("billing",)
    assert ancestors(axis, "billing/invoices/late-fees") == ("billing/invoices", "billing")
    assert axis.children("billing") == ("billing/invoices", "billing/refunds")


def test_from_nested_takes_a_forest_custom_keys_and_nothing_at_all() -> None:
    """Three shapes a hand-maintained file takes, none of them malformed."""
    forest = MappingHierarchy.from_nested([{"name": "One"}, {"name": "Two"}])
    assert forest.roots() == ("one", "two")

    renamed = MappingHierarchy.from_nested(
        {"title": "Top", "sub": [{"title": "Under"}]}, child_key="sub", name_key="title"
    )
    assert renamed.parents("top/under") == ("top",)

    assert MappingHierarchy.from_nested(None).roots() == ()


def test_from_nested_refuses_two_paths_that_slug_to_one_id() -> None:
    """The refusal the ontology door has, through the door that has no source.

    Shipping this constructor without it would reintroduce a closed defect on a
    new path: the slug collapses punctuation and case, so ``Late Fees`` and
    ``late-fees`` are two nodes to the person editing the file and one to the
    slug, and the second silently replaced the first.
    """
    with pytest.raises(ValidationError) as excinfo:
        MappingHierarchy.from_nested(
            {"name": "Billing", "children": [{"name": "Late Fees"}, {"name": "late-fees"}]}
        )

    assert "'billing/late-fees'" in str(excinfo.value)
    assert "this tree" in str(excinfo.value)


# --------------------------------------------------------------------------
# snapshot -- a live axis walked once
# --------------------------------------------------------------------------


class _MovingAxis:
    """A hierarchy whose edges change, which no shipped class is.

    Every member delegates to a :class:`MappingHierarchy` this object replaces,
    so it re-derives nothing: what it adds is the one property a frozen value
    cannot have, which is the property a snapshot exists to be tested against.
    """

    def __init__(self, parent_map: dict[str, tuple[str, ...]]) -> None:
        self.rebuild(parent_map)

    def rebuild(self, parent_map: dict[str, tuple[str, ...]]) -> None:
        self._axis = MappingHierarchy(parent_map)

    def roots(self) -> tuple[str, ...]:
        return tuple(self._axis.roots())

    def parents(self, node_id: str) -> tuple[str, ...]:
        return tuple(self._axis.parents(node_id))

    def children(self, node_id: str) -> tuple[str, ...]:
        return tuple(self._axis.children(node_id))

    def contains(self, node_id: str) -> bool:
        return self._axis.contains(node_id)


def test_a_snapshot_answers_what_the_live_axis_answered() -> None:
    live: Hierarchy[str] = _MovingAxis(SPECIES)

    snapshot = MappingHierarchy.snapshot(live)

    for node_id in ("mammal", "dog", "beagle", "retriever"):
        assert snapshot.parents(node_id) == live.parents(node_id)
        assert snapshot.children(node_id) == live.children(node_id)
        assert snapshot.contains(node_id)
    assert snapshot.roots() == live.roots()


def test_a_snapshot_is_a_copy_and_the_axis_moving_does_not_move_it() -> None:
    """The one thing a live axis cannot do: say what has changed since.

    A snapshot that quietly tracked its source would pass every test written for
    the source, which is why this is asserted from the direction that fails.
    """
    live = _MovingAxis(SPECIES)
    snapshot = MappingHierarchy.snapshot(live)

    live.rebuild({"dog": ("reptile",), "beagle": ("dog",)})

    assert live.parents("dog") == ("reptile",)
    assert snapshot.parents("dog") == ("mammal",)


def test_a_snapshot_keeps_every_parent_of_a_node_that_has_two() -> None:
    """A DAG copied into a tree would be a different axis wearing the name."""
    snapshot = MappingHierarchy.snapshot(MappingHierarchy(DIAMOND))

    assert snapshot.parents("d") == ("b", "c")
    assert snapshot.children("a") == ("b", "c")


def test_a_snapshot_sees_what_descends_from_the_roots_and_says_so() -> None:
    """The protocol's limit, pinned rather than left to be discovered.

    A ``Hierarchy`` has no extent member -- ``roots()`` says as much in its own
    docstring -- so descending is the only enumeration available and a cyclic
    component with no root above it is invisible to it. Absent from the copy is
    the honest outcome; asserting it is what keeps the docstring true.
    """
    orphaned_cycle = MappingHierarchy({"a": ("b",), "b": ("a",)})

    assert orphaned_cycle.roots() == ()
    assert dict(MappingHierarchy.snapshot(orphaned_cycle).parent_map) == {}


def test_a_snapshot_of_a_snapshot_is_the_same_axis() -> None:
    """Idempotent on the answers, which is what a copy of a copy must be."""
    once = MappingHierarchy.snapshot(MappingHierarchy(SPECIES))
    twice = MappingHierarchy.snapshot(once)

    assert dict(twice.parent_map) == dict(once.parent_map)
    assert twice.roots() == once.roots()


# --------------------------------------------------------------------------
# The asynchronous twin -- same answers, one await
# --------------------------------------------------------------------------


def test_the_async_twin_answers_what_the_sync_one_does() -> None:
    """Same mapping, same answers. The slot is what differs, not the axis."""
    sync_axis = MappingHierarchy(SPECIES)
    async_axis = AsyncMappingHierarchy(SPECIES)

    async def _read() -> None:
        assert await async_axis.roots() == sync_axis.roots()
        assert await async_axis.parents("beagle") == sync_axis.parents("beagle")
        assert await async_axis.children("dog") == sync_axis.children("dog")
        assert await async_axis.contains("mammal") == sync_axis.contains("mammal")
        assert await async_axis.parents_many(("beagle", "wolf")) == sync_axis.parents_many(
            ("beagle", "wolf")
        )
        assert await async_ancestors(async_axis, "beagle") == ancestors(sync_axis, "beagle")

    asyncio.run(_read())


def test_the_async_twin_satisfies_the_async_protocol_and_nests_the_same() -> None:
    async_axis = AsyncMappingHierarchy.from_nested(PRODUCT_AREAS_TREE)

    assert isinstance(async_axis, AsyncHierarchy)
    assert dict(async_axis.parent_map) == dict(
        MappingHierarchy.from_nested(PRODUCT_AREAS_TREE).parent_map
    )


def test_the_async_snapshot_walks_a_live_async_axis() -> None:
    """The same lift, over the flavour that has something to await."""
    live = AsyncMappingHierarchy(SPECIES)

    snapshot = asyncio.run(AsyncMappingHierarchy.snapshot(live))

    assert isinstance(snapshot, AsyncMappingHierarchy)
    assert asyncio.run(snapshot.roots()) == ("mammal",)
    assert asyncio.run(snapshot.parents("beagle")) == ("dog",)
