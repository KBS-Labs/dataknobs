"""Reaching an axis from the ontology, walking it, and being refused one.

``taxonomy(name)`` is an accessor, not a factory: everything it needs is a
field the ontology already holds, which is why it takes only the name. The
tests here assert that in both directions -- what a caller does *not* have to
supply, and what the accessor refuses to build rather than downgrading in
silence.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from dataknobs_common.exceptions import NotFoundError, ValidationError
from dataknobs_common.hierarchy import AsyncMappingHierarchy, MappingHierarchy, flatten
from dataknobs_common.ontology import async_load_ontology, load_ontology
from dataknobs_common.ontology.hierarchy import (
    AssertionHierarchy,
    AsyncAssertionHierarchy,
)
from dataknobs_common.ontology.model import (
    AttributeDef,
    EntityType,
    InferenceMode,
    Materialization,
    TaxonomyDefinition,
    split_qualified,
)
from dataknobs_common.ontology.sources import MappingEntitySource
from dataknobs_common.ontology.values import AsyncOntology
from dataknobs_common.ontology.taxonomy import AsyncTaxonomy, Taxonomy
from dataknobs_common.testing import assert_twins_agree

#: The smallest loadable vocabulary: one edge, and no ``taxonomies:`` at all.
#:
#: The axis under test is supplied by hand, so a document declaring one would
#: be a second definition of it that nothing reads.
MINIMAL_DOCUMENT = {
    "ontology": {
        "id": "minimal",
        "version": "1.0",
        "entity_types": [{"id": "Species"}],
        "entities": [
            {"id": "dog", "type": "Species"},
            {"id": "mammal", "type": "Species"},
        ],
        "assertions": [{"subject": "dog", "relation": "isa", "object": "mammal"}],
    }
}

# --------------------------------------------------------------------------
# The axis is reachable, and it takes nothing but its name
# --------------------------------------------------------------------------


def test_the_axis_walks_the_authored_terms(mammals_v11_path: Path) -> None:
    """Every term the file declares, breadth first from the axis's root."""
    onto = load_ontology(mammals_v11_path)

    walked = tuple(onto.taxonomy("species").walk())

    assert walked == ("mammal", "dog", "retriever", "beagle", "golden_retriever")


def test_the_accessor_takes_only_the_axis_name(mammals_v11_path: Path) -> None:
    """No store, no loop, and no consumer-supplied collaborator.

    The whole of what makes this an accessor rather than a factory: every
    argument a ``Taxonomy`` needs is already a field of the ontology, so the
    caller supplies a name and nothing else.
    """
    onto = load_ontology(mammals_v11_path)

    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()

    parameters = inspect.signature(type(onto).taxonomy).parameters
    assert list(parameters) == ["self", "name"]

    axis = onto.taxonomy("species")
    assert isinstance(axis, Taxonomy)
    assert isinstance(axis.structure, AssertionHierarchy)
    assert axis.entities is onto.entities
    assert axis.assertions is onto.assertions
    assert axis.definition is onto.taxonomies["species"]


def test_the_walk_starts_where_it_is_told(mammals_v11_path: Path) -> None:
    """``from_id`` anchors the walk, and the anchor is included."""
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    assert tuple(axis.walk(from_id="dog")) == (
        "dog",
        "retriever",
        "beagle",
        "golden_retriever",
    )


def test_max_depth_bounds_the_levels_expanded(mammals_v11_path: Path) -> None:
    """Zero is the anchor alone, and each step adds one level."""
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    assert tuple(axis.walk(from_id="dog", max_depth=0)) == ("dog",)
    assert tuple(axis.walk(from_id="dog", max_depth=1)) == ("dog", "retriever", "beagle")


def test_the_walk_yields_each_node_once(mammals_v11_path: Path) -> None:
    """A DAG node reachable by two paths is still one node."""
    walked = tuple(load_ontology(mammals_v11_path).taxonomy("species").walk())

    assert len(walked) == len(set(walked))


@pytest.mark.asyncio
async def test_the_async_twin_walks_the_same_terms(mammals_v11_path: Path) -> None:
    """Same order, same terms, awaited -- and ``taxonomy()`` is not awaited.

    A plain ``def`` on the asynchronous twin, because it constructs over fields
    the object is already holding. Asserted rather than assumed: making it
    awaitable would cost every caller an ``await`` for a dictionary lookup.
    """
    onto = await async_load_ontology(mammals_v11_path)

    axis = onto.taxonomy("species")

    assert isinstance(axis, AsyncTaxonomy)
    assert isinstance(axis.structure, AsyncAssertionHierarchy)
    assert not inspect.iscoroutinefunction(type(onto).taxonomy)

    walked = [node async for node in axis.walk()]
    assert tuple(walked) == ("mammal", "dog", "retriever", "beagle", "golden_retriever")


# --------------------------------------------------------------------------
# Refused at the accessor, naming the axis
# --------------------------------------------------------------------------


def test_a_materialized_content_axis_is_refused_naming_the_axis(
    materialized_content_path: Path,
) -> None:
    """Refused where the caller asked, not at the first walk.

    A loader that silently downgrades is a loader whose output nobody can
    reason about, and a failure at first walk is a failure at a call site with
    no idea why. The message names the axis, so the answer is actionable
    without reading this code.
    """
    onto = load_ontology(materialized_content_path)

    with pytest.raises(ValidationError) as raised:
        onto.taxonomy("species")

    assert "species" in str(raised.value)
    assert raised.value.context["taxonomy"] == "species"
    assert raised.value.context["axis"] == "content"


def test_a_materialized_structure_axis_is_a_copy_taken_at_load(
    materialized_structure_path: Path,
) -> None:
    """``materialized`` is honoured rather than refused, and by the door.

    The mode was refused for as long as nothing took the snapshot. Taking it at
    load is the only place both flavours can: ``taxonomy()`` is a plain ``def``
    on both twins while the asynchronous snapshot is ``async``, and it builds
    afresh per call, so a copy taken there would be a new copy with a new build
    time on every fetch.
    """
    onto = load_ontology(materialized_structure_path)

    axis = onto.taxonomy("species")

    assert isinstance(axis.structure, MappingHierarchy)
    assert tuple(axis.walk()) == ("mammal", "dog", "retriever", "beagle", "golden_retriever")


def test_a_materialized_axis_answers_what_the_live_one_answers(
    mammals_v11_path: Path,
    materialized_structure_path: Path,
) -> None:
    """One vocabulary, two modes, and the same axis under both.

    The property that makes ``materialization`` a *freshness* decision rather
    than a semantic one: what changes is when the edges were read, not which
    edges they are. Asserted against the same document with the block flipped,
    which is the only difference between the two files.
    """
    live = load_ontology(mammals_v11_path).taxonomy("species")
    copied = load_ontology(materialized_structure_path).taxonomy("species")

    assert tuple(copied.walk()) == tuple(live.walk())
    for node in ("mammal", "dog", "retriever", "beagle", "golden_retriever"):
        assert copied.structure.parents(node) == live.structure.parents(node)
        assert copied.structure.children(node) == live.structure.children(node)
        assert copied.structure.contains(node)
    assert copied.structure.roots() == live.structure.roots()


def test_a_materialized_axis_is_fixed_where_the_live_one_follows_its_source(
    materialized_structure_path: Path,
    mammals_v11_path: Path,
) -> None:
    """The one thing a copy is for: a build time the live read does not have.

    Both ontologies are asked for their axis twice. The live one rebuilds from
    the source each time, so a source that moved would move it; the materialized
    one hands back the structure taken at load, so the two fetches are the same
    object rather than two reads that happen to agree.
    """
    materialized = load_ontology(materialized_structure_path)
    live = load_ontology(mammals_v11_path)

    assert materialized.taxonomy("species").structure is materialized.taxonomy("species").structure
    assert live.taxonomy("species").structure is not live.taxonomy("species").structure


def test_a_materialized_axis_keeps_a_component_no_root_reaches() -> None:
    """The copy is the axis, not the part of it a descent can reach.

    ``isa`` is asserted, not constrained, so a document may name a cycle -- and
    a cycle with nothing above it has no root, which is the enumeration a
    descent starts from. Copying by descending would drop it, and the axis would
    then refuse an anchor the same axis read live accepts. The assertion source
    can say what it holds, so the copy asks it.
    """
    document = {
        "ontology": {
            "id": "loops",
            "version": "1.0",
            "entity_types": [{"id": "Node"}],
            "entities": [
                {"id": "a", "type": "Node", "name": "A"},
                {"id": "b", "type": "Node", "name": "B"},
            ],
            "assertions": [
                {"subject": "a", "relation": "isa", "object": "b"},
                {"subject": "b", "relation": "isa", "object": "a"},
            ],
            "taxonomies": [
                {
                    "id": "loop",
                    "relation": "isa",
                    "materialization": {"structure": "materialized"},
                }
            ],
        }
    }

    axis = load_ontology(document).taxonomy("loop")

    assert axis.structure.roots() == ()
    assert axis.structure.contains("a")
    assert tuple(axis.walk(from_id="a")) == ("a", "b")


@pytest.mark.asyncio
async def test_the_async_door_materializes_the_same_axis(
    materialized_structure_path: Path,
) -> None:
    """Same document, same copy, one ``await`` -- so the mode means one thing.

    The asynchronous snapshot is a coroutine where the synchronous one is not,
    which is why the copy is taken at the door: this is the only member of the
    pair with somewhere to put the ``await``.
    """
    onto = await async_load_ontology(materialized_structure_path)

    axis = onto.taxonomy("species")

    assert isinstance(axis.structure, AsyncMappingHierarchy)
    walked = [node async for node in axis.walk()]
    assert tuple(walked) == ("mammal", "dog", "retriever", "beagle", "golden_retriever")


def test_a_hand_built_ontology_that_declares_materialized_is_refused(
    materialized_structure_path: Path,
) -> None:
    """The invariant the doors keep, asserted where it can be broken.

    ``Ontology`` is a public dataclass, so a caller can build one directly and
    hand it taxonomies whose definitions ask for a copy it carries none of.
    Falling back to the live axis there is the silent downgrade the refusal
    existed to prevent, so the accessor says so instead -- naming the axis, at
    the call that asked for it.
    """
    loaded = load_ontology(materialized_structure_path)
    hand_built = replace(loaded, structures={})

    with pytest.raises(ValidationError) as raised:
        hand_built.taxonomy("species")

    assert "species" in str(raised.value)
    assert raised.value.context["taxonomy"] == "species"
    assert raised.value.context["axis"] == "structure"


def test_a_copy_is_found_under_the_name_the_axis_was_asked_for() -> None:
    """``structures`` is keyed like ``taxonomies``, not by ``definition.id``.

    A loader door keys both the same way, so nothing it builds can tell the two
    apart. ``taxonomies`` is a mapping a caller may build directly, though, and
    a definition filed under an alias is reached by that alias -- so a copy
    looked up by ``definition.id`` would be missing for exactly the ontology
    that supplied it.
    """
    copied = MappingHierarchy({"dog": ("mammal",)})
    onto = replace(
        load_ontology(MINIMAL_DOCUMENT),
        taxonomies={
            "under_an_alias": TaxonomyDefinition(
                id="species",
                relation="isa",
                materialization=Materialization(structure=InferenceMode.MATERIALIZED),
            )
        },
        structures={"under_an_alias": copied},
    )

    assert onto.taxonomy("under_an_alias").structure is copied


def test_the_default_axes_are_the_ones_that_exist(mammals_v11_path: Path) -> None:
    """A vocabulary that declares no ``materialization:`` still builds.

    The half a refusal is most likely to break, and it is the half the caller
    never types: both defaults must name a branch that exists, or every
    hand-edited file in the plan is refused for asking nothing.

    ``structure`` defaults to the live read because that is what the axis
    built here *is* -- an ``AssertionHierarchy``, which opens nothing and
    caches nothing. A default naming the snapshot would be a default no
    consumer could get.
    """
    definition = load_ontology(mammals_v11_path).taxonomies["species"]

    assert definition.materialization.structure.value == "on_demand"
    assert definition.materialization.content.value == "on_demand"
    assert tuple(load_ontology(mammals_v11_path).taxonomy("species").walk())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture_name", "without_its_copies"),
    [("materialized_content_path", False), ("materialized_structure_path", True)],
)
async def test_both_flavours_refuse_identically(
    request: pytest.FixtureRequest, fixture_name: str, without_its_copies: bool
) -> None:
    """One rule per refusal, and neither twin carries its own copy of either.

    Over both materialization refusals rather than one: they are
    ``_refuse_a_materialized_content_axis`` and ``_structure_for``, and a
    second refusal added to only one twin is exactly what this is here to
    catch.

    The structure case has to have its copies taken away first, because a
    loader door supplies them -- which is the point of the parameter rather
    than a detail of it: the refusal survives only for an ontology built by
    hand, and it is still one rule read by two accessors.
    """
    path: Path = request.getfixturevalue(fixture_name)
    sync_onto = load_ontology(path)
    async_onto = await async_load_ontology(path)
    if without_its_copies:
        sync_onto = replace(sync_onto, structures={})
        async_onto = replace(async_onto, structures={})

    with pytest.raises(ValidationError) as sync_raised:
        sync_onto.taxonomy("species")
    with pytest.raises(ValidationError) as async_raised:
        async_onto.taxonomy("species")

    assert str(sync_raised.value) == str(async_raised.value)
    assert sync_raised.value.context == async_raised.value.context


def test_an_undeclared_axis_is_refused_listing_what_is_declared(
    mammals_v11_path: Path,
) -> None:
    """The useful answer to a typo is the set it was nearly one of."""
    onto = load_ontology(mammals_v11_path)

    with pytest.raises(NotFoundError) as raised:
        onto.taxonomy("speceis")

    assert "speceis" in str(raised.value)
    assert raised.value.context["declared"] == ["species"]


# --------------------------------------------------------------------------
# The twins, and what the axis carries for the cursor that comes later
# --------------------------------------------------------------------------


def test_the_taxonomy_twins_expose_the_same_annotated_surface() -> None:
    """Same fields, same walk surface; the one difference is declared.

    ``max_concurrency`` bounds a frontier read with no bulk member to use, and
    the synchronous walk issues no concurrent calls at all -- a knob that does
    nothing is worse than an asymmetry that is stated. The guard compares that
    set by equality, so a second divergence fails rather than joining it.

    No ``compare_return`` on ``walk``: that pair is the one that streams, so its
    return annotations differ by flavour (``Iterator[str]`` against
    ``AsyncIterator[str]``) and asserting on them would pin the flavour rather
    than the contract. ``subtree_keys`` collects, returns ``list[str]`` on both
    halves, and is checked the other way -- which is the whole reason it gets
    its own call rather than joining the first.

    **All three members, because this type's surface is not one walk.** A guard
    over ``walk`` alone is green about ``subtree_keys`` by saying nothing about
    it, and a parameter added to one half of the member it does not name is
    exactly the drift it exists to catch. The field lists are compared too, so
    a field added to one flavour and not the other fails here -- which is the
    assertion the fifth field arrived under.
    """
    sync_fields = Taxonomy.__dataclass_fields__
    async_fields = AsyncTaxonomy.__dataclass_fields__

    assert list(sync_fields) == list(async_fields)

    assert_twins_agree(
        Taxonomy.walk,
        AsyncTaxonomy.walk,
        async_only={"max_concurrency"},
        label="Taxonomy.walk",
    )

    assert_twins_agree(
        Taxonomy.subtree_keys,
        AsyncTaxonomy.subtree_keys,
        async_only={"max_concurrency"},
        compare_return=True,
        label="Taxonomy.subtree_keys",
    )

    # The third member, and the one that is synchronous on both flavours: it
    # reads a mapping the caller already holds, so there is nothing to await.
    # `unflavoured` is how that is *declared* rather than skipped -- the guard
    # refuses a synchronous async-half by default, and an exception written
    # down is what keeps it from also being how a member drops out of the
    # comparison unnoticed.
    assert_twins_agree(
        Taxonomy.inherited_attributes,
        AsyncTaxonomy.inherited_attributes,
        unflavoured=True,
        compare_return=True,
        label="Taxonomy.inherited_attributes",
    )


def test_the_axis_carries_the_assertions_it_was_built_from(
    mammals_v11_path: Path,
) -> None:
    """What lets a cursor report the edge rather than only its endpoints.

    ``assertions is None`` is the question that tells *this edge carries no
    annotation* from *this axis has no annotations to give* -- a hierarchy over
    a ``parent_id`` column has rows and no assertions at all.
    """
    onto = load_ontology(mammals_v11_path)

    assert onto.taxonomy("species").assertions is onto.assertions

    unannotated = Taxonomy(
        definition=onto.taxonomies["species"],
        structure=AssertionHierarchy(onto.assertions, "isa"),
        entities=onto.entities,
    )
    assert unannotated.assertions is None


# --------------------------------------------------------------------------
# The anchor, and the member that was declared to answer for it
# --------------------------------------------------------------------------


def test_an_anchor_the_axis_does_not_contain_is_refused(mammals_v11_path: Path) -> None:
    """An unknown ``from_id`` is refused rather than yielded back.

    The anchor is *included* in the output by design, so seeding the frontier
    with it unchecked emits an id the axis does not contain as though it were a
    term of the axis -- and the caller cannot tell the difference, because a
    walk is exactly what they asked for.

    Yielding nothing was the other candidate and is the worse one. It collapses
    *nothing below this node* into *this node is not here*, which are the two
    answers ``Hierarchy.contains`` says in its own docstring it exists to keep
    apart. An axis that owns that member and then destroys the distinction in
    the one walk that needs it is refuting itself.
    """
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    assert not axis.structure.contains("marmoset")

    with pytest.raises(NotFoundError) as raised:
        tuple(axis.walk(from_id="marmoset"))

    assert "marmoset" in str(raised.value)
    assert raised.value.context["anchor"] == "marmoset"
    assert raised.value.context["taxonomy"] == "species"


@pytest.mark.asyncio
async def test_both_flavours_refuse_an_unknown_anchor_alike(
    mammals_v11_path: Path,
) -> None:
    """The twins refuse with the same type and the same words.

    The differential rather than either surface alone: one algorithm behaving
    two ways by flavour is what this pair's shape exists to prevent, and a
    check written into one streaming walk and forgotten in the other is the
    specific way that happens here -- the two walks are twinned by hand.
    """
    onto = load_ontology(mammals_v11_path)
    async_onto = await async_load_ontology(mammals_v11_path)

    with pytest.raises(NotFoundError) as sync_raised:
        tuple(onto.taxonomy("species").walk(from_id="marmoset"))

    with pytest.raises(NotFoundError) as async_raised:
        [node async for node in async_onto.taxonomy("species").walk(from_id="marmoset")]

    assert str(sync_raised.value) == str(async_raised.value)
    assert sync_raised.value.context == async_raised.value.context


def test_a_known_anchor_still_walks(mammals_v11_path: Path) -> None:
    """The check costs the walk nothing it was already answering.

    Pinned beside the refusal because a containment check placed wrongly --
    against the entity source rather than the structure, say -- would refuse
    every anchor and still satisfy the test above.
    """
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    assert tuple(axis.walk(from_id="dog")) == (
        "dog",
        "retriever",
        "beagle",
        "golden_retriever",
    )


# --------------------------------------------------------------------------
# A negated edge is not walked
# --------------------------------------------------------------------------
#
# One test and not four. Absent from `parents` and `children`, absent from the
# bulk forms, a root where the only parent edge is negated, `contains` False
# for a node named only by negated edges, and absent from a snapshot -- every
# one of them follows from one sentence: an axis is made of ASSERTED edges
# between entities.
#
# It was red before the change it guards, and the note that used to stand here
# said the opposite. These tests take a DOCUMENT, not an `Assertion`: a
# document could always write `polarity: negated`, the loader dropped it, and a
# walk that then places `whale` under `fish` IS the red state. Measured against
# the tree before an assertion could carry a polarity, every assertion below
# fails, with no `Polarity` import anywhere. What the change created is the
# PASS state.
#
# The correction is kept rather than quietly deleted, because it is the lesson:
# *this cannot be written as a failing test yet* is itself a claim, and the way
# to check it is to run the test. It was written without running it. Run it red
# first -- which is what the reproduce-first rule already asks for, and what
# would have caught this one.

#: Five ``isa`` edges, one of which the document states as a negation.
#:
#: ``whale`` is the case worth having: it is named by an asserted edge (as
#: ``orca``'s parent) so it is on the axis, and its own only parent edge is
#: negated -- which is what makes *is it a root* a real question rather than
#: *is it here at all*. ``fish`` is the other case: named by nothing but the
#: negation.
NEGATED_EDGE_DOCUMENT = {
    "id": "mammals",
    "entity_types": [{"id": "Species"}],
    "entities": [
        {"id": "mammal", "type": "Species"},
        {"id": "dog", "type": "Species"},
        {"id": "cat", "type": "Species"},
        {"id": "beagle", "type": "Species"},
        {"id": "whale", "type": "Species"},
        {"id": "orca", "type": "Species"},
        {"id": "fish", "type": "Species"},
    ],
    "assertions": [
        {"subject": "dog", "relation": "isa", "object": "mammal"},
        {"subject": "cat", "relation": "isa", "object": "mammal"},
        {"subject": "beagle", "relation": "isa", "object": "dog"},
        {"subject": "orca", "relation": "isa", "object": "whale"},
        {"subject": "whale", "relation": "isa", "object": "fish", "polarity": "negated"},
    ],
}


def test_a_negated_edge_is_not_a_parent_or_a_child() -> None:
    """The direct reads, and a positive control beside each.

    The control is not decoration: a filter that returned nothing at all would
    satisfy the two negative assertions on its own.
    """
    onto = load_ontology(NEGATED_EDGE_DOCUMENT)
    axis = AssertionHierarchy(onto.assertions, "isa")

    assert axis.parents("whale") == ()
    assert axis.parents("dog") == ("mammal",)
    assert axis.children("fish") == ()
    assert axis.children("dog") == ("beagle",)


def test_a_negated_edge_is_absent_from_the_bulk_reads_too() -> None:
    """A frontier read is what a walk actually calls, so it is where a missing
    filter would do its damage.
    """
    onto = load_ontology(NEGATED_EDGE_DOCUMENT)
    axis = AssertionHierarchy(onto.assertions, "isa")

    assert axis.parents_many(["whale", "dog"]) == ((), ("mammal",))
    assert axis.children_many(["fish", "dog"]) == ((), ("beagle",))


def test_a_node_whose_only_parent_edge_is_negated_is_a_root() -> None:
    """It has no asserted parent, and ``roots()`` is *the nodes this relation
    leaves unplaced*. Reporting ``whale`` as placed would report a placement
    no edge makes.
    """
    onto = load_ontology(NEGATED_EDGE_DOCUMENT)

    assert sorted(AssertionHierarchy(onto.assertions, "isa").roots()) == ["mammal", "whale"]


def test_a_node_named_only_by_a_negated_edge_is_absent() -> None:
    """``contains`` keeps *nothing below this node* apart from *this node is
    not here*, and a negation places nothing -- so ``fish`` is not on the axis
    at all, the way a literal object is not.
    """
    onto = load_ontology(NEGATED_EDGE_DOCUMENT)
    axis = AssertionHierarchy(onto.assertions, "isa")

    assert axis.contains("fish") is False
    assert axis.contains("whale") is True


def test_a_negated_edge_is_absent_from_a_snapshot() -> None:
    """A copy that held edges the live read excludes would be a different
    shape from the axis it copies, which is the one property a copy exists to
    not have.
    """
    onto = load_ontology(NEGATED_EDGE_DOCUMENT)
    axis = AssertionHierarchy(onto.assertions, "isa")

    edges = axis.parent_edges()
    assert "fish" not in edges
    assert edges["whale"] == ()

    copied = MappingHierarchy.snapshot(axis)
    assert copied.contains("fish") is False
    assert sorted(copied.roots()) == ["mammal", "whale"]


@pytest.mark.asyncio
async def test_the_async_twin_declines_a_negated_edge_alike() -> None:
    """The twins write the filter out twice, because they share no runtime
    code -- so the assertion that they agree is written out too.
    """
    onto = await async_load_ontology(NEGATED_EDGE_DOCUMENT)
    axis = AsyncAssertionHierarchy(onto.assertions, "isa")

    assert await axis.parents("whale") == ()
    assert await axis.parents("dog") == ("mammal",)
    assert await axis.children("fish") == ()
    assert await axis.parents_many(["whale", "dog"]) == ((), ("mammal",))
    assert sorted(await axis.roots()) == ["mammal", "whale"]
    assert await axis.contains("fish") is False
    assert "fish" not in await axis.parent_edges()


# --------------------------------------------------------------------------
# Spending the axis on a query -- subtree_keys
# --------------------------------------------------------------------------


def test_subtree_keys_is_the_axis_from_a_point_including_the_point(
    mammals_v11_path: Path,
) -> None:
    """What a filter over a foreign table is built from.

    The root is included because naming an interior node means *this and
    everything under it*. An off-by-one here under-counts silently while the
    count is the whole answer, which is why the inclusion is asserted rather
    than left to whichever walk the method happens to delegate to.
    """
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    keys = axis.subtree_keys("dog")

    assert keys[0] == "dog"
    assert set(keys) == {"dog", "retriever", "beagle", "golden_retriever"}
    assert len(keys) == len(set(keys)), "a repeated key changes a length a caller reports"


def test_subtree_keys_emits_the_delegated_pre_order_rather_than_the_walks_breadth(
    mammals_v11_path: Path,
) -> None:
    """It delegates, and the delegation is observable.

    ``walk`` is breadth first because it streams; ``subtree_keys`` collects, so
    it emits what the descending flattened walks emit. Over this vocabulary the
    two are different tuples, which is what makes the delegation assertable at
    all rather than merely plausible.
    """
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    assert axis.subtree_keys("dog") == ["dog", "retriever", "golden_retriever", "beagle"]
    assert tuple(axis.walk(from_id="dog")) == (
        "dog",
        "retriever",
        "beagle",
        "golden_retriever",
    )


def test_subtree_keys_bounds_on_depth_and_is_unbounded_by_default(
    mammals_v11_path: Path,
) -> None:
    """``depth`` is the bound, and the ordinary question is *everything under this*."""
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    assert axis.subtree_keys("dog", depth=0) == ["dog"]
    assert set(axis.subtree_keys("dog", depth=1)) == {"dog", "retriever", "beagle"}
    assert axis.subtree_keys("dog", depth=None) == axis.subtree_keys("dog")


def test_subtree_keys_refuses_an_unknown_root(mammals_v11_path: Path) -> None:
    """``walk``'s refusal, for ``walk``'s reason and not a new one.

    This walk *includes* its anchor, so an unknown one comes back as a
    one-element list -- a wrong answer indistinguishable from a leaf, arriving
    as the filter a caller is about to count with.
    """
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    with pytest.raises(NotFoundError) as raised:
        axis.subtree_keys("marmoset")

    assert raised.value.context["anchor"] == "marmoset"
    assert raised.value.context["taxonomy"] == "species"


@pytest.mark.asyncio
async def test_both_flavours_of_subtree_keys_agree(mammals_v11_path: Path) -> None:
    """The differential: what is twinned here is the driving and nothing else.

    Including the refusal, which is the half a hand-twinned pair loses first --
    the containment question is the ``await``, and the refusal below it is not.
    """
    onto = load_ontology(mammals_v11_path)
    async_onto = await async_load_ontology(mammals_v11_path)

    assert onto.taxonomy("species").subtree_keys("dog") == await async_onto.taxonomy(
        "species"
    ).subtree_keys("dog")
    assert onto.taxonomy("species").subtree_keys("dog", depth=1) == await async_onto.taxonomy(
        "species"
    ).subtree_keys("dog", depth=1)

    with pytest.raises(NotFoundError) as sync_raised:
        onto.taxonomy("species").subtree_keys("marmoset")
    with pytest.raises(NotFoundError) as async_raised:
        await async_onto.taxonomy("species").subtree_keys("marmoset")

    assert str(sync_raised.value) == str(async_raised.value)


#: A structure axis whose keys a localising parse would change, and which is
#: nonetheless entirely local-keyed.
#:
#: The two are different claims, which is the whole reason the guard below is a
#: round trip rather than a parse: ``split_qualified('mammal:carnivore')`` reads
#: ``mammal`` as an ontology segment and answers ``carnivore``, over an axis
#: where nothing was ever qualified.
COLON_KEYED = {"mammal:felid": ("mammal:carnivore",), "mammal:carnivore": ()}


def test_every_key_subtree_keys_returns_is_one_the_axis_answers_for(
    mammals_v11_path: Path,
) -> None:
    """The keys are the axis's own, and that is a round trip rather than a spelling.

    ``root_id`` arrives in the structure axis's space, so a key returned in any
    other space could not be fed back -- not to :meth:`Taxonomy.at`, not to
    ``structure.contains``, not to this method. Asserted as the round trip
    rather than as a parse, because *the axis is local-keyed* and *no key holds
    a colon* are different claims and only the second is one a parse can check.
    """
    axis = load_ontology(mammals_v11_path).taxonomy("species")

    for key in axis.subtree_keys("dog"):
        assert axis.structure.contains(key)
        assert axis.subtree_keys(key)[0] == key
        assert axis.at(key).exists()


def test_the_round_trip_is_what_a_localising_parse_would_break(
    mammals_v11_path: Path,
) -> None:
    """The control, without which the guard above is green by saying nothing.

    Over the mammals vocabulary a localising parse changes no key, so that
    guard would pass against a member that localised. This axis is the case
    that separates them: entirely local-keyed, and every key changed by the
    parse. The contract holds, and the assertion at the end is what it would
    cost to break it -- every key handed out would be one the axis refuses.
    """
    onto = load_ontology(mammals_v11_path)
    colonised = Taxonomy(
        definition=onto.taxonomies["species"],
        structure=MappingHierarchy(COLON_KEYED),
        entities=onto.entities,
    )

    keys = colonised.subtree_keys("mammal:carnivore")

    assert keys == ["mammal:carnivore", "mammal:felid"]
    for key in keys:
        assert colonised.structure.contains(key)

    localised = [split_qualified(key).local_id for key in keys]
    assert localised == ["carnivore", "felid"], "the parse changes every one of them"
    assert not any(colonised.structure.contains(key) for key in localised)


# --------------------------------------------------------------------------
# The memo, and the walk that goes through no driver
# --------------------------------------------------------------------------


class CountingStructure:
    """A structure axis that records every ``children`` call.

    Offers no bulk member deliberately: ``children_many`` answers a frontier in
    one call and hides the per-node question these tests ask.
    """

    def __init__(self, parents: dict[str, tuple[str, ...]]) -> None:
        self._inner = MappingHierarchy(parents)
        self.asked: list[str] = []

    def roots(self) -> tuple[str, ...]:
        return tuple(self._inner.roots())

    def parents(self, node_id: str) -> tuple[str, ...]:
        return tuple(self._inner.parents(node_id))

    def children(self, node_id: str) -> tuple[str, ...]:
        self.asked.append(node_id)
        return tuple(self._inner.children(node_id))

    def contains(self, node_id: str) -> bool:
        return self._inner.contains(node_id)


class AsyncCountingStructure:
    """:class:`CountingStructure`, awaited."""

    def __init__(self, parents: dict[str, tuple[str, ...]]) -> None:
        self._inner = CountingStructure(parents)

    @property
    def asked(self) -> list[str]:
        return self._inner.asked

    async def roots(self) -> tuple[str, ...]:
        return self._inner.roots()

    async def parents(self, node_id: str) -> tuple[str, ...]:
        return self._inner.parents(node_id)

    async def children(self, node_id: str) -> tuple[str, ...]:
        return self._inner.children(node_id)

    async def contains(self, node_id: str) -> bool:
        return self._inner.contains(node_id)


#: mammal -> dog -> {retriever, beagle}. Two levels, so the streaming walk
#: issues more than one frontier read and a memo has something to answer.
STRUCTURE = {
    "mammal": (),
    "dog": ("mammal",),
    "retriever": ("dog",),
    "beagle": ("dog",),
}


def _axis_over(structure: object, path: Path) -> Taxonomy:
    onto = load_ontology(path)
    return replace(onto.taxonomy("species"), structure=structure)  # type: ignore[arg-type]


def test_the_streaming_walk_reads_a_caller_supplied_memo(mammals_v11_path: Path) -> None:
    """The memo lives in the frontier read *because* this walk calls it directly.

    ``walk`` cannot go through the collecting core, so a memo written into the
    drivers would have left it as the only walk without one -- which is the
    stated reason the memo is in the shared step rather than in either driver.
    A reason is only load-bearing if the walk it names can actually be given a
    cache, so this is what makes the claim true rather than merely available.
    """
    structure = CountingStructure(STRUCTURE)
    axis = _axis_over(structure, mammals_v11_path)
    warm: dict[tuple[str, object], object] = {}

    first = tuple(axis.walk())
    asked_once = len(structure.asked)
    assert asked_once > 0

    structure.asked.clear()
    second = tuple(axis.walk(cache=warm))
    third = tuple(axis.walk(cache=warm))

    assert first == second == third
    assert len(structure.asked) == asked_once, (
        f"the warmed walk re-asked the backing: {structure.asked}"
    )


@pytest.mark.asyncio
async def test_the_async_streaming_walk_reads_one_too(mammals_v11_path: Path) -> None:
    """The memo is in the step both flavours of the streaming walk call."""
    structure = AsyncCountingStructure(STRUCTURE)
    onto = await async_load_ontology(mammals_v11_path)
    axis = replace(onto.taxonomy("species"), structure=structure)  # type: ignore[arg-type]
    warm: dict[tuple[str, object], object] = {}

    first = [node async for node in axis.walk(cache=warm)]
    asked_once = len(structure.asked)

    structure.asked.clear()
    second = [node async for node in axis.walk(cache=warm)]

    assert first == second
    assert asked_once > 0
    assert structure.asked == [], "the warmed walk re-asked the backing"


def test_a_memo_warmed_by_a_collecting_walk_answers_the_streaming_one(
    mammals_v11_path: Path,
) -> None:
    """One memo, one axis, both kinds of walk -- which is what sharing the step buys.

    The collecting walks and the streaming one issue the same request against
    the same step, so a cache filled by either is readable by the other. A memo
    written into the drivers could not have done this.
    """
    structure = CountingStructure(STRUCTURE)
    axis = _axis_over(structure, mammals_v11_path)
    shared: dict[tuple[str, object], object] = {}

    flatten(structure, cache=shared)  # type: ignore[arg-type]
    warmed = len(structure.asked)

    walked = tuple(axis.walk(cache=shared))

    assert walked == ("mammal", "dog", "retriever", "beagle")
    assert len(structure.asked) == warmed, "the streaming walk re-asked the backing"


def test_subtree_keys_forwards_a_memo_to_the_walks_it_delegates_to(
    mammals_v11_path: Path,
) -> None:
    """The two delegations carry the caller's cache, or the delegation is partial.

    ``subtree_keys`` *is* ``flatten`` unbounded and ``descendants_to_depth``
    bounded, and both of those take a memo. A member that delegates to a walk
    and drops the one parameter that walk grew is the same drift one layer up:
    the caller holding a long-lived cache pays the backing again on every call.
    """
    structure = CountingStructure(STRUCTURE)
    axis = _axis_over(structure, mammals_v11_path)
    shared: dict[tuple[str, object], object] = {}

    first = axis.subtree_keys("dog", cache=shared)
    asked_once = len(structure.asked)
    second = axis.subtree_keys("dog", cache=shared)
    bounded = axis.subtree_keys("dog", depth=1, cache=shared)

    assert first == second == ["dog", "retriever", "beagle"]
    assert bounded == ["dog", "retriever", "beagle"]
    assert len(structure.asked) == asked_once, (
        f"a delegated walk re-asked the backing: {structure.asked}"
    )


@pytest.mark.asyncio
async def test_the_async_subtree_keys_forwards_one_too(mammals_v11_path: Path) -> None:
    """The twin, for the reason every twin here is checked: a parameter on one half."""
    structure = AsyncCountingStructure(STRUCTURE)
    onto = await async_load_ontology(mammals_v11_path)
    axis = replace(onto.taxonomy("species"), structure=structure)  # type: ignore[arg-type]
    shared: dict[tuple[str, object], object] = {}

    first = await axis.subtree_keys("dog", cache=shared)
    asked_once = len(structure.asked)
    second = await axis.subtree_keys("dog", cache=shared)

    assert first == second == ["dog", "retriever", "beagle"]
    assert len(structure.asked) == asked_once


# --------------------------------------------------------------------------
# inherited_attributes -- the OTHER lattice, and the store that answers for it
# --------------------------------------------------------------------------


def test_the_two_lattices_answer_different_questions_about_one_node(
    mammals_v11_path: Path,
) -> None:
    """The criterion this member exists for, asserted rather than described.

    ``beagle`` sits on both lattices and they are different stores. The
    **instance** lattice -- ``isa`` assertions between entities -- puts ``dog``
    and ``mammal`` above it; the **type** lattice -- the ``isa:`` field on an
    entity type declaration -- puts ``Species`` above ``Breed``. Walking one
    and reading the other are not the same operation, and a reader who reaches
    for ``inherited_attributes`` expecting ancestors gets declarations.

    Both in one test, because a paragraph saying so is what this replaces.
    """
    onto = load_ontology(mammals_v11_path)
    axis = onto.taxonomy("species")

    beagle = axis.at("beagle")
    entity = beagle.entity()
    assert entity is not None

    ancestors = [above.node for above in beagle.ancestors()]
    inherited = [attribute.name for attribute in axis.inherited_attributes(entity.type)]

    assert ancestors == ["dog", "mammal"], "the instance lattice, as entity ids"
    assert inherited == ["akc_group", "latin_name", "lifespan_years"], (
        "the type lattice, as attribute declarations -- Breed's own first, then Species'"
    )
    assert set(ancestors).isdisjoint(inherited), "and they name different things entirely"


def test_the_nearer_declaration_shadows_the_farther_one(mammals_v11_path: Path) -> None:
    """A subtype redeclaring an attribute is specialising it, not adding a second.

    Returning both would build an extraction schema with two fields of one
    name, and the caller has no rule for choosing between them. The nearer one
    wins because it is the more specific statement about the type that was
    asked about.
    """
    onto = load_ontology(
        {
            "ontology": {
                "id": "shadowed",
                "version": "1.0",
                "entity_types": [
                    {
                        "id": "Item",
                        "attributes": [
                            {"name": "sku", "type": "string", "description": "the general one"},
                            {"name": "weight", "type": "number"},
                        ],
                    },
                    {
                        "id": "Product",
                        "isa": "Item",
                        "attributes": [
                            {
                                "name": "sku",
                                "type": "string",
                                "required": True,
                                "description": "the specific one",
                            }
                        ],
                    },
                ],
            }
        }
    )
    axis = Taxonomy(
        definition=TaxonomyDefinition(id="species", relation="isa"),
        structure=MappingHierarchy({}),
        entities=onto.entities,
        entity_types=onto.entity_types,
    )

    inherited = axis.inherited_attributes("Product")

    assert [attribute.name for attribute in inherited] == ["sku", "weight"], "one sku, not two"
    assert inherited[0].description == "the specific one"
    assert inherited[0].required is True


def test_a_cyclic_type_lattice_terminates(mammals_v11_path: Path) -> None:
    """The loader does not refuse one, so the walk has to survive it.

    Measured rather than assumed: ``_refuse_undeclared_isa`` checks that a
    declared parent *exists* and nothing checks that the chain *ends*, so a
    document naming ``A isa B`` and ``B isa A`` loads without complaint. The
    visited set here is unconditional for the same reason every walk over the
    structure axis carries one.
    """
    onto = load_ontology(
        {
            "ontology": {
                "id": "cyclic",
                "version": "1.0",
                "entity_types": [
                    {"id": "A", "isa": "B", "attributes": [{"name": "a", "type": "string"}]},
                    {"id": "B", "isa": "A", "attributes": [{"name": "b", "type": "string"}]},
                ],
            }
        }
    )
    axis = Taxonomy(
        definition=TaxonomyDefinition(id="species", relation="isa"),
        structure=MappingHierarchy({}),
        entities=onto.entities,
        entity_types=onto.entity_types,
    )

    assert [attribute.name for attribute in axis.inherited_attributes("A")] == ["a", "b"]
    assert [attribute.name for attribute in axis.inherited_attributes("B")] == ["b", "a"]


def test_a_type_the_store_does_not_declare_is_refused(mammals_v11_path: Path) -> None:
    """Refused, not answered with ``[]``, because the two are different states.

    A type declared with no attributes and no parent legitimately inherits
    nothing; a type the store has never heard of is a caller error, and an
    empty list would report the second as the first. It is also what makes the
    field safe to default: an axis built without types answers nothing at all
    rather than answering *nothing is declared*.
    """
    onto = load_ontology(mammals_v11_path)
    axis = onto.taxonomy("species")

    with pytest.raises(NotFoundError) as refusal:
        axis.inherited_attributes("NoSuchType")

    assert refusal.value.context["entity_type"] == "NoSuchType"
    assert refusal.value.context["taxonomy"] == "species"

    typeless = replace(axis, entity_types={})
    with pytest.raises(NotFoundError):
        typeless.inherited_attributes("Breed")


def test_the_vocabulary_answers_the_lattice_question_without_building_an_axis(
    mammals_v11_path: Path,
) -> None:
    """The store is the ontology's, so the ontology is a place to ask about it.

    ``inherited_attributes`` reads ``entity_types`` and nothing else -- which is
    why every taxonomy of one vocabulary answers it identically, and why
    reaching it only through :meth:`Ontology.taxonomy` meant building an axis to
    ask a question the axis has no part in. Both surfaces are one line over the
    same walk, so the two answers are the same object-for-object.
    """
    onto = load_ontology(mammals_v11_path)

    direct = onto.inherited_attributes("Breed")
    via_axis = onto.taxonomy("species").inherited_attributes("Breed")

    assert [a.name for a in direct] == ["akc_group", "latin_name", "lifespan_years"]
    assert direct == via_axis

    # And the refusal travels with it rather than being re-argued per surface.
    with pytest.raises(NotFoundError) as refusal:
        onto.inherited_attributes("NoSuchType")
    assert refusal.value.context["entity_type"] == "NoSuchType"


@pytest.mark.asyncio
async def test_the_async_vocabulary_answers_it_without_awaiting(mammals_v11_path: Path) -> None:
    """A plain ``def`` on this twin too, for the mapping's reason."""
    onto = await async_load_ontology(mammals_v11_path)

    assert not inspect.iscoroutinefunction(AsyncOntology.inherited_attributes)
    assert [a.name for a in onto.inherited_attributes("Breed")] == [
        "akc_group",
        "latin_name",
        "lifespan_years",
    ]


def test_an_ancestor_the_store_does_not_declare_is_refused_too() -> None:
    """The same rule as the anchor's, applied to the ancestor the walk reaches for.

    ``_refuse_an_undeclared_type`` argues that answering ``[]`` for a type the
    store never heard of collapses *this type declares nothing* into *this type
    is not here*, and that the reading a caller reaches for is the one that is
    not their fault. Two lines into the walk the same collapse was performed on
    the anchor's **parent**, silently: a ``Breed`` whose declared ``Species`` is
    absent answered with Breed's own attributes and said nothing, so a caller
    could not tell a complete answer from a truncated one.

    It is reachable without a malformed document. ``entity_types`` is a mapping
    the caller supplies and it defaults to empty -- a partial store is invited
    by the field rather than guarded against -- and :class:`EntityType` is a
    public dataclass anyone may construct.
    """
    types = {
        "Breed": EntityType(
            id="Breed",
            name="Breed",
            isa="Species",
            attributes=[AttributeDef(name="sku", value_type="string")],
        ),
        # `Species` deliberately absent: the store is partial, not malformed.
    }
    axis = Taxonomy(
        definition=TaxonomyDefinition(id="species", relation="isa"),
        structure=MappingHierarchy({}),
        entities=MappingEntitySource({}),
        entity_types=types,
    )

    with pytest.raises(NotFoundError) as refusal:
        axis.inherited_attributes("Breed")

    # The refusal names both ends: what was asked about, and what is missing.
    assert refusal.value.context["entity_type"] == "Species"
    assert refusal.value.context["asked_about"] == "Breed"
    assert refusal.value.context["taxonomy"] == "species"


def test_an_isa_is_a_declared_field_rather_than_an_open_metadata_key() -> None:
    """A parent the loader cannot see is a parent the loader cannot check.

    ``isa`` was parked in :attr:`EntityType.metadata` under a documented key
    while nothing read it. Once a walk reads it back, the open dict is a hole:
    ``_type_metadata`` copied a row's ``metadata`` **wholesale** before folding
    the top-level ``isa:`` in, so a document writing the parent one level down
    reached the lattice without passing ``_refuse_undeclared_isa`` -- which
    reads only ``row.get("isa")``.

    A declared field closes it at the source rather than adding a second
    check: there is no longer a second place a parent can be written.
    """
    onto = load_ontology(
        {
            "ontology": {
                "id": "smuggled",
                "version": "1.0",
                "entity_types": [
                    {
                        "id": "Breed",
                        "metadata": {"isa": "NotDeclaredAnywhere"},
                        "attributes": [{"name": "sku", "type": "string"}],
                    }
                ],
            }
        }
    )

    # The key in `metadata` is now inert -- it is not where a parent is kept.
    assert onto.entity_types["Breed"].isa is None
    assert onto.entity_types["Breed"].metadata == {"isa": "NotDeclaredAnywhere"}

    axis = Taxonomy(
        definition=TaxonomyDefinition(id="species", relation="isa"),
        structure=MappingHierarchy({}),
        entities=onto.entities,
        entity_types=onto.entity_types,
    )
    assert [a.name for a in axis.inherited_attributes("Breed")] == ["sku"]


def test_the_loader_refuses_an_undeclared_parent_however_it_is_written() -> None:
    """And the top-level spelling is still refused, which is the half that shipped."""
    with pytest.raises(ValidationError) as refusal:
        load_ontology(
            {
                "ontology": {
                    "id": "dangling",
                    "version": "1.0",
                    "entity_types": [{"id": "Breed", "isa": "NotDeclaredAnywhere"}],
                }
            }
        )

    assert "NotDeclaredAnywhere" in str(refusal.value)


def test_a_type_declaring_nothing_inherits_nothing_and_says_so(mammals_v11_path: Path) -> None:
    """The other half of the pair above: declared, with nothing to give."""
    onto = load_ontology(
        {
            "ontology": {
                "id": "bare",
                "version": "1.0",
                "entity_types": [{"id": "Thing"}],
            }
        }
    )
    axis = Taxonomy(
        definition=TaxonomyDefinition(id="species", relation="isa"),
        structure=MappingHierarchy({}),
        entities=onto.entities,
        entity_types=onto.entity_types,
    )

    assert axis.inherited_attributes("Thing") == []


def test_the_accessor_hands_the_axis_the_type_store(mammals_v11_path: Path) -> None:
    """A field the accessor forgets to pass is an axis that refuses everything.

    The field defaults to empty, which is what makes a hand-built axis cheap
    and what makes this assertion necessary: nothing else in the suite would
    notice ``Ontology.taxonomy`` dropping it, because every other member
    answers the same either way.
    """
    onto = load_ontology(mammals_v11_path)

    assert onto.taxonomy("species").entity_types == onto.entity_types
    assert onto.taxonomy("species").inherited_attributes("Breed")


@pytest.mark.asyncio
async def test_the_async_twin_answers_without_awaiting(mammals_v11_path: Path) -> None:
    """A plain ``def`` on both flavours, because a mapping awaits nothing.

    The same rule that makes ``at()`` synchronous on the asynchronous twin:
    making this awaitable would cost every caller an ``await`` for a walk over
    a mapping they are already holding.
    """
    onto = await async_load_ontology(mammals_v11_path)
    axis = onto.taxonomy("species")

    assert not inspect.iscoroutinefunction(AsyncTaxonomy.inherited_attributes)
    assert [a.name for a in axis.inherited_attributes("Breed")] == [
        "akc_group",
        "latin_name",
        "lifespan_years",
    ]
