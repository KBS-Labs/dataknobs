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
from dataknobs_common.hierarchy import AsyncMappingHierarchy, MappingHierarchy
from dataknobs_common.ontology import async_load_ontology, load_ontology
from dataknobs_common.ontology.hierarchy import (
    AssertionHierarchy,
    AsyncAssertionHierarchy,
)
from dataknobs_common.ontology.model import (
    InferenceMode,
    Materialization,
    TaxonomyDefinition,
)
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

    No ``compare_return``: this pair is the one that streams, so its return
    annotations differ by flavour (``Iterator[str]`` against
    ``AsyncIterator[str]``) and asserting on them would pin the flavour rather
    than the contract. The hierarchy members, which do not stream, are checked
    the other way.
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
