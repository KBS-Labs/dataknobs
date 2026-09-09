"""The cursor over an axis: the door in, the forwards, and the edge it walked.

Four claims:

* **the door** -- ``Taxonomy.at()`` returns the cursor without checking
  containment, on both flavours, and is a plain ``def`` on both;
* **the forwards** -- every structural member invokes the ``HierarchyView``
  member of the same name, proven by patching that member and watching this one
  move (the guard for the shape a maintainer reimplements without noticing the
  forward is there);
* **the edge members** -- keyed per assertion, ``()`` from three states, and
  never a stated negation;
* **one narrowing** -- the criteria that make an axis *asserted edges only*
  have one home, and patching it moves both hierarchy twins and both cursors.
"""

from __future__ import annotations

import inspect
from collections.abc import Hashable
from dataclasses import FrozenInstanceError, replace
from typing import TYPE_CHECKING

import pytest

from dataknobs_common.exceptions import NotFoundError
from dataknobs_common.hierarchy import AsyncHierarchyView, HierarchyView, MappingHierarchy
from dataknobs_common.ontology import async_load_ontology, load_ontology
from dataknobs_common.ontology import hierarchy as axis_module
from dataknobs_common.ontology import taxonomy as taxonomy_module
from dataknobs_common.ontology.hierarchy import (
    AssertionHierarchy,
    AsyncAssertionHierarchy,
    edge_criteria,
)
from dataknobs_common.ontology.model import EntityRef, Polarity
from dataknobs_common.ontology.taxonomy import (
    AsyncTaxonomy,
    AsyncTaxonomyView,
    Taxonomy,
    TaxonomyView,
)
from dataknobs_common.testing import assert_twin_types_agree

if TYPE_CHECKING:
    from pathlib import Path

    from dataknobs_common.ontology import Ontology


@pytest.fixture
def onto(mammals_v11_path: Path) -> Ontology:
    """The worked vocabulary at 1.1, which declares the ``species`` axis."""
    return load_ontology(mammals_v11_path)


@pytest.fixture
def axis(onto: Ontology) -> Taxonomy:
    """``mammal`` above ``dog`` above ``retriever`` and ``beagle``; one assertion per edge."""
    return onto.taxonomy("species")


#: One edge annotated twice, with distinct ids and different metadata.
#:
#: The fixture that fails a cursor keyed per *parent*: a collapse to one pair
#: per neighbour returns one pair here, and the test wants two.
TWICE_ANNOTATED = {
    "id": "twice",
    "assertions": [
        {"subject": "dog", "relation": "isa", "object": "mammal"},
        {
            "id": "by-the-kennel-club",
            "subject": "beagle",
            "relation": "isa",
            "object": "dog",
            "metadata": {"source": "akc"},
        },
        {
            "id": "by-the-federation",
            "subject": "beagle",
            "relation": "isa",
            "object": "dog",
            "metadata": {"source": "fci"},
        },
    ],
    "taxonomies": [{"id": "species", "relation": "isa"}],
}

#: One edge stated in both polarities -- contradictory, and statable.
#:
#: The structure excludes the negation, so ``parents()`` still reports ``dog``;
#: what the edge read returns for that parent is the whole question.
BOTH_POLARITIES = {
    "id": "contradiction",
    "assertions": [
        {"subject": "dog", "relation": "isa", "object": "mammal"},
        {"id": "yes", "subject": "beagle", "relation": "isa", "object": "dog"},
        {
            "id": "no",
            "subject": "beagle",
            "relation": "isa",
            "object": "dog",
            "polarity": "negated",
        },
    ],
    "taxonomies": [{"id": "species", "relation": "isa"}],
}


# --------------------------------------------------------------------------
# Criterion 20 -- the door, and absence is not childlessness
# --------------------------------------------------------------------------


def test_at_on_an_unknown_id_returns_a_view_that_is_neither_root_nor_leaf(axis: Taxonomy) -> None:
    """The door does not refuse; the view it returns says the node is not here.

    ``walk(from_id=...)`` refuses the same id, and the difference is not
    inconsistency: a walk *includes* its anchor and would emit it as a term of
    the axis, where a cursor answers ``exists()`` for itself.
    """
    there = axis.at("no_such_node")

    assert there.exists() is False
    assert there.is_leaf() is False
    assert there.is_root() is False
    assert there.children() == ()

    assert axis.at("beagle").is_leaf() is True
    assert axis.at("mammal").is_root() is True


def test_the_door_is_pure_over_the_axis_and_a_plain_def_on_both_twins(axis: Taxonomy) -> None:
    """It takes the node and nothing else, and awaits nothing on either flavour.

    Checking containment on the asynchronous twin would need an ``await`` for
    a question the caller has not asked yet; the check lives on the view.
    """
    view = axis.at("dog")
    assert isinstance(view, TaxonomyView)
    assert view.taxonomy is axis
    assert view.node == "dog"

    assert not inspect.iscoroutinefunction(AsyncTaxonomy.at)
    assert inspect.signature(Taxonomy.at).parameters.keys() == {"self", "node_id"}
    assert inspect.signature(AsyncTaxonomy.at).parameters.keys() == {"self", "node_id"}


def test_the_structural_members_answer_over_the_declared_axis(axis: Taxonomy) -> None:
    here = axis.at("dog")

    assert [above.node for above in here.parents()] == ["mammal"]
    assert [below.node for below in here.children()] == ["retriever", "beagle"]
    assert all(isinstance(below, TaxonomyView) for below in here.children())
    assert here.at("beagle").parents()[0] == axis.at("dog")


# --------------------------------------------------------------------------
# A cursor is a value, and a value hashes
# --------------------------------------------------------------------------


def test_the_cursor_hashes_so_a_walk_can_key_a_seen_set_on_it(axis: Taxonomy) -> None:
    """``hash()`` answers rather than raising, which ``Hashable`` already promised.

    A frozen dataclass gets a generated ``__hash__``, so ``isinstance(view,
    Hashable)`` was ``True`` before this was true -- the type passed the check
    and raised at the call, because the generated hash hashes the field tuple
    and :class:`Taxonomy` was not frozen. That is the shape worth a test: not
    an absent capability, which a caller discovers by reading, but a promised
    one that fails only when exercised.

    The cost is paid by whoever writes the walk the guide defers -- every walk
    in this module carries ``seen: set[...]``, and keying it on the cursor
    rather than the bare id is the obvious reading of a type sold as a value.
    """
    here = axis.at("dog")

    assert isinstance(here, Hashable)
    assert hash(here) == hash(axis.at("dog"))
    assert len({axis.at("dog"), axis.at("dog"), axis.at("beagle")}) == 2

    seen: set[TaxonomyView] = set()
    frontier = [axis.at("mammal")]
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        frontier.extend(node.children())
    assert {view.node for view in seen} == {
        "mammal",
        "dog",
        "retriever",
        "golden_retriever",
        "beagle",
    }


def test_the_axis_it_holds_is_frozen_and_compared_by_identity(axis: Taxonomy) -> None:
    """Both halves of what makes the cursor hashable, asserted where they live.

    Frozen is the half a reader expects. Identity is the half that does the
    work: field-wise equality would generate a ``__hash__`` reaching
    ``definition.metadata``, and a dict does not hash however frozen its
    owner is. Two separately built axes over one ontology are therefore not
    equal, which is the price and is asserted here rather than left implicit.
    """
    with pytest.raises(FrozenInstanceError):
        axis.structure = MappingHierarchy({})  # type: ignore[misc]

    assert replace(axis, assertions=None).assertions is None
    assert axis == axis
    assert axis != replace(axis)

# --------------------------------------------------------------------------
# Criterion 17 -- the forwards, one patch per flavour
# --------------------------------------------------------------------------

_BOOL_FORWARDS = ("exists", "is_root", "is_leaf")
_VIEW_FORWARDS = ("parents", "children")


@pytest.mark.parametrize("member", _BOOL_FORWARDS)
def test_patching_a_hierarchy_view_question_moves_the_taxonomy_view(
    monkeypatch: pytest.MonkeyPatch, axis: Taxonomy, member: str
) -> None:
    """``TaxonomyView.<m>()`` *invokes* ``HierarchyView.<m>`` rather than agreeing with it.

    Agreement is what the shorter alternative -- one protocol call, correct
    today -- also gives, and it drifts the first time either side is edited.
    """
    sentinel = object()
    monkeypatch.setattr(HierarchyView, member, lambda self: sentinel)

    assert getattr(axis.at("dog"), member)() is sentinel


@pytest.mark.parametrize("member", _VIEW_FORWARDS)
def test_patching_a_hierarchy_view_step_moves_the_taxonomy_view(
    monkeypatch: pytest.MonkeyPatch, axis: Taxonomy, member: str
) -> None:
    """The step members re-wrap what the hierarchy cursor returns.

    ``beagle``'s real parent is ``dog`` and it has no children, so an answer of
    ``mammal`` from either member can only have come through the patch.
    """
    monkeypatch.setattr(
        HierarchyView, member, lambda self: (HierarchyView(self.structure, "mammal"),)
    )

    moved = getattr(axis.at("beagle"), member)()

    assert [view.node for view in moved] == ["mammal"]
    assert all(isinstance(view, TaxonomyView) for view in moved)


def test_the_edge_members_forward_within_the_class(axis: Taxonomy) -> None:
    """``parent_edges`` invokes ``parents()`` and ``child_edges`` invokes ``children()``.

    The structural half of an edge member is a forward *within* this class,
    so the patch target is ``TaxonomyView.parents`` rather than anything on
    ``HierarchyView``. Unpatched, ``beagle`` has one edge up and ``dog`` two
    down; with the step patched to report nothing, both read nothing.
    """
    assert len(axis.at("beagle").parent_edges()) == 1
    assert len(axis.at("dog").child_edges()) == 2

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(TaxonomyView, "parents", lambda self: ())
        patch.setattr(TaxonomyView, "children", lambda self: ())

        assert axis.at("beagle").parent_edges() == ()
        assert axis.at("dog").child_edges() == ()


@pytest.mark.asyncio
async def test_patching_the_async_hierarchy_view_moves_the_async_taxonomy_view(
    monkeypatch: pytest.MonkeyPatch, mammals_v11_path: Path
) -> None:
    """The same guard over the other flavour: the protocol beneath is twinned,
    so no single patch can move both, and each flavour gets its own.
    """
    onto = await async_load_ontology(mammals_v11_path)
    axis = onto.taxonomy("species")
    sentinel = object()

    async def patched_exists(self: AsyncHierarchyView[str]) -> object:
        return sentinel

    async def patched_parents(
        self: AsyncHierarchyView[str],
    ) -> tuple[AsyncHierarchyView[str], ...]:
        return (AsyncHierarchyView(self.structure, "mammal"),)

    monkeypatch.setattr(AsyncHierarchyView, "exists", patched_exists)
    monkeypatch.setattr(AsyncHierarchyView, "parents", patched_parents)

    assert await axis.at("dog").exists() is sentinel
    moved = await axis.at("beagle").parents()
    assert [view.node for view in moved] == ["mammal"]
    assert all(isinstance(view, AsyncTaxonomyView) for view in moved)


# --------------------------------------------------------------------------
# Criteria 19 and 16 -- the edge the cursor walked, keyed per assertion
# --------------------------------------------------------------------------


def test_parent_edges_reports_the_assertion_that_placed_the_node(axis: Taxonomy) -> None:
    """One pair, because the file writes one assertion on that edge.

    The pair rather than the assertion alone: a bare ``Assertion`` loses the
    position to keep walking from, and every other member here composes.
    """
    pairs = axis.at("beagle").parent_edges()

    assert len(pairs) == 1
    neighbour, placed_by = pairs[0]
    assert neighbour == axis.at("dog")
    assert (placed_by.subject, placed_by.relation, placed_by.object) == (
        "beagle",
        "isa",
        EntityRef("dog"),
    )


def test_an_axis_with_no_assertions_answers_empty_rather_than_raising(axis: Taxonomy) -> None:
    """A hierarchy built from a ``parent_id`` column has rows and no assertions.

    The same call returns ``()`` there -- and ``taxonomy.assertions is None``
    is what distinguishes *this axis has no annotations to give* from a root.
    """
    unannotated = replace(axis, assertions=None)

    assert unannotated.at("beagle").parent_edges() == ()
    assert unannotated.at("dog").child_edges() == ()
    assert unannotated.assertions is None
    assert unannotated.at("beagle").parents()[0].node == "dog"


def test_the_edge_members_are_keyed_per_assertion_not_per_parent() -> None:
    """One parent annotated twice gives two pairs with the same neighbour.

    A collapse to one pair per parent returns one here and passes criterion 19
    quietly; this is the fixture that fails it.
    """
    axis = load_ontology(TWICE_ANNOTATED).taxonomy("species")

    up = axis.at("beagle").parent_edges()
    assert [neighbour.node for neighbour, _ in up] == ["dog", "dog"]
    assert [placed_by.id for _, placed_by in up] == ["by-the-kennel-club", "by-the-federation"]
    assert [placed_by.metadata["source"] for _, placed_by in up] == ["akc", "fci"]

    down = axis.at("dog").child_edges()
    assert [neighbour.node for neighbour, _ in down] == ["beagle", "beagle"]


def test_empty_from_three_states_and_the_question_that_separates_them() -> None:
    """A root, an absent node, and an axis with no assertions all answer ``()``.

    Three states, one value, and the value is honest: it says *nothing is
    written on any edge here*. The three are told apart by other members --
    ``exists()`` for the second, ``taxonomy.assertions is None`` for the
    third -- rather than by a wider return type that would make iterating the
    result unsafe.
    """
    axis = load_ontology(TWICE_ANNOTATED).taxonomy("species")

    root = axis.at("mammal")
    assert root.parent_edges() == ()
    assert root.is_root() is True

    absent = axis.at("no_such_node")
    assert absent.parent_edges() == ()
    assert absent.exists() is False

    unannotated = replace(axis, assertions=None).at("beagle")
    assert unannotated.parent_edges() == ()
    assert unannotated.taxonomy.assertions is None
    assert unannotated.exists() is True


def test_a_literal_object_of_the_same_relation_is_not_an_edge() -> None:
    """A value on the relation names no neighbour, so it contributes no pair."""
    axis = load_ontology(
        {
            "id": "mixed",
            "assertions": [
                {"subject": "beagle", "relation": "isa", "object": "dog"},
                {"subject": "beagle", "relation": "isa", "object": {"value": 1, "type": "integer"}},
            ],
            "taxonomies": [{"id": "species", "relation": "isa"}],
        }
    ).taxonomy("species")

    assert [neighbour.node for neighbour, _ in axis.at("beagle").parent_edges()] == ["dog"]


# --------------------------------------------------------------------------
# A stated negation is not a pair -- and the narrowing has one home
# --------------------------------------------------------------------------


def test_a_stated_negation_is_not_the_edge_the_cursor_walked() -> None:
    """Both polarities on one edge: the structure walks the asserted one, and
    so does the edge read.

    Without the narrowing this returns two pairs, one of them the negation,
    handed back as the assertion that placed the node. Nothing would raise.
    """
    axis = load_ontology(BOTH_POLARITIES).taxonomy("species")

    up = axis.at("beagle").parent_edges()
    assert [placed_by.id for _, placed_by in up] == ["yes"]
    assert up[0][1].polarity is Polarity.ASSERTED

    down = axis.at("dog").child_edges()
    assert [placed_by.id for _, placed_by in down] == ["yes"]


def test_the_narrowing_holds_when_the_structure_is_a_copy_and_the_assertions_are_live() -> None:
    """The case the hierarchy twins' own helper cannot reach.

    A structure copied at load reports ``dog`` as a parent and filters
    nothing; the assertion source is live and holds both polarities. The
    cursor reads the source directly, so the narrowing has to be its own.
    """
    onto = load_ontology(BOTH_POLARITIES)
    copied = Taxonomy(
        definition=onto.taxonomies["species"],
        structure=MappingHierarchy({"beagle": ("dog",), "dog": ("mammal",), "mammal": ()}),
        entities=onto.entities,
        assertions=onto.assertions,
    )

    up = copied.at("beagle").parent_edges()
    assert [placed_by.id for _, placed_by in up] == ["yes"]


def test_edge_criteria_is_this_relation_asserted() -> None:
    """Criteria rather than a read, so it has no flavour and unpacks into either."""
    criteria = edge_criteria("isa")

    assert criteria == {"relation": "isa", "polarity": Polarity.ASSERTED}
    onto = load_ontology(BOTH_POLARITIES)
    assert [a.id for a in onto.assertions.find(subject="beagle", **criteria)] == ["yes"]


def test_patching_the_criteria_moves_both_synchronous_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One decision, one home: replace it and every reader of an axis flips.

    Patched to select the *negated* edge instead, the hierarchy twin stops
    seeing ``dog isa mammal`` and starts seeing ``beagle isa dog`` through the
    negation, and the cursor hands back the negated assertion as the edge --
    which is only possible if both unpack the function rather than restating
    its constant. Patched at both bindings, because the cursor's module
    imports the name.
    """
    onto = load_ontology(BOTH_POLARITIES)
    twin = AssertionHierarchy(onto.assertions, "isa")
    assert twin.parents("dog") == ("mammal",)
    before = onto.taxonomy("species").at("beagle").parent_edges()
    assert [placed_by.id for _, placed_by in before] == ["yes"]

    def negations_only(relation: str) -> dict[str, object]:
        return {"relation": relation, "polarity": Polarity.NEGATED}

    monkeypatch.setattr(axis_module, "edge_criteria", negations_only)
    monkeypatch.setattr(taxonomy_module, "edge_criteria", negations_only)

    assert twin.parents("dog") == ()
    assert twin.parents("beagle") == ("dog",)
    after = onto.taxonomy("species").at("beagle").parent_edges()
    assert [placed_by.id for _, placed_by in after] == ["no"]


@pytest.mark.asyncio
async def test_patching_the_criteria_moves_both_asynchronous_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same, awaited: the async twin and the async cursor share the home too."""
    onto = await async_load_ontology(BOTH_POLARITIES)
    before = await onto.taxonomy("species").at("beagle").parent_edges()
    assert [placed_by.id for _, placed_by in before] == ["yes"]

    def negations_only(relation: str) -> dict[str, object]:
        return {"relation": relation, "polarity": Polarity.NEGATED}

    monkeypatch.setattr(axis_module, "edge_criteria", negations_only)
    monkeypatch.setattr(taxonomy_module, "edge_criteria", negations_only)

    twin = AsyncAssertionHierarchy(onto.assertions, "isa")
    assert await twin.contains("mammal") is False  # only the asserted edge named it
    after = await onto.taxonomy("species").at("beagle").parent_edges()
    assert [placed_by.id for _, placed_by in after] == ["no"]


# --------------------------------------------------------------------------
# The twin, and parity
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_async_cursor_answers_the_same(mammals_v11_path: Path) -> None:
    onto = await async_load_ontology(mammals_v11_path)
    here = onto.taxonomy("species").at("dog")

    assert await here.exists() is True
    assert await here.is_root() is False
    assert [above.node for above in await here.parents()] == ["mammal"]
    assert [below.node for below in await here.children()] == ["retriever", "beagle"]

    up = await here.at("beagle").parent_edges()
    assert [(neighbour.node, placed_by.subject) for neighbour, placed_by in up] == [
        ("dog", "beagle")
    ]
    down = await here.child_edges()
    assert [neighbour.node for neighbour, _ in down] == ["retriever", "beagle"]

    absent = here.at("no_such_node")
    assert await absent.is_leaf() is False
    assert await absent.parent_edges() == ()


def test_the_cursor_twins_expose_the_same_annotated_surface() -> None:
    """Return compared where it is the same type; the view-returning members
    each land in their own flavour's slot, so theirs is not.
    """
    assert_twin_types_agree(
        TaxonomyView, AsyncTaxonomyView, ("exists", "is_root", "is_leaf"), compare_return=True
    )
    assert_twin_types_agree(
        TaxonomyView, AsyncTaxonomyView, ("parents", "children", "parent_edges", "child_edges")
    )


def test_at_is_a_plain_def_with_one_signature_on_every_twin() -> None:
    """Four ``at()``s, all ``def``: the twin guard refuses a synchronous async-half,
    so the signatures are compared directly, as A2's nested constructors are.
    """
    for sync_owner, async_owner in ((TaxonomyView, AsyncTaxonomyView), (Taxonomy, AsyncTaxonomy)):
        assert not inspect.iscoroutinefunction(async_owner.at)
        sync_at = inspect.signature(sync_owner.at)
        async_at = inspect.signature(async_owner.at)
        assert list(sync_at.parameters) == list(async_at.parameters) == ["self", "node_id"]
        assert sync_at.parameters["node_id"].annotation == async_at.parameters["node_id"].annotation


def test_the_walk_still_refuses_what_the_door_admits(axis: Taxonomy) -> None:
    """Two members, opposite answers to one id, and both are right.

    A walk includes its anchor, so an unknown one would be emitted as a term
    of the axis; a cursor answers ``exists()`` for itself and needs no refusal.
    """
    assert axis.at("no_such_node").exists() is False
    with pytest.raises(NotFoundError):
        list(axis.walk(from_id="no_such_node"))
