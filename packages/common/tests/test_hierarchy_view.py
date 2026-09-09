"""The anchored view: one hierarchy, one node, and six questions asked from there.

Three claims, and the second is the one worth a file of its own:

* **the answers** -- each member reports what the protocol member beneath it
  reports, re-wrapped as views;
* **absence is not childlessness** -- a node the structure does not contain is
  neither a root nor a leaf, which is the guard that stops an empty
  ``children()`` from reading as *fully specified*;
* **parity** -- the twins expose one surface, with ``at()`` a plain ``def`` on
  both.

The taxonomy cursor that forwards to these members, and the door that returns
it, are tested beside the taxonomy.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING

import pytest

from dataknobs_common.hierarchy import (
    AsyncHierarchyView,
    AsyncMappingHierarchy,
    HierarchyView,
    MappingHierarchy,
    _MappingBacking,
)
from dataknobs_common.ontology import (
    AsyncMappingAssertionSource,
    MappingAssertionSource,
    RelationType,
    async_load_ontology,
    load_ontology,
)
from dataknobs_common.ontology.hierarchy import AssertionHierarchy, AsyncAssertionHierarchy
from dataknobs_common.ontology.taxonomy import AsyncTaxonomyView, TaxonomyView
from dataknobs_common.testing import assert_twin_types_agree

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def species(mammals_path: Path) -> AssertionHierarchy:
    """The worked file as an ``isa`` axis: ``mammal`` above ``dog`` above ``beagle``."""
    return AssertionHierarchy(load_ontology(mammals_path).assertions, "isa")


# --------------------------------------------------------------------------
# The answers -- over authored assertions, with nothing standing up behind them
# --------------------------------------------------------------------------


def test_the_members_answer_with_no_store_no_embedder_and_no_loop(
    species: AssertionHierarchy,
) -> None:
    """The six members this increment builds, over a file and nothing else.

    The loop assertion is inside the test: a test that merely *is not* async
    proves nothing about the members, since a synchronous call can still be
    made from inside a running loop.
    """
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()

    view = HierarchyView(species, "dog")

    assert view.exists() is True
    assert view.is_root() is False
    assert view.is_leaf() is False
    assert [above.node for above in view.parents()] == ["mammal"]
    assert [below.node for below in view.children()] == ["beagle"]
    assert view.at("beagle").node == "beagle"


def test_parents_are_plural() -> None:
    """A node with two parents gives two views, never a chosen one.

    A single-parent answer anywhere in the view would silently pick one edge
    of a DAG, which is the lie the structure axis exists not to tell.
    """
    view = HierarchyView(MappingHierarchy({"f": ("c", "d"), "c": (), "d": ()}), "f")

    assert [above.node for above in view.parents()] == ["c", "d"]


def test_a_non_string_key_round_trips() -> None:
    """The key parameter reaches the cursor: an integer-keyed axis gives integer views."""
    view = HierarchyView(MappingHierarchy({3: (2,), 2: (1,), 1: ()}), 3)

    assert view.parents()[0].node == 2
    assert view.at(1).is_root() is True


# --------------------------------------------------------------------------
# Absence is not childlessness
# --------------------------------------------------------------------------


def test_an_absent_node_is_neither_a_root_nor_a_leaf(species: AssertionHierarchy) -> None:
    """Re-anchoring at an id the structure does not contain does not raise.

    It returns a view that answers ``exists()`` for itself -- and answers
    ``False`` to both structural questions, because an empty ``children()``
    is what an absent node returns too, and without the guard a consumer would
    read *nothing below this node* as *fully specified*.
    """
    absent = HierarchyView(species, "dog").at("no_such_node")

    assert absent.exists() is False
    assert absent.is_leaf() is False
    assert absent.is_root() is False
    assert absent.children() == ()
    assert absent.parents() == ()


def test_the_guard_separates_absent_from_childless(species: AssertionHierarchy) -> None:
    """The half that proves the guard is a guard and not a member that always says no.

    ``beagle`` and the absent node both have an empty ``children()``; only one
    of them is a leaf. ``mammal`` and the absent node both have an empty
    ``parents()``; only one of them is a root. And ``dog``, present with both,
    is neither -- which is why ``False`` alone cannot mean *absent*, and
    ``exists()`` is the member that does.
    """
    assert HierarchyView(species, "beagle").is_leaf() is True
    assert HierarchyView(species, "dog").is_leaf() is False
    assert HierarchyView(species, "mammal").is_root() is True
    assert HierarchyView(species, "dog").is_root() is False

    assert HierarchyView(species, "beagle").children() == HierarchyView(species, "nope").children()


def test_the_guard_invokes_exists_rather_than_restating_it(
    monkeypatch: pytest.MonkeyPatch, species: AssertionHierarchy
) -> None:
    """One containment check, invoked twice, rather than three copies of it.

    Patching ``exists`` must move both guarded members: a copy of the check
    inside ``is_root`` would answer ``True`` for ``mammal`` here, and drift
    from ``exists`` the first time either was edited alone.
    """
    monkeypatch.setattr(HierarchyView, "exists", lambda self: False)

    assert HierarchyView(species, "mammal").is_root() is False
    assert HierarchyView(species, "beagle").is_leaf() is False


# --------------------------------------------------------------------------
# A cursor, not a node
# --------------------------------------------------------------------------


def test_two_views_are_equal_iff_they_name_the_same_node(species: AssertionHierarchy) -> None:
    """Equality is the pair, and the pair is frozen."""
    assert HierarchyView(species, "dog") == HierarchyView(species, "dog")
    assert HierarchyView(species, "dog") != HierarchyView(species, "beagle")

    with pytest.raises(dataclasses.FrozenInstanceError):
        HierarchyView(species, "dog").node = "beagle"  # type: ignore[misc]


class _LiveParents:
    """A hierarchy over a mapping the test can change after a view is taken.

    A real implementation of the four members rather than a double: the
    property under test is that a view reads *through* to whatever its
    structure currently answers, and only a backing that can change shows it.
    """

    def __init__(self) -> None:
        self.parents_of: dict[str, tuple[str, ...]] = {"dog": ("mammal",), "mammal": ()}

    def roots(self) -> Sequence[str]:
        return tuple(n for n, above in self.parents_of.items() if not above)

    def parents(self, node_id: str) -> Sequence[str]:
        return self.parents_of.get(node_id, ())

    def children(self, node_id: str) -> Sequence[str]:
        return tuple(n for n, above in self.parents_of.items() if node_id in above)

    def contains(self, node_id: str) -> bool:
        return node_id in self.parents_of or any(node_id in a for a in self.parents_of.values())


def test_a_view_reads_the_structure_beneath_it_rather_than_a_copy() -> None:
    """A view taken before the axis changes reads the axis after it.

    That is the reason it holds the hierarchy rather than a snapshot, and the
    reason it can be frozen: there is nothing in it to go stale.
    """
    live = _LiveParents()
    view = HierarchyView(live, "dog")
    assert view.is_leaf() is True

    live.parents_of["beagle"] = ("dog",)

    assert view.is_leaf() is False
    assert [below.node for below in view.children()] == ["beagle"]


# --------------------------------------------------------------------------
# A cursor a walk can key a set on
# --------------------------------------------------------------------------


#: Every backing this package ships, by name, both flavours of each.
#:
#: The parametrisation *is* the test. A frozen cursor gets a generated
#: ``__hash__`` over its field tuple, and the first field is whatever the
#: caller handed it -- so whether a cursor hashes is a property of the
#: **backing**, not of the cursor. A test written against the one backing that
#: happened to hash is the test that was here when two that did not shipped
#: beside it.
_SHIPPED_BACKINGS = (
    "MappingHierarchy",
    "AsyncMappingHierarchy",
    "AssertionHierarchy",
    "AsyncAssertionHierarchy",
)

#: The mapping the guide's own worked example builds.
BILLING = {"late-fees": ("billing",), "refunds": ("billing",), "billing": ()}


def _cursor_over(backing: str, path: Path) -> HierarchyView[str] | AsyncHierarchyView[str]:
    """A cursor over the named backing, anchored somewhere it exists."""
    if backing == "MappingHierarchy":
        return HierarchyView(MappingHierarchy(BILLING), "billing")
    if backing == "AsyncMappingHierarchy":
        return AsyncHierarchyView(AsyncMappingHierarchy(BILLING), "billing")
    edges = load_ontology(path).assertions.find(relation="isa")
    if backing == "AssertionHierarchy":
        return HierarchyView(AssertionHierarchy(MappingAssertionSource(edges), "isa"), "dog")
    return AsyncHierarchyView(
        AsyncAssertionHierarchy(AsyncMappingAssertionSource(edges), "isa"), "dog"
    )


@pytest.mark.parametrize("backing", _SHIPPED_BACKINGS)
def test_a_cursor_hashes_over_every_backing_this_package_ships(
    backing: str, mammals_path: Path
) -> None:
    """``hash()`` answers rather than raising, which ``Hashable`` already promised.

    The shape worth a test is not an absent capability -- a caller discovers
    that by reading -- but a promised one that fails only when exercised:
    ``isinstance(view, Hashable)`` is ``True`` for a frozen cursor whatever it
    holds, so a backing whose own hash reaches a ``dict`` passes the check and
    raises at the call.

    Asserted over every backing rather than one, because the promise is made
    once, in the cursor's own docstring and the guide, for all of them.

    Either spelling of a relation reaches the same axis, which the test below
    pins: :func:`_cursor_over` hands the assertion-backed pair an **id**, and
    the definition would name the same one.
    """
    view = _cursor_over(backing, mammals_path)
    same = view.at(view.node)

    assert isinstance(view, Hashable)
    assert hash(view) == hash(same)
    assert {view, same} == {view}
    assert len({view, view.at("no_such_node")}) == 2


def test_a_relation_given_as_its_definition_names_the_same_axis(mammals_path: Path) -> None:
    """``RelationRef`` is two spellings of one name; the axis keeps the canonical one.

    :func:`relation_id` is where this package decides which spelling it was
    handed, and its own docstring says *every comparison goes through here
    rather than each site deciding what it was handed*. The two members
    ``@dataclass`` generates are comparisons, and they went through nothing:
    an axis named by the definition read identically to one named by the id
    and compared unequal to it, while a ``RelationType`` -- an ``Entity``,
    honestly unhashable -- withheld the hash a frozen field tuple promises.

    Canonicalised at construction, both spellings name one axis. That is the
    rule the other backings already follow, said for a field shape they do not
    have: **compare by the identity of what you hold, and the value of what
    you name.** The mapping twins hold a live mapping and nothing else, so
    identity is the whole of it; this pair holds a handle and a name, so the
    handle is compared by identity and the name by value -- which is only
    coherent once the name has one spelling.
    """
    edges = load_ontology(mammals_path).assertions.find(relation="isa")
    source = MappingAssertionSource(edges)
    isa = RelationType(id="isa")
    assert not isinstance(isa, Hashable)  # honest, while it is held on its own

    by_id = AssertionHierarchy(source, "isa")
    by_definition = AssertionHierarchy(source, isa)

    assert by_definition.relation == "isa"  # the door still takes either
    assert by_id.parents("dog") == by_definition.parents("dog") == ("mammal",)
    assert by_id == by_definition
    assert HierarchyView(by_id, "dog") == HierarchyView(by_definition, "dog")
    assert hash(HierarchyView(by_id, "dog")) == hash(HierarchyView(by_definition, "dog"))


@pytest.mark.asyncio
async def test_the_async_axis_canonicalises_its_relation_too(mammals_path: Path) -> None:
    """The twin, canonicalising on its own account because no base does it for it.

    The mapping side needed ``eq=False`` written on all three of its
    decorators; this pair has no shared body at all, so the same obligation
    arrives from the other direction and is asserted rather than assumed.
    """
    edges = load_ontology(mammals_path).assertions.find(relation="isa")
    source = AsyncMappingAssertionSource(edges)

    by_id = AsyncAssertionHierarchy(source, "isa")
    by_definition = AsyncAssertionHierarchy(source, RelationType(id="isa"))

    assert by_definition.relation == "isa"
    assert await by_id.parents("dog") == await by_definition.parents("dog") == ("mammal",)
    assert by_id == by_definition
    assert AsyncHierarchyView(by_id, "dog") == AsyncHierarchyView(by_definition, "dog")
    assert hash(AsyncHierarchyView(by_id, "dog")) == hash(AsyncHierarchyView(by_definition, "dog"))


@dataclasses.dataclass(frozen=True)
class _Labelled:
    """A frozen dataclass over a ``dict``: the shape that answers and raises.

    One class in both roles, because the point is that the role does not
    matter -- it is a legal key and a legal (if empty) structure, and a cursor
    holding it in either field fails identically.
    """

    name: str
    labels: dict[str, str]

    def roots(self) -> Sequence[str]:
        return ()

    def parents(self, node_id: object) -> Sequence[str]:
        return ()

    def children(self, node_id: object) -> Sequence[str]:
        return ()

    def contains(self, node_id: object) -> bool:
        return False


def test_the_limit_is_both_fields_and_the_key_bound_does_not_catch_it() -> None:
    """A cursor hashes as far as *what it holds* does -- and it holds two things.

    The generated ``__hash__`` is over the **field tuple**, so the structure
    and the key each carry a veto. The key half is the one that reads as
    impossible: ``K`` is bound to ``Hashable``, and a frozen dataclass over a
    ``dict`` satisfies that bound while raising when hashed -- so the bound
    documents the requirement rather than enforcing it, and a consumer binding
    ``K`` to such a type type-checks cleanly and fails at the call.

    Pinned as the stated **limit** rather than as a defect: what is asserted
    here is exactly what the class docstring and the guide promise, so a
    sentence that widened the promise without the code widening with it fails
    here.
    """
    unhashable = _Labelled("billing", {"team": "core"})

    assert isinstance(unhashable, Hashable)
    with pytest.raises(TypeError, match="unhashable type: 'dict'"):
        hash(unhashable)

    from_the_structure = HierarchyView(unhashable, "billing")
    assert isinstance(from_the_structure, Hashable)
    with pytest.raises(TypeError, match="unhashable type: 'dict'"):
        hash(from_the_structure)

    from_the_key = HierarchyView(MappingHierarchy(BILLING), unhashable)
    assert isinstance(hash(from_the_key.structure), int)
    assert isinstance(from_the_key, Hashable)
    with pytest.raises(TypeError, match="unhashable type: 'dict'"):
        hash(from_the_key)


def test_a_walk_keys_its_seen_set_on_cursors_over_a_mapping_a_consumer_holds() -> None:
    """The guide's deferred walk, run over the guide's own worked backing.

    The three fences it is made of are each well-formed; what was wrong was
    the sequence, and only running them in order shows it. Kept here as the
    executable form of that page's ``seen: set[...]`` walk.
    """
    seen: set[HierarchyView[str]] = set()
    frontier = [HierarchyView(MappingHierarchy(BILLING), "billing")]
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        frontier.extend(node.children())

    assert sorted(view.node for view in seen) == ["billing", "late-fees", "refunds"]


def test_a_materialized_axis_hands_back_a_structure_a_cursor_can_hash(
    materialized_structure_path: Path,
) -> None:
    """The reach that is nobody's hand-built example: a document asking for a copy.

    An axis declaring ``materialization.structure: materialized`` is given a
    :class:`MappingHierarchy` by the loader, so ``axis.structure`` is a carried
    backing for every such axis a document declares -- reached without a
    consumer ever naming the class, and read through the structure rather than
    through the axis cursor, which was fixed on the axis.
    """
    axis = load_ontology(materialized_structure_path).taxonomy("species")

    assert isinstance(axis.structure, MappingHierarchy)
    assert hash(HierarchyView(axis.structure, "dog")) == hash(HierarchyView(axis.structure, "dog"))


def test_no_mapping_twin_gets_its_hash_by_inheriting_the_shared_body() -> None:
    """Each twin's own decorator carries ``eq=False``; the base's does not travel.

    ``@dataclass(frozen=True)`` on a subclass regenerates ``__eq__`` and, with
    it, a field-wise ``__hash__`` -- so a twin declared without ``eq=False``
    reaches ``parent_map`` again however the shared body is declared. That is
    the trap this asserts against: the property reads as inherited and is not.
    Enumerated rather than listed, so a third twin added later fails here
    rather than at a consumer's ``hash()``.
    """
    bodies = (_MappingBacking, *_MappingBacking.__subclasses__())

    assert len(bodies) >= 3
    for cls in bodies:
        assert cls.__hash__ is object.__hash__, f"{cls.__name__} hashes its fields"


def test_the_mapping_twins_are_compared_by_identity_which_is_the_price() -> None:
    """The cost of the line above, asserted where it lands rather than left implicit.

    Two separately built mappings holding the same edges are no longer equal,
    the way two separately built axes are not. Nothing in this package compared
    two of them, and a consumer wanting *same edges* has ``parent_edges()``.
    """
    held = MappingHierarchy(BILLING)

    assert held in {held}
    assert held != MappingHierarchy(BILLING)
    assert held.parent_edges() == MappingHierarchy(BILLING).parent_edges()


# --------------------------------------------------------------------------
# The word the members must not carry
# --------------------------------------------------------------------------


def _every_name(cls: type) -> set[str]:
    """Every attribute, field and parameter name a class exposes."""
    names = set(vars(cls)) | {f.name for f in dataclasses.fields(cls)}
    for attribute in vars(cls).values():
        if callable(attribute):
            names |= set(inspect.signature(attribute).parameters)
    return names


@pytest.mark.parametrize(
    "cls", [HierarchyView, AsyncHierarchyView, TaxonomyView, AsyncTaxonomyView]
)
def test_nothing_on_the_view_is_named_granularity(cls: type) -> None:
    """The word belongs to a projection policy over a match set, not to a node.

    Asserted by introspection over all four cursors, so the collision cannot
    come back silently on any of them.
    """
    assert not [name for name in _every_name(cls) if "granularity" in name.lower()]


# --------------------------------------------------------------------------
# The twin, and parity
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_async_twin_answers_the_same(mammals_path: Path) -> None:
    """Same answers, awaited -- and ``at()`` awaited by nobody."""
    onto = await async_load_ontology(mammals_path)
    view = AsyncHierarchyView(AsyncAssertionHierarchy(onto.assertions, "isa"), "dog")

    assert await view.exists() is True
    assert await view.is_root() is False
    assert await view.is_leaf() is False
    assert [above.node for above in await view.parents()] == ["mammal"]
    assert [below.node for below in await view.children()] == ["beagle"]

    absent = view.at("no_such_node")
    assert await absent.exists() is False
    assert await absent.is_leaf() is False
    assert await absent.is_root() is False
    assert await view.at("beagle").is_leaf() is True


#: The members whose return type is the same in both flavours.
_BOOL_MEMBERS = ("exists", "is_root", "is_leaf")

#: The members that return views -- each flavour's own, so the return
#: annotation differs by flavour and is not compared.
_VIEW_MEMBERS = ("parents", "children")


def test_the_view_twins_expose_the_same_annotated_surface() -> None:
    """Same members, same parameters, same annotations; return compared where it can be."""
    assert_twin_types_agree(HierarchyView, AsyncHierarchyView, _BOOL_MEMBERS, compare_return=True)
    assert_twin_types_agree(HierarchyView, AsyncHierarchyView, _VIEW_MEMBERS)


def test_at_is_a_plain_def_on_both_twins_with_one_signature() -> None:
    """Both are ``def``, so the twin guard is the wrong instrument.

    ``assert_twins_agree`` refuses a synchronous async-half by design, which
    is exactly what ``at()`` is: it constructs, so it awaits nothing on either
    flavour. The signatures are compared directly instead.
    """
    assert not inspect.iscoroutinefunction(AsyncHierarchyView.at)

    sync_at = inspect.signature(HierarchyView.at)
    async_at = inspect.signature(AsyncHierarchyView.at)
    assert list(sync_at.parameters) == list(async_at.parameters) == ["self", "node_id"]
    assert sync_at.parameters["node_id"].annotation == async_at.parameters["node_id"].annotation
