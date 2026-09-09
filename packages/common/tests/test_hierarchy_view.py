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
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from dataknobs_common.hierarchy import (
    AsyncHierarchyView,
    HierarchyView,
    MappingHierarchy,
)
from dataknobs_common.ontology import async_load_ontology, load_ontology
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
