"""The structure axis walks, and the three claims a walk has to make.

Four kinds of test, and they are not substitutes for each other:

* **the answer** -- ``ancestors`` returns the right ids in the right order;
* **the differential** -- both flavours return the *same* answer over one
  deliberately cyclic fixture, which is the only input that tells a walk that
  terminates by construction from one that terminates by luck;
* **the delegation** -- patching the shared core moves both flavours, which is
  what distinguishes *calls the core* from *agrees with the core today*;
* **parity** -- the twins expose the same members with the same annotations.

Surface equality cannot make the third claim and results cannot make the
fourth, which is why all four are here.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dataknobs_common import hierarchy as hierarchy_module
from dataknobs_common.hierarchy import (
    AsyncHierarchy,
    Hierarchy,
    ancestors,
    async_ancestors,
)
from dataknobs_common.ontology import async_load_ontology, load_ontology
from dataknobs_common.ontology.hierarchy import (
    AssertionHierarchy,
    AsyncAssertionHierarchy,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# A DAG with a cycle in it: root -> a -> c -> f -> root. Nothing in the walks
# consults an acyclicity claim, because the data may be cyclic whatever one
# says, and a cycle is the input that separates a correct walk from a lucky one.
CYCLIC_PARENTS: Mapping[str, tuple[str, ...]] = {
    "f": ("c", "d"),
    "c": ("a",),
    "d": ("b", "root"),
    "a": ("root",),
    "b": ("root",),
    "root": ("f",),
}


class MappingParents:
    """A minimal synchronous hierarchy over a parent mapping."""

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        self._parents = parents

    def roots(self) -> Sequence[str]:
        return tuple(n for n in self._parents if not self._parents[n])

    def parents(self, node_id: str) -> Sequence[str]:
        return self._parents.get(node_id, ())

    def children(self, node_id: str) -> Sequence[str]:
        return tuple(n for n, ps in self._parents.items() if node_id in ps)

    def contains(self, node_id: str) -> bool:
        return node_id in self._parents


class AsyncMappingParents:
    """The same, awaited -- the differential test's other half."""

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        self._inner = MappingParents(parents)

    async def roots(self) -> Sequence[str]:
        return self._inner.roots()

    async def parents(self, node_id: str) -> Sequence[str]:
        return self._inner.parents(node_id)

    async def children(self, node_id: str) -> Sequence[str]:
        return self._inner.children(node_id)

    async def contains(self, node_id: str) -> bool:
        return self._inner.contains(node_id)


class IntParents:
    """A hierarchy whose keys are not strings, for the key parameter."""

    def roots(self) -> Sequence[int]:
        return (1,)

    def parents(self, node_id: int) -> Sequence[int]:
        return (node_id - 1,) if node_id > 1 else ()

    def children(self, node_id: int) -> Sequence[int]:
        return (node_id + 1,)

    def contains(self, node_id: int) -> bool:
        return node_id >= 1


# --------------------------------------------------------------------------
# Criterion 12 -- the answer, over the authored assertions
# --------------------------------------------------------------------------


def test_ancestors_over_authored_assertions_with_no_store(mammals_path: Path) -> None:
    """``dog`` then ``mammal``, in that order, and nothing stood up first.

    Order is the assertion, not membership: a set comparison passes on a
    reversed walk, and *nearest first* is what a consumer folding ancestors
    into a prompt is relying on.
    """
    onto = load_ontology(mammals_path)

    structure = AssertionHierarchy(onto.assertions, "isa")

    assert ancestors(structure, "beagle") == ("dog", "mammal")


def test_the_walk_needs_no_event_loop(mammals_path: Path) -> None:
    """No store, no embedder, and no loop -- asserted, not assumed."""
    onto = load_ontology(mammals_path)

    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()

    assert ancestors(AssertionHierarchy(onto.assertions, "isa"), "beagle")


def test_the_type_lattice_does_not_leak_into_the_instance_walk(
    mammals_path: Path,
) -> None:
    """``Breed isa Species`` is a field; ``beagle isa dog`` is a row.

    Both spell the relation ``isa`` and the axis sees only the second. A leak
    would put ``Species`` in a walk over instances, where a consumer folding
    ancestors into a prompt could not detect it.
    """
    onto = load_ontology(mammals_path)

    walked = ancestors(AssertionHierarchy(onto.assertions, "isa"), "beagle")

    assert "Species" not in walked
    assert "Breed" not in walked


# --------------------------------------------------------------------------
# The assertion-backed hierarchy's own four members
# --------------------------------------------------------------------------


def test_the_axis_answers_all_four_members(mammals_v11_path: Path) -> None:
    """Roots, parents, children and containment over one relation."""
    onto = load_ontology(mammals_v11_path)
    axis = AssertionHierarchy(onto.assertions, "isa")

    assert tuple(axis.roots()) == ("mammal",)
    assert tuple(axis.parents("beagle")) == ("dog",)
    assert tuple(axis.children("dog")) == ("retriever", "beagle")
    assert axis.contains("beagle")
    assert axis.contains("mammal")


def test_a_node_no_edge_mentions_is_absent(mammals_v11_path: Path) -> None:
    """Absence and childlessness are opposite answers, not the same one."""
    onto = load_ontology(mammals_v11_path)
    axis = AssertionHierarchy(onto.assertions, "isa")

    assert not axis.contains("no_such_node")
    assert tuple(axis.children("no_such_node")) == ()
    # ...and a node that IS present with no children reports the same empty
    # children, which is exactly why `contains` has to exist.
    assert axis.contains("beagle")
    assert tuple(axis.children("beagle")) == ()


def test_roots_is_not_an_extent(mammals_v11_path: Path) -> None:
    """The nodes this relation leaves unplaced, not the members of a type.

    ``Species`` has two entities and the axis has one root, so the two answers
    are visibly different rather than accidentally equal.
    """
    onto = load_ontology(mammals_v11_path)
    axis = AssertionHierarchy(onto.assertions, "isa")

    assert tuple(axis.roots()) == ("mammal",)
    assert onto.entities.by_type("Species") == frozenset({"mammal", "dog"})


def test_a_literal_object_contributes_no_parent(mammals_v11_path: Path) -> None:
    """``dog lifespan_years 12`` is a value, not a place in a structure."""
    onto = load_ontology(mammals_v11_path)

    axis = AssertionHierarchy(onto.assertions, "lifespan_years")

    assert tuple(axis.parents("dog")) == ()
    assert tuple(axis.roots()) == ("dog",)
    # The node is still one this relation mentions, which is what `contains`
    # answers -- an axis that has heard of a node and places it nowhere.
    assert axis.contains("dog")


# --------------------------------------------------------------------------
# The differential -- both flavours, one cyclic fixture, equal answers
# --------------------------------------------------------------------------


def test_both_flavours_agree_over_a_cyclic_hierarchy() -> None:
    """The same walk, driven two ways, over data that would trap a naive one.

    The test that would be skipped, because both flavours passing their own
    assertions reads as sufficient. It is not: the claim is that they are *the
    same walk*, and only a comparison over one fixture asserts that.
    """
    walked = ancestors(MappingParents(CYCLIC_PARENTS), "f")
    awaited = asyncio.run(async_ancestors(AsyncMappingParents(CYCLIC_PARENTS), "f"))

    assert walked == awaited
    assert walked == ("c", "d", "a", "b", "root")
    # `root`'s parent is the anchor, and the anchor is in the visited set from
    # the first line -- so the cycle closes without the walk re-emitting `f`.
    assert "f" not in walked


def test_a_cyclic_walk_returns_each_node_once() -> None:
    """The visited set is unconditional, so a cycle terminates and dedups."""
    walked = ancestors(MappingParents(CYCLIC_PARENTS), "f")

    assert len(walked) == len(set(walked))


def test_the_anchor_is_excluded_from_its_own_ancestors(mammals_path: Path) -> None:
    """Excluded at the boundary, not by a rule inside the walk."""
    onto = load_ontology(mammals_path)

    assert "beagle" not in ancestors(AssertionHierarchy(onto.assertions, "isa"), "beagle")


@pytest.mark.asyncio
async def test_the_async_axis_walks_the_same_vocabulary(mammals_path: Path) -> None:
    """The asynchronous door's ontology, walked by the asynchronous twin."""
    onto = await async_load_ontology(mammals_path)

    structure = AsyncAssertionHierarchy(onto.assertions, "isa")

    assert await async_ancestors(structure, "beagle") == ("dog", "mammal")
    assert tuple(await structure.children("dog")) == ("beagle",)
    assert await structure.contains("beagle")


# --------------------------------------------------------------------------
# The delegation -- one core, two surfaces
# --------------------------------------------------------------------------


def test_patching_the_core_moves_both_flavours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both wrappers *invoke* the shared generator rather than agreeing with it.

    Equality of results cannot make this claim. A flavour that inlined the
    traversal would return the same answer on the day it was written and drift
    a year later, silently, which is the failure this shape exists to catch.
    """

    def sentinel_walk(node_id: str) -> object:
        return iter(())  # a walk that asks nothing and returns None

    monkeypatch.setattr(hierarchy_module, "_ancestors", sentinel_walk)

    assert ancestors(MappingParents(CYCLIC_PARENTS), "f") is None
    assert asyncio.run(async_ancestors(AsyncMappingParents(CYCLIC_PARENTS), "f")) is None


# --------------------------------------------------------------------------
# Parity, the protocol check, and the key parameter
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sync_type", "async_type"),
    [
        (Hierarchy, AsyncHierarchy),
        (AssertionHierarchy, AsyncAssertionHierarchy),
    ],
)
def test_the_twins_expose_the_same_annotated_surface(sync_type: type, async_type: type) -> None:
    """Same member names, same parameters, same annotations.

    Annotations rather than the whole signature: a twin's return type differs
    by flavour where it streams, and comparing the rendered signature would
    then compare the ``async`` rather than the contract.
    """
    for name in ("roots", "parents", "children", "contains"):
        sync_member = getattr(sync_type, name)
        async_member = getattr(async_type, name)
        assert inspect.signature(sync_member).parameters.keys() == (
            inspect.signature(async_member).parameters.keys()
        ), name
        assert sync_member.__annotations__ == async_member.__annotations__, name
        assert not inspect.iscoroutinefunction(sync_member), name
        assert inspect.iscoroutinefunction(async_member), name


def test_the_concretes_satisfy_the_protocols_at_runtime(mammals_path: Path) -> None:
    """``@runtime_checkable``, and the one thing it does not reach.

    ``isinstance`` compares member *names* and nothing else, so a synchronous
    implementation satisfies the asynchronous protocol at runtime too. Asserted
    rather than left to be discovered: it is the reason the static check is the
    one that separates the flavours.
    """
    onto = load_ontology(mammals_path)
    axis = AssertionHierarchy(onto.assertions, "isa")

    assert isinstance(axis, Hierarchy)
    assert isinstance(axis, AsyncHierarchy)  # names only -- see the docstring

    with pytest.raises(TypeError):
        isinstance(axis, Hierarchy[str])


def test_a_non_string_key_round_trips() -> None:
    """The walks hash a key and never inspect one, so any hashable will do."""
    assert ancestors(IntParents(), 4) == (3, 2, 1)


def test_hierarchy_does_not_import_the_ontology_package() -> None:
    """The general module does not depend on the specific package.

    A runtime edge back would close a cycle through ``ontology/__init__``,
    which imports the values module that imports this one. Checkable, so it is
    checked rather than left as a convention.
    """
    import subprocess
    import sys

    probe = (
        "import sys, dataknobs_common.hierarchy, dataknobs_common.taxonomy; "
        "print('dataknobs_common.ontology' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"
