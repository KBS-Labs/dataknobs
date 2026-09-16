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
from collections.abc import Callable, Generator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common import _walk_core as walk_core
from dataknobs_common import hierarchy as hierarchy_module
from dataknobs_common.exceptions import NotFoundError, OperationError
from dataknobs_common.hierarchy import (
    AsyncBulkHierarchy,
    AsyncEnumerableHierarchy,
    AsyncHierarchy,
    AsyncHierarchyView,
    AsyncMappingHierarchy,
    BulkHierarchy,
    EnumerableHierarchy,
    Hierarchy,
    HierarchyView,
    MappingHierarchy,
    ancestors,
    async_ancestors,
    async_children_at_depth,
    async_deepest_common_ancestor,
    async_descendants,
    async_descendants_to_depth,
    async_drive,
    async_flatten,
    async_leaves,
    async_paths_to_root,
    children_at_depth,
    deepest_common_ancestor,
    descendants,
    descendants_to_depth,
    drive,
    flatten,
    leaves,
    paths_to_root,
)
from dataknobs_common.ontology import async_load_ontology, load_ontology
from dataknobs_common.ontology.hierarchy import (
    AssertionHierarchy,
    AsyncAssertionHierarchy,
)
from dataknobs_common.ontology.model import TaxonomyDefinition
from dataknobs_common.ontology.sources import AsyncMappingEntitySource, MappingEntitySource
from dataknobs_common.ontology.taxonomy import AsyncTaxonomy, Taxonomy
from dataknobs_common.testing import assert_twin_types_agree, assert_twins_agree

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
# The answer, over the authored assertions
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


#: Every walk this module publishes, called two ways over one cyclic fixture.
#:
#: **Eight rows, one fixture, and both of those were the work.** This was two
#: parametrisations over two identical copies of the mapping below -- five
#: descending walks in the neighbouring suite, ``ancestors`` here -- which is
#: six walks of eight and two fixtures where the claim is eight over one. A
#: differential that covers some of the walks is green about the ones it does
#: not, and two copies of a fixture are two fixtures the day one is edited.
_CYCLIC_WALKS: tuple[tuple[str, Callable[..., Any], Callable[..., Any]], ...] = (
    ("ancestors", lambda h: ancestors(h, "f"), lambda h: async_ancestors(h, "f")),
    ("descendants", lambda h: descendants(h, "root"), lambda h: async_descendants(h, "root")),
    (
        "descendants_to_depth",
        lambda h: descendants_to_depth(h, "root", 2),
        lambda h: async_descendants_to_depth(h, "root", 2),
    ),
    (
        "children_at_depth",
        lambda h: children_at_depth(h, "root", 2),
        lambda h: async_children_at_depth(h, "root", 2),
    ),
    (
        "flatten",
        lambda h: flatten(h, from_id="root"),
        lambda h: async_flatten(h, from_id="root"),
    ),
    ("leaves", lambda h: leaves(h, under="root"), lambda h: async_leaves(h, under="root")),
    ("paths_to_root", lambda h: paths_to_root(h, "f"), lambda h: async_paths_to_root(h, "f")),
    (
        "deepest_common_ancestor",
        lambda h: deepest_common_ancestor(h, "c", "d"),
        lambda h: async_deepest_common_ancestor(h, "c", "d"),
    ),
)

#: The six walks whose answer is a flat tuple of node ids.
#:
#: Split out rather than asserted inside the differential, because the dedup
#: claim is not true of all eight. ``paths_to_root`` returns paths that *share*
#: nodes by construction -- that is the whole of what a path walk answers that
#: an ancestor walk cannot -- and ``deepest_common_ancestor`` returns one key
#: or ``None``. Asserting ``len(set(x)) == len(x)`` over either is not merely
#: inapplicable; over a DAG it is false of a correct answer.
_FLAT_CYCLIC_WALKS = tuple(
    row for row in _CYCLIC_WALKS if row[0] not in {"paths_to_root", "deepest_common_ancestor"}
)


#: The two backing kinds a differential has to cover, and why both.
#:
#: ``MappingHierarchy`` implements ``parents_many``, so the frontier read
#: returns on the **bulk** branch and never builds the bounded ``gather`` the
#: singular branch does. Driving only that pair leaves the asynchronous
#: semaphore path unexercised over cyclic data -- which is exactly the branch a
#: walk taken node-at-a-time never reaches and a walk taken a frontier at a
#: time depends on. The pair is parametrised rather than chosen because the
#: claim is about the walks, and a walk that terminates against one backing and
#: not the other is the drift this suite exists to catch.
_CYCLIC_BACKINGS: tuple[tuple[str, Callable[..., Any], Callable[..., Any]], ...] = (
    ("bulk", MappingHierarchy, AsyncMappingHierarchy),
    ("singular", MappingParents, AsyncMappingParents),
)


@pytest.mark.parametrize(
    ("backing", "sync_axis", "async_axis"),
    _CYCLIC_BACKINGS,
    ids=[name for name, _, _ in _CYCLIC_BACKINGS],
)
@pytest.mark.parametrize(
    ("walk", "sync_walk", "async_walk"),
    _CYCLIC_WALKS,
    ids=[name for name, _, _ in _CYCLIC_WALKS],
)
def test_every_walk_terminates_on_cyclic_data_and_both_flavours_agree(
    walk: str,
    sync_walk: Callable[..., Any],
    async_walk: Callable[..., Any],
    backing: str,
    sync_axis: Callable[..., Any],
    async_axis: Callable[..., Any],
) -> None:
    """A cycle is the input that separates a correct walk from a lucky one.

    Both halves in one test because they are one claim: a visited set that is
    unconditional in one flavour and conditional in the other is a difference
    the answers show and the surfaces do not. Termination is asserted by the
    call returning at all -- a walk that did not terminate would hang here
    rather than fail, which is the one failure mode a test cannot phrase as an
    assertion.

    **The claim this can make of all eight is termination and agreement**, and
    that is why the shape assertions are elsewhere. Two of the eight do not
    return a flat tuple of ids, so the dedup assertion that used to live in
    this body covers six of them and moved to the row set it is true of.

    **Both backing kinds**, because the frontier read branches on them: a
    backing offering the bulk members never reaches the bounded ``gather``, so
    a differential run only against one of the two is green about a branch it
    never entered.
    """
    del walk, backing  # named for the failure message pytest prints
    result = sync_walk(sync_axis(CYCLIC_PARENTS))

    assert result == asyncio.run(async_walk(async_axis(CYCLIC_PARENTS)))


@pytest.mark.parametrize(
    ("backing", "sync_axis", "async_axis"),
    _CYCLIC_BACKINGS,
    ids=[name for name, _, _ in _CYCLIC_BACKINGS],
)
@pytest.mark.parametrize(
    ("walk", "sync_walk", "async_walk"),
    _FLAT_CYCLIC_WALKS,
    ids=[name for name, _, _ in _FLAT_CYCLIC_WALKS],
)
def test_a_cyclic_walk_returns_each_node_once(
    walk: str,
    sync_walk: Callable[..., Any],
    async_walk: Callable[..., Any],
    backing: str,
    sync_axis: Callable[..., Any],
    async_axis: Callable[..., Any],
) -> None:
    """The visited set is unconditional, so a cycle terminates and dedups."""
    del walk, async_walk, backing, async_axis
    result = sync_walk(sync_axis(CYCLIC_PARENTS))

    assert len(set(result)) == len(result), "a node was emitted twice"


def test_both_flavours_agree_over_a_cyclic_hierarchy() -> None:
    """The same walk, driven two ways, over data that would trap a naive one.

    The row above asserts that the two flavours agree; this asserts *what* they
    agree on, which a differential cannot: two identically wrong walks agree.
    """
    walked = ancestors(MappingParents(CYCLIC_PARENTS), "f")
    awaited = asyncio.run(async_ancestors(AsyncMappingParents(CYCLIC_PARENTS), "f"))

    assert walked == awaited
    assert walked == ("c", "d", "a", "b", "root")
    # `root`'s parent is the anchor, and the anchor is in the visited set from
    # the first line -- so the cycle closes without the walk re-emitting `f`.
    assert "f" not in walked


def test_the_anchor_is_excluded_by_the_walks_and_included_by_subtree_keys(
    mammals_v11_path: Path,
) -> None:
    """The boundary, asserted from both sides in one test.

    ``ancestors`` and ``descendants`` walk *away* from the anchor and exclude
    it; ``subtree_keys`` answers *this and everything under it* and includes
    it. Either alone is a sentence about one function. Together they are the
    boundary, and the boundary is what a caller has to know: a filter built
    from a walk that silently dropped the node the user named under-counts, and
    the count is the whole answer.

    One test because the difference is the subject. Two tests asserting one
    inclusion each would both keep passing if the two walks were made to agree.
    """
    axis = load_ontology(mammals_v11_path).taxonomy("species")
    view = axis.at("dog")
    structure = axis.structure

    assert view.node not in ancestors(structure, view.node)
    assert view.node not in descendants(structure, view.node)
    assert view.node in axis.subtree_keys(view.node)

    assert set(axis.subtree_keys(view.node)) == {view.node, *descendants(structure, view.node)}


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


class RaisesStopIteration:
    """A hierarchy whose members raise ``StopIteration`` out of a plain ``def``.

    Not a contrived shape: ``return next(r["parent"] for r in rows if ...)``
    over a row set that does not contain ``node_id`` raises exactly this, and a
    column-backed hierarchy is a case the guide invites.
    """

    def roots(self) -> Sequence[str]:
        return (next(x for x in ()),)  # type: ignore[unreachable]

    def parents(self, node_id: str) -> Sequence[str]:
        return (next(x for x in ()),)  # type: ignore[unreachable]

    def children(self, node_id: str) -> Sequence[str]:
        return (next(x for x in ()),)  # type: ignore[unreachable]

    def contains(self, node_id: str) -> bool:
        return False


class AsyncRaisesStopIteration:
    """:class:`RaisesStopIteration`'s asynchronous twin."""

    async def roots(self) -> Sequence[str]:
        return (next(x for x in ()),)  # type: ignore[unreachable]

    async def parents(self, node_id: str) -> Sequence[str]:
        return (next(x for x in ()),)  # type: ignore[unreachable]

    async def children(self, node_id: str) -> Sequence[str]:
        return (next(x for x in ()),)  # type: ignore[unreachable]

    async def contains(self, node_id: str) -> bool:
        return False


def _asking_for(
    member: str,
) -> Generator[tuple[str, tuple[str, ...]], tuple[Sequence[str], ...], tuple[str, ...]]:
    """A one-question walk, in the shape the guide invites a consumer to write."""
    (reply,) = yield (member, ("anything",))
    return tuple(reply)


@pytest.mark.parametrize("member", ["roots", "parents", "children"])
def test_a_collaborators_stop_iteration_never_reads_as_the_walk_finishing(
    member: str,
) -> None:
    """A ``StopIteration`` out of the *hierarchy* must not end the walk quietly.

    ``drive`` catches ``StopIteration`` to learn the walk's return value, and
    the collaborator's calls sat inside that same ``try``. A hierarchy is
    arbitrary consumer code and a plain ``def`` is not covered by PEP 479, so
    one escaping ``roots()`` was indistinguishable from the generator returning:
    ``drive`` returned ``stop.value`` -- ``None``, typed as the walk's result --
    and the caller failed later at ``for x in None``, nowhere near the cause.

    Parametrised over all three members because only ``roots`` was exposed. The
    other two are protected *by accident*: their replies are built with
    generator expressions, where PEP 479 does fire. Rewriting either as a list
    comprehension would silently reinstate the defect, so all three are pinned.
    """
    with pytest.raises(RuntimeError, match="StopIteration"):
        drive(RaisesStopIteration(), _asking_for(member))

    with pytest.raises(RuntimeError, match="StopIteration"):
        asyncio.run(async_drive(AsyncRaisesStopIteration(), _asking_for(member)))


@pytest.mark.parametrize("member", ["roots", "parents", "children"])
def test_both_flavours_fail_alike_on_a_collaborators_stop_iteration(
    member: str,
) -> None:
    """The twins raise the *same* type, which is the property the pair rests on.

    Before the fix they did not: for ``roots`` the synchronous driver returned
    ``None`` while the asynchronous one raised ``RuntimeError`` from its own
    coroutine frame. One algorithm behaving two ways by flavour is exactly what
    this module's shape exists to prevent, so the differential is the assertion
    that matters -- more than either surface's behaviour taken alone.
    """
    sync_raised: type[BaseException] | None = None
    async_raised: type[BaseException] | None = None

    try:
        drive(RaisesStopIteration(), _asking_for(member))
    except Exception as exc:  # the exception's type is what is being compared
        sync_raised = type(exc)

    try:
        asyncio.run(async_drive(AsyncRaisesStopIteration(), _asking_for(member)))
    except Exception as exc:  # the exception's type is what is being compared
        async_raised = type(exc)

    assert sync_raised is not None, "the synchronous driver answered instead of raising"
    assert sync_raised is async_raised


class CountingParents:
    """A hierarchy that records how it was asked, with no bulk members."""

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        self._inner = MappingParents(parents)
        self.singular_calls = 0
        self.contains_calls = 0

    def roots(self) -> Sequence[str]:
        return self._inner.roots()

    def parents(self, node_id: str) -> Sequence[str]:
        self.singular_calls += 1
        return self._inner.parents(node_id)

    def children(self, node_id: str) -> Sequence[str]:
        self.singular_calls += 1
        return self._inner.children(node_id)

    def contains(self, node_id: str) -> bool:
        self.contains_calls += 1
        return self._inner.contains(node_id)


class BulkParents(CountingParents):
    """The same, offering the optional frontier members."""

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        super().__init__(parents)
        self.bulk_calls = 0
        self.widest_frontier = 0

    def parents_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        self.bulk_calls += 1
        self.widest_frontier = max(self.widest_frontier, len(node_ids))
        return tuple(self._inner.parents(n) for n in node_ids)

    def children_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        self.bulk_calls += 1
        self.widest_frontier = max(self.widest_frontier, len(node_ids))
        return tuple(self._inner.children(n) for n in node_ids)


class AsyncCountingParents:
    """:class:`CountingParents`, awaited -- singular-only, and counting.

    Singular-only for :class:`ConcurrencyProbe`'s reason turned around: a
    backing that answers a level in one call is asked once per *level*, so a
    count over one is a count of levels and says nothing about how many nodes
    were asked about. This counts nodes, which is what a memo saves.
    """

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        self._inner = MappingParents(parents)
        self.singular_calls = 0

    async def roots(self) -> Sequence[str]:
        return self._inner.roots()

    async def parents(self, node_id: str) -> Sequence[str]:
        self.singular_calls += 1
        return self._inner.parents(node_id)

    async def children(self, node_id: str) -> Sequence[str]:
        self.singular_calls += 1
        return self._inner.children(node_id)

    async def contains(self, node_id: str) -> bool:
        return self._inner.contains(node_id)


class AsyncBulkParents:
    """An asynchronous hierarchy offering the optional frontier members."""

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        self._inner = MappingParents(parents)
        self.bulk_calls = 0
        self.singular_calls = 0

    async def roots(self) -> Sequence[str]:
        return self._inner.roots()

    async def parents(self, node_id: str) -> Sequence[str]:
        self.singular_calls += 1
        return self._inner.parents(node_id)

    async def children(self, node_id: str) -> Sequence[str]:
        self.singular_calls += 1
        return self._inner.children(node_id)

    async def parents_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        self.bulk_calls += 1
        return tuple(self._inner.parents(n) for n in node_ids)

    async def children_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        self.bulk_calls += 1
        return tuple(self._inner.children(n) for n in node_ids)

    async def contains(self, node_id: str) -> bool:
        return self._inner.contains(node_id)


class ConcurrencyProbe:
    """An asynchronous hierarchy that records how many calls overlap.

    **Singular-only, and that is load-bearing rather than incidental.** The
    frontier read returns on the bulk path *before* the bound is constructed --
    a backing answering a whole level in one call has nothing to bound -- so a
    ``max_concurrency`` test written against a backing that offers
    ``parents_many`` measures nothing and passes. Every bound test below uses
    this class for that reason. Give it bulk members and they all go quiet
    without failing.
    """

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        self._inner = MappingParents(parents)
        self._in_flight = 0
        self.widest_overlap = 0

    async def _record(self) -> None:
        self._in_flight += 1
        self.widest_overlap = max(self.widest_overlap, self._in_flight)
        await asyncio.sleep(0)  # a suspension point for the others to reach
        self._in_flight -= 1

    async def roots(self) -> Sequence[str]:
        return self._inner.roots()

    async def parents(self, node_id: str) -> Sequence[str]:
        await self._record()
        return self._inner.parents(node_id)

    async def children(self, node_id: str) -> Sequence[str]:
        await self._record()
        return self._inner.children(node_id)

    async def contains(self, node_id: str) -> bool:
        return self._inner.contains(node_id)


# A wide level: one root with four children, so a frontier read is visibly
# different from four node reads.
WIDE_PARENTS: Mapping[str, tuple[str, ...]] = {
    "root": (),
    "a": ("root",),
    "b": ("root",),
    "c": ("root",),
    "d": ("root",),
}


def test_a_frontier_is_one_bulk_call_rather_than_one_call_per_node() -> None:
    """A backing offering the optional members is asked once per level.

    The protocol's request shape is a frontier and ``find_many``-style backings
    answer a frontier, but ``Hierarchy`` only offered ``children(node_id)`` --
    so the driver fanned a level back out to N calls and a row-backed hierarchy
    paid N queries per level with no way to fix it from inside the protocol.
    """
    bulk = BulkParents(WIDE_PARENTS)

    assert set(ancestors(bulk, "a")) == {"root"}
    assert bulk.bulk_calls > 0
    assert bulk.singular_calls == 0


def test_a_backing_without_the_bulk_members_is_still_driven() -> None:
    """The members are optional: the singular pair remains sufficient."""
    plain = CountingParents(WIDE_PARENTS)

    assert set(ancestors(plain, "a")) == {"root"}
    assert plain.singular_calls > 0


def test_the_bulk_reply_is_positional() -> None:
    """One reply per node asked about, including nodes with no answer.

    The walk zips replies against the frontier it sent, so a backing that
    dropped empty answers would silently shift every later node's parents onto
    the wrong id.
    """
    bulk = BulkParents(WIDE_PARENTS)

    replies = bulk.parents_many(("a", "root", "b"))

    assert len(replies) == 3
    assert tuple(replies[1]) == ()


class DroppingBulk(BulkParents):
    """A bulk member that drops a node rather than answering it empty.

    The plausible way to break the positional contract, and therefore the one
    worth a test: a query returning one row per match returns *no* row for a
    node with no parents, so the natural implementation silently shortens the
    reply.
    """

    def parents_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        self.bulk_calls += 1
        return tuple(reply for n in node_ids if (reply := self._inner.parents(n)))


def test_a_bulk_member_that_breaks_the_positional_contract_is_named() -> None:
    """The walk refuses, and the refusal says whose contract was broken.

    Every path here pairs a frontier with its replies positionally, so a short
    reply is caught -- but caught by ``zip(strict=True)``, whose message names
    an argument number and no backing, no member and no counts. The contract is
    ``BulkHierarchy``'s, so the refusal belongs where a bulk member answers,
    not three frames later where the pairing happens to notice.
    """
    dropping = DroppingBulk(WIDE_PARENTS)

    with pytest.raises(ValueError) as caught:
        ancestors(dropping, "a")

    message = str(caught.value)
    assert "parents_many" in message, message
    assert "DroppingBulk" in message, message
    assert "one reply per node" in message, message


class AsyncDroppingBulk(AsyncBulkParents):
    """:class:`DroppingBulk`, awaited."""

    async def parents_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        self.bulk_calls += 1
        return tuple(reply for n in node_ids if (reply := self._inner.parents(n)))


def test_the_async_flavour_names_it_too() -> None:
    """The check is in the step both fetches call, so neither flavour has it alone."""
    with pytest.raises(ValueError) as caught:
        asyncio.run(async_ancestors(AsyncDroppingBulk(WIDE_PARENTS), "a"))

    assert "parents_many" in str(caught.value)
    assert "AsyncDroppingBulk" in str(caught.value)


def test_the_async_driver_asks_a_wide_level_concurrently() -> None:
    """Without bulk members, a level is gathered rather than awaited in turn.

    ``async_drive``'s stated benefit is one round of concurrency per depth. A
    sequential await per node returns the same answer, so only overlap
    distinguishes them.
    """
    probe = ConcurrencyProbe(WIDE_PARENTS)

    asyncio.run(async_ancestors(probe, "a"))

    assert probe.widest_overlap == 1, "one node per level here, so nothing to overlap"

    wide = ConcurrencyProbe({**WIDE_PARENTS, "deep": ("a", "b", "c", "d")})
    asyncio.run(async_ancestors(wide, "deep"))

    assert wide.widest_overlap > 1, "a four-node level was awaited one at a time"


def test_the_streaming_walk_shares_the_frontier_read(mammals_v11_path: Path) -> None:
    """``Taxonomy.walk`` gets bulk too, rather than deciding it a second time.

    It cannot go through the collecting core, so the risk is that the two drift
    over which backings answer a level in one query. Sharing the frontier read
    is what stops that.
    """
    onto = load_ontology(mammals_v11_path)
    bulk = BulkParents({"dog": (), "retriever": ("dog",), "beagle": ("dog",)})
    axis = replace(onto.taxonomy("species"), structure=bulk)

    assert set(axis.walk(from_id="dog")) == {"dog", "retriever", "beagle"}
    assert bulk.bulk_calls > 0
    assert bulk.singular_calls == 0


def test_the_async_streaming_walk_shares_it_too(mammals_v11_path: Path) -> None:
    """The asynchronous twin was reading a level one sequential await at a time.

    Its own module sells per-depth behaviour and says a twinned walk gets it
    only if somebody writes it into each copy. This is the copy that had not.
    """

    async def run() -> tuple[set[str], int, int]:
        onto = await async_load_ontology(mammals_v11_path)
        bulk = AsyncBulkParents({"dog": (), "retriever": ("dog",), "beagle": ("dog",)})
        axis = replace(onto.taxonomy("species"), structure=bulk)
        seen = {node_id async for node_id in axis.walk(from_id="dog")}
        return seen, bulk.bulk_calls, bulk.singular_calls

    seen, bulk_calls, singular_calls = asyncio.run(run())

    assert seen == {"dog", "retriever", "beagle"}
    assert bulk_calls > 0
    assert singular_calls == 0


def test_the_assertion_axis_bulk_members_agree_with_the_singular_ones(
    mammals_v11_path: Path,
) -> None:
    """The concrete's two forms answer the same thing for every node.

    The bulk pair goes through ``find_many`` and the singular pair through
    ``find``; agreement is the property that makes the optional members safe
    for the driver to prefer without asking the caller.
    """
    onto = load_ontology(mammals_v11_path)
    axis = AssertionHierarchy(onto.assertions, "isa")
    nodes = ("dog", "retriever", "beagle", "mammal", "nonesuch")

    assert tuple(tuple(r) for r in axis.parents_many(nodes)) == tuple(
        tuple(axis.parents(n)) for n in nodes
    )
    assert tuple(tuple(r) for r in axis.children_many(nodes)) == tuple(
        tuple(axis.children(n)) for n in nodes
    )


# --------------------------------------------------------------------------
# The seventh reading -- routes, where the other six give a set
# --------------------------------------------------------------------------


#: One node above two, joining again at a root. Two ways up from ``x``.
DIAMOND: Mapping[str, tuple[str, ...]] = {
    "x": ("a", "b"),
    "a": ("root",),
    "b": ("root",),
    "root": (),
}


def _stacked_diamonds(count: int) -> dict[str, tuple[str, ...]]:
    """``count`` diamonds end to end: ``3 * count + 1`` nodes, ``2 ** count`` paths.

    The shape that separates the two costs. Paths double per diamond and nodes
    grow by three, so a walk whose requests follow its paths and one whose
    requests follow its nodes diverge by an amount no small fixture shows.
    """
    parents: dict[str, tuple[str, ...]] = {"n0": ()}
    for level in range(count):
        parents[f"a{level}"] = (f"n{level}",)
        parents[f"b{level}"] = (f"n{level}",)
        parents[f"n{level + 1}"] = (f"a{level}", f"b{level}")
    return parents


def test_two_parents_give_two_paths_where_ancestors_gives_one_set() -> None:
    """The claim the path walk exists to make, beside the one it does not make.

    ``ancestors`` deduplicates: ``root`` is reachable two ways and appears
    once, which is right for *what is above me* and destroys *how did I get
    here*. Both are asserted in one test because either alone reads as a
    preference rather than as the boundary between two questions.
    """
    hierarchy = MappingHierarchy(DIAMOND)

    assert paths_to_root(hierarchy, "x") == (("x", "a", "root"), ("x", "b", "root"))
    assert ancestors(hierarchy, "x") == ("a", "b", "root")


def test_the_cycle_guard_is_scoped_to_the_path_and_not_to_the_walk() -> None:
    """The one thing this walk could get wrong, asserted rather than reviewed.

    Written over the shared descent it would compile, terminate on every cyclic
    fixture in this file, and return **one** path where two are owed -- because
    that descent's visited set is scoped to the walk, which is right for the
    six walks composed over it and wrong for this one. The node the two paths
    share is what shows it.
    """
    paths = paths_to_root(MappingHierarchy(DIAMOND), "x")

    assert len(paths) == 2
    assert all(path[-1] == "root" for path in paths), "both paths reach the shared root"
    assert sum("root" in path for path in paths) == 2, (
        "a walk-scoped visited set would have kept `root` on whichever path "
        "reached it first and dropped the other path entirely"
    )


def test_a_repeated_parent_in_one_reply_is_one_route() -> None:
    """``parents`` is consumer code, and a join answers one row per match.

    The same hazard the seed read guards against and the shared descent absorbs
    with its visited set. This walk has neither, because a node reached twice
    along *different* paths is genuinely two routes -- so the dedup has to be
    inside the one reply and nowhere wider, which is what the second assertion
    pins.
    """

    class DoubleCountingParents(MappingParents):
        """A backing whose join returns ``root`` twice for ``x``."""

        def parents(self, node_id: str) -> Sequence[str]:
            above = super().parents(node_id)
            return (*above, *above) if node_id == "x" else above

    assert paths_to_root(DoubleCountingParents({"x": ("root",), "root": ()}), "x") == (
        ("x", "root"),
    )
    assert paths_to_root(MappingHierarchy(DIAMOND), "x") == (
        ("x", "a", "root"),
        ("x", "b", "root"),
    ), "`root` is reached on two different paths, and that is two routes"


def test_the_paths_come_back_in_parent_order() -> None:
    """Emission order is a decision, and a stack answers it backwards by default.

    What a walk is made of and what it emits are two questions. A depth-first
    walk over a stack returns the *last* parent's paths first unless somebody
    reverses the pushes, and an undeclared order is one a caller comes to rely
    on before anyone notices it was never chosen.
    """
    hierarchy = MappingHierarchy({"z": ("first", "second"), "first": (), "second": ()})

    assert hierarchy.parents("z") == ("first", "second")
    assert paths_to_root(hierarchy, "z") == (("z", "first"), ("z", "second"))


def test_every_path_through_the_first_parent_precedes_every_path_through_the_second() -> None:
    """Outermost first: the branch nearest the anchor dominates the ordering.

    One diamond cannot tell *parent order at each node* from *outermost branch
    first*, because with one branch point the two coincide. Two stacked
    diamonds separate them.
    """
    hierarchy = MappingHierarchy(_stacked_diamonds(2))

    paths = paths_to_root(hierarchy, "n2")

    assert tuple(path[1] for path in paths) == ("a1", "a1", "b1", "b1")


def test_a_path_is_emitted_only_where_it_cannot_be_extended() -> None:
    """No path here is a prefix of another, which is what maximal means.

    A node with two parents, one of them already on the path, extends through
    the other -- and the truncated path is not also emitted. Emitting it would
    hand a caller counting *the ways up from here* a way that is really the
    first half of another.
    """
    #   t -> u -> t   is a cycle, and u also has a root above it
    hierarchy = MappingHierarchy({"t": ("u",), "u": ("t", "top"), "top": ()})

    paths = paths_to_root(hierarchy, "t")

    assert paths == (("t", "u", "top"),)
    assert not any(a != b and a == b[: len(a)] for a in paths for b in paths)

    # The degenerate cycle is the same rule: a node naming itself is a parent
    # already on the path, so the route extends through the other parent and
    # the one-element prefix is not also emitted.
    self_looped = MappingHierarchy({"n": ("n", "top"), "top": ()})

    assert self_looped.parents("n") == ("n", "top")
    assert paths_to_root(self_looped, "n") == (("n", "top"),)


def test_a_cyclic_component_with_no_root_returns_paths_that_reach_none() -> None:
    """Termination is by the guard, and the shape says the walk found no root.

    Nothing is raised: the far end of each path is a node whose every parent is
    already behind it, which is a fact about the axis rather than a failure of
    the walk.
    """
    paths = paths_to_root(MappingHierarchy({"p": ("q",), "q": ("p",)}), "p")

    assert paths == (("p", "q"),)
    assert MappingHierarchy({"p": ("q",), "q": ("p",)}).parents("q") == ("p",)


def test_an_unknown_anchor_is_refused_because_this_walk_emits_it() -> None:
    """``((node,),)`` is exactly the shape a root gives, so it cannot be returned.

    It reaches the refusal the way every emitting walk does: one that handed
    back an unknown anchor would hand back a term of the axis the caller cannot
    tell from a real one. The count of such walks is deliberately not stated --
    it was four when the rule was written and it moved twice while this branch
    was open, which is a number a docstring cannot keep.
    """
    hierarchy = MappingHierarchy(DIAMOND)

    assert paths_to_root(hierarchy, "root") == (("root",),)

    with pytest.raises(NotFoundError) as refusal:
        paths_to_root(hierarchy, "marmoset")

    assert refusal.value.context == {"anchor": "marmoset"}


def test_the_path_walk_asks_each_node_once_and_a_frontier_at_a_time() -> None:
    """The cost that decided the shape: per node, not per route.

    A walk that descends route by route asks about a node once per route that
    reaches it -- 125 requests for the sixteen nodes carrying these thirty-two
    routes -- and has to buy that back with a memo of its own. Reading the
    routes off a level-synchronous ascent instead, the ascent visits each node
    once and the enumeration issues no request at all.

    Asserted against the answer as well as the count: a cheaper walk that
    returned something else would be a different walk rather than a faster one.
    """
    axis = _stacked_diamonds(5)
    counting = CountingParents(axis)
    cached = CountingParents(axis)

    walked = paths_to_root(counting, "n5")

    assert len(walked) == 32, "thirty-two ways up five stacked diamonds"
    assert counting.singular_calls == len(axis) == 16, "one request per node, not per route"
    assert walked == paths_to_root(cached, "n5", cache={})
    assert cached.singular_calls == counting.singular_calls, (
        "a caller's cache has nothing here to save -- the walk already asks each node once"
    )


def test_the_path_walk_asks_a_bulk_backing_one_query_per_level() -> None:
    """The frontier property is the module's, and this walk is not exempt.

    Every walk here asks about a frontier rather than a node, which is what
    earns a bulk backing its per-level query and the asynchronous driver its
    round of concurrency per depth. A route-at-a-time walk asks about one node
    per request, so a backing offering ``parents_many`` gets a one-element
    frontier N times -- green on every answer test, and paying N queries where
    it should pay one per level.

    Eight roots above one node is the shape that shows it: two levels, nine
    nodes, and eight routes.
    """
    wide: dict[str, tuple[str, ...]] = {"x": tuple(f"p{i}" for i in range(8))}
    wide.update({f"p{i}": () for i in range(8)})

    axis = BulkParents(wide)
    routes = paths_to_root(axis, "x")

    assert len(routes) == 8
    assert axis.bulk_calls == 2, "two levels, so two queries -- not one per node"
    assert axis.widest_frontier == 8, "the whole level went in one request"
    assert axis.singular_calls == 0, "a bulk backing never sees the singular member"


def test_the_async_path_walks_bound_binds() -> None:
    """A keyword a twin accepts and ignores is a surface that lies quietly.

    The bound is real here because the walk reads a frontier: eight parents of
    one node go out in a single bounded round. A route-at-a-time walk awaits
    them one after another, so the widest overlap is one however the bound is
    set -- which reads as *the keyword is taken for parity* rather than as
    *the walk forgot to use a level*.
    """
    wide: dict[str, tuple[str, ...]] = {"x": tuple(f"p{i}" for i in range(8))}
    wide.update({f"p{i}": () for i in range(8)})

    probe = ConcurrencyProbe(wide)
    routes = asyncio.run(async_paths_to_root(probe, "x", max_concurrency=4))

    assert len(routes) == 8
    assert probe.widest_overlap == 4, "the level was read concurrently, within the bound"


def test_the_two_endings_are_told_apart_by_one_call() -> None:
    """The result is routes, not a verdict, and the recovery is stated.

    A path ends at a root or at a node whose every parent is already behind it,
    and the tuple looks the same either way. A caller counting *routes that
    reach a root* over possibly-cyclic data has to ask, so the question that
    answers it is pinned here rather than left in prose.
    """
    #   t -> u -> t   is a cycle, and `u` also has a root above it
    axis = MappingHierarchy(
        {"t": ("u",), "u": ("t", "top"), "top": (), "loop": ("spin",), "spin": ("loop",)}
    )

    reaches_a_root = paths_to_root(axis, "t")
    reaches_nothing = paths_to_root(axis, "loop")

    assert reaches_a_root == (("t", "u", "top"),)
    assert reaches_nothing == (("loop", "spin"),)
    assert axis.parents(reaches_a_root[0][-1]) == (), "empty parents is the root ending"
    assert axis.parents(reaches_nothing[0][-1]) == ("loop",), "non-empty is the cycle ending"


def test_the_guard_is_per_route_and_the_ascent_is_per_node() -> None:
    """Two scopes, and collapsing either one shows up in exactly one line.

    A walk-scoped guard over the *routes* returns one path where a diamond owes
    two. A route-scoped visited set over the *ascent* asks ``root`` once per
    route rather than once. Both are asserted together because a test asserting
    only the answer would pass the second defect and one asserting only the
    count would pass the first.
    """
    hierarchy = CountingParents(DIAMOND)

    routes = paths_to_root(hierarchy, "x")

    assert routes == (("x", "a", "root"), ("x", "b", "root")), "the guard is per route"
    assert hierarchy.singular_calls == 4, "the ascent is per node: `root` is asked once"


def test_a_route_bound_refuses_rather_than_truncating() -> None:
    """The answer's own size is the one cost the ascent cannot bound.

    The fetch is linear in nodes however branchy the axis is; what doubles per
    stacked branch point is the *result*. A caller reading an axis they do not
    control needs a ceiling, and a ceiling that silently returned the first
    ``k`` routes would collapse *there were exactly k ways up* into *there were
    at least k*, which is the pair of answers this module adds a member or a
    refusal to keep apart everywhere else.
    """
    hierarchy = MappingHierarchy(_stacked_diamonds(3))

    assert len(paths_to_root(hierarchy, "n3")) == 8

    with pytest.raises(OperationError) as refusal:
        paths_to_root(hierarchy, "n3", max_paths=4)

    assert refusal.value.context == {"anchor": "n3", "max_paths": 4}


def test_a_bound_at_or_above_the_count_returns_every_route() -> None:
    """The bounded walk is the unbounded one with something to stop it.

    A ceiling the axis stays under changes nothing at all -- so a caller who
    sets one defensively and never reaches it holds exactly the answer they
    would have held without it, rather than a differently-ordered or
    differently-truncated one.
    """
    hierarchy = MappingHierarchy(_stacked_diamonds(3))
    every = paths_to_root(hierarchy, "n3")

    assert paths_to_root(hierarchy, "n3", max_paths=8) == every, "exactly the count"
    assert paths_to_root(hierarchy, "n3", max_paths=9) == every
    assert paths_to_root(hierarchy, "n3", max_paths=10_000) == every


def test_the_bound_refuses_before_building_the_answer_it_would_refuse() -> None:
    """A ceiling checked after enumerating is a ceiling that already spent the memory.

    Twenty stacked diamonds is a sixty-one node axis with 1,048,576 routes up
    it. The ascent is sixty-one requests either way; what this asserts is that
    the *enumeration* stops at the ceiling rather than completing and then
    reporting. Like the termination tests above, the failure mode is one an
    assertion cannot phrase -- an eager enumeration does not fail here, it
    spends a million tuples first, and the test's own runtime is the proof.
    """
    hierarchy = MappingHierarchy(_stacked_diamonds(20))

    assert 2**20 == 1_048_576, "the answer this never builds"

    with pytest.raises(OperationError):
        paths_to_root(hierarchy, "n20", max_paths=10)

    # ...and the same refusal against the projection directly, over an edge map
    # that counts what the enumeration touched. An eager enumeration passes the
    # assertion above and fails this one: it reaches every node of every route
    # before it looks at the ceiling, which is upward of a million lookups
    # against the few hundred a bounded one needs.
    class CountingEdges(dict):  # type: ignore[type-arg]
        reads = 0

        def __getitem__(self, key: object) -> object:
            type(self).reads += 1
            return super().__getitem__(key)

    _, edges = drive(hierarchy, walk_core._expand(("n20",), "parents"))

    with pytest.raises(OperationError):
        walk_core._paths_of(CountingEdges(edges), "n20", 10)

    assert CountingEdges.reads < 1_000, (
        f"a bounded enumeration reads a few hundred edges, not {CountingEdges.reads}"
    )


def test_the_ceiling_bounds_routes_because_depth_is_not_what_explodes() -> None:
    """Why the bound is not ``max_depth``, which is the module's other bound.

    ``descendants_to_depth`` bounds its sibling by depth, so depth is the
    vocabulary a reader expects -- and it is the wrong dimension here. Routes
    multiply with *branching*, not with distance: three levels of ten parents
    each is a thirty-one node axis carrying a thousand routes, all of them
    within ``max_depth=3``. A descent bound would leave that unbounded, and
    would also end each route at a node the walk never asked about, which is
    *unknown* rather than *unextendable*.
    """
    wide: dict[str, tuple[str, ...]] = {"x": tuple(f"p{i}" for i in range(10))}
    wide.update({f"p{i}": tuple(f"q{j}" for j in range(10)) for i in range(10)})
    wide.update({f"q{j}": tuple(f"r{k}" for k in range(10)) for j in range(10)})
    wide.update({f"r{k}": () for k in range(10)})

    hierarchy = MappingHierarchy(wide)
    routes = paths_to_root(hierarchy, "x")

    assert len(wide) == 31, "a small axis"
    assert max(len(route) for route in routes) - 1 == 3, "and a shallow one"
    assert len(routes) == 1000, "...carrying a thousand routes a depth bound would admit"

    with pytest.raises(OperationError):
        paths_to_root(hierarchy, "x", max_paths=100)


def test_a_route_bound_below_one_is_refused() -> None:
    """A ceiling that cannot admit an answer is refused, not clamped.

    ``max_paths=0`` rejects every axis including a bare root, which is a
    guaranteed failure rather than a narrow budget -- the same reason a
    frontier bound of zero is refused rather than raised to one.
    """
    hierarchy = MappingHierarchy(DIAMOND)

    for bound in (0, -1):
        with pytest.raises(ValueError, match="max_paths must be at least 1"):
            paths_to_root(hierarchy, "x", max_paths=bound)


def test_an_unusable_route_bound_is_refused_before_the_ascent_is_spent() -> None:
    """A bound that can admit nothing is wrong at call time, not at emission time.

    ``max_paths=0`` is a guaranteed failure whatever the axis holds, so the
    ascent it would refuse is work already known to be wasted. Refusing it
    after that ascent spends one request per node above the anchor -- over a
    remote backing, an arbitrary number of queries -- to report a mistake in
    the literal the caller typed.

    This is the posture ``_refuse_an_unusable_bound`` states for the frontier
    bound in its own docstring: a width that cannot admit anybody is refused on
    the way past. One rule, so both bounds keep it.
    """
    hierarchy = CountingParents(_stacked_diamonds(5))

    with pytest.raises(ValueError, match="max_paths must be at least 1"):
        paths_to_root(hierarchy, "n5", max_paths=0)

    assert hierarchy.singular_calls == 0, "refused before a single edge was asked for"


def test_both_flavours_refuse_an_unusable_route_bound_before_the_ascent() -> None:
    """The twin refuses on the same terms, and as early."""
    hierarchy = AsyncBulkParents(_stacked_diamonds(5))

    with pytest.raises(ValueError, match="max_paths must be at least 1"):
        asyncio.run(async_paths_to_root(hierarchy, "n5", max_paths=0))

    assert hierarchy.bulk_calls == 0, "refused before a single frontier was asked for"


def test_both_flavours_bound_the_routes_alike() -> None:
    """The ceiling is on the twin too, and refuses on the same axis."""
    axis = _stacked_diamonds(3)

    assert asyncio.run(
        async_paths_to_root(AsyncMappingHierarchy(axis), "n3", max_paths=8)
    ) == paths_to_root(MappingHierarchy(axis), "n3")

    with pytest.raises(OperationError) as refusal:
        asyncio.run(async_paths_to_root(AsyncMappingHierarchy(axis), "n3", max_paths=4))

    assert refusal.value.context == {"anchor": "n3", "max_paths": 4}


# --------------------------------------------------------------------------
# The eighth reading -- one ascent seeded by both, and what `deepest` means
# --------------------------------------------------------------------------


def test_the_deepest_common_ancestor_is_the_nearest_node_above_both() -> None:
    """Nearest, not any: a chain gives a whole set of common ancestors."""
    hierarchy = MappingHierarchy({"top": (), "mid": ("top",), "left": ("mid",), "right": ("mid",)})

    assert deepest_common_ancestor(hierarchy, "left", "right") == "mid"
    assert ancestors(hierarchy, "left") == ("mid", "top"), "`top` is common too, and higher"


def test_deepest_is_not_nearest_once_the_axis_has_a_shortcut_edge() -> None:
    """The defect a tree cannot show, and the reason this is not a distance.

    ``beagle`` names ``mammal`` directly *and* reaches it the long way through
    ``hound``, ``dog`` and ``canine``. Read as a distance from ``beagle``,
    ``mammal`` is one hop and ``dog`` is two, so *the nearest common ancestor*
    answers ``mammal`` -- while ``dog`` is a common ancestor of both arguments
    standing strictly **below** it. The two candidates are perfectly
    comparable, which is the case the asymmetric tie-break tells a reader is
    safe, so nothing about the published contract warned that this could
    happen.

    The shape is ordinary rather than pathological: a term asserted under both
    a narrow and a broad category is what a ``broader`` relation collects, and
    a consumer reaching for the most specific shared category would have been
    handed the least specific one.
    """
    hierarchy = MappingHierarchy(
        {
            "beagle": ("mammal", "hound"),
            "hound": ("dog",),
            "dog": ("canine",),
            "canine": ("mammal",),
            "puppy": ("dog",),
            "mammal": (),
        }
    )

    assert deepest_common_ancestor(hierarchy, "beagle", "puppy") == "dog"
    assert ancestors(hierarchy, "beagle")[0] == "mammal", (
        "`mammal` is nearest by distance, and is the answer a level-ordered "
        "chain read first-match-wins returns"
    )
    assert "mammal" in ancestors(hierarchy, "dog"), "...and it stands above `dog`"


def test_the_answer_does_not_depend_on_the_order_a_reply_names_parents_in() -> None:
    """A tie-break is a decision; a reply's row order deciding the answer is not.

    Reading a level-ordered chain first-match-wins puts both of ``beagle``'s
    parents at distance one, so whichever the backing named first won -- and a
    query over a join does not promise an order. Minimality is a property of
    the axis, so reversing the reply must not move the answer.
    """
    forward = {
        "beagle": ("mammal", "dog"),
        "dog": ("mammal",),
        "puppy": ("dog",),
        "mammal": (),
    }
    reversed_reply = {**forward, "beagle": ("dog", "mammal")}

    assert deepest_common_ancestor(MappingHierarchy(forward), "beagle", "puppy") == "dog"
    assert deepest_common_ancestor(MappingHierarchy(reversed_reply), "beagle", "puppy") == "dog"


def test_either_argument_may_be_the_answer() -> None:
    """*Deepest common ancestor*, not *deepest proper common ancestor*."""
    hierarchy = MappingHierarchy(DIAMOND)

    assert deepest_common_ancestor(hierarchy, "a", "x") == "a"
    assert deepest_common_ancestor(hierarchy, "x", "a") == "a"


def test_the_tie_break_is_asymmetric_over_a_dag() -> None:
    """Swapping the arguments can swap the answer, and that is the definition.

    A real tie needs two **minimal** common ancestors -- neither above the
    other, so there is nothing to choose between them on depth -- and then the
    rule picks the one nearer the *first* argument. The fixture makes the tie
    by giving both nodes the same two roots in opposite orders, so neither
    argument's chain agrees with the other's about which comes first.
    """
    hierarchy = MappingHierarchy(
        {
            "l": ("shared_l", "shared_r"),
            "r": ("shared_r", "shared_l"),
            "shared_l": (),
            "shared_r": (),
        }
    )

    assert deepest_common_ancestor(hierarchy, "l", "r") == "shared_l"
    assert deepest_common_ancestor(hierarchy, "r", "l") == "shared_r"
    assert ancestors(hierarchy, "shared_l") == (), "neither candidate stands above the other"
    assert ancestors(hierarchy, "shared_r") == ()


def test_a_cycle_is_a_tie_rather_than_an_exclusion() -> None:
    """Mutual ancestry has no deepest member, so it falls through to the tie-break.

    Two nodes on one cycle are each strictly above the other. A minimality rule
    reading only *is some other common ancestor below me* drops both and
    answers ``None`` where a common ancestor plainly exists, so the rule reads
    *below me, and I am not also below it* -- which makes mutual ancestry a tie.
    """
    hierarchy = MappingHierarchy({"p": ("q",), "q": ("p",), "x": ("p",), "y": ("q",)})

    assert deepest_common_ancestor(hierarchy, "x", "y") == "p", "`x`'s chain reaches `p` first"
    assert deepest_common_ancestor(hierarchy, "y", "x") == "q"


def test_an_unknown_argument_is_refused_because_this_walk_emits_it() -> None:
    """Either argument may be the answer, which makes both of them anchors.

    This answered ``None`` for an unknown node and the ambiguity was defended
    the way :func:`ancestors`' is -- nothing false returned, one ``contains``
    call to resolve it. That held for two *different* unknown nodes and failed
    for the same one twice: both chains contained it, so the intersection did
    too and it came back as a term of the axis. A walk that can emit an
    argument has to know it.
    """
    hierarchy = MappingHierarchy({"p": (), "q": ()})

    assert deepest_common_ancestor(hierarchy, "p", "q") is None, "disjoint is still `None`"

    for a, b in (("marmoset", "p"), ("p", "marmoset"), ("marmoset", "marmoset")):
        with pytest.raises(NotFoundError) as refusal:
            deepest_common_ancestor(hierarchy, a, b)
        assert refusal.value.context == {"anchor": "marmoset"}

    assert (
        asyncio.run(async_deepest_common_ancestor(AsyncMappingHierarchy({"p": ()}), "p", "p"))
        == "p"
    )
    with pytest.raises(NotFoundError):
        asyncio.run(
            async_deepest_common_ancestor(AsyncMappingHierarchy({"p": ()}), "marmoset", "p")
        )


def test_one_ascent_seeded_by_both_asks_each_node_once() -> None:
    """The cost the composition published rather than fixed, now absent.

    Two composed ancestor walks asked the ancestry the two nodes share twice --
    44 requests against 23 distinct edges here -- because ``yield from``
    delegates a request past the composing frame and there is nowhere to keep
    what comes back. One descent seeded by both asks each node once, and each
    argument's own chain is recovered from the replies at no request.

    A caller's cache is asserted to change nothing, which is the whole claim:
    the walk no longer needs one to reach this count.
    """
    depth = 20
    axis: dict[str, tuple[str, ...]] = {"r0": ()}
    for level in range(1, depth + 1):
        axis[f"r{level}"] = (f"r{level - 1}",)
    axis["x"] = (f"r{depth}",)
    axis["y"] = (f"r{depth}",)

    plain = CountingParents(axis)
    cached = CountingParents(axis)

    assert deepest_common_ancestor(plain, "x", "y") == f"r{depth}"
    assert deepest_common_ancestor(cached, "x", "y", cache={}) == f"r{depth}"

    assert len(axis) == 23
    assert plain.singular_calls == 23, "each node above either argument, once"
    assert cached.singular_calls == plain.singular_calls, "a cache has nothing to save"


def test_a_repeated_seed_does_not_lose_an_argument_to_the_visited_set() -> None:
    """``a == b`` is a fair question, and the descent records against the first reach.

    :func:`_expand` records discovery against the first node to reach
    something, so a frontier carrying the same seed twice records the second
    one as reaching nothing -- the failure ``_seeds_under`` dedups ``roots()``
    to avoid. This walk seeds a descent with both arguments, so it dedups them.
    """
    hierarchy = CountingParents(DIAMOND)

    assert deepest_common_ancestor(hierarchy, "x", "x") == "x"
    assert hierarchy.singular_calls == 4, "four nodes, asked once each"
    assert hierarchy.contains_calls == 1, "one anchor, so one containment question"


def test_the_ascent_seeded_by_both_asks_a_bulk_backing_one_query_per_level() -> None:
    """The eighth walk on the frontier path, which a singular counter cannot see.

    ``CountingParents`` has no bulk members, so every count test above it
    measures the fan-out path. The claim *one frontier per level* is about the
    other one: two seeds go up as a single widening frontier rather than as two
    ascents, and a backing that answers a level in one query sees one query.
    """
    bulk = BulkParents(_stacked_diamonds(3))

    assert deepest_common_ancestor(bulk, "a2", "b2") == "n2"

    # Six levels: {a2,b2} {n2} {a1,b1} {n1} {a0,b0} {n0}. The ascent runs to the
    # root rather than stopping at the answer -- minimality is a question about
    # the whole shared ancestry, so there is no level it can stop early at.
    assert bulk.bulk_calls == 6, "one query per level of the shared ascent"
    assert bulk.widest_frontier == 2, "both arguments in one frontier"
    assert bulk.singular_calls == 0, "and nothing fell back to per-node asks"


# --------------------------------------------------------------------------
# The delegation -- one core, two surfaces
# --------------------------------------------------------------------------


#: The key a patched core answers with, and nothing else can.
#:
#: A *value* assertion rather than an identity one, because four of the six
#: surfaces re-wrap what the core returns and a re-wrapped tuple is a new
#: object. What makes the value sound is that it is not a node: no walk over
#: ``CYCLIC_PARENTS`` can produce it, so an unpatched surface cannot answer
#: with it by agreeing rather than by delegating -- which is the reason the
#: sentinel is not ``None``.
_PATCHED: tuple[str, ...] = ("__the_core_was_patched__",)


def _nodes(views: Sequence[Any]) -> tuple[str, ...]:
    """The keys a tuple of cursors is anchored on."""
    return tuple(view.node for view in views)


def _as_axis(structure: Hierarchy[str]) -> Taxonomy:
    """Any structure, in a taxonomy-shaped slot, with an empty content axis.

    The taxonomy cursor's walk members forward to the structural cursor's, so
    what a delegation test needs from the axis is the structure and nothing
    else -- an entity source that carries nothing answers every question these
    surfaces ask.
    """
    return Taxonomy(
        definition=TaxonomyDefinition(id="delegation", relation="isa"),
        structure=structure,
        entities=MappingEntitySource({}),
    )


def _as_async_axis(structure: AsyncHierarchy[str]) -> AsyncTaxonomy:
    """:func:`_as_axis`'s twin."""
    return AsyncTaxonomy(
        definition=TaxonomyDefinition(id="delegation", relation="isa"),
        structure=structure,
        entities=AsyncMappingEntitySource({}),
    )


def test_the_snapshot_walk_is_a_reading_of_the_shared_descent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim this package makes everywhere, asserted of the walk that made it false.

    ``_parent_edges`` is driven by both snapshot constructors, so it is a walk
    this package ships -- and it ran its own frontier loop, its own visited set
    and its own ``roots`` fetch, which is a second expansion beside the one
    every docstring here says is the only one. Equality of snapshots cannot
    tell a reading of the descent from a copy of it that agrees today, which is
    the same reason ``_CORES`` exists for the eight public walks.

    It could not have been a reading before: a spanning record drops the second
    parent of a DAG node, and that edge is the one thing a parent-edge snapshot
    cannot lose. Keeping the whole reply is what made this possible, so the
    walk that motivated the change is the walk that now demonstrates it.

    **Not a ``_CORES`` row, and the shape is why.** Every surface in that table
    answers *with* what the core returned, so a sentinel travels out unchanged
    and equality catches it. ``snapshot`` answers with a ``MappingHierarchy``
    built *around* the reply, so a sentinel would be wrapped rather than
    returned -- and the two things this asserts beyond delegation, that the
    descent is seeded once from the roots and that both parents of a DAG node
    survive it, are claims no sentinel can make. It is kept here for the same
    reason the table exists and in the shape that walk needs.
    """
    seen: list[tuple[str, tuple[str, ...]]] = []
    unpatched = walk_core._expand

    def recording_expand(
        seeds: tuple[str, ...], direction: str, **kwargs: Any
    ) -> Generator[Any, Any, Any]:
        seen.append((direction, tuple(seeds)))
        return unpatched(seeds, direction, **kwargs)

    monkeypatch.setattr(walk_core, "_expand", recording_expand)

    # A walk-only axis: `MappingHierarchy` publishes `parent_edges`, so
    # snapshotting one takes the enumerable branch and walks nothing at all.
    snapshot = MappingHierarchy.snapshot(MappingParents(DIAMOND))

    assert seen == [("children", ("root",))], "one descent, seeded by the roots"
    assert snapshot.parents("x") == ("a", "b"), "...and both parents of the DAG node kept"


#: Each core this module wraps, and one call into **every** surface written
#: over it.
#:
#: Parametrised over **all eight** walks rather than over one. A parametrised
#: guard run over a set of one is green for the same reason an empty one is,
#: and this table was that until the six compositions existed -- so a walk
#: added without a row here is a walk whose delegation nothing checks.
#:
#: **And over every surface rather than the two module functions**, which is
#: the other axis and was a set of two until the cursors gained walk members.
#: Five of the eight walks now have six surfaces -- two module functions, the
#: structural cursor, the taxonomy cursor, and both asynchronous twins -- and
#: a cursor member that reimplemented the traversal would pass every answer
#: test in this file while drifting from the core it claims to call. The other
#: three have two surfaces because no cursor member is declared over them --
#: ``deepest_common_ancestor`` takes two anchors and a cursor names one, and
#: ``flatten`` and ``leaves`` take an *optional* anchor that defaults to every
#: root, so neither reads the cursor's node as *where you are* -- and a row
#: claiming six for those would be asserting a member nobody declared.
#:
#: The last two walk rows are the ones a reader should expect to be absent.
#: They arrived as separate algorithms and were re-read as projections of the
#: same descent, which is the point in a walk's life where a wrapper quietly
#: keeps the old traversal. That is exactly the claim equality of results
#: cannot make.
_CORES: tuple[tuple[str, tuple[Callable[..., Any], ...], tuple[Callable[..., Any], ...]], ...] = (
    (
        "_ancestors",
        (
            lambda h: ancestors(h, "f"),
            lambda h: _nodes(HierarchyView(h, "f").ancestors()),
            lambda h: _nodes(_as_axis(h).at("f").ancestors()),
        ),
        (
            lambda h: async_ancestors(h, "f"),
            lambda h: _async_nodes(AsyncHierarchyView(h, "f").ancestors()),
            lambda h: _async_nodes(_as_async_axis(h).at("f").ancestors()),
        ),
    ),
    (
        "_descendants",
        (
            lambda h: descendants(h, "f"),
            lambda h: _nodes(HierarchyView(h, "f").descendants()),
            lambda h: _nodes(_as_axis(h).at("f").descendants()),
        ),
        (
            lambda h: async_descendants(h, "f"),
            lambda h: _async_nodes(AsyncHierarchyView(h, "f").descendants()),
            lambda h: _async_nodes(_as_async_axis(h).at("f").descendants()),
        ),
    ),
    (
        "_descendants_to_depth",
        (
            lambda h: descendants_to_depth(h, "f", 2),
            lambda h: _nodes(HierarchyView(h, "f").descendants_to_depth(2)),
            lambda h: _nodes(_as_axis(h).at("f").descendants_to_depth(2)),
        ),
        (
            lambda h: async_descendants_to_depth(h, "f", 2),
            lambda h: _async_nodes(AsyncHierarchyView(h, "f").descendants_to_depth(2)),
            lambda h: _async_nodes(_as_async_axis(h).at("f").descendants_to_depth(2)),
        ),
    ),
    (
        "_paths_to_root",
        (
            lambda h: paths_to_root(h, "f"),
            lambda h: HierarchyView(h, "f").paths_to_root(),
            lambda h: _as_axis(h).at("f").paths_to_root(),
        ),
        (
            lambda h: async_paths_to_root(h, "f"),
            lambda h: AsyncHierarchyView(h, "f").paths_to_root(),
            lambda h: _as_async_axis(h).at("f").paths_to_root(),
        ),
    ),
    (
        "_children_at_depth",
        (
            lambda h: children_at_depth(h, "f", 2),
            lambda h: _nodes(HierarchyView(h, "f").children_at_depth(2)),
            lambda h: _nodes(_as_axis(h).at("f").children_at_depth(2)),
        ),
        (
            lambda h: async_children_at_depth(h, "f", 2),
            lambda h: _async_nodes(AsyncHierarchyView(h, "f").children_at_depth(2)),
            lambda h: _async_nodes(_as_async_axis(h).at("f").children_at_depth(2)),
        ),
    ),
    ("_flatten", (flatten,), (async_flatten,)),
    ("_leaves", (leaves,), (async_leaves,)),
    (
        "_deepest_common_ancestor",
        (lambda h: deepest_common_ancestor(h, "c", "d"),),
        (lambda h: async_deepest_common_ancestor(h, "c", "d"),),
    ),
)


async def _async_nodes(awaitable: Any) -> tuple[str, ...]:
    """:func:`_nodes` over a coroutine that answers with cursors."""
    return _nodes(await awaitable)


def test_a_snapshot_forwards_a_cache_so_a_later_walk_spends_its_replies() -> None:
    """The constructor drives a walk, so it takes the walk's memo.

    ``snapshot`` is the one ``drive()`` call site in this module that is not a
    walk, and it was the one that forwarded no ``cache=`` -- the drift the
    module docstring names for ``max_concurrency`` and the memo closed for the
    twelve walks, reopened at the one construct kind a census over *walks*
    cannot see.

    **The reproduction is a count rather than an exception**, which is what
    makes it the kind of defect a green suite keeps: every answer is identical
    either way. Thirteen nodes, each asked for its children once by the
    snapshot and once again by the walk after it; with one memo across both the
    second asks nothing.
    """
    axis = _stacked_diamonds(4)
    assert len(axis) == 13, "the count below is the axis's size, so it is stated"

    cold = CountingParents(axis)
    MappingHierarchy.snapshot(cold)
    uncached_leaves = leaves(cold)
    assert cold.singular_calls == 26, "thirteen nodes asked twice"

    warm = CountingParents(axis)
    memo: dict[tuple[str, Any], Sequence[str]] = {}
    snapshot = MappingHierarchy.snapshot(warm, cache=memo)

    assert leaves(warm, cache=memo) == uncached_leaves, "the memo changed an answer"
    assert warm.singular_calls == 13, "thirteen nodes asked once"
    assert snapshot.parent_edges() == MappingHierarchy.snapshot(cold).parent_edges()

    # And the direction the docstring rules out: what a snapshot holds is
    # `children` replies, so an ascending walk after one asks `parents` as it
    # would have. Asserted rather than implied -- a memo keyed without the
    # member name would satisfy every line above and silently answer an
    # ascent from a descent's replies.
    ascending = CountingParents(axis)
    memo_from_a_descent: dict[tuple[str, Any], Sequence[str]] = {}
    MappingHierarchy.snapshot(ascending, cache=memo_from_a_descent)
    after_snapshot = ascending.singular_calls

    ancestors(ascending, "n4", cache=memo_from_a_descent)

    assert ascending.singular_calls > after_snapshot, (
        "the ascent asked `parents`, which a descent's memo cannot hold"
    )


@pytest.mark.asyncio
async def test_the_async_snapshot_forwards_it_too() -> None:
    """The twin, because a parameter on one half is the drift this closes.

    ``max_concurrency`` was already here and ``cache`` was not, which is the
    asymmetry rather than a second one: the asynchronous constructor had the
    keyword its flavour needs and neither had the keyword both flavours need.
    """
    axis = _stacked_diamonds(4)

    cold = AsyncCountingParents(axis)
    await AsyncMappingHierarchy.snapshot(cold)
    uncached_leaves = await async_leaves(cold)
    assert cold.singular_calls == 26, "thirteen nodes asked twice"

    warm = AsyncCountingParents(axis)
    memo: dict[tuple[str, Any], Sequence[str]] = {}
    await AsyncMappingHierarchy.snapshot(warm, cache=memo)

    assert await async_leaves(warm, cache=memo) == uncached_leaves
    assert warm.singular_calls == 13, "thirteen nodes asked once"


@pytest.mark.parametrize(
    ("core", "sync_calls", "async_calls"), _CORES, ids=[name for name, _, _ in _CORES]
)
def test_patching_the_core_moves_every_surface_over_it(
    monkeypatch: pytest.MonkeyPatch,
    core: str,
    sync_calls: tuple[Callable[..., Any], ...],
    async_calls: tuple[Callable[..., Any], ...],
) -> None:
    """Every surface *invokes* the shared generator rather than agreeing with it.

    Equality of results cannot make this claim. A surface that inlined the
    traversal would return the same answer on the day it was written and drift
    a year later, silently, which is the failure this shape exists to catch --
    and a cursor member is where it is likeliest, because re-walking is one
    line and forwarding is one line.

    Patching the **core** rather than the public walk is the point: the public
    walk is per flavour, so patching it would not move the twin, while the
    thing both are written over is below both.
    """

    def sentinel_walk(*_args: object, **_kwargs: object) -> object:
        """A walk that asks nothing and returns a value nothing else produces.

        A generator rather than ``iter(())``, and returning a key rather than
        ``None``, for the same reason in both halves: ``None`` out of an empty
        iterator is an answer the *unpatched* driver could also give, so it
        proves the surfaces agree rather than that the patch was reached.

        It takes whatever it is handed, because the cores do not share a
        signature -- a bound, a depth and an optional anchor are three
        different shapes and the patch is about neither.
        """

        def _asks_nothing() -> object:
            return _PATCHED
            yield  # unreachable, and what makes this a generator

        return _asks_nothing()

    monkeypatch.setattr(hierarchy_module, core, sentinel_walk)

    for surface in sync_calls:
        assert surface(MappingParents(CYCLIC_PARENTS)) == _PATCHED
    for async_surface in async_calls:
        assert asyncio.run(async_surface(AsyncMappingParents(CYCLIC_PARENTS))) == _PATCHED


# --------------------------------------------------------------------------
# The keywords a cursor member carries, and whether it forwards them
# --------------------------------------------------------------------------
#
# ``assert_twin_types_agree`` compares parameter names, annotations and
# defaults, so it fails a member that **drops** a keyword from its signature.
# It cannot see a member that declares one and never passes it on, and
# ``_CORES`` cannot either: those surfaces are called with no keywords at all.
# So a member accepting ``max_paths=`` and forwarding nothing turns a
# documented refusal into an unbounded enumeration while every other test in
# this file stays green -- which is the shape the module docstring names for
# ``max_concurrency`` and ``snapshot`` closed for the constructor.


#: Each walk-shaped cursor member, with the positional arguments it needs.
#:
#: All five take ``cache=``, so one table drives the memo test over every one
#: of them; the two bounds are their own tests because only one member takes
#: each.
_CURSOR_WALK_MEMBERS: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("ancestors", ()),
    ("descendants", ()),
    ("descendants_to_depth", (2,)),
    ("children_at_depth", (2,)),
    ("paths_to_root", ()),
)


@pytest.mark.parametrize(
    ("member", "args"), _CURSOR_WALK_MEMBERS, ids=[name for name, _ in _CURSOR_WALK_MEMBERS]
)
def test_a_cursor_walk_member_spends_the_memo_it_is_handed(
    member: str, args: tuple[Any, ...]
) -> None:
    """The memo reaches the walk, on the structural cursor and the taxonomy one.

    **A count rather than an answer**, for :func:`snapshot`'s reason: a member
    that accepted ``cache=`` and dropped it returns exactly what it returns
    now, so nothing about the result can carry this claim. The second call
    asking the backing nothing is what the forward buys, and it is the only
    observable difference between forwarding and not.
    """
    axis = _stacked_diamonds(4)

    for build in (lambda b: HierarchyView(b, "n2"), lambda b: _as_axis(b).at("n2")):
        backing = CountingParents(axis)
        cursor = build(backing)
        memo: dict[tuple[str, Any], Sequence[str]] = {}

        getattr(cursor, member)(*args, cache=memo)
        after_first = backing.singular_calls
        assert after_first > 0, "the walk asked the backing at all, so there is something to save"

        getattr(cursor, member)(*args, cache=memo)
        assert backing.singular_calls == after_first, "the second walk spent the memo"


@pytest.mark.parametrize(
    ("member", "args"), _CURSOR_WALK_MEMBERS, ids=[name for name, _ in _CURSOR_WALK_MEMBERS]
)
def test_an_async_cursor_walk_member_spends_the_memo_too(
    member: str, args: tuple[Any, ...]
) -> None:
    """The twin, because a keyword forwarded on one flavour is the drift this closes."""
    axis = _stacked_diamonds(4)

    for build in (
        lambda b: AsyncHierarchyView(b, "n2"),
        lambda b: _as_async_axis(b).at("n2"),
    ):
        backing = AsyncCountingParents(axis)
        cursor = build(backing)
        memo: dict[tuple[str, Any], Sequence[str]] = {}

        asyncio.run(getattr(cursor, member)(*args, cache=memo))
        after_first = backing.singular_calls
        assert after_first > 0

        asyncio.run(getattr(cursor, member)(*args, cache=memo))
        assert backing.singular_calls == after_first, "the second walk spent the memo"


def test_a_cursor_forwards_the_route_ceiling_rather_than_declaring_it() -> None:
    """``max_paths`` reaches the walk on all four cursors.

    Four stacked diamonds carry sixteen maximal routes, so a ceiling of four
    is one the axis exceeds -- and a member that declared the keyword and
    forwarded nothing would return all sixteen here rather than raising. That
    is the failure this asserts against: an unbounded enumeration where the
    caller asked for a bound, reported as a correct answer.
    """
    axis = _stacked_diamonds(4)

    assert len(paths_to_root(MappingParents(axis), "n4")) == 16, "the ceiling below is exceeded"

    for cursor in (
        HierarchyView(MappingParents(axis), "n4"),
        _as_axis(MappingParents(axis)).at("n4"),
    ):
        with pytest.raises(OperationError):
            cursor.paths_to_root(max_paths=4)

    for async_cursor in (
        AsyncHierarchyView(AsyncMappingParents(axis), "n4"),
        _as_async_axis(AsyncMappingParents(axis)).at("n4"),
    ):
        with pytest.raises(OperationError):
            asyncio.run(async_cursor.paths_to_root(max_paths=4))


@pytest.mark.parametrize(
    ("member", "args"),
    _CURSOR_WALK_MEMBERS,
    ids=[name for name, _ in _CURSOR_WALK_MEMBERS],
)
def test_an_async_cursor_forwards_the_frontier_bound(member: str, args: tuple[Any, ...]) -> None:
    """``max_concurrency`` reaches the walk on both asynchronous cursors -- including
    :meth:`paths_to_root`, which a docstring once called inert.

    The ascent that walk makes is the one :meth:`ancestors` makes, and the
    bound is on the ascent: eight parents of one node go out in one bounded
    round, so a bound of four is visible as an overlap of four. It is
    ``max_paths`` that is reached afterwards, from replies already in hand --
    two bounds, and they bound different things.

    ``ConcurrencyProbe`` is singular-only deliberately; give it bulk members
    and every assertion here goes quiet without failing.
    """
    wide: dict[str, tuple[str, ...]] = {"x": tuple(f"p{i}" for i in range(8))}
    wide.update({f"p{i}": () for i in range(8)})
    # A node under ``x`` as well, so the descending members have a frontier of
    # their own to bound rather than answering from the anchor alone.
    wide.update({f"c{i}": ("x",) for i in range(8)})

    for build in (lambda b: AsyncHierarchyView(b, "x"), lambda b: _as_async_axis(b).at("x")):
        probe = ConcurrencyProbe(wide)

        asyncio.run(getattr(build(probe), member)(*args, max_concurrency=4))

        assert probe.widest_overlap == 4, f"{member} read a level within the bound it was given"


# --------------------------------------------------------------------------
# Parity, the protocol check, and the key parameter
# --------------------------------------------------------------------------


#: Every member a hierarchy twin pair must expose identically.
#:
#: Listed rather than discovered, so that a member added to one flavour and not
#: the other fails here. A comparison that walked whatever both already had
#: would go quiet on exactly that case.
_HIERARCHY_MEMBERS = ("roots", "parents", "children", "contains")


@pytest.mark.parametrize(
    ("sync_type", "async_type"),
    [
        (Hierarchy, AsyncHierarchy),
        (AssertionHierarchy, AsyncAssertionHierarchy),
        (MappingHierarchy, AsyncMappingHierarchy),
    ],
)
def test_the_twins_expose_the_same_annotated_surface(sync_type: type, async_type: type) -> None:
    """Same member names, same parameters, same annotations, same return type.

    ``compare_return=True`` because none of these members streams: each returns
    the same type in both flavours, so a difference there is contract drift
    rather than the flavour showing through. ``Taxonomy.walk`` is where that
    stops being true, and its own test says so.
    """
    assert_twin_types_agree(sync_type, async_type, _HIERARCHY_MEMBERS, compare_return=True)


#: The two members a bulk pair adds. Listed for the reason the four are.
_BULK_MEMBERS = ("parents_many", "children_many")


@pytest.mark.parametrize(
    ("sync_type", "async_type"),
    [
        (BulkHierarchy, AsyncBulkHierarchy),
        (MappingHierarchy, AsyncMappingHierarchy),
    ],
)
def test_the_bulk_twins_expose_the_same_annotated_surface(
    sync_type: type, async_type: type
) -> None:
    """The optional capability is twinned on the same terms as the required one.

    Checked on the protocol pair *and* on the first non-assertion adopter of it,
    because a concrete is where the two halves are written out by hand and
    therefore where they can disagree.
    """
    assert_twin_types_agree(sync_type, async_type, _BULK_MEMBERS, compare_return=True)


#: The member an enumerable pair adds. Listed for the reason the others are.
_ENUMERABLE_MEMBERS = ("parent_edges",)


@pytest.mark.parametrize(
    ("sync_type", "async_type"),
    [
        (EnumerableHierarchy, AsyncEnumerableHierarchy),
        (MappingHierarchy, AsyncMappingHierarchy),
        (AssertionHierarchy, AsyncAssertionHierarchy),
    ],
)
def test_the_enumerable_twins_expose_the_same_annotated_surface(
    sync_type: type, async_type: type
) -> None:
    """Both adopters, because this member decides how complete a copy is.

    A flavour whose ``parent_edges`` disagreed with its twin's would make
    ``materialized`` mean one thing through the synchronous door and another
    through the asynchronous one, which is the failure a parity check over a
    capability is for.
    """
    assert_twin_types_agree(sync_type, async_type, _ENUMERABLE_MEMBERS, compare_return=True)


#: The query pair every read on an assertion-backed axis goes through.
#:
#: Private, and checked anyway. It is the one place each twin says which
#: relation it is and that a negated edge is not part of the axis -- written
#: out twice because the twins share no runtime code -- so a difference here
#: is the two flavours disagreeing about what the axis *contains*, which no
#: public-surface comparison would report.
_QUERY_MEMBERS = ("_find", "_find_many")


def test_the_assertion_axis_query_helpers_are_twins() -> None:
    """Same parameters, same annotations, same return type on both flavours."""
    assert_twin_types_agree(
        AssertionHierarchy, AsyncAssertionHierarchy, _QUERY_MEMBERS, compare_return=True
    )


def test_the_snapshot_constructors_are_twins() -> None:
    """One stated difference, and it is the one every async entry point has.

    ``max_concurrency`` bounds a frontier read with no bulk member to use, and
    the synchronous driver issues no concurrent calls at all. ``hierarchy`` is
    flavoured by definition. The return type differs by flavour too -- each
    lands in its own slot -- so ``compare_return`` stays off here where the
    member comparisons above have it on.
    """
    assert_twins_agree(
        MappingHierarchy.snapshot,
        AsyncMappingHierarchy.snapshot,
        async_only={"max_concurrency"},
        flavour_typed={"hierarchy"},
    )


def test_the_nested_constructors_take_the_same_arguments() -> None:
    """Both are plain ``def``, so the twin guard is the wrong instrument.

    ``assert_twins_agree`` refuses a synchronous "async half", which is exactly
    right and exactly why it cannot be used here: a tree held in memory has
    nothing to await, so making the asynchronous constructor awaitable would
    cost every caller an ``await`` for a traversal of their own data. What still
    has to hold is that the two take the same arguments, so that is asserted
    directly.
    """
    assert not inspect.iscoroutinefunction(AsyncMappingHierarchy.from_nested)
    assert inspect.signature(MappingHierarchy.from_nested).parameters == (
        inspect.signature(AsyncMappingHierarchy.from_nested).parameters
    )


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
    checked rather than left as a convention. The taxonomy module was in this
    probe once and is not now: it lives under ``ontology/``, because its
    cursor needs the model at runtime -- the same reason the assertion backing
    does.

    **The probe stubs the package door, and must.** ``dataknobs_common``
    publishes the vocabulary surface, so its ``__init__`` imports
    ``dataknobs_common.ontology`` -- and importing a submodule runs the
    parent's ``__init__`` first. Left to run, the door puts the vocabulary in
    ``sys.modules`` whatever this module does, and the probe answers a
    question about the door instead of about the edge. A bare parent module
    carrying the real ``__path__`` is all the import machinery needs, and it
    leaves ``__init__`` unrun.

    The second half is the control, and it names a module the probe was not
    handed. Pointing the probe at ``dataknobs_common.ontology`` and asking
    whether it reached ``dataknobs_common.ontology`` would pass on an
    instrument that can see nothing but its own argument -- which is the one
    failure that would make the absence above vacuous. ``ontology.taxonomy``
    reaches this module only through an import chain, so a stub that broke the
    measurement rather than the door reports ``False`` here.
    """
    import subprocess
    import sys

    def reaches(module: str, target: str) -> str:
        probe = (
            "import importlib, importlib.util, sys, types; "
            "_spec = importlib.util.find_spec('dataknobs_common'); "
            "_stub = types.ModuleType('dataknobs_common'); "
            "_stub.__path__ = list(_spec.submodule_search_locations); "
            "_stub.__spec__ = _spec; "
            "sys.modules['dataknobs_common'] = _stub; "
            f"importlib.import_module({module!r}); "
            f"print({target!r} in sys.modules)"
        )
        return subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        ).stdout.strip()

    assert reaches("dataknobs_common.hierarchy", "dataknobs_common.ontology") == "False"
    assert reaches("dataknobs_common.ontology.taxonomy", "dataknobs_common.hierarchy") == "True"


# --------------------------------------------------------------------------
# The walk's own exhaustion, and the width of a frontier read
# --------------------------------------------------------------------------


def test_driving_a_spent_walk_is_refused_rather_than_answered() -> None:
    """A second drive of one generator must not answer ``None`` as a result.

    A walk is single-use and the drivers did not say so. ``drive`` opens with
    ``next(walk)``; on an exhausted generator that raises ``StopIteration``
    with ``value=None``, which the driver's own ``except`` reads as the walk
    returning and hands back cast to the declared type. The caller then holds a
    ``None`` typed ``tuple[str, ...]`` and fails at ``for x in None``, nowhere
    near the driver that produced it.

    Distinct from the collaborator's ``StopIteration`` pinned above, which is
    an exception raised by the *hierarchy*. This is the walk's own exhaustion,
    which is not an error condition anywhere else -- so it is checked before
    the ``try`` rather than converted inside it.
    """
    hierarchy = MappingParents({"a": (), "b": ("a",)})
    walk = _asking_for("parents")

    assert drive(hierarchy, walk) == ()

    with pytest.raises(RuntimeError, match="already been driven"):
        drive(hierarchy, walk)


def test_both_flavours_refuse_a_spent_walk_alike() -> None:
    """The twins raise the same type, which is the property the pair rests on."""
    hierarchy = AsyncMappingParents({"a": (), "b": ("a",)})
    walk = _asking_for("parents")

    assert asyncio.run(async_drive(hierarchy, walk)) == ()

    with pytest.raises(RuntimeError, match="already been driven"):
        asyncio.run(async_drive(hierarchy, walk))


class DelegatingWalk(
    Generator[tuple[str, tuple[str, ...]], tuple[Sequence[str], ...], tuple[str, ...]]
):
    """A walk by protocol rather than by construction, wrapping a real one.

    ``Walk`` is spelled ``Generator[...]``, and ``collections.abc.Generator``
    is satisfiable by a class -- so this is what a consumer reaching for an
    instrumented or filtered walk writes first. The drivers themselves would
    drive it: they only ever ``next`` and ``send``, which this forwards.
    """

    def __init__(
        self,
        inner: Generator[tuple[str, tuple[str, ...]], tuple[Sequence[str], ...], tuple[str, ...]],
    ) -> None:
        self._inner = inner

    def send(self, value: tuple[Sequence[str], ...]) -> tuple[str, tuple[str, ...]]:
        return self._inner.send(value)

    def throw(self, *args: object, **kwargs: object) -> tuple[str, tuple[str, ...]]:
        return self._inner.throw(*args, **kwargs)  # type: ignore[arg-type]


def test_driving_a_suspended_walk_is_refused_rather_than_answered() -> None:
    """A walk stopped mid-question is refused, and exhaustion does not cover it.

    The guard reads ``!= GEN_CREATED`` rather than ``== GEN_CLOSED``, and this
    is the half of that which a second drive of a finished walk cannot reach.
    It is also the worse half: an exhausted walk sends ``None`` where a
    *result* belongs, where a suspended one sends ``None`` where the reply to
    its outstanding question belongs -- so it would answer, out of a frontier
    it never read, rather than fail.

    Both flavours are asserted against the *same* suspended walk, which the
    first refusal leaves untouched: the guard runs before the driver's opening
    ``next``, so nothing has advanced it.
    """
    walk = _asking_for("parents")

    assert next(walk) == ("parents", ("anything",))
    assert inspect.getgeneratorstate(walk) == inspect.GEN_SUSPENDED

    with pytest.raises(RuntimeError, match=inspect.GEN_SUSPENDED):
        drive(MappingParents({"a": (), "b": ("a",)}), walk)

    with pytest.raises(RuntimeError, match=inspect.GEN_SUSPENDED):
        asyncio.run(async_drive(AsyncMappingParents({"a": (), "b": ("a",)}), walk))

    assert inspect.getgeneratorstate(walk) == inspect.GEN_SUSPENDED


def test_a_walk_that_is_not_a_generator_is_refused_by_kind() -> None:
    """The freshness check needs a generator *object*, and says so itself.

    ``inspect.getgeneratorstate`` reads ``gi_running`` off its argument, so a
    ``Walk`` satisfying the alias without being a generator reached it and
    raised ``AttributeError`` naming a CPython slot -- a failure that tells the
    caller nothing about walks. The drivers would otherwise have driven this
    one, so the refusal is a real narrowing and belongs where it can be read:
    a ``TypeError`` naming the kind wanted and the one-line way to get it.

    Silently skipping the check for what it cannot inspect was the other
    candidate. It reopens the ``None``-as-a-result hazard precisely for the
    walks the driver cannot warn about, which is the wrong direction.
    """
    walk = DelegatingWalk(_asking_for("parents"))

    with pytest.raises(TypeError, match="generator"):
        drive(MappingParents({"a": (), "b": ("a",)}), walk)

    with pytest.raises(TypeError, match="generator"):
        asyncio.run(async_drive(AsyncMappingParents({"a": (), "b": ("a",)}), walk))


#: A level wider than any sensible fan-out bound, so an unbounded gather and a
#: bounded one are distinguishable by overlap alone.
_WIDE_LEVEL = 64

_VERY_WIDE_PARENTS: Mapping[str, tuple[str, ...]] = {
    "root": (),
    **{f"n{i}": ("root",) for i in range(_WIDE_LEVEL)},
    "deep": tuple(f"n{i}" for i in range(_WIDE_LEVEL)),
}


def test_a_wide_frontier_is_gathered_within_a_bound() -> None:
    """Concurrency is capped by configuration rather than by the data's shape.

    ``_async_reply`` gathered one call per node over the whole frontier, so how
    hard a walk hits a backing was a property of the *tree* -- a node with ten
    thousand children issued ten thousand concurrent calls into whatever the
    backing was. The bulk path does not help: a backing offering
    ``parents_many`` gets one call, so the unbounded path was exactly the one
    taken by backings least able to absorb it.
    """
    probe = ConcurrencyProbe(_VERY_WIDE_PARENTS)

    asyncio.run(async_ancestors(probe, "deep"))

    assert probe.widest_overlap > 1, "the level must still be gathered, not serialised"
    assert probe.widest_overlap < _WIDE_LEVEL, (
        f"{probe.widest_overlap} calls overlapped over a {_WIDE_LEVEL}-node level: "
        f"the fan-out is the width of the data rather than of a bound"
    )


def test_the_bound_is_the_callers_to_set() -> None:
    """The width is an argument, not a constant the core holds.

    The walk core deliberately carries no configuration, so the bound is
    threaded in from the driver rather than read there -- which is what lets a
    caller who knows their backing raise or lower it.
    """
    probe = ConcurrencyProbe(_VERY_WIDE_PARENTS)

    asyncio.run(async_ancestors(probe, "deep", max_concurrency=4))

    assert probe.widest_overlap <= 4


def test_a_bound_below_one_is_refused() -> None:
    """A width of zero would deadlock rather than serialise.

    ``asyncio.Semaphore(0)`` never admits anyone, so an unchecked zero turns a
    walk into a hang -- the one failure mode worse than the unbounded fan-out
    the bound was added to stop.
    """
    probe = ConcurrencyProbe(WIDE_PARENTS)

    with pytest.raises(ValueError, match="max_concurrency"):
        asyncio.run(async_ancestors(probe, "a", max_concurrency=0))


@pytest.mark.parametrize(
    ("sync_fn", "async_fn"),
    [
        (drive, async_drive),
        (ancestors, async_ancestors),
        (descendants, async_descendants),
        (descendants_to_depth, async_descendants_to_depth),
        (children_at_depth, async_children_at_depth),
        (flatten, async_flatten),
        (leaves, async_leaves),
        (paths_to_root, async_paths_to_root),
        (deepest_common_ancestor, async_deepest_common_ancestor),
    ],
    ids=[
        "drive",
        "ancestors",
        "descendants",
        "descendants_to_depth",
        "children_at_depth",
        "flatten",
        "leaves",
        "paths_to_root",
        "deepest_common_ancestor",
    ],
)
def test_the_module_twins_differ_by_two_declared_things(
    sync_fn: Callable[..., Any], async_fn: Callable[..., Any]
) -> None:
    """The driving pair and every walk over it stay signature-compatible.

    **One row per pair, and the set is the module's public surface.** A
    parametrised guard run over a subset is green for the reason an empty one
    is: it asserts about the pairs it lists and says nothing about the rest, so
    a walk added without a row here is a walk whose twin nothing checks. The
    same argument the core table makes, applied to the surface that table
    delegates to. Nine pairs is the whole of it: the driving pair, and the
    eight walks this module publishes.

    A caller writing flavour-agnostic code against these needs the difference
    to be exactly what is declared, not merely small. Two things are declared
    and both are real:

    ``max_concurrency`` bounds a frontier read with no bulk member to use, and
    the synchronous driver issues no concurrent calls at all -- a knob that does
    nothing is worse than an asymmetry that is stated. ``hierarchy`` is
    flavoured by definition: each driver takes the protocol of its own flavour.

    Both are compared by equality inside the guard, so a *second* divergence of
    either kind fails here rather than quietly joining the first. That property
    used to be argued by naming the set in a module constant; it is now a
    property of the assertion, which is why the constant is gone.
    """
    assert_twins_agree(
        sync_fn,
        async_fn,
        async_only={"max_concurrency"},
        flavour_typed={"hierarchy"},
        compare_return=True,
    )


def test_an_empty_ancestors_does_not_distinguish_a_root_from_an_unknown_node() -> None:
    """The ambiguity the docstring documents, pinned so it stays documented.

    ``Taxonomy.walk`` refuses an anchor its axis does not contain and this does
    not, which is a real difference between two neighbouring surfaces. It is
    defensible -- ``walk`` includes its anchor, so an unknown one is emitted as
    a term of the axis, where this excludes it and returns nothing false -- but
    a reader who met the refusal first will expect it here. Pinning the
    behaviour keeps the difference deliberate: change it and this test says so.
    """
    hierarchy = MappingParents({"root": (), "child": ("root",)})

    assert ancestors(hierarchy, "root") == ()
    assert ancestors(hierarchy, "nonesuch") == ()

    assert hierarchy.contains("root")
    assert not hierarchy.contains("nonesuch")


# --------------------------------------------------------------------------
# The memo reaches every walk, including the oldest one
# --------------------------------------------------------------------------


def test_ancestors_takes_the_memo_its_siblings_take() -> None:
    """Every walk takes the memo, rather than the ones that happened to arrive with it.

    ``ancestors`` shipped before the memo existed, which is the whole way a
    surface drifts: a capability arrives with the walks that prompted it and
    the incumbent keeps the older signature. The module's own docstring names
    that drift for ``max_concurrency``, so the same gap reopened one parameter
    over is the one worth closing rather than recording.
    """
    hierarchy = CountingParents(
        {"gg": (), "g": ("gg",), "p": ("g",), "z": ("p",)},
    )
    warm: dict[tuple[str, Any], Sequence[str]] = {}

    first = ancestors(hierarchy, "z", cache=warm)
    asked_once = hierarchy.singular_calls
    second = ancestors(hierarchy, "z", cache=warm)

    assert first == second == ("p", "g", "gg")
    assert hierarchy.singular_calls == asked_once, "the second walk re-asked the backing"


@pytest.mark.asyncio
async def test_async_ancestors_takes_it_too() -> None:
    """The twin, because a parameter on one half is the drift this closes."""
    hierarchy = AsyncMappingHierarchy({"gg": (), "g": ("gg",), "p": ("g",), "z": ("p",)})
    warm: dict[tuple[str, Any], Sequence[str]] = {}

    first = await async_ancestors(hierarchy, "z", cache=warm)
    second = await async_ancestors(hierarchy, "z", cache=warm)

    assert first == second == ("p", "g", "gg")
    assert warm, "the walk filled no memo"
