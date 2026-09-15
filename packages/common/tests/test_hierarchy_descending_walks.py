"""The five descending walks, and the three claims each of them makes.

``ancestors`` shipped alone, so the suite beside this one is about *a* walk.
This one is about the six together, and it is organised around the claims that
only become claims once there is more than one walk to make them:

* **what each emits.** Four of the five are flattened descents and all four
  emit *pre-order by discovery*; ``children_at_depth`` returns a level and has
  no emission order to choose. The order is a published contract rather than an
  artefact of how the descent happens to fetch, so it is asserted against a
  fixture where it differs from level order and against a DAG where it differs
  from depth-first;
* **where the anchor is.** Excluded from ``descendants``, included by
  ``flatten`` and ``descendants_to_depth``, included at depth ``0`` by
  ``children_at_depth``, and included by ``leaves`` only if it is one. Five
  walks and four answers, none of them inferable from the others;
* **what the backing is asked.** The descent asks each node once, and
  ``leaves`` confirms childlessness over nodes the descent already asked
  about -- so the second ask is answered from the walk's memo and the backing
  sees each node exactly once.

The differential and the parity claims the neighbouring suite makes are not
repeated here; what is repeated, deliberately, is that every walk runs over a
cyclic fixture, because a walk that terminates by luck is indistinguishable
from one that terminates by construction until it does not.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.bounded_cache import BoundedLRUCache
from dataknobs_common.hierarchy import (
    MappingHierarchy,
    ancestors,
    async_children_at_depth,
    async_descendants,
    async_descendants_to_depth,
    async_flatten,
    async_leaves,
    children_at_depth,
    descendants,
    descendants_to_depth,
    drive,
    flatten,
    leaves,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# --------------------------------------------------------------------------
# Fixtures -- each one exists to make a specific pair of answers differ
# --------------------------------------------------------------------------

#: Branching at two levels, so pre-order and level order are different tuples.
#: A tree that branches only at the root cannot tell them apart, which is the
#: shape of fixture that makes an order assertion pass under either discipline.
BRANCHING: Mapping[str, tuple[str, ...]] = {
    "root": (),
    "a": ("root",),
    "b": ("root",),
    "a1": ("a",),
    "a2": ("a",),
    "b1": ("b",),
}

PRE_ORDER = ("root", "a", "a1", "a2", "b", "b1")
LEVEL_ORDER = ("root", "a", "b", "a1", "a2", "b1")

#: A DAG where the two descending disciplines dedup at different first visits.
#: ``x`` is reachable from ``b`` at depth two and from ``y`` at depth three; a
#: level-synchronous descent discovers it under ``b``, a depth-first one under
#: ``y``. The orders then differ, which is why the published contract is
#: *pre-order by discovery* and not *depth-first*.
DIAMOND_DAG: Mapping[str, tuple[str, ...]] = {
    "root": (),
    "a": ("root",),
    "b": ("root",),
    "y": ("a",),
    "x": ("b", "y"),
}

#: Two parents at distance one, and a chain above only one of them. Level order
#: gives distances 1, 1, 2, 3 and pre-order gives 1, 2, 3, 1 -- so *nearest
#: first*, which is what ``ancestors`` publishes, is a claim only one of the two
#: keeps. This is the fixture behind that walk's exclusion from the pre-order
#: rule.
ASCENDING_DIAMOND: Mapping[str, tuple[str, ...]] = {
    "gg": (),
    "g": ("gg",),
    "p1": ("g",),
    "p2": (),
    "z": ("p1", "p2"),
}

#: root -> a -> c -> f -> root. Every walk here runs over it.
CYCLIC: Mapping[str, tuple[str, ...]] = {
    "f": ("c", "d"),
    "c": ("a",),
    "d": ("b", "root"),
    "a": ("root",),
    "b": ("root",),
    "root": ("f",),
}


class CountingChildren:
    """A hierarchy that records every ``children`` call, per node.

    Offers no bulk members deliberately: ``children_many`` would answer a
    frontier in one call and hide the per-node question this suite asks.
    """

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        self._inner = MappingHierarchy(parents)
        self.asked: list[str] = []
        self.root_calls = 0

    def roots(self) -> Sequence[str]:
        self.root_calls += 1
        return self._inner.roots()

    def parents(self, node_id: str) -> Sequence[str]:
        return self._inner.parents(node_id)

    def children(self, node_id: str) -> Sequence[str]:
        self.asked.append(node_id)
        return self._inner.children(node_id)

    def contains(self, node_id: str) -> bool:
        return self._inner.contains(node_id)


class AsyncCountingChildren:
    """:class:`CountingChildren`, awaited."""

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        self._inner = CountingChildren(parents)

    @property
    def asked(self) -> list[str]:
        return self._inner.asked

    async def roots(self) -> Sequence[str]:
        return self._inner.roots()

    async def parents(self, node_id: str) -> Sequence[str]:
        return self._inner.parents(node_id)

    async def children(self, node_id: str) -> Sequence[str]:
        return self._inner.children(node_id)

    async def contains(self, node_id: str) -> bool:
        return self._inner.contains(node_id)


def _depth_first(parents: Mapping[str, tuple[str, ...]], seed: str) -> tuple[str, ...]:
    """A reference depth-first walk, written out so it can be disagreed with.

    Deliberately the shape the shared core refuses -- one recursion per node,
    its own visited set -- because it is the *other* discipline, and a control
    that is computed by the thing under test controls nothing.
    """
    hierarchy = MappingHierarchy(parents)
    seen = {seed}
    out: list[str] = []

    def go(node_id: str) -> None:
        out.append(node_id)
        for child in hierarchy.children(node_id):
            if child not in seen:
                seen.add(child)
                go(child)

    go(seed)
    return tuple(out)


# --------------------------------------------------------------------------
# What each one emits
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("walk", "expected"),
    [
        (lambda h: flatten(h, from_id="root"), PRE_ORDER),
        (lambda h: descendants(h, "root"), PRE_ORDER[1:]),
        (lambda h: descendants_to_depth(h, "root", 9), PRE_ORDER),
        (flatten, PRE_ORDER),
    ],
    ids=["flatten", "descendants", "descendants_to_depth", "flatten-from-roots"],
)
def test_the_descending_flattened_walks_emit_pre_order_by_discovery(
    walk: Any, expected: tuple[str, ...]
) -> None:
    """One rule across the four, and across the bound.

    The control is the second assertion: level order selects the same nodes, so
    a test asserting membership -- or asserting an order over a fixture that
    branches once -- passes under either discipline and proves neither.
    """
    hierarchy = MappingHierarchy(BRANCHING)

    assert walk(hierarchy) == expected
    assert sorted(expected) == sorted(LEVEL_ORDER if len(expected) == 6 else LEVEL_ORDER[1:])
    assert expected != (LEVEL_ORDER if len(expected) == 6 else LEVEL_ORDER[1:])


def test_leaves_emits_the_same_pre_order() -> None:
    """``leaves`` is that order, filtered -- not an order of its own."""
    hierarchy = MappingHierarchy(BRANCHING)

    assert leaves(hierarchy) == ("a1", "a2", "b1")
    assert leaves(hierarchy) == tuple(n for n in PRE_ORDER if n in {"a1", "a2", "b1"})


def test_the_bound_is_a_bound_and_never_an_order_selector() -> None:
    """Every bounded answer is the unbounded one, cut -- same order, fewer nodes.

    The failure this forbids is a walk that is level-ordered when bounded and
    pre-ordered when not, which is what an implementation reaching for
    ``_levels`` at one end and the discovery edges at the other would produce.
    """
    hierarchy = MappingHierarchy(BRANCHING)
    whole = flatten(hierarchy, from_id="root")

    for depth in range(4):
        bounded = descendants_to_depth(hierarchy, "root", depth)
        assert bounded == tuple(n for n in whole if n in set(bounded))

    assert descendants_to_depth(hierarchy, "root", 9) == whole


def test_pre_order_by_discovery_is_not_depth_first_over_a_dag() -> None:
    """The negative control that keeps the published name honest.

    Over a tree the two disciplines agree, so a suite of trees would let either
    name be written on the contract. They diverge exactly where a node is
    reachable by two paths of different lengths, and this is that fixture.
    """
    hierarchy = MappingHierarchy(DIAMOND_DAG)

    assert flatten(hierarchy, from_id="root") == ("root", "a", "y", "b", "x")
    assert _depth_first(DIAMOND_DAG, "root") == ("root", "a", "y", "x", "b")
    assert flatten(hierarchy, from_id="root") != _depth_first(DIAMOND_DAG, "root")

    assert sorted(flatten(hierarchy, from_id="root")) == sorted(_depth_first(DIAMOND_DAG, "root"))


def test_the_two_disciplines_agree_over_a_tree() -> None:
    """The other half of the control: the divergence is the DAG's, not a bug."""
    hierarchy = MappingHierarchy(BRANCHING)

    assert flatten(hierarchy, from_id="root") == _depth_first(BRANCHING, "root")


def test_ancestors_keeps_level_order_because_nearest_first_is_a_distance_claim() -> None:
    """The walk the pre-order rule excludes, and the measurement that excludes it.

    Over this fixture level order gives distances 1, 1, 2, 3 and pre-order gives
    1, 2, 3, 1. ``ancestors`` publishes *nearest first*, so the two promises
    conflict and the docstring already chose -- which is why this walk reads the
    same descent as levels while its five neighbours read it as a pre-order.
    """
    hierarchy = MappingHierarchy(ASCENDING_DIAMOND)

    assert ancestors(hierarchy, "z") == ("p1", "p2", "g", "gg")

    distance = {"p1": 1, "p2": 1, "g": 2, "gg": 3}
    walked = [distance[n] for n in ancestors(hierarchy, "z")]
    assert walked == sorted(walked), "nearest first is the claim; it must not regress"
    assert walked != [distance[n] for n in ("p1", "g", "gg", "p2")]


def test_children_at_depth_returns_one_level_and_has_no_order_to_choose() -> None:
    """A level is a set of equals; what orders it is the order the backing gave."""
    hierarchy = MappingHierarchy(BRANCHING)

    assert children_at_depth(hierarchy, "root", 0) == ("root",)
    assert children_at_depth(hierarchy, "root", 1) == ("a", "b")
    assert children_at_depth(hierarchy, "root", 2) == ("a1", "a2", "b1")
    assert children_at_depth(hierarchy, "root", 3) == ()


def test_a_depth_the_axis_does_not_reach_is_an_answer_rather_than_a_failure() -> None:
    """``()`` means *nothing is that far below*, which is a fact about the axis."""
    hierarchy = MappingHierarchy(BRANCHING)

    assert children_at_depth(hierarchy, "a1", 1) == ()
    assert children_at_depth(hierarchy, "root", 99) == ()


def test_a_negative_bound_reads_as_zero_on_both_bounded_walks() -> None:
    """What the recursion these replace does, kept rather than tightened.

    A refusal would be defensible and is a different decision; this records the
    one taken, so changing it fails a test rather than a consumer.
    """
    hierarchy = MappingHierarchy(BRANCHING)

    assert children_at_depth(hierarchy, "root", -1) == ("root",)
    assert descendants_to_depth(hierarchy, "root", -1) == ("root",)


# --------------------------------------------------------------------------
# Where the anchor is -- five walks, four answers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("walk", "carries_the_anchor"),
    [
        (lambda h: descendants(h, "a"), False),
        (lambda h: flatten(h, from_id="a"), True),
        (lambda h: descendants_to_depth(h, "a", 1), True),
        (lambda h: children_at_depth(h, "a", 0), True),
    ],
    ids=["descendants", "flatten", "descendants_to_depth", "children_at_depth-0"],
)
def test_the_anchor_boundary_is_declared_per_walk(walk: Any, carries_the_anchor: bool) -> None:
    """Four answers, none of them inferable from the others.

    Three of the four are pinned by a released test rather than by a signature,
    which is why the boundary is asserted here rather than left to whichever
    implementation happens to be in the file.
    """
    hierarchy = MappingHierarchy(BRANCHING)

    assert ("a" in walk(hierarchy)) is carries_the_anchor


def test_leaves_includes_the_anchor_only_if_it_is_one() -> None:
    """The fifth answer: conditional, and on a property of the node."""
    hierarchy = MappingHierarchy(BRANCHING)

    assert leaves(hierarchy, under="a1") == ("a1",)
    assert "a" not in leaves(hierarchy, under="a")
    assert leaves(hierarchy, under="a") == ("a1", "a2")


def test_childlessness_is_asked_rather_than_read_off_the_descent() -> None:
    """Over a DAG a node can discover nothing and still have children.

    ``y`` reaches only ``x``, which the descent had already found under ``b`` --
    so ``y``'s discovery edges are empty while ``y`` is not a leaf. A walk that
    read leafness off the descent would return it.
    """
    hierarchy = MappingHierarchy(DIAMOND_DAG)

    assert hierarchy.children("y") == ("x",)
    assert leaves(hierarchy, under="root") == ("x",)


# --------------------------------------------------------------------------
# What the backing is asked -- the memo
# --------------------------------------------------------------------------


def test_leaves_asks_the_backing_about_each_node_once() -> None:
    """The descent asks every node; the confirmation asks the candidates again.

    Without a memo those are two asks for every leaf -- the descent already had
    the reply and threw it away. With one, the backing sees each node exactly
    once and the second ask is answered from the walk's own memory.
    """
    hierarchy = CountingChildren(BRANCHING)

    assert leaves(hierarchy) == ("a1", "a2", "b1")

    assert sorted(hierarchy.asked) == sorted(BRANCHING)
    assert len(hierarchy.asked) == len(BRANCHING), (
        f"the backing was asked {len(hierarchy.asked)} times for "
        f"{len(BRANCHING)} nodes: {hierarchy.asked}"
    )


@pytest.mark.asyncio
async def test_the_async_leaves_asks_each_node_once_too() -> None:
    """The memo is in the step both drivers call, so neither flavour has it alone."""
    hierarchy = AsyncCountingChildren(BRANCHING)

    assert await async_leaves(hierarchy) == ("a1", "a2", "b1")
    assert len(hierarchy.asked) == len(BRANCHING)


def test_roots_is_not_remembered() -> None:
    """A decision, not an omission.

    One walk asks for it, once, and remembering it across walks would cost a
    caller their only chance to notice the axis grew a root -- which is the
    promise about the world that a per-walk memo exists in order not to make.
    """
    hierarchy = CountingChildren(BRANCHING)
    cache: dict[tuple[str, Any], Sequence[str]] = {}

    flatten(hierarchy, cache=cache)
    flatten(hierarchy, cache=cache)

    assert hierarchy.root_calls == 2
    assert not [key for key in cache if key[0] == "roots"]


def test_a_caller_supplied_cache_outlives_the_walk_that_filled_it() -> None:
    """The seam, and the thing a per-walk default cannot do.

    The caller owns the lifetime because the caller is the only party that knows
    how fast its data moves.
    """
    hierarchy = CountingChildren(BRANCHING)
    cache: dict[tuple[str, Any], Sequence[str]] = {}

    first = flatten(hierarchy, from_id="root", cache=cache)
    asked_once = len(hierarchy.asked)
    second = flatten(hierarchy, from_id="root", cache=cache)

    assert first == second
    assert len(hierarchy.asked) == asked_once, "the second walk re-asked the backing"


def test_a_bounded_cache_smaller_than_the_frontier_still_answers_in_order() -> None:
    """A caller's cache may evict its own entries while it is being filled.

    ``BoundedLRUCache`` is a real construct of this package and satisfies the
    two-member seam without inheriting ``MutableMapping`` -- which is why the
    seam is a Protocol. Sized below the frontier it drops entries mid-request,
    so a step that filled the cache and then read every id back would miss one
    it had answered a moment earlier. Nothing about that is visible when the
    cache is a ``dict``.
    """
    tiny: BoundedLRUCache[tuple[str, Any], Sequence[str]] = BoundedLRUCache(max_size=1)
    hierarchy = MappingHierarchy(BRANCHING)

    assert flatten(hierarchy, from_id="root", cache=tiny) == PRE_ORDER


def test_the_memo_answers_in_request_order() -> None:
    """The drivers hand replies back positionally, so a memo can corrupt a walk.

    A reply in the wrong slot is a wrong answer with no exception anywhere, and
    the mixed case -- some ids remembered, some fetched, interleaved -- is the
    one a cache built by an earlier walk produces.
    """
    hierarchy = MappingHierarchy(BRANCHING)
    warm: dict[tuple[str, Any], Sequence[str]] = {}

    descendants(hierarchy, "a", cache=warm)

    assert flatten(hierarchy, from_id="root", cache=warm) == PRE_ORDER


# --------------------------------------------------------------------------
# Cycles, and both flavours
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sync_walk", "async_walk"),
    [
        (lambda h: descendants(h, "root"), lambda h: async_descendants(h, "root")),
        (lambda h: flatten(h, from_id="root"), lambda h: async_flatten(h, from_id="root")),
        (
            lambda h: descendants_to_depth(h, "root", 2),
            lambda h: async_descendants_to_depth(h, "root", 2),
        ),
        (
            lambda h: children_at_depth(h, "root", 2),
            lambda h: async_children_at_depth(h, "root", 2),
        ),
        (lambda h: leaves(h, under="root"), lambda h: async_leaves(h, under="root")),
    ],
    ids=["descendants", "flatten", "descendants_to_depth", "children_at_depth", "leaves"],
)
def test_every_walk_terminates_on_cyclic_data_and_both_flavours_agree(
    sync_walk: Any, async_walk: Any
) -> None:
    """A cycle is the input that separates a correct walk from a lucky one.

    Both halves in one test because they are one claim: a visited set that is
    unconditional in one flavour and conditional in the other is a difference
    the answers show and the surfaces do not.
    """
    result = sync_walk(MappingHierarchy(CYCLIC))

    assert len(set(result)) == len(result), "a node was emitted twice"
    assert result == asyncio.run(async_walk(_AsyncMapping(CYCLIC)))


class _AsyncMapping:
    """The cyclic fixture, awaited. Local because only this file's twin needs it."""

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        self._inner = MappingHierarchy(parents)

    async def roots(self) -> Sequence[str]:
        return self._inner.roots()

    async def parents(self, node_id: str) -> Sequence[str]:
        return self._inner.parents(node_id)

    async def children(self, node_id: str) -> Sequence[str]:
        return self._inner.children(node_id)

    async def contains(self, node_id: str) -> bool:
        return self._inner.contains(node_id)


def test_an_empty_axis_is_walked_rather_than_refused() -> None:
    """``flatten`` and ``leaves`` descend from the roots, and there may be none."""
    empty: MappingHierarchy[str] = MappingHierarchy({})

    assert flatten(empty) == ()
    assert leaves(empty) == ()


def test_a_walk_is_still_single_use() -> None:
    """The refusal the drivers make is not weakened by there being more walks."""
    hierarchy = MappingHierarchy(BRANCHING)
    walk = _a_walk()
    drive(hierarchy, walk)

    with pytest.raises(RuntimeError, match="already been driven"):
        drive(hierarchy, walk)


def _a_walk() -> Any:
    from dataknobs_common._walk_core import _flatten

    return _flatten("root")
