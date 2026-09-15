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
* **what the backing is asked.** The descent asks each node once and no walk
  here asks a second time -- ``leaves`` reads childlessness off the reply the
  descent already received -- so the backing sees each node exactly once
  whatever cache the caller did or did not supply.

The differential and the parity claims are the neighbouring suite's and are not
repeated here. The cyclic differential used to be: five of these walks were run
over a cyclic fixture of this file's own while ``ancestors`` was run over an
identical one next door, which is two tests over two copies making one claim
about six of eight walks. It is now one parametrisation over one fixture
reaching all eight, and it lives beside the other differential because the
walks it covers are not all descending ones.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import pytest

from dataknobs_common.bounded_cache import BoundedLRUCache
from dataknobs_common.exceptions import NotFoundError
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

    from dataknobs_common.hierarchy import WalkCache

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


def test_childlessness_is_the_reply_and_not_the_discovery_edges() -> None:
    """Over a DAG a node can discover nothing and still have children.

    ``y`` reaches only ``x``, which the descent had already found under ``b`` --
    so ``y``'s discovery edges are empty while ``y`` is not a leaf. A walk
    reading leafness off *discovery* would return it. The descent keeps the
    other fact, which is whether the reply itself was empty, and those two
    differ on exactly this node.
    """
    hierarchy = MappingHierarchy(DIAMOND_DAG)

    assert hierarchy.children("y") == ("x",)
    assert leaves(hierarchy, under="root") == ("x",)


def test_a_bounded_descent_records_childlessness_only_for_what_it_asked() -> None:
    """The invariant the descent's pair carries, and the one way to misread it.

    ``childless`` is a fact about *replies*, so it covers exactly the nodes a
    reply arrived for -- and a bounded descent stops before asking its last
    level. ``discovered`` takes a key for every node that entered a frontier
    and for no other, so the two together say which is which: absent from
    ``discovered`` means *never asked*, never *has children*. ``a1`` is the
    case, a genuine leaf the bound stopped short of, and a reader concluding
    from its absence that it has children would be wrong about it.
    """
    from dataknobs_common._walk_core import _expand

    hierarchy = MappingHierarchy(BRANCHING)

    discovered, childless = drive(hierarchy, _expand(("root",), "children", max_depth=2))

    assert set(discovered) == {"root", "a", "b"}, "a key per node that entered a frontier"
    assert childless == set(), "nothing that was asked answered empty"
    assert "a1" not in discovered, "the bound stopped before asking it"
    assert hierarchy.children("a1") == (), "and it is childless all the same"


# --------------------------------------------------------------------------
# What the backing is asked -- the memo
# --------------------------------------------------------------------------


def _a_dict() -> dict[tuple[str, Any], Sequence[str]]:
    return {}


def _a_cache_below_the_frontier() -> BoundedLRUCache[tuple[str, Any], Sequence[str]]:
    return BoundedLRUCache(max_size=1)


@pytest.mark.parametrize(
    "make_cache",
    [lambda: None, _a_dict, _a_cache_below_the_frontier],
    ids=["no cache", "a dict", "a cache bounded below the frontier"],
)
def test_leaves_asks_the_backing_about_each_node_once(
    make_cache: Callable[[], WalkCache | None],
) -> None:
    """And whatever the caller hands it, because it never asks a second time.

    Childlessness is the **reply** the descent already received, not a question
    the walk asks again -- so there is no second ask for a memo to answer and
    nothing a caller's cache can do to this count. The three rows are the three
    shapes that used to differ: a walk whose second ask went to the backing, one
    whose driver happened to remember the first, and one whose caller supplied a
    cache too small to hold the frontier and so evicted the answer before it was
    read back.
    """
    hierarchy = CountingChildren(BRANCHING)

    assert leaves(hierarchy, cache=make_cache()) == ("a1", "a2", "b1")

    assert sorted(hierarchy.asked) == sorted(BRANCHING)
    assert len(hierarchy.asked) == len(BRANCHING), (
        f"the backing was asked {len(hierarchy.asked)} times for "
        f"{len(BRANCHING)} nodes: {hierarchy.asked}"
    )


@pytest.mark.asyncio
async def test_the_async_leaves_asks_each_node_once_too() -> None:
    """The descent is the shared one, so neither flavour has this alone."""
    hierarchy = AsyncCountingChildren(BRANCHING)

    assert await async_leaves(hierarchy) == ("a1", "a2", "b1")
    assert len(hierarchy.asked) == len(BRANCHING)


def test_roots_is_not_remembered() -> None:
    """A decision, not an omission.

    One walk asks for it, once, and remembering it across walks would cost a
    caller their only chance to notice the axis grew a root -- which is the
    promise about the *world* that a cache over edge replies does not make.
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
# The awaited fixture these walks are driven over
# --------------------------------------------------------------------------


class _AsyncMapping:
    """A parent mapping, awaited. Local because only this file's twins need it."""

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


# --------------------------------------------------------------------------
# What the backing is allowed to be -- a roots() that repeats itself
# --------------------------------------------------------------------------


class DuplicateRoots:
    """A backing whose ``roots()`` names the same node twice.

    Not perverse. ``roots()`` is arbitrary consumer code, and a query over a
    join returns one row per match -- a ``DISTINCT`` nobody wrote is the whole
    of the difference. The protocol asks for *the nodes with no parent* and
    says nothing about a backing counting one of them once.

    Every frontier after the seeds is deduplicated by the descent's visited
    set, so this is the one request whose reply can reach the core repeated.
    """

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        self._inner = MappingHierarchy(parents)

    def roots(self) -> Sequence[str]:
        found = tuple(self._inner.roots())
        return (*found, *found)

    def parents(self, node_id: str) -> Sequence[str]:
        return self._inner.parents(node_id)

    def children(self, node_id: str) -> Sequence[str]:
        return self._inner.children(node_id)

    def contains(self, node_id: str) -> bool:
        return self._inner.contains(node_id)


class AsyncDuplicateRoots:
    """:class:`DuplicateRoots`, awaited."""

    def __init__(self, parents: Mapping[str, tuple[str, ...]]) -> None:
        self._inner = DuplicateRoots(parents)

    async def roots(self) -> Sequence[str]:
        return self._inner.roots()

    async def parents(self, node_id: str) -> Sequence[str]:
        return self._inner.parents(node_id)

    async def children(self, node_id: str) -> Sequence[str]:
        return self._inner.children(node_id)

    async def contains(self, node_id: str) -> bool:
        return self._inner.contains(node_id)


def test_a_root_named_twice_still_yields_the_whole_axis() -> None:
    """A repeated seed must not cost the walk everything under it.

    The descent records discovery against the node that reached it, and a
    frontier entry visited twice reaches nothing the second time -- everything
    below was already claimed by the first visit. Kept unguarded, the second
    visit's empty record replaces the first's real one and the whole subtree
    stops being reachable from the seed, while the backing is still asked for
    every edge of it.

    Both flavours, because the seeds are read by one flavour-agnostic step and
    a fix written into either driver would be the wrong layer.
    """
    hierarchy = DuplicateRoots(BRANCHING)

    assert flatten(hierarchy) == PRE_ORDER
    assert leaves(hierarchy) == ("a1", "a2", "b1")

    assert asyncio.run(async_flatten(AsyncDuplicateRoots(BRANCHING))) == PRE_ORDER
    assert asyncio.run(async_leaves(AsyncDuplicateRoots(BRANCHING))) == ("a1", "a2", "b1")


def test_a_repeated_seed_is_not_emitted_twice() -> None:
    """Deduplicated *in walk order* is a claim about the seeds as well.

    A repeated key changes no membership answer and does change a length a
    caller is reporting, which is the reason the descent dedups at all.
    """
    assert flatten(DuplicateRoots(BRANCHING)).count("root") == 1


# --------------------------------------------------------------------------
# Where the anchor is, at the boundary of the axis -- an anchor it does not hold
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("walk", "async_walk"),
    [
        (
            lambda h: flatten(h, from_id="marmoset"),
            lambda h: async_flatten(h, from_id="marmoset"),
        ),
        (
            lambda h: descendants_to_depth(h, "marmoset", 3),
            lambda h: async_descendants_to_depth(h, "marmoset", 3),
        ),
        (
            lambda h: children_at_depth(h, "marmoset", 0),
            lambda h: async_children_at_depth(h, "marmoset", 0),
        ),
        (
            lambda h: leaves(h, under="marmoset"),
            lambda h: async_leaves(h, under="marmoset"),
        ),
    ],
    ids=["flatten", "descendants_to_depth", "children_at_depth-0", "leaves"],
)
def test_a_walk_that_includes_its_anchor_refuses_one_the_axis_does_not_hold(
    walk: Any, async_walk: Any
) -> None:
    """Every walk here that *includes* its anchor, and the whole reason it must.

    Seeding a frontier with an unchecked anchor emits an id the axis does not
    contain as though it were a term of it, and the caller cannot tell: a
    one-element result is exactly what a childless node returns. That collapses
    *nothing below this node* into *this node is not here* -- the two answers
    ``contains`` says in its own docstring it exists to keep apart -- inside the
    one walk that needs the distinction.

    ``leaves`` belongs to the list for the same reason and by a longer route:
    an unknown anchor discovers nothing, is confirmed childless because the
    backing has no children for a node it has never heard of, and comes back as
    a leaf of the axis.
    """
    hierarchy = MappingHierarchy(BRANCHING)

    with pytest.raises(NotFoundError) as raised:
        walk(hierarchy)

    assert raised.value.context["anchor"] == "marmoset"

    with pytest.raises(NotFoundError):
        asyncio.run(async_walk(_AsyncMapping(BRANCHING)))


@pytest.mark.parametrize(
    ("walk", "async_walk"),
    [
        (lambda h: descendants(h, "marmoset"), lambda h: async_descendants(h, "marmoset")),
        (lambda h: ancestors(h, "marmoset"), None),
    ],
    ids=["descendants", "ancestors"],
)
def test_a_walk_that_excludes_its_anchor_still_does_not_refuse_one(
    walk: Any, async_walk: Any
) -> None:
    """The other half of the boundary, pinned so the refusal cannot spread.

    An excluding walk returns nothing false about an unknown anchor -- the
    answer is ambiguous, not incorrect, and one ``contains`` call resolves it.
    Refusing here would cost a caller the cheap ambiguous answer and buy
    nothing, so the difference between the two halves is deliberate and this is
    what keeps it so.
    """
    hierarchy = MappingHierarchy(BRANCHING)

    assert walk(hierarchy) == ()

    if async_walk is not None:
        assert asyncio.run(async_walk(_AsyncMapping(BRANCHING))) == ()


def test_the_anchor_check_does_not_reach_a_walk_that_has_no_anchor() -> None:
    """``flatten`` and ``leaves`` descend from the roots when the anchor is omitted.

    There is nothing to refuse there, and an axis with no roots at all is still
    walked rather than refused -- which the empty-axis test pins from the other
    side.
    """
    hierarchy = MappingHierarchy(BRANCHING)

    assert flatten(hierarchy) == PRE_ORDER
    assert leaves(hierarchy) == ("a1", "a2", "b1")


def test_a_caller_supplied_cache_is_scoped_to_one_axis() -> None:
    """The limit the key has, pinned so it stays a stated one.

    ``WalkCacheKey`` is ``(member, node_id)`` and names no hierarchy, so a
    cache spent on a second axis answers it from the first's edges. There is no
    discriminator to add: a ``Hierarchy`` is arbitrary consumer code and need
    not be hashable, and ``id()`` is reused after a collection -- which would
    trade a documented scope for a silent wrong answer that depends on garbage
    collection.

    This asserts the consequence rather than wishing it away, so a future key
    that *can* discriminate fails here and takes the docstrings with it.
    """
    one = MappingHierarchy(BRANCHING)
    other = MappingHierarchy({"root": (), "z": ("root",)})
    shared: dict[tuple[str, Any], Sequence[str]] = {}

    assert flatten(one, from_id="root", cache=shared) == PRE_ORDER

    assert flatten(other, from_id="root", cache=shared) == PRE_ORDER
    assert flatten(other, from_id="root") == ("root", "z")

    fresh: dict[tuple[str, Any], Sequence[str]] = {}
    assert flatten(other, from_id="root", cache=fresh) == ("root", "z")
