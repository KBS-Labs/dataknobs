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
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dataknobs_common import hierarchy as hierarchy_module
from dataknobs_common.hierarchy import (
    AsyncHierarchy,
    Hierarchy,
    ancestors,
    async_ancestors,
    async_drive,
    drive,
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

    def roots(self) -> Sequence[str]:
        return self._inner.roots()

    def parents(self, node_id: str) -> Sequence[str]:
        self.singular_calls += 1
        return self._inner.parents(node_id)

    def children(self, node_id: str) -> Sequence[str]:
        self.singular_calls += 1
        return self._inner.children(node_id)

    def contains(self, node_id: str) -> bool:
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
    axis = onto.taxonomy("species")
    axis.structure = bulk  # type: ignore[assignment]

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
        axis = onto.taxonomy("species")
        axis.structure = bulk  # type: ignore[assignment]
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
    sentinel = object()

    def sentinel_walk(node_id: str) -> object:
        """A walk that asks nothing and returns a value nothing else produces.

        A generator rather than ``iter(())``, and returning a sentinel rather
        than ``None``, for the same reason in both halves: ``None`` out of an
        empty iterator is an answer the *unpatched* driver could also give, so
        it proves the surfaces agree rather than that the patch was reached.
        """

        def _asks_nothing() -> object:
            return sentinel
            yield  # unreachable, and what makes this a generator

        return _asks_nothing()

    monkeypatch.setattr(hierarchy_module, "_ancestors", sentinel_walk)

    assert ancestors(MappingParents(CYCLIC_PARENTS), "f") is sentinel
    assert asyncio.run(async_ancestors(AsyncMappingParents(CYCLIC_PARENTS), "f")) is sentinel


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
