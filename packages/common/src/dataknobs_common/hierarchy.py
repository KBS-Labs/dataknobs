"""The structure axis: what a node's parents and children are, and the walks over it.

Two protocols and the traversals over them. A hierarchy here is **read-only,
key-addressed and multi-parent-tolerant** -- ``parents(node_id)`` returns a
sequence and never a single node, because an open-world relation yields a DAG
and a structure that silently picks one parent out of two is a lie the caller
cannot detect.

**The asynchronous twin returns a ``Sequence``, not an ``AsyncIterable``**, and
that choice is what makes everything below possible. Because ``parents()`` and
``children()`` hand back *values*, a walk needs to await nothing in the middle
of its own logic: each traversal is written **once**, as a flavour-free
generator that yields a request and receives the answer, and the only twinned
code in this module is the driver pair that decides whether that answer needs
awaiting.

That pair is a **fixed** cost: it does not grow when a walk is added, which is
the whole of the bet. State it as a bet rather than a saving, because one
public traversal ships today -- ``ancestors`` -- and at one walk the
arrangement costs more than a hand-twinned pair would. It pays from the second
on, and the shape it exists to avoid is ``dataknobs_data``'s
``_search_with_complex_query``: 57 lines in each flavour, differing in three.

The key type is a parameter with ``str`` **defaulted**, so a bare ``Hierarchy``
is ``Hierarchy[str]`` and reads as it always did. It exists because the walks
never *inspect* a node id -- they only hash one -- so an object tree with no id
at all can bind ``K`` to its own node type and share the same traversals.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Generator, Hashable, Sequence
from typing import TYPE_CHECKING, Generic, Literal, Protocol, cast, runtime_checkable

if sys.version_info >= (3, 13):  # pragma: no cover - 3.12 is the floor and what runs
    from typing import TypeVar
else:
    # PEP 696 defaults are ``typing``'s only from 3.13 and ``requires-python``
    # is >=3.12, so this is a *runtime* need rather than a typing-only one:
    # ``typing.TypeVar`` raises TypeError on ``default=``. The branch above
    # deletes this dependency the day the floor rises.
    from typing_extensions import TypeVar

__all__ = [
    "AsyncBulkHierarchy",
    "AsyncHierarchy",
    "BulkHierarchy",
    "Hierarchy",
    "K",
    "ancestors",
    "async_ancestors",
    "async_drive",
    "drive",
]

#: A node key. Defaults to ``str``, which is what every ontology-backed
#: hierarchy uses; the bound is ``Hashable`` because hashing is the only thing
#: the walks below ever do to a key.
K = TypeVar("K", bound=Hashable, default=str)

#: The core's own key parameter, with no default. A default may not precede a
#: parameter without one, and the generic aliases below pair a key with a walk
#: result type -- so the private layer takes the undefaulted form and the
#: public fence keeps the default that makes a bare ``Hierarchy`` mean
#: ``Hierarchy[str]``.
_K = TypeVar("_K", bound=Hashable)

_T = TypeVar("_T")

#: The three protocol members a walk may ask for.
Member = Literal["parents", "children", "roots"]

#: One request: a member, and the nodes to ask it about. ``roots`` takes none.
Ask = tuple[Member, tuple[_K, ...]]

#: A walk: a generator that yields an :data:`Ask`, receives one reply per node
#: asked about, and finally returns its result. It contains no ``await`` and no
#: knowledge of which flavour is driving it.
Walk = Generator[Ask[_K], tuple[Sequence[_K], ...], _T]


@runtime_checkable
class Hierarchy(Protocol, Generic[K]):
    """A read-only, key-addressed partial order over nodes.

    ``K`` defaults to ``str``, so a bare ``Hierarchy`` *is* ``Hierarchy[str]``
    and every unparameterised annotation means what it says. It is invariant,
    appearing in both argument and return position, which is what this protocol
    wants.

    ``isinstance`` is written against the **bare** name --
    ``isinstance(x, Hierarchy[str])`` raises ``TypeError``, as it does for every
    subscripted generic. Note also what the runtime check does and does not
    reach: it compares member *names* and nothing else, so a synchronous
    implementation satisfies :class:`AsyncHierarchy` at runtime. The static
    check is the one that separates the flavours.
    """

    def roots(self) -> Sequence[K]:
        """The nodes this axis has with no parent.

        Not an extent. A node absent from every edge of this axis is not in
        this hierarchy at all, so ``roots()`` equals a type's membership only
        by coincidence.
        """
        ...

    def parents(self, node_id: K) -> Sequence[K]:
        """The nodes ``node_id`` is directly under. Plural, always."""
        ...

    def children(self, node_id: K) -> Sequence[K]:
        """The nodes directly under ``node_id``."""
        ...

    def contains(self, node_id: K) -> bool:
        """Whether this axis knows the node at all.

        The member that keeps *nothing below this node* distinguishable from
        *this node is not here* -- opposite answers that an empty
        ``children()`` alone cannot tell apart.
        """
        ...


@runtime_checkable
class AsyncHierarchy(Protocol, Generic[K]):
    """The asynchronous twin. Same four members, same return types.

    ``contains`` is ``async`` for symmetry rather than need: a backing that
    answers it from memory awaits nothing, and one that answers it from a row
    cannot.
    """

    async def roots(self) -> Sequence[K]:
        """The nodes this axis has with no parent."""
        ...

    async def parents(self, node_id: K) -> Sequence[K]:
        """The nodes ``node_id`` is directly under. Plural, always."""
        ...

    async def children(self, node_id: K) -> Sequence[K]:
        """The nodes directly under ``node_id``."""
        ...

    async def contains(self, node_id: K) -> bool:
        """Whether this axis knows the node at all."""
        ...


@runtime_checkable
class BulkHierarchy(Hierarchy[K], Protocol):
    """A :class:`Hierarchy` that can answer a whole frontier in one call.

    **Optional, and deliberately a separate protocol.** Every walk here asks
    about a frontier rather than a node -- :data:`Ask` carries a tuple -- so a
    backing that can answer one query per level instead of one per node is
    asking to be told. Requiring the members on :class:`Hierarchy` would break
    every implementation that has only the singular pair, including the
    hand-written one the guide invites, so the capability is declared where a
    backing can opt into it and the drivers use it when it is there.

    A row- or document-backed hierarchy is the case: ``N`` queries per level
    against a database is the cost the singular members force, and it cannot be
    fixed from inside them.

    Each member returns **one sequence per node asked about, in the order
    asked** -- a node with no answer contributes an empty sequence rather than
    being dropped, because the reply is positional.
    """

    def parents_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`Hierarchy.parents` for a whole frontier, one reply per node."""
        ...

    def children_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`Hierarchy.children` for a whole frontier, one reply per node."""
        ...


@runtime_checkable
class AsyncBulkHierarchy(AsyncHierarchy[K], Protocol):
    """:class:`BulkHierarchy`'s asynchronous twin, with the same contract.

    Worth having even though :func:`async_drive` already gathers per frontier:
    ``gather`` gives one *round trip* per node run concurrently, where these
    give one *query* for the level. A backing with a bulk read wants the
    second, and concurrency does not substitute for it.
    """

    async def parents_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`AsyncHierarchy.parents` for a whole frontier."""
        ...

    async def children_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`AsyncHierarchy.children` for a whole frontier."""
        ...


# --------------------------------------------------------------------------
# The two drivers -- the only twinned code here, and their count is constant
# --------------------------------------------------------------------------


def _sync_reply(
    hierarchy: Hierarchy[_K], member: Member, node_ids: tuple[_K, ...]
) -> tuple[Sequence[_K], ...]:
    """Answer one request in bulk where the backing offers it, else one by one.

    The single implementation of that rule: :class:`Taxonomy`'s streaming walk
    calls it too rather than deciding again, so the carve-out walk cannot drift
    from the driver on which backings get a per-level query.

    It also guarantees no ``StopIteration`` escapes.

    A ``Hierarchy`` is arbitrary consumer code, and a plain ``def`` ending in
    ``next(r for r in rows if ...)`` raises ``StopIteration`` on a miss.
    Reaching :func:`drive`'s ``except`` it would be read as "the walk
    finished", so it is converted here into the ``RuntimeError`` PEP 479
    already raises for the same mistake one frame further in -- which is what
    :func:`async_drive` gets for free, its own coroutine frame performing the
    conversion. Same failure, same type, in both flavours.
    """
    try:
        if member == "roots":
            return (hierarchy.roots(),)
        if member in ("parents", "children"):
            bulk = getattr(hierarchy, f"{member}_many", None)
            if bulk is not None:
                return tuple(bulk(node_ids))
            one = hierarchy.parents if member == "parents" else hierarchy.children
            return tuple(one(node_id) for node_id in node_ids)
    except StopIteration as stop:
        raise RuntimeError(
            f"hierarchy {type(hierarchy).__name__}.{member}() raised StopIteration"
        ) from stop
    raise ValueError(f"unknown hierarchy member {member!r}")


def drive(hierarchy: Hierarchy[_K], walk: Walk[_K, _T]) -> _T:
    """Run a walk against a synchronous hierarchy.

    The ``except`` below must see the walk's own return and nothing else.
    ``StopIteration`` from the *core* cannot reach it -- PEP 479 converts one
    escaping a generator body into ``RuntimeError`` at the frame boundary --
    and one from the *hierarchy* is converted by :func:`_sync_reply` before it
    is raised, so the reply call is safe inside the ``try``.
    """
    try:
        member, node_ids = next(walk)
        while True:
            member, node_ids = walk.send(_sync_reply(hierarchy, member, node_ids))
    except StopIteration as stop:
        # ``StopIteration.value`` is typed ``Any`` by the standard library.
        # Not a suppression -- there is no finding to suppress; it is a
        # boundary the language does not type.
        return cast("_T", stop.value)


async def _async_reply(
    hierarchy: AsyncHierarchy[_K], member: Member, node_ids: tuple[_K, ...]
) -> tuple[Sequence[_K], ...]:
    """:func:`_sync_reply`'s twin: bulk where offered, else one round of
    concurrency per frontier.

    ``gather`` runs one round trip per node concurrently; ``children_many``
    is one *query* for the level. The second is what a row-backed hierarchy
    wants, and concurrency does not substitute for it.

    No explicit ``StopIteration`` conversion here: this is a coroutine, so the
    language performs it at this frame's boundary and a collaborator's
    ``StopIteration`` surfaces as the same ``RuntimeError`` the synchronous
    side raises by hand.
    """
    if member == "roots":
        return (await hierarchy.roots(),)
    if member in ("parents", "children"):
        bulk = getattr(hierarchy, f"{member}_many", None)
        if bulk is not None:
            return tuple(await bulk(node_ids))
        one = hierarchy.parents if member == "parents" else hierarchy.children
        return tuple(await asyncio.gather(*(one(n) for n in node_ids)))
    raise ValueError(f"unknown hierarchy member {member!r}")


async def async_drive(hierarchy: AsyncHierarchy[_K], walk: Walk[_K, _T]) -> _T:
    """Run a walk against an asynchronous hierarchy.

    One round of concurrency **per depth** rather than per node, because the
    core asks for a whole frontier at a time. A twinned walk gets that only if
    somebody remembers to write it into each copy.
    """
    try:
        member, node_ids = next(walk)
        while True:
            member, node_ids = walk.send(await _async_reply(hierarchy, member, node_ids))
    except StopIteration as stop:
        return cast("_T", stop.value)


# --------------------------------------------------------------------------
# The primitive the walks are written as
# --------------------------------------------------------------------------

#: Which way a level expansion goes.
Direction = Literal["parents", "children"]


def _levels(
    seeds: tuple[_K, ...],
    direction: Direction,
    *,
    exclude_seeds: bool,
) -> Walk[_K, tuple[tuple[_K, ...], ...]]:
    """Expand level by level, returning one tuple per level.

    Three properties, each of which is a way to get a walk wrong:

    * the visited set is **unconditional** -- the data may be cyclic whatever
      any acyclicity constraint claims, and a walk that terminates by luck is
      indistinguishable from one that terminates by construction until it does
      not;
    * results are **deduplicated in walk order** -- a DAG node is reachable by
      several paths, and a repeated id changes no membership answer but does
      change a count someone is reporting;
    * the seeds are excluded or included **at the boundary**, by the caller's
      flag, rather than by a rule inside the walk.

    Levels rather than a flat tuple because levels are what a frontier-at-a-time
    expansion produces, and they are what makes the driver's one-round-per-depth
    concurrency possible. Flattening is a composition's choice -- :func:`_flat`
    is the only one taken today -- rather than a decision this primitive makes
    on every caller's behalf.
    """
    seen = set(seeds)
    levels: list[tuple[_K, ...]] = [] if exclude_seeds else [seeds]
    frontier = seeds
    while frontier:
        replies = yield (direction, frontier)
        fresh: list[_K] = []
        for reply in replies:
            for node_id in reply:
                if node_id not in seen:
                    seen.add(node_id)
                    fresh.append(node_id)
        if fresh:
            levels.append(tuple(fresh))
        frontier = tuple(fresh)
    return tuple(levels)


def _flat(levels: tuple[tuple[_K, ...], ...]) -> tuple[_K, ...]:
    """One tuple, in level order."""
    return tuple(node_id for level in levels for node_id in level)


def _ancestors(node_id: _K) -> Walk[_K, tuple[_K, ...]]:
    """Every node above this one, nearest first, excluding the node itself."""
    return _flat((yield from _levels((node_id,), "parents", exclude_seeds=True)))


# --------------------------------------------------------------------------
# The public walks
# --------------------------------------------------------------------------


def ancestors(hierarchy: Hierarchy[K], node_id: K) -> tuple[K, ...]:
    """Every node above ``node_id``, nearest first.

    Excludes ``node_id`` itself, terminates on cyclic data, and returns each
    node once. ``K`` is inferred from ``hierarchy``.
    """
    return drive(hierarchy, _ancestors(node_id))


async def async_ancestors(hierarchy: AsyncHierarchy[K], node_id: K) -> tuple[K, ...]:
    """:func:`ancestors` over an asynchronous hierarchy.

    The same generator drives both flavours -- this is not a second
    implementation of the walk, and a test asserts that patching the core moves
    both surfaces.
    """
    return await async_drive(hierarchy, _ancestors(node_id))


if TYPE_CHECKING:  # pragma: no cover - checked by the type checker, not run
    from typing import assert_type

    def _the_key_defaults_to_str() -> None:
        """A bare ``Hierarchy`` annotation means ``Hierarchy[str]``.

        Written here rather than as a test because this file is type-checked
        and the test tree is not -- the same reason
        ``ontology/sources.py`` carries its protocol-satisfaction check inline.

        This is the half of the key parameter that regresses **silently**.
        Without the default a bare ``Hierarchy`` is ``Hierarchy[Any]``,
        ``disallow_any_generics`` is not set in this repository, and a
        wrong-typed key would then type-check with no error at all.
        """

        def _bare(hierarchy: Hierarchy) -> None:
            assert_type(hierarchy, "Hierarchy[str]")

        def _bare_async(hierarchy: AsyncHierarchy) -> None:
            assert_type(hierarchy, "AsyncHierarchy[str]")

        def _inferred(hierarchy: Hierarchy[int]) -> None:
            assert_type(ancestors(hierarchy, 3), "tuple[int, ...]")

        del _bare, _bare_async, _inferred
