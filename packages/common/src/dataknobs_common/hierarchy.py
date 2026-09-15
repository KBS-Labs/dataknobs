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
the whole of the bet. It was stated as a bet rather than a saving because one
traversal shipped -- ``ancestors`` -- and at one walk the arrangement costs
more than a hand-twinned pair would. The second has since arrived: the walk
behind :meth:`MappingHierarchy.snapshot` is a generator in the core, driven by
both constructors, and it cost one generator rather than a pair. What the
arrangement buys is a lower *marginal* cost rather than a lower total -- the
pair is paid once, it is not small, and it grows when a **capability** is added
to it (bulk frontier dispatch did) where a walk added *over* it does not. That
distinction is the bet, and it is the part to keep watching. The shape it
exists to avoid is ``dataknobs_data``'s ``_search_with_complex_query``: 57
lines in each flavour, differing in three.

Both halves have since been paid again, and the bet held both times. Five more
walks arrived -- ``descendants``, ``descendants_to_depth``,
``children_at_depth``, ``flatten`` and ``leaves`` -- and each cost one
generator in the core and two one-expression wrappers here, the driver pair
unchanged. And a **capability** arrived too, which is the half that was
supposed to be expensive: the walk cache went into the frontier read both
drivers already call rather than into either of them, so it cost one
implementation instead of a twinned pair -- and the streaming walk that goes
through no driver at all reaches the same seam for the cost of forwarding a
parameter, rather than needing a second implementation of its own.

The key type is a parameter with ``str`` **defaulted**, so a bare ``Hierarchy``
is ``Hierarchy[str]`` and reads as it always did. It exists because the walks
never *inspect* a node id -- they only hash one -- so an object tree with no id
at all can bind ``K`` to its own node type and share the same traversals.
"""

from __future__ import annotations

import sys
from collections.abc import Generator, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NoReturn,
    Protocol,
    cast,
    runtime_checkable,
)

# The algorithms and the frontier read live in the private sibling: a core that
# is only a paragraph is satisfied by two twins that each implement it, so what
# is shared is code both flavours call. The runtime dependency runs this way
# only -- the core names these protocols in annotations and nothing more.
from dataknobs_common._nested_core import _mint_tree
from dataknobs_common._walk_core import (
    WalkCache,
    WalkCacheKey,
    _ancestors,
    _async_reply,
    _children_at_depth,
    _deepest_common_ancestor,
    _descendants,
    _descendants_to_depth,
    _flatten,
    _leaves,
    _parent_edges,
    _paths_to_root,
    _refuse_a_spent_walk,
    _refuse_an_unusable_bound,
    _sync_reply,
)
from dataknobs_common.exceptions import NotFoundError

if sys.version_info >= (3, 13):  # pragma: no cover - 3.12 is the floor and what runs
    from typing import TypeVar
else:
    # PEP 696 defaults are ``typing``'s only from 3.13 and ``requires-python``
    # is >=3.12, so this is a *runtime* need rather than a typing-only one:
    # ``typing.TypeVar`` raises TypeError on ``default=``. The branch above
    # deletes this dependency the day the floor rises.
    from typing_extensions import TypeVar

__all__ = [
    "DEFAULT_FRONTIER_CONCURRENCY",
    "Ask",
    "AsyncBulkHierarchy",
    "AsyncEnumerableHierarchy",
    "AsyncHierarchy",
    "AsyncHierarchyView",
    "AsyncMappingHierarchy",
    "BulkHierarchy",
    "EnumerableHierarchy",
    "Hierarchy",
    "HierarchyView",
    "K",
    "MappingHierarchy",
    "Member",
    "Walk",
    "WalkCache",
    "WalkCacheKey",
    "ancestors",
    "async_ancestors",
    "async_children_at_depth",
    "async_deepest_common_ancestor",
    "async_descendants",
    "async_descendants_to_depth",
    "async_drive",
    "async_flatten",
    "async_leaves",
    "async_paths_to_root",
    "children_at_depth",
    "deepest_common_ancestor",
    "descendants",
    "descendants_to_depth",
    "drive",
    "flatten",
    "leaves",
    "paths_to_root",
]

#: How many singular calls an asynchronous frontier read may have outstanding.
#:
#: A bound is needed because without one the fan-out is the width of the
#: *level*, which is a property of the data rather than of anything anyone
#: configured: a node with ten thousand children issued ten thousand concurrent
#: calls into whatever the backing was. A backing offering the bulk members
#: takes one call and never had the problem, so the unbounded path was the one
#: taken by exactly the backings least able to absorb it.
#:
#: The number is a default rather than a limit -- every entry point below takes
#: ``max_concurrency``, because only the caller knows what their backing is.
#:
#: It is deliberately small, and sized against what this repository actually
#: ships rather than against a general intuition: ``dataknobs-data``'s
#: asynchronous Postgres pool defaults to ``max_size=5`` and its pgvector store
#: to ``pool_max_size=10``. A walk should not be the thing that saturates a pool
#: it does not own, so the default stays in that order of magnitude -- while
#: being comfortably above 1, so a wide level is still gathered rather than
#: serialised one node at a time.
#:
#: A caller who knows their backing should set it. That is the whole reason it
#: is a keyword on every entry point rather than a constant read in the core.
DEFAULT_FRONTIER_CONCURRENCY = 8

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
#:
#: A generator *object*, and the drivers enforce it. They refuse a walk that is
#: not fresh, and a walk's freshness lives on the generator -- so a class
#: implementing ``collections.abc.Generator`` satisfies this alias but is
#: refused by :func:`drive` and :func:`async_drive` with a ``TypeError``. Write
#: a walk that wraps another as a generator function delegating with ``yield
#: from``, which is one, rather than as a class, which is not.
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


@runtime_checkable
class EnumerableHierarchy(Hierarchy[K], Protocol):
    """A :class:`Hierarchy` that can say what it holds, not only what descends.

    **The extent ``roots()`` is not.** A bare hierarchy offers no way to ask
    what is in it: ``roots()`` is *the nodes this relation leaves unplaced*, so
    the only enumeration available against the four members is to start there
    and descend -- and a cyclic component with no root above it is unreachable
    that way. The consequence is not academic. It lands on
    :meth:`MappingHierarchy.snapshot`, whose copy would then *refuse an anchor
    the axis it copied accepts*, because
    :meth:`~dataknobs_common.ontology.taxonomy.Taxonomy.walk` refuses an unknown anchor
    by asking ``contains``.

    A backing that is holding its edges, or can fetch them in one query, is not
    subject to that limit and this is where it says so. Optional and separate
    for the same reason :class:`BulkHierarchy` is: requiring it on
    :class:`Hierarchy` would break every implementation with only the four
    members, including the hand-written one the guide invites -- and such an
    implementation is *right* to lack it, since an axis behind a paged API may
    genuinely have no way to answer.

    One entry per node the axis knows, **including a node that appears only as
    somebody's parent**, whose entry is empty. A mapping keyed by children
    alone would omit exactly the nodes ``roots()`` reports, which is the half a
    naive reading misses.
    """

    def parent_edges(self) -> Mapping[K, Sequence[K]]:
        """Every node this axis knows, and what each is directly under."""
        ...


@runtime_checkable
class AsyncEnumerableHierarchy(AsyncHierarchy[K], Protocol):
    """:class:`EnumerableHierarchy`'s asynchronous twin, same contract.

    ``async`` because a backing that answers this from a query has a round trip
    to make, where one holding a mapping does not -- the same asymmetry every
    member of the asynchronous protocol carries.
    """

    async def parent_edges(self) -> Mapping[K, Sequence[K]]:
        """Every node this axis knows, and what each is directly under."""
        ...


# --------------------------------------------------------------------------
# The two drivers -- the only twinned code here, and their count is constant
# --------------------------------------------------------------------------


def drive(hierarchy: Hierarchy[_K], walk: Walk[_K, _T], *, cache: WalkCache | None = None) -> _T:
    """Run a walk against a synchronous hierarchy.

    ``cache`` is where the walk remembers an edge reply, and there is **no
    default one**: a walk composed here asks the backing about a node exactly
    once, so a memo built per call would hold a second copy of every reply for
    a walk that will never read it back. A whole-axis walk is where that shows
    -- :meth:`MappingHierarchy.snapshot` is one -- and it measured 107% of what
    the walk returns.

    So a cache here is for spending edge replies **across** walks, and it is
    the caller's because only they can say when it is stale. It is also theirs
    to scope: the key names the member and the node and not the hierarchy, so
    one cache spent on two axes answers the second from the first.
    :data:`~dataknobs_common._walk_core.WalkCacheKey` says why no discriminator
    is available to put there.

    The ``except`` below must see the walk's own return and nothing else.
    ``StopIteration`` from the *core* cannot reach it -- PEP 479 converts one
    escaping a generator body into ``RuntimeError`` at the frame boundary --
    and one from the *hierarchy* is converted by :func:`_sync_reply` before it
    is raised, so the reply call is safe inside the ``try``.

    The walk's own state is checked *before* the ``try``, for the same reason:
    a spent generator raises ``StopIteration(value=None)`` from the opening
    ``next``, which lands in that ``except`` and reads as the walk finishing.
    That check pins the walk's *kind* as well -- the state it reads exists only
    on a generator object -- so a :data:`Walk` written as a class is refused
    with ``TypeError`` rather than driven.
    """
    _refuse_a_spent_walk(walk)
    try:
        member, node_ids = next(walk)
        while True:
            member, node_ids = walk.send(_sync_reply(hierarchy, member, node_ids, cache=cache))
    except StopIteration as stop:
        # ``StopIteration.value`` is typed ``Any`` by the standard library.
        # Not a suppression -- there is no finding to suppress; it is a
        # boundary the language does not type.
        return cast("_T", stop.value)


async def async_drive(
    hierarchy: AsyncHierarchy[_K],
    walk: Walk[_K, _T],
    *,
    max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
    cache: WalkCache | None = None,
) -> _T:
    """Run a walk against an asynchronous hierarchy.

    One round of concurrency **per depth** rather than per node, because the
    core asks for a whole frontier at a time. A twinned walk gets that only if
    somebody remembers to write it into each copy.

    ``max_concurrency`` bounds that round where the backing offers no bulk
    members; see :data:`DEFAULT_FRONTIER_CONCURRENCY` for why a bound is not
    optional. It is a keyword here and a constant nowhere the core can read,
    which is what keeps the walk core free of configuration.

    ``cache`` is :func:`drive`'s, and means the same thing: nothing unless the
    caller supplies one they own -- including the one-cache-one-axis scope that
    function states.
    """
    _refuse_a_spent_walk(walk)
    try:
        member, node_ids = next(walk)
        while True:
            member, node_ids = walk.send(
                await _async_reply(
                    hierarchy,
                    member,
                    node_ids,
                    max_concurrency=max_concurrency,
                    cache=cache,
                )
            )
    except StopIteration as stop:
        return cast("_T", stop.value)


# --------------------------------------------------------------------------
# The anchor boundary -- what an including walk owes an anchor it cannot find
# --------------------------------------------------------------------------


def _refuse_an_unknown_anchor(node_id: object) -> NoReturn:
    """Refuse an anchor the axis does not contain, for an *including* walk.

    The anchor is emitted by four of the walks below, so seeding a frontier
    with it unchecked returns an id the hierarchy does not contain as though it
    were a term of it -- and the caller cannot tell, because a one-element
    result is exactly what a childless node gives. That collapses *nothing
    below this node* into *this node is not here*, which are the two answers
    :meth:`Hierarchy.contains` says in its own docstring it exists to keep
    apart.

    Yielding nothing was the other candidate and is the worse one for the same
    reason: it destroys the same distinction at the other end. An *excluding*
    walk needs neither -- :func:`ancestors` and :func:`descendants` return
    nothing false about an unknown anchor, so the answer there is ambiguous
    rather than incorrect and one ``contains`` call resolves it.

    Shared by both flavours rather than written into each: the ``await`` is on
    the containment question, not on the refusal, so the message is
    single-sourced while each twin keeps its own call.
    """
    raise NotFoundError(
        f"no node {node_id!r} in this hierarchy, so it cannot anchor a walk "
        f"that includes its anchor",
        context={"anchor": node_id},
    )


def _refuse_unless_known(hierarchy: Hierarchy[K], node_id: K | None) -> None:
    """Ask the containment question an including walk needs, where there is one.

    ``None`` is not an anchor: :func:`flatten` and :func:`leaves` descend from
    the axis's roots when it is omitted, and there is nothing to refuse.
    """
    if node_id is not None and not hierarchy.contains(node_id):
        _refuse_an_unknown_anchor(node_id)


async def _async_refuse_unless_known(hierarchy: AsyncHierarchy[K], node_id: K | None) -> None:
    """:func:`_refuse_unless_known`'s twin. The ``await`` is the whole difference."""
    if node_id is not None and not await hierarchy.contains(node_id):
        _refuse_an_unknown_anchor(node_id)


# --------------------------------------------------------------------------
# The public walks
# --------------------------------------------------------------------------


def ancestors(
    hierarchy: Hierarchy[K], node_id: K, *, cache: WalkCache | None = None
) -> tuple[K, ...]:
    """Every node above ``node_id``, nearest first.

    Excludes ``node_id`` itself, terminates on cyclic data, and returns each
    node once. ``K`` is inferred from ``hierarchy``.

    **An empty result means either a root or an unknown node**, and this does
    not refuse the second — unlike the four walks here that *include* their
    anchor, which do. The difference is recoverability rather than taste. An
    including walk emits an unknown anchor as a term of the axis, so the caller
    receives a wrong answer they cannot detect; see
    :func:`_refuse_an_unknown_anchor`. This excludes its anchor, so nothing
    false is returned: the answer is ambiguous, not incorrect, and
    ``hierarchy.contains(node_id)`` — a member every backing must implement —
    resolves it in one call. Ask it first where the distinction matters.

    ``cache`` is :func:`drive`'s, on every walk here rather than on the ones
    that happened to arrive with it. This walk shipped first and the seam came
    later, which is exactly the shape the module docstring describes a surface
    drifting into.
    """
    return drive(hierarchy, _ancestors(node_id), cache=cache)


async def async_ancestors(
    hierarchy: AsyncHierarchy[K],
    node_id: K,
    *,
    max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
    cache: WalkCache | None = None,
) -> tuple[K, ...]:
    """:func:`ancestors` over an asynchronous hierarchy.

    The same generator drives both flavours -- this is not a second
    implementation of the walk, and a test asserts that patching the core moves
    both surfaces. ``max_concurrency`` is forwarded to the driver; the
    synchronous twin has no counterpart because it issues no concurrent calls.

    An empty result carries the same ambiguity :func:`ancestors` describes, and
    is resolved the same way.
    """
    return await async_drive(
        hierarchy, _ancestors(node_id), max_concurrency=max_concurrency, cache=cache
    )


def descendants(
    hierarchy: Hierarchy[K], node_id: K, *, cache: WalkCache | None = None
) -> tuple[K, ...]:
    """Every node below ``node_id``, **pre-order by discovery**.

    Excludes ``node_id`` itself -- the exclusion :func:`ancestors` makes, at
    the other end of the same axis. :meth:`~dataknobs_common.ontology.taxonomy.Taxonomy.subtree_keys`
    is the including counterpart, and the difference between the two is a
    boundary rather than a preference.

    Terminates on cyclic data and returns each node once. An empty result means
    either a leaf or an unknown node, and ``hierarchy.contains(node_id)``
    separates them -- see :func:`_refuse_an_unknown_anchor` for why that
    ambiguity is acceptable here and is a refusal in the walks that include
    their anchor.

    **Pre-order by discovery, not depth-first**, and over a DAG the two differ:
    a node reachable by several paths is emitted under whichever reached it
    first. Over a tree they coincide.
    """
    return drive(hierarchy, _descendants(node_id), cache=cache)


async def async_descendants(
    hierarchy: AsyncHierarchy[K],
    node_id: K,
    *,
    max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
    cache: WalkCache | None = None,
) -> tuple[K, ...]:
    """:func:`descendants` over an asynchronous hierarchy.

    The same generator drives both flavours -- this is not a second
    implementation of the walk.
    """
    return await async_drive(
        hierarchy, _descendants(node_id), max_concurrency=max_concurrency, cache=cache
    )


def descendants_to_depth(
    hierarchy: Hierarchy[K],
    node_id: K,
    max_depth: int,
    *,
    cache: WalkCache | None = None,
) -> tuple[K, ...]:
    """``node_id`` and everything at most ``max_depth`` levels below it.

    **Includes** ``node_id``: ``max_depth=0`` returns the anchor alone and
    ``1`` returns the anchor and its children. A negative bound reads as ``0``.

    :func:`flatten` is this walk with no bound, in the same order --
    ``max_depth`` bounds the descent and never selects an order, so a caller
    cannot get one ordering by asking for a depth and another by asking for
    all of them.

    **An anchor the axis does not contain is refused**, because this walk emits
    it: see :func:`_refuse_an_unknown_anchor` for why an including walk owes
    that and an excluding one does not.
    """
    _refuse_unless_known(hierarchy, node_id)
    return drive(hierarchy, _descendants_to_depth(node_id, max_depth), cache=cache)


async def async_descendants_to_depth(
    hierarchy: AsyncHierarchy[K],
    node_id: K,
    max_depth: int,
    *,
    max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
    cache: WalkCache | None = None,
) -> tuple[K, ...]:
    """:func:`descendants_to_depth` over an asynchronous hierarchy."""
    await _async_refuse_unless_known(hierarchy, node_id)
    return await async_drive(
        hierarchy,
        _descendants_to_depth(node_id, max_depth),
        max_concurrency=max_concurrency,
        cache=cache,
    )


def children_at_depth(
    hierarchy: Hierarchy[K], node_id: K, depth: int, *, cache: WalkCache | None = None
) -> tuple[K, ...]:
    """Exactly the nodes ``depth`` levels below ``node_id`` -- one level.

    ``depth=0`` is ``node_id`` itself, ``1`` its children, ``2`` its
    grandchildren. A negative depth reads as ``0`` -- the anchor alone -- as it
    does on :func:`descendants_to_depth`, rather than as an error. A depth the
    axis does not reach returns ``()``, which is an answer and not a failure:
    nothing is that far below the anchor. The two are different answers, and
    the difference is the point -- ``()`` says *nothing is that deep*, while a
    one-element result says *this node is*.

    **The one walk here with no emission order to choose.** A level is a set of
    equals; what orders it is the order the backing answered in.

    **An anchor the axis does not contain is refused**, for the reason
    :func:`_refuse_an_unknown_anchor` gives -- at ``depth=0`` this walk is the
    anchor alone, which is where an unchecked one would be returned as a term
    of the axis with nothing to distinguish it.
    """
    _refuse_unless_known(hierarchy, node_id)
    return drive(hierarchy, _children_at_depth(node_id, depth), cache=cache)


async def async_children_at_depth(
    hierarchy: AsyncHierarchy[K],
    node_id: K,
    depth: int,
    *,
    max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
    cache: WalkCache | None = None,
) -> tuple[K, ...]:
    """:func:`children_at_depth` over an asynchronous hierarchy."""
    await _async_refuse_unless_known(hierarchy, node_id)
    return await async_drive(
        hierarchy,
        _children_at_depth(node_id, depth),
        max_concurrency=max_concurrency,
        cache=cache,
    )


def flatten(
    hierarchy: Hierarchy[K],
    *,
    from_id: K | None = None,
    cache: WalkCache | None = None,
) -> tuple[K, ...]:
    """The axis from a point, or the whole of it, **pre-order by discovery**.

    ``from_id`` is **included**. Omitting it descends from every root, and each
    root's subtree is emitted whole before the next begins.

    What ``roots()`` reaches is what this returns: a node no edge mentions is
    not in the hierarchy at all, and a cyclic component with no root above it
    is unreachable from here. That is a property of the protocol rather than of
    this walk.

    **A ``from_id`` the axis does not contain is refused**, because this walk
    emits it; see :func:`_refuse_an_unknown_anchor`. Omitting it refuses
    nothing, there being no anchor to check -- an axis with no roots at all is
    walked and returns ``()``.
    """
    _refuse_unless_known(hierarchy, from_id)
    return drive(hierarchy, _flatten(from_id), cache=cache)


async def async_flatten(
    hierarchy: AsyncHierarchy[K],
    *,
    from_id: K | None = None,
    max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
    cache: WalkCache | None = None,
) -> tuple[K, ...]:
    """:func:`flatten` over an asynchronous hierarchy."""
    await _async_refuse_unless_known(hierarchy, from_id)
    return await async_drive(
        hierarchy, _flatten(from_id), max_concurrency=max_concurrency, cache=cache
    )


def leaves(
    hierarchy: Hierarchy[K],
    *,
    under: K | None = None,
    cache: WalkCache | None = None,
) -> tuple[K, ...]:
    """Every node with no children at or under ``under``, in :func:`flatten`'s order.

    ``under`` is **included if it is one**: a childless node is its own only
    leaf. Omitting it takes the leaves of the whole axis.

    **Childlessness is the reply, not the discovery edges.** A node the descent
    discovered nothing through is either a leaf or a node whose every child had
    already been reached along another path, and over a DAG those are different
    answers -- so what separates them is whether the *reply* was empty, which
    the descent sees and keeps. This walk therefore costs exactly the descent:
    the backing is asked about each node once, with a cache or without one.

    **An ``under`` the axis does not contain is refused**, and this walk is the
    one that reaches that answer by the longest route: an unknown anchor
    discovers nothing, is confirmed childless because a backing has no children
    for a node it has never heard of, and would come back as a leaf of the
    axis. See :func:`_refuse_an_unknown_anchor`.
    """
    _refuse_unless_known(hierarchy, under)
    return drive(hierarchy, _leaves(under), cache=cache)


async def async_leaves(
    hierarchy: AsyncHierarchy[K],
    *,
    under: K | None = None,
    max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
    cache: WalkCache | None = None,
) -> tuple[K, ...]:
    """:func:`leaves` over an asynchronous hierarchy."""
    await _async_refuse_unless_known(hierarchy, under)
    return await async_drive(
        hierarchy, _leaves(under), max_concurrency=max_concurrency, cache=cache
    )


def paths_to_root(
    hierarchy: Hierarchy[K], node_id: K, *, cache: WalkCache | None = None
) -> tuple[tuple[K, ...], ...]:
    """Every way up from ``node_id``, one tuple per path, the node at index 0.

    A node with two parents gives **two paths**, which is the answer
    :func:`ancestors` cannot give: that walk returns the set of nodes above,
    deduplicated, and a consumer asking *how did we get here* needs the routes
    rather than the membership. The cycle guard is scoped to the path for the
    same reason -- a node reachable two ways appears on both.

    Each path runs from ``node_id`` outward to a node that cannot be extended:
    a root, or a node whose every parent is already on that path. **Cyclic
    data therefore returns paths that reach no root**, and nothing is raised:
    the far end is simply a node whose parents are all behind it.

    Paths come back in **parent order, outermost first** -- every path through
    a node's first parent precedes every path through its second, with
    ``parents()``'s own order deciding which is which.

    **An anchor the axis does not contain is refused**, because this walk emits
    it: an unknown node comes back as ``((node_id,),)``, which is exactly the
    shape a root gives. See :func:`_refuse_an_unknown_anchor`.

    **This walk keeps its own edge replies**, because it is the one walk here
    that asks about a node more than once by construction -- a node on ``k``
    paths is reached ``k`` times. The memo is per call and costs the caller
    nothing to ask for; ``cache`` is still :func:`drive`'s and still for
    spending replies across *other* walks.
    """
    _refuse_unless_known(hierarchy, node_id)
    return drive(hierarchy, _paths_to_root(node_id), cache=cache)


async def async_paths_to_root(
    hierarchy: AsyncHierarchy[K],
    node_id: K,
    *,
    max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
    cache: WalkCache | None = None,
) -> tuple[tuple[K, ...], ...]:
    """:func:`paths_to_root` over an asynchronous hierarchy.

    ``max_concurrency`` is forwarded and **never binds**, which is worth saying
    rather than leaving to be measured: this walk is a depth-first descent and
    asks about one node at a time, so there is no frontier for a bound to
    narrow. It is taken because the driver takes it and because a twin that
    quietly refused a keyword its siblings accept is the surface drift the
    parity guard exists to catch -- not because setting it changes anything
    here.
    """
    await _async_refuse_unless_known(hierarchy, node_id)
    return await async_drive(
        hierarchy, _paths_to_root(node_id), max_concurrency=max_concurrency, cache=cache
    )


def deepest_common_ancestor(
    hierarchy: Hierarchy[K], a: K, b: K, *, cache: WalkCache | None = None
) -> K | None:
    """The nearest node above both ``a`` and ``b``, or ``None`` if there is none.

    **Either argument may be the answer**: each chain includes its own end, so
    an ancestor of the other node is returned rather than passed over.

    **The tie-break is asymmetric.** The answer is the first of ``a``'s
    ancestry, nearest first, that also stands above ``b``. Over a tree that
    names one node however the arguments are ordered; over a DAG two common
    ancestors can be incomparable, and then the one nearer ``a`` wins --
    swapping the arguments can swap the answer.

    **``None`` is ambiguous and is not refused.** A disjoint pair, an unknown
    ``a`` and an unknown ``b`` all answer ``None``. That is ambiguous rather
    than wrong -- nothing false is returned -- and ``contains()`` separates
    them in one call, which is :func:`ancestors`' treatment and for the same
    reason.

    **This walk re-asks the ancestry the two nodes share, and a ``cache`` is
    the only thing that stops it.** It composes two ancestor walks, and a
    composed walk cannot see the replies its parts receive, so the shared chain
    above the meeting point is asked for twice: 44 requests against 23 distinct
    edges over a 23-node axis, and 23 with ``cache={}``. The excess is bounded
    by the depth of the shared chain rather than by the size of the axis. Pass
    a cache where that matters; there is no other mechanism.
    """
    return drive(hierarchy, _deepest_common_ancestor(a, b), cache=cache)


async def async_deepest_common_ancestor(
    hierarchy: AsyncHierarchy[K],
    a: K,
    b: K,
    *,
    max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
    cache: WalkCache | None = None,
) -> K | None:
    """:func:`deepest_common_ancestor` over an asynchronous hierarchy.

    The two ancestor walks it composes run **in sequence**, not concurrently:
    ``max_concurrency`` bounds the frontier read inside each of them, which is
    where the concurrency in this walk is. The re-ask its synchronous twin
    documents is the same here, and so is the remedy.
    """
    return await async_drive(
        hierarchy,
        _deepest_common_ancestor(a, b),
        max_concurrency=max_concurrency,
        cache=cache,
    )


# --------------------------------------------------------------------------
# The anchored view -- a cursor over one node
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HierarchyView(Generic[K]):
    """One :class:`Hierarchy`, one node. A cursor, not a node.

    It owns nothing and copies nothing: the structure is the caller's, and two
    views over one hierarchy are equal iff they name the same node. There is no
    state between calls, so nothing is cached and nothing needs invalidating
    when the structure beneath it changes -- a view obtained before a rebuild
    and used after it reads the new structure. That is why it holds the
    hierarchy rather than a snapshot of it, and why it can be frozen at all.

    **No member below has an algorithm of its own.** ``exists``, ``parents``
    and ``children`` each invoke one protocol member and re-wrap; ``is_root``
    and ``is_leaf`` invoke ``exists`` and one more; ``at`` constructs. The
    walks -- ``ancestors``, ``descendants``, ``paths_to_root`` -- are not on it
    yet, and ``roots`` never will be: it is the protocol's, asked as
    ``view.structure.roots()``.

    **A node that is not here is neither a root nor a leaf.** ``is_root()`` and
    ``is_leaf()`` are ``False`` wherever ``exists()`` is ``False``, and that
    guard is the whole reason they are not one line each. Without it
    ``is_leaf()`` answers *fully specified* about a node the structure has
    never heard of -- an absent node has an empty ``children()`` too -- and a
    consumer deciding whether a placement is specific enough to act on would
    decline to ask a narrowing question on exactly the terms it knows nothing
    about. Measured on a public vocabulary whose ``isa`` axis named 92 of its
    170 terms, that was 46% of it. The cost, accepted: after the guard,
    ``False`` no longer separates *absent* from *has both parents and
    children*. ``exists()`` is what does, and it is safe to call alone.

    **It hashes, so a walk can key its visited set on it -- and it hashes
    exactly as far as what it holds does.** Frozen means a generated
    ``__hash__`` over the **field tuple**, which is both fields: the
    :class:`Hierarchy` and the key. So the capability is reported here rather
    than required, and either field can withhold it.

    Every backing this package ships gives it: both mapping twins are
    compared by identity for that purpose, and the assertion-backed pair holds
    a source that hashes as any object does over a relation it canonicalises
    to an id. ``str`` keys give it. A
    structure of your own that does not, or a key type of your own that does
    not, makes a cursor over it answer ``isinstance(view, Hashable)`` with
    ``True`` and raise at ``hash()``. **The ``Hashable`` bound on ``K`` does
    not catch the key half**: a frozen dataclass over a ``dict`` satisfies the
    bound and raises, so the bound documents the requirement rather than
    enforcing it. Declare such a type ``eq=False``, or hold its contents in
    something hashable.

    Generic in the key with ``str`` defaulted, like the protocol it holds, so
    a bare ``HierarchyView`` is ``HierarchyView[str]``.
    """

    structure: Hierarchy[K]
    node: K

    def exists(self) -> bool:
        """Whether the structure knows this node at all."""
        return self.structure.contains(self.node)

    def is_root(self) -> bool:
        """Present, with nothing above it. ``False`` for an absent node."""
        return self.exists() and not self.structure.parents(self.node)

    def is_leaf(self) -> bool:
        """Present, with nothing below it. ``False`` for an absent node."""
        return self.exists() and not self.structure.children(self.node)

    def parents(self) -> tuple[HierarchyView[K], ...]:
        """One view per node directly above this one. Plural, always."""
        return tuple(
            HierarchyView(self.structure, above) for above in self.structure.parents(self.node)
        )

    def children(self) -> tuple[HierarchyView[K], ...]:
        """One view per node directly below this one."""
        return tuple(
            HierarchyView(self.structure, below) for below in self.structure.children(self.node)
        )

    def at(self, node_id: K) -> HierarchyView[K]:
        """Re-anchor at another node of the same structure.

        Constructs rather than reads, so it checks nothing: the view it
        returns answers :meth:`exists` itself, at the point a caller asks,
        which is what keeps *nothing below this node* and *this node is not
        here* apart.
        """
        return HierarchyView(self.structure, node_id)


@dataclass(frozen=True)
class AsyncHierarchyView(Generic[K]):
    """The twin: the same six members over an :class:`AsyncHierarchy`.

    Every one is ``async def`` bar :meth:`at`, which awaits nothing because it
    constructs rather than reads -- the same rule that makes
    :meth:`AsyncMappingHierarchy.from_nested` a plain ``def``.

    It hashes on the same terms :class:`HierarchyView` states, from both of the
    same fields: both asynchronous backings this package ships hash, and so
    does a ``str`` key.
    """

    structure: AsyncHierarchy[K]
    node: K

    async def exists(self) -> bool:
        """Whether the structure knows this node at all."""
        return await self.structure.contains(self.node)

    async def is_root(self) -> bool:
        """Present, with nothing above it. ``False`` for an absent node."""
        if not await self.exists():
            return False
        return not await self.structure.parents(self.node)

    async def is_leaf(self) -> bool:
        """Present, with nothing below it. ``False`` for an absent node."""
        if not await self.exists():
            return False
        return not await self.structure.children(self.node)

    async def parents(self) -> tuple[AsyncHierarchyView[K], ...]:
        """One view per node directly above this one. Plural, always."""
        return tuple(
            AsyncHierarchyView(self.structure, above)
            for above in await self.structure.parents(self.node)
        )

    async def children(self) -> tuple[AsyncHierarchyView[K], ...]:
        """One view per node directly below this one."""
        return tuple(
            AsyncHierarchyView(self.structure, below)
            for below in await self.structure.children(self.node)
        )

    def at(self, node_id: K) -> AsyncHierarchyView[K]:
        """:meth:`HierarchyView.at`, and a plain ``def`` for the same reason."""
        return AsyncHierarchyView(self.structure, node_id)


# --------------------------------------------------------------------------
# The mapping backings -- a structure carried in memory
# --------------------------------------------------------------------------


def _invert(parent_map: Mapping[K, Sequence[K]]) -> dict[K, tuple[K, ...]]:
    """A child mapping from a parent mapping.

    Once, at construction. ``children()`` over a parent mapping *is* this
    inversion, and doing it per call makes a downward walk quadratic in the
    size of the axis -- which is the walk a snapshot takes, so the cost would
    land on the constructor that exists to make walking cheap.
    """
    children: dict[K, list[K]] = {}
    for node_id, above in parent_map.items():
        for parent_id in above:
            children.setdefault(parent_id, []).append(node_id)
    return {parent_id: tuple(below) for parent_id, below in children.items()}


def _nested_parent_map(tree: Any, child_key: str, name_key: str) -> dict[str, tuple[str, ...]]:
    """A parent mapping from a nested tree, keyed by the ids the path mints.

    The keys are :mod:`dataknobs_common._nested_core`'s and not this module's,
    which is the whole point of that core: a document read through an ontology's
    ``kind: nested`` and the same tree held in memory here mint one set of ids,
    so a consumer holding both is holding one vocabulary rather than two.
    """
    return {
        minted.id: () if minted.parent_id is None else (minted.parent_id,)
        for minted in _mint_tree(tree, child_key=child_key, name_key=name_key)
    }


@dataclass(frozen=True, eq=False)
class _MappingBacking(Generic[K]):
    """The mapping, its inversion, and the six answers computed from them.

    Shared by the twins below rather than written into each, for the reason
    every shared thing here is shared: a rule a twin re-implements is a rule
    that drifts, and ``contains`` in particular is now a *refusal's evidence*
    -- :meth:`~dataknobs_common.ontology.taxonomy.Taxonomy.walk` refuses an anchor its
    axis does not contain by asking it. Two implementations of that answer is
    two chances to make the refusal fire on a node that is genuinely there.

    Not public and not a base anyone should reach for: it is the body of two
    classes, and the classes are the surface.

    **Compared by identity, and every twin's own decorator has to say so.**
    A backing holding a ``Mapping`` cannot be hashed field-wise -- a dict does
    not hash however frozen its owner is -- and a cursor over it inherits that:
    :class:`HierarchyView` is frozen, so its generated ``__hash__`` reaches
    whatever structure it was handed, and a backing that raises there makes the
    cursor promise the capability at ``isinstance`` and fail at the call. That
    is the trade :class:`~dataknobs_common.ontology.taxonomy.Taxonomy` already
    took, for the same reason, and it costs nothing that was being used: two
    mappings holding the same edges are compared with ``parent_edges()``.

    **Forced rather than chosen**, which is the part worth keeping when this
    looks like a restriction to lift. The mapping is the caller's, held by
    reference: its contents move under this object, so a hash derived from
    them would change while a cursor over it sits in the very ``seen`` set the
    hash exists to serve -- the hash invariant broken in the one place it is
    load-bearing. ``child_map`` is inverted once at construction besides, so
    after a caller mutates, ``parents()`` moves and ``children()`` does not.
    This was never a value. Restoring field-wise equality would reintroduce
    both defects at once.

    ``eq=False`` **on this class does not travel to a subclass.**
    ``@dataclass`` regenerates ``__eq__`` on every class it decorates, and a
    frozen dataclass with ``eq=True`` gets the field-wise ``__hash__`` back
    with it -- so a twin declared ``@dataclass(frozen=True)`` reaches
    ``parent_map`` again however this body is declared. Each twin below
    therefore carries the flag itself, and a test enumerates the subclasses so
    that a third added without it fails there rather than at a consumer's
    ``hash()``.
    """

    #: A node's parents. **Not named ``parents``**: a dataclass field and a
    #: protocol member may not share a name, and ``parents`` is a member. The
    #: caller's spelling is unchanged, because the argument was always
    #: positional.
    parent_map: Mapping[K, Sequence[K]]

    #: :func:`_invert` of :attr:`parent_map`, computed once at construction.
    child_map: Mapping[K, Sequence[K]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_map", _invert(self.parent_map))

    def _nodes(self) -> tuple[K, ...]:
        """Every node this axis knows, in a stable order.

        A key of either mapping. The second half is the one a naive reading
        misses: a node that appears **only as a parent** is never a key of
        ``parent_map``, and it is as much a term of the axis as any other.
        """
        return tuple(dict.fromkeys((*self.parent_map, *self.child_map)))

    def _roots(self) -> Sequence[K]:
        return tuple(node_id for node_id in self._nodes() if not self.parent_map.get(node_id))

    def _parents(self, node_id: K) -> Sequence[K]:
        return tuple(self.parent_map.get(node_id, ()))

    def _children(self, node_id: K) -> Sequence[K]:
        return tuple(self.child_map.get(node_id, ()))

    def _contains(self, node_id: K) -> bool:
        """Exactly the set a walk can reach: a key, or a value under any key.

        The inversion is what makes that one lookup rather than a scan --
        ``child_map`` is keyed by every node that appears as somebody's parent,
        which is precisely the second half of the set.
        """
        return node_id in self.parent_map or node_id in self.child_map

    def _edges(self) -> Mapping[K, Sequence[K]]:
        """Every node, and what each is directly under.

        Built over :meth:`_nodes` rather than handed back as
        :attr:`parent_map`, and the difference is the whole of what makes this
        an *extent*: a node that appears only as somebody's parent is never a
        key of that mapping, so returning it would omit precisely the nodes
        ``roots()`` reports. A fresh dict also keeps a caller from holding this
        backing's own mapping.
        """
        return {node_id: self._parents(node_id) for node_id in self._nodes()}

    def _parents_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        return tuple(self._parents(node_id) for node_id in node_ids)

    def _children_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        return tuple(self._children(node_id) for node_id in node_ids)


@dataclass(frozen=True, eq=False)
class MappingHierarchy(_MappingBacking[K]):
    """A structure carried in memory: a node's parents, as a mapping.

    The hierarchy for a vocabulary somebody typed, or holds, or has just
    finished walking. It opens nothing, so it lives beside the protocols rather
    than with a backing package -- a concrete goes where its dependency is, and
    this one has none.

    It implements :class:`BulkHierarchy` as well as :class:`Hierarchy`, because
    one dict lookup per node is the cheapest ``parents_many`` there is and
    declining to offer it would be the decision needing an argument.

    Three ways in, and each answers a different question::

        MappingHierarchy({"beagle": ("dog",), "dog": ("mammal",)})   # carried
        MappingHierarchy.from_nested({"name": "Billing", "children": [...]})
        MappingHierarchy.snapshot(live_axis)

    ``roots()`` is *every node with no parents*, which includes a node that
    appears only as somebody's parent and is never a key.

    Frozen and **compared by identity** -- ``eq=False`` written here rather
    than inherited, for the reason :class:`_MappingBacking` states. It is what
    lets a cursor over this backing be put in a set.
    """

    def roots(self) -> Sequence[K]:
        """The nodes this axis has with no parent."""
        return self._roots()

    def parents(self, node_id: K) -> Sequence[K]:
        """The nodes ``node_id`` is directly under."""
        return self._parents(node_id)

    def children(self, node_id: K) -> Sequence[K]:
        """The nodes directly under ``node_id``."""
        return self._children(node_id)

    def contains(self, node_id: K) -> bool:
        """Whether this axis knows the node at all."""
        return self._contains(node_id)

    def parents_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`parents` for a whole frontier, one reply per node."""
        return self._parents_many(node_ids)

    def children_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`children` for a whole frontier, one reply per node."""
        return self._children_many(node_ids)

    def parent_edges(self) -> Mapping[K, Sequence[K]]:
        """Every node this axis knows, and what each is directly under."""
        return self._edges()

    @classmethod
    def from_nested(
        cls,
        tree: Any,
        *,
        child_key: str = "children",
        name_key: str = "name",
    ) -> MappingHierarchy[str]:
        """A hierarchy from a nested tree, keyed by the ids the paths mint.

        The shape a person maintains by hand: a node *is* its path, and there is
        no id field anywhere. The id is a slug of the path and is stable across
        a rename, because a name that renamed itself would re-key every
        descendant of a renamed interior node at once.

        Keyed by ``str`` whatever ``K`` this is called on, since a minted id is
        a slug. Two paths that slug to one id are refused, naming both -- the
        slug collapses punctuation and case, so ``Late Fees`` and ``late-fees``
        are two nodes to the person editing the file and one to the slug.

        Args:
            tree: A mapping, or a list of them for a forest. ``None`` is empty.
            child_key: Where a node keeps its children.
            name_key: Where a node keeps its name.

        Returns:
            A hierarchy over the minted ids -- **the same ids** an ontology
            gives the same tree under ``kind: nested``.

        Raises:
            ValidationError: On two paths that slug to one id.
        """
        return MappingHierarchy(_nested_parent_map(tree, child_key, name_key))

    @classmethod
    def snapshot(cls, hierarchy: Hierarchy[K]) -> MappingHierarchy[K]:
        """A live axis read once, kept as a mapping.

        The cheap copy -- ids and edges, not content -- and the one thing a
        live axis cannot do: say what has changed since it was taken. Walking
        it afterwards asks the mapping rather than the backing.

        **How complete the copy is, is a property of the axis.** An axis that
        implements :class:`EnumerableHierarchy` is asked what it holds and is
        copied whole. One that does not is *descended from its roots*, which is
        the only enumeration the four members offer -- and a cyclic component
        with no root above it is unreachable that way, so it is absent from the
        copy, and the copy then refuses an anchor the live axis accepts. Where
        that matters, give the backing ``parent_edges()``.

        The dispatch is written into both twins rather than into the drivers,
        and the asymmetry is deliberate: a fourth :data:`Member` would be a
        *capability* added to the driver pair, which the module docstring
        prices as the expensive kind of change. Two three-line branches is the
        cheaper shape, and it is the last place either flavour needs one.
        """
        edges = getattr(hierarchy, "parent_edges", None)
        if edges is not None:
            return cls(dict(cast("Mapping[K, Sequence[K]]", edges())))
        return cls(drive(hierarchy, _parent_edges()))


@dataclass(frozen=True, eq=False)
class AsyncMappingHierarchy(_MappingBacking[K]):
    """The same mapping, in an :class:`AsyncHierarchy`-shaped slot.

    It awaits nothing, and that is the point rather than an oversight: a
    consumer typed against :class:`AsyncHierarchy` -- because the *rest* of
    their vocabulary is by-reference -- still needs something hand-built to put
    in the slot, and hand-writing one per test is what this exists to stop.

    ``eq=False`` here too, and written out for the same reason: a difference
    between the twins on this point is one :func:`assert_twin_types_agree`
    cannot see, since it reads members and this is a decision about the type.
    """

    async def roots(self) -> Sequence[K]:
        """The nodes this axis has with no parent."""
        return self._roots()

    async def parents(self, node_id: K) -> Sequence[K]:
        """The nodes ``node_id`` is directly under."""
        return self._parents(node_id)

    async def children(self, node_id: K) -> Sequence[K]:
        """The nodes directly under ``node_id``."""
        return self._children(node_id)

    async def contains(self, node_id: K) -> bool:
        """Whether this axis knows the node at all."""
        return self._contains(node_id)

    async def parents_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`parents` for a whole frontier, one reply per node."""
        return self._parents_many(node_ids)

    async def children_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`children` for a whole frontier, one reply per node."""
        return self._children_many(node_ids)

    async def parent_edges(self) -> Mapping[K, Sequence[K]]:
        """Every node this axis knows, and what each is directly under."""
        return self._edges()

    @classmethod
    def from_nested(
        cls,
        tree: Any,
        *,
        child_key: str = "children",
        name_key: str = "name",
    ) -> AsyncMappingHierarchy[str]:
        """:meth:`MappingHierarchy.from_nested`, into an asynchronous slot.

        A plain ``def``, and deliberately: a tree held in memory has nothing to
        await, so making this awaitable would cost every caller an ``await``
        for a traversal of their own data.
        """
        return AsyncMappingHierarchy(_nested_parent_map(tree, child_key, name_key))

    @classmethod
    async def snapshot(
        cls,
        hierarchy: AsyncHierarchy[K],
        *,
        max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
    ) -> AsyncMappingHierarchy[K]:
        """:meth:`MappingHierarchy.snapshot`, over a live asynchronous axis.

        ``async`` where the nested constructor is not, because this one reads
        the axis: the awaiting is real here and absent there.

        ``max_concurrency`` is forwarded to :func:`async_drive` and carries that
        function's meaning exactly -- see :data:`DEFAULT_FRONTIER_CONCURRENCY`.
        It is a keyword here for the same reason it is one on every other
        asynchronous entry point: only the caller knows what their backing is,
        and a snapshot walks the *whole* axis rather than one branch of it. An
        axis that can enumerate never reaches the walk, so it never reaches the
        bound either -- one query has no frontier to fan out over.
        """
        _refuse_an_unusable_bound(max_concurrency)
        edges = getattr(hierarchy, "parent_edges", None)
        if edges is not None:
            return cls(dict(cast("Mapping[K, Sequence[K]]", await edges())))
        return cls(await async_drive(hierarchy, _parent_edges(), max_concurrency=max_concurrency))


if TYPE_CHECKING:  # pragma: no cover - checked by the type checker, not run
    from typing import assert_type

    from dataknobs_common.bounded_cache import BoundedLRUCache

    def _the_cache_seam_admits_the_cache_this_package_ships() -> None:
        """``BoundedLRUCache`` satisfies :class:`WalkCache`, checked not claimed.

        The reason that seam is a two-member Protocol rather than
        ``MutableMapping`` is that this class implements every member the ABC
        requires *without inheriting it*, and an ABC matches nominally. That is
        a claim about a relationship between two modules, and nothing executed
        can notice it lapsing: the behavioural test hands one to ``flatten``
        and would go on passing whatever the annotations said.

        **The value parameter is the half that is easy to get wrong.** It has
        to be the reply type, because ``get`` is read covariantly: annotate the
        cache ``BoundedLRUCache[..., object]`` and ``get`` returns
        ``object | None`` where the seam promises ``Sequence[Any] | None``, so
        the cache does not satisfy the seam a caller is about to hand it to.
        This package's own guide carried that annotation until this proof was
        written, which is the argument for writing it here.
        """

        def _satisfies(cache: BoundedLRUCache[WalkCacheKey, Sequence[Any]]) -> WalkCache:
            return cache

        def _spends(
            axis: Hierarchy[str], cache: BoundedLRUCache[WalkCacheKey, Sequence[Any]]
        ) -> None:
            """And it is accepted where a caller would actually pass one."""
            flatten(axis, cache=cache)
            leaves(axis, cache=cache)

        del _satisfies, _spends

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

        def _bare_concrete(axis: MappingHierarchy, async_axis: AsyncMappingHierarchy) -> None:
            """The concretes inherit the default, and would lose it silently.

            ``_MappingBacking`` is where ``K`` is declared for both, so a
            parameter list that stopped naming it here would widen an
            unannotated backing to ``Any`` with nothing to report it.
            """
            assert_type(axis.roots(), "Sequence[str]")
            assert_type(async_axis.parent_map, "Mapping[str, Sequence[str]]")

        def _bare_view(view: HierarchyView, async_view: AsyncHierarchyView) -> None:
            """The cursors inherit the default the same way."""
            assert_type(view, "HierarchyView[str]")
            assert_type(view.parents(), "tuple[HierarchyView[str], ...]")
            assert_type(async_view.node, "str")

        del _bare, _bare_async, _inferred, _bare_concrete, _bare_view
