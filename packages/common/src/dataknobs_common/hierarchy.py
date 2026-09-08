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
arrangement costs more than a hand-twinned pair would. What it buys from the
second walk on is a lower *marginal* cost rather than a lower total: the pair
is paid once, but it is not small, and it grows when a **capability** is added
to it -- bulk frontier dispatch did -- where a walk added *over* it does not.
That distinction is the bet. The shape it exists to avoid is
``dataknobs_data``'s ``_search_with_complex_query``: 57 lines in each flavour,
differing in three.

The key type is a parameter with ``str`` **defaulted**, so a bare ``Hierarchy``
is ``Hierarchy[str]`` and reads as it always did. It exists because the walks
never *inspect* a node id -- they only hash one -- so an object tree with no id
at all can bind ``K`` to its own node type and share the same traversals.
"""

from __future__ import annotations

import sys
from collections.abc import Generator, Hashable, Sequence
from typing import TYPE_CHECKING, Generic, Literal, Protocol, cast, runtime_checkable

# The algorithms and the frontier read live in the private sibling: a core that
# is only a paragraph is satisfied by two twins that each implement it, so what
# is shared is code both flavours call. The runtime dependency runs this way
# only -- the core names these protocols in annotations and nothing more.
from dataknobs_common._walk_core import (
    _ancestors,
    _async_reply,
    _refuse_a_spent_walk,
    _sync_reply,
)

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


def drive(hierarchy: Hierarchy[_K], walk: Walk[_K, _T]) -> _T:
    """Run a walk against a synchronous hierarchy.

    The ``except`` below must see the walk's own return and nothing else.
    ``StopIteration`` from the *core* cannot reach it -- PEP 479 converts one
    escaping a generator body into ``RuntimeError`` at the frame boundary --
    and one from the *hierarchy* is converted by :func:`_sync_reply` before it
    is raised, so the reply call is safe inside the ``try``.

    The walk's own state is checked *before* the ``try``, for the same reason:
    a spent generator raises ``StopIteration(value=None)`` from the opening
    ``next``, which lands in that ``except`` and reads as the walk finishing.
    """
    _refuse_a_spent_walk(walk)
    try:
        member, node_ids = next(walk)
        while True:
            member, node_ids = walk.send(_sync_reply(hierarchy, member, node_ids))
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
) -> _T:
    """Run a walk against an asynchronous hierarchy.

    One round of concurrency **per depth** rather than per node, because the
    core asks for a whole frontier at a time. A twinned walk gets that only if
    somebody remembers to write it into each copy.

    ``max_concurrency`` bounds that round where the backing offers no bulk
    members; see :data:`DEFAULT_FRONTIER_CONCURRENCY` for why a bound is not
    optional. It is a keyword here and a constant nowhere the core can read,
    which is what keeps the walk core free of configuration.
    """
    _refuse_a_spent_walk(walk)
    try:
        member, node_ids = next(walk)
        while True:
            member, node_ids = walk.send(
                await _async_reply(hierarchy, member, node_ids, max_concurrency=max_concurrency)
            )
    except StopIteration as stop:
        return cast("_T", stop.value)


# --------------------------------------------------------------------------
# The public walks
# --------------------------------------------------------------------------


def ancestors(hierarchy: Hierarchy[K], node_id: K) -> tuple[K, ...]:
    """Every node above ``node_id``, nearest first.

    Excludes ``node_id`` itself, terminates on cyclic data, and returns each
    node once. ``K`` is inferred from ``hierarchy``.

    **An empty result means either a root or an unknown node**, and this does
    not refuse the second — unlike
    :meth:`~dataknobs_common.taxonomy.Taxonomy.walk`, which refuses an anchor
    its axis does not contain. The difference is recoverability rather than
    taste. ``walk`` *includes* its anchor, so an unknown one is emitted as a
    term of the axis and the caller receives a wrong answer they cannot
    detect. This excludes its anchor, so nothing false is returned: the answer
    is ambiguous, not incorrect, and ``hierarchy.contains(node_id)`` — a member
    every backing must implement — resolves it in one call. Ask it first where
    the distinction matters.
    """
    return drive(hierarchy, _ancestors(node_id))


async def async_ancestors(
    hierarchy: AsyncHierarchy[K],
    node_id: K,
    *,
    max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
) -> tuple[K, ...]:
    """:func:`ancestors` over an asynchronous hierarchy.

    The same generator drives both flavours -- this is not a second
    implementation of the walk, and a test asserts that patching the core moves
    both surfaces. ``max_concurrency`` is forwarded to the driver; the
    synchronous twin has no counterpart because it issues no concurrent calls.

    An empty result carries the same ambiguity :func:`ancestors` describes, and
    is resolved the same way.
    """
    return await async_drive(hierarchy, _ancestors(node_id), max_concurrency=max_concurrency)


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
