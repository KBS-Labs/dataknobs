"""The walk core: the algorithms, and the step that reads a frontier.

Private, and a module rather than a convention. A core that is only a
paragraph is satisfied by two twins that each implement the paragraph -- so
what is shared here is *code both flavours call*, not a rule both flavours
follow.

Three shared cores live here, and they are shared with different callers:

* the **algorithms** -- ``_expand``, the two ways of reading what it
  discovers, the walks composed from those, ``_paths_to_root``, and
  ``_parent_edges`` -- which :mod:`dataknobs_common.hierarchy`'s public
  wrappers and its snapshot constructors drive. There are **two** expansions
  here and not one: ``_expand`` dedups per walk, which six walks want, and
  ``_paths_to_root`` guards per path, which is the whole of why a walk that
  returns paths cannot be composed over the other one;
* the **frontier read** -- ``_sync_reply`` / ``_async_reply`` -- which the
  drivers call, and which :class:`~dataknobs_common.ontology.taxonomy.Taxonomy`'s
  streaming walk calls too. That walk cannot go through the collecting core,
  because streaming would widen the core's request type for every walk that
  does not stream; sharing this one step is what stops it drifting from the
  drivers over which backings answer a level in one query. It is also where
  the walk **memo** lives, and for that reason: written into the drivers it
  would leave the one walk that does not use them as the only walk without it;
* the **freshness check** -- ``_refuse_a_spent_walk`` -- which both drivers
  make before their opening ``next``. Shared for the reason every refusal here
  is: a rule a twin re-implements is a rule that drifts, and the twins' whole
  claim is that they behave alike.

Both consumers import from here, which is why this is a sibling of
``hierarchy.py`` rather than something inside it: a private name reached out
of a *public* module is a boundary crossed, and the same name reached out of
a private core is two consumers of one core.

**Fetching and emission are separate questions**, and keeping them separate is
what lets six walks share one descent. ``_expand`` decides what is asked and in
what rounds; ``_levels_of`` and ``_preorder_of`` decide what order the answer
comes back in. Conflating them makes a pre-ordered walk look like a second
algorithm the level-synchronous core cannot afford, when it is a second reading
of edges the core already had in hand.

The dependency runs one way at runtime -- ``hierarchy`` imports this, never
the reverse. The protocols are annotations here and nothing more.
"""

from __future__ import annotations

import asyncio
import inspect
import types
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataknobs_common.hierarchy import (
        AsyncHierarchy,
        Hierarchy,
        Member,
        Walk,
        _K,
    )


def _refuse_a_spent_walk(walk: Walk[_K, object]) -> None:
    """Refuse a walk that is not fresh, before a driver's first ``next``.

    A walk is a generator and therefore single-use, and the drivers did not say
    so. ``drive`` opens with ``next(walk)`` inside the ``try`` whose ``except
    StopIteration`` learns the walk's return value -- so an already-exhausted
    generator raised ``StopIteration(value=None)`` there and was read as *the
    walk finished*, handing the caller ``None`` cast to the declared result
    type. The failure then surfaced wherever that value was next used.

    A *suspended* walk is refused too, and for a worse reason than exhaustion:
    resuming one mid-traversal sends ``None`` where the reply to its
    outstanding question belongs, so it would answer rather than fail.

    A walk that is not a generator *object* is refused first, and by kind. The
    state this reads lives on the generator itself, so a ``Walk`` satisfying the
    alias structurally -- a class implementing ``collections.abc.Generator``,
    which the drivers would otherwise drive, since they only ever ``next`` and
    ``send`` -- reached ``inspect.getgeneratorstate`` and raised
    ``AttributeError`` naming a CPython slot.

    Refusing it narrows what the drivers accept, and the narrowing is the point.
    Skipping the check for what it cannot inspect was the other candidate: it
    reopens the ``None``-as-a-result hazard above for exactly the walks the
    driver would be unable to warn about, which is the wrong direction to fail
    in. So the kind is part of the contract, and the message carries the
    one-line way to keep a delegating walk -- a generator function using
    ``yield from`` is a generator, where a class is not.

    Shared by both drivers rather than written into each -- the refusals either
    twin makes are the same refusals, and a rule a twin re-implements is a rule
    that drifts. Deliberately not folded into the ``StopIteration`` conversion
    below: that converts an exception raised by the *hierarchy*, where this is
    the walk's own state, which is not an error condition anywhere else.
    """
    if not isinstance(walk, types.GeneratorType):
        raise TypeError(
            f"a walk must be a generator object, not {type(walk).__name__}; "
            f"the drivers read a walk's state to refuse a spent one, and only "
            f"a generator carries it. Write a delegating walk as a generator "
            f"function -- 'def wrapped(): return (yield from inner)' -- rather "
            f"than as a class"
        )

    state = inspect.getgeneratorstate(walk)
    if state != inspect.GEN_CREATED:
        raise RuntimeError(
            f"this walk has already been driven (generator state {state}); "
            f"a walk is single-use, so build a fresh one per drive"
        )


def _refuse_an_unusable_bound(max_concurrency: int) -> None:
    """Refuse a frontier bound that cannot admit anybody.

    A semaphore of zero admits nothing, so a walk given one waits forever
    rather than failing -- which is why this is a refusal and not a clamp.

    Shared rather than written at each site because there are now two, and they
    are reached by *different* routes: :func:`_async_reply` validates on the way
    to building the semaphore, and :meth:`AsyncMappingHierarchy.snapshot`
    validates before a branch that never gets there. A backing that can answer
    without a frontier -- bulk members, or ``parent_edges()`` -- is exactly the
    one whose caller would otherwise have a deadlocking width silently accepted,
    because nothing on their path ever looked at it.
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be at least 1, got {max_concurrency}")


#: How a walk cache keys one remembered reply: the member that was asked, and
#: the node it was asked about.
#:
#: ``roots`` never appears in one. It is not a per-node reply, and it is
#: deliberately not cached: exactly one walk asks for it, once, and remembering
#: it would cost that walk its only chance to notice the axis grew a root --
#: which is precisely the promise about the *world* that a memo over one walk
#: refuses to make.
#:
#: **What is not in it: which hierarchy was asked.** A cache is therefore
#: scoped to *one axis*, and handing one to walks over two different axes
#: answers the second from the first's edges -- a wrong answer with no
#: exception anywhere. That is a limit rather than an omission: the only
#: discriminator available is the backing's own identity, and a
#: :class:`~dataknobs_common.hierarchy.Hierarchy` is arbitrary consumer code
#: that need not be hashable -- a plain ``@dataclass`` backing has
#: ``__hash__`` of ``None`` -- while ``id()`` is reused after a collection and
#: would answer a *new* axis from a dead one's entries. So the scope is stated
#: and kept by the caller, who is also the only party that knows how many axes
#: they hold.
WalkCacheKey = tuple["Member", Any]


class WalkCache(Protocol):
    """Somewhere edge replies outlive the walk that fetched them. Two members.

    **Always the caller's, and never a default.** No walk composed here asks
    its backing about a node twice -- the descent asks each node once and
    :func:`_leaves` reads childlessness off the reply rather than asking again
    -- so a memo built per walk would fill with entries that walk will never
    read. What a cache is *for* is the next walk: a second walk over the same
    axis, a streaming walk warmed by a collecting one, or the two walks
    ``Taxonomy.subtree_keys`` delegates to.

    That is also why the lifetime cannot live here. A cache that outlives a
    walk is a claim about how fast the axis changes, and the caller is the only
    party who knows.

    **Two members because the walk only ever reads and fills.** It never
    invalidates, never expires an entry and never asks whether one is still
    good. If a caller's cache turns out to need any of that, then the thing
    being handed in is not a mapping and this is the wrong seam -- the answer
    then is a narrower object with an explicit contract, not a wider Protocol.

    **A Protocol rather than ``MutableMapping``, and that is measured rather
    than preferred.** ``MutableMapping`` is the annotation this invites, and it
    excludes this package's own :class:`~dataknobs_common.bounded_cache.BoundedLRUCache`:
    the class implements every member the ABC requires and derives from
    ``Generic``, and an ABC matches nominally, so implementing the whole
    interface is not enough. Two members admit a ``dict``, admit that cache,
    and admit whatever a consumer already holds.

    What it holds is **edge replies, not results**. A caller who needs a
    result's *order* to survive keeps the result.

    **It may be bounded, and nothing here assumes it is not.** A cache smaller
    than a frontier evicts its own entries while that frontier is being filled,
    which is why :func:`_partition` carries its hits out rather than reading
    them back after the fetch -- and why a walk's cost never depends on a hit.

    **One cache, one axis.** :data:`WalkCacheKey` says why the hierarchy is not
    in the key and cannot be; the consequence is that a cache outliving a walk
    belongs to the axis it was filled from, and a caller holding several axes
    holds several caches.
    """

    def get(self, key: WalkCacheKey, default: None = None) -> Sequence[Any] | None:
        """The reply remembered for ``key``, or ``None`` when there is none."""
        ...

    def __setitem__(self, key: WalkCacheKey, value: Sequence[Any]) -> None:
        """Remember ``value`` as the reply for ``key``."""
        ...


def _partition(
    cache: WalkCache, member: Member, node_ids: tuple[_K, ...]
) -> tuple[dict[_K, Sequence[_K]], tuple[_K, ...]]:
    """What the cache can answer, and which ids it cannot -- in one pass.

    The hits are carried *out* rather than read back after the fetch, and that
    is not tidiness. A caller-supplied cache may be **bounded**: a frontier
    wider than the cache evicts the cache's own earlier entries while it is
    being filled, so a fill-then-read would miss an id it had answered a moment
    earlier. Nothing about that failure is visible in a test whose cache is a
    ``dict``.

    Deduplicated, because a reply is per node and asking twice in one request
    buys nothing. Every walk here already hands over a deduplicated frontier;
    this makes it a property of the step rather than of each caller.
    """
    known: dict[_K, Sequence[_K]] = {}
    missing: list[_K] = []
    asked: set[_K] = set()
    for node_id in node_ids:
        if node_id in asked:
            continue
        asked.add(node_id)
        remembered = cache.get((member, node_id))
        if remembered is None:
            missing.append(node_id)
        else:
            known[node_id] = remembered
    return known, tuple(missing)


def _remember(
    cache: WalkCache,
    member: Member,
    node_ids: tuple[_K, ...],
    replies: tuple[Sequence[_K], ...],
) -> None:
    """Fill the cache with what was just fetched, one entry per node."""
    for node_id, reply in zip(node_ids, replies, strict=True):
        cache[(member, node_id)] = reply


def _stitch(
    known: dict[_K, Sequence[_K]],
    node_ids: tuple[_K, ...],
    fetched_ids: tuple[_K, ...],
    fetched: tuple[Sequence[_K], ...],
) -> tuple[Sequence[_K], ...]:
    """One reply per id **in request order**, from the two sources together.

    The drivers hand replies back positionally, so this is the step where a
    memo could silently corrupt a walk: a reply in the wrong slot is a wrong
    answer with no exception anywhere.
    """
    answers: dict[_K, Sequence[_K]] = dict(known)
    answers.update(zip(fetched_ids, fetched, strict=True))
    return tuple(answers[node_id] for node_id in node_ids)


def _checked_bulk_reply(
    hierarchy: object,
    member: Member,
    node_ids: tuple[_K, ...],
    replies: tuple[Sequence[_K], ...],
) -> tuple[Sequence[_K], ...]:
    """Refuse a bulk reply that is not one sequence per node, naming whose it is.

    :class:`~dataknobs_common.hierarchy.BulkHierarchy` states the contract --
    one reply per node asked about, in the order asked, an empty sequence where
    there is no answer -- and a backing that drops the empty ones breaks it in
    the way a query naturally does, since a join returns no row for a node with
    no match.

    **Every path already catches it, and catches it badly.** The frontier and
    its replies are paired with ``zip(..., strict=True)`` three frames further
    on, so a short reply raises ``zip() argument 2 is shorter than argument 1``
    -- a message naming an argument position and neither the backing, the
    member, nor the counts. The contract is the bulk member's, so this is where
    its breach is reported; the ``strict`` pairings stay, as the assertion that
    this ran.

    Both flavours call it for the reason every rule here is shared: a check one
    twin re-implements is a check that drifts.
    """
    if len(replies) != len(node_ids):
        raise ValueError(
            f"{type(hierarchy).__name__}.{member}_many() answered {len(replies)} "
            f"replies for {len(node_ids)} nodes. A bulk member answers one reply "
            f"per node, in the order asked, with an empty sequence where there is "
            f"no answer -- a dropped reply silently shifts every later node's "
            f"edges onto the wrong id."
        )
    return replies


def _sync_fetch(
    hierarchy: Hierarchy[_K], member: Member, node_ids: tuple[_K, ...]
) -> tuple[Sequence[_K], ...]:
    """One bulk call where the backing offers one, else one call per node.

    The single implementation of that rule -- :class:`Taxonomy`'s streaming
    walk reaches it through :func:`_sync_reply` rather than deciding again, so
    the carve-out walk cannot drift from the driver on which backings get a
    per-level query.
    """
    bulk = getattr(hierarchy, f"{member}_many", None)
    if bulk is not None:
        return _checked_bulk_reply(hierarchy, member, node_ids, tuple(bulk(node_ids)))
    one = hierarchy.parents if member == "parents" else hierarchy.children
    return tuple(one(node_id) for node_id in node_ids)


async def _async_fetch(
    hierarchy: AsyncHierarchy[_K],
    member: Member,
    node_ids: tuple[_K, ...],
    *,
    max_concurrency: int,
) -> tuple[Sequence[_K], ...]:
    """:func:`_sync_fetch`'s twin: bulk where offered, else a *bounded* round.

    ``gather`` runs one round trip per node concurrently; ``children_many`` is
    one *query* for the level. The second is what a row-backed hierarchy wants,
    and concurrency does not substitute for it.

    ``max_concurrency`` is an argument with no default rather than a constant
    read here, which is the whole of why this module still holds no
    configuration: the number is policy, it lives with the public surface, and
    a caller who knows their backing can move it. Without a bound the fan-out
    was the width of the *level* -- a property of the data -- so a node with
    ten thousand children issued ten thousand concurrent calls into whatever
    the backing was. The bulk path never had the problem, which means the
    unbounded path was exactly the one taken by backings least able to absorb
    it.
    """
    bulk = getattr(hierarchy, f"{member}_many", None)
    if bulk is not None:
        return _checked_bulk_reply(hierarchy, member, node_ids, tuple(await bulk(node_ids)))
    one = hierarchy.parents if member == "parents" else hierarchy.children
    limit = asyncio.Semaphore(max_concurrency)

    async def _bounded(node_id: _K) -> Sequence[_K]:
        async with limit:
            return await one(node_id)

    return tuple(await asyncio.gather(*(_bounded(n) for n in node_ids)))


def _sync_reply(
    hierarchy: Hierarchy[_K],
    member: Member,
    node_ids: tuple[_K, ...],
    *,
    cache: WalkCache | None = None,
) -> tuple[Sequence[_K], ...]:
    """Answer one request, from the memo where it can and the backing where it must.

    **The memo lives here rather than in either driver**, which is the whole of
    why it reaches everything: the drivers both call this step, and
    :class:`Taxonomy`'s streaming walk calls it *directly* rather than going
    through a driver. A memo written into the drivers would leave the one walk
    that does not use them as the only walk without it.

    ``cache=None`` means no memo at all, and it is the default everywhere: no
    walk composed here asks about a node twice, so there is nothing for a
    per-walk memo to answer. A cache is the caller's, and it is for the walks
    that come after this one.

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
            if cache is None:
                return _sync_fetch(hierarchy, member, node_ids)
            known, missing = _partition(cache, member, node_ids)
            fetched = _sync_fetch(hierarchy, member, missing) if missing else ()
            _remember(cache, member, missing, fetched)
            return _stitch(known, node_ids, missing, fetched)
    except StopIteration as stop:
        raise RuntimeError(
            f"hierarchy {type(hierarchy).__name__}.{member}() raised StopIteration"
        ) from stop
    raise ValueError(f"unknown hierarchy member {member!r}")


async def _async_reply(
    hierarchy: AsyncHierarchy[_K],
    member: Member,
    node_ids: tuple[_K, ...],
    *,
    max_concurrency: int,
    cache: WalkCache | None = None,
) -> tuple[Sequence[_K], ...]:
    """:func:`_sync_reply`'s twin, cache and all.

    The bound is validated here rather than in :func:`_async_fetch`, because a
    request answered entirely from the memo never reaches the fetch -- and a
    width that cannot admit anybody should be refused on the way past, not only
    on the walks unlucky enough to miss.

    No explicit ``StopIteration`` conversion here: this is a coroutine, so the
    language performs it at this frame's boundary and a collaborator's
    ``StopIteration`` surfaces as the same ``RuntimeError`` the synchronous
    side raises by hand.
    """
    _refuse_an_unusable_bound(max_concurrency)
    if member == "roots":
        return (await hierarchy.roots(),)
    if member in ("parents", "children"):
        if cache is None:
            return await _async_fetch(hierarchy, member, node_ids, max_concurrency=max_concurrency)
        known, missing = _partition(cache, member, node_ids)
        fetched = (
            await _async_fetch(hierarchy, member, missing, max_concurrency=max_concurrency)
            if missing
            else ()
        )
        _remember(cache, member, missing, fetched)
        return _stitch(known, node_ids, missing, fetched)
    raise ValueError(f"unknown hierarchy member {member!r}")


#: Which way a level expansion goes.
Direction = Literal["parents", "children"]


def _expand(
    seeds: tuple[_K, ...],
    direction: Direction,
    *,
    max_depth: int | None = None,
) -> Walk[_K, tuple[dict[_K, tuple[_K, ...]], set[_K]]]:
    """Expand level by level, keeping **which node discovered which**.

    The one descent every walk below is written over, and the only place a
    request is issued. Four properties, each of which is a way to get a walk
    wrong:

    * the visited set is **unconditional** -- the data may be cyclic whatever
      any acyclicity constraint claims, and a walk that terminates by luck is
      indistinguishable from one that terminates by construction until it does
      not;
    * discovery is **deduplicated in walk order** -- a DAG node is reachable by
      several paths, and a repeated id changes no membership answer but does
      change a count someone is reporting. It is recorded against the *first*
      node to reach it, which is what makes the pre-order projection below
      well defined;
    * the frontier is asked **one level at a time**, which is what lets
      :func:`~dataknobs_common.hierarchy.async_drive` issue one round of
      concurrency per depth and a bulk backing answer a level in one query;
    * **childlessness is recorded here, because here is the only place it is
      visible.** A node whose discovery edges are empty is either childless or
      a node whose every child had already been reached by another path, and
      over a DAG those are two different answers. Which one it is is in the
      *reply*, and the reply is in this frame -- so the one bit is kept rather
      than thrown away and asked for again. That is the whole of why
      :func:`_leaves` issues no second request.

    **What it keeps and a flat level list discards is the pairing** between a
    frontier and its replies. Both projections below are recoverable from it
    and it is not recoverable from either, which is why the descent returns
    this and the choice of order is made afterwards. Fetching and emission are
    separate questions, and conflating them is what made pre-order look like a
    second algorithm that this core could not afford.

    ``max_depth`` counts **expansions**, so ``0`` asks nothing at all and the
    seeds are the whole answer. It is a bound on the descent and never an
    order selector -- both projections read the same discovery edges whether
    the descent stopped early or ran out.

    **What ``childless`` covers is exactly what was asked**, and ``discovered``
    says which nodes those were: it takes a key for every node that entered a
    frontier and for no other. A bounded descent stops before asking its last
    level, so a node absent from ``discovered`` has had no reply about it and
    is neither childless nor known to have children. A reader wanting leafness
    over a bound has to consult both, which is why both are returned.
    """
    seen = set(seeds)
    discovered: dict[_K, tuple[_K, ...]] = {}
    childless: set[_K] = set()
    frontier = seeds
    depth = 0
    while frontier and (max_depth is None or depth < max_depth):
        replies = yield (direction, frontier)
        fresh: list[_K] = []
        for node_id, reply in zip(frontier, replies, strict=True):
            mine: list[_K] = []
            for neighbour in reply:
                if neighbour not in seen:
                    seen.add(neighbour)
                    mine.append(neighbour)
                    fresh.append(neighbour)
            discovered[node_id] = tuple(mine)
            if not reply:
                childless.add(node_id)
        frontier = tuple(fresh)
        depth += 1
    return discovered, childless


def _levels_of(
    discovered: dict[_K, tuple[_K, ...]],
    seeds: tuple[_K, ...],
    *,
    exclude_seeds: bool,
) -> tuple[tuple[_K, ...], ...]:
    """The discovery edges read as levels -- one tuple per depth.

    Level order, which is a claim about **distance**: everything at depth one
    precedes everything at depth two. That is what ``ancestors`` publishes as
    *nearest first*, and it is why that walk reads the descent this way rather
    than the other.

    The seeds are excluded or included **at the boundary**, by the caller's
    flag, rather than by a rule inside the descent.
    """
    levels: list[tuple[_K, ...]] = [] if exclude_seeds else [seeds]
    frontier = seeds
    while True:
        fresh = tuple(found for node_id in frontier for found in discovered.get(node_id, ()))
        if not fresh:
            return tuple(levels)
        levels.append(fresh)
        frontier = fresh


def _preorder_of(
    discovered: dict[_K, tuple[_K, ...]],
    seeds: tuple[_K, ...],
    *,
    exclude_seeds: bool,
) -> tuple[_K, ...]:
    """The same discovery edges read as **pre-order by discovery** -- flat.

    Each node, then everything first reached through it, then the next. Over a
    tree this is the depth-first order a recursive walk produces, and the
    equivalence is a property of **trees**: on a DAG the two disciplines dedup
    at different first visits and diverge, which is why this publishes itself
    as *pre-order by discovery* rather than as depth-first. It costs no second
    request, no second visited set and no second algorithm -- the edges it
    reads were already in hand.

    Seeds in order, each followed by its own subtree, so a forest -- ``leaves``
    and ``flatten`` descend from every root -- comes back grouped rather than
    interleaved. A seed is never discovered by anything, so excluding one drops
    exactly it and keeps everything beneath.
    """
    skip = frozenset(seeds) if exclude_seeds else frozenset()
    out: list[_K] = []
    stack: list[_K] = list(reversed(seeds))
    while stack:
        node_id = stack.pop()
        if node_id not in skip:
            out.append(node_id)
        stack.extend(reversed(discovered.get(node_id, ())))
    return tuple(out)


def _levels(
    seeds: tuple[_K, ...],
    direction: Direction,
    *,
    exclude_seeds: bool,
    max_depth: int | None = None,
) -> Walk[_K, tuple[tuple[_K, ...], ...]]:
    """:func:`_expand`, read as levels. The level-ordered half of the core.

    Levels rather than a flat tuple because levels are what a frontier-at-a-time
    expansion produces, and because one caller wants a single one of them:
    ``children_at_depth`` indexes where ``ancestors`` flattens. Flattening is a
    composition's choice rather than a decision this makes on every caller's
    behalf.
    """
    discovered, _ = yield from _expand(seeds, direction, max_depth=max_depth)
    return _levels_of(discovered, seeds, exclude_seeds=exclude_seeds)


def _flat(levels: tuple[tuple[_K, ...], ...]) -> tuple[_K, ...]:
    """One tuple, in level order."""
    return tuple(node_id for level in levels for node_id in level)


def _ancestors(node_id: _K) -> Walk[_K, tuple[_K, ...]]:
    """Every node above this one, nearest first, excluding the node itself."""
    return _flat((yield from _levels((node_id,), "parents", exclude_seeds=True)))


def _seeds_under(anchor: _K | None) -> Walk[_K, tuple[_K, ...]]:
    """``anchor`` alone, or the axis's roots when there is none.

    The two walks whose anchor is optional -- ``flatten`` and ``leaves`` --
    descend from a forest when it is omitted, and this is the one request that
    difference costs. It is deliberately **not** memoised: ``roots`` is asked
    once per walk, and remembering it across walks would cost a caller their
    only chance to notice the axis grew a root.

    **The reply is deduplicated here**, which is where the descent's promise
    about repeated ids begins rather than a tidiness. The seed tuple is read by
    three steps -- the descent and both of its projections -- so it is the one
    frontier none of them can dedup on its own behalf: :func:`_expand` records
    discovery against the *first* node to reach something, and a seed visited
    twice reaches nothing the second time, so an unguarded repeat replaces a
    real record with an empty one and the whole subtree stops being reachable.
    ``roots()`` is arbitrary consumer code -- a query over a join answers one
    row per match -- and every later frontier is already deduplicated by the
    visited set, so this reply is the only one that can arrive repeated.
    """
    if anchor is not None:
        return (anchor,)
    (roots,) = yield ("roots", ())
    return tuple(dict.fromkeys(roots))


def _descendants(node_id: _K) -> Walk[_K, tuple[_K, ...]]:
    """Every node below this one, pre-order by discovery, **excluding** it.

    The exclusion is the half of a boundary: ``subtree_keys`` supplies the
    including half over the same descent, and the difference between them is
    the whole of what that boundary is.
    """
    discovered, _ = yield from _expand((node_id,), "children")
    return _preorder_of(discovered, (node_id,), exclude_seeds=True)


def _flatten(from_id: _K | None) -> Walk[_K, tuple[_K, ...]]:
    """Everything at or under ``from_id``, or the whole axis from its roots.

    The anchor is **included** -- this is the axis from a point rather than the
    strict descendants of it -- and the order is the same pre-order every
    descending flattened walk here emits, unbounded being no different from
    bounded.
    """
    seeds = yield from _seeds_under(from_id)
    discovered, _ = yield from _expand(seeds, "children")
    return _preorder_of(discovered, seeds, exclude_seeds=False)


def _descendants_to_depth(node_id: _K, max_depth: int) -> Walk[_K, tuple[_K, ...]]:
    """:func:`_flatten` from one node, bounded. The anchor is **included**.

    ``max_depth`` counts levels below the anchor, so ``0`` is the anchor alone
    and ``1`` is the anchor and its children. A negative bound reads as ``0``
    rather than as an error, which is what the recursion this replaces does.

    **The bound is a bound and never an order selector**: this and
    :func:`_flatten` are one rule read at two depths, so the unbounded walk is
    exactly this one with nothing to stop it.
    """
    discovered, _ = yield from _expand((node_id,), "children", max_depth=max(max_depth, 0))
    return _preorder_of(discovered, (node_id,), exclude_seeds=False)


def _children_at_depth(node_id: _K, depth: int) -> Walk[_K, tuple[_K, ...]]:
    """Exactly the nodes ``depth`` levels below ``node_id``. A level, unflattened.

    ``depth=0`` is the anchor itself, ``1`` its children. **The one walk here
    with no emission order to choose** -- a level is a set of equals, and the
    order within it is the order the backing answered in.

    Returns ``()`` for a depth the axis does not reach, which is a real answer
    rather than a missing one: nothing is that far below the anchor.
    """
    wanted = max(depth, 0)
    levels = yield from _levels((node_id,), "children", exclude_seeds=False, max_depth=wanted)
    return levels[wanted] if wanted < len(levels) else ()


def _leaves(under: _K | None) -> Walk[_K, tuple[_K, ...]]:
    """Every node with no children at or under ``under``, in the same pre-order.

    The anchor is **included if it is one**: a childless node is its own only
    leaf.

    **Childlessness is the reply, not the discovery edges** -- and this walk
    reads it rather than asking for it again. A node whose discovery edges are
    empty is either a leaf or a node whose every child had already been reached
    by another path, and over a DAG those are different answers; what tells
    them apart is whether the *reply* was empty, which :func:`_expand` sees and
    now keeps.

    **So there is no second request**, and the walk costs exactly the descent.
    There was one: the candidates -- the nodes that discovered nothing -- were
    asked again, for edges the descent already had and discarded. A memo made
    that ask free when a caller happened to supply one big enough to hold the
    frontier, which is two conditions a walk should not have to meet to avoid
    fetching the same edge twice.
    """
    seeds = yield from _seeds_under(under)
    discovered, childless = yield from _expand(seeds, "children")
    order = _preorder_of(discovered, seeds, exclude_seeds=False)
    return tuple(node_id for node_id in order if node_id in childless)


def _paths_to_root(node_id: _K) -> Walk[_K, tuple[tuple[_K, ...], ...]]:
    """Every maximal path upward from ``node_id``, the anchor at the near end.

    **The second algorithm**, and the whole of what makes it one is the scope
    of its cycle guard. :func:`_expand` dedups **per walk**, which is right for
    the six walks composed over it and wrong here: a node reachable by two
    paths is two answers, not one, and a walk-scoped visited set collapses them
    into whichever path arrived first. So the guard is a membership test
    against **the path being extended**, and the same node appears on as many
    paths as reach it.

    **A path is emitted when it cannot be extended**, which is one rule
    covering two endings: a node with no parents is a root, and a node whose
    every parent is already on the path closed a cycle. Neither is emitted as a
    *prefix* of a longer path -- a path truncated where one of several parents
    cycled would be exactly that, and a caller reading the result as *the ways
    up from here* would count a way that is really the beginning of another.

    **Cyclic data therefore terminates without reaching a root**, and the
    result says so by shape rather than by a flag: the far end of such a path
    is a node whose parents are all behind it. An axis with no root above a
    cyclic component has no root to offer, which is a property of the data --
    :func:`_parent_edges` says the same thing from the other direction.

    **Parent order, outermost first.** The stack reverses, so parents are
    pushed reversed to come back in the order ``parents()`` gave them: every
    path through the first parent precedes every path through the second. That
    is a decision rather than an accident of the container -- what a walk is
    made of and what it emits are two questions, and a stack answers the second
    one by default if nobody answers it deliberately.

    **A repeated parent in one reply is one edge, not two routes.** ``parents``
    is arbitrary consumer code -- a query over a join answers one row per match
    -- and an unguarded repeat here would return the same path twice, where
    :func:`_expand`'s visited set absorbs one silently and
    :func:`_parent_edges` guards against one by name. The dedup is inside the
    reply and nothing wider: a parent reached again on a *different* path is
    two routes and stays two.

    **The reply memo is scoped to the walk, and that is not the guard's
    scope.** The two structures sit four lines apart and hold opposite scopes
    on purpose: reachability is a property of a *path*, so the guard is
    per-path; an edge is not, so ``parents(n)`` is the same reply whichever
    path arrived at ``n`` and asking again buys nothing. Sharing one scope
    between them is a defect in either direction -- a path-scoped memo
    memoises nothing, and a walk-scoped guard returns one path where two are
    owed.

    This walk is the reason that memo exists at all. Every walk over
    :func:`_expand` asks each node once by construction, so a memo built for
    one would fill with entries it never reads back; this one re-asks by
    construction, because a node on ``k`` paths is popped ``k`` times. Measured
    over a chain of stacked diamonds: **125 requests for 16 nodes** without it,
    **16** with, identical to what a caller's ``cache={}`` achieves -- and the
    memo is kept here rather than left to that cache because a walk that can
    keep its own replies should not need a caller to notice.
    """
    paths: list[tuple[_K, ...]] = []
    stack: list[tuple[_K, ...]] = [(node_id,)]
    replies: dict[_K, Sequence[_K]] = {}
    while stack:
        path = stack.pop()
        tip = path[-1]
        if tip in replies:
            above = replies[tip]
        else:
            (above,) = yield ("parents", (tip,))
            replies[tip] = above
        onward = tuple(dict.fromkeys(p for p in above if p not in path))
        if not onward:
            paths.append(path)
            continue
        stack.extend((*path, parent) for parent in reversed(onward))
    return tuple(paths)


def _deepest_common_ancestor(a: _K, b: _K) -> Walk[_K, _K | None]:
    """The nearest node above both ``a`` and ``b``, or ``None``.

    **A composition, not a third algorithm**: :func:`_ancestors` twice, joined
    by a membership test. ``yield from`` has no flavour, which is the property
    this is the worked example of -- written as twins the asynchronous half
    would have to call an asynchronous ``ancestors`` where the synchronous half
    calls a synchronous one, and two wrappers deep is where a re-walk gets
    written by accident.

    **Either argument may be the answer.** Each chain includes its own end, so
    a node that is an ancestor of the other is returned rather than skipped
    over -- *deepest common ancestor* and not *deepest proper common
    ancestor*.

    **The tie-break is asymmetric and is part of the definition.** The answer
    is the first of ``a``'s chain, nearest first, that also stands above ``b``.
    Over a tree there is only one candidate and the asymmetry is invisible;
    over a DAG two common ancestors can be incomparable, and then the one
    nearer ``a`` wins. Swapping the arguments can therefore swap the answer,
    which is why this is stated rather than left to be discovered.

    **It re-asks the chain the two share, and cannot stop itself.** ``yield
    from`` delegates a request straight past this frame, so the replies its two
    sub-walks receive are never observed here and there is nowhere to keep
    them: measured, this frame sees **0** of them. The cost is the depth of the
    shared chain and not the size of the axis -- 44 requests against 23
    distinct edges over a 23-node axis, 23 of them with a caller's cache. For
    this walk a cache is not a refinement a careful caller reaches for; it is
    the only mechanism there is.
    """
    a_chain = (a, *(yield from _ancestors(a)))
    b_chain = {b, *(yield from _ancestors(b))}
    return next((above for above in a_chain if above in b_chain), None)


def _parent_edges() -> Walk[_K, dict[_K, tuple[_K, ...]]]:
    """Every edge the axis has, as a parent mapping, by descending from roots.

    The walk behind a snapshot. It is written here, as a walk over the shared
    core, rather than twinned into the two constructors that take one -- which
    is the arrangement ``hierarchy``'s module docstring calls the bet: a walk
    added *over* the drivers costs one generator, where a capability added *to*
    them costs a pair.

    Descending is the only enumeration a :class:`Hierarchy` offers. There is no
    extent member -- ``roots()`` says so in its own docstring -- so what a
    snapshot can see is what is reachable downward from the roots. **A cyclic
    component with no root above it is therefore not in the snapshot**, and
    that is a property of the protocol rather than of this walk: nothing here
    can name a node no member mentions.

    Deduplication is unconditional, like :func:`_levels`: a node is expanded
    once however many parents reach it, while *every* parent that reaches it is
    kept -- because a DAG node with two parents is the case the whole structure
    axis exists to represent, and dropping one would make the snapshot a
    different shape from the axis it copied.
    """
    (roots,) = yield ("roots", ())
    parents: dict[_K, list[_K]] = {node_id: [] for node_id in roots}
    frontier = tuple(parents)
    while frontier:
        replies = yield ("children", frontier)
        fresh: list[_K] = []
        for node_id, children in zip(frontier, replies, strict=True):
            for child in children:
                if child not in parents:
                    parents[child] = []
                    fresh.append(child)
                if node_id not in parents[child]:
                    parents[child].append(node_id)
        frontier = tuple(fresh)
    return {node_id: tuple(above) for node_id, above in parents.items()}
