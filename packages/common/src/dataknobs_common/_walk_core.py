"""The walk core: the algorithms, and the step that reads a frontier.

Private, and a module rather than a convention. A core that is only a
paragraph is satisfied by two twins that each implement the paragraph -- so
what is shared here is *code both flavours call*, not a rule both flavours
follow.

Three shared cores live here, and they are shared with different callers:

* the **algorithms** -- ``_expand``, the ways of reading what it collects,
  and the walks composed from those -- which
  :mod:`dataknobs_common.hierarchy`'s public wrappers and its snapshot
  constructors drive. There is **one** expansion here and not two: every walk
  below is a reading of it, the one behind a snapshot included, and including
  the two that dedup differently from the rest, because what they dedup
  differently is the *emission* and not the fetch;
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
what lets every walk here share one descent. ``_expand`` decides what is asked
and in what rounds; ``_levels_of``, ``_preorder_of`` and ``_paths_of`` decide
what order the answer comes back in, and ``_upward_order`` and ``_above_within``
read the same replies for questions that are not about order at all. Conflating them makes a differently-ordered
walk look like a second algorithm the level-synchronous core cannot afford,
when it is a third reading of edges the core already had in hand -- and the
descent keeps the **replies** rather than a spanning subset of them precisely
so that a reading wanting routes rather than membership has them to read.

The dependency runs one way at runtime -- ``hierarchy`` imports this, never
the reverse. The protocols are annotations here and nothing more.
"""

from __future__ import annotations

import asyncio
import inspect
import types
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol

from dataknobs_common.exceptions import OperationError

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


def _refuse_an_unusable_bound(bound: int, name: str) -> None:
    """Refuse a bound that cannot admit anybody. **A refusal and never a clamp.**

    Two bounds follow this rule and neither has a sensible reading below one. A
    semaphore of zero admits nothing, so a walk given one waits forever rather
    than failing. A route ceiling of zero rejects every axis including a bare
    root, so a walk given one fails whatever it is pointed at. Raising either
    silently to one would answer a question the caller did not ask, which is
    the difference between a bound and a clamp.

    Shared rather than written at each site because a rule re-implemented is a
    rule that drifts, and these sites are reached by *different* routes:
    :func:`_async_reply` validates on the way to building the semaphore,
    :meth:`AsyncMappingHierarchy.snapshot` validates before a branch that never
    gets there, and :func:`_paths_of` validates before an enumeration that
    might otherwise refuse for the wrong reason. A backing that can answer
    without a frontier -- bulk members, or ``parent_edges()`` -- is exactly the
    one whose caller would otherwise have a deadlocking width silently
    accepted, because nothing on their path ever looked at it.

    ``name`` is the caller's parameter rather than this function's, because the
    message a consumer reads has to name the keyword they typed.
    """
    if bound < 1:
        raise ValueError(f"{name} must be at least 1, got {bound}")


def _refuse_an_unaffordable_answer(anchor: object, max_paths: int) -> NoReturn:
    """Refuse an enumeration the caller declared they could not hold.

    **The only cost a walk here publishes that is a property of the answer**
    rather than of the fetch. Every ascent costs one request per node however
    branchy the axis is; the number of maximal routes up it doubles per stacked
    branch point, so a sixty-one node axis can carry a million ways up. That is
    a correct answer and an unaffordable one, and the two are not
    distinguishable in advance without enumerating -- which is the thing being
    refused.

    Raised rather than returned short, for the reason stated at the site: a
    truncated tuple says *at least this many* where the walk's contract says
    *these and no others*. The context carries both the anchor and the ceiling,
    because the useful next step is a different question over the same node --
    :func:`_ancestors` for membership -- and a message naming only the limit
    does not say which node was too branchy.
    """
    raise OperationError(
        f"more than {max_paths} maximal paths above {anchor!r}; the ascent is "
        f"one request per node however branchy the axis, but the number of "
        f"routes up it is not bounded by that -- raise max_paths if the answer "
        f"is affordable, or ask `ancestors` for membership instead of routes",
        context={"anchor": anchor, "max_paths": max_paths},
    )


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

    **Always the caller's, and never a default.** No walk here asks its
    backing about a node twice, and that is structural rather than a tally over
    the walks that happen to exist: every one of them is a *reading* of one
    level-synchronous descent, the descent visits a node once, and a reading
    issues no request at all. A walk wanting an edge the descent discarded is
    the shape that breaks it -- so :func:`_expand` keeps the replies, and the
    two readings that want routes and comparability read them there. A memo
    built per walk would therefore fill with entries that walk never reads.
    What a cache is *for* is the next walk: a second walk over the same axis, a
    streaming walk warmed by a collecting one, or the two walks
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

    ``cache=None`` means no memo at all, and it is the default everywhere: a
    walk here is a reading of one descent that visits a node once, so there is
    nothing for a per-walk memo to answer. A cache is the caller's, and it is
    for the walks that come after this one.

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
    _refuse_an_unusable_bound(max_concurrency, "max_concurrency")
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
) -> Walk[_K, tuple[dict[_K, tuple[_K, ...]], dict[_K, tuple[_K, ...]]]]:
    """Expand level by level, keeping **the replies** and which node discovered which.

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
      concurrency per depth and a bulk backing answer a level in one query.
      **Every walk here is a reading of this descent**, so that property is the
      module's and not one each walk has to re-earn;
    * **the reply is kept, because this frame is the only place it is
      visible.** The discovery edges are a *spanning* subset of it: every edge
      into a node already seen is dropped, and three different readers want an
      edge this frame threw away. :func:`_leaves` wants to know whether the
      reply was *empty* -- over a DAG a node can discover nothing and still
      have children, so childlessness is the reply rather than the discovery
      edges. :func:`_paths_of` wants every edge, because a node reached by two
      parents is two routes where it is one member. :func:`_above_within` wants
      the induced subgraph, because *is this common ancestor above that one*
      cannot be answered from a spanning tree.

    That last property is the childlessness rule generalised, and generalised
    because keeping one bit of a reply is what made the next two readers look
    like second algorithms. It costs the difference between the induced edge
    set and a spanning one -- nothing over a tree, and the non-tree edges over
    a DAG -- which is the memory this walk had already fetched and was
    discarding.

    **The reply is deduplicated here.** ``parents()`` and ``children()`` are
    arbitrary consumer code and a query over a join answers one row per match,
    so a repeated neighbour is one edge rather than two. ``discovered``
    absorbed a repeat silently through the visited set and ``edges`` cannot,
    because a reader that counts routes would count that one twice -- so the
    guard is a property of the descent rather than of whichever projection
    remembers to ask for it.

    **What it keeps and a flat level list discards is the pairing** between a
    frontier and its replies. All three projections below are recoverable from
    it and it is not recoverable from any of them, which is why the descent
    returns this and the choice of order is made afterwards. Fetching and
    emission are separate questions, and conflating them is what made pre-order
    look like a second algorithm that this core could not afford.

    ``max_depth`` counts **expansions**, so ``0`` asks nothing at all and the
    seeds are the whole answer. It is a bound on the descent and never an
    order selector -- every projection reads the same edges whether the descent
    stopped early or ran out.

    **``edges`` covers exactly what was asked**: it takes a key for every node
    that entered a frontier and for no other, so its keys are the boundary a
    bounded descent stopped at. A node absent from it has had no reply about it
    and is neither childless, nor known to have children, nor known to be the
    far end of a route -- which is why **every reader below indexes it** rather
    than reaching for a default, and why a bounded descent is not a thing those
    readings can be given. A default would answer *nothing above it* for a node
    nobody asked about, which is a bound reported as a property of the axis.
    """
    seen = set(seeds)
    discovered: dict[_K, tuple[_K, ...]] = {}
    edges: dict[_K, tuple[_K, ...]] = {}
    frontier = seeds
    depth = 0
    while frontier and (max_depth is None or depth < max_depth):
        replies = yield (direction, frontier)
        fresh: list[_K] = []
        for node_id, reply in zip(frontier, replies, strict=True):
            neighbours = tuple(dict.fromkeys(reply))
            mine: list[_K] = []
            for neighbour in neighbours:
                if neighbour not in seen:
                    seen.add(neighbour)
                    mine.append(neighbour)
                    fresh.append(neighbour)
            # ``mine`` is a subsequence of ``neighbours`` by construction, so
            # equal lengths mean nothing was dropped -- which is every node of a
            # tree. Rebuilding there would hold two equal tuples per node and
            # double the memory against the shape the docstring above promises
            # pays nothing. Both are immutable, so sharing one is invisible.
            discovered[node_id] = neighbours if len(mine) == len(neighbours) else tuple(mine)
            edges[node_id] = neighbours
        frontier = tuple(fresh)
        depth += 1
    return discovered, edges


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


def _paths_of(
    edges: dict[_K, tuple[_K, ...]], anchor: _K, max_paths: int | None = None
) -> tuple[tuple[_K, ...], ...]:
    """The same replies read as **every maximal route** from ``anchor``. One tuple each.

    The third projection, and the one that shows why the descent keeps replies
    rather than the spanning subset of them. :func:`_expand` records a node
    against the *first* frontier to reach it and drops every later edge into
    it, which is right for the two projections above -- a node reachable two
    ways is one member of a set -- and is exactly what a route is not. Reading
    the replies instead, a node reached by two parents is two routes and stays
    two.

    **The cycle guard is scoped to the route being extended**, not to the walk,
    for the same reason: reachability is a property of a route, so a
    walk-scoped visited set here returns one path where a diamond owes two.
    That scope is what makes this a different *reading* and not a different
    algorithm -- the guard is a membership test against a tuple this frame
    already holds, and the edges it reads were in hand before it started. No
    request is issued here at all.

    **A route is emitted where it cannot be extended**, which is one rule
    covering two endings: a node with no parents is a root, and a node whose
    every parent is already behind it closed a cycle. Neither is emitted as a
    *prefix* of a longer route -- a route truncated where one of several
    parents cycled would be exactly that, and a caller reading the result as
    *the ways up from here* would count a way that is really the beginning of
    another. **The two endings are not distinguished in the result**, because
    the result is routes and not a verdict about the axis; ``edges[route[-1]]``
    tells them apart for a caller holding the descent, and
    ``hierarchy.parents(route[-1])`` for one holding only the answer.

    **Parent order, outermost first.** The stack reverses, so parents are
    pushed reversed to come back in the order the reply gave them: every route
    through the first parent precedes every route through the second. That is a
    decision rather than an accident of the container -- what a projection is
    made of and what it emits are two questions, and a stack answers the second
    by default if nobody answers it deliberately.

    **The descent must be unbounded**, which is why this indexes ``edges``
    rather than reaching for a default. A bounded one stops before asking its
    last level, and a node it never asked about is not *unextendable* -- it is
    unknown, and emitting a route that ends there would report a bound as a
    property of the axis. :func:`_upward_order` and :func:`_above_within` index
    for the same reason: a reader that accepted an input this one refuses would
    answer a wrong question rather than raise, which is the worse of the two.

    **The result is the question's own size**, not an inefficiency: the number
    of maximal routes doubles per stacked branch point while the descent that
    feeds it stays one request per node. That is the one cost the ascent cannot
    bound, so ``max_paths`` bounds it here -- and **refuses rather than
    truncating**, which is the whole of why the ceiling is a parameter and not
    a slice the caller takes afterwards. A ceiling that returned the first
    ``k`` routes would collapse *there were exactly k ways up* into *there were
    at least k*, and a pair of answers one shape cannot tell apart is what
    :meth:`Hierarchy.contains` and :func:`_refuse_an_unknown_anchor` exist to
    keep separate everywhere else here. Refusing keeps the published property
    intact: what comes back is every maximal route or nothing at all.

    **The ceiling is checked as routes are emitted**, not after, because a
    ceiling read at the end has already spent the memory it was set to save --
    and the memory is the harm. What it bounds is the term that grows
    exponentially: the emitted routes, and with them the work, at ``max_paths``
    plus one emissions rather than all of them. The stack is *not* bounded by
    it and is not bounded by the edge map either -- it holds a whole route per
    unexplored sibling on the current descent, so it is quadratic in the depth
    where the map is linear in the nodes. Polynomial against exponential is the
    trade, and it is the one worth making rather than a claim that the stack
    is free.

    **An unusable ceiling is refused earlier than this**, at the walk's door in
    :func:`_paths_to_root`, before the ascent it would throw away. The check
    here is the projection's own, for a caller reaching the core directly; it
    is the same helper and so cannot disagree with it.

    ``max_paths=None`` is unbounded and is the default: the walk's contract is
    *every way up*, and a ceiling that turned a correct answer into an
    exception for a caller who never asked for one would be a clamp on the
    question rather than a budget on the answer. A caller who cannot afford
    even the bounded enumeration is asking about membership, which
    :func:`_ancestors` answers over the same descent for the size of the axis.
    """
    if max_paths is not None:
        _refuse_an_unusable_bound(max_paths, "max_paths")
    routes: list[tuple[_K, ...]] = []
    stack: list[tuple[_K, ...]] = [(anchor,)]
    while stack:
        route = stack.pop()
        onward = tuple(above for above in edges[route[-1]] if above not in route)
        if not onward:
            routes.append(route)
            if max_paths is not None and len(routes) > max_paths:
                _refuse_an_unaffordable_answer(anchor, max_paths)
            continue
        stack.extend((*route, above) for above in reversed(onward))
    return tuple(routes)


def _upward_order(edges: dict[_K, tuple[_K, ...]], node_id: _K) -> tuple[_K, ...]:
    """``node_id`` and everything above it, nearest first, deduplicated.

    :func:`_ancestors` read off a descent that may have been seeded by somebody
    else. ``_levels_of`` cannot answer this when the descent carried two seeds,
    because its levels are distances from the *frontier* rather than from
    either seed -- so the one walk that expands two nodes at once recovers each
    one's chain here, from the replies, at no request.

    ``edges`` is indexed rather than defaulted, for the reason :func:`_expand`
    gives: a node it has no key for was never asked about, and reading that as
    *nothing above it* is a bound mistaken for the shape of the axis.
    """
    order: list[_K] = [node_id]
    seen = {node_id}
    frontier: tuple[_K, ...] = (node_id,)
    while frontier:
        fresh: list[_K] = []
        for current in frontier:
            for above in edges[current]:
                if above not in seen:
                    seen.add(above)
                    fresh.append(above)
        order.extend(fresh)
        frontier = tuple(fresh)
    return tuple(order)


def _above_within(edges: dict[_K, tuple[_K, ...]], node_id: _K) -> frozenset[_K]:
    """Everything **strictly** above ``node_id`` in the replies already collected.

    Strictly, and over cyclic data that includes ``node_id`` itself -- a node on
    a cycle really is above itself, and a reader comparing two nodes needs to be
    told so rather than protected from it. :func:`_deepest_common_of` is the
    reader, and it is what turns the two directions of that answer into a
    tie rather than an exclusion.

    Indexed rather than defaulted, like every other reader of a reply map.
    """
    above: set[_K] = set()
    frontier = edges[node_id]
    while frontier:
        fresh = tuple(dict.fromkeys(above_id for above_id in frontier if above_id not in above))
        above.update(fresh)
        frontier = tuple(onward for above_id in fresh for onward in edges[above_id])
    return frozenset(above)


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


def _deduped_seeds(seeds: Sequence[_K]) -> tuple[_K, ...]:
    """A seed frontier with repeats removed, in first-named order.

    **The one frontier no later step can dedup on its own behalf**, which is
    where the descent's promise about repeated ids begins rather than a
    tidiness. :func:`_expand` records discovery against the *first* node to
    reach something, so a seed named twice reaches nothing the second time: an
    unguarded repeat replaces a real record with an empty one and the whole
    subtree stops being reachable. Every later frontier is already deduplicated
    by the visited set, so a seed reply is the only one that can arrive
    repeated -- and it arrives from ``roots()`` or from a caller's two
    arguments, both of which are arbitrary consumer code.

    Shared by the three walks that build one rather than written into each, for
    the reason every refusal here is shared: a rule a caller re-implements is a
    rule that drifts, and a third site is where that stops being hypothetical.
    """
    return tuple(dict.fromkeys(seeds))


def _seeds_under(anchor: _K | None) -> Walk[_K, tuple[_K, ...]]:
    """``anchor`` alone, or the axis's roots when there is none.

    The two walks whose anchor is optional -- ``flatten`` and ``leaves`` --
    descend from a forest when it is omitted, and this is the one request that
    difference costs. It is deliberately **not** memoised: ``roots`` is asked
    once per walk, and remembering it across walks would cost a caller their
    only chance to notice the axis grew a root.

    **The reply is deduplicated**, by :func:`_deduped_seeds`, which carries the
    reason: ``roots()`` is arbitrary consumer code and a seed named twice
    reaches nothing the second time.
    """
    if anchor is not None:
        return (anchor,)
    (roots,) = yield ("roots", ())
    return _deduped_seeds(roots)


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
    now keeps. It keeps the whole reply rather than that one bit, so this reads
    ``not edges[node_id]`` where it once read a set the descent maintained
    alongside -- one structure fewer, and the same answer.

    **So there is no second request**, and the walk costs exactly the descent.
    There was one: the candidates -- the nodes that discovered nothing -- were
    asked again, for edges the descent already had and discarded. A memo made
    that ask free when a caller happened to supply one big enough to hold the
    frontier, which is two conditions a walk should not have to meet to avoid
    fetching the same edge twice.
    """
    seeds = yield from _seeds_under(under)
    discovered, edges = yield from _expand(seeds, "children")
    order = _preorder_of(discovered, seeds, exclude_seeds=False)
    return tuple(node_id for node_id in order if not edges[node_id])


def _paths_to_root(
    node_id: _K, max_paths: int | None = None
) -> Walk[_K, tuple[tuple[_K, ...], ...]]:
    """Every maximal path upward from ``node_id``, the anchor at the near end.

    **A reading of the shared descent, and the reply map is what makes it one.**
    The question it answers is not the one the other readings answer --
    :func:`_ancestors` returns *what is above me*, deduplicated, and this
    returns *how did I get here*, where a node reachable two ways is two
    answers. What differs between them is the **projection**: the ascent that
    feeds both asks each node exactly once, one frontier per level, and
    :func:`_paths_of` scopes its cycle guard to the route being extended where
    :func:`_levels_of` leaves the descent's walk-scoped visited set in place.

    It was written as a second algorithm first, and that is worth recording
    because the mistake is a natural one: the guard really does have to be
    per-route, and a walk that descends route by route really does need one. It
    also needs a request per route rather than per node -- 125 for a sixteen-node
    axis carrying thirty-two routes -- which it then has to buy back with a memo
    of its own, and it asks about one node at a time, which costs a bulk backing
    its per-level query and the asynchronous driver its round of concurrency.
    Reading the replies instead, every one of those costs is somebody else's
    already-solved problem: **16 requests for sixteen nodes, one frontier per
    level, and no memo anywhere.**

    **The ceiling is checked here, before the ascent.** ``max_paths`` below one
    can admit no answer from any axis, so the descent it would refuse is work
    already known to be wasted -- and over a remote backing that is one query
    per node above the anchor, spent to report a mistake in the literal the
    caller typed. It is the posture :func:`_refuse_an_unusable_bound` states
    for the frontier bound, kept by the second bound to take it.
    """
    if max_paths is not None:
        _refuse_an_unusable_bound(max_paths, "max_paths")
    _, edges = yield from _expand((node_id,), "parents")
    return _paths_of(edges, node_id, max_paths)


def _deepest_common_of(edges: dict[_K, tuple[_K, ...]], a: _K, b: _K) -> _K | None:
    """The deepest common ancestor, read off replies already collected.

    **Deepest is a claim about the partial order, not about distance**, and
    those are the same thing only over a tree. This walk was written as *the
    first of ``a``'s chain, nearest first, that also stands above ``b``* -- and
    a chain read nearest-first is level order, which is distance from ``a``.
    One shortcut edge separates them: an axis where ``beagle`` names both
    ``mammal`` and ``hound``, with ``dog`` two hops away under ``hound`` and
    ``mammal`` above ``dog``, answered ``mammal`` for ``beagle`` and ``puppy``
    while ``dog`` was a common ancestor standing strictly below it. Both
    candidates were perfectly *comparable*, which is the case the asymmetry
    below explicitly told a reader was safe.

    So the candidates are the **minimal** common ancestors: those with no other
    common ancestor strictly below them. ``c`` is dropped when some other
    common ancestor ``d`` stands below it **and** ``c`` does not also stand
    below ``d``. That second clause is what keeps cyclic data answerable: two
    nodes on one cycle are each above the other, and a rule reading only the
    first clause would drop both and return ``None`` where a common ancestor
    plainly exists. With it, mutual ancestry is a *tie* and falls through to
    the tie-break, which is the only sense *deepest* has inside a cycle.

    **The tie-break is asymmetric and is part of the definition.** Among the
    minimal candidates -- which are pairwise incomparable, so there is nothing
    to choose between them on depth -- the answer is the first in ``a``'s own
    chain, nearest first. Over a tree there is one candidate and the asymmetry
    is invisible; over a DAG swapping the arguments can swap the answer, which
    is why this is stated rather than left to be discovered.

    That relation -- *strictly above, and not also below* -- is a strict partial
    order: irreflexive because no node is both in and out of its own ancestry,
    and transitive because reachability composes. A finite non-empty set under
    one has minimal elements, so the candidate set is never empty when
    ``common`` is not, and the ``None`` below is reachable only through the
    empty intersection above it.

    **No request is issued here**, and the cost is quadratic in the shared
    ancestry rather than in the axis: one upward traversal of the collected
    replies per common ancestor, then one comparison per ordered pair of them.
    **The memory is quadratic in the same term** and is not bounded -- the
    ancestry of every common ancestor is materialised at once, because
    establishing that one candidate is minimal already consults every other, so
    there is nothing for laziness to skip. Two deep nodes under a broad shared
    ancestry is the shape that pays it: the ascent stays one cheap request per
    node while the comparison grows with the square of what they share.
    """
    a_order = _upward_order(edges, a)
    # Bound to a local rather than inlined: ``frozenset.__and__`` takes an
    # ``AbstractSet[object]``, which solves the nested call's key type as
    # ``object`` and then fails against an invariant ``dict`` key.
    b_chain = frozenset(_upward_order(edges, b))
    common = frozenset(a_order) & b_chain
    if not common:
        return None
    above = {node_id: _above_within(edges, node_id) for node_id in common}
    return next(
        (
            node_id
            for node_id in a_order
            if node_id in common
            and not any(
                node_id in above[other] and other not in above[node_id]
                for other in common
                if other != node_id
            )
        ),
        None,
    )


def _deepest_common_ancestor(a: _K, b: _K) -> Walk[_K, _K | None]:
    """The deepest node above both ``a`` and ``b``, or ``None``.

    **One ascent seeded by both**, which is the whole of what makes this the
    eighth reading of the shared descent rather than a walk with a cost to
    publish. Written as two composed ``_ancestors`` walks it asked the ancestry
    the two nodes share **twice** -- 44 requests over a 23-node axis, which
    has 23 nodes to ask about -- because ``yield from`` delegates a request past
    the composing frame and there was nowhere to keep what came back. Seeding
    one descent with both nodes asks each node once, and each argument's own
    chain is recovered from the replies by :func:`_upward_order` at no request
    at all.

    The seeds go through :func:`_deduped_seeds`, which carries the reason.
    ``a == b`` is the repeat here, and it is a question a caller may
    legitimately ask rather than a malformed one.

    :func:`_deepest_common_of` is where the answer is chosen, and it says why
    *deepest* is not *nearest*.
    """
    _, edges = yield from _expand(_deduped_seeds((a, b)), "parents")
    return _deepest_common_of(edges, a, b)


def _parent_edges() -> Walk[_K, dict[_K, tuple[_K, ...]]]:
    """Every edge the axis has, as a parent mapping, by descending from roots.

    The walk behind a snapshot. It is written here, as a walk over the shared
    core, rather than twinned into the two constructors that take one -- which
    is the arrangement ``hierarchy``'s module docstring calls the bet: a walk
    added *over* the drivers costs one generator, where a capability added *to*
    them costs a pair.

    **A reading of :func:`_expand`, like every other walk here**, and it was
    the last one that was not. It ran its own frontier loop and its own visited
    set because it could not be a reading: a spanning record drops the edge
    into a node another frontier reached first, and that edge is the one thing
    a parent-edge snapshot cannot lose -- a DAG copied into a tree is a
    different axis wearing the name. Keeping the whole reply is what closed
    that gap, so the walk that most needed the induced edge set is the walk
    that stopped being a second expansion because of it.

    Inverting the reply map is the whole of what remains, and it is a rewrite
    of edges already in hand rather than a traversal: ``edges`` names every
    node that entered a frontier and pairs it with the children it answered,
    which is the same relation this returns read from the other end.

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
    different shape from the axis it copied. Both halves are the descent's
    now: the visited set expands a node once, and the reply map keeps every
    edge it was told about.
    """
    (roots,) = yield ("roots", ())
    _, edges = yield from _expand(_deduped_seeds(roots), "children")
    parents: dict[_K, list[_K]] = {node_id: [] for node_id in edges}
    for node_id, children in edges.items():
        for child in children:
            parents[child].append(node_id)
    return {node_id: tuple(above) for node_id, above in parents.items()}
