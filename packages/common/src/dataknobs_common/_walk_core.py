"""The walk core: the algorithms, and the step that reads a frontier.

Private, and a module rather than a convention. A core that is only a
paragraph is satisfied by two twins that each implement the paragraph -- so
what is shared here is *code both flavours call*, not a rule both flavours
follow.

Three shared cores live here, and they are shared with different callers:

* the **algorithms** -- ``_levels`` and the walks composed from it -- which
  :mod:`dataknobs_common.hierarchy`'s public wrappers drive;
* the **frontier read** -- ``_sync_reply`` / ``_async_reply`` -- which the
  drivers call, and which :class:`~dataknobs_common.taxonomy.Taxonomy`'s
  streaming walk calls too. That walk cannot go through the collecting core,
  because streaming would widen the core's request type for every walk that
  does not stream; sharing this one step is what stops it drifting from the
  drivers over which backings answer a level in one query;
* the **freshness check** -- ``_refuse_a_spent_walk`` -- which both drivers
  make before their opening ``next``. Shared for the reason every refusal here
  is: a rule a twin re-implements is a rule that drifts, and the twins' whole
  claim is that they behave alike.

Both consumers import from here, which is why this is a sibling of
``hierarchy.py`` rather than something inside it: a private name reached out
of a *public* module is a boundary crossed, and the same name reached out of
a private core is two consumers of one core.

The dependency runs one way at runtime -- ``hierarchy`` imports this, never
the reverse. The protocols are annotations here and nothing more.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Literal

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

    Shared by both drivers rather than written into each -- the refusals either
    twin makes are the same refusals, and a rule a twin re-implements is a rule
    that drifts. Deliberately not folded into the ``StopIteration`` conversion
    below: that converts an exception raised by the *hierarchy*, where this is
    the walk's own state, which is not an error condition anywhere else.
    """
    state = inspect.getgeneratorstate(walk)
    if state != inspect.GEN_CREATED:
        raise RuntimeError(
            f"this walk has already been driven (generator state {state}); "
            f"a walk is single-use, so build a fresh one per drive"
        )


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


async def _async_reply(
    hierarchy: AsyncHierarchy[_K],
    member: Member,
    node_ids: tuple[_K, ...],
    *,
    max_concurrency: int,
) -> tuple[Sequence[_K], ...]:
    """:func:`_sync_reply`'s twin: bulk where offered, else a *bounded* round
    of concurrency per frontier.

    ``gather`` runs one round trip per node concurrently; ``children_many``
    is one *query* for the level. The second is what a row-backed hierarchy
    wants, and concurrency does not substitute for it.

    ``max_concurrency`` is an argument with no default rather than a constant
    read here, which is the whole of why this module still holds no
    configuration: the number is policy, it lives with the public surface, and
    a caller who knows their backing can move it. Without a bound the fan-out
    was the width of the *level* -- a property of the data -- so a node with
    ten thousand children issued ten thousand concurrent calls into whatever
    the backing was. The bulk path never had the problem, which means the
    unbounded path was exactly the one taken by backings least able to absorb
    it.

    No explicit ``StopIteration`` conversion here: this is a coroutine, so the
    language performs it at this frame's boundary and a collaborator's
    ``StopIteration`` surfaces as the same ``RuntimeError`` the synchronous
    side raises by hand.
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be at least 1, got {max_concurrency}")
    if member == "roots":
        return (await hierarchy.roots(),)
    if member in ("parents", "children"):
        bulk = getattr(hierarchy, f"{member}_many", None)
        if bulk is not None:
            return tuple(await bulk(node_ids))
        one = hierarchy.parents if member == "parents" else hierarchy.children
        limit = asyncio.Semaphore(max_concurrency)

        async def _bounded(node_id: _K) -> Sequence[_K]:
            async with limit:
                return await one(node_id)

        return tuple(await asyncio.gather(*(_bounded(n) for n in node_ids)))
    raise ValueError(f"unknown hierarchy member {member!r}")


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
