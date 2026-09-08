"""A taxonomy: one relation of a vocabulary, reified as a walkable axis.

A :class:`Taxonomy` is a *definition* plus the three backings that answer it --
structure, content, and the assertions the edges were made of. It holds no
copy: the sources are the authority, so a rebuild beneath one is visible on the
next read.

**Two flavours, not a choice.** The synchronous twin is what a vocabulary a
person typed uses -- a file already read has nothing to await -- and the
asynchronous one is what a by-reference backing needs. Everything that is not
an ``await`` is shared: the walks over the structure axis live once in
:mod:`dataknobs_common.hierarchy`, and what is twinned here is the driving.

This module lives beside the hierarchy protocols rather than inside
``ontology/`` because a taxonomy is reachable without one: the definition is a
value, and the three backings are protocols. What it must not do is import the
ontology package at runtime, which would close a cycle through that package's
``__init__`` -- so every ontology name below is an annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

# The walk core's frontier read: the one implementation of "ask the frontier in
# bulk where the backing offers it, else one node at a time". The streaming
# walks below cannot go through the collecting core, but they must not decide
# this again -- a second copy is how the two drift over which backings get a
# per-level query. Both this module and the drivers import it from the private
# core, which is why it lives there rather than inside ``hierarchy``.
from dataknobs_common._walk_core import _async_reply, _sync_reply

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from dataknobs_common.hierarchy import AsyncHierarchy, Hierarchy
    from dataknobs_common.ontology.model import TaxonomyDefinition
    from dataknobs_common.ontology.sources import (
        AssertionSource,
        AsyncAssertionSource,
        AsyncEntitySource,
        EntitySource,
    )

__all__ = ["AsyncTaxonomy", "Taxonomy"]


@dataclass
class Taxonomy:
    """One relation of a vocabulary, walkable, with synchronous backings.

    ``assertions`` is what lets a cursor over this axis report the **edge** it
    walked rather than only its endpoints. It is optional because not every
    hierarchy has assertions behind it -- one built from a ``parent_id`` column
    has rows and no ``Assertion`` -- and ``assertions is None`` is the question
    that tells an unannotated edge from an axis that has no annotations to give.
    """

    definition: TaxonomyDefinition
    structure: Hierarchy
    entities: EntitySource
    assertions: AssertionSource | None = None

    def walk(self, *, from_id: str | None = None, max_depth: int | None = None) -> Iterator[str]:
        """Every node at or under ``from_id``, breadth first, each one once.

        From the axis's roots when ``from_id`` is omitted. The anchor is
        **included** -- this is the whole axis from a point rather than the
        strict descendants of it. ``max_depth`` bounds how many levels are
        expanded, so ``0`` yields the seeds alone.

        A streaming member, and the reason it is not written through the shared
        walk core: the core collects and returns, and a streaming core is a
        wider request type for a member with its own consumers. That extension
        is measured and deliberately not taken here. What it does share is the
        step that reads a frontier, so this walk and the drivers cannot drift
        over which backings answer a level in one query.
        """
        seen: set[str] = set()
        frontier = (from_id,) if from_id is not None else tuple(self.structure.roots())
        depth = 0
        while frontier and (max_depth is None or depth <= max_depth):
            fresh: list[str] = []
            for node_id in frontier:
                if node_id in seen:
                    continue
                seen.add(node_id)
                fresh.append(node_id)
                yield node_id
            if max_depth is not None and depth == max_depth:
                return
            replies = _sync_reply(self.structure, "children", tuple(fresh))
            frontier = tuple(child for reply in replies for child in reply)
            depth += 1


@dataclass
class AsyncTaxonomy:
    """The asynchronous twin. Same four fields, asynchronous backings."""

    definition: TaxonomyDefinition
    structure: AsyncHierarchy
    entities: AsyncEntitySource
    assertions: AsyncAssertionSource | None = None

    async def walk(
        self, *, from_id: str | None = None, max_depth: int | None = None
    ) -> AsyncIterator[str]:
        """:meth:`Taxonomy.walk`, awaited.

        The same traversal, and it is duplicated for one reason: a streaming
        member cannot go through the shared collecting core without widening
        that core's request type for every walk that does not stream. The
        frontier read is shared, so this gets the driver's per-level behaviour
        -- one bulk query where the backing offers one, concurrency where it
        does not -- rather than a sequential await per node.
        """
        seen: set[str] = set()
        if from_id is not None:
            frontier: tuple[str, ...] = (from_id,)
        else:
            frontier = tuple(await self.structure.roots())
        depth = 0
        while frontier and (max_depth is None or depth <= max_depth):
            fresh: list[str] = []
            for node_id in frontier:
                if node_id in seen:
                    continue
                seen.add(node_id)
                fresh.append(node_id)
                yield node_id
            if max_depth is not None and depth == max_depth:
                return
            replies = await _async_reply(self.structure, "children", tuple(fresh))
            frontier = tuple(child for reply in replies for child in reply)
            depth += 1
