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
from typing import TYPE_CHECKING, NoReturn

# The walk core's frontier read: the one implementation of "ask the frontier in
# bulk where the backing offers it, else one node at a time". The streaming
# walks below cannot go through the collecting core, but they must not decide
# this again -- a second copy is how the two drift over which backings get a
# per-level query. Both this module and the drivers import it from the private
# core, which is why it lives there rather than inside ``hierarchy``.
from dataknobs_common._walk_core import _async_reply, _sync_reply
from dataknobs_common.exceptions import NotFoundError
from dataknobs_common.hierarchy import DEFAULT_FRONTIER_CONCURRENCY

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


def _refuse_an_unknown_anchor(taxonomy_id: str, anchor: str) -> NoReturn:
    """Refuse a ``from_id`` the structure axis does not contain.

    The anchor is *included* in a walk's output by design, so seeding a
    frontier with it unchecked emits an id the axis does not contain as though
    it were a term of the axis -- and the caller cannot tell, because a walk is
    exactly what they asked for.

    Yielding nothing was the other candidate and is the worse one: it collapses
    *nothing below this node* into *this node is not here*, which are the two
    answers :meth:`~dataknobs_common.hierarchy.Hierarchy.contains` says in its
    own docstring it exists to keep apart. An axis that carries that member and
    then destroys the distinction in the one walk that needs it is refuting
    itself -- and ``contains`` had no caller anywhere until this one.

    Shared by both flavours rather than written into each: the ``await`` is on
    the containment question, not on the refusal, so the message and its
    context are single-sourced while each twin keeps its own call.
    """
    raise NotFoundError(
        f"no node {anchor!r} in the structure axis of taxonomy {taxonomy_id!r}, "
        f"so it cannot anchor a walk",
        context={"taxonomy": taxonomy_id, "anchor": anchor},
    )


@dataclass
class Taxonomy:
    """One relation of a vocabulary, walkable, with synchronous backings.

    ``assertions`` is what lets a cursor over this axis report the **edge** it
    walked rather than only its endpoints. It is optional because not every
    hierarchy has assertions behind it -- one built from a ``parent_id`` column
    has rows and no ``Assertion`` -- and ``assertions is None`` is the question
    that tells an unannotated edge from an axis that has no annotations to give.

    **The key is pinned to ``str``, and the pin is written out.**
    :class:`~dataknobs_common.hierarchy.Hierarchy` is generic in its key because
    the walks only ever *hash* a node id, so an object tree with no ids can bind
    the parameter to its own node type and share the traversals. A taxonomy
    cannot: it is both axes at once, and the content axis is an
    ``EntitySource`` addressed by ``str`` because an ``Entity`` has a ``str``
    id. Making the structure axis generic here would let a caller hold an
    integer-keyed hierarchy beside an entity lookup that cannot be asked about
    an integer.

    So ``K`` serves the walks and the module-level drivers, and this pin is a
    boundary rather than an oversight -- spelled ``Hierarchy[str]`` rather than
    bare so that it reads as one. Moving it is a design question with a
    prerequisite: the content side has to become generic first, or explicitly
    stay behind.
    """

    definition: TaxonomyDefinition
    structure: Hierarchy[str]
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
        if from_id is not None:
            if not self.structure.contains(from_id):
                _refuse_an_unknown_anchor(self.definition.id, from_id)
            frontier: tuple[str, ...] = (from_id,)
        else:
            frontier = tuple(self.structure.roots())
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
    """The asynchronous twin. Same four fields, asynchronous backings.

    The key is pinned to ``str`` here too, for the reason
    :class:`Taxonomy` states: a taxonomy carries the content axis as well as
    the structure one, and the content axis is ``str``-addressed.
    """

    definition: TaxonomyDefinition
    structure: AsyncHierarchy[str]
    entities: AsyncEntitySource
    assertions: AsyncAssertionSource | None = None

    async def walk(
        self,
        *,
        from_id: str | None = None,
        max_depth: int | None = None,
        max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
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
            if not await self.structure.contains(from_id):
                _refuse_an_unknown_anchor(self.definition.id, from_id)
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
            replies = await _async_reply(
                self.structure, "children", tuple(fresh), max_concurrency=max_concurrency
            )
            frontier = tuple(child for reply in replies for child in reply)
            depth += 1
