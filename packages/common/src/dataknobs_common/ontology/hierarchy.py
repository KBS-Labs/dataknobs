"""A hierarchy whose edges are assertions of one relation.

The component behind the sentence *a taxonomy is a view of an ontology's
relation*. It holds an :class:`~dataknobs_common.ontology.sources.AssertionSource`
and a relation, opens nothing, and caches nothing: the source is the authority,
so a rebuild beneath it is visible immediately -- which is the property an
anchored view relies on when it holds the structure rather than a snapshot of
it.

**Why this lives under ``ontology/`` and the protocol it satisfies does not.**
``dataknobs_common.hierarchy`` is general: a hierarchy over a ``parent_id``
column has no ontology behind it, and nesting the protocol here would make the
general construct a member of the specific one. This concrete is the mirror
case -- it is *made of* ontology types, and reading a parent means asking
whether an assertion's object is an entity or a literal, which needs
:class:`~dataknobs_common.ontology.model.EntityRef` at runtime rather than as
an annotation. Importing that from ``dataknobs_common.hierarchy`` would run
this package's ``__init__`` from inside a module that ``__init__`` transitively
imports, which is a cycle that fails at import time. So the general module
imports nothing from here and this one imports it freely, which is the same
rule that sends a database-backed hierarchy to ``dataknobs-data``: a concrete
goes where its dependency is, not beside its protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dataknobs_common.ontology.model import EntityRef
from dataknobs_common.ontology.sources import object_entity_id

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dataknobs_common.ontology.model import Assertion, RelationRef
    from dataknobs_common.ontology.sources import AssertionSource, AsyncAssertionSource

__all__ = ["AssertionHierarchy", "AsyncAssertionHierarchy"]


def _ordered(node_ids: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate in first-appearance order.

    A DAG node is reachable by several paths, so a repeated id changes no
    membership answer and does change a count someone is reporting.
    """
    return tuple(dict.fromkeys(node_ids))


def _parents_of(assertions: Sequence[Assertion]) -> tuple[str, ...]:
    """The entity objects of these assertions, deduplicated in order.

    An assertion whose object is a literal contributes no parent: an axis is
    made of edges between entities, and ``dog lifespan_years 12`` is a value
    rather than a place in a structure.
    """
    return _ordered(
        entity_id
        for assertion in assertions
        if (entity_id := object_entity_id(assertion.object)) is not None
    )


def _children_of(assertions: Sequence[Assertion]) -> tuple[str, ...]:
    """The subjects of these assertions, deduplicated in order.

    The mirror of :func:`_parents_of`: a child is an edge's subject, and no
    literal check is needed because a subject is always an entity id.
    """
    return _ordered(assertion.subject for assertion in assertions)


def _nodes_of(edges: Sequence[Assertion]) -> tuple[str, ...]:
    """Every node these edges mention, deduplicated in first-appearance order.

    Declaration order, because it is the only order a hand-edited file gives
    and the alternative -- sorting -- would make ``roots()`` report an order
    the author did not write.
    """
    nodes: list[str] = []
    for edge in edges:
        nodes.append(edge.subject)
        entity_id = object_entity_id(edge.object)
        if entity_id is not None:
            nodes.append(entity_id)
    return _ordered(nodes)


@dataclass(frozen=True)
class AssertionHierarchy:
    """A :class:`~dataknobs_common.hierarchy.Hierarchy` over one relation.

    ``parents(x)`` is ``find(subject=x, relation=...)`` read for its objects;
    ``children(x)`` is ``find(object=x, relation=...)`` read for its subjects
    -- the mirrored query, not the same one.
    """

    source: AssertionSource
    relation: RelationRef

    def roots(self) -> Sequence[str]:
        """The nodes this axis has with no parent.

        Not an extent, and the distinction matters: this is *the nodes this
        relation leaves unplaced*, which equals a type's membership only by
        coincidence. ``EntitySource.by_type`` is the extent.
        """
        edges = self.source.find(relation=self.relation)
        placed = {edge.subject for edge in edges if object_entity_id(edge.object) is not None}
        return tuple(node for node in _nodes_of(edges) if node not in placed)

    def parents(self, node_id: str) -> Sequence[str]:
        """The nodes ``node_id`` is directly under."""
        return _parents_of(self.source.find(subject=node_id, relation=self.relation))

    def children(self, node_id: str) -> Sequence[str]:
        """The nodes directly under ``node_id``."""
        return _children_of(self.source.find(object=EntityRef(node_id), relation=self.relation))

    def parents_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        """:meth:`parents` for a whole frontier, in one query.

        ``AssertionSource.find_many`` is the bulk form the singular member
        cannot reach, so a level costs one call rather than one per node. The
        reply is positional -- a node the query returned nothing for gets an
        empty tuple, not a missing slot.
        """
        found = self.source.find_many(subjects=tuple(node_ids), relation=self.relation)
        return tuple(_parents_of(found.get(node_id, [])) for node_id in node_ids)

    def children_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        """:meth:`children` for a whole frontier, in one query."""
        found = self.source.find_many(objects=tuple(node_ids), relation=self.relation)
        return tuple(_children_of(found.get(node_id, [])) for node_id in node_ids)

    def contains(self, node_id: str) -> bool:
        """Whether any assertion of this relation names the node.

        What keeps *nothing below this node* distinguishable from *this node is
        not here*. An id no edge of this relation mentions is absent, and an
        absent node is neither a root nor a leaf.
        """
        if self.source.find(subject=node_id, relation=self.relation):
            return True
        return bool(self.source.find(object=EntityRef(node_id), relation=self.relation))


@dataclass(frozen=True)
class AsyncAssertionHierarchy:
    """The asynchronous twin, over an ``AsyncAssertionSource``.

    Same two fields and the same four members, every one ``async def``,
    returning a ``Sequence`` rather than an async iterable -- which is what
    lets both flavours share one implementation of every walk.
    """

    source: AsyncAssertionSource
    relation: RelationRef

    async def roots(self) -> Sequence[str]:
        """The nodes this axis has with no parent."""
        edges = await self.source.find(relation=self.relation)
        placed = {edge.subject for edge in edges if object_entity_id(edge.object) is not None}
        return tuple(node for node in _nodes_of(edges) if node not in placed)

    async def parents(self, node_id: str) -> Sequence[str]:
        """The nodes ``node_id`` is directly under."""
        return _parents_of(await self.source.find(subject=node_id, relation=self.relation))

    async def children(self, node_id: str) -> Sequence[str]:
        """The nodes directly under ``node_id``."""
        edges = await self.source.find(object=EntityRef(node_id), relation=self.relation)
        return _children_of(edges)

    async def parents_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        """:meth:`AssertionHierarchy.parents_many`, awaited."""
        found = await self.source.find_many(subjects=tuple(node_ids), relation=self.relation)
        return tuple(_parents_of(found.get(node_id, [])) for node_id in node_ids)

    async def children_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        """:meth:`AssertionHierarchy.children_many`, awaited."""
        found = await self.source.find_many(objects=tuple(node_ids), relation=self.relation)
        return tuple(_children_of(found.get(node_id, [])) for node_id in node_ids)

    async def contains(self, node_id: str) -> bool:
        """Whether any assertion of this relation names the node."""
        if await self.source.find(subject=node_id, relation=self.relation):
            return True
        found = await self.source.find(object=EntityRef(node_id), relation=self.relation)
        return bool(found)


if TYPE_CHECKING:  # pragma: no cover - checked by the type checker, not run

    def _twins_satisfy_their_protocols() -> None:
        """Each twin is assignable to the hierarchy flavour it claims.

        Inline rather than in a test, because this file is type-checked and
        the test tree is not -- the arrangement ``sources.py`` already uses for
        the source protocols.
        """
        from dataknobs_common.hierarchy import (
            AsyncBulkHierarchy,
            AsyncHierarchy,
            BulkHierarchy,
            Hierarchy,
        )
        from dataknobs_common.ontology.sources import (
            AsyncMappingAssertionSource,
            MappingAssertionSource,
        )

        sync: Hierarchy = AssertionHierarchy(MappingAssertionSource([]), "isa")
        asynchronous: AsyncHierarchy = AsyncAssertionHierarchy(
            AsyncMappingAssertionSource([]), "isa"
        )
        bulk: BulkHierarchy = AssertionHierarchy(MappingAssertionSource([]), "isa")
        async_bulk: AsyncBulkHierarchy = AsyncAssertionHierarchy(
            AsyncMappingAssertionSource([]), "isa"
        )
        del sync, asynchronous, bulk, async_bulk
