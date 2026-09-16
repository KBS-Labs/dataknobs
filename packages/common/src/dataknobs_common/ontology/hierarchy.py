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
from typing import TYPE_CHECKING, Generic, TypedDict

from dataknobs_common.hierarchy import K
from dataknobs_common.ontology.model import EntityRef, Polarity, relation_id
from dataknobs_common.ontology.sources import object_entity_id

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from dataknobs_common.ontology.model import Assertion, RelationRef, Term
    from dataknobs_common.ontology.sources import AssertionSource, AsyncAssertionSource

__all__ = ["AssertionHierarchy", "AsyncAssertionHierarchy", "EdgeCriteria", "edge_criteria"]


class EdgeCriteria(TypedDict):
    """The two criteria that select the edges of one axis -- :func:`edge_criteria`."""

    relation: RelationRef
    polarity: Polarity


def edge_criteria(relation: RelationRef) -> EdgeCriteria:
    """The criteria that select the edges of an axis: this relation, **asserted**.

    Unpack it into a read rather than writing the two keywords::

        source.find(subject="beagle", **edge_criteria("isa"))
        source.find_many(subjects=frontier, **edge_criteria("isa"))

    *An axis is made of asserted edges between entities* is one decision, and
    this is its one home. The two concretes below read through it, the
    taxonomy cursor's edge members read through it from another module, and a
    consumer building an axis over ``onto.assertions.find(relation="isa")``
    reaches the same narrowing by unpacking the same criteria -- where writing
    ``polarity=`` at each site is how a reader added later omits it silently,
    and a query that does not narrow returns a stated negation as an edge.

    A function returning criteria rather than one performing the read, so it
    has no flavour: the synchronous and asynchronous twins share it the way
    every walk shares the core, without sharing a line of runtime code.
    """
    return {"relation": relation, "polarity": Polarity.ASSERTED}


def _ordered(node_ids: Iterable[K]) -> tuple[K, ...]:
    """Deduplicate in first-appearance order.

    A DAG node is reachable by several paths, so a repeated id changes no
    membership answer and does change a count someone is reporting.
    """
    return tuple(dict.fromkeys(node_ids))


def _parents_of(assertions: Sequence[Assertion[K]]) -> tuple[K, ...]:
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


def _children_of(assertions: Sequence[Assertion[K]]) -> tuple[K, ...]:
    """The subjects of these assertions, deduplicated in order.

    The mirror of :func:`_parents_of`: a child is an edge's subject, and no
    literal check is needed because a subject is always an entity id.
    """
    return _ordered(assertion.subject for assertion in assertions)


def _parent_edges_of(edges: Sequence[Assertion[K]]) -> dict[K, tuple[K, ...]]:
    """Every node these edges mention, and what each is directly under.

    The extent :meth:`AssertionHierarchy.roots` is not, which is what lets a
    snapshot of this axis be the *whole* axis rather than the part a descent
    from the roots reaches -- a cyclic component with nothing above it has no
    root to be found from, and ``isa`` is asserted rather than constrained, so a
    document may name one.

    Every node gets an entry, including one that only ever appears as a parent;
    its entry is empty, which is the same thing :meth:`roots` reports about it.
    An assertion whose object is a literal contributes a node and no edge, for
    the reason :func:`_parents_of` gives: ``dog lifespan_years 12`` is a value
    rather than a place in a structure.
    """
    above: dict[K, list[K]] = {}
    for edge in edges:
        above.setdefault(edge.subject, [])
        parent = object_entity_id(edge.object)
        if parent is None:
            continue
        above.setdefault(parent, [])
        if parent not in above[edge.subject]:
            above[edge.subject].append(parent)
    return {node_id: tuple(parents) for node_id, parents in above.items()}


def _nodes_of(edges: Sequence[Assertion[K]]) -> tuple[K, ...]:
    """Every node these edges mention, deduplicated in first-appearance order.

    Declaration order, because it is the only order a hand-edited file gives
    and the alternative -- sorting -- would make ``roots()`` report an order
    the author did not write.
    """
    nodes: list[K] = []
    for edge in edges:
        nodes.append(edge.subject)
        entity_id = object_entity_id(edge.object)
        if entity_id is not None:
            nodes.append(entity_id)
    return _ordered(nodes)


@dataclass(frozen=True)
class AssertionHierarchy(Generic[K]):
    """A :class:`~dataknobs_common.hierarchy.Hierarchy` over one relation.

    **Generic in the node key**, and it has to be rather than merely may be:
    this is the axis an :class:`~dataknobs_common.ontology.Ontology` builds
    when a definition asks for a live read, so a ``K``-keyed ontology whose
    structure came back ``str``-keyed would be an axis in one space beside a
    content lookup in another. A consumer's own concrete is free to bind
    ``str`` and compile unchanged; this one is on the path from a widened
    surface to a widened axis, which is the difference.

    ``parents(x)`` is ``find(subject=x, ...)`` read for its objects;
    ``children(x)`` is ``find(object=x, ...)`` read for its subjects -- the
    mirrored query, not the same one. Both go through :meth:`_find`, which
    supplies this axis's relation to :func:`edge_criteria`, where *asserted*
    is decided once for every reader of an axis, inside this module and out.

    **The relation is canonicalised at construction.** ``RelationRef`` is two
    spellings of one name -- an id, or the definition itself -- and
    :func:`~dataknobs_common.ontology.model.relation_id` is where this
    package decides which it was handed. Every *read* went through it. The two
    members ``@dataclass`` generates did not, so an axis named by the
    definition read identically to one named by the id, compared unequal to
    it, and withheld the hash a frozen field tuple promises -- a
    ``RelationType`` is an :class:`~dataknobs_common.ontology.model.Entity`,
    honestly unhashable, and a field is where that honesty inverts.

    Normalised here, this axis follows the rule the whole family follows, for
    a field shape none of the others has: **compare by the identity of what
    you hold, and the value of what you name.** The handle is the source,
    compared by identity as any object is; the name is the relation, compared
    by value -- which is only coherent once the name has one spelling. The
    door still takes either, and :attr:`relation` hands back the id.
    """

    source: AssertionSource[K]
    relation: RelationRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", relation_id(self.relation))

    def _find(
        self, *, subject: K | None = None, object: Term[K] | None = None
    ) -> Sequence[Assertion[K]]:
        """Every **asserted** edge of this relation matching the criteria.

        Every read below goes through here, and this goes through
        :func:`edge_criteria` -- so a member added later cannot omit the
        relation or the polarity by writing a ``self.source.find(...)`` that
        looks complete, which is exactly how :meth:`parent_edges`, the newest
        of them, would have arrived unfiltered. The narrowing itself is not
        this method's: it is the function's, so a reader outside this class
        narrows the same way by unpacking the same criteria.

        The module already shared its *result* side -- :func:`_parents_of`,
        :func:`_children_of`, :func:`_nodes_of`, :func:`_parent_edges_of` --
        and shared nothing on the query side, which is why one decision was
        once written at eight call sites per flavour.
        """
        return self.source.find(subject=subject, object=object, **edge_criteria(self.relation))

    def _find_many(
        self, *, subjects: Sequence[K] | None = None, objects: Sequence[K] | None = None
    ) -> Mapping[K, Sequence[Assertion[K]]]:
        """:meth:`_find`'s bulk form, narrowed identically."""
        return self.source.find_many(
            subjects=subjects, objects=objects, **edge_criteria(self.relation)
        )

    def roots(self) -> Sequence[K]:
        """The nodes this axis has with no parent.

        Not an extent, and the distinction matters: this is *the nodes this
        relation leaves unplaced*, which equals a type's membership only by
        coincidence. ``EntitySource.by_type`` is the extent.

        A node whose only parent edge is negated **is** a root here. It has no
        asserted parent, and reporting it as placed would report a placement
        no edge makes.
        """
        edges = self._find()
        placed = {edge.subject for edge in edges if object_entity_id(edge.object) is not None}
        return tuple(node for node in _nodes_of(edges) if node not in placed)

    def parents(self, node_id: K) -> Sequence[K]:
        """The nodes ``node_id`` is directly under."""
        return _parents_of(self._find(subject=node_id))

    def children(self, node_id: K) -> Sequence[K]:
        """The nodes directly under ``node_id``."""
        return _children_of(self._find(object=EntityRef(node_id)))

    def parents_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`parents` for a whole frontier, in one query.

        ``AssertionSource.find_many`` is the bulk form the singular member
        cannot reach, so a level costs one call rather than one per node. The
        reply is positional -- a node the query returned nothing for gets an
        empty tuple, not a missing slot.
        """
        found = self._find_many(subjects=tuple(node_ids))
        return tuple(_parents_of(found.get(node_id, [])) for node_id in node_ids)

    def children_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`children` for a whole frontier, in one query."""
        found = self._find_many(objects=tuple(node_ids))
        return tuple(_children_of(found.get(node_id, [])) for node_id in node_ids)

    def contains(self, node_id: K) -> bool:
        """Whether any **asserted** edge of this relation names the node.

        What keeps *nothing below this node* distinguishable from *this node is
        not here*. An id no edge of this relation mentions is absent, and an
        absent node is neither a root nor a leaf.

        A node named only by negated edges is absent, for the reason a node
        named only as a literal object is: an axis is made of asserted edges
        between entities, and a negation places nothing.
        """
        if self._find(subject=node_id):
            return True
        return bool(self._find(object=EntityRef(node_id)))

    def parent_edges(self) -> Mapping[K, Sequence[K]]:
        """Every asserted edge of this relation, in one query.

        The member that makes a copy of this axis complete. It is one ``find``
        for the whole relation -- the same query :meth:`roots` already makes --
        rather than a descent, so what it can see does not depend on anything
        being reachable from a root.

        Negated edges are absent, and that is what keeps a copy the same
        *shape* as the axis it copies: a snapshot holding edges the live read
        excludes would answer differently from the thing it is a snapshot of.
        """
        return _parent_edges_of(self._find())


@dataclass(frozen=True)
class AsyncAssertionHierarchy(Generic[K]):
    """The asynchronous twin, over an ``AsyncAssertionSource``.

    Same two fields and the same members, every one ``async def``, returning a
    ``Sequence`` rather than an async iterable -- which is what lets both
    flavours share one implementation of every walk.

    Its :meth:`_find` pair is written out rather than shared with the
    synchronous twin's: the twins share no runtime code, because one awaits
    and the other cannot. What they do share is :func:`edge_criteria`, which
    performs no read and so has no flavour.
    """

    source: AsyncAssertionSource[K]
    relation: RelationRef

    def __post_init__(self) -> None:
        """Canonicalise the relation, on this class's own account.

        The twins share no body, so there is no base to inherit this from --
        the same obligation the mapping side meets by writing ``eq=False`` on
        each of its three decorators, arrived at from the other direction.
        """
        object.__setattr__(self, "relation", relation_id(self.relation))

    async def _find(
        self, *, subject: K | None = None, object: Term[K] | None = None
    ) -> Sequence[Assertion[K]]:
        """:meth:`AssertionHierarchy._find`, awaited."""
        return await self.source.find(
            subject=subject, object=object, **edge_criteria(self.relation)
        )

    async def _find_many(
        self, *, subjects: Sequence[K] | None = None, objects: Sequence[K] | None = None
    ) -> Mapping[K, Sequence[Assertion[K]]]:
        """:meth:`AssertionHierarchy._find_many`, awaited."""
        return await self.source.find_many(
            subjects=subjects, objects=objects, **edge_criteria(self.relation)
        )

    async def roots(self) -> Sequence[K]:
        """The nodes this axis has with no parent."""
        edges = await self._find()
        placed = {edge.subject for edge in edges if object_entity_id(edge.object) is not None}
        return tuple(node for node in _nodes_of(edges) if node not in placed)

    async def parents(self, node_id: K) -> Sequence[K]:
        """The nodes ``node_id`` is directly under."""
        return _parents_of(await self._find(subject=node_id))

    async def children(self, node_id: K) -> Sequence[K]:
        """The nodes directly under ``node_id``."""
        return _children_of(await self._find(object=EntityRef(node_id)))

    async def parents_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`AssertionHierarchy.parents_many`, awaited."""
        found = await self._find_many(subjects=tuple(node_ids))
        return tuple(_parents_of(found.get(node_id, [])) for node_id in node_ids)

    async def children_many(self, node_ids: Sequence[K]) -> Sequence[Sequence[K]]:
        """:meth:`AssertionHierarchy.children_many`, awaited."""
        found = await self._find_many(objects=tuple(node_ids))
        return tuple(_children_of(found.get(node_id, [])) for node_id in node_ids)

    async def contains(self, node_id: K) -> bool:
        """Whether any asserted edge of this relation names the node."""
        if await self._find(subject=node_id):
            return True
        return bool(await self._find(object=EntityRef(node_id)))

    async def parent_edges(self) -> Mapping[K, Sequence[K]]:
        """:meth:`AssertionHierarchy.parent_edges`, awaited."""
        return _parent_edges_of(await self._find())


if TYPE_CHECKING:  # pragma: no cover - checked by the type checker, not run

    def _twins_satisfy_their_protocols() -> None:
        """Each twin is assignable to the hierarchy flavour it claims.

        Inline rather than in a test, because this file is type-checked and
        the test tree is not -- the arrangement ``sources.py`` already uses for
        the source protocols.
        """
        from dataknobs_common.hierarchy import (
            AsyncBulkHierarchy,
            AsyncEnumerableHierarchy,
            AsyncHierarchy,
            BulkHierarchy,
            EnumerableHierarchy,
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
        enumerable: EnumerableHierarchy = AssertionHierarchy(MappingAssertionSource([]), "isa")
        async_enumerable: AsyncEnumerableHierarchy = AsyncAssertionHierarchy(
            AsyncMappingAssertionSource([]), "isa"
        )
        del sync, asynchronous, bulk, async_bulk, enumerable, async_enumerable
