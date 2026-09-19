# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A hierarchy whose edges are a parent column of a table someone else owns.

The row-backed axis the bulk and enumerable hierarchy protocols were written
for. ``dataknobs_common.ontology.AsyncAssertionHierarchy`` reads its edges out
of an assertion store; this one reads them out of two columns of one table, and
the two are held against each other by a test in this package -- a taxonomy is
an ``AsyncTaxonomy`` either way, and only its ``structure`` differs, so a walk
that disagreed would mean the hierarchy family had two implementations of one
walk rather than two backings under it.

**Why this lives here and not beside the protocol it satisfies.**
``dataknobs_common.hierarchy`` is general: a hierarchy over a ``parent_id``
column has no ontology behind it, and the protocols say so. This concrete needs
an :class:`~dataknobs_data.database.AsyncDatabase` at runtime, so it goes where
its dependency is -- the same rule that keeps ``AssertionHierarchy`` under
``common/ontology/`` beside the ontology types it is made of. The mirror is
exact: ``data/ontology/hierarchy.py`` is to ``common/ontology/hierarchy.py``
what ``data/ontology/sources.py`` is to ``common/ontology/sources.py``.

A column name from configuration reaches a query builder and is never
interpolated into SQL: :class:`~dataknobs_data.query.Filter` is the only path.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING, Any

from dataknobs_common.exceptions import ValidationError

from dataknobs_data.ontology.sources import READ_BATCH_SIZE
from dataknobs_data.query import Filter, Operator, Query

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from dataknobs_data.database import AsyncDatabase

#: The axis kind this module binds. One value, spelled the way the published
#: ``taxonomies:`` grammar spells it, and read by the registry's axis dispatch
#: -- which computes its refusal from the declared kind alone, so a document
#: means the same thing whether or not this package has been imported.
COLUMN_AXIS_KIND = "column"

#: A ``(child, parent)`` pair, as one row of the table yields one.
_Edge = tuple[str, str]


@dataclass(frozen=True)
class ColumnAxisBinding:
    """The parsed ``kind: column`` row of a ``taxonomies:`` section.

    Configuration, and only the part the *binder* consumes. ``kind:`` is the
    discriminator rather than a field, and ``id:`` and ``relation:`` are read
    by ``TaxonomyDefinition`` -- a second reader for either would be a second
    place the same key is spelled.

    **There is no ``child:``.** The child column is the source's own ``id:``,
    which is what keeps the axis and the entity source keyed alike by
    construction: the ids a walk answers with are the ids ``entity()`` takes.
    A ``child:`` on this row would be a second place to say that, and the first
    place two id spaces could diverge.

    Attributes:
        source: The id of a ``kind: record`` source **this document declares**
        parent_key: The column holding the parent's key, in the space the
            source's projection ``id:`` names
    """

    source: str
    parent_key: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, binding: str) -> ColumnAxisBinding:
        """Parse an axis row, naming the axis on a refusal.

        Args:
            row: The ``taxonomies:`` row as written
            binding: The axis id, so every refusal names what to go and fix

        Returns:
            The parsed binding

        Raises:
            ValidationError: On a missing ``source:`` or ``parent_key:``
        """
        return cls(
            source=str(_required(row, "source", binding=binding)),
            parent_key=str(_required(row, "parent_key", binding=binding)),
        )


def _required(row: Mapping[str, Any], key: str, *, binding: str) -> Any:
    value = row.get(key)
    if value is None or value == "":
        raise ValidationError(
            f"taxonomy {binding!r} declares `kind: {COLUMN_AXIS_KIND}` and no "
            f"`{key}:`. A column axis names the source whose table it reads and "
            f"the column holding the parent's key. Declared: {sorted(row)}",
            context={"taxonomy": binding, "kind": COLUMN_AXIS_KIND, "key": key},
        )
    return value


def _ordered(node_ids: Iterable[str]) -> tuple[str, ...]:
    """Deduplicate in first-appearance order.

    A DAG node is reachable by several paths, so a repeated id changes no
    membership answer and does change a count someone is reporting.
    """
    return tuple(dict.fromkeys(node_ids))


def _parent_edges_of(edges: Sequence[_Edge]) -> dict[str, tuple[str, ...]]:
    """Every node these pairs mention, and what each is directly under.

    The extent :meth:`ColumnHierarchy.roots` is not. Every node gets an entry,
    including one that only ever appears as somebody's parent; its entry is
    empty, which is the same thing ``roots()`` reports about it.

    The pair-shaped twin of ``dataknobs_common.ontology.hierarchy``'s
    assertion-shaped ``_parent_edges_of``, and it is the same rule over a
    different edge representation: there an edge is an asserted relation
    between two entities, here it is two columns of one row. The rule the two
    share is stated once, in
    :meth:`~dataknobs_common.hierarchy.Hierarchy.roots`'s own docstring, which
    both implementations are written against.
    """
    above: dict[str, list[str]] = {}
    for child, parent in edges:
        above.setdefault(child, [])
        above.setdefault(parent, [])
        if parent not in above[child]:
            above[child].append(parent)
    return {node_id: tuple(parents) for node_id, parents in above.items()}


def _eq(column: str, node_id: str) -> Filter:
    """One end of an edge, pinned to a key.

    ``EQ`` is a byte comparison, which is the one operation that means the same
    thing on every backend in this package -- the reason
    :meth:`~dataknobs_data.ontology.sources.RecordEntitySource.by_surface_form`
    gives for taking it over a query-time fold.
    """
    return Filter(column, Operator.EQ, node_id)


def _nodes_of(edges: Sequence[_Edge]) -> tuple[str, ...]:
    """Every node these pairs mention, deduplicated in first-appearance order.

    Read order, because it is the only order a table gives and the alternative
    -- sorting -- would make ``roots()`` report an order nothing chose.
    """
    nodes: list[str] = []
    for child, parent in edges:
        nodes.append(child)
        nodes.append(parent)
    return _ordered(nodes)


@dataclass(frozen=True)
class ColumnHierarchy:
    """An :class:`~dataknobs_common.hierarchy.AsyncHierarchy` over a parent column.

    ``parents(x)`` is *the row whose child column is x, read for its parent
    column*; ``children(x)`` is *the rows whose parent column is x, read for
    their child column* -- the mirrored query, not the same one. Both go
    through :meth:`_edges`, which is the one place a column name is spent.

    **The bulk and enumerable members are not an optimisation, they are what
    the two optional protocols were written for.** Their own docstrings name
    the case: *"a row- or document-backed hierarchy … N queries per level
    against a database is the cost the singular members force, and it cannot be
    fixed from inside them."* This is the first row-backed axis in the
    workspace, so shipping it with the four singular members alone would be
    shipping the construct those protocols exist for without them.

    ``parent_edges()`` is load-bearing twice. It is what lets
    :meth:`~dataknobs_common.hierarchy.AsyncMappingHierarchy.snapshot` take a
    copy in **one** query rather than by descending from the roots -- and a
    descent cannot reach a cyclic component with nothing above it, so the copy
    would refuse an anchor the live axis accepts. A ``parent_id`` column with a
    cycle in it is not hypothetical; nothing constrains one.

    **A node is in this axis iff it appears in an edge**, which is the
    protocol's rule and not this class's: *"A node absent from every edge of
    this axis is not in this hierarchy at all, so ``roots()`` equals a type's
    membership only by coincidence."* So a row whose parent column is null
    **and** which nothing names as a parent is not here: ``contains()`` is
    False, ``roots()`` omits it, and a cursor over it reports ``exists()``
    False. The alternative -- every row of the table is a node -- reads more
    naturally for a catalogue and would make this axis disagree with the same
    edges read as assertions on every isolated row.

    **An edge is a row with both ends**, and that is one rule with three
    consequences. It is spelled as two ``EXISTS`` filters, which mean *is not
    null* on every backend in this package -- ``IS NOT NULL`` on the three SQL
    ones, ``exists`` on Elasticsearch, ``is not None`` in process. It excludes
    a row with no parent, which is what makes the node set edge-derived. And
    over a **shared store**, where the handle *is* the store and surface-form
    rows live beside entity rows, it excludes the form rows for free: a form
    row carries the projection's id column and no parent column, so no
    narrowing of this axis's own has to be written for an arrangement the
    entity source needed one for.

    Referential integrity is not this axis's subject. A parent column naming a
    key no row carries places a node that ``entity()`` will not find, exactly
    as an assertion may name an entity no ``entities:`` row declares --
    ``_publish_taxonomy_delta`` states that case from the other side.

    Args:
        database: The handle the rows are read through -- the **source's**,
            not one of this axis's own. The registry caches a handle per
            resolved block and table, and this axis names the source's table,
            so binding one opens nothing and closes nothing
        table: The table the handle addresses. Carried for the reason
            ``EntityProjection.table`` is carried: it is what decides whether a
            second handle was opened, and it is what a report names. **No query
            here carries it** -- on ``memory``, ``file``, ``s3`` and
            ``elasticsearch`` a handle is the store and the name narrows
            nothing; on the three SQL backends the handle already addresses it
        child: The column holding a row's own key -- the source's projection
            ``id:``, so that a walk answers in the space ``entity()`` takes
        parent: The column holding the parent's key, in that same space
    """

    database: AsyncDatabase
    table: str
    child: str
    parent: str

    async def _edges(self, *narrowing: Filter) -> list[_Edge]:
        """Every ``(child, parent)`` pair matching the criteria, in one read.

        Every member below goes through here, so a member added later cannot
        omit an end of the edge by writing a search that looks complete --
        which is how ``parent_edges``, the one whose whole job is completeness,
        would have arrived carrying rows that are not edges.

        **Streamed rather than searched, and the reason is a shipped defect one
        module over.** What this returns is bounded by the *data* and not by
        the caller's input: ``parent_edges()`` is the whole table, and
        ``children(k)`` over a wide node is however many rows name it. An
        unbounded ``search`` is the read a backend is free to cap, and one does
        -- ``AsyncElasticsearchDatabase`` answers a query carrying no ``limit``
        with ``size=10000``, so a large axis would have answered with the cap's
        worth of edges and reported nothing wrong.
        :meth:`~dataknobs_data.ontology.sources.RecordEntitySource.by_type`
        carries the same reasoning for the same reason, and the streaming door
        honours the full operator set on every async backend.

        A row is skipped where either end is absent after the read as well as
        before it: the filters do the work, and the check below is what keeps
        the answer's type honest rather than a second opinion about which rows
        are edges.
        """
        query = Query(
            filters=[
                Filter(self.child, Operator.EXISTS),
                Filter(self.parent, Operator.EXISTS),
                *narrowing,
            ]
        )
        edges: list[_Edge] = []
        async for record in self.database.stream_read(query):
            child = record.get_value(self.child)
            parent = record.get_value(self.parent)
            if child is None or parent is None:
                continue
            edges.append((str(child), str(parent)))
        return edges

    async def _edges_for(self, column: str, node_ids: Sequence[str]) -> list[_Edge]:
        """:meth:`_edges` for a whole frontier, one ``IN`` read per batch.

        The shared half of the two bulk members, so the pair cannot drift on
        what a frontier read narrows by. One filter is what makes a bulk member
        worth having, but one filter is one read and a read is a thing a
        backend bounds -- by result size on Elasticsearch, by bind parameters
        on the SQL backends -- so the list is split at
        :data:`~dataknobs_data.ontology.sources.READ_BATCH_SIZE`, which is this
        package's existing answer to *how many rows per read*.
        """
        found: list[_Edge] = []
        for chunk in batched(_ordered(node_ids), READ_BATCH_SIZE):
            found.extend(await self._edges(Filter(column, Operator.IN, list(chunk))))
        return found

    async def roots(self) -> Sequence[str]:
        """The nodes this axis has with no parent.

        Not an extent. This is *the rows this column leaves unplaced*, which
        equals the table's membership only by coincidence -- a row with a null
        parent that nothing names as a parent is not in the axis at all.
        """
        edges = await self._edges()
        placed = {child for child, _ in edges}
        return tuple(node for node in _nodes_of(edges) if node not in placed)

    async def parents(self, node_id: str) -> Sequence[str]:
        """The nodes ``node_id`` is directly under.

        Plural, always: the column holds one key per row, and nothing here
        promises the child column is unique, so two rows carrying one key are
        two edges rather than an error this axis is in a position to raise.
        """
        return _ordered(parent for _, parent in await self._edges(_eq(self.child, node_id)))

    async def children(self, node_id: str) -> Sequence[str]:
        """The nodes directly under ``node_id``."""
        return _ordered(child for child, _ in await self._edges(_eq(self.parent, node_id)))

    async def parents_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        """:meth:`parents` for a whole frontier, in one query per batch.

        The reply is positional -- a node the read returned nothing for gets an
        empty tuple, not a missing slot -- and it is in the order asked, which
        is what the walk drivers index it by.
        """
        found = _parent_edges_of(await self._edges_for(self.child, node_ids))
        return tuple(found.get(node_id, ()) for node_id in node_ids)

    async def children_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        """:meth:`children` for a whole frontier, in one query per batch."""
        below: dict[str, list[str]] = {}
        for child, parent in await self._edges_for(self.parent, node_ids):
            beneath = below.setdefault(parent, [])
            if child not in beneath:
                beneath.append(child)
        return tuple(tuple(below.get(node_id, ())) for node_id in node_ids)

    async def contains(self, node_id: str) -> bool:
        """Whether any edge of this column names the node, as a child or a parent.

        What keeps *nothing below this node* distinguishable from *this node is
        not here* -- opposite answers an empty ``children()`` alone cannot tell
        apart. Two reads rather than one, and the second only where the first
        found nothing: a disjunction over two columns is not a narrowing every
        backend in this package compiles the same way, and a byte comparison
        per column is.
        """
        if await self._edges(_eq(self.child, node_id)):
            return True
        return bool(await self._edges(_eq(self.parent, node_id)))

    async def parent_edges(self) -> Mapping[str, Sequence[str]]:
        """Every node this axis knows, and what each is directly under, in one read.

        The member that makes a copy of this axis complete, and the one a
        delta over its population is computed from. One read for the whole
        column rather than a descent, so what it can see does not depend on
        anything being reachable from a root.
        """
        return _parent_edges_of(await self._edges())


if TYPE_CHECKING:  # pragma: no cover - checked by the type checker, not run

    def _it_satisfies_the_protocols_it_claims() -> None:
        """The axis is assignable to all three asynchronous hierarchy flavours.

        Inline rather than in a test, because this file is type-checked and the
        test tree is not -- the arrangement ``common/ontology/hierarchy.py``
        uses for its own twins, and ``sources.py`` for the source protocols.
        The two optional protocols are opt-in **by member presence**: nothing
        registers, and ``_walk_core`` dispatches on ``getattr``, so an
        assignment is the only place a missing member is reported at all.
        """
        from dataknobs_common.hierarchy import (
            AsyncBulkHierarchy,
            AsyncEnumerableHierarchy,
            AsyncHierarchy,
        )

        from dataknobs_data.backends.memory import AsyncMemoryDatabase

        axis = ColumnHierarchy(AsyncMemoryDatabase(), "products", "sku", "parent_sku")
        plain: AsyncHierarchy[str] = axis
        bulk: AsyncBulkHierarchy[str] = axis
        enumerable: AsyncEnumerableHierarchy[str] = axis
        del plain, bulk, enumerable


__all__ = ["COLUMN_AXIS_KIND", "ColumnAxisBinding", "ColumnHierarchy"]
