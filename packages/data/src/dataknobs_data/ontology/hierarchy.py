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

from dataknobs_common.async_iter import aclosing_iter
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.hierarchy import dedupe_ordered, nodes_of, parent_edges_of
from dataknobs_common.ontology import TAXONOMY_ROW_KEYS

from dataknobs_data.ontology.sources import READ_BATCH_SIZE
from dataknobs_data.query import Filter, Operator, Query

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dataknobs_data.database import AsyncDatabase

#: The axis kind this module binds. One value, spelled the way the published
#: ``taxonomies:`` grammar spells it, and read by the registry's axis dispatch.
#:
#: The refusal that holds whether or not this package has been imported is the
#: **loader's** -- ``_refuse_unbindable_axes`` computes it from the declared
#: kind alone, in a distribution that binds no backing at all, so a document
#: means the same thing either way. The registry's own dispatch is the other
#: half of that split and lives here by definition: its ``LIVE_AXIS_KINDS``
#: cannot be consulted without importing the code that fills it.
COLUMN_AXIS_KIND = "column"

#: A ``(child, parent)`` pair, as one row of the table yields one.
#:
#: Both ends are present rather than ``str | None``, which is the shape
#: :func:`~dataknobs_common.hierarchy.parent_edges_of` takes: the two ``EXISTS``
#: filters are what make that true, and this alias is where the guarantee they
#: buy is written down.
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
            ValidationError: On a missing ``source:`` or ``parent_key:``, or on
                a key nothing reading this row reads
        """
        _refuse_an_unread_key(row, binding=binding)
        return cls(
            source=str(_required(row, "source", binding=binding)),
            parent_key=str(_required(row, "parent_key", binding=binding)),
        )


#: Every key a ``kind: column`` row is read for, across both readers of it.
#:
#: ``TaxonomyDefinition``'s six, which ``dataknobs_common`` publishes rather
#: than this module re-spelling, plus the discriminator and the two this
#: binding parses. One set because one row: the split between the two readers
#: is an implementation detail of where the code lives, and an author looking
#: at the line has to be told what the *line* may carry.
_KNOWN_KEYS: frozenset[str] = TAXONOMY_ROW_KEYS | {"kind", "source", "parent_key"}


def _refuse_an_unread_key(row: Mapping[str, Any], *, binding: str) -> None:
    """Refuse a key on this row that neither reader of it reads.

    The rule ``kind:`` is refused under, applied to the rest of the row --
    ``_refuse_unbindable_axes`` in ``dataknobs_common`` states it: *a dropped
    key and an unsupported key have to look different, or the file says one
    thing and the vocabulary means another.* Enforced for ``kind:`` alone, it
    left a misspelt ``parent_col:`` caught only by the coincidence that the
    canonical spelling is then absent and required, and a key that is nobody's
    typo dropped in silence.

    **``child:`` gets its own message**, because this class's own docstring
    teaches it as the one key that deliberately does not exist, and an author
    who writes it anyway has read that and disagreed. Telling them the set of
    keys they could have written answers a question they did not ask; telling
    them where the child column actually comes from answers the one they did.
    """
    unread = sorted(set(row) - _KNOWN_KEYS)
    if not unread:
        return
    if "child" in unread:
        raise ValidationError(
            f"taxonomy {binding!r} declares `child:`, which a column axis does "
            f"not have. The child column is the source's projection `id:`, "
            f"which is what keeps the ids a walk answers with the ids "
            f"`entity()` takes -- a second place to say it is the first place "
            f"the two could diverge. Drop it, or change the source's `id:` if "
            f"the axis really is keyed on another column",
            context={"taxonomy": binding, "kind": COLUMN_AXIS_KIND, "key": "child"},
        )
    raise ValidationError(
        f"taxonomy {binding!r} declares {unread}, which nothing reading a "
        f"`kind: {COLUMN_AXIS_KIND}` row reads. Such a row is read for "
        f"{sorted(_KNOWN_KEYS)}",
        context={"taxonomy": binding, "kind": COLUMN_AXIS_KIND, "keys": unread},
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


def _eq(column: str, node_id: str) -> Filter:
    """One end of an edge, pinned to a key.

    ``EQ`` is a byte comparison, which is the one operation that means the same
    thing on every backend in this package -- the reason
    :meth:`~dataknobs_data.ontology.sources.RecordEntitySource.by_surface_form`
    gives for taking it over a query-time fold.
    """
    return Filter(column, Operator.EQ, node_id)


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

    **An edge is a row with both ends**, spelled as two ``EXISTS`` filters,
    which mean *is not null* on every backend in this package -- ``IS NOT
    NULL`` on the three SQL ones, ``exists`` on Elasticsearch, ``is not None``
    in process. It excludes a row with no parent, which is what makes the node
    set edge-derived.

    **Which rows are the binding's is a separate question, and it is the
    binding's to answer** -- :attr:`narrowing`, not this rule. Over a shared
    store, where the handle *is* the store and surface-form rows live beside
    entity rows, ``EXISTS`` on the parent column was relied on to exclude the
    form rows: a form row carries the projection's id column and, in the
    arrangements written so far, no parent column. That is a property of the
    *data* rather than of the declaration, and nothing enforces it -- a side
    table denormalised by a join carries whatever it was joined from. The axis
    then read edges out of rows
    :meth:`~dataknobs_data.ontology.sources.RecordEntitySource.entity_filters`
    excludes, and ``parents()`` disagreed with ``entity()`` about one id. Both
    readers of one binding now narrow by one filter, taken from the binding.

    Referential integrity is not this axis's subject. A parent column naming a
    key no row carries places a node that ``entity()`` will not find, exactly
    as an assertion may name an entity no ``entities:`` row declares --
    ``_publish_taxonomy_delta`` states that case from the other side.

    **The keys are read as strings**, matching
    :meth:`~dataknobs_data.ontology.sources.RecordEntitySource._projected`,
    which is what the *keyed alike by construction* claim above rests on for an
    id column that is not one. An integer ``id:``/``parent_key:`` pair walks in
    the string space, and a consumer holding integer keys converts once at the
    boundary rather than per member.

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
        narrowing: What every read of this axis is additionally bounded by, so
            that it reaches the same rows the entity source does. The
            binding's, handed over by the registry; empty where the handle
            addresses the entity table alone. It is a tuple of
            :class:`~dataknobs_data.query.Filter`, which is frozen and hashes
            over a projected value, so a narrowed axis hashes exactly as an
            unnarrowed one does. That is a property of the field's type rather
            than of this class, and it is named here because it was once the
            other way round: a filter was a mutable dataclass whose value may
            be a list, and adding this field made a *narrowed* axis withhold
            the hash its frozen field tuple promises -- the failure
            ``AssertionHierarchy`` normalises its own relation to avoid
    """

    database: AsyncDatabase
    table: str
    child: str
    parent: str
    narrowing: tuple[Filter, ...] = ()

    async def _edges(self, *criteria: Filter, limit: int | None = None) -> list[_Edge]:
        """Every ``(child, parent)`` pair matching the criteria, in one read.

        Every member below goes through here, so a member added later cannot
        omit an end of the edge by writing a search that looks complete --
        which is how ``parent_edges``, the one whose whole job is completeness,
        would have arrived carrying rows that are not edges. It is also the one
        place :attr:`narrowing` is spent, which is what makes *which rows of
        this store are the binding's* a property of the binding rather than of
        whichever member remembered to ask.

        ``limit`` bounds a read whose caller does not need all of it --
        :meth:`contains` is the case, and it is the only one: every other
        member's answer *is* the set. The bound is applied to the query and to
        the collection, which are the same number because the two ``EXISTS``
        filters are what decide whether a row is an edge; the check below
        cannot subtract from a bounded read without contradicting them.

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

        **Driven under :func:`~dataknobs_common.async_iter.aclosing_iter`,
        because the bounded branch breaks on the ordinary path.** ``limit``
        means :meth:`contains` stops at the first edge and stops on *success*,
        so a bare ``async for`` would leave a read suspended on every True
        this class answers --- and a bound on the query does not finish the
        generator either, since it is suspended at the row it was asked for
        rather than past it. On the backends whose ``stream_read`` holds a
        real resource across its yields that is a pooled connection inside an
        open transaction, released when the interpreter finalizes the
        abandoned read.
        """
        query = Query(
            filters=[
                Filter(self.child, Operator.EXISTS),
                Filter(self.parent, Operator.EXISTS),
                *self.narrowing,
                *criteria,
            ]
        )
        if limit is not None:
            query = query.limit(limit)
        edges: list[_Edge] = []
        async with aclosing_iter(self.database.stream_read(query)) as records:
            async for record in records:
                child = record.get_value(self.child)
                parent = record.get_value(self.parent)
                if child is None or parent is None:
                    continue
                edges.append((str(child), str(parent)))
                if limit is not None and len(edges) >= limit:
                    break
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
        for chunk in batched(dedupe_ordered(node_ids), READ_BATCH_SIZE):
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
        return tuple(node for node in nodes_of(edges) if node not in placed)

    async def parents(self, node_id: str) -> Sequence[str]:
        """The nodes ``node_id`` is directly under.

        Plural, always: the column holds one key per row, and nothing here
        promises the child column is unique, so two rows carrying one key are
        two edges rather than an error this axis is in a position to raise.
        """
        return dedupe_ordered(parent for _, parent in await self._edges(_eq(self.child, node_id)))

    async def children(self, node_id: str) -> Sequence[str]:
        """The nodes directly under ``node_id``."""
        return dedupe_ordered(child for child, _ in await self._edges(_eq(self.parent, node_id)))

    async def parents_many(self, node_ids: Sequence[str]) -> Sequence[Sequence[str]]:
        """:meth:`parents` for a whole frontier, in one query per batch.

        The reply is positional -- a node the read returned nothing for gets an
        empty tuple, not a missing slot -- and it is in the order asked, which
        is what the walk drivers index it by.
        """
        found = parent_edges_of(await self._edges_for(self.child, node_ids))
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

        **Bounded at one edge each, because the question is a bool.** Unbounded,
        the second read is *the rows whose parent column is this node*, which
        is the node's whole fan-out -- so a root with two hundred thousand
        children streamed two hundred thousand rows to answer ``True``. That is
        not a corner: ``AsyncHierarchyView.exists`` asks this, and
        ``Taxonomy.walk`` checks its anchor with it, so ``at(root)`` paid the
        fan-out before the walk began.
        """
        if await self._edges(_eq(self.child, node_id), limit=1):
            return True
        return bool(await self._edges(_eq(self.parent, node_id), limit=1))

    async def parent_edges(self) -> Mapping[str, Sequence[str]]:
        """Every node this axis knows, and what each is directly under, in one read.

        The member that makes a copy of this axis complete, and the one a
        delta over its population is computed from. One read for the whole
        column rather than a descent, so what it can see does not depend on
        anything being reachable from a root.
        """
        return parent_edges_of(await self._edges())


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
