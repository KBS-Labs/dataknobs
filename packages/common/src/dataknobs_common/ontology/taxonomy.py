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

**Why this lives under ``ontology/`` and the protocols it holds do not.** A
taxonomy is reachable without an *ontology* -- the definition is a value and
the three backings are protocols, so one built by hand over a mapping loads no
vocabulary -- but not without the ontology *model*: every field is one of its
types, and the cursor below builds an ``EntityRef`` to ask what is written on
an edge and reads the answer through ``object_entity_id``. Those are runtime
needs, and the general module ``dataknobs_common.hierarchy`` may not reach the
specific package for them -- this package's ``__init__`` imports it, through the
values module, so an edge back would close a cycle at import time. A concrete
goes where its dependency is; this one's is the model, so it sits beside the
assertion backing here rather than beside the protocols.
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
from dataknobs_common.hierarchy import (
    DEFAULT_FRONTIER_CONCURRENCY,
    AsyncHierarchyView,
    HierarchyView,
)
from dataknobs_common.ontology.hierarchy import edge_criteria
from dataknobs_common.ontology.model import EntityRef
from dataknobs_common.ontology.sources import object_entity_id

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from dataknobs_common.hierarchy import AsyncHierarchy, Hierarchy
    from dataknobs_common.ontology.model import Assertion, TaxonomyDefinition
    from dataknobs_common.ontology.sources import (
        AssertionSource,
        AsyncAssertionSource,
        AsyncEntitySource,
        EntitySource,
    )

__all__ = ["AsyncTaxonomy", "AsyncTaxonomyView", "Taxonomy", "TaxonomyView"]


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


@dataclass(frozen=True, eq=False)
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

    **Frozen, and compared by identity.** Frozen because nothing mutates a
    built axis: swapping a backing is :func:`dataclasses.replace`, which says
    at the call site that a second axis now exists. Identity because
    :class:`TaxonomyView` is frozen too and therefore hashes what it holds:
    field-wise equality would generate a ``__hash__`` that reaches
    ``definition.metadata`` and raises, and no amount of freezing fixes a dict.
    See :class:`TaxonomyView` for why that trade is the cheap one.
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

    def at(self, node_id: str) -> TaxonomyView:
        """The cursor over this axis, anchored at ``node_id`` -- the door in.

        Pure over fields this object already holds, so it takes nothing but
        the node. It does **not** check containment, and that is the whole
        precision of it: the view answers ``exists()`` itself, at the point a
        caller asks, which keeps *nothing below this node* and *this node is
        not here* apart -- where :meth:`walk` must refuse an unknown anchor,
        because a walk *includes* it and would emit it as a term of the axis.
        """
        return TaxonomyView(self, node_id)


@dataclass(frozen=True, eq=False)
class AsyncTaxonomy:
    """The asynchronous twin. Same four fields, asynchronous backings.

    The key is pinned to ``str`` here too, for the reason
    :class:`Taxonomy` states: a taxonomy carries the content axis as well as
    the structure one, and the content axis is ``str``-addressed. Frozen and
    identity-compared for the reason it states as well -- a difference here
    would be one :func:`assert_twin_types_agree` cannot see, since it reads
    members and these are decisions about the type.
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

    def at(self, node_id: str) -> AsyncTaxonomyView:
        """:meth:`Taxonomy.at`, and a plain ``def`` for the same reason.

        Checking containment here would drag an accessor into the loop for a
        question the caller has not asked; ``structure.contains`` is ``async``
        on this flavour. The check lives on the view.
        """
        return AsyncTaxonomyView(self, node_id)


# --------------------------------------------------------------------------
# The anchored view over an axis -- the cursor that can also report an edge
# --------------------------------------------------------------------------


def _written_by_object(written: Sequence[Assertion]) -> dict[str, list[Assertion]]:
    """Edge assertions grouped by the entity their object names, in source order.

    An assertion whose object is a literal names no neighbour and is dropped,
    for the reason the assertion backing gives: ``dog lifespan_years 12`` is a
    value rather than a place in a structure.
    """
    grouped: dict[str, list[Assertion]] = {}
    for assertion in written:
        neighbour = object_entity_id(assertion.object)
        if neighbour is not None:
            grouped.setdefault(neighbour, []).append(assertion)
    return grouped


def _written_by_subject(written: Sequence[Assertion]) -> dict[str, list[Assertion]]:
    """The mirror: edge assertions grouped by subject, in source order."""
    grouped: dict[str, list[Assertion]] = {}
    for assertion in written:
        grouped.setdefault(assertion.subject, []).append(assertion)
    return grouped


@dataclass(frozen=True)
class TaxonomyView:
    """The cursor over a taxonomy, so it can also report what is written on an edge.

    **Not generic in a key.** A taxonomy binds an ``EntitySource`` and an
    ``AssertionSource`` whose keys are the ontology's local entity ids --
    ``taxonomy.structure`` is a ``Hierarchy[str]``, and this cursor's node is a
    ``str``.

    **Every structural member invokes the ``HierarchyView`` member of the same
    name over ``taxonomy.structure`` and re-wraps; it does not re-walk.** That
    is exactly the shape a maintainer reimplements without noticing the
    forward is there, so a test patches each ``HierarchyView`` member and
    asserts the member here moves. ``at`` is the one exception: it constructs
    a ``TaxonomyView``, which no ``HierarchyView`` member can return, so it
    forwards to nothing.

    The two edge members are the reason this class exists over
    ``HierarchyView``: they read the assertions the edges were made of, which
    a bare ``Hierarchy`` has none of.

    **The axis is compared by identity, and that is what makes this hashable.**
    A frozen dataclass hashes its field tuple, so a cursor can only hash if the
    axis it holds can. Two of a taxonomy's four fields cannot be made to:
    ``definition`` carries a ``metadata`` dict, and a ``MappingHierarchy``
    structure carries two more, so freezing them would leave ``__hash__``
    generated and still raising -- promising the capability at ``isinstance``
    and failing at the call. Identity closes that for every backing at once,
    and it costs nothing that was being used: an axis is *built* by
    :meth:`Ontology.taxonomy` from a definition, and it is the definition that
    this module calls the value.

    So two cursors are equal exactly when they name the same node **of the same
    built axis**, which is the sentence this class was already documented by.
    """

    taxonomy: Taxonomy
    node: str

    def _structural(self) -> HierarchyView[str]:
        """The cursor every structural member forwards to."""
        return HierarchyView(self.taxonomy.structure, self.node)

    def _wrap(self, views: tuple[HierarchyView[str], ...]) -> tuple[TaxonomyView, ...]:
        return tuple(TaxonomyView(self.taxonomy, view.node) for view in views)

    def exists(self) -> bool:
        """Whether the structure axis knows this node at all."""
        return self._structural().exists()

    def is_root(self) -> bool:
        """Present, with nothing above it. ``False`` for an absent node."""
        return self._structural().is_root()

    def is_leaf(self) -> bool:
        """Present, with nothing below it. ``False`` for an absent node."""
        return self._structural().is_leaf()

    def parents(self) -> tuple[TaxonomyView, ...]:
        """One view per node directly above this one. Plural, always."""
        return self._wrap(self._structural().parents())

    def children(self) -> tuple[TaxonomyView, ...]:
        """One view per node directly below this one."""
        return self._wrap(self._structural().children())

    def at(self, node_id: str) -> TaxonomyView:
        """Re-anchor at another node of the same axis. Constructs; checks nothing."""
        return TaxonomyView(self.taxonomy, node_id)

    def parent_edges(self) -> tuple[tuple[TaxonomyView, Assertion], ...]:
        """What is written on each edge up from here: one pair per assertion.

        The neighbour cursor and the ``Assertion`` together, so one call gives
        both the annotation and a position to keep walking from. **Keyed per
        assertion, not per parent**: two parents with one assertion each give
        two pairs, and one parent annotated twice also gives two. ``()``
        therefore means *nothing is written on any edge here* -- true for a
        root, for an absent node, for a taxonomy whose ``assertions`` is
        ``None``, and for parents nothing annotates. ``taxonomy.assertions is
        None`` is the question that separates the third; :meth:`parents` and
        :meth:`exists` answer the other two.

        The structural half is :meth:`parents`: the structure decides which
        parents there are, and this reads only what is written on the edges to
        them -- so a structure copied at load and a live assertion source
        disagree in the structure's favour. The read narrows to asserted
        assertions the way the structure does, through
        :func:`~dataknobs_common.ontology.hierarchy.edge_criteria`, so a
        document stating both polarities on one edge does not hand the
        negation back as the edge walked.
        """
        source = self.taxonomy.assertions
        if source is None:
            return ()
        above = self.parents()
        if not above:
            return ()
        written = _written_by_object(
            source.find(subject=self.node, **edge_criteria(self.taxonomy.definition.relation))
        )
        return tuple(
            (parent, assertion) for parent in above for assertion in written.get(parent.node, ())
        )

    def child_edges(self) -> tuple[tuple[TaxonomyView, Assertion], ...]:
        """:meth:`parent_edges` in the other direction: the edges down from here."""
        source = self.taxonomy.assertions
        if source is None:
            return ()
        below = self.children()
        if not below:
            return ()
        written = _written_by_subject(
            source.find(
                object=EntityRef(self.node),
                **edge_criteria(self.taxonomy.definition.relation),
            )
        )
        return tuple(
            (child, assertion) for child in below for assertion in written.get(child.node, ())
        )


@dataclass(frozen=True)
class AsyncTaxonomyView:
    """The twin, over an :class:`AsyncTaxonomy`.

    The structural members invoke :class:`~dataknobs_common.hierarchy.AsyncHierarchyView`
    and the edge members read an ``AsyncAssertionSource``; every one is
    ``async def`` bar :meth:`at`, which constructs.
    """

    taxonomy: AsyncTaxonomy
    node: str

    def _structural(self) -> AsyncHierarchyView[str]:
        """The cursor every structural member forwards to."""
        return AsyncHierarchyView(self.taxonomy.structure, self.node)

    def _wrap(self, views: tuple[AsyncHierarchyView[str], ...]) -> tuple[AsyncTaxonomyView, ...]:
        return tuple(AsyncTaxonomyView(self.taxonomy, view.node) for view in views)

    async def exists(self) -> bool:
        """Whether the structure axis knows this node at all."""
        return await self._structural().exists()

    async def is_root(self) -> bool:
        """Present, with nothing above it. ``False`` for an absent node."""
        return await self._structural().is_root()

    async def is_leaf(self) -> bool:
        """Present, with nothing below it. ``False`` for an absent node."""
        return await self._structural().is_leaf()

    async def parents(self) -> tuple[AsyncTaxonomyView, ...]:
        """One view per node directly above this one. Plural, always."""
        return self._wrap(await self._structural().parents())

    async def children(self) -> tuple[AsyncTaxonomyView, ...]:
        """One view per node directly below this one."""
        return self._wrap(await self._structural().children())

    def at(self, node_id: str) -> AsyncTaxonomyView:
        """Re-anchor at another node of the same axis. Constructs; checks nothing."""
        return AsyncTaxonomyView(self.taxonomy, node_id)

    async def parent_edges(self) -> tuple[tuple[AsyncTaxonomyView, Assertion], ...]:
        """:meth:`TaxonomyView.parent_edges`, awaited."""
        source = self.taxonomy.assertions
        if source is None:
            return ()
        above = await self.parents()
        if not above:
            return ()
        written = _written_by_object(
            await source.find(subject=self.node, **edge_criteria(self.taxonomy.definition.relation))
        )
        return tuple(
            (parent, assertion) for parent in above for assertion in written.get(parent.node, ())
        )

    async def child_edges(self) -> tuple[tuple[AsyncTaxonomyView, Assertion], ...]:
        """:meth:`TaxonomyView.child_edges`, awaited."""
        source = self.taxonomy.assertions
        if source is None:
            return ()
        below = await self.children()
        if not below:
            return ()
        written = _written_by_subject(
            await source.find(
                object=EntityRef(self.node),
                **edge_criteria(self.taxonomy.definition.relation),
            )
        )
        return tuple(
            (child, assertion) for child in below for assertion in written.get(child.node, ())
        )
