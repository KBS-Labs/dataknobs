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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, NoReturn

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
    K,
    AsyncHierarchyView,
    HierarchyView,
    async_descendants_to_depth,
    async_flatten,
    descendants_to_depth,
    flatten,
)
from dataknobs_common.ontology.hierarchy import edge_criteria
from dataknobs_common.ontology.model import EntityRef
from dataknobs_common.ontology.sources import object_entity_id

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence

    from dataknobs_common._walk_core import WalkCache
    from dataknobs_common.hierarchy import AsyncHierarchy, Hierarchy
    from dataknobs_common.ontology.model import (
        Assertion,
        AttributeDef,
        Entity,
        EntityType,
        TaxonomyDefinition,
    )
    from dataknobs_common.ontology.sources import (
        AssertionSource,
        AsyncAssertionSource,
        AsyncEntitySource,
        EntitySource,
    )

__all__ = ["AsyncTaxonomy", "AsyncTaxonomyView", "Taxonomy", "TaxonomyView"]


def _refuse_an_unknown_anchor(taxonomy_id: str, anchor: object) -> NoReturn:
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

    **Not a second copy of the rule.**
    :func:`~dataknobs_common.hierarchy._refuse_an_unknown_anchor` keeps it for
    every walk that includes its anchor, and :meth:`Taxonomy.subtree_keys`
    reaches that one through the walk it delegates to. This exists because
    :meth:`Taxonomy.walk` delegates to no walk -- it streams, and drives
    itself -- and because only this frame can name the taxonomy in the
    refusal. What differs between the two is the message; what does not is
    which walks owe one.
    """
    raise NotFoundError(
        f"no node {anchor!r} in the structure axis of taxonomy {taxonomy_id!r}, "
        f"so it cannot anchor a walk",
        context={"taxonomy": taxonomy_id, "anchor": anchor},
    )


def _refuse_an_undeclared_type(
    taxonomy_id: str, entity_type: str, *, asked_about: str | None = None
) -> NoReturn:
    """Refuse an entity type the type store does not declare.

    Returning ``[]`` was the other candidate and is the worse one, for
    :func:`_refuse_an_unknown_anchor`'s reason at one remove: a type declared
    with no attributes and no parent **legitimately** inherits nothing, so an
    empty list for an undeclared one collapses *this type declares nothing*
    into *this type is not here*. The caller cannot tell those apart, and the
    reading they will reach for is the one that is not their fault.

    It is also what makes the store safe to default. An axis built without one
    answers nothing at all rather than answering *nothing is declared* about
    every type in the vocabulary.

    **Both ends of the walk are refused by this one rule**, which is the whole
    reason ``asked_about`` exists. The argument above is about a name the store
    does not hold, and it does not care whether that name arrived from the
    caller or from an ``isa:`` two hops up: a truncated answer and a complete
    one are indistinguishable either way. Refusing only the anchor left the
    ancestor case collapsing exactly as this docstring says it must not --
    silently, and reachable from any partial store.

    Args:
        taxonomy_id: The axis the question was asked of.
        entity_type: The type that is absent.
        asked_about: The type the caller actually named, when the absent one
            was reached from it. ``None`` when they are the same, which is the
            anchor case and needs no second name.
    """
    reached = (
        f", reached from {asked_about!r}"
        if asked_about is not None and asked_about != entity_type
        else ""
    )
    raise NotFoundError(
        f"no entity type {entity_type!r} in the type store of taxonomy "
        f"{taxonomy_id!r}{reached}, so nothing can be said about what it inherits",
        context={
            "taxonomy": taxonomy_id,
            "entity_type": entity_type,
            "asked_about": asked_about if asked_about is not None else entity_type,
        },
    )


def _inherited_attributes(
    entity_types: Mapping[str, EntityType], taxonomy_id: str, entity_type: str
) -> list[AttributeDef]:
    """The attribute declarations ``entity_type`` may be asked for, nearest first.

    **Shared by both flavours rather than written into each, and it is shared
    entire rather than in its frontier step**: the store is a mapping the
    caller is already holding, so there is no read here to await and therefore
    no half of this that differs between the twins. :meth:`Taxonomy.walk` is
    duplicated because it streams over a backing; this is not.

    **A nearer declaration shadows a farther one of the same name.** A subtype
    redeclaring ``sku`` is *specialising* it -- a different description, a
    different ``required`` -- and returning both would build a schema with two
    fields of one name and no rule for choosing between them.

    **The visited set is unconditional**, like every walk over the structure
    axis, and for a reason that is measured rather than defensive: a document
    declaring ``A isa B`` and ``B isa A`` **loads**. The loader refuses an
    ``isa:`` naming an undeclared type and nothing refuses one that closes a
    loop, so a cycle here is reachable from a valid document.

    **An ancestor the store does not hold is refused, exactly as the anchor
    is** -- by the same line, which is why they cannot drift. Stopping quietly
    at an absent parent was the other candidate and is the same collapse
    :func:`_refuse_an_undeclared_type` rules out one frame down: a truncated
    list and a complete one are the same list, so a caller reading a schema off
    a partial store got a short answer with nothing to say it was short. The
    loader validates every ``isa:`` in a document it reads, but ``entity_types``
    is a mapping the **caller** supplies and it defaults to empty -- a partial
    store is invited by the field rather than excluded by it, and
    :class:`~dataknobs_common.ontology.model.EntityType` is public and
    constructible.

    The anchor needs no separate check ahead of the loop: an absent one is
    absent on the first iteration, and the refusal reads ``asked_about`` equal
    to ``entity_type`` and says nothing about a route. A guard there would be a
    second site to keep in step with this one, for a message it already gives.

    Each type has at most one parent -- :attr:`EntityType.isa` is a scalar
    field on the declaration -- so there is no ordering to choose between
    branches, which is the one way this differs from the lattice ``structure``
    walks.
    """
    collected: list[AttributeDef] = []
    claimed: set[str] = set()
    seen: set[str] = set()
    current: str | None = entity_type
    while current is not None and current not in seen:
        seen.add(current)
        declaration = entity_types.get(current)
        if declaration is None:
            _refuse_an_undeclared_type(taxonomy_id, current, asked_about=entity_type)
        for attribute in declaration.attributes:
            if attribute.name in claimed:
                continue
            claimed.add(attribute.name)
            collected.append(attribute)
        current = declaration.isa
    return collected


@dataclass(frozen=True, eq=False)
class Taxonomy(Generic[K]):
    """One relation of a vocabulary, walkable, with synchronous backings.

    ``assertions`` is what lets a cursor over this axis report the **edge** it
    walked rather than only its endpoints. It is optional because not every
    hierarchy has assertions behind it -- one built from a ``parent_id`` column
    has rows and no ``Assertion`` -- and ``assertions is None`` is the question
    that tells an unannotated edge from an axis that has no annotations to give.

    **The key is a parameter, defaulted to ``str``.** It was pinned, and the
    pin was written out with its prerequisite: *the content side has to become
    generic first, or explicitly stay behind*. It became generic, so the pin is
    gone and the sentence it was written in is what this paragraph replaces.

    The reason for the pin survives the change and is worth keeping, because it
    is what the codec now carries. An ``EntitySource``'s keys are the
    ontology's local entity ids: they have an owner, a space and a
    qualification rule, and none of that survives being *parameterised*. What
    it does survive is being made an **object** -- a
    :class:`~dataknobs_common.ontology.values.KeyCodec` the ontology holds --
    so the rule has a home rather than a pin. A bare ``Taxonomy`` is
    ``Taxonomy[str]`` and reads exactly as it did.

    **Frozen, and compared by identity.** Frozen because nothing mutates a
    built axis: swapping a backing is :func:`dataclasses.replace`, which says
    at the call site that a second axis now exists. Identity because
    :class:`TaxonomyView` is frozen too and therefore hashes what it holds:
    field-wise equality would generate a ``__hash__`` that reaches
    ``definition.metadata`` and raises, and no amount of freezing fixes a dict.
    See :class:`TaxonomyView` for why that trade is the cheap one.
    """

    definition: TaxonomyDefinition
    structure: Hierarchy[K]
    entities: EntitySource[K]
    assertions: AssertionSource[K] | None = None

    #: The **type** lattice and the attribute declarations on it -- a different
    #: store from the ``isa`` assertions :attr:`structure` walks, and the one
    #: :meth:`inherited_attributes` reads.
    #:
    #: A ``Mapping`` rather than a source, and the asymmetry with
    #: :attr:`entities` is the design: a vocabulary's *instances* may be
    #: millions behind a store, and its *types* are tens, authored in the
    #: document and loaded whole. :class:`~dataknobs_common.ontology.Ontology`
    #: already carries them exactly this way.
    #:
    #: **Optional, because the member refuses rather than answering emptily.**
    #: An axis with no type store is an ordinary thing -- one built from a
    #: ``parent_id`` column has no ``EntityType`` any more than it has an
    #: ``Assertion`` -- and asking it what a type inherits is refused, naming
    #: the type. **Appended last** for the reason
    #: :attr:`~dataknobs_common.ontology.model.Assertion.polarity` carries: any
    #: earlier position moves ``assertions`` under a caller who passes it
    #: positionally.
    entity_types: Mapping[str, EntityType] = field(default_factory=dict)

    def walk(
        self,
        *,
        from_id: K | None = None,
        max_depth: int | None = None,
        cache: WalkCache | None = None,
    ) -> Iterator[K]:
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

        **Breadth first because it streams, which is arithmetic and not a
        preference.** :meth:`subtree_keys` answers this same question and emits
        *pre-order by discovery*, which places a node after everything under
        its earlier siblings -- so a stream in that order would hold a root's
        second child until the whole first branch was in, a lag bounded by that
        branch's depth and by nothing else. Emitting pre-order with a bounded
        lag means one request per node instead of one per level, which is what
        the shared frontier read exists not to do. The two therefore diverge
        wherever a branch has a branch under it, and both are right.

        ``cache`` is that shared step's memo, and this walk reaching it is the
        whole reason the memo lives there rather than in either driver. There is
        no default one: a streaming walk asks about each node exactly once, so a
        per-walk memo would buy nothing and retain the axis for the caller's
        iteration. What a caller *supplies* outlives the walk, which is what
        makes a cache filled by a collecting walk readable here.
        """
        seen: set[K] = set()
        if from_id is not None:
            if not self.structure.contains(from_id):
                _refuse_an_unknown_anchor(self.definition.id, from_id)
            frontier: tuple[K, ...] = (from_id,)
        else:
            frontier = tuple(self.structure.roots())
        depth = 0
        while frontier and (max_depth is None or depth <= max_depth):
            fresh: list[K] = []
            for node_id in frontier:
                if node_id in seen:
                    continue
                seen.add(node_id)
                fresh.append(node_id)
                yield node_id
            if max_depth is not None and depth == max_depth:
                return
            replies = _sync_reply(self.structure, "children", tuple(fresh), cache=cache)
            frontier = tuple(child for reply in replies for child in reply)
            depth += 1

    def subtree_keys(
        self,
        root_id: K,
        *,
        depth: int | None = None,
        cache: WalkCache | None = None,
    ) -> list[K]:
        """``root_id`` and everything under it, as the keys a query filters on.

        The member a taxonomy is usually resolved *for*: one line then builds
        the filter -- ``Filter(column, Operator.IN, axis.subtree_keys(node))``
        -- and a count over a foreign table covers the subtree rather than the
        one node a synonym list would have matched just as well.

        Four properties, each of which is a way to get it wrong:

        * **the root is included.** Naming an interior node means *this and
          everything under it*, and an off-by-one here under-counts silently
          while the count is the whole answer. This is the including half of
          the boundary :func:`~dataknobs_common.hierarchy.descendants`
          supplies the excluding half of;
        * **deduplicated, in walk order** -- a DAG node is reachable by several
          paths, so a raw walk repeats. A repeated key changes no ``IN`` result
          and does change a length a caller may be reporting;
        * **``depth`` is the bound, and it is optional.** Unbounded by default,
          because the ordinary question is *everything under this*;
        * **an unknown root is refused**, for :meth:`walk`'s reason and not a
          new one: this walk includes its anchor, so an unknown one would come
          back as a one-element list that the caller cannot tell from a leaf.

        **The keys are the structure axis's own, and that is the whole of the
        contract.** ``Filter(column, Operator.IN, axis.subtree_keys(node))`` is
        right exactly when the axis and that column are keyed alike, which is a
        property of how the axis was *bound* and not of this call: this frame
        knows the axis and has never been told which foreign table it is about
        to be filtered against, so a translation here would be a guess wearing
        a service's clothes. Where the two spaces do differ it is the caller who
        holds both, and
        :func:`~dataknobs_common.ontology.model.split_qualified` is on the
        package door for exactly that.

        It is also what keeps this surface coherent, which is why the
        translation could not be bolted on later either: ``root_id`` arrives in the
        axis's space, so a return in another one could not be fed back --
        neither to :meth:`at`, nor to :meth:`walk`, nor to
        ``structure.contains``, nor to this method. Every key it handed out
        would be a key it refuses.

        Two delegations rather than an algorithm: unbounded this *is*
        :func:`~dataknobs_common.hierarchy.flatten` and bounded it *is*
        :func:`~dataknobs_common.hierarchy.descendants_to_depth`, which is why
        it emits their pre-order and needs no rule of its own at either end of
        the bound -- which is a different order from :meth:`walk`'s over the
        same question, for the reason that method gives. Nothing here depends
        on the difference, since an ``IN`` clause has none. ``cache`` is
        forwarded to whichever one runs, because a delegation that drops a
        parameter its delegate grew is how the layer above a walk ends up
        unable to do what the walk can.

        The containment question is asked **here as well as** by the walk this
        delegates to, which also refuses an unknown anchor. Only this frame can
        name the taxonomy in the refusal, and the axis a caller filters on is
        where that context is worth one lookup; the walk's own refusal is what
        makes the rule true for somebody who reaches it directly.
        """
        if not self.structure.contains(root_id):
            _refuse_an_unknown_anchor(self.definition.id, root_id)
        if depth is None:
            return list(flatten(self.structure, from_id=root_id, cache=cache))
        return list(descendants_to_depth(self.structure, root_id, depth, cache=cache))

    def inherited_attributes(self, entity_type: str) -> list[AttributeDef]:
        """Every attribute declaration ``entity_type`` may be asked for, nearest first.

        **The other lattice.** :attr:`structure` walks the ``isa`` *assertions*
        between entities; this walks the ``isa:`` *field* on entity type
        declarations, which is a different store carrying a different kind of
        thing. ``ancestors`` over ``beagle`` gives ``dog`` and ``mammal``;
        this over ``Breed`` gives ``akc_group``, ``latin_name`` and
        ``lifespan_years``. A reader who reaches for this expecting ancestors
        gets declarations, which is why the two are asserted to differ in one
        test rather than described in a paragraph.

        Its own declarations first, then each ancestor's going up, and **a
        nearer declaration shadows a farther one of the same name** -- see
        :func:`_inherited_attributes` for why, and for the visited set.

        **An undeclared type is refused, and a type declared with nothing
        returns ``[]``.** Those are answers to different questions and this
        keeps them apart, the way ``contains()`` and the unknown-anchor
        refusal do everywhere else in this family.

        **It reads none of :attr:`structure`**, and that is worth saying
        rather than leaving to be noticed: two taxonomies over one vocabulary
        answer identically here, because the type lattice is neither of their
        relations -- it is the schema both are declared in. The member is on
        the axis because that is the surface a schema projector can reach
        without holding a resolution, which is the whole of the ruling that
        placed it.
        """
        return _inherited_attributes(self.entity_types, self.definition.id, entity_type)

    def at(self, node_id: K) -> TaxonomyView[K]:
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
class AsyncTaxonomy(Generic[K]):
    """The asynchronous twin. Same five fields, asynchronous backings.

    The key is a parameter here too, defaulted to ``str``, for the reason
    :class:`Taxonomy` states. Frozen and identity-compared for the reason it
    states as well -- a difference here would be one
    :func:`assert_twin_types_agree` cannot see, since it reads members and
    these are decisions about the type.
    """

    definition: TaxonomyDefinition
    structure: AsyncHierarchy[K]
    entities: AsyncEntitySource[K]
    assertions: AsyncAssertionSource[K] | None = None

    #: :attr:`Taxonomy.entity_types`, unflavoured -- a mapping awaits nothing.
    entity_types: Mapping[str, EntityType] = field(default_factory=dict)

    async def walk(
        self,
        *,
        from_id: K | None = None,
        max_depth: int | None = None,
        max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
        cache: WalkCache | None = None,
    ) -> AsyncIterator[K]:
        """:meth:`Taxonomy.walk`, awaited.

        The same traversal, and it is duplicated for one reason: a streaming
        member cannot go through the shared collecting core without widening
        that core's request type for every walk that does not stream. The
        frontier read is shared, so this gets the driver's per-level behaviour
        -- one bulk query where the backing offers one, concurrency where it
        does not -- rather than a sequential await per node.

        ``cache`` is that shared step's memo, and means what it means on the
        synchronous twin.
        """
        seen: set[K] = set()
        if from_id is not None:
            if not await self.structure.contains(from_id):
                _refuse_an_unknown_anchor(self.definition.id, from_id)
            frontier: tuple[K, ...] = (from_id,)
        else:
            frontier = tuple(await self.structure.roots())
        depth = 0
        while frontier and (max_depth is None or depth <= max_depth):
            fresh: list[K] = []
            for node_id in frontier:
                if node_id in seen:
                    continue
                seen.add(node_id)
                fresh.append(node_id)
                yield node_id
            if max_depth is not None and depth == max_depth:
                return
            replies = await _async_reply(
                self.structure,
                "children",
                tuple(fresh),
                max_concurrency=max_concurrency,
                cache=cache,
            )
            frontier = tuple(child for reply in replies for child in reply)
            depth += 1

    async def subtree_keys(
        self,
        root_id: K,
        *,
        depth: int | None = None,
        max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
        cache: WalkCache | None = None,
    ) -> list[K]:
        """:meth:`Taxonomy.subtree_keys`, awaited.

        The four properties are that method's; what is twinned here is the
        driving and nothing else.
        """
        if not await self.structure.contains(root_id):
            _refuse_an_unknown_anchor(self.definition.id, root_id)
        if depth is None:
            return list(
                await async_flatten(
                    self.structure,
                    from_id=root_id,
                    max_concurrency=max_concurrency,
                    cache=cache,
                )
            )
        return list(
            await async_descendants_to_depth(
                self.structure,
                root_id,
                depth,
                max_concurrency=max_concurrency,
                cache=cache,
            )
        )

    def inherited_attributes(self, entity_type: str) -> list[AttributeDef]:
        """:meth:`Taxonomy.inherited_attributes`, and a plain ``def``.

        Synchronous on this flavour for :meth:`at`'s reason: the store is a
        mapping the caller is already holding, so there is nothing here to
        await, and making it awaitable would cost every caller an ``await``
        for a walk over their own data. The parity guard compares it as an
        unflavoured member rather than skipping it.
        """
        return _inherited_attributes(self.entity_types, self.definition.id, entity_type)

    def at(self, node_id: K) -> AsyncTaxonomyView[K]:
        """:meth:`Taxonomy.at`, and a plain ``def`` for the same reason.

        Checking containment here would drag an accessor into the loop for a
        question the caller has not asked; ``structure.contains`` is ``async``
        on this flavour. The check lives on the view.
        """
        return AsyncTaxonomyView(self, node_id)


# --------------------------------------------------------------------------
# The anchored view over an axis -- the cursor that can also report an edge
# --------------------------------------------------------------------------


def _written_by_object(written: Sequence[Assertion[K]]) -> dict[K, list[Assertion[K]]]:
    """Edge assertions grouped by the entity their object names, in source order.

    An assertion whose object is a literal names no neighbour and is dropped,
    for the reason the assertion backing gives: ``dog lifespan_years 12`` is a
    value rather than a place in a structure.
    """
    grouped: dict[K, list[Assertion[K]]] = {}
    for assertion in written:
        neighbour = object_entity_id(assertion.object)
        if neighbour is not None:
            grouped.setdefault(neighbour, []).append(assertion)
    return grouped


def _written_by_subject(written: Sequence[Assertion[K]]) -> dict[K, list[Assertion[K]]]:
    """The mirror: edge assertions grouped by subject, in source order."""
    grouped: dict[K, list[Assertion[K]]] = {}
    for assertion in written:
        grouped.setdefault(assertion.subject, []).append(assertion)
    return grouped


@dataclass(frozen=True)
class TaxonomyView(Generic[K]):
    """The cursor over a taxonomy, so it can also report what is written on an edge.

    **Generic in the key, defaulted to ``str``.** It was not, and the sentence
    that said so gave the reason: a taxonomy binds an ``EntitySource`` and an
    ``AssertionSource`` whose keys are the ontology's local entity ids, which
    carry an owner, a space and a qualification rule. That reason is kept and
    moved -- it is a :class:`~dataknobs_common.ontology.values.KeyCodec` the
    ontology holds, so it survives as an object rather than as a pin. A bare
    ``TaxonomyView`` is ``TaxonomyView[str]``.

    **Every structural member invokes the ``HierarchyView`` member of the same
    name over ``taxonomy.structure`` and re-wraps; it does not re-walk.** That
    is exactly the shape a maintainer reimplements without noticing the
    forward is there, so a test patches each ``HierarchyView`` member and
    asserts the member here moves. ``at`` is the one exception: it constructs
    a ``TaxonomyView``, which no ``HierarchyView`` member can return, so it
    forwards to nothing.

    **Three members are the reason this class exists over ``HierarchyView``**,
    and they read the two axes a bare ``Hierarchy`` does not carry:
    :meth:`entity` reads the *content* axis, and the two edge members read the
    *assertions* the edges were made of.

    ``paths_to_root`` is the one walk-shaped member that does not re-wrap: it
    answers with routes, whose meaning is their order, so it hands back keys
    for the reason its own docstring gives.

    **The axis is compared by identity, and that is what makes this hashable.**
    A frozen dataclass hashes its field tuple, so a cursor can only hash if the
    axis it holds can. Three of a taxonomy's five fields cannot be made to:
    ``definition`` carries a ``metadata`` dict, a ``MappingHierarchy``
    structure carries two more, and ``entity_types`` is a ``Mapping``, so
    freezing them would leave ``__hash__`` generated and still raising --
    promising the capability at ``isinstance`` and failing at the call. Identity closes that for every backing at once,
    and it costs nothing that was being used: an axis is *built* by
    :meth:`Ontology.taxonomy` from a definition, and it is the definition that
    this module calls the value.

    So two cursors are equal exactly when they name the same node **of the same
    built axis**, which is the sentence this class was already documented by.
    """

    taxonomy: Taxonomy[K]
    node: K

    def _structural(self) -> HierarchyView[K]:
        """The cursor every structural member forwards to."""
        return HierarchyView(self.taxonomy.structure, self.node)

    def _wrap(self, views: tuple[HierarchyView[K], ...]) -> tuple[TaxonomyView[K], ...]:
        return tuple(TaxonomyView(self.taxonomy, view.node) for view in views)

    def entity(self) -> Entity[K] | None:
        """What this node **is**, by way of the content axis. ``None`` if it carries none.

        The member the cursor over a taxonomy has and the cursor over a bare
        structure cannot: an axis is two axes, and this reads the second.

        **``None`` is a state rather than an error, and it is not
        :meth:`exists`' answer.** ``exists()`` asks the *structure*; this asks
        the *content*, and the four combinations are all reachable. A node in
        the structure with no entity behind it is ordinary under a live
        backing -- an axis built from a parent-id column knows an id the entity
        store has not been given -- so the two questions are asked separately
        and answered separately. A caller wanting *is this node here at all*
        asks ``exists()``; one wanting *what is it* asks this and reads
        ``None`` as **nothing is written about it**.

        An accessor, in the sense :meth:`Ontology.entity` states: it invokes
        the one lookup rather than reproducing it.
        """
        return self.taxonomy.entities.get(self.node)

    def exists(self) -> bool:
        """Whether the structure axis knows this node at all."""
        return self._structural().exists()

    def is_root(self) -> bool:
        """Present, with nothing above it. ``False`` for an absent node."""
        return self._structural().is_root()

    def is_leaf(self) -> bool:
        """Present, with nothing below it. ``False`` for an absent node."""
        return self._structural().is_leaf()

    def parents(self) -> tuple[TaxonomyView[K], ...]:
        """One view per node directly above this one. Plural, always."""
        return self._wrap(self._structural().parents())

    def children(self) -> tuple[TaxonomyView[K], ...]:
        """One view per node directly below this one."""
        return self._wrap(self._structural().children())

    def ancestors(self, *, cache: WalkCache | None = None) -> tuple[TaxonomyView[K], ...]:
        """:meth:`~dataknobs_common.hierarchy.HierarchyView.ancestors`, re-wrapped.

        The structural member, not a second walk: it invokes the cursor over
        ``taxonomy.structure`` and re-anchors each answer on this axis, so a
        caller who walks up can keep asking what a node *is*.
        """
        return self._wrap(self._structural().ancestors(cache=cache))

    def descendants(self, *, cache: WalkCache | None = None) -> tuple[TaxonomyView[K], ...]:
        """:meth:`~dataknobs_common.hierarchy.HierarchyView.descendants`, re-wrapped."""
        return self._wrap(self._structural().descendants(cache=cache))

    def descendants_to_depth(
        self, max_depth: int, *, cache: WalkCache | None = None
    ) -> tuple[TaxonomyView[K], ...]:
        """:meth:`~dataknobs_common.hierarchy.HierarchyView.descendants_to_depth`, re-wrapped.

        Includes this node and refuses one the structure axis does not contain,
        both inherited. :meth:`Taxonomy.subtree_keys` answers the same question
        as *keys* and refuses with the taxonomy's own message; this answers it
        as cursors and refuses with the structure's.
        """
        return self._wrap(self._structural().descendants_to_depth(max_depth, cache=cache))

    def children_at_depth(
        self, depth: int, *, cache: WalkCache | None = None
    ) -> tuple[TaxonomyView[K], ...]:
        """:meth:`~dataknobs_common.hierarchy.HierarchyView.children_at_depth`, re-wrapped.

        One level rather than a span, and the anchor means *where you are*.
        Both are the structural member's, inherited rather than restated.
        """
        return self._wrap(self._structural().children_at_depth(depth, cache=cache))

    def paths_to_root(
        self, *, max_paths: int | None = None, cache: WalkCache | None = None
    ) -> tuple[tuple[K, ...], ...]:
        """:meth:`~dataknobs_common.hierarchy.HierarchyView.paths_to_root` -- keys, not cursors.

        The one walk-shaped member here that does not re-wrap, for the reason
        the cursor below it gives: a path's meaning is its order.
        """
        return self._structural().paths_to_root(max_paths=max_paths, cache=cache)

    def at(self, node_id: K) -> TaxonomyView[K]:
        """Re-anchor at another node of the same axis. Constructs; checks nothing."""
        return TaxonomyView(self.taxonomy, node_id)

    def parent_edges(self) -> tuple[tuple[TaxonomyView[K], Assertion[K]], ...]:
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

    def child_edges(self) -> tuple[tuple[TaxonomyView[K], Assertion[K]], ...]:
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
class AsyncTaxonomyView(Generic[K]):
    """The twin, over an :class:`AsyncTaxonomy`.

    The structural members invoke :class:`~dataknobs_common.hierarchy.AsyncHierarchyView`,
    :meth:`entity` reads an ``AsyncEntitySource`` and the edge members read an
    ``AsyncAssertionSource``; every one is ``async def`` bar :meth:`at`, which
    constructs.

    The walk-shaped members each also carry ``max_concurrency``, which the
    cursor below them carries and the synchronous flavour has no equivalent
    of.
    """

    taxonomy: AsyncTaxonomy[K]
    node: K

    def _structural(self) -> AsyncHierarchyView[K]:
        """The cursor every structural member forwards to."""
        return AsyncHierarchyView(self.taxonomy.structure, self.node)

    def _wrap(self, views: tuple[AsyncHierarchyView[K], ...]) -> tuple[AsyncTaxonomyView[K], ...]:
        return tuple(AsyncTaxonomyView(self.taxonomy, view.node) for view in views)

    async def entity(self) -> Entity[K] | None:
        """:meth:`TaxonomyView.entity`, awaited."""
        return await self.taxonomy.entities.get(self.node)

    async def exists(self) -> bool:
        """Whether the structure axis knows this node at all."""
        return await self._structural().exists()

    async def is_root(self) -> bool:
        """Present, with nothing above it. ``False`` for an absent node."""
        return await self._structural().is_root()

    async def is_leaf(self) -> bool:
        """Present, with nothing below it. ``False`` for an absent node."""
        return await self._structural().is_leaf()

    async def parents(self) -> tuple[AsyncTaxonomyView[K], ...]:
        """One view per node directly above this one. Plural, always."""
        return self._wrap(await self._structural().parents())

    async def children(self) -> tuple[AsyncTaxonomyView[K], ...]:
        """One view per node directly below this one."""
        return self._wrap(await self._structural().children())

    async def ancestors(
        self,
        *,
        max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
        cache: WalkCache | None = None,
    ) -> tuple[AsyncTaxonomyView[K], ...]:
        """:meth:`TaxonomyView.ancestors`, awaited."""
        return self._wrap(
            await self._structural().ancestors(max_concurrency=max_concurrency, cache=cache)
        )

    async def descendants(
        self,
        *,
        max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
        cache: WalkCache | None = None,
    ) -> tuple[AsyncTaxonomyView[K], ...]:
        """:meth:`TaxonomyView.descendants`, awaited."""
        return self._wrap(
            await self._structural().descendants(max_concurrency=max_concurrency, cache=cache)
        )

    async def descendants_to_depth(
        self,
        max_depth: int,
        *,
        max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
        cache: WalkCache | None = None,
    ) -> tuple[AsyncTaxonomyView[K], ...]:
        """:meth:`TaxonomyView.descendants_to_depth`, awaited."""
        return self._wrap(
            await self._structural().descendants_to_depth(
                max_depth, max_concurrency=max_concurrency, cache=cache
            )
        )

    async def children_at_depth(
        self,
        depth: int,
        *,
        max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
        cache: WalkCache | None = None,
    ) -> tuple[AsyncTaxonomyView[K], ...]:
        """:meth:`TaxonomyView.children_at_depth`, awaited."""
        return self._wrap(
            await self._structural().children_at_depth(
                depth, max_concurrency=max_concurrency, cache=cache
            )
        )

    async def paths_to_root(
        self,
        *,
        max_paths: int | None = None,
        max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
        cache: WalkCache | None = None,
    ) -> tuple[tuple[K, ...], ...]:
        """:meth:`TaxonomyView.paths_to_root`, awaited -- keys, not cursors.

        ``max_concurrency`` bounds the **ascent** and ``max_paths`` the
        **answer**, for the reason the cursor below it states. Both are
        forwarded.
        """
        return await self._structural().paths_to_root(
            max_paths=max_paths, max_concurrency=max_concurrency, cache=cache
        )

    def at(self, node_id: K) -> AsyncTaxonomyView[K]:
        """Re-anchor at another node of the same axis. Constructs; checks nothing."""
        return AsyncTaxonomyView(self.taxonomy, node_id)

    async def parent_edges(self) -> tuple[tuple[AsyncTaxonomyView[K], Assertion[K]], ...]:
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

    async def child_edges(self) -> tuple[tuple[AsyncTaxonomyView[K], Assertion[K]], ...]:
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
