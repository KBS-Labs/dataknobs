# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""What a *set* of rows is about, rolled up onto the vocabulary that placed them.

The reading direction is the whole of it. A row already knows which entities it
is about --- that is what its tags say, and reading them needs no vocabulary at
all. This asks the next question: given a page of such rows, **which entities do
they support, and on what evidence?**

Two levels, one measure. :func:`roll_up` groups a corpus's tags by *node*, within
one axis of one vocabulary, and answers a :class:`SupportSet`.
:func:`ontology_support` groups the same tags by *vocabulary* and answers one
entry per vocabulary the rows touched. They are the same counting at two
heights, which is why they are written beside each other rather than one in each
package that happens to want one.

**Pure over a vocabulary and a sequence of tags; opens nothing.** Nothing here
imports anything outside the standard library and this package, so every
question it answers is askable with no store, no embedder and no event loop ---
the same rule the tag reader next door keeps, one level up.

**Two flavours of the roll-up, and one of the vocabulary count.**
:func:`roll_up` reads a :class:`~dataknobs_common.ontology.values.Ontology` and
:func:`async_roll_up` an
:class:`~dataknobs_common.ontology.values.AsyncOntology`, because the roll-up
*does* have something to await: whether an axis carries a node, and what stands
above it, are both reads, and on the asynchronous flavour both are ``async
def``. That flavour is the one a by-reference backing needs --- an axis over
rows is asynchronous --- so without the twin the operation is unreachable from
the vocabularies it was designed around. Everything that is not an ``await`` is
shared: :func:`_grouped` counts, :func:`_assemble` builds the answer, and what
is twinned is the driving. :func:`ontology_support` has no twin because it
genuinely has nothing to await --- it reads the tags' own ``ontology_id`` and
holds no vocabulary at all.

**The rows are referred to by position**, because the caller passed a sequence
and the position is the only reference this module is entitled to invent. A
generator has no positions to report, so the argument is a ``Sequence``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from dataknobs_common._walk_core import _refuse_an_unusable_bound
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.hierarchy import DEFAULT_FRONTIER_CONCURRENCY, K

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from dataknobs_common._walk_core import WalkCache, WalkCacheKey
    from dataknobs_common.ontology.tags import NodeTag
    from dataknobs_common.ontology.values import AsyncOntology, Ontology

__all__ = [
    "Granularity",
    "NodeSupport",
    "OntologySupport",
    "SupportSet",
    "async_roll_up",
    "ontology_support",
    "roll_up",
]

#: What :func:`_grouped` groups by --- a vocabulary id at one height and a node
#: key at the other.
_G = TypeVar("_G")


class Granularity(Enum):
    """Which of several supported entities to present.

    **The vocabulary is reused; the enum is new.** The four words were chosen
    in a review before anything shipped them, so what is inherited here is the
    naming decision --- *deepest* is what :attr:`MOST_SPECIFIC` means --- rather
    than a type being moved. No other ``Granularity`` exists to migrate from.

    **This is therefore the enum's first shipped home**, and a later reader
    imports it from here rather than the enum moving. Moving a published
    vocabulary is a migration, and the point of putting it here first is to not
    owe one.
    """

    MOST_SPECIFIC = "most_specific"
    """Keep the entries nothing more specific was named for."""

    MOST_GENERAL = "most_general"
    """Keep the entries nothing in the set stands above."""

    AT_TYPE = "at_type"
    """Project onto the entity *type*. Refused --- see :meth:`SupportSet.prune`."""

    ALL = "all"
    """Keep everything, ranked. The identity of the projection."""


@dataclass(frozen=True)
class NodeSupport(Generic[K]):
    """One node a set of rows supports, and which rows support it."""

    node_id: K
    """In the **ontology's** key space, already localized.

    The tag carries a rendered id and
    :meth:`~dataknobs_common.ontology.values.Ontology.localize` is the only door
    that turns one into a key while refusing an id belonging to another
    vocabulary. The roll-up goes through it, so a caller never does --- which is
    why a caller spends this on ``axis.at(...)`` directly rather than localizing
    first.
    """

    rows: tuple[int, ...]
    """Positions in the sequence the caller passed, ascending, without duplicates.

    **Positional because the position is the only honest reference.** The row is
    recoverable from its position by indexing and the position is never
    recoverable from the row, because two rows with the same content have one
    content and two places. The corpus is the consumer's, so there is no row id
    this module is entitled to invent.

    The caller must therefore pass a ``Sequence``. A generator has no positions
    to report.
    """

    above: tuple[K, ...] = ()
    """The nodes **of this same set** standing above this one, nearest first.

    Restricted to the set, and the first clause of this sentence is the whole of
    it: this is not ``ancestors()``. A parent appears here when some row named
    it; an ancestor no row named does not, however near it stands.

    It is carried rather than derived because deriving it is a hierarchy walk,
    which is I/O, and :meth:`SupportSet.prune` is a member on a value. It is also
    the half a caller cannot cheaply reconstruct, which is the test a value's
    fields are earned by.

    **Empty for every entry in** :attr:`SupportSet.unplaced`, and that is a
    property of the field rather than of those nodes: the walk runs over the
    axis, and an unplaced node is one the axis does not carry, so there was
    nothing to walk. A caller reading ``above`` off an entry needs to know which
    of the two collections it came out of; within :attr:`SupportSet.supported`
    an empty tuple means what this docstring's first line says.
    """


@dataclass(frozen=True)
class SupportSet(Generic[K]):
    """What one axis of one vocabulary makes of a set of rows.

    **Everything the rows supported, and the projection is a reader over it.**
    Pruning changes what is *presented*, never what is *reachable*, so the record
    and the presentation are two members rather than one answer.
    """

    ontology_id: str
    """Which vocabulary this is an answer about."""

    taxonomy_id: str
    """Which axis of it.

    A deployment holding two vocabularies gets two of these, and a caller
    holding two has the names rather than a convention.
    """

    supported: tuple[NodeSupport[K], ...] = ()
    """Every node the rows named that this axis carries, in the order the rows named them.

    **The record, not the presentation.** First-seen order. What to put in front
    of a person is :meth:`prune`'s answer, and the two are deliberately different
    members.
    """

    unplaced: tuple[NodeSupport[K], ...] = ()
    """Nodes this vocabulary's tags named and this axis does **not** carry.

    Reported rather than refused, on the split between what is *operational* and
    what is *malformed* --- and this is the operational half: a corpus tagged
    before a vocabulary was reorganised is the ordinary case, not a bug, and the
    corpus belongs to a consumer, so it is not even a bug anybody here can go and
    fix.

    **Operational, and only operational.** A node id that is not a string is
    refused by the read one frame earlier, and a local id this vocabulary's key
    space cannot parse never becomes a key at all --- its row lands in
    :attr:`unsupported_rows` instead. So this field's population is what its
    first paragraph says it is rather than a mixture: nodes, in this
    vocabulary's key space, that this axis does not carry.

    It carries the rows, not just the ids, because the rows are what a maintainer
    needs in order to find the tagger.
    """

    unsupported_rows: tuple[int, ...] = ()
    """Positions that contributed to no entry in :attr:`supported`.

    A row carrying no tag, a row carrying a tag for another vocabulary or another
    axis, a row whose only tag was unplaced, and a row whose only tag named a
    local id this vocabulary's :class:`~dataknobs_common.ontology.values.KeyCodec`
    could not read.

    **A report, not a verdict**: an answer drawn from five rows of which three
    touched nothing is exactly the case a consumer must be able to see. It is
    derivable, and it is kept because the caller who has to derive it derives it
    every time.

    The four causes are deliberately **not** distinguished. The caller holds the
    tags, so *no tag* and *another vocabulary's tag* are one lookup apart for
    them --- and so is an unreadable id, which the caller can reproduce by
    spending :meth:`~dataknobs_common.ontology.values.Ontology.localize` on the
    row's own tags. Which vocabularies are in play at all is
    :func:`ontology_support`'s question.
    """

    def _strictly_above(self) -> dict[K, tuple[K, ...]]:
        """:attr:`NodeSupport.above` with mutual pairs removed, per supported node.

        *Strictly* above: a node that this one also stands above is at the same
        height rather than over it, so it is dropped from both readings. Over an
        acyclic axis nothing is dropped and this is :attr:`NodeSupport.above`
        unchanged, which is why the two policies can read it unconditionally
        instead of testing for a cycle they usually do not have.

        Computed here rather than carried on :class:`NodeSupport`, because it is
        a property of the *set* --- which pairs happen to be mutual within it ---
        and reading it costs a pass over entries already in hand.
        """
        above_of = {entry.node_id: frozenset(entry.above) for entry in self.supported}
        return {
            entry.node_id: tuple(
                node for node in entry.above if entry.node_id not in above_of.get(node, frozenset())
            )
            for entry in self.supported
        }

    def prune(self, policy: Granularity = Granularity.MOST_SPECIFIC) -> tuple[NodeSupport[K], ...]:
        """The entries to present under ``policy``, ranked.

        **One member, because there is one operation.** :attr:`Granularity.ALL`
        is the identity, so a caller wanting everything ranked asks for it here
        rather than through a second door.

        **The ranking is by row support, descending, ties in first-seen order**,
        and the reason is worth stating because the word a caller arrives with is
        *ranked by specificity*: after :attr:`Granularity.MOST_SPECIFIC` the
        survivors are mutually incomparable by construction, so specificity
        cannot order them. The policy decides membership; the support decides
        order.

        **Mutual ancestry is not a specificity relation**, and both policies read
        :attr:`NodeSupport.above` with that subtracted. A cyclic axis loads ---
        nothing refuses a document declaring ``A isa B`` alongside ``B isa A`` ---
        so a support set in which each of two nodes stands above the other is
        reachable from a valid vocabulary. Read literally, every projection then
        empties: each is an ancestor of something in the set, and neither has an
        empty ``above``. A page holding evidence for two entities would present
        as a page about nothing, which is the one answer that is certainly wrong,
        because :attr:`supported` plainly holds them. So a pair standing above
        each other is read as what it is --- the same height --- and the ranking,
        which always exists, decides the order. :attr:`NodeSupport.above` is
        unchanged: it is the record of what the axis says.

        **It survives everything.** There is no scoring anywhere in this family:
        the number being ordered is a count of rows, and a count is comparable to
        another count whatever produced either.

        **It performs no I/O.** :attr:`Granularity.MOST_SPECIFIC` and
        :attr:`Granularity.MOST_GENERAL` are computed from
        :attr:`NodeSupport.above` alone, and :attr:`Granularity.ALL` from
        nothing.

        Raises:
            ValidationError: For :attr:`Granularity.AT_TYPE`, naming the entity
                source it would need. Projecting to a type reads the type of a
                node that may not be in this set, which is a source read, and
                this value holds no source.
        """
        if policy is Granularity.AT_TYPE:
            raise ValidationError(
                f"projecting the support for {self.ontology_id}:{self.taxonomy_id} onto "
                f"entity types needs the vocabulary's entity source, which this value "
                f"does not hold: a support set carries node ids and row positions, and "
                f"reading what a node IS is a source read. Ask the ontology for the "
                f"entity of each node instead"
            )

        strictly = self._strictly_above()
        if policy is Granularity.MOST_GENERAL:
            kept = [entry for entry in self.supported if not strictly[entry.node_id]]
        elif policy is Granularity.MOST_SPECIFIC:
            covered = {above for aboves in strictly.values() for above in aboves}
            kept = [entry for entry in self.supported if entry.node_id not in covered]
        else:
            kept = list(self.supported)

        order = {entry.node_id: position for position, entry in enumerate(self.supported)}
        return tuple(sorted(kept, key=lambda entry: (-len(entry.rows), order[entry.node_id])))


@dataclass(frozen=True)
class OntologySupport:
    """One vocabulary a set of rows touches, and which rows touch it.

    Declared beside :class:`NodeSupport` so that the two levels are one shape
    rather than two conventions, and **kept a separate type** rather than
    generalised over one. ``ontology_id`` and ``node_id`` are different id
    spaces, and two id spaces in one dataclass need the names to carry the
    difference. A generic pair would let a caller pass either where either is
    expected, which is the one mistake the names exist to prevent.

    **One producer today**, which is :func:`ontology_support`. The shape is the
    one a registry narrowing this answer to the vocabularies it holds would
    report in, and it is published from here so that such a narrowing is a
    filter over this module's answer rather than a second count --- but nothing
    outside this package produces one yet, and this sentence says so rather than
    describing an integration as though it existed.
    """

    ontology_id: str
    """Which vocabulary."""

    rows: tuple[int, ...]
    """Positions that named it, ascending, without duplicates --- :attr:`NodeSupport.rows`
    one level up, and positional for its reason."""


@dataclass(frozen=True)
class _AxisScopedCache:
    """A caller's cache, keyed so that one of them may serve several axes.

    **A walk cache carries no hierarchy in its key and cannot**, because a
    backing is arbitrary consumer code that need not be hashable and ``id()`` is
    reused after a collection. The consequence is stated on
    :class:`~dataknobs_common._walk_core.WalkCache` as *one cache, one axis*, and
    it is left to the caller there for a good reason: a walk driver takes the
    hierarchy itself, so the caller is the only party that knows how many of them
    they hold.

    **That reasoning does not reach this module.** :func:`roll_up` takes the axis
    *by name* off a vocabulary it was handed, so the party holding the
    discriminator is this function rather than its caller --- and a caller with
    one vocabulary and two axis names holds one object and would naturally spend
    one cache on both. Scoping here makes that call shape correct instead of
    silently wrong, which is the only kind of wrong it could otherwise be: the
    second axis would be answered from the first's edges with no exception
    anywhere.

    The cost is one sentence of documented behaviour: entries this puts in a
    caller's cache are not the entries a bare
    :func:`~dataknobs_common.hierarchy.deepest_common_ancestor` over the same
    axis would read, because the keys are scoped and its are not. A cache is a
    cost optimisation whose entries no caller inspects, so what is traded is
    sharing between two functions for correctness across two axes.
    """

    inner: WalkCache
    """The caller's own cache, which receives the scoped keys."""

    scope: str
    """The axis name every key written through here is qualified by."""

    def get(self, key: WalkCacheKey, default: None = None) -> Sequence[Any] | None:
        """:meth:`~dataknobs_common._walk_core.WalkCache.get`, under the scoped key."""
        member, node = key
        return self.inner.get((member, (self.scope, node)), default)

    def __setitem__(self, key: WalkCacheKey, value: Sequence[Any]) -> None:
        """:meth:`~dataknobs_common._walk_core.WalkCache.__setitem__`, under the scoped key."""
        member, node = key
        self.inner[(member, (self.scope, node))] = value


def _grouped(
    tagged: Sequence[Sequence[NodeTag]],
    key: Callable[[NodeTag], tuple[_G, ...]],
) -> dict[_G, list[int]]:
    """Positions grouped by ``key``, in first-seen order, ascending and deduplicated.

    **The one counting loop in this module**, because there is one measure. The
    two public answers differ in what they group *by* --- a vocabulary id at one
    height and a node key at the other --- and in nothing else: both want each
    group's rows, ascending, without duplicates, with the groups in the order the
    rows first named them. Written twice, the two drift on exactly those
    properties, which is the class of difference no assertion about either alone
    would catch.

    ``key`` answers the groups a tag belongs to --- one, or none for a tag this
    grouping is not about, which is how :func:`roll_up` filters another
    vocabulary's tags and another axis's without a second pass. It answers a
    tuple rather than an optional key because the key space is the consumer's:
    ``None`` is a value a consumer's ``K`` may legitimately take, so a
    ``None``-means-absent convention would be unable to say which of the two a
    key of ``None`` was.

    **The deduplication is against the last position rather than the whole
    list**, which is equivalence rather than optimisation: ``position`` ascends,
    so a repeat can only be the entry just appended. The scan it replaces was
    quadratic in the rows naming one group --- a page of twenty thousand hits all
    naming one node is the ordinary case for a popular entity, and that is the
    case it was slowest on.
    """
    grouped: dict[_G, list[int]] = {}
    for position, row in enumerate(tagged):
        for tag in row:
            for group in key(tag):
                rows = grouped.setdefault(group, [])
                if not rows or rows[-1] != position:
                    rows.append(position)
    return grouped


def _localizer(
    ontology: Ontology[K] | AsyncOntology[K], taxonomy_id: str
) -> Callable[[NodeTag], tuple[K, ...]]:
    """:func:`_grouped`'s key for a roll-up: this call's tags, localized.

    Three dispositions, and which one applies is decided here rather than by the
    caller. A tag naming another vocabulary or another axis is **not this call's**
    and is dropped without reaching
    :meth:`~dataknobs_common.ontology.values.Ontology.localize`, so that door's
    refusal stays available to a caller who hands it one directly. A tag whose
    local id this vocabulary's codec cannot read is **dropped too**, and the
    difference is worth stating: ``localize`` lets a codec's own failure through
    unwrapped, by a documented contract, so over a non-``str`` key space one row
    tagged before the key space changed would otherwise raise out of the whole
    operation --- no supported set, no unplaced, no residue, over one row in a
    corpus the caller did not write and cannot fix. That is the same operational
    class ``unplaced`` exists for, disposed of the way
    :attr:`SupportSet.unsupported_rows` disposes of the others.

    The refusal is caught broadly because the contract it is complementing is
    broad: a codec is a consumer's function and ``from_id`` may raise whatever
    it likes. ``BaseException`` still passes, so a cancellation or an interrupt
    is not swallowed.
    """

    def _key(tag: NodeTag) -> tuple[K, ...]:
        if tag.ontology_id != ontology.id or tag.taxonomy_id != taxonomy_id:
            return ()
        try:
            return (ontology.localize(tag.qualified_id),)
        # A codec is a consumer's function and ``from_id`` may raise whatever it
        # likes, so the catch is as broad as the contract it complements.
        # ``BaseException`` still passes: a cancellation is not a bad id.
        except Exception:
            return ()

    return _key


def _assemble(
    ontology_id: str,
    taxonomy_id: str,
    named: Mapping[K, list[int]],
    above: Mapping[K, tuple[K, ...]],
    total: int,
) -> SupportSet[K]:
    """The answer, built from a grouping and the axis's two replies about it.

    The half of a roll-up that awaits nothing, so both flavours end here. The
    split between the collections is ``above``'s membership: a node the axis
    carries has an entry there --- possibly an empty one --- and a node it does
    not carry has none.

    First-seen order survives into both collections because ``named`` is in it
    and this filters rather than re-sorts.
    """
    supported = tuple(
        NodeSupport(node_id=node, rows=tuple(rows), above=above[node])
        for node, rows in named.items()
        if node in above
    )
    unplaced = tuple(
        NodeSupport(node_id=node, rows=tuple(rows))
        for node, rows in named.items()
        if node not in above
    )
    contributing = {position for entry in supported for position in entry.rows}
    return SupportSet(
        ontology_id=ontology_id,
        taxonomy_id=taxonomy_id,
        supported=supported,
        unplaced=unplaced,
        unsupported_rows=tuple(
            position for position in range(total) if position not in contributing
        ),
    )


def roll_up(
    ontology: Ontology[K],
    taxonomy_id: str,
    tagged: Sequence[Sequence[NodeTag]],
    *,
    cache: WalkCache | None = None,
) -> SupportSet[K]:
    """Which entities of one axis a set of rows supports, and on what evidence.

    **Not** ``deepest_common_ancestor``. That walk is pairwise and, folded over
    the *n* entities a row set supports, **generalises**: what comes back stands
    above every one of them and is frequently an entity no row named. This
    **selects** --- of the entities the rows named, keep the ones nothing more
    specific was named for. Over one vocabulary the two return opposite answers.

    **The axis is asked about each node once**, not once per tag that named it.
    Membership is a question about a node and takes no cache, so a page of
    twenty thousand hits naming one popular entity is one round trip rather than
    twenty thousand. It also makes the split between :attr:`SupportSet.supported`
    and :attr:`SupportSet.unplaced` a reading of the axis at one instant: asked
    per occurrence, a backing that changed mid-call could put one node in both.

    Args:
        ontology: The vocabulary to roll up for. Needed rather than a bare axis
            because :meth:`~dataknobs_common.ontology.values.Ontology.localize`
            is the only door that turns a tag's rendered id into a key.
        taxonomy_id: Which axis. A tag carries one; a vocabulary may declare
            several, and a roll-up answers about one.

            **This one is refused where a tag's is filtered.** It is the
            caller's value, so a name this vocabulary does not declare is a name
            somebody typed, and
            :meth:`~dataknobs_common.ontology.values.Ontology.taxonomy` refuses
            it --- see Raises below. A tag's axis id is a third party's and is
            filtered, one line down. The two are disposed of differently because
            of whose they are.
        tagged: One row's tags per position, as
            :func:`~dataknobs_common.ontology.tags.read_node_tags` answered. A
            ``Sequence``: positions are the row references. Tags naming another
            ontology or another axis are not this call's, and their rows land in
            :attr:`SupportSet.unsupported_rows`. ``localize`` is never called on
            one --- the filter runs first, so its refusal stays the door's for a
            caller who hands it one directly. A tag whose local id this
            vocabulary's codec cannot read lands there too --- see
            :func:`_localizer`.
        cache: The walk cache, forwarded. One is used within a call whether or
            not one is given, because the *n* ascents share a frontier.

            **Safe to spend across axes**, which is this function's own doing
            rather than the cache's: a walk cache carries no hierarchy in its
            key, so one handed to two axes would otherwise answer the second
            from the first's edges. This call holds the axis name, so it scopes
            what it forwards --- see :class:`_AxisScopedCache`, which also states
            what that trades away.

    Returns:
        A :class:`SupportSet`. Empty in every field for an empty input, which is
        an answer rather than an error --- a set of rows that touched no
        vocabulary is the ordinary majority of a corpus.

    Raises:
        NotFoundError: Naming ``taxonomy_id`` and the axes this vocabulary does
            declare, where it declares no such axis. Reached through
            :meth:`~dataknobs_common.ontology.values.Ontology.taxonomy`, on the
            **caller's** value. A *tag* naming an undeclared axis raises
            nothing: its row lands in :attr:`SupportSet.unsupported_rows`,
            indistinguishably from a tag naming a declared axis this call was
            not asked about, because the filter compares against ``taxonomy_id``
            and asks the vocabulary nothing.
        ValidationError: From
            :meth:`~dataknobs_common.ontology.values.Ontology.taxonomy`, where
            ``taxonomy_id`` names an axis this vocabulary *does* declare whose
            ``materialization`` it cannot supply. The vocabulary author's
            mistake rather than the caller's or the corpus's, and named here
            because a caller reading this clause is entitled to know the
            accessor has two refusals and not one.
    """
    axis = ontology.taxonomy(taxonomy_id)
    walked: WalkCache = {} if cache is None else _AxisScopedCache(cache, taxonomy_id)

    named = _grouped(tagged, _localizer(ontology, taxonomy_id))
    placed = [node for node in named if axis.at(node).exists()]
    within = set(placed)
    above = {
        node: tuple(
            view.node for view in axis.at(node).ancestors(cache=walked) if view.node in within
        )
        for node in placed
    }
    return _assemble(ontology.id, taxonomy_id, named, above, len(tagged))


async def async_roll_up(
    ontology: AsyncOntology[K],
    taxonomy_id: str,
    tagged: Sequence[Sequence[NodeTag]],
    *,
    max_concurrency: int = DEFAULT_FRONTIER_CONCURRENCY,
    cache: WalkCache | None = None,
) -> SupportSet[K]:
    """:func:`roll_up` over an asynchronous vocabulary.

    **The flavour a by-reference backing needs.** An axis over rows is an
    :class:`~dataknobs_common.hierarchy.AsyncHierarchy`, so a vocabulary whose
    structure is read rather than authored arrives as an
    :class:`~dataknobs_common.ontology.values.AsyncOntology` and can reach the
    roll-up only through here. Everything that is not an ``await`` is shared with
    the synchronous half --- the grouping, the localizing, the cache scoping and
    the assembly --- so what is twinned is the driving and nothing else.

    ``localize`` and ``taxonomy`` are unflavoured on both halves, which is why
    only the two axis reads differ.

    Args:
        ontology: The asynchronous vocabulary to roll up for.
        taxonomy_id: :func:`roll_up`'s, unchanged.
        tagged: :func:`roll_up`'s, unchanged. Still a ``Sequence``, and still
            already read: the tags are the caller's and reading them awaits
            nothing.
        max_concurrency: How many membership questions may be in flight at once.
            The frontier bound the walks take, applied to the one read that is
            not a walk --- ``exists()`` is per node and takes no cache, so a page
            naming many distinct nodes is where the bound earns its keep.
            Refused below one, never clamped to it.
        cache: :func:`roll_up`'s, scoped the same way.

    Returns:
        A :class:`SupportSet`, equal to what :func:`roll_up` answers over an
        equivalent synchronous vocabulary.

    Raises:
        NotFoundError: :func:`roll_up`'s.
        ValidationError: :func:`roll_up`'s, and for a ``max_concurrency`` below
            one, which admits nobody rather than merely being small.
    """
    _refuse_an_unusable_bound(max_concurrency, "max_concurrency")
    axis = ontology.taxonomy(taxonomy_id)
    walked: WalkCache = {} if cache is None else _AxisScopedCache(cache, taxonomy_id)

    named = _grouped(tagged, _localizer(ontology, taxonomy_id))
    limit = asyncio.Semaphore(max_concurrency)

    async def _carried(node: K) -> bool:
        async with limit:
            return await axis.at(node).exists()

    carried = await asyncio.gather(*(_carried(node) for node in named))
    placed = [node for node, exists in zip(named, carried, strict=True) if exists]
    within = set(placed)

    above: dict[K, tuple[K, ...]] = {}
    for node in placed:
        ascent = await axis.at(node).ancestors(max_concurrency=max_concurrency, cache=walked)
        above[node] = tuple(view.node for view in ascent if view.node in within)

    return _assemble(ontology.id, taxonomy_id, named, above, len(tagged))


def ontology_support(tagged: Sequence[Sequence[NodeTag]]) -> tuple[OntologySupport, ...]:
    """Which vocabularies a set of rows touches, ranked by how many rows named each.

    :func:`roll_up` one level up: the same corpus, the same evidence and the same
    measure, grouped by *vocabulary* rather than by node. Descending by row
    count, ties in first-seen order --- the ranking
    :meth:`SupportSet.prune` uses, for its reason: the number being ordered is a
    count of rows, and there is nothing else here to order by.

    **It holds no vocabulary and loads none.** It reads the tags' own
    ``ontology_id`` and counts, so it answers over vocabularies the caller has
    never loaded and over ones nobody holds --- which is what a caller with a
    page of results and no idea what is on it actually has. A registry narrowing
    this to the vocabularies it holds is a filter over this answer rather than a
    second count, which keeps one implementation of the measure.

    **No asynchronous twin, and here the reason holds**: the tags are the
    caller's and their ``ontology_id`` is a field, so there is nothing to await.
    :func:`roll_up` has one because it reads an axis; this reads a string.

    Args:
        tagged: One row's tags per position, as
            :func:`~dataknobs_common.ontology.tags.read_node_tags` answered.
            A ``Sequence``, for :attr:`NodeSupport.rows`' reason.

    Returns:
        One :class:`OntologySupport` per vocabulary any tag named, ranked. Empty
        for a corpus whose rows touched none, which is an answer rather than an
        error.
    """
    touched = _grouped(tagged, lambda tag: (tag.ontology_id,))
    order = {ontology_id: position for position, ontology_id in enumerate(touched)}
    entries = [
        OntologySupport(ontology_id=ontology_id, rows=tuple(rows))
        for ontology_id, rows in touched.items()
    ]
    return tuple(sorted(entries, key=lambda entry: (-len(entry.rows), order[entry.ontology_id])))
