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
the same rule the tag reader next door keeps, one level up. There is no
asynchronous twin because there is nothing to await.

**The rows are referred to by position**, because the caller passed a sequence
and the position is the only reference this module is entitled to invent. A
generator has no positions to report, so the argument is a ``Sequence``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Generic

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.hierarchy import K

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataknobs_common._walk_core import WalkCache
    from dataknobs_common.ontology.tags import NodeTag
    from dataknobs_common.ontology.values import Ontology

__all__ = [
    "Granularity",
    "NodeSupport",
    "OntologySupport",
    "SupportSet",
    "ontology_support",
    "roll_up",
]


class Granularity(Enum):
    """Which of several supported entities to present.

    **Reused, not re-minted.** This projection was reified once so that a
    decision made once in a review is then made once, permanently, and the word
    it chose is the word this operation needs: *deepest* is what
    :attr:`MOST_SPECIFIC` means.

    **This is the enum's first shipped home**, and a later reader imports it
    from here rather than the enum moving. Moving a published vocabulary is a
    migration, and the point of putting it here first is to not owe one.
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
    refused by the read one frame earlier, so this field's population is what its
    first paragraph says it is rather than a mixture.

    It carries the rows, not just the ids, because the rows are what a maintainer
    needs in order to find the tagger.
    """

    unsupported_rows: tuple[int, ...] = ()
    """Positions that contributed to no entry in :attr:`supported`.

    A row carrying no tag, a row carrying a tag for another vocabulary or another
    axis, and a row whose only tag was unplaced.

    **A report, not a verdict**: an answer drawn from five rows of which three
    touched nothing is exactly the case a consumer must be able to see. It is
    derivable, and it is kept because the caller who has to derive it derives it
    every time.

    The three causes are deliberately **not** distinguished. The caller holds the
    tags, so *no tag* and *another vocabulary's tag* are one lookup apart for
    them; which vocabularies are in play at all is :func:`ontology_support`'s
    question.
    """

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

        if policy is Granularity.MOST_GENERAL:
            kept = [entry for entry in self.supported if not entry.above]
        elif policy is Granularity.MOST_SPECIFIC:
            covered = {above for entry in self.supported for above in entry.above}
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

    Its other producer is the ontology registry over in ``dataknobs-data``,
    which answers this shape filtered to the vocabularies it holds; the type is
    here because a value type travels with the operation that answers in it, and
    that package already depends on this one.
    """

    ontology_id: str
    """Which vocabulary."""

    rows: tuple[int, ...]
    """Positions that named it, ascending, without duplicates --- :attr:`NodeSupport.rows`
    one level up, and positional for its reason."""


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
            caller who hands it one directly.
        cache: The walk cache, forwarded. One is used within a call whether or
            not one is given, because the *n* ascents share a frontier;
            supplying one spends those replies across other walks as well.

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
    walked: WalkCache = {} if cache is None else cache

    placed: dict[K, list[int]] = {}
    missing: dict[K, list[int]] = {}

    for position, row in enumerate(tagged):
        for tag in row:
            if tag.ontology_id != ontology.id or tag.taxonomy_id != taxonomy_id:
                continue
            node = ontology.localize(tag.qualified_id)
            seen = placed if axis.at(node).exists() else missing
            rows = seen.setdefault(node, [])
            if position not in rows:
                rows.append(position)

    within = set(placed)
    supported = tuple(
        NodeSupport(
            node_id=node,
            rows=tuple(rows),
            above=tuple(
                view.node for view in axis.at(node).ancestors(cache=walked) if view.node in within
            ),
        )
        for node, rows in placed.items()
    )
    unplaced = tuple(NodeSupport(node_id=node, rows=tuple(rows)) for node, rows in missing.items())

    contributing = {position for rows in placed.values() for position in rows}
    return SupportSet(
        ontology_id=ontology.id,
        taxonomy_id=taxonomy_id,
        supported=supported,
        unplaced=unplaced,
        unsupported_rows=tuple(
            position for position in range(len(tagged)) if position not in contributing
        ),
    )


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

    Args:
        tagged: One row's tags per position, as
            :func:`~dataknobs_common.ontology.tags.read_node_tags` answered.
            A ``Sequence``, for :attr:`NodeSupport.rows`' reason.

    Returns:
        One :class:`OntologySupport` per vocabulary any tag named, ranked. Empty
        for a corpus whose rows touched none, which is an answer rather than an
        error.
    """
    touched: dict[str, list[int]] = {}
    for position, row in enumerate(tagged):
        for tag in row:
            rows = touched.setdefault(tag.ontology_id, [])
            if position not in rows:
                rows.append(position)

    order = {ontology_id: position for position, ontology_id in enumerate(touched)}
    entries = [
        OntologySupport(ontology_id=ontology_id, rows=tuple(rows))
        for ontology_id, rows in touched.items()
    ]
    return tuple(sorted(entries, key=lambda entry: (-len(entry.rows), order[entry.ontology_id])))
