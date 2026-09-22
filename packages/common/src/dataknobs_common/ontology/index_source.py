# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Every entity a vocabulary holds, as one indexable item each.

The adapter between a vocabulary and a text index. It takes the **ontology**
rather than its entity source, because producing an item needs two things and
the entity source has only one of them: the set of entities to yield, and a
**qualified** id per item. The second lives on the ontology --- ``qualify`` is
one of the doors a key leaves by --- and re-spelling it from a codec and an id
is the thing that breaks over a vocabulary whose keys are not strings.

It adds no member to any protocol. The enumeration is the pair that already
shipped: ``describe().declares`` says which type ids may be named, ``by_type``
returns what is in one, and they are the same set seen from two sides.

**In ``ontology/`` rather than beside the protocol it satisfies**, because it
reads an ``AsyncOntology`` and writes ``ONTOLOGY_ID_KEY`` --- the two things
this subpackage holds. The protocol and the pure sources are a level up, where
a reader who has no vocabulary can reach them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import islice
from typing import TYPE_CHECKING, Generic

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.hierarchy import K
from dataknobs_common.index import IndexItem, join_non_empty
from dataknobs_common.ontology.sources import AUTHORED_SOURCE_ID
from dataknobs_common.ontology.tags import ALIAS_FORMS_KEY, ONTOLOGY_ID_KEY

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from dataknobs_common.ontology.values import AsyncOntology

logger = logging.getLogger(__name__)

__all__ = ["EntitySourceIndexSource"]

#: How many ids one ``get_many`` asks for.
#:
#: The adapter's own batching, and it is the adapter's rather than the entity
#: source's on purpose: ``get_many`` batches its *reads* and returns a ``dict``
#: of everything, so handing it the whole union materialises the vocabulary
#: inside a member called ``stream_items``. Matches the read batch a live
#: entity source already uses, so the two do not disagree about what one read
#: costs.
STREAM_BATCH_SIZE = 1000

#: The ``Entity`` attributes ``fields`` may name.
#:
#: Two, and they are the two an entity carries free text in. Validated at
#: construction rather than at the first read, so the caller is told while
#: still holding the mistake --- the alternative is a stream that yields empty
#: text for every row and looks like an empty vocabulary.
TEXT_FIELDS = ("name", "description")


# `eq=False` for the reason the pure sources carry: `fields` is declared a
# `Sequence[str]`, so a caller passing a list gives a frozen instance that
# answers `Hashable` and raises at `hash()`. Identity is the honest answer
# for a source, and it is what `AsyncOntology` beside it already says.
@dataclass(frozen=True, eq=False)
class EntitySourceIndexSource(Generic[K]):
    """Every entity an ontology holds, as one indexable item each.

    Example:
        ```python
        source = EntitySourceIndexSource(onto, fields=("name", "description"))
        async for item in source.stream_items():
            ...  # IndexItem(id="mammals:beagle", text="Beagle -- a hound", ...)
        ```

    Attributes:
        ontology: The vocabulary to enumerate. Its ``entities`` supplies the
            rows and its ``qualify`` supplies the id space.
        fields: Which of :data:`TEXT_FIELDS` to read off each entity, in
            order. Validated at construction.
        join: What to put between two non-empty field values. Applied
            **between** values rather than after each, so an entity carrying
            only one of two fields produces no dangling separator --- which is
            the common case over a real vocabulary rather than an edge one.
        aliases_key: The metadata key the surface forms are written under, and
            the one a decorating source is pointed at. Defaulted to the
            published constant rather than spelled, because it is one key with
            two ends.
    """

    ontology: AsyncOntology[K]
    fields: Sequence[str] = ("name",)
    join: str = " -- "
    aliases_key: str = ALIAS_FORMS_KEY

    #: Resolved once at construction, so the per-item read is a lookup rather
    #: than a repeated membership test. Not a parameter.
    _fields: tuple[str, ...] = field(init=False, repr=False, compare=False)

    #: The entity types validated at construction, kept rather than re-asked.
    #:
    #: ``__post_init__`` refuses a source that cannot enumerate and a source
    #: declaring a type the schema does not; both refusals are about *this*
    #: set. ``stream_items`` used to call ``describe()`` a second time and
    #: write ``or frozenset()`` over the answer, which turned the first
    #: refusal into a suggestion --- a source answering a set here and
    #: ``None`` there produced the silently-empty index the refusal exists to
    #: prevent, and skipped the second refusal on the way. Streaming the
    #: validated set is what makes both of them bind the read. Not a
    #: parameter.
    _declared: frozenset[str] = field(init=False, repr=False, compare=False)

    #: The id of the source that was described, kept for the same reason.
    #:
    #: **``describe()`` is asked once per instance, and this is the half of
    #: that rule the neighbour above does not state.** The second call came
    #: back to read one field off a fresh description for a log line, which
    #: is the cheap end of the same defect: ``describe()`` builds a new value
    #: every time and the protocol promises nothing about two of them
    #: agreeing, so a report naming the id read the second time names a
    #: source no refusal here ever looked at. Not a parameter.
    _source_id: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Refuse a field name, an unenumerable source, and an undeclared type.

        Three refusals, all at construction, all naming the source. The order
        matters only in that the cheapest comes first; each is a different
        question and none subsumes another.
        """
        chosen = tuple(self.fields)
        unknown = [name for name in chosen if name not in TEXT_FIELDS]
        if unknown or not chosen:
            raise ValidationError(
                f"index source over ontology {self.ontology.id!r} was asked for "
                f"{sorted(unknown) or 'no'} entity field(s); an entity carries free text "
                f"in {', '.join(TEXT_FIELDS)} and nothing else",
                context={"ontology_id": self.ontology.id, "fields": list(chosen)},
            )
        object.__setattr__(self, "_fields", chosen)

        description = self.ontology.entities.describe()
        declared = description.declares
        if declared is None:
            # `None` is *cannot enumerate*, which is a different answer from
            # *holds none* and the reason that field has no default. A source
            # that cannot enumerate cannot be indexed whole, and saying so
            # here beats building an index that is silently empty -- an entity
            # absent from an index resolves to nothing, and "not in the
            # corpus" is a legitimate answer nothing reports as an error.
            raise ValidationError(
                f"source {description.source_id!r} of ontology {self.ontology.id!r} cannot "
                f"enumerate its entity types, so an index over it would be silently partial. "
                f"A source holding no types declares an empty set; this one declares nothing",
                context={"ontology_id": self.ontology.id, "source_id": description.source_id},
            )

        # Skipped where the document declared no `entity_types:` at all: an
        # empty schema section is no schema, not an empty one, and refusing
        # here would forbid the whole by-reference regime -- a vocabulary
        # backed by rows nobody wrote a schema for is the case this adapter
        # exists to index.
        if self.ontology.entity_types:
            undeclared = sorted(set(declared) - set(self.ontology.entity_types))
            if undeclared:
                raise ValidationError(
                    f"source {description.source_id!r} declares entity type(s) {undeclared} "
                    f"that ontology {self.ontology.id!r} does not declare in `entity_types:`; "
                    f"an index over them would hold entities nothing can say anything about",
                    context={
                        "ontology_id": self.ontology.id,
                        "source_id": description.source_id,
                        "undeclared": undeclared,
                    },
                )

            # The other direction, and it is a warning rather than a refusal.
            # A schema declaring two types over a binding that holds one is an
            # ordinary configuration -- one live table under a vocabulary that
            # also names types nothing is bound to -- so refusing it would
            # forbid the common case. But the resulting index *is* partial in
            # exactly the way the `None` refusal is about: a query about a
            # type nobody indexed answers "not in the corpus", which is a
            # legitimate answer nothing reports as an error.
            #
            # **Only for a live binding**, and the exclusion is the whole
            # reason this is not a general check. An authored source derives
            # `declares` from the entities it actually holds, so a declared
            # type with no entities is absent from it -- and that is a
            # document saying a type has no members, which is a vocabulary
            # being explicit rather than an index being partial. Warning
            # there would fire on the ordinary authored document and say
            # nothing true. A live binding is the other case: its projection
            # names one constant type, so a schema declaring several is a
            # genuine gap between what is named and what any source can
            # answer for.
            missing = sorted(set(self.ontology.entity_types) - set(declared))
            if missing and description.source_id != AUTHORED_SOURCE_ID:
                logger.warning(
                    "index source over ontology %s covers %d of %d declared entity "
                    "type(s); source %s holds nothing under %s, so an index over it "
                    "will answer nothing for them",
                    self.ontology.id,
                    len(set(declared)),
                    len(self.ontology.entity_types),
                    description.source_id,
                    ", ".join(missing),
                )

        object.__setattr__(self, "_declared", frozenset(declared))
        object.__setattr__(self, "_source_id", description.source_id)

    @property
    def source_field(self) -> str:
        """Which entity fields this source composed its text from, as one name.

        Not a protocol member, for the reason the protocol has two: a required
        third breaks ``isinstance`` for every structural implementor outside
        this tree. An index reads it with a fall-back and hands it to the
        store, which writes it beside the source text --- so a hit from this
        index is distinguishable from a row some other caller wrote through
        the raw vector door, which it otherwise is not.

        **Comma-joined, and deliberately not :attr:`join`.** The key's reader
        splits it on commas, so composing it with the display separator wrote
        a grammar nothing parses --- and tied the key to a cosmetic choice,
        where changing how the text reads changed how the key encodes.
        """
        return ",".join(self._fields)

    def declares(self) -> frozenset[str]:
        """One named set: this ontology's id.

        A source over a vocabulary declares an ontology id and writes
        ``ONTOLOGY_ID_KEY``; the two are the same fact, one for a scope filter
        to translate against and one stored on the row.
        """
        return frozenset({self.ontology.id})

    async def stream_items(self) -> AsyncIterator[IndexItem]:
        """One item per entity, batched, in type order.

        **Order is stable across types and unspecified within one.**
        ``by_type`` answers a ``frozenset`` and the key type need not be
        orderable, so no within-type order is available to promise --- and
        nothing needs one, because every row is written with its own id and is
        independent of the rest.

        The union is read in batches rather than handed over whole: the entity
        source's own bulk read batches what it sends the backend but returns a
        mapping of everything, so a single call would hold the entire
        vocabulary in memory inside the member whose name says it does not.

        **The batching bounds the entities and not the ids.** ``by_type``
        answers a set of every id of one type --- its own docstring says *"on
        a million-row binding this returns a million-element set and there is
        no ``limit:`` that would make it something else"* --- and the sort
        below builds a second list of the same size, both held for that
        type's whole stream. So the memory this method holds is proportional
        to the largest *type*, not to the vocabulary and not to
        :data:`STREAM_BATCH_SIZE`. Fixing that needs a streaming enumerator on
        the entity source, which is a protocol change rather than something
        this adapter can arrange; until there is one, the claim above is about
        entity objects and this paragraph is the part it does not cover.

        **A build whose every row composes empty text is reported, once.**
        :data:`TEXT_FIELDS`' construction check catches a *misspelled* field
        name for the reason it states --- *"the alternative is a stream that
        yields empty text for every row and looks like an empty
        vocabulary"* --- and cannot catch a correctly spelled one the
        entities carry nothing under, which is the ordinary case for a live
        binding whose projection fills whatever columns the table has. The
        result is not an empty index: it is an index full of rows
        equidistant from every query, which is worse, because an empty index
        has a report and this has an answer. So this reports it, at the same
        level and for the same reason as the partial-coverage warning at
        construction --- and needs no live-binding restriction as that one
        has, because an authored document whose every entity yields empty
        text under ``fields:`` is a document with no names either.

        **Every row, not some.** A binding holding the field on a third of
        its rows is legitimate and common, and reporting there would fire on
        the ordinary case. A stream that yields nothing reports nothing
        either: that is the empty-index condition, which the construction
        refusals already speak for.

        **The report follows the close, not the end.** Written as a statement
        after the loop it was reachable only by a consumer that ran the
        stream out, so the two consumers that most need it --- one stopping
        on a failure, one stopping on purpose --- got silence. A build that
        fails partway is exactly where *every row composed empty text* is
        worth knowing, because it is a candidate cause and the caller is
        about to decide whether to retry. So it is emitted from a ``finally``
        and says which of the two happened, since a count off an unfinished
        stream describes what was consumed rather than what the vocabulary
        holds.

        The stated cost: a caller sampling a short prefix for a preview can
        now be warned about a binding that fills the field on most of its
        rows but not its first few. That is the same weak-sample reading the
        paragraph above declines for *some* rows and accepts for *few* ---
        accepted here because the consumer of this protocol is an index
        build, the count is in the message for the reader to weigh, and the
        condition it reports is a property of the request (a ``fields:``
        naming something the projection does not fill) rather than of the
        data.

        **Closed, rather than abandoned.** An async generator a consumer
        walks away from runs its ``finally`` when the interpreter finalizes
        it --- a later turn of the loop, unordered against the consumer's own
        reporting. A consumer that wants the report at its failure closes the
        stream there, which is what ``SemanticIndex.build`` does with
        :func:`~dataknobs_common.async_iter.aclosing_iter`.
        """
        entities = self.ontology.entities
        yielded = 0
        with_text = 0
        finished = False
        try:
            for type_id in sorted(self._declared):
                ids = sorted(await entities.by_type(type_id), key=str)
                cursor = iter(ids)
                while batch := list(islice(cursor, STREAM_BATCH_SIZE)):
                    found = await entities.get_many(batch)
                    for entity_id in batch:
                        entity = found.get(entity_id)
                        if entity is None:
                            continue
                        text = self._text_for(entity)
                        yielded += 1
                        with_text += bool(text)
                        yield IndexItem(
                            id=self.ontology.qualify(entity.id),
                            text=text,
                            metadata={
                                ONTOLOGY_ID_KEY: self.ontology.id,
                                self.aliases_key: list(entity.aliases),
                            },
                        )
            finished = True
        finally:
            # In `finally` rather than after the loop, so the report follows
            # the *close* rather than the exhaustion -- see the docstring.
            # Nothing here awaits: a `finally` that awaits during a close
            # raises `async generator ignored GeneratorExit` and loses both
            # the report and the close.
            if yielded and not with_text:
                plural = "y" if yielded == 1 else "ies"
                fields = ", ".join(self._fields)
                if finished:
                    logger.warning(
                        "index source over ontology %s streamed %d entit%s and every one "
                        "composed empty text from field(s) %s of source %s; the index this "
                        "builds is not empty, it is full of rows equidistant from every query",
                        self.ontology.id,
                        yielded,
                        plural,
                        fields,
                        self._source_id,
                    )
                else:
                    logger.warning(
                        "index source over ontology %s was closed after %d entit%s and every "
                        "one composed empty text from field(s) %s of source %s; the stream "
                        "did not finish, so that is what it produced rather than what the "
                        "vocabulary holds -- but a row composing no text is not absent from "
                        "an index, it is in it and equidistant from every query",
                        self.ontology.id,
                        yielded,
                        plural,
                        fields,
                        self._source_id,
                    )

    def _text_for(self, entity: object) -> str:
        """The chosen fields' values, non-empty ones only, joined.

        Empty values are dropped **before** the join rather than joined and
        stripped afterwards, which is what keeps one value from arriving with
        a separator hanging off it. Over a real vocabulary most entities carry
        a name and no description, so this is the ordinary path.
        """
        return join_non_empty([getattr(entity, name, "") for name in self._fields], self.join)
