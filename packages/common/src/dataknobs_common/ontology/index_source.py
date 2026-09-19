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

from dataclasses import dataclass, field
from itertools import islice
from typing import TYPE_CHECKING, Generic

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.hierarchy import K
from dataknobs_common.index import IndexItem
from dataknobs_common.ontology.tags import ALIAS_FORMS_KEY, ONTOLOGY_ID_KEY

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from dataknobs_common.ontology.values import AsyncOntology

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


@dataclass(frozen=True)
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

    @property
    def source_field(self) -> str:
        """Which entity fields this source composed its text from, as one name.

        Not a protocol member, for the reason the protocol has two: a required
        third breaks ``isinstance`` for every structural implementor outside
        this tree. An index reads it with a fall-back and hands it to the
        store, which writes it beside the source text --- so a hit from this
        index is distinguishable from a row some other caller wrote through
        the raw vector door, which it otherwise is not.
        """
        return self.join.join(self._fields)

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
        """
        entities = self.ontology.entities
        declared = entities.describe().declares or frozenset()
        for type_id in sorted(declared):
            ids = sorted(await entities.by_type(type_id), key=str)
            cursor = iter(ids)
            while batch := list(islice(cursor, STREAM_BATCH_SIZE)):
                found = await entities.get_many(batch)
                for entity_id in batch:
                    entity = found.get(entity_id)
                    if entity is None:
                        continue
                    yield IndexItem(
                        id=self.ontology.qualify(entity.id),
                        text=self._text_for(entity),
                        metadata={
                            ONTOLOGY_ID_KEY: self.ontology.id,
                            self.aliases_key: list(entity.aliases),
                        },
                    )

    def _text_for(self, entity: object) -> str:
        """The chosen fields' values, non-empty ones only, joined.

        Empty values are dropped **before** the join rather than joined and
        stripped afterwards, which is what keeps one value from arriving with
        a separator hanging off it. Over a real vocabulary most entities carry
        a name and no description, so this is the ordinary path.
        """
        values = [str(getattr(entity, name, "") or "").strip() for name in self._fields]
        return self.join.join(value for value in values if value)
