# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The keys an indexed row carries to say what it is about, and the read of them.

Four string constants, a value type per thing a reader gets back, and two
functions. A vector row written from a vocabulary needs to say which
vocabulary, which axis, which node, and which surface forms it stood for ---
and every one of those is a key spelled into somebody else's corpus, so it is
published here rather than written as a literal at each end. A key spelled at
each end is a reader reaching for the wrong one, and a key nothing wrote reads
as absent, which every reader treats as *unknown, assume current*.

**The read is published from the same place as the keys**, so that a writer who
came here for the constants finds what will be done with them. It is a pure
function over a mapping: it holds no vocabulary, opens nothing, and cannot tell
a node that exists from one that does not --- judging that needs an ontology,
and having one is the caller's next line rather than this module's.

**Nothing here imports anything outside this package and the standard
library.** That is what makes every question this module answers askable with
no store, no embedder and no event loop, and it is a rule rather than a
convenience: the keys describe a row that a consumer's own retrieval handed
back, and a reader of them that needed a backend would be a reader nobody on
that side of the boundary can call.

**The family is split across two modules, and this docstring is half of what
pays for that.** The fifth key of the same row --- ``MODEL_NAME_KEY``, the
model that produced the vector --- lives in
``dataknobs_data.vector.content``, because it is about an **embedder** rather
than about identity, it already ships, and five files import it. Moving it
would be a migration. So the two halves name each other, in both directions:
``vector/content.py``'s module docstring names this one, and this one names
it. A reader reaching for one module finds the other.

**The values are namespaced and underscore-separated**, because they are
written into a store we do not own and a family spelled two ways is the defect
a published family exists to prevent. Each value is its constant's name less
``_KEY``, lower-cased.
"""

from __future__ import annotations

from collections.abc import Buffer, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology.model import _refuse_colon, qualify

__all__ = [
    "ALIAS_FORMS_KEY",
    "NODE_ID_KEY",
    "ONTOLOGY_ID_KEY",
    "TAXONOMY_ID_KEY",
    "MalformedRow",
    "NodeTag",
    "TagReading",
    "read_alias_forms",
    "read_node_tags",
    "read_node_tags_many",
]

#: Which vocabulary this row's id belongs to.
#:
#: The one key every row written from a vocabulary carries, whatever else it
#: does: an id without its namespace is a local id, and a reader that cannot
#: tell the two apart resolves a hit to nothing while reporting *not found*.
ONTOLOGY_ID_KEY = "dk_ontology_id"

#: Which structure axis of that vocabulary this row is placed on.
TAXONOMY_ID_KEY = "dk_taxonomy_id"

#: Which node or nodes of that axis this row is about.
#:
#: **List-valued**, because a row may be about several nodes at once. A reader
#: takes a bare string as one node rather than as its characters.
NODE_ID_KEY = "dk_node_id"

#: The surface forms this row's entity is known by.
#:
#: **List-valued.** Written by a source that yields entities, and read by the
#: decorator that turns one entity into one indexed row per form --- one key,
#: two ends, which is why it is a constant rather than a literal at either.
#:
#: A reader takes a bare string as one form rather than as its characters, as
#: :data:`NODE_ID_KEY` above states for the same family. The rule is restated
#: here because this key's reader is the one that had not applied it: a
#: consumer-written ``"ACME"`` became four one-character rows, all under the
#: entity's id, leaving one row holding ``"E"``.
ALIAS_FORMS_KEY = "dk_alias_forms"


#: The three keys that describe *what a row is about*, written together or not
#: at all. Named once, because three readings of the same set is how a reader
#: comes to require two of them and refuse on the third.
#:
#: ``ALIAS_FORMS_KEY`` is not among them: it describes the *entity* a row
#: stands for rather than which node the row is placed on, and a row carrying
#: forms and no placement is an ordinary alias row rather than a half-written
#: tag.
_TAG_KEYS = (ONTOLOGY_ID_KEY, TAXONOMY_ID_KEY, NODE_ID_KEY)


@dataclass(frozen=True)
class NodeTag:
    """One node a content row is tagged with, and whose vocabulary it is.

    A frozen dataclass rather than a ``NamedTuple``, and the difference from
    :class:`~dataknobs_common.ontology.model.QualifiedId` next door is its
    stated reason: that one is a ``NamedTuple`` *"because every reader unpacks
    it"*, and this one's readers filter it and project from it. A tuple with a
    property that is not a field also makes ``len()`` and iteration disagree
    with the object's own surface.
    """

    #: Which vocabulary. Never ``None``: a node id without one addresses
    #: nothing, which is the whole of why the pair is written rather than
    #: composed.
    ontology_id: str

    #: Which axis of it.
    taxonomy_id: str

    #: The node, **as written** --- a rendered key in the space
    #: :meth:`~dataknobs_common.ontology.taxonomy.Taxonomy.subtree_keys`
    #: returns, not a ``K``. Turning it back into a key is
    #: :meth:`~dataknobs_common.ontology.Ontology.localize`'s, because only an
    #: ontology knows whose ids it is parsing.
    node_id: str

    @property
    def qualified_id(self) -> str:
        """The pair, spelled as the one string ``Ontology.localize`` takes.

        Composed through the free
        :func:`~dataknobs_common.ontology.model.qualify` and never with an
        f-string, for that function's own stated reason: *"a malformed id is
        unfixable once written into stored data."* The two spellings of
        a namespaced id --- this family's two keys, and ``qualify``'s one
        string --- meet here and nowhere else, so this is the member that says
        they are the same thing.
        """
        return qualify(self.ontology_id, self.node_id)


def read_node_tags(metadata: Mapping[str, Any]) -> tuple[NodeTag, ...]:
    """Which nodes this row is about, and whose vocabulary each one is.

    The read half of the key family, published from the same place as the keys
    so that a writer reading them finds it.

    **Empty for a row carrying none of the three keys.** A row that touched no
    vocabulary is a first-class answer: most of a corpus is such rows, and a
    caller that has to tell an exception from an answer on the common path
    wraps every call in a ``try``.

    **A key whose value is ``None`` is a key nothing wrote.** Presence is read
    off the value rather than off the mapping, because the argument is a row
    *as the consumer's own store handed it back* and a relational or columnar
    one materialises what nothing wrote as a null rather than omitting it.
    Reading such a row as half-written would refuse an untagged corpus
    wholesale --- the opposite of the distinction the two answers exist to
    draw --- and a null under one key of three still refuses, with the more
    accurate of the two messages.

    **Refuses a row carrying some of them and not all.** That is not an
    operational state the way a stale id is --- it is a writer that got the
    contract wrong, and silence lets it scale to a whole corpus before anyone
    reads a row back. The refusal names the keys that are absent, because the
    writer this contract is addressed to has not read this tree.

    A bare ``str`` under :data:`NODE_ID_KEY` is one node, not its characters.

    **Refuses a value that is not a string.** A ``bytes`` --- or any other
    ``Buffer``: a ``memoryview`` off a binary column, an ``array`` --- is a
    sequence of integers and reaches the failure the line above forecloses; a
    ``Mapping`` reads as its keys and would otherwise succeed; and a scalar
    that is not a ``str`` would reach ``TypeError`` from the iteration rather
    than the class this contract documents, which is why the type is tested
    before anything is iterated. ``[]`` is not a malformation: it is a row
    about no node of this axis, and it answers ``()``.

    **Refuses a ``:`` in :data:`ONTOLOGY_ID_KEY`**, which the loader already
    refuses in an id it mints. :attr:`NodeTag.qualified_id` joins the pair
    with that character and ``Ontology.localize`` splits on the first one, so
    an ontology id carrying one composes an id that a *different* vocabulary
    accepts and localizes to a local id nobody wrote. This is the last frame
    that can see the shape, exactly as it is the last that can see the type.

    Args:
        metadata: A content row's metadata mapping, as the consumer's own
            store handed it back.

    Returns:
        One tag per node, in the order the row wrote them. Empty where the row
        carries no tag at all.

    Raises:
        ValidationError: Where the row carries some of the three keys and not
            all three --- counting a ``None`` as none --- naming the ones that
            are missing; where :data:`ONTOLOGY_ID_KEY` or
            :data:`TAXONOMY_ID_KEY` is not a ``str``, or
            :data:`NODE_ID_KEY` is neither a ``str`` nor a non-``Buffer``
            ``Sequence`` of them --- naming the key, the type found, and every
            offending position; or where :data:`ONTOLOGY_ID_KEY` carries a
            ``:``. The row refuses whole; one bad entry does not yield the
            readable ones.
    """
    written = [key for key in _TAG_KEYS if metadata.get(key) is not None]
    if not written:
        return ()
    if len(written) != len(_TAG_KEYS):
        absent = [key for key in _TAG_KEYS if metadata.get(key) is None]
        raise ValidationError(
            f"a content row carrying {_named(written)} declares no {_named(absent)}: "
            f"a row carrying any of the three tag keys carries all three",
            context={"written": written, "absent": absent},
        )

    ontology_id = metadata[ONTOLOGY_ID_KEY]
    taxonomy_id = metadata[TAXONOMY_ID_KEY]
    _refuse_a_value_that_is_not_a_key(ONTOLOGY_ID_KEY, ontology_id)
    _refuse_a_value_that_is_not_a_key(TAXONOMY_ID_KEY, taxonomy_id)
    _refuse_colon(ONTOLOGY_ID_KEY, ontology_id)
    return tuple(
        NodeTag(ontology_id=ontology_id, taxonomy_id=taxonomy_id, node_id=node_id)
        for node_id in _strings_under(NODE_ID_KEY, metadata[NODE_ID_KEY])
    )


def read_alias_forms(metadata: Mapping[str, Any], key: str = ALIAS_FORMS_KEY) -> tuple[str, ...]:
    """The surface forms this row's entity is known by.

    The read of :data:`ALIAS_FORMS_KEY`, over the same core
    :func:`read_node_tags` reads its node list with --- which is the point of
    publishing it. The rule is that **a bare ``str`` is one form, not its
    characters**, and this key's docstring exists because the reader that had
    to apply it did not: a consumer-written ``"ACME"`` became four
    one-character rows. A rule implemented privately is a rule the consumer
    who needs it has to write again.

    **Standalone, unlike the three tag keys.** This key describes the *entity*
    a row stands for rather than which node the row is placed on, so a row
    carrying it and nothing else is an ordinary alias row and not a
    half-written tag. There is no companion key for it to be absent against,
    and so no half-written state: a row without it answers ``()``.

    Args:
        metadata: A content row's metadata mapping, as the consumer's own
            store handed it back.
        key: Where the forms were written. A parameter rather than the
            constant because
            :attr:`~dataknobs_common.ontology.index_source.EntitySourceIndexSource.aliases_key`
            is a configurable field defaulting to it --- a read hard-coded to
            the constant could not open the rows the writer beside it
            produces.

    Returns:
        One form per entry, in the order the row wrote them. Empty where the
        row carries no forms, including where it carries ``None``.

    Raises:
        ValidationError: Where the value is neither a ``str`` nor a
            non-``Buffer`` ``Sequence`` of them --- naming the key, the type
            found, and every offending position. The row refuses whole.
    """
    written = metadata.get(key)
    if written is None:
        return ()
    return tuple(_strings_under(key, written))


@dataclass(frozen=True)
class MalformedRow:
    """One position :func:`read_node_tags` refused, and what it said.

    **The message, not the exception.** An exception inside a frozen value
    makes two readings of the same input unequal and puts a mutable object in
    an immutable one; and the context a caller might act on is one *localized*
    re-read away --- ``read_node_tags(metadatas[m.row])`` --- which is the
    property the batch reader buys.
    """

    row: int
    """Position in the sequence the caller passed."""

    reason: str
    """What the row reader said about this row, verbatim.

    Carried rather than re-composed so that the batch caller is told no less
    than the per-row caller, and so that a malformation the row reader learns
    to refuse later arrives here already spelled.
    """


@dataclass(frozen=True)
class TagReading:
    """What a set of content rows declared, and which of them were written wrong.

    Returned by :func:`read_node_tags_many`. It holds no corpus, no store and
    no row --- only what the caller's own sequence said, in the caller's own
    positions.
    """

    tags: tuple[tuple[NodeTag, ...], ...]
    """One entry per input position, **in the order asked**.

    ``()`` where the row carried no tag and ``()`` where it was refused, which
    is :class:`~dataknobs_common.hierarchy.BulkHierarchy`'s shipped rule for
    every positional reply in this package: *"a node with no answer contributes
    an empty sequence rather than being dropped, because the reply is
    positional."* Which of the two a given ``()`` is, is :attr:`malformed`'s
    answer and is the reason that field exists.

    **A third source is not separated, deliberately.** A fully tagged row
    whose node list is empty also holds ``()``, and :attr:`malformed` does not
    distinguish it from an untagged one --- both are *this row is about no
    node*, one said by omission and one said explicitly, and neither is a
    fault anybody can act on. A caller who needs the difference has the row:
    it is whether :data:`~dataknobs_common.ontology.tags.NODE_ID_KEY` is in
    it. The field separates the answers that differ in *who can fix them*,
    which is the only distinction a reader of a corpus can act on.
    """

    malformed: tuple[MalformedRow, ...] = ()
    """Every position the row reader refused, ascending.

    Empty for a reading with nothing wrong in it, which is the ordinary case
    and is why it defaults.
    """

    def require_readable(self) -> tuple[tuple[NodeTag, ...], ...]:
        """:attr:`tags`, or a refusal naming **every** malformed position.

        The row reader's stance, taken over a batch by a caller who wants it.
        One call rather than a loop, and the message is composed once here
        rather than invented by each caller --- the reason the loader's own
        shared reader exists, one module over: *"a shared reader is what makes
        the eleventh required field inherit the behaviour instead of repeating
        the omission."*

        It performs no I/O and reads nothing but this value.

        Returns:
            :attr:`tags`, unchanged, where nothing was refused.

        Raises:
            ValidationError: Where any row was refused, naming each position
                and quoting what the row reader said about it.
        """
        if self.malformed:
            named = "; ".join(f"row {bad.row}: {bad.reason}" for bad in self.malformed)
            raise ValidationError(
                f"{len(self.malformed)} of {len(self.tags)} rows are not readable: {named}",
                context={"malformed": [bad.row for bad in self.malformed], "rows": len(self.tags)},
            )
        return self.tags


def read_node_tags_many(metadatas: Sequence[Mapping[str, Any]]) -> TagReading:
    """:func:`read_node_tags` over a sequence of rows, one answer per position.

    **The bulk form, for a corpus rather than a row**, and ``_many`` is the
    name this package's other batch members already carry.

    **It calls the row reader and re-implements nothing.** What *malformed*
    means is defined in exactly one place, which matters because that set is
    not closed.

    **It reports rather than raises, and that is not the row reader relaxed.**
    The row reader refuses a half-written row because silence lets the mistake
    scale to a whole corpus --- and over a batch, raising is the quieter of the
    two: it names the first bad row and discards every good one. The other
    clause is what settles it, because it is the same argument: an untagged row
    answers ``()`` rather than raising so that a caller is not made to wrap
    every call in a ``try``. Over ten thousand rows the common path *contains*
    a malformed row.

    A caller who wants the refusal asks for it, on the next line and once:
    :meth:`TagReading.require_readable`.

    Args:
        metadatas: One content row's metadata mapping per position, as the
            consumer's own store handed them back. A ``Sequence``: the
            positions are the only row reference this contract is entitled to.

    Returns:
        A :class:`TagReading`. Empty in both fields for an empty input.
    """
    read: list[tuple[NodeTag, ...]] = []
    malformed: list[MalformedRow] = []
    for position, metadata in enumerate(metadatas):
        try:
            read.append(read_node_tags(metadata))
        except ValidationError as refusal:
            read.append(())
            malformed.append(MalformedRow(row=position, reason=str(refusal)))
    return TagReading(tags=tuple(read), malformed=tuple(malformed))


def _named(keys: Sequence[str]) -> str:
    """The keys as a refusal names them: quoted, in the family's own order."""
    return ", ".join(repr(key) for key in keys)


def _refuse_a_value_that_is_not_a_key(key: str, value: Any) -> None:
    """:data:`ONTOLOGY_ID_KEY` and :data:`TAXONOMY_ID_KEY`: a string, or a refusal.

    Neither is list-valued, so there is no iteration to hide a value of the
    wrong type inside. The message names the type found, because naming only
    the key would send a writer to look at a value they can already see.

    Raises:
        ValidationError: Naming the key and the type found.
    """
    if isinstance(value, str):
        return
    raise ValidationError(
        f"{key!r} is {type(value).__name__}, and a {key!r} is a string",
        context={"key": key, "type": type(value).__name__},
    )


def _strings_under(key: str, value: Any) -> list[str]:
    """``key``'s entries, or a refusal naming every offending position.

    **The key is a parameter because two keys carry this rule.**
    :data:`NODE_ID_KEY` and :data:`ALIAS_FORMS_KEY` are both list-valued, both
    take a bare ``str`` as one entry, and both refuse the same shapes for the
    same reasons --- and the second one's docstring exists because a reader
    that had written the rule itself got it wrong. One implementation is what
    keeps the two from drifting; the key is what lets each refusal name the
    one the writer actually wrote.

    A bare ``str`` is one node. A ``Sequence`` is its entries, **except** a
    ``Buffer`` --- ``bytes``, ``bytearray``, ``memoryview``, ``array.array``
    --- which are sequences of integers and are the foreclosed failure
    arriving through a second door. **The exclusion is that property and not a
    list of types**, because a list is the types somebody thought of:
    ``memoryview`` is what a driver hands back for a binary column, it is
    registered on ``Sequence``, and an empty one would otherwise answer *this
    row is about no node* over a value that says nothing of the kind.

    A ``Mapping``, a ``set`` and an iterator are not sequences: the first
    reads as its keys, the second cannot satisfy this function's documented
    order, and the third is consumed by being read once.

    **Every offending position is named, not the first**, because the extent of
    a tagger's bug should be learnable in one read.

    Raises:
        ValidationError: Naming the key and the type found for a value that is
            neither; or the count and each offending position for a sequence
            carrying an entry that is not a string.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, Buffer):
        offending = [
            (position, entry) for position, entry in enumerate(value) if not isinstance(entry, str)
        ]
        if offending:
            named = ", ".join(f"{position} is {type(e).__name__}" for position, e in offending)
            raise ValidationError(
                f"{key!r} has {len(offending)} of {len(value)} entries that are "
                f"not strings: {named}",
                context={
                    "key": key,
                    "entries": len(value),
                    "offending": [position for position, _ in offending],
                },
            )
        return list(value)
    raise ValidationError(
        f"{key!r} is {type(value).__name__}, and a {key!r} is a string or a sequence of strings",
        context={"key": key, "type": type(value).__name__},
    )
