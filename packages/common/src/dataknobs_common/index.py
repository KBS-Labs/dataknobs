# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""What "some other data source" means, for anything that indexes text.

One protocol, so every layer above it is source-agnostic: a thing that can
stream ``(id, text, metadata)`` triples and say which named sets its ids fall
in. A vector index binds one of these to a store; nothing above this module
knows whether the text came from a database column, a dict, or a vocabulary.

**Top level rather than inside a subpackage.** The two members are a value and
a protocol with no dependency beyond the standard library, and both
``ontology`` and consumers outside it reach them --- a taxonomy's node set is
one of these, and so is a bare table's name column. ``hierarchy`` sits at this
level for the same reason and is the precedent followed here.

**Qualification happens in the source, never in a consumer.** A source emits
ids in whatever space it was constructed for, and it is the source's job to
emit the space its reader resolves in. An index stores what it is given and
returns what it stored; it never rewrites an id and never infers a namespace.
So one class over a bare table emits local ids, and the same class bound
inside a vocabulary emits qualified ones --- one class, two configurations,
and the difference visible at construction rather than inferred from what
comes out.

**The sources compare and hash by identity; the item does not.** A source is
a configured behaviour, not a value --- two of them over one mapping are
interchangeable, and nothing asks whether they are equal. Frozen with equality
left on, each would instead claim ``Hashable`` and raise at the call the moment
it held a ``dict`` or a caller passed a list where a ``Sequence`` was declared,
which is the one combination that guards a caller against nothing. So the
sources take ``eq=False`` and hash by identity, and :class:`IndexItem` --- a
record two of which really can be equal, and which tests compare --- stays
unfrozen with equality on, where the check answers False honestly.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from dataknobs_common.async_iter import aclosing_iter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "AliasSource",
    "AsyncIndexSource",
    "CallableSource",
    "IndexItem",
    "MappingSource",
]


def join_non_empty(values: Sequence[Any], join: str) -> str:
    """The non-empty values, joined --- and no dangling separator over one.

    One helper rather than one spelling per caller, because the property it
    carries is the one that is easy to get wrong in the same way twice: a row
    with a name and no description must produce the name, not the name
    followed by a separator. Dropping before the join is what makes that true
    by construction rather than by a strip afterwards.

    **Here rather than beside either caller.** It was written twice --- once
    in ``dataknobs-data``'s bare-table sources and once in the ontology
    adapter's ``_text_for``, one package apart, with different default
    separators --- which is the duplication its own docstring argued against.
    It is a pure algorithm over values, so it belongs where the protocol it
    serves is declared, and both composing sources reach it from there.

    Args:
        values: The values to join, in order. ``None`` and empty strings are
            dropped; everything else is rendered with ``str``.
        join: What to put between two surviving values.

    Returns:
        The surviving values joined, or ``""`` where none survived.
    """
    rendered = [str(value).strip() for value in values if value is not None]
    return join.join(value for value in rendered if value)


@dataclass
class IndexItem:
    """One indexable thing: an id, the text that stands for it, and its tags.

    Attributes:
        id: What a hit over this item resolves to. In whichever space the
            source that produced it was constructed for --- see the module
            docstring. Nothing downstream rewrites it.
        text: What gets embedded. One entity may produce several items with
            several texts and one id, which is what a surface-form decorator
            is for.
        metadata: Tags stored beside the vector and returned with a hit.
            Whatever the source writes, plus whatever a decorator forwards;
            a reader that needs a key to be there asks the source that
            declares it.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AsyncIndexSource(Protocol):
    """A stream of indexable items, and the named sets its ids fall in.

    Two members. ``stream_items`` is written as an ``async def`` generator by
    every implementation, which is why it is declared here as a plain ``def``
    returning an :class:`~collections.abc.AsyncIterator`: calling an async
    generator function returns the iterator without awaiting, so that is the
    signature a structural implementation actually has.

    ``isinstance(x, AsyncIndexSource)`` works and checks that both members are
    present; it does not check their signatures. Treat it as a smoke test,
    which is what a family validating against a protocol needs and all it
    needs.
    """

    def stream_items(self) -> AsyncIterator[IndexItem]:
        """Every item this source holds, one at a time.

        Streamed rather than returned as a list, because the point of the
        protocol is that a source may be larger than memory. An
        implementation that reads a backend does so in batches of its own
        choosing and yields one item per row; materialising the whole read
        inside a member called ``stream_items`` is the mistake the name
        exists to prevent.
        """
        ...

    def declares(self) -> frozenset[str]:
        """The named sets this source's ids belong to.

        The set a scope filter is translated against: a caller asking for
        results *within* a named set needs the source to have said which sets
        it writes, or membership cannot be decided and the filter silently
        empties the corpus.

        A source over a bare table declares nothing --- ``frozenset()`` --- and
        that is a complete answer rather than an omission: its ids are local
        and fall in no named space. A source over a vocabulary declares that
        vocabulary's id.
        """
        ...


@dataclass(frozen=True, eq=False)
class MappingSource:
    """Every entry of an in-memory mapping, as one item each.

    The smallest real source there is, and the one that lets an index be
    tested without a database. Ordered by key, which costs nothing here ---
    the mapping is already in memory --- and makes two builds over one mapping
    produce rows in one order.

    Example:
        ```python
        source = MappingSource({"beagle": "Beagle", "dog": "Dog"})
        ```
    """

    #: Id to the text that stands for it.
    items: Mapping[str, str]

    #: The named sets these ids fall in. Empty by default, because a bare
    #: mapping's keys are local ids in no named space; a caller binding one
    #: inside a namespace passes the name it qualified them with.
    declared: frozenset[str] = frozenset()

    def declares(self) -> frozenset[str]:
        """Whatever this source was constructed to declare."""
        return self.declared

    async def stream_items(self) -> AsyncIterator[IndexItem]:
        """One item per entry, in key order."""
        for key in sorted(self.items):
            yield IndexItem(id=key, text=self.items[key])


@dataclass(frozen=True, eq=False)
class CallableSource:
    """Whatever a callable produces, as items --- the escape hatch.

    For the source that is none of the others: a generator over an API, a
    file being parsed, a queue being drained. The callable returns or yields
    :class:`IndexItem` values and this class is the three lines that make it
    satisfy the protocol.

    Built rather than left to a consumer because a published family of
    reference implementations with one member missing is a homework
    assignment: the protocol ships in this module, and the adapter over a
    callable is the cheapest thing in the family to get wrong by hand.

    Example:
        ```python
        async def rows() -> AsyncIterator[IndexItem]:
            async for row in api.pages():
                yield IndexItem(id=row["id"], text=row["title"])

        source = CallableSource(rows)
        ```
    """

    #: Called with no arguments. May return an async iterable, a sync
    #: iterable, or an awaitable of either --- all four are what a caller
    #: reaching for an escape hatch already has in hand.
    fn: Callable[[], Any]

    #: As :attr:`MappingSource.declared`.
    declared: frozenset[str] = frozenset()

    def declares(self) -> frozenset[str]:
        """Whatever this source was constructed to declare."""
        return self.declared

    async def stream_items(self) -> AsyncIterator[IndexItem]:
        """Drive the callable, whichever of the four shapes it has.

        The async shape is driven under
        :func:`~dataknobs_common.async_iter.aclosing_iter` --- see
        :meth:`AliasSource.stream_items`, which states why for the whole
        family. It applies here most of all: the documented use is *"a
        generator over an API, a file being parsed, a queue being drained"*,
        and all three hold something a close is what returns.
        """
        produced = self.fn()
        if hasattr(produced, "__await__"):
            produced = await produced
        if hasattr(produced, "__aiter__"):
            async with aclosing_iter(produced) as rows:
                async for item in rows:
                    yield item
            return
        for item in produced:
            yield item


@dataclass(frozen=True, eq=False)
class AliasSource:
    """One entity, many surface forms, one id --- a decorator over any source.

    An entity named "Acme Corp" with aliases ``["ACME", "Acme Corporation"]``
    is three things a person might type and one thing the vocabulary names.
    Indexing only the canonical name means alias matching becomes a separate
    lookup table that drifts from the entity store; indexing the forms as
    separate entities means a hit resolves to something that does not exist.
    So this yields one item per surface form, **all carrying the inner item's
    id**.

    **It decorates ``text`` and nothing else.** The id, the metadata and the
    declared sets are the inner source's, forwarded unchanged. A decorator
    that re-declares is a second translation site for a scope filter, and two
    translation sites start disagreeing; a decorator that rewrites an id
    breaks the one guarantee a hit carries. It decorates text; it does not
    decorate identity or scope.

    The forms are read from the inner item's own metadata, under
    :attr:`aliases_field`, so the leaf source is what decides which key
    carries them and this class needs no vocabulary of its own. An item whose
    metadata has no such key, or whose value is empty, yields its own text
    alone --- an entity with no aliases is not an error.

    .. warning::

        **A store keyed on id holds one row per id, so the forms collide
        there.** This class produces N items for one entity by design, and a
        vector store upserts on id conflict --- measured: three items sharing
        an id leave one row, carrying the *last* form rather than the
        canonical name. Whether the extra rows need vector ids distinct from
        the entity id, and under what grammar, is not settled by any ruling
        this class could read, so it is not decided here. Use this source
        where the reader de-duplicates by id, or where the store is keyed on
        something else, until it is.

    Example:
        ```python
        source = AliasSource(EntitySourceIndexSource(ontology), "dk_alias_forms")
        ```
    """

    #: The source whose items are decorated. Typed as the protocol declared
    #: in this module rather than ``Any``: it is ``runtime_checkable``, it is
    #: right here, and the one place a decorator can be handed something that
    #: is not a source is the one place the annotation is worth having.
    inner: AsyncIndexSource

    #: The metadata key on an inner item whose value is a sequence of surface
    #: forms. Published rather than assumed, because the leaf source writes
    #: it and this one reads it --- one key, two ends, and a literal at either
    #: end is how the two stop agreeing.
    aliases_field: str

    def declares(self) -> frozenset[str]:
        """The inner source's declared sets, unchanged."""
        declared: frozenset[str] = self.inner.declares()
        return declared

    async def stream_items(self) -> AsyncIterator[IndexItem]:
        """The inner item, then one per surface form it names.

        The canonical text first, so a reader that keeps the first of a
        collision keeps the entity's own name rather than whichever alias the
        source happened to list last.

        **Driven under :func:`~dataknobs_common.async_iter.aclosing_iter`,
        which is the rule for every source in this family that drives
        another.** A bare ``async for`` leaves the inner iterator suspended
        when this one is closed, so the inner's cleanup waits for the
        interpreter to finalize it --- a later turn of the loop, unordered
        against whatever the consumer does next. Both things a leaf reaches at its close are then lost to
        that gap: ``EntitySourceIndexSource`` reports an all-empty stream
        from a ``finally``, and ``RecordFieldSource`` over PostgreSQL yields
        from inside an acquired connection and an open transaction.
        Measured with this class in front of the ontology adapter --- the
        composition an ``index:`` block with ``aliases: true`` builds --- the
        leaf's report arrived after the consumer had already handled the
        failure instead of with it.
        """
        async with aclosing_iter(self.inner.stream_items()) as items:
            async for item in items:
                yield item
                for form in self._forms_of(item):
                    if not form or form == item.text:
                        continue
                    yield IndexItem(id=item.id, text=form, metadata=dict(item.metadata))

    def _forms_of(self, item: IndexItem) -> tuple[str, ...]:
        """The surface forms on one item, reading the key's published rule.

        The alias key is **list-valued**, and the rule its family states is
        that *a reader takes a bare string as one form rather than as its
        characters*. This loop iterated the value directly, so a metadata
        value of ``"ACME"`` produced four one-character items --- all under
        the entity's id, which a store keyed on id upserts into one row
        holding ``"E"``. The in-tree writer always writes a ``list``, so the
        only population that could reach it is the consumer-supplied source
        this class is documented to decorate: the one that cannot read the
        constant's docstring at the point of failure.

        A value that is neither a string nor a sequence of them --- a mapping,
        a number --- is not a malformed list to be salvaged. Iterating a
        mapping yields its keys and rendering a number yields a digit, and
        both would be embedded as surface forms under the entity's id. It is
        logged and dropped, so the entity keeps its canonical text.
        """
        value = item.metadata.get(self.aliases_field)
        if value is None or value == "":
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, MappingABC) or not hasattr(value, "__iter__"):
            logger.warning(
                "alias field %r on item %r holds a %s; it is list-valued, so the value "
                "is ignored and the entity keeps its canonical text alone",
                self.aliases_field,
                item.id,
                type(value).__name__,
            )
            return ()
        return tuple(str(form) for form in value)
