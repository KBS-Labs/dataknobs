# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Index sources over a database table --- the bare-table case.

Two of the reference implementations of
:class:`~dataknobs_common.index.AsyncIndexSource`, and the two that hold a
backend. They live in ``dataknobs-data`` rather than beside the protocol for
the reason every placement in this family follows: a value type or a pure
algorithm over one belongs in ``dataknobs-common``; a concrete that opens a
connection belongs here.

**They emit local ids.** A row's own id is what a hit resolves to, which is
correct for a table read on its own terms and is a deliberate boundary rather
than an oversight --- qualification happens in the source, and a source
constructed over a bare table was not given a namespace to qualify into. A
reader that needs qualified ids indexes the vocabulary rather than the table.

Plural on purpose: the singular reads as *the* index, and neither of these is
one. Three files, three dependencies --- these hold a database, the index
holds a store, the adapter holds a vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dataknobs_common.index import IndexItem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from dataknobs_data.database import AsyncDatabase
    from dataknobs_data.query import Query

__all__ = ["MultiFieldSource", "RecordFieldSource"]

#: What goes between two field values when more than one is read.
#:
#: An em dash surrounded by spaces, which is what the design fence writes and
#: what reads naturally inside an embedded sentence.
DEFAULT_JOIN = " — "


def join_non_empty(values: Sequence[Any], join: str) -> str:
    """The non-empty values, joined --- and no dangling separator over one.

    One helper and two callers rather than two spellings, because the property
    it carries is the one that is easy to get wrong in the same way twice: a
    row with a name and no description must produce the name, not the name
    followed by a separator. Dropping before the join is what makes that true
    by construction rather than by a strip afterwards.

    Args:
        values: The values to join, in order. ``None`` and empty strings are
            dropped; everything else is rendered with ``str``.
        join: What to put between two surviving values.

    Returns:
        The surviving values joined, or ``""`` where none survived.
    """
    rendered = [str(value).strip() for value in values if value is not None]
    return join.join(value for value in rendered if value)


@dataclass(frozen=True)
class RecordFieldSource:
    """One text per row, read from one field of a table.

    The named case --- *"a database column"* --- and the smallest live source
    there is.

    Example:
        ```python
        source = RecordFieldSource(db, "title")
        ```

    Attributes:
        database: The table to stream. Streamed rather than read whole,
            because a source exists to be larger than memory.
        field: Which field carries the text.
        query: An optional narrowing. A source over half a table is a
            legitimate thing to index, and expressing it as a query rather
            than as a filter callback keeps the narrowing in the backend.
        declared: The named sets these ids fall in. Empty by default: a bare
            table's ids are local and fall in no named space.
    """

    database: AsyncDatabase
    field: str
    query: Query | None = None
    declared: frozenset[str] = frozenset()

    @property
    def source_field(self) -> str:
        """What this source composed its text from, as one name.

        Not a protocol member --- the protocol has two, and a required third
        would break ``isinstance`` for every structural implementor outside
        this tree. An index reads this with a fall-back and passes it to the
        store, which writes it beside the source text it already writes; a
        source that cannot say goes without, and its rows are simply not
        distinguishable from any other caller's.
        """
        return self.field

    def declares(self) -> frozenset[str]:
        """Whatever this source was constructed to declare."""
        return self.declared

    async def stream_items(self) -> AsyncIterator[IndexItem]:
        """One item per row that has an id and non-empty text.

        A row missing the field, or holding blank text in it, is skipped
        rather than yielded empty: an embedded empty string is a vector that
        matches everything weakly and nothing well, which is worse in a corpus
        than an absence.
        """
        async for record in self.database.stream_read(self.query):
            if record.id is None:
                continue
            text = join_non_empty([record.get_value(self.field)], DEFAULT_JOIN)
            if text:
                yield IndexItem(id=str(record.id), text=text)


@dataclass(frozen=True)
class MultiFieldSource:
    """One text per row, composed from several fields of a table.

    The same class of thing as :class:`RecordFieldSource` over more than one
    column --- a name and a description embedded together, so a query matching
    either reaches the row.

    Example:
        ```python
        source = MultiFieldSource(db, ["title", "summary"])
        ```

    Attributes:
        database: As :attr:`RecordFieldSource.database`.
        fields: Which fields carry the text, in order.
        join: What to put between two non-empty values. Applied **between**
            them, so a row carrying one of two produces no dangling separator.
        query: As :attr:`RecordFieldSource.query`.
        declared: As :attr:`RecordFieldSource.declared`.
    """

    database: AsyncDatabase
    fields: Sequence[str]
    join: str = DEFAULT_JOIN
    query: Query | None = None
    declared: frozenset[str] = frozenset()

    @property
    def source_field(self) -> str:
        """What this source composed its text from, as one name.

        Not a protocol member --- the protocol has two, and a required third
        would break ``isinstance`` for every structural implementor outside
        this tree. An index reads this with a fall-back and passes it to the
        store, which writes it beside the source text it already writes; a
        source that cannot say goes without, and its rows are simply not
        distinguishable from any other caller's.
        """
        return self.join.join(self.fields)

    def declares(self) -> frozenset[str]:
        """Whatever this source was constructed to declare."""
        return self.declared

    async def stream_items(self) -> AsyncIterator[IndexItem]:
        """One item per row whose chosen fields compose to something."""
        async for record in self.database.stream_read(self.query):
            if record.id is None:
                continue
            text = join_non_empty(
                [record.get_value(name) for name in self.fields],
                self.join,
            )
            if text:
                yield IndexItem(id=str(record.id), text=text)
