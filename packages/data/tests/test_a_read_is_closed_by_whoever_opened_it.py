# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Every consumer of ``stream_read`` in this package closes the read it opened.

An async generator a consumer walks away from --- a ``break``, a raise, an
early return --- is left suspended at its ``yield``, and its ``finally`` runs
only when the interpreter finalizes it: a later turn of the loop, unordered
against whatever the consumer does next. Two shipped backends hold a real
resource across those yields, so for them the gap is an acquired pool
connection inside an open transaction, or an uncleared scroll.

The rule these tests pin is one sentence: **a frame closes the stream it
opened, and a generator closes the stream it drives.** Each test asserts
``held == 0`` at the point the consumer finished rather than eventually,
because "released eventually" is what the unfixed code already does.
"""

from __future__ import annotations

from typing import Any

import pytest
from dataknobs_common.async_iter import aclosing_iter
from dataknobs_data import Query, Record
from dataknobs_data.keyed_store import AsyncKeyedRecordStore
from dataknobs_data.migration.migrator import Migrator
from dataknobs_data.ontology.hierarchy import ColumnHierarchy
from dataknobs_data.ontology.sources import EntityProjection, RecordEntitySource
from dataknobs_data.streaming import StreamConfig, StreamResult
from dataknobs_data.testing import HoldingStreamDatabase

ROWS = [
    {"id": "a", "parent": "root"},
    {"id": "b", "parent": "a"},
    {"id": "c", "parent": "a"},
]


async def test_stream_transform_closes_the_read_it_opened() -> None:
    """The default on ``AsyncDatabase``, so every backend inherits the answer.

    A delegating generator: closing it throws ``GeneratorExit`` at its own
    ``yield``, which leaves the loop --- and a bare ``async for`` does not
    close what it was iterating, so the read below it stays suspended.

    Asserted **at the close** rather than after it, because "released
    eventually" is what the unfixed code already does. The close has to
    arrive from above: a consumer that abandons this generator too leaves
    the whole chain suspended, however each link drives the next.
    """
    database = HoldingStreamDatabase(ROWS)

    async with aclosing_iter(database.stream_transform()) as transformed:
        async for _record in transformed:
            break
        assert database.held == 1, "the read is open while the transform is being read"

    assert database.held == 0, "stream_transform did not close the read below it"


async def test_a_keyed_store_stream_closes_the_read_it_opened() -> None:
    """Same shape one layer up: the store's ``stream`` drives the database's."""
    database = HoldingStreamDatabase(ROWS)
    store: AsyncKeyedRecordStore[str] = AsyncKeyedRecordStore(
        database,
        serializer=lambda value: ({"id": value}, {}),
        deserializer=lambda record: str(record.get_value("id")),
    )

    async with aclosing_iter(store.stream()) as values:
        async for _value in values:
            break

    assert database.held == 0, "the store's stream did not close the database's"


async def test_a_bounded_edge_read_closes_the_read_it_opened() -> None:
    """``contains`` breaks at the first edge, by construction and on success.

    Not an error path: this is what an ordinary True answer costs. The read
    carries ``limit=1`` as well, so the generator is suspended immediately
    after the row it was asked for --- a bound on the query does not finish it.
    """
    database = HoldingStreamDatabase(ROWS)
    axis = ColumnHierarchy(database=database, table="t", child="id", parent="parent")

    assert await axis.contains("a") is True
    assert database.held == 0, "the bounded edge read was left open"
    assert database.opens == 1, "a True on the first read should not issue a second"


async def test_a_type_scan_closes_the_read_when_a_row_will_not_project() -> None:
    """The scan runs to exhaustion ordinarily; a row that will not read abandons it."""

    class Armed(Record):
        """Ordinary while the stream builds it, raising once the scan reads it."""

        _armed = False

        def get_value(self, field: str, default: Any = None) -> Any:
            if self._armed:
                raise RuntimeError("row is malformed")
            return super().get_value(field, default)

    class Malformed(HoldingStreamDatabase):
        async def stream_read(self, query: Query | None = None, config: Any = None) -> Any:
            self.held += 1
            self.opens += 1
            try:
                for row in self.rows:
                    record = Armed(data=dict(row))
                    record._armed = True
                    yield record
            finally:
                self.held -= 1

    database = Malformed(ROWS)
    source = RecordEntitySource(
        database=database,
        projection=EntityProjection(table="t", id="id", const_type="thing"),
        source_id="s",
    )

    with pytest.raises(RuntimeError):
        await source.by_type("thing")

    assert database.held == 0, "the abandoned scan left its read open"


async def test_a_migration_closes_the_source_when_the_target_stops_early() -> None:
    """``stream_write`` stops on a failed batch, which abandons the whole chain.

    The source read is the migration's own, so the migration is what closes
    it: a target is handed an iterator it did not open and does not own.
    """

    class StopsAfterOne(HoldingStreamDatabase):
        async def stream_write(self, records: Any, config: Any = None) -> StreamResult:
            async for _record in records:
                break
            return StreamResult(successful=1, failed=0, skipped=0, errors=[])

    source = HoldingStreamDatabase(ROWS)
    target = StopsAfterOne()
    await source.connect()
    await target.connect()

    await Migrator().migrate_async(source, target, config=StreamConfig())

    assert source.held == 0, "the migration left its source read open"
