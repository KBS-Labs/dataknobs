# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Every consumer of ``stream_read`` in this package closes the read it opened.

An async generator a consumer walks away from --- a ``break``, a raise, an
early return --- is left suspended at its ``yield``, and its ``finally`` runs
only when the interpreter finalizes it: a later turn of the loop, unordered
against whatever the consumer does next. ``AsyncPostgresDatabase.stream_read``
yields from inside an acquired pool connection and an open transaction, so
that gap is a connection the pool does not have back.

The rule these tests pin is one sentence: **a frame closes the stream it
opened, and a generator closes the stream it drives.** Each test asserts
``held == 0`` at the point the consumer finished rather than eventually.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest import mock

import pytest
from dataknobs_common.async_iter import aclosing_iter
from dataknobs_data import Query
from dataknobs_data.testing import HoldingStreamDatabase
from dataknobs_fsm.core.exceptions import ETLError
from dataknobs_fsm.io.adapters import AsyncDatabaseProvider, AsyncFileProvider
from dataknobs_fsm.io.base import IOConfig, IOFormat, IOMode
from dataknobs_fsm.patterns import etl as etl_module
from dataknobs_fsm.patterns.etl import DatabaseETL
from dataknobs_fsm.resources.database import AsyncDatabaseResourceAdapter

ROWS = [{"id": str(index)} for index in range(6)]


def _io_config(source: Any) -> IOConfig:
    return IOConfig(mode=IOMode.READ, format=IOFormat.JSON, source=source)


def _database_provider(database: HoldingStreamDatabase) -> AsyncDatabaseProvider:
    provider = AsyncDatabaseProvider(_io_config({"backend": "memory"}))
    provider.db = database
    return provider


async def test_etl_batching_closes_the_read_it_opened() -> None:
    """The extract generator drives the source read, so it owns closing it.

    Exercised directly because that is where the rule lives: any consumer of
    the batches --- the FSM run, a checkpoint that gives up, a caller taking a
    sample --- leaves the source read suspended if this frame does not close
    it.
    """
    database = HoldingStreamDatabase(ROWS)
    etl = DatabaseETL.from_config(
        {
            "source_db": {"backend": "memory"},
            "target_db": {"backend": "memory"},
            "batch_size": 2,
        }
    )

    async with aclosing_iter(etl._extract_batches(database, Query())) as batches:
        async for _batch in batches:
            break

    assert database.held == 0, "the extract generator left the source read open"


async def test_an_etl_run_closes_the_extract_stream_when_it_gives_up() -> None:
    """The threshold check raises out of the loop, and the ``finally`` then closes the source.

    The order is what makes this more than an abandoned generator: without
    the close, ``source_db.close()`` runs while a read on that database is
    still suspended inside it. The extract generator closing its own read is
    not enough --- nothing closes *it* unless this frame does, which is the
    same defect one frame up.

    The source is substituted at the factory rather than the object: what
    runs is the real ``run()`` over a real ``AsyncDatabase``, and nothing
    here stands in for a code path.
    """
    database = HoldingStreamDatabase(ROWS)

    async def _from_backend(_backend: str, _config: Any) -> HoldingStreamDatabase:
        return database

    etl = DatabaseETL.from_config(
        {
            "source_db": {"type": "memory"},
            "target_db": {"type": "memory"},
            "batch_size": 2,
            "error_threshold": -1.0,
        }
    )

    with mock.patch.object(etl_module.AsyncDatabase, "from_backend", _from_backend):
        with pytest.raises(ETLError):
            await etl.run()

    assert database.held == 0, "the run left its source read open while closing the source"


async def test_a_provider_stream_closes_the_read_it_opened() -> None:
    """``AsyncDatabaseProvider.stream_read`` is a generator over another."""
    database = HoldingStreamDatabase(ROWS)

    async with aclosing_iter(_database_provider(database).stream_read(None)) as records:
        async for _record in records:
            break

    assert database.held == 0, "the provider's stream left the database's open"


async def test_provider_batching_closes_the_read_it_opened() -> None:
    """Two delegating frames over one read, so both have to say it."""
    database = HoldingStreamDatabase(ROWS)
    provider = _database_provider(database)

    async with aclosing_iter(provider.batch_read(None, batch_size=2)) as batches:
        async for _batch in batches:
            break

    assert database.held == 0, "the provider's batching left the database read open"


async def test_file_batching_closes_the_stream_it_drives() -> None:
    """The file provider's batching drives ``stream_read``, which is an override point.

    The shipped file ``stream_read`` holds only the provider's own handle,
    which ``close()`` owns --- so what this pins is the contract rather than a
    leak in this class. It is the same three characters as its database
    sibling, where the stream below *is* a database read, and a rule the twins
    state differently is one a reader cannot rely on.
    """

    class HoldingFileProvider(AsyncFileProvider):
        def __init__(self, config: IOConfig) -> None:
            super().__init__(config)
            self.held = 0

        async def stream_read(self, **kwargs: Any) -> AsyncIterator[Any]:
            self.held += 1
            try:
                for row in ROWS:
                    yield row
            finally:
                self.held -= 1

    provider = HoldingFileProvider(_io_config("/dev/null"))

    async with aclosing_iter(provider.batch_read(batch_size=2)) as batches:
        async for _batch in batches:
            break

    assert provider.held == 0, "the file provider's batching left its stream open"


async def test_a_fetch_one_query_closes_the_read_it_opened() -> None:
    """``fetch_one`` breaks at the first row, by construction and on success.

    Not an error path: this is what an ordinary answer costs, on every call.
    """
    database = HoldingStreamDatabase(ROWS)
    adapter = AsyncDatabaseResourceAdapter(name="r", backend="memory")
    adapter._database = database

    answered = await adapter.execute_query(fetch_one=True)
    assert answered["id"] == "0", "fetch_one answered from somewhere other than the read"
    assert database.held == 0, "the fetch_one read was left open"
