# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""The registry adapter's ``stream`` closes the store read below it.

A delegating async generator: closing it throws ``GeneratorExit`` at its own
``yield``, which leaves the loop --- and a bare ``async for`` does not close
what it was iterating. Three links are involved here, and each has to say it
or the chain breaks at whichever one does not: this adapter's ``stream``, the
keyed store's ``stream``, and the database's ``stream_read``. On a
Postgres-backed registry the read at the bottom holds a pooled connection
inside an open transaction, so a caller that takes one page and stops holds a
connection until the interpreter finalizes three abandoned generators.
"""

from __future__ import annotations

from dataknobs_common.async_iter import aclosing_iter
from dataknobs_data.testing import HoldingStreamDatabase

from dataknobs_bots.registry.adapter import DataKnobsRegistryAdapter

ROWS = [{"bot_id": f"bot-{index}", "config": {}, "status": "active"} for index in range(5)]


async def test_a_registry_stream_closes_the_read_it_opened() -> None:
    """Taking one page and stopping is the expected use, not an error path."""
    database = HoldingStreamDatabase(ROWS)
    adapter = DataKnobsRegistryAdapter(database=database)
    await adapter.initialize()

    async with aclosing_iter(adapter.stream()) as registrations:
        async for registration in registrations:
            assert registration.bot_id == "bot-0"
            break
        assert database.held == 1, "the read is open while the registry is being streamed"

    assert database.held == 0, "the registry stream left the database read open"

    await adapter.close()
