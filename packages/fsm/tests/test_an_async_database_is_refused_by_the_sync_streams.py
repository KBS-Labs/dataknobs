"""The database stream classes are sync, so they take a sync database.

All three declared ``database: Union[SyncDatabase, AsyncDatabase]`` while being
sync themselves --- ``IStreamSource`` and ``IStreamSink`` are sync Protocols and
the package ships no async database source or sink, so the union was a promise
nothing kept. An async database passed to them lost every row in silence, in
both directions:

* ``write_chunk`` returned ``True`` having written nothing, because ``create()``
  returned a coroutine that was discarded and never awaited.
* ``read_chunk``'s ``self.database.search(...)`` was a coroutine --- truthy, so
  ``if not records`` did not fire --- and ``records[-1]`` raised ``TypeError``
  into the trailing ``except Exception``, which returned an empty chunk marked
  ``is_last``. Iterating yielded that once and stopped, so the caller saw a
  stream that ended normally.

Measured before the fix against a ``SyncMemoryDatabase`` control: the sync run
wrote 2 rows and read back 2; the async run returned ``True`` from the same
``write_chunk`` call, wrote 0, and read back 0.

The type is narrowed and the refusal is explicit because an annotation alone
stops nobody --- the package is not type-checked by its consumers, and this
failure is silent rather than loud.
"""

from __future__ import annotations

import pytest

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase

from dataknobs_fsm.streaming import (
    DatabaseBulkLoader,
    DatabaseStreamSink,
    DatabaseStreamSource,
)
from dataknobs_fsm.streaming.core import StreamChunk

CONSTRUCTORS = [DatabaseStreamSource, DatabaseStreamSink, DatabaseBulkLoader]
IDS = [cls.__name__ for cls in CONSTRUCTORS]


@pytest.mark.parametrize("constructor", CONSTRUCTORS, ids=IDS)
def test_an_async_database_is_refused(constructor: type) -> None:
    """Every one of the three, because the union was on all three."""
    with pytest.raises(ConfigurationError) as caught:
        constructor(AsyncMemoryDatabase())

    message = str(caught.value)
    assert "AsyncMemoryDatabase" in message, "the message must name what was passed"
    assert "sync" in message.lower(), "and say what it needed instead"


@pytest.mark.parametrize("constructor", CONSTRUCTORS, ids=IDS)
def test_a_sync_database_is_accepted(constructor: type) -> None:
    """The control: narrowing must not refuse the databases that always worked."""
    assert constructor(SyncMemoryDatabase()) is not None


def test_a_sync_database_still_round_trips() -> None:
    """End to end, so the refusal is not the only thing the guard proves."""
    database = SyncMemoryDatabase()

    sink = DatabaseStreamSink(database=database, batch_size=2)
    assert sink.write_chunk(
        StreamChunk(data=[{"id": "a", "v": 1}, {"id": "b", "v": 2}], chunk_id="0", is_last=True)
    )

    read = [
        row
        for chunk in DatabaseStreamSource(database=database, batch_size=10)
        for row in chunk.data
    ]

    assert len(read) == 2, "a sync database writes two rows and reads two back"
