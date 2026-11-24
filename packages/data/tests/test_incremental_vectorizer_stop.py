"""Stopping an idle vectorizer is immediate, and loses no queued record.

A worker waiting on an empty queue looked at the shutdown event once a second,
so ``stop()`` took up to that long. Its wait now races the shutdown, as every
other wait in the class does.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.vector.migration import IncrementalVectorizer


async def _embed(text: str) -> np.ndarray:
    return np.array([float(len(text)), 1.0, 2.0])


def _vectorizer(db: AsyncMemoryDatabase, max_workers: int = 4) -> IncrementalVectorizer:
    return IncrementalVectorizer(
        database=db,
        embedding_fn=_embed,
        text_fields="content",
        vector_field="embedding",
        max_workers=max_workers,
    )


async def test_an_idle_vectorizer_stops_at_once() -> None:
    vectorizer = _vectorizer(AsyncMemoryDatabase(config={"vector_enabled": True}))
    await vectorizer.start()
    await asyncio.sleep(0.05)  # every worker is now waiting on the queue

    loop = asyncio.get_running_loop()
    started = loop.time()
    await vectorizer.stop()

    assert loop.time() - started < 0.5


@pytest.mark.parametrize("shutdown_first", [True, False])
async def test_a_record_arriving_with_the_shutdown_is_processed_or_left_queued(
    shutdown_first: bool,
) -> None:
    """Both ends of the race are ready at once; the record goes to one of them."""
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    await db.create(Record(data={"content": "a document"}))
    [record] = await db.all()
    vectorizer = _vectorizer(db)
    worker = asyncio.create_task(vectorizer._worker(0))
    await asyncio.sleep(0.05)  # the worker is now waiting on the queue

    if shutdown_first:
        vectorizer._shutdown_event.set()
        vectorizer._queue.put_nowait(record)
    else:
        vectorizer._queue.put_nowait(record)
        vectorizer._shutdown_event.set()
    await asyncio.wait_for(worker, timeout=5.0)

    processed = vectorizer._stats["processed"]
    assert processed + vectorizer._queue.qsize() == 1
    [stored] = await db.all()
    assert ("embedding" in stored.fields) is bool(processed)


async def test_a_restarted_vectorizer_finishes_what_a_stop_left() -> None:
    """No record is taken and dropped: a queue that owed one could never join."""
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    for i in range(20):
        await db.create(Record(data={"content": f"document {i}"}))
    vectorizer = _vectorizer(db, max_workers=2)
    await vectorizer.start()
    await asyncio.sleep(0)
    await vectorizer.stop()

    await vectorizer.start()
    try:
        assert await vectorizer.wait_for_completion(timeout=10.0) is True
    finally:
        await vectorizer.stop()
    assert all("embedding" in r.fields for r in await db.all())
