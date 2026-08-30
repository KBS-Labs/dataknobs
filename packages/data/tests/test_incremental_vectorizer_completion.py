"""``wait_for_completion`` waits for the work, not for the queue to look empty.

It read ``self._queue.qsize()`` and returned as soon as that was zero. Two
distinct states satisfy that and neither means "done":

* **Nothing has been enqueued yet.** ``start()`` creates the loader as a task;
  the caller's very next ``await wait_for_completion()`` runs before the loader
  has issued its first query. Measured before the fix: it returns immediately,
  with zero records vectorized, on a database full of pending ones.
* **Everything is in flight.** A worker takes a record off the queue and then
  embeds and writes it. Between those, ``qsize()`` is zero while the work is
  not done.

The fix gives the queue the two halves of the standard contract --- workers
call ``task_done()``, waiters call ``join()`` --- and adds the piece
``join()`` alone cannot express: whether the *source* is drained, which is the
loader's knowledge, not the queue's.

Prior art: ``8309ef37`` waived ``ASYNC110`` for this module for a "cheap 5s
queue-drain poll", noting the idiomatic ``Queue.join()``/``task_done()``
migration was tracked separately. This completes that intent, and the waiver
comes out with it.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.vector.migration import IncrementalVectorizer

CORPUS_SIZE = 12


class SlowEmbedder:
    """An embedder with a real delay, so "in flight" is a state that exists.

    Without one, a worker takes a record and finishes it within the same
    scheduling slot, and the in-flight window this module is about never opens
    wide enough to observe.
    """

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.calls = 0

    async def __call__(self, text: str) -> np.ndarray:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return np.array([float(len(text)), 1.0, 2.0])


async def _corpus() -> AsyncMemoryDatabase:
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    for i in range(CORPUS_SIZE):
        await db.create(Record(data={"content": f"document number {i}"}))
    return db


async def _vectorized_count(db: AsyncMemoryDatabase) -> int:
    return sum(1 for record in await db.all() if "embedding" in record.fields)


class TestItDoesNotReturnBeforeTheWorkStarts:
    """The reproduce cell: called right after ``start()``, on a full corpus."""

    async def test_returns_only_once_every_record_is_vectorized(self) -> None:
        db = await _corpus()
        embedder = SlowEmbedder()
        vectorizer = IncrementalVectorizer(
            database=db,
            embedding_fn=embedder,
            text_fields=["content"],
            vector_field="embedding",
            batch_size=5,
            max_workers=2,
        )

        await vectorizer.start()
        try:
            completed = await vectorizer.wait_for_completion(timeout=20.0)
        finally:
            await vectorizer.stop(timeout=5.0)

        assert completed is True, "wait_for_completion timed out"
        assert await _vectorized_count(db) == CORPUS_SIZE, (
            "wait_for_completion returned while records were still unvectorized"
        )


class TestItDoesNotReturnWhileRecordsAreInFlight:
    """The second state ``qsize()`` cannot distinguish.

    Drives one record with a long embed so the queue is empty for most of the
    call while the work is emphatically not done.
    """

    async def test_a_single_slow_record_is_waited_for(self) -> None:
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        await db.create(Record(data={"content": "the only document"}))

        embedder = SlowEmbedder(delay=0.3)
        vectorizer = IncrementalVectorizer(
            database=db,
            embedding_fn=embedder,
            text_fields=["content"],
            vector_field="embedding",
            max_workers=1,
        )

        await vectorizer.start()
        try:
            # Let the worker take the record off the queue. A tenth of the
            # embedder's 0.3s delay, so the record is reliably taken and just
            # as reliably unfinished when the wait below begins.
            await asyncio.sleep(0.03)
            assert vectorizer.get_stats()["queue_size"] == 0, "not taken yet"

            completed = await vectorizer.wait_for_completion(timeout=20.0)
        finally:
            await vectorizer.stop(timeout=5.0)

        assert completed is True
        assert await _vectorized_count(db) == 1


class TestATimeoutIsReportedRatherThanHung:
    """It used to have no timeout at all, so a stalled worker hung the caller."""

    async def test_returns_false_when_the_work_cannot_finish(self) -> None:
        db = await _corpus()
        # 10s per record against a 0.2s budget: the deadline arrives first.
        vectorizer = IncrementalVectorizer(
            database=db,
            embedding_fn=SlowEmbedder(delay=10.0),
            text_fields=["content"],
            vector_field="embedding",
            max_workers=1,
        )

        await vectorizer.start()
        try:
            completed = await asyncio.wait_for(
                vectorizer.wait_for_completion(timeout=0.2), timeout=5.0
            )
        finally:
            # A short grace period on purpose: the worker is parked inside a
            # 10s embed, and how long `stop` waits before cancelling it is not
            # what this cell is about.
            await vectorizer.stop(timeout=0.1)

        assert completed is False


class TestAnEmptyCorpusCompletesImmediately:
    """A companion: the fix must not turn "nothing to do" into a hang.

    Passes before and after; it is what stops the obvious over-correction ---
    waiting for a record count that will never arrive.
    """

    async def test_no_pending_records_is_completion(self) -> None:
        db = AsyncMemoryDatabase(config={"vector_enabled": True})
        vectorizer = IncrementalVectorizer(
            database=db,
            embedding_fn=SlowEmbedder(),
            text_fields=["content"],
            vector_field="embedding",
        )

        await vectorizer.start()
        try:
            completed = await asyncio.wait_for(
                vectorizer.wait_for_completion(timeout=10.0), timeout=15.0
            )
        finally:
            await vectorizer.stop(timeout=5.0)

        assert completed is True


class TestShutdownReleasesAWaiter:
    """A companion: ``stop()`` must not strand a caller inside the wait."""

    async def test_a_waiter_returns_when_the_vectorizer_stops(self) -> None:
        db = await _corpus()
        vectorizer = IncrementalVectorizer(
            database=db,
            embedding_fn=SlowEmbedder(delay=10.0),
            text_fields=["content"],
            vector_field="embedding",
            max_workers=1,
        )

        await vectorizer.start()
        waiter = asyncio.create_task(vectorizer.wait_for_completion(timeout=30.0))
        await asyncio.sleep(0.05)

        await vectorizer.stop(timeout=0.1)

        completed = await asyncio.wait_for(waiter, timeout=5.0)
        assert completed is False, "shutdown reported as completion"


class TestEveryRecordIsVectorizedExactlyOnce:
    """A companion for the ``task_done()`` change, guarding the drain rewrite.

    ``_load_queue`` waited for the queue to *empty* before re-querying, which
    left a window where a record taken but not yet written came back from the
    next query and was embedded twice. Joining instead of polling ``qsize()``
    closes it, and this is what says so.
    """

    async def test_no_record_is_embedded_twice(self) -> None:
        db = await _corpus()
        embedder = SlowEmbedder(delay=0.05)
        vectorizer = IncrementalVectorizer(
            database=db,
            embedding_fn=embedder,
            text_fields=["content"],
            vector_field="embedding",
            batch_size=4,
            max_workers=3,
        )

        await vectorizer.start()
        try:
            assert await vectorizer.wait_for_completion(timeout=20.0) is True
        finally:
            await vectorizer.stop(timeout=5.0)

        assert await _vectorized_count(db) == CORPUS_SIZE
        assert embedder.calls == CORPUS_SIZE, (
            f"{embedder.calls} embeddings for {CORPUS_SIZE} records"
        )


@pytest.mark.parametrize("workers", [1, 4])
class TestStatsAgreeWithTheDatabase:
    """The counters a caller reads after waiting describe what actually happened."""

    async def test_processed_matches_the_corpus(self, workers: int) -> None:
        db = await _corpus()
        vectorizer = IncrementalVectorizer(
            database=db,
            embedding_fn=SlowEmbedder(delay=0.01),
            text_fields=["content"],
            vector_field="embedding",
            batch_size=5,
            max_workers=workers,
        )

        await vectorizer.start()
        try:
            assert await vectorizer.wait_for_completion(timeout=20.0) is True
            stats = vectorizer.get_stats()
        finally:
            await vectorizer.stop(timeout=5.0)

        assert stats["processed"] == CORPUS_SIZE
        assert stats["failed"] == 0
