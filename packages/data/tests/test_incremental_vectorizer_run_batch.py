"""``run_batch`` has to be able to stop.

Its loop was ``while self._stats["processed"] < (limit or float("inf"))`` with
one ``break``, and the break required ``self._processing_task.done()``. That
task is the loader, whose own loop runs ``while not
self._shutdown_event.is_set()`` and *idles* rather than returning when the
source drains --- and the only thing that sets the event is ``stop()``, which
``run_batch`` calls after the loop. So the break could never be taken:

* ``run_batch()`` --- the documented no-argument form --- never returned,
  because ``None or float("inf")`` is infinity.
* ``run_batch(0)`` never returned either, for the same reason via ``or``.
* ``run_batch(n)`` never returned whenever fewer than ``n`` records succeeded,
  because a failure bumps ``failed`` and not ``processed``.

Every cell here is bracketed by ``asyncio.wait_for``: without it a regression
does not fail this module, it hangs the suite.

The second half is what the loop was polling with. ``Queue.empty()`` goes
false the moment a worker *takes* the last record, before it has embedded or
written it --- the same distinction ``wait_for_completion`` was rewritten
around one method above, using ``join()``/``task_done()`` and a
``_source_drained`` event. ``TestNothingIsCountedBeforeItIsWritten`` is what
holds the two methods to one answer.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.vector.migration import IncrementalVectorizer

CORPUS_SIZE = 6

# Long enough that a regression is unmistakable, short enough that a hung test
# is not the slowest thing in the suite.
DEADLINE = 5.0


class SlowEmbedder:
    """A delay, so "taken but not yet written" is a state that exists.

    Without one a worker finishes a record inside the scheduling slot it took
    it in, and the in-flight window ``empty()`` misreads never opens.
    """

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.calls = 0

    async def __call__(self, text: str) -> np.ndarray:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return np.array([float(len(text)), 1.0, 2.0])


class NeverReturns:
    """An embedder that hangs, so a timeout is the only way out."""

    async def __call__(self, text: str) -> np.ndarray:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def _corpus(size: int = CORPUS_SIZE) -> AsyncMemoryDatabase:
    db = AsyncMemoryDatabase(config={"vector_enabled": True})
    for i in range(size):
        await db.create(Record(data={"content": f"document number {i}"}))
    return db


async def _vectorized_count(db: AsyncMemoryDatabase) -> int:
    return sum(1 for record in await db.all() if "embedding" in record.fields)


def _vectorizer(db: AsyncMemoryDatabase, **kwargs: object) -> IncrementalVectorizer:
    return IncrementalVectorizer(
        database=db,
        embedding_fn=kwargs.pop("embedding_fn", SlowEmbedder()),  # type: ignore[arg-type]
        text_fields=["content"],
        **kwargs,  # type: ignore[arg-type]
    )


class TestTheNoArgumentFormReturns:
    """The reproduce cell. ``run_batch()`` is the form the class docstring
    advertises a sibling of, and it never returned.
    """

    @pytest.mark.asyncio
    async def test_it_returns_once_the_source_is_drained(self) -> None:
        db = await _corpus()
        vectorizer = _vectorizer(db)

        result = await asyncio.wait_for(vectorizer.run_batch(), timeout=DEADLINE)

        assert result.processed == CORPUS_SIZE
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_it_actually_vectorized_the_corpus(self) -> None:
        db = await _corpus()

        await asyncio.wait_for(_vectorizer(db).run_batch(), timeout=DEADLINE)

        assert await _vectorized_count(db) == CORPUS_SIZE


class TestALimitIsRespectedAndReturns:
    @pytest.mark.asyncio
    async def test_it_stops_at_the_limit_without_draining_the_corpus(self) -> None:
        db = await _corpus()
        vectorizer = _vectorizer(db, max_workers=1, batch_size=CORPUS_SIZE)

        result = await asyncio.wait_for(vectorizer.run_batch(limit=2), timeout=DEADLINE)

        assert result.processed >= 2
        assert result.processed < CORPUS_SIZE

    @pytest.mark.asyncio
    async def test_a_limit_of_zero_returns_immediately(self) -> None:
        """``0 or float("inf")`` is infinity --- the falsy-limit hole."""
        db = await _corpus()

        result = await asyncio.wait_for(_vectorizer(db).run_batch(limit=0), timeout=DEADLINE)

        assert result.processed == 0

    @pytest.mark.asyncio
    async def test_a_limit_larger_than_the_corpus_still_returns(self) -> None:
        """The drain has to win when the count never can."""
        db = await _corpus()

        result = await asyncio.wait_for(
            _vectorizer(db).run_batch(limit=CORPUS_SIZE * 10), timeout=DEADLINE
        )

        assert result.processed == CORPUS_SIZE


class TestNothingIsCountedBeforeItIsWritten:
    """``empty()`` was the predicate; it is true of a record still in flight."""

    @pytest.mark.asyncio
    async def test_every_counted_record_is_in_the_database(self) -> None:
        db = await _corpus()

        result = await asyncio.wait_for(_vectorizer(db).run_batch(), timeout=DEADLINE)

        assert await _vectorized_count(db) == result.processed


class TestATimeoutIsAWayOut:
    @pytest.mark.asyncio
    async def test_a_stalled_embedder_does_not_hang_the_caller(self) -> None:
        db = await _corpus()
        vectorizer = _vectorizer(db, embedding_fn=NeverReturns(), max_workers=1)

        result = await asyncio.wait_for(vectorizer.run_batch(timeout=0.2), timeout=DEADLINE)

        assert result.processed == 0


class TestTheVectorizerIsLeftStopped:
    """A method that starts workers owns stopping them, on every exit."""

    @pytest.mark.asyncio
    async def test_after_a_drain(self) -> None:
        db = await _corpus()
        vectorizer = _vectorizer(db)

        await asyncio.wait_for(vectorizer.run_batch(), timeout=DEADLINE)

        assert vectorizer.get_stats()["is_running"] is False

    @pytest.mark.asyncio
    async def test_after_a_timeout(self) -> None:
        db = await _corpus()
        vectorizer = _vectorizer(db, embedding_fn=NeverReturns(), max_workers=1)

        await asyncio.wait_for(vectorizer.run_batch(timeout=0.2), timeout=DEADLINE)

        assert vectorizer.get_stats()["is_running"] is False

    @pytest.mark.asyncio
    async def test_the_batch_size_is_restored(self) -> None:
        db = await _corpus()
        vectorizer = _vectorizer(db, batch_size=64)

        await asyncio.wait_for(vectorizer.run_batch(limit=2), timeout=DEADLINE)

        assert vectorizer.batch_size == 64
