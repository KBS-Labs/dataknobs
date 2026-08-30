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


class ReturnsNothing:
    """An embedder that declines rather than fails.

    ``_process_record`` returns without writing when this happens, and the
    record still matches the loader's ``NOT_EXISTS`` query.
    """

    async def __call__(self, text: str) -> np.ndarray | None:
        return None


class TestARecordThePipelineDeclinesToWriteDoesNotStallTheDrain:
    """``_source_drained`` meant "the query found nothing".

    The condition the class needs is "the query found nothing *new*". A
    record can complete without being written --- its assembled text is
    empty, the embedder returned ``None``, it has no id, or the ``update``
    reported that nothing is stored under that id --- and it then matches
    the loader's ``NOT_EXISTS(vector_field)`` query on every subsequent
    pass. The drain never completed, so both ``run_batch()`` and
    ``wait_for_completion()`` hung on their ``timeout=None`` defaults.

    Each cell here needs no failing embedder: the pipeline declines these
    records deliberately, and ``_load_pending_records`` documents the
    over-fetch that produces them as intended behaviour.
    """

    @pytest.mark.asyncio
    async def test_a_record_with_no_text_does_not_stall_the_others(self) -> None:
        db = await _corpus()
        await db.create(Record(data={"content": ""}))

        result = await asyncio.wait_for(_vectorizer(db).run_batch(), timeout=DEADLINE)

        assert result.processed == CORPUS_SIZE
        assert await _vectorized_count(db) == CORPUS_SIZE

    @pytest.mark.asyncio
    async def test_a_record_missing_every_text_field_does_not_stall(self) -> None:
        """No `content` key at all, rather than an empty one."""
        db = await _corpus()
        await db.create(Record(data={"title": "no content field here"}))

        result = await asyncio.wait_for(_vectorizer(db).run_batch(), timeout=DEADLINE)

        assert result.processed == CORPUS_SIZE

    @pytest.mark.asyncio
    async def test_an_embedder_that_returns_none_does_not_stall(self) -> None:
        db = await _corpus(2)
        vectorizer = _vectorizer(db, embedding_fn=ReturnsNothing())

        result = await asyncio.wait_for(vectorizer.run_batch(), timeout=DEADLINE)

        assert result.processed == 0
        assert await _vectorized_count(db) == 0

    @pytest.mark.asyncio
    async def test_wait_for_completion_returns_too(self) -> None:
        """The sibling method races the same condition, so it hung too."""
        db = await _corpus()
        await db.create(Record(data={"content": ""}))
        vectorizer = _vectorizer(db)
        await vectorizer.start()
        try:
            assert await asyncio.wait_for(vectorizer.wait_for_completion(), timeout=DEADLINE)
        finally:
            await vectorizer.stop()

    @pytest.mark.asyncio
    async def test_a_decline_is_reported_as_a_decline(self) -> None:
        """Not as a success, and not as a failure.

        A declined record counted as ``processed`` while no vector was
        written, so the result claimed more work than the database could
        show --- and a caller had no way to tell an unvectorizable corpus
        from a vectorized one.
        """
        db = await _corpus()
        await db.create(Record(data={"content": ""}))

        result = await asyncio.wait_for(_vectorizer(db).run_batch(), timeout=DEADLINE)

        assert result.processed == CORPUS_SIZE
        assert result.skipped == 1
        assert result.failed == 0
        assert await _vectorized_count(db) == result.processed

    @pytest.mark.asyncio
    async def test_a_declined_record_is_not_embedded_over_and_over(self) -> None:
        """The stall was a busy loop, not a quiet wait.

        The declined record was re-queried and re-dispatched on every pass,
        so the embedder saw it once per pass for as long as the caller
        waited.
        """
        db = await _corpus(1)
        await db.create(Record(data={"content": ""}))
        embedder = SlowEmbedder()
        vectorizer = _vectorizer(db, embedding_fn=embedder)

        await asyncio.wait_for(vectorizer.run_batch(), timeout=DEADLINE)

        assert embedder.calls == 1


class TestALimitIsMeasuredPerCall:
    """``_stats`` is the vectorizer's lifetime counter and is never reset.

    So a limit compared against it directly is a budget the *instance* has
    already spent, and the natural consumer shape --- a loop of successive
    batches over one corpus --- stopped doing work after the first call
    while still reporting a plausible-looking total.
    """

    @pytest.mark.asyncio
    async def test_successive_batches_each_do_work(self) -> None:
        db = await _corpus(10)
        vectorizer = _vectorizer(db, max_workers=1)

        first = await asyncio.wait_for(vectorizer.run_batch(limit=2), timeout=DEADLINE)
        second = await asyncio.wait_for(vectorizer.run_batch(limit=2), timeout=DEADLINE)
        third = await asyncio.wait_for(vectorizer.run_batch(limit=2), timeout=DEADLINE)

        assert first.processed >= 2
        assert second.processed >= 2
        assert third.processed >= 2

    @pytest.mark.asyncio
    async def test_the_result_reports_this_call_not_the_instance_total(self) -> None:
        db = await _corpus(10)
        vectorizer = _vectorizer(db, max_workers=1)

        await asyncio.wait_for(vectorizer.run_batch(limit=2), timeout=DEADLINE)
        before = await _vectorized_count(db)
        second = await asyncio.wait_for(vectorizer.run_batch(limit=2), timeout=DEADLINE)
        after = await _vectorized_count(db)

        assert second.processed == after - before

    @pytest.mark.asyncio
    async def test_a_loop_of_batches_drains_the_corpus(self) -> None:
        db = await _corpus(10)
        vectorizer = _vectorizer(db, max_workers=1)

        for _ in range(10):
            result = await asyncio.wait_for(vectorizer.run_batch(limit=3), timeout=DEADLINE)
            if result.processed == 0:
                break

        assert await _vectorized_count(db) == 10


class TestTheRacerDoesNotCallAFailureAWin:
    """``_until_shutdown`` reads ``done`` to decide who won the race.

    A task that *raised* is in ``done`` too, so counting it as "the work
    finished" reports success for a wait that failed --- and the cleanup
    gathers with ``return_exceptions=True``, which retrieves the exception
    and discards it, so not even the "never retrieved" warning fires. None
    of the conditions raced today can raise; the method is documented as a
    general racer, and these cells are what keep that true.
    """

    @pytest.mark.asyncio
    async def test_an_awaitable_that_raises_propagates(self) -> None:
        vectorizer = _vectorizer(await _corpus(1))

        async def boom() -> None:
            raise RuntimeError("the condition itself failed")

        with pytest.raises(RuntimeError, match="the condition itself failed"):
            await asyncio.wait_for(vectorizer._until_shutdown(boom()), timeout=DEADLINE)

    @pytest.mark.asyncio
    async def test_a_winner_still_reports_a_win(self) -> None:
        vectorizer = _vectorizer(await _corpus(1))

        async def fine() -> None:
            return None

        assert await asyncio.wait_for(vectorizer._until_shutdown(fine()), timeout=DEADLINE)

    @pytest.mark.asyncio
    async def test_a_timeout_still_reports_a_loss(self) -> None:
        vectorizer = _vectorizer(await _corpus(1))

        won = await asyncio.wait_for(
            vectorizer._until_shutdown(asyncio.Event().wait(), timeout=0.05), timeout=DEADLINE
        )

        assert won is False


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
