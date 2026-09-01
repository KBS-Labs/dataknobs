"""A callback that is an object, not a function, still gets awaited.

Every class here takes a callback from its caller and, inside an ``async
def``, has to decide what to do with it. There are two ways to get that
wrong, and this package held both.

**Asking the wrong question.** ``asyncio.iscoroutinefunction`` answers
correctly for a function and reports a callable *object* whose ``__call__``
is an ``async def`` as synchronous --- which is how anything holding state is
written, and holding state is the ordinary reason to pass an object rather
than a function.

**Asking no question at all.** The more common spelling: call the callback
and move on. That is the same defect one step earlier, and it is the one a
guard keyed to ``iscoroutinefunction`` cannot see, because the token is not
there to find.

The failure is silent in every direction. An async callable invoked without
being awaited returns a coroutine: truthy, non-``None``, and discarded.
Handed to ``asyncio.to_thread`` it is invoked on a worker thread and the
coroutine is discarded there instead. Stored, it is persisted --- and where
the callback's *return value* is used, the coroutine object becomes the
transformed record, the yielded item, or the answer to a filter predicate.
Nothing raises anywhere along that path.

Every callback in the existing suites for these classes is a plain ``async
def`` or a plain ``def``, which are exactly the two shapes the broken
branches get right --- so the suites passed throughout.

The recurrence guard is not here. It is a workspace-level census over
``packages/data/src`` in ``tests/test_async_callable_adoption.py``, because
the question "did this dispatch classify its callback?" is not a question
about the vector subpackage.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.migration.migrator import Migrator
from dataknobs_data.records import Record
from dataknobs_data.streaming import StreamProcessor
from dataknobs_data.vector.migration import VectorMigration
from dataknobs_data.vector.optimizations import BatchConfig, BatchProcessor, ConnectionPool
from dataknobs_data.vector.stores.memory import MemoryVectorStore
from dataknobs_data.vector.sync import VectorTextSynchronizer
from dataknobs_data.vector.tracker import ChangeTracker


class StatefulAsyncCallback:
    """An ``async def __call__`` on an object that remembers what it saw.

    The state is the point. A plain ``async def`` closure could record the
    same thing, and would be dispatched correctly by a branch that asks
    ``iscoroutinefunction``; it is the *object* wrapper that is misread, and
    it is the shape a caller reaches for precisely because it accumulates.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[Any, ...]] = []

    async def __call__(self, *args: Any) -> None:
        self.seen.append(args)


class StatefulAsyncTransform:
    """An async transform whose *return value* the caller goes on to use.

    Worse than a dropped notification: an un-awaited call here does not
    discard the coroutine, it substitutes it for the value. The record that
    gets stored, or the item that gets yielded, is the coroutine object.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, record: Record) -> Record:
        self.calls += 1
        record.set_value("transformed", True)
        return record


class StatefulAsyncPredicate:
    """An async predicate whose answer decides whether an item survives.

    A coroutine is always truthy, so an un-awaited predicate does not fail
    closed or open at random --- it admits everything, every time.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, record: Record) -> bool:
        self.calls += 1
        return bool(record.get_value("keep"))


async def _database(*records: Record) -> AsyncMemoryDatabase:
    db = AsyncMemoryDatabase()
    await db.connect()
    for record in records:
        await db.create(record)
    return db


def _embedding(_text: str) -> np.ndarray:
    return np.array([0.1, 0.2, 0.3], dtype=np.float32)


async def _resolved(value: Any) -> Any:
    """``ConnectionPool`` awaits its factory, so the factory returns this."""
    return value


# --------------------------------------------------------------------- #
# Dispatches that asked `iscoroutinefunction`
# --------------------------------------------------------------------- #


class TestBatchProcessorAwaitsACallableObject:
    """``BatchProcessor._process_sequential`` dispatches the item callback."""

    async def test_the_sequential_path_runs_it(self) -> None:
        processor = BatchProcessor(BatchConfig(size=2, parallel_workers=1))
        callback = StatefulAsyncCallback()

        await processor.add("a", callback)
        await processor.add("b", callback)  # reaching `size` auto-flushes

        assert callback.seen == [("a",), ("b",)]

    async def test_the_parallel_path_runs_it(self) -> None:
        """The default ``parallel_workers=4`` fans out to the same dispatch.

        Worth its own test because it is the *default* configuration: a
        caller who does not pass a ``BatchConfig`` at all takes this path.
        """
        processor = BatchProcessor(BatchConfig(size=4, parallel_workers=2))
        callback = StatefulAsyncCallback()

        for item in ("a", "b", "c", "d"):
            await processor.add(item, callback)

        assert sorted(callback.seen) == [("a",), ("b",), ("c",), ("d",)]

    async def test_a_plain_async_function_still_runs(self) -> None:
        """The shape that always worked keeps working."""
        processor = BatchProcessor(BatchConfig(size=2, parallel_workers=1))
        seen: list[Any] = []

        async def callback(item: Any) -> None:
            seen.append(item)

        await processor.add("a", callback)
        await processor.add("b", callback)

        assert seen == ["a", "b"]

    async def test_a_synchronous_callback_still_runs(self) -> None:
        """And so does the other arm of the branch."""
        processor = BatchProcessor(BatchConfig(size=2, parallel_workers=1))
        seen: list[Any] = []

        await processor.add("a", seen.append)
        await processor.add("b", seen.append)

        assert seen == ["a", "b"]

    @requires_blockbuster
    async def test_a_blocking_callback_does_not_stall_the_loop(self, tmp_path: Path) -> None:
        """The per-item callback is the caller's chance to record the item.

        Its sibling ``ChangeTracker.process_batch`` has always offloaded the
        same kind of dispatch, and this one called it inline --- so the two
        surfaces disagreed about whether a consumer's callback may block, with
        neither of them saying which was right.
        """
        log = tmp_path / "processed.txt"
        processor = BatchProcessor(BatchConfig(size=2, parallel_workers=1))

        def record(item: Any) -> None:
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(f"{item}\n")

        with assert_no_blocking():
            await processor.add("a", record)
            await processor.add("b", record)

        assert log.read_text() == "a\nb\n"

    async def test_the_callback_runs_off_the_event_loop(self) -> None:
        """Structural proof, for a callback that blocks in a way blockbuster does not patch.

        The detector covers the common syscalls; it does not cover an
        arbitrary CPU-bound consumer callback. Naming the thread pins the
        offload itself rather than one class of symptom.
        """
        processor = BatchProcessor(BatchConfig(size=1, parallel_workers=1))
        loop_thread = threading.current_thread()
        seen: list[threading.Thread] = []

        await processor.add("a", lambda _: seen.append(threading.current_thread()))

        assert seen and seen[0] is not loop_thread


class TestConnectionPoolClosesAnObjectsCloseMethod:
    """``ConnectionPool.close`` dispatches each connection's ``close``.

    The connection is whatever the caller's ``factory`` returns, so its
    ``close`` is an attribute of a consumer-supplied object and may be
    anything callable --- including an object with an ``async def
    __call__``, which is how a closer holding its own state is written. An
    un-awaited close leaks the connection and logs nothing.

    ``ConnectionPool`` is deprecated in favour of
    :class:`~dataknobs_data.pooling.ConnectionPoolManager`, so these pin
    surface we still ship rather than surface we recommend. The dispatch
    defect was real while it lasted, and the class remains importable
    until its removal, so the fix and its proof stay.
    """

    async def test_an_async_callable_close_is_awaited(self) -> None:
        class AsyncCloser:
            def __init__(self) -> None:
                self.closed = False

            async def __call__(self) -> None:
                self.closed = True

        class Connection:
            def __init__(self) -> None:
                self.close = AsyncCloser()

        connection = Connection()
        pool = ConnectionPool(factory=lambda: _resolved(connection))

        assert await pool.acquire() is connection
        await pool.close()

        assert connection.close.closed is True

    async def test_a_plain_async_close_still_runs(self) -> None:
        """The shape that always worked keeps working."""
        closed: list[str] = []

        class Connection:
            async def close(self) -> None:
                closed.append("yes")

        pool = ConnectionPool(factory=lambda: _resolved(Connection()))
        await pool.acquire()
        await pool.close()

        assert closed == ["yes"]

    async def test_a_synchronous_close_still_runs(self) -> None:
        closed: list[str] = []

        class Connection:
            def close(self) -> None:
                closed.append("yes")

        pool = ConnectionPool(factory=lambda: _resolved(Connection()))
        await pool.acquire()
        await pool.close()

        assert closed == ["yes"]


class TestChangeTrackerAwaitsACallableObject:
    """``ChangeTracker.process_batch`` dispatches each update callback."""

    async def _tracker(self) -> ChangeTracker:
        return ChangeTracker(await _database(), tracked_fields=["content"])

    async def test_process_batch_runs_it(self) -> None:
        tracker = await self._tracker()
        callback = StatefulAsyncCallback()
        tracker.add_update_callback(callback)

        assert tracker.track_change("r-1", "content", "old", "new") is True
        processed = await tracker.process_batch()

        assert processed == 1
        assert len(callback.seen) == 1
        assert callback.seen[0][0].record_id == "r-1"

    async def test_the_count_does_not_outrun_the_callbacks(self) -> None:
        """``processed`` claims the callbacks ran, so it must not lead them.

        This is the half that made the defect invisible: the loop increments
        its counter after dispatching, and a discarded coroutine raises
        nothing, so ``process_batch`` reported every task done while none of
        the work attached to them had happened.
        """
        tracker = await self._tracker()
        callback = StatefulAsyncCallback()
        tracker.add_update_callback(callback)

        for i in range(3):
            tracker.track_change(f"r-{i}", "content", "old", f"new-{i}")
        processed = await tracker.process_batch()

        assert processed == len(callback.seen) == 3

    async def test_a_synchronous_callback_still_runs(self) -> None:
        """The sync arm here is a thread offload, not a direct call."""
        tracker = await self._tracker()
        seen: list[Any] = []
        tracker.add_update_callback(seen.append)

        tracker.track_change("r-1", "content", "old", "new")
        assert await tracker.process_batch() == 1
        assert len(seen) == 1


# --------------------------------------------------------------------- #
# Dispatches that asked nothing --- progress notifications
# --------------------------------------------------------------------- #


class TestVectorMigrationRunsItsProgressCallback:
    """``VectorMigration``'s three public entry points each report progress.

    A fourth --- ``IncrementalVectorizer.run``, in the same module --- has
    always classified its callback correctly. That sibling is why these three
    are a defect rather than a policy.
    """

    async def _migration(self) -> VectorMigration:
        source = await _database(Record({"id": "r-1", "content": "alpha"}))
        return VectorMigration(
            source_db=source,
            target_db=await _database(),
            embedding_fn=_embedding,
            text_fields=["content"],
        )

    async def test_run(self) -> None:
        callback = StatefulAsyncCallback()

        status = await (await self._migration()).run(progress_callback=callback)

        assert status.migrated_records == 1
        assert len(callback.seen) == 1

    async def test_add_vectors_to_existing(self) -> None:
        migration = VectorMigration(
            source_db=await _database(Record({"id": "r-1", "content": "alpha"})),
            embedding_fn=_embedding,
            text_fields=["content"],
        )
        callback = StatefulAsyncCallback()

        await migration.add_vectors_to_existing(
            {"embedding": "content"}, progress_callback=callback
        )

        assert len(callback.seen) == 1

    async def test_migrate_between_backends(self) -> None:
        callback = StatefulAsyncCallback()

        await (await self._migration()).migrate_between_backends(progress_callback=callback)

        assert len(callback.seen) == 1


class TestVectorSynchronizerRunsItsProgressCallback:
    """Both of ``VectorTextSynchronizer``'s sweep entry points report progress."""

    async def _synchronizer(self) -> VectorTextSynchronizer:
        db = await _database(Record({"id": "r-1", "content": "alpha"}))
        return VectorTextSynchronizer(
            database=db,
            embedding_fn=_embedding,
            text_fields=["content"],
        )

    async def test_sync_all(self) -> None:
        callback = StatefulAsyncCallback()

        await (await self._synchronizer()).sync_all(progress_callback=callback)

        assert callback.seen == [(1, 1)]

    async def test_bulk_sync(self) -> None:
        callback = StatefulAsyncCallback()

        await (await self._synchronizer()).bulk_sync(progress_callback=callback)

        assert len(callback.seen) == 1


class TestDataMigratorRunsItsProgressCallback:
    """``DataMigrator.migrate_async`` reports once, at the end of the stream."""

    async def test_migrate_async(self) -> None:
        callback = StatefulAsyncCallback()

        await Migrator().migrate_async(
            source=await _database(Record({"id": "r-1", "content": "alpha"})),
            target=await _database(),
            on_progress=callback,
        )

        assert len(callback.seen) == 1


# --------------------------------------------------------------------- #
# Dispatches that asked nothing --- and then used the answer
# --------------------------------------------------------------------- #


class TestTransformResultsAreNotCoroutines:
    """Where a callback's return value is used, an un-awaited call substitutes
    a coroutine object for it --- so the coroutine is what gets stored or
    yielded. Strictly worse than a dropped notification, and equally silent.
    """

    async def test_vector_migration_transform_fn(self) -> None:
        """The transformed record is created in the target database."""
        migration = VectorMigration(
            source_db=await _database(Record({"id": "r-1", "content": "alpha"})),
            target_db=await _database(),
            embedding_fn=_embedding,
            text_fields=["content"],
        )
        transform = StatefulAsyncTransform()

        await migration.migrate_between_backends(transform_fn=transform)

        assert transform.calls == 1
        stored = await migration.target_db.read("r-1")
        assert stored is not None
        assert stored.get_value("transformed") is True

    async def test_database_stream_transform(self) -> None:
        """``stream_transform`` yields whatever the transform returned."""
        db = await _database(Record({"id": "r-1", "content": "alpha"}))
        transform = StatefulAsyncTransform()

        produced = [record async for record in db.stream_transform(transform=transform)]

        assert transform.calls == 1
        assert [type(record) for record in produced] == [Record]
        assert produced[0].get_value("transformed") is True

    async def test_stream_processor_async_transform_stream(self) -> None:
        transform = StatefulAsyncTransform()

        async def source():
            yield Record({"id": "r-1", "content": "alpha"})

        produced = [
            record async for record in StreamProcessor.async_transform_stream(source(), transform)
        ]

        assert transform.calls == 1
        assert [type(record) for record in produced] == [Record]
        assert produced[0].get_value("transformed") is True


class TestVectorStoreAcceptsAnAsyncRecordFetcher:
    """``VectorStore.search_similar_records`` calls the caller's fetcher.

    This one did not fail silently --- iterating a coroutine raises
    ``TypeError`` --- so the defect is that the store *refuses* the shape a
    consumer actually has. Fetching records by id is I/O, so the fetcher for
    an async database is an async callable, and there was no way to pass one.
    """

    async def test_an_async_fetcher_is_awaited(self) -> None:
        store = MemoryVectorStore(dimensions=3)
        await store.add_vectors(
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            ids=["v-1"],
            metadata=[{"record_id": "r-1"}],
        )
        db = await _database(Record({"id": "r-1", "content": "alpha"}))

        async def fetch_records(ids: list[str]) -> list[Record]:
            return [record for record in [await db.read(rid) for rid in ids] if record]

        results = await store.search_similar_records(
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            k=1,
            fetch_records=fetch_records,
        )

        assert [result.record.get_value("content") for result in results] == ["alpha"]

    async def test_a_synchronous_fetcher_still_works(self) -> None:
        store = MemoryVectorStore(dimensions=3)
        await store.add_vectors(
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            ids=["v-1"],
            metadata=[{"record_id": "r-1"}],
        )
        record = Record({"id": "r-1", "content": "alpha"})

        results = await store.search_similar_records(
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            k=1,
            fetch_records=lambda ids: [record],
        )

        assert [result.record.get_value("content") for result in results] == ["alpha"]


class TestPredicateResultsAreNotCoroutines:
    """A coroutine is always truthy, so an un-awaited predicate admits
    everything --- the filter silently stops filtering rather than failing.
    """

    async def test_stream_processor_async_filter_stream(self) -> None:
        predicate = StatefulAsyncPredicate()

        async def source():
            yield Record({"id": "keep-me", "keep": True})
            yield Record({"id": "drop-me", "keep": False})

        survived = [
            record async for record in StreamProcessor.async_filter_stream(source(), predicate)
        ]

        assert predicate.calls == 2
        assert [record.id for record in survived] == ["keep-me"]


# --------------------------------------------------------------------- #
# The shapes that already worked keep working
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", ["function", "object"])
class TestSynchronousCallbacksAreUnaffected:
    """Widening a dispatch to accept an async callable must not change what a
    synchronous one does. Both spellings of "synchronous" are checked, since
    the object form is the one every branch here used to misclassify.
    """

    @staticmethod
    def _sync_callback(shape: str, seen: list[Any]) -> Any:
        if shape == "function":
            return lambda *args: seen.append(args)

        class Recorder:
            def __call__(self, *args: Any) -> None:
                seen.append(args)

        return Recorder()

    async def test_vector_migration_run(self, shape: str) -> None:
        seen: list[Any] = []
        migration = VectorMigration(
            source_db=await _database(Record({"id": "r-1", "content": "alpha"})),
            target_db=await _database(),
            embedding_fn=_embedding,
            text_fields=["content"],
        )

        await migration.run(progress_callback=self._sync_callback(shape, seen))

        assert len(seen) == 1

    async def test_stream_processor_async_filter_stream(self, shape: str) -> None:
        seen: list[Any] = []
        recorder = self._sync_callback(shape, seen)

        def predicate(record: Record) -> bool:
            recorder(record)
            return bool(record.get_value("keep"))

        async def source():
            yield Record({"id": "keep-me", "keep": True})
            yield Record({"id": "drop-me", "keep": False})

        survived = [
            record async for record in StreamProcessor.async_filter_stream(source(), predicate)
        ]

        assert len(seen) == 2
        assert [record.id for record in survived] == ["keep-me"]
