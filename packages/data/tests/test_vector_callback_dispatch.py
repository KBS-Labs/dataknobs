"""A callback that is an object, not a function, still gets awaited.

Both classes here take a callback from their caller and branch on whether it
is asynchronous. ``asyncio.iscoroutinefunction`` answers that correctly for a
function and incorrectly for a callable *object* whose ``__call__`` is an
``async def`` --- which is how anything holding state is written, and holding
state is the ordinary reason to pass an object rather than a function.

The failure is silent in both directions the branch can go. Called without
being awaited, an async callable returns a coroutine: truthy, non-``None``,
and discarded. Handed to ``asyncio.to_thread``, it is invoked on a worker
thread and the coroutine it returns is discarded there instead. Neither path
raises, and both loops go on to count the item as processed.

Every callback in the existing suites for these two classes is a plain
``async def``, which is exactly the shape the broken branch gets right --- so
the suites passed throughout.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

import dataknobs_data.vector as vector_pkg
from dataknobs_data.backends.memory import AsyncMemoryDatabase
from dataknobs_data.vector.optimizations import BatchConfig, BatchProcessor
from dataknobs_data.vector.tracker import ChangeTracker


class StatefulAsyncCallback:
    """An ``async def __call__`` on an object that remembers what it saw.

    The state is the point. A plain ``async def`` closure could record the
    same thing, and would be dispatched correctly by the broken branch; it is
    the *object* wrapper that `iscoroutinefunction` misreads.
    """

    def __init__(self) -> None:
        self.seen: list[Any] = []

    async def __call__(self, item: Any) -> None:
        self.seen.append(item)


class TestBatchProcessorAwaitsACallableObject:
    """``BatchProcessor._process_sequential`` dispatches the item callback."""

    async def test_the_sequential_path_runs_it(self) -> None:
        processor = BatchProcessor(BatchConfig(size=2, parallel_workers=1))
        callback = StatefulAsyncCallback()

        await processor.add("a", callback)
        await processor.add("b", callback)  # reaching `size` auto-flushes

        assert callback.seen == ["a", "b"]

    async def test_the_parallel_path_runs_it(self) -> None:
        """The default ``parallel_workers=4`` fans out to the same dispatch.

        Worth its own test because it is the *default* configuration: a
        caller who does not pass a ``BatchConfig`` at all takes this path.
        """
        processor = BatchProcessor(BatchConfig(size=4, parallel_workers=2))
        callback = StatefulAsyncCallback()

        for item in ("a", "b", "c", "d"):
            await processor.add(item, callback)

        assert sorted(callback.seen) == ["a", "b", "c", "d"]

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


class TestChangeTrackerAwaitsACallableObject:
    """``ChangeTracker.process_batch`` dispatches each update callback."""

    async def _tracker(self) -> ChangeTracker:
        db = AsyncMemoryDatabase()
        await db.connect()
        return ChangeTracker(db, tracked_fields=["content"])

    async def test_process_batch_runs_it(self) -> None:
        tracker = await self._tracker()
        callback = StatefulAsyncCallback()
        tracker.add_update_callback(callback)

        assert tracker.track_change("r-1", "content", "old", "new") is True
        processed = await tracker.process_batch()

        assert processed == 1
        assert len(callback.seen) == 1
        assert callback.seen[0].record_id == "r-1"

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


class TestNoVectorModuleReadsCallableShapeItself:
    """A recurrence guard over the whole subpackage.

    ``dataknobs_data.vector`` already answered this question correctly in two
    places --- ``embedding_fn`` and ``migration``'s progress callback --- and
    incorrectly in two others. Adoption stopping partway through one
    subpackage is what this asserts against, since nothing about the next
    callback-dispatching site would make its author look at the other four.
    """

    #: Raw ``iscoroutinefunction`` uses that are correct, and why. A bound
    #: method of an ``async def`` is answered correctly by the stdlib check;
    #: it is *callable objects* that it misreads, so a site whose subject can
    #: only ever be a method needs no helper.
    DECLARED: ClassVar[dict[tuple[str, str], str]] = {
        ("optimizations.py", "conn.close"): (
            "a connection's own `close`, so always a bound method or a plain "
            "function -- never a caller-supplied callable object"
        ),
    }

    def test_every_raw_use_is_declared(self) -> None:
        pattern = re.compile(r"iscoroutinefunction\(\s*([^)]*)\)")
        root = Path(vector_pkg.__file__).parent

        undeclared: list[str] = []
        for source in sorted(root.rglob("*.py")):
            for lineno, line in enumerate(source.read_text().splitlines(), 1):
                match = pattern.search(line)
                if match is None:
                    continue
                key = (source.name, match.group(1).strip())
                if key not in self.DECLARED:
                    undeclared.append(f"{source.name}:{lineno}: {line.strip()}")

        assert not undeclared, (
            "raw `iscoroutinefunction` in dataknobs_data.vector, on a subject "
            "that may be a callable object:\n  "
            + "\n  ".join(undeclared)
            + "\n\nUse `dataknobs_common.callbacks.is_async_callable`, which is "
            "a TypeGuard and so is a drop-in. If the subject genuinely cannot "
            "be a callable object, add it to DECLARED with that reason."
        )

    def test_the_declaration_has_no_stale_entries(self) -> None:
        """A declared site that no longer exists would license a future one."""
        pattern = re.compile(r"iscoroutinefunction\(\s*([^)]*)\)")
        root = Path(vector_pkg.__file__).parent

        found = {
            (source.name, match.group(1).strip())
            for source in root.rglob("*.py")
            for match in [pattern.search(line) for line in source.read_text().splitlines()]
            if match is not None
        }

        assert set(self.DECLARED) <= found, (
            f"declared but no longer present: {set(self.DECLARED) - found}"
        )
