"""Every consumer callable in ``io/utils`` is dispatched, and off the loop where it may block.

Three defects share these lines, and each is silent in its own way.

**The callable is never judged.** ``IORouter.route`` calls its ``condition``
and its ``transform`` and uses what comes back. An async one returns a
coroutine: always truthy, so the condition matches every record, and it is
that coroutine object rather than the transformed data that gets written.

**The callable is judged by the wrong question.**
``asyncio.iscoroutinefunction`` answers for functions and reports a callable
*object* whose ``__call__`` is an ``async def`` as synchronous --- which is
how anything holding state is written. The buffer's overflow handler takes
the sync arm, and its items are already off the buffer by then, so they are
lost with no copy anywhere.

**The callable is judged correctly and then run on the event loop.**
``ParallelIOExecutor`` did not run a synchronous provider at all; a provider's
read or write is a file, a socket or a database round-trip, so running one
inline once the branch exists would stall every other task on the loop for
its duration. That defect is non-functional --- the value returned is right
--- so only :func:`assert_no_blocking` can see it.

Real constructs throughout: the blocking sites are driven with
``SyncFileProvider``, which this package ships and which really does open and
read a file, rather than with a stand-in that pretends to.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from dataknobs_common.testing import assert_no_blocking, requires_blockbuster
from dataknobs_fsm.io.adapters import SyncFileProvider
from dataknobs_fsm.io.base import IOConfig, IOFormat, IOMode
from dataknobs_fsm.io.utils import (
    async_transform_pipeline,
    IOBuffer,
    IORouter,
    ParallelIOExecutor,
)


class StatefulAsyncCallable:
    """An ``async def __call__`` on an object that remembers what it saw.

    The state is the point. A plain ``async def`` closure could record the
    same thing, and the ``iscoroutinefunction`` sites classify *that* shape
    correctly; it is the object wrapper they misread.
    """

    def __init__(self, returning: Any = None) -> None:
        self.seen: list[Any] = []
        self._returning = returning

    async def __call__(self, item: Any) -> Any:
        self.seen.append(item)
        return self._returning


class RecordingAsyncProvider:
    """The narrow slice of a provider that ``IORouter`` and the executor use."""

    def __init__(self) -> None:
        self.written: list[Any] = []

    async def write(self, data: Any, **kwargs: Any) -> None:
        self.written.append(data)

    async def read(self, **kwargs: Any) -> Any:
        return "async-read"


def _read_config(path: Path) -> IOConfig:
    return IOConfig(mode=IOMode.READ, format=IOFormat.TEXT, source=str(path))


def _write_config(path: Path) -> IOConfig:
    return IOConfig(mode=IOMode.WRITE, format=IOFormat.TEXT, source=str(path))


class TestIORouterJudgesItsRouteCallables:
    """``add_route`` takes a condition and a transform and never asked."""

    async def test_an_async_condition_does_not_match_everything(self) -> None:
        """A coroutine is truthy, so an unjudged condition admits every record."""
        router = IORouter()
        provider = RecordingAsyncProvider()

        async def never(data: Any) -> bool:
            return False

        router.add_route(condition=never, provider=provider)
        results = await router.route("payload")

        assert results == []
        assert provider.written == []

    async def test_an_async_transform_is_awaited_not_written_as_data(self) -> None:
        """The value the transform produced, not the coroutine that produces it."""
        router = IORouter()
        provider = RecordingAsyncProvider()

        async def shout(data: Any) -> str:
            return str(data).upper()

        router.add_route(condition=lambda _: True, provider=provider, transform=shout)
        results = await router.route("payload")

        assert results == ["PAYLOAD"]
        assert provider.written == ["PAYLOAD"]

    async def test_a_callable_object_condition_is_judged(self) -> None:
        """The shape `iscoroutinefunction` misreads, on the site that never asked."""
        router = IORouter()
        provider = RecordingAsyncProvider()
        condition = StatefulAsyncCallable(returning=False)

        router.add_route(condition=condition, provider=provider)
        results = await router.route("payload")

        assert condition.seen == ["payload"]
        assert results == []

    @requires_blockbuster
    async def test_a_synchronous_provider_write_does_not_stall_the_loop(
        self, tmp_path: Path
    ) -> None:
        """A provider's write is disk or network I/O, whoever supplied it."""
        target = tmp_path / "routed.txt"
        provider = SyncFileProvider(_write_config(target))
        router = IORouter()
        router.add_route(condition=lambda _: True, provider=provider)

        with assert_no_blocking():
            await router.route("payload")
        provider.close()

        assert target.read_text() == "payload"


class TestIOBufferOverflowHandler:
    """The items are off the buffer before the handler is called."""

    async def test_a_callable_object_handler_receives_the_items(self) -> None:
        handler = StatefulAsyncCallable()
        buffer = IOBuffer(max_size=4, overflow_handler=handler)

        for item in ("a", "b", "c", "d"):
            await buffer.add(item)

        assert handler.seen == [["a", "b"]]
        assert await buffer.flush() == ["c", "d"]

    @requires_blockbuster
    async def test_a_blocking_handler_does_not_stall_the_loop(self, tmp_path: Path) -> None:
        """An overflow handler's ordinary job is to put the items somewhere."""
        spill = tmp_path / "overflow.txt"

        def flush_to_disk(items: list[Any]) -> None:
            with open(spill, "a", encoding="utf-8") as handle:
                handle.write("".join(str(i) for i in items))

        buffer = IOBuffer(max_size=4, overflow_handler=flush_to_disk)

        with assert_no_blocking():
            for item in ("a", "b", "c", "d"):
                await buffer.add(item)

        assert spill.read_text() == "ab"


class TestAsyncTransformPipeline:
    """A pipeline of transforms, which may be sync or async in any mix."""

    async def test_a_callable_object_transform_is_awaited(self) -> None:
        """An un-awaited coroutine here feeds the *next* transform in the chain."""
        double = StatefulAsyncCallable(returning=8)
        pipeline = async_transform_pipeline(lambda x: x * 2, double, lambda x: x + 1)

        assert await pipeline(2) == 9
        assert double.seen == [4]

    async def test_a_synchronous_transform_runs_on_the_event_loop(self) -> None:
        """Deliberate, and pinned so that reversing it has to argue.

        A transform is named for computing rather than for doing, and it runs
        once per item, so it pays for the offload on every record while a
        provider write pays once per batch. The sibling dispatches in this
        module go the other way for exactly that reason --- the difference is
        the surface, not an oversight in one of them.
        """
        loop_thread = threading.current_thread()
        seen: list[threading.Thread] = []

        def record_thread(value: Any) -> Any:
            seen.append(threading.current_thread())
            return value

        pipeline = async_transform_pipeline(record_thread)
        await pipeline("payload")

        assert seen == [loop_thread]


class TestParallelIOExecutorIncludesSyncProviders:
    """Its providers are typed ``IOProvider``, which ``SyncIOProvider`` is."""

    async def test_read_all_reads_a_synchronous_provider(self, tmp_path: Path) -> None:
        source = tmp_path / "in.txt"
        source.write_text("from-disk")
        executor = ParallelIOExecutor([SyncFileProvider(_read_config(source))])

        assert await executor.read_all() == ["from-disk"]

    async def test_read_all_reads_both_kinds_together(self, tmp_path: Path) -> None:
        source = tmp_path / "in.txt"
        source.write_text("from-disk")
        executor = ParallelIOExecutor(
            [RecordingAsyncProvider(), SyncFileProvider(_read_config(source))]
        )

        assert sorted(await executor.read_all()) == ["async-read", "from-disk"]

    async def test_write_all_writes_to_a_synchronous_provider(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        provider = SyncFileProvider(_write_config(target))
        executor = ParallelIOExecutor([provider])

        await executor.write_all("payload")
        provider.close()

        assert target.read_text() == "payload"

    @requires_blockbuster
    async def test_a_synchronous_read_does_not_stall_the_loop(self, tmp_path: Path) -> None:
        source = tmp_path / "in.txt"
        source.write_text("from-disk")
        executor = ParallelIOExecutor([SyncFileProvider(_read_config(source))])

        with assert_no_blocking():
            results = await executor.read_all()

        assert results == ["from-disk"]

    async def test_a_provider_with_neither_method_is_still_skipped(self) -> None:
        """The `hasattr` gate is the documented one and is not what changed."""

        class Inert:
            pass

        assert await ParallelIOExecutor([Inert()]).read_all() == []
