"""The streaming surfaces call their consumer's callbacks without judging them.

Not the ``iscoroutinefunction`` mistake --- the commoner one underneath it:
these sites never ask at all. A callback is called, and whatever comes back is
used. For an async callable that is a coroutine object, which raises nothing,
is always truthy, and is never the value the caller meant.

Three of the four consequences are visible in the data rather than in a log:

**A sink's answer is a boolean.** ``AsyncStreamContext.stream_async`` writes
``if not sink(chunk): self.metrics.errors_count += 1``. A coroutine is truthy,
so every write is recorded as having succeeded --- including the ones that
never happened, since the coroutine was discarded rather than awaited.

**A transform's answer is the data.** Both the async stream context and the
streaming file processor put the transform's return value straight into the
chunk they hand onward. An async transform therefore writes coroutine objects
to the output file, and the transform's body never runs.

**A progress callback's answer is nothing**, which is the one case where only
the side effect is lost rather than the payload. It is still the difference
between a caller that sees progress and one that does not.

Real files throughout for the file processor: it is a file processor, and a
test that avoids the disk would not be exercising it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataknobs_fsm.streaming.core import (
    AsyncStreamContext,
    StreamChunk,
    StreamConfig,
)
from dataknobs_fsm.utils.streaming_file_utils import StreamingFileProcessor


class AsyncSink:
    """A sink holding a connection --- hence an object, hence misread."""

    def __init__(self, *, accepts: bool = True) -> None:
        self.written: list[StreamChunk] = []
        self._accepts = accepts

    async def __call__(self, chunk: StreamChunk) -> bool:
        self.written.append(chunk)
        return self._accepts


class AsyncTransform:
    """A transform holding a lookup table it consults per chunk."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, data: Any) -> Any:
        self.calls += 1
        return {"transformed": data}


class AsyncProgress:
    """A progress reporter accumulating into a metrics endpoint."""

    def __init__(self) -> None:
        self.reports: list[tuple[int, int]] = []

    async def __call__(self, items: int, chunks: int) -> None:
        self.reports.append((items, chunks))


async def _one_chunk_source(chunks: list[StreamChunk]) -> Any:
    for chunk in chunks:
        yield chunk


def _chunk(data: Any, *, is_last: bool = True) -> StreamChunk:
    return StreamChunk(data=data, chunk_id="c0", sequence_number=0, is_last=is_last)


# --------------------------------------------------------------------- #
# AsyncStreamContext.stream_async
# --------------------------------------------------------------------- #


async def test_an_async_sink_actually_receives_the_chunk() -> None:
    context = AsyncStreamContext(StreamConfig(parallelism=1))
    sink = AsyncSink()

    await context.stream_async(_one_chunk_source([_chunk({"id": 1})]), sink)

    assert len(sink.written) == 1, "the sink was called but its coroutine was discarded"
    assert sink.written[0].data == {"id": 1}


async def test_a_rejecting_async_sink_is_counted_as_an_error() -> None:
    """The metric a caller reads to decide whether the stream worked.

    ``not coroutine`` is ``False``, so a sink that refused every chunk reports
    zero errors --- and it reports them over writes that never happened.
    """
    context = AsyncStreamContext(StreamConfig(parallelism=1))
    sink = AsyncSink(accepts=False)

    metrics = await context.stream_async(_one_chunk_source([_chunk({"id": 1})]), sink)

    assert metrics.errors_count == 1, "a refused write was recorded as a success"


async def test_an_async_transform_produces_data_rather_than_a_coroutine() -> None:
    context = AsyncStreamContext(StreamConfig(parallelism=1))
    sink = AsyncSink()
    transform = AsyncTransform()

    await context.stream_async(_one_chunk_source([_chunk({"id": 1})]), sink, transform=transform)

    assert transform.calls == 1
    assert sink.written[0].data == {"transformed": {"id": 1}}, (
        "the coroutine object was carried into the chunk in place of the data"
    )


async def test_a_synchronous_sink_and_transform_still_work() -> None:
    """Regression guard for the shape that was already correct."""
    context = AsyncStreamContext(StreamConfig(parallelism=1))
    written: list[StreamChunk] = []

    def sink(chunk: StreamChunk) -> bool:
        written.append(chunk)
        return True

    def transform(data: Any) -> Any:
        return {"sync": data}

    await context.stream_async(_one_chunk_source([_chunk({"id": 1})]), sink, transform=transform)

    assert [c.data for c in written] == [{"sync": {"id": 1}}]


# --------------------------------------------------------------------- #
# StreamingFileProcessor.process
# --------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    return [json.loads(line) for line in text.splitlines() if line]


async def test_an_async_transform_reaches_the_output_file(tmp_path: Path) -> None:
    """The coroutine would be serialized --- or fail to be --- into the output."""
    source = tmp_path / "in.jsonl"
    target = tmp_path / "out.jsonl"
    _write_jsonl(source, [{"id": 1}, {"id": 2}])

    class AsyncRecordTransform:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            return {**record, "seen": True}

    transform = AsyncRecordTransform()
    processor = StreamingFileProcessor(source, target, transform_fn=transform)

    await processor.process()

    assert transform.calls == 2
    assert _read_jsonl(target) == [
        {"id": 1, "seen": True},
        {"id": 2, "seen": True},
    ]


async def test_an_async_progress_callback_is_reported_to(tmp_path: Path) -> None:
    source = tmp_path / "in.jsonl"
    target = tmp_path / "out.jsonl"
    _write_jsonl(source, [{"id": 1}])

    progress = AsyncProgress()
    processor = StreamingFileProcessor(source, target, chunk_size=1)

    await processor.process(progress_callback=progress)

    assert progress.reports, "the progress callback's coroutine was discarded"
    assert progress.reports[-1][0] == 1


async def test_a_synchronous_transform_and_progress_still_work(tmp_path: Path) -> None:
    """Regression guard, and a check that the sync path keeps its ordering."""
    source = tmp_path / "in.jsonl"
    target = tmp_path / "out.jsonl"
    _write_jsonl(source, [{"id": 1}, {"id": 2}])

    reports: list[tuple[int, int]] = []
    processor = StreamingFileProcessor(
        source, target, transform_fn=lambda r: {**r, "seen": True}, chunk_size=1
    )

    await processor.process(progress_callback=lambda items, chunks: reports.append((items, chunks)))

    assert _read_jsonl(target) == [{"id": 1, "seen": True}, {"id": 2, "seen": True}]
    assert [items for items, _ in reports] == [1, 2]
