"""Reproduce-first async-correctness tests for ``StreamingFileReader``.

``StreamingFileReader.read_chunks`` streams a large file in bounded
chunks, but each format reader (``_read_jsonl/json/csv/text_chunks``)
opens the file with a blocking ``open()`` and line-iterates it on the
event loop. The offload drives each reader's extracted sync generator on
a worker thread via ``aiter_sync_in_thread``, so the open and every read
happen off the loop while the bounded-queue hand-off preserves streaming.

Each blocking test consumes ``read_chunks`` inside :func:`assert_no_blocking`.
Pre-offload these FAIL on the ``open()``; post-offload they PASS. The
functional tests guard the real regression surface — chunk boundaries,
the ``is_last`` flag on the final chunk, and record fidelity — and the
multi-chunk streaming test proves a file larger than one chunk is read
without being materialised whole.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_fsm.streaming.core import StreamChunk
from dataknobs_fsm.utils import streaming_file_utils as _module
from dataknobs_fsm.utils.streaming_file_utils import (
    StreamingFileReader,
    StreamingFileWriter,
)

if TYPE_CHECKING:
    from pathlib import Path


async def _read_all(reader: StreamingFileReader) -> list:
    return [chunk async for chunk in reader.read_chunks()]


def _write_jsonl(path: Path, n: int) -> None:
    path.write_text("".join(f'{{"i": {i}}}\n' for i in range(n)))


def _thread_recording_open(record: list[str]):
    """A real ``open`` that records the invoking thread (see file_utils test)."""
    real_open = open

    def spy(*args, **kwargs):
        record.append(threading.current_thread().name)
        return real_open(*args, **kwargs)

    return spy


# --------------------------------------------------------------------------
# Reproduce-first: read_chunks must not open/read on the event loop.
# --------------------------------------------------------------------------


@requires_blockbuster
async def test_jsonl_chunks_do_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, 5)
    reader = StreamingFileReader(path, chunk_size=2, input_format="jsonl")
    with assert_no_blocking():
        chunks = await _read_all(reader)
    assert sum(len(c.data) for c in chunks) == 5


@requires_blockbuster
async def test_json_chunks_do_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('[{"i": 0}, {"i": 1}, {"i": 2}]')
    reader = StreamingFileReader(path, chunk_size=2, input_format="json")
    with assert_no_blocking():
        chunks = await _read_all(reader)
    assert sum(len(c.data) for c in chunks) == 3


@requires_blockbuster
async def test_json_single_object_chunks_do_not_block(tmp_path: Path) -> None:
    # Exercises the ijson-fails-then-json.load fallback branch (second open()).
    path = tmp_path / "data.json"
    path.write_text('{"i": 0}')
    reader = StreamingFileReader(path, chunk_size=2, input_format="json")
    with assert_no_blocking():
        chunks = await _read_all(reader)
    assert sum(len(c.data) for c in chunks) == 1


@requires_blockbuster
async def test_csv_chunks_do_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,value\nalice,1\nbob,2\ncarol,3\n")
    reader = StreamingFileReader(path, chunk_size=2, input_format="csv")
    with assert_no_blocking():
        chunks = await _read_all(reader)
    assert sum(len(c.data) for c in chunks) == 3


@requires_blockbuster
async def test_text_chunks_do_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("a\nb\nc\n")
    reader = StreamingFileReader(path, chunk_size=2, input_format="text")
    with assert_no_blocking():
        chunks = await _read_all(reader)
    assert sum(len(c.data) for c in chunks) == 3


# --------------------------------------------------------------------------
# Functional: chunk boundaries, is_last flag, streaming preserved.
# --------------------------------------------------------------------------


async def test_jsonl_chunk_boundaries_and_last_flag(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, 5)
    reader = StreamingFileReader(path, chunk_size=2, input_format="jsonl")
    chunks = await _read_all(reader)

    # 5 records, chunk_size 2 -> [2, 2, 1]; only the final chunk is is_last.
    assert [len(c.data) for c in chunks] == [2, 2, 1]
    assert [c.is_last for c in chunks] == [False, False, True]
    flat = [rec["i"] for c in chunks for rec in c.data]
    assert flat == [0, 1, 2, 3, 4]


async def test_csv_records_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,value\nalice,1\nbob,2\n")
    reader = StreamingFileReader(path, chunk_size=10, input_format="csv")
    chunks = await _read_all(reader)
    records = [rec for c in chunks for rec in c.data]
    assert records == [
        {"name": "alice", "value": "1"},
        {"name": "bob", "value": "2"},
    ]
    assert chunks[-1].is_last is True


# --------------------------------------------------------------------------
# Structural reproduce-first for line-iterating chunk readers (detector-blind).
# Pre-offload the open runs on the event-loop thread (FAIL); post-offload it
# runs on the aiter_sync_in_thread worker (PASS).
# --------------------------------------------------------------------------


async def test_jsonl_chunks_open_on_worker_thread(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, 3)
    threads: list[str] = []
    monkeypatch.setattr(_module, "open", _thread_recording_open(threads), raising=False)
    reader = StreamingFileReader(path, chunk_size=2, input_format="jsonl")
    await _read_all(reader)
    assert threads, "open() was never called"
    assert threading.current_thread().name not in threads
    assert all(name.startswith("dk-aiter-sync-pump") for name in threads)


async def test_csv_chunks_open_on_worker_thread(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,value\nalice,1\nbob,2\n")
    threads: list[str] = []
    monkeypatch.setattr(_module, "open", _thread_recording_open(threads), raising=False)
    reader = StreamingFileReader(path, chunk_size=10, input_format="csv")
    await _read_all(reader)
    assert threads
    assert all(name.startswith("dk-aiter-sync-pump") for name in threads)


async def test_text_chunks_open_on_worker_thread(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "data.txt"
    path.write_text("a\nb\nc\n")
    threads: list[str] = []
    monkeypatch.setattr(_module, "open", _thread_recording_open(threads), raising=False)
    reader = StreamingFileReader(path, chunk_size=2, input_format="text")
    await _read_all(reader)
    assert threads
    assert all(name.startswith("dk-aiter-sync-pump") for name in threads)


async def test_streams_larger_than_one_chunk(tmp_path: Path) -> None:
    """A file larger than one chunk yields multiple chunks lazily.

    Proves the ``AsyncIterator`` streaming contract is preserved: with a
    small chunk size over many records the reader emits many bounded
    chunks rather than one whole-file chunk.
    """
    path = tmp_path / "big.jsonl"
    _write_jsonl(path, 250)
    reader = StreamingFileReader(path, chunk_size=10, input_format="jsonl")
    chunk_count = 0
    seen = 0
    async for chunk in reader.read_chunks():
        chunk_count += 1
        seen += len(chunk.data)
        assert len(chunk.data) <= 10
    assert chunk_count == 25
    assert seen == 250


# --------------------------------------------------------------------------
# Reproduce-first: the sibling WRITER must not open/flush/close on the loop.
# ``StreamingFileReader`` was offloaded above; ``StreamingFileWriter`` lives
# in the same module and had the same defect (blocking ``open`` + buffer
# flush + close on the event loop). The offload moves each onto a worker
# thread via ``asyncio.to_thread``.
# --------------------------------------------------------------------------


@requires_blockbuster
async def test_writer_does_not_block(tmp_path: Path) -> None:
    """The writer's open + flush + close must not run on the event loop.

    Pre-offload the auto-``open()`` (and the buffer flush + close) ran
    inline on the loop, so blockbuster trips on the ``open()``. Post-offload
    they run via ``asyncio.to_thread`` and this PASSES.
    """
    path = tmp_path / "out.jsonl"
    writer = StreamingFileWriter(path, output_format="jsonl")
    chunk = StreamChunk(data=[{"i": 0}, {"i": 1}], is_last=True)
    with assert_no_blocking():
        await writer.write_chunk(chunk)  # auto-opens off-loop, flushes (is_last)
        await writer.close()
    records = [json.loads(ln) for ln in path.read_text().splitlines() if ln]
    assert records == [{"i": 0}, {"i": 1}]


async def test_writer_opens_on_worker_thread(
    tmp_path: Path, monkeypatch
) -> None:
    """Structural proof the writer's blocking ``open`` runs off the loop.

    Records the thread ``open()`` is invoked on (the flush's ``json.dump`` /
    ``f.write`` may be detector-blind). Pre-offload ``open`` ran on the
    event-loop thread (FAIL); post-offload it runs on a ``to_thread``
    worker, never the loop thread (PASS).
    """
    path = tmp_path / "out.jsonl"
    threads: list[str] = []
    monkeypatch.setattr(_module, "open", _thread_recording_open(threads), raising=False)
    writer = StreamingFileWriter(path, output_format="jsonl")
    await writer.write_chunk(StreamChunk(data=[{"i": 0}], is_last=True))
    await writer.close()
    assert threads, "open() was never called"
    assert threading.current_thread().name not in threads


async def test_writer_json_format_round_trips(tmp_path: Path) -> None:
    """The JSON-array writer still produces a single valid array post-offload.

    Guards the offloaded ``open`` (writes ``[``), flush (comma-joined
    elements), and ``_close_file`` (writes ``]``) across two chunks.
    """
    path = tmp_path / "out.json"
    writer = StreamingFileWriter(path, output_format="json")
    await writer.write_chunk(StreamChunk(data=[{"i": 0}, {"i": 1}]))
    await writer.write_chunk(StreamChunk(data=[{"i": 2}], is_last=True))
    await writer.close()
    assert json.loads(path.read_text()) == [{"i": 0}, {"i": 1}, {"i": 2}]
