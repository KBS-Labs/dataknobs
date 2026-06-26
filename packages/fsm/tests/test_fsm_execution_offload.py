"""Reproduce-first tests for the FSM-execution sync-bridge offload.

``FileProcessor`` previously built a synchronous ``SimpleFSM`` and called
its sync ``process``/``process_batch``/``process_stream`` from inside its
own ``async def`` methods. Those sync methods bridge to a background-thread
loop via ``asyncio.run_coroutine_threadsafe(...).result()``, which blocks
the *calling* thread on a ``threading.Lock.acquire`` — and since the caller
is the event-loop thread, the loop stalls. The fix builds an
``AsyncSimpleFSM`` and ``await``s its async methods directly (the sync
methods were only ever thin bridges over those same async methods).

Each blocking test wraps ``await processor.process()`` in
:func:`assert_no_blocking`. Against the pre-fix code these FAIL with
``blockbuster.BlockingError`` (``lock.acquire`` on the loop); after the
swap they PASS. The functional tests assert deterministic execution
outcomes per mode so the engine swap (and the stream-path crash fixes it
required) cannot silently regress — this path had no end-to-end coverage
before.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.patterns.file_processing import (
    FileFormat,
    FileProcessingConfig,
    FileProcessor,
    ProcessingMode,
)

if TYPE_CHECKING:
    from pathlib import Path

_JSON_ARRAY = '[{"i": 0}, {"i": 1}, {"i": 2}]'
_JSONL = '{"i": 0}\n{"i": 1}\n{"i": 2}\n'

# Minimal passthrough FSM (start -> end) for exercising process_stream
# directly, independent of FileProcessor's config-typed str paths.
_PASSTHROUGH_CONFIG = {
    "name": "passthrough",
    "main_network": "main",
    "networks": [
        {
            "name": "main",
            "states": [
                {"name": "input", "is_start": True},
                {"name": "output", "is_end": True},
            ],
            "arcs": [{"from": "input", "to": "output", "name": "done"}],
        }
    ],
}


def _processor(
    input_path: Path,
    output_path: Path | None,
    mode: ProcessingMode,
) -> FileProcessor:
    return FileProcessor(
        FileProcessingConfig(
            input_path=str(input_path),
            output_path=str(output_path) if output_path else None,
            format=FileFormat.JSON,
            mode=mode,
            chunk_size=2,
        )
    )


# --------------------------------------------------------------------------
# Reproduce-first: process() must not block the event loop in any mode.
# --------------------------------------------------------------------------


@requires_blockbuster
async def test_whole_mode_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "in.json"
    path.write_text(_JSON_ARRAY)
    proc = _processor(path, None, ProcessingMode.WHOLE)
    with assert_no_blocking():
        metrics = await proc.process()
    assert metrics["records_processed"] == 3


@requires_blockbuster
async def test_batch_mode_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text(_JSONL)
    out = tmp_path / "out.jsonl"
    proc = _processor(path, out, ProcessingMode.BATCH)
    with assert_no_blocking():
        metrics = await proc.process()
    assert metrics["lines_read"] == 3


@requires_blockbuster
async def test_stream_mode_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text(_JSONL)
    out = tmp_path / "out.jsonl"
    proc = _processor(path, out, ProcessingMode.STREAM)
    with assert_no_blocking():
        metrics = await proc.process()
    # STREAM now reports the unified metrics shape (records_processed), the
    # same keys as BATCH/WHOLE.
    assert metrics["records_processed"] == 3


# --------------------------------------------------------------------------
# Functional: the engine swap (+ stream-path crash fixes) executes each
# mode end-to-end. This path had no prior end-to-end coverage.
# --------------------------------------------------------------------------


async def test_whole_mode_processes_records(tmp_path: Path) -> None:
    path = tmp_path / "in.json"
    path.write_text(_JSON_ARRAY)
    proc = _processor(path, None, ProcessingMode.WHOLE)
    metrics = await proc.process()
    assert metrics["records_processed"] == 3
    assert metrics["errors"] == 0


async def test_batch_mode_reads_all_lines(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text(_JSONL)
    out = tmp_path / "out.jsonl"
    proc = _processor(path, out, ProcessingMode.BATCH)
    metrics = await proc.process()
    assert metrics["lines_read"] == 3


async def test_stream_mode_processes_all_records(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text(_JSONL)
    out = tmp_path / "out.jsonl"
    proc = _processor(path, out, ProcessingMode.STREAM)
    metrics = await proc.process()
    # STREAM reports the unified metrics shape (records_processed /
    # records_written), the same keys as BATCH/WHOLE.
    assert metrics["records_processed"] == 3
    assert metrics["records_written"] == 3


# --------------------------------------------------------------------------
# Review-feedback: the regular-mode JSON sink cleanup, the compression
# guard, and Path sources on process_stream.
# --------------------------------------------------------------------------


@requires_blockbuster
async def test_stream_mode_json_sink_cleanup_does_not_block(
    tmp_path: Path,
) -> None:
    """A ``.json`` stream sink must not flush its whole-file cleanup on the loop.

    This exercises the non-streaming sink branch (``use_streaming`` False):
    ``process_stream`` routes a ``.json`` sink through the sync
    ``create_file_writer``, which yields ``create_json_writer`` whose
    ``cleanup()`` accumulates every result and writes the whole array at
    once. ``process_stream`` runs that sync cleanup in its ``finally`` —
    pre-fix inline on the loop (``open`` + ``json.dump``), so blockbuster
    trips; post-fix offloaded via ``asyncio.to_thread``, so it PASSES. The
    other stream tests use a ``.jsonl`` sink (whose writer has no cleanup),
    so none exercised this.
    """
    path = tmp_path / "in.jsonl"
    path.write_text(_JSONL)
    out = tmp_path / "out.json"  # .json -> create_json_writer (whole-file cleanup)
    proc = _processor(path, out, ProcessingMode.STREAM)
    with assert_no_blocking():
        await proc.process()
    # The cleanup ran in the finally and produced a valid JSON-array sink.
    assert isinstance(json.loads(out.read_text()), list)


async def test_process_rejects_compression(tmp_path: Path) -> None:
    """Compressed output is rejected loudly, not silently dropped.

    No FileProcessor execution path writes compressed output (the former
    stream-mode ``FileStreamSink`` path was removed when the pattern moved
    onto the async engine), so a configured ``compression`` would otherwise
    be silently ignored — the caller would get uncompressed bytes believing
    they were compressed. ``process`` now raises ``NotImplementedError``.
    """
    path = tmp_path / "in.jsonl"
    path.write_text(_JSONL)
    proc = FileProcessor(
        FileProcessingConfig(
            input_path=str(path),
            output_path=str(tmp_path / "out.jsonl.gz"),
            format=FileFormat.JSON,
            mode=ProcessingMode.STREAM,
            compression="gzip",
        )
    )
    with pytest.raises(NotImplementedError, match="compressed output"):
        await proc.process()


async def test_process_stream_accepts_path_source(tmp_path: Path) -> None:
    """``process_stream`` accepts a ``Path`` source, not only ``str``.

    The mode branch gated on ``isinstance(source, str)``; a ``Path`` fell
    through to the "treat as async iterator" branch and broke when the
    executor tried to ``async for`` over it. Now ``str`` and ``Path`` are
    both routed to the file reader. Pre-fix this raises; post-fix it reads
    the file and processes every record.
    """
    src = tmp_path / "in.jsonl"
    src.write_text(_JSONL)
    fsm = AsyncSimpleFSM(_PASSTHROUGH_CONFIG)
    try:
        result = await fsm.process_stream(source=src, chunk_size=2)  # Path, not str
    finally:
        await fsm.close()
    assert result["total_processed"] == 3
