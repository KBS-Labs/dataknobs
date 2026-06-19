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

from typing import TYPE_CHECKING

from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

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
    assert metrics["total_processed"] == 3


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
    assert metrics["total_processed"] == 3
