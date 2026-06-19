"""Reproduce-first async-correctness tests for ``patterns/file_processing.py``.

``FileProcessor``'s three I/O methods run blocking file access on the
event loop: ``_process_whole`` reads the whole file (``f.read()``),
``_read_batches`` opens and line-iterates lazily, and ``_write_output``
writes the whole result set. The offload drives the lazy batch reader on
a worker thread via ``aiter_sync_in_thread`` and the whole read/write via
``asyncio.to_thread``.

Each blocking test wraps the awaited method in :func:`assert_no_blocking`.
Pre-offload these FAIL on the ``open()``; post-offload they PASS. The
functional tests guard the extraction: batch boundaries are honoured, the
whole-file path processes every record, and written output round-trips.
These call the processor's I/O methods directly to isolate the disk I/O
from the FSM execution machinery (a legitimate internal-logic unit test).
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_fsm.patterns import file_processing as _module
from dataknobs_fsm.patterns.file_processing import (
    FileFormat,
    FileProcessingConfig,
    FileProcessor,
    ProcessingMode,
)

if TYPE_CHECKING:
    from pathlib import Path


def _thread_recording_open(record: list[str]):
    """A real ``open`` that records the invoking thread (see file_utils test)."""
    real_open = open

    def spy(*args, **kwargs):
        record.append(threading.current_thread().name)
        return real_open(*args, **kwargs)

    return spy


def _processor(
    input_path: Path | None = None,
    output_path: Path | None = None,
    *,
    fmt: FileFormat = FileFormat.JSON,
    output_format: FileFormat | None = None,
    mode: ProcessingMode = ProcessingMode.BATCH,
    chunk_size: int = 1000,
) -> FileProcessor:
    return FileProcessor(
        FileProcessingConfig(
            input_path=str(input_path) if input_path else "unused.jsonl",
            output_path=str(output_path) if output_path else None,
            format=fmt,
            output_format=output_format,
            mode=mode,
            chunk_size=chunk_size,
        )
    )


async def _read_batches(proc: FileProcessor) -> list:
    return [batch async for batch in proc._read_batches()]


# --------------------------------------------------------------------------
# Reproduce-first: the I/O methods must not open/read/write on the loop.
# --------------------------------------------------------------------------


@requires_blockbuster
async def test_read_batches_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text("".join(f'{{"i": {i}}}\n' for i in range(5)))
    proc = _processor(path, fmt=FileFormat.JSON, mode=ProcessingMode.BATCH, chunk_size=2)
    with assert_no_blocking():
        batches = await _read_batches(proc)
    assert sum(len(b) for b in batches) == 5


@requires_blockbuster
async def test_process_whole_does_not_block(tmp_path: Path) -> None:
    """``_process_whole`` must not block the loop end-to-end.

    Two offloads make this pass: the whole-file read moved off the loop
    (proven structurally by ``test_process_whole_read_runs_off_event_loop``),
    and the FSM execution that follows it switched from the synchronous
    ``SimpleFSM`` sync-bridge (which blocked the loop on
    ``threading.Lock.acquire`` via ``run_coroutine_threadsafe(...).result()``)
    to awaiting ``AsyncSimpleFSM`` directly. See
    ``test_fsm_execution_offload.py`` for the per-mode FSM-execution
    reproduce-first coverage.
    """
    path = tmp_path / "data.json"
    path.write_text('[{"i": 0}, {"i": 1}, {"i": 2}]')
    proc = _processor(path, fmt=FileFormat.JSON, mode=ProcessingMode.WHOLE)
    with assert_no_blocking():
        metrics = await proc._process_whole()
    assert metrics["records_processed"] == 3


async def test_process_whole_read_runs_off_event_loop(
    tmp_path: Path, monkeypatch
) -> None:
    """Tighter reproduce-first for the in-scope fix: the whole-file read.

    The detector cannot prove this in ``_process_whole`` because the FSM
    execution that follows the read blocks first (see the standing-signal
    test above). So this proves the read offload structurally: the read's
    ``open()`` must run on an ``asyncio.to_thread`` worker, not the
    event-loop thread. Pre-offload it ran on the loop thread (FAIL);
    post-offload it runs on a worker thread (PASS).
    """
    path = tmp_path / "data.json"
    path.write_text('[{"i": 0}, {"i": 1}, {"i": 2}]')
    proc = _processor(path, fmt=FileFormat.JSON, mode=ProcessingMode.WHOLE)
    threads: list[str] = []
    monkeypatch.setattr(_module, "open", _thread_recording_open(threads), raising=False)
    await proc._process_whole()
    assert threads, "open() was never called"
    assert threading.current_thread().name not in threads


async def test_process_whole_processes_all_records(tmp_path: Path) -> None:
    """Functional: the offloaded read + parse still processes every record."""
    path = tmp_path / "data.json"
    path.write_text('[{"i": 0}, {"i": 1}, {"i": 2}]')
    proc = _processor(path, fmt=FileFormat.JSON, mode=ProcessingMode.WHOLE)
    metrics = await proc._process_whole()
    assert metrics["records_processed"] == 3


@requires_blockbuster
async def test_write_output_does_not_block(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    proc = _processor(
        output_path=out, fmt=FileFormat.JSON, output_format=FileFormat.JSON
    )
    results = [
        {"data": {"x": 1}, "success": True},
        {"data": {"x": 2}, "success": True},
    ]
    with assert_no_blocking():
        await proc._write_output(results)
    assert json.loads(out.read_text()) == [{"x": 1}, {"x": 2}]


# --------------------------------------------------------------------------
# Functional: batch boundaries, whole-file fidelity, output round-trip.
# --------------------------------------------------------------------------


async def test_read_batches_honours_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text("".join(f'{{"i": {i}}}\n' for i in range(5)))
    proc = _processor(path, fmt=FileFormat.JSON, mode=ProcessingMode.BATCH, chunk_size=2)
    batches = await _read_batches(proc)
    assert [len(b) for b in batches] == [2, 2, 1]
    flat = [rec["i"] for b in batches for rec in b]
    assert flat == [0, 1, 2, 3, 4]
    assert proc._metrics["lines_read"] == 5


async def test_read_batches_opens_on_worker_thread(
    tmp_path: Path, monkeypatch
) -> None:
    """Structural reproduce-first for the line-iterating batch reader.

    The detector cannot see ``readline``; this proves the open + read run
    on the ``aiter_sync_in_thread`` worker. Pre-offload it FAILS (open on
    the event-loop thread); post-offload it PASSES (dk-aiter-sync-pump).
    """
    path = tmp_path / "data.jsonl"
    path.write_text("".join(f'{{"i": {i}}}\n' for i in range(3)))
    proc = _processor(path, fmt=FileFormat.JSON, mode=ProcessingMode.BATCH, chunk_size=2)
    threads: list[str] = []
    monkeypatch.setattr(_module, "open", _thread_recording_open(threads), raising=False)
    await _read_batches(proc)
    assert threads, "open() was never called"
    assert threading.current_thread().name not in threads
    assert all(name.startswith("dk-aiter-sync-pump") for name in threads)


async def test_read_batches_counts_json_errors(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"i": 0}\nnot json\n{"i": 1}\n')
    proc = _processor(path, fmt=FileFormat.JSON, mode=ProcessingMode.BATCH, chunk_size=10)
    batches = await _read_batches(proc)
    flat = [rec for b in batches for rec in b]
    assert flat == [{"i": 0}, {"i": 1}]
    assert proc._metrics["errors"] == 1


async def test_write_output_csv_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    proc = _processor(
        output_path=out, fmt=FileFormat.CSV, output_format=FileFormat.CSV
    )
    results = [
        {"data": {"name": "alice", "value": "1"}, "success": True},
        {"data": {"name": "bob", "value": "2"}, "success": True},
        {"data": {"name": "skip", "value": "9"}, "success": False},
    ]
    await proc._write_output(results)
    lines = [ln for ln in out.read_text().splitlines() if ln]
    assert lines[0] == "name,value"
    assert "alice,1" in lines
    assert "bob,2" in lines
    assert "skip,9" not in lines
