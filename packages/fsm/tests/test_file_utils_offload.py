"""Reproduce-first async-correctness tests for ``utils/file_utils.py``.

The module-level file readers are ``async def`` generators, but each one
opens its file with a blocking ``open()`` and iterates it on the event
loop — stalling every other task for the duration of the read. Run on the
loop, that is exactly the blocking-under-async defect this offload closes.

Each blocking test drives a reader to exhaustion inside
:func:`assert_no_blocking`. Against the pre-offload code these FAIL with
``blockbuster.BlockingError`` (the ``open()`` runs on the loop); after the
offload — lazy readers via ``aiter_sync_in_thread``, the whole-file JSON
read via ``asyncio.to_thread`` — they PASS. The functional tests guard the
real regression risk of extracting sync helpers: records still round-trip,
malformed lines are still skipped, and JSON arrays/objects still flatten
correctly.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_fsm.utils import file_utils as _module
from dataknobs_fsm.utils.file_utils import (
    create_file_reader,
    read_csv_file,
    read_json_file,
    read_jsonl_file,
    read_text_file,
)

if TYPE_CHECKING:
    from pathlib import Path


async def _drain(aiter) -> list:
    return [item async for item in aiter]


def _thread_recording_open(record: list[str]):
    """A real ``open`` that records the thread it is invoked on.

    Not a mock — it opens the file for real and returns the real handle;
    it only notes ``threading.current_thread().name`` so a test can prove
    the read ran on the ``aiter_sync_in_thread`` worker (``dk-aiter-sync-pump``)
    rather than on the event-loop thread. This is the structural proof for
    line-iterating readers, which the blockbuster detector cannot see
    (it patches ``read``/``write`` but not ``readline``/line iteration).
    """
    real_open = open

    def spy(*args, **kwargs):
        record.append(threading.current_thread().name)
        return real_open(*args, **kwargs)

    return spy


# --------------------------------------------------------------------------
# Reproduce-first: the readers must not open/read on the event loop.
# --------------------------------------------------------------------------


@requires_blockbuster
async def test_read_jsonl_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n')
    with assert_no_blocking():
        records = await _drain(read_jsonl_file(path))
    assert records == [{"a": 1}, {"a": 2}]


@requires_blockbuster
async def test_read_json_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('[{"a": 1}, {"a": 2}]')
    with assert_no_blocking():
        records = await _drain(read_json_file(path))
    assert records == [{"a": 1}, {"a": 2}]


@requires_blockbuster
async def test_read_csv_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,value\nalice,1\nbob,2\n")
    with assert_no_blocking():
        records = await _drain(read_csv_file(path))
    assert records == [
        {"name": "alice", "value": "1"},
        {"name": "bob", "value": "2"},
    ]


@requires_blockbuster
async def test_read_text_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("first\nsecond\n")
    with assert_no_blocking():
        records = await _drain(read_text_file(path))
    assert records == [{"text": "first"}, {"text": "second"}]


@requires_blockbuster
async def test_create_file_reader_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n')
    with assert_no_blocking():
        records = await _drain(create_file_reader(path))
    assert records == [{"a": 1}, {"a": 2}]


# --------------------------------------------------------------------------
# Structural reproduce-first for the line-iterating readers. The detector
# cannot see ``readline``/line iteration, so these prove the open + read run
# on the helper's worker thread. Pre-offload they FAIL (open runs on the
# event-loop thread); post-offload they PASS (open runs on dk-aiter-sync-pump).
# --------------------------------------------------------------------------


async def test_read_jsonl_opens_on_worker_thread(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n')
    threads: list[str] = []
    monkeypatch.setattr(_module, "open", _thread_recording_open(threads), raising=False)
    await _drain(read_jsonl_file(path))
    assert threads, "open() was never called"
    assert threading.current_thread().name not in threads
    assert all(name.startswith("dk-aiter-sync-pump") for name in threads)


async def test_read_csv_opens_on_worker_thread(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,value\nalice,1\n")
    threads: list[str] = []
    monkeypatch.setattr(_module, "open", _thread_recording_open(threads), raising=False)
    await _drain(read_csv_file(path))
    assert threads
    assert all(name.startswith("dk-aiter-sync-pump") for name in threads)


async def test_read_text_opens_on_worker_thread(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "data.txt"
    path.write_text("first\n")
    threads: list[str] = []
    monkeypatch.setattr(_module, "open", _thread_recording_open(threads), raising=False)
    await _drain(read_text_file(path))
    assert threads
    assert all(name.startswith("dk-aiter-sync-pump") for name in threads)


# --------------------------------------------------------------------------
# Functional: extracting sync helpers must not change behaviour.
# --------------------------------------------------------------------------


async def test_jsonl_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\nnot json\n\n{"a": 2}\n')
    records = await _drain(read_jsonl_file(path))
    assert records == [{"a": 1}, {"a": 2}]


async def test_json_single_object_yields_once(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('{"a": 1}')
    records = await _drain(read_json_file(path))
    assert records == [{"a": 1}]


async def test_csv_headerless_columns(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("alice,1\nbob,2\n")
    records = await _drain(read_csv_file(path, has_header=False))
    assert records == [
        {"col_0": "alice", "col_1": "1"},
        {"col_0": "bob", "col_1": "2"},
    ]


async def test_text_keeps_empty_when_not_skipping(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("first\n\nsecond\n")
    kept = await _drain(read_text_file(path, skip_empty=False))
    assert {"text": ""} in kept
    skipped = await _drain(read_text_file(path, skip_empty=True))
    assert {"text": ""} not in skipped


async def test_create_file_reader_autodetects_csv(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,value\nalice,1\n")
    records = await _drain(create_file_reader(path))
    assert records == [{"name": "alice", "value": "1"}]


async def test_create_file_reader_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="Unsupported input format"):
        await _drain(create_file_reader(path, input_format="parquet"))
