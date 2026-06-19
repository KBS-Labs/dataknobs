"""Reproduce-first async-correctness tests for ``functions/library/streaming.py``.

``ChunkReader.transform`` and ``FileAppender._write_buffer`` are
``async def`` methods whose bodies are entirely synchronous file I/O:
``Path.exists`` / ``Path.stat`` stats plus blocking ``open()`` reads and
writes. Each is a single bounded read/write (not a lazy stream), so the
offload runs the whole body on a worker thread via ``asyncio.to_thread``.

Each blocking test wraps a single awaited call in
:func:`assert_no_blocking`. Pre-offload these FAIL on the stat/``open()``;
post-offload they PASS. The functional tests confirm the extracted sync
helpers still chunk by offset, signal ``has_more`` correctly, and append
records faithfully across formats.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_fsm.functions.library.streaming import ChunkReader, FileAppender

if TYPE_CHECKING:
    from pathlib import Path


# --------------------------------------------------------------------------
# Reproduce-first: ChunkReader.transform must not stat/open on the loop.
# --------------------------------------------------------------------------


@requires_blockbuster
async def test_chunk_reader_json_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('[{"i": 0}, {"i": 1}, {"i": 2}]')
    reader = ChunkReader(str(path), chunk_size=2, format="json")
    with assert_no_blocking():
        result = await reader.transform({})
    assert result["chunk"] == [{"i": 0}, {"i": 1}]
    assert result["has_more"] is True


@requires_blockbuster
async def test_chunk_reader_csv_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("name,value\nalice,1\nbob,2\n")
    reader = ChunkReader(str(path), chunk_size=10, format="csv")
    with assert_no_blocking():
        result = await reader.transform({})
    assert result["chunk"] == [
        {"name": "alice", "value": "1"},
        {"name": "bob", "value": "2"},
    ]
    assert result["has_more"] is False


@requires_blockbuster
async def test_chunk_reader_lines_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("first\nsecond\n")
    reader = ChunkReader(str(path), chunk_size=10, format="lines")
    with assert_no_blocking():
        result = await reader.transform({})
    assert result["chunk"] == [{"line": "first"}, {"line": "second"}]


# --------------------------------------------------------------------------
# Reproduce-first: FileAppender._write_buffer must not stat/open on the loop.
# --------------------------------------------------------------------------


@requires_blockbuster
async def test_file_appender_json_flush_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    appender = FileAppender(str(path), format="json", field="data", buffer_size=100)
    await appender.transform({"data": [{"x": 1}, {"x": 2}]})
    with assert_no_blocking():
        written = await appender.flush()
    assert written == 2
    assert json.loads(path.read_text()) == [{"x": 1}, {"x": 2}]


@requires_blockbuster
async def test_file_appender_lines_flush_does_not_block(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    appender = FileAppender(str(path), format="lines", field="data", buffer_size=100)
    await appender.transform({"data": [{"x": 1}]})
    with assert_no_blocking():
        await appender.flush()
    assert path.read_text().strip() == json.dumps({"x": 1})


# --------------------------------------------------------------------------
# Functional: offset chunking, has_more, multi-format append.
# --------------------------------------------------------------------------


async def test_chunk_reader_advances_by_offset(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('[{"i": 0}, {"i": 1}, {"i": 2}]')
    reader = ChunkReader(str(path), chunk_size=2, format="json")

    first = await reader.transform({})
    assert first["chunk"] == [{"i": 0}, {"i": 1}]
    assert first["has_more"] is True

    second = await reader.transform(first)
    assert second["chunk"] == [{"i": 2}]
    assert second["has_more"] is False


async def test_file_appender_json_accumulates_across_flushes(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    appender = FileAppender(str(path), format="json", field="data", buffer_size=100)

    await appender.transform({"data": [{"x": 1}]})
    await appender.flush()
    await appender.transform({"data": [{"x": 2}]})
    await appender.flush()

    assert json.loads(path.read_text()) == [{"x": 1}, {"x": 2}]


async def test_file_appender_csv_writes_header_once(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    appender = FileAppender(str(path), format="csv", field="data", buffer_size=100)

    await appender.transform({"data": [{"name": "alice", "value": "1"}]})
    await appender.flush()
    await appender.transform({"data": [{"name": "bob", "value": "2"}]})
    await appender.flush()

    lines = [ln for ln in path.read_text().splitlines() if ln]
    assert lines[0] == "name,value"
    assert lines.count("name,value") == 1
    assert "alice,1" in lines
    assert "bob,2" in lines
