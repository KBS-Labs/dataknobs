# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""``ChunkReader`` reads a stream source through the interface it declares.

``ChunkReader`` takes ``source: Union[str, IStreamSource]``. The file branch is
covered and works; the stream branch called ``source.read(chunk_size)`` and
iterated the result with ``async for``. :class:`IStreamSource` declares
``read_chunk() -> StreamChunk | None``, ``__iter__`` and ``close`` --- there is
no ``read``, and no shipped source has one --- so every stream source raised
``AttributeError`` on the first chunk. mypy named it (``"IStreamSource" has no
attribute "read"``) for as long as the branch has existed.

Real constructs only: a real :class:`FileStreamSource` over a real file on
disk, which is the shipped implementation of the interface.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_fsm.functions.library.streaming import ChunkReader
from dataknobs_fsm.streaming.file_stream import FileStreamSource

ROWS = [{"id": i, "name": f"row-{i}"} for i in range(5)]


def _jsonl(tmp_path: Path) -> Path:
    import json

    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in ROWS) + "\n")
    return path


async def test_a_stream_source_yields_its_first_chunk(tmp_path: Path) -> None:
    """The reproduce: the stream branch, exercised at all."""
    source = FileStreamSource(_jsonl(tmp_path), chunk_size=2)
    try:
        result = await ChunkReader(source, chunk_size=2).transform({})
    finally:
        source.close()

    assert result["chunk"] == ROWS[:2], result
    assert result["has_more"] is True, result


async def test_a_stream_source_reports_its_last_chunk(tmp_path: Path) -> None:
    """``has_more`` has to become False, or a pipeline over a stream never ends."""
    source = FileStreamSource(_jsonl(tmp_path), chunk_size=2)
    reader = ChunkReader(source, chunk_size=2)
    try:
        seen = []
        for _ in range(4):
            result = await reader.transform({})
            seen.extend(result["chunk"])
            if not result["has_more"]:
                break
    finally:
        source.close()

    assert seen == ROWS, seen
    assert result["has_more"] is False, result


async def test_an_exhausted_stream_source_yields_nothing(tmp_path: Path) -> None:
    """A source with nothing left answers an empty chunk rather than raising."""
    source = FileStreamSource(_jsonl(tmp_path), chunk_size=10)
    reader = ChunkReader(source, chunk_size=10)
    try:
        first = await reader.transform({})
        second = await reader.transform({})
    finally:
        source.close()

    assert first["chunk"] == ROWS, first
    assert second["chunk"] == [], second
    assert second["has_more"] is False, second
