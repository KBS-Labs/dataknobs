"""Reproduce-first flow-correctness tests for ``patterns/file_processing.py``.

These pin the *functional* behavior the pattern lacked: records must
traverse the FSM to ``complete`` and land in the output for every
processing mode, transforms/filters/validators must actually execute, and
filtered/invalid records must be excluded from the output.

Pre-fix the pattern is broken end-to-end: with *any* config (even a pure
passthrough) records dead-end at the ``filter`` state and never reach
``write``/``complete`` — BATCH/STREAM mark every record errored/failed and
write nothing, while the configured filter/transform functions never run
(they were referenced by name from inline ``eval`` code that cannot see
them). These tests FAIL against that code and PASS once:

- ``_build_arcs`` connects only the *enabled* stages so no stage dead-ends;
- transforms/aggregators are wired as ``ITransformFunction`` state
  transforms and filters/validators as registered arc conditions (the
  proven ``custom_functions=`` idiom);
- BATCH writes its output; filtered records route to a non-emitting
  terminal so they are excluded from the output in every mode;
- STREAM runs on the same async engine as BATCH/WHOLE (one shared path).

Real constructs only — a real ``FileProcessor`` over real temp files, with
the output reopened and asserted (no mocks).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from dataknobs_fsm.patterns.file_processing import (
    FileFormat,
    FileProcessingConfig,
    FileProcessor,
    ProcessingMode,
)

if TYPE_CHECKING:
    from pathlib import Path


def _seed_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _seed_json_array(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records))


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Read output records back, tolerating a JSON array or JSONL sink."""
    text = path.read_text().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _processor(
    src: Path,
    out: Path,
    mode: ProcessingMode,
    **config_kwargs: Any,
) -> FileProcessor:
    return FileProcessor(
        FileProcessingConfig(
            input_path=str(src),
            output_path=str(out),
            format=FileFormat.JSON,
            output_format=FileFormat.JSON,
            mode=mode,
            chunk_size=2,
            **config_kwargs,
        )
    )


_RECORDS = [{"id": 0, "v": 0}, {"id": 1, "v": 10}, {"id": 2, "v": 20}]


# --------------------------------------------------------------------------
# Passthrough: every record must reach the output in every mode.
# --------------------------------------------------------------------------


async def test_batch_passthrough_writes_all_records(tmp_path: Path) -> None:
    src, out = tmp_path / "in.jsonl", tmp_path / "out.json"
    _seed_jsonl(src, _RECORDS)
    metrics = await _processor(src, out, ProcessingMode.BATCH).process()
    assert metrics["errors"] == 0
    assert metrics["records_written"] == 3
    assert {r["id"] for r in _read_records(out)} == {0, 1, 2}


async def test_stream_passthrough_writes_all_records(tmp_path: Path) -> None:
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    _seed_jsonl(src, _RECORDS)
    await _processor(src, out, ProcessingMode.STREAM).process()
    assert {r["id"] for r in _read_records(out)} == {0, 1, 2}


async def test_whole_passthrough_writes_all_records(tmp_path: Path) -> None:
    src, out = tmp_path / "in.json", tmp_path / "out.json"
    _seed_json_array(src, _RECORDS)
    metrics = await _processor(src, out, ProcessingMode.WHOLE).process()
    assert metrics["records_processed"] == 3
    assert {r["id"] for r in _read_records(out)} == {0, 1, 2}


# --------------------------------------------------------------------------
# Transform: the configured transform must execute and reach the output.
# --------------------------------------------------------------------------


def _tag(record: dict[str, Any]) -> dict[str, Any]:
    return {**record, "tag": "X"}


async def test_batch_transform_applied_to_output(tmp_path: Path) -> None:
    src, out = tmp_path / "in.jsonl", tmp_path / "out.json"
    _seed_jsonl(src, _RECORDS)
    await _processor(
        src, out, ProcessingMode.BATCH, transformations=[_tag]
    ).process()
    rows = _read_records(out)
    assert len(rows) == 3
    assert all(r["tag"] == "X" for r in rows)


async def test_stream_transform_applied_to_output(tmp_path: Path) -> None:
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    _seed_jsonl(src, _RECORDS)
    await _processor(
        src, out, ProcessingMode.STREAM, transformations=[_tag]
    ).process()
    rows = _read_records(out)
    assert len(rows) == 3
    assert all(r["tag"] == "X" for r in rows)


async def test_whole_transform_applied_to_output(tmp_path: Path) -> None:
    src, out = tmp_path / "in.json", tmp_path / "out.json"
    _seed_json_array(src, _RECORDS)
    await _processor(
        src, out, ProcessingMode.WHOLE, transformations=[_tag]
    ).process()
    rows = _read_records(out)
    assert len(rows) == 3
    assert all(r["tag"] == "X" for r in rows)


# --------------------------------------------------------------------------
# Filter: a configured filter must gate; filtered records excluded from
# output in every mode (the cross-engine sink-policy guarantee).
# --------------------------------------------------------------------------


def _keep_even(record: dict[str, Any]) -> bool:
    return record["id"] % 2 == 0


async def test_batch_filter_excludes_records(tmp_path: Path) -> None:
    src, out = tmp_path / "in.jsonl", tmp_path / "out.json"
    _seed_jsonl(src, _RECORDS)
    metrics = await _processor(
        src, out, ProcessingMode.BATCH, filters=[_keep_even]
    ).process()
    assert {r["id"] for r in _read_records(out)} == {0, 2}
    assert metrics["records_written"] == 2
    assert metrics["skipped"] == 1
    assert metrics["errors"] == 0


async def test_stream_filter_excludes_records(tmp_path: Path) -> None:
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    _seed_jsonl(src, _RECORDS)
    await _processor(
        src, out, ProcessingMode.STREAM, filters=[_keep_even]
    ).process()
    assert {r["id"] for r in _read_records(out)} == {0, 2}


# --------------------------------------------------------------------------
# Validate: invalid records are routed away from the output, not written.
# --------------------------------------------------------------------------


async def test_batch_validation_excludes_invalid(tmp_path: Path) -> None:
    src, out = tmp_path / "in.jsonl", tmp_path / "out.json"
    _seed_jsonl(
        src,
        [
            {"id": 0, "name": "alice"},
            {"id": 1},  # missing required name -> invalid
            {"id": 2, "name": "carol"},
        ],
    )
    metrics = await _processor(
        src,
        out,
        ProcessingMode.BATCH,
        validation_schema={"name": {"required": True}},
    ).process()
    assert {r["id"] for r in _read_records(out)} == {0, 2}
    assert metrics["errors"] == 1
