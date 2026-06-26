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


# --------------------------------------------------------------------------
# Whole-file mode: filter/validation exclusion must behave identically to
# batch/stream (the cross-mode sink-policy guarantee). Pre-fix only batch was
# covered end-to-end; whole/stream exclusion went untested.
# --------------------------------------------------------------------------


async def test_whole_filter_excludes_records(tmp_path: Path) -> None:
    src, out = tmp_path / "in.json", tmp_path / "out.json"
    _seed_json_array(src, _RECORDS)
    metrics = await _processor(
        src, out, ProcessingMode.WHOLE, filters=[_keep_even]
    ).process()
    assert {r["id"] for r in _read_records(out)} == {0, 2}
    assert metrics["records_written"] == 2
    assert metrics["skipped"] == 1
    assert metrics["errors"] == 0


async def test_whole_validation_excludes_invalid(tmp_path: Path) -> None:
    src, out = tmp_path / "in.json", tmp_path / "out.json"
    _seed_json_array(
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
        ProcessingMode.WHOLE,
        validation_schema={"name": {"required": True}},
    ).process()
    assert {r["id"] for r in _read_records(out)} == {0, 2}
    assert metrics["errors"] == 1


# --------------------------------------------------------------------------
# STREAM metrics shape + explicit-format honoring. STREAM is the default mode,
# so its metrics must populate the same keys as batch/whole, and an explicit
# config ``format`` must override extension-based auto-detection.
# --------------------------------------------------------------------------


async def test_stream_metrics_report_written_and_processed(
    tmp_path: Path,
) -> None:
    """STREAM mode must return the unified metrics shape.

    Pre-fix the stream path did ``self._metrics.update(result)`` with a result
    carrying only ``total_processed``/``successful``/``failed``, leaving
    ``records_processed``/``records_written`` permanently 0. This FAILS pre-fix
    and PASSES once the stream stats are mapped onto the shared keys.
    """
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    _seed_jsonl(src, _RECORDS)
    metrics = await _processor(src, out, ProcessingMode.STREAM).process()
    assert metrics["records_processed"] == 3
    assert metrics["records_written"] == 3
    assert metrics["skipped"] == 0
    assert metrics["errors"] == 0


async def test_stream_filter_metrics_count_skipped(tmp_path: Path) -> None:
    """STREAM filter exclusion must surface in the unified metrics."""
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    _seed_jsonl(src, _RECORDS)
    metrics = await _processor(
        src, out, ProcessingMode.STREAM, filters=[_keep_even]
    ).process()
    assert metrics["records_written"] == 2
    assert metrics["skipped"] == 1
    assert metrics["errors"] == 0


async def test_stream_validation_metrics_count_errors(tmp_path: Path) -> None:
    """STREAM validation rejections must count as ``errors``, not ``skipped``.

    A record that fails validation routes cleanly to the non-emitting ``error``
    terminal (``success=True``, ``emit=False``) — distinct from a record routed
    to ``filtered``. Batch/whole classify the ``error`` terminal as an error;
    STREAM must do the same so ``metrics['errors']`` is a reliable data-quality
    signal across modes. Pre-fix the stream path inferred ``skipped`` as a
    ``total - failed - written`` remainder, so a rejected record landed in
    ``skipped`` with ``errors == 0``. This FAILS pre-fix and PASSES once the
    executor reports excluded records bucketed by terminal name.
    """
    src, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    _seed_jsonl(
        src,
        [
            {"id": 0, "name": "alice"},
            {"id": 1},  # missing required name -> error terminal
            {"id": 2, "name": "carol"},
        ],
    )
    metrics = await _processor(
        src,
        out,
        ProcessingMode.STREAM,
        validation_schema={"name": {"required": True}},
    ).process()
    assert {r["id"] for r in _read_records(out)} == {0, 2}
    assert metrics["records_written"] == 2
    assert metrics["errors"] == 1
    assert metrics["skipped"] == 0
    assert metrics["records_processed"] == 2


async def test_stream_validation_metrics_match_batch_and_whole(
    tmp_path: Path,
) -> None:
    """The same validation input yields identical metrics in all three modes.

    Pins the cross-mode uniformity the unified-metrics contract advertises: a
    rejected record is counted the same way (``errors``) whether processed via
    STREAM, BATCH, or WHOLE.
    """
    records = [
        {"id": 0, "name": "alice"},
        {"id": 1},  # invalid
        {"id": 2, "name": "carol"},
    ]
    schema = {"name": {"required": True}}
    keys = ("records_written", "skipped", "errors", "records_processed")

    def _seed(path: Path, mode: ProcessingMode) -> None:
        if mode is ProcessingMode.WHOLE:
            _seed_json_array(path, records)
        else:
            _seed_jsonl(path, records)

    metrics_by_mode = {}
    for mode in (
        ProcessingMode.STREAM,
        ProcessingMode.BATCH,
        ProcessingMode.WHOLE,
    ):
        src = tmp_path / f"in_{mode.value}.jsonl"
        out = tmp_path / f"out_{mode.value}.jsonl"
        _seed(src, mode)
        metrics = await _processor(
            src, out, mode, validation_schema=schema
        ).process()
        metrics_by_mode[mode] = {k: metrics[k] for k in keys}

    expected = {
        "records_written": 2,
        "skipped": 0,
        "errors": 1,
        "records_processed": 2,
    }
    for mode, observed in metrics_by_mode.items():
        assert observed == expected, f"{mode.value} metrics diverged: {observed}"


async def test_stream_honors_explicit_input_format_over_extension(
    tmp_path: Path,
) -> None:
    """An explicit ``format`` must override extension-based auto-detection.

    A ``.log``-extensioned file whose lines are JSON: auto-detection treats
    ``.log`` as text (records arrive as ``{'text': line}``), so the parsed
    fields never appear in the output. With ``format=JSON`` the streaming
    reader must parse each line as JSON. This FAILS pre-fix (the stream path
    hardcoded ``input_format='auto'``) and PASSES once the resolved format is
    forwarded.
    """
    src, out = tmp_path / "events.log", tmp_path / "out.jsonl"
    _seed_jsonl(src, _RECORDS)  # JSON lines written into a .log file
    proc = FileProcessor(
        FileProcessingConfig(
            input_path=str(src),
            output_path=str(out),
            format=FileFormat.JSON,
            output_format=FileFormat.JSON,
            mode=ProcessingMode.STREAM,
            chunk_size=2,
        )
    )
    await proc.process()
    rows = _read_records(out)
    assert {r.get("id") for r in rows} == {0, 1, 2}, (
        "explicit format=JSON was ignored; .log lines were not parsed as JSON"
    )
