# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Reproduce-first tests for the ETL incremental watermark.

``ETLMode.INCREMENTAL`` promises to extract only what changed. Two things
stood between it and that promise:

* the watermark field was the literal ``"updated_at"``, so a source that
  timestamps its rows under any other name could not be read incrementally
  at all; and
* the watermark *value* was read from ``self._checkpoint_data["last_timestamp"]``
  --- a key nothing in the class ever wrote. The filter was therefore never
  added, and every incremental run re-extracted the whole source.

So the configurable name is half of it: a name for a value that is never
produced narrows nothing. These tests pin both halves, the decisions that
make an advancing watermark safe --- an errored batch does not advance it
(the rows are re-extracted), a rejected row does not stall it (a
permanently-invalid row would freeze the pipeline forever) --- and the two
hazards a watermark brings with it: a row with no value in the column is
lost from the second run on, and a column holding two types is refused by
name rather than aborting on an ordering error deep in a batch.

Real constructs only: file-backed ``AsyncDatabase`` source and target.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from dataknobs_data import AsyncDatabase, Query, Record
from dataknobs_fsm.core.exceptions import ETLError
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode


async def _seed(path: str, rows: list[dict]) -> None:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        for row in rows:
            await db.upsert(row["id"], Record(dict(row)))
    finally:
        await db.close()


async def _loaded_ids(path: str) -> set[str]:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        return {record.to_dict()["id"] async for record in db.stream_read(Query())}
    finally:
        await db.close()


def _config(src: str, tgt: str, **overrides: Any) -> ETLConfig:
    return ETLConfig(
        source_db={"type": "file", "path": src},
        target_db={"type": "file", "path": tgt},
        target_table="records",
        key_columns=["id"],
        mode=ETLMode.INCREMENTAL,
        **overrides,
    )


async def test_a_second_run_extracts_only_what_changed(tmp_path: Path) -> None:
    """The promise of the mode, on the shipped default field name.

    The reproduce: nothing wrote the watermark, so the second run
    re-extracted every row the first had already loaded.
    """
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    await _seed(
        src,
        [
            {"id": "1", "name": "A", "updated_at": "2026-01-01"},
            {"id": "2", "name": "B", "updated_at": "2026-01-01"},
        ],
    )
    etl = DatabaseETL(_config(src, tgt))

    first = await etl.run()
    assert first["extracted"] == 2, first

    await _seed(src, [{"id": "3", "name": "C", "updated_at": "2026-02-01"}])
    second = await etl.run()

    assert second["extracted"] == 1, f"the second run re-extracted unchanged rows: {second}"
    assert await _loaded_ids(tgt) == {"1", "2", "3"}


async def test_the_watermark_field_is_configurable(tmp_path: Path) -> None:
    """A source that timestamps rows under its own name reads incrementally.

    ``updated_at`` was the literal in the filter, so this source --- which
    names the column ``modified_at`` --- had no incremental mode available
    to it at all.
    """
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    await _seed(
        src,
        [
            {"id": "1", "name": "A", "modified_at": "2026-01-01"},
            {"id": "2", "name": "B", "modified_at": "2026-01-01"},
        ],
    )
    etl = DatabaseETL(_config(src, tgt, watermark_field="modified_at"))

    await etl.run()
    await _seed(src, [{"id": "3", "name": "C", "modified_at": "2026-02-01"}])
    second = await etl.run()

    assert second["extracted"] == 1, f"the configured watermark field narrowed nothing: {second}"
    assert await _loaded_ids(tgt) == {"1", "2", "3"}


async def test_an_errored_batch_does_not_advance_the_watermark(tmp_path: Path) -> None:
    """A transient failure must not carry the watermark past the rows it lost.

    An advancing watermark is only safe if it advances over what actually
    completed. The row that errored is re-extracted on the next run; the
    load is an upsert keyed on ``key_columns``, so re-delivery is idempotent.
    """
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    await _seed(
        src,
        [
            {"id": "1", "name": "A", "updated_at": "2026-01-01"},
            {"id": "2", "name": "B", "updated_at": "2026-02-01"},
        ],
    )
    failing = {"2"}

    def _explode_once(record: dict) -> dict:
        if record["id"] in failing:
            raise RuntimeError("target write failed")
        return record

    etl = DatabaseETL(
        _config(
            src,
            tgt,
            batch_size=1,
            error_threshold=1.0,
            transformations=[_explode_once],
        )
    )

    first = await etl.run()
    assert first["errors"] == 1, first
    assert await _loaded_ids(tgt) == {"1"}

    failing.clear()
    second = await etl.run()

    assert second["extracted"] == 1, f"the errored row was not re-extracted: {second}"
    assert await _loaded_ids(tgt) == {"1", "2"}


async def test_a_rejected_row_does_not_stall_the_watermark(tmp_path: Path) -> None:
    """A validation reject is permanent, so it must not hold the watermark back.

    Rejections are a data-quality outcome, not a pipeline failure --- the
    same distinction ``_update_metrics`` already draws. Blocking on one
    would freeze the pipeline on the first invalid row forever, re-reading
    and re-rejecting it on every run.
    """
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    await _seed(
        src,
        [
            {"id": "1", "name": "A", "age": 30, "updated_at": "2026-01-01"},
            {"id": "2", "name": "B", "age": 9, "updated_at": "2026-02-01"},
        ],
    )
    etl = DatabaseETL(_config(src, tgt, validation_schema=lambda r: r.get("age", 0) >= 18))

    first = await etl.run()
    assert first["rejected"] == 1, first
    assert first["loaded"] == 1, first

    second = await etl.run()

    assert second["extracted"] == 0, f"the rejected row stalled the watermark: {second}"


async def test_the_watermark_survives_a_checkpoint_round_trip(tmp_path: Path) -> None:
    """Resuming from a checkpoint resumes the position, not just the counts.

    ``_save_checkpoint`` recorded metrics and a position but not the
    watermark, so a resumed run restarted from the beginning of the source
    while reporting the checkpoint's totals.
    """
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    etl = DatabaseETL(_config(src, tgt))

    etl._watermark = "2026-05-05"
    checkpoint_id = await etl._save_checkpoint()
    etl._watermark = None

    await etl._load_checkpoint(checkpoint_id)

    assert etl._watermark == "2026-05-05"


async def test_incremental_says_so_when_no_row_carries_the_watermark_field(
    tmp_path: Path, caplog: Any
) -> None:
    """A full scan wearing the word "incremental" is worth one line of log.

    The source names no such column, so the watermark cannot advance and
    every run re-reads everything. That is a configuration mistake the run
    can see and the operator cannot.
    """
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    await _seed(src, [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}])
    etl = DatabaseETL(_config(src, tgt))

    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm.patterns.etl"):
        await etl.run()

    assert "updated_at" in caplog.text, (
        f"nothing reported the missing watermark field: {caplog.text!r}"
    )


async def test_a_row_carrying_no_watermark_value_is_named(tmp_path: Path, caplog: Any) -> None:
    """A nullable watermark column loses rows, quietly, from the second run on.

    The row loads on the first run --- there is no watermark yet to exclude
    it --- and then ``value > watermark`` excludes a ``None`` as surely as
    an old value, so no later run sees it. Measured rather than reasoned
    about: the second run below extracts nothing at all.
    """
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    await _seed(
        src,
        [
            {"id": "1", "name": "A", "updated_at": None},
            {"id": "2", "name": "B", "updated_at": "2026-01-01"},
        ],
    )
    etl = DatabaseETL(_config(src, tgt))

    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm.patterns.etl"):
        first = await etl.run()

    assert first["loaded"] == 2, first
    assert "later runs will not see them" in caplog.text, caplog.text

    second = await etl.run()
    assert second["extracted"] == 0, (
        f"the unpositioned row came back, so the warning overstates the hazard: {second}"
    )


async def test_a_watermark_column_of_mixed_types_is_refused_by_name(tmp_path: Path) -> None:
    """Two types in the watermark column, named --- not a bare ``TypeError``.

    The comparison happens deep in a batch loop, so the default failure is
    an ordering error from whichever batch happens to straddle the two
    types, naming neither the field nor the values.
    """
    src, tgt = str(tmp_path / "src.json"), str(tmp_path / "tgt.json")
    await _seed(
        src,
        [
            {"id": "1", "name": "A", "updated_at": "2026-01-01"},
            {"id": "2", "name": "B", "updated_at": 7},
        ],
    )

    with pytest.raises(ETLError, match="cannot be ordered"):
        await DatabaseETL(_config(src, tgt)).run()
