"""Reproduce-first tests: per-record transform/load failures must be visible.

Before this fix the ETL pipeline reached a final state even when a state
transform raised — :meth:`BaseExecutionEngine.handle_transform_error` recorded
the offending state in ``context.failed_states`` but nothing consulted it. The
record still transitioned to ``complete``, was counted as both ``transformed``
and ``loaded``, ``errors`` stayed ``0``, and ``error_threshold`` never tripped.
A target-write outage during a migration would report a fully successful run —
the same silent-data-loss class the transform-wiring fix set out to eliminate,
just shifted from "nothing runs" to "failures are invisible".

These tests drive a user ``transformations`` callable that raises for a chosen
row. They FAIL against the swallow-and-count-as-loaded behavior (``errors == 0``)
and PASS once a failed record is reported as ``success=False`` and counted as an
error.

Real constructs only: file-backed ``AsyncDatabase`` source/target.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_data import AsyncDatabase, Record
from dataknobs_fsm.core.exceptions import ETLError
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode


async def _seed_source(path: str, rows: list[dict]) -> None:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        for row in rows:
            await db.upsert(row["id"], Record(dict(row)))
    finally:
        await db.close()


def _raise_on_id(target_id: str):
    def _transform(row: dict) -> dict:
        if row.get("id") == target_id:
            raise ValueError(f"boom on {target_id}")
        return {**row, "tag": "ok"}

    return _transform


@pytest.mark.asyncio
async def test_raising_transformation_counts_as_error(tmp_path: Path) -> None:
    """A raising transformation is counted as an error, not a loaded record."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(
        src,
        [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}, {"id": "3", "name": "C"}],
    )

    etl = DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": src},
            target_db={"type": "file", "path": tgt},
            source_query=None,
            target_table="records",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
            # High threshold so the run completes and we can inspect metrics.
            error_threshold=1.0,
            transformations=[_raise_on_id("2")],
        )
    )

    metrics = await etl.run()

    assert metrics["errors"] == 1, (
        f"a raising transformation was swallowed instead of counted as an error: {metrics}"
    )
    assert metrics["loaded"] == 2, (
        f"only the two clean rows should count as loaded: {metrics}"
    )
    assert metrics["transformed"] == 2
    assert metrics["extracted"] == 3


@pytest.mark.asyncio
async def test_non_dict_transformation_return_counts_as_error(tmp_path: Path) -> None:
    """A transformation returning a non-dict raises and is counted as an error."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, [{"id": "1", "name": "A"}])

    etl = DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": src},
            target_db={"type": "file", "path": tgt},
            source_query=None,
            target_table="records",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
            error_threshold=1.0,
            transformations=[lambda _r: None],  # type: ignore[return-value]
        )
    )

    metrics = await etl.run()

    assert metrics["errors"] == 1, (
        f"a non-dict transformation return must surface as an error: {metrics}"
    )
    assert metrics["loaded"] == 0


@pytest.mark.asyncio
async def test_error_threshold_trips_on_transform_failures(tmp_path: Path) -> None:
    """Transform failures push the error rate over the threshold and abort."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(
        src,
        [{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}],
    )

    etl = DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": src},
            target_db={"type": "file", "path": tgt},
            source_query=None,
            target_table="records",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
            error_threshold=0.05,  # 5% — any failure in a small batch trips it
            transformations=[_raise_on_id("2")],
        )
    )

    with pytest.raises(ETLError):
        await etl.run()
