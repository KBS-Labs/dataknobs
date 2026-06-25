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

from dataknobs_data import AsyncDatabase, Query, Record
from dataknobs_fsm.core.exceptions import ETLError
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode


async def _seed_source(path: str, rows: list[dict]) -> None:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        for row in rows:
            await db.upsert(row["id"], Record(dict(row)))
    finally:
        await db.close()


async def _read_target(path: str) -> dict[str, dict]:
    """Reopen the target and return its rows keyed by id."""
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        return {
            r.to_dict()["id"]: r.to_dict()
            async for r in db.stream_read(Query())
        }
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


@pytest.mark.asyncio
async def test_failed_transform_record_not_loaded_to_target(tmp_path: Path) -> None:
    """A record whose transform raises must NOT be upserted to the target.

    Reproduce-first for the "loaded untransformed" divergence: before the
    engine skipped downstream transforms on a failed record, a raising
    transform still let traversal reach the ``load`` state, which upserted the
    *original, untransformed* record into the target. The record was counted as
    an ``error`` (not ``loaded``) — so a consumer trusting ``loaded`` believed
    the row was absent while the target actually held a stale copy.

    This test FAILS against that behavior (id ``2`` present in the target) and
    PASSES once the load step is skipped for the failed record (id ``2``
    absent; only the two clean, transformed rows persisted).
    """
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
            error_threshold=1.0,
            transformations=[_raise_on_id("2")],
        )
    )

    metrics = await etl.run()
    assert metrics["loaded"] == 2
    assert metrics["errors"] == 1

    target = await _read_target(tgt)
    assert "2" not in target, (
        "the failed record was upserted to the target instead of being "
        f"skipped: {target}"
    )
    assert set(target) == {"1", "3"}, f"only the clean rows should persist: {target}"
    # The persisted rows must carry the transform's output, not raw source data.
    assert target["1"]["tag"] == "ok"
    assert target["3"]["tag"] == "ok"
