"""Reproduce-first persistence test for the ETL pattern (W2 + W3).

The headline acceptance criterion: ``DatabaseETL.run()`` must actually upsert
the extracted/transformed rows into the target store. Historically the FSM
traversed skeleton states without invoking any load function, so the target
was never written (the ``loaded`` metric was hollow). This test reopens the
target store independently and asserts the rows are present — it fails against
the un-wired pipeline and passes once the DatabaseUpsert load step runs with
its injected async ``target_db`` resource.

Uses real constructs: file-backed ``AsyncDatabase`` source/target (reopenable)
and a real ``DatabaseETL`` build — no mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_common.testing import assert_no_blocking
from dataknobs_data import AsyncDatabase, Query, Record
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode


async def _seed_source(path: str, rows: list[dict]) -> None:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        for row in rows:
            await db.upsert(row["id"], Record(dict(row)))
    finally:
        await db.close()


async def _read_target(path: str) -> list[dict]:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        return [record.to_dict() async for record in db.stream_read(Query())]
    finally:
        await db.close()


def _etl(src: str, tgt: str) -> DatabaseETL:
    return DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": src},
            target_db={"type": "file", "path": tgt},
            source_query=None,
            target_table="records",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
        )
    )


@pytest.mark.asyncio
async def test_etl_run_persists_rows_to_target(tmp_path: Path) -> None:
    """AC#1: run() upserts every extracted row into the target store."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    seed = [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}, {"id": "3", "name": "Cara"}]
    await _seed_source(src, seed)

    metrics = await _etl(src, tgt).run()

    rows = await _read_target(tgt)
    assert len(rows) == 3, (
        f"expected 3 rows upserted to the target, found {len(rows)} — "
        f"the ETL load step did not persist (metrics={metrics})"
    )
    assert {r.get("name") for r in rows} == {"Alice", "Bob", "Cara"}

    # Metrics must be truthful: every loaded row was transformed + loaded
    # (no longer the hollow {'loaded': N, 'transformed': 0}).
    assert metrics["loaded"] == 3
    assert metrics["transformed"] == 3
    assert metrics["errors"] == 0


@pytest.mark.asyncio
async def test_etl_run_persists_without_blocking(tmp_path: Path) -> None:
    """The end-to-end run must not block the event loop."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, [{"id": "1", "name": "Alice"}])

    with assert_no_blocking():
        await _etl(src, tgt).run()

    rows = await _read_target(tgt)
    assert len(rows) == 1
