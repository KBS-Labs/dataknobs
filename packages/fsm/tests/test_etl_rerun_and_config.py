"""Reproduce-first tests for ETL re-runnability and field-mapping validation.

#3 — A single ``DatabaseETL`` instance must be re-runnable. ``run()``'s finally
closes the FSM, which clears its resource providers; reusing the same closed
FSM left the load step with no ``target_db`` to upsert into, so a second
``run()`` reported ``loaded`` rows that were never persisted. ``run()`` now
rebuilds the FSM and resets metrics each call.

#4 — ``field_mappings`` that rename a key column out from under the load step
collapse every row onto a single ``"None"`` id (silently overwriting the whole
target with one record). That destructive config is now rejected at
construction.

Real constructs only: file-backed ``AsyncDatabase`` source/target.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dataknobs_data import AsyncDatabase, Query, Record
from dataknobs_fsm.core.exceptions import InvalidConfigurationError
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


def _make_etl(src: str, tgt: str, **overrides) -> DatabaseETL:
    return DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": src},
            target_db={"type": "file", "path": tgt},
            source_query=None,
            target_table="records",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
            **overrides,
        )
    )


@pytest.mark.asyncio
async def test_etl_is_rerunnable(tmp_path: Path) -> None:
    """A second run() on the same instance persists and reports honestly."""
    src1 = str(tmp_path / "src1.json")
    src2 = str(tmp_path / "src2.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src1, [{"id": "1", "name": "A"}])
    await _seed_source(src2, [{"id": "2", "name": "B"}, {"id": "3", "name": "C"}])

    etl = _make_etl(src1, tgt, transformations=[lambda r: {**r, "tag": "X"}])

    metrics1 = await etl.run()
    assert metrics1["loaded"] == 1

    # Repoint the source and run again on the SAME instance.
    etl.config.source_db["path"] = src2
    metrics2 = await etl.run()

    # Metrics reflect only the second run (not accumulated).
    assert metrics2["loaded"] == 2, f"second run did not load freshly: {metrics2}"
    assert metrics2["errors"] == 0

    rows = await _read_target(tgt)
    persisted_ids = {r.get("id") for r in rows}
    assert {"2", "3"} <= persisted_ids, (
        f"second run's rows were not actually persisted: {rows}"
    )
    assert all(r.get("tag") == "X" for r in rows if r.get("id") in {"2", "3"})


def test_field_mapping_renaming_key_column_is_rejected(tmp_path: Path) -> None:
    """Renaming a key column via field_mappings is a destructive config."""
    with pytest.raises(InvalidConfigurationError):
        ETLConfig(
            source_db={"type": "file", "path": str(tmp_path / "s.json")},
            target_db={"type": "file", "path": str(tmp_path / "t.json")},
            key_columns=["id"],
            field_mappings={"id": "order_id"},
        )


def test_field_mapping_on_non_key_field_is_allowed(tmp_path: Path) -> None:
    """Renaming a non-key field is fine."""
    cfg = ETLConfig(
        source_db={"type": "file", "path": str(tmp_path / "s.json")},
        target_db={"type": "file", "path": str(tmp_path / "t.json")},
        key_columns=["id"],
        field_mappings={"name": "full_name"},
    )
    assert cfg.field_mappings == {"name": "full_name"}
