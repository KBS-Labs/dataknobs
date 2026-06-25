"""Reproduce-first tests: the per-record transform stage must actually run.

``DatabaseETL`` accepts ``field_mappings`` and a list of ``transformations``
callables, but historically neither reached the per-record FSM: only the
``load`` step was wired through ``custom_functions=``, while the ``transform``
arc carried an inline-code reference that set ``transformed=True`` and ignored
the user's callables entirely (the ``_create_transformer`` composer that would
have applied them was dead code). A user passing ``transformations=[...]`` saw
their logic silently dropped — a data-correctness bug.

These tests reopen the target store and assert the transformed shape is
persisted. They FAIL against the unwired transform stage and PASS once the
field-mapping + user transformations run as a registered per-record transform.

Real constructs only: file-backed ``AsyncDatabase`` source/target.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.asyncio
async def test_etl_applies_user_transformations(tmp_path: Path) -> None:
    """A user ``transformations`` callable must be applied before load."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}])

    etl = DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": src},
            target_db={"type": "file", "path": tgt},
            source_query=None,
            target_table="records",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
            transformations=[lambda r: {**r, "tag": "X"}],
        )
    )
    await etl.run()

    rows = await _read_target(tgt)
    assert rows, "no rows persisted"
    assert all(r.get("tag") == "X" for r in rows), (
        f"user transformation was not applied before load — persisted rows: {rows}"
    )


@pytest.mark.asyncio
async def test_etl_applies_field_mappings(tmp_path: Path) -> None:
    """``field_mappings`` must rename keys before load."""
    src = str(tmp_path / "source.json")
    tgt = str(tmp_path / "target.json")
    await _seed_source(src, [{"id": "1", "name": "Alice"}])

    etl = DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": src},
            target_db={"type": "file", "path": tgt},
            source_query=None,
            target_table="records",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
            field_mappings={"name": "full_name"},
        )
    )
    await etl.run()

    rows = await _read_target(tgt)
    assert rows, "no rows persisted"
    row = rows[0]
    assert row.get("full_name") == "Alice", f"field_mappings not applied: {row}"
    assert "name" not in row, f"old key survived the rename: {row}"
