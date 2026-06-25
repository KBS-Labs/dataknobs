"""Reproduce-first tests for W2 — engine resource-injection into FSM functions.

A resource-bearing transform function (e.g. ``DatabaseUpsert``) wired into a
state via the proven ``custom_functions=`` + state ``functions`` block idiom
must receive its declared resource through ``FunctionContext.resources`` and
actually perform its side effect (persist the row). Today the async engine
builds ``FunctionContext(..., resources={})`` — always empty — and never
acquires ``state.resource_requirements``, so the function cannot reach its
resource and silently no-ops (the row is never written).

These tests assert the *effect* (the row lands in the target resource) and
that the write does not block the event loop. They use real constructs (a
real ``AsyncDatabase`` file backend, a real FSM build — no mocks) and read
the target back through a fresh ``AsyncDatabase.from_backend`` to prove
persistence independently of the writing adapter instance.

They FAIL against the un-wired engine (resources never injected → row never
written) and PASS once W2 acquires the state resources into
``FunctionContext.resources`` and the database functions read from
``context.resources``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_common.testing import assert_no_blocking
from dataknobs_data import AsyncDatabase
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode
from dataknobs_fsm.functions.library.database import DatabaseUpsert


def _upsert_fsm(target_cfg: dict[str, Any]) -> AsyncSimpleFSM:
    """Build a minimal FSM whose start state upserts the record to a resource.

    The ``load`` state declares the ``target_db`` resource and references the
    ``load`` transform (a ``DatabaseUpsert``) through the supported
    ``custom_functions=`` + state ``functions`` idiom.
    """
    config = {
        "name": "upsert_only",
        "data_mode": DataHandlingMode.COPY.value,
        "resources": [
            {"name": "target_db", "type": "async_database", "config": target_cfg},
        ],
        "states": [
            {
                "name": "load",
                "is_start": True,
                "resources": ["target_db"],
                "functions": {"transform": {"type": "registered", "name": "load"}},
            },
            {"name": "done", "is_end": True},
        ],
        "arcs": [{"from": "load", "to": "done", "name": "loaded"}],
    }
    return AsyncSimpleFSM(
        config,
        data_mode=DataHandlingMode.COPY,
        custom_functions={
            "load": DatabaseUpsert(
                resource_name="target_db",
                table="rows",
                key_columns=["id"],
            )
        },
    )


@pytest.mark.asyncio
async def test_database_upsert_state_transform_persists_row(tmp_path: Path) -> None:
    """B3/B4: a DatabaseUpsert state transform must persist via its resource."""
    target = {"type": "file", "path": str(tmp_path / "target.json")}

    fsm = _upsert_fsm(target)
    try:
        result = await fsm.process({"id": "1", "name": "Alice"})
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()

    # Reopen the target independently and assert the row persisted.
    db = await AsyncDatabase.from_backend("file", target)
    try:
        record = await db.read("1")
    finally:
        await db.close()

    assert record is not None, (
        "DatabaseUpsert did not persist the row — its declared 'target_db' "
        "resource was never injected into FunctionContext.resources, so the "
        "transform could not reach the database"
    )
    assert record.to_dict().get("name") == "Alice"


@pytest.mark.asyncio
async def test_database_upsert_state_transform_does_not_block(tmp_path: Path) -> None:
    """The resource-bearing transform must not block the event loop."""
    target = {"type": "file", "path": str(tmp_path / "target.json")}

    fsm = _upsert_fsm(target)
    try:
        with assert_no_blocking():
            result = await fsm.process({"id": "1", "name": "Alice"})
        assert result["success"], f"FSM did not complete cleanly: {result}"
    finally:
        await fsm.close()
