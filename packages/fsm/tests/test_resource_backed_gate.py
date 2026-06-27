"""Canonical sample: a resource-backed ``validate`` arc gate.

The pure-data validation forms (dict schema / library validator / callable)
read only the record. The moment a gate must validate against a *reference
table* — "reject any row whose country is not in the ``valid_countries``
table" — the arc condition needs a resource. This is the canonical
resource-bearing-gate pattern, riding the arc-condition resource injection the
engine provides: an arc declares its resource under ``resources``, and the
engine injects it into the condition's :class:`FunctionContext` so the predicate
resolves it via ``ctx.resource_for_role(...)`` / ``ctx.require_resource(...)``.

This test is the executable worked example referenced from the ETL/file-
processing docs. Real constructs only: a file-backed ``AsyncDatabase`` reference
table, a real ``AsyncSimpleFSM`` build — no mocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_data import AsyncDatabase, Query, Record
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
from dataknobs_fsm.functions.base import FunctionContext


async def _seed_reference(path: str, valid_codes: list[str]) -> None:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": path})
    try:
        for code in valid_codes:
            await db.upsert(code, Record({"id": code}))
    finally:
        await db.close()


# The resource-backed gate predicate. It is an async arc condition: the engine
# injects the arc's declared resource into ``ctx`` (keyed by name, with the
# ``{role: name}`` map exposed for role lookup), so the predicate resolves the
# reference table and checks membership. A raising ``require_resource`` (the
# reference resource missing or down) is treated by the engine as a genuine
# evaluation failure: it surfaces as a record *error*, not a silent reject —
# an infrastructure outage must not masquerade as a clean data-quality drop.
async def country_in_reference(data: dict[str, Any], ctx: FunctionContext) -> bool:
    reference = ctx.resource_for_role("reference")
    rows = await reference.execute_query(Query())
    valid_codes = {row.get("id") for row in rows}
    return data.get("country") in valid_codes


def _build_gate_fsm(ref_path: str) -> AsyncSimpleFSM:
    config = {
        "name": "reference_gate",
        "states": [
            {"name": "read", "is_start": True},
            {"name": "validate"},
            {"name": "complete", "is_end": True},
            {"name": "rejected", "is_end": True, "emit_output": False},
        ],
        "arcs": [
            {"from": "read", "to": "validate", "name": "start"},
            # The gate arc declares the reference resource by role; the engine
            # acquires it and injects it into the condition's context.
            {
                "from": "validate",
                "to": "complete",
                "name": "valid",
                "condition": {"type": "registered", "name": "country_in_reference"},
                "resources": {"reference": "valid_countries"},
                "priority": 10,
            },
            {"from": "validate", "to": "rejected", "name": "invalid"},
        ],
        "resources": [
            {
                "name": "valid_countries",
                "type": "async_database",
                "config": {"type": "file", "path": ref_path},
            }
        ],
    }
    return AsyncSimpleFSM(
        config,
        custom_functions={"country_in_reference": country_in_reference},
    )


@pytest.mark.asyncio
async def test_resource_backed_gate_routes_on_reference_lookup(
    tmp_path: Path,
) -> None:
    """Records are gated by a lookup against an injected reference resource."""
    ref = str(tmp_path / "valid_countries.json")
    await _seed_reference(ref, ["US", "CA", "GB"])

    fsm = _build_gate_fsm(ref)
    try:
        results = {
            row["id"]: await fsm.process(row)
            for row in [
                {"id": "1", "country": "US"},   # in reference -> complete
                {"id": "2", "country": "ZZ"},   # not in reference -> rejected
                {"id": "3", "country": "CA"},   # in reference -> complete
            ]
        }
    finally:
        await fsm.close()

    assert all(r["success"] for r in results.values()), results
    assert results["1"]["final_state"] == "complete", results["1"]
    assert results["2"]["final_state"] == "rejected", results["2"]
    assert results["3"]["final_state"] == "complete", results["3"]
