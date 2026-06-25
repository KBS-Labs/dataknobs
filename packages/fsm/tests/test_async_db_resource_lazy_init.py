"""Reproduce-first test: concurrent first-use must open exactly one backend.

``AsyncDatabaseResourceAdapter`` lazily opens its backend on first use. The
single adapter instance is shared across all concurrent records of a batch
(``acquire()`` returns ``self``), and the async batch executor launches records
concurrently via ``asyncio.gather``. With an unguarded check-then-await-then-set
lazy init, every racer observes ``self._database is None`` (the None-check is
synchronous and there is an ``await`` before the assignment, so under
``gather`` all racers pass the check before any assigns), each opens its own
``AsyncDatabase``, and all but the last are orphaned and never closed — a
connection/pool leak for pooled backends.

This test counts real backend opens under concurrent first-use. It FAILS
against the unguarded init (N opens for N racers) and PASSES once the lazy
open is serialized with a double-checked lock (exactly one open).

Real constructs only: a real file-backed ``AsyncDatabase``; the spy delegates
to the real ``from_backend`` and only counts invocations.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dataknobs_data import AsyncDatabase
from dataknobs_fsm.resources.database import AsyncDatabaseResourceAdapter


@pytest.mark.asyncio
async def test_concurrent_first_use_opens_one_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opens = 0
    real_from_backend = AsyncDatabase.from_backend

    async def counting_from_backend(backend, config):  # type: ignore[no-untyped-def]
        nonlocal opens
        # Yield control so the race window opens — models a pooled/network
        # backend whose open actually suspends (file/memory opens complete
        # without yielding, which would hide the race on those backends).
        await asyncio.sleep(0)
        opens += 1
        return await real_from_backend(backend, config)

    monkeypatch.setattr(AsyncDatabase, "from_backend", counting_from_backend)

    adapter = AsyncDatabaseResourceAdapter(
        "target_db", type="file", path=str(tmp_path / "t.json")
    )
    try:
        dbs = await asyncio.gather(*(adapter._ensure_db() for _ in range(8)))
    finally:
        await adapter.aclose()

    assert opens == 1, f"concurrent first-use opened {opens} backends, expected 1"
    # Every racer must get the same shared instance.
    assert all(db is dbs[0] for db in dbs)
