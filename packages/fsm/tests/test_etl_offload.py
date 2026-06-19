"""Reproduce-first tests for ``DatabaseETL.run()`` — repair + sync-bridge offload.

``DatabaseETL.run()`` had never been executed end-to-end: it called the
abstract ``AsyncDatabase.create`` (a no-op ``NotImplementedError``) instead
of the ``AsyncDatabase.from_backend`` factory, so it died before reaching
its FSM step. Once that is repaired, the FSM step exhibited the same
sync-bridge defect as ``FileProcessor``: it built a synchronous
``SimpleFSM`` and called ``process_batch`` from inside the async ``run()``,
blocking the loop on ``threading.Lock.acquire``. The fix builds an
``AsyncSimpleFSM`` and ``await``s it.

The blocking test wraps ``await etl.run()`` in :func:`assert_no_blocking`
(GREEN only after the bridge swap; against the sync-bridge it FAILS with
``blockbuster.BlockingError``). The functional tests assert what the
repair actually establishes: ``run()`` now executes end-to-end, extracts
every source row, and completes without error — coverage that did not
exist before, since ``run()`` could not previously execute at all.

Scope note: the ETL FSM's fetch/transform/load *logic* (the
``DatabaseFetch``/``DatabaseUpsert`` functions) is not wired into the
pattern's FSM and does not execute — records traverse skeleton states to
``complete`` without anything being written to the target. That is a
separate, deeper pre-existing defect (the ETL pattern's core was never
implemented/wired), independent of this blocking/factory repair, so these
tests deliberately do NOT assert target-store persistence (the ``loaded``
metric is hollow today).

The source uses the on-disk ``file`` async backend so the rows seeded
before the run are the same data ``run()`` reopens by path (the ``memory``
backend would hand ``run()`` a fresh, empty store).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_data import AsyncDatabase, Record
from dataknobs_fsm.patterns.etl import DatabaseETL, ETLConfig, ETLMode

if TYPE_CHECKING:
    from pathlib import Path


async def _seed_source(path: Path, n: int) -> None:
    db = await AsyncDatabase.from_backend("file", {"type": "file", "path": str(path)})
    for i in range(n):
        await db.create(Record({"id": str(i), "v": i * 10}))
    await db.close()


def _etl(src: Path, tgt: Path) -> DatabaseETL:
    return DatabaseETL(
        ETLConfig(
            source_db={"type": "file", "path": str(src)},
            target_db={"type": "file", "path": str(tgt)},
            source_query=None,  # -> Query() (all rows); the default is a SQL string
            target_table="t",
            key_columns=["id"],
            mode=ETLMode.FULL_REFRESH,
            batch_size=2,
        )
    )


# --------------------------------------------------------------------------
# Reproduce-first: run() must not block the event loop on the FSM step.
# --------------------------------------------------------------------------


@requires_blockbuster
async def test_etl_run_does_not_block(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    tgt = tmp_path / "tgt.json"
    await _seed_source(src, 3)
    etl = _etl(src, tgt)
    with assert_no_blocking():
        metrics = await etl.run()
    assert metrics["extracted"] == 3


# --------------------------------------------------------------------------
# Functional: the repaired run() executes end-to-end and extracts the rows.
# (Target-store persistence is intentionally not asserted — see module
# docstring: the ETL FSM's load logic is a separate, unwired pre-existing
# defect.)
# --------------------------------------------------------------------------


async def test_etl_run_extracts_all_records(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    tgt = tmp_path / "tgt.json"
    await _seed_source(src, 3)
    etl = _etl(src, tgt)

    metrics = await etl.run()

    assert metrics["extracted"] == 3
    assert metrics["errors"] == 0


async def test_etl_run_empty_source(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    tgt = tmp_path / "tgt.json"
    await _seed_source(src, 0)  # create + close an empty store
    etl = _etl(src, tgt)

    metrics = await etl.run()

    assert metrics["extracted"] == 0
    assert metrics["loaded"] == 0
    assert metrics["errors"] == 0


@requires_blockbuster
async def test_etl_run_batches_do_not_block(tmp_path: Path) -> None:
    """Many records across multiple batches still never block the loop."""
    src = tmp_path / "src.json"
    tgt = tmp_path / "tgt.json"
    await _seed_source(src, 7)  # batch_size=2 -> 4 batches
    etl = _etl(src, tgt)
    with assert_no_blocking():
        metrics = await etl.run()
    assert metrics["extracted"] == 7
