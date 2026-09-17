# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""``BatchOperations`` reaches an async database through one loop per operation.

``BatchOperations`` fronts an ``AsyncDatabase | SyncDatabase`` and drove the
async half with ``asyncio.run`` at six sites across three methods. That has two
consequences, and only the first is what the class was reported for:

1. **Every public method raised for a caller already on a loop** ---
   ``RuntimeError: asyncio.run() cannot be called from a running event loop``
   --- which is the one situation a synchronous helper over an async store
   exists to serve. Seven entry points reach the database, so seven tests.
2. **Each ``asyncio.run`` built and closed its own loop**, so a single
   ``bulk_insert_dataframe`` ran its chunks --- and, in the per-record fallback
   path, its rows --- on a different loop each time. Against a backend holding
   loop-bound state that is not a cost, it is a break: an ``asyncpg`` pool
   acquired by ``connect()`` is bound to the loop that acquired it, and the
   next loop finds it unusable. Measured against ``AsyncPostgresDatabase``
   before the fix, the *first* operation after ``connect()`` raised
   ``InterfaceError: cannot perform operation: another operation is in
   progress``, from plain synchronous code with no running loop anywhere. So
   the class had never worked against a pooled backend at all, in either
   direction.

The loop-identity tests below are what pin (2) without a service: a real
``AsyncMemoryDatabase`` subclass that records which loop served each call.
``AsyncMemoryDatabase`` itself survives the churn --- its ``asyncio.Lock`` is
uncontended, and an uncontended lock never reaches ``_get_loop`` --- which is
exactly why a test suite that only ever exercised the memory backend reported
green while the pooled one was broken. The ``requires_postgres`` test at the
end is the same claim against the backend that actually binds.

The fix is an **operation-scoped** :class:`~dataknobs_common.sync_bridge.SyncLoopBridge`:
one loop for one public call, including the private helpers and the composite
methods that reach the database more than once. Operation-scoped rather than
held, because a bridge this object owned past the call would be a teardown
obligation on a class that has never had one --- and rather than per-call,
because ``_insert_chunk``'s fallback path calls the database once per row and
``run_coro_sync`` there would be one daemon thread per row.

That fixes the churn *within* an operation. It cannot fix it *across* the
database's life, because ``BatchOperations`` does not own the database and
therefore does not own the loop its ``connect()`` bound. ``bridge=`` is what
lets the owner say which loop that is; ``the_caller_supplied_bridge_*`` tests
are its contract.
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
from typing import Any

import pandas as pd
import pytest
from dataknobs_common import SyncLoopBridge
from dataknobs_common.testing import assert_no_leaked_bridge_threads, requires_postgres

from dataknobs_data import Record
from dataknobs_data.backends.memory import AsyncMemoryDatabase, SyncMemoryDatabase
from dataknobs_data.pandas.batch_ops import BatchConfig, BatchOperations
from dataknobs_data.query import Query


def frame(n: int = 3) -> pd.DataFrame:
    """A DataFrame with ``n`` rows, small enough to read in a failure message."""
    return pd.DataFrame({"name": [f"row-{i}" for i in range(n)], "n": list(range(n))})


class LoopWitness(AsyncMemoryDatabase):
    """A real async database that records which loop served each call.

    Not a stand-in for a database --- it *is* one, and every assertion about
    stored rows below goes through its real storage. The subclass adds one
    thing: the identity of the running loop at each entry point, which is the
    property under test and the one an ordinary database has no reason to
    expose.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loops: list[int] = []
        self.threads: set[str] = set()

    def _witness(self) -> None:
        self.loops.append(id(asyncio.get_running_loop()))
        self.threads.add(threading.current_thread().name)

    async def create(self, record: Record, **kwargs: Any) -> str:
        self._witness()
        return await super().create(record, **kwargs)

    async def create_batch(self, records: list[Record], **kwargs: Any) -> list[str]:
        self._witness()
        return await super().create_batch(records, **kwargs)

    async def search(self, query: Query, **kwargs: Any) -> list[Record]:
        self._witness()
        return await super().search(query, **kwargs)

    async def update(self, record_id: str, record: Record, **kwargs: Any) -> bool:
        self._witness()
        return await super().update(record_id, record, **kwargs)

    async def update_batch(self, updates: list[tuple[str, Record]], **kwargs: Any) -> list[bool]:
        self._witness()
        return await super().update_batch(updates, **kwargs)

    @property
    def distinct_loops(self) -> int:
        """How many different event loops served this database so far."""
        return len(set(self.loops))


# --------------------------------------------------------------------------
# 1. The reported defect: every public entry point raised on a running loop.
# --------------------------------------------------------------------------


#: Whether this environment can write Parquet at all. Resolved once, so a
#: test that sweeps every entry point can pass over the one method pandas
#: cannot serve here without skipping the other six with it.
HAVE_PARQUET_ENGINE = importlib.util.find_spec("pyarrow") is not None


def _export_to_parquet(ops: BatchOperations, path: str) -> None:
    """``export_to_parquet`` needs an engine pandas does not ship.

    Skipping here rather than dropping the entry point keeps the table below
    an honest census of what reaches the database: the method is one of the
    seven whether or not this environment can write the file.
    """
    pytest.importorskip("pyarrow")
    ops.export_to_parquet(Query(), path)


def _entry_points(ops: BatchOperations, tmp_path: Any) -> dict[str, Any]:
    """Every public method that reaches the database, as zero-argument calls.

    Named rather than parametrized over the class because the arguments
    differ, and a table of lambdas is the honest way to say "these seven, and
    no others, reach the database" --- ``aggregate``, ``transform_and_save``
    and the two exporters reach it only through ``query_as_dataframe``, which
    is why they are here at all: fixing the three methods that spell
    ``asyncio.run`` would have fixed them too, and a test that did not name
    them could not show it.
    """
    return {
        "bulk_insert_dataframe": lambda: ops.bulk_insert_dataframe(frame()),
        "query_as_dataframe": lambda: ops.query_as_dataframe(Query()),
        "update_from_dataframe": lambda: ops.update_from_dataframe(frame(), id_column=None),
        "aggregate": lambda: ops.aggregate(Query(), {"n": "sum"}),
        "transform_and_save": lambda: ops.transform_and_save(Query(), lambda df: df),
        "export_to_csv": lambda: ops.export_to_csv(Query(), str(tmp_path / "out.csv")),
        "export_to_parquet": lambda: _export_to_parquet(ops, str(tmp_path / "out.pq")),
    }


@pytest.mark.parametrize(
    "entry_point",
    [
        "bulk_insert_dataframe",
        "query_as_dataframe",
        "update_from_dataframe",
        "aggregate",
        "transform_and_save",
        "export_to_csv",
        "export_to_parquet",
    ],
)
def test_every_entry_point_works_from_inside_a_running_loop(entry_point, tmp_path):
    """The one case a synchronous helper over an async store exists to serve.

    Red before the fix with ``RuntimeError: asyncio.run() cannot be called
    from a running event loop``, for all seven.
    """

    async def main() -> None:
        db = AsyncMemoryDatabase()
        ops = BatchOperations(db)
        assert ops.is_async is True
        _entry_points(ops, tmp_path)[entry_point]()

    asyncio.run(main())


def test_a_sync_database_is_untouched_by_any_of_this(tmp_path):
    """The over-correction guard: no bridge, no thread, same answers.

    ``BatchOperations`` fronts either flavour, and the sync branch never had
    the defect. It must not acquire a loop to serve a database that does not
    need one.
    """
    db = SyncMemoryDatabase()
    ops = BatchOperations(db)
    assert ops.is_async is False

    with assert_no_leaked_bridge_threads():
        before = threading.active_count()
        for name, call in _entry_points(ops, tmp_path).items():
            if name == "export_to_parquet" and not HAVE_PARQUET_ENGINE:
                continue
            call()
        assert threading.active_count() == before

    assert len(db.search(Query())) > 0


# --------------------------------------------------------------------------
# 2. The measured consequence: one operation, one loop.
# --------------------------------------------------------------------------


def test_one_operation_drives_every_row_on_one_loop():
    """``_insert_chunk``'s per-record fallback ran each row on its own loop.

    Red before the fix: ``chunk_size=1`` over four rows put four chunks
    through ``_insert_chunk``, each one building and closing a loop of its
    own. Against a backend whose ``connect()`` bound a pool, the second loop
    is where it stops working.
    """
    db = LoopWitness()
    ops = BatchOperations(db)

    stats = ops.bulk_insert_dataframe(frame(4), config=BatchConfig(chunk_size=1))

    assert stats["inserted"] == 4
    assert db.loops, "the database was never reached"
    assert db.distinct_loops == 1, f"{db.distinct_loops} loops served one operation"


def test_a_composite_operation_drives_both_halves_on_one_loop():
    """``transform_and_save`` queries, then writes --- one operation, one loop.

    The composite methods are the reason the bridge is acquired by the public
    entry point and passed down rather than taken per database call: a scope
    per ``asyncio.run`` site would have given this method two loops, which is
    the defect in miniature.
    """
    db = LoopWitness()
    ops = BatchOperations(db)
    ops.bulk_insert_dataframe(frame(3))
    db.loops.clear()

    ops.transform_and_save(Query(), lambda df: df)

    assert len(db.loops) > 1, "the composite reached the database only once"
    assert db.distinct_loops == 1, f"{db.distinct_loops} loops served one operation"


def test_an_operations_loop_is_not_the_callers_thread():
    """The loop is on the bridge's daemon thread, which is the whole mechanism."""
    db = LoopWitness()
    ops = BatchOperations(db)

    ops.bulk_insert_dataframe(frame(2))

    assert db.threads and threading.current_thread().name not in db.threads


def test_an_operation_leaves_no_thread_behind():
    """Operation-scoped means the thread ends with the call.

    This is what buys ``BatchOperations`` its freedom from a ``close()``: the
    class has never had a teardown and does not acquire one here.
    """
    db = LoopWitness()
    ops = BatchOperations(db)

    with assert_no_leaked_bridge_threads():
        ops.bulk_insert_dataframe(frame(2))
        ops.query_as_dataframe(Query())

    assert not hasattr(ops, "close"), "BatchOperations must not grow a teardown obligation"


# --------------------------------------------------------------------------
# 3. ``bridge=`` --- the owner of the database says which loop it is bound to.
# --------------------------------------------------------------------------


def test_the_caller_supplied_bridge_serves_every_operation():
    """One loop across operations, which only the database's owner can supply.

    ``BatchOperations`` does not own the database, so it cannot own the loop
    that ``connect()`` bound. A caller who connects on their own bridge and
    passes it here gets every operation on that loop.
    """
    db = LoopWitness()
    with SyncLoopBridge() as bridge:
        bridge.run(db.connect())
        ops = BatchOperations(db, bridge=bridge)

        ops.bulk_insert_dataframe(frame(2))
        ops.query_as_dataframe(Query())
        ops.transform_and_save(Query(), lambda df: df)

        assert db.distinct_loops == 1


def test_the_caller_supplied_bridge_outlives_the_operations():
    """A bridge handed in belongs to the caller; nothing here closes it."""
    db = LoopWitness()
    bridge = SyncLoopBridge()
    try:
        ops = BatchOperations(db, bridge=bridge)
        ops.bulk_insert_dataframe(frame(2))
        assert not bridge.is_closed
        ops.query_as_dataframe(Query())
        assert not bridge.is_closed
    finally:
        bridge.close()


def test_a_timeout_bounds_a_wait_the_caller_cannot_otherwise_cancel():
    """The only upper bound a blocked synchronous caller has."""

    class Slow(AsyncMemoryDatabase):
        async def search(self, query: Query, **kwargs: Any) -> list[Record]:
            await asyncio.sleep(5)
            return await super().search(query, **kwargs)

    ops = BatchOperations(Slow(), timeout=0.05)

    with pytest.raises(TimeoutError):
        ops.query_as_dataframe(Query())


# --------------------------------------------------------------------------
# 4. The same claim against a backend that really does bind a loop.
# --------------------------------------------------------------------------


@requires_postgres
def test_a_pooled_backend_is_usable_when_the_owner_supplies_the_loop(
    postgres_connection_params,
):
    """The measurement that set the design, against the real thing.

    The pre-fix equivalent --- ``asyncio.run(db.connect())`` and then these
    same operations, each on a loop of its own --- raised
    ``asyncpg.InterfaceError: cannot perform operation: another operation is
    in progress`` on the *first* operation, from synchronous code with no
    running loop anywhere. It cannot be written as a red test here because
    ``bridge=`` is what this change adds; what it can do is pin the shape that
    works, so a later change that reintroduces per-operation loops for a
    supplied bridge fails.

    It is the only test in this file that needs a service, and the memory
    backend cannot stand in for it: an uncontended ``asyncio.Lock`` binds to
    no loop, so the defect is invisible there.
    """
    from dataknobs_data.backends.postgres import AsyncPostgresDatabase

    db = AsyncPostgresDatabase(**postgres_connection_params, table="batch_ops_bridge_probe")
    with SyncLoopBridge() as bridge:
        bridge.run(db.connect())
        try:
            # The table outlives the run, so start from empty rather than
            # asserting a lower bound that would pass on last run's rows.
            bridge.run(db.clear())
            ops = BatchOperations(db, bridge=bridge)

            # chunk_size=1 is the shape that matters: four chunks, so four
            # separate reaches into the pool where `asyncio.run` gave each
            # one a loop of its own.
            stats = ops.bulk_insert_dataframe(frame(4), config=BatchConfig(chunk_size=1))
            assert stats["inserted"] == 4

            assert len(ops.query_as_dataframe(Query())) == 4
        finally:
            bridge.run(db.clear())
            bridge.run(db.close())
