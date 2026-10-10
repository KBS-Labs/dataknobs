"""Async SQLite's ``close`` refuses what has not started and waits for what has.

``close`` refuses every operation from the moment it starts, as async DuckDB's
does, and closes the connection once the operations already running on it are
done. An operation is several statements, each its own await, so ``close``
can start between two of them; it then waits rather than closing the
connection under the rest.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase
from dataknobs_data.exceptions import DuplicateRecordError
from dataknobs_data.records import Record

pytestmark = pytest.mark.asyncio


class HeldPartway(AsyncSQLiteDatabase):
    """Stops an operation at one of its awaits until the test lets it go on.

    ``hold`` names the method to stop in; it stops there once, after the
    method has done its work, as an await partway through the operation.
    """

    hold: str = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.started, self.release = asyncio.Event(), asyncio.Event()

    async def _held(self) -> None:
        self.hold = ""
        self.started.set()
        await self.release.wait()

    async def _existing_ids(self, *args: Any) -> set[str]:
        found = await super()._existing_ids(*args)
        if self.hold == "_existing_ids":
            await self._held()
        return found

    async def read(self, id: str) -> Record | None:
        found = await super().read(id)
        if self.hold == "read":
            await self._held()
        return found


async def _stored(path: Path) -> dict[str, Any]:
    db = AsyncSQLiteDatabase({"path": str(path)})
    await db.connect()
    try:
        return {str(r.storage_id): r["k"] for r in await db.all()}
    finally:
        await db.close()


async def _seeded(path: Path, cls: type[AsyncSQLiteDatabase] = AsyncSQLiteDatabase) -> Any:
    db = cls({"path": str(path)})
    await db.connect()
    await db.create(Record({"k": 1}, storage_id="a"))
    return db


async def _in_a_transaction(db: AsyncSQLiteDatabase) -> list[str]:
    async with db._transaction() as tx:
        return await db.create_batch([Record({"k": 2}, storage_id="b")], _tx=tx)


#: An operation that stops partway, where it stops, and what is stored after it.
HELD: dict[str, tuple[str, Any, dict[str, Any]]] = {
    "delete_batch": ("_existing_ids", lambda db, v: db.delete_batch(["a"]), {}),
    "create_batch": (
        "_existing_ids",
        lambda db, v: db.create_batch([Record({"k": 1}, storage_id="a")]),
        {"a": 1},
    ),
    "update-expected": (
        "read",
        lambda db, v: db.update("a", Record({"k": 2}), expected_version=v),
        {"a": 2},
    ),
    "delete-expected": ("read", lambda db, v: db.delete("a", expected_version=v), {}),
}


@pytest.mark.parametrize("operation", list(HELD))
async def test_close_waits_for_an_operation_it_starts_partway_through(
    tmp_path: Path, operation: str
) -> None:
    """Bug: ``close`` closed the connection while an operation was suspended
    between two of its statements, and refused nothing until it had: the
    operation then failed with the driver's ``ProgrammingError: Cannot operate
    on a closed database``, or with ``AttributeError`` on the cleared
    connection, and an operation started meanwhile met the closing one.
    """
    path = tmp_path / "records.db"
    db = await _seeded(path, HeldPartway)
    version = await db.get_version("a")
    hold, run, stored = HELD[operation]
    db.hold = hold
    running = asyncio.ensure_future(run(db, version))
    await asyncio.wait_for(db.started.wait(), timeout=10)
    closing = asyncio.create_task(db.close())
    for _ in range(5):
        await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="not connected"):
        await db.count()
    assert not closing.done(), "close waits for the operation already running"
    db.release.set()
    if operation == "create_batch":
        # Its id is stored: refused as a duplicate, after rolling back.
        with pytest.raises(DuplicateRecordError):
            await running
    else:
        await running
    await closing
    assert await _stored(path) == stored


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


async def test_a_transaction_closed_in_its_body_is_refused_by_name_once(tmp_path: Path) -> None:
    """Bug: the commit read the cleared connection and raised
    ``AttributeError``, and so did the rollback that handled it. Closing
    discards the open transaction, so the rollback now has nothing to do, and
    the commit's refusal is the one error.
    """
    db = await _seeded(tmp_path / "records.db")
    with pytest.raises(RuntimeError, match="not connected") as raised:
        async with db._transaction():
            await db.close()
    assert [type(e) for e in _chain(raised.value)] == [RuntimeError]


async def test_a_batch_in_a_transaction_close_starts_during(tmp_path: Path) -> None:
    """The batch is one operation and finishes; the transaction is not one,
    and its commit, after the close, is refused by name. Nothing is written.
    """
    path = tmp_path / "records.db"
    db = await _seeded(path, HeldPartway)
    db.hold = "_existing_ids"
    running = asyncio.ensure_future(_in_a_transaction(db))
    await asyncio.wait_for(db.started.wait(), timeout=10)
    closing = asyncio.create_task(db.close())
    await asyncio.sleep(0)
    db.release.set()
    with pytest.raises(RuntimeError, match="not connected") as raised:
        await running
    assert [type(e) for e in _chain(raised.value)] == [RuntimeError]
    await closing
    assert await _stored(path) == {"a": 1}


#: An operation, and what it leaves stored when it runs.
RACED: dict[str, tuple[Any, dict[str, Any]]] = {
    "create": (lambda db: db.create(Record({"k": 2}, storage_id="b")), {"a": 1, "b": 2}),
    "update_batch": (lambda db: db.update_batch([("a", Record({"k": 2}))]), {"a": 2}),
}


@pytest.mark.parametrize("yields", range(6))
@pytest.mark.parametrize("operation", list(RACED))
async def test_an_operation_racing_close_runs_whole_or_is_refused_by_name(
    tmp_path: Path, operation: str, yields: int
) -> None:
    """Bug: ``close`` awaited the driver's close before it refused anything, so
    an operation racing it met the driver's own errors -- ``ProgrammingError:
    Cannot operate on a closed database`` or ``ValueError: no active
    connection`` -- partway through.
    """
    path = tmp_path / "records.db"
    db = await _seeded(path)
    run, stored = RACED[operation]

    async def close_after() -> None:
        for _ in range(yields):
            await asyncio.sleep(0)
        await db.close()

    answer, closed = await asyncio.gather(run(db), close_after(), return_exceptions=True)
    assert closed is None
    if isinstance(answer, BaseException):
        assert isinstance(answer, RuntimeError) and "not connected" in str(answer), answer
        assert await _stored(path) == {"a": 1}
    else:
        assert await _stored(path) == stored
