"""Reproduce-first async-correctness tests for ``AsyncFileDatabase``.

``AsyncFileDatabase``'s CRUD surface is ``async def``, but every method
routes through ``_load_data`` / ``_save_data`` whose bodies perform
synchronous file I/O (``open``/read/write, ``tempfile.mkstemp``,
``os.replace``) and acquire a blocking inter-process ``FileLock``
(``fcntl.lockf`` on POSIX). Run on the event loop, that stalls every
other task for the duration of the disk operation.

Each test wraps a single awaited CRUD call in
:func:`assert_no_blocking`. Against the pre-offload code these FAIL with
``blockbuster.BlockingError`` (the disk syscall runs on the loop); after
the ``asyncio.to_thread`` offload they PASS. The final concurrency test
guards the real regression risk — that offloading must not break the
``asyncio.Lock`` serialisation that prevents lost writes.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import pytest
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_data.backends.file import AsyncFileDatabase
from dataknobs_data.query import Query
from dataknobs_data.records import Record

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


def _record(name: str, value: int) -> Record:
    return Record({"name": name, "value": value})


@pytest.fixture
async def db(tmp_path: Path):
    """A tmp-file-backed AsyncFileDatabase, closed on teardown."""
    database = AsyncFileDatabase({"path": str(tmp_path / "records.json")})
    yield database
    await database.close()


@requires_blockbuster
async def test_create_does_not_block(db: AsyncFileDatabase) -> None:
    with assert_no_blocking():
        await db.create(_record("alice", 1))


@requires_blockbuster
async def test_read_does_not_block(db: AsyncFileDatabase) -> None:
    rec_id = await db.create(_record("alice", 1))
    with assert_no_blocking():
        result = await db.read(rec_id)
    assert result is not None


@requires_blockbuster
async def test_update_does_not_block(db: AsyncFileDatabase) -> None:
    rec_id = await db.create(_record("alice", 1))
    with assert_no_blocking():
        await db.update(rec_id, _record("alice", 2))


@requires_blockbuster
async def test_delete_does_not_block(db: AsyncFileDatabase) -> None:
    rec_id = await db.create(_record("alice", 1))
    with assert_no_blocking():
        await db.delete(rec_id)


@requires_blockbuster
async def test_search_does_not_block(db: AsyncFileDatabase) -> None:
    await db.create(_record("alice", 1))
    with assert_no_blocking():
        await db.search(Query())


@requires_blockbuster
async def test_create_batch_does_not_block(db: AsyncFileDatabase) -> None:
    with assert_no_blocking():
        await db.create_batch([_record("a", 1), _record("b", 2)])


@requires_blockbuster
async def test_close_temp_file_cleanup_does_not_block() -> None:
    """Temp-file cleanup on close() must not block the loop.

    A temp-file-backed AsyncFileDatabase removes its data + ``.lock`` files
    on close(). Those existence stats + unlinks are blocking disk I/O;
    pre-offload they ran on the loop (``os.path.exists`` trips blockbuster),
    post-offload they run via ``to_thread``. Afterwards both files are gone.
    """
    database = AsyncFileDatabase({})  # no path → temp-file backed
    await database.create(_record("alice", 1))  # materialise the data file
    filepath = database.filepath
    assert os.path.exists(filepath)

    with assert_no_blocking():
        await database.close()

    assert not os.path.exists(filepath)
    assert not os.path.exists(filepath + ".lock")


async def test_concurrent_creates_do_not_lose_writes(db: AsyncFileDatabase) -> None:
    """``asyncio.gather`` of N creates must persist all N records.

    The offload moves the read-modify-write critical section onto a
    worker thread, but the surrounding ``async with self._lock``
    (``asyncio.Lock``) still serialises the async API. This proves the
    lock + offload interplay does not drop concurrent writes.
    """
    n = 25
    await asyncio.gather(*(db.create(_record(f"r{i}", i)) for i in range(n)))
    assert await db._count_all() == n
