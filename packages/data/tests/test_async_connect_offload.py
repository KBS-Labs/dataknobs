"""Reproduce-first async-correctness tests for file-backed ``connect()``.

``AsyncSQLiteDatabase.connect`` and ``AsyncDuckDBDatabase.connect`` create
the parent directory of a file-based database with
``db_file.parent.mkdir(...)``. Because the call is on an attribute-bound
Path (not a ``Path(...)`` literal), ruff's ``ASYNC240`` cannot see it — so
the blocking ``mkdir`` ran directly on the event loop, stalling every other
task while the directory was created.

Each test points the backend at a not-yet-existing nested directory and
wraps the awaited ``connect()`` in :func:`assert_no_blocking`: against the
pre-fix code these FAIL with ``blockbuster.BlockingError`` (``os.mkdir`` on
the loop); after the ``asyncio.to_thread`` offload they PASS.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


@requires_blockbuster
async def test_sqlite_async_connect_does_not_block(tmp_path: Path) -> None:
    db = AsyncSQLiteDatabase({"path": str(tmp_path / "nested" / "dir" / "records.sqlite")})
    try:
        with assert_no_blocking():
            await db.connect()
    finally:
        await db.close()
    assert (tmp_path / "nested" / "dir").is_dir()


@requires_blockbuster
async def test_duckdb_async_connect_does_not_block(tmp_path: Path) -> None:
    db = AsyncDuckDBDatabase(
        {"path": str(tmp_path / "nested" / "dir" / "records.duckdb"), "table": "records"}
    )
    try:
        with assert_no_blocking():
            await db.connect()
    finally:
        await db.close()
    assert (tmp_path / "nested" / "dir").is_dir()


def _native(path: Path, table: str) -> dict[str, object]:
    return {
        "path": str(path),
        "table": table,
        "layout": "native",
        "id_column": "k",
        "schema": {"fields": {"k": "string"}},
    }


@requires_blockbuster
@pytest.mark.parametrize("refusal", ["missing file", "missing table"])
async def test_duckdb_async_refused_connect_does_not_block(tmp_path: Path, refusal: str) -> None:
    """Bug: a refused connect shut its pool down with ``shutdown(wait=True)``
    on the loop -- a thread join -- and left the pool's thread running when the
    file would not open.
    """
    import duckdb

    path = tmp_path / "owner.duckdb"
    if refusal == "missing table":
        duckdb.connect(str(path)).close()
    db = AsyncDuckDBDatabase(_native(path, "tickets"))
    threads = set(threading.enumerate())
    with pytest.raises(RuntimeError), assert_no_blocking():
        await db.connect()
    # The store's pool names its threads ``ThreadPoolExecutor-N_M``; the loop's
    # own default executor, which ``asyncio.to_thread`` uses, is the loop's.
    left = {t.name for t in set(threading.enumerate()) - threads}
    assert not {name for name in left if name.startswith("ThreadPoolExecutor")}, left


@requires_blockbuster
async def test_duckdb_async_close_does_not_block(tmp_path: Path) -> None:
    """Bug: ``close()`` shut the pool down with ``shutdown(wait=True)`` on the loop."""
    db = AsyncDuckDBDatabase({"path": str(tmp_path / "records.duckdb"), "table": "records"})
    await db.connect()
    with assert_no_blocking():
        await db.close()
