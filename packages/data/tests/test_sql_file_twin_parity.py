"""The SQLite twins and the DuckDB twins read through one surface.

A caller writing flavour-agnostic code over a native table passes the same
arguments to either twin. What each twin answers is the parametrised native
store suite's to prove; this proves the questions agree.
"""

from __future__ import annotations

import pytest

from dataknobs_common.testing import assert_twin_types_agree
from dataknobs_data.backends.duckdb import AsyncDuckDBDatabase, SyncDuckDBDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.backends.sqlite_async import AsyncSQLiteDatabase

READS = ("read", "exists", "count", "search", "stream_read", "_count_all")
#: Synchronous on both halves: they run no statement.
LAYOUT = ("set_schema", "instance_capabilities")


@pytest.mark.parametrize(
    ("sync_type", "async_type"),
    [(SyncSQLiteDatabase, AsyncSQLiteDatabase), (SyncDuckDBDatabase, AsyncDuckDBDatabase)],
    ids=["sqlite", "duckdb"],
)
def test_the_read_surface_agrees(sync_type: type, async_type: type) -> None:
    assert_twin_types_agree(sync_type, async_type, READS + LAYOUT, unflavoured_members=LAYOUT)
