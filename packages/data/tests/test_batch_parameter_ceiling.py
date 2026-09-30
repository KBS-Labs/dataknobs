"""A batch of any size is one call, on every SQL backend.

The batch verbs (``create_batch``, ``upsert_batch``, ``update_batch``,
``delete_batch``) and a membership filter (``IN`` / ``NOT IN``) bind values as
query parameters, and some drivers cap how many one statement may carry:
SQLite at its connection's ``SQLITE_LIMIT_VARIABLE_NUMBER`` (32766 by default),
asyncpg at 32767. A batch or a list past that cap must still succeed, and a
batch write must stay all-or-nothing however many statements it takes.

``N`` is past both caps for every verb: a batch write binds at least one
parameter per record, and a membership list one per member.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from dataknobs_common.testing import requires_postgres

from dataknobs_data import AsyncDatabase, Query, Record, SyncDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase
from dataknobs_data.exceptions import DuplicateRecordError
from dataknobs_data.query import Filter, Operator

if TYPE_CHECKING:
    from collections.abc import Iterator

N = 33_000

SQL_BACKENDS = [
    "sqlite",
    "duckdb",
    pytest.param("postgres", marks=requires_postgres),
]


@pytest.fixture(params=SQL_BACKENDS)
def backend(request: pytest.FixtureRequest) -> Iterator[tuple[str, dict[str, Any]]]:
    """One SQL backend's kind and constructor config."""
    kind = request.param
    if kind == "postgres":
        yield from (
            (kind, c) for c in request.getfixturevalue("make_postgres_test_db")("test_ceiling_")
        )
        return
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        yield (
            kind,
            {
                "sqlite": {"path": str(root / "records.db")},
                "duckdb": {"path": str(root / "records.duckdb"), "table": "records"},
            }[kind],
        )


@pytest.fixture(params=["memory", "file", *SQL_BACKENDS])
def any_backend(request: pytest.FixtureRequest) -> Iterator[tuple[str, dict[str, Any]]]:
    """One backend of any kind, in-process or SQL."""
    kind = request.param
    if kind == "postgres":
        yield from (
            (kind, c) for c in request.getfixturevalue("make_postgres_test_db")("test_ceiling_")
        )
        return
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        yield (
            kind,
            {
                "memory": {},
                "file": {"path": str(root / "records.json")},
                "sqlite": {"path": str(root / "records.db")},
                "duckdb": {"path": str(root / "records.duckdb"), "table": "records"},
            }[kind],
        )


def _ids(n: int) -> list[str]:
    return [f"r{i}" for i in range(n)]


def _records(n: int, offset: int = 0) -> list[Record]:
    return [Record({"v": i + offset}, storage_id=f"r{i}") for i in range(n)]


def _found(records: list[Record]) -> set[str]:
    return {str(r.storage_id) for r in records}


def test_every_batch_verb_takes_a_batch_past_the_ceiling_sync(
    backend: tuple[str, dict[str, Any]],
) -> None:
    """Each verb and a membership filter take N items in one call."""
    kind, config = backend
    db = SyncDatabase.from_backend(kind, config=config)
    try:
        assert db.create_batch(_records(N)) == _ids(N)
        assert db.count() == N

        assert db.upsert_batch(_records(N, offset=1)) == _ids(N)
        assert db.read(f"r{N - 1}").get_value("v") == N

        updates = [(f"r{i}", Record({"v": -i})) for i in range(N)]
        assert db.update_batch(updates) == [True] * N
        assert db.read(f"r{N - 1}").get_value("v") == -(N - 1)

        db.create(Record({"v": 1}, storage_id="extra"))
        wanted = [-i for i in range(N)]
        assert _found(db.search(Query(filters=[Filter("v", Operator.IN, wanted)]))) == set(_ids(N))
        assert _found(db.search(Query(filters=[Filter("v", Operator.NOT_IN, wanted)]))) == {"extra"}
        by_id = Query(filters=[Filter("id", Operator.IN, _ids(N))])
        assert _found(db.search(by_id)) == set(_ids(N))

        assert db.delete_batch([*_ids(N), "absent"]) == [True] * N + [False]
        assert db.count() == 1
    finally:
        db.close()


async def test_every_batch_verb_takes_a_batch_past_the_ceiling_async(
    backend: tuple[str, dict[str, Any]],
) -> None:
    """The async twin of the sync test."""
    kind, config = backend
    db = await AsyncDatabase.from_backend(kind, config=config)
    try:
        assert await db.create_batch(_records(N)) == _ids(N)
        assert await db.count() == N

        assert await db.upsert_batch(_records(N, offset=1)) == _ids(N)
        assert (await db.read(f"r{N - 1}")).get_value("v") == N

        updates = [(f"r{i}", Record({"v": -i})) for i in range(N)]
        assert await db.update_batch(updates) == [True] * N
        assert (await db.read(f"r{N - 1}")).get_value("v") == -(N - 1)

        await db.create(Record({"v": 1}, storage_id="extra"))
        wanted = [-i for i in range(N)]
        found = await db.search(Query(filters=[Filter("v", Operator.IN, wanted)]))
        assert _found(found) == set(_ids(N))
        found = await db.search(Query(filters=[Filter("v", Operator.NOT_IN, wanted)]))
        assert _found(found) == {"extra"}
        found = await db.search(Query(filters=[Filter("id", Operator.IN, _ids(N))]))
        assert _found(found) == set(_ids(N))

        assert await db.delete_batch([*_ids(N), "absent"]) == [True] * N + [False]
        assert await db.count() == 1
    finally:
        await db.close()


def test_a_batch_past_the_ceiling_is_all_or_nothing_sync(
    backend: tuple[str, dict[str, Any]],
) -> None:
    """A colliding id at the end of a long batch writes none of it."""
    kind, config = backend
    db = SyncDatabase.from_backend(kind, config=config)
    try:
        db.create(Record({"v": "kept"}, storage_id=f"r{N - 1}"))
        with pytest.raises(DuplicateRecordError):
            db.create_batch(_records(N))
        assert db.count() == 1
        assert db.read(f"r{N - 1}").get_value("v") == "kept"
    finally:
        db.close()


async def test_a_batch_past_the_ceiling_is_all_or_nothing_async(
    backend: tuple[str, dict[str, Any]],
) -> None:
    """The async twin of the sync test."""
    kind, config = backend
    db = await AsyncDatabase.from_backend(kind, config=config)
    try:
        await db.create(Record({"v": "kept"}, storage_id=f"r{N - 1}"))
        with pytest.raises(DuplicateRecordError):
            await db.create_batch(_records(N))
        assert await db.count() == 1
        assert (await db.read(f"r{N - 1}")).get_value("v") == "kept"
    finally:
        await db.close()


def _own_updates(n: int) -> list[tuple[str, Record]]:
    return [(f"r{i}", Record({"v": -i - 1}, metadata={"m": -i - 1})) for i in range(n)]


def test_update_batch_writes_each_record_its_own_update_sync(
    any_backend: tuple[str, dict[str, Any]],
) -> None:
    """Every record in the batch ends with its own data and its own metadata."""
    kind, config = any_backend
    db = SyncDatabase.from_backend(kind, config=config)
    try:
        db.create_batch([Record({"v": i}, storage_id=f"r{i}", metadata={"m": i}) for i in range(4)])
        assert db.update_batch(_own_updates(4)) == [True] * 4
        stored = {f"r{i}": db.read(f"r{i}") for i in range(4)}
        assert {k: (r.get_value("v"), r.metadata.get("m")) for k, r in stored.items()} == {
            f"r{i}": (-i - 1, -i - 1) for i in range(4)
        }
    finally:
        db.close()


async def test_update_batch_writes_each_record_its_own_update_async(
    any_backend: tuple[str, dict[str, Any]],
) -> None:
    """The async twin of the sync test."""
    kind, config = any_backend
    db = await AsyncDatabase.from_backend(kind, config=config)
    try:
        await db.create_batch(
            [Record({"v": i}, storage_id=f"r{i}", metadata={"m": i}) for i in range(4)]
        )
        assert await db.update_batch(_own_updates(4)) == [True] * 4
        stored = {f"r{i}": await db.read(f"r{i}") for i in range(4)}
        assert {k: (r.get_value("v"), r.metadata.get("m")) for k, r in stored.items()} == {
            f"r{i}": (-i - 1, -i - 1) for i in range(4)
        }
    finally:
        await db.close()


def test_update_batch_keeps_the_last_update_of_a_repeated_id_sync(
    any_backend: tuple[str, dict[str, Any]],
) -> None:
    """A repeated id ends as its last update, as a loop of ``update`` would."""
    kind, config = any_backend
    db = SyncDatabase.from_backend(kind, config=config)
    try:
        db.create(Record({"v": 0}, storage_id="a"))
        updates = [("a", Record({"v": 1})), ("absent", Record({"v": 9})), ("a", Record({"v": 2}))]
        assert db.update_batch(updates) == [True, False, True]
        assert db.read("a").get_value("v") == 2
        assert db.read("absent") is None
    finally:
        db.close()


async def test_update_batch_keeps_the_last_update_of_a_repeated_id_async(
    any_backend: tuple[str, dict[str, Any]],
) -> None:
    """The async twin of the sync test."""
    kind, config = any_backend
    db = await AsyncDatabase.from_backend(kind, config=config)
    try:
        await db.create(Record({"v": 0}, storage_id="a"))
        updates = [("a", Record({"v": 1})), ("absent", Record({"v": 9})), ("a", Record({"v": 2}))]
        assert await db.update_batch(updates) == [True, False, True]
        assert (await db.read("a")).get_value("v") == 2
        assert await db.read("absent") is None
    finally:
        await db.close()


def test_sqlite_reads_its_ceiling_from_the_connection(tmp_path: Path) -> None:
    """A connection's own lower limit is the one the batch verbs respect.

    SQLite's variable limit is a per-connection setting, and a build may ship
    with a lower default than 32766, so the ceiling is read, not assumed.
    """
    db = SyncSQLiteDatabase({"path": str(tmp_path / "records.db")})
    db.connect()
    try:
        db.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 7)
        n = 10
        assert db.create_batch(_records(n)) == _ids(n)
        assert db.upsert_batch(_records(n, offset=1)) == _ids(n)
        assert db.update_batch([(f"r{i}", Record({"v": -i})) for i in range(n)]) == [True] * n
        found = db.search(Query(filters=[Filter("v", Operator.IN, [-i for i in range(n)])]))
        assert _found(found) == set(_ids(n))
        assert db.delete_batch(_ids(n)) == [True] * n
    finally:
        db.close()


def test_update_batch_costs_in_proportion_to_its_size(tmp_path: Path) -> None:
    """Twice the batch is about twice the work, not four times.

    Counted in SQLite virtual-machine steps through its progress handler, so
    the measure is deterministic: a statement that tests every update against
    every row grows with the square of the batch.
    """
    db = SyncSQLiteDatabase({"path": str(tmp_path / "records.db")})
    db.connect()
    try:
        db.create_batch(_records(4000))
        steps = [0]

        def count() -> int:
            steps[0] += 1
            return 0

        def work(n: int) -> int:
            steps[0] = 0
            db.conn.set_progress_handler(count, 100)
            try:
                db.update_batch([(f"r{i}", Record({"v": -i})) for i in range(n)])
            finally:
                db.conn.set_progress_handler(None, 0)
            return steps[0]

        small, large = work(2000), work(4000)
        assert large < 3 * small, (small, large)
    finally:
        db.close()
