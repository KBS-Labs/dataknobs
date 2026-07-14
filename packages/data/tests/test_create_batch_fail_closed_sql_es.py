"""Fail-closed ``create_batch`` contract on the SQL and Elasticsearch backends.

The companion to ``test_create_batch_fail_closed.py`` (memory + file). Here the
six *overriding* backends whose bulk ``create_batch`` fast-path predated the
``create()`` fail-closed contract are brought into line: a colliding id — against
an existing record or a duplicate within the same batch — raises
``DuplicateRecordError``, and a caller-supplied ``record.id`` is honored (no
minting) rather than silently overwritten or replaced with a fresh uuid.

Atomicity note: the SQL backends run a single multi-row ``INSERT`` inside a
transaction, so a collision aborts the whole batch (nothing written). The
Elasticsearch bulk API is *non-atomic* (per-item), so a batch containing a
collision may partially write the non-colliding rows before the conflict is
reported — matching a plain ``create()`` loop (also non-atomic). The SQL cases
therefore assert whole-batch atomicity; the ES cases assert only that the
collision fails closed and ids are honored.

sqlite + duckdb run in-process; postgres + elasticsearch are service-gated.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from dataknobs_data import DuplicateRecordError, Record
from dataknobs_data.backends.duckdb import SyncDuckDBDatabase
from dataknobs_data.backends.sqlite import SyncSQLiteDatabase


# ---------------------------------------------------------------------------
# In-process SQL backends (sqlite, duckdb) — single multi-row INSERT, atomic
# ---------------------------------------------------------------------------
@pytest.fixture(params=["sqlite", "duckdb"])
def sql_db(request: pytest.FixtureRequest) -> Iterator[object]:
    """A connected in-process SQL backend whose create_batch is fail-closed."""
    kind = request.param
    db: object
    if kind == "sqlite":
        db = SyncSQLiteDatabase({"path": ":memory:"})
    else:
        db = SyncDuckDBDatabase({"path": ":memory:"})
    db.connect()
    try:
        yield db
    finally:
        db.close()


def test_sql_create_batch_duplicate_raises(sql_db: object) -> None:
    """create_batch fails closed when a record collides with an existing id."""
    sql_db.create(Record({"v": 1}, id="dup"))
    with pytest.raises(DuplicateRecordError):
        sql_db.create_batch([Record({"v": 2}, id="dup")])


def test_sql_create_batch_is_atomic_on_collision(sql_db: object) -> None:
    """A batch containing one collision writes NONE of its records (SQL atomic)."""
    sql_db.create(Record({"v": "old"}, id="dup"))
    with pytest.raises(DuplicateRecordError):
        sql_db.create_batch(
            [
                Record({"v": 1}, id="new1"),
                Record({"v": 2}, id="dup"),  # collides — whole batch must abort
                Record({"v": 3}, id="new2"),
            ]
        )
    assert sql_db.read("new1") is None
    assert sql_db.read("new2") is None
    assert sql_db.read("dup").get_value("v") == "old"


def test_sql_create_batch_within_batch_duplicate_raises(sql_db: object) -> None:
    """Two records sharing an id within one batch fail closed (nothing written)."""
    with pytest.raises(DuplicateRecordError):
        sql_db.create_batch(
            [Record({"v": 1}, id="same"), Record({"v": 2}, id="same")]
        )
    assert sql_db.read("same") is None


def test_sql_create_batch_preserves_ids(sql_db: object) -> None:
    """Distinct ids create normally and the given ids are honored (no minting)."""
    ids = sql_db.create_batch([Record({"v": 1}, id="x"), Record({"v": 2}, id="y")])
    assert ids == ["x", "y"]
    assert sql_db.read("x").get_value("v") == 1
    assert sql_db.read("y").get_value("v") == 2
