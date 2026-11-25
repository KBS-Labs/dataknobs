"""An update replaces the stored record, on both twins, as on every backend.

Elasticsearch merges a partial update into the stored document, so a field the
new record dropped was still stored, searched and returned.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest
from dataknobs_common.testing import requires_real_elasticsearch
from elasticsearch import Elasticsearch

from dataknobs_data import ConcurrencyError, Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_data.fields import VectorField

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

STORED = {"t": 1, "o": {"x": 1, "y": 2}}
NEW: dict[str, Any] = {"o": {"x": 3}}


async def settle(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


@pytest.fixture(params=[SyncElasticsearchDatabase, AsyncElasticsearchDatabase])
async def db(
    request: pytest.FixtureRequest, make_elasticsearch_test_index: Any
) -> AsyncIterator[Any]:
    for config in make_elasticsearch_test_index("test_update_replaces_"):
        database = request.param(config)
        await settle(database.connect())
        await settle(database.upsert("a", Record(STORED, metadata={"m": 1, "k": 2})))
        yield database
        await settle(database.close())


async def _update(db: Any, how: str, record: Record) -> None:
    if how == "update_batch":
        assert await settle(db.update_batch([("a", record)])) == [True]
        return
    version = await settle(db.get_version("a")) if "expected_version" in how else None
    write = db.upsert if how.startswith("upsert") else db.update
    assert await settle(write("a", record, expected_version=version)) in (True, "a")


HOW = ["update", "update expected_version", "upsert expected_version", "update_batch"]


@pytest.mark.parametrize("how", HOW)
async def test_what_the_new_record_dropped_is_gone(db: Any, how: str) -> None:
    memory = SyncMemoryDatabase()
    memory.upsert("a", Record(STORED, metadata={"m": 1, "k": 2}))
    for database in (memory, db):
        await _update(database, how, Record(NEW, metadata={"k": 3}))
        read = await settle(database.read("a"))
        assert {name: f.value for name, f in read.fields.items()} == NEW
        assert read.metadata == {"k": 3}
    for gone in ("t", "o.y"):
        query = Query(filters=[Filter(gone, Operator.EXISTS)])
        assert await settle(db.search(query)) == []
    by_id = Query(filters=[Filter("id", Operator.EQ, "a")])
    assert [r.storage_id for r in await settle(db.search(by_id))] == ["a"]


@pytest.mark.parametrize("how", HOW)
async def test_an_empty_record_leaves_nothing(db: Any, how: str) -> None:
    await _update(db, how, Record({}))
    assert await settle(db.search(Query(filters=[Filter("t", Operator.EXISTS)]))) == []


@pytest.mark.parametrize("how", HOW)
async def test_values_are_stored_as_given(db: Any, how: str) -> None:
    """The record travels as a script parameter, which reads it as JSON."""
    data = {
        "n": 2**53 + 1,
        "f": 0.1,
        "whole": 1.0,
        "s": "é ${x} {{y}} \" \\ '",
        "flag": False,
        "none": None,
        "list": ["a", None],
        "objects": [{"k": 1}],
        "empty": {},
    }
    await _update(db, how, Record(data))
    read = await settle(db.read("a"))
    assert {name: f.value for name, f in read.fields.items()} == data
    assert isinstance(read.get_value("whole"), float)


async def test_a_vector_survives_an_update(db: Any) -> None:
    vector = VectorField([0.1, 0.2, 0.3], name="v", source_field="text", model_name="m")
    record = Record({"t": 1})
    record.fields["v"] = vector
    await settle(db.upsert("a", record))
    read = await settle(db.read("a"))
    read.set_value("t", 2)
    assert await settle(db.update("a", read))
    stored = await settle(db.read("a"))
    assert stored.get_value("t") == 2
    assert stored.fields["v"].value == pytest.approx([0.1, 0.2, 0.3])
    assert (stored.fields["v"].source_field, stored.fields["v"].model_name) == ("text", "m")


async def test_a_repeated_id_takes_its_last_update(db: Any) -> None:
    updates = [("a", Record({"p": 1})), ("missing", Record({"p": 2})), ("a", Record({"q": 3}))]
    assert await settle(db.update_batch(updates)) == [True, False, True]
    read = await settle(db.read("a"))
    assert {name: f.value for name, f in read.fields.items()} == {"q": 3}
    assert not await settle(db.exists("missing"))


async def test_a_stale_version_changes_nothing(db: Any) -> None:
    stale = await settle(db.get_version("a"))
    await settle(db.update("a", Record({"p": 1})))
    with pytest.raises(ConcurrencyError):
        await settle(db.update("a", Record({"q": 1}), expected_version=stale))
    assert not await settle(db.update("missing", Record({"q": 1})))
    read = await settle(db.read("a"))
    assert {name: f.value for name, f in read.fields.items()} == {"p": 1}


@pytest.mark.parametrize("how", HOW)
async def test_a_top_level_key_is_replaced_too(db: Any, how: str) -> None:
    """One another writer stored, as indexing the record would drop it."""
    with Elasticsearch(f"http://{db.config.host}:{db.config.port}") as admin:
        admin.index(index=db.index_name, id="a", document={"id": "a", "other": 1}, refresh=True)
        await _update(db, how, Record(NEW))
        assert "other" not in admin.get(index=db.index_name, id="a")["_source"]


async def test_an_update_that_changes_nothing_advances_the_version(db: Any) -> None:
    before = await settle(db.get_version("a"))
    assert await settle(db.update("a", await settle(db.read("a"))))
    assert await settle(db.get_version("a")) != before
