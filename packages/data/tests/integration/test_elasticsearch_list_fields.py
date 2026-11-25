# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A field holding a list is compared whole, on both twins, as on every backend.

Elasticsearch has no list type: it indexes each element as a value of the field,
so a filter matched a list when any one element did, and an empty list was no
value at all.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
from dataknobs_common.testing import requires_real_elasticsearch
from elasticsearch import Elasticsearch

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.backends.memory import SyncMemoryDatabase
from dataknobs_data.fields import VectorField
from dataknobs_data.query_logic import (
    ComplexQuery,
    FilterCondition,
    LogicCondition,
    LogicOperator,
)

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

ROWS: dict[str, dict[str, Any]] = {
    "pair": {"t": ["a", "b"], "s": 1},
    "one": {"t": ["b"]},
    "scalar": {"t": "a", "s": 2},
    "empty": {"t": []},
    "null_member": {"t": [None]},
    "absent": {"z": 1},
    "null": {"t": None},
    "in_object": {"o": {"t": ["a"]}},
    "object": {"o": {"t": "a"}},
    "empty_in_object": {"o": {"t": [], "u": 1}},
    "of_objects": {"l": [{"k": "a"}]},
    "in_objects": {"l": [{"k": ["a"]}]},
    "object_not_list": {"l": {"k": "a"}},
    "numbers": {"n": [1, 2]},
    "number": {"n": 1},
    "flags": {"b": [True]},
    "flag": {"b": True},
    "nested": {"t": [["a"]]},
    "only_empty": {"never": []},
}

COMPARISONS: list[tuple[Operator, Any]] = [
    (Operator.EQ, "a"),
    (Operator.NEQ, "a"),
    (Operator.IN, ["a"]),
    (Operator.IN, []),
    (Operator.NOT_IN, ["a"]),
    (Operator.NOT_IN, []),
    (Operator.GT, "a"),
    (Operator.GTE, "a"),
    (Operator.LT, "b"),
    (Operator.LTE, "a"),
    (Operator.BETWEEN, ["a", "b"]),
    (Operator.NOT_BETWEEN, ["a", "a"]),
    (Operator.LIKE, "a%"),
    (Operator.NOT_LIKE, "a%"),
    (Operator.NOT_LIKE, "x%"),
    (Operator.STARTS_WITH, "a"),
    (Operator.REGEX, "a"),
    (Operator.NEQ, None),
    (Operator.EXISTS, None),
    (Operator.NOT_EXISTS, None),
    (Operator.CONTAINS, "b"),
    (Operator.NOT_CONTAINS, "b"),
    (Operator.CONTAINS, None),
    (Operator.NOT_CONTAINS, None),
]
NUMBERS: list[tuple[Operator, Any]] = [
    (Operator.EQ, 1),
    (Operator.NEQ, 1),
    (Operator.NEQ, "a"),
    (Operator.IN, [1]),
    (Operator.NOT_IN, [1]),
    (Operator.GT, 0),
    (Operator.BETWEEN, [0, 5]),
    (Operator.NOT_BETWEEN, [0, 5]),
    (Operator.NOT_LIKE, "a%"),
    (Operator.EXISTS, None),
    (Operator.NOT_EXISTS, None),
    (Operator.CONTAINS, 1),
    (Operator.NOT_CONTAINS, 1),
    (Operator.NOT_CONTAINS, "a"),
]
FILTERS = [
    *[Filter(f, op, v) for f in ("t", "o.t", "l.k", "never") for op, v in COMPARISONS],
    *[Filter("n", op, v) for op, v in NUMBERS],
    *[
        Filter("b", op, True)
        for op in (Operator.EQ, Operator.NEQ, Operator.CONTAINS, Operator.NOT_CONTAINS)
    ],
    *[Filter(f, op) for f in ("o", "l") for op in (Operator.EXISTS, Operator.NOT_EXISTS)],
]


async def settle(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def ids(db: Any, query: Any) -> list[str]:
    return sorted(str(r.storage_id) for r in await settle(db.search(query)))


def memory(rows: dict[str, dict[str, Any]] = ROWS) -> SyncMemoryDatabase:
    db = SyncMemoryDatabase()
    for record_id, data in rows.items():
        db.upsert(record_id, Record(data))
    return db


@pytest.fixture(params=[SyncElasticsearchDatabase, AsyncElasticsearchDatabase])
async def db(
    request: pytest.FixtureRequest, make_elasticsearch_test_index: Any
) -> AsyncIterator[Any]:
    for config in make_elasticsearch_test_index("test_list_fields_"):
        database = request.param(config)
        await settle(database.connect())
        await settle(
            database.upsert_batch(
                [Record(data, storage_id=record_id) for record_id, data in ROWS.items()]
            )
        )
        yield database
        await settle(database.close())


async def test_every_filter_answers_as_memory_does(db: Any) -> None:
    expected = memory()
    wrong = {}
    for spec in FILTERS:
        query = Query(filters=[spec])
        found, right = await ids(db, query), await ids(expected, query)
        if found != right:
            wrong[f"{spec.field} {spec.operator.name} {spec.value!r}"] = (found, right)
        assert await settle(db.count(query)) == len(found)
    assert wrong == {}


async def test_a_condition_tree_answers_as_memory_does(db: Any) -> None:
    a, not_a = Query().filter("t", "==", "a"), Query().filter("t", "!=", "a")
    for query in (
        ComplexQuery.OR([a, not_a]),
        ComplexQuery.AND([not_a, Query().filter("s", "exists")]),
        ComplexQuery(condition=LogicCondition(LogicOperator.NOT, [FilterCondition(a.filters[0])])),
    ):
        assert await ids(db, query) == await ids(memory(), query)


WRITES = ["create", "upsert", "update", "create_batch", "upsert_batch", "update_batch"]


@pytest.mark.parametrize("how", WRITES)
async def test_every_write_says_which_fields_are_lists(db: Any, how: str) -> None:
    record = Record({"w": ["a"]}, storage_id="new")
    if how.startswith("update"):
        await settle(db.upsert("new", Record({"w": "a"})))
    if how == "create":
        await settle(db.create(record))
    elif how in ("create_batch", "upsert_batch"):
        await settle(getattr(db, how)([record]))
    elif how == "update_batch":
        await settle(db.update_batch([("new", record)]))
    else:
        await settle(getattr(db, how)("new", record))

    assert await ids(db, Query().filter("w", "==", "a")) == []
    assert await ids(db, Query().filter("w", "!=", "x")) == ["new"]


async def test_a_list_replaced_by_one_value_is_compared_as_it(db: Any) -> None:
    await settle(db.upsert("w", Record({"w": ["a"]})))
    await settle(db.update("w", Record({"w": "a"})))

    assert await ids(db, Query().filter("w", "==", "a")) == ["w"]


async def test_a_record_read_and_written_back_is_compared_the_same(db: Any) -> None:
    # A list of numbers or booleans reads back as a vector, of floats, which a
    # boolean field refuses.
    for record_id in ROWS.keys() - {"flags"}:
        await settle(db.update(record_id, await settle(db.read(record_id))))

    await test_every_filter_answers_as_memory_does(db)


async def test_a_record_reads_back_as_written(db: Any) -> None:
    record = await settle(db.read("pair"))

    assert {name: field.value for name, field in record.fields.items()} == ROWS["pair"]


async def test_a_list_inside_a_list_is_read_as_its_elements(db: Any) -> None:
    assert await ids(db, Query().filter("t", "contains", "a")) == ["nested", "pair"]


async def test_contains_does_not_reach_a_value_or_an_object_in_a_list(db: Any) -> None:
    assert await ids(db, Query().filter("o.t", "contains", "a")) == ["in_object"]
    assert await ids(db, Query().filter("l.k", "contains", "a")) == []


@pytest.mark.parametrize(
    "mappings",
    [
        None,
        {"properties": {"data": {"properties": {"t": {"type": "keyword"}}}}},
        {"date_detection": True},
    ],
)
@pytest.mark.parametrize("kind", [SyncElasticsearchDatabase, AsyncElasticsearchDatabase])
async def test_an_index_made_without_the_mapping(
    kind: Any, mappings: Any, make_elasticsearch_test_index: Any
) -> None:
    """One made by an earlier version, or elsewhere, maps the field dynamically."""
    for config in make_elasticsearch_test_index("test_list_fields_"):
        with Elasticsearch(f"http://{config['host']}:{config['port']}") as client:
            client.indices.delete(index=config["index"], ignore_unavailable=True)
            client.indices.create(index=config["index"], mappings=mappings)
            client.index(
                index=config["index"],
                id="old",
                document={"id": "old", "data": {"t": ["a"]}, "metadata": {}},
                refresh=True,
            )
        db = kind(config)
        await settle(db.connect())
        try:
            equals = Query().filter("t", "==", "a")
            assert await ids(db, equals) == ["old"], "stored before: compared as it was"

            await settle(db.upsert("new", Record({"t": ["a"], "2024-01-01": ["x"]})))
            assert await ids(db, equals) == ["old"]
            assert await ids(db, Query().filter("t", "!=", "a")) == ["new"]
            contains, not_contains = (
                Query().filter("t", op, "a") for op in ("contains", "not contains")
            )
            assert await ids(db, contains) == ["new"]
            assert await ids(db, not_contains) == ["old"]

            await settle(db.update("old", await settle(db.read("old"))))
            assert await ids(db, equals) == []
            assert await ids(db, contains) == ["new", "old"]
            assert await ids(db, not_contains) == []
        finally:
            await settle(db.close())


async def test_a_vector_search_filter_compares_the_whole_list(
    make_elasticsearch_test_index: Any,
) -> None:
    for config in make_elasticsearch_test_index("test_list_fields_"):
        vector = {"type": "dense_vector", "dims": 2, "index": True, "similarity": "cosine"}
        with Elasticsearch(f"http://{config['host']}:{config['port']}") as client:
            client.indices.delete(index=config["index"], ignore_unavailable=True)
        db = AsyncElasticsearchDatabase(
            {**config, "mappings": {"properties": {"data": {"properties": {"v": vector}}}}}
        )
        await db.connect()
        try:
            for record_id, tags in (("list", ["a"]), ("one", "a")):
                record = Record({"t": tags})
                record.fields["v"] = VectorField([1.0, 0.0], name="v")
                await db.upsert(record_id, record)
            found = await db.vector_search(
                np.array([1.0, 0.0]), vector_field="v", k=5, filter=Query().filter("t", "==", "a")
            )
            assert [r.record.storage_id or r.record.id for r in found] == ["one"]
        finally:
            await db.close()
