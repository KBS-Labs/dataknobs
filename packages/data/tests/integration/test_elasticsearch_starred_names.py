"""A ``*`` in a field name is part of the name, on both twins.

Elasticsearch's ``exists`` reads it as any run of characters, so a filter built
from one answered for every field the name matched as a pattern.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest
from dataknobs_common.testing import requires_real_elasticsearch

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.backends.memory import SyncMemoryDatabase

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

#: ``p*`` and ``r*`` are fields; only ``r*`` is also a pattern another one matches.
ROWS: dict[str, dict[str, Any]] = {
    "a": {"day": "x", "obj": {"k": "v"}},
    "b": {"z": 1},
    "c": {"p*": "x", "q*": {"k": "v"}},
    "d": {"p*": "y"},
    "e": {"r*": "x", "rank": 1},
}

FILTERS: list[tuple[Operator, Any]] = [
    (Operator.EXISTS, None),
    (Operator.NOT_EXISTS, None),
    (Operator.EQ, "x"),
    (Operator.NEQ, "x"),
    (Operator.IN, ["x"]),
    (Operator.NOT_IN, ["x"]),
    (Operator.LIKE, "x%"),
    (Operator.NOT_LIKE, "x%"),
    (Operator.BETWEEN, ["x", "xx"]),
    (Operator.NOT_BETWEEN, ["x", "xx"]),
]
ASK_FOR_A_VALUE = [
    (op, value)
    for op, value in FILTERS
    if op.name.startswith("NOT_") or op in (Operator.EXISTS, Operator.NEQ)
]


async def settle(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


@pytest.fixture(params=[SyncElasticsearchDatabase, AsyncElasticsearchDatabase])
async def db(
    request: pytest.FixtureRequest, make_elasticsearch_test_index: Any
) -> AsyncIterator[Any]:
    for config in make_elasticsearch_test_index("test_starred_"):
        database = request.param(config)
        await settle(database.connect())
        for key, row in ROWS.items():
            await settle(database.upsert(key, Record(row)))
        yield database
        await settle(database.close())


async def _found(db: Any, name: str, op: Operator, value: Any) -> list[str]:
    records = await settle(db.search(Query(filters=[Filter(name, op, value)])))
    return sorted(r.storage_id for r in records)


@pytest.mark.parametrize(
    "name",
    ["*", "d*", "*y", "obj.*", "o*.k", "x*", "p*", "q*", "q*.k"],
)
async def test_a_starred_name_answers_as_it_does_in_memory(db: Any, name: str) -> None:
    memory = SyncMemoryDatabase()
    for key, row in ROWS.items():
        memory.upsert(key, Record(row))
    for op, value in FILTERS:
        expected = await _found(memory, name, op, value)
        assert await _found(db, name, op, value) == expected, op


@pytest.mark.parametrize(("op", "value"), ASK_FOR_A_VALUE)
async def test_a_starred_field_another_one_matches_is_refused(
    db: Any, op: Operator, value: Any
) -> None:
    with pytest.raises(ValueError, match=r"'r\*'.*reads the '\*'"):
        await _found(db, "r*", op, value)


async def test_a_comparison_with_a_starred_field_is_answered(db: Any) -> None:
    assert await _found(db, "r*", Operator.EQ, "x") == ["e"]
    assert await _found(db, "r*", Operator.LIKE, "x%") == ["e"]


async def test_a_field_mapped_later_is_seen(db: Any) -> None:
    assert await _found(db, "p*", Operator.EXISTS, None) == ["c", "d"]
    await settle(db.upsert("f", Record({"price": 1})))
    with pytest.raises(ValueError, match="cannot answer EXISTS"):
        await _found(db, "p*", Operator.EXISTS, None)
