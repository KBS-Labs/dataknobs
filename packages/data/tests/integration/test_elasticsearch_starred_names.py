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
from elasticsearch import Elasticsearch

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.backends.memory import SyncMemoryDatabase

BACKENDS = [SyncElasticsearchDatabase, AsyncElasticsearchDatabase]

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


@pytest.fixture(params=BACKENDS)
async def db(
    request: pytest.FixtureRequest, make_elasticsearch_test_index: Any
) -> AsyncIterator[Any]:
    for config in make_elasticsearch_test_index("test_starred_"):
        database = request.param(config)
        await settle(database.connect())
        await settle(
            database.upsert_batch([Record(row, storage_id=key) for key, row in ROWS.items()])
        )
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


async def test_a_starred_field_another_one_matches_is_refused(db: Any) -> None:
    for op, value in ASK_FOR_A_VALUE:
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


async def test_a_field_matched_only_by_its_keyword_sub_field_is_refused(db: Any) -> None:
    """``exists`` on ``d*d`` finds ``day`` through ``day.keyword``."""
    await settle(db.upsert("f", Record({"d*d": "x"})))
    for op, value in ASK_FOR_A_VALUE:
        with pytest.raises(ValueError, match="reads the"):
            await _found(db, "d*d", op, value)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_fields_in_separate_indices_of_an_alias_are_refused(
    backend: Any, make_elasticsearch_test_index: Any
) -> None:
    """The index without ``d*`` would answer for its ``day``."""
    for config in make_elasticsearch_test_index("test_starred_alias_"):
        starred, other, alias = (config["index"] + suffix for suffix in ("", "_other", "_alias"))
        with Elasticsearch(f"http://{config['host']}:{config['port']}") as admin:
            try:
                for index, row in ((starred, {"d*": "x"}), (other, {"day": "x"})):
                    with SyncElasticsearchDatabase({**config, "index": index}) as database:
                        database.upsert(index, Record(row))
                admin.indices.put_alias(index=[starred, other], name=alias)
                db = backend({**config, "index": alias})
                await settle(db.connect())
                try:
                    for op, value in ASK_FOR_A_VALUE:
                        with pytest.raises(ValueError, match="reads the"):
                            await _found(db, "d*", op, value)
                finally:
                    await settle(db.close())
            finally:
                admin.indices.delete(index=other, ignore_unavailable=True)
