"""A search that some shard did not answer raises.

Elasticsearch answers it with a 200 holding what the other shards found, which
read as though the rest of the records did not exist. Each case here loses one
of two shards on a live node: one that can never be assigned, or one restarted
between two pages of a read.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import numpy as np
import pytest
from dataknobs_common.testing import requires_real_elasticsearch
from elasticsearch import Elasticsearch

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.exceptions import DatabaseError
from dataknobs_data.streaming import StreamConfig
from dataknobs_data.vector.hybrid import FusionStrategy, HybridSearchConfig

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

Database = SyncElasticsearchDatabase | AsyncElasticsearchDatabase
BACKENDS = [SyncElasticsearchDatabase, AsyncElasticsearchDatabase]
TWO_SHARDS = {"number_of_shards": 2, "number_of_replicas": 0}


async def settle(value: Any) -> Any:
    """The result of a call on either twin."""
    return await value if inspect.isawaitable(value) else value


async def streamed(db: Database, config: StreamConfig | None = None) -> list[Record]:
    stream = db.stream_read(config=config)
    if isinstance(stream, AsyncIterator):
        return [record async for record in stream]
    return list(stream)


@pytest.fixture
def es_config(make_elasticsearch_test_index: Any) -> Iterator[dict[str, Any]]:
    yield from make_elasticsearch_test_index("test_partial_")


@pytest.fixture
def admin(es_config: dict[str, Any]) -> Iterator[Elasticsearch]:
    with Elasticsearch(f"http://{es_config['host']}:{es_config['port']}") as client:
        yield client


@pytest.fixture(params=BACKENDS)
async def short(
    request: pytest.FixtureRequest, es_config: dict[str, Any], admin: Elasticsearch
) -> AsyncIterator[Database]:
    """A database on an index with one of its two shards unavailable."""
    index = es_config["index"]
    admin.indices.create(
        index=index,
        settings={**TWO_SHARDS, "index.routing.allocation.total_shards_per_node": 1},
        mappings={
            "properties": {
                "data": {
                    "properties": {
                        "n": {"type": "long"},
                        "v": {"type": "dense_vector", "dims": 3, "similarity": "cosine"},
                    }
                }
            }
        },
        wait_for_active_shards=0,
    )
    admin.cluster.health(index=index, wait_for_active_shards=1, timeout="10s")
    database = request.param(es_config)
    await settle(database.connect())
    # Only a record of the shard that is there can be written.
    reachable = [
        name
        for name in (f"r{n}" for n in range(12))
        if any(admin.search_shards(index=index, routing=name)["shards"])
    ]
    for name in reachable[:3]:
        await settle(database.create(Record(data={"n": 1, "v": [1.0, 0.0, 0.0]}, id=name)))
    yield database
    await settle(database.close())


@pytest.fixture(params=BACKENDS)
async def whole(
    request: pytest.FixtureRequest, es_config: dict[str, Any], admin: Elasticsearch
) -> AsyncIterator[Database]:
    """A database on an index of two working shards, read two records at a time."""
    admin.indices.create(index=es_config["index"], settings=TWO_SHARDS)
    database = request.param({**es_config, "search_page_size": 2})
    await settle(database.connect())
    for n in range(12):
        await settle(database.create(Record(data={"n": n}, id=f"r{n}")))
    yield database
    await settle(database.close())


def _restart_a_shard(admin: Elasticsearch, index: str) -> None:
    """Restart shard 0, which forgets the reads in progress on it."""
    admin.cluster.reroute(
        commands=[
            {
                "cancel": {
                    "index": index,
                    "shard": 0,
                    "node": next(iter(admin.nodes.info()["nodes"].values()))["name"],
                    "allow_primary": True,
                }
            }
        ],
        metric="none",
    )
    admin.cluster.health(index=index, wait_for_status="green", timeout="10s")


ONE = Filter("n", Operator.EQ, 1)
INCOMPLETE: dict[str, Callable[[Any], Any]] = {
    "search": lambda db: db.search(Query(limit_value=5)),
    "search-filtered": lambda db: db.search(Query(filters=[ONE], limit_value=5)),
    "count": lambda db: db.count(),
    "count-filtered": lambda db: db.count(Query(filters=[ONE])),
    "clear": lambda db: db.clear(),
    "vector_search": lambda db: db.vector_search(np.ones(3), vector_field="v", k=2),
    "hybrid_search": lambda db: db.hybrid_search("x", np.ones(3), vector_field="v", k=2),
    "hybrid_search-native": lambda db: db.hybrid_search(
        "x",
        np.ones(3),
        vector_field="v",
        k=2,
        config=HybridSearchConfig(fusion_strategy=FusionStrategy.NATIVE),
    ),
}


@pytest.mark.parametrize("name", INCOMPLETE)
async def test_a_shard_that_is_not_there_raises(short: Database, name: str) -> None:
    with pytest.raises(DatabaseError, match="no_shard_available") as caught:
        await settle(INCOMPLETE[name](short))
    assert caught.value.context["failures"]


@pytest.mark.parametrize("short", [AsyncElasticsearchDatabase], indirect=True)
async def test_a_stream_missing_a_shard_raises(short: Database) -> None:
    """The sync twin streams what ``search`` returns."""
    with pytest.raises(DatabaseError, match="no_shard_available"):
        await streamed(short)


async def test_a_text_search_missing_a_shard_raises(short: Database) -> None:
    """The text half of a hybrid search, whose vector half would raise for it."""
    with pytest.raises(DatabaseError, match="no_shard_available"):
        await short._text_search_for_hybrid("x", ["s"], k=2)


async def test_a_record_that_is_there_is_still_read(short: Database) -> None:
    """``get`` asks one shard, so the other's absence is not its concern."""
    written = [f"r{n}" for n in range(12)]
    reachable = []
    for record_id in written:
        try:
            if await settle(short.read(record_id)) is not None:
                reachable.append(record_id)
        except Exception as error:
            assert "no_shard_available" in str(error) or "503" in str(error)
    assert len(reachable) == 3


async def test_a_shard_lost_between_pages_raises(whole: Database, admin: Elasticsearch) -> None:
    send = whole._search_hits
    pages = 0

    def before_page() -> None:
        nonlocal pages
        pages += 1
        if pages == 2:
            _restart_a_shard(admin, whole.index_name)

    def sync_page(body: dict[str, Any]) -> Any:
        before_page()
        return send(body)

    whole._search_hits = sync_page  # type: ignore[method-assign,assignment]
    with pytest.raises(DatabaseError, match="search_context_missing"):
        await settle(whole.search(Query()))


@pytest.mark.parametrize("whole", [AsyncElasticsearchDatabase], indirect=True)
async def test_a_shard_lost_during_a_stream_raises(
    whole: AsyncElasticsearchDatabase, admin: Elasticsearch
) -> None:
    """The sync twin has read every record before it yields one."""
    stream = whole.stream_read(config=StreamConfig(batch_size=2))
    with pytest.raises(DatabaseError, match="search_context_missing"):
        seen = 0
        async for _ in stream:
            seen += 1
            if seen == 1:
                _restart_a_shard(admin, whole.index_name)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_mapping_some_index_did_not_give_is_not_remembered(
    backend: type[Database], es_config: dict[str, Any], admin: Elasticsearch
) -> None:
    """An alias: the index that is down may map the field differently."""
    up, down, alias = (es_config["index"] + suffix for suffix in ("", "_down", "_alias"))
    admin.indices.create(index=up, mappings={"properties": {"data.n": {"type": "long"}}})
    try:
        admin.indices.create(
            index=down,
            settings={"index.routing.allocation.require._name": "no_such_node"},
            wait_for_active_shards=0,
        )
        admin.indices.put_alias(index=[up, down], name=alias)
        db = backend({**es_config, "index": alias})
        await settle(db.connect())
        try:
            for operation in (db.count, db.search):
                with pytest.raises(DatabaseError, match="no_shard_available"):
                    await settle(operation(Query(filters=[ONE], limit_value=5)))
            assert db._known_types.unknown(["n"]) == ["n"]
        finally:
            await settle(db.close())
    finally:
        admin.indices.delete(index=down, ignore_unavailable=True)


async def test_a_complete_answer_is_unchanged(whole: Database) -> None:
    assert len(await settle(whole.search(Query()))) == 12
    assert len(await settle(whole.search(Query(limit_value=5)))) == 5
    assert await settle(whole.count()) == 12
    assert len(await streamed(whole, StreamConfig(batch_size=2))) == 12
    assert await settle(whole.clear()) == 12
