"""An Elasticsearch request that fails raises; only a missing document is absent.

Both backends answered a failed request with an ordinary result: the sync one
read a rejected query as no matches, and both read a closed or write-blocked
index as a missing record or a write that did nothing. Each case here breaks
the index on a live node, through a separate client, and runs both twins
through the same operations.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import numpy as np
import pytest
from dataknobs_common.testing import requires_real_elasticsearch
from elasticsearch import (
    ApiError,
    BadRequestError,
    ConflictError,
    Elasticsearch,
    NotFoundError,
)

from dataknobs_data import Filter, Operator, Query, Record
from dataknobs_data.backends.elasticsearch import SyncElasticsearchDatabase
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.exceptions import DatabaseError, DuplicateRecordError
from dataknobs_data.streaming import ConflictPolicy, StreamConfig

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

Database = SyncElasticsearchDatabase | AsyncElasticsearchDatabase

#: ``terms`` takes no null, so Elasticsearch answers this one with a 400.
REJECTED = Query(filters=[Filter("colour", Operator.NOT_IN, [None, "blue"])])


async def settle(value: Any) -> Any:
    """The result of a call on either twin."""
    return await value if inspect.isawaitable(value) else value


@pytest.fixture
def es_config(make_elasticsearch_test_index: Any) -> Iterator[dict[str, Any]]:
    yield from make_elasticsearch_test_index("test_failures_")


@pytest.fixture
def admin(es_config: dict[str, Any]) -> Iterator[Elasticsearch]:
    with Elasticsearch(f"http://{es_config['host']}:{es_config['port']}") as client:
        yield client


@pytest.fixture(params=[SyncElasticsearchDatabase, AsyncElasticsearchDatabase])
async def db(request: pytest.FixtureRequest, es_config: dict[str, Any]) -> AsyncIterator[Database]:
    database = request.param(es_config)
    await settle(database.connect())
    await settle(database.create(_record("a")))
    yield database
    await settle(database.close())


def _record(record_id: str, n: Any = 1) -> Record:
    return Record(data={"n": n}, id=record_id)


READS: dict[str, Callable[[Any], Any]] = {
    "search": lambda db: db.search(Query(limit_value=5)),
    "search-paged": lambda db: db.search(Query()),
    "count": lambda db: db.count(),
    "read": lambda db: db.read("a"),
    "read_batch": lambda db: db.read_batch(["a"]),
    "get_version": lambda db: db.get_version("a"),
    "exists": lambda db: db.exists("a"),
}
WRITES: dict[str, Callable[[Any], Any]] = {
    "create": lambda db: db.create(_record("z")),
    "upsert": lambda db: db.upsert("a", _record("a")),
    "update": lambda db: db.update("a", _record("a", 2)),
    "conditional-update": lambda db: db.update("a", _record("a", 2), expected_version="0:1"),
    "delete": lambda db: db.delete("a"),
    "conditional-delete": lambda db: db.delete("a", expected_version="0:1"),
    "clear": lambda db: db.clear(),
}
BATCHES: dict[str, Callable[[Any], Any]] = {
    "create_batch": lambda db: db.create_batch([_record("z")]),
    "upsert_batch": lambda db: db.upsert_batch([_record("a")]),
    "update_batch": lambda db: db.update_batch([("a", _record("a", 2))]),
    "delete_batch": lambda db: db.delete_batch(["a"]),
}
VECTOR: dict[str, Callable[[Any], Any]] = {
    "vector_search": lambda db: db.vector_search(np.ones(3), vector_field="v", k=1),
    "create_vector_index": lambda db: db.create_vector_index("v", 3),
}


@pytest.mark.parametrize("query", [REJECTED, Query(filters=REJECTED.filters, limit_value=5)])
async def test_a_rejected_query_raises(db: Database, query: Query) -> None:
    with pytest.raises(BadRequestError):
        await settle(db.search(query))
    with pytest.raises(BadRequestError):
        await settle(db.count(query))


@pytest.mark.parametrize("name", [*READS, *WRITES])
async def test_a_closed_index_raises(db: Database, admin: Elasticsearch, name: str) -> None:
    admin.indices.close(index=db.index_name)
    with pytest.raises(BadRequestError):
        await settle({**READS, **WRITES}[name](db))


@pytest.mark.parametrize("name", BATCHES)
async def test_a_batch_on_a_closed_index_raises(
    db: Database, admin: Elasticsearch, name: str
) -> None:
    admin.indices.close(index=db.index_name)
    with pytest.raises(DatabaseError, match="closed"):
        await settle(BATCHES[name](db))


@pytest.mark.parametrize("name", WRITES)
async def test_a_blocked_write_raises(db: Database, admin: Elasticsearch, name: str) -> None:
    admin.indices.put_settings(index=db.index_name, settings={"index.blocks.write": True})
    with pytest.raises(ApiError) as caught:
        await settle(WRITES[name](db))
    assert caught.value.status_code == 403


@pytest.mark.parametrize("name", BATCHES)
async def test_a_blocked_batch_raises(db: Database, admin: Elasticsearch, name: str) -> None:
    admin.indices.put_settings(index=db.index_name, settings={"index.blocks.write": True})
    with pytest.raises(DatabaseError, match="cluster_block_exception"):
        await settle(BATCHES[name](db))


@pytest.mark.parametrize("name", [*READS, "delete", "conditional-delete"])
async def test_a_missing_index_is_not_a_missing_record(
    db: Database, admin: Elasticsearch, name: str
) -> None:
    """Elasticsearch answers 404 to both."""
    admin.indices.delete(index=db.index_name)
    with pytest.raises(NotFoundError, match="index_not_found"):
        await settle({**READS, **WRITES}[name](db))


async def test_a_batch_delete_from_a_missing_index_raises(
    db: Database, admin: Elasticsearch
) -> None:
    admin.indices.delete(index=db.index_name)
    with pytest.raises(DatabaseError, match="index_not_found"):
        await settle(db.delete_batch(["a"]))


@pytest.mark.parametrize("name", VECTOR)
async def test_a_vector_operation_on_a_missing_index_raises(
    db: Database, admin: Elasticsearch, name: str
) -> None:
    admin.indices.delete(index=db.index_name)
    with pytest.raises(ValueError, match="index_not_found"):
        await settle(VECTOR[name](db))


async def test_a_hybrid_search_on_a_missing_index_raises(
    db: Database, admin: Elasticsearch
) -> None:
    """The sync text leg is ``search``, which raises the client's own error."""
    admin.indices.delete(index=db.index_name)
    with pytest.raises((ValueError, NotFoundError), match="index_not_found"):
        await settle(db.hybrid_search("blue", np.ones(3), vector_field="v", k=1))


async def test_a_vector_index_blocked_by_the_server_raises(
    db: Database, admin: Elasticsearch
) -> None:
    """A 403 is none of the errors the vector methods wrap."""
    admin.indices.put_settings(index=db.index_name, settings={"index.blocks.metadata": True})
    with pytest.raises(ApiError) as caught:
        await settle(db.create_vector_index("v", 3))
    assert caught.value.status_code == 403
    admin.indices.put_settings(index=db.index_name, settings={"index.blocks.metadata": False})


async def test_a_vector_search_blocked_by_the_server_raises(
    db: Database, admin: Elasticsearch
) -> None:
    admin.indices.put_settings(index=db.index_name, settings={"index.blocks.read": True})
    with pytest.raises(ApiError) as caught:
        await settle(db.vector_search(np.ones(3), vector_field="v", k=1))
    assert caught.value.status_code == 403


async def test_a_clear_that_leaves_a_record_raises(db: Database, admin: Elasticsearch) -> None:
    """A write the search has not seen yet is a conflict for the delete."""
    admin.indices.put_settings(index=db.index_name, settings={"index.refresh_interval": "-1"})
    await settle(db.create(_record("b")))
    stored = admin.get(index=db.index_name, id="a")["_source"]
    admin.index(index=db.index_name, id="a", document=stored, refresh=False)
    with pytest.raises(ConflictError):
        await settle(db.clear())
    assert await settle(db.exists("a")) is True


@pytest.mark.parametrize("record_id", ["absent", ""])
async def test_a_missing_record_is_absent(db: Database, record_id: str) -> None:
    assert await settle(db.read(record_id)) is None
    assert await settle(db.get_version(record_id)) is None
    assert await settle(db.exists(record_id)) is False
    assert await settle(db.update(record_id, _record("absent"))) is False
    assert await settle(db.delete(record_id)) is False


async def test_a_conditional_write_to_a_missing_record_is_false(db: Database) -> None:
    assert await settle(db.update("absent", _record("absent"), expected_version="0:1")) is False
    assert await settle(db.delete("absent", expected_version="0:1")) is False


async def test_exists_is_a_bool(db: Database) -> None:
    assert await settle(db.exists("a")) is True


async def test_a_batch_reports_a_missing_record_as_false(db: Database) -> None:
    updates = [("a", _record("a", 2)), ("absent", _record("absent"))]
    assert await settle(db.update_batch(updates)) == [True, False]
    assert await settle(db.delete_batch(["absent", "a"])) == [False, True]


@pytest.mark.parametrize("name", ["create_batch", "upsert_batch"])
async def test_a_failed_item_is_named_after_the_rest_are_written(db: Database, name: str) -> None:
    """``n`` is mapped as a number by the first record, so text is refused."""
    records = [_record("ok"), _record("bad", "not a number"), _record("ok2")]
    with pytest.raises(DatabaseError, match="bad") as caught:
        await settle(getattr(db, name)(records))
    assert [failure["id"] for failure in caught.value.context["failures"]] == ["bad"]
    assert await settle(db.exists("ok")) and await settle(db.exists("ok2"))


async def test_a_failed_update_item_raises(db: Database) -> None:
    with pytest.raises(DatabaseError, match="'a'"):
        await settle(db.update_batch([("a", _record("a", "not a number"))]))


async def test_a_batch_collision_is_a_duplicate(db: Database) -> None:
    with pytest.raises(DuplicateRecordError):
        await settle(db.create_batch([_record("new"), _record("a")]))


async def test_a_failure_beside_a_collision_is_not_only_a_duplicate(db: Database) -> None:
    """A caller that skips duplicates would otherwise lose the failed write."""
    with pytest.raises(DatabaseError) as caught:
        await settle(db.create_batch([_record("a"), _record("bad", "not a number")]))
    assert [failure["id"] for failure in caught.value.context["failures"]] == ["a", "bad"]


async def test_a_batch_larger_than_one_request_keeps_its_order(db: Database) -> None:
    ids = [f"r{n}" for n in range(501)]
    assert await settle(db.upsert_batch([_record(i) for i in ids])) == ids


async def test_a_stream_counts_a_failed_record(db: Database) -> None:
    records = [_record("ok"), _record("bad", "not a number")]
    config = StreamConfig(on_conflict=ConflictPolicy.UPSERT, on_error=lambda _e, _r: True)
    if isinstance(db, SyncElasticsearchDatabase):
        result = db.stream_write(iter(records), config)
    else:

        async def source() -> AsyncIterator[Record]:
            for record in records:
                yield record

        result = await db.stream_write(source(), config)
    assert (result.successful, result.failed) == (1, 1)


async def test_connect_reports_an_index_it_could_not_create(es_config: dict[str, Any]) -> None:
    db = SyncElasticsearchDatabase({**es_config, "mappings": {"properties": {"n": {"type": "x"}}}})
    with pytest.raises(BadRequestError):
        db.connect()


@pytest.mark.parametrize("backend", [SyncElasticsearchDatabase, AsyncElasticsearchDatabase])
async def test_connect_accepts_an_index_another_process_just_created(
    backend: type[Database], es_config: dict[str, Any], admin: Elasticsearch
) -> None:
    """Between the existence check and the create, another process may win."""
    db = backend(es_config)
    db._index_exists = _absent_then_created(db, admin, es_config["index"])  # type: ignore[union-attr]
    await settle(db.connect())
    try:
        assert await settle(db.create(_record("a"))) == "a"
    finally:
        await settle(db.close())


def _absent_then_created(db: Database, admin: Elasticsearch, index: str) -> Callable[[], Any]:
    def check() -> bool:
        admin.indices.create(index=index)
        return False

    async def acheck() -> bool:
        return check()

    return check if isinstance(db, SyncElasticsearchDatabase) else acheck


async def test_nothing_is_printed(db: Database, capsys: pytest.CaptureFixture[str]) -> None:
    """Every sync request wrote its response, documents included, to stderr."""
    await settle(db.read("a"))
    await settle(db.search(Query(limit_value=5)))
    assert capsys.readouterr() == ("", "")


def test_a_closed_sync_database_refuses_and_reconnects(es_config: dict[str, Any]) -> None:
    db = SyncElasticsearchDatabase(es_config)
    db.connect()
    db.create(_record("a"))
    db.close()
    with pytest.raises(RuntimeError, match="not connected"):
        db.read("a")
    db.connect()
    assert db.read("a") is not None
    db.close()
