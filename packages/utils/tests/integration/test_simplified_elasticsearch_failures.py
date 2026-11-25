"""A request ``SimplifiedElasticsearchIndex`` sends and Elasticsearch refuses.

Each method used to answer a refusal with its empty result: ``search`` with no
hits, ``count`` with 0, ``get`` with ``None`` and the writes with ``False``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
import requests
from dataknobs_common.testing import requires_real_elasticsearch

from dataknobs_utils.elasticsearch_utils import (
    ElasticsearchConflictError,
    ElasticsearchRequestError,
    SimplifiedElasticsearchIndex,
)

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]

REJECTED = {"query": {"terms": {"colour": [None]}}}


@pytest.fixture
def index(make_elasticsearch_test_index: Any) -> Iterator[SimplifiedElasticsearchIndex]:
    for config in make_elasticsearch_test_index("test_simplified_"):
        wrapper = SimplifiedElasticsearchIndex(config["index"], config["host"], config["port"])
        assert wrapper.create() is True
        wrapper.index({"n": 1}, doc_id="a", refresh=True)
        yield wrapper


def _close(index: SimplifiedElasticsearchIndex) -> None:
    index.request_helper.request("post", f"{index.index_name}/_close", verbose=False)


OPERATIONS: dict[str, Callable[[SimplifiedElasticsearchIndex], Any]] = {
    "count": lambda i: i.count(),
    "get": lambda i: i.get("a"),
    "exists": lambda i: i.exists("a"),
    "index": lambda i: i.index({"n": 2}, doc_id="a"),
    "update": lambda i: i.update("a", {"doc": {"n": 2}}),
    "delete": lambda i: i.delete("a"),
    "delete_by_query": lambda i: i.delete_by_query({"query": {"match_all": {}}}),
    "refresh": lambda i: i.refresh(),
}


@pytest.mark.parametrize("name", OPERATIONS)
def test_a_refused_request_raises(index: SimplifiedElasticsearchIndex, name: str) -> None:
    _close(index)
    with pytest.raises(ElasticsearchRequestError) as caught:
        OPERATIONS[name](index)
    assert caught.value.status == 400
    if name != "exists":  # a HEAD has no body to give a reason in
        assert "index_closed_exception" in str(caught.value)


@pytest.mark.parametrize("name", ["count", "get", "delete", "delete_by_query", "refresh"])
def test_a_missing_index_is_not_a_missing_document(
    index: SimplifiedElasticsearchIndex, name: str
) -> None:
    assert index.delete() is True
    with pytest.raises(ElasticsearchRequestError, match="index_not_found_exception"):
        OPERATIONS[name](index)


def test_a_rejected_search_returns_the_error(index: SimplifiedElasticsearchIndex) -> None:
    response = index.search(REJECTED)
    assert not response.succeeded
    assert response.status == 400
    assert "error" in response.result and "hits" not in response.result
    with pytest.raises(ElasticsearchRequestError):
        index.count(REJECTED)


def test_what_is_still_an_answer(index: SimplifiedElasticsearchIndex) -> None:
    assert index.get("absent") is None
    assert index.exists("absent") is False
    assert index.update("absent", {"doc": {"n": 1}}) is False
    assert index.delete("absent") is False
    assert index.create() is False
    assert index.delete() is True
    assert index.delete() is False
    assert index.exists() is False


def test_a_create_conflict_is_still_its_own_error(index: SimplifiedElasticsearchIndex) -> None:
    with pytest.raises(ElasticsearchConflictError):
        index.index({"n": 2}, doc_id="a", op_type="create")


def test_a_negative_retry_count_is_refused(index: SimplifiedElasticsearchIndex) -> None:
    with pytest.raises(ValueError, match="max_retries"):
        index.index({"n": 2}, doc_id="a", max_retries=-1)


def test_an_index_that_cannot_be_created_raises(index: SimplifiedElasticsearchIndex) -> None:
    refused = SimplifiedElasticsearchIndex(
        index.index_name + "_x",
        index.host,
        index.port,
        mappings={"properties": {"n": {"type": "no_such_type"}}},
    )
    with pytest.raises(ElasticsearchRequestError, match="mapper_parsing_exception"):
        refused.create()


def test_no_server_is_not_a_missing_index() -> None:
    """A HEAD that never got an answer used to read as ``False``."""
    with pytest.raises(ElasticsearchRequestError, match="no response"):
        SimplifiedElasticsearchIndex("test_simplified_none", "localhost", 1).exists()


def test_nothing_is_printed(
    index: SimplifiedElasticsearchIndex, capsys: pytest.CaptureFixture[str]
) -> None:
    index.get("a")
    index.search()
    index.exists()
    assert capsys.readouterr() == ("", "")


@pytest.fixture
def short(make_elasticsearch_test_index: Any) -> Iterator[SimplifiedElasticsearchIndex]:
    """An index with one of its two shards unavailable."""
    for config in make_elasticsearch_test_index("test_simplified_short_"):
        base = f"http://{config['host']}:{config['port']}/{config['index']}"
        settings = {
            "number_of_shards": 2,
            "number_of_replicas": 0,
            "index.routing.allocation.total_shards_per_node": 1,
        }
        created = requests.put(
            f"{base}?wait_for_active_shards=0", json={"settings": settings}, timeout=10
        )
        created.raise_for_status()
        health = base.replace(config["index"], f"_cluster/health/{config['index']}")
        requests.get(f"{health}?wait_for_active_shards=1&timeout=5s", timeout=10)
        wrapper = SimplifiedElasticsearchIndex(config["index"], config["host"], config["port"])
        yield wrapper


def test_a_count_missing_a_shard_raises(short: SimplifiedElasticsearchIndex) -> None:
    with pytest.raises(ElasticsearchRequestError, match=r"1 of 2 shards.*no_shard_available"):
        short.count()


def test_a_search_missing_a_shard_says_so(short: SimplifiedElasticsearchIndex) -> None:
    response = short.search()
    assert response.succeeded
    assert response.result["_shards"]["failed"] == 1
