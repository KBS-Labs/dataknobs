"""elasticsearch_index_fingerprint changes on every write to an index, and on no read.

The provisioning count and the shared corpus both decide "did this test write?"
by comparing two fingerprints, so the fingerprint has to see every way a test
can change an index: documents written or deleted, settings, mappings, aliases,
and the index being closed. A search, a count or a refresh must leave it alone,
or every read-only test would look like a writer.

Against the real dev node; no fake can say which counters a real node moves.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from dataknobs_common.testing import elasticsearch_index_fingerprint, requires_elasticsearch

pytestmark = requires_elasticsearch


@pytest.fixture
def index(elasticsearch_connection_params: dict[str, Any]) -> Iterator[str]:
    """A fresh index with one document, removed afterwards."""
    name = f"test_fingerprint_{uuid.uuid4().hex[:8]}"
    helper = _helper(elasticsearch_connection_params)
    assert helper.put(name, payload=json.dumps({"settings": {"number_of_replicas": 0}})).succeeded
    assert helper.put(
        f"{name}/_doc/a", payload=json.dumps({"n": 1}), params={"refresh": "true"}
    ).succeeded
    yield name
    helper.delete(name)


def _helper(params: dict[str, Any]) -> Any:
    from dataknobs_utils.requests_utils import HEADERS, RequestHelper

    # A str body is sent raw; a dict would be form-encoded, which the node refuses.
    return RequestHelper(params["host"], params["port"], timeout=5, headers=HEADERS)


def _fingerprint(params: dict[str, Any], name: str) -> Any:
    return elasticsearch_index_fingerprint(params["host"], params["port"], name)


_WRITES: dict[str, Callable[[Any, str], Any]] = {
    "index-a-document": lambda h, n: h.put(f"{n}/_doc/b", payload=json.dumps({"n": 2})),
    "index-the-same-document-again": lambda h, n: h.put(
        f"{n}/_doc/a", payload=json.dumps({"n": 1})
    ),
    "delete-a-document": lambda h, n: h.delete(f"{n}/_doc/a"),
    "delete-by-query": lambda h, n: h.post(
        f"{n}/_delete_by_query", payload=json.dumps({"query": {"match_all": {}}})
    ),
    "block-writes": lambda h, n: h.put(
        f"{n}/_settings", payload=json.dumps({"index.blocks.write": True})
    ),
    "close": lambda h, n: h.post(f"{n}/_close"),
    "put-mapping": lambda h, n: h.put(
        f"{n}/_mapping", payload=json.dumps({"properties": {"m": {"type": "long"}}})
    ),
    "put-alias": lambda h, n: h.put(f"{n}/_alias/{n}_alias"),
    "delete-the-index": lambda h, n: h.delete(n),
}


@pytest.mark.parametrize("write", list(_WRITES.values()), ids=list(_WRITES))
def test_a_write_changes_the_fingerprint(
    elasticsearch_connection_params: dict[str, Any],
    index: str,
    write: Callable[[Any, str], Any],
) -> None:
    before = _fingerprint(elasticsearch_connection_params, index)
    response = write(_helper(elasticsearch_connection_params), index)
    assert response.succeeded, response.json
    assert _fingerprint(elasticsearch_connection_params, index) != before


_READS: dict[str, Callable[[Any, str], Any]] = {
    "search": lambda h, n: h.post(f"{n}/_search", payload=json.dumps({"query": {"match_all": {}}})),
    "count": lambda h, n: h.get(f"{n}/_count"),
    "get-a-document": lambda h, n: h.get(f"{n}/_doc/a"),
    "refresh": lambda h, n: h.post(f"{n}/_refresh"),
    "read-the-mapping": lambda h, n: h.get(f"{n}/_mapping"),
}


@pytest.mark.parametrize("read", list(_READS.values()), ids=list(_READS))
def test_a_read_leaves_the_fingerprint_alone(
    elasticsearch_connection_params: dict[str, Any],
    index: str,
    read: Callable[[Any, str], Any],
) -> None:
    before = _fingerprint(elasticsearch_connection_params, index)
    assert read(_helper(elasticsearch_connection_params), index).succeeded
    assert _fingerprint(elasticsearch_connection_params, index) == before


def test_an_absent_index_has_a_fingerprint_that_says_so(
    elasticsearch_connection_params: dict[str, Any],
) -> None:
    fingerprint = _fingerprint(elasticsearch_connection_params, f"test_absent_{uuid.uuid4().hex}")
    assert fingerprint.exists is False


def test_creating_an_index_changes_its_fingerprint(
    elasticsearch_connection_params: dict[str, Any],
) -> None:
    name = f"test_fingerprint_{uuid.uuid4().hex[:8]}"
    before = _fingerprint(elasticsearch_connection_params, name)
    helper = _helper(elasticsearch_connection_params)
    try:
        assert helper.put(name).succeeded
        assert _fingerprint(elasticsearch_connection_params, name) != before
    finally:
        helper.delete(name)


def test_a_node_that_cannot_answer_raises() -> None:
    """A fingerprint that could not be taken is not one that matched."""
    with pytest.raises(ConnectionError):
        elasticsearch_index_fingerprint("127.0.0.1", 1, "test_anything")
