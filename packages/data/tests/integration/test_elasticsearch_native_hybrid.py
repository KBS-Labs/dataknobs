"""A native RRF hybrid search that Elasticsearch does not answer.

A node that cannot run one is asked for the two searches to fuse here instead.
Anything else raises: it was logged as "Native RRF not available" and the two
searches were sent whatever had gone wrong.

Whether the node under test supports native RRF depends on its version and
licence, so a proxy answers that one request. The answers in ``UNSUPPORTED``
are those Elasticsearch 8.11, 8.14, 8.16 and 8.19 gave. Every other request, and
the client, is real.
"""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest
import requests
from dataknobs_common.testing import requires_real_elasticsearch
from elasticsearch import ApiError, AuthorizationException, Elasticsearch

from dataknobs_data import Record
from dataknobs_data.backends.elasticsearch_async import AsyncElasticsearchDatabase
from dataknobs_data.exceptions import DatabaseError
from dataknobs_data.vector.hybrid import FusionStrategy, HybridSearchConfig, HybridSearchResult

pytestmark = [pytest.mark.integration, requires_real_elasticsearch]


def _error(status: int, error_type: str, reason: str, **more: Any) -> tuple[int, dict[str, Any]]:
    cause = {"type": error_type, "reason": reason, **more}
    return status, {"error": {"root_cause": [cause], **cause}, "status": status}


RRF = "Reciprocal Rank Fusion (RRF)"
UNSUPPORTED = {
    "before-retrievers": _error(
        400, "parsing_exception", "Unknown key for a START_OBJECT in [retriever]."
    ),
    "before-rank_window_size": _error(
        400,
        "x_content_parse_exception",
        "[1:237] [rrf] unknown field [rank_window_size] did you mean [window_size]?",
    ),
    "not-licensed": _error(
        403,
        "security_exception",
        f"current license is non-compliant for [{RRF}]",
        **{"license.expired.feature": RRF},
    ),
}
PARTIAL = (
    200,
    {
        "took": 1,
        "timed_out": False,
        "_shards": {
            "total": 2,
            "successful": 1,
            "skipped": 0,
            "failed": 1,
            "failures": [{"shard": 1, "reason": {"type": "no_shard_available_action_exception"}}],
        },
        "hits": {"hits": []},
    },
)


class Script:
    """What the proxy answers a native search with, and the searches it saw.

    ``answer`` is a status and body, or ``None`` to close the connection.
    """

    def __init__(self) -> None:
        self.answer: tuple[int, dict[str, Any]] | None = None
        self.searches: list[str] = []


@pytest.fixture
def es_config(make_elasticsearch_test_index: Any) -> Iterator[dict[str, Any]]:
    yield from make_elasticsearch_test_index("test_native_hybrid_")


@pytest.fixture
def script() -> Script:
    return Script()


@pytest.fixture
def proxied(es_config: dict[str, Any], script: Script) -> Iterator[dict[str, Any]]:
    upstream = f"http://{es_config['host']}:{es_config['port']}"

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def forward(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            native = b'"retriever"' in body
            if self.path.split("?")[0].endswith("/_search"):
                script.searches.append("native" if native else "other")
            if native and script.answer is None:
                self.close_connection = True
                return
            if native:
                status, content = script.answer[0], json.dumps(script.answer[1]).encode()
                content_type = self.headers["Accept"]
            else:
                kept = ("Content-Type", "Accept")
                sent = {name: self.headers[name] for name in kept if name in self.headers}
                got = requests.request(
                    self.command, upstream + self.path, data=body, headers=sent, timeout=30
                )
                status, content = got.status_code, got.content
                content_type = got.headers.get("Content-Type", "application/json")
            self.send_response(status)
            self.send_header("X-Elastic-Product", "Elasticsearch")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    for method in ("GET", "POST", "PUT", "DELETE", "HEAD"):
        setattr(Proxy, f"do_{method}", Proxy.forward)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield {**es_config, "host": "127.0.0.1", "port": server.server_port}
    server.shutdown()
    server.server_close()
    thread.join()


async def _filled(
    config: dict[str, Any], es_config: dict[str, Any]
) -> AsyncIterator[AsyncElasticsearchDatabase]:
    vector = {"type": "dense_vector", "dims": 3, "index": True, "similarity": "cosine"}
    with Elasticsearch(f"http://{es_config['host']}:{es_config['port']}") as admin:
        admin.indices.create(
            index=es_config["index"],
            mappings={"properties": {"data": {"properties": {"v": vector}}}},
        )
    database = AsyncElasticsearchDatabase(config)
    await database.connect()
    await database.create(Record(data={"content": "x", "v": [1.0, 0.0, 0.0]}, id="r"))
    yield database
    await database.close()


@pytest.fixture
async def db(
    proxied: dict[str, Any], es_config: dict[str, Any]
) -> AsyncIterator[AsyncElasticsearchDatabase]:
    async for database in _filled(proxied, es_config):
        yield database


@pytest.fixture
async def direct(es_config: dict[str, Any]) -> AsyncIterator[AsyncElasticsearchDatabase]:
    async for database in _filled(es_config, es_config):
        yield database


async def search(database: AsyncElasticsearchDatabase) -> list[HybridSearchResult]:
    return await database.hybrid_search(
        "x",
        np.array([1.0, 0.0, 0.0]),
        vector_field="v",
        k=2,
        config=HybridSearchConfig(fusion_strategy=FusionStrategy.NATIVE),
    )


async def test_the_node_under_test_is_answered(direct: AsyncElasticsearchDatabase) -> None:
    """Natively or not, by the node's own answer."""
    assert [result.record.storage_id for result in await search(direct)] == ["r"]


@pytest.mark.parametrize("name", UNSUPPORTED)
async def test_a_node_without_native_rrf_falls_back(
    db: AsyncElasticsearchDatabase, script: Script, name: str
) -> None:
    script.answer = UNSUPPORTED[name]
    assert [result.record.storage_id for result in await search(db)] == ["r"]
    assert script.searches == ["native", "other", "other"]


REFUSALS = {
    "permission": (
        _error(403, "security_exception", "action [indices:data/read/search] is unauthorized"),
        AuthorizationException,
        "unauthorized",
    ),
    "other-licence": (
        _error(
            403,
            "security_exception",
            "current license is non-compliant for [field and document level security]",
            **{"license.expired.feature": "field and document level security"},
        ),
        AuthorizationException,
        "document level security",
    ),
    "query": (
        _error(400, "parsing_exception", "unknown query [nope]"),
        ValueError,
        r"unknown query \[nope\]",
    ),
    "vector": (
        _error(400, "status_exception", "[rrf] search failed - retrievers '[knn]' returned errors"),
        ValueError,
        "returned errors",
    ),
    "window": (
        _error(
            400,
            "action_request_validation_exception",
            "Validation Failed: 1: [rrf] requires [rank_window_size: 1] be greater than or "
            "equal to [size: 2];",
        ),
        ValueError,
        "rank_window_size",
    ),
    "missing-index": (
        _error(404, "index_not_found_exception", "no such index [p]"),
        ValueError,
        "no such index",
    ),
    "server": (_error(500, "exception", "broke"), ApiError, "broke"),
    "dropped": (None, RuntimeError, "Failed to connect"),
    "partial": (PARTIAL, DatabaseError, "hybrid search is incomplete"),
}


@pytest.mark.parametrize("name", REFUSALS)
async def test_anything_else_raises(
    db: AsyncElasticsearchDatabase,
    script: Script,
    caplog: pytest.LogCaptureFixture,
    name: str,
) -> None:
    script.answer, error, match = REFUSALS[name]
    with pytest.raises(error, match=match) as caught:
        await search(db)
    assert set(script.searches) == {"native"}
    assert "not available" not in caplog.text
    if name == "partial":
        assert caught.value.context["failures"] == PARTIAL[1]["_shards"]["failures"]
