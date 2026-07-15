"""Behavioral tests for ``sweep_stale_test_indices`` against a live cluster.

The sweep reclaims single-node shard budget by deleting stale ``test_*``
indices left behind by runs killed mid-test. These tests exercise the real
helper against the dev Elasticsearch service — no mocks (Elasticsearch is the
documented external-dependency exception) — and are skipped via
``@requires_elasticsearch`` when the cluster is unreachable.

Each test isolates its blast radius with a **unique run-scoped sub-prefix**
(``test_sweep_<uuid8>_``) passed as ``prefixes=`` to the sweep, so it never
touches other suites' ``test_*`` residue and other suites never touch its
fixtures.

The one exception is :func:`test_sweep_unreachable_host_is_non_fatal`, which
targets a closed port and therefore needs no live cluster — it runs
unconditionally.
"""

from __future__ import annotations

import contextlib
import http.server
import logging
import threading
import time
import uuid
from collections.abc import Iterator

import pytest

from dataknobs_common.testing import (
    requires_elasticsearch,
    sweep_stale_test_indices,
)


def _run_prefix() -> str:
    """A collision-free per-test sub-prefix within the ``test_`` namespace."""
    return f"test_sweep_{uuid.uuid4().hex[:8]}_"


def _create_index(host: str, port: int, name: str) -> None:
    from dataknobs_utils.requests_utils import RequestHelper

    helper = RequestHelper(host, port, timeout=5)
    resp = helper.put(name)
    assert resp.succeeded, f"index create failed for {name}: status {resp.status}"


def _index_exists(host: str, port: int, name: str) -> bool:
    from dataknobs_utils.requests_utils import RequestHelper

    helper = RequestHelper(host, port, timeout=5)
    return helper.head(name).status == 200


def _creation_ms(host: str, port: int, name: str) -> int:
    from dataknobs_utils.requests_utils import RequestHelper

    helper = RequestHelper(host, port, timeout=5)
    resp = helper.get(
        f"_cat/indices/{name}",
        params={"h": "index,creation.date", "format": "json"},
    )
    assert resp.succeeded and isinstance(resp.json, list) and resp.json
    return int(resp.json[0]["creation.date"])


def _es_now_ms(host: str, port: int, names: list[str]) -> int:
    """A ``now_ms`` derived from ES's own clock, safely past every named index.

    ``creation.date`` is stamped by the ES container's clock; the sweep's
    default ``now_ms`` uses the host's clock. For a "sweep everything"
    assertion on *just-created* indices the two-clock comparison has zero
    margin, so a few tens of ms of host↔container skew would spuriously spare
    the newest index. Deriving ``now_ms`` from the same clock the timestamps
    came from removes skew from the test entirely. (Production is immune by a
    different mechanism: the 300s default threshold dwarfs sub-second skew.)
    """
    return max(_creation_ms(host, port, name) for name in names) + 10_000


@pytest.fixture
def es_host_port(elasticsearch_connection_params) -> tuple[str, int]:
    return (
        elasticsearch_connection_params["host"],
        elasticsearch_connection_params["port"],
    )


@requires_elasticsearch
def test_sweep_reclaims_stale_indices(es_host_port, ensure_elasticsearch_ready):
    """Core reclamation: stale indices under the swept prefix are deleted."""
    host, port = es_host_port
    prefix = _run_prefix()
    names = [f"{prefix}{i}" for i in range(3)]
    for name in names:
        _create_index(host, port, name)

    # now_ms just past every index (ES clock) => all count as stale.
    deleted = sweep_stale_test_indices(
        host,
        port,
        prefixes=(prefix,),
        min_age_seconds=0,
        now_ms=_es_now_ms(host, port, names),
    )

    assert set(deleted) == set(names)
    for name in names:
        assert not _index_exists(host, port, name)


@requires_elasticsearch
def test_sweep_age_gating_protects_fresh_index(
    es_host_port, ensure_elasticsearch_ready
):
    """Load-bearing safety property: an index younger than the threshold survives.

    The sweep deletes an index only when ``created_ms <= now_ms -
    min_age_seconds * 1000``. With ``now_ms`` injected as ``old_ms + 1000`` and
    ``min_age_seconds=1``, the cutoff lands exactly on the older index's
    creation instant: the old index is eligible, the strictly-newer fresh
    index is not. A brief real gap between the two creates guarantees distinct
    millisecond timestamps (``creation.date`` is ms-precision), so the boundary
    is well-defined; the cutoff itself is deterministic via ``now_ms`` — no
    dependence on the wall clock at sweep time.
    """
    host, port = es_host_port
    prefix = _run_prefix()
    old_name = f"{prefix}old"
    fresh_name = f"{prefix}fresh"

    _create_index(host, port, old_name)
    time.sleep(0.05)  # ensure fresh_ms > old_ms at ms precision
    _create_index(host, port, fresh_name)

    old_ms = _creation_ms(host, port, old_name)
    fresh_ms = _creation_ms(host, port, fresh_name)
    assert fresh_ms > old_ms, "fresh index must be strictly newer than old"

    # cutoff = now_ms - 1000 = old_ms: old is eligible, fresh is not.
    now_ms = old_ms + 1000

    deleted = sweep_stale_test_indices(
        host, port, prefixes=(prefix,), min_age_seconds=1, now_ms=now_ms
    )

    assert old_name in deleted
    assert fresh_name not in deleted
    assert not _index_exists(host, port, old_name)
    assert _index_exists(host, port, fresh_name)

    # Cleanup the survivor.
    sweep_stale_test_indices(
        host,
        port,
        prefixes=(prefix,),
        min_age_seconds=0,
        now_ms=_es_now_ms(host, port, [fresh_name]),
    )


@requires_elasticsearch
def test_sweep_deletes_by_exact_name(es_host_port, ensure_elasticsearch_ready):
    """Exact-name deletion guard.

    A wildcard ``DELETE`` is rejected by the cluster under
    ``action.destructive_requires_name`` (ES default). That both indices are
    individually removed proves the sweep deletes by exact name rather than by
    pattern. A direct wildcard-DELETE probe documents why exact-name is
    required.
    """
    from dataknobs_utils.requests_utils import RequestHelper

    host, port = es_host_port
    prefix = _run_prefix()
    names = [f"{prefix}a", f"{prefix}b"]
    for name in names:
        _create_index(host, port, name)

    # Document the constraint: a wildcard DELETE must NOT succeed.
    helper = RequestHelper(host, port, timeout=5)
    wildcard = helper.delete(f"{prefix}*")
    assert not wildcard.succeeded, (
        "wildcard DELETE unexpectedly succeeded; destructive_requires_name "
        "is not enforced, so the sweep's exact-name path is under-tested"
    )
    # Both still present after the rejected wildcard delete.
    for name in names:
        assert _index_exists(host, port, name)

    deleted = sweep_stale_test_indices(
        host,
        port,
        prefixes=(prefix,),
        min_age_seconds=0,
        now_ms=_es_now_ms(host, port, names),
    )
    assert set(deleted) == set(names)
    for name in names:
        assert not _index_exists(host, port, name)


@requires_elasticsearch
def test_sweep_scopes_to_given_prefixes(es_host_port, ensure_elasticsearch_ready):
    """A prefix not passed to the sweep is left untouched."""
    host, port = es_host_port
    swept_prefix = _run_prefix()
    kept_prefix = _run_prefix()
    swept = f"{swept_prefix}target"
    kept = f"{kept_prefix}bystander"
    _create_index(host, port, swept)
    _create_index(host, port, kept)

    try:
        deleted = sweep_stale_test_indices(
            host,
            port,
            prefixes=(swept_prefix,),
            min_age_seconds=0,
            now_ms=_es_now_ms(host, port, [swept]),
        )
        assert deleted == [swept]
        assert not _index_exists(host, port, swept)
        assert _index_exists(host, port, kept)
    finally:
        sweep_stale_test_indices(
            host,
            port,
            prefixes=(kept_prefix,),
            min_age_seconds=0,
            now_ms=_es_now_ms(host, port, [kept]),
        )


_ES_FIX_LOGGER = "dataknobs_common.testing.elasticsearch_fixtures"


def test_sweep_unreachable_host_is_non_fatal(caplog):
    """Best-effort: an unreachable cluster yields ``[]`` and logs, never raises.

    Needs no live ES — targets a closed port — so it runs unconditionally.
    """
    with caplog.at_level(logging.WARNING, logger=_ES_FIX_LOGGER):
        result = sweep_stale_test_indices(
            "127.0.0.1", 1, prefixes=("test_sweep_nope_",), min_age_seconds=0
        )
    assert result == []
    assert any(
        "Could not list stale Elasticsearch test indices" in rec.getMessage()
        for rec in caplog.records
    )


@contextlib.contextmanager
def _canned_http_server(status: int, body: bytes) -> Iterator[tuple[str, int]]:
    """A real local HTTP server returning a fixed status + body to any request.

    Not a mock of any dataknobs interface — an actual ``http.server`` on an
    ephemeral port — so the sweep exercises its genuine request/response and
    JSON-parsing code path against a controllable, ES-independent endpoint.
    Yields ``(host, port)``.
    """

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _respond(self) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # Names mandated by BaseHTTPRequestHandler's do_<METHOD> dispatch.
        do_GET = _respond  # noqa: N815
        do_DELETE = _respond  # noqa: N815

        def log_message(self, *_args: object) -> None:  # silence stderr noise
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        yield host, port
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_sweep_non_json_response_is_non_fatal(caplog):
    """A non-JSON response body (e.g. a proxy error page) is swallowed, not raised.

    The response handler calls ``json.loads`` on the body, raising
    ``json.JSONDecodeError`` (a ``ValueError``). The sweep must catch it like
    any other failure — otherwise it aborts the session-scoped fixture. Uses a
    real local server, no ES.
    """
    with (
        _canned_http_server(200, b"<html>502 Bad Gateway</html>") as (host, port),
        caplog.at_level(logging.WARNING, logger=_ES_FIX_LOGGER),
    ):
        result = sweep_stale_test_indices(
            host, port, prefixes=("test_sweep_nonjson_",), min_age_seconds=0
        )
    assert result == []
    assert any(
        "Could not list stale Elasticsearch test indices" in rec.getMessage()
        for rec in caplog.records
    )


def test_sweep_non_2xx_list_logs_status_and_returns_empty(caplog):
    """A non-2xx list response is distinguished from 'nothing to sweep' via a log.

    A valid-JSON body with a 503 status must not be mistaken for an empty
    match: the sweep returns ``[]`` and logs the status so a real server-side
    rejection is diagnosable. Uses a real local server, no ES.
    """
    with (
        _canned_http_server(503, b"{}") as (host, port),
        caplog.at_level(logging.WARNING, logger=_ES_FIX_LOGGER),
    ):
        result = sweep_stale_test_indices(
            host, port, prefixes=("test_sweep_503_",), min_age_seconds=0
        )
    assert result == []
    assert any(
        "returned status 503" in rec.getMessage() for rec in caplog.records
    )


def test_sweep_malformed_max_age_env_is_non_fatal(caplog, monkeypatch):
    """A malformed ``DK_ES_TEST_INDEX_MAX_AGE_SECONDS`` falls back, never raises.

    The env-var parse runs before any request, so a bad value would abort the
    session fixture before it even connects. Points at a closed port so the
    fallback path completes without ES; asserts the malformed value is logged
    and the sweep returns ``[]``.
    """
    monkeypatch.setenv("DK_ES_TEST_INDEX_MAX_AGE_SECONDS", "not-a-number")
    with caplog.at_level(logging.WARNING, logger=_ES_FIX_LOGGER):
        # min_age_seconds omitted => the env var is consulted (and is malformed).
        result = sweep_stale_test_indices(
            "127.0.0.1", 1, prefixes=("test_sweep_badenv_",)
        )
    assert result == []
    assert any(
        "malformed DK_ES_TEST_INDEX_MAX_AGE_SECONDS" in rec.getMessage()
        for rec in caplog.records
    )
