"""Shared Elasticsearch pytest fixtures for dataknobs integration tests.

This module is a pytest11 plugin (registered in
``packages/common/pyproject.toml``) so any package depending on
``dataknobs-common`` automatically gets these fixtures via pytest's plugin
discovery — no explicit ``conftest.py`` imports required.

Consumers wrap :func:`make_elasticsearch_test_index` with a thin per-prefix
fixture to get a clean per-test index:

    @pytest.fixture
    def elasticsearch_test_index(make_elasticsearch_test_index):
        yield from make_elasticsearch_test_index("test_records_")

The ``requests`` and ``dataknobs_utils.elasticsearch_utils`` imports are
deferred to fixture-body execution so this module does not impose either
dep on consumers that don't run Elasticsearch tests.

Environment variables (read at fixture-creation time):

- ``ELASTICSEARCH_HOST`` (default: ``elasticsearch`` in Docker,
  ``localhost`` otherwise)
- ``ELASTICSEARCH_PORT`` (default: ``9200``)
- ``DOCKER_CONTAINER`` (any truthy value forces ``elasticsearch`` host default)
"""

from __future__ import annotations

import logging
import os
import socket
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)


def wait_for_elasticsearch(
    host: str,
    port: int,
    max_retries: int = 30,
) -> bool:
    """Wait for Elasticsearch to accept connections and report a ready cluster.

    First polls the TCP port; once the port is open, queries
    ``_cluster/health`` and accepts ``yellow`` or ``green`` cluster status
    as ready. Tolerates transient connection / timeout errors during startup.

    Args:
        host: Elasticsearch host.
        port: Elasticsearch port.
        max_retries: Maximum number of attempts (1-second sleep between).

    Returns:
        True once the cluster reports ready.

    Raises:
        ConnectionError: If the cluster is still not ready after ``max_retries``.
    """
    import requests
    from dataknobs_utils.requests_utils import RequestHelper

    for i in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex((host, port))
        finally:
            sock.close()

        if result == 0:
            helper = RequestHelper(host, port, timeout=5)
            try:
                response = helper.get("_cluster/health")
                if response.succeeded:
                    if response.json and "status" in response.json:
                        status = response.json["status"]
                        if status in ("yellow", "green"):
                            return True
                        logger.info(
                            "Elasticsearch cluster status is %s, waiting...",
                            status,
                        )
                    else:
                        return True
            except requests.exceptions.ConnectionError as exc:
                logger.info(
                    "Connection error to Elasticsearch at %s:%s: %s",
                    host,
                    port,
                    exc,
                )
            except requests.exceptions.Timeout as exc:
                logger.info(
                    "Timeout connecting to Elasticsearch at %s:%s: %s",
                    host,
                    port,
                    exc,
                )
        else:
            logger.info(
                "Port %s on %s is not open yet (attempt %s/%s)",
                port,
                host,
                i + 1,
                max_retries,
            )

        if i == max_retries - 1:
            raise ConnectionError(
                f"Could not connect to Elasticsearch at {host}:{port} "
                f"after {max_retries} attempts. Please ensure Elasticsearch "
                f"is running and accessible."
            )
        time.sleep(1)

    return False


#: Default staleness threshold (seconds) below which an index is considered
#: potentially in-flight and is NOT swept. Overridable per-call and via the
#: ``DK_ES_TEST_INDEX_MAX_AGE_SECONDS`` environment variable.
_DEFAULT_MIN_AGE_SECONDS = 300


def sweep_stale_test_indices(
    host: str,
    port: int,
    *,
    prefixes: tuple[str, ...] = ("test_",),
    min_age_seconds: int | None = None,
    now_ms: int | None = None,
) -> list[str]:
    """Delete stale dataknobs test indices to reclaim single-node shard budget.

    dataknobs Elasticsearch integration tests create uniquely-suffixed
    ``test_*`` indices that are deleted best-effort on fixture teardown. A run
    killed mid-test (Ctrl-C, crash, timeout) never reaches teardown, so it
    leaks its indices. Because the dev/CI cluster uses a persistent data
    volume, leaked indices accumulate across runs; each holds a shard, and a
    single-node cluster's ``cluster.max_shards_per_node`` ceiling (default
    1000) eventually rejects all new index creation — reddening the whole ES
    suite. This sweep runs once at session start to reclaim that residue.

    Mechanism: list indices matching each prefix via
    ``_cat/indices/<prefix>*`` (a *read* — the wildcard is not a destructive
    op, so ``action.destructive_requires_name`` does not apply), then delete,
    **by exact name**, each index whose ES ``creation.date`` (epoch millis) is
    older than the age threshold. Deletion is always one ``DELETE /<index>``
    per stale index; a wildcard ``DELETE`` is deliberately never issued
    because ES rejects it under ``destructive_requires_name``.

    Age-gating is the load-bearing safety property under pytest-xdist: session
    fixtures run once per worker, so a blanket "delete all matching" would race
    a concurrent worker's live index. An in-flight index is seconds old and is
    never swept; accumulated residue is minutes-to-months old and always is.
    The comparison uses a *local* ``now_ms`` (wall clock at sweep start)
    against the cluster's ``creation.date``; dev/CI ES runs in Docker on the
    same host as the test process, so host↔container clock skew is sub-second —
    negligible against the minutes-scale default threshold.

    Best-effort and non-fatal: any request / connection / parse error is
    logged at WARNING and the function returns the indices deleted so far (or
    an empty list). It never raises — a cleanup hiccup must not fail an
    otherwise-green run.

    Args:
        host: Elasticsearch host.
        port: Elasticsearch port.
        prefixes: Index-name prefixes to sweep. Each is matched as
            ``<prefix>*``. Defaults to the whole ``test_`` namespace; narrow it
            for a shared (non-dataknobs-dedicated) cluster.
        min_age_seconds: Minimum age, in seconds, for an index to be swept.
            Defaults to ``$DK_ES_TEST_INDEX_MAX_AGE_SECONDS`` if set, else 300.
        now_ms: Reference "now" in epoch millis (injectable for deterministic
            tests). Defaults to ``int(time.time() * 1000)`` at call time.

    Returns:
        The list of deleted index names (empty on any failure or if nothing
        was stale).
    """
    import requests
    from dataknobs_utils.requests_utils import RequestHelper

    if min_age_seconds is None:
        min_age_seconds = int(
            os.environ.get(
                "DK_ES_TEST_INDEX_MAX_AGE_SECONDS",
                str(_DEFAULT_MIN_AGE_SECONDS),
            )
        )
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - min_age_seconds * 1000

    helper = RequestHelper(host, port, timeout=5)
    # Comma-joined wildcard patterns: "_cat/indices/test_*,other_*".
    pattern = ",".join(f"{prefix}*" for prefix in prefixes)
    deleted: list[str] = []

    try:
        response = helper.get(
            f"_cat/indices/{pattern}",
            params={"h": "index,creation.date", "format": "json"},
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        logger.warning(
            "Could not list stale Elasticsearch test indices at %s:%s: %s",
            host,
            port,
            exc,
        )
        return deleted

    if not response.succeeded or not isinstance(response.json, list):
        # No matching indices (404 on an empty pattern) or an unexpected shape;
        # nothing to reclaim.
        return deleted

    for entry in response.json:
        name = entry.get("index")
        raw_created = entry.get("creation.date")
        if not name or raw_created is None:
            continue
        try:
            created_ms = int(raw_created)
        except (TypeError, ValueError):
            # Unparseable creation date — skip rather than guess its age.
            logger.warning(
                "Skipping Elasticsearch index %s: unparseable creation.date %r",
                name,
                raw_created,
            )
            continue
        if created_ms > cutoff_ms:
            # Younger than the threshold: possibly an in-flight index. Skip.
            continue
        try:
            del_response = helper.delete(name)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            logger.warning(
                "Failed to delete stale Elasticsearch index %s: %s", name, exc
            )
            continue
        if del_response.succeeded:
            deleted.append(name)
        else:
            logger.warning(
                "Delete of stale Elasticsearch index %s returned status %s",
                name,
                del_response.status,
            )

    if deleted:
        logger.info(
            "Swept %d stale Elasticsearch test index(es): %s",
            len(deleted),
            ", ".join(deleted),
        )
    return deleted


try:
    import pytest

    @pytest.fixture(scope="session")
    def elasticsearch_connection_params() -> dict[str, Any]:
        """Elasticsearch connection parameters for integration tests.

        Detects whether the test process is running inside a Docker
        container (presence of ``/.dockerenv`` or ``DOCKER_CONTAINER`` env
        var) and defaults the host to ``elasticsearch`` (the typical compose
        service name) in that case, ``localhost`` otherwise.
        """
        if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
            default_host = "elasticsearch"
        else:
            default_host = "localhost"

        return {
            "host": os.environ.get("ELASTICSEARCH_HOST", default_host),
            "port": int(os.environ.get("ELASTICSEARCH_PORT", "9200")),
        }

    @pytest.fixture(scope="session")
    def ensure_elasticsearch_ready(
        elasticsearch_connection_params: dict[str, Any],
    ) -> None:
        """Ensure Elasticsearch is reachable before integration tests run.

        After confirming the cluster is up, sweeps stale ``test_*`` indices
        left by prior runs killed mid-test — reclaiming single-node shard
        budget so accumulated residue can't exhaust it. The sweep is
        age-gated and best-effort (see :func:`sweep_stale_test_indices`), so
        it never touches an in-flight index nor fails an otherwise-green run.
        """
        wait_for_elasticsearch(
            host=elasticsearch_connection_params["host"],
            port=elasticsearch_connection_params["port"],
        )
        sweep_stale_test_indices(
            elasticsearch_connection_params["host"],
            elasticsearch_connection_params["port"],
        )

    @pytest.fixture
    def make_elasticsearch_test_index(
        ensure_elasticsearch_ready: None,
        elasticsearch_connection_params: dict[str, Any],
    ) -> Callable[[str], Iterator[dict[str, Any]]]:
        """Factory fixture for per-test Elasticsearch indices.

        Returns a callable ``factory(index_prefix)`` that yields a
        connection-config dict including a unique ``index`` name and deletes
        that index on teardown. Consumer fixtures use ``yield from`` to
        thread the cleanup through:

            @pytest.fixture
            def elasticsearch_test_index(make_elasticsearch_test_index):
                yield from make_elasticsearch_test_index("test_records_")

        The yielded config dict has the same shape as
        ``elasticsearch_connection_params`` plus:

        - ``index``: ``f"{index_prefix}{uuid8}"``
        - ``refresh``: ``True`` (immediate visibility for tests)

        Cleanup is best-effort: ``ConnectionError`` / ``ValueError`` during
        teardown are logged at WARNING and swallowed so a teardown failure
        doesn't mask the real test outcome. Other exceptions propagate so
        unexpected failures surface.

        Args:
            ensure_elasticsearch_ready: Session fixture ensuring the cluster
                is up.
            elasticsearch_connection_params: Session-scoped connection params.

        Returns:
            A callable that, given an ``index_prefix``, yields a config dict
            and tears down the index on completion.
        """

        def factory(index_prefix: str) -> Iterator[dict[str, Any]]:
            from dataknobs_utils.elasticsearch_utils import (
                SimplifiedElasticsearchIndex,
            )

            test_id = uuid.uuid4().hex[:8]
            config = elasticsearch_connection_params.copy()
            config["index"] = f"{index_prefix}{test_id}"
            config["refresh"] = True

            try:
                yield config
            finally:
                try:
                    es_index = SimplifiedElasticsearchIndex(
                        index_name=config["index"],
                        host=config["host"],
                        port=config["port"],
                    )
                    if es_index.exists():
                        es_index.delete()
                except (ConnectionError, ValueError) as exc:
                    logger.warning(
                        "Failed to clean up Elasticsearch index %s: %s",
                        config["index"],
                        exc,
                    )

        return factory

except ImportError:
    # pytest not installed — fixture decorators unavailable.
    # The wait_for_elasticsearch helper above remains usable.
    elasticsearch_connection_params = None  # type: ignore[assignment]
    ensure_elasticsearch_ready = None  # type: ignore[assignment]
    make_elasticsearch_test_index = None  # type: ignore[assignment]
