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
        """Ensure Elasticsearch is reachable before integration tests run."""
        wait_for_elasticsearch(
            host=elasticsearch_connection_params["host"],
            port=elasticsearch_connection_params["port"],
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
