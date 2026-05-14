"""Pytest configuration for dataknobs_data tests."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add the package source to path for testing
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set the event loop policy for the test session."""
    return asyncio.DefaultEventLoopPolicy()


_TEST_INDEX_PREFIXES = ("test_records_", "test_factory_vectors_")


def _cleanup_leaked_elasticsearch_indices() -> None:
    """Backstop: delete any leaked ``test_*`` indices at session end.

    Per-test fixtures should clean up their own indices, but failed or
    incorrectly-written cleanup logic can leave behind orphans that
    eventually exhaust the cluster's per-node shard limit (default 1000)
    and cause silent index-creation failures (``'_id': None, 'result':
    'error'``). This sweep runs once per session as the last line of
    defense.
    """
    import os

    if os.environ.get("TEST_ELASTICSEARCH", "").lower() != "true":
        return

    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        return

    if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
        host = os.environ.get("ELASTICSEARCH_HOST", "elasticsearch")
    else:
        host = os.environ.get("ELASTICSEARCH_HOST", "localhost")
    port = int(os.environ.get("ELASTICSEARCH_PORT", "9200"))

    try:
        es = Elasticsearch([{"host": host, "port": port, "scheme": "http"}])
        if not es.ping():
            return
        for prefix in _TEST_INDEX_PREFIXES:
            try:
                es.indices.delete(index=f"{prefix}*", ignore_unavailable=True)
            except Exception:
                # Best-effort sweep — don't mask real test outcomes.
                pass
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):
    """Cleanup connection pools and leaked test indices at session end."""
    import asyncio

    async def cleanup():
        # Clean up any remaining pools
        try:
            from dataknobs_data.backends.elasticsearch_async import _client_manager as es_manager
            await es_manager.close_all()
        except Exception:
            pass  # Ignore cleanup errors

        try:
            from dataknobs_data.backends.s3_async import _session_manager as s3_manager
            await s3_manager.close_all()
        except Exception:
            pass  # Ignore cleanup errors

        try:
            from dataknobs_data.backends.postgres_native import _pool_manager as pg_manager
            await pg_manager.close_all()
        except Exception:
            pass  # Ignore cleanup errors

    # Run cleanup in a new event loop
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cleanup())
        loop.close()
    except Exception:
        pass  # Ignore cleanup errors

    _cleanup_leaked_elasticsearch_indices()
