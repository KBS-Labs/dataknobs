"""Pytest configuration for dataknobs_data tests."""

import asyncio
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

# Add the package source to path for testing
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
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


# The module-level connection-pool managers closed once at session end, as
# ``(module path, attribute name)`` pairs.
#
# Held as data rather than as one try/import/close block per backend. There
# were three such blocks, and one named a module that does not exist — which
# no run could distinguish from the two that worked, because each swallowed
# its own ImportError as though the backend's optional driver were merely
# absent. ``test_session_pool_cleanup.py`` checks that every entry here still
# resolves.
_POOL_MANAGERS: tuple[tuple[str, str], ...] = (
    ("dataknobs_data.backends.elasticsearch_async", "_client_manager"),
    ("dataknobs_data.backends.s3_async", "_session_manager"),
    ("dataknobs_data.backends.postgres", "_pool_manager"),
)


def _load_pool_manager(module_path: str, attribute: str) -> Any | None:
    """Return the named pool manager, or ``None`` if its driver is not installed.

    Each backend module imports an optional third-party driver (``asyncpg``,
    ``aioboto3``, ``elasticsearch``), so an absent driver is an ordinary skip.
    An absent *backend module* is not: it means this table names something that
    does not exist, and treating the two alike is how a cleanup target sits
    dead while reporting nothing. They are told apart by which module the
    ``ModuleNotFoundError`` actually names — the one asked for (or a parent of
    it) is a wrong path; anything else is a driver the backend imported.

    A renamed attribute raises ``AttributeError`` for the same reason.
    """
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        missing = exc.name
        if missing is None or missing == module_path or module_path.startswith(f"{missing}."):
            raise
        return None
    return getattr(module, attribute)


async def _close_pool_managers() -> None:
    """Close every registered pool manager; one failure does not stop the rest."""
    for module_path, attribute in _POOL_MANAGERS:
        try:
            manager = _load_pool_manager(module_path, attribute)
        except (AttributeError, ModuleNotFoundError):
            logger.exception(
                "Session cleanup cannot reach %s.%s, so the pools it holds were not closed",
                module_path,
                attribute,
            )
            continue
        if manager is None:
            continue
        try:
            await manager.close_all()
        except Exception:
            logger.exception("Failed to close the pools held by %s.%s", module_path, attribute)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Cleanup connection pools and leaked test indices at session end."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_close_pool_managers())
    finally:
        loop.close()

    _cleanup_leaked_elasticsearch_indices()
