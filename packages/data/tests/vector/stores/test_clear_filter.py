"""Item 118 Option B: ``VectorStore.clear(filter=...)`` cross-backend tests.

The unscoped ``clear()`` API has been a known limitation since
multi-tenant scoping shipped (Item 8); the consumer-reachable bug at
``KnowledgeIngestionManager.ingest()`` (which calls unscoped
``clear()`` while reloading a single domain in a shared store)
made it production-affecting.  These tests pin the post-fix
contract: filtered clear is the production-default for tenant-scoped
ingest, and unfiltered clear continues to wipe everything.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio

from dataknobs_common.testing import (
    is_chromadb_available,
    is_faiss_available,
    requires_postgres,
    safe_sql_ident,
)
from dataknobs_data.vector.stores.memory import MemoryVectorStore

if is_faiss_available():
    from dataknobs_data.vector.stores.faiss import FaissVectorStore

if is_chromadb_available():
    from dataknobs_data.vector.stores.chroma import ChromaVectorStore

try:
    import asyncpg  # noqa: F401

    from dataknobs_data.vector.stores.pgvector import PgVectorStore

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


logger = logging.getLogger(__name__)


_pgvector_marks = [
    requires_postgres,
    pytest.mark.skipif(
        os.environ.get("TEST_POSTGRES", "").lower() != "true"
        or not ASYNCPG_AVAILABLE,
        reason="pgvector tests require TEST_POSTGRES=true and asyncpg",
    ),
]


def _get_test_connection_string() -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    database = os.environ.get("POSTGRES_DB", "test_dataknobs")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


@pytest.fixture(scope="session")
def _ensure_pgvector_extension() -> None:
    if not ASYNCPG_AVAILABLE:
        return

    async def _setup() -> None:
        conn = await asyncpg.connect(_get_test_connection_string())
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await conn.close()

    try:
        asyncio.run(_setup())
    except (OSError, asyncpg.PostgresError):
        pass


@pytest.fixture
def pgvector_config(_ensure_pgvector_extension: None) -> dict[str, Any]:
    return {
        "connection_string": _get_test_connection_string(),
        "dimensions": 4,
        "metric": "cosine",
        "schema": "public",
        "table_name": f"test_clear_filter_{uuid.uuid4().hex[:8]}",
        "auto_create_table": True,
        "id_type": "text",
    }


async def _teardown_backend(backend: str, store: Any) -> None:
    """Drop the per-test collection/table created by a fixture."""
    if backend == "chroma":
        try:
            store.client.delete_collection(name=store.collection_name)
        except Exception as exc:
            logger.warning(
                "Chroma teardown failed for collection %r: %s",
                store.collection_name,
                exc,
            )
    elif backend == "pgvector":
        conn = None
        try:
            conn = await asyncpg.connect(_get_test_connection_string())
            await conn.execute(
                f"DROP TABLE IF EXISTS "
                f"{safe_sql_ident(store.schema)}.{safe_sql_ident(store.table_name)}"
            )
        except (OSError, asyncpg.PostgresError) as exc:
            logger.warning(
                "pgvector teardown failed for table %s.%s: %s",
                store.schema,
                store.table_name,
                exc,
            )
        finally:
            if conn is not None:
                await conn.close()


# Three vectors split across two tenants. Vector values are
# orthogonal unit vectors so any backend's similarity ordering is
# stable and well-defined.
SEED_IDS = ["a-1", "a-2", "b-1"]
SEED_METADATA = [
    {"tenant": "A"},
    {"tenant": "A"},
    {"tenant": "B"},
]


def _seed_vectors() -> np.ndarray:
    return np.eye(3, 4, dtype=np.float32)


@pytest_asyncio.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param(
            "faiss",
            id="faiss",
            marks=pytest.mark.skipif(
                not is_faiss_available(), reason="faiss not installed"
            ),
        ),
        pytest.param(
            "chroma",
            id="chroma",
            marks=pytest.mark.skipif(
                not is_chromadb_available(), reason="chromadb not installed"
            ),
        ),
        pytest.param("pgvector", id="pgvector", marks=_pgvector_marks),
    ]
)
async def any_vector_store(
    request: pytest.FixtureRequest, pgvector_config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Yield a freshly-seeded VectorStore for each backend param."""
    backend = request.param
    store: Any
    if backend == "memory":
        store = MemoryVectorStore({"dimensions": 4})
    elif backend == "faiss":
        store = FaissVectorStore({"dimensions": 4, "metric": "cosine"})
    elif backend == "chroma":
        store = ChromaVectorStore(
            {
                "dimensions": 4,
                "collection_name": f"test_clear_filter_{uuid.uuid4().hex[:8]}",
            }
        )
    elif backend == "pgvector":
        store = PgVectorStore(pgvector_config)
    else:
        pytest.fail(f"Unknown backend param: {backend}")

    await store.initialize()
    try:
        await store.add_vectors(
            _seed_vectors(), ids=list(SEED_IDS), metadata=list(SEED_METADATA)
        )
        yield store
    finally:
        await _teardown_backend(backend, store)
        await store.close()


@pytest.mark.asyncio
async def test_clear_with_filter_removes_only_matching(
    any_vector_store: Any,
) -> None:
    """clear(filter={"tenant": "A"}) leaves tenant-B vectors intact."""
    await any_vector_store.clear(filter={"tenant": "A"})

    # Tenant-A vectors removed.
    assert await any_vector_store.count(filter={"tenant": "A"}) == 0
    # Tenant-B vector survives.
    assert await any_vector_store.count(filter={"tenant": "B"}) == 1
    # Total reflects the partial wipe.
    assert await any_vector_store.count() == 1


@pytest.mark.asyncio
async def test_clear_without_filter_truncates(any_vector_store: Any) -> None:
    """Backward-compat: unfiltered clear still wipes everything."""
    await any_vector_store.clear()
    assert await any_vector_store.count() == 0


@pytest.mark.asyncio
async def test_clear_filter_no_match_is_noop(any_vector_store: Any) -> None:
    """A filter matching no records leaves the store untouched."""
    before = await any_vector_store.count()
    await any_vector_store.clear(filter={"tenant": "NONEXISTENT"})
    assert await any_vector_store.count() == before
