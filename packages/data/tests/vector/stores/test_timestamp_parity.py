"""Cross-backend parity tests for timestamp exposure (Item 36, Phase 7).

The Phase 2 abstraction exists so consumers can runtime-swap between
vector store backends without behavioral surprises. These tests
parameterize the same body over every shipping backend and assert
identical timestamp semantics:

- ``_created_at`` / ``_updated_at`` are present when
  ``include_timestamps=True`` and absent by default.
- Upsert preserves ``_created_at`` and advances ``_updated_at``.

Only **memory** and **pgvector** run today. FAISS and Chroma are
deferred (see Item 36 plan, §Deferred Follow-ups) — when they land,
each is added as a single ``pytest.param`` line; the test bodies do
not change.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio

from dataknobs_common.testing import requires_postgres
from dataknobs_data.vector.stores.memory import MemoryVectorStore

try:
    import asyncpg  # noqa: F401

    from dataknobs_data.vector.stores.pgvector import PgVectorStore

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False


# Gate pgvector parameterization on the same env flag as
# test_pgvector_store.py. ``requires_postgres`` only checks that the
# TCP port is open; ``TEST_POSTGRES=true`` is the canonical "postgres
# is fully provisioned (db + extension)" signal and is what
# ``bin/test.sh`` sets when it brings services up.
_pgvector_marks = [
    requires_postgres,
    pytest.mark.skipif(
        os.environ.get("TEST_POSTGRES", "").lower() != "true"
        or not ASYNCPG_AVAILABLE,
        reason="pgvector parity tests require TEST_POSTGRES=true and asyncpg",
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
    """Install the pgvector extension once per session.

    No-op when asyncpg or postgres is unavailable — the
    ``requires_postgres`` marker on the pgvector param will skip
    pgvector-parameterized tests cleanly.
    """
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
        # Postgres unavailable or extension install blocked; the
        # requires_postgres marker handles skipping.
        pass


@pytest.fixture
def pgvector_config(_ensure_pgvector_extension: None) -> dict[str, Any]:
    """Per-test pgvector config with a unique table name for isolation."""
    return {
        "connection_string": _get_test_connection_string(),
        "dimensions": 4,
        "metric": "cosine",
        "schema": "public",
        "table_name": f"test_parity_{uuid.uuid4().hex[:8]}",
        "auto_create_table": True,
        "id_type": "text",
    }


@pytest_asyncio.fixture(
    params=[
        pytest.param("memory", id="memory"),
        pytest.param("pgvector", id="pgvector", marks=_pgvector_marks),
    ]
)
async def any_vector_store(
    request: pytest.FixtureRequest, pgvector_config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Yield a freshly-initialized VectorStore for each backend param.

    FAISS and Chroma will be added to ``params`` when Phases 5 and 6
    of Item 36 land.
    """
    backend = request.param
    store: Any
    if backend == "memory":
        store = MemoryVectorStore({"dimensions": 4})
        await store.initialize()
        try:
            yield store
        finally:
            await store.close()
    elif backend == "pgvector":
        store = PgVectorStore(pgvector_config)
        await store.initialize()
        try:
            yield store
        finally:
            try:
                async with store._pool.acquire() as conn:
                    await conn.execute(
                        f"DROP TABLE IF EXISTS "
                        f"{store.schema}.{store.table_name}"
                    )
            except Exception:
                pass
            await store.close()
    else:
        pytest.fail(f"Unknown backend param: {backend}")


@pytest.mark.asyncio
async def test_timestamps_present_and_ordered(any_vector_store: Any) -> None:
    """include_timestamps=True exposes _created_at / _updated_at on every backend."""
    vec = np.random.rand(4).astype(np.float32)
    await any_vector_store.add_vectors(
        [vec], ids=["t1"], metadata=[{"k": "v"}]
    )

    results = await any_vector_store.get_vectors(
        ["t1"], include_timestamps=True
    )
    _, meta = results[0]

    assert meta is not None
    assert "_created_at" in meta
    assert "_updated_at" in meta
    assert meta["_created_at"] is not None
    assert meta["_updated_at"] is not None
    assert meta["k"] == "v"


@pytest.mark.asyncio
async def test_timestamps_absent_by_default(any_vector_store: Any) -> None:
    """Default get_vectors() omits timestamp keys on every backend."""
    vec = np.random.rand(4).astype(np.float32)
    await any_vector_store.add_vectors(
        [vec], ids=["t1"], metadata=[{"k": "v"}]
    )

    results = await any_vector_store.get_vectors(["t1"])
    _, meta = results[0]

    assert meta is not None
    assert "_created_at" not in meta
    assert "_updated_at" not in meta
    assert meta["k"] == "v"


@pytest.mark.asyncio
async def test_upsert_refreshes_updated_consistently(
    any_vector_store: Any,
) -> None:
    """Second add_vectors with same id: created preserved, updated advances."""
    vec1 = np.random.rand(4).astype(np.float32)
    vec2 = np.random.rand(4).astype(np.float32)

    await any_vector_store.add_vectors([vec1], ids=["t1"])
    first_results = await any_vector_store.get_vectors(
        ["t1"], include_timestamps=True
    )
    first = first_results[0][1]
    assert first is not None

    # Sleep longer than the backend clock resolution so updated_at
    # must strictly advance under ISO-string lexicographic comparison.
    await asyncio.sleep(0.05)

    await any_vector_store.add_vectors([vec2], ids=["t1"])
    second_results = await any_vector_store.get_vectors(
        ["t1"], include_timestamps=True
    )
    second = second_results[0][1]
    assert second is not None

    assert second["_created_at"] == first["_created_at"], (
        "created_at must not change on upsert "
        "(backend-dependent parity violation)"
    )
    assert second["_updated_at"] > first["_updated_at"], (
        "updated_at must advance on upsert"
    )
