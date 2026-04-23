"""Cross-backend metadata-filter semantics tests (Item 98).

The four-quadrant semantics are:

* ``scalar`` filter, ``scalar`` metadata — equality.
* ``scalar`` filter, ``list`` metadata — contains (the scalar appears in
  the stored list).
* ``list`` filter, ``scalar`` metadata — IN (the scalar is one of the
  filter elements).
* ``list`` filter, ``list`` metadata — non-empty intersection.

A missing metadata key fails the filter; an empty filter matches every
record. All filter keys must match (AND across keys). Behavior is the
same on every shipping backend (``MemoryVectorStore``,
``FaissVectorStore``, ``ChromaVectorStore``, ``PgVectorStore``).

PgVector additionally must preserve metadata types (booleans stay
booleans, numbers stay numbers) — covered by a small set of
pgvector-only cases.
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
        "table_name": f"test_filter_{uuid.uuid4().hex[:8]}",
        "auto_create_table": True,
        "id_type": "text",
    }


async def _teardown_backend(backend: str, store: Any) -> None:
    """Drop the per-test collection/table created by a fixture.

    Uses out-of-band connections so we don't depend on private store
    attributes (notably ``PgVectorStore._pool``). Failures are logged
    rather than swallowed so orphaned test collections/tables become
    visible in pytest output.
    """
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
                f"DROP TABLE IF EXISTS {store.schema}.{store.table_name}"
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


# Five seed records exercising every metadata shape used by the
# four-quadrant matrix below.
SEED_IDS = ["A", "B", "C", "D", "E"]
SEED_METADATA: list[dict[str, Any]] = [
    {"type": "tension", "tags": ["urgent", "blocker"]},
    {"type": "gap", "tags": ["urgent"]},
    {"type": "tension", "tags": ["later"]},
    {"type": "gap", "tags": []},
    {"type": "terminology"},  # no "tags" key
]


def _seed_vectors() -> np.ndarray:
    """Five 4-d unit vectors (rows of identity, padded). Deterministic."""
    return np.eye(5, 4, dtype=np.float32)


def _query_vector() -> np.ndarray:
    """Constant query vector — distance ordering not asserted."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


# (filter, expected ids) — applied to all four backends.
FOUR_QUADRANT_CASES: list[tuple[dict[str, Any], set[str]]] = [
    ({"type": "tension"}, {"A", "C"}),  # scalar/scalar EQ
    ({"tags": "urgent"}, {"A", "B"}),  # scalar/list contains (NEW)
    ({"type": ["tension", "gap"]}, {"A", "B", "C", "D"}),  # list/scalar IN
    ({"tags": ["urgent", "later"]}, {"A", "B", "C"}),  # list/list intersect
    ({"tags": "missing"}, set()),  # scalar not in list
    ({"type": "tension", "tags": "urgent"}, {"A"}),  # AND across keys
    ({"missing_key": "value"}, set()),  # missing key fails
    ({"tags": []}, set()),  # empty-list filter never satisfied
]

CASE_IDS = [f"case{i + 1}" for i in range(len(FOUR_QUADRANT_CASES))]


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
                "collection_name": f"test_filter_{uuid.uuid4().hex[:8]}",
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
@pytest.mark.parametrize(
    "filter_dict,expected", FOUR_QUADRANT_CASES, ids=CASE_IDS
)
async def test_search_filter_quadrants(
    any_vector_store: Any,
    filter_dict: dict[str, Any],
    expected: set[str],
) -> None:
    """search() returns the four-quadrant-correct id set for each filter."""
    results = await any_vector_store.search(
        _query_vector(), k=10, filter=filter_dict
    )
    assert {r[0] for r in results} == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filter_dict,expected", FOUR_QUADRANT_CASES, ids=CASE_IDS
)
async def test_count_filter_quadrants(
    any_vector_store: Any,
    filter_dict: dict[str, Any],
    expected: set[str],
) -> None:
    """count(filter=...) matches the search result-set size on every backend."""
    n = await any_vector_store.count(filter=filter_dict)
    assert n == len(expected)


# ---------------------------------------------------------------------------
# PgVector type-safety cases (booleans/numerics stop silently returning empty)
#
# These also run under Memory/FAISS/Chroma as a sanity check — Python ``==``
# between like-typed values matches trivially on those backends, and the
# previously broken pgvector text-cast path now matches them too.
# ---------------------------------------------------------------------------


TYPE_SAFETY_IDS = ["X1", "X2"]
TYPE_SAFETY_METADATA: list[dict[str, Any]] = [
    {"active": True, "count": 5},
    {"active": False, "count": 7},
]


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
async def type_safety_store(
    request: pytest.FixtureRequest, pgvector_config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Two-record store for type-roundtrip cases."""
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
                "collection_name": f"test_typesafe_{uuid.uuid4().hex[:8]}",
            }
        )
    elif backend == "pgvector":
        store = PgVectorStore(pgvector_config)
    else:
        pytest.fail(f"Unknown backend param: {backend}")

    await store.initialize()
    try:
        vectors = np.eye(2, 4, dtype=np.float32)
        await store.add_vectors(
            vectors,
            ids=list(TYPE_SAFETY_IDS),
            metadata=list(TYPE_SAFETY_METADATA),
        )
        yield store
    finally:
        await _teardown_backend(backend, store)
        await store.close()


@pytest.mark.asyncio
async def test_boolean_true_roundtrip(type_safety_store: Any) -> None:
    n = await type_safety_store.count(filter={"active": True})
    assert n == 1


@pytest.mark.asyncio
async def test_boolean_false_roundtrip(type_safety_store: Any) -> None:
    n = await type_safety_store.count(filter={"active": False})
    assert n == 1


@pytest.mark.asyncio
async def test_numeric_int_roundtrip(type_safety_store: Any) -> None:
    n = await type_safety_store.count(filter={"count": 5})
    assert n == 1


@pytest.mark.asyncio
async def test_numeric_no_implicit_string_coercion(
    type_safety_store: Any,
) -> None:
    """Filter ``{"count": "5"}`` against integer metadata never matches.

    Pre-fix pgvector's text-cast translation also returned 0 here (for
    the wrong reason — both sides stringified). Post-fix it returns 0
    because JSONB ``@>`` is type-preserving and ``"5"`` (string) does
    not contain ``5`` (number).
    """
    n = await type_safety_store.count(filter={"count": "5"})
    assert n == 0
