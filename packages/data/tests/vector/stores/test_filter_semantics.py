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

import logging
import os
import uuid
from collections.abc import AsyncIterator, Iterator
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


@pytest.fixture
def pgvector_config(make_pgvector_test_table: Any) -> Iterator[dict[str, Any]]:
    """Per-test pgvector config from the shared ``dataknobs-common``
    fixture (pre-drop + teardown drop + pgvector-extension ensure live
    there now). ``metric`` is preserved at ``cosine`` to keep behavior
    byte-identical to the prior hand-rolled config.
    """
    gen = make_pgvector_test_table("test_filter_", dimensions=4)
    cfg = next(gen)
    cfg["metric"] = "cosine"
    try:
        yield cfg
    finally:
        gen.close()


async def _teardown_backend(backend: str, store: Any) -> None:
    """Drop the per-test Chroma collection created by a fixture.

    pgvector tables are owned by the shared ``make_pgvector_test_table``
    fixture (pre-drop + teardown drop), so only Chroma needs explicit
    cleanup here. The Chroma failure is logged rather than swallowed so
    an orphaned test collection becomes visible in pytest output.
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


# ---------------------------------------------------------------------------
# Config-level ``domain_id`` scoping.
#
# ``PgVectorStore`` honors a config-level ``domain_id``: every
# read/count/clear/update_metadata_where is implicitly scoped to that
# domain, and ``add_vectors`` defaults a row's ``domain_id`` to it.
# Memory/FAISS/Chroma historically ignored config ``domain_id``
# entirely — the multi-tenant isolation a consumer configures was a
# silent no-op on those backends, and a runtime backend swap changed
# isolation semantics. These reproduce-first tests pin the fixed
# symmetric contract; they fail on memory/faiss/chroma pre-fix and
# pass on pgvector (the expected asymmetry split).
# ---------------------------------------------------------------------------

_DOMAIN_SCOPED_IDS = ["s1", "s2", "o1"]


def _domain_scoped_metadata() -> list[dict[str, Any]]:
    """s1/s2 carry NO ``domain_id`` (must default to config ``t1``);
    o1 explicitly belongs to ``t2`` (must be scoped out).
    """
    return [
        {"k": "v"},
        {"k": "v"},
        {"domain_id": "t2", "k": "v"},
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
async def domain_scoped_store(
    request: pytest.FixtureRequest, pgvector_config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Store configured with ``domain_id="t1"``, seeded across t1/t2."""
    backend = request.param
    store: Any
    if backend == "memory":
        store = MemoryVectorStore({"dimensions": 4, "domain_id": "t1"})
    elif backend == "faiss":
        store = FaissVectorStore(
            {"dimensions": 4, "metric": "cosine", "domain_id": "t1"}
        )
    elif backend == "chroma":
        store = ChromaVectorStore(
            {
                "dimensions": 4,
                "domain_id": "t1",
                "collection_name": f"test_domain_{uuid.uuid4().hex[:8]}",
            }
        )
    elif backend == "pgvector":
        store = PgVectorStore({**pgvector_config, "domain_id": "t1"})
    else:
        pytest.fail(f"Unknown backend param: {backend}")

    await store.initialize()
    try:
        await store.add_vectors(
            _seed_vectors()[:3],
            ids=list(_DOMAIN_SCOPED_IDS),
            metadata=_domain_scoped_metadata(),
        )
        yield store
    finally:
        await _teardown_backend(backend, store)
        await store.close()


# NOTE on the asserted contract. #8 delivers *isolation* symmetry: a
# configured ``domain_id`` confines every read/count/clear/update to
# that domain on every backend, and a cross-domain request is empty.
# The behavior of a caller *explicitly* passing ``domain_id`` in the
# filter is intentionally NOT asserted as uniform: pgvector scopes via
# a dedicated ``domain_id`` column and stores caller metadata JSONB
# verbatim, so an explicit ``{"domain_id": "t1"}`` filter is a
# JSONB-containment probe there, orthogonal to the column scope —
# whereas memory/faiss/chroma carry ``domain_id`` in metadata. That
# divergence is inherent to pgvector's richer schema and is documented
# in VECTOR_FILTER_SEMANTICS.md, not pinned here.


@pytest.mark.asyncio
async def test_config_domain_id_scopes_count(
    domain_scoped_store: Any,
) -> None:
    """count() is implicitly scoped to the configured domain; a
    cross-domain probe intersects to empty on every backend.
    """
    # s1/s2 defaulted to t1; o1 is t2 and scoped out.
    assert await domain_scoped_store.count() == 2
    # Caller asking for a different domain than the configured scope
    # intersects to empty (pgvector: column='t1' AND JSONB-probe 't2';
    # memory/faiss/chroma: AND-merged unsatisfiable filter).
    assert await domain_scoped_store.count(filter={"domain_id": "t2"}) == 0


@pytest.mark.asyncio
async def test_config_domain_id_scopes_search(
    domain_scoped_store: Any,
) -> None:
    """search() never returns rows outside the configured domain."""
    results = await domain_scoped_store.search(_query_vector(), k=10)
    assert {r[0] for r in results} == {"s1", "s2"}
    # Cross-domain request → empty on every backend.
    cross = await domain_scoped_store.search(
        _query_vector(), k=10, filter={"domain_id": "t2"}
    )
    assert cross == []


@pytest.mark.asyncio
async def test_config_domain_id_scopes_update_metadata_where(
    domain_scoped_store: Any,
) -> None:
    """update_metadata_where(None, ...) only touches the configured
    domain — the count of affected rows is exactly the in-domain set,
    and a cross-domain request is a no-op.
    """
    affected = await domain_scoped_store.update_metadata_where(
        None, {"_stale": True}
    )
    assert affected == 2
    # An explicit cross-domain update never escapes the configured
    # scope (intersects to empty on every backend).
    cross = await domain_scoped_store.update_metadata_where(
        {"domain_id": "t2"}, {"_stale": True}
    )
    assert cross == 0
    # The scoped store still sees exactly its two in-domain rows.
    assert await domain_scoped_store.count() == 2
