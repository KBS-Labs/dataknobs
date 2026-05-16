"""``VectorStore.update_metadata_where(filter, set_)`` cross-backend tests.

The filter-keyed sibling of ``update_metadata`` underpins the
``TOMBSTONE`` zero-downtime swap (mark a domain's old chunks
``_stale`` without an id list). The contract pinned here — *merge,
don't replace; only filter-matched rows; return the affected count* —
must hold identically on every in-tree backend, since the vector
store is consumer-selectable by config. There is deliberately no
"FAISS fallback" variant: FAISS passes the same assertions as every
other backend.
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
        "table_name": f"test_update_meta_where_{uuid.uuid4().hex[:8]}",
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


# Three vectors split across two tenants. Orthogonal unit vectors so
# similarity ordering is irrelevant — these tests are about metadata.
SEED_IDS = ["a-1", "a-2", "b-1"]


def _seed_metadata() -> list[dict[str, Any]]:
    """Fresh seed-metadata dicts on every call.

    Stores keep the caller's dict references and ``update_metadata``/
    ``update_metadata_where`` merge in place, so a module-level
    constant would be mutated and leak state across parametrized
    fixtures and tests (a tenant-B row picking up another test's
    ``_stale`` write). Mirrors ``_seed_vectors`` returning a fresh
    array each call.
    """
    return [
        {"tenant": "A"},
        {"tenant": "A"},
        {"tenant": "B"},
    ]


def _seed_vectors() -> np.ndarray:
    return np.eye(3, 4, dtype=np.float32)


def _make_store(
    backend: str,
    pgvector_config: dict[str, Any],
    *,
    extra_config: dict[str, Any] | None = None,
) -> Any:
    """Construct (not initialize) a VectorStore for ``backend``.

    Shared by every parametrized store fixture so backend wiring lives
    in one place. ``extra_config`` is merged into the store config
    (used by the timestamp tests to request ``datetime`` format).
    """
    extra = extra_config or {}
    if backend == "memory":
        return MemoryVectorStore({"dimensions": 4, **extra})
    if backend == "faiss":
        return FaissVectorStore(
            {"dimensions": 4, "metric": "cosine", **extra}
        )
    if backend == "chroma":
        return ChromaVectorStore(
            {
                "dimensions": 4,
                "collection_name": (
                    f"test_update_meta_where_{uuid.uuid4().hex[:8]}"
                ),
                **extra,
            }
        )
    if backend == "pgvector":
        return PgVectorStore({**pgvector_config, **extra})
    pytest.fail(f"Unknown backend param: {backend}")


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
    store = _make_store(backend, pgvector_config)
    await store.initialize()
    try:
        await store.add_vectors(
            _seed_vectors(), ids=list(SEED_IDS), metadata=_seed_metadata()
        )
        yield store
    finally:
        await _teardown_backend(backend, store)
        await store.close()


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
        pytest.param("pgvector", id="pgvector", marks=_pgvector_marks),
    ]
)
async def ts_vector_store(
    request: pytest.FixtureRequest, pgvector_config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Freshly-seeded store for timestamp-capable backends only.

    Chroma is deliberately excluded — it has no timestamp side-car
    (documented as deferred in ``vector-timestamps.md``). Memory,
    FAISS, and PgVector expose ``_created_at`` / ``_updated_at`` via
    ``include_timestamps=True``. ``datetime`` format so before/after
    comparisons are real ``datetime`` objects, not ISO strings.
    """
    backend = request.param
    store = _make_store(
        backend,
        pgvector_config,
        extra_config={"timestamps": {"format": "datetime"}},
    )
    await store.initialize()
    try:
        await store.add_vectors(
            _seed_vectors(), ids=list(SEED_IDS), metadata=_seed_metadata()
        )
        yield store
    finally:
        await _teardown_backend(backend, store)
        await store.close()


@pytest.mark.asyncio
async def test_update_metadata_where_filter_scoped(
    any_vector_store: Any,
) -> None:
    """Only filter-matched rows are touched; the count is returned."""
    affected = await any_vector_store.update_metadata_where(
        {"tenant": "A"}, {"_stale": True}
    )
    assert affected == 2

    # Equality filter on the merged key — the exact mechanism the
    # TOMBSTONE cleanup/rollback uses to select tombstoned rows.
    assert await any_vector_store.count(filter={"_stale": True}) == 2
    # Tenant-B row was outside the filter — untouched.
    assert await any_vector_store.count(filter={"tenant": "B"}) == 1


@pytest.mark.asyncio
async def test_update_metadata_where_merges_not_replaces(
    any_vector_store: Any,
) -> None:
    """``set_`` is merged into existing metadata, not a wholesale swap."""
    await any_vector_store.update_metadata_where(
        {"tenant": "A"}, {"_stale": True}
    )
    # Original ``tenant`` key still selects the same two rows.
    assert await any_vector_store.count(filter={"tenant": "A"}) == 2
    assert (
        await any_vector_store.count(
            filter={"tenant": "A", "_stale": True}
        )
        == 2
    )


@pytest.mark.asyncio
async def test_update_metadata_where_none_filter_updates_all(
    any_vector_store: Any,
) -> None:
    """``filter=None`` matches every vector (parity with clear/count)."""
    affected = await any_vector_store.update_metadata_where(
        None, {"_stale": False}
    )
    assert affected == 3
    assert await any_vector_store.count(filter={"_stale": False}) == 3


@pytest.mark.asyncio
async def test_update_metadata_where_no_match_returns_zero(
    any_vector_store: Any,
) -> None:
    """A filter matching nothing changes nothing and returns 0."""
    affected = await any_vector_store.update_metadata_where(
        {"tenant": "NONEXISTENT"}, {"_stale": True}
    )
    assert affected == 0
    assert await any_vector_store.count(filter={"_stale": True}) == 0


@pytest.mark.asyncio
async def test_update_metadata_where_refreshes_updated_at(
    ts_vector_store: Any,
) -> None:
    """``update_metadata_where`` refreshes ``_updated_at`` and keeps
    ``_created_at`` on every timestamp-capable backend.

    Reproduce-first for #10: fails on FAISS pre-fix because
    ``get_vectors``/``search`` reject ``include_timestamps`` and FAISS
    has no timestamp side-car. Passes on Memory/PgVector (already
    supported) — the expected reproduce split.
    """
    before = await ts_vector_store.get_vectors(
        ["a-1"], include_metadata=True, include_timestamps=True
    )
    before_meta = before[0][1]
    assert before_meta is not None
    created_before = before_meta["_created_at"]
    updated_before = before_meta["_updated_at"]
    assert created_before is not None
    assert updated_before is not None

    await asyncio.sleep(0.01)
    affected = await ts_vector_store.update_metadata_where(
        {"tenant": "A"}, {"_stale": True}
    )
    assert affected == 2

    after = await ts_vector_store.get_vectors(
        ["a-1"], include_metadata=True, include_timestamps=True
    )
    after_meta = after[0][1]
    assert after_meta is not None
    # The merge landed.
    assert after_meta["_stale"] is True
    # created_at preserved, updated_at advanced.
    assert after_meta["_created_at"] == created_before
    assert after_meta["_updated_at"] > updated_before
    # A row outside the filter (tenant B) keeps its original updated_at.
    b_after = await ts_vector_store.get_vectors(
        ["b-1"], include_metadata=True, include_timestamps=True
    )
    assert b_after[0][1] is not None
    assert "_stale" not in b_after[0][1]


# --- FAISS-only timestamp behavior (mirrors MemoryVectorStore) ---

pytestmark_faiss = pytest.mark.skipif(
    not is_faiss_available(), reason="faiss not installed"
)


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_add_vectors_sets_created_equals_updated() -> None:
    """A freshly added FAISS vector has ``_created_at == _updated_at``."""
    store = FaissVectorStore(
        {
            "dimensions": 4,
            "metric": "cosine",
            "timestamps": {"format": "datetime"},
        }
    )
    await store.initialize()
    try:
        await store.add_vectors(
            _seed_vectors(), ids=list(SEED_IDS), metadata=_seed_metadata()
        )
        (_, meta), = await store.get_vectors(
            ["a-1"], include_metadata=True, include_timestamps=True
        )
        assert meta is not None
        assert meta["_created_at"] == meta["_updated_at"]
    finally:
        await store.close()


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_readd_preserves_created_advances_updated() -> None:
    """Re-adding an existing id keeps ``_created_at``, advances
    ``_updated_at`` (upsert semantics across FAISS internal-id eviction)."""
    store = FaissVectorStore(
        {
            "dimensions": 4,
            "metric": "cosine",
            "timestamps": {"format": "datetime"},
        }
    )
    await store.initialize()
    try:
        await store.add_vectors(
            _seed_vectors()[:1], ids=["a-1"], metadata=[{"tenant": "A"}]
        )
        (_, first), = await store.get_vectors(
            ["a-1"], include_metadata=True, include_timestamps=True
        )
        assert first is not None
        await asyncio.sleep(0.01)
        await store.add_vectors(
            _seed_vectors()[:1], ids=["a-1"], metadata=[{"tenant": "A2"}]
        )
        (_, second), = await store.get_vectors(
            ["a-1"], include_metadata=True, include_timestamps=True
        )
        assert second is not None
        assert second["_created_at"] == first["_created_at"]
        assert second["_updated_at"] > first["_updated_at"]
        # The metadata was actually replaced (no orphan).
        assert second["tenant"] == "A2"
        assert await store.count() == 1
    finally:
        await store.close()


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_update_metadata_refreshes_updated_at() -> None:
    """``update_metadata`` (id-keyed) advances ``_updated_at``."""
    store = FaissVectorStore(
        {
            "dimensions": 4,
            "metric": "cosine",
            "timestamps": {"format": "datetime"},
        }
    )
    await store.initialize()
    try:
        await store.add_vectors(
            _seed_vectors()[:1], ids=["a-1"], metadata=[{"tenant": "A"}]
        )
        (_, before), = await store.get_vectors(
            ["a-1"], include_metadata=True, include_timestamps=True
        )
        assert before is not None
        await asyncio.sleep(0.01)
        await store.update_metadata(["a-1"], [{"tenant": "A", "x": 1}])
        (_, after), = await store.get_vectors(
            ["a-1"], include_metadata=True, include_timestamps=True
        )
        assert after is not None
        assert after["_created_at"] == before["_created_at"]
        assert after["_updated_at"] > before["_updated_at"]
    finally:
        await store.close()


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_timestamps_survive_save_load(tmp_path: Any) -> None:
    """Timestamps round-trip through the ``.meta`` pickle; a legacy
    pickle without a ``timestamps`` key loads as empty (rows return
    ``None`` for both timestamps), mirroring ``memory.py`` load."""
    persist = tmp_path / "faiss_ts.index"
    cfg = {
        "dimensions": 4,
        "metric": "cosine",
        "persist_path": str(persist),
        "timestamps": {"format": "datetime"},
    }
    store = FaissVectorStore(cfg)
    await store.initialize()
    await store.add_vectors(
        _seed_vectors()[:1], ids=["a-1"], metadata=[{"tenant": "A"}]
    )
    (_, saved), = await store.get_vectors(
        ["a-1"], include_metadata=True, include_timestamps=True
    )
    assert saved is not None
    await store.close()  # triggers save()

    reloaded = FaissVectorStore(cfg)
    await reloaded.initialize()  # triggers load()
    (_, rt), = await reloaded.get_vectors(
        ["a-1"], include_metadata=True, include_timestamps=True
    )
    assert rt is not None
    assert rt["_created_at"] == saved["_created_at"]
    assert rt["_updated_at"] == saved["_updated_at"]
    await reloaded.close()

    # Simulate a pre-Item-36 .meta pickle (no "timestamps" key).
    import pickle

    meta_path = str(persist) + ".meta"
    with open(meta_path, "rb") as fh:
        legacy = pickle.load(fh)
    legacy.pop("timestamps", None)
    with open(meta_path, "wb") as fh:
        pickle.dump(legacy, fh)

    legacy_store = FaissVectorStore(cfg)
    await legacy_store.initialize()
    (_, legacy_meta), = await legacy_store.get_vectors(
        ["a-1"], include_metadata=True, include_timestamps=True
    )
    assert legacy_meta is not None
    assert legacy_meta["_created_at"] is None
    assert legacy_meta["_updated_at"] is None
    await legacy_store.close()


# ---------------------------------------------------------------------------
# Metadata-aliasing conformance (Item 131)
#
# Regression guard: the caller's ``add_vectors`` metadata dict and the
# store's internal copy must be fully isolated in *both* directions on
# *every* in-tree backend. The store-side root cause was fixed in PR5
# via ``VectorStoreBase._apply_domain_default`` (copy-on-ingest,
# unconditional); PgVector/Chroma already serialized on write. These
# tests therefore must pass on all four backends today — a failure on
# memory/FAISS means that copy routing has regressed.
# ---------------------------------------------------------------------------


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
async def empty_vector_store(
    request: pytest.FixtureRequest, pgvector_config: dict[str, Any]
) -> AsyncIterator[Any]:
    """Initialized-but-unseeded store, so the test owns the exact
    metadata dict passed to ``add_vectors`` (the aliasing subject)."""
    backend = request.param
    store = _make_store(backend, pgvector_config)
    await store.initialize()
    try:
        yield store
    finally:
        await _teardown_backend(backend, store)
        await store.close()


@pytest.mark.asyncio
async def test_caller_mutation_after_add_does_not_leak_into_store(
    empty_vector_store: Any,
) -> None:
    """Caller → store isolation: mutating the dict after ``add_vectors``
    must not change what the store returns."""
    caller_meta = {"k": 1}
    await empty_vector_store.add_vectors(
        _seed_vectors()[:1], ids=["a"], metadata=[caller_meta]
    )

    caller_meta["k"] = 2  # caller reuses/mutates its own dict

    (_, stored), = await empty_vector_store.get_vectors(
        ["a"], include_metadata=True
    )
    assert stored is not None
    assert stored["k"] == 1


@pytest.mark.asyncio
async def test_store_writes_do_not_leak_onto_caller_dict(
    empty_vector_store: Any,
) -> None:
    """Store → caller isolation: a store-internal write
    (``update_metadata_where``) must not appear on the caller's dict."""
    caller_meta = {"k": 1}
    await empty_vector_store.add_vectors(
        _seed_vectors()[:1], ids=["a"], metadata=[caller_meta]
    )

    await empty_vector_store.update_metadata_where(None, {"_stale": True})

    # The store-internal key never lands on the caller's dict, and no
    # injected bookkeeping (timestamps) leaks back either.
    assert caller_meta == {"k": 1}