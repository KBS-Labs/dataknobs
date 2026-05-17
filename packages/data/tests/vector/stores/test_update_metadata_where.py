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
    import faiss

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
# Metadata-aliasing conformance
#
# Regression guard: the caller's ``add_vectors`` metadata dict and the
# store's internal copy must be fully isolated in *both* directions on
# *every* in-tree backend. The store-side isolation is owned by
# ``VectorStoreBase._apply_domain_default`` (copy-on-ingest,
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


# ---------------------------------------------------------------------------
# FAISS IVF get_vectors
#
# ``get_vectors`` serves the stored vector from the internal-id-keyed
# side-car, not FAISS ``reconstruct``. For the IVF index types
# (``ivfflat``/``ivfpq``, auto-selected for ``dimensions >= 100`` — the
# production 384/768/1024 case) FAISS reconstruct-by-id needs a
# maintained direct map that this faiss build refuses to combine with
# ``remove_ids``, so the index is kept for ``search`` only and these
# tests guard that IVF get_vectors / delete / upsert / clear / reload
# all stay correct via the side-car.
# ---------------------------------------------------------------------------


def _ivf_vectors(n: int, dim: int) -> np.ndarray:
    """``n`` distinct unit-ish rows of width ``dim`` (>= nlist to train)."""
    arr = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        arr[i, i % dim] = 1.0
        arr[i, (i + 1) % dim] = 0.5
    return arr


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_ivfflat_get_vectors_reconstructs() -> None:
    """IVF index types return stored vectors + metadata + timestamps.

    The regression guarded here: IVF ``get_vectors`` previously
    returned ``(None, None)`` for every id (FAISS reconstruct-by-id
    unusable for IVF); it now serves from the side-car.
    """
    dim = 128
    n = 8  # >= nlist so the IVF index trains on first add
    store = FaissVectorStore(
        {
            "dimensions": dim,
            "metric": "cosine",
            "index_type": "ivfflat",
            "index_params": {"nlist": 4},
            "timestamps": {"format": "datetime"},
        }
    )
    await store.initialize()
    try:
        ids = [f"v-{i}" for i in range(n)]
        await store.add_vectors(
            _ivf_vectors(n, dim),
            ids=ids,
            metadata=[{"i": i} for i in range(n)],
        )
        got = await store.get_vectors(
            ids, include_metadata=True, include_timestamps=True
        )
        assert len(got) == n
        for i, (vec, meta) in enumerate(got):
            assert vec is not None, (
                f"id v-{i} returned None — IVF get_vectors broken"
            )
            assert vec.shape == (dim,)
            assert meta is not None
            assert meta["i"] == i
            assert meta["_created_at"] is not None
            assert meta["_updated_at"] is not None
    finally:
        await store.close()


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_ivfflat_get_vectors_survives_save_load(
    tmp_path: Any,
) -> None:
    """IVF ``get_vectors`` still returns stored vectors after a
    save/reload cycle (the side-car is pickled and restored)."""
    dim = 128
    n = 8
    persist = tmp_path / "ivf.index"
    cfg = {
        "dimensions": dim,
        "metric": "cosine",
        "index_type": "ivfflat",
        "index_params": {"nlist": 4},
        "persist_path": str(persist),
    }
    store = FaissVectorStore(cfg)
    await store.initialize()
    ids = [f"v-{i}" for i in range(n)]
    await store.add_vectors(
        _ivf_vectors(n, dim), ids=ids, metadata=[{"i": i} for i in range(n)]
    )
    await store.close()  # triggers save()

    reloaded = FaissVectorStore(cfg)
    await reloaded.initialize()  # triggers load()
    try:
        got = await reloaded.get_vectors(ids, include_metadata=True)
        for i, (vec, meta) in enumerate(got):
            assert vec is not None, (
                "IVF get_vectors lost the vector after save/reload"
            )
            assert meta is not None and meta["i"] == i
    finally:
        await reloaded.close()


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_ivfflat_get_vectors_after_clear_and_repopulate() -> None:
    """An unfiltered ``clear()`` recreates a fresh index; IVF
    ``get_vectors`` must stay correct across a clear/re-ingest cycle.

    Guards the clear-then-repopulate path on an IVF-dimensioned store
    (the side-car must be cleared and rebuilt in lockstep with the
    index so re-added ids resolve, not return ``(None, None)``).
    """
    dim = 128
    n = 8  # >= nlist so the IVF index trains on first add
    cfg = {
        "dimensions": dim,
        "metric": "cosine",
        "index_type": "ivfflat",
        "index_params": {"nlist": 4},
    }
    store = FaissVectorStore(cfg)
    await store.initialize()
    try:
        ids = [f"v-{i}" for i in range(n)]
        await store.add_vectors(
            _ivf_vectors(n, dim),
            ids=ids,
            metadata=[{"i": i} for i in range(n)],
        )
        # Sanity: get_vectors works before the clear/re-ingest cycle.
        before = await store.get_vectors(ids)
        assert all(vec is not None for vec, _ in before)

        await store.clear()  # unfiltered: recreates the index
        assert await store.count() == 0

        await store.add_vectors(
            _ivf_vectors(n, dim),
            ids=ids,
            metadata=[{"i": i} for i in range(n)],
        )
        got = await store.get_vectors(ids, include_metadata=True)
        assert len(got) == n
        for i, (vec, meta) in enumerate(got):
            assert vec is not None, (
                f"id v-{i} returned None after clear()+re-add "
                "— side-car not rebuilt with the fresh index"
            )
            assert vec.shape == (dim,)
            assert meta is not None and meta["i"] == i
    finally:
        await store.close()


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_ivfflat_delete_vectors_then_get() -> None:
    """``delete_vectors`` on an IVF index removes only the targeted ids
    and the survivors still resolve via ``get_vectors``.

    Every prior FAISS delete test used ``dimensions=4`` (the ``flat``
    path), so the IVF ``remove_ids`` path (``faiss.py`` line ~330) was
    untested — the earlier IVF fix attempt broke it. This covers it on
    a real IVF index, alongside side-car/index consistency.
    """
    dim = 128
    n = 8  # >= nlist so the IVF index trains on first add
    store = FaissVectorStore(
        {
            "dimensions": dim,
            "metric": "cosine",
            "index_type": "ivfflat",
            "index_params": {"nlist": 4},
        }
    )
    await store.initialize()
    try:
        ids = [f"v-{i}" for i in range(n)]
        await store.add_vectors(
            _ivf_vectors(n, dim),
            ids=ids,
            metadata=[{"i": i} for i in range(n)],
        )
        removed = await store.delete_vectors(["v-1", "v-3"])
        assert removed == 2
        assert await store.count() == n - 2

        gone = await store.get_vectors(["v-1", "v-3"])
        assert gone == [(None, None), (None, None)]

        survivors = [f"v-{i}" for i in range(n) if i not in (1, 3)]
        got = await store.get_vectors(survivors, include_metadata=True)
        for ext_id, (vec, meta) in zip(survivors, got):
            assert vec is not None, (
                f"survivor {ext_id} lost its vector after IVF "
                "delete_vectors"
            )
            assert meta is not None and meta["i"] == int(ext_id[2:])
    finally:
        await store.close()


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_ivfflat_upsert_evicts_and_gets() -> None:
    """Re-adding an existing id on an IVF index evicts the old internal
    id (``remove_ids``, ``faiss.py`` line ~228) and the refreshed row
    resolves via ``get_vectors`` with carried-over ``_created_at``.

    Exercises the upsert-eviction ``remove_ids`` on a real IVF index
    (no flat-index test reached it) and that the side-car drops the
    orphan and stores the new row in lockstep.
    """
    dim = 128
    n = 8
    store = FaissVectorStore(
        {
            "dimensions": dim,
            "metric": "cosine",
            "index_type": "ivfflat",
            "index_params": {"nlist": 4},
            "timestamps": {"format": "datetime"},
        }
    )
    await store.initialize()
    try:
        ids = [f"v-{i}" for i in range(n)]
        await store.add_vectors(
            _ivf_vectors(n, dim),
            ids=ids,
            metadata=[{"gen": 1} for _ in range(n)],
        )
        first = await store.get_vectors(
            ["v-0"], include_metadata=True, include_timestamps=True
        )
        created_v0 = first[0][1]["_created_at"]

        # Re-add v-0 with a fresh vector + metadata: triggers the
        # orphan-eviction remove_ids on the IVF index.
        new_vec = _ivf_vectors(n, dim)[1:2]
        await store.add_vectors(new_vec, ids=["v-0"], metadata=[{"gen": 2}])

        assert await store.count() == n  # upsert, not insert
        got = await store.get_vectors(
            ["v-0"], include_metadata=True, include_timestamps=True
        )
        vec, meta = got[0]
        assert vec is not None, (
            "re-added id lost its vector after IVF upsert eviction"
        )
        assert meta["gen"] == 2
        assert meta["_created_at"] == created_v0  # created carried over
        assert meta["_updated_at"] >= created_v0
        # Untouched neighbor still reconstructs.
        other = await store.get_vectors(["v-5"], include_metadata=True)
        assert other[0][0] is not None and other[0][1]["gen"] == 1
    finally:
        await store.close()


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_ivf_small_first_batch_defers_then_migrates() -> None:
    """An IVF store whose first ``add_vectors`` batch is smaller than
    ``nlist`` must succeed (search + get_vectors correct) and then
    migrate to a real trained IVF index once enough vectors accumulate.

    Reproduce-first: pre-fix the train-skip ``else`` was a misleading
    ``pass`` and execution fell through to ``add_with_ids`` on an
    untrained ``IndexIVF*`` — FAISS raised ``RuntimeError``. The store
    now serves a temporary flat index until >= ``nlist`` vectors exist,
    then trains the IVF and migrates from the side-car.
    """
    dim = 128
    nlist = 4
    store = FaissVectorStore(
        {
            "dimensions": dim,
            "metric": "cosine",
            "index_type": "ivfflat",
            "index_params": {"nlist": nlist},
        }
    )
    await store.initialize()
    try:
        # First batch of 2 < nlist=4: pre-fix this raised RuntimeError.
        small = _ivf_vectors(2, dim)
        await store.add_vectors(
            small, ids=["a", "b"], metadata=[{"i": 0}, {"i": 1}]
        )
        assert store._deferred_ivf is True  # temp-flat, IVF deferred

        # Search + get_vectors must already be correct on the temp-flat.
        hits = await store.search(small[0], k=1)
        assert hits and hits[0][0] == "a"
        got = await store.get_vectors(["a", "b"], include_metadata=True)
        assert [v is not None for v, _ in got] == [True, True]
        assert [m["i"] for _, m in got] == [0, 1]

        # Cross the nlist threshold (cumulative 5 >= 4): migrate to IVF.
        more = _ivf_vectors(5, dim)[2:]  # 3 fresh rows
        await store.add_vectors(
            more,
            ids=["c", "d", "e"],
            metadata=[{"i": 2}, {"i": 3}, {"i": 4}],
        )
        assert store._deferred_ivf is False  # migrated to real IVF
        inner = faiss.downcast_index(store.index.index)
        assert isinstance(inner, faiss.IndexIVF)
        assert inner.is_trained

        # All five survive the migration (side-car drives the rebuild).
        all_ids = ["a", "b", "c", "d", "e"]
        got = await store.get_vectors(all_ids, include_metadata=True)
        assert all(v is not None for v, _ in got)
        assert [m["i"] for _, m in got] == [0, 1, 2, 3, 4]
        assert await store.count() == 5
        hits = await store.search(more[-1], k=1)
        assert hits and hits[0][0] == "e"
    finally:
        await store.close()


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_ivf_deferred_state_survives_save_load(
    tmp_path: Any,
) -> None:
    """A still-deferred IVF store (first batch < nlist) round-trips
    through save/load and migrates to a real IVF after reload once the
    threshold is crossed — the deferred flag and side-car persist.
    """
    dim = 128
    nlist = 4
    persist = tmp_path / "ivf_deferred.index"
    cfg = {
        "dimensions": dim,
        "metric": "cosine",
        "index_type": "ivfflat",
        "index_params": {"nlist": nlist},
        "persist_path": str(persist),
    }
    store = FaissVectorStore(cfg)
    await store.initialize()
    await store.add_vectors(
        _ivf_vectors(2, dim), ids=["a", "b"], metadata=[{"i": 0}, {"i": 1}]
    )
    assert store._deferred_ivf is True
    await store.close()  # save() while deferred

    reloaded = FaissVectorStore(cfg)
    await reloaded.initialize()  # load(): deferred state restored
    try:
        assert reloaded._deferred_ivf is True
        got = await reloaded.get_vectors(["a", "b"], include_metadata=True)
        assert [m["i"] for _, m in got] == [0, 1]

        # Crossing the threshold post-reload migrates from the side-car.
        await reloaded.add_vectors(
            _ivf_vectors(5, dim)[2:],
            ids=["c", "d", "e"],
            metadata=[{"i": 2}, {"i": 3}, {"i": 4}],
        )
        assert reloaded._deferred_ivf is False
        assert await reloaded.count() == 5
        got = await reloaded.get_vectors(
            ["a", "b", "c", "d", "e"], include_metadata=True
        )
        assert all(v is not None for v, _ in got)
        assert [m["i"] for _, m in got] == [0, 1, 2, 3, 4]
    finally:
        await reloaded.close()


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_ivf_search_applies_configured_nprobe() -> None:
    """The configured ``nprobe`` must reach the trained IVF index.

    Reproduce-first: ``self.index`` is always ``IndexIDMap2``, which
    does not proxy ``nprobe``, so the pre-fix
    ``if hasattr(self.index, "nprobe")`` guard was always False and the
    ``nprobe`` assignment was dead — every IVF store searched at FAISS's
    default ``nprobe=1`` regardless of config. Post-fix ``search()``
    unwraps the inner IVF and sets ``nprobe`` there.
    """
    dim = 128
    nlist = 4
    store = FaissVectorStore(
        {
            "dimensions": dim,
            "metric": "cosine",
            "index_type": "ivfflat",
            "index_params": {"nlist": nlist},
            "search_params": {"nprobe": 3},
        }
    )
    await store.initialize()
    try:
        n = 8  # >= nlist so the IVF trains and migrates off temp-flat
        await store.add_vectors(
            _ivf_vectors(n, dim),
            ids=[f"v{i}" for i in range(n)],
            metadata=[{"i": i} for i in range(n)],
        )
        assert store._deferred_ivf is False  # real trained IVF
        inner = faiss.downcast_index(store.index.index)
        assert isinstance(inner, faiss.IndexIVF)
        # Search is what applies nprobe; pre-fix it stayed at 1.
        await store.search(_ivf_vectors(n, dim)[0], k=1)
        assert inner.nprobe == 3
    finally:
        await store.close()


@pytestmark_faiss
@pytest.mark.asyncio
async def test_faiss_get_vectors_internal_id_desync_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A resolved id whose internal id has no stored vector is logged
    at WARNING and surfaced as ``(None, None)`` — not silently
    indistinguishable from an absent id.

    The external id still resolves (so the absent-id short-circuit is
    skipped) but is repointed at an internal id the side-car never
    stored — exactly the shape of the post-delete internal-id reuse
    race the code comment describes.
    """
    store = FaissVectorStore({"dimensions": 4, "metric": "cosine"})
    await store.initialize()
    try:
        await store.add_vectors(
            _seed_vectors()[:1], ids=["a"], metadata=[{"k": 1}]
        )

        # ``"a"`` stays in ``id_map`` (passes the absent-id guard) but
        # now maps to an internal id the side-car never stored.
        store.id_map["a"] = 10_000_000

        with caplog.at_level(
            logging.WARNING, logger="dataknobs_data.vector.stores.faiss"
        ):
            got = await store.get_vectors(["a"])

        assert got == [(None, None)]
        assert any(
            "no stored vector" in rec.getMessage().lower()
            for rec in caplog.records
        ), "expected a WARNING for the internal-id desync"
    finally:
        await store.close()
