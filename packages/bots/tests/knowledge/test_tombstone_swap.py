"""TOMBSTONE swap mode (Items 125+126 Phase 2 / PR5).

Real constructs only — ``InMemoryKnowledgeBackend`` is the documented
testing backend; ``RAGKnowledgeBase`` uses a real vector store
(parametrized) and the ``echo`` embedding provider; no mocks.

The TOMBSTONE flow is store-agnostic — it only calls
``destination.update_metadata_where`` / ``clear`` /
``ingest_from_backend``. Per-store correctness of
``update_metadata_where`` is pinned separately by the data-layer
``test_update_metadata_where.py`` suite (all four in-tree stores);
here we parametrize the *manager flow* over Memory + FAISS always
(two genuinely different real store impls, no external services) and
Chroma / PgVector under their service markers, so the orchestration
(mark → ingest → retire, rollback, read-hiding, per-file scope) is
proven end-to-end against real stores.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest

from dataknobs_bots.knowledge import (
    IngestSwapMode,
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
)
from dataknobs_bots.knowledge.storage import (
    IngestionStatus,
    InMemoryKnowledgeBackend,
)
from dataknobs_common.testing import (
    is_chromadb_available,
    is_faiss_available,
    requires_postgres,
)
from dataknobs_data.vector.stores.memory import MemoryVectorStore

logger = logging.getLogger(__name__)

try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

_pgvector_marks = [
    requires_postgres,
    pytest.mark.skipif(
        os.environ.get("TEST_POSTGRES", "").lower() != "true"
        or not ASYNCPG_AVAILABLE,
        reason="pgvector tests require TEST_POSTGRES=true and asyncpg",
    ),
]

_EMBED_DIM = 768  # EchoProvider's default embedding dimension.

# EchoProvider embeds via SHA-256, so query/chunk cosine similarity is
# effectively random in [-1, 1]. These tests exercise the swap/stale
# logic, not retrieval quality, so reads pass min_similarity=-1.0 to
# keep every candidate regardless of similarity sign — deterministic.
_ALL = -1.0


def _pg_connection_string() -> str:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    database = os.environ.get("POSTGRES_DB", "test_dataknobs")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _vector_store_config(backend: str) -> dict[str, Any]:
    if backend == "memory":
        return {"backend": "memory", "dimensions": _EMBED_DIM}
    if backend == "faiss":
        # Flat index: exact, no training step (auto would pick ivfflat
        # at 384 dims and need >= nlist vectors to train).
        return {
            "backend": "faiss",
            "dimensions": _EMBED_DIM,
            "metric": "cosine",
            "index_type": "flat",
        }
    if backend == "chroma":
        return {
            "backend": "chroma",
            "dimensions": _EMBED_DIM,
            "collection_name": f"tombstone_{uuid.uuid4().hex[:8]}",
        }
    if backend == "pgvector":
        return {
            "backend": "pgvector",
            "connection_string": _pg_connection_string(),
            "dimensions": _EMBED_DIM,
            "metric": "cosine",
            "schema": "public",
            "table_name": f"test_tombstone_{uuid.uuid4().hex[:8]}",
            "auto_create_table": True,
            "id_type": "text",
        }
    raise AssertionError(f"unknown backend {backend}")


async def _seed(backend: InMemoryKnowledgeBackend) -> None:
    await backend.initialize()
    await backend.create_kb("d")
    await backend.put_file("d", "docs/a.md", b"# A\n\nAlpha apple uniquea.\n")
    await backend.put_file("d", "docs/b.md", b"# B\n\nAlpha banana uniqueb.\n")


async def _drop_pg_table(schema: str, table_name: str) -> None:
    """Drop a pgvector test table if it exists.

    The shared ``dataknobs_test`` database accumulates leftover tables
    from suites whose teardown did not run (a killed/timed-out session).
    ``PgVectorStore`` uses ``CREATE TABLE IF NOT EXISTS``, so a stale
    same-named table at a different ``vector(N)`` would silently shadow
    the store's intended dimension and surface as
    ``asyncpg.DataError: expected <stale> dimensions, not <echo>``.
    Calling this both *before* store construction and on teardown makes
    the table's dimension deterministic regardless of leftover state.
    """
    conn = None
    try:
        from dataknobs_common.testing import safe_sql_ident

        conn = await asyncpg.connect(_pg_connection_string())
        await conn.execute(
            f"DROP TABLE IF EXISTS "
            f"{safe_sql_ident(schema)}.{safe_sql_ident(table_name)}"
        )
    except (OSError, asyncpg.PostgresError) as exc:
        logger.warning("pgvector table drop failed: %s", exc)
    finally:
        if conn is not None:
            await conn.close()


async def _teardown_store(backend_name: str, store: Any) -> None:
    if backend_name == "chroma":
        try:
            store.client.delete_collection(name=store.collection_name)
        except Exception as exc:
            logger.warning("Chroma teardown failed: %s", exc)
    elif backend_name == "pgvector":
        await _drop_pg_table(store.schema, store.table_name)


@pytest.fixture(
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
                not is_chromadb_available(),
                reason="chromadb not installed",
            ),
        ),
        pytest.param("pgvector", id="pgvector", marks=_pgvector_marks),
    ]
)
async def wired(
    request: pytest.FixtureRequest,
) -> AsyncIterator[
    tuple[
        InMemoryKnowledgeBackend,
        RAGKnowledgeBase,
        KnowledgeIngestionManager,
    ]
]:
    backend_name = request.param
    src = InMemoryKnowledgeBackend()
    await _seed(src)
    vs_config = _vector_store_config(backend_name)
    if backend_name == "pgvector":
        # Pre-drop: a leftover same-named table at a different
        # vector(N) would shadow CREATE TABLE IF NOT EXISTS and make
        # the column dimension non-deterministic in the shared DB.
        await _drop_pg_table(vs_config["schema"], vs_config["table_name"])
    rag = await RAGKnowledgeBase.from_config(
        {
            "vector_store": vs_config,
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )
    manager = KnowledgeIngestionManager(source=src, destination=rag)
    try:
        yield src, rag, manager
    finally:
        await _teardown_store(backend_name, rag.vector_store)
        await rag.close()
        await src.close()


async def test_tombstone_roundtrip_retires_old_generation(wired) -> None:
    """A TOMBSTONE re-ingest leaves exactly one clean generation.

    After the swap commits: no chunk is still ``_stale`` (the old
    generation was physically retired), the domain count matches a
    single generation, and the revised content is queryable.
    """
    backend, rag, manager = wired

    first = await manager.ingest("d", swap_mode=IngestSwapMode.CLEAR_FIRST)
    assert first.success
    count_before = await rag.count(filter={"domain_id": "d"})
    assert count_before > 0

    await backend.put_file(
        "d", "docs/b.md", b"# B\n\nAlpha banana REVISEDtoken.\n"
    )

    result = await manager.ingest("d", swap_mode=IngestSwapMode.TOMBSTONE)

    assert result.success
    # Old generation physically gone — nothing left tombstoned.
    assert await rag.vector_store.count(filter={"_stale": True}) == 0
    # Exactly one fresh generation (same file set).
    assert await rag.count(filter={"domain_id": "d"}) == count_before
    # The live generation is queryable and none of it is tombstoned.
    hits = await rag.query("Alpha", k=10, min_similarity=_ALL)
    assert hits
    assert all(h["metadata"].get("_stale") is not True for h in hits)
    info = await backend.get_info("d")
    assert info is not None
    assert info.ingestion_status == IngestionStatus.READY


async def test_query_and_hybrid_query_both_hide_stale(wired) -> None:
    """RC3 regression: ``query()`` hides stale too, not only hybrid.

    Marking ``docs/a.md`` chunks ``_stale`` (simulating the mid-swap
    state) must hide them from **both** read paths; ``include_stale``
    brings them back.
    """
    _backend, rag, manager = wired
    await manager.ingest("d", swap_mode=IngestSwapMode.CLEAR_FIRST)

    marked = await rag.vector_store.update_metadata_where(
        {"domain_id": "d", "source_path": "docs/a.md"}, {"_stale": True}
    )
    assert marked > 0

    def _sources(results: list[dict[str, Any]]) -> set[str]:
        return {r["metadata"].get("source_path") for r in results}

    q = await rag.query("Alpha", k=10, min_similarity=_ALL)
    assert "docs/a.md" not in _sources(q)
    assert "docs/b.md" in _sources(q)

    h = await rag.hybrid_query("Alpha", k=10, min_similarity=_ALL)
    assert "docs/a.md" not in _sources(h)
    assert "docs/b.md" in _sources(h)

    # Escape hatch returns the tombstoned chunks on both paths.
    q_all = await rag.query(
        "Alpha", k=10, min_similarity=_ALL, include_stale=True
    )
    assert "docs/a.md" in _sources(q_all)
    h_all = await rag.hybrid_query(
        "Alpha", k=10, min_similarity=_ALL, include_stale=True
    )
    assert "docs/a.md" in _sources(h_all)


class _FailIngestRAG:
    """Real destination wrapper exercising the rollback path.

    Delegates the destination-side primitives the TOMBSTONE flow uses
    to a live :class:`RAGKnowledgeBase` (real store state), but raises
    from ``ingest_from_backend`` so the new generation never commits.
    Not a mock — every other call hits the real knowledge base.
    """

    def __init__(self, inner: RAGKnowledgeBase) -> None:
        self._inner = inner

    async def update_metadata_where(
        self, filter: dict[str, Any] | None, set_: dict[str, Any]
    ) -> int:
        return await self._inner.update_metadata_where(filter, set_)

    async def clear(self, filter: dict[str, Any] | None = None) -> None:
        await self._inner.clear(filter=filter)

    async def ingest_from_backend(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom: embed failed mid-swap")


async def test_tombstone_rollback_restores_old_generation(wired) -> None:
    """An upsert failure mid-swap restores the previous generation.

    The old chunks are un-tombstoned (readable again) and never
    deleted; the failure surfaces and the status is ERROR.
    """
    backend, rag, manager = wired
    await manager.ingest("d", swap_mode=IngestSwapMode.CLEAR_FIRST)
    count_before = await rag.count(filter={"domain_id": "d"})
    assert count_before > 0

    failing = KnowledgeIngestionManager(
        source=backend, destination=_FailIngestRAG(rag)
    )
    with pytest.raises(RuntimeError, match="embed failed mid-swap"):
        await failing.ingest("d", swap_mode=IngestSwapMode.TOMBSTONE)

    # Rolled back: nothing left tombstoned, old generation intact and
    # readable, status ERROR.
    assert await rag.vector_store.count(filter={"_stale": True}) == 0
    assert await rag.count(filter={"domain_id": "d"}) == count_before
    assert await rag.query("Alpha", k=10, min_similarity=_ALL)
    info = await backend.get_info("d")
    assert info is not None
    assert info.ingestion_status == IngestionStatus.ERROR


async def test_ingest_changes_tombstone_per_file_scope(wired) -> None:
    """Per-file delta + TOMBSTONE swaps only the changed file's chunks.

    ``ingest_changes(..., swap_mode=TOMBSTONE)`` routes through the
    same apply-core, so the changed file gets a crash-safe swap while
    the untouched file's chunks are never tombstoned or deleted.
    """
    backend, rag, manager = wired
    await manager.ingest("d", swap_mode=IngestSwapMode.CLEAR_FIRST)
    version = await manager.get_current_version("d")
    assert version
    a_before = await rag.count(
        filter={"domain_id": "d", "source_path": "docs/a.md"}
    )
    assert a_before > 0

    await backend.put_file(
        "d", "docs/b.md", b"# B\n\nAlpha banana DELTAtoken.\n"
    )

    result = await manager.ingest_changes(
        "d", version, swap_mode=IngestSwapMode.TOMBSTONE
    )

    assert result.success
    assert result.files_processed == 1
    # No residual tombstones; untouched file untouched; delta is live.
    assert await rag.vector_store.count(filter={"_stale": True}) == 0
    assert (
        await rag.count(
            filter={"domain_id": "d", "source_path": "docs/a.md"}
        )
        == a_before
    )
    hits = await rag.query("Alpha", k=10, min_similarity=_ALL)
    assert hits
    assert "docs/b.md" in {
        h["metadata"].get("source_path") for h in hits
    }
    assert all(h["metadata"].get("_stale") is not True for h in hits)


class _RetireProbeRAG:
    """Real destination; snapshots store state at the post-commit retire.

    The TOMBSTONE retire is the first ``clear`` carrying ``_stale: True``.
    At that instant the tombstoned old generation and the freshly
    written new generation must coexist *physically* — that overlap is
    the entire point of a zero-downtime swap. Everything delegates to a
    real :class:`RAGKnowledgeBase`; nothing is mocked.
    """

    def __init__(self, inner: RAGKnowledgeBase) -> None:
        self._inner = inner
        self.at_retire: dict[str, int] | None = None

    async def update_metadata_where(
        self, filter: dict[str, Any] | None, set_: dict[str, Any]
    ) -> int:
        return await self._inner.update_metadata_where(filter, set_)

    async def ingest_from_backend(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.ingest_from_backend(*args, **kwargs)

    async def clear(self, filter: dict[str, Any] | None = None) -> None:
        if (
            self.at_retire is None
            and filter is not None
            and filter.get("_stale") is True
        ):
            vs = self._inner.vector_store
            self.at_retire = {
                "total": await vs.count(filter=None),
                "stale": await vs.count(filter={"_stale": True}),
            }
        await self._inner.clear(filter=filter)


async def test_tombstone_preserves_old_generation_during_swap(
    wired,
) -> None:
    """Root cause #1: the old generation must survive the swap window.

    Deterministic chunk ids make a re-embed an in-place *overwrite* of
    the tombstoned rows (clearing ``_stale``), so the old generation is
    destroyed the instant the new one is written and TOMBSTONE degrades
    to a no-op. With generation-distinct ids the old rows stay
    physically present (still tombstoned) until the clean-commit retire
    — total transiently doubles and the old generation is fully
    ``_stale`` at the retire instant.
    """
    backend, rag, manager = wired
    await manager.ingest("d", swap_mode=IngestSwapMode.CLEAR_FIRST)
    count_before = await rag.vector_store.count(
        filter={"domain_id": "d"}
    )
    assert count_before > 0

    await backend.put_file(
        "d", "docs/b.md", b"# B\n\nAlpha banana REVISEDtwo.\n"
    )

    probe = _RetireProbeRAG(rag)
    pmgr = KnowledgeIngestionManager(source=backend, destination=probe)
    result = await pmgr.ingest("d", swap_mode=IngestSwapMode.TOMBSTONE)

    assert result.success
    assert probe.at_retire is not None
    # Old generation physically preserved and still hidden at retire.
    assert probe.at_retire["stale"] == count_before
    # Old (tombstoned) + new (live) coexist — total transiently doubles.
    assert probe.at_retire["total"] == 2 * count_before
    # Post-commit steady state: exactly one clean generation.
    assert await rag.vector_store.count(filter={"_stale": True}) == 0
    assert (
        await rag.vector_store.count(filter={"domain_id": "d"})
        == count_before
    )


class _PartialWriteFailRAG:
    """Writes one real chunk carrying the swap generation, then raises.

    Unlike :class:`_FailIngestRAG` (raises *before* any write), this
    reproduces the partial-write leak: a chunk physically lands in the
    store before the upsert fails. The generation token is read from
    the threaded ``extra_metadata`` — so the rollback can target
    exactly the crashed swap's rows.
    """

    def __init__(self, inner: RAGKnowledgeBase) -> None:
        self._inner = inner
        self.gen: str | None = None

    async def update_metadata_where(
        self, filter: dict[str, Any] | None, set_: dict[str, Any]
    ) -> int:
        return await self._inner.update_metadata_where(filter, set_)

    async def clear(self, filter: dict[str, Any] | None = None) -> None:
        await self._inner.clear(filter=filter)

    async def ingest_from_backend(
        self,
        source: Any,
        domain_id: str,
        *,
        config: Any = None,
        progress_callback: Any = None,
        extra_metadata: dict[str, Any] | None = None,
        file_filter: Any = None,
    ) -> Any:
        meta = dict(extra_metadata or {})
        self.gen = meta.get("_generation")
        await self._inner.vector_store.add_vectors(
            np.zeros((1, _EMBED_DIM), dtype=np.float32),
            ids=[f"{domain_id}\x1f{self.gen}\x1fpartial\x1f0"],
            metadata=[
                {
                    **meta,
                    "source_path": "docs/a.md",
                    "text": "partial",
                    "chunk_index": 0,
                }
            ],
        )
        raise RuntimeError("boom: embed failed after partial write")


async def test_tombstone_partial_write_does_not_leak(wired) -> None:
    """Root cause #2: a partial write must not leak live chunks.

    The new generation never committed, so the rollback must remove
    every chunk it wrote (targeted by ``_generation``) *and* restore
    the old generation. Today the partial chunk has no ``_stale`` key,
    so the un-tombstone rollback never touches it and it leaks live.
    """
    backend, rag, manager = wired
    await manager.ingest("d", swap_mode=IngestSwapMode.CLEAR_FIRST)
    count_before = await rag.vector_store.count(
        filter={"domain_id": "d"}
    )
    assert count_before > 0

    dest = _PartialWriteFailRAG(rag)
    failing = KnowledgeIngestionManager(source=backend, destination=dest)
    with pytest.raises(RuntimeError, match="after partial write"):
        await failing.ingest("d", swap_mode=IngestSwapMode.TOMBSTONE)

    assert await rag.vector_store.count(filter={"_stale": True}) == 0
    assert (
        await rag.vector_store.count(filter={"domain_id": "d"})
        == count_before
    )
    if dest.gen is not None:
        assert (
            await rag.vector_store.count(
                filter={"_generation": dest.gen}
            )
            == 0
        )
    assert await rag.query("Alpha", k=10, min_similarity=_ALL)
    info = await backend.get_info("d")
    assert info is not None
    assert info.ingestion_status == IngestionStatus.ERROR


async def test_ingest_changes_tombstone_partial_write_keeps_deleted_gone(
    wired,
) -> None:
    """Root cause #3: rollback must not resurrect source-deleted files.

    ``ingest_changes`` tombstones the union of modified + deleted
    paths. Today a partial-write rollback un-tombstones the *whole*
    scope, so a file deleted at the source comes back from the dead.
    The fix restores only the modified scope and unconditionally purges
    the purely-deleted scope.
    """
    backend, rag, manager = wired
    await manager.ingest("d", swap_mode=IngestSwapMode.CLEAR_FIRST)
    version = await manager.get_current_version("d")
    assert version

    await backend.delete_file("d", "docs/b.md")
    await backend.put_file(
        "d", "docs/a.md", b"# A\n\nAlpha apple CHANGEDthree.\n"
    )

    dest = _PartialWriteFailRAG(rag)
    failing = KnowledgeIngestionManager(source=backend, destination=dest)
    with pytest.raises(RuntimeError, match="after partial write"):
        await failing.ingest_changes(
            "d", version, swap_mode=IngestSwapMode.TOMBSTONE
        )

    # b.md was deleted at the source — it must stay gone (not merely
    # hidden); a.md (modified, re-embed failed) is restored & visible.
    assert (
        await rag.vector_store.count(
            filter={"domain_id": "d", "source_path": "docs/b.md"}
        )
        == 0
    )
    hits = await rag.query("Alpha", k=10, min_similarity=_ALL)
    sources = {h["metadata"].get("source_path") for h in hits}
    assert "docs/b.md" not in sources
    assert "docs/a.md" in sources
    assert await rag.vector_store.count(filter={"_stale": True}) == 0
    info = await backend.get_info("d")
    assert info is not None
    assert info.ingestion_status == IngestionStatus.ERROR


class _CrashMidSwap(BaseException):
    """Not an ``Exception`` — :meth:`_apply_tombstone`'s
    ``except Exception`` does **not** catch it, so ``_rollback_swap``
    never runs. This faithfully simulates a SIGKILL between the upsert
    and the commit: a real process crash bypasses Python-level
    ``except``/``finally`` exactly the same way, leaving the store +
    status in the genuine interrupted state on disk.
    """


class _CrashAfterOrphanWriteRAG:
    """Writes a real orphan chunk carrying the swap generation, then
    crashes hard (``BaseException``) so no rollback runs. Everything
    else delegates to a real :class:`RAGKnowledgeBase`.
    """

    def __init__(self, inner: RAGKnowledgeBase) -> None:
        self._inner = inner
        self.gen: str | None = None

    async def update_metadata_where(
        self, filter: dict[str, Any] | None, set_: dict[str, Any]
    ) -> int:
        return await self._inner.update_metadata_where(filter, set_)

    async def clear(self, filter: dict[str, Any] | None = None) -> None:
        await self._inner.clear(filter=filter)

    async def ingest_from_backend(
        self,
        source: Any,
        domain_id: str,
        *,
        config: Any = None,
        progress_callback: Any = None,
        extra_metadata: dict[str, Any] | None = None,
        file_filter: Any = None,
    ) -> Any:
        meta = dict(extra_metadata or {})
        self.gen = meta.get("_generation")
        await self._inner.vector_store.add_vectors(
            np.zeros((1, _EMBED_DIM), dtype=np.float32),
            ids=[f"{domain_id}\x1f{self.gen}\x1forphan\x1f0"],
            metadata=[
                {
                    **meta,
                    "source_path": "docs/a.md",
                    "text": "orphan",
                    "chunk_index": 0,
                }
            ],
        )
        raise _CrashMidSwap()


async def test_interrupted_swap_reconciled_on_next_ingest_changes(
    wired,
) -> None:
    """Finding #7: a crash mid-swap is reconciled by the next ingest.

    A SIGKILL between ``_upsert`` and the commit (a full-domain
    TOMBSTONE) leaves the *whole domain's* old generation tombstoned,
    orphan new-generation chunks written, and status stuck SWAPPING
    (no Python-level rollback ran). The recovery op here is a per-file
    ``ingest_changes`` for ``docs/b.md`` only — its swap scope does
    **not** cover ``docs/a.md`` or the orphan, so without reconcile:
    ``docs/a.md``'s real chunks stay tombstoned (invisible *forever*)
    and the orphan leaks live. The next ingest must reconcile first —
    restore the previous generation and drop exactly the crashed
    swap's orphans by its persisted token — before applying the delta.
    """
    backend, rag, manager = wired
    await manager.ingest("d", swap_mode=IngestSwapMode.CLEAR_FIRST)
    count_before = await rag.vector_store.count(
        filter={"domain_id": "d"}
    )
    assert count_before > 0
    version = await manager.get_current_version("d")
    assert version

    crashed = _CrashAfterOrphanWriteRAG(rag)
    cmgr = KnowledgeIngestionManager(source=backend, destination=crashed)
    with pytest.raises(_CrashMidSwap):
        await cmgr.ingest("d", swap_mode=IngestSwapMode.TOMBSTONE)

    # Genuine interrupted state: whole-domain old gen tombstoned,
    # status SWAPPING, no rollback ran.
    info = await backend.get_info("d")
    assert info is not None
    assert info.ingestion_status == IngestionStatus.SWAPPING
    assert (
        await rag.vector_store.count(
            filter={"domain_id": "d", "_stale": True}
        )
        == count_before
    )

    # Per-file recovery op: only docs/b.md changed.
    await backend.put_file(
        "d", "docs/b.md", b"# B\n\nAlpha banana RECOVERtoken.\n"
    )
    result = await manager.ingest_changes(
        "d", version, swap_mode=IngestSwapMode.TOMBSTONE
    )
    assert result.success

    # Reconcile restored docs/a.md, dropped the orphan, and the delta
    # swapped docs/b.md cleanly: no residue, a.md visible again.
    assert await rag.vector_store.count(filter={"_stale": True}) == 0
    if crashed.gen:
        assert (
            await rag.vector_store.count(
                filter={"_generation": crashed.gen}
            )
            == 0
        )
    hits = await rag.query("Alpha", k=10, min_similarity=_ALL)
    sources = {h["metadata"].get("source_path") for h in hits}
    assert "docs/a.md" in sources
    assert "docs/b.md" in sources
    assert all(h["metadata"].get("_stale") is not True for h in hits)
    info = await backend.get_info("d")
    assert info is not None
    assert info.ingestion_status == IngestionStatus.READY


async def test_reconcile_public_api_recovers_stuck_domain(wired) -> None:
    """``manager.reconcile`` recovers a never-re-ingested stuck domain.

    The real recovery mechanism for the case where no further ingest
    is scheduled. Idempotent: a second call is a no-op returning
    ``False``.
    """
    backend, rag, manager = wired
    await manager.ingest("d", swap_mode=IngestSwapMode.CLEAR_FIRST)
    count_before = await rag.vector_store.count(
        filter={"domain_id": "d"}
    )

    crashed = _CrashAfterOrphanWriteRAG(rag)
    cmgr = KnowledgeIngestionManager(source=backend, destination=crashed)
    with pytest.raises(_CrashMidSwap):
        await cmgr.ingest("d", swap_mode=IngestSwapMode.TOMBSTONE)

    reconciled = await manager.reconcile("d")
    assert reconciled is True
    assert await rag.vector_store.count(filter={"_stale": True}) == 0
    assert (
        await rag.vector_store.count(filter={"domain_id": "d"})
        == count_before
    )
    if crashed.gen:
        assert (
            await rag.vector_store.count(
                filter={"_generation": crashed.gen}
            )
            == 0
        )
    hits = await rag.query("Alpha", k=10, min_similarity=_ALL)
    assert hits
    info = await backend.get_info("d")
    assert info is not None
    assert info.ingestion_status == IngestionStatus.READY

    # Idempotent — nothing left to reconcile.
    assert await manager.reconcile("d") is False


class _NativeHybridMemoryStore(MemoryVectorStore):
    """Real ``MemoryVectorStore`` plus a native ``hybrid_search``.

    No in-tree store implements ``hybrid_search``, so the
    ``RAGKnowledgeBase`` native-fusion path is otherwise unexercised
    by real code. This is **not** a mock: ``hybrid_search`` reads the
    real stored vectors/metadata and honors the real ``filter`` /
    ``k`` it is given. It orders deterministically by chunk id (not by
    EchoProvider's pseudo-random cosine) so the test controls exactly
    which rows fall in the requested ``k``.
    """

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: Any,
        text_fields: list[str],
        k: int,
        config: Any,
        filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        from dataknobs_data import Record
        from dataknobs_data.vector.hybrid import HybridSearchResult

        eff = self._effective_filter(filter)
        rows = sorted(self.metadata_store.items(), key=lambda kv: kv[0])
        out: list[Any] = []
        for _id, meta in rows:
            if eff and not self._match_metadata_filter(meta, eff):
                continue
            out.append(
                HybridSearchResult(
                    record=Record(data=dict(meta)),
                    combined_score=1.0,
                    text_score=1.0,
                    vector_score=1.0,
                )
            )
            if len(out) >= k:
                break
        return out


async def test_native_hybrid_overfetches_past_stale() -> None:
    """Finding #4: native hybrid fusion must not under-return mid-swap.

    The native path requested exactly ``k`` from ``hybrid_search``
    then post-filtered ``_stale`` — so when tombstoned chunks rank in
    the top ``k`` it returned **fewer than k** visible results during
    a swap. The fix over-fetches ``k * _STALE_OVERFETCH`` before the
    stale gate (the same contract the vector path already had).
    """
    rag = await RAGKnowledgeBase.from_config(
        {
            "vector_store": {
                "backend": "memory",
                "dimensions": _EMBED_DIM,
            },
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )
    # Swap in the native-hybrid store (real code, same interface)
    # before any ingest so the native fusion path is exercised.
    store = _NativeHybridMemoryStore({"dimensions": _EMBED_DIM})
    await store.initialize()
    rag.vector_store = store
    src = InMemoryKnowledgeBackend()
    await src.initialize()
    await src.create_kb("d")
    # 12 files → 12 chunks; ids sort as docs/f00..docs/f11.
    for i in range(12):
        await src.put_file(
            "d", f"docs/f{i:02d}.md", f"# F{i}\n\nAlpha token{i}.\n".encode()
        )
    mgr = KnowledgeIngestionManager(source=src, destination=rag)
    await mgr.ingest("d", swap_mode=IngestSwapMode.CLEAR_FIRST)
    total = await store.count(filter={"domain_id": "d"})
    assert total == 12

    # Tombstone the 6 lexicographically-first chunks — exactly the
    # ones the native store returns first for any small k.
    first_six = sorted(store.metadata_store)[:6]
    for cid in first_six:
        meta = store.metadata_store[cid]
        store.metadata_store[cid] = {**meta, "_stale": True}

    k = 5
    hits = await rag.hybrid_query(
        "Alpha", k=k, fusion_strategy="native", min_similarity=_ALL
    )
    # Pre-fix: store returns first 5 (all stale) → 0 visible.
    assert len(hits) == k
    assert all(h["metadata"].get("_stale") is not True for h in hits)

    await rag.close()
    await src.close()


async def test_count_excludes_stale_by_default(wired) -> None:
    """Finding #5: ``count()`` must exclude tombstoned chunks.

    Mid-swap, ``count(filter)`` delegated straight to the store and so
    reported old+new (double). It now excludes ``_stale`` by default
    (``visible = count(f) - count(f | _stale)``); ``include_stale=True``
    restores the prior single-count.
    """
    _backend, rag, manager = wired
    await manager.ingest("d", swap_mode=IngestSwapMode.CLEAR_FIRST)
    total = await rag.vector_store.count(filter={"domain_id": "d"})
    assert total > 0

    marked = await rag.vector_store.update_metadata_where(
        {"domain_id": "d", "source_path": "docs/a.md"}, {"_stale": True}
    )
    assert marked > 0

    visible = total - marked
    assert await rag.count(filter={"domain_id": "d"}) == visible
    assert (
        await rag.count(filter={"domain_id": "d"}, include_stale=True)
        == total
    )


async def test_clear_existing_deprecated_still_works(wired) -> None:
    """``clear_existing=`` warns but maps to the equivalent swap mode."""
    _backend, rag, manager = wired

    with pytest.warns(DeprecationWarning, match="clear_existing"):
        r1 = await manager.ingest("d", clear_existing=True)
    assert r1.success
    assert await rag.count(filter={"domain_id": "d"}) > 0

    # False still maps to APPEND (no full-domain clear) and still warns.
    with pytest.warns(DeprecationWarning, match="clear_existing"):
        r2 = await manager.ingest("d", clear_existing=False)
    assert r2.success
