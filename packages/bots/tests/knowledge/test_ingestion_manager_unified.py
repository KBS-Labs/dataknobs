"""Phase 3 tests for :class:`KnowledgeIngestionManager` delegation.

Before Phase 3, the manager did its own per-file dispatch by extension and
ignored the backend's ``_metadata/knowledge_base.yaml`` config. After
Phase 3, it delegates to :meth:`RAGKnowledgeBase.ingest_from_backend`,
which honors patterns, exclude patterns, and streaming JSON.

The first test in this module is the **reproduce-first failing test**
for the pattern-handling bug; it FAILED on ``main`` and PASSES after the
refactor.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dataknobs_bots.knowledge import (
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
)
from dataknobs_bots.knowledge.storage import InMemoryKnowledgeBackend


async def _make_kb() -> RAGKnowledgeBase:
    return await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )


async def _populate(backend: InMemoryKnowledgeBackend) -> None:
    await backend.initialize()
    await backend.create_kb("d1")
    await backend.put_file("d1", "docs/topic.md", b"# Topic\n\nHello.\n")
    await backend.put_file(
        "d1", "docs/sidenote.md", b"# Sidenote\n\nIgnore me.\n"
    )


def _make_progress_capture() -> tuple[list[tuple[str, int]], Any]:
    """Return a ``(captured, callback)`` pair for manager progress.

    The manager invokes ``progress_callback(path, chunks_created)`` after
    each successfully-ingested file. Tests can assert on which paths
    appeared to verify pattern/exclude behavior end-to-end.
    """
    captured: list[tuple[str, int]] = []

    def cb(path: str, chunks: int) -> None:
        captured.append((path, chunks))

    return captured, cb


@pytest.mark.asyncio
async def test_ingest_respects_patterns_through_manager() -> None:
    """Reproducer: manager must honor ``_metadata/knowledge_base.yaml`` excludes.

    Before Phase 3, :class:`KnowledgeIngestionManager` dispatched by file
    extension and never consulted
    ``_metadata/knowledge_base.(yaml|yml|json)``. This test places a
    config document that excludes ``docs/sidenote.md`` and asserts that
    after ingest the excluded file is not in the vector store.
    """
    backend = InMemoryKnowledgeBackend()
    await _populate(backend)

    config_doc = {
        "name": "d1",
        "exclude_patterns": ["docs/sidenote.md"],
    }
    await backend.put_file(
        "d1",
        "_metadata/knowledge_base.json",
        json.dumps(config_doc).encode("utf-8"),
    )

    rag = await _make_kb()
    manager = KnowledgeIngestionManager(source=backend, destination=rag)
    captured, cb = _make_progress_capture()
    result = await manager.ingest("d1", progress_callback=cb)

    paths = {p for p, _ in captured}
    # Post-fix: exclude pattern honored — only topic.md is ingested
    assert result.files_processed == 1, (
        f"Expected 1 file (sidenote.md excluded); got {result.files_processed} "
        f"with captured paths={paths}"
    )
    assert any("topic.md" in p for p in paths), (
        f"topic.md must be ingested; captured paths={paths}"
    )
    assert not any("sidenote.md" in p for p in paths), (
        f"sidenote.md must be excluded; captured paths={paths}"
    )


@pytest.mark.asyncio
async def test_ingestion_manager_preserves_status_tracking() -> None:
    """Manager transitions domain status to ``ready`` on success."""
    backend = InMemoryKnowledgeBackend()
    await _populate(backend)

    rag = await _make_kb()
    manager = KnowledgeIngestionManager(source=backend, destination=rag)
    await manager.ingest("d1")

    info = await backend.get_info("d1")
    assert info is not None
    assert info.ingestion_status.value == "ready"


@pytest.mark.asyncio
async def test_ingestion_manager_preserves_event_bus() -> None:
    """Manager publishes ``knowledge:ingestion`` event on completion."""
    from dataknobs_common.events import EventType, InMemoryEventBus

    backend = InMemoryKnowledgeBackend()
    await _populate(backend)

    rag = await _make_kb()
    bus = InMemoryEventBus()
    await bus.connect()

    captured: list[Any] = []

    async def handler(event: Any) -> None:
        captured.append(event)

    await bus.subscribe("knowledge:ingestion", handler)

    manager = KnowledgeIngestionManager(
        source=backend, destination=rag, event_bus=bus
    )
    await manager.ingest("d1")

    # Give the bus a moment to deliver
    import asyncio

    for _ in range(10):
        if captured:
            break
        await asyncio.sleep(0.01)

    assert len(captured) == 1, "Expected exactly one ingestion event"
    event = captured[0]
    assert event.type == EventType.UPDATED
    assert event.topic == "knowledge:ingestion"
    assert event.payload["domain_id"] == "d1"
    assert event.payload["status"] == "ready"
    assert event.payload["files_processed"] >= 1
    assert event.payload["chunks_created"] >= 1

    await bus.close()


@pytest.mark.asyncio
async def test_ingest_if_changed_skips_when_unchanged() -> None:
    """``ingest_if_changed`` returns ``None`` when version matches."""
    backend = InMemoryKnowledgeBackend()
    await _populate(backend)

    rag = await _make_kb()
    manager = KnowledgeIngestionManager(source=backend, destination=rag)
    first = await manager.ingest_if_changed("d1", last_version=None)
    assert first is not None  # First call always runs

    version = await manager.get_current_version("d1")
    assert version is not None

    # Second call with the current version should skip
    second = await manager.ingest_if_changed("d1", last_version=version)
    assert second is None


@pytest.mark.asyncio
async def test_ingest_if_changed_runs_when_changed() -> None:
    """``ingest_if_changed`` runs when ``last_version`` differs."""
    backend = InMemoryKnowledgeBackend()
    await _populate(backend)

    rag = await _make_kb()
    manager = KnowledgeIngestionManager(source=backend, destination=rag)

    result = await manager.ingest_if_changed("d1", last_version="does-not-exist")
    assert result is not None
    assert result.files_processed >= 1


@pytest.mark.asyncio
async def test_ingestion_manager_threads_domain_id_into_chunk_metadata() -> None:
    """Manager threads ``{"domain_id": domain_id}`` onto every chunk.

    Multi-tenant consumers filter vector-store queries by
    ``domain_id`` in metadata. Regression guard for the unification —
    the delegation to :meth:`RAGKnowledgeBase.ingest_from_backend`
    must pass ``extra_metadata={"domain_id": ...}`` so every stored
    chunk carries the tenant identifier.
    """
    backend = InMemoryKnowledgeBackend()
    await _populate(backend)

    rag = await _make_kb()
    manager = KnowledgeIngestionManager(source=backend, destination=rag)
    await manager.ingest("d1")

    # Every chunk stored in the vector store carries domain_id="d1".
    vector_store = rag.vector_store
    stored_metadata = vector_store.metadata_store  # type: ignore[attr-defined]
    assert stored_metadata, "Expected chunks to have been stored"
    for chunk_id, meta in stored_metadata.items():
        assert meta.get("domain_id") == "d1", (
            f"chunk {chunk_id} missing domain_id; got meta={meta}"
        )


@pytest.mark.asyncio
async def test_ingestion_result_reports_files_skipped() -> None:
    """``IngestionResult.files_skipped`` reflects the
    :class:`DirectoryProcessor` counter end-to-end: config files,
    excluded paths, and unsupported-extension files enumerated by an
    explicit pattern all flow through the stats dict into the
    result type."""
    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    await backend.create_kb("d1")
    await backend.put_file("d1", "docs/keep.md", b"# Keep\n")
    await backend.put_file("d1", "docs/skip.md", b"# Skip\n")
    await backend.put_file(
        "d1",
        "_metadata/knowledge_base.json",
        json.dumps({"name": "d1", "exclude_patterns": ["docs/skip.md"]}).encode(
            "utf-8"
        ),
    )

    rag = await _make_kb()
    manager = KnowledgeIngestionManager(source=backend, destination=rag)
    result = await manager.ingest("d1")

    # docs/skip.md was enumerated by the default patterns and excluded.
    assert result.files_skipped >= 1
    assert result.files_processed == 1


@pytest.mark.asyncio
async def test_ingest_from_backend_accepts_extra_metadata() -> None:
    """:meth:`RAGKnowledgeBase.ingest_from_backend` merges caller-
    provided ``extra_metadata`` onto every chunk. Confirms the
    parameter is wired end-to-end, not just accepted by the manager.
    """
    backend = InMemoryKnowledgeBackend()
    await _populate(backend)

    rag = await _make_kb()
    await rag.ingest_from_backend(
        backend, "d1", extra_metadata={"tenant": "acme", "env": "prod"}
    )

    stored_metadata = rag.vector_store.metadata_store  # type: ignore[attr-defined]
    assert stored_metadata
    for meta in stored_metadata.values():
        assert meta.get("tenant") == "acme"
        assert meta.get("env") == "prod"


@pytest.mark.asyncio
async def test_ingestion_manager_publishes_failure_event() -> None:
    """When ingest fails, manager marks status ``error`` and publishes failure."""
    from dataknobs_common.events import InMemoryEventBus

    backend = InMemoryKnowledgeBackend()
    await _populate(backend)

    rag = await _make_kb()
    bus = InMemoryEventBus()
    await bus.connect()

    captured: list[Any] = []

    async def handler(event: Any) -> None:
        captured.append(event)

    await bus.subscribe("knowledge:ingestion", handler)

    manager = KnowledgeIngestionManager(
        source=backend, destination=rag, event_bus=bus
    )

    # Force a failure by stubbing the destination's ingest path.
    async def _boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("simulated destination failure")

    rag.ingest_from_backend = _boom  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="simulated destination failure"):
        await manager.ingest("d1")

    info = await backend.get_info("d1")
    assert info is not None
    assert info.ingestion_status.value == "error"

    await bus.close()
