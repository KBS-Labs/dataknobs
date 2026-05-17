"""Per-file delta ingest.

Real constructs only — ``InMemoryKnowledgeBackend`` is the documented
testing backend, ``RAGKnowledgeBase`` uses the in-memory vector store
and the ``echo`` embedding provider; no mocks.

``KnowledgeIngestionManager.ingest_changes`` re-embeds only the files
that changed since a captured canonical version (``get_checksum``),
purges chunks for deleted *and* modified files, and falls back to a
full re-ingest when the version predates the backend's snapshot
retention. The full and per-file paths share one apply-core, so swap
semantics cannot diverge between them.
"""

from __future__ import annotations

import logging

import pytest

from dataknobs_bots.knowledge import (
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
)
from dataknobs_bots.knowledge.ingestion import IngestionResult
from dataknobs_bots.knowledge.storage import InMemoryKnowledgeBackend


async def _make_rag() -> RAGKnowledgeBase:
    return await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )


async def _seed(backend: InMemoryKnowledgeBackend) -> None:
    await backend.initialize()
    await backend.create_kb("d")
    await backend.put_file("d", "docs/a.md", b"# A\n\nAlpha.\n")
    await backend.put_file("d", "docs/b.md", b"# B\n\nBeta.\n")
    await backend.put_file("d", "docs/c.md", b"# C\n\nGamma.\n")


def _progress_capture() -> tuple[list[str], object]:
    seen: list[str] = []

    def cb(path: str, chunks: int) -> None:
        seen.append(path)

    return seen, cb


@pytest.fixture
async def wired() -> tuple[
    InMemoryKnowledgeBackend, RAGKnowledgeBase, KnowledgeIngestionManager
]:
    backend = InMemoryKnowledgeBackend()
    await _seed(backend)
    rag = await _make_rag()
    manager = KnowledgeIngestionManager(source=backend, destination=rag)
    yield backend, rag, manager
    await rag.close()
    await backend.close()


async def test_ingest_changes_reembeds_only_modified(wired) -> None:
    """A single modified file in N → only that file re-embedded.

    Proven precisely by the progress callback (the manager invokes it
    once per file that actually went through the embed pipeline) and
    by ``files_processed``.
    """
    backend, rag, manager = wired

    first = await manager.ingest("d")
    assert first.files_processed == 3
    total_before = await rag.count(filter={"domain_id": "d"})

    version = await manager.get_current_version("d")
    assert version

    await backend.put_file("d", "docs/b.md", b"# B\n\nBeta REVISED.\n")

    seen, cb = _progress_capture()
    result = await manager.ingest_changes(
        "d", version, progress_callback=cb
    )

    assert result.files_processed == 1
    assert result.files_deleted == 0
    assert result.success
    # Exactly one file went through the embed pipeline, and it was b.md
    # (progress reports the source display URI, suffixed by the path).
    assert len(seen) == 1
    assert seen[0].endswith("docs/b.md")
    # a.md / c.md were never re-enumerated, so their chunks are intact
    # and the total is unchanged (old b chunk purged, new one added).
    assert await rag.count(filter={"domain_id": "d"}) == total_before
    assert await rag.count(
        filter={"domain_id": "d", "source_path": "docs/b.md"}
    ) == 1
    assert await rag.count(
        filter={"domain_id": "d", "source_path": "docs/a.md"}
    ) == 1


async def test_ingest_changes_purges_deleted_file_chunks(wired) -> None:
    """A file removed at the source → its chunks are deleted, others stay."""
    backend, rag, manager = wired

    await manager.ingest("d")
    version = await manager.get_current_version("d")
    assert version

    assert await backend.delete_file("d", "docs/c.md") is True

    seen, cb = _progress_capture()
    result = await manager.ingest_changes(
        "d", version, progress_callback=cb
    )

    assert result.files_deleted == 1
    assert result.files_processed == 0
    assert result.success
    assert seen == []  # nothing re-embedded — pure deletion
    assert await rag.count(
        filter={"domain_id": "d", "source_path": "docs/c.md"}
    ) == 0
    assert await rag.count(
        filter={"domain_id": "d", "source_path": "docs/a.md"}
    ) == 1
    assert await rag.count(filter={"domain_id": "d"}) == 2


async def test_ingest_changes_invalid_version_falls_back_to_full(
    wired, caplog
) -> None:
    """An unresolvable version → full re-ingest, not a silent skip."""
    backend, rag, manager = wired

    await manager.ingest("d")

    with caplog.at_level(logging.WARNING):
        result = await manager.ingest_changes(
            "d", "not-a-retained-snapshot-id"
        )

    assert any(
        "predates snapshot retention" in r.message for r in caplog.records
    )
    # Full re-ingest of every file (the fallback ran ingest()).
    assert result.files_processed == 3
    assert result.success
    assert await rag.count(filter={"domain_id": "d"}) == 3


async def test_ingest_changes_noop_when_unchanged(wired) -> None:
    """No change since the captured version → successful no-op."""
    backend, rag, manager = wired

    await manager.ingest("d")
    version = await manager.get_current_version("d")
    assert version

    seen, cb = _progress_capture()
    result = await manager.ingest_changes(
        "d", version, progress_callback=cb
    )

    assert result.files_processed == 0
    assert result.files_deleted == 0
    assert result.success
    assert seen == []
    info = await backend.get_info("d")
    assert info is not None
    assert info.ingestion_status.value == "ready"


def test_ingestion_result_files_deleted_serialized() -> None:
    """``files_deleted`` is a first-class field and survives to_dict."""
    r = IngestionResult(domain_id="d", files_deleted=4)
    assert r.files_deleted == 4
    assert r.to_dict()["files_deleted"] == 4
