"""Item 118 Option B: KnowledgeIngestionManager cross-domain isolation.

Pre-fix, ``KnowledgeIngestionManager.ingest()`` with
``clear_existing=True`` called ``self._destination.clear()`` (unscoped),
so refreshing one domain wiped every other domain's chunks in any
multi-tenant shared store.  The smoking-gun log line in ``ingest()``
already says ``"Cleared existing vectors for domain: %s"`` — the
intent was domain-scoped; the implementation could not honor it
because the underlying ``VectorStore.clear()`` API didn't support
filters.

This test pins cross-domain isolation post-fix.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.knowledge import (
    InMemoryKnowledgeBackend,
    IngestSwapMode,
    KnowledgeIngestionManager,
    RAGKnowledgeBase,
)


async def _make_kb() -> RAGKnowledgeBase:
    return await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )


@pytest.mark.asyncio
async def test_reingest_one_domain_preserves_others():
    """Re-ingesting one domain with clear_existing=True does not wipe siblings."""
    kb = await _make_kb()

    backend = InMemoryKnowledgeBackend()
    await backend.initialize()

    # Pre-seed two domains with distinct content.
    await backend.create_kb("domain-a")
    await backend.put_file(
        "domain-a", "doc.md", b"# A heading\n\nDomain A body content.\n"
    )
    await backend.create_kb("domain-b")
    await backend.put_file(
        "domain-b", "doc.md", b"# B heading\n\nDomain B body content.\n"
    )

    mgr = KnowledgeIngestionManager(source=backend, destination=kb)

    # Ingest domain-a first (the store is empty so the clear
    # is a no-op). Pin its chunk count.
    await mgr.ingest("domain-a", swap_mode=IngestSwapMode.CLEAR_FIRST)
    domain_a_count_before = await kb.vector_store.count(
        filter={"domain_id": "domain-a"}
    )
    assert domain_a_count_before > 0, (
        "expected domain-a to have chunks after first ingest"
    )

    # Re-ingest domain-b with CLEAR_FIRST. Pre-fix this
    # wipes every domain in the shared store (including domain-a);
    # post-fix it scopes the clear to ``domain_id=domain-b`` only.
    await mgr.ingest("domain-b", swap_mode=IngestSwapMode.CLEAR_FIRST)

    # domain-a chunks must survive.
    domain_a_count_after = await kb.vector_store.count(
        filter={"domain_id": "domain-a"}
    )
    assert domain_a_count_after == domain_a_count_before, (
        "Re-ingesting domain-b wiped domain-a chunks "
        f"(expected {domain_a_count_before}, got {domain_a_count_after})"
    )
    # domain-b chunks present.
    domain_b_count_after = await kb.vector_store.count(
        filter={"domain_id": "domain-b"}
    )
    assert domain_b_count_after > 0
