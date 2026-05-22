"""Embedder rate-limit seam.

The production consumer ingests through an OpenAI-rate-limited
embedder; unbounded per-chunk embed calls fail the whole ingest. The
infra (:class:`dataknobs_common.ratelimit.RateLimiter` +
``InMemoryRateLimiter``) already exists — this seam only wires an
optional, injected ``rate_limiter`` through the ingest path
(:meth:`KnowledgeIngestionManager.__init__` /
:meth:`RAGKnowledgeBase.ingest_from_backend` →
``_embed_and_store_chunks``). ``None`` (the default, Ollama-local)
keeps today's behaviour byte-for-byte.

Real constructs only: real ``InMemoryRateLimiter`` (and a thin
recording subclass — a real impl on the real code path, not a mock),
``EchoProvider`` embeddings, ``MemoryVectorStore``,
``InMemoryKnowledgeBackend``.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.knowledge import RAGKnowledgeBase
from dataknobs_bots.knowledge.ingestion import KnowledgeIngestionManager
from dataknobs_bots.knowledge.storage import InMemoryKnowledgeBackend
from dataknobs_common.ratelimit import (
    InMemoryRateLimiter,
    RateLimit,
    RateLimiterConfig,
    create_rate_limiter,
)


class RecordingRateLimiter(InMemoryRateLimiter):
    """Real ``InMemoryRateLimiter`` that records the category names it
    is asked to acquire. Subclass (not a mock) so the real acquire
    code path still runs — it proves the seam *and* exercises the
    actual limiter."""

    def __init__(self) -> None:
        super().__init__(
            RateLimiterConfig(
                default_rates=[RateLimit(limit=10_000, interval=60.0)]
            )
        )
        self.acquired: list[str] = []

    async def acquire(
        self,
        name: str = "default",
        weight: int = 1,
        timeout: float | None = None,
    ) -> None:
        self.acquired.append(name)
        await super().acquire(name, weight, timeout)


async def _make_kb() -> RAGKnowledgeBase:
    return await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )


@pytest.fixture
async def backend() -> InMemoryKnowledgeBackend:
    be = InMemoryKnowledgeBackend()
    await be.initialize()
    await be.create_kb("d1")
    await be.put_file("d1", "a.md", b"# Alpha\n\nAlpha body text.\n")
    await be.put_file("d1", "b.md", b"# Beta\n\nBeta body text.\n")
    return be


@pytest.mark.asyncio
async def test_manager_rate_limiter_acquired_once_per_embedded_chunk(
    backend: InMemoryKnowledgeBackend,
) -> None:
    """Every ingest-path embed is preceded by ``acquire("embed")``."""
    kb = await _make_kb()
    rl = RecordingRateLimiter()
    mgr = KnowledgeIngestionManager(
        source=backend, destination=kb, rate_limiter=rl
    )

    result = await mgr.ingest("d1")

    assert result.chunks_created >= 2
    assert rl.acquired == ["embed"] * result.chunks_created


@pytest.mark.asyncio
async def test_ingest_from_backend_accepts_rate_limiter(
    backend: InMemoryKnowledgeBackend,
) -> None:
    """The seam is reachable directly on
    :meth:`RAGKnowledgeBase.ingest_from_backend`, not only via the
    manager."""
    kb = await _make_kb()
    rl = RecordingRateLimiter()

    stats = await kb.ingest_from_backend(
        backend, "d1", rate_limiter=rl
    )

    assert stats["total_chunks"] >= 2
    assert rl.acquired == ["embed"] * stats["total_chunks"]


@pytest.mark.asyncio
async def test_real_inmemory_rate_limiter_consumes_capacity(
    backend: InMemoryKnowledgeBackend,
) -> None:
    """A plain (non-recording) real ``InMemoryRateLimiter`` is actually
    exercised once per chunk — proven via its own status, no timing
    assertion (deterministic)."""
    kb = await _make_kb()
    rl = create_rate_limiter({"rates": [{"limit": 1000, "interval": 60}]})
    mgr = KnowledgeIngestionManager(
        source=backend, destination=kb, rate_limiter=rl
    )

    result = await mgr.ingest("d1")

    status = await rl.get_status("embed")
    assert status.current_count == result.chunks_created


@pytest.mark.asyncio
async def test_no_rate_limiter_is_unchanged(
    backend: InMemoryKnowledgeBackend,
) -> None:
    """Regression guard: omitting ``rate_limiter`` (the default) leaves
    ingest behaviour identical to prior releases."""
    kb = await _make_kb()
    mgr = KnowledgeIngestionManager(source=backend, destination=kb)

    result = await mgr.ingest("d1")

    assert result.chunks_created >= 2
    assert result.errors == []
