"""A refused ``domain_id`` is not a missing domain.

:meth:`KnowledgeIngestionManager.ingest_if_changed` wraps its
change-detection call in ``except ValueError`` to turn "this domain does
not exist" into a benign ``None``. The identifier guards raise
:class:`~dataknobs_common.paths.PathEscapeError`, which **is** a
``ValueError`` — deliberately, so one ``except`` reaches every refusal in
this series — so that broad clause swallowed them.

The result was the worst available outcome: the caller asked for an
ingest, the name was refused, and the manager logged "Domain not found"
and returned ``None``. No ingest happened, nothing raised, and the one
message that named the real problem was replaced by one naming a
different problem the operator would then go looking for.

The narrower clause has to come first. Ordering is the whole fix, which
is why it is pinned: reintroducing the bug requires only deleting three
lines that look redundant beside a broader clause that would still catch
"the domain is missing" perfectly well.
"""

from __future__ import annotations

import pytest

from dataknobs_bots.knowledge import KnowledgeIngestionManager, RAGKnowledgeBase
from dataknobs_bots.knowledge.storage import InMemoryKnowledgeBackend
from dataknobs_common.paths import SegmentEscapeError


async def _make_rag() -> RAGKnowledgeBase:
    return await RAGKnowledgeBase.from_config(
        {
            "vector_store": {"backend": "memory", "dimensions": 384},
            "embedding_provider": "echo",
            "embedding_model": "test",
        }
    )


@pytest.fixture
async def manager():
    backend = InMemoryKnowledgeBackend()
    await backend.initialize()
    await backend.create_kb("d")
    await backend.put_file("d", "a.md", b"# A\n")
    rag = await _make_rag()
    yield KnowledgeIngestionManager(source=backend, destination=rag)
    await rag.close()
    await backend.close()


@pytest.mark.parametrize("domain_id", ["acme/content", "_scoped", ".."])
async def test_an_inadmissible_domain_id_raises_rather_than_reporting_absence(
    manager: KnowledgeIngestionManager, domain_id: str
) -> None:
    with pytest.raises(SegmentEscapeError):
        await manager.ingest_if_changed(domain_id, last_version="whatever")


async def test_a_genuinely_missing_domain_still_reports_absence(
    manager: KnowledgeIngestionManager,
) -> None:
    """The behaviour the broad clause was there for, unchanged.

    A legal name for a knowledge base that does not exist is the case
    ``None`` is the right answer to, and narrowing the refusal ahead of
    it must not have cost that.
    """
    assert await manager.ingest_if_changed("no-such-domain", last_version="whatever") is None
