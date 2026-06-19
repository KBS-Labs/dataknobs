"""Async-correctness tests for ``RAGKnowledgeBase`` document ingest.

The ``load_*_document`` methods read files from disk. Run directly on the
event loop those reads stall it; the fix offloads them via
``asyncio.to_thread``. These reproduce-first tests wrap the awaited ingest
call in ``assert_no_blocking()`` (FAIL before the offload, PASS after),
using an in-memory vector store + echo embedder so no external service is
needed. Functional assertions guard the chunk counts. No mocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataknobs_bots.knowledge.rag import RAGKnowledgeBase
from dataknobs_common.testing import assert_no_blocking, is_blockbuster_available

requires_blockbuster = pytest.mark.skipif(
    not is_blockbuster_available(),
    reason="blockbuster not installed",
)

_CONFIG = {
    "vector_store": {"backend": "memory", "dimensions": 384},
    "embedding_provider": "echo",
    "embedding_model": "test",
}


async def _make_kb() -> RAGKnowledgeBase:
    return await RAGKnowledgeBase.from_config(dict(_CONFIG))


@requires_blockbuster
async def test_load_markdown_document_does_not_block(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("# Title\n\nSome content for the knowledge base.")
    kb = await _make_kb()
    try:
        with assert_no_blocking():
            chunks = await kb.load_markdown_document(doc)
        assert chunks > 0
    finally:
        await kb.close()


@requires_blockbuster
async def test_load_json_document_does_not_block(tmp_path: Path) -> None:
    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps({"title": "Doc", "body": "content here"}))
    kb = await _make_kb()
    try:
        with assert_no_blocking():
            chunks = await kb.load_json_document(doc)
        assert chunks > 0
    finally:
        await kb.close()
