"""Async-correctness tests for ``RAGKnowledgeBase`` document ingest.

The ``load_*_document`` methods read files from disk, and
``load_from_directory`` (with ``config=None``) additionally probes the
filesystem and reads/parses ``knowledge_base.{yaml,yml,json}`` via
``KnowledgeBaseConfig.load``. Run directly on the event loop those reads
stall it; the fix offloads them via ``asyncio.to_thread``. These
reproduce-first tests wrap the awaited ingest call in ``assert_no_blocking()``
(FAIL before the offload, PASS after), using an in-memory vector store + echo
embedder so no external service is needed. Functional assertions guard the
chunk counts. No mocks.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataknobs_bots.knowledge.rag import RAGKnowledgeBase
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

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


@requires_blockbuster
async def test_load_yaml_document_does_not_block(tmp_path: Path) -> None:
    doc = tmp_path / "doc.yaml"
    doc.write_text("title: Doc\nbody: content here for the knowledge base\n")
    kb = await _make_kb()
    try:
        with assert_no_blocking():
            chunks = await kb.load_yaml_document(doc)
        assert chunks > 0
    finally:
        await kb.close()


@requires_blockbuster
async def test_load_csv_document_does_not_block(tmp_path: Path) -> None:
    doc = tmp_path / "doc.csv"
    doc.write_text("question,answer\nWhat is it?,A knowledge base entry.\n")
    kb = await _make_kb()
    try:
        with assert_no_blocking():
            chunks = await kb.load_csv_document(doc)
        assert chunks > 0
    finally:
        await kb.close()


@requires_blockbuster
async def test_load_from_directory_auto_config_does_not_block(tmp_path: Path) -> None:
    # Exercise the ``config is None`` branch: KnowledgeBaseConfig.load probes
    # the filesystem and reads/parses the config file, both of which block the
    # loop unless offloaded.
    (tmp_path / "knowledge_base.yaml").write_text(
        "name: test-corpus\ndefault_chunking:\n  max_chunk_size: 500\n"
    )
    (tmp_path / "guide.md").write_text("# Guide\n\nThis is a test guide.")
    kb = await _make_kb()
    try:
        with assert_no_blocking():
            results = await kb.load_from_directory(tmp_path)
        assert results["total_files"] >= 1
        assert results["total_chunks"] > 0
    finally:
        await kb.close()
