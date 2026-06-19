"""Async-correctness tests for ``DirectoryProcessor.process_async``.

The async ingest path reads file bytes through the offloaded
``DocumentSource`` and then converts YAML/CSV to markdown via
``ContentTransformer``. The transformer disambiguates "path vs inline
content" with ``Path(content).exists()`` — a blocking ``os.stat`` that
runs on the event loop even when the content is already in memory. The
fix offloads the transform call via ``asyncio.to_thread`` (mirroring
``RAGKnowledgeBase.load_yaml_document`` / ``load_csv_document``).

These reproduce-first tests drive ``process_async`` over a YAML and a CSV
file under ``assert_no_blocking()``: they FAIL against the inline
transform call (the detector catches the ``stat``) and PASS once it is
offloaded. A temp directory is the whole fixture. Functional assertions
confirm chunks are produced without errors. No mocks.

Note: the streaming-JSON local path (``_stream_json_chunks`` handing the
on-disk path to a synchronous chunker generator) is a separate, more
involved blocking site tracked for its own fix and is intentionally not
covered here.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.testing import assert_no_blocking, requires_blockbuster
from dataknobs_xization.ingestion import (
    DirectoryProcessor,
    KnowledgeBaseConfig,
    ProcessedDocument,
)


async def _process(directory: Path) -> list[ProcessedDocument]:
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), directory)
    return [doc async for doc in processor.process_async()]


@requires_blockbuster
async def test_process_async_yaml_does_not_block(tmp_path: Path) -> None:
    (tmp_path / "data.yaml").write_text(
        "title: Doc\nbody: content for the knowledge base\n"
    )
    with assert_no_blocking():
        docs = await _process(tmp_path)

    assert len(docs) == 1
    assert not docs[0].errors
    assert docs[0].chunks


@requires_blockbuster
async def test_process_async_csv_does_not_block(tmp_path: Path) -> None:
    (tmp_path / "table.csv").write_text(
        "question,answer\nWhat is it?,A knowledge base entry.\n"
    )
    with assert_no_blocking():
        docs = await _process(tmp_path)

    assert len(docs) == 1
    assert not docs[0].errors
    assert docs[0].chunks
