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

The streaming-JSON local path (``_stream_json_chunks`` handing the
on-disk path to a synchronous chunker generator) is covered too: that
generator ``open``/``gzip.open``s the file and reads forward on every
``__next__``, blocking the loop on each chunk pull. The fix drives the
sync generator on a worker thread and pumps chunks across a bounded
queue. The streaming-JSON tests below drive a ``.jsonl``, a
``.jsonl.gz``, and a plain ``.json`` array (forced down the streaming
branch by a tiny threshold) under ``assert_no_blocking()``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from dataknobs_common.testing import assert_no_blocking, requires_blockbuster
from dataknobs_xization.ingestion import (
    DirectoryProcessor,
    KnowledgeBaseConfig,
    ProcessedDocument,
)
from dataknobs_xization.ingestion import processor as processor_module


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


@requires_blockbuster
async def test_stream_json_local_jsonl_does_not_block(tmp_path: Path) -> None:
    # A ``.jsonl`` file always takes the streaming branch (``is_jsonl``),
    # regardless of size. One JSON object per line -> one chunk per line.
    #
    # Note: ``blockbuster`` patches ``io.*.read`` but NOT ``readline`` /
    # line-iteration, so this plain-text path's pre-fix blocking is its
    # one blind spot -- this test is GREEN even against the old blocking
    # code. The same ``_stream_jsonl`` code path *is* proven off-loaded by
    # ``test_stream_json_local_gzip_does_not_block`` below (gzip reads go
    # through the patched ``BufferedReader.read``). This case stands as
    # functional coverage that the thread-pump yields correct jsonl output.
    lines = [
        {"id": 1, "text": "first knowledge entry"},
        {"id": 2, "text": "second knowledge entry"},
        {"id": 3, "text": "third knowledge entry"},
    ]
    (tmp_path / "data.jsonl").write_text(
        "".join(f"{json.dumps(obj)}\n" for obj in lines)
    )

    with assert_no_blocking():
        docs = await _process(tmp_path)

    assert len(docs) == 1
    assert not docs[0].errors
    assert len(docs[0].chunks) == len(lines)


@requires_blockbuster
async def test_stream_json_local_gzip_does_not_block(tmp_path: Path) -> None:
    # A gzipped ``.jsonl.gz`` exercises the streaming branch AND the
    # chunker's gzip handling -- the fix must keep decompression off-loop.
    lines = [
        {"id": 1, "text": "alpha"},
        {"id": 2, "text": "beta"},
    ]
    payload = "".join(f"{json.dumps(obj)}\n" for obj in lines)
    with gzip.open(tmp_path / "data.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(payload)

    with assert_no_blocking():
        docs = await _process(tmp_path)

    assert len(docs) == 1
    assert not docs[0].errors
    # gzip decoded correctly off-loop -> the entries survived round-trip.
    assert len(docs[0].chunks) == len(lines)
    texts = " ".join(chunk["text"] for chunk in docs[0].chunks)
    assert "alpha" in texts
    assert "beta" in texts


@requires_blockbuster
async def test_stream_json_local_large_array_does_not_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a plain ``.json`` array down the streaming branch without a
    # 10 MiB fixture by lowering the threshold to a single byte.
    monkeypatch.setattr(processor_module, "STREAMING_THRESHOLD_BYTES", 1)

    array = [
        {"id": 1, "text": "first array record"},
        {"id": 2, "text": "second array record"},
    ]
    (tmp_path / "records.json").write_text(json.dumps(array))

    with assert_no_blocking():
        docs = await _process(tmp_path)

    assert len(docs) == 1
    assert not docs[0].errors
    # The streaming JSON-array path expands objects into field-level
    # chunks, so assert content survived (not an exact count).
    assert docs[0].chunks
    texts = " ".join(chunk["text"] for chunk in docs[0].chunks)
    assert "first array record" in texts
    assert "second array record" in texts
