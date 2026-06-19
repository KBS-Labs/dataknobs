"""Async-correctness tests for ``LocalDocumentSource``.

The source's async methods do filesystem I/O: ``iter_files`` walks the
tree with ``Path.glob`` + per-path ``stat``, and ``read_bytes`` /
``read_streaming`` read file contents. Run directly on the event loop
those calls stall it; each is offloaded via ``asyncio.to_thread``.

These reproduce-first tests wrap each awaited operation in
``assert_no_blocking()`` — they FAIL against a synchronous implementation
(the detector catches the blocking syscall) and PASS once the call is
offloaded. A temp directory is the whole fixture, so they run in the
always-on unit pass. Functional assertions guard the enumerated refs and
read contents. No mocks.

``iter_files`` is the load-bearing case here: its glob/stat walk used to
run on the loop (caught end-to-end by the bots ingest-offload suite); this
guards the fix at the source layer where the defect lived. ``read_bytes``
/ ``read_streaming`` were already offloaded — covered here so the whole
``DocumentSource`` read surface has a loop-safety guard.
"""

from __future__ import annotations

from pathlib import Path

from dataknobs_common.testing import assert_no_blocking, requires_blockbuster
from dataknobs_xization.ingestion.source import (
    DocumentFileRef,
    LocalDocumentSource,
)


def _build_corpus(root: Path) -> None:
    (root / "top.md").write_text("# Top\n")
    (root / "data.json").write_text('{"k": "v"}')
    sub = root / "docs"
    sub.mkdir()
    (sub / "guide.md").write_text("# Guide\n")
    (sub / "nested").mkdir()
    (sub / "nested" / "deep.md").write_text("# Deep\n")


@requires_blockbuster
async def test_iter_files_does_not_block(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    source = LocalDocumentSource(tmp_path)

    refs: list[DocumentFileRef] = []
    with assert_no_blocking():
        async for ref in source.iter_files(["**/*.md"]):
            refs.append(ref)

    assert sorted(r.path for r in refs) == [
        "docs/guide.md",
        "docs/nested/deep.md",
        "top.md",
    ]


@requires_blockbuster
async def test_read_bytes_does_not_block(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    source = LocalDocumentSource(tmp_path)
    refs = [ref async for ref in source.iter_files(["top.md"])]
    assert len(refs) == 1

    with assert_no_blocking():
        data = await source.read_bytes(refs[0])

    assert data == (tmp_path / "top.md").read_bytes()


@requires_blockbuster
async def test_read_streaming_does_not_block(tmp_path: Path) -> None:
    payload = b"x" * 50_000
    (tmp_path / "big.bin").write_bytes(payload)
    source = LocalDocumentSource(tmp_path)
    refs = [ref async for ref in source.iter_files(["big.bin"])]
    assert len(refs) == 1

    chunks: list[bytes] = []
    with assert_no_blocking():
        async for chunk in source.read_streaming(refs[0], chunk_size=8192):
            chunks.append(chunk)

    assert b"".join(chunks) == payload
