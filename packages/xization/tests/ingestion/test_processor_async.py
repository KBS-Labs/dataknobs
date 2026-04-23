"""Async-specific processor tests for the Phase 1 refactor.

Existing sync ``test_ingestion.py::TestDirectoryProcessor`` coverage
is preserved unchanged by the sync ``process()`` wrapper. These tests
add coverage for the new async API and the widened constructor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataknobs_xization.ingestion import (
    DirectoryProcessor,
    FilePatternConfig,
    KnowledgeBaseConfig,
    LocalDocumentSource,
    ProcessedDocument,
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Minimal mixed-type corpus."""
    (tmp_path / "intro.md").write_text("# Intro\n\nWelcome.\n")
    (tmp_path / "data.json").write_text('[{"title": "A"}, {"title": "B"}]')
    return tmp_path


def test_process_async_yields_same_documents_as_process(
    corpus: Path,
) -> None:
    """Collected list equality between async and sync entrypoints."""
    import asyncio

    config = KnowledgeBaseConfig(name="t")

    async def _run_async() -> list[ProcessedDocument]:
        processor = DirectoryProcessor(config, corpus)
        return [doc async for doc in processor.process_async()]

    async_docs = asyncio.run(_run_async())

    processor_sync_wrapped = DirectoryProcessor(config, corpus)
    sync_docs = list(processor_sync_wrapped.process())

    assert len(async_docs) == len(sync_docs)
    for a, s in zip(async_docs, sync_docs, strict=True):
        assert a.source_file == s.source_file
        assert a.document_type == s.document_type
        assert len(a.chunks) == len(s.chunks)


@pytest.mark.asyncio
async def test_constructor_accepts_document_source(corpus: Path) -> None:
    source = LocalDocumentSource(corpus)
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), source)
    docs = [doc async for doc in processor.process_async()]
    assert len(docs) == 2


@pytest.mark.asyncio
async def test_constructor_sets_root_dir_none_for_document_source(
    corpus: Path,
) -> None:
    source = LocalDocumentSource(corpus)
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), source)
    assert processor.root_dir is None
    assert processor.source is source


@pytest.mark.asyncio
async def test_process_async_respects_excludes(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("# keep\n")
    (tmp_path / "drop.md").write_text("# drop\n")
    config = KnowledgeBaseConfig(name="t", exclude_patterns=["drop.md"])
    processor = DirectoryProcessor(config, tmp_path)
    docs = [doc async for doc in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].source_file.endswith("keep.md")


# ---------------------------------------------------------------------------
# Multi-extension dispatch — regression coverage for the extensions the
# pre-unification ``KnowledgeIngestionManager`` handled (markdown / txt /
# yaml / csv) that the default-pattern list initially dropped when we moved
# to the unified pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_async_dispatches_markdown_extension(
    tmp_path: Path,
) -> None:
    """``.markdown`` (long-form) routes through the markdown pipeline."""
    (tmp_path / "guide.markdown").write_text("# Guide\n\nBody.\n")
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), tmp_path)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].document_type == "markdown"
    assert docs[0].source_file.endswith("guide.markdown")
    assert docs[0].chunk_count >= 1
    assert not docs[0].has_errors


@pytest.mark.asyncio
async def test_process_async_dispatches_txt_as_markdown(
    tmp_path: Path,
) -> None:
    """``.txt`` is chunked as single-section markdown — matches the
    pre-unification ``KnowledgeIngestionManager._load_text`` path."""
    (tmp_path / "notes.txt").write_text("Some plain notes content.\n")
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), tmp_path)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].document_type == "markdown"
    assert docs[0].source_file.endswith("notes.txt")
    assert docs[0].chunk_count >= 1
    assert not docs[0].has_errors


@pytest.mark.asyncio
async def test_process_async_dispatches_yaml_via_transformer(
    tmp_path: Path,
) -> None:
    """YAML is transformed to markdown via ``ContentTransformer``."""
    pytest.importorskip("yaml")
    (tmp_path / "settings.yaml").write_text(
        "name: demo\nvalues:\n  - a\n  - b\n"
    )
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), tmp_path)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].document_type == "markdown"
    assert docs[0].source_file.endswith("settings.yaml")
    assert docs[0].chunk_count >= 1
    assert not docs[0].has_errors


@pytest.mark.asyncio
async def test_process_async_dispatches_yml_via_transformer(
    tmp_path: Path,
) -> None:
    """``.yml`` is treated as YAML — same dispatch as ``.yaml``."""
    pytest.importorskip("yaml")
    (tmp_path / "short.yml").write_text("key: value\n")
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), tmp_path)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].document_type == "markdown"


@pytest.mark.asyncio
async def test_process_async_dispatches_csv_via_transformer(
    tmp_path: Path,
) -> None:
    """CSV rows become markdown sections via ``ContentTransformer``."""
    (tmp_path / "people.csv").write_text(
        "name,role\nAlice,Engineer\nBob,Designer\n"
    )
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), tmp_path)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].document_type == "markdown"
    assert docs[0].source_file.endswith("people.csv")
    assert docs[0].chunk_count >= 1
    assert not docs[0].has_errors


@pytest.mark.asyncio
async def test_process_async_skips_unsupported_extensions(
    tmp_path: Path,
) -> None:
    """Unsupported extensions are skipped and counted in
    ``files_skipped`` — the default-pattern list doesn't match them, so
    they don't even enter the dispatch path."""
    (tmp_path / "keep.md").write_text("# Keep\n")
    # ``.xyz`` is not in the default patterns → never reaches dispatch.
    (tmp_path / "ignore.xyz").write_text("not handled")
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), tmp_path)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].source_file.endswith("keep.md")
    # ``.xyz`` isn't in the default patterns, so it never reaches the
    # extension-dispatch block — it just isn't enumerated by the
    # source. ``files_skipped`` tracks files that DID pass the source
    # enumeration.
    assert processor.files_skipped == 0


@pytest.mark.asyncio
async def test_process_async_explicit_pattern_routes_unsupported_ext(
    tmp_path: Path,
) -> None:
    """When an explicit pattern enumerates a file with an extension the
    dispatcher doesn't know, the file is enumerated and then skipped
    via the unsupported-extension branch, incrementing
    ``files_skipped``."""
    (tmp_path / "keep.md").write_text("# Keep\n")
    (tmp_path / "stray.xyz").write_text("not handled")
    config = KnowledgeBaseConfig(
        name="t",
        patterns=[
            FilePatternConfig(pattern="**/*.md"),
            FilePatternConfig(pattern="**/*.xyz"),
        ],
    )
    processor = DirectoryProcessor(config, tmp_path)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].source_file.endswith("keep.md")
    assert processor.files_skipped == 1


@pytest.mark.asyncio
async def test_process_async_markdown_like_mix(tmp_path: Path) -> None:
    """Mixed markdown-like files (``.md`` / ``.markdown`` / ``.txt``)
    are all emitted as ``document_type='markdown'``."""
    (tmp_path / "a.md").write_text("# A\n")
    (tmp_path / "b.markdown").write_text("# B\n")
    (tmp_path / "c.txt").write_text("Plain C\n")
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), tmp_path)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 3
    assert all(d.document_type == "markdown" for d in docs)


# ---------------------------------------------------------------------------
# JSONL remote streaming — line-by-line parse from an async byte iterator.
#
# Tests a remote :class:`DocumentSource` (non-local) whose ``read_streaming``
# yields bytes in arbitrary piece sizes. The processor's
# ``_stream_jsonl_from_remote`` must assemble lines across piece boundaries
# without buffering the whole file.
# ---------------------------------------------------------------------------


import asyncio as _asyncio
from collections.abc import AsyncIterator as _AsyncIterator
from dataknobs_xization.ingestion.source import (  # noqa: E402
    DocumentFileRef,
    DocumentSource,
)


class _RemoteJSONLSource:
    """In-memory :class:`DocumentSource` whose ``read_streaming`` yields
    bytes in small configurable pieces.

    NOT a :class:`LocalDocumentSource` — that matters because the
    processor's ``_stream_json_chunks`` has a short-circuit for local
    sources that bypasses ``_stream_jsonl_from_remote``. This stub
    forces the remote path under test.
    """

    def __init__(self, files: dict[str, bytes], piece_size: int = 4) -> None:
        self._files = files
        self._piece_size = piece_size

    async def iter_files(
        self, patterns: Any
    ) -> _AsyncIterator[DocumentFileRef]:
        for path, data in self._files.items():
            yield DocumentFileRef(
                path=path,
                size_bytes=len(data),
                source_uri=f"memory://{path}",
            )

    async def read_bytes(self, ref: DocumentFileRef) -> bytes:
        return self._files[ref.path]

    async def read_streaming(
        self, ref: DocumentFileRef, chunk_size: int = 8192
    ) -> _AsyncIterator[bytes]:
        data = self._files[ref.path]
        # Ignore chunk_size — slice at piece_size to stress the
        # line-reassembly logic in ``_stream_jsonl_from_remote``.
        for i in range(0, len(data), self._piece_size):
            # Await between pieces to simulate network backpressure.
            await _asyncio.sleep(0)
            yield data[i : i + self._piece_size]


@pytest.mark.asyncio
async def test_remote_jsonl_streaming_splits_lines_across_pieces() -> None:
    """Pieces yielded smaller than a single JSONL line — the processor
    still yields one chunk per line.
    """
    content = b'{"title": "A"}\n{"title": "B"}\n{"title": "C"}\n'
    source = _RemoteJSONLSource({"data.jsonl": content}, piece_size=3)
    # Source conforms to the protocol.
    assert isinstance(source, DocumentSource)
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), source)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].document_type == "jsonl"
    assert docs[0].chunk_count == 3
    assert not docs[0].has_errors


@pytest.mark.asyncio
async def test_remote_jsonl_streaming_tolerates_missing_trailing_newline() -> None:
    """Last line may lack a trailing newline — remote path emits it."""
    content = b'{"title": "A"}\n{"title": "B"}'
    source = _RemoteJSONLSource({"data.jsonl": content}, piece_size=7)
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), source)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].chunk_count == 2


@pytest.mark.asyncio
async def test_remote_jsonl_streaming_skips_malformed_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Malformed lines are logged and skipped — a single bad line must
    not abort the batch.
    """
    content = (
        b'{"title": "good_1"}\n'
        b"this is not json\n"
        b'{"title": "good_2"}\n'
    )
    source = _RemoteJSONLSource({"data.jsonl": content}, piece_size=5)
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), source)
    with caplog.at_level(
        "WARNING", logger="dataknobs_xization.ingestion.processor"
    ):
        docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    # Two good lines emitted, malformed one skipped.
    assert docs[0].chunk_count == 2
    assert not docs[0].has_errors  # file-level errors distinct from line skips
    assert any(
        "Skipping malformed JSONL line" in record.message
        for record in caplog.records
    ), f"Expected malformed-line warning; got {caplog.records}"


@pytest.mark.asyncio
async def test_files_skipped_counts_config_files(tmp_path: Path) -> None:
    """``DirectoryProcessor.files_skipped`` counts ``knowledge_base.*``
    config files that were enumerated but intentionally not ingested.
    """
    (tmp_path / "guide.md").write_text("# Guide\n")
    (tmp_path / "knowledge_base.json").write_text('{"name": "t"}')

    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), tmp_path)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].source_file.endswith("guide.md")
    assert processor.files_skipped == 1


@pytest.mark.asyncio
async def test_files_skipped_counts_excluded_paths(tmp_path: Path) -> None:
    """Excluded files are enumerated by the source (they match the
    default patterns) and then skipped by the exclude filter —
    bumping ``files_skipped``.
    """
    (tmp_path / "keep.md").write_text("# Keep\n")
    (tmp_path / "drop.md").write_text("# Drop\n")
    (tmp_path / "also_drop.md").write_text("# Also drop\n")

    config = KnowledgeBaseConfig(
        name="t", exclude_patterns=["drop.md", "also_drop.md"]
    )
    processor = DirectoryProcessor(config, tmp_path)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert processor.files_skipped == 2


@pytest.mark.asyncio
async def test_files_skipped_resets_between_runs(tmp_path: Path) -> None:
    """Counter is reset at the start of each :meth:`process_async` so
    successive runs don't accumulate stale skip counts.
    """
    (tmp_path / "guide.md").write_text("# Guide\n")
    (tmp_path / "knowledge_base.json").write_text('{"name": "t"}')

    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), tmp_path)
    _ = [d async for d in processor.process_async()]
    assert processor.files_skipped == 1
    # Second run must start at 0 before counting.
    _ = [d async for d in processor.process_async()]
    assert processor.files_skipped == 1


@pytest.mark.asyncio
async def test_remote_jsonl_streaming_ignores_blank_lines() -> None:
    """Blank / whitespace-only lines are skipped silently."""
    content = (
        b'{"title": "A"}\n'
        b"\n"
        b"   \n"
        b'{"title": "B"}\n'
    )
    source = _RemoteJSONLSource({"data.jsonl": content}, piece_size=2)
    processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), source)
    docs = [d async for d in processor.process_async()]
    assert len(docs) == 1
    assert docs[0].chunk_count == 2
