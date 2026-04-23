"""Tests for :mod:`dataknobs_xization.ingestion.source`."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dataknobs_xization.ingestion.source import (
    BackendDocumentSource,
    DocumentFileRef,
    DocumentSource,
    LocalDocumentSource,
)


async def _collect_refs(
    source: LocalDocumentSource, patterns: list[str]
) -> list[DocumentFileRef]:
    refs: list[DocumentFileRef] = []
    async for ref in source.iter_files(patterns):
        refs.append(ref)
    return refs


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Build a tiny corpus with md/json/nested files."""
    (tmp_path / "top.md").write_text("# Top\n")
    (tmp_path / "data.json").write_text('{"k": "v"}')
    sub = tmp_path / "docs"
    sub.mkdir()
    (sub / "guide.md").write_text("# Guide\n")
    (sub / "notes.md").write_text("# Notes\n")
    (sub / "nested").mkdir()
    (sub / "nested" / "deep.md").write_text("# Deep\n")
    return tmp_path


@pytest.mark.asyncio
async def test_local_iter_files_matches_path_glob(corpus: Path) -> None:
    source = LocalDocumentSource(corpus)
    refs = await _collect_refs(source, ["**/*.md"])
    yielded = sorted(r.path for r in refs)
    expected = sorted(
        str(p.relative_to(corpus).as_posix())
        for p in corpus.glob("**/*.md")
        if p.is_file()
    )
    assert yielded == expected


@pytest.mark.asyncio
async def test_local_iter_files_deduplicates_across_patterns(
    corpus: Path,
) -> None:
    """Source yields duplicates; processor-level dedup is separate."""
    source = LocalDocumentSource(corpus)
    refs = await _collect_refs(source, ["**/*.md", "docs/*.md"])
    guide_count = sum(1 for r in refs if r.path == "docs/guide.md")
    # Both patterns match docs/guide.md → yielded twice. DirectoryProcessor
    # is responsible for dedup via _collect_files_async.
    assert guide_count == 2


@pytest.mark.asyncio
async def test_local_iter_files_skips_directories(corpus: Path) -> None:
    source = LocalDocumentSource(corpus)
    refs = await _collect_refs(source, ["**/*"])
    # "docs/nested" would match **/* but is a directory
    assert all("nested" != Path(r.path).name for r in refs)
    # And the directory entry itself shouldn't appear
    assert not any(r.path == "docs/nested" for r in refs)


@pytest.mark.asyncio
async def test_local_read_bytes_matches_file_contents(corpus: Path) -> None:
    source = LocalDocumentSource(corpus)
    refs = await _collect_refs(source, ["top.md"])
    assert len(refs) == 1
    assert await source.read_bytes(refs[0]) == (corpus / "top.md").read_bytes()


@pytest.mark.asyncio
async def test_local_read_streaming_yields_correct_chunks(
    corpus: Path,
) -> None:
    payload = b"x" * 50_000
    (corpus / "big.bin").write_bytes(payload)
    source = LocalDocumentSource(corpus)
    refs = await _collect_refs(source, ["big.bin"])
    assert len(refs) == 1

    collected = b""
    async for piece in source.read_streaming(refs[0], chunk_size=8192):
        collected += piece
    assert collected == payload


@pytest.mark.asyncio
async def test_local_read_streaming_chunk_size_respected(
    corpus: Path,
) -> None:
    payload = b"y" * 20_000
    (corpus / "pay.bin").write_bytes(payload)
    source = LocalDocumentSource(corpus)
    refs = await _collect_refs(source, ["pay.bin"])

    sizes: list[int] = []
    async for piece in source.read_streaming(refs[0], chunk_size=4096):
        sizes.append(len(piece))
    # all but last chunk should be 4096
    assert all(s == 4096 for s in sizes[:-1])
    assert sum(sizes) == len(payload)


@pytest.mark.asyncio
async def test_local_read_streaming_does_not_block_other_async_work(
    corpus: Path,
) -> None:
    """Regression: :meth:`LocalDocumentSource.read_streaming` must not
    block the event loop between chunks.

    An earlier implementation used a thread-producer +
    ``run_coroutine_threadsafe(queue.put(...)).result()`` pattern that
    could deadlock the event loop when the thread was waiting for a
    put while the loop was blocked on the next read. The current
    implementation dispatches each read via :func:`asyncio.to_thread`,
    which yields control between reads. This test exercises that
    property by interleaving the stream with a concurrent
    ``asyncio.sleep`` that must make progress during the stream.
    """
    payload = b"a" * 200_000
    (corpus / "bulk.bin").write_bytes(payload)
    source = LocalDocumentSource(corpus)
    refs = await _collect_refs(source, ["bulk.bin"])

    ticks: list[float] = []
    stop = asyncio.Event()

    async def _ticker() -> None:
        while not stop.is_set():
            ticks.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0)  # yield

    ticker_task = asyncio.create_task(_ticker())
    try:
        collected = b""
        async for piece in source.read_streaming(refs[0], chunk_size=4096):
            collected += piece
        assert collected == payload
    finally:
        stop.set()
        await ticker_task

    # The ticker must have made multiple passes — proves the event
    # loop was not starved while the stream was running. One tick
    # would be consistent with the stream blocking the loop start to
    # finish; we require more than a trivial handful.
    assert len(ticks) > 5, (
        f"Event loop appears to have been blocked during streaming "
        f"(only {len(ticks)} ticker iterations)"
    )


@pytest.mark.asyncio
async def test_local_read_streaming_concurrent_iterations(
    corpus: Path,
) -> None:
    """Two simultaneous streams from the same source interleave
    cleanly via ``asyncio.gather`` — regression guard for the
    thread/queue deadlock pattern that would serialize or hang when
    multiple consumers ran on the same loop."""
    payload_a = b"A" * 50_000
    payload_b = b"B" * 50_000
    (corpus / "a.bin").write_bytes(payload_a)
    (corpus / "b.bin").write_bytes(payload_b)

    source = LocalDocumentSource(corpus)
    ref_a = (await _collect_refs(source, ["a.bin"]))[0]
    ref_b = (await _collect_refs(source, ["b.bin"]))[0]

    async def _drain(ref: DocumentFileRef) -> bytes:
        buf = b""
        async for piece in source.read_streaming(ref, chunk_size=4096):
            buf += piece
        return buf

    got_a, got_b = await asyncio.wait_for(
        asyncio.gather(_drain(ref_a), _drain(ref_b)),
        timeout=5.0,
    )
    assert got_a == payload_a
    assert got_b == payload_b


@pytest.mark.asyncio
async def test_local_read_streaming_closes_on_early_exit(
    corpus: Path,
) -> None:
    """Early consumer exit must release the underlying file handle
    via the generator's ``finally``. If the close were skipped, tests
    on strict filesystems (Windows CI) would fail to remove the
    tempdir. We verify by reopening the file in write mode after the
    early exit — a leaked handle would still hold a read lock on some
    platforms, but on POSIX it mainly serves as a behavioral check
    that the generator finalizes without error.
    """
    payload = b"x" * 100_000
    (corpus / "abort.bin").write_bytes(payload)
    source = LocalDocumentSource(corpus)
    refs = await _collect_refs(source, ["abort.bin"])

    stream = source.read_streaming(refs[0], chunk_size=4096)
    first = await stream.__anext__()
    assert first == b"x" * 4096
    await stream.aclose()

    # Generator finalized cleanly; file is writable again.
    (corpus / "abort.bin").write_bytes(b"done")
    assert (corpus / "abort.bin").read_bytes() == b"done"


@pytest.mark.asyncio
async def test_local_document_file_ref_source_uri_is_file_uri(
    corpus: Path,
) -> None:
    source = LocalDocumentSource(corpus)
    refs = await _collect_refs(source, ["top.md"])
    assert refs[0].source_uri.startswith("file://")
    assert refs[0].source_uri.endswith("top.md")


def test_local_document_source_is_runtime_checkable(tmp_path: Path) -> None:
    source = LocalDocumentSource(tmp_path)
    assert isinstance(source, DocumentSource)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_process_sync_wrapper_inside_running_loop_raises(
    corpus: Path,
) -> None:
    """Sync ``process()`` cannot be called from inside a running loop.

    ``asyncio.run()`` raises RuntimeError; the wrapper does not
    attempt any loop-nesting workaround.
    """
    from dataknobs_xization.ingestion import (
        DirectoryProcessor,
        KnowledgeBaseConfig,
    )

    async def _call() -> None:
        processor = DirectoryProcessor(KnowledgeBaseConfig(name="t"), corpus)
        with pytest.raises(RuntimeError):
            list(processor.process())

    asyncio.run(_call())


# ---------------------------------------------------------------------------
# BackendDocumentSource — direct unit tests
#
# ``BackendDocumentSource`` adapts any object satisfying the
# ``KnowledgeResourceBackend`` structural protocol. xization does not depend
# on ``dataknobs-bots``, so the tests below implement minimal real backends
# that satisfy the structural shape (``list_files`` + ``get_file`` + optional
# ``stream_file`` / ``uri``). These are real code exercising real code paths
# in ``BackendDocumentSource`` — not mocks or fakes that bypass logic.
# ---------------------------------------------------------------------------


@dataclass
class _StandardKF:
    """Minimal ``KnowledgeFile``-like record with ``size_bytes``."""

    path: str
    size_bytes: int = 0


@dataclass
class _LegacyKF:
    """``KnowledgeFile``-like record exposing ``size`` only.

    Exercises the fallback tier in ``BackendDocumentSource.iter_files``
    that looks for ``size`` when ``size_bytes`` is missing.
    """

    path: str
    size: int = 0


@dataclass
class _BareKF:
    """``KnowledgeFile``-like record with no size attribute at all."""

    path: str


class _StaticBackend:
    """Backend with ``list_files`` + ``get_file`` but no ``stream_file``.

    Used to exercise ``BackendDocumentSource.read_streaming``'s fallback
    path (chunk the result of ``get_file`` in-memory).
    """

    def __init__(
        self,
        files: dict[str, bytes],
        file_records: list[Any] | None = None,
    ) -> None:
        self._files = files
        self._records = file_records or [
            _StandardKF(path=p, size_bytes=len(c)) for p, c in files.items()
        ]

    async def list_files(self, domain_id: str) -> list[Any]:
        return list(self._records)

    async def get_file(self, domain_id: str, path: str) -> bytes | None:
        return self._files.get(path)


class _StreamingBackend(_StaticBackend):
    """Backend whose ``stream_file`` yields the file in chunks.

    Exercises the preferred-stream path in
    ``BackendDocumentSource.read_streaming``.
    """

    async def stream_file(
        self, domain_id: str, path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes] | None:
        data = self._files.get(path)
        if data is None:
            return None

        async def _gen() -> AsyncIterator[bytes]:
            for offset in range(0, len(data), chunk_size):
                yield data[offset : offset + chunk_size]

        return _gen()


class _NullStreamBackend(_StaticBackend):
    """``stream_file`` exists but always returns ``None``.

    ``BackendDocumentSource`` must fall back to ``get_file`` + in-memory
    chunking when the backend signals "no streaming available" this way.
    """

    async def stream_file(
        self, domain_id: str, path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes] | None:
        return None


class _UriBackend(_StaticBackend):
    """Backend exposing a ``uri`` attribute used for ``source_uri``."""

    uri = "s3://my-bucket/prefix"


async def _collect_backend_refs(
    source: BackendDocumentSource, patterns: list[str]
) -> list[DocumentFileRef]:
    refs: list[DocumentFileRef] = []
    async for ref in source.iter_files(patterns):
        refs.append(ref)
    return refs


# --- Glob / pattern-matching semantics ------------------------------------


@pytest.mark.parametrize(
    "path,pattern,expected",
    [
        # ** — any depth including zero
        ("top.md", "**/*.md", True),
        ("docs/guide.md", "**/*.md", True),
        ("docs/nested/deep.md", "**/*.md", True),
        ("docs/guide.txt", "**/*.md", False),
        # Anchored subtree
        ("docs/guide.md", "docs/**", True),
        ("docs/nested/deep.md", "docs/**", True),
        ("other/guide.md", "docs/**", False),
        # Bare * matches within a single segment only
        ("top.md", "*.md", True),
        ("docs/guide.md", "*.md", False),
        # ? matches exactly one non-slash char
        ("a.md", "?.md", True),
        ("ab.md", "?.md", False),
        ("a/b.md", "?.md", False),
        # Literal path
        ("exact/match.json", "exact/match.json", True),
        ("exact/mismatch.json", "exact/match.json", False),
        # Prefix anchor with wildcard
        ("docs/guide.md", "docs/*.md", True),
        ("docs/nested/deep.md", "docs/*.md", False),
    ],
)
def test_backend_matches_glob_semantics(
    path: str, pattern: str, expected: bool
) -> None:
    assert BackendDocumentSource._matches(path, pattern) is expected


# --- iter_files -----------------------------------------------------------


@pytest.fixture
def backend_corpus_files() -> dict[str, bytes]:
    return {
        "top.md": b"# Top\n",
        "data.json": b'{"k": "v"}',
        "docs/guide.md": b"# Guide\n",
        "docs/notes.md": b"# Notes\n",
        "docs/nested/deep.md": b"# Deep\n",
    }


@pytest.mark.asyncio
async def test_backend_iter_files_glob_filters(
    backend_corpus_files: dict[str, bytes],
) -> None:
    backend = _StaticBackend(backend_corpus_files)
    source = BackendDocumentSource(backend, "test-domain")

    refs = await _collect_backend_refs(source, ["**/*.md"])
    paths = sorted(r.path for r in refs)
    assert paths == [
        "docs/guide.md",
        "docs/nested/deep.md",
        "docs/notes.md",
        "top.md",
    ]


@pytest.mark.asyncio
async def test_backend_iter_files_empty_patterns_yields_all(
    backend_corpus_files: dict[str, bytes],
) -> None:
    backend = _StaticBackend(backend_corpus_files)
    source = BackendDocumentSource(backend, "test-domain")

    refs = await _collect_backend_refs(source, [])
    paths = sorted(r.path for r in refs)
    assert paths == sorted(backend_corpus_files.keys())


@pytest.mark.asyncio
async def test_backend_iter_files_non_matching_pattern_yields_none() -> None:
    backend = _StaticBackend({"top.md": b"x"})
    source = BackendDocumentSource(backend, "d")

    refs = await _collect_backend_refs(source, ["**/*.xyz"])
    assert refs == []


@pytest.mark.asyncio
async def test_backend_iter_files_size_bytes_from_record() -> None:
    backend = _StaticBackend(
        {"a.md": b"hello"},
        file_records=[_StandardKF(path="a.md", size_bytes=5)],
    )
    source = BackendDocumentSource(backend, "d")

    refs = await _collect_backend_refs(source, ["*.md"])
    assert len(refs) == 1
    assert refs[0].size_bytes == 5


@pytest.mark.asyncio
async def test_backend_iter_files_size_falls_back_to_size_attr() -> None:
    """When record lacks ``size_bytes``, fall back to ``size``."""
    backend = _StaticBackend(
        {"a.md": b"hello"},
        file_records=[_LegacyKF(path="a.md", size=5)],
    )
    source = BackendDocumentSource(backend, "d")

    refs = await _collect_backend_refs(source, ["*.md"])
    assert len(refs) == 1
    assert refs[0].size_bytes == 5


@pytest.mark.asyncio
async def test_backend_iter_files_size_defaults_to_minus_one() -> None:
    """When record has neither ``size_bytes`` nor ``size``, defaults to -1."""
    backend = _StaticBackend(
        {"a.md": b"hello"},
        file_records=[_BareKF(path="a.md")],
    )
    source = BackendDocumentSource(backend, "d")

    refs = await _collect_backend_refs(source, ["*.md"])
    assert len(refs) == 1
    assert refs[0].size_bytes == -1


# --- _backend_uri / source_uri -------------------------------------------


@pytest.mark.asyncio
async def test_backend_source_uri_uses_class_name_by_default() -> None:
    backend = _StaticBackend({"top.md": b"x"})
    source = BackendDocumentSource(backend, "my-domain")
    refs = await _collect_backend_refs(source, ["*.md"])
    assert refs[0].source_uri == "_StaticBackend://my-domain/top.md"


@pytest.mark.asyncio
async def test_backend_source_uri_uses_backend_uri_attr() -> None:
    backend = _UriBackend({"top.md": b"x"})
    source = BackendDocumentSource(backend, "my-domain")
    refs = await _collect_backend_refs(source, ["*.md"])
    assert refs[0].source_uri == "s3://my-bucket/prefix/my-domain/top.md"


# --- read_bytes ----------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_read_bytes_returns_file_contents() -> None:
    backend = _StaticBackend({"top.md": b"hello world"})
    source = BackendDocumentSource(backend, "d")
    refs = await _collect_backend_refs(source, ["*.md"])
    assert await source.read_bytes(refs[0]) == b"hello world"


@pytest.mark.asyncio
async def test_backend_read_bytes_raises_when_backend_returns_none() -> None:
    """``read_bytes`` must raise ``FileNotFoundError`` if the backend
    returns ``None`` — e.g., race where the file was deleted between
    ``list_files`` and ``read_bytes``.
    """
    backend = _StaticBackend({})
    source = BackendDocumentSource(backend, "d")
    ref = DocumentFileRef(path="missing.md", size_bytes=0, source_uri="x")
    with pytest.raises(FileNotFoundError):
        await source.read_bytes(ref)


# --- read_streaming ------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_read_streaming_prefers_backend_stream_file() -> None:
    payload = b"x" * 20_000
    backend = _StreamingBackend({"big.bin": payload})
    source = BackendDocumentSource(backend, "d")
    refs = await _collect_backend_refs(source, ["*.bin"])

    collected = b""
    async for piece in source.read_streaming(refs[0], chunk_size=4096):
        collected += piece
    assert collected == payload


@pytest.mark.asyncio
async def test_backend_read_streaming_falls_back_without_stream_file() -> None:
    """Backend without ``stream_file`` falls back to ``get_file`` +
    in-memory chunking.
    """
    payload = b"y" * 20_000
    backend = _StaticBackend({"pay.bin": payload})
    source = BackendDocumentSource(backend, "d")
    refs = await _collect_backend_refs(source, ["*.bin"])

    sizes: list[int] = []
    collected = b""
    async for piece in source.read_streaming(refs[0], chunk_size=4096):
        sizes.append(len(piece))
        collected += piece
    assert collected == payload
    assert sum(sizes) == len(payload)
    assert all(s <= 4096 for s in sizes)


@pytest.mark.asyncio
async def test_backend_read_streaming_falls_back_when_stream_returns_none() -> None:
    """When ``stream_file`` exists but returns ``None``, fall back to
    ``get_file``.
    """
    payload = b"z" * 10_000
    backend = _NullStreamBackend({"pay.bin": payload})
    source = BackendDocumentSource(backend, "d")
    refs = await _collect_backend_refs(source, ["*.bin"])

    collected = b""
    async for piece in source.read_streaming(refs[0], chunk_size=2048):
        collected += piece
    assert collected == payload


# --- properties / protocol conformance -----------------------------------


def test_backend_document_source_is_runtime_checkable() -> None:
    backend = _StaticBackend({})
    source = BackendDocumentSource(backend, "d")
    assert isinstance(source, DocumentSource)


def test_backend_document_source_exposes_backend_and_domain() -> None:
    backend = _StaticBackend({})
    source = BackendDocumentSource(backend, "my-domain")
    assert source.backend is backend
    assert source.domain_id == "my-domain"


# --- list_files prefix optimization ---------------------------------------


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("docs/*.md", "docs/"),
        ("docs/api/*.md", "docs/api/"),
        ("docs/guide.md", "docs/guide.md"),
        ("*.md", ""),
        ("**/*.md", ""),
        ("?.md", ""),
        ("[abc].md", ""),
    ],
)
def test_literal_prefix(pattern: str, expected: str) -> None:
    assert BackendDocumentSource._literal_prefix(pattern) == expected


@pytest.mark.parametrize(
    "patterns,expected",
    [
        # All share a prefix, bounded at ``/``.
        (["docs/*.md", "docs/*.txt"], "docs/"),
        (["docs/api/*.md", "docs/guide/*.md"], "docs/"),
        # Only a sub-segment is common — truncate to last ``/`` boundary.
        (["docs/api/*.md", "docs/api2/*.md"], "docs/"),
        # One pattern starts with a glob → no common literal prefix.
        (["*.md", "docs/*.md"], ""),
        # Patterns don't share anything.
        (["docs/*.md", "api/*.md"], ""),
        # Single pattern.
        (["docs/api/*.md"], "docs/api/"),
        # Empty input.
        ([], ""),
    ],
)
def test_common_prefix(patterns: list[str], expected: str) -> None:
    assert BackendDocumentSource._common_prefix(patterns) == expected


class _PrefixRecordingBackend(_StaticBackend):
    """Backend that records the ``prefix`` argument passed to
    :meth:`list_files`, so tests can assert the optimization fired.
    """

    def __init__(
        self,
        files: dict[str, bytes],
        file_records: list[Any] | None = None,
    ) -> None:
        super().__init__(files, file_records)
        self.list_calls: list[dict[str, Any]] = []

    async def list_files(
        self, domain_id: str, prefix: str | None = None
    ) -> list[Any]:
        self.list_calls.append({"domain_id": domain_id, "prefix": prefix})
        if prefix:
            return [r for r in self._records if r.path.startswith(prefix)]
        return list(self._records)


@pytest.mark.asyncio
async def test_backend_iter_files_passes_common_prefix_to_list_files() -> None:
    """When all patterns share a literal prefix, ``iter_files`` narrows
    the backend listing by passing ``prefix=...``.

    Regression for M#8: the pre-optimization path always listed the
    whole namespace, which was fine for the in-memory backend but
    expensive against S3 for large buckets.
    """
    backend = _PrefixRecordingBackend(
        {
            "docs/guide.md": b"# Guide",
            "docs/api.md": b"# API",
            "other/readme.md": b"# Other",
        }
    )
    source = BackendDocumentSource(backend, "d")

    refs = await _collect_backend_refs(source, ["docs/*.md"])
    paths = sorted(r.path for r in refs)
    assert paths == ["docs/api.md", "docs/guide.md"]
    assert len(backend.list_calls) == 1
    assert backend.list_calls[0]["prefix"] == "docs/"


@pytest.mark.asyncio
async def test_backend_iter_files_no_prefix_when_patterns_differ() -> None:
    """Patterns without a common literal prefix trigger a full listing
    (no ``prefix`` kwarg). Ensures the optimization doesn't
    over-narrow the listing and miss files.
    """
    backend = _PrefixRecordingBackend(
        {
            "docs/guide.md": b"# Guide",
            "api/ref.md": b"# Ref",
        }
    )
    source = BackendDocumentSource(backend, "d")

    refs = await _collect_backend_refs(source, ["docs/*.md", "api/*.md"])
    paths = sorted(r.path for r in refs)
    assert paths == ["api/ref.md", "docs/guide.md"]
    # Called without prefix — patterns didn't share a literal prefix.
    assert len(backend.list_calls) == 1
    assert backend.list_calls[0]["prefix"] is None


@pytest.mark.asyncio
async def test_backend_iter_files_empty_patterns_no_prefix() -> None:
    """Empty ``patterns`` → full listing, no prefix narrowing."""
    backend = _PrefixRecordingBackend({"a.md": b"x"})
    source = BackendDocumentSource(backend, "d")

    refs = await _collect_backend_refs(source, [])
    assert [r.path for r in refs] == ["a.md"]
    assert len(backend.list_calls) == 1
    assert backend.list_calls[0]["prefix"] is None
