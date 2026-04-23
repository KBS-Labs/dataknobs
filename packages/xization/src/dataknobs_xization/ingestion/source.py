"""Storage-agnostic document source protocol for ingestion.

Decouples ``DirectoryProcessor`` from the local filesystem so the same
pattern-based chunking/excludes/per-pattern metadata pipeline can drive
any storage backend (local Path, in-memory, S3, etc.).

The :class:`DocumentSource` protocol has two implementations in this
module:

* :class:`LocalDocumentSource` — backed by a local :class:`pathlib.Path`.
* :class:`BackendDocumentSource` — backed by any
  :class:`KnowledgeResourceBackend` (file, memory, S3).
  Added in Phase 2.

Both are async-native. :class:`DirectoryProcessor` drives the protocol
through :meth:`DirectoryProcessor.process_async`; the sync
:meth:`~DirectoryProcessor.process` wrapper collects results via
``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dataknobs_bots.knowledge.storage.backend import KnowledgeResourceBackend

logger = logging.getLogger(__name__)


def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a pathlib-style glob pattern into a regex.

    ``*`` matches any characters except ``/``; ``?`` matches a single
    non-``/`` character; ``**`` matches any sequence of characters
    including ``/``. A trailing ``**/`` or leading ``**/`` is
    recognized and maps to "any sequence of path segments, possibly
    empty" — matching :meth:`pathlib.Path.glob` semantics.
    """
    parts: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                # "**" — consume, then check for trailing "/" → "any
                # depth including zero" form, else match anything.
                i += 2
                if i < n and pattern[i] == "/":
                    parts.append("(?:.*/)?")
                    i += 1
                else:
                    parts.append(".*")
            else:
                parts.append("[^/]*")
                i += 1
        elif c == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(parts) + "$")


@dataclass(frozen=True)
class DocumentFileRef:
    """Reference to a file inside a :class:`DocumentSource`.

    Attributes:
        path: Path relative to the source root. Uses forward slashes.
        size_bytes: File size. ``-1`` when the source cannot report size
            cheaply (some remote backends).
        source_uri: Opaque identifier for logging and metadata.
            Local: ``file:///abs/path``. Backend:
            ``{BackendClassName}://{domain_id}/{path}`` unless the
            backend exposes a ``uri`` property (Phase 2).
    """

    path: str
    size_bytes: int
    source_uri: str


@runtime_checkable
class DocumentSource(Protocol):
    """Async protocol for storage-agnostic file enumeration and reads.

    Implementations enumerate files by glob pattern and provide full
    reads + streaming reads. Patterns use shell-style globs (``*.md``,
    ``**/*.json``, ``docs/**/*.md``). Implementations are responsible
    for pattern semantics that match :func:`pathlib.Path.glob`.
    """

    async def iter_files(
        self, patterns: Iterable[str]
    ) -> AsyncIterator[DocumentFileRef]:
        """Yield :class:`DocumentFileRef` for each file matching any
        of ``patterns``. Deduplication across patterns is the caller's
        responsibility; implementations may yield the same file twice
        if it matches multiple patterns.
        """
        ...

    async def read_bytes(self, ref: DocumentFileRef) -> bytes:
        """Read the full contents of a file."""
        ...

    async def read_streaming(
        self, ref: DocumentFileRef, chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """Stream file contents in byte-sized chunks."""
        ...


class LocalDocumentSource:
    """Local filesystem-backed :class:`DocumentSource`.

    Wraps a :class:`pathlib.Path` root; ``iter_files`` uses
    :meth:`Path.glob` and filters to files only.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The root directory this source wraps."""
        return self._root

    async def iter_files(
        self, patterns: Iterable[str]
    ) -> AsyncIterator[DocumentFileRef]:
        """Enumerate files under ``root`` matching any of ``patterns``.

        Uses :meth:`Path.glob` for each pattern. Directories are
        skipped. The same file may be yielded more than once if it
        matches multiple patterns; callers that need deduplication
        should track seen paths.
        """
        for pattern in patterns:
            for path in self._root.glob(pattern):
                if not path.is_file():
                    continue
                rel = path.relative_to(self._root).as_posix()
                try:
                    size = path.stat().st_size
                except OSError:
                    size = -1
                yield DocumentFileRef(
                    path=rel,
                    size_bytes=size,
                    source_uri=path.resolve().as_uri(),
                )

    async def read_bytes(self, ref: DocumentFileRef) -> bytes:
        """Read the full contents of ``ref``."""
        path = self._root / ref.path
        return await asyncio.to_thread(path.read_bytes)

    async def read_streaming(
        self, ref: DocumentFileRef, chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """Stream file contents in ``chunk_size`` byte pieces.

        Each read happens in a worker thread via :func:`asyncio.to_thread`
        so the event loop isn't blocked on file I/O. Compared to a
        threaded-producer + bounded-queue design, this is simpler and
        safe under abandoned iteration: if the async consumer exits
        early (via ``break``, exception, or cancellation), the file
        handle is released by the generator's finalizer and no thread
        is left waiting to hand off a chunk.
        """
        path = self._root / ref.path

        def _open_and_read() -> tuple[Any, bytes]:
            f = open(path, "rb")
            try:
                chunk = f.read(chunk_size)
            except BaseException:
                f.close()
                raise
            return f, chunk

        f, chunk = await asyncio.to_thread(_open_and_read)
        try:
            while chunk:
                yield chunk
                chunk = await asyncio.to_thread(f.read, chunk_size)
        finally:
            await asyncio.to_thread(f.close)


class BackendDocumentSource:
    """:class:`DocumentSource` backed by a
    :class:`KnowledgeResourceBackend`.

    Adapts any in-tree backend (file, memory, S3) to the
    :class:`DocumentSource` protocol. Pattern matching uses a custom
    :func:`_compile_glob` regex compiler that mirrors
    :meth:`pathlib.Path.glob` semantics (``*`` bounded at ``/``,
    ``?`` matches a single non-``/`` character, ``**`` crosses
    directory separators) — consistent with :class:`LocalDocumentSource`
    and strictly different from :func:`fnmatch.fnmatch`, which doesn't
    treat ``/`` as a boundary.

    When ``patterns`` is empty, all files under ``domain_id`` are
    yielded (matches :class:`DirectoryProcessor`'s behavior for configs
    without explicit patterns).

    :meth:`iter_files` derives a common prefix from the supplied
    patterns (the portion before the first glob metacharacter) and
    passes it to :meth:`backend.list_files` so backends like S3 can
    avoid listing the whole bucket when every pattern shares a prefix.
    """

    def __init__(
        self,
        backend: KnowledgeResourceBackend,
        domain_id: str,
    ) -> None:
        self._backend = backend
        self._domain_id = domain_id

    @property
    def backend(self) -> KnowledgeResourceBackend:
        """The wrapped backend."""
        return self._backend

    @property
    def domain_id(self) -> str:
        """The domain identifier within the backend."""
        return self._domain_id

    def _backend_uri(self) -> str:
        """Stable identifier prefix for ``DocumentFileRef.source_uri``.

        Backends MAY expose a ``uri`` attribute/property; when
        available it is used verbatim. Otherwise we build
        ``{ClassName}://{domain_id}``.
        """
        backend_uri = getattr(self._backend, "uri", None)
        if backend_uri:
            return f"{backend_uri}/{self._domain_id}"
        return f"{type(self._backend).__name__}://{self._domain_id}"

    async def iter_files(
        self, patterns: Iterable[str]
    ) -> AsyncIterator[DocumentFileRef]:
        """Enumerate files matching any of ``patterns``.

        Derives a common prefix from the pattern list (the literal
        portion shared by every pattern before the first glob
        metacharacter) and passes it to
        :meth:`backend.list_files(domain_id, prefix=...)` so prefix-
        aware backends (notably S3) can narrow the listing. After
        listing, matches are filtered in Python via :meth:`_matches`.
        Empty ``patterns`` yields every file in the domain.
        """
        patterns_list = list(patterns)
        prefix = self._common_prefix(patterns_list)
        if prefix:
            files = await self._backend.list_files(
                self._domain_id, prefix=prefix
            )
        else:
            files = await self._backend.list_files(self._domain_id)
        base_uri = self._backend_uri()

        for kf in files:
            path = kf.path
            if patterns_list and not any(
                self._matches(path, p) for p in patterns_list
            ):
                continue
            size = getattr(kf, "size_bytes", None)
            if size is None:
                size = getattr(kf, "size", -1)
            yield DocumentFileRef(
                path=path,
                size_bytes=int(size),
                source_uri=f"{base_uri}/{path}",
            )

    @staticmethod
    def _common_prefix(patterns: list[str]) -> str:
        """Return the longest literal path-segment prefix shared by
        every pattern.

        Used to narrow :meth:`backend.list_files` calls. Returns the
        empty string when patterns is empty, when any pattern starts
        with a glob metacharacter, or when patterns don't agree on a
        common leading path segment. Always ends at a ``/`` boundary
        so the result is a valid path prefix (not a partial segment).
        """
        if not patterns:
            return ""
        per_pattern = [
            BackendDocumentSource._literal_prefix(p) for p in patterns
        ]
        if not all(per_pattern):
            return ""
        common = per_pattern[0]
        for p in per_pattern[1:]:
            limit = min(len(common), len(p))
            i = 0
            while i < limit and common[i] == p[i]:
                i += 1
            common = common[:i]
            if not common:
                return ""
        boundary = common.rfind("/")
        return common[: boundary + 1] if boundary >= 0 else ""

    @staticmethod
    def _literal_prefix(pattern: str) -> str:
        """Return the portion of ``pattern`` before the first glob
        metacharacter (``*``, ``?``, ``[``). Backend listing can use
        this as an inclusive prefix.
        """
        for i, ch in enumerate(pattern):
            if ch in "*?[":
                return pattern[:i]
        return pattern

    @staticmethod
    def _matches(path: str, pattern: str) -> bool:
        """Match ``path`` against a glob pattern using path-aware
        semantics that mirror :meth:`pathlib.Path.glob`:

        * ``*`` matches any characters except ``/``.
        * ``?`` matches any single character except ``/``.
        * ``**`` matches any sequence of path segments (including
          none); ``**/`` at segment boundaries matches "any depth".
        """
        return _compile_glob(pattern).match(path) is not None

    async def read_bytes(self, ref: DocumentFileRef) -> bytes:
        """Read the full contents of ``ref`` from the backend."""
        data = await self._backend.get_file(self._domain_id, ref.path)
        if data is None:
            raise FileNotFoundError(
                f"Backend returned no data for {self._domain_id}/{ref.path}"
            )
        return data

    async def read_streaming(
        self, ref: DocumentFileRef, chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """Stream the file's contents in ``chunk_size`` byte pieces.

        Prefers :meth:`backend.stream_file` when the backend supports
        it; otherwise falls back to a full :meth:`get_file` followed by
        in-memory chunking.
        """
        stream_fn = getattr(self._backend, "stream_file", None)
        if stream_fn is not None:
            stream = await stream_fn(
                self._domain_id, ref.path, chunk_size=chunk_size
            )
            if stream is not None:
                async for piece in stream:
                    yield piece
                return
        data = await self.read_bytes(ref)
        for offset in range(0, len(data), chunk_size):
            yield data[offset : offset + chunk_size]


__all__ = [
    "BackendDocumentSource",
    "DocumentFileRef",
    "DocumentSource",
    "LocalDocumentSource",
]
