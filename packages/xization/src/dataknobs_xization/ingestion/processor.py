"""Directory processor for knowledge base ingestion.

This module provides the DirectoryProcessor class for processing
documents from a directory into chunks ready for embedding.

The implementation is async-primary: :meth:`DirectoryProcessor.process_async`
yields :class:`ProcessedDocument` values as files are read from the
underlying :class:`DocumentSource`. The sync :meth:`DirectoryProcessor.process`
is a thin wrapper that collects the async iterator via
:func:`asyncio.run`; it cannot be called from inside a running event
loop.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from dataknobs_xization.chunking import create_chunker
from dataknobs_xization.chunking.base import Chunker, DocumentInfo
from dataknobs_xization.content_transformer import ContentTransformer
from dataknobs_xization.ingestion.config import (
    FilePatternConfig,
    KnowledgeBaseConfig,
)
from dataknobs_xization.ingestion.source import (
    DocumentFileRef,
    DocumentSource,
    LocalDocumentSource,
)
from dataknobs_xization.json import JSONChunk, JSONChunkConfig, JSONChunker

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """A processed document ready for embedding and storage.

    Contains chunks from a single source file along with metadata
    about the processing.

    Attributes:
        source_file: Display path for the source file (absolute path for
            local sources; opaque URI for backend sources).
        source_path: Source-relative path (``ref.path``). Stable across
            local and backend sources; suitable for metadata filtering.
        document_type: Type of document (markdown, json, jsonl)
        chunks: List of processed chunks
        metadata: Document-level metadata
        errors: Any errors encountered during processing
    """

    source_file: str
    source_path: str
    document_type: Literal["markdown", "json", "jsonl"]
    chunks: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        """Number of chunks in this document."""
        return len(self.chunks)

    @property
    def has_errors(self) -> bool:
        """Whether processing encountered errors."""
        return len(self.errors) > 0


# File size threshold for streaming (10MB)
STREAMING_THRESHOLD_BYTES = 10 * 1024 * 1024

# Config file names to always exclude from processing
CONFIG_FILE_NAMES = {"knowledge_base.json", "knowledge_base.yaml", "knowledge_base.yml"}

# Default glob patterns used when no explicit patterns are configured.
_DEFAULT_PATTERNS = (
    "**/*.md",
    "**/*.markdown",
    "**/*.txt",
    "**/*.yaml",
    "**/*.yml",
    "**/*.csv",
    "**/*.json",
    "**/*.jsonl",
    "**/*.ndjson",
    "**/*.json.gz",
    "**/*.jsonl.gz",
    "**/*.ndjson.gz",
)

# Extensions routed through the markdown pipeline (optional transform
# step prepends for non-markdown types).
_MARKDOWN_LIKE_EXTS = (".md", ".markdown", ".txt")
_YAML_EXTS = (".yaml", ".yml")
_CSV_EXTS = (".csv",)


class DirectoryProcessor:
    """Process documents from a directory for knowledge base ingestion.

    Handles markdown and JSON files with configurable chunking,
    supporting both in-memory and streaming processing for large files.

    Accepts any :class:`DocumentSource` as the file source; a raw
    ``str | Path`` is wrapped automatically in a
    :class:`LocalDocumentSource`.

    Attributes:
        config: Knowledge base configuration
        root_dir: Local root directory (``None`` when constructed
            from a non-local :class:`DocumentSource`).
    """

    def __init__(
        self,
        config: KnowledgeBaseConfig,
        root_dir: str | Path | DocumentSource,
        chunker: Chunker | None = None,
    ):
        """Initialize the directory processor.

        Args:
            config: Knowledge base configuration.
            root_dir: Root directory containing documents, or a
                :class:`DocumentSource` instance. A ``str`` or
                :class:`~pathlib.Path` is wrapped in a
                :class:`LocalDocumentSource`.
            chunker: Optional pre-built chunker for markdown files.
                When provided, used as the default chunker instead of
                creating one from ``config.default_chunking``.
        """
        self.config = config
        if isinstance(root_dir, (str, Path)):
            self.root_dir: Path | None = Path(root_dir)
            self._source: DocumentSource = LocalDocumentSource(self.root_dir)
        else:
            self.root_dir = None
            self._source = root_dir

        # Counter updated during :meth:`process_async` for files
        # skipped before dispatch (config files, excluded paths,
        # unsupported extensions). Consumers read it after iteration
        # completes.
        self.files_skipped: int = 0

        # Build the default chunker once for markdown files.  Per-file
        # chunkers are only created when a pattern overrides the default.
        self._default_chunking_config = self._build_effective_config(
            config.default_chunking
        )
        self._default_config_key = self._config_cache_key(
            self._default_chunking_config
        )
        self._default_chunker = chunker or create_chunker(
            self._default_chunking_config
        )

    @property
    def source(self) -> DocumentSource:
        """The underlying :class:`DocumentSource`."""
        return self._source

    def _build_effective_config(
        self, chunking_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge default quality filter into a chunking config dict."""
        effective = dict(chunking_config)
        if (
            self.config.default_quality_filter
            and "quality_filter" not in effective
        ):
            effective["quality_filter"] = self.config.default_quality_filter
        return effective

    @staticmethod
    def _config_cache_key(config: dict[str, Any]) -> str:
        """Canonical JSON string for config comparison.

        Using JSON serialization instead of dict equality handles
        nested structures robustly and fails visibly if a non-
        serializable object is placed in the config.
        """
        return json.dumps(config, sort_keys=True, default=str)

    def process(self) -> Iterator[ProcessedDocument]:
        """Process all documents in the directory (sync wrapper).

        Collects the async iterator from :meth:`process_async` via
        :func:`asyncio.run` and yields the collected list. Cannot be
        called from inside a running event loop — async callers should
        use :meth:`process_async` directly.

        Yields:
            ProcessedDocument for each processed file.
        """
        async def _collect() -> list[ProcessedDocument]:
            return [doc async for doc in self.process_async()]

        return iter(asyncio.run(_collect()))

    async def process_async(self) -> AsyncIterator[ProcessedDocument]:
        """Process all documents in the directory (async primary).

        Yields :class:`ProcessedDocument` for each supported file as
        it is read from the underlying :class:`DocumentSource`.
        Automatically uses streaming for large JSON files to avoid
        memory exhaustion.

        Yields:
            ProcessedDocument for each processed file.
        """
        self.files_skipped = 0
        async for ref in self._collect_files_async():
            # Skip config files (matched by filename only).
            if Path(ref.path).name in CONFIG_FILE_NAMES:
                logger.debug("Skipping config file: %s", ref.path)
                self.files_skipped += 1
                continue

            # Skip excluded files.
            if self.config.is_excluded(ref.path):
                logger.debug("Skipping excluded file: %s", ref.path)
                self.files_skipped += 1
                continue

            pattern_config = self.config.get_pattern_config(ref.path)

            suffix = Path(ref.path).suffix.lower()
            stem_suffix = Path(Path(ref.path).stem).suffix.lower()
            if suffix in _MARKDOWN_LIKE_EXTS:
                yield await self._process_markdown_file_async(
                    ref, pattern_config
                )
            elif suffix in _YAML_EXTS:
                yield await self._process_yaml_async(ref, pattern_config)
            elif suffix in _CSV_EXTS:
                yield await self._process_csv_async(ref, pattern_config)
            elif suffix in (".json", ".jsonl", ".ndjson") or (suffix == ".gz" and stem_suffix in (
                ".json",
                ".jsonl",
                ".ndjson",
            )):
                async for doc in self._process_json_async(ref, pattern_config):
                    yield doc
            else:
                logger.debug("Skipping unsupported file type: %s", ref.path)
                self.files_skipped += 1

    def _collect_patterns(self) -> list[str]:
        """Return the enabled glob patterns from config, or defaults.

        Matches the behavior of the previous ``_collect_files``
        implementation: if explicit patterns are configured, use their
        ``pattern`` strings (only enabled ones); otherwise fall back
        to :data:`_DEFAULT_PATTERNS`.
        """
        if self.config.patterns:
            return [p.pattern for p in self.config.patterns if p.enabled]
        return list(_DEFAULT_PATTERNS)

    async def _collect_files_async(self) -> AsyncIterator[DocumentFileRef]:
        """Enumerate candidate files via the :class:`DocumentSource`.

        Deduplicates across patterns so a file matching two patterns
        is yielded only once. Order within a pattern follows the
        source; cross-pattern order is insertion order of first
        appearance, which yields stable results for a given source.
        """
        patterns = self._collect_patterns()
        seen: set[str] = set()
        async for ref in self._source.iter_files(patterns):
            if ref.path in seen:
                continue
            seen.add(ref.path)
            yield ref

    def _chunk_markdown_text(
        self,
        content: str,
        ref_path: str,
        source_display: str,
    ) -> list[dict[str, Any]]:
        """Chunk a markdown (or markdown-converted) string.

        Shared by :meth:`_process_markdown_file_async`,
        :meth:`_process_yaml_async`, and :meth:`_process_csv_async`.
        Picks the configured chunker (default or per-pattern override)
        and emits ready-to-embed chunk dicts with the same shape as
        chunks produced directly from markdown files.
        """
        chunking_config = self.config.get_chunking_config(ref_path)
        effective_config = self._build_effective_config(chunking_config)

        if self._config_cache_key(effective_config) == self._default_config_key:
            chunker = self._default_chunker
        else:
            chunker = create_chunker(effective_config)

        doc_info = DocumentInfo(
            source=source_display,
            content_type="text/markdown",
        )
        md_chunks = chunker.chunk(content, doc_info)

        chunk_dicts: list[dict[str, Any]] = []
        for i, chunk in enumerate(md_chunks):
            chunk_dicts.append(
                {
                    "text": chunk.text,
                    "embedding_text": chunk.metadata.embedding_text
                    or chunk.text,
                    "chunk_index": i,
                    "source_path": source_display,
                    "metadata": {
                        "heading_path": chunk.metadata.heading_display
                        or chunk.metadata.get_heading_path(),
                        "headings": chunk.metadata.headings,
                        "heading_levels": chunk.metadata.heading_levels,
                        "line_number": chunk.metadata.line_number,
                        "char_start": chunk.metadata.char_start,
                        "char_end": chunk.metadata.char_end,
                        "chunk_size": chunk.metadata.chunk_size,
                        "content_length": chunk.metadata.content_length,
                    },
                }
            )
        return chunk_dicts

    async def _process_markdown_file_async(
        self,
        ref: DocumentFileRef,
        pattern_config: FilePatternConfig | None,
    ) -> ProcessedDocument:
        """Process a markdown-like text file (``.md``, ``.markdown``,
        ``.txt``).

        ``.txt`` files are treated as single-section markdown by the
        default chunker — the same behavior as the pre-unification
        ``KnowledgeIngestionManager._load_text`` path.
        """
        errors: list[str] = []
        chunks: list[dict[str, Any]] = []
        source_file = self._display_path(ref)

        try:
            content = (await self._source.read_bytes(ref)).decode("utf-8")
            chunks = self._chunk_markdown_text(content, ref.path, source_file)
        except Exception as e:
            errors.append(f"Failed to process markdown: {e}")
            logger.exception("Error processing %s", ref.path)

        return ProcessedDocument(
            source_file=source_file,
            source_path=ref.path,
            document_type="markdown",
            chunks=chunks,
            metadata=self.config.get_metadata(ref.path),
            errors=errors,
        )

    async def _process_yaml_async(
        self,
        ref: DocumentFileRef,
        pattern_config: FilePatternConfig | None,
    ) -> ProcessedDocument:
        """Process a YAML file by transforming to markdown, then chunking.

        Requires PyYAML (raised as a :class:`ImportError` from
        :meth:`ContentTransformer.transform_yaml`). The transform error
        is captured in ``ProcessedDocument.errors`` rather than aborting
        the ingest.
        """
        errors: list[str] = []
        chunks: list[dict[str, Any]] = []
        source_file = self._display_path(ref)

        try:
            content = (await self._source.read_bytes(ref)).decode("utf-8")
            transformer = ContentTransformer()
            markdown = transformer.transform_yaml(
                content,
                title=Path(ref.path).stem.replace("_", " ").title(),
            )
            chunks = self._chunk_markdown_text(markdown, ref.path, source_file)
        except Exception as e:
            errors.append(f"Failed to process YAML: {e}")
            logger.exception("Error processing %s", ref.path)

        return ProcessedDocument(
            source_file=source_file,
            source_path=ref.path,
            document_type="markdown",
            chunks=chunks,
            metadata=self.config.get_metadata(ref.path),
            errors=errors,
        )

    async def _process_csv_async(
        self,
        ref: DocumentFileRef,
        pattern_config: FilePatternConfig | None,
    ) -> ProcessedDocument:
        """Process a CSV file by transforming each row to a markdown
        section, then chunking.

        ``pattern_config.metadata_fields`` is not wired here yet —
        per-row field selection happens inside
        :meth:`ContentTransformer.transform_csv`.
        """
        errors: list[str] = []
        chunks: list[dict[str, Any]] = []
        source_file = self._display_path(ref)

        try:
            content = (await self._source.read_bytes(ref)).decode("utf-8")
            transformer = ContentTransformer()
            markdown = transformer.transform_csv(
                content,
                title=Path(ref.path).stem.replace("_", " ").title(),
            )
            chunks = self._chunk_markdown_text(markdown, ref.path, source_file)
        except Exception as e:
            errors.append(f"Failed to process CSV: {e}")
            logger.exception("Error processing %s", ref.path)

        return ProcessedDocument(
            source_file=source_file,
            source_path=ref.path,
            document_type="markdown",
            chunks=chunks,
            metadata=self.config.get_metadata(ref.path),
            errors=errors,
        )

    async def _process_json_async(
        self,
        ref: DocumentFileRef,
        pattern_config: FilePatternConfig | None,
    ) -> AsyncIterator[ProcessedDocument]:
        """Process a JSON or JSONL file (async).

        Automatically uses streaming for large files or JSONL format.
        Streaming from a non-local :class:`DocumentSource` buffers
        bytes through :class:`io.BytesIO` before feeding the streaming
        parser; the parser still yields chunks incrementally.

        Args:
            ref: Reference to the JSON file in the
                :class:`DocumentSource`.
            pattern_config: Optional pattern-specific configuration.

        Yields:
            ProcessedDocument for the file.
        """
        errors: list[str] = []
        chunks: list[dict[str, Any]] = []
        source_file = self._display_path(ref)

        try:
            chunking_config = self.config.get_chunking_config(ref.path)

            json_config = JSONChunkConfig(
                max_chunk_size=chunking_config.get("max_chunk_size", 1000),
                nested_separator=chunking_config.get("nested_separator", "."),
                array_handling=chunking_config.get("array_handling", "expand"),
                include_field_names=chunking_config.get("include_field_names", True),
                skip_technical_fields=chunking_config.get("skip_technical_fields", True),
            )

            if pattern_config:
                if pattern_config.text_template:
                    json_config.text_template = pattern_config.text_template
                if pattern_config.text_fields:
                    json_config.text_fields = pattern_config.text_fields

            chunker = JSONChunker(json_config)

            is_jsonl = self._is_jsonl_file(ref.path)
            file_size = ref.size_bytes if ref.size_bytes >= 0 else 0
            should_stream = is_jsonl or file_size > STREAMING_THRESHOLD_BYTES

            if should_stream:
                async for json_chunk in self._stream_json_chunks(
                    ref, chunker, source_file
                ):
                    chunks.append(self._json_chunk_to_dict(json_chunk))
            else:
                import json as json_lib
                raw = await self._source.read_bytes(ref)
                data = json_lib.loads(raw.decode("utf-8"))

                for json_chunk in chunker.chunk(data, source=source_file):
                    chunks.append(self._json_chunk_to_dict(json_chunk))

        except Exception as e:
            errors.append(f"Failed to process JSON: {e}")
            logger.exception("Error processing %s", ref.path)

        metadata = self.config.get_metadata(ref.path)

        doc_type: Literal["json", "jsonl"] = "jsonl" if self._is_jsonl_file(ref.path) else "json"

        yield ProcessedDocument(
            source_file=source_file,
            source_path=ref.path,
            document_type=doc_type,
            chunks=chunks,
            metadata=metadata,
            errors=errors,
        )

    async def _stream_json_chunks(
        self,
        ref: DocumentFileRef,
        chunker: JSONChunker,
        source_display: str,
    ) -> AsyncIterator[JSONChunk]:
        """Stream :class:`JSONChunk` values for a JSON/JSONL ref.

        Three paths, all forward-sequential in chunk emission:

        * :class:`LocalDocumentSource`: pass the on-disk path directly
          to :meth:`JSONChunker.stream_chunks` so gzip/URL/path-based
          dispatch works unchanged.
        * Remote source + JSONL (``.jsonl``, ``.ndjson``, ``.jsonl.gz``,
          ``.ndjson.gz``): parse one object per line from the async
          byte iterator directly, without buffering the whole file.
        * Remote source + single JSON tree: buffer bytes into
          :class:`io.BytesIO` then parse — whole-tree parsing cannot
          be incremental. This path holds one file-sized buffer in
          memory; callers with multi-hundred-MB single-object JSON
          should prefer JSONL for that reason.
        """
        if isinstance(self._source, LocalDocumentSource):
            path = self._source.root / ref.path
            for chunk in chunker.stream_chunks(path):
                if not chunk.source_file:
                    chunk.source_file = source_display
                yield chunk
            return

        if self._is_jsonl_file(ref.path):
            async for chunk in self._stream_jsonl_from_remote(
                ref, chunker, source_display
            ):
                yield chunk
            return

        buf = io.BytesIO()
        async for piece in self._source.read_streaming(ref):
            buf.write(piece)
        buf.seek(0)
        text_stream = io.TextIOWrapper(buf, encoding="utf-8")
        try:
            for chunk in chunker.stream_chunks(
                text_stream,
                source_file=source_display,
            ):
                yield chunk
        finally:
            text_stream.detach()

    async def _stream_jsonl_from_remote(
        self,
        ref: DocumentFileRef,
        chunker: JSONChunker,
        source_display: str,
    ) -> AsyncIterator[JSONChunk]:
        """Parse JSONL from an async byte iterator one line at a time.

        Accumulates only the tail of the current (incomplete) line
        between async reads, never the whole file. Malformed lines are
        logged and skipped so a bad line doesn't abort the ingest.
        """
        pending = b""
        async for piece in self._source.read_streaming(ref):
            pending += piece
            while True:
                nl = pending.find(b"\n")
                if nl < 0:
                    break
                raw_line, pending = pending[:nl], pending[nl + 1 :]
                for chunk in self._emit_jsonl_line(
                    raw_line, chunker, source_display
                ):
                    yield chunk

        if pending.strip():
            for chunk in self._emit_jsonl_line(pending, chunker, source_display):
                yield chunk

    def _emit_jsonl_line(
        self,
        raw_line: bytes,
        chunker: JSONChunker,
        source_display: str,
    ) -> Iterator[JSONChunk]:
        """Parse a single JSONL line and yield its chunks.

        Empty / whitespace-only lines and malformed JSON are skipped
        with a warning.
        """
        stripped = raw_line.strip()
        if not stripped:
            return
        try:
            obj = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning(
                "Skipping malformed JSONL line in %s: %s", source_display, e
            )
            return
        for chunk in chunker.chunk(obj, source=source_display):
            yield chunk

    def _display_path(self, ref: DocumentFileRef) -> str:
        """Best-effort user-facing path for the source file.

        For a local source, return the absolute on-disk path so
        existing callers that rely on ``source_file`` being a real
        filesystem path keep working. For other sources, return the
        ``source_uri`` which is opaque but identifies the origin.
        """
        if isinstance(self._source, LocalDocumentSource):
            return str(self._source.root / ref.path)
        return ref.source_uri

    def _json_chunk_to_dict(self, chunk: JSONChunk) -> dict[str, Any]:
        """Convert a JSONChunk to a dictionary.

        Args:
            chunk: JSONChunk instance

        Returns:
            Dictionary representation
        """
        return {
            "text": chunk.text,
            "embedding_text": chunk.embedding_text or chunk.text,
            "chunk_index": chunk.chunk_index,
            "source_path": chunk.source_path,
            "metadata": chunk.metadata,
        }

    def _is_jsonl_file(self, filepath: str) -> bool:
        """Check if a file is JSONL format based on extension.

        Args:
            filepath: Path to check

        Returns:
            True if file is JSONL format
        """
        filepath_lower = filepath.lower()
        return any(
            filepath_lower.endswith(ext)
            for ext in [".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz"]
        )


def process_directory(
    directory: str | Path,
    config: KnowledgeBaseConfig | None = None,
    chunker: Chunker | None = None,
) -> Iterator[ProcessedDocument]:
    """Convenience function to process a directory.

    Args:
        directory: Directory to process
        config: Optional configuration (loads from directory if not provided)
        chunker: Optional pre-built chunker for markdown files

    Yields:
        ProcessedDocument for each file
    """
    directory = Path(directory)

    if config is None:
        config = KnowledgeBaseConfig.load(directory)

    processor = DirectoryProcessor(config, directory, chunker=chunker)
    yield from processor.process()
