"""Directory processor for knowledge base ingestion.

This module provides the DirectoryProcessor class for processing
documents from a directory into chunks ready for embedding.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from dataknobs_xization.chunking import create_chunker
from dataknobs_xization.chunking.base import Chunker, DocumentInfo
from dataknobs_xization.ingestion.config import (
    FilePatternConfig,
    KnowledgeBaseConfig,
)
from dataknobs_xization.json import JSONChunk, JSONChunkConfig, JSONChunker

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """A processed document ready for embedding and storage.

    Contains chunks from a single source file along with metadata
    about the processing.

    Attributes:
        source_file: Path to the source file
        document_type: Type of document (markdown, json, jsonl)
        chunks: List of processed chunks
        metadata: Document-level metadata
        errors: Any errors encountered during processing
    """

    source_file: str
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


class DirectoryProcessor:
    """Process documents from a directory for knowledge base ingestion.

    Handles markdown and JSON files with configurable chunking,
    supporting both in-memory and streaming processing for large files.

    Attributes:
        config: Knowledge base configuration
        root_dir: Root directory for processing
    """

    def __init__(
        self,
        config: KnowledgeBaseConfig,
        root_dir: str | Path,
        chunker: Chunker | None = None,
    ):
        """Initialize the directory processor.

        Args:
            config: Knowledge base configuration
            root_dir: Root directory containing documents
            chunker: Optional pre-built chunker for markdown files.
                When provided, used as the default chunker instead of
                creating one from ``config.default_chunking``.
        """
        self.config = config
        self.root_dir = Path(root_dir)

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
        """Process all documents in the directory.

        Yields ProcessedDocument for each file, automatically using
        streaming for large JSON files to avoid memory exhaustion.

        Yields:
            ProcessedDocument for each processed file
        """
        # Collect all files first
        files = self._collect_files()

        for filepath in files:
            rel_path = filepath.relative_to(self.root_dir)

            # Skip config files
            if filepath.name in CONFIG_FILE_NAMES:
                logger.debug("Skipping config file: %s", rel_path)
                continue

            # Skip excluded files
            if self.config.is_excluded(rel_path):
                logger.debug("Skipping excluded file: %s", rel_path)
                continue

            # Get pattern config if any
            pattern_config = self.config.get_pattern_config(rel_path)

            # Process based on file type
            suffix = filepath.suffix.lower()
            if suffix == ".md":
                yield from self._process_markdown(filepath, pattern_config)
            elif suffix in (".json", ".jsonl", ".ndjson"):
                yield from self._process_json(filepath, pattern_config)
            elif suffix == ".gz":
                # Check inner extension for compressed files
                inner_suffix = Path(filepath.stem).suffix.lower()
                if inner_suffix in (".json", ".jsonl", ".ndjson"):
                    yield from self._process_json(filepath, pattern_config)
                else:
                    logger.debug("Skipping unsupported compressed file: %s", rel_path)
            else:
                logger.debug("Skipping unsupported file type: %s", rel_path)

    def _collect_files(self) -> list[Path]:
        """Collect all files to process from the directory.

        Returns:
            List of file paths
        """
        files = []

        # If patterns are defined, use them to find files
        if self.config.patterns:
            for pattern_config in self.config.patterns:
                if pattern_config.enabled:
                    for filepath in self.root_dir.glob(pattern_config.pattern):
                        if filepath.is_file() and filepath not in files:
                            files.append(filepath)
        else:
            # Default: find all supported files
            for pattern in ["**/*.md", "**/*.json", "**/*.jsonl", "**/*.ndjson",
                           "**/*.json.gz", "**/*.jsonl.gz", "**/*.ndjson.gz"]:
                for filepath in self.root_dir.glob(pattern):
                    if filepath.is_file() and filepath not in files:
                        files.append(filepath)

        return sorted(files)

    def _process_markdown(
        self,
        filepath: Path,
        pattern_config: FilePatternConfig | None,
    ) -> Iterator[ProcessedDocument]:
        """Process a markdown file.

        Args:
            filepath: Path to markdown file
            pattern_config: Optional pattern-specific configuration

        Yields:
            ProcessedDocument for the file
        """
        errors: list[str] = []
        chunks: list[dict[str, Any]] = []

        try:
            # Read file
            with open(filepath, encoding="utf-8") as f:
                content = f.read()

            # Get chunking config — reuse the default chunker when config
            # matches to avoid per-file construction overhead.
            chunking_config = self.config.get_chunking_config(
                filepath.relative_to(self.root_dir)
            )
            effective_config = self._build_effective_config(chunking_config)

            if self._config_cache_key(effective_config) == self._default_config_key:
                chunker = self._default_chunker
            else:
                chunker = create_chunker(effective_config)

            # Chunk the document
            doc_info = DocumentInfo(
                source=str(filepath),
                content_type="text/markdown",
            )
            md_chunks = chunker.chunk(content, doc_info)

            # Convert to dictionaries
            for i, chunk in enumerate(md_chunks):
                chunk_dict = {
                    "text": chunk.text,
                    "embedding_text": chunk.metadata.embedding_text or chunk.text,
                    "chunk_index": i,
                    "source_path": str(filepath),
                    "metadata": {
                        "heading_path": chunk.metadata.heading_display or chunk.metadata.get_heading_path(),
                        "headings": chunk.metadata.headings,
                        "heading_levels": chunk.metadata.heading_levels,
                        "line_number": chunk.metadata.line_number,
                        "char_start": chunk.metadata.char_start,
                        "char_end": chunk.metadata.char_end,
                        "chunk_size": chunk.metadata.chunk_size,
                        "content_length": chunk.metadata.content_length,
                    },
                }
                chunks.append(chunk_dict)

        except Exception as e:
            errors.append(f"Failed to process markdown: {e}")
            logger.exception("Error processing %s", filepath)

        # Build metadata
        metadata = self.config.get_metadata(filepath.relative_to(self.root_dir))

        yield ProcessedDocument(
            source_file=str(filepath),
            document_type="markdown",
            chunks=chunks,
            metadata=metadata,
            errors=errors,
        )

    def _process_json(
        self,
        filepath: Path,
        pattern_config: FilePatternConfig | None,
    ) -> Iterator[ProcessedDocument]:
        """Process a JSON or JSONL file.

        Automatically uses streaming for large files or JSONL format.

        Args:
            filepath: Path to JSON file
            pattern_config: Optional pattern-specific configuration

        Yields:
            ProcessedDocument for the file
        """
        errors: list[str] = []
        chunks: list[dict[str, Any]] = []

        try:
            # Build JSON chunker config
            chunking_config = self.config.get_chunking_config(
                filepath.relative_to(self.root_dir)
            )

            json_config = JSONChunkConfig(
                max_chunk_size=chunking_config.get("max_chunk_size", 1000),
                nested_separator=chunking_config.get("nested_separator", "."),
                array_handling=chunking_config.get("array_handling", "expand"),
                include_field_names=chunking_config.get("include_field_names", True),
                skip_technical_fields=chunking_config.get("skip_technical_fields", True),
            )

            # Apply pattern-specific overrides
            if pattern_config:
                if pattern_config.text_template:
                    json_config.text_template = pattern_config.text_template
                if pattern_config.text_fields:
                    json_config.text_fields = pattern_config.text_fields

            chunker = JSONChunker(json_config)

            # Determine if we should stream
            is_jsonl = self._is_jsonl_file(str(filepath))
            file_size = os.path.getsize(filepath)
            should_stream = is_jsonl or file_size > STREAMING_THRESHOLD_BYTES

            if should_stream:
                # Stream chunks for large files or JSONL
                for json_chunk in chunker.stream_chunks(filepath):
                    chunk_dict = self._json_chunk_to_dict(json_chunk)
                    chunks.append(chunk_dict)
            else:
                # Load and chunk in memory for small files
                import json as json_lib
                with open(filepath, encoding="utf-8") as f:
                    data = json_lib.load(f)

                for json_chunk in chunker.chunk(data, source=str(filepath)):
                    chunk_dict = self._json_chunk_to_dict(json_chunk)
                    chunks.append(chunk_dict)

        except Exception as e:
            errors.append(f"Failed to process JSON: {e}")
            logger.exception("Error processing %s", filepath)

        # Build metadata
        metadata = self.config.get_metadata(filepath.relative_to(self.root_dir))

        # Determine document type
        doc_type: Literal["json", "jsonl"] = "jsonl" if self._is_jsonl_file(str(filepath)) else "json"

        yield ProcessedDocument(
            source_file=str(filepath),
            document_type=doc_type,
            chunks=chunks,
            metadata=metadata,
            errors=errors,
        )

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
