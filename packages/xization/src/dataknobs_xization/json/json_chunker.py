# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""JSON chunker for generating RAG-optimized chunks from JSON data.

This module provides functionality to chunk JSON data (objects, arrays, JSONL files)
into units suitable for RAG (Retrieval-Augmented Generation) applications, with
preserved metadata and configurable text generation.

Supports both in-memory and streaming modes for handling large files.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import queue
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterator, Literal

# Patterns for detecting technical/non-text fields
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)
BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{20,}={0,2}$")
TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"  # ISO format
)

# Field names commonly containing text content
TEXT_FIELD_NAMES = frozenset(
    {
        "title",
        "name",
        "description",
        "content",
        "text",
        "summary",
        "body",
        "message",
        "comment",
        "note",
        "notes",
        "abstract",
        "overview",
        "details",
        "explanation",
        "definition",
        "label",
    }
)

# Field names to skip (technical/metadata)
SKIP_FIELD_NAMES = frozenset(
    {
        "id",
        "uuid",
        "guid",
        "_id",
        "created_at",
        "updated_at",
        "created",
        "updated",
        "timestamp",
        "modified",
        "hash",
        "checksum",
        "signature",
        "token",
        "key",
        "secret",
    }
)


@dataclass
class JSONChunkConfig:
    """Configuration for JSON chunking.

    Attributes:
        max_chunk_size: Maximum size of generated text in characters
        text_template: Optional Jinja2 template for text generation (overrides auto-flatten)
        text_fields: Specific fields to use for text (None = auto-detect)
        nested_separator: Separator for flattened nested keys (e.g., "config.database.host")
        array_handling: How to handle arrays - expand into multiple chunks, join values, or take first
        include_field_names: Whether to include field names in generated text
        skip_technical_fields: Whether to skip UUIDs, timestamps, base64 in text generation
    """

    max_chunk_size: int = 1000
    text_template: str | None = None
    text_fields: list[str] | None = None
    nested_separator: str = "."
    array_handling: Literal["expand", "join", "first"] = "expand"
    include_field_names: bool = True
    skip_technical_fields: bool = True


@dataclass
class JSONChunk:
    """A chunk generated from JSON data.

    Attributes:
        text: Generated embeddable text
        metadata: All original fields (flattened for nested objects)
        source_path: JSON path to this chunk's source (e.g., "[0].products[2]")
        source_file: Original file path (if from file)
        embedding_text: Enriched text optimized for embedding
        chunk_index: Index of this chunk in the sequence
    """

    text: str
    metadata: dict[str, Any]
    source_path: str = ""
    source_file: str = ""
    embedding_text: str = ""
    chunk_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "source_path": self.source_path,
            "source_file": self.source_file,
            "embedding_text": self.embedding_text,
            "chunk_index": self.chunk_index,
        }


_RECORD_QUEUE_SIZE = 16
"""How many reassembled elements the streaming worker may hold ahead of the consumer."""

_PRODUCER_POLL_SECONDS = 0.05
"""How often a blocked streaming worker re-checks whether the consumer has stopped."""

_STREAM_DONE = object()
_UNSET = object()


class _StopStreamingError(Exception):
    """Raised inside the streaming worker to abandon parsing once the consumer stops."""


def _new_container(step: Any) -> Any:
    """Create the container a path step addresses: a list for an index, else a dict."""
    return [] if isinstance(step, int) else {}


def _child(container: Any, step: Any, next_step: Any) -> Any:
    """Get (creating if needed) the child of ``container`` at ``step``."""
    if isinstance(container, list):
        if step == len(container):
            container.append(_new_container(next_step))
        return container[step]
    if step not in container:
        container[step] = _new_container(next_step)
    return container[step]


def _set_leaf(container: Any, step: Any, item: Any) -> None:
    """Set ``item`` at ``step`` of ``container``."""
    if isinstance(container, list) and step == len(container):
        container.append(item)
    else:
        container[step] = item


def _insert(root: Any, rel_path: tuple[Any, ...], item: Any) -> Any:
    """Insert a leaf value into a structure being rebuilt from streamed leaves.

    Args:
        root: The structure so far, or ``_UNSET`` before the first leaf.
        rel_path: The leaf's path relative to the structure's root. Array
            indices arrive consecutively from 0, in document order.
        item: The leaf value (a scalar, or an empty object or array).

    Returns:
        The structure's root, created on the first leaf.
    """
    if not rel_path:
        return item
    if root is _UNSET:
        root = _new_container(rel_path[0])
    container = root
    for step, next_step in itertools.pairwise(rel_path):
        container = _child(container, step, next_step)
    _set_leaf(container, rel_path[-1], item)
    return root


class _RecordElementAssembler:
    """Group streamed leaf values into records and rebuild each record's element.

    A value's record key is its path up to and including the path's first array
    index; a new record starts each time that key changes. A value whose path
    has no index is shared context and belongs to no element. Each record's
    element (the value at its record key) is passed to ``emit`` when the record
    ends.
    """

    def __init__(self, emit: Callable[[tuple[Any, ...], Any], None]) -> None:
        self._emit = emit
        self._key: tuple[Any, ...] | None = None
        self._element: Any = _UNSET

    def visit(self, item: Any, path: tuple[Any, ...]) -> None:
        """Take one streamed leaf value at ``path``."""
        first_index = next((i for i, step in enumerate(path) if isinstance(step, int)), None)
        if first_index is None:
            return
        key = path[: first_index + 1]
        if key != self._key:
            self.finish()
            self._key = key
        self._element = _insert(self._element, path[first_index + 1 :], item)

    def finish(self) -> None:
        """End the current record, if any, and emit its element."""
        if self._key is not None:
            self._emit(self._key, self._element)
        self._key = None
        self._element = _UNSET


class JSONChunker:
    """Chunker for generating chunks from JSON data with preserved metadata.

    Supports both in-memory processing and streaming for large files.

    Example:
        >>> chunker = JSONChunker()
        >>> # In-memory processing
        >>> chunks = chunker.chunk({"title": "Hello", "content": "World"})
        >>> # Streaming from file
        >>> for chunk in chunker.stream_chunks("large_data.jsonl"):
        ...     process(chunk)
    """

    def __init__(self, config: JSONChunkConfig | None = None):
        """Initialize the JSON chunker.

        Args:
            config: Configuration for chunking behavior
        """
        self.config = config or JSONChunkConfig()
        self._chunk_index = 0
        self._jinja_env: Any = None  # Lazy loaded

    def chunk(
        self,
        data: dict[str, Any] | list[dict[str, Any]],
        source: str = "",
    ) -> list[JSONChunk]:
        """Chunk in-memory JSON data.

        Args:
            data: JSON object or array of objects to chunk
            source: Optional source identifier (e.g., file path)

        Returns:
            List of JSONChunk objects
        """
        self._chunk_index = 0

        if isinstance(data, dict):
            return [self._process_item(data, source_path="", source_file=source)]
        elif isinstance(data, list):
            chunks = []
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    chunks.append(
                        self._process_item(item, source_path=f"[{idx}]", source_file=source)
                    )
            return chunks
        else:
            raise ValueError(f"Expected dict or list, got {type(data).__name__}")

    def stream_chunks(
        self,
        source: str | Path | IO[str] | IO[bytes],
        timeout: int = 10,
        is_jsonl: bool | None = None,
        source_file: str | None = None,
    ) -> Iterator[JSONChunk]:
        """Stream chunks from large JSON files without loading into memory.

        Leverages dataknobs_utils.json_utils streaming infrastructure.

        Supported formats:
        - JSON: values are grouped into records by their first array index
          (the outermost array each value's path passes through, at any
          depth); each record whose array element is an object becomes one
          chunk, with ``source_path`` set to that element's jq path (e.g.
          ``.[0]`` or ``.items[2]``). Other elements and index-free values
          yield no chunk. Chunks are produced as each record ends.
        - JSONL files: Each line is a complete JSON object
        - Compressed files: .gz files auto-detected and decompressed
          (only when ``source`` is a path; file-like sources must be
          pre-decompressed by the caller)
        - URLs: Remote JSON fetched with streaming
        - File-like objects: any object with a ``read()`` method is
          streamed forward-sequentially; ``is_jsonl`` is required to
          select JSONL vs JSON-array parsing.

        Args:
            source: File path, URL, JSON string, or file-like object.
            timeout: Request timeout for URLs (seconds).
            is_jsonl: Override automatic JSONL detection. Ignored when
                ``source`` is a path or URL (detection uses the
                extension). Required when ``source`` is a file-like
                object; defaults to ``False`` (JSON array) when omitted.
            source_file: Logical identifier attached to emitted chunks
                as ``source_file``. When ``source`` is a path, defaults
                to the path string; when ``source`` is file-like,
                defaults to an empty string.

        Yields:
            JSONChunk objects as they are processed.
        """
        self._chunk_index = 0

        if (
            not isinstance(source, (str, Path))
            and hasattr(source, "read")
            and callable(source.read)
        ):
            jsonl = bool(is_jsonl)
            source_label = source_file or ""
            if jsonl:
                yield from self._stream_jsonl(source, source_file=source_label)
            else:
                yield from self._stream_json_array(source, timeout, source_file=source_label)
            return

        source_str = str(source)
        source_label = source_file if source_file is not None else source_str
        jsonl = is_jsonl if is_jsonl is not None else self._is_jsonl_file(source_str)
        if jsonl:
            yield from self._stream_jsonl(source_str, source_file=source_label)
        else:
            yield from self._stream_json_array(source_str, timeout, source_file=source_label)

    def _is_jsonl_file(self, source: str) -> bool:
        """Check if source is a JSONL file based on extension."""
        lower = source.lower()
        return (
            lower.endswith(".jsonl")
            or lower.endswith(".jsonl.gz")
            or lower.endswith(".ndjson")
            or lower.endswith(".ndjson.gz")
        )

    def _stream_jsonl(
        self,
        source: str | IO[str] | IO[bytes],
        source_file: str | None = None,
    ) -> Iterator[JSONChunk]:
        """Stream from a JSONL file or file-like object.

        One JSON object per line. Path sources auto-decompress
        ``.gz``; file-like sources are consumed as-is (callers
        pre-decompress).
        """
        import gzip

        label = (
            source_file if source_file is not None else (source if isinstance(source, str) else "")
        )

        if hasattr(source, "read") and callable(source.read):  # type: ignore[union-attr]
            yield from self._iter_jsonl_lines(source, label)  # type: ignore[arg-type]
            return

        source_str = str(source)
        source_path = Path(source_str)

        def open_gzip(p: Path) -> Any:
            return gzip.open(p, "rt", encoding="utf-8")

        def open_text(p: Path) -> Any:
            return open(p, encoding="utf-8")

        opener = open_gzip if source_str.lower().endswith(".gz") else open_text

        with opener(source_path) as f:
            yield from self._iter_jsonl_lines(f, label)

    def _iter_jsonl_lines(
        self,
        stream: IO[str] | IO[bytes],
        source_label: str,
    ) -> Iterator[JSONChunk]:
        """Yield chunks from each non-empty JSONL line in ``stream``."""
        lines: Iterable[str | bytes] = stream
        for line_num, raw in enumerate(lines):
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            line = text.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield self._process_item(
                    item,
                    source_path=f"[{line_num}]",
                    source_file=source_label,
                )

    def _stream_json_array(
        self,
        source: str | IO[str] | IO[bytes],
        timeout: int,
        source_file: str | None = None,
    ) -> Iterator[JSONChunk]:
        """Stream chunks from JSON (not JSONL) using json_utils infrastructure.

        The source is parsed by ``stream_json_data`` on a worker thread, which
        groups the streamed values into records by their first array index and
        reassembles each record's element (the array element itself, keys in
        source order; index-free values are not part of any element). Elements
        are handed to this generator through a bounded queue as each record
        ends, so chunks are produced incrementally. One chunk is yielded per
        element that is a JSON object. Closing the generator early stops the
        worker; a parsing or fetching error is re-raised here.

        Args:
            source: Path, URL, JSON string, or file-like object.
            timeout: Request timeout for URL sources (seconds).
            source_file: Label attached to each chunk as ``source_file``.
                Defaults to ``source`` when it is a string, else "".

        Yields:
            JSONChunk objects, in record order.
        """
        # Imported here so importing this module does not pull in json_utils'
        # heavier dependencies until JSON streaming is actually used.
        from dataknobs_utils.json_utils import build_jq_path, stream_json_data

        label = (
            source_file if source_file is not None else (source if isinstance(source, str) else "")
        )
        records: queue.Queue[Any] = queue.Queue(maxsize=_RECORD_QUEUE_SIZE)
        stop = threading.Event()

        def put(entry: Any) -> None:
            while not stop.is_set():
                try:
                    records.put(entry, timeout=_PRODUCER_POLL_SECONDS)
                    return
                except queue.Full:
                    continue
            raise _StopStreamingError

        assembler = _RecordElementAssembler(lambda key, element: put((key, element)))

        def visit(item: Any, path: tuple[Any, ...]) -> None:
            if stop.is_set():
                raise _StopStreamingError
            assembler.visit(item, path)

        def produce() -> None:
            try:
                stream_json_data(source, visit, timeout=timeout)
                assembler.finish()
                put(_STREAM_DONE)
            except _StopStreamingError:
                return
            except BaseException as err:  # handed to the consumer, which re-raises it
                with contextlib.suppress(_StopStreamingError):
                    put(err)

        worker = threading.Thread(target=produce, name="json-chunker-stream", daemon=True)
        worker.start()
        try:
            while True:
                entry = records.get()
                if entry is _STREAM_DONE:
                    return
                if isinstance(entry, BaseException):
                    raise entry
                record_key, element = entry
                if not isinstance(element, dict):
                    continue
                jq_path = build_jq_path(record_key, keep_list_idxs=True)
                yield self._process_item(
                    element,
                    source_path=f".{jq_path}" if jq_path.startswith("[") else jq_path,
                    source_file=label,
                )
        finally:
            stop.set()
            worker.join()

    def _fallback_load(self, source: str) -> Iterator[JSONChunk]:
        """Fallback: load entire file when streaming utils unavailable."""
        import gzip
        from pathlib import Path

        source_path = Path(source)
        if not source_path.exists():
            return

        if source.lower().endswith(".gz"):
            with gzip.open(source_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(source_path, encoding="utf-8") as f:
                data = json.load(f)

        yield from self.chunk(data, source=source)

    def _process_item(
        self,
        item: dict[str, Any],
        source_path: str,
        source_file: str,
    ) -> JSONChunk:
        """Process a single JSON object into a chunk.

        Args:
            item: JSON object to process
            source_path: JSON path to this item
            source_file: Source file path

        Returns:
            JSONChunk with generated text and preserved metadata
        """
        # Flatten nested structure for metadata
        flat_metadata = self._flatten(item)

        # Generate text
        if self.config.text_template:
            text = self._render_template(item)
        else:
            text = self._auto_generate_text(item)

        # Truncate if needed
        if len(text) > self.config.max_chunk_size:
            text = text[: self.config.max_chunk_size - 3] + "..."

        # Generate embedding text (enriched with context)
        embedding_text = self._build_embedding_text(item, text)

        chunk = JSONChunk(
            text=text,
            metadata=flat_metadata,
            source_path=source_path,
            source_file=source_file,
            embedding_text=embedding_text,
            chunk_index=self._chunk_index,
        )
        self._chunk_index += 1
        return chunk

    def _flatten(
        self,
        obj: dict[str, Any],
        prefix: str = "",
    ) -> dict[str, Any]:
        """Flatten nested dict/list structure using dot notation.

        Args:
            obj: Object to flatten
            prefix: Current key prefix

        Returns:
            Flattened dictionary
        """
        result: dict[str, Any] = {}
        sep = self.config.nested_separator

        for key, value in obj.items():
            full_key = f"{prefix}{sep}{key}" if prefix else key

            if isinstance(value, dict):
                result.update(self._flatten(value, full_key))
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    # List of objects - store count and flatten first
                    result[f"{full_key}._count"] = len(value)
                    if value:
                        result.update(self._flatten(value[0], f"{full_key}[0]"))
                else:
                    # List of primitives - store as-is
                    result[full_key] = value
            else:
                result[full_key] = value

        return result

    def _auto_generate_text(self, item: dict[str, Any]) -> str:
        """Auto-generate embeddable text from JSON object.

        Algorithm:
        1. Extract title/name/id field as primary identifier
        2. Concatenate text-like fields (description, content, summary)
        3. Format nested objects with field names
        4. Handle arrays based on config

        Args:
            item: JSON object to convert to text

        Returns:
            Generated text string
        """
        parts: list[str] = []

        # Use specific fields if configured
        if self.config.text_fields:
            for field_name in self.config.text_fields:
                value = self._get_nested_value(item, field_name)
                if value is not None:
                    parts.append(self._format_value(field_name, value))
            return "\n".join(parts)

        # Auto-detect: prioritize known text fields
        used_keys: set[str] = set()

        # First pass: extract primary identifier
        for key in ["title", "name", "label"]:
            if key in item and isinstance(item[key], str):
                parts.append(item[key])
                used_keys.add(key)
                break

        # Second pass: extract text content fields
        for key, value in item.items():
            if key in used_keys:
                continue
            lower_key = key.lower()
            if lower_key in TEXT_FIELD_NAMES:
                if isinstance(value, str) and value.strip():
                    if not self._is_technical_value(value):
                        if self.config.include_field_names and key not in (
                            "content",
                            "text",
                            "body",
                        ):
                            parts.append(f"{key}: {value}")
                        else:
                            parts.append(value)
                        used_keys.add(key)

        # Third pass: include other non-technical fields
        for key, value in item.items():
            if key in used_keys:
                continue
            lower_key = key.lower()
            if lower_key in SKIP_FIELD_NAMES:
                continue
            if key.startswith("_"):
                continue

            formatted = self._format_value(key, value)
            if formatted:
                parts.append(formatted)

        return "\n".join(parts)

    def _format_value(self, key: str, value: Any, depth: int = 0) -> str:
        """Format a value for text generation.

        Args:
            key: Field name
            value: Field value
            depth: Nesting depth (for indentation)

        Returns:
            Formatted string
        """
        if value is None:
            return ""

        if isinstance(value, str):
            if self.config.skip_technical_fields and self._is_technical_value(value):
                return ""
            if self.config.include_field_names:
                return f"{key}: {value}"
            return value

        if isinstance(value, bool):
            if self.config.include_field_names:
                return f"{key}: {'yes' if value else 'no'}"
            return "yes" if value else "no"

        if isinstance(value, (int, float)):
            if self.config.include_field_names:
                return f"{key}: {value}"
            return str(value)

        if isinstance(value, list):
            if not value:
                return ""
            if isinstance(value[0], dict):
                # List of objects - summarize
                return f"{key}: {len(value)} items"
            # List of primitives
            if self.config.array_handling == "join":
                joined = ", ".join(str(v) for v in value[:10])
                if len(value) > 10:
                    joined += f"... ({len(value)} total)"
                if self.config.include_field_names:
                    return f"{key}: {joined}"
                return joined
            elif self.config.array_handling == "first":
                return self._format_value(key, value[0], depth)
            # "expand" - return all items
            items = [str(v) for v in value]
            if self.config.include_field_names:
                return f"{key}: {', '.join(items)}"
            return ", ".join(items)

        if isinstance(value, dict):
            # Nested object - format recursively
            sub_parts = []
            for k, v in value.items():
                formatted = self._format_value(k, v, depth + 1)
                if formatted:
                    sub_parts.append(formatted)
            if sub_parts:
                if self.config.include_field_names:
                    return f"{key}: {'; '.join(sub_parts)}"
                return "; ".join(sub_parts)
            return ""

        return ""

    def _is_technical_value(self, value: str) -> bool:
        """Check if a string value appears to be technical/non-text."""
        if not self.config.skip_technical_fields:
            return False

        if len(value) < 10:
            return False

        if UUID_PATTERN.match(value):
            return True
        if BASE64_PATTERN.match(value) and len(value) > 50:
            return True
        if TIMESTAMP_PATTERN.match(value):
            return True

        return False

    def _get_nested_value(self, obj: dict[str, Any], path: str) -> Any:
        """Get a value from a nested dict using dot notation path.

        Args:
            obj: Object to traverse
            path: Dot-notation path (e.g., "config.database.host")

        Returns:
            Value at path, or None if not found
        """
        parts = path.split(self.config.nested_separator)
        current: Any = obj

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _render_template(self, item: dict[str, Any]) -> str:
        """Render text using Jinja2 template.

        Args:
            item: JSON object to render

        Returns:
            Rendered text string
        """
        if self._jinja_env is None:
            try:
                from jinja2 import Environment

                self._jinja_env = Environment()
            except ImportError as err:
                raise ImportError(
                    "jinja2 is required for template-based text generation. "
                    "Install it with: pip install jinja2"
                ) from err

        template = self._jinja_env.from_string(self.config.text_template)
        return template.render(**item)

    def _build_embedding_text(self, item: dict[str, Any], base_text: str) -> str:
        """Build enriched text optimized for embedding.

        Adds context that improves semantic search quality.

        Args:
            item: Original JSON object
            base_text: Generated base text

        Returns:
            Enriched text for embedding
        """
        parts = []

        # Add type/category context if available
        for key in ["type", "category", "kind", "class"]:
            if key in item and isinstance(item[key], str):
                parts.append(f"[{item[key].upper()}]")
                break

        parts.append(base_text)

        # Add tags/keywords if available
        for key in ["tags", "keywords", "labels"]:
            if key in item and isinstance(item[key], list):
                tags = [str(t) for t in item[key][:5] if isinstance(t, str)]
                if tags:
                    parts.append(f"Tags: {', '.join(tags)}")
                break

        return " ".join(parts)
