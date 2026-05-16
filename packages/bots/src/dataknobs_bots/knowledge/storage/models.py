"""Data models for knowledge resource storage.

These models represent metadata about files and knowledge bases stored
in a KnowledgeResourceBackend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class IngestionStatus(Enum):
    """Status of knowledge base ingestion into vector storage.

    The ingestion process loads files from KnowledgeResourceBackend
    and processes them into the RAGKnowledgeBase vector store.
    """

    PENDING = "pending"
    """Files uploaded but not yet ingested."""

    INGESTING = "ingesting"
    """Currently processing files into vectors."""

    READY = "ready"
    """Ingestion complete, knowledge base is ready for queries."""

    ERROR = "error"
    """Ingestion failed with an error."""

    SWAPPING = "swapping"
    """A zero-downtime swap is in progress (old + new chunks coexist).

    Reserved for the later-phase ``TOMBSTONE`` re-ingest path (no
    ingestion flow transitions to this value yet): the previous
    generation stays query-visible while the new one is written, then
    is atomically retired. Reads stay served throughout.
    """


class InvalidVersionError(ValueError):
    """A version string cannot be interpreted by a backend.

    Raised by :meth:`KnowledgeResourceBackend.list_changes_since` when
    the supplied version predates the backend's snapshot retention (or
    is otherwise unknown), so a minimal diff cannot be produced.
    Consumers catch this and fall back to a full re-ingest. It subclasses
    :class:`ValueError` for backward compatibility with callers that
    already treat version problems as ``ValueError``.
    """


@dataclass
class KnowledgeFile:
    """Metadata about a file in a knowledge base.

    Represents a single file stored in a KnowledgeResourceBackend.
    Files are organized by domain_id and relative path.

    Attributes:
        path: Relative path within KB (e.g., "curriculum/topics.json")
        content_type: MIME type (e.g., "text/markdown", "application/json")
        size_bytes: Size of the file in bytes
        checksum: MD5 or SHA256 hash for change detection
        uploaded_at: When the file was uploaded/last modified
        metadata: Optional custom metadata attached to the file

    Example:
        ```python
        file = KnowledgeFile(
            path="content/introduction.md",
            content_type="text/markdown",
            size_bytes=1234,
            checksum="abc123...",
            uploaded_at=datetime.now(timezone.utc),
            metadata={"author": "Jane Doe"}
        )
        ```
    """

    path: str
    content_type: str
    size_bytes: int
    checksum: str
    uploaded_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "uploaded_at": self.uploaded_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeFile:
        """Create from dictionary."""
        uploaded_at = data.get("uploaded_at")
        if isinstance(uploaded_at, str):
            uploaded_at = datetime.fromisoformat(uploaded_at)
        elif uploaded_at is None:
            uploaded_at = datetime.now(timezone.utc)

        return cls(
            path=data["path"],
            content_type=data.get("content_type", "application/octet-stream"),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum", ""),
            uploaded_at=uploaded_at,
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class ChangeSet:
    """File-level diff between two snapshots of a knowledge base.

    Returned by :meth:`KnowledgeResourceBackend.list_changes_since`.
    ``added``/``modified`` carry full :class:`KnowledgeFile` metadata;
    ``deleted`` carries paths only (the files no longer exist to
    describe). The three collections are disjoint by construction
    (a path appears in at most one) and are stored as tuples so the
    instance is genuinely immutable (``frozen=True`` alone would not
    stop ``cs.added.append(...)``). Lists may be passed in; they are
    normalized to tuples. ``version`` is the canonical
    snapshot identity of the *current* state (equal to
    :meth:`KnowledgeResourceBackend.get_checksum`), so it can be
    captured and passed back to a later ``list_changes_since`` call.

    Attributes:
        added: Files present now but absent at the compared version
        modified: Files present at both but with a changed checksum
        deleted: Paths present at the compared version but absent now
        version: Canonical snapshot id of the current state

    Example:
        ```python
        v = await backend.get_checksum("my-domain")
        await backend.put_file("my-domain", "new.md", b"# New")
        change = await backend.list_changes_since("my-domain", v)
        assert [f.path for f in change.added] == ["new.md"]
        assert not change.is_empty
        ```
    """

    added: Sequence[KnowledgeFile]
    modified: Sequence[KnowledgeFile]
    deleted: Sequence[str]
    version: str

    def __post_init__(self) -> None:
        # frozen=True blocks rebinding but not in-place mutation of a
        # list field; store tuples so the value object is truly immutable.
        object.__setattr__(self, "added", tuple(self.added))
        object.__setattr__(self, "modified", tuple(self.modified))
        object.__setattr__(self, "deleted", tuple(self.deleted))

    @property
    def is_empty(self) -> bool:
        """``True`` when nothing was added, modified, or deleted."""
        return not (self.added or self.modified or self.deleted)


@dataclass
class KnowledgeBaseInfo:
    """Metadata about a knowledge base.

    A knowledge base is a collection of files under a domain_id,
    representing the knowledge for a particular domain or bot.

    Attributes:
        domain_id: Unique identifier for this knowledge base
        file_count: Number of files in the knowledge base
        total_size_bytes: Total size of all files
        last_updated: When any file was last added/modified
        version: Monotonic counter, incremented on any change. Retained
            for cache-invalidation and display only. **No longer the
            change-detection key** — change detection uses the canonical
            content snapshot (see
            :meth:`KnowledgeResourceBackend.get_checksum` /
            :meth:`~KnowledgeResourceBackend.list_changes_since`). Do not
            pass this counter to ``has_changes_since``; pass a
            ``get_checksum`` value.
        ingestion_status: Current ingestion status
        ingestion_error: Error message if ingestion failed
        vector_store_path: Optional path/identifier for persisted vector store
        metadata: Optional custom metadata for the knowledge base

    Example:
        ```python
        info = KnowledgeBaseInfo(
            domain_id="cooking-assistant",
            file_count=42,
            total_size_bytes=123456,
            last_updated=datetime.now(timezone.utc),
            version="1",
            ingestion_status=IngestionStatus.READY,
        )
        ```
    """

    domain_id: str
    file_count: int
    total_size_bytes: int
    last_updated: datetime
    version: str
    ingestion_status: IngestionStatus = IngestionStatus.PENDING
    ingestion_error: str | None = None
    vector_store_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "domain_id": self.domain_id,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "last_updated": self.last_updated.isoformat(),
            "version": self.version,
            "ingestion_status": self.ingestion_status.value,
            "ingestion_error": self.ingestion_error,
            "vector_store_path": self.vector_store_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeBaseInfo:
        """Create from dictionary."""
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        elif last_updated is None:
            last_updated = datetime.now(timezone.utc)

        status = data.get("ingestion_status", "pending")
        if isinstance(status, str):
            status = IngestionStatus(status)

        return cls(
            domain_id=data["domain_id"],
            file_count=data.get("file_count", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
            last_updated=last_updated,
            version=data.get("version", "1"),
            ingestion_status=status,
            ingestion_error=data.get("ingestion_error"),
            vector_store_path=data.get("vector_store_path"),
            metadata=data.get("metadata", {}),
        )

    def increment_version(self) -> None:
        """Increment the version string."""
        try:
            self.version = str(int(self.version) + 1)
        except ValueError:
            # If version is not numeric, append a timestamp
            self.version = f"{self.version}.{int(datetime.now(timezone.utc).timestamp())}"
