"""Data models for knowledge resource storage.

These models represent metadata about files and knowledge bases stored
in a KnowledgeResourceBackend.
"""

from __future__ import annotations

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
        version: Incremented on any change (for cache invalidation)
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
