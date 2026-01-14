"""Protocol definition for knowledge resource backends.

This module defines the KnowledgeResourceBackend protocol, which provides
a unified interface for storing and retrieving knowledge resource files.

Unlike RegistryBackend (for config records) or dataknobs_data (for structured
data), this protocol handles raw file storage with directory-like organization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, BinaryIO, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .models import KnowledgeBaseInfo, KnowledgeFile


@runtime_checkable
class KnowledgeResourceBackend(Protocol):
    r"""Protocol for knowledge resource file storage.

    This protocol defines the interface for storing and retrieving raw files
    organized by domain_id and path. Implementations include:

    - InMemoryKnowledgeBackend: For testing
    - FileKnowledgeBackend: For local development
    - S3KnowledgeBackend: For production deployments

    Structure:
        {domain_id}/
            _metadata.json       # KB info and file index
            content/
                file1.md
                subdir/
                    file2.json

    Example:
        ```python
        from dataknobs_bots.knowledge.storage import create_knowledge_backend

        # Create backend
        backend = create_knowledge_backend("file", {"path": "./data/knowledge"})
        await backend.initialize()

        # Create knowledge base
        await backend.create_kb("my-domain")

        # Upload files
        await backend.put_file(
            "my-domain",
            "content/intro.md",
            b"# Introduction\n\nWelcome to the docs."
        )

        # List files
        files = await backend.list_files("my-domain")
        for f in files:
            print(f"{f.path}: {f.size_bytes} bytes")

        # Get file content
        content = await backend.get_file("my-domain", "content/intro.md")
        if content:
            print(content.decode("utf-8"))

        # Cleanup
        await backend.close()
        ```
    """

    async def initialize(self) -> None:
        """Initialize the backend.

        Creates connections, verifies access, and performs any necessary setup.
        Must be called before using other methods.
        """
        ...

    async def close(self) -> None:
        """Close connections and release resources.

        Should be called when done using the backend.
        """
        ...

    # --- File Operations ---

    async def put_file(
        self,
        domain_id: str,
        path: str,
        content: bytes | BinaryIO,
        content_type: str | None = None,
        metadata: dict | None = None,
    ) -> KnowledgeFile:
        """Upload or update a file.

        If the file already exists, it will be overwritten. The knowledge base's
        version is incremented and last_updated is set to now.

        Args:
            domain_id: Knowledge base identifier
            path: Relative path within KB (e.g., "curriculum/topics.json")
            content: File content as bytes or file-like object
            content_type: MIME type (auto-detected if not provided)
            metadata: Optional custom metadata

        Returns:
            KnowledgeFile with upload details including checksum and timestamp

        Raises:
            ValueError: If domain_id doesn't exist (call create_kb first)
        """
        ...

    async def get_file(self, domain_id: str, path: str) -> bytes | None:
        """Get file content.

        Args:
            domain_id: Knowledge base identifier
            path: Relative path within KB

        Returns:
            File content as bytes, or None if file doesn't exist
        """
        ...

    async def stream_file(
        self, domain_id: str, path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes] | None:
        """Stream file content for large files.

        More memory-efficient than get_file for large files.

        Args:
            domain_id: Knowledge base identifier
            path: Relative path within KB
            chunk_size: Size of chunks to yield

        Returns:
            Async iterator yielding bytes chunks, or None if file doesn't exist
        """
        ...

    async def delete_file(self, domain_id: str, path: str) -> bool:
        """Delete a file.

        Updates the knowledge base's version and last_updated.

        Args:
            domain_id: Knowledge base identifier
            path: Relative path within KB

        Returns:
            True if file was deleted, False if not found
        """
        ...

    async def list_files(
        self, domain_id: str, prefix: str | None = None
    ) -> list[KnowledgeFile]:
        """List all files in a knowledge base.

        Args:
            domain_id: Knowledge base identifier
            prefix: Optional path prefix to filter (e.g., "curriculum/")

        Returns:
            List of KnowledgeFile metadata objects
        """
        ...

    async def file_exists(self, domain_id: str, path: str) -> bool:
        """Check if a file exists.

        Args:
            domain_id: Knowledge base identifier
            path: Relative path within KB

        Returns:
            True if file exists
        """
        ...

    # --- Knowledge Base Operations ---

    async def create_kb(
        self, domain_id: str, metadata: dict | None = None
    ) -> KnowledgeBaseInfo:
        """Create a new knowledge base.

        Args:
            domain_id: Unique identifier for the knowledge base
            metadata: Optional metadata for the knowledge base

        Returns:
            KnowledgeBaseInfo for the newly created KB

        Raises:
            ValueError: If a KB with this domain_id already exists
        """
        ...

    async def get_info(self, domain_id: str) -> KnowledgeBaseInfo | None:
        """Get knowledge base metadata.

        Args:
            domain_id: Knowledge base identifier

        Returns:
            KnowledgeBaseInfo, or None if KB doesn't exist
        """
        ...

    async def delete_kb(self, domain_id: str) -> bool:
        """Delete entire knowledge base and all files.

        Warning: This permanently deletes all files in the knowledge base.

        Args:
            domain_id: Knowledge base identifier

        Returns:
            True if KB was deleted, False if not found
        """
        ...

    async def list_kbs(self) -> list[KnowledgeBaseInfo]:
        """List all knowledge bases.

        Returns:
            List of KnowledgeBaseInfo for all knowledge bases
        """
        ...

    # --- Ingestion Status ---

    async def set_ingestion_status(
        self,
        domain_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        """Update ingestion status for a knowledge base.

        Called by KnowledgeIngestionManager to track ingestion progress.

        Args:
            domain_id: Knowledge base identifier
            status: Status string ("pending", "ingesting", "ready", "error")
            error: Error message if status is "error"

        Raises:
            ValueError: If domain_id doesn't exist
        """
        ...

    # --- Change Detection ---

    async def get_checksum(self, domain_id: str) -> str:
        """Get combined checksum of all files for change detection.

        The checksum changes whenever any file is added, modified, or deleted.
        Useful for determining if re-ingestion is needed.

        Args:
            domain_id: Knowledge base identifier

        Returns:
            Combined checksum string (e.g., MD5 of all file checksums)

        Raises:
            ValueError: If domain_id doesn't exist
        """
        ...

    async def has_changes_since(self, domain_id: str, version: str) -> bool:
        """Check if KB has changed since given version.

        Args:
            domain_id: Knowledge base identifier
            version: Previous version string to compare against

        Returns:
            True if current version differs from given version

        Raises:
            ValueError: If domain_id doesn't exist
        """
        ...
