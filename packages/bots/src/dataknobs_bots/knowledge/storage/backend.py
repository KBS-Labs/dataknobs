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

    from .models import (
        ChangeSet,
        IngestionStatus,
        KnowledgeBaseInfo,
        KnowledgeFile,
    )


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
        status: IngestionStatus | str,
        error: str | None = None,
        *,
        generation: str | None = None,
    ) -> None:
        """Update ingestion status for a knowledge base.

        Called by KnowledgeIngestionManager to track ingestion progress.

        Args:
            domain_id: Knowledge base identifier
            status: An :class:`IngestionStatus` member (the preferred,
                typed form) or its string value (e.g. ``"ready"``).
                Implementations normalize via
                :func:`normalize_ingestion_status`.
            error: Error message if status is the error state
            generation: In-flight TOMBSTONE swap token. Passed by the
                SWAPPING transition so an interrupted swap can be
                reconciled; defaults to ``None`` and is **always**
                written through (so any non-SWAPPING transition clears
                a stale token). Implementations store it on
                :attr:`KnowledgeBaseInfo.generation`.

        Raises:
            ValueError: If ``domain_id`` doesn't exist.
            ValidationError: If ``status`` is a string with no matching
                :class:`IngestionStatus` member (raised by
                :func:`normalize_ingestion_status`; carries the list of
                accepted values). ``ValidationError`` is a
                :class:`~dataknobs_common.exceptions.DataknobsError`,
                not a ``ValueError`` subclass.
        """
        ...

    # --- Change Detection ---

    async def get_checksum(self, domain_id: str) -> str:
        """Canonical content-snapshot identity of the whole KB.

        A stable hash over every file's ``path:checksum``, so it changes
        whenever any file is added, modified, or deleted. **This value is
        the version**: capture it and pass it back to
        :meth:`has_changes_since` / :meth:`list_changes_since`. The empty
        KB has identity ``""``.

        Note: this is *not* the monotonic ``KnowledgeBaseInfo.version``
        counter, which is retained for cache-invalidation/display only
        and must not be passed to change-detection methods.

        Args:
            domain_id: Knowledge base identifier

        Returns:
            Canonical snapshot identity string (MD5 of file checksums)

        Raises:
            ValueError: If domain_id doesn't exist
        """
        ...

    async def has_changes_since(self, domain_id: str, version: str) -> bool:
        """Check if the KB changed since the given snapshot version.

        The degenerate case of :meth:`list_changes_since`:
        ``not (await list_changes_since(domain_id, version)).is_empty``.
        An unresolvable version is treated as "assume changed" so callers
        safely re-ingest (no exception for that case).

        Args:
            domain_id: Knowledge base identifier
            version: A value previously returned by :meth:`get_checksum`

        Returns:
            True if the current snapshot differs from ``version``

        Raises:
            ValueError: If domain_id doesn't exist
        """
        ...

    async def list_changes_since(
        self, domain_id: str, version: str
    ) -> ChangeSet:
        """File-level diff of the KB since the given snapshot version.

        ``version`` is a value previously returned by
        :meth:`get_checksum`. When it equals the current identity the
        result is empty. Otherwise the current files are diffed against
        the snapshot at ``version``; backends without a retained snapshot
        for that version either report every current file as ``added``
        (correct, non-minimal) or raise :class:`InvalidVersionError`.

        Args:
            domain_id: Knowledge base identifier
            version: A value previously returned by :meth:`get_checksum`

        Returns:
            A :class:`ChangeSet` (added / modified / deleted + the
            current canonical version)

        Raises:
            ValueError: If domain_id doesn't exist
            InvalidVersionError: If ``version`` differs from the current
                identity and the backend cannot resolve it to a snapshot
                (predates retention / unknown). Consumers fall back to a
                full re-ingest.
        """
        ...
