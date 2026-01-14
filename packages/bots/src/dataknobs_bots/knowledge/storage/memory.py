"""In-memory knowledge resource backend for testing.

This backend stores all files in memory, making it ideal for unit tests
and development scenarios where persistence is not needed.
"""

from __future__ import annotations

import hashlib
import mimetypes
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from io import BytesIO
from typing import BinaryIO

from .models import IngestionStatus, KnowledgeBaseInfo, KnowledgeFile


class InMemoryKnowledgeBackend:
    """In-memory implementation of KnowledgeResourceBackend.

    Stores all files and metadata in dictionaries. Ideal for:
    - Unit testing
    - Development
    - Short-lived processes

    Note: All data is lost when the instance is garbage collected.

    Example:
        ```python
        backend = InMemoryKnowledgeBackend()
        await backend.initialize()

        await backend.create_kb("test-domain")
        await backend.put_file("test-domain", "doc.md", b"# Hello")

        content = await backend.get_file("test-domain", "doc.md")
        assert content == b"# Hello"

        await backend.close()
        ```
    """

    def __init__(self) -> None:
        """Initialize the in-memory backend."""
        # domain_id -> KnowledgeBaseInfo
        self._kb_info: dict[str, KnowledgeBaseInfo] = {}

        # domain_id -> path -> file content (bytes)
        self._files: dict[str, dict[str, bytes]] = {}

        # domain_id -> path -> KnowledgeFile
        self._file_metadata: dict[str, dict[str, KnowledgeFile]] = {}

        self._initialized = False

    @classmethod
    def from_config(cls, _config: dict | None = None) -> InMemoryKnowledgeBackend:
        """Create from configuration dict.

        Args:
            _config: Not used, exists for API compatibility

        Returns:
            New InMemoryKnowledgeBackend instance
        """
        return cls()

    async def initialize(self) -> None:
        """Initialize the backend. No-op for in-memory."""
        self._initialized = True

    async def close(self) -> None:
        """Close the backend. No-op for in-memory."""
        self._initialized = False

    # --- File Operations ---

    async def put_file(
        self,
        domain_id: str,
        path: str,
        content: bytes | BinaryIO,
        content_type: str | None = None,
        metadata: dict | None = None,
    ) -> KnowledgeFile:
        """Upload or update a file."""
        if domain_id not in self._kb_info:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        # Get content as bytes
        if isinstance(content, bytes):
            data = content
        else:
            data = content.read()

        # Auto-detect content type
        if content_type is None:
            guessed_type, _ = mimetypes.guess_type(path)
            content_type = guessed_type or "application/octet-stream"

        # Calculate checksum
        checksum = hashlib.md5(data).hexdigest()

        # Check if this is an update
        is_new = path not in self._files.get(domain_id, {})

        # Store file
        if domain_id not in self._files:
            self._files[domain_id] = {}
        if domain_id not in self._file_metadata:
            self._file_metadata[domain_id] = {}

        self._files[domain_id][path] = data

        # Create file metadata
        file_info = KnowledgeFile(
            path=path,
            content_type=content_type,
            size_bytes=len(data),
            checksum=checksum,
            uploaded_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        self._file_metadata[domain_id][path] = file_info

        # Update KB info
        kb_info = self._kb_info[domain_id]
        kb_info.last_updated = datetime.now(timezone.utc)
        kb_info.increment_version()
        if is_new:
            kb_info.file_count += 1
        kb_info.total_size_bytes = sum(len(f) for f in self._files[domain_id].values())

        return file_info

    async def get_file(self, domain_id: str, path: str) -> bytes | None:
        """Get file content."""
        if domain_id not in self._files:
            return None
        return self._files[domain_id].get(path)

    async def stream_file(
        self, domain_id: str, path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes] | None:
        """Stream file content."""
        content = await self.get_file(domain_id, path)
        if content is None:
            return None

        async def _generator() -> AsyncIterator[bytes]:
            stream = BytesIO(content)
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        return _generator()

    async def delete_file(self, domain_id: str, path: str) -> bool:
        """Delete a file."""
        if domain_id not in self._files:
            return False

        if path not in self._files[domain_id]:
            return False

        # Remove file
        del self._files[domain_id][path]
        if path in self._file_metadata[domain_id]:
            del self._file_metadata[domain_id][path]

        # Update KB info
        kb_info = self._kb_info[domain_id]
        kb_info.last_updated = datetime.now(timezone.utc)
        kb_info.increment_version()
        kb_info.file_count = len(self._files[domain_id])
        kb_info.total_size_bytes = sum(len(f) for f in self._files[domain_id].values())

        return True

    async def list_files(
        self, domain_id: str, prefix: str | None = None
    ) -> list[KnowledgeFile]:
        """List all files in a knowledge base."""
        if domain_id not in self._file_metadata:
            return []

        files = list(self._file_metadata[domain_id].values())

        if prefix:
            files = [f for f in files if f.path.startswith(prefix)]

        # Sort by path for consistent ordering
        files.sort(key=lambda f: f.path)
        return files

    async def file_exists(self, domain_id: str, path: str) -> bool:
        """Check if a file exists."""
        return domain_id in self._files and path in self._files[domain_id]

    # --- Knowledge Base Operations ---

    async def create_kb(
        self, domain_id: str, metadata: dict | None = None
    ) -> KnowledgeBaseInfo:
        """Create a new knowledge base."""
        if domain_id in self._kb_info:
            raise ValueError(f"Knowledge base '{domain_id}' already exists")

        kb_info = KnowledgeBaseInfo(
            domain_id=domain_id,
            file_count=0,
            total_size_bytes=0,
            last_updated=datetime.now(timezone.utc),
            version="1",
            ingestion_status=IngestionStatus.PENDING,
            metadata=metadata or {},
        )
        self._kb_info[domain_id] = kb_info
        self._files[domain_id] = {}
        self._file_metadata[domain_id] = {}

        return kb_info

    async def get_info(self, domain_id: str) -> KnowledgeBaseInfo | None:
        """Get knowledge base metadata."""
        return self._kb_info.get(domain_id)

    async def delete_kb(self, domain_id: str) -> bool:
        """Delete entire knowledge base and all files."""
        if domain_id not in self._kb_info:
            return False

        del self._kb_info[domain_id]
        if domain_id in self._files:
            del self._files[domain_id]
        if domain_id in self._file_metadata:
            del self._file_metadata[domain_id]

        return True

    async def list_kbs(self) -> list[KnowledgeBaseInfo]:
        """List all knowledge bases."""
        return sorted(self._kb_info.values(), key=lambda kb: kb.domain_id)

    # --- Ingestion Status ---

    async def set_ingestion_status(
        self,
        domain_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        """Update ingestion status for a knowledge base."""
        if domain_id not in self._kb_info:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        kb_info = self._kb_info[domain_id]
        kb_info.ingestion_status = IngestionStatus(status)
        kb_info.ingestion_error = error

    # --- Change Detection ---

    async def get_checksum(self, domain_id: str) -> str:
        """Get combined checksum of all files."""
        if domain_id not in self._kb_info:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        if domain_id not in self._file_metadata or not self._file_metadata[domain_id]:
            return ""

        # Combine all file checksums sorted by path
        checksums = sorted(
            f"{f.path}:{f.checksum}" for f in self._file_metadata[domain_id].values()
        )
        combined = ":".join(checksums)
        return hashlib.md5(combined.encode()).hexdigest()

    async def has_changes_since(self, domain_id: str, version: str) -> bool:
        """Check if KB has changed since given version."""
        if domain_id not in self._kb_info:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        return self._kb_info[domain_id].version != version
