"""In-memory knowledge resource backend for testing.

This backend stores all files in memory, making it ideal for unit tests
and development scenarios where persistence is not needed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from io import BytesIO
from typing import BinaryIO

from .key_layout import KnowledgeKeyKind
from .mixin import KnowledgeResourceBackendMixin
from .models import (
    IngestionStatus,
    InvalidVersionError,
    KnowledgeBaseInfo,
    KnowledgeFile,
    normalize_ingestion_status,
)


class InMemoryKnowledgeBackend(KnowledgeResourceBackendMixin):
    """In-memory implementation of KnowledgeResourceBackend.

    Stores all files and metadata in dictionaries. Ideal for:
    - Unit testing
    - Development
    - Short-lived processes

    Note: All data is lost when the instance is garbage collected.

    Event triggers: not meaningful for in-process storage — there is no
    external observer to filter. :meth:`key_pattern` exists for protocol
    symmetry and returns the empty string ``""``; :meth:`classify_key`
    inherits the canonical implementation from the mixin and works
    against the same constants every other in-tree backend uses.

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

        # domain_id -> canonical version (get_checksum) -> {path: checksum}.
        # In-process per-version store backing _load_snapshot so memory
        # produces minimal diffs (a native per-version diff for the
        # file/S3 backends is a possible future enhancement).
        self._snapshots: dict[str, dict[str, dict[str, str]]] = {}

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

        # Get content as bytes (file-like reads are offloaded off-loop).
        data = await self._read_content_bytes(content)

        # Auto-detect content type. The first mimetypes lookup lazily
        # reads the system mime database from disk, so offload it.
        if content_type is None:
            content_type = await asyncio.to_thread(
                self._guess_content_type, path
            )

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
        await self._record_snapshot(domain_id)

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
        await self._record_snapshot(domain_id)

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
        await self._record_snapshot(domain_id)  # baseline: "" -> {}

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
        self._snapshots.pop(domain_id, None)

        return True

    async def list_kbs(self) -> list[KnowledgeBaseInfo]:
        """List all knowledge bases."""
        return sorted(self._kb_info.values(), key=lambda kb: kb.domain_id)

    # --- Ingestion Status ---

    async def set_ingestion_status(
        self,
        domain_id: str,
        status: IngestionStatus | str,
        error: str | None = None,
        *,
        generation: str | None = None,
    ) -> None:
        """Update ingestion status for a knowledge base."""
        if domain_id not in self._kb_info:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        kb_info = self._kb_info[domain_id]
        kb_info.ingestion_status = normalize_ingestion_status(status)
        kb_info.ingestion_error = error
        # Always written through: a non-SWAPPING transition passes the
        # default None and so clears any stale in-flight swap token.
        kb_info.generation = generation

        # In-process backend: no serialized metadata file is written,
        # but the state-write event fires for surface parity with the
        # file / S3 backends (so consumers compose one observability
        # path across every backend). ``byte_size`` reflects the
        # would-be serialized status payload.
        body = json.dumps(
            {
                "ingestion_status": kb_info.ingestion_status,
                "ingestion_error": kb_info.ingestion_error,
                "generation": kb_info.generation,
            },
            default=str,
        )
        await self._fire_state_write(
            domain_id=domain_id,
            key=f"{domain_id}/{self.METADATA_FILE}",
            kind=KnowledgeKeyKind.METADATA,
            byte_size=len(body.encode("utf-8")),
        )

    # --- Key Layout ---
    #
    # classify_key comes from KnowledgeResourceBackendMixin against the
    # canonical METADATA_FILE / CONTENT_DIR / SNAPSHOTS_DIR constants
    # (same as every in-tree backend). key_pattern returns the empty
    # string because no external observer can filter against in-process
    # storage; the method exists for protocol symmetry so consumer code
    # can call it uniformly across every backend without branching.

    def key_pattern(
        self,
        kind: KnowledgeKeyKind = KnowledgeKeyKind.CONTENT,
        domain_id: str | None = None,
    ) -> str:
        """No event-source filter is meaningful for in-process storage.

        Returns ``""`` (an explicitly empty sentinel) so contract tests
        can assert the method exists and is callable for every backend
        without conflating "method missing" with "method present, value
        meaningfully empty." The ``kind`` and ``domain_id`` arguments
        are accepted for protocol symmetry but ignored.
        """
        del kind, domain_id  # accepted for protocol symmetry, ignored
        return ""

    # --- Change Detection ---
    #
    # get_checksum / has_changes_since / list_changes_since come from
    # KnowledgeResourceBackendMixin (one canonical algorithm). Memory
    # additionally retains a per-version snapshot so it produces minimal
    # diffs rather than the mixin's full-set default.

    async def _record_snapshot(self, domain_id: str) -> None:
        """Snapshot the current file→checksum map under its version.

        Called after every mutation so a later
        ``list_changes_since(domain_id, that_version)`` can diff against
        the exact state. The version key is the canonical
        :meth:`get_checksum` value (computed once, here, by the mixin).
        """
        version = await self.get_checksum(domain_id)
        meta = self._file_metadata.get(domain_id, {})
        snapshot = {path: f.checksum for path, f in meta.items()}
        self._snapshots.setdefault(domain_id, {})[version] = snapshot
        # The empty-KB baseline (version "") writes no real snapshot in
        # the file / S3 backends, so skip the event there too for parity.
        if not version:
            return
        body = json.dumps(snapshot)
        await self._fire_state_write(
            domain_id=domain_id,
            key=f"{domain_id}/{self.SNAPSHOTS_DIR}/{version}.json",
            kind=KnowledgeKeyKind.SNAPSHOT,
            byte_size=len(body.encode("utf-8")),
        )

    async def _load_snapshot(
        self, domain_id: str, version: str
    ) -> dict[str, str]:
        """Return the retained ``{path: checksum}`` map for ``version``."""
        snaps = self._snapshots.get(domain_id, {})
        if version not in snaps:
            raise InvalidVersionError(
                f"Version {version!r} is not retained for domain "
                f"{domain_id!r}"
            )
        return snaps[version]
