"""File-based knowledge resource backend for local development.

This backend stores files on the local filesystem, making it ideal for
local development and single-instance deployments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from .models import IngestionStatus, KnowledgeBaseInfo, KnowledgeFile

logger = logging.getLogger(__name__)


class FileKnowledgeBackend:
    """File system-based knowledge resource storage.

    Structure on disk:
        {base_path}/
            {domain_id}/
                _metadata.json       # KB info and file index
                content/
                    file1.md
                    subdir/
                        file2.json

    Suitable for:
    - Local development
    - Testing
    - Single-instance deployments

    Example:
        ```python
        backend = FileKnowledgeBackend(base_path="./data/knowledge")
        await backend.initialize()

        await backend.create_kb("my-domain")
        await backend.put_file("my-domain", "intro.md", b"# Hello")

        content = await backend.get_file("my-domain", "intro.md")
        print(content.decode())

        await backend.close()
        ```
    """

    METADATA_FILE = "_metadata.json"
    CONTENT_DIR = "content"

    def __init__(self, base_path: str | Path) -> None:
        """Initialize the file backend.

        Args:
            base_path: Root directory for all knowledge bases
        """
        self._base_path = Path(base_path)
        self._initialized = False

    @classmethod
    def from_config(cls, config: dict) -> FileKnowledgeBackend:
        """Create from configuration dict.

        Args:
            config: Configuration with "path" key

        Returns:
            New FileKnowledgeBackend instance
        """
        return cls(base_path=config.get("path", "./data/knowledge"))

    async def initialize(self) -> None:
        """Initialize the backend. Creates base directory if needed."""
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    async def close(self) -> None:
        """Close the backend. No-op for file backend."""
        self._initialized = False

    def _kb_path(self, domain_id: str) -> Path:
        """Get the path to a knowledge base directory."""
        return self._base_path / domain_id

    def _metadata_path(self, domain_id: str) -> Path:
        """Get the path to a KB's metadata file."""
        return self._kb_path(domain_id) / self.METADATA_FILE

    def _content_path(self, domain_id: str) -> Path:
        """Get the path to a KB's content directory."""
        return self._kb_path(domain_id) / self.CONTENT_DIR

    def _file_path(self, domain_id: str, path: str) -> Path:
        """Get the full path to a file."""
        return self._content_path(domain_id) / path

    def _load_metadata(self, domain_id: str) -> dict:
        """Load metadata from disk."""
        meta_path = self._metadata_path(domain_id)
        if not meta_path.exists():
            return {}
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def _save_metadata(self, domain_id: str, metadata: dict) -> None:
        """Save metadata to disk atomically."""
        meta_path = self._metadata_path(domain_id)

        # Write to temp file first, then rename for atomicity
        fd, tmp_path = tempfile.mkstemp(dir=meta_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
            os.replace(tmp_path, meta_path)
        except Exception:
            # Clean up temp file on error
            tmp = Path(tmp_path)
            if tmp.exists():
                tmp.unlink()
            raise

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
        kb_path = self._kb_path(domain_id)
        if not kb_path.exists():
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

        # Ensure parent directory exists
        file_path = self._file_path(domain_id, path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file atomically
        fd, tmp_path = tempfile.mkstemp(dir=file_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, file_path)
        except Exception:
            tmp = Path(tmp_path)
            if tmp.exists():
                tmp.unlink()
            raise

        # Create file metadata
        file_info = KnowledgeFile(
            path=path,
            content_type=content_type,
            size_bytes=len(data),
            checksum=checksum,
            uploaded_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        # Update KB metadata
        kb_metadata = self._load_metadata(domain_id)
        files = kb_metadata.get("files", {})
        is_new = path not in files
        files[path] = file_info.to_dict()
        kb_metadata["files"] = files

        # Update KB info
        kb_info = KnowledgeBaseInfo.from_dict(kb_metadata.get("info", {"domain_id": domain_id}))
        kb_info.last_updated = datetime.now(timezone.utc)
        kb_info.increment_version()
        kb_info.file_count = len(files)
        kb_info.total_size_bytes = sum(f.get("size_bytes", 0) for f in files.values())
        kb_metadata["info"] = kb_info.to_dict()

        self._save_metadata(domain_id, kb_metadata)

        logger.debug(
            "%s file: %s/%s (%d bytes)",
            "Created" if is_new else "Updated",
            domain_id,
            path,
            len(data),
        )

        return file_info

    async def get_file(self, domain_id: str, path: str) -> bytes | None:
        """Get file content."""
        file_path = self._file_path(domain_id, path)
        if not file_path.exists():
            return None
        return file_path.read_bytes()

    async def stream_file(
        self, domain_id: str, path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes] | None:
        """Stream file content."""
        file_path = self._file_path(domain_id, path)
        if not file_path.exists():
            return None

        async def _generator() -> AsyncIterator[bytes]:
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        return _generator()

    async def delete_file(self, domain_id: str, path: str) -> bool:
        """Delete a file."""
        file_path = self._file_path(domain_id, path)
        if not file_path.exists():
            return False

        # Remove file
        file_path.unlink()

        # Update metadata
        kb_metadata = self._load_metadata(domain_id)
        files = kb_metadata.get("files", {})
        if path in files:
            del files[path]
        kb_metadata["files"] = files

        # Update KB info
        kb_info = KnowledgeBaseInfo.from_dict(kb_metadata.get("info", {"domain_id": domain_id}))
        kb_info.last_updated = datetime.now(timezone.utc)
        kb_info.increment_version()
        kb_info.file_count = len(files)
        kb_info.total_size_bytes = sum(f.get("size_bytes", 0) for f in files.values())
        kb_metadata["info"] = kb_info.to_dict()

        self._save_metadata(domain_id, kb_metadata)

        # Clean up empty parent directories
        self._cleanup_empty_dirs(file_path.parent, self._content_path(domain_id))

        logger.debug("Deleted file: %s/%s", domain_id, path)
        return True

    def _cleanup_empty_dirs(self, start: Path, stop: Path) -> None:
        """Remove empty directories from start up to (but not including) stop."""
        current = start
        while current != stop and current.exists():
            if not any(current.iterdir()):
                current.rmdir()
                current = current.parent
            else:
                break

    async def list_files(
        self, domain_id: str, prefix: str | None = None
    ) -> list[KnowledgeFile]:
        """List all files in a knowledge base."""
        kb_metadata = self._load_metadata(domain_id)
        files_dict = kb_metadata.get("files", {})

        files = [KnowledgeFile.from_dict(f) for f in files_dict.values()]

        if prefix:
            files = [f for f in files if f.path.startswith(prefix)]

        # Sort by path for consistent ordering
        files.sort(key=lambda f: f.path)
        return files

    async def file_exists(self, domain_id: str, path: str) -> bool:
        """Check if a file exists."""
        return self._file_path(domain_id, path).exists()

    # --- Knowledge Base Operations ---

    async def create_kb(
        self, domain_id: str, metadata: dict | None = None
    ) -> KnowledgeBaseInfo:
        """Create a new knowledge base."""
        kb_path = self._kb_path(domain_id)
        if kb_path.exists():
            raise ValueError(f"Knowledge base '{domain_id}' already exists")

        # Create directories
        kb_path.mkdir(parents=True)
        self._content_path(domain_id).mkdir()

        # Create initial metadata
        kb_info = KnowledgeBaseInfo(
            domain_id=domain_id,
            file_count=0,
            total_size_bytes=0,
            last_updated=datetime.now(timezone.utc),
            version="1",
            ingestion_status=IngestionStatus.PENDING,
            metadata=metadata or {},
        )

        kb_metadata = {
            "info": kb_info.to_dict(),
            "files": {},
        }
        self._save_metadata(domain_id, kb_metadata)

        logger.info("Created knowledge base: %s", domain_id)
        return kb_info

    async def get_info(self, domain_id: str) -> KnowledgeBaseInfo | None:
        """Get knowledge base metadata."""
        kb_path = self._kb_path(domain_id)
        if not kb_path.exists():
            return None

        kb_metadata = self._load_metadata(domain_id)
        info_dict = kb_metadata.get("info", {"domain_id": domain_id})
        return KnowledgeBaseInfo.from_dict(info_dict)

    async def delete_kb(self, domain_id: str) -> bool:
        """Delete entire knowledge base and all files."""
        kb_path = self._kb_path(domain_id)
        if not kb_path.exists():
            return False

        shutil.rmtree(kb_path)
        logger.info("Deleted knowledge base: %s", domain_id)
        return True

    async def list_kbs(self) -> list[KnowledgeBaseInfo]:
        """List all knowledge bases."""
        kbs = []
        if not self._base_path.exists():
            return kbs

        for path in self._base_path.iterdir():
            if path.is_dir() and (path / self.METADATA_FILE).exists():
                info = await self.get_info(path.name)
                if info:
                    kbs.append(info)

        return sorted(kbs, key=lambda kb: kb.domain_id)

    # --- Ingestion Status ---

    async def set_ingestion_status(
        self,
        domain_id: str,
        status: str,
        error: str | None = None,
    ) -> None:
        """Update ingestion status for a knowledge base."""
        kb_path = self._kb_path(domain_id)
        if not kb_path.exists():
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        kb_metadata = self._load_metadata(domain_id)
        info_dict = kb_metadata.get("info", {"domain_id": domain_id})
        info_dict["ingestion_status"] = status
        info_dict["ingestion_error"] = error
        kb_metadata["info"] = info_dict

        self._save_metadata(domain_id, kb_metadata)

    # --- Change Detection ---

    async def get_checksum(self, domain_id: str) -> str:
        """Get combined checksum of all files."""
        kb_path = self._kb_path(domain_id)
        if not kb_path.exists():
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        kb_metadata = self._load_metadata(domain_id)
        files = kb_metadata.get("files", {})

        if not files:
            return ""

        # Combine all file checksums sorted by path
        checksums = sorted(f"{path}:{info['checksum']}" for path, info in files.items())
        combined = ":".join(checksums)
        return hashlib.md5(combined.encode()).hexdigest()

    async def has_changes_since(self, domain_id: str, version: str) -> bool:
        """Check if KB has changed since given version."""
        info = await self.get_info(domain_id)
        if info is None:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")
        return info.version != version
