"""File-based knowledge resource backend for local development.

This backend stores files on the local filesystem, making it ideal for
local development and single-instance deployments.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
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

logger = logging.getLogger(__name__)


class FileKnowledgeBackend(KnowledgeResourceBackendMixin):
    """File system-based knowledge resource storage.

    Structure on disk:
        {base_path}/
            {domain_id}/
                content/              # consumer-controlled (put_file)
                    file1.md
                    subdir/
                        file2.json
                _metadata.json        # DK-managed state
                _snapshots/           # DK-managed state
                    <version>.json

    Suitable for:
    - Local development
    - Testing
    - Single-instance deployments

    Event triggers (filesystem inotify wrappers, polling watchers) MUST
    filter to the ``content/`` subtree to avoid retriggering on the
    DK-managed ``_metadata.json`` and ``_snapshots/`` writes the manager
    emits during ingest. Call :meth:`key_pattern` to derive a glob
    suitable for ``pathlib.Path.glob`` or an inotify wrapper; see the
    "Event triggers for knowledge backends" docs page for per-source
    recipes. Use :meth:`classify_key` for per-event filtering when the
    watcher cannot pattern-match.

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
        await asyncio.to_thread(
            self._base_path.mkdir, parents=True, exist_ok=True
        )
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

    def _snapshots_path(self, domain_id: str) -> Path:
        """Get the path to a KB's per-version snapshot directory."""
        return self._kb_path(domain_id) / self.SNAPSHOTS_DIR

    def _snapshot_file(self, domain_id: str, version: str) -> Path:
        """Path of the snapshot JSON for ``version`` (an MD5 hex id)."""
        return self._snapshots_path(domain_id) / f"{version}.json"

    def _file_path(self, domain_id: str, path: str) -> Path:
        """Get the full path to a file."""
        return self._content_path(domain_id) / path

    def key_pattern(
        self,
        kind: KnowledgeKeyKind = KnowledgeKeyKind.CONTENT,
        domain_id: str | None = None,
    ) -> str:
        """Filesystem glob pattern matching keys of the given kind.

        Suitable for ``pathlib.Path.glob`` and most inotify wrappers.
        See the "Event triggers for knowledge backends" docs page for
        per-source recipes.

        :attr:`KnowledgeKeyKind.UNKNOWN` raises :class:`ValueError`
        (fails closed — there is no shape for "unrecognized keys").
        """
        domain_segment = domain_id if domain_id else "*"
        base = str(self._base_path)
        if kind is KnowledgeKeyKind.CONTENT:
            return f"{base}/{domain_segment}/{self.CONTENT_DIR}/**"
        if kind is KnowledgeKeyKind.METADATA:
            return f"{base}/{domain_segment}/{self.METADATA_FILE}"
        if kind is KnowledgeKeyKind.SNAPSHOT:
            return f"{base}/{domain_segment}/{self.SNAPSHOTS_DIR}/*"
        raise ValueError(
            f"key_pattern is not defined for kind {kind!r} "
            f"(only CONTENT / METADATA / SNAPSHOT)"
        )

    def _load_metadata_sync(self, domain_id: str) -> dict:
        """Load metadata from disk (blocking; call via :meth:`_load_metadata`)."""
        meta_path = self._metadata_path(domain_id)
        if not meta_path.exists():
            return {}
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    async def _load_metadata(self, domain_id: str) -> dict:
        """Load metadata from disk, off the event loop."""
        return await asyncio.to_thread(self._load_metadata_sync, domain_id)

    @staticmethod
    def _atomic_write_text(target: Path, body: str) -> None:
        """Write ``body`` to ``target`` atomically (temp file + rename).

        Blocking — call via ``asyncio.to_thread`` from async methods.
        """
        fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
            os.replace(tmp_path, target)
        except Exception:
            tmp = Path(tmp_path)
            if tmp.exists():
                tmp.unlink()
            raise

    @staticmethod
    def _write_content_file(target: Path, data: bytes) -> None:
        """Create ``target``'s parent and write ``data`` atomically.

        Blocking — call via ``asyncio.to_thread`` from async methods.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, target)
        except Exception:
            tmp = Path(tmp_path)
            if tmp.exists():
                tmp.unlink()
            raise

    async def _save_metadata(self, domain_id: str, metadata: dict) -> None:
        """Save metadata to disk atomically, then fire a state-write event."""
        meta_path = self._metadata_path(domain_id)
        body = json.dumps(metadata, indent=2, default=str)

        await asyncio.to_thread(self._atomic_write_text, meta_path, body)

        await self._fire_state_write(
            domain_id=domain_id,
            key=str(meta_path),
            kind=KnowledgeKeyKind.METADATA,
            byte_size=len(body.encode("utf-8")),
        )

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
        if not await asyncio.to_thread(self._kb_path(domain_id).exists):
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

        # Ensure parent directory exists, then write file atomically —
        # both off the event loop.
        file_path = self._file_path(domain_id, path)
        await asyncio.to_thread(self._write_content_file, file_path, data)

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
        kb_metadata = await self._load_metadata(domain_id)
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

        await self._save_metadata(domain_id, kb_metadata)
        await self._record_snapshot(domain_id, files)

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
        return await asyncio.to_thread(self._read_bytes_or_none, file_path)

    @staticmethod
    def _read_bytes_or_none(file_path: Path) -> bytes | None:
        """Read ``file_path`` or return ``None`` if it does not exist.

        Blocking — call via ``asyncio.to_thread`` from async methods.
        """
        if not file_path.exists():
            return None
        return file_path.read_bytes()

    async def stream_file(
        self, domain_id: str, path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes] | None:
        """Stream file content, reading each chunk off the event loop."""
        file_path = self._file_path(domain_id, path)
        if not await asyncio.to_thread(file_path.exists):
            return None

        async def _generator() -> AsyncIterator[bytes]:
            handle = await asyncio.to_thread(open, file_path, "rb")
            try:
                while True:
                    chunk = await asyncio.to_thread(handle.read, chunk_size)
                    if not chunk:
                        break
                    yield chunk
            finally:
                await asyncio.to_thread(handle.close)

        return _generator()

    async def delete_file(self, domain_id: str, path: str) -> bool:
        """Delete a file."""
        file_path = self._file_path(domain_id, path)
        if not await asyncio.to_thread(self._unlink_if_exists, file_path):
            return False

        # Update metadata
        kb_metadata = await self._load_metadata(domain_id)
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

        await self._save_metadata(domain_id, kb_metadata)
        await self._record_snapshot(domain_id, files)

        # Clean up empty parent directories
        await asyncio.to_thread(
            self._cleanup_empty_dirs,
            file_path.parent,
            self._content_path(domain_id),
        )

        logger.debug("Deleted file: %s/%s", domain_id, path)
        return True

    @staticmethod
    def _unlink_if_exists(file_path: Path) -> bool:
        """Unlink ``file_path`` if present; return whether it existed.

        Blocking — call via ``asyncio.to_thread`` from async methods.
        """
        if not file_path.exists():
            return False
        file_path.unlink()
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
        kb_metadata = await self._load_metadata(domain_id)
        files_dict = kb_metadata.get("files", {})

        files = [KnowledgeFile.from_dict(f) for f in files_dict.values()]

        if prefix:
            files = [f for f in files if f.path.startswith(prefix)]

        # Sort by path for consistent ordering
        files.sort(key=lambda f: f.path)
        return files

    async def file_exists(self, domain_id: str, path: str) -> bool:
        """Check if a file exists."""
        return await asyncio.to_thread(self._file_path(domain_id, path).exists)

    # --- Knowledge Base Operations ---

    async def create_kb(
        self, domain_id: str, metadata: dict | None = None
    ) -> KnowledgeBaseInfo:
        """Create a new knowledge base."""
        if not await asyncio.to_thread(self._create_kb_dirs, domain_id):
            raise ValueError(f"Knowledge base '{domain_id}' already exists")

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
        await self._save_metadata(domain_id, kb_metadata)

        logger.info("Created knowledge base: %s", domain_id)
        return kb_info

    def _create_kb_dirs(self, domain_id: str) -> bool:
        """Create the KB + content directories; return whether created.

        Returns ``False`` if the KB already exists (no directories made).
        Blocking — call via ``asyncio.to_thread`` from async methods.
        """
        kb_path = self._kb_path(domain_id)
        if kb_path.exists():
            return False
        kb_path.mkdir(parents=True)
        self._content_path(domain_id).mkdir()
        return True

    async def get_info(self, domain_id: str) -> KnowledgeBaseInfo | None:
        """Get knowledge base metadata."""
        if not await asyncio.to_thread(self._kb_path(domain_id).exists):
            return None

        kb_metadata = await self._load_metadata(domain_id)
        info_dict = kb_metadata.get("info", {"domain_id": domain_id})
        return KnowledgeBaseInfo.from_dict(info_dict)

    async def delete_kb(self, domain_id: str) -> bool:
        """Delete entire knowledge base and all files."""
        deleted = await asyncio.to_thread(self._rmtree_if_exists, domain_id)
        if deleted:
            logger.info("Deleted knowledge base: %s", domain_id)
        return deleted

    def _rmtree_if_exists(self, domain_id: str) -> bool:
        """Remove the KB tree if present; return whether it existed.

        Blocking — call via ``asyncio.to_thread`` from async methods.
        """
        kb_path = self._kb_path(domain_id)
        if not kb_path.exists():
            return False
        shutil.rmtree(kb_path)
        return True

    async def list_kbs(self) -> list[KnowledgeBaseInfo]:
        """List all knowledge bases."""
        domain_ids = await asyncio.to_thread(self._list_kb_domain_ids)
        kbs = []
        for domain_id in domain_ids:
            info = await self.get_info(domain_id)
            if info:
                kbs.append(info)

        return sorted(kbs, key=lambda kb: kb.domain_id)

    def _list_kb_domain_ids(self) -> list[str]:
        """Scan the base directory for KB domain ids (those with metadata).

        Blocking — call via ``asyncio.to_thread`` from async methods.
        """
        if not self._base_path.exists():
            return []
        return [
            path.name
            for path in self._base_path.iterdir()
            if path.is_dir() and (path / self.METADATA_FILE).exists()
        ]

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
        if not await asyncio.to_thread(self._kb_path(domain_id).exists):
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        kb_metadata = await self._load_metadata(domain_id)
        info_dict = kb_metadata.get("info", {"domain_id": domain_id})
        # Persist the string value so the JSON round-trips and
        # KnowledgeBaseInfo.from_dict rehydrates the enum.
        info_dict["ingestion_status"] = normalize_ingestion_status(
            status
        ).value
        info_dict["ingestion_error"] = error
        # Always written through: a non-SWAPPING transition passes the
        # default None and so clears any stale in-flight swap token.
        info_dict["generation"] = generation
        kb_metadata["info"] = info_dict

        await self._save_metadata(domain_id, kb_metadata)

    # --- Change Detection ---
    #
    # get_checksum / has_changes_since / list_changes_since come from
    # KnowledgeResourceBackendMixin (one canonical algorithm over
    # list_files()). This backend overrides _load_snapshot with a real
    # per-version store so list_changes_since produces a minimal
    # file-level diff rather than the mixin's full-set default. The
    # snapshot for a version is the {path: checksum} map captured at the
    # moment the KB had that canonical identity; it is written after
    # every mutation, mirroring InMemoryKnowledgeBackend.

    async def _record_snapshot(
        self, domain_id: str, files: dict[str, dict]
    ) -> None:
        """Persist the post-mutation ``{path: checksum}`` map.

        Keyed by the canonical :meth:`get_checksum` identity (computed
        from the same map via the shared mixin formula, so a later
        ``list_changes_since(domain_id, that_version)`` can diff against
        the exact state). The empty KB has identity ``""`` and needs no
        file — :meth:`_load_snapshot` resolves ``""`` to the empty
        snapshot directly.

        Called after :meth:`put_file` / :meth:`delete_file` write
        metadata. ``files`` is the KB's file index (the value already in
        hand at the call site) — ``{path: KnowledgeFile.to_dict()}``.
        """
        snapshot = {
            path: info.get("checksum", "") for path, info in files.items()
        }
        version = self._identity_of_snapshot(snapshot)
        if not version:
            return
        snap_path = self._snapshot_file(domain_id, version)
        body = json.dumps(snapshot)
        wrote = await asyncio.to_thread(
            self._write_snapshot_file, snap_path, body
        )
        if not wrote:
            return  # identical content state already captured

        await self._fire_state_write(
            domain_id=domain_id,
            key=str(snap_path),
            kind=KnowledgeKeyKind.SNAPSHOT,
            byte_size=len(body.encode("utf-8")),
        )

    def _write_snapshot_file(self, snap_path: Path, body: str) -> bool:
        """Create the snapshot dir and write ``body`` atomically.

        Returns ``False`` (no write) if the snapshot already exists.
        Blocking — call via ``asyncio.to_thread`` from async methods.
        """
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        if snap_path.exists():
            return False
        self._atomic_write_text(snap_path, body)
        return True

    async def _load_snapshot(
        self, domain_id: str, version: str
    ) -> dict[str, str]:
        """Resolve ``version`` to its retained ``{path: checksum}`` map.

        ``""`` is the empty-KB baseline (no file written for it — every
        current file diffs as ``added``). Any other version with no
        retained snapshot predates retention / is unknown ⇒
        :class:`InvalidVersionError` (callers fall back to a full
        re-ingest).
        """
        if not version:
            return {}
        snap_path = self._snapshot_file(domain_id, version)
        data = await asyncio.to_thread(self._read_snapshot_file, snap_path)
        if data is None:
            raise InvalidVersionError(
                f"Version {version!r} is not retained for domain "
                f"{domain_id!r}"
            )
        return data

    @staticmethod
    def _read_snapshot_file(snap_path: Path) -> dict[str, str] | None:
        """Read a snapshot ``{path: checksum}`` map, or ``None`` if absent.

        Blocking — call via ``asyncio.to_thread`` from async methods.
        """
        if not snap_path.exists():
            return None
        with open(snap_path, encoding="utf-8") as f:
            data: dict[str, str] = json.load(f)
        return data
