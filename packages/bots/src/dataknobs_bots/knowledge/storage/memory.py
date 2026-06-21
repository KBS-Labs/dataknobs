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
from typing import TYPE_CHECKING, BinaryIO, ClassVar

from dataknobs_common.capabilities import Capability, CapabilityLike
from dataknobs_common.exceptions import ConcurrencyError

from .key_layout import KnowledgeKeyKind
from .mixin import KnowledgeResourceBackendMixin
from .models import (
    IngestionStatus,
    InvalidVersionError,
    KnowledgeBaseInfo,
    KnowledgeFile,
    normalize_ingestion_status,
)

if TYPE_CHECKING:
    from dataknobs_common.tenancy import TenantContext


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

    # Tenant-aware state methods honor ``ctx.state_key_prefix()``; the
    # in-process state dicts ARE tenant-scoped under the prefix, so the
    # capability surface matches the file / S3 backends (consumers can
    # test against memory and deploy against file/s3). Unions onto the
    # mixin's base set (does not replace it).
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[CapabilityLike]] = (
        KnowledgeResourceBackendMixin.SUPPORTED_CAPABILITIES
        | frozenset({
            Capability.TENANT_SCOPED_STATE,
            Capability.SNAPSHOT_ISOLATION,
            # Conditional state writes are trivially race-free in a
            # single-process backend: the version-counter check and
            # increment in set_ingestion_status run synchronously.
            Capability.TRANSACTIONAL_METADATA,
        })
    )

    def __init__(self) -> None:
        """Initialize the in-memory backend."""
        # domain_id -> KnowledgeBaseInfo. Domain-keyed: KB existence /
        # identity plus the single-tenant (ctx=None) ingest-state view.
        self._kb_info: dict[str, KnowledgeBaseInfo] = {}

        # state-key (prefix + domain_id) -> KnowledgeBaseInfo. Per-tenant
        # ingest-state overlay, created lazily on the first tenant-scoped
        # write. Only populated for a non-empty state prefix; the empty
        # prefix (ctx=None / SingleTenantContext) uses ``_kb_info``
        # directly so single-tenant behavior is byte-identical.
        self._tenant_info: dict[str, KnowledgeBaseInfo] = {}

        # domain_id -> path -> file content (bytes)
        self._files: dict[str, dict[str, bytes]] = {}

        # domain_id -> path -> KnowledgeFile
        self._file_metadata: dict[str, dict[str, KnowledgeFile]] = {}

        # state-key (prefix + domain_id) -> version (get_checksum) ->
        # {path: checksum}. In-process per-version store backing
        # _load_snapshot so memory produces minimal diffs. Tenant-scoped
        # under the state prefix; content mutations (ctx=None) record
        # under the bare domain_id, byte-identical to the pre-tenancy
        # store.
        self._snapshots: dict[str, dict[str, dict[str, str]]] = {}

        # state-key (prefix + domain_id) -> monotonic version counter.
        # The native optimistic-concurrency token for this backend:
        # get_state_version returns the current counter (as str) and
        # set_ingestion_status increments it on every write. Tenant-
        # scoped under the state prefix, mirroring _snapshots / the
        # _tenant_info overlay; the single-tenant (ctx=None) counter is
        # keyed by the bare domain_id and seeded at create_kb so the
        # single-tenant document has a token from creation onward (the
        # in-process analog of the file/S3 metadata document existing
        # after create_kb).
        self._state_versions: dict[str, int] = {}

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
        # Seed the single-tenant state-version counter: the metadata
        # document now exists, so get_state_version(ctx=None) returns a
        # token (the file/S3 analog: _metadata.json exists after
        # create_kb). Per-tenant counters stay unseeded (lazy on first
        # tenant write), mirroring the lazy per-tenant metadata document.
        self._state_versions[domain_id] = 0
        await self._record_snapshot(domain_id)  # baseline: "" -> {}

        return kb_info

    def _info_overlay_key(
        self, domain_id: str, ctx: TenantContext | None
    ) -> str | None:
        """Per-tenant overlay key, or ``None`` for the single-tenant case.

        Returns ``None`` when the context contributes no state prefix
        (``ctx=None`` or a ``SingleTenantContext``) so single-tenant
        reads/writes go straight to the domain-keyed ``_kb_info`` and
        stay byte-identical to the pre-tenancy behavior.
        """
        prefix = self._state_prefix(ctx)
        return f"{prefix}{domain_id}" if prefix else None

    async def get_info(
        self, domain_id: str, *, ctx: TenantContext | None = None
    ) -> KnowledgeBaseInfo | None:
        """Get knowledge base metadata.

        KB existence/identity is keyed by ``domain_id`` (guarded first).
        With ``ctx=None`` / a no-prefix context the shared domain-keyed
        view is returned (byte-identical to the pre-tenancy behavior).
        With a tenant prefix the per-tenant overlay is returned — or, when
        that tenant has not written ingest state yet, a *fresh default*
        view (``PENDING`` / no ``generation``), NOT the shared domain
        view. Returning the domain view here would leak the domain's
        ``ingestion_status`` and in-flight ``generation`` token (load-
        bearing for TOMBSTONE-swap reconciliation) to a tenant that never
        wrote, and would diverge from the file / S3 backends (whose
        per-tenant metadata-document miss yields the same fresh default).
        """
        base = self._kb_info.get(domain_id)
        if base is None:
            return None
        overlay_key = self._info_overlay_key(domain_id, ctx)
        if overlay_key is not None:
            overlay = self._tenant_info.get(overlay_key)
            if overlay is None:
                return KnowledgeBaseInfo.from_dict({"domain_id": domain_id})
            return overlay
        return base

    async def delete_kb(self, domain_id: str) -> bool:
        """Delete entire knowledge base and all files."""
        if domain_id not in self._kb_info:
            return False

        del self._kb_info[domain_id]
        if domain_id in self._files:
            del self._files[domain_id]
        if domain_id in self._file_metadata:
            del self._file_metadata[domain_id]
        # Drop the single-tenant snapshot store plus every per-tenant
        # state-keyed store/overlay for this domain (state keys are
        # ``{prefix}{domain_id}`` where the prefix ends in ``/``).
        suffix = f"/{domain_id}"

        def _is_domain_key(key: str) -> bool:
            return key == domain_id or key.endswith(suffix)

        for snap_key in [k for k in self._snapshots if _is_domain_key(k)]:
            self._snapshots.pop(snap_key, None)
        for info_key in [k for k in self._tenant_info if _is_domain_key(k)]:
            self._tenant_info.pop(info_key, None)
        for ver_key in [k for k in self._state_versions if _is_domain_key(k)]:
            self._state_versions.pop(ver_key, None)

        return True

    async def list_kbs(self) -> list[KnowledgeBaseInfo]:
        """List all knowledge bases."""
        return sorted(self._kb_info.values(), key=lambda kb: kb.domain_id)

    # --- Ingestion Status ---

    def _state_version_key(
        self, domain_id: str, ctx: TenantContext | None
    ) -> str:
        """State-version counter key (``{state-prefix}{domain_id}``).

        Matches the snapshot store's ``state_key`` shape: the bare
        ``domain_id`` for the single-tenant case (empty prefix) and
        ``{prefix}{domain_id}`` for a tenant-scoped call.
        """
        return f"{self._state_prefix(ctx)}{domain_id}"

    async def get_state_version(
        self, domain_id: str, *, ctx: TenantContext | None = None
    ) -> str | None:
        """Current monotonic version counter as an opaque ``str`` token.

        ``None`` when the KB does not exist or — for a tenant context —
        when that tenant has not written ingest state yet (its counter is
        lazy, mirroring the file / S3 per-tenant metadata document being
        absent until first write). The single-tenant counter is seeded at
        :meth:`create_kb`, so it returns ``"0"`` immediately after
        creation. See the protocol for the round-trip contract.
        """
        if domain_id not in self._kb_info:
            return None
        counter = self._state_versions.get(
            self._state_version_key(domain_id, ctx)
        )
        return str(counter) if counter is not None else None

    async def set_ingestion_status(
        self,
        domain_id: str,
        status: IngestionStatus | str,
        error: str | None = None,
        *,
        generation: str | None = None,
        ctx: TenantContext | None = None,
        expected_version: str | None = None,
    ) -> None:
        """Update ingestion status for a knowledge base.

        KB existence stays keyed by ``domain_id``. With ``ctx=None`` the
        status is written in place on the shared domain-keyed info
        (byte-identical to the pre-tenancy behavior); with a tenant
        context the status is written to a per-tenant overlay created
        lazily on first write — mirroring the file / S3 lazy per-tenant
        metadata document.

        When ``expected_version`` is supplied, the write is conditional:
        it proceeds only if the current state-version counter still
        matches the token, otherwise raises :class:`ConcurrencyError`
        without mutating state. The check-then-increment runs
        synchronously (single process, single loop), so the guard is
        race-free. Every write — conditional or not — advances the
        counter, so a stale token from before any write conflicts.
        """
        if domain_id not in self._kb_info:
            raise ValueError(f"Knowledge base '{domain_id}' does not exist")

        version_key = self._state_version_key(domain_id, ctx)
        current_counter = self._state_versions.get(version_key)
        current_token = (
            str(current_counter) if current_counter is not None else None
        )
        if expected_version is not None and expected_version != current_token:
            raise ConcurrencyError(
                "Knowledge-base state document was modified by a "
                "concurrent writer",
                context={
                    "domain_id": domain_id,
                    "expected_version": expected_version,
                    "actual_version": current_token,
                },
            )

        overlay_key = self._info_overlay_key(domain_id, ctx)
        if overlay_key is None:
            kb_info = self._kb_info[domain_id]
        else:
            kb_info = self._tenant_info.get(overlay_key)
            if kb_info is None:
                kb_info = KnowledgeBaseInfo.from_dict(
                    {"domain_id": domain_id}
                )
                self._tenant_info[overlay_key] = kb_info

        kb_info.ingestion_status = normalize_ingestion_status(status)
        kb_info.ingestion_error = error
        # Always written through: a non-SWAPPING transition passes the
        # default None and so clears any stale in-flight swap token.
        kb_info.generation = generation

        # Advance the version counter on every write (the token must
        # change so a concurrent writer holding the prior token
        # conflicts). A lazy per-tenant counter starts at None -> 1.
        self._state_versions[version_key] = (current_counter or 0) + 1

        # In-process backend: no serialized metadata file is written,
        # but the state-write event fires for surface parity with the
        # file / S3 backends (so consumers compose one observability
        # path across every backend). ``byte_size`` reflects the
        # would-be serialized status payload. The event key folds in the
        # tenant state prefix for parity with the file/S3 observed keys.
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
            key=f"{self._state_prefix(ctx)}{domain_id}/{self.METADATA_FILE}",
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

    async def _record_snapshot(
        self, domain_id: str, *, ctx: TenantContext | None = None
    ) -> None:
        """Snapshot the current file→checksum map under its version.

        Called after every mutation so a later
        ``list_changes_since(domain_id, that_version)`` can diff against
        the exact state. The version key is the canonical
        :meth:`get_checksum` value (computed once, here, by the mixin).

        The snapshot store is tenant-scoped under ``ctx`` — content
        mutations (``ctx=None``) record under the bare ``domain_id``,
        byte-identical to the pre-tenancy store; a tenant-bound call
        records under ``ctx.state_key_prefix()`` so per-tenant snapshot
        lineage cannot collide. The captured ``{path: checksum}`` map is
        the shared (domain-keyed) content state in either case.
        """
        version = await self.get_checksum(domain_id, ctx=ctx)
        state_key = f"{self._state_prefix(ctx)}{domain_id}"
        meta = self._file_metadata.get(domain_id, {})
        snapshot = {path: f.checksum for path, f in meta.items()}
        self._snapshots.setdefault(state_key, {})[version] = snapshot
        # The empty-KB baseline (version "") writes no real snapshot in
        # the file / S3 backends, so skip the event there too for parity.
        if not version:
            return
        body = json.dumps(snapshot)
        await self._fire_state_write(
            domain_id=domain_id,
            key=f"{state_key}/{self.SNAPSHOTS_DIR}/{version}.json",
            kind=KnowledgeKeyKind.SNAPSHOT,
            byte_size=len(body.encode("utf-8")),
        )

    async def _load_snapshot(
        self,
        domain_id: str,
        version: str,
        *,
        ctx: TenantContext | None = None,
    ) -> dict[str, str]:
        """Return the retained ``{path: checksum}`` map for ``version``.

        Resolved from the tenant-scoped snapshot store (``ctx``); a
        version recorded under a different tenant (or single-tenant) is
        not visible here.
        """
        state_key = f"{self._state_prefix(ctx)}{domain_id}"
        snaps = self._snapshots.get(state_key, {})
        if version not in snaps:
            raise InvalidVersionError(
                f"Version {version!r} is not retained for domain "
                f"{domain_id!r}"
            )
        return snaps[version]
