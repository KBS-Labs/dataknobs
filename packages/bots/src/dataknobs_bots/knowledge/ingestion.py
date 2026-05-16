"""Knowledge ingestion manager for coordinating file storage to vector storage.

This module provides the KnowledgeIngestionManager which coordinates loading
files from a KnowledgeResourceBackend into a RAGKnowledgeBase.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataknobs_common.events import EventBus
    from dataknobs_xization.ingestion import KnowledgeBaseConfig

    from .rag import RAGKnowledgeBase
    from .storage import KnowledgeFile, KnowledgeResourceBackend

logger = logging.getLogger(__name__)


class IngestSwapMode(Enum):
    """How a re-ingest replaces the destination's existing chunks.

    The swap policy is a property of the shared apply-core
    (:meth:`KnowledgeIngestionManager._apply_file_ops`), *not* of the
    entry point — full-domain (:meth:`~KnowledgeIngestionManager.ingest`)
    and per-file (:meth:`~KnowledgeIngestionManager.ingest_changes`)
    re-ingests honor the same modes through the same code path.

    Members ship incrementally with the same value space:

    - ``CLEAR_FIRST`` — delete the (scoped) existing chunks, then
      ingest. The legacy ``clear_existing=True`` behavior; carries a
      temporal zero-results window for concurrent reads.
    - ``APPEND`` — ingest without a preceding full-domain clear (the
      legacy ``clear_existing=False`` behavior). Per-file deletions
      requested explicitly still happen.

    ``TOMBSTONE`` (zero-downtime swap) and ``SHADOW_GENERATION`` are
    introduced in a later phase; this enum is the stable type the
    apply-core is written against so those modes are additive rather
    than a signature change.
    """

    CLEAR_FIRST = "clear_first"
    APPEND = "append"


@dataclass
class IngestionResult:
    """Result of an ingestion operation.

    Contains statistics about files processed, chunks created, and any errors
    encountered during ingestion. ``files_deleted`` is the count of
    source files whose chunks were removed because the file no longer
    exists at the source (populated by
    :meth:`KnowledgeIngestionManager.ingest_changes`; ``0`` for a
    full :meth:`~KnowledgeIngestionManager.ingest`).

    Invariants:
        ``completed_at`` is populated on every terminal state — call
        :meth:`finish` at every return site (or in a ``finally`` block)
        to enforce this. ``IngestionResult`` is the lower-level sibling
        of :class:`EnsureIngestionResult` and shares this invariant.
    """

    domain_id: str
    files_processed: int = 0
    chunks_created: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def finish(self) -> IngestionResult:
        """Stamp ``completed_at`` on the first call; idempotent thereafter.

        Returns ``self`` so it can be chained at return sites or used
        from a ``finally`` block.
        """
        if self.completed_at is None:
            self.completed_at = datetime.now(timezone.utc)
        return self

    @property
    def success(self) -> bool:
        """Check if ingestion completed without errors."""
        return len(self.errors) == 0

    @property
    def duration_seconds(self) -> float | None:
        """Get duration in seconds if completed."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "domain_id": self.domain_id,
            "files_processed": self.files_processed,
            "chunks_created": self.chunks_created,
            "files_skipped": self.files_skipped,
            "files_deleted": self.files_deleted,
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
        }


class KnowledgeIngestionManager:
    """Coordinates loading files from storage backend into RAG knowledge base.

    This manager bridges :class:`KnowledgeResourceBackend` (file storage)
    and :class:`RAGKnowledgeBase` (vector storage). It delegates the
    actual document processing (pattern matching, chunking, streaming
    JSON, exclude patterns, per-pattern metadata) to
    :meth:`RAGKnowledgeBase.ingest_from_backend`, and owns:

    - Ingestion status tracking on the source backend
    - Event-bus publishing for hot-reload consumers
    - Version-based skip (`ingest_if_changed`)

    When ``config`` is omitted, :meth:`ingest_from_backend` reads any
    ``_metadata/knowledge_base.(yaml|yml|json)`` in the backend's domain
    namespace and falls back to defaults. This is a strict superset of
    the pre-unification behavior, which silently ignored patterns and
    excludes.

    Example:
        ```python
        from dataknobs_bots.knowledge import (
            KnowledgeIngestionManager,
            create_knowledge_backend,
            RAGKnowledgeBase,
        )

        # Create components
        file_backend = create_knowledge_backend("file", {"path": "./data/kb"})
        rag_kb = await RAGKnowledgeBase.from_config(rag_config)

        # Create manager
        manager = KnowledgeIngestionManager(
            source=file_backend,
            destination=rag_kb,
        )

        # Ingest all files for a domain
        result = await manager.ingest("my-domain")
        print(f"Ingested {result.chunks_created} chunks from {result.files_processed} files")
        ```
    """

    def __init__(
        self,
        source: KnowledgeResourceBackend,
        destination: RAGKnowledgeBase,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize the ingestion manager.

        Args:
            source: Backend storing knowledge files
            destination: RAG knowledge base for vector storage
            event_bus: Optional event bus for publishing ingestion events
        """
        self._source = source
        self._destination = destination
        self._event_bus = event_bus

    async def ingest(
        self,
        domain_id: str,
        clear_existing: bool = True,
        progress_callback: Callable[[str, int], None] | None = None,
        config: KnowledgeBaseConfig | None = None,
    ) -> IngestionResult:
        """Ingest all documents from a domain into the knowledge base.

        Delegates to :meth:`RAGKnowledgeBase.ingest_from_backend` for
        the actual processing — pattern-based chunking, exclude
        patterns, per-pattern metadata, streaming JSON — and wraps the
        result in an :class:`IngestionResult` for backward compatibility.

        Args:
            domain_id: Domain to ingest
            clear_existing: Clear existing vectors before ingesting
                (default: ``True``)
            progress_callback: Optional callback invoked as
                ``(file_path, num_chunks)`` after each ingested document
            config: Optional :class:`KnowledgeBaseConfig` overriding any
                backend-hosted ``_metadata/knowledge_base.(yaml|yml|json)``

        Returns:
            :class:`IngestionResult` with aggregate statistics
        """
        result = IngestionResult(domain_id=domain_id)

        try:
            await self._source.set_ingestion_status(domain_id, "ingesting")
            logger.info("Starting ingestion for domain: %s", domain_id)

            await self._apply_file_ops(
                domain_id,
                upsert_filter=None,
                delete_paths=None,
                swap_mode=(
                    IngestSwapMode.CLEAR_FIRST
                    if clear_existing
                    else IngestSwapMode.APPEND
                ),
                config=config,
                progress_callback=progress_callback,
                result=result,
            )
            await self._finalize(domain_id, result)

        except Exception as e:
            logger.error("Ingestion failed for %s: %s", domain_id, e)
            await self._source.set_ingestion_status(domain_id, "error", str(e))
            raise
        finally:
            result.finish()

        return result

    async def _apply_file_ops(
        self,
        domain_id: str,
        *,
        upsert_filter: Callable[[KnowledgeFile], bool] | None,
        delete_paths: list[str] | None,
        swap_mode: IngestSwapMode,
        config: KnowledgeBaseConfig | None,
        progress_callback: Callable[[str, int], None] | None,
        result: IngestionResult,
    ) -> None:
        """The single place a re-ingest mutates the destination.

        Used by **both** :meth:`ingest` (full domain) and
        :meth:`ingest_changes` (per-file delta) so swap semantics never
        diverge between the two entry points.

        ``swap_mode`` governs the *full-domain* clear:

        - ``CLEAR_FIRST`` — scoped
          ``clear(filter={"domain_id": ...})`` before the upsert (the
          legacy ``clear_existing=True`` behavior). The filter scopes
          the wipe to this domain so a shared multi-tenant store keeps
          every other tenant's chunks.
        - ``APPEND`` — no full-domain clear.

        ``delete_paths`` (when non-empty) additionally purges every
        chunk whose ``source_path`` matches — used by
        :meth:`ingest_changes` to drop chunks for *deleted and
        modified* files before re-embedding. ``source_path`` is the
        backend-stable source-relative path
        (``DocumentFileRef.path``); ``source`` is a display path and
        is not used as the delete key.

        The upsert re-embeds the source files selected by
        ``upsert_filter`` (``None`` = every file) via
        :meth:`RAGKnowledgeBase.ingest_from_backend`. ``result`` is
        populated in place from the returned statistics;
        ``files_deleted`` is owned by the caller, which knows the
        delete semantics for its entry point.
        """
        if swap_mode is IngestSwapMode.CLEAR_FIRST:
            await self._destination.clear(filter={"domain_id": domain_id})
            logger.debug(
                "Cleared existing vectors for domain: %s", domain_id
            )

        for path in delete_paths or ():
            await self._destination.clear(
                filter={"domain_id": domain_id, "source_path": path}
            )
        if delete_paths:
            logger.debug(
                "Purged chunks for %d source path(s) in domain: %s",
                len(delete_paths),
                domain_id,
            )

        stats = await self._destination.ingest_from_backend(
            self._source,
            domain_id,
            config=config,
            progress_callback=progress_callback,
            extra_metadata={"domain_id": domain_id},
            file_filter=upsert_filter,
        )

        result.files_processed = int(stats.get("total_files", 0))
        result.chunks_created = int(stats.get("total_chunks", 0))
        result.files_skipped = int(stats.get("files_skipped", 0))
        result.errors = list(stats.get("errors", []))

    async def _finalize(
        self, domain_id: str, result: IngestionResult
    ) -> str:
        """Set the terminal ingestion status, log, and publish.

        Shared tail of :meth:`ingest` and :meth:`ingest_changes` — the
        success/error decision, backend status write, completion log,
        and event publish are one behavior, not duplicated per entry
        point. Returns the resolved status string.
        """
        status = "ready" if result.success else "error"
        error_msg = str(result.errors) if result.errors else None
        await self._source.set_ingestion_status(
            domain_id, status, error_msg
        )

        logger.info(
            "Ingestion completed for %s: %d files, %d chunks, "
            "%d deleted, %d errors",
            domain_id,
            result.files_processed,
            result.chunks_created,
            result.files_deleted,
            len(result.errors),
        )

        if self._event_bus:
            await self._publish_ingestion_event(domain_id, result, status)
        return status

    async def _publish_ingestion_event(
        self,
        domain_id: str,
        result: IngestionResult,
        status: str,
    ) -> None:
        """Publish an ingestion completed event."""
        if not self._event_bus:
            return

        from dataknobs_common.events import Event, EventType

        await self._event_bus.publish(
            "knowledge:ingestion",
            Event(
                type=EventType.UPDATED,
                topic="knowledge:ingestion",
                payload={
                    "domain_id": domain_id,
                    "files_processed": result.files_processed,
                    "chunks_created": result.chunks_created,
                    "files_deleted": result.files_deleted,
                    "status": status,
                },
                timestamp=result.completed_at or datetime.now(timezone.utc),
            ),
        )

    async def ingest_if_changed(
        self,
        domain_id: str,
        last_version: str | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
        config: KnowledgeBaseConfig | None = None,
    ) -> IngestionResult | None:
        """Ingest only if the knowledge base has changed.

        Useful for hot-reload scenarios where re-ingestion should be
        skipped if nothing has changed.

        Args:
            domain_id: Domain to check and potentially ingest
            last_version: Last known version string; if ``None``, always
                ingests
            progress_callback: Optional callback ``(file_path, chunks)``
                passed through to :meth:`ingest`
            config: Optional :class:`KnowledgeBaseConfig` passed through
                to :meth:`ingest`

        Returns:
            :class:`IngestionResult` if ingestion occurred, ``None`` if
            skipped
        """
        if last_version is not None:
            try:
                has_changes = await self._source.has_changes_since(domain_id, last_version)
                if not has_changes:
                    logger.debug(
                        "No changes since version %s for domain: %s",
                        last_version,
                        domain_id,
                    )
                    return None
            except ValueError:
                logger.warning("Domain not found: %s", domain_id)
                return None

        return await self.ingest(
            domain_id,
            progress_callback=progress_callback,
            config=config,
        )

    async def ingest_changes(
        self,
        domain_id: str,
        since_version: str,
        *,
        progress_callback: Callable[[str, int], None] | None = None,
        config: KnowledgeBaseConfig | None = None,
    ) -> IngestionResult:
        """Ingest only the files changed since ``since_version``.

        ``since_version`` must be a canonical snapshot id previously
        returned by :meth:`get_current_version` (equivalently
        ``backend.get_checksum``) — *not* the monotonic
        ``KnowledgeBaseInfo.version`` counter. The source backend's
        :meth:`~dataknobs_bots.knowledge.storage.KnowledgeResourceBackend.list_changes_since`
        computes the file-level delta; chunks for **deleted and
        modified** files are purged, then **added and modified** files
        are re-embedded through the shared apply-core, so a per-file
        re-ingest gets the same swap semantics as a full re-ingest.

        If ``since_version`` predates the backend's snapshot retention
        (the backend raises
        :class:`~dataknobs_bots.knowledge.storage.InvalidVersionError`),
        this falls back to a full :meth:`ingest` (``CLEAR_FIRST``)
        after a warning — it never silently skips the update.

        Args:
            domain_id: Domain to ingest changes for
            since_version: Canonical snapshot id to diff against
            progress_callback: Optional ``(file_path, num_chunks)``
                callback, passed through to the re-embed
            config: Optional :class:`KnowledgeBaseConfig` overriding
                any backend-hosted ``knowledge_base.*``

        Returns:
            :class:`IngestionResult` where ``files_processed`` counts
            added + modified files and ``files_deleted`` carries the
            count of files removed because they no longer exist at the
            source.
        """
        from .storage import InvalidVersionError

        # Resolve the file-level delta *before* entering this method's
        # ingest lifecycle. A ``since_version`` that predates the
        # backend's snapshot retention is a full handoff to
        # :meth:`ingest`, which runs its own complete lifecycle (status
        # write, ``_finalize``, ``result.finish()``). That handoff must
        # stay outside the lifecycle ``try`` below — otherwise a failure
        # inside the delegated ``ingest`` would get its terminal "error"
        # status and ``result.finish()`` written a second time here.
        try:
            change = await self._source.list_changes_since(
                domain_id, since_version
            )
        except InvalidVersionError:
            logger.warning(
                "Version %s for domain %s predates snapshot "
                "retention; falling back to a full re-ingest",
                since_version,
                domain_id,
            )
            return await self.ingest(
                domain_id,
                progress_callback=progress_callback,
                config=config,
            )
        except Exception as e:
            logger.error(
                "Incremental ingestion failed for %s: %s", domain_id, e
            )
            await self._source.set_ingestion_status(
                domain_id, "error", str(e)
            )
            raise

        result = IngestionResult(domain_id=domain_id)
        try:
            await self._source.set_ingestion_status(domain_id, "ingesting")

            changed_paths = {f.path for f in change.added} | {
                f.path for f in change.modified
            }
            # Modified files' stale chunks must be purged before
            # re-embedding, alongside the deleted ones.
            delete_paths = sorted(
                set(change.deleted)
                | {f.path for f in change.modified}
            )
            logger.info(
                "Ingesting changes for %s: +%d ~%d -%d (since %s)",
                domain_id,
                len(change.added),
                len(change.modified),
                len(change.deleted),
                since_version,
            )

            if not changed_paths and not change.deleted:
                # Nothing added, modified, or deleted — a successful
                # no-op (still records terminal status + event).
                await self._finalize(domain_id, result)
                return result

            await self._apply_file_ops(
                domain_id,
                upsert_filter=lambda f: f.path in changed_paths,
                delete_paths=delete_paths,
                swap_mode=IngestSwapMode.APPEND,
                config=config,
                progress_callback=progress_callback,
                result=result,
            )
            # Must precede _finalize: _finalize publishes files_deleted
            # in the completion event payload (it is owned by this
            # caller, not by _apply_file_ops).
            result.files_deleted = len(change.deleted)
            await self._finalize(domain_id, result)

        except Exception as e:
            logger.error(
                "Incremental ingestion failed for %s: %s", domain_id, e
            )
            await self._source.set_ingestion_status(
                domain_id, "error", str(e)
            )
            raise
        finally:
            result.finish()

        return result

    async def get_current_version(self, domain_id: str) -> str | None:
        """Get the current version of a knowledge base.

        Returns the canonical content-snapshot identity
        (``backend.get_checksum``) — the value
        :meth:`ingest_if_changed` compares via
        ``has_changes_since``. Capturing this and passing it back as
        ``last_version`` is now a correct round-trip (previously it
        returned the monotonic ``info.version`` counter, which lived in
        a different space than ``has_changes_since`` compared against,
        causing every checksum-keyed check to spuriously re-ingest).

        Args:
            domain_id: Domain to get version for

        Returns:
            Canonical snapshot identity, or ``None`` if domain doesn't
            exist
        """
        if await self._source.get_info(domain_id) is None:
            return None
        return await self._source.get_checksum(domain_id)
