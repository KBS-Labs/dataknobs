"""Knowledge ingestion manager for coordinating file storage to vector storage.

This module provides the KnowledgeIngestionManager which coordinates loading
files from a KnowledgeResourceBackend into a RAGKnowledgeBase.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataknobs_common.events import EventBus
    from dataknobs_xization.ingestion import KnowledgeBaseConfig

    from .rag import RAGKnowledgeBase
    from .storage import KnowledgeResourceBackend

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of an ingestion operation.

    Contains statistics about files processed, chunks created, and any errors
    encountered during ingestion.
    """

    domain_id: str
    files_processed: int = 0
    chunks_created: int = 0
    files_skipped: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

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

            if clear_existing:
                await self._destination.clear()
                logger.debug("Cleared existing vectors for domain: %s", domain_id)

            stats = await self._destination.ingest_from_backend(
                self._source,
                domain_id,
                config=config,
                progress_callback=progress_callback,
                extra_metadata={"domain_id": domain_id},
            )

            result.files_processed = int(stats.get("total_files", 0))
            result.chunks_created = int(stats.get("total_chunks", 0))
            result.files_skipped = int(stats.get("files_skipped", 0))
            result.errors = list(stats.get("errors", []))

            status = "ready" if result.success else "error"
            error_msg = str(result.errors) if result.errors else None
            await self._source.set_ingestion_status(domain_id, status, error_msg)

            logger.info(
                "Ingestion completed for %s: %d files, %d chunks, %d errors",
                domain_id,
                result.files_processed,
                result.chunks_created,
                len(result.errors),
            )

            if self._event_bus:
                await self._publish_ingestion_event(domain_id, result, status)

        except Exception as e:
            logger.error("Ingestion failed for %s: %s", domain_id, e)
            await self._source.set_ingestion_status(domain_id, "error", str(e))
            raise
        finally:
            result.completed_at = datetime.now(timezone.utc)

        return result

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

    async def get_current_version(self, domain_id: str) -> str | None:
        """Get the current version of a knowledge base.

        Useful for caching the version to use with
        :meth:`ingest_if_changed`.

        Args:
            domain_id: Domain to get version for

        Returns:
            Current version string, or ``None`` if domain doesn't exist
        """
        info = await self._source.get_info(domain_id)
        if info is None:
            return None
        return info.version
