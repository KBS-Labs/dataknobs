"""Knowledge ingestion service for automated knowledge base population.

Provides a high-level interface for:
- Checking if ingestion is needed
- Running ingestion from configuration
- Ensuring knowledge bases are populated before use
- Domain-scoped ingestion management

This service complements KnowledgeIngestionManager by adding:
- Automatic population checks (skip if already has content)
- Configuration-driven ingestion from documents_path
- Singleton pattern for convenience

Example:
    service = KnowledgeIngestionService()

    # Check and ingest if needed
    result = await service.ensure_ingested(knowledge_base, kb_config)

    if result.skipped:
        print(f"Already populated: {result.reason}")
    else:
        print(f"Ingested {result.total_chunks} chunks")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataknobs_bots.knowledge.rag import RAGKnowledgeBase
    from dataknobs_bots.knowledge.ingestion import IngestionResult

logger = logging.getLogger(__name__)


@dataclass
class EnsureIngestionResult:
    """Result of an ensure_ingested operation.

    Relationship to IngestionResult:
    --------------------------------
    DataKnobs has TWO result types for ingestion at different abstraction levels:

    1. `IngestionResult` (existing, in ingestion.py):
       - Used by KnowledgeIngestionManager for file-backend-to-vector-store coordination
       - Requires domain_id (operates on storage backend domains)
       - Tracks files_processed, files_skipped, chunks_created
       - Has duration_seconds property
       - No "skip" concept - always processes files when called

    2. `EnsureIngestionResult` (new, in service.py):
       - Used by KnowledgeIngestionService for high-level "ensure populated" operations
       - Operates at knowledge base level (no domain_id required)
       - Has skipped/reason for skip-if-populated semantics
       - Tracks total_files, total_chunks
       - Has error field for top-level failures

    The service can internally use KnowledgeIngestionManager and transform its
    IngestionResult into an EnsureIngestionResult, or it can use
    RAGKnowledgeBase.load_documents_from_directory() directly for simpler cases.

    Example transformation (if using KnowledgeIngestionManager internally):
        manager_result: IngestionResult = await manager.ingest(domain_id)
        return EnsureIngestionResult(
            total_files=manager_result.files_processed,
            total_chunks=manager_result.chunks_created,
            errors=manager_result.errors,
        )
    """

    skipped: bool = False
    reason: str | None = None
    total_files: int = 0
    total_chunks: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @property
    def success(self) -> bool:
        """Check if operation succeeded (skipped counts as success)."""
        return self.skipped or (self.error is None and len(self.errors) == 0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "skipped": self.skipped,
            "reason": self.reason,
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "errors": self.errors,
            "error": self.error,
            "success": self.success,
        }

    @classmethod
    def from_ingestion_result(
        cls,
        result: IngestionResult,
    ) -> EnsureIngestionResult:
        """Create from a KnowledgeIngestionManager IngestionResult.

        Args:
            result: IngestionResult from KnowledgeIngestionManager

        Returns:
            Equivalent EnsureIngestionResult
        """
        return cls(
            total_files=result.files_processed,
            total_chunks=result.chunks_created,
            errors=result.errors,
            started_at=result.started_at,
            completed_at=result.completed_at,
        )


class KnowledgeIngestionService:
    """Service for managing knowledge base ingestion.

    This service provides high-level ingestion management for use cases like:
    - Auto-ingestion when bots are registered
    - Ensuring knowledge bases are populated before use
    - Configuration-driven ingestion from file paths

    For lower-level file-backend-to-vector-store coordination, use
    KnowledgeIngestionManager instead.

    Args:
        force_reingest: If True, always reingest even if populated
    """

    def __init__(self, force_reingest: bool = False) -> None:
        self._force_reingest = force_reingest
        self._logger = logging.getLogger(f"{__name__}.KnowledgeIngestionService")

    async def check_needs_ingestion(
        self,
        knowledge_base: RAGKnowledgeBase,
        min_chunks: int = 1,
    ) -> bool:
        """Check if a knowledge base needs ingestion.

        Uses RAGKnowledgeBase.count() to determine if ingestion is needed.

        Args:
            knowledge_base: The RAGKnowledgeBase to check
            min_chunks: Minimum number of chunks to consider populated

        Returns:
            True if ingestion is needed, False if already populated
        """
        if self._force_reingest:
            return True

        try:
            # Use the count() method
            count = await knowledge_base.count()
            needs_ingestion = count < min_chunks
            self._logger.debug(
                "Knowledge base has %d chunks, needs_ingestion=%s",
                count,
                needs_ingestion,
            )
            return needs_ingestion

        except Exception as e:
            self._logger.warning("Error checking ingestion status: %s", e)
            return True

    async def ingest_from_config(
        self,
        knowledge_base: RAGKnowledgeBase,
        kb_config: dict[str, Any],
    ) -> EnsureIngestionResult:
        """Ingest documents based on knowledge_base configuration.

        Args:
            knowledge_base: The RAGKnowledgeBase to populate
            kb_config: Knowledge base configuration dict with:
                - documents_path: Path to documents directory
                - document_pattern: Glob pattern (default: "**/*.md")

        Returns:
            EnsureIngestionResult with operation details
        """
        result = EnsureIngestionResult()

        # Validate configuration
        documents_path = kb_config.get("documents_path")
        if not documents_path:
            result.error = "No documents_path specified in knowledge_base config"
            result.completed_at = datetime.now(timezone.utc)
            return result

        # Resolve path
        docs_path = Path(documents_path)
        if not docs_path.is_absolute():
            # Try to resolve relative to common locations
            if not docs_path.exists():
                cwd_path = Path.cwd() / documents_path
                if cwd_path.exists():
                    docs_path = cwd_path

        if not docs_path.exists():
            result.error = f"Documents path does not exist: {documents_path}"
            result.completed_at = datetime.now(timezone.utc)
            return result

        document_pattern = kb_config.get("document_pattern", "**/*.md")

        self._logger.info(
            "Starting ingestion from %s with pattern %s",
            docs_path,
            document_pattern,
        )

        try:
            # Run ingestion using RAGKnowledgeBase method
            load_result = await knowledge_base.load_documents_from_directory(
                directory=docs_path,
                pattern=document_pattern,
            )

            result.total_files = load_result.get("total_files", 0)
            result.total_chunks = load_result.get("total_chunks", 0)
            result.errors = load_result.get("errors", [])

            # Save the knowledge base
            if hasattr(knowledge_base, "save"):
                await knowledge_base.save()
                self._logger.info("Knowledge base saved")

            self._logger.info(
                "Ingestion complete: %d files, %d chunks",
                result.total_files,
                result.total_chunks,
            )

        except Exception as e:
            self._logger.error("Ingestion failed: %s", e)
            result.error = str(e)

        result.completed_at = datetime.now(timezone.utc)
        return result

    async def ensure_ingested(
        self,
        knowledge_base: RAGKnowledgeBase,
        kb_config: dict[str, Any],
        force: bool = False,
    ) -> EnsureIngestionResult:
        """Ensure a knowledge base is populated, ingesting if necessary.

        This is the main entry point for automatic ingestion. It checks if
        the knowledge base needs ingestion and runs it if so.

        Args:
            knowledge_base: The RAGKnowledgeBase to check/populate
            kb_config: Knowledge base configuration dict
            force: Force re-ingestion even if already populated

        Returns:
            EnsureIngestionResult with operation details
        """
        # Check if knowledge base is enabled
        if not kb_config.get("enabled", False):
            return EnsureIngestionResult(
                skipped=True,
                reason="knowledge_base_disabled",
            )

        # Check if we need to ingest
        if not force:
            needs_ingestion = await self.check_needs_ingestion(knowledge_base)
            if not needs_ingestion:
                self._logger.info("Knowledge base already populated, skipping ingestion")
                return EnsureIngestionResult(
                    skipped=True,
                    reason="already_populated",
                )

        # Run ingestion
        self._logger.info("Running knowledge base ingestion")
        return await self.ingest_from_config(knowledge_base, kb_config)


# Module-level convenience using a class to avoid global statement
class _ServiceHolder:
    """Holder for the default service instance."""

    instance: KnowledgeIngestionService | None = None


def get_ingestion_service(force_reingest: bool = False) -> KnowledgeIngestionService:
    """Get the default ingestion service instance.

    Args:
        force_reingest: If True, always reingest even if populated

    Returns:
        Singleton KnowledgeIngestionService instance
    """
    if _ServiceHolder.instance is None:
        _ServiceHolder.instance = KnowledgeIngestionService(force_reingest=force_reingest)
    return _ServiceHolder.instance


async def ensure_knowledge_base_ingested(
    knowledge_base: RAGKnowledgeBase,
    config: dict[str, Any],
    force: bool = False,
) -> EnsureIngestionResult:
    """Convenience function to ensure a knowledge base is ingested.

    Args:
        knowledge_base: The RAGKnowledgeBase to check/populate
        config: Knowledge base configuration dict
        force: Force re-ingestion

    Returns:
        EnsureIngestionResult with operation details
    """
    service = get_ingestion_service()
    return await service.ensure_ingested(knowledge_base, config, force=force)
