"""Knowledge ingestion manager for coordinating file storage to vector storage.

This module provides the KnowledgeIngestionManager which coordinates loading
files from a KnowledgeResourceBackend into a RAGKnowledgeBase.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from dataknobs_common.events import EventBus

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

    This manager bridges KnowledgeResourceBackend (file storage) and
    RAGKnowledgeBase (vector storage), handling:
    - File discovery and download
    - Decompression of .gz files
    - Content transformation and chunking
    - Status tracking
    - Event publishing for hot-reload

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

    # Supported file extensions and their handlers
    SUPPORTED_EXTENSIONS: dict[str, str] = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".csv": "csv",
        ".txt": "text",
    }

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
    ) -> IngestionResult:
        """Ingest all files from a domain into the knowledge base.

        Args:
            domain_id: Domain to ingest
            clear_existing: Clear existing vectors before ingesting (default: True)
            progress_callback: Optional callback(file_path, chunks_created)

        Returns:
            IngestionResult with statistics
        """
        result = IngestionResult(domain_id=domain_id)

        try:
            # Update status
            await self._source.set_ingestion_status(domain_id, "ingesting")
            logger.info("Starting ingestion for domain: %s", domain_id)

            # Clear existing content if requested
            if clear_existing:
                await self._destination.clear()
                logger.debug("Cleared existing vectors for domain: %s", domain_id)

            # List and process files
            files = await self._source.list_files(domain_id)
            logger.info("Found %d files to process for domain: %s", len(files), domain_id)

            for file_info in files:
                try:
                    chunks = await self._ingest_file(domain_id, file_info.path)
                    if chunks > 0:
                        result.files_processed += 1
                        result.chunks_created += chunks
                        logger.debug(
                            "Ingested %s: %d chunks",
                            file_info.path,
                            chunks,
                        )
                        if progress_callback:
                            progress_callback(file_info.path, chunks)
                    else:
                        result.files_skipped += 1
                        logger.debug("Skipped %s: unsupported type", file_info.path)

                except Exception as e:
                    logger.warning("Failed to ingest %s: %s", file_info.path, e)
                    result.errors.append({
                        "file": file_info.path,
                        "error": str(e),
                    })

            # Update status
            result.completed_at = datetime.now(timezone.utc)
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

            # Publish event
            if self._event_bus:
                await self._publish_ingestion_event(domain_id, result, status)

        except Exception as e:
            logger.error("Ingestion failed for %s: %s", domain_id, e)
            await self._source.set_ingestion_status(domain_id, "error", str(e))
            raise

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

    async def _ingest_file(self, domain_id: str, path: str) -> int:
        """Ingest a single file. Returns number of chunks created."""
        # Get content
        content = await self._source.get_file(domain_id, path)
        if content is None:
            return 0

        # Handle compression
        actual_path = path
        if path.endswith(".gz"):
            content = gzip.decompress(content)
            actual_path = path[:-3]  # Remove .gz

        # Determine file type
        suffix = Path(actual_path).suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(suffix)
        if file_type is None:
            logger.debug("Skipping unsupported file type: %s", path)
            return 0

        # Dispatch to appropriate loader
        metadata = {"domain_id": domain_id, "source_path": path}

        if file_type == "markdown":
            return await self._load_markdown(content, path, metadata)
        elif file_type == "json":
            return await self._load_json(content, path, metadata)
        elif file_type == "yaml":
            return await self._load_yaml(content, path, metadata)
        elif file_type == "csv":
            return await self._load_csv(content, path, metadata)
        elif file_type == "text":
            return await self._load_text(content, path, metadata)

        return 0

    async def _load_markdown(
        self,
        content: bytes,
        source: str,
        metadata: dict[str, Any],
    ) -> int:
        """Load markdown content into the knowledge base."""
        text = content.decode("utf-8")
        return await self._destination._load_markdown_text(
            text,
            source=source,
            metadata=metadata,
        )

    async def _load_json(
        self,
        content: bytes,
        source: str,
        metadata: dict[str, Any],
    ) -> int:
        """Load JSON content by converting to markdown first."""
        from dataknobs_xization import ContentTransformer

        data = json.loads(content)
        transformer = ContentTransformer()
        markdown = transformer.transform_json(data)

        return await self._destination._load_markdown_text(
            markdown,
            source=source,
            metadata=metadata,
        )

    async def _load_yaml(
        self,
        content: bytes,
        source: str,
        metadata: dict[str, Any],
    ) -> int:
        """Load YAML content by converting to markdown first."""
        import yaml

        from dataknobs_xization import ContentTransformer

        data = yaml.safe_load(content)
        transformer = ContentTransformer()
        # YAML loads as dict, same as JSON
        markdown = transformer.transform_json(data)

        return await self._destination._load_markdown_text(
            markdown,
            source=source,
            metadata=metadata,
        )

    async def _load_csv(
        self,
        content: bytes,
        source: str,
        metadata: dict[str, Any],
    ) -> int:
        """Load CSV content by converting to markdown table."""
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
        rows = list(reader)
        if not rows:
            return 0

        # Build markdown table
        headers = list(rows[0].keys())
        md_lines = ["| " + " | ".join(headers) + " |"]
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            md_lines.append("| " + " | ".join(str(v) for v in row.values()) + " |")
        markdown = "\n".join(md_lines)

        return await self._destination._load_markdown_text(
            markdown,
            source=source,
            metadata=metadata,
        )

    async def _load_text(
        self,
        content: bytes,
        source: str,
        metadata: dict[str, Any],
    ) -> int:
        """Load plain text content."""
        text = content.decode("utf-8")
        return await self._destination._load_markdown_text(
            text,
            source=source,
            metadata=metadata,
        )

    async def ingest_if_changed(
        self,
        domain_id: str,
        last_version: str | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> IngestionResult | None:
        """Ingest only if the knowledge base has changed.

        Useful for hot-reload scenarios where you want to skip
        re-ingestion if nothing has changed.

        Args:
            domain_id: Domain to check and potentially ingest
            last_version: Last known version string (if None, always ingests)
            progress_callback: Optional callback(file_path, chunks_created)

        Returns:
            IngestionResult if ingestion occurred, None if skipped
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
                # Domain doesn't exist, skip
                logger.warning("Domain not found: %s", domain_id)
                return None

        return await self.ingest(domain_id, progress_callback=progress_callback)

    async def get_current_version(self, domain_id: str) -> str | None:
        """Get the current version of a knowledge base.

        Useful for caching the version to use with ingest_if_changed.

        Args:
            domain_id: Domain to get version for

        Returns:
            Current version string, or None if domain doesn't exist
        """
        info = await self._source.get_info(domain_id)
        if info is None:
            return None
        return info.version
