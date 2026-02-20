"""Artifact corpus for managing collections of related artifacts.

A corpus is a named, typed collection of artifacts (e.g., a quiz bank is a
corpus of quiz questions). It provides corpus-level operations: add items
with optional dedup, query items, get summaries, and finalize.

Example:
    >>> from dataknobs_bots.artifacts import ArtifactCorpus, ArtifactRegistry
    >>> from dataknobs_bots.artifacts.corpus import CorpusConfig
    >>> corpus = await ArtifactCorpus.create(
    ...     registry=registry,
    ...     config=CorpusConfig(
    ...         corpus_type="quiz_bank",
    ...         item_type="quiz_question",
    ...         name="Chapter 1 Quiz",
    ...     ),
    ... )
    >>> artifact, dedup = await corpus.add_item({"stem": "What is 2+2?"})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dataknobs_data import Filter, Operator

from .models import Artifact, ArtifactStatus
from .registry import ArtifactRegistry

if TYPE_CHECKING:
    from dataknobs_data.dedup import DedupChecker, DedupConfig, DedupResult

    from .provenance import ProvenanceRecord

logger = logging.getLogger(__name__)


@dataclass
class CorpusConfig:
    """Configuration for a corpus.

    Attributes:
        corpus_type: Artifact type for the parent corpus (e.g., ``"quiz_bank"``).
        item_type: Artifact type for child items (e.g., ``"quiz_question"``).
        name: Human-readable corpus name.
        rubric_ids: Rubric IDs to apply to items.
        auto_review: Automatically submit items for review on creation.
        dedup_config: Optional dedup configuration for the corpus.
        metadata: Additional metadata stored on the corpus artifact.
    """

    corpus_type: str
    item_type: str
    name: str
    rubric_ids: list[str] = field(default_factory=list)
    auto_review: bool = False
    dedup_config: DedupConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactCorpus:
    """A collection of related artifacts managed through an ArtifactRegistry.

    ArtifactCorpus wraps an ``ArtifactRegistry`` to provide collection-level
    operations. The corpus itself is stored as an artifact, and items link to
    it via ``content["corpus_id"]``.

    Use ``create()`` to start a new corpus or ``load()`` to resume an existing one.
    """

    def __init__(
        self,
        registry: ArtifactRegistry,
        corpus_artifact: Artifact,
        config: CorpusConfig,
        dedup_checker: DedupChecker | None = None,
    ) -> None:
        """Initialize corpus.

        Prefer using ``create()`` or ``load()`` factory methods.

        Args:
            registry: The artifact registry backing this corpus.
            corpus_artifact: The parent corpus artifact.
            config: Corpus configuration.
            dedup_checker: Optional dedup checker for duplicate detection.
        """
        self._registry = registry
        self._corpus_artifact = corpus_artifact
        self._config = config
        self._dedup_checker = dedup_checker

    @classmethod
    async def create(
        cls,
        registry: ArtifactRegistry,
        config: CorpusConfig,
        dedup_checker: DedupChecker | None = None,
    ) -> ArtifactCorpus:
        """Create a new corpus with a parent artifact.

        Args:
            registry: The artifact registry.
            config: Corpus configuration.
            dedup_checker: Optional dedup checker.

        Returns:
            A new ArtifactCorpus instance.
        """
        corpus_artifact = await registry.create(
            artifact_type=config.corpus_type,
            name=config.name,
            content={
                "item_type": config.item_type,
                "metadata": config.metadata,
            },
            tags=["corpus"],
        )
        logger.info(
            "Created corpus '%s' (id=%s, item_type=%s)",
            config.name,
            corpus_artifact.id,
            config.item_type,
        )
        return cls(
            registry=registry,
            corpus_artifact=corpus_artifact,
            config=config,
            dedup_checker=dedup_checker,
        )

    @classmethod
    async def load(
        cls,
        registry: ArtifactRegistry,
        corpus_id: str,
    ) -> ArtifactCorpus:
        """Load an existing corpus by its artifact ID.

        Args:
            registry: The artifact registry.
            corpus_id: The ID of the corpus artifact to load.

        Returns:
            A restored ArtifactCorpus instance.

        Raises:
            ValueError: If the corpus artifact is not found.
        """
        corpus_artifact = await registry.get(corpus_id)
        if corpus_artifact is None:
            raise ValueError(f"Corpus '{corpus_id}' not found")

        content = corpus_artifact.content
        config = CorpusConfig(
            corpus_type=corpus_artifact.type,
            item_type=content.get("item_type", ""),
            name=corpus_artifact.name,
            metadata=content.get("metadata", {}),
        )
        return cls(
            registry=registry,
            corpus_artifact=corpus_artifact,
            config=config,
        )

    @property
    def id(self) -> str:
        """The corpus artifact ID."""
        return self._corpus_artifact.id

    async def add_item(
        self,
        content: dict[str, Any],
        provenance: ProvenanceRecord | None = None,
        tags: list[str] | None = None,
        skip_dedup: bool = False,
    ) -> tuple[Artifact, DedupResult | None]:
        """Add an item to the corpus.

        Performs dedup checking if a ``DedupChecker`` is configured. On exact
        duplicate, returns the existing artifact instead of creating a new one.

        Args:
            content: The item content dictionary.
            provenance: Optional provenance record.
            tags: Optional tags for the item.
            skip_dedup: Skip dedup checking even if configured.

        Returns:
            Tuple of (artifact, dedup_result). ``dedup_result`` is None when
            dedup checking is disabled or skipped.
        """
        dedup_result: DedupResult | None = None

        # Step 1: Dedup check
        if not skip_dedup and self._dedup_checker:
            dedup_result = await self._dedup_checker.check(content)
            if dedup_result.is_exact_duplicate:
                logger.info(
                    "Exact duplicate detected in corpus '%s': matches %s",
                    self.id,
                    dedup_result.exact_match_id,
                )
                existing = await self._registry.get(dedup_result.exact_match_id)
                if existing:
                    return existing, dedup_result

        # Step 2: Create the item artifact with corpus_id in content
        item_tags = list(tags or [])
        item_content = dict(content)
        item_content["corpus_id"] = self.id

        artifact = await self._registry.create(
            artifact_type=self._config.item_type,
            name=f"{self._config.name} item",
            content=item_content,
            provenance=provenance,
            tags=item_tags,
        )

        # Step 3: Register with dedup checker for future checks
        if self._dedup_checker:
            await self._dedup_checker.register(content, artifact.id)

        # Step 4: Auto-review if configured
        if self._config.auto_review and artifact.rubric_ids:
            await self._registry.submit_for_review(artifact.id)

        logger.info(
            "Added item to corpus '%s': artifact_id=%s",
            self.id,
            artifact.id,
        )
        return artifact, dedup_result

    async def get_items(
        self,
        status: ArtifactStatus | None = None,
    ) -> list[Artifact]:
        """Get items belonging to this corpus.

        Args:
            status: Optional status filter.

        Returns:
            List of matching artifacts.
        """
        return await self._registry.query(
            artifact_type=self._config.item_type,
            status=status,
            filters=[Filter("content.corpus_id", Operator.EQ, self.id)],
        )

    async def count(
        self,
        status: ArtifactStatus | None = None,
    ) -> int:
        """Count items in this corpus.

        Args:
            status: Optional status filter.

        Returns:
            Number of matching items.
        """
        items = await self.get_items(status=status)
        return len(items)

    async def remove_item(self, artifact_id: str) -> None:
        """Archive an item from the corpus.

        The item's status is set to ARCHIVED. It remains in storage but
        will no longer appear in unfiltered queries.

        Args:
            artifact_id: The artifact ID to archive.
        """
        await self._registry.set_status(
            artifact_id,
            ArtifactStatus.ARCHIVED,
            reason=f"Removed from corpus '{self.id}'",
        )

    async def finalize(self) -> Artifact:
        """Finalize the corpus, recording item summary and approving.

        Updates the corpus artifact content with item IDs and count,
        then transitions it to APPROVED status.

        Returns:
            The finalized corpus artifact.
        """
        items = await self.get_items()
        item_ids = [item.id for item in items]

        summary_content = {
            **self._corpus_artifact.content,
            "item_ids": item_ids,
            "item_count": len(item_ids),
            "finalized": True,
        }

        revised = await self._registry.revise(
            artifact_id=self.id,
            new_content=summary_content,
            reason="Corpus finalized",
            triggered_by="system:corpus_finalize",
        )

        # Transition through valid chain: DRAFT -> PENDING_REVIEW -> IN_REVIEW -> APPROVED
        await self._registry.set_status(
            revised.id,
            ArtifactStatus.PENDING_REVIEW,
            reason="Corpus finalization review",
        )
        await self._registry.set_status(
            revised.id,
            ArtifactStatus.IN_REVIEW,
            reason="Corpus finalization review",
        )
        await self._registry.set_status(
            revised.id,
            ArtifactStatus.APPROVED,
            reason="Corpus finalized with all items",
        )

        self._corpus_artifact = revised
        # Refresh to get updated status
        refreshed = await self._registry.get(revised.id)
        if refreshed:
            self._corpus_artifact = refreshed

        logger.info(
            "Finalized corpus '%s' with %d items", self.id, len(item_ids)
        )
        return self._corpus_artifact

    async def get_summary(self) -> dict[str, Any]:
        """Get corpus statistics.

        Returns:
            Dictionary with corpus metadata and item status breakdown.
        """
        all_items = await self.get_items()
        status_counts: dict[str, int] = {}
        for item in all_items:
            status = item.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "corpus_id": self.id,
            "corpus_name": self._config.name,
            "corpus_type": self._config.corpus_type,
            "item_type": self._config.item_type,
            "total_items": len(all_items),
            "status_breakdown": status_counts,
            "corpus_status": self._corpus_artifact.status.value,
            "metadata": self._config.metadata,
        }
