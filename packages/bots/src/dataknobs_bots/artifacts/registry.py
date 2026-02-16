"""Artifact registry for managing artifacts with async database backing.

The registry provides:
- Artifact creation with provenance tracking
- Versioning (creating new versions preserves history)
- Status transitions enforced by TransitionValidator
- Rubric-based evaluation via submit_for_review
- Querying by type, status, tags

Example:
    >>> from dataknobs_data.backends.memory import AsyncMemoryDatabase
    >>> db = AsyncMemoryDatabase()
    >>> registry = ArtifactRegistry(db)
    >>> artifact = await registry.create(
    ...     artifact_type="content",
    ...     name="Questions",
    ...     content={"questions": [...]},
    ...     provenance=create_provenance("bot:edubot", "generator"),
    ... )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from dataknobs_data import AsyncDatabase, Filter, Operator, Query, Record

from .models import Artifact, ArtifactStatus, ArtifactTypeDefinition, _generate_artifact_id, _now_iso
from .provenance import ProvenanceRecord, RevisionRecord, create_provenance
from .transitions import validate_transition

logger = logging.getLogger(__name__)

# Type alias for lifecycle hook callbacks
LifecycleHook = Callable[[Artifact], Awaitable[None]]


class ArtifactRegistry:
    """Async artifact registry backed by AsyncDatabase.

    Manages artifact lifecycle including creation, versioning, status
    transitions, and rubric-based evaluation.

    Args:
        db: The async database backend for artifact storage.
        rubric_registry: Optional rubric registry for evaluation lookups.
        rubric_executor: Optional rubric executor for running evaluations.
        type_definitions: Optional mapping of type IDs to their definitions.
    """

    def __init__(
        self,
        db: AsyncDatabase,
        rubric_registry: Any | None = None,
        rubric_executor: Any | None = None,
        type_definitions: dict[str, ArtifactTypeDefinition] | None = None,
    ) -> None:
        self._db = db
        self._rubric_registry = rubric_registry
        self._rubric_executor = rubric_executor
        self._type_definitions = type_definitions or {}
        self._on_create_hooks: list[LifecycleHook] = []
        self._on_status_change_hooks: list[LifecycleHook] = []
        self._on_review_complete_hooks: list[LifecycleHook] = []

    # --- Creation ---

    async def create(
        self,
        artifact_type: str,
        name: str,
        content: dict[str, Any],
        provenance: ProvenanceRecord | None = None,
        tags: list[str] | None = None,
    ) -> Artifact:
        """Create a new artifact.

        Args:
            artifact_type: The type of artifact to create.
            name: Human-readable name.
            content: The artifact content.
            provenance: Provenance record. If None, a minimal one is created.
            tags: Searchable tags. Merged with type definition defaults.

        Returns:
            The created artifact.
        """
        type_def = self._type_definitions.get(artifact_type)

        effective_tags = list(tags or [])
        rubric_ids: list[str] = []
        if type_def:
            effective_tags = list(set(effective_tags + type_def.tags))
            rubric_ids = list(type_def.rubrics)

        effective_provenance = provenance or create_provenance(
            created_by="system",
            creation_method="manual",
        )

        artifact = Artifact(
            id=_generate_artifact_id(),
            type=artifact_type,
            name=name,
            content=content,
            provenance=effective_provenance,
            tags=effective_tags,
            rubric_ids=rubric_ids,
        )

        await self._store(artifact)

        for hook in self._on_create_hooks:
            await hook(artifact)

        logger.info(
            "Created artifact '%s' (type=%s, id=%s)",
            name,
            artifact_type,
            artifact.id,
        )

        return artifact

    # --- Retrieval ---

    async def get(self, artifact_id: str) -> Artifact | None:
        """Retrieve an artifact by ID.

        Returns the latest version of the artifact.
        """
        record = await self._db.read(artifact_id)
        if record is None:
            return None
        return Artifact.from_dict(record.data)

    async def get_version(
        self, artifact_id: str, version: str
    ) -> Artifact | None:
        """Retrieve a specific version of an artifact."""
        key = f"{artifact_id}:{version}"
        record = await self._db.read(key)
        if record is None:
            return None
        return Artifact.from_dict(record.data)

    async def query(
        self,
        artifact_type: str | None = None,
        status: ArtifactStatus | None = None,
        tags: list[str] | None = None,
    ) -> list[Artifact]:
        """Query artifacts by type, status, and/or tags.

        Args:
            artifact_type: Filter by artifact type.
            status: Filter by status.
            tags: Filter by tags (artifact must have all specified tags).

        Returns:
            List of matching artifacts.
        """
        filters: list[Filter] = []
        if artifact_type:
            filters.append(Filter("type", Operator.EQ, artifact_type))
        if status:
            filters.append(Filter("status", Operator.EQ, status.value))

        records = await self._db.search(Query(filters=filters))

        artifacts: list[Artifact] = []
        seen_ids: set[str] = set()
        for record in records:
            data = record.data
            # Skip versioned records
            version_key = data.get("_version_key", "")
            record_key = record.storage_id or record.id
            if version_key and record_key == version_key:
                continue

            artifact = Artifact.from_dict(data)

            if tags and not all(t in artifact.tags for t in tags):
                continue

            if artifact.id not in seen_ids:
                seen_ids.add(artifact.id)
                artifacts.append(artifact)

        return artifacts

    # --- Updates ---

    async def revise(
        self,
        artifact_id: str,
        new_content: dict[str, Any],
        reason: str,
        triggered_by: str,
    ) -> Artifact:
        """Create a new version of an artifact with revised content.

        The old version is marked as superseded. The new version inherits
        provenance and adds a revision record.

        Args:
            artifact_id: The artifact to revise.
            new_content: The updated content.
            reason: Why the revision was made.
            triggered_by: Who or what triggered the revision.

        Returns:
            The new artifact version.

        Raises:
            ValueError: If the artifact is not found.
        """
        current = await self.get(artifact_id)
        if current is None:
            raise ValueError(f"Artifact '{artifact_id}' not found")

        # Bump version
        parts = current.version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_version = ".".join(parts)

        # Mark old version as superseded
        await self.set_status(
            artifact_id, ArtifactStatus.SUPERSEDED, reason="Superseded by revision"
        )

        # Create revision record
        revision = RevisionRecord(
            previous_version=current.version,
            reason=reason,
            changes_summary=f"Revised content for artifact '{current.name}'",
            triggered_by=triggered_by,
        )

        # Build new provenance from current
        new_provenance = ProvenanceRecord.from_dict(current.provenance.to_dict())
        new_provenance.revision_history.append(revision)

        new_artifact = Artifact(
            id=artifact_id,
            type=current.type,
            name=current.name,
            version=new_version,
            status=ArtifactStatus.DRAFT,
            content=new_content,
            content_schema=current.content_schema,
            provenance=new_provenance,
            tags=list(current.tags),
            rubric_ids=list(current.rubric_ids),
        )

        await self._store(new_artifact)

        logger.info(
            "Revised artifact '%s' from v%s to v%s",
            artifact_id,
            current.version,
            new_version,
        )

        return new_artifact

    async def set_status(
        self,
        artifact_id: str,
        status: ArtifactStatus,
        reason: str | None = None,
    ) -> None:
        """Update an artifact's status with transition validation.

        Args:
            artifact_id: The artifact to update.
            status: The target status.
            reason: Optional reason for the status change.

        Raises:
            ValueError: If the artifact is not found.
            InvalidTransitionError: If the transition is not allowed.
        """
        artifact = await self.get(artifact_id)
        if artifact is None:
            raise ValueError(f"Artifact '{artifact_id}' not found")

        validate_transition(artifact.status, status)

        artifact.status = status
        artifact.updated_at = _now_iso()
        await self._store(artifact)

        for hook in self._on_status_change_hooks:
            await hook(artifact)

        logger.info(
            "Artifact '%s' status changed to '%s'%s",
            artifact_id,
            status.value,
            f" (reason: {reason})" if reason else "",
        )

    # --- Review Integration ---

    async def submit_for_review(
        self, artifact_id: str
    ) -> list[dict[str, Any]]:
        """Submit an artifact for rubric-based evaluation.

        Transitions the artifact to IN_REVIEW, runs applicable rubrics,
        then transitions to APPROVED or NEEDS_REVISION based on results.

        Args:
            artifact_id: The artifact to evaluate.

        Returns:
            List of evaluation result dicts.

        Raises:
            ValueError: If the artifact is not found or review components
                are not configured.
        """
        artifact = await self.get(artifact_id)
        if artifact is None:
            raise ValueError(f"Artifact '{artifact_id}' not found")

        # Transition to pending_review then in_review
        if artifact.status == ArtifactStatus.DRAFT:
            await self.set_status(artifact_id, ArtifactStatus.PENDING_REVIEW)
            artifact = await self.get(artifact_id)
            if artifact is None:
                raise ValueError(f"Artifact '{artifact_id}' not found after status update")
        if artifact.status in (
            ArtifactStatus.PENDING_REVIEW,
            ArtifactStatus.NEEDS_REVISION,
        ):
            artifact = await self.get(artifact_id)
            if artifact is None:
                raise ValueError(f"Artifact '{artifact_id}' not found after status update")
            if artifact.status == ArtifactStatus.PENDING_REVIEW:
                await self.set_status(artifact_id, ArtifactStatus.IN_REVIEW)

        if not self._rubric_registry or not self._rubric_executor:
            logger.warning(
                "Review requested for '%s' but no rubric registry/executor configured",
                artifact_id,
            )
            return []

        # Re-read after status changes
        artifact = await self.get(artifact_id)
        if artifact is None:
            raise ValueError(f"Artifact '{artifact_id}' not found after status update")

        evaluations: list[dict[str, Any]] = []
        all_passed = True

        for rubric_id in artifact.rubric_ids:
            rubric = await self._rubric_registry.get(rubric_id)
            if rubric is None:
                logger.warning("Rubric '%s' not found, skipping", rubric_id)
                continue

            evaluation = await self._rubric_executor.evaluate(
                rubric, artifact.content, target_id=artifact.id, target_type=artifact.type
            )
            evaluations.append(evaluation.to_dict())
            artifact.evaluation_ids.append(evaluation.id)

            if not evaluation.passed:
                all_passed = False

        # Update artifact with evaluation IDs
        await self._store(artifact)

        # Transition based on results
        if evaluations:
            if all_passed:
                await self.set_status(artifact_id, ArtifactStatus.APPROVED)
            else:
                await self.set_status(artifact_id, ArtifactStatus.NEEDS_REVISION)

        for hook in self._on_review_complete_hooks:
            current = await self.get(artifact_id)
            if current:
                await hook(current)

        return evaluations

    async def get_evaluations(
        self, artifact_id: str
    ) -> list[dict[str, Any]]:
        """Get evaluation results for an artifact.

        Returns evaluation dicts stored alongside the artifact.
        """
        artifact = await self.get(artifact_id)
        if artifact is None:
            return []

        evaluations: list[dict[str, Any]] = []
        for eval_id in artifact.evaluation_ids:
            record = await self._db.read(f"eval:{eval_id}")
            if record:
                evaluations.append(record.data)
        return evaluations

    # --- Lifecycle Hooks ---

    def on_create(self, callback: LifecycleHook) -> None:
        """Register a callback for artifact creation events."""
        self._on_create_hooks.append(callback)

    def on_status_change(self, callback: LifecycleHook) -> None:
        """Register a callback for status change events."""
        self._on_status_change_hooks.append(callback)

    def on_review_complete(self, callback: LifecycleHook) -> None:
        """Register a callback for review completion events."""
        self._on_review_complete_hooks.append(callback)

    # --- Configuration ---

    @classmethod
    async def from_config(
        cls,
        config: dict[str, Any],
        db: AsyncDatabase,
        rubric_registry: Any | None = None,
        rubric_executor: Any | None = None,
    ) -> ArtifactRegistry:
        """Create a registry from configuration.

        The config should have an ``"artifact_types"`` key mapping type IDs
        to their configuration dicts.

        Args:
            config: Configuration dictionary.
            db: The async database backend.
            rubric_registry: Optional rubric registry.
            rubric_executor: Optional rubric executor.

        Returns:
            A configured ArtifactRegistry.
        """
        type_defs: dict[str, ArtifactTypeDefinition] = {}
        for type_id, type_config in config.get("artifact_types", {}).items():
            type_defs[type_id] = ArtifactTypeDefinition.from_config(
                type_id, type_config
            )

        return cls(
            db=db,
            rubric_registry=rubric_registry,
            rubric_executor=rubric_executor,
            type_definitions=type_defs,
        )

    # --- Internal ---

    async def _store(self, artifact: Artifact) -> None:
        """Store an artifact, creating both a latest pointer and a versioned record."""
        data = artifact.to_dict()
        version_key = f"{artifact.id}:{artifact.version}"
        data["_version_key"] = version_key

        await self._db.upsert(artifact.id, Record(data))
        await self._db.upsert(version_key, Record(data))
