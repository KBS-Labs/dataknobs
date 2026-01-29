"""Artifact registry for managing artifacts within conversations.

The registry provides:
- Artifact creation with definition lookup
- Versioning (creating new versions preserves history)
- Status transitions
- Querying by type, status, stage, etc.
- Persistence to conversation metadata
- Lifecycle hooks for artifact events

Example:
    >>> # Create registry with definitions from config
    >>> registry = ArtifactRegistry.from_config(bot_config)
    >>>
    >>> # Create an artifact
    >>> artifact = registry.create(
    ...     definition_id="assessment_questions",
    ...     content=questions,
    ...     stage="build_questions",
    ... )
    >>>
    >>> # Query artifacts
    >>> pending = registry.get_pending_review()
    >>> questions = registry.get_by_definition("assessment_questions")
    >>>
    >>> # Create new version
    >>> updated = registry.update(artifact.id, new_content)
    >>> assert updated.lineage.version == 2
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .models import (
    Artifact,
    ArtifactDefinition,
    ArtifactMetadata,
    ArtifactLineage,
    ArtifactReview,
    ArtifactStatus,
    ArtifactType,
)


logger = logging.getLogger(__name__)

# Type alias for lifecycle hook callbacks
LifecycleHook = Callable[..., None]


class ArtifactRegistry:
    """Manages artifacts within a conversation context.

    The registry provides:
    - Artifact creation with definition lookup
    - Versioning (creating new versions preserves history)
    - Status transitions
    - Querying by type, status, stage, etc.
    - Persistence to conversation metadata

    Attributes:
        _artifacts: Dict mapping artifact ID to Artifact
        _definitions: Dict mapping definition ID to ArtifactDefinition
        _hooks: Lifecycle hooks for artifact events
    """

    def __init__(
        self,
        definitions: dict[str, ArtifactDefinition] | None = None,
    ) -> None:
        """Initialize registry.

        Args:
            definitions: Optional artifact definitions from config
        """
        self._artifacts: dict[str, Artifact] = {}
        self._definitions = definitions or {}
        self._hooks: dict[str, list[LifecycleHook]] = {
            "on_create": [],
            "on_update": [],
            "on_status_change": [],
            "on_review": [],
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ArtifactRegistry:
        """Create registry from bot configuration.

        Args:
            config: Bot configuration with 'artifacts.definitions' section

        Returns:
            Configured ArtifactRegistry
        """
        definitions: dict[str, ArtifactDefinition] = {}
        artifacts_config = config.get("artifacts", {})
        for def_id, def_config in artifacts_config.get("definitions", {}).items():
            definitions[def_id] = ArtifactDefinition.from_config(def_id, def_config)
        return cls(definitions=definitions)

    # =========================================================================
    # Definition Management
    # =========================================================================

    def register_definition(self, definition: ArtifactDefinition) -> None:
        """Register an artifact definition at runtime.

        Args:
            definition: ArtifactDefinition to register
        """
        self._definitions[definition.id] = definition
        logger.debug("Registered artifact definition: %s", definition.id)

    def get_definition(self, definition_id: str) -> ArtifactDefinition | None:
        """Get an artifact definition by ID.

        Args:
            definition_id: Definition ID to look up

        Returns:
            ArtifactDefinition if found, None otherwise
        """
        return self._definitions.get(definition_id)

    def get_available_definitions(self) -> list[str]:
        """Get IDs of all registered definitions."""
        return list(self._definitions.keys())

    # =========================================================================
    # Artifact Creation
    # =========================================================================

    def create(
        self,
        content: Any,
        definition_id: str | None = None,
        artifact_type: ArtifactType = "content",
        name: str = "",
        content_type: str = "application/json",
        stage: str | None = None,
        task_id: str | None = None,
        purpose: str | None = None,
        source_ids: list[str] | None = None,
        tags: list[str] | None = None,
        **metadata_kwargs: Any,
    ) -> Artifact:
        """Create a new artifact.

        If definition_id is provided, applies definition defaults.

        Args:
            content: The artifact content
            definition_id: Optional reference to artifact definition
            artifact_type: Artifact type (if no definition)
            name: Human-readable name
            content_type: MIME type
            stage: Wizard stage creating this
            task_id: Task creating this
            purpose: Why this artifact exists
            source_ids: IDs of source artifacts
            tags: Categorization tags
            **metadata_kwargs: Additional metadata

        Returns:
            Created Artifact
        """
        # Apply definition defaults
        definition = self._definitions.get(definition_id) if definition_id else None
        if definition:
            artifact_type = definition.type
            name = name or definition.name
            tags = list(set((tags or []) + definition.tags))

        # Build metadata
        final_tags = list(set(tags or []))
        metadata = ArtifactMetadata(
            stage=stage,
            task_id=task_id,
            purpose=purpose,
            tags=final_tags,
            custom=metadata_kwargs,
        )

        # Build lineage
        lineage = ArtifactLineage(
            source_ids=source_ids or [],
            version=1,
        )

        # Create artifact
        artifact = Artifact(
            type=artifact_type,
            name=name,
            content=content,
            content_type=content_type,
            status="draft",
            schema_id=definition.schema_ref if definition else None,
            metadata=metadata,
            lineage=lineage,
            definition_id=definition_id,
        )

        self._artifacts[artifact.id] = artifact
        self._trigger_hooks("on_create", artifact)

        # Auto-submit for review if configured
        if definition and definition.auto_submit_for_review:
            self.submit_for_review(artifact.id)

        logger.info(
            "Created artifact",
            extra={
                "artifact_id": artifact.id,
                "type": artifact.type,
                "definition_id": definition_id,
            },
        )

        return artifact

    # =========================================================================
    # Artifact Updates
    # =========================================================================

    def update(
        self,
        artifact_id: str,
        content: Any,
        derived_from: str | None = None,
    ) -> Artifact:
        """Create a new version of an artifact.

        The original artifact is marked as superseded.
        A new artifact is created with incremented version.

        Args:
            artifact_id: ID of artifact to update
            content: New content
            derived_from: Description of changes

        Returns:
            New Artifact (new version)

        Raises:
            KeyError: If artifact_id not found
        """
        original = self._artifacts.get(artifact_id)
        if not original:
            raise KeyError(f"Artifact not found: {artifact_id}")

        # Mark original as superseded
        original.status = "superseded"

        # Create new version
        new_artifact = Artifact(
            type=original.type,
            name=original.name,
            content=content,
            content_type=original.content_type,
            status="draft",
            schema_id=original.schema_id,
            metadata=ArtifactMetadata(
                created_by=original.metadata.created_by,
                stage=original.metadata.stage,
                task_id=original.metadata.task_id,
                purpose=original.metadata.purpose,
                tags=original.metadata.tags.copy(),
                custom=original.metadata.custom.copy(),
            ),
            lineage=ArtifactLineage(
                parent_id=original.id,
                source_ids=original.lineage.source_ids.copy(),
                version=original.lineage.version + 1,
                derived_from=derived_from,
            ),
            definition_id=original.definition_id,
        )

        self._artifacts[new_artifact.id] = new_artifact
        self._trigger_hooks("on_update", new_artifact, original)

        logger.info(
            "Updated artifact",
            extra={
                "artifact_id": new_artifact.id,
                "original_id": original.id,
                "version": new_artifact.lineage.version,
            },
        )

        return new_artifact

    # =========================================================================
    # Status Transitions
    # =========================================================================

    def submit_for_review(self, artifact_id: str) -> Artifact:
        """Submit artifact for review.

        Args:
            artifact_id: ID of artifact to submit

        Returns:
            Updated Artifact

        Raises:
            KeyError: If artifact_id not found
            ValueError: If artifact not in reviewable state
        """
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            raise KeyError(f"Artifact not found: {artifact_id}")
        if not artifact.is_reviewable:
            raise ValueError(f"Artifact not reviewable: status={artifact.status}")

        old_status = artifact.status
        artifact.status = "pending_review"
        self._trigger_hooks("on_status_change", artifact, old_status, "pending_review")

        logger.debug(
            "Submitted artifact for review",
            extra={"artifact_id": artifact_id, "old_status": old_status},
        )

        return artifact

    def set_status(
        self,
        artifact_id: str,
        status: ArtifactStatus,
    ) -> Artifact:
        """Set artifact status directly.

        Args:
            artifact_id: ID of artifact
            status: New status

        Returns:
            Updated Artifact

        Raises:
            KeyError: If artifact_id not found
        """
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            raise KeyError(f"Artifact not found: {artifact_id}")

        old_status = artifact.status
        artifact.status = status
        self._trigger_hooks("on_status_change", artifact, old_status, status)

        logger.debug(
            "Changed artifact status",
            extra={
                "artifact_id": artifact_id,
                "old_status": old_status,
                "new_status": status,
            },
        )

        return artifact

    # =========================================================================
    # Review Management
    # =========================================================================

    def add_review(
        self,
        artifact_id: str,
        review: ArtifactReview,
    ) -> Artifact:
        """Add a review to an artifact.

        Updates artifact status based on review result and definition thresholds.

        Args:
            artifact_id: ID of artifact
            review: Review to add

        Returns:
            Updated Artifact

        Raises:
            KeyError: If artifact_id not found
        """
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            raise KeyError(f"Artifact not found: {artifact_id}")

        review.artifact_id = artifact_id
        artifact.reviews.append(review)
        self._trigger_hooks("on_review", artifact, review)

        # Check if we should update status based on definition
        definition = (
            self._definitions.get(artifact.definition_id)
            if artifact.definition_id
            else None
        )
        if definition:
            self._evaluate_approval(artifact, definition)

        logger.debug(
            "Added review to artifact",
            extra={
                "artifact_id": artifact_id,
                "reviewer": review.reviewer,
                "passed": review.passed,
            },
        )

        return artifact

    def _evaluate_approval(
        self,
        artifact: Artifact,
        definition: ArtifactDefinition,
    ) -> None:
        """Evaluate if artifact should be approved based on reviews.

        Args:
            artifact: Artifact to evaluate
            definition: Definition with review requirements
        """
        required_reviews = set(definition.reviews)
        completed_reviews = {r.reviewer for r in artifact.reviews}

        # Check if all required reviews are done
        if not required_reviews.issubset(completed_reviews):
            return  # Not all reviews completed yet

        # Get relevant reviews (most recent per reviewer)
        latest_reviews: dict[str, ArtifactReview] = {}
        for review in artifact.reviews:
            if review.reviewer in required_reviews:
                latest_reviews[review.reviewer] = review

        # Check if all passed or meets threshold
        if definition.require_all_reviews:
            all_passed = all(r.passed for r in latest_reviews.values())
            if all_passed:
                artifact.status = "approved"
            else:
                artifact.status = "needs_revision"
        else:
            # Check average score against threshold
            scores = [
                r.score for r in latest_reviews.values() if r.score is not None
            ]
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score >= definition.approval_threshold:
                    artifact.status = "approved"
                else:
                    artifact.status = "needs_revision"

    # =========================================================================
    # Queries
    # =========================================================================

    def get(self, artifact_id: str) -> Artifact | None:
        """Get artifact by ID.

        Args:
            artifact_id: ID to look up

        Returns:
            Artifact if found, None otherwise
        """
        return self._artifacts.get(artifact_id)

    def get_by_definition(self, definition_id: str) -> list[Artifact]:
        """Get all artifacts for a definition.

        Args:
            definition_id: Definition ID to filter by

        Returns:
            List of matching artifacts
        """
        return [
            a for a in self._artifacts.values()
            if a.definition_id == definition_id
        ]

    def get_by_type(self, artifact_type: ArtifactType) -> list[Artifact]:
        """Get all artifacts of a type.

        Args:
            artifact_type: Type to filter by

        Returns:
            List of matching artifacts
        """
        return [a for a in self._artifacts.values() if a.type == artifact_type]

    def get_by_status(self, status: ArtifactStatus) -> list[Artifact]:
        """Get all artifacts with a status.

        Args:
            status: Status to filter by

        Returns:
            List of matching artifacts
        """
        return [a for a in self._artifacts.values() if a.status == status]

    def get_by_stage(self, stage: str) -> list[Artifact]:
        """Get all artifacts created in a stage.

        Args:
            stage: Wizard stage to filter by

        Returns:
            List of matching artifacts
        """
        return [
            a for a in self._artifacts.values()
            if a.metadata.stage == stage
        ]

    def get_pending_review(self) -> list[Artifact]:
        """Get artifacts pending review.

        Returns:
            List of artifacts with pending_review status
        """
        return self.get_by_status("pending_review")

    def get_approved(self) -> list[Artifact]:
        """Get approved artifacts.

        Returns:
            List of artifacts with approved status
        """
        return self.get_by_status("approved")

    def get_all(self) -> list[Artifact]:
        """Get all artifacts.

        Returns:
            List of all artifacts
        """
        return list(self._artifacts.values())

    # =========================================================================
    # Version Navigation
    # =========================================================================

    def get_latest_version(self, artifact_id: str) -> Artifact | None:
        """Get the latest version of an artifact.

        Follows the version chain to find the most recent.

        Args:
            artifact_id: ID of any version of the artifact

        Returns:
            Latest version artifact, or None if not found
        """
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return None

        # Find any artifact that has this as parent
        while True:
            child = next(
                (a for a in self._artifacts.values()
                 if a.lineage.parent_id == artifact.id),
                None
            )
            if child:
                artifact = child
            else:
                return artifact

    def get_version_history(self, artifact_id: str) -> list[Artifact]:
        """Get full version history for an artifact.

        Args:
            artifact_id: ID of any version of the artifact

        Returns:
            List of all versions ordered by version number
        """
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return []

        # Walk back to find root
        root = artifact
        while root.lineage.parent_id:
            parent = self._artifacts.get(root.lineage.parent_id)
            if parent:
                root = parent
            else:
                break

        # Walk forward to collect all versions
        history = [root]
        current = root
        while True:
            child = next(
                (a for a in self._artifacts.values()
                 if a.lineage.parent_id == current.id),
                None
            )
            if child:
                history.append(child)
                current = child
            else:
                break

        return history

    # =========================================================================
    # Lifecycle Hooks
    # =========================================================================

    def on(self, event: str, callback: LifecycleHook) -> None:
        """Register a lifecycle hook.

        Supported events:
        - on_create: Called when artifact is created (artifact)
        - on_update: Called when artifact is updated (new_artifact, original)
        - on_status_change: Called on status change (artifact, old_status, new_status)
        - on_review: Called when review is added (artifact, review)

        Args:
            event: Event name
            callback: Callback function

        Raises:
            ValueError: If event name not recognized
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown event: {event}. Valid events: {list(self._hooks.keys())}")
        self._hooks[event].append(callback)

    def _trigger_hooks(self, event: str, *args: Any) -> None:
        """Trigger hooks for an event.

        Args:
            event: Event name
            *args: Arguments to pass to callbacks
        """
        for callback in self._hooks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                logger.warning("Hook error for %s: %s", event, e)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry state to dictionary.

        Returns:
            Dictionary representation of all artifacts
        """
        return {
            "artifacts": {
                aid: a.to_dict() for aid, a in self._artifacts.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        definitions: dict[str, ArtifactDefinition] | None = None,
    ) -> ArtifactRegistry:
        """Restore registry from dictionary.

        Args:
            data: Serialized registry data
            definitions: Optional artifact definitions

        Returns:
            ArtifactRegistry instance
        """
        registry = cls(definitions=definitions)
        for aid, adata in data.get("artifacts", {}).items():
            registry._artifacts[aid] = Artifact.from_dict(adata)
        return registry

    def clear(self) -> None:
        """Clear all artifacts from the registry."""
        self._artifacts.clear()
