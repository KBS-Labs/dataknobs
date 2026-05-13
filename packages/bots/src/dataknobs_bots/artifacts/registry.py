"""Artifact registry for managing artifacts with async database backing.

The registry provides:
- Artifact creation with provenance tracking
- Versioning (creating new versions preserves history)
- Status transitions enforced by TransitionValidator
- Rubric-based evaluation via submit_for_review
- Querying by type, status, tags

Internally composes :class:`AsyncKeyedRecordStore[Artifact]` so every
``Record`` is built in one place (the store's serializer) and the
``metadata`` channel is preserved by construction.  The serializer
signature ``(Artifact) -> (data, metadata)`` makes the metadata column
part of the function's type — a future change to the model cannot
silently drop the metadata channel without a type-visible diff.

Example:
    >>> from dataknobs_data.backends.memory import AsyncMemoryDatabase
    >>> db = AsyncMemoryDatabase()
    >>> registry = ArtifactRegistry(db)
    >>> artifact = await registry.create(
    ...     artifact_type="content",
    ...     name="Questions",
    ...     content={"questions": [...]},
    ...     provenance=create_provenance("bot:edubot", "generator"),
    ...     metadata={"tenant_id": "acme"},
    ... )
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from dataknobs_data import (
    AsyncDatabase,
    AsyncKeyedRecordStore,
    Filter,
    Operator,
    Query,
    Record,
    SortSpec,
)

from ..utils.versioned_records import iter_latest_records
from .models import Artifact, ArtifactStatus, ArtifactTypeDefinition, _generate_artifact_id, _now_iso
from .provenance import ProvenanceRecord, RevisionRecord, create_provenance
from .transitions import validate_transition

logger = logging.getLogger(__name__)

# Type alias for lifecycle hook callbacks
LifecycleHook = Callable[[Artifact], Awaitable[None]]


def _artifact_to_columns(
    artifact: Artifact,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split an :class:`Artifact` into the ``(data, metadata)`` channels.

    The two-channel return type is load-bearing: the metadata column is
    part of the function's *signature*, so a future change to the
    Artifact model can't accidentally drop the metadata channel without
    a type-visible diff at this site.

    ``_version_key`` is a storage-layer field used by :meth:`query` to
    distinguish "latest pointer" records (stored at ``artifact.id``)
    from versioned snapshots (stored at ``f"{artifact.id}:{version}"``).
    It is always ``f"{artifact.id}:{artifact.version}"`` regardless of
    which key the record is written under, so the deduplication test
    ``storage_id == _version_key`` correctly identifies snapshots.
    """
    data = artifact.to_dict()
    metadata = dict(data.pop("metadata", None) or {})
    data["_version_key"] = f"{artifact.id}:{artifact.version}"
    return data, metadata


def _artifact_from_record(record: Record) -> Artifact:
    """Reconstruct an :class:`Artifact` from a stored ``Record``.

    Strips the storage-layer ``_version_key`` field before deserializing
    (it is not part of the model surface) and injects ``record.metadata``
    into the ``metadata`` slot of the artifact dict so the model field
    is populated.
    """
    data = dict(record.data or {})
    data.pop("_version_key", None)
    data["metadata"] = dict(record.metadata or {})
    return Artifact.from_dict(data)


class ArtifactRegistry:
    """Async artifact registry backed by AsyncDatabase.

    Manages artifact lifecycle including creation, versioning, status
    transitions, and rubric-based evaluation.

    Internally composes :class:`AsyncKeyedRecordStore[Artifact]` so
    every ``Record`` is constructed in one place (the store's
    serializer) and the metadata column is preserved by construction.

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
        self._store: AsyncKeyedRecordStore[Artifact] = AsyncKeyedRecordStore[
            Artifact
        ](
            db,
            serializer=_artifact_to_columns,
            deserializer=_artifact_from_record,
        )
        self._rubric_registry = rubric_registry
        self._rubric_executor = rubric_executor
        self._type_definitions = type_definitions or {}
        self._on_create_hooks: list[LifecycleHook] = []
        self._on_status_change_hooks: list[LifecycleHook] = []
        self._on_review_complete_hooks: list[LifecycleHook] = []
        # Per-artifact-id locks serialize read-modify-write paths
        # (revise, set_status, submit_for_review) in-process so two
        # concurrent callers on the same id don't interleave a stale
        # read with each other's write.  Locks are created lazily and
        # never released, so the dict grows monotonically with the
        # number of distinct artifact ids touched — acceptable for
        # typical population sizes; consumers writing millions of
        # distinct ids per process should consider an eviction policy.
        # **Scope:** in-process only.  Two processes (e.g. two app
        # servers) writing to the same backing database still race.
        # The full fix is an optimistic-version / row-lock check at
        # the database layer, which is a separate cross-cutting work
        # item.
        self._entity_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _entity_lock(self, artifact_id: str) -> asyncio.Lock:
        """Return the per-artifact lock, creating it on first use."""
        async with self._locks_guard:
            lock = self._entity_locks.get(artifact_id)
            if lock is None:
                lock = asyncio.Lock()
                self._entity_locks[artifact_id] = lock
            return lock

    # --- Creation ---

    async def create(
        self,
        artifact_type: str,
        name: str,
        content: dict[str, Any],
        provenance: ProvenanceRecord | None = None,
        tags: list[str] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Artifact:
        """Create a new artifact.

        Args:
            artifact_type: The type of artifact to create.
            name: Human-readable name.
            content: The artifact content.
            provenance: Provenance record. If None, a minimal one is created.
            tags: Searchable tags. Merged with type definition defaults.
            metadata: Cross-cutting context (``tenant_id``,
                ``correlation_id``, audit info, feature flags). Routed
                to the underlying record's ``metadata`` column so it is
                independently filterable via ``filter_metadata`` on
                :meth:`query` without scanning every row.

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
            metadata=dict(metadata or {}),
        )

        await self._store_artifact(artifact)

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
        return await self._store.get(artifact_id)

    async def get_version(
        self, artifact_id: str, version: str
    ) -> Artifact | None:
        """Retrieve a specific version of an artifact."""
        return await self._store.get(f"{artifact_id}:{version}")

    async def query(
        self,
        artifact_type: str | None = None,
        status: ArtifactStatus | None = None,
        tags: list[str] | None = None,
        filters: list[Filter] | None = None,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
        sort: list[SortSpec] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Artifact]:
        """Query artifacts by type, status, tags, and/or arbitrary filters.

        Args:
            artifact_type: Filter by artifact type.
            status: Filter by status.
            tags: Filter by tags (artifact must have all specified tags).
            filters: Additional filters on any field, including nested content
                fields via dot notation
                (e.g., ``Filter("content.corpus_id", Operator.EQ, "abc")``).
            filter_metadata: Equality filter over the ``metadata`` column.
                Entries are AND-combined with structural filters and
                routed via the ``metadata.X`` field-path convention so
                SQL/JSONB backends push the filter into the indexable
                column instead of scanning every row.
            sort: Optional multi-key sort specification.  Pushed down to
                the database query so SQL backends can use indexes when
                available.  The first-occurrence-per-id deduplication
                pass preserves the database's ordering — pointer and
                snapshot rows for the same artifact carry identical
                sort-relevant fields, so whichever pointer comes first
                wins and the overall order is the database's order.
            limit: Optional row limit applied **after** dedup.  Not
                pushed to the database because the dual-write storage
                shape (latest-pointer + versioned snapshots) makes the
                pre-dedup row count diverge from the post-dedup
                artifact count: a database-level ``LIMIT 5`` could
                return 5 snapshot rows that all dedup away, leaving an
                empty result when artifacts exist further down.
            offset: Optional row offset applied **after** dedup; same
                reason as ``limit``.

        Returns:
            List of matching artifacts.
        """
        q = Query(
            filters=self._build_filters(
                artifact_type=artifact_type,
                status=status,
                filters=filters,
                filter_metadata=filter_metadata,
            )
        )
        if sort is not None:
            q.sort_specs = list(sort)

        records = await self._store.search(q)

        artifacts: list[Artifact] = []
        for record in iter_latest_records(records):
            # Tag filtering is structural (tags is a list field) and the
            # query system has no ARRAY_CONTAINS operator, so it must be
            # applied client-side.  Read directly from ``record.data``
            # to avoid deserializing artifacts that won't match.
            if tags:
                record_tags = record.data.get("tags") or []
                if not all(t in record_tags for t in tags):
                    continue
            artifacts.append(_artifact_from_record(record))

        # Apply pagination after dedup.  See docstring for why these
        # cannot be pushed to the database alongside the filters.
        if offset is not None:
            artifacts = artifacts[offset:]
        if limit is not None:
            artifacts = artifacts[:limit]

        return artifacts

    async def count(
        self,
        artifact_type: str | None = None,
        status: ArtifactStatus | None = None,
        tags: list[str] | None = None,
        filters: list[Filter] | None = None,
        *,
        filter_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Count artifacts matching the same filter shape as :meth:`query`.

        Mirrors :meth:`query` parameter-for-parameter (minus
        ``sort``/``limit``/``offset``, which don't affect the count)
        and is equivalent to ``len(await self.query(...))`` after
        dedup.  ``sort``/``limit``/``offset`` are intentionally omitted
        because none of them change the total count — they change
        which rows are returned, not how many match.

        Cost note:
            The storage shape (dual-write: latest pointer + versioned
            snapshot per version) means the database's row count is
            not the artifact count — pre-dedup it includes snapshots,
            post-dedup it does not.  A pushdown-only count is
            therefore not safe.  This implementation runs the same
            search as :meth:`query`, applies the dedup pass, and
            returns the count.  Performance scales with the matching
            row count, not the count value itself.  Backends that ship
            ``SELECT COUNT(*) WHERE ...`` pushdown do NOT benefit
            until the storage shape is changed to mark "is latest"
            on the row itself — out of scope here.

        Args:
            artifact_type: Filter by artifact type.
            status: Filter by status.
            tags: Filter by tags (artifact must have all specified tags).
            filters: Additional filters on any field.
            filter_metadata: Equality filter over the ``metadata`` column.

        Returns:
            Number of matching artifacts (deduplicated to one per
            artifact ID).
        """
        records = await self._store.search(
            Query(
                filters=self._build_filters(
                    artifact_type=artifact_type,
                    status=status,
                    filters=filters,
                    filter_metadata=filter_metadata,
                )
            )
        )

        count = 0
        for record in iter_latest_records(records):
            # Tag filter is structural; see :meth:`query` for the same
            # client-side handling.  Read from ``record.data`` to avoid
            # the cost of deserializing artifacts in count paths.
            if tags:
                record_tags = record.data.get("tags") or []
                if not all(t in record_tags for t in tags):
                    continue
            count += 1
        return count

    # --- Updates ---

    async def revise(
        self,
        artifact_id: str,
        new_content: dict[str, Any],
        reason: str,
        triggered_by: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Artifact:
        """Create a new version of an artifact with revised content.

        The old version is marked as superseded. The new version inherits
        provenance and adds a revision record.

        Args:
            artifact_id: The artifact to revise.
            new_content: The updated content.
            reason: Why the revision was made.
            triggered_by: Who or what triggered the revision.
            metadata: Optional metadata to attach to the new version.
                If ``None``, the current version's metadata is inherited.

        Returns:
            The new artifact version.

        Raises:
            ValueError: If the artifact is not found.

        Note:
            Holds the per-artifact lock for the full read-bump-write
            flow so two concurrent ``revise`` calls on the same id
            cannot both compute the same ``new_version`` from the
            same ``current``.  In-process safety only — see class
            docstring on ``_entity_locks`` for the multi-process
            caveat.
        """
        lock = await self._entity_lock(artifact_id)
        async with lock:
            current = await self.get(artifact_id)
            if current is None:
                raise ValueError(f"Artifact '{artifact_id}' not found")

            # Bump version
            parts = current.version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            new_version = ".".join(parts)

            # Mark old version as superseded (we already hold the lock)
            await self._set_status_unlocked(
                artifact_id,
                ArtifactStatus.SUPERSEDED,
                reason="Superseded by revision",
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
                metadata=dict(metadata if metadata is not None else current.metadata),
            )

            await self._store_artifact(new_artifact)

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
        lock = await self._entity_lock(artifact_id)
        async with lock:
            await self._set_status_unlocked(artifact_id, status, reason)

    async def _set_status_unlocked(
        self,
        artifact_id: str,
        status: ArtifactStatus,
        reason: str | None = None,
    ) -> None:
        """Body of :meth:`set_status` that assumes the entity lock is held.

        Internal helper for :meth:`revise` and
        :meth:`submit_for_review`, which already hold the entity lock
        for their full read-modify-write flow and would deadlock if
        they re-acquired it via the public :meth:`set_status`.
        """
        artifact = await self.get(artifact_id)
        if artifact is None:
            raise ValueError(f"Artifact '{artifact_id}' not found")

        validate_transition(artifact.status, status)

        artifact.status = status
        artifact.updated_at = _now_iso()
        await self._store_artifact(artifact)

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

        Note:
            Holds the per-artifact lock for the full transition →
            evaluate → transition flow so a concurrent ``revise`` or
            ``set_status`` on the same artifact cannot interleave
            with the review.  The lock is held across rubric
            evaluation (which may include LLM calls), so other writes
            on the same artifact block until the review completes —
            this is the correct semantics for serializing
            review/revision on a single entity.  In-process safety
            only — see class docstring on ``_entity_locks``.
        """
        lock = await self._entity_lock(artifact_id)
        async with lock:
            artifact = await self.get(artifact_id)
            if artifact is None:
                raise ValueError(f"Artifact '{artifact_id}' not found")

            # Transition to pending_review then in_review
            if artifact.status == ArtifactStatus.DRAFT:
                await self._set_status_unlocked(
                    artifact_id, ArtifactStatus.PENDING_REVIEW
                )
                artifact = await self.get(artifact_id)
                if artifact is None:
                    raise ValueError(
                        f"Artifact '{artifact_id}' not found after status update"
                    )
            if artifact.status in (
                ArtifactStatus.PENDING_REVIEW,
                ArtifactStatus.NEEDS_REVISION,
            ):
                artifact = await self.get(artifact_id)
                if artifact is None:
                    raise ValueError(
                        f"Artifact '{artifact_id}' not found after status update"
                    )
                if artifact.status == ArtifactStatus.PENDING_REVIEW:
                    await self._set_status_unlocked(
                        artifact_id, ArtifactStatus.IN_REVIEW
                    )

            if not self._rubric_registry or not self._rubric_executor:
                logger.warning(
                    "Review requested for '%s' but no rubric registry/executor configured",
                    artifact_id,
                )
                return []

            # Re-read after status changes
            artifact = await self.get(artifact_id)
            if artifact is None:
                raise ValueError(
                    f"Artifact '{artifact_id}' not found after status update"
                )

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
            await self._store_artifact(artifact)

            # Transition based on results
            if evaluations:
                if all_passed:
                    await self._set_status_unlocked(
                        artifact_id, ArtifactStatus.APPROVED
                    )
                else:
                    await self._set_status_unlocked(
                        artifact_id, ArtifactStatus.NEEDS_REVISION
                    )

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

        Evaluation records use a separate key namespace (``eval:<id>``)
        and are not artifact records, so this method reads through the
        store's underlying database directly rather than the typed
        artifact surface.
        """
        artifact = await self.get(artifact_id)
        if artifact is None:
            return []

        evaluations: list[dict[str, Any]] = []
        for eval_id in artifact.evaluation_ids:
            record = await self._store.db.read(f"eval:{eval_id}")
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

    def _build_filters(
        self,
        *,
        artifact_type: str | None,
        status: ArtifactStatus | None,
        filters: list[Filter] | None,
        filter_metadata: Mapping[str, Any] | None,
    ) -> list[Filter]:
        """Compose the structural filter list shared by :meth:`query` and :meth:`count`.

        Routes ``filter_metadata`` entries through the ``metadata.X``
        field-path convention so SQL/JSONB backends can push them into
        the indexable metadata column.  Tag filtering is **not**
        included here — it is structural over a list field and the
        query system has no ARRAY_CONTAINS operator, so callers apply
        it client-side after the search.
        """
        all_filters: list[Filter] = []
        if artifact_type:
            all_filters.append(Filter("type", Operator.EQ, artifact_type))
        if status:
            all_filters.append(Filter("status", Operator.EQ, status.value))
        if filters:
            all_filters.extend(filters)
        for k, v in (filter_metadata or {}).items():
            all_filters.append(Filter(f"metadata.{k}", Operator.EQ, v))
        return all_filters

    async def _store_artifact(self, artifact: Artifact) -> None:
        """Persist an artifact via two writes: versioned snapshot + latest pointer.

        Both writes route through the keyed store so the metadata
        channel is preserved by construction (the serializer is the
        single ``Record`` build site).

        Write order: **snapshot first, then pointer**.  This matches
        :class:`RubricRegistry.register` and is the transactionally
        safer ordering — at any partial-write observation point the
        pointer either does not yet exist or references a snapshot
        that already does, never a missing snapshot.  Pointer-first
        would briefly let ``get_version(id, v)`` return ``None`` for
        the new version while ``get(id)`` shows it as latest.
        """
        await self._store.put(f"{artifact.id}:{artifact.version}", artifact)
        await self._store.put(artifact.id, artifact)
