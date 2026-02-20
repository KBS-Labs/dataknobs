"""Artifact lifecycle transforms for wizard workflows.

Pre-built async transform functions for creating, reviewing, revising,
and approving artifacts within wizard configurations. Each transform
operates on a wizard data dict and uses a TransformContext for access
to the artifact registry and rubric system.

Example:
    >>> from dataknobs_data.backends.memory import AsyncMemoryDatabase
    >>> from dataknobs_bots.artifacts.registry import ArtifactRegistry
    >>> context = TransformContext(
    ...     artifact_registry=ArtifactRegistry(AsyncMemoryDatabase()),
    ... )
    >>> data = {"title": "My Quiz", "questions": [...]}
    >>> await create_artifact(data, context, config={
    ...     "artifact_type": "quiz",
    ...     "name_template": "Quiz: {{ title }}",
    ...     "content_fields": ["questions"],
    ... })
    >>> data["_artifact_id"]  # Set by the transform
    'art_...'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .corpus import ArtifactCorpus, CorpusConfig
from .models import ArtifactStatus
from .provenance import create_provenance

logger = logging.getLogger(__name__)


@dataclass
class TransformContext:
    """Context available to transform functions during wizard execution.

    Provides access to the artifact registry, rubric system, and generator
    registry without requiring transforms to have direct dependencies on
    wizard infrastructure.

    Attributes:
        artifact_registry: Registry for artifact CRUD operations.
        rubric_registry: Registry for looking up rubric definitions.
        rubric_executor: Executor for running rubric evaluations.
        generator_registry: Registry for content generation.
        config: Additional configuration for transforms.
        user_id: Identifier of the current user.
        session_id: Identifier of the current session.
    """

    artifact_registry: Any | None = None
    rubric_registry: Any | None = None
    rubric_executor: Any | None = None
    generator_registry: Any | None = None
    config: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None
    session_id: str | None = None


async def create_artifact(
    data: dict[str, Any],
    context: TransformContext,
    config: dict[str, Any] | None = None,
) -> None:
    """Create an artifact from wizard data.

    Reads configuration to determine artifact type, name, and which fields
    from ``data`` to include in content. Sets ``data["_artifact_id"]`` for
    downstream stages.

    Config keys:
        artifact_type (str): Artifact type identifier (default: "content").
        name_template (str): Jinja2 template for artifact name, rendered
            with ``data`` as context. Falls back to ``name_field``.
        name_field (str): Key in ``data`` to use as name (default: "name").
        content_fields (list[str]): Keys to extract from ``data`` for content.
            If not specified, entire ``data`` (excluding ``_``-prefixed keys)
            is used.
        tags (list[str]): Tags to apply to the artifact.

    Args:
        data: Wizard data dictionary (modified in place).
        context: Transform context with artifact registry.
        config: Transform-specific configuration.

    Raises:
        ValueError: If artifact_registry is not configured.
    """
    if context.artifact_registry is None:
        raise ValueError("artifact_registry is required for create_artifact")

    cfg = config or context.config.get("create_artifact", {})
    artifact_type = cfg.get("artifact_type", "content")
    tags = cfg.get("tags", [])

    # Build name
    name = _resolve_name(data, cfg)

    # Build content
    content_fields = cfg.get("content_fields")
    if content_fields:
        content = {k: data[k] for k in content_fields if k in data}
    else:
        content = {k: v for k, v in data.items() if not k.startswith("_")}

    # Build provenance
    created_by = f"user:{context.user_id}" if context.user_id else "system:wizard"
    provenance = create_provenance(
        created_by=created_by,
        creation_method="wizard",
        creation_context={
            "session_id": context.session_id,
            "transform": "create_artifact",
        },
    )

    artifact = await context.artifact_registry.create(
        artifact_type=artifact_type,
        name=name,
        content=content,
        provenance=provenance,
        tags=tags,
    )

    data["_artifact_id"] = artifact.id
    logger.info(
        "Transform create_artifact: created '%s' (id=%s, type=%s)",
        name,
        artifact.id,
        artifact_type,
    )


async def submit_for_review(
    data: dict[str, Any],
    context: TransformContext,
    config: dict[str, Any] | None = None,
) -> None:
    """Submit an artifact for rubric-based review.

    Gets the artifact ID from ``data`` and submits it for evaluation.
    Sets ``data["_evaluation_results"]`` with evaluation results and
    ``data["_review_passed"]`` with a boolean pass/fail.

    Config keys:
        artifact_id_field (str): Key in ``data`` holding the artifact ID
            (default: "_artifact_id").

    Args:
        data: Wizard data dictionary (modified in place).
        context: Transform context with artifact registry.
        config: Transform-specific configuration.

    Raises:
        ValueError: If artifact_registry is not configured or artifact ID
            is missing.
    """
    if context.artifact_registry is None:
        raise ValueError("artifact_registry is required for submit_for_review")

    cfg = config or context.config.get("submit_for_review", {})
    id_field = cfg.get("artifact_id_field", "_artifact_id")
    artifact_id = data.get(id_field)

    if not artifact_id:
        raise ValueError(f"No artifact ID found in data['{id_field}']")

    evaluations = await context.artifact_registry.submit_for_review(artifact_id)

    data["_evaluation_results"] = evaluations
    data["_review_passed"] = all(
        e.get("passed", False) for e in evaluations
    ) if evaluations else True

    logger.info(
        "Transform submit_for_review: artifact '%s' review_passed=%s",
        artifact_id,
        data["_review_passed"],
    )


async def revise_artifact(
    data: dict[str, Any],
    context: TransformContext,
    config: dict[str, Any] | None = None,
) -> None:
    """Create a new version of an artifact with revised content.

    Gets the artifact ID from ``data``, extracts revision fields, and
    creates a new version. Updates ``data["_artifact_id"]`` to the new
    artifact ID (same ID, new version).

    Config keys:
        artifact_id_field (str): Key in ``data`` holding the artifact ID
            (default: "_artifact_id").
        content_fields (list[str]): Keys to extract from ``data`` for
            revised content. If not specified, entire ``data`` (excluding
            ``_``-prefixed keys) is used.
        reason_field (str): Key in ``data`` holding the revision reason
            (default: "_revision_reason").

    Args:
        data: Wizard data dictionary (modified in place).
        context: Transform context with artifact registry.
        config: Transform-specific configuration.

    Raises:
        ValueError: If artifact_registry is not configured or artifact ID
            is missing.
    """
    if context.artifact_registry is None:
        raise ValueError("artifact_registry is required for revise_artifact")

    cfg = config or context.config.get("revise_artifact", {})
    id_field = cfg.get("artifact_id_field", "_artifact_id")
    artifact_id = data.get(id_field)

    if not artifact_id:
        raise ValueError(f"No artifact ID found in data['{id_field}']")

    # Build revised content
    content_fields = cfg.get("content_fields")
    if content_fields:
        new_content = {k: data[k] for k in content_fields if k in data}
    else:
        new_content = {k: v for k, v in data.items() if not k.startswith("_")}

    reason = data.get(cfg.get("reason_field", "_revision_reason"), "Revised via wizard")
    triggered_by = f"user:{context.user_id}" if context.user_id else "system:wizard"

    new_artifact = await context.artifact_registry.revise(
        artifact_id=artifact_id,
        new_content=new_content,
        reason=reason,
        triggered_by=triggered_by,
    )

    data["_artifact_id"] = new_artifact.id

    logger.info(
        "Transform revise_artifact: revised '%s' to v%s",
        artifact_id,
        new_artifact.version,
    )


async def approve_artifact(
    data: dict[str, Any],
    context: TransformContext,
    config: dict[str, Any] | None = None,
) -> None:
    """Set an artifact's status to approved.

    Config keys:
        artifact_id_field (str): Key in ``data`` holding the artifact ID
            (default: "_artifact_id").

    Args:
        data: Wizard data dictionary (modified in place).
        context: Transform context with artifact registry.
        config: Transform-specific configuration.

    Raises:
        ValueError: If artifact_registry is not configured or artifact ID
            is missing.
    """
    if context.artifact_registry is None:
        raise ValueError("artifact_registry is required for approve_artifact")

    cfg = config or context.config.get("approve_artifact", {})
    id_field = cfg.get("artifact_id_field", "_artifact_id")
    artifact_id = data.get(id_field)

    if not artifact_id:
        raise ValueError(f"No artifact ID found in data['{id_field}']")

    await context.artifact_registry.set_status(
        artifact_id, ArtifactStatus.APPROVED, reason="Approved via wizard"
    )

    logger.info("Transform approve_artifact: approved '%s'", artifact_id)


async def save_artifact_draft(
    data: dict[str, Any],
    context: TransformContext,
    config: dict[str, Any] | None = None,
) -> None:
    """Incrementally save wizard data as a draft artifact.

    If an artifact already exists (``data["_artifact_id"]`` is set),
    revises it with updated content. Otherwise creates a new draft.

    This is designed to run on every transition for data safety.

    Config keys:
        artifact_type (str): Artifact type for new artifacts
            (default: "content").
        name_template (str): Jinja2 template for artifact name.
        name_field (str): Key in ``data`` to use as name (default: "name").
        content_fields (list[str]): Keys to extract from ``data``.
        tags (list[str]): Tags for new artifacts.

    Args:
        data: Wizard data dictionary (modified in place).
        context: Transform context with artifact registry.
        config: Transform-specific configuration.
    """
    if context.artifact_registry is None:
        logger.debug("save_artifact_draft: no artifact_registry, skipping")
        return

    artifact_id = data.get("_artifact_id")

    if artifact_id:
        # Update existing artifact
        await revise_artifact(data, context, config)
    else:
        # Create new draft
        await create_artifact(data, context, config)


async def create_corpus(
    data: dict[str, Any],
    context: TransformContext,
    config: dict[str, Any] | None = None,
) -> None:
    """Create a new ArtifactCorpus and store its ID in wizard data.

    Config keys:
        corpus_type (str): Artifact type for the corpus (default: "corpus").
        item_type (str): Artifact type for items (default: "item").
        name_template (str): Jinja2 template for corpus name.
        name_field (str): Key in ``data`` to use as name (default: "name").
        rubric_ids (list[str]): Rubrics to apply to items (default: []).
        auto_review (bool): Auto-evaluate items on add (default: False).
        dedup (dict): DedupConfig settings (optional). Keys:
            hash_fields, hash_algorithm, semantic_check, similarity_threshold.

    Args:
        data: Wizard data dictionary (modified in place).
        context: Transform context with artifact registry.
        config: Transform-specific configuration.

    Raises:
        ValueError: If artifact_registry is not configured.
    """
    if context.artifact_registry is None:
        raise ValueError("create_corpus requires artifact_registry in context")

    cfg = config or {}
    name = _resolve_name(data, cfg)
    corpus_type = cfg.get("corpus_type", "corpus")
    item_type = cfg.get("item_type", "item")

    # Build dedup checker if configured
    dedup_checker = None
    dedup_config_dict = cfg.get("dedup")
    if dedup_config_dict:
        from dataknobs_data.backends.memory import AsyncMemoryDatabase
        from dataknobs_data.dedup import DedupChecker, DedupConfig

        dedup_cfg = DedupConfig(
            hash_fields=dedup_config_dict.get("hash_fields", ["content"]),
            hash_algorithm=dedup_config_dict.get("hash_algorithm", "md5"),
            semantic_check=dedup_config_dict.get("semantic_check", False),
            similarity_threshold=dedup_config_dict.get(
                "similarity_threshold", 0.92
            ),
        )
        dedup_db = AsyncMemoryDatabase()
        dedup_checker = DedupChecker(db=dedup_db, config=dedup_cfg)

    corpus_config = CorpusConfig(
        corpus_type=corpus_type,
        item_type=item_type,
        name=name,
        rubric_ids=cfg.get("rubric_ids", []),
        auto_review=cfg.get("auto_review", False),
    )

    corpus = await ArtifactCorpus.create(
        registry=context.artifact_registry,
        config=corpus_config,
        dedup_checker=dedup_checker,
    )

    data["_corpus_id"] = corpus.id
    data["_corpus"] = corpus
    data["_corpus_item_count"] = 0
    logger.info(
        "Transform create_corpus: created '%s' (id=%s, type=%s)",
        name,
        corpus.id,
        corpus_type,
    )


async def add_to_corpus(
    data: dict[str, Any],
    context: TransformContext,
    config: dict[str, Any] | None = None,
) -> None:
    """Add content from wizard data to the corpus.

    Config keys:
        content_key (str): Key in ``data`` holding the content dict to add
            (default: "_current_item").
        corpus_key (str): Key in ``data`` holding the ArtifactCorpus instance
            (default: "_corpus").
        tags (list[str]): Additional tags for the item (default: []).

    Args:
        data: Wizard data dictionary (modified in place).
        context: Transform context with artifact registry.
        config: Transform-specific configuration.

    Raises:
        ValueError: If no corpus or content is found in data.
    """
    cfg = config or {}
    content_key = cfg.get("content_key", "_current_item")
    corpus_key = cfg.get("corpus_key", "_corpus")
    extra_tags = cfg.get("tags", [])

    corpus: ArtifactCorpus | None = data.get(corpus_key)
    if corpus is None:
        corpus_id = data.get("_corpus_id")
        if corpus_id and context.artifact_registry:
            corpus = await ArtifactCorpus.load(
                registry=context.artifact_registry,
                corpus_id=corpus_id,
            )
            data[corpus_key] = corpus
        else:
            raise ValueError(
                f"No corpus found at data['{corpus_key}'] "
                "and no _corpus_id to load from"
            )

    content = data.get(content_key)
    if content is None:
        raise ValueError(f"No content found at data['{content_key}']")

    artifact, dedup_result = await corpus.add_item(
        content=content,
        tags=extra_tags,
    )

    data["_last_added_artifact_id"] = artifact.id
    data["_corpus_item_count"] = await corpus.count()
    if dedup_result:
        data["_dedup_result"] = {
            "is_exact_duplicate": dedup_result.is_exact_duplicate,
            "recommendation": dedup_result.recommendation,
            "exact_match_id": dedup_result.exact_match_id,
            "similar_count": len(dedup_result.similar_items),
        }
    else:
        data["_dedup_result"] = None

    logger.info(
        "Transform add_to_corpus: added artifact '%s' to corpus '%s' (count=%d)",
        artifact.id,
        corpus.id,
        data["_corpus_item_count"],
    )


async def finalize_corpus(
    data: dict[str, Any],
    context: TransformContext,
    config: dict[str, Any] | None = None,
) -> None:
    """Finalize the corpus, approving the parent artifact.

    Config keys:
        corpus_key (str): Key in ``data`` holding the ArtifactCorpus instance
            (default: "_corpus").

    Args:
        data: Wizard data dictionary (modified in place).
        context: Transform context with artifact registry.
        config: Transform-specific configuration.

    Raises:
        ValueError: If no corpus is found in data.
    """
    cfg = config or {}
    corpus_key = cfg.get("corpus_key", "_corpus")

    corpus: ArtifactCorpus | None = data.get(corpus_key)
    if corpus is None:
        corpus_id = data.get("_corpus_id")
        if corpus_id and context.artifact_registry:
            corpus = await ArtifactCorpus.load(
                registry=context.artifact_registry,
                corpus_id=corpus_id,
            )
        else:
            raise ValueError("No corpus found to finalize")

    await corpus.finalize()
    data["_corpus_summary"] = await corpus.get_summary()
    logger.info(
        "Transform finalize_corpus: finalized corpus '%s'",
        corpus.id,
    )


def _resolve_name(data: dict[str, Any], config: dict[str, Any]) -> str:
    """Resolve artifact name from config template or field.

    Args:
        data: Wizard data dictionary.
        config: Transform configuration.

    Returns:
        Resolved artifact name.
    """
    name_template = config.get("name_template")
    if name_template:
        import jinja2

        try:
            template = jinja2.Template(name_template)
            return template.render(**data)
        except jinja2.UndefinedError:
            logger.warning(
                "Name template rendering failed, falling back to name_field"
            )

    name_field = config.get("name_field", "name")
    return str(data.get(name_field, "Untitled"))
