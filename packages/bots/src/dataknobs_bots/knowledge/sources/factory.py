"""Config-driven construction of :class:`GroundedSource` instances.

Translates :class:`~dataknobs_bots.reasoning.grounded_config.GroundedSourceConfig`
declarations into concrete source objects, using ``dataknobs-data`` factories
for database backends and ``dataknobs-bots`` for vector KB sources.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dataknobs_data.sources.base import GroundedSource

from dataknobs_bots.reasoning.grounded_config import GroundedSourceConfig

if TYPE_CHECKING:
    from dataknobs_data.fields import FieldType
    from dataknobs_data.schema import DatabaseSchema

logger = logging.getLogger(__name__)


async def create_source_from_config(
    config: GroundedSourceConfig,
    *,
    knowledge_base: Any | None = None,
) -> GroundedSource:
    """Construct a :class:`GroundedSource` from a config declaration.

    Args:
        config: Source configuration specifying type, name, and options.
        knowledge_base: Pre-built :class:`KnowledgeBase` instance.
            Required when ``config.source_type == "vector_kb"``; ignored
            for other types.

    Returns:
        A fully initialized source ready for use in the grounded pipeline.

    Raises:
        ValueError: If the source type is unknown or required dependencies
            are missing.

    Supported source types:

    ``vector_kb``
        Wraps an existing :class:`KnowledgeBase` as a
        :class:`VectorKnowledgeSource`.  Requires ``knowledge_base``
        to be provided.

    ``database``
        Creates a :class:`DatabaseSource` from the config options.
        Required options: ``backend`` (database backend key),
        ``content_field``.  Optional: ``connection``,
        ``text_search_fields``, ``schema`` (field definitions).

    Example config (YAML)::

        sources:
          - type: vector_kb
            name: docs

          - type: database
            name: courses
            backend: sqlite
            connection: "courses.db"
            content_field: description
            text_search_fields: [title, description]
            schema:
              fields:
                department: {type: string, enum: [CS, Math, Physics]}
                level: {type: integer}
                description: {type: text}
    """
    source_type = config.source_type

    if source_type == "vector_kb":
        return _create_vector_kb_source(config, knowledge_base)

    if source_type == "database":
        return await _create_database_source(config)

    raise ValueError(
        f"Unknown grounded source type: {source_type!r}. "
        f"Supported types: vector_kb, database"
    )


def _create_vector_kb_source(
    config: GroundedSourceConfig,
    knowledge_base: Any | None,
) -> GroundedSource:
    """Wrap a KnowledgeBase as a VectorKnowledgeSource."""
    from dataknobs_bots.tools.resolve import resolve_optional_callable

    from .vector import VectorKnowledgeSource

    if knowledge_base is None:
        raise ValueError(
            f"Source {config.name!r} has type 'vector_kb' but no "
            f"knowledge_base was provided. Either configure a "
            f"knowledge_base in the bot config or use a different "
            f"source type."
        )

    # Resolve optional identity callables from dotted import paths.
    # Defaults are applied inside VectorKnowledgeSource and
    # _build_vector_query_fn when these are None.
    opts = config.options
    dedup_key = resolve_optional_callable(
        opts.get("dedup_key"),
        field_name="dedup_key",
        owner=config.name,
    )
    source_id_fn = resolve_optional_callable(
        opts.get("source_id_fn"),
        field_name="source_id_fn",
        owner=config.name,
    )
    metadata_fn = resolve_optional_callable(
        opts.get("metadata_fn"),
        field_name="metadata_fn",
        owner=config.name,
    )

    topic_index = build_topic_index(
        config.topic_index, knowledge_base,
        source_name=config.name,
        source_id_fn=source_id_fn,
        metadata_fn=metadata_fn,
    )
    return VectorKnowledgeSource(
        knowledge_base,
        name=config.name,
        topic_index=topic_index,
        dedup_key=dedup_key,
        source_id_fn=source_id_fn,
        metadata_fn=metadata_fn,
    )


def build_topic_index(
    topic_index_config: dict[str, Any] | None,
    kb: Any,
    *,
    source_name: str = "knowledge_base",
    source_id_fn: Callable[[dict[str, Any]], str] | None = None,
    metadata_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Any | None:
    """Build a topic index from config + KnowledgeBase.

    Both topic index types are constructed in lazy mode — they store
    query functions but build structures per-turn from seed results.

    This is the shared construction logic used by both the factory path
    (``create_source_from_config``) and the legacy path
    (``GroundedReasoning.set_knowledge_base``).

    Args:
        topic_index_config: The ``topic_index`` dict from source config.
            ``None`` means no topic index is configured.
        kb: KnowledgeBase instance for wiring query/embed functions.
        source_name: Source name for logging and provenance.
        source_id_fn: Optional identity callable forwarded to the
            underlying ``vector_query_fn`` so topic-index retrieval
            emits the same ``source_id`` as the main
            :class:`VectorKnowledgeSource.query` path.
        metadata_fn: Optional metadata callable forwarded to the
            underlying ``vector_query_fn`` so topic-index retrieval
            emits the same ``SourceResult.metadata`` as the main path.

    Returns:
        A ``HeadingTreeIndex`` or ``ClusterTopicIndex``, or ``None``
        if no topic index is configured.
    """
    if topic_index_config is None:
        return None

    index_type = topic_index_config.get("type", "heading_tree")

    # Build a vector_query_fn from the KB, threading the identity
    # callables so the topic-index path uses the same identity rule as
    # the main VectorKnowledgeSource.query path.
    vector_query_fn = _build_vector_query_fn(
        kb, source_name,
        source_id_fn=source_id_fn,
        metadata_fn=metadata_fn,
    )

    if index_type == "heading_tree":
        from dataknobs_bots.knowledge.sources.heading_tree import (
            HeadingTreeConfig,
            HeadingTreeIndex,
        )

        config = HeadingTreeConfig.from_dict(topic_index_config)

        # Build a dedicated heading-selection LLM if configured
        heading_selection_llm = _build_heading_selection_llm(config)

        idx = HeadingTreeIndex(
            vector_query_fn=vector_query_fn,
            source_name=source_name,
            config=config,
            heading_selection_llm=heading_selection_llm,
        )
        logger.info(
            "Built heading_tree topic index for source '%s' "
            "(entry_strategy=%s, expansion_mode=%s, "
            "heading_selection_llm=%s)",
            source_name, config.entry_strategy, config.expansion_mode,
            "dedicated" if heading_selection_llm else "main",
        )
        return idx

    if index_type == "cluster":
        from dataknobs_data.sources.cluster_index import (
            ClusterTopicConfig,
            ClusterTopicIndex,
        )

        embed_fn = _build_embed_fn(kb)
        config = ClusterTopicConfig.from_dict(topic_index_config)
        idx = ClusterTopicIndex(
            embed_fn=embed_fn,
            vector_query_fn=vector_query_fn,
            source_name=source_name,
            config=config,
        )
        logger.info(
            "Built cluster topic index for source '%s' "
            "(cluster_threshold=%.2f, top_clusters=%d)",
            source_name, config.cluster_threshold, config.top_clusters,
        )
        return idx

    logger.warning(
        "Unknown topic_index type %r for source '%s', skipping",
        index_type, source_name,
    )
    return None


def _build_vector_query_fn(
    kb: Any,
    source_name: str,
    *,
    source_id_fn: Callable[[dict[str, Any]], str] | None = None,
    metadata_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Any:
    """Build a vector_query_fn closure from a KnowledgeBase.

    Uses the same identity callables as :class:`VectorKnowledgeSource`
    when provided, falling back to the shared module-level defaults.
    Consumers who customize identity on the source get the same
    behavior on the topic-index path without extra configuration.
    """
    from dataknobs_data.sources.base import SourceResult

    from .vector import default_metadata, default_source_id

    sid_fn = source_id_fn or default_source_id
    md_fn = metadata_fn or default_metadata

    async def vector_query_fn(
        query: str,
        top_k: int,
        *,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SourceResult]:
        raw_results = await kb.query(
            query, k=top_k, filter_metadata=filter_metadata,
        )
        results: list[SourceResult] = []
        for r in raw_results:
            # Identity callables are consumer-supplied and may raise.
            # Mirror VectorKnowledgeSource.query: skip failing records
            # rather than aborting the entire topic-index lookup.
            try:
                source_id = sid_fn(r)
                metadata = md_fn(r)
            except Exception:
                logger.warning(
                    "Identity callable raised for a topic-index "
                    "result in source '%s' (query=%r); skipping record",
                    source_name, query, exc_info=True,
                )
                continue
            results.append(SourceResult(
                content=r.get("text", ""),
                source_id=source_id,
                source_name=source_name,
                source_type="vector_kb",
                relevance=r.get("similarity", 1.0),
                metadata=metadata,
            ))
        return results

    return vector_query_fn


def _build_embed_fn(kb: Any) -> Any | None:
    """Build an embed_fn from KB's embedding capability.

    Returns ``None`` if the KB doesn't support embedding.
    """
    if hasattr(kb, "embed"):
        async def embed_fn(text: str) -> list[float]:
            return await kb.embed(text)
        return embed_fn
    return None


def _build_heading_selection_llm(config: Any) -> Any | None:
    """Build a dedicated LLM provider for heading selection.

    When ``heading_selection_config`` is set on the config, creates a
    lightweight provider for the constrained heading classification task.
    This avoids using the main (often expensive/slow) thinking model for
    a simple selection task.

    Returns ``None`` if no dedicated config is set (falls back to main LLM).
    """
    hs_config = getattr(config, "heading_selection_config", None)
    if not hs_config:
        return None

    from dataknobs_llm import create_llm_provider

    try:
        provider = create_llm_provider(hs_config)
        logger.info(
            "Built dedicated heading selection LLM: provider=%s, model=%s",
            hs_config.get("provider"), hs_config.get("model"),
        )
        return provider
    except Exception:
        logger.warning(
            "Failed to build heading selection LLM from config %r, "
            "will fall back to main LLM",
            hs_config, exc_info=True,
        )
        return None


async def _create_database_source(
    config: GroundedSourceConfig,
) -> GroundedSource:
    """Create a DatabaseSource from config options.

    Expected options:
        backend: Database backend key (e.g. "memory", "sqlite").
        connection: Connection string (backend-specific).
        content_field: Field whose value becomes SourceResult.content.
        text_search_fields: Fields for LIKE text search.
        schema: Dict with "fields" mapping field names to type defs.
        description: Human-readable source description.
    """
    from dataknobs_data import async_database_factory
    from dataknobs_data.sources.database import DatabaseSource

    opts = config.options

    # Build the database backend
    backend = opts.get("backend", "memory")
    db_config: dict[str, Any] = {"backend": backend}

    connection = opts.get("connection")
    if connection:
        db_config["connection"] = connection

    db = async_database_factory.create(**db_config)

    # Build the schema from config
    schema_config = opts.get("schema", {})
    field_defs = schema_config.get("fields", {})
    schema = _build_database_schema(field_defs)

    # Set schema on the database if it supports it
    if hasattr(db, "set_schema"):
        db.set_schema(schema)

    content_field = opts.get("content_field", "content")
    text_search_fields = opts.get("text_search_fields", [])
    description = opts.get("description", "")

    return DatabaseSource(
        db=db,
        schema=schema,
        name=config.name,
        content_field=content_field,
        text_search_fields=text_search_fields,
        description=description,
    )


@functools.cache
def _get_field_type_names() -> dict[str, FieldType]:
    """Build the field type name mapping (cached after first call)."""
    from dataknobs_data.fields import FieldType

    result: dict[str, FieldType] = {}
    for ft in FieldType:
        result[ft.name.lower()] = ft
        result[ft.value.lower()] = ft
    return result


def _build_database_schema(
    field_defs: dict[str, Any],
) -> DatabaseSchema:
    """Build a DatabaseSchema from config field definitions.

    Each field definition can be:
        - A string type name: ``"string"``, ``"integer"``, ``"text"``, etc.
        - A dict with ``type`` and optional ``enum``:
          ``{type: string, enum: [CS, Math]}``

    Returns:
        A populated DatabaseSchema.
    """
    from dataknobs_data.fields import FieldType
    from dataknobs_data.schema import DatabaseSchema

    type_map = _get_field_type_names()
    kwargs: dict[str, FieldType] = {}
    enum_fields: dict[str, list[Any]] = {}

    for name, definition in field_defs.items():
        if isinstance(definition, str):
            ft = type_map.get(definition.lower())
            if ft is None:
                logger.warning(
                    "Unknown field type %r for field %r, defaulting to STRING",
                    definition, name,
                )
                ft = FieldType.STRING
            kwargs[name] = ft
        elif isinstance(definition, dict):
            type_str = definition.get("type", "string")
            ft = type_map.get(type_str.lower(), FieldType.STRING)
            kwargs[name] = ft

            enum_values = definition.get("enum")
            if enum_values:
                enum_fields[name] = enum_values
        else:
            logger.warning(
                "Unexpected field definition for %r: %r, skipping",
                name, definition,
            )

    schema = DatabaseSchema.create(**kwargs)

    # Apply enum metadata after creation
    for name, values in enum_fields.items():
        if name in schema.fields:
            schema.fields[name].metadata["enum"] = values

    return schema
