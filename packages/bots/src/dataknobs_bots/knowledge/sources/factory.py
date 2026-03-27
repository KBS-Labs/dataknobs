"""Config-driven construction of :class:`GroundedSource` instances.

Translates :class:`~dataknobs_bots.reasoning.grounded_config.GroundedSourceConfig`
declarations into concrete source objects, using ``dataknobs-data`` factories
for database backends and ``dataknobs-bots`` for vector KB sources.
"""

from __future__ import annotations

import functools
import logging
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
    from .vector import VectorKnowledgeSource

    if knowledge_base is None:
        raise ValueError(
            f"Source {config.name!r} has type 'vector_kb' but no "
            f"knowledge_base was provided. Either configure a "
            f"knowledge_base in the bot config or use a different "
            f"source type."
        )

    # Topic index is constructed lazily by the source or externally
    # after all chunks are available — the factory stores the config.
    return VectorKnowledgeSource(knowledge_base, name=config.name)


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
