"""Intent schema composition and parsing for grounded retrieval.

Bridges :class:`GroundedSource` schema declarations (from ``dataknobs-data``)
with :class:`SchemaExtractor` (from ``dataknobs-llm``).  Composes per-source
schema fragments into a single JSON schema for intent extraction, and
parses the extraction result back into a :class:`RetrievalIntent`.
"""

from __future__ import annotations

import copy
from typing import Any

from dataknobs_data.sources.base import GroundedSource, RetrievalIntent

# Base schema properties present in every intent extraction.
_BASE_PROPERTIES: dict[str, Any] = {
    "text_queries": {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "Key topics or phrases to search for (1-3 concise phrases). "
            "These are used for semantic/text search across all sources."
        ),
    },
    "scope": {
        "type": "string",
        "enum": ["broad", "focused", "exact"],
        "description": (
            "How broad the search should be. 'broad' for exploratory questions, "
            "'focused' for specific topics, 'exact' for precise lookups."
        ),
    },
}


def compose_intent_schema(
    sources: list[GroundedSource],
    *,
    domain_context: str = "",
) -> dict[str, Any]:
    """Compose a single JSON schema from all source schema fragments.

    Always includes ``text_queries`` (array of search phrases) and
    ``scope`` (broad/focused/exact).  For each source that declares
    a schema via :meth:`GroundedSource.get_schema`, its fields are
    nested under a property named after the source to prevent
    field name collisions.

    Args:
        sources: List of grounded sources to compose schemas from.
        domain_context: Optional domain hint added to the schema
            description to improve extraction quality.

    Returns:
        Complete JSON schema dict ready for :meth:`SchemaExtractor.extract`.

    Example::

        schema = compose_intent_schema([vector_source, db_source])
        # Returns:
        # {
        #   "type": "object",
        #   "properties": {
        #     "text_queries": {"type": "array", ...},
        #     "scope": {"type": "string", ...},
        #     "courses": {
        #       "type": "object",
        #       "properties": {
        #         "department": {"type": "string", "enum": [...]}
        #       }
        #     }
        #   },
        #   "required": ["text_queries"]
        # }
    """
    properties = copy.deepcopy(_BASE_PROPERTIES)

    if domain_context:
        properties["text_queries"]["description"] += (
            f" Domain context: {domain_context}"
        )

    source_descriptions: list[str] = []

    for source in sources:
        source_schema = source.get_schema()
        if source_schema is None or not source_schema.fields:
            continue

        # Nest source fields under source name
        source_prop: dict[str, Any] = {
            "type": "object",
            "properties": copy.deepcopy(source_schema.fields),
        }
        if source_schema.description:
            source_prop["description"] = source_schema.description
            source_descriptions.append(
                f"{source_schema.source_name}: {source_schema.description}"
            )
        if source_schema.required_fields:
            source_prop["required"] = list(source_schema.required_fields)

        properties[source_schema.source_name] = source_prop

    description = "Extract the user's search intent as structured data."
    if source_descriptions:
        description += " Available data sources: " + "; ".join(source_descriptions) + "."

    return {
        "type": "object",
        "description": description,
        "properties": properties,
        "required": ["text_queries"],
    }


def parse_intent(extraction_data: dict[str, Any]) -> RetrievalIntent:
    """Convert :class:`SchemaExtractor` output to :class:`RetrievalIntent`.

    Separates the base fields (``text_queries``, ``scope``) from
    source-specific filter dicts (everything else).

    Args:
        extraction_data: The ``data`` dict from an ``ExtractionResult``.

    Returns:
        Parsed :class:`RetrievalIntent`.
    """
    text_queries = extraction_data.get("text_queries", [])
    if isinstance(text_queries, str):
        text_queries = [text_queries]

    scope = extraction_data.get("scope", "focused")

    # Everything that isn't a base field is a source filter dict
    filters: dict[str, Any] = {}
    base_keys = {"text_queries", "scope"}
    for key, value in extraction_data.items():
        if key not in base_keys and isinstance(value, dict):
            filters[key] = value

    return RetrievalIntent(
        text_queries=text_queries,
        filters=filters,
        scope=scope,
        raw_data=extraction_data,
    )
