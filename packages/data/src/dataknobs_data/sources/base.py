"""Grounded source abstraction for structured retrieval.

Defines the core types and ABC for queryable data sources that participate
in a grounded retrieval pipeline.  Sources declare schemas of queryable
dimensions, receive structured intent, and translate it deterministically
to native queries.

The LLM never generates query syntax.  It extracts structured intent
against declared schemas (handled by ``dataknobs-llm``), and each source
translates that intent to its native query language in code.

This module is intentionally LLM-free — it lives in ``dataknobs-data``
so any project using the data layer can define and query sources without
depending on the LLM or bots packages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass
class SourceSchema:
    """Schema fragment a source declares for intent extraction.

    Each source publishes its queryable dimensions as a JSON schema
    ``properties`` fragment.  The orchestrating layer composes fragments
    from all sources into a single schema for the intent extractor.

    Attributes:
        source_name: Unique identifier matching :attr:`GroundedSource.name`.
        fields: JSON schema ``properties`` fragment.  Keys are field names,
            values are JSON schema property definitions.  May include
            ``x-extraction`` hints (e.g. ``{"normalize": true}``) for
            enum normalization.
        required_fields: Fields that must be present in the extracted
            intent for this source to receive meaningful queries.
        description: Human-readable description included in the
            extraction prompt to help the LLM understand this source.
    """

    source_name: str
    fields: dict[str, Any]
    required_fields: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RetrievalIntent:
    """Source-agnostic structured intent for retrieval.

    Produced by an intent extractor (e.g. ``SchemaExtractor`` in
    ``dataknobs-llm``) from a user message against a composed schema.
    Each source reads the slice of ``filters`` keyed by its name.

    Attributes:
        text_queries: Semantic search phrases.  Always present — even
            sources with structured filters receive these for text search.
        filters: Structured conditions keyed by ``source_name``.
            Values are dicts of ``{field_name: value}`` extracted from
            the user message.
        scope: Retrieval breadth hint.
        raw_data: The full extraction dict, preserved for provenance.
    """

    text_queries: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    scope: str = "focused"
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceResult:
    """Normalized result that all sources produce.

    Provides a uniform shape regardless of whether the backing store
    is a vector KB, SQL database, Elasticsearch index, or anything else.

    Attributes:
        content: The text content of this result, for inclusion in
            the synthesis prompt.
        source_id: Unique identifier within the source (chunk ID,
            primary key, document ID, etc.).
        source_name: Which :class:`GroundedSource` produced this.
        source_type: Category string (``"vector_kb"``, ``"database"``,
            ``"elasticsearch"``).
        relevance: Score from 0.0 to 1.0.  Similarity for vector
            sources, 1.0 for exact-match database results.
        metadata: Source-specific metadata (heading paths, table names,
            field values, etc.).
    """

    content: str
    source_id: str
    source_name: str
    source_type: str
    relevance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for compatibility with existing formatters."""
        return {
            "text": self.content,
            "source": self.source_name,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "similarity": self.relevance,
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------
# ABC
# ------------------------------------------------------------------


class GroundedSource(ABC):
    """Abstract base for retrieval sources in a grounded pipeline.

    Each source declares:

    - **Schema** — queryable dimensions (filter fields, types, enums)
      via :meth:`get_schema`.
    - **Query execution** — deterministic translation of structured
      intent to native queries via :meth:`query`.

    The LLM never generates query syntax.  It extracts structured
    intent against declared schemas, and each source translates that
    intent deterministically.

    Subclasses must implement :attr:`name`, :attr:`source_type`, and
    :meth:`query`.  Override :meth:`get_schema` to declare filter
    dimensions (default returns ``None`` — text-only source).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique source identifier, used in provenance and schema composition."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Source category string (e.g. ``"vector_kb"``, ``"database"``)."""

    def get_schema(self) -> SourceSchema | None:
        """Declare queryable dimensions for intent extraction.

        Sources with no structured filters (e.g. pure vector KB)
        return ``None``.  The orchestrator still passes ``text_queries``
        to them.

        .. note::

            Schema declarations are consumed by
            ``compose_intent_schema()`` in ``dataknobs-llm`` to build
            a single JSON schema for ``SchemaExtractor``-based structured
            intent extraction.  This pipeline is planned infrastructure
            for a future ``mode: "schema"`` intent mode in
            ``GroundedReasoning``.  The current ``mode: "extract"`` uses
            ``QueryTransformer`` and does not call ``get_schema()``.

        Returns:
            Schema fragment, or ``None`` for text-only sources.
        """
        return None

    @abstractmethod
    async def query(
        self,
        intent: RetrievalIntent,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SourceResult]:
        """Execute retrieval against this source using extracted intent.

        Translation from intent to native query is deterministic code,
        not LLM-generated.

        Args:
            intent: Structured intent from the extraction layer.
            top_k: Maximum results to return.
            score_threshold: Minimum relevance score to include.

        Returns:
            Normalized results sorted by relevance (descending).
        """

    async def close(self) -> None:  # noqa: B027
        """Release resources.  Default no-op."""
