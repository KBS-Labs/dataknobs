"""Configuration dataclasses for the grounded reasoning strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroundedIntentConfig:
    """Configuration for intent resolution — how retrieval intent is built.

    Three modes control how the user message becomes a
    :class:`~dataknobs_data.sources.base.RetrievalIntent`:

    ``extract`` (default)
        An LLM generates search queries from the user message.  This is the
        most flexible mode — the LLM interprets the question and produces
        targeted queries.

    ``static``
        Intent is fully defined in configuration.  No LLM call.  Useful when
        the bot has a fixed domain and the config author knows exactly what
        to search for.  The user message can optionally be appended as an
        additional text query.

    ``template``
        A Jinja2 template produces a YAML dict that is parsed into a
        ``RetrievalIntent``.  Template variables include ``message`` (user
        message) and ``metadata`` (conversation metadata).  No LLM call.

    In all modes, ``default_filters`` are merged into the final intent,
    providing config-defined constraints that cannot be overridden.

    Attributes:
        mode: Intent resolution mode — ``"extract"``, ``"static"``,
            or ``"template"``.
        num_queries: (extract mode) Number of queries to generate.
        domain_context: (extract mode) Domain hint for the query gen prompt.
        use_conversation_context: (extract mode) Include conversation history.
        extraction_config: (extract mode) Optional dict for creating a
            dedicated :class:`~dataknobs_llm.extraction.SchemaExtractor`
            for intent extraction.  When present, query generation uses
            structured JSON extraction with confidence scoring instead
            of raw text parsing.  Same shape as wizard ``extraction_config``
            (``provider``, ``model``, ``temperature``, etc.).  When absent,
            falls back to the ``QueryTransformer`` text-parsing path.
        text_queries: (static mode) Fixed text queries.
        filters: (static mode) Fixed structured filters keyed by source name.
        scope: (static/template mode) Retrieval scope.
        include_message_as_query: (static mode) Append the user message as
            an additional text query.
        template: (template mode) Jinja2 template producing a YAML dict
            with ``text_queries``, ``filters``, ``scope`` keys.
        expand_ambiguous_queries: (extract mode) Use
            :class:`~dataknobs_bots.knowledge.query.ContextualExpander` to
            enrich ambiguous queries with conversation-context keywords
            before passing to the LLM.  Triggered only when
            :func:`~dataknobs_bots.knowledge.query.is_ambiguous_query`
            returns ``True``.
        max_context_turns: (extract mode) How many conversation turns
            the expander considers when extracting keywords.
        include_assistant_context: (extract mode) Whether the expander
            includes assistant messages when extracting keywords.
        default_filters: Filters merged into the final intent regardless
            of mode.  Config-defined constraints the LLM cannot override.
    """

    mode: str = "extract"

    # Extract mode
    num_queries: int = 3
    domain_context: str = ""
    use_conversation_context: bool = True
    extraction_config: dict[str, Any] | None = None

    # Extract mode — query enrichment (optional)
    expand_ambiguous_queries: bool = False
    max_context_turns: int = 3
    include_assistant_context: bool = False

    # Static mode
    text_queries: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    scope: str = "focused"
    include_message_as_query: bool = True

    # Template mode
    template: str | None = None

    # All modes
    default_filters: dict[str, Any] = field(default_factory=dict)


# Backward-compatible alias — GroundedQueryConfig was the Phase 1 name.
GroundedQueryConfig = GroundedIntentConfig


@dataclass
class GroundedRetrievalConfig:
    """Configuration for the deterministic retrieval phase.

    Attributes:
        top_k: Maximum results per query.
        score_threshold: Minimum similarity score to include a result.
        merge_adjacent: Whether to merge adjacent chunks sharing
            the same heading path via ``ChunkMerger``.
        deduplicate: Whether to deduplicate results returned by
            multiple queries (keyed on source + chunk_index).
    """

    top_k: int = 5
    score_threshold: float = 0.3
    merge_adjacent: bool = True
    deduplicate: bool = True


@dataclass
class GroundedSynthesisConfig:
    """Configuration for the synthesis phase.

    Two modes:

    ``llm`` (default)
        The LLM synthesizes a natural-language response grounded in the
        retrieved results.  Citation and parametric-knowledge controls apply.

    ``template``
        A Jinja2 template formats the results deterministically.  No LLM
        call.  Template variables: ``results`` (list of result dicts),
        ``results_by_source`` (dict), ``message`` (user message),
        ``metadata`` (conversation metadata), ``intent`` (resolved intent).

    Attributes:
        mode: Synthesis mode — ``"llm"`` or ``"template"``.
        require_citations: (llm mode) Instruct the LLM to cite sources.
        allow_parametric: (llm mode) Allow the LLM to supplement with
            its own parametric knowledge.
        citation_format: (llm mode) ``"section"`` (heading paths) or
            ``"source"`` (file paths).
        template: (template mode) Jinja2 template string.
    """

    mode: str = "llm"
    require_citations: bool = True
    allow_parametric: bool = False
    citation_format: str = "section"
    template: str | None = None


@dataclass
class GroundedSourceConfig:
    """Configuration for a single grounded source.

    Used by the config-driven source factory to construct
    :class:`~dataknobs_data.sources.base.GroundedSource` instances.

    Attributes:
        source_type: Source type key — ``"vector_kb"``, ``"database"``.
        name: Unique source name for provenance tracking.
        weight: Relative weight for round-robin result merging.
            A source with ``weight: 3`` contributes 3 results per
            round-robin cycle vs 1 for ``weight: 1`` (default).
            Higher weights give a source proportionally more
            representation in the merged result set.
        options: Source-specific options passed to the constructor.
            For ``"database"``: ``backend``, ``connection``,
            ``content_field``, ``text_search_fields``, ``schema``.
    """

    source_type: str = "vector_kb"
    name: str = "knowledge_base"
    weight: int = 1
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundedSourceConfig:
        """Build from a config dict."""
        return cls(
            source_type=data.get("type", "vector_kb"),
            name=data.get("name", "knowledge_base"),
            weight=int(data.get("weight", 1)),
            options={
                k: v for k, v in data.items()
                if k not in ("type", "name", "weight")
            },
        )


@dataclass
class GroundedReasoningConfig:
    """Top-level configuration for :class:`GroundedReasoning`.

    Attributes:
        intent: Intent resolution configuration (replaces ``query_generation``).
        retrieval: Retrieval phase configuration.
        synthesis: Synthesis phase configuration.
        sources: Optional list of source configurations for config-driven
            source construction.  When empty, sources are injected
            programmatically via ``set_knowledge_base()`` / ``add_source()``.
        store_provenance: Whether to record retrieval provenance
            (queries, chunks, scores, timing) in conversation metadata.
        greeting_template: Optional Jinja2 template for bot-initiated
            greetings (same semantics as other strategies).
    """

    intent: GroundedIntentConfig = field(
        default_factory=GroundedIntentConfig,
    )
    retrieval: GroundedRetrievalConfig = field(
        default_factory=GroundedRetrievalConfig,
    )
    synthesis: GroundedSynthesisConfig = field(
        default_factory=GroundedSynthesisConfig,
    )
    sources: list[GroundedSourceConfig] = field(default_factory=list)
    store_provenance: bool = True
    greeting_template: str | None = None

    @property
    def query_generation(self) -> GroundedIntentConfig:
        """Backward-compatible alias for :attr:`intent`."""
        return self.intent

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundedReasoningConfig:
        """Build config from a flat reasoning config dict.

        Accepts both the new ``intent`` key and the legacy
        ``query_generation`` key (which maps to extract mode).

        New shape::

            reasoning:
              strategy: grounded
              intent:
                mode: extract
                num_queries: 3
              retrieval: {top_k: 5}
              synthesis: {mode: llm, require_citations: true}
              sources:
                - type: vector_kb
                  name: docs

        Legacy shape (still supported)::

            reasoning:
              strategy: grounded
              query_generation: {num_queries: 3}
              retrieval: {top_k: 5}
              synthesis: {require_citations: true}
        """
        # Intent config: prefer "intent", fall back to "query_generation"
        intent_data = data.get("intent") or data.get("query_generation", {})

        # Source configs
        source_configs = [
            GroundedSourceConfig.from_dict(s)
            for s in data.get("sources", [])
        ]

        return cls(
            intent=GroundedIntentConfig(**intent_data),
            retrieval=GroundedRetrievalConfig(
                **data.get("retrieval", {}),
            ),
            synthesis=GroundedSynthesisConfig(
                **data.get("synthesis", {}),
            ),
            sources=source_configs,
            store_provenance=data.get("store_provenance", True),
            greeting_template=data.get("greeting_template"),
        )
