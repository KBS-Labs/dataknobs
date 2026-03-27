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
        output_style_hint: (extract mode) Custom description for the
            ``output_style`` field in the intent extraction schema.
            Overrides the built-in description to tune how the extraction
            model classifies synthesis style from the user's phrasing.
            Useful when the default is too aggressive or too conservative
            for a particular model or domain.  When ``None``, the built-in
            default is used (strongly favors ``conversational``).
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
    output_style_hint: str | None = None

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

    **Legacy modes** (``mode`` field):

    ``llm`` (default)
        The LLM synthesizes a natural-language response grounded in the
        retrieved results.  Citation and parametric-knowledge controls apply.

    ``template``
        A Jinja2 template formats the results deterministically.  No LLM
        call.  Template variables: ``results`` (list of result dicts),
        ``results_by_source`` (dict), ``message`` (user message),
        ``metadata`` (conversation metadata), ``intent`` (resolved intent).

    **Synthesis styles** (``style`` field — runtime-switchable):

    When ``style`` is set, it takes precedence over ``mode``.  Three styles:

    ``conversational``
        Same as ``mode: llm`` — LLM synthesis grounded in retrieved
        results.

    ``structured``
        Same as ``mode: template`` — provenance-rich formatted output.
        When no custom ``template`` is configured, a built-in default
        template is used (shows results grouped by source with headings
        and relevance scores).

    ``hybrid``
        LLM synthesis followed by a provenance appendix.  The appendix
        uses ``provenance_template`` (or the built-in default).

    **Runtime resolution cascade** (highest to lowest priority):

    1. Per-turn ``output_style`` from intent extraction (extract mode).
    2. Session-level ``metadata["synthesis_style"]``.
    3. Config-level ``style`` field.
    4. Legacy ``mode`` mapping (``llm`` → conversational,
       ``template`` → structured).
    5. Default: ``conversational``.

    Attributes:
        mode: Legacy synthesis mode — ``"llm"`` or ``"template"``.
            Superseded by ``style`` when set.
        style: Runtime-switchable synthesis style —
            ``"conversational"``, ``"structured"``, or ``"hybrid"``.
            When ``None``, falls back to ``mode``.
        require_citations: (conversational/hybrid) Instruct the LLM to
            cite sources.
        allow_parametric: (conversational/hybrid) Controls whether the
            LLM may supplement with knowledge beyond the retrieved content.

            - ``False``: Strict grounding -- only KB content, flag gaps.
            - ``True``: Relaxed -- may supplement, but distinguish sources.
            - ``"bridge"``: May connect concepts across retrieved content
              but must not introduce external facts.  When clustering is
              active, cluster annotations provide structural guidance.
        citation_format: (conversational/hybrid) ``"section"`` (heading
            paths) or ``"source"`` (file paths).
        template: Custom Jinja2 template for structured output.
        provenance_template: Custom Jinja2 template for the provenance
            appendix (hybrid mode) or structured output.  When ``None``,
            a built-in default is used.
        instruction: Optional domain-specific instruction appended to the
            synthesis system prompt after grounding instructions.  Use this
            to guide the model toward the config author's desired synthesis
            focus (e.g. "prioritize content that directly addresses the
            question").  Applied for ``conversational`` and ``hybrid``
            styles.  When ``None``, no extra instruction is appended.
    """

    mode: str = "llm"
    style: str | None = None
    require_citations: bool = True
    allow_parametric: bool | str = False
    citation_format: str = "section"
    template: str | None = None
    provenance_template: str | None = None
    instruction: str | None = None


@dataclass
class GroundedResultProcessingConfig:
    """Configuration for the result processing pipeline.

    Stages run in order: normalize -> filter -> rerank -> cluster
    (Phase 3) -> cluster-query score (Phase 3).  Each stage is
    enabled by setting its config.  Omitting a field disables that
    stage entirely.

    Stages that support multiple strategies accept either a string
    shorthand (single strategy) or a list of dicts (strategy chain
    with explicit fallback order).  Every strategy is a first-class
    choice, not a fallback.

    Attributes:
        normalize_strategy: Normalization method or strategy chain.
            String shorthand: ``"min_max"``, ``"z_score"``, ``"rank"``.
            List form: ``[{"method": "z_score"}, {"method": "min_max"}]``.
        relative_threshold: Drop results below this fraction of the
            best score (0.0-1.0).  ``None`` disables filtering.
        min_results: Never drop below this count regardless of scores.
        query_rerank_weight: Blend weight for query term overlap
            (0.0-1.0).  ``None`` disables re-ranking.
        cluster_strategy: Clustering method or strategy chain (Phase 3).
            String shorthand: ``"embedding"``, ``"tfidf"``,
            ``"term_overlap"``.  ``None`` disables clustering.
        cluster_min_size: Minimum results to form a cluster.
        cluster_threshold: Intra-cluster similarity threshold.
    """

    # Level 1
    normalize_strategy: str | list[dict[str, Any]] | None = None
    relative_threshold: float | None = None
    min_results: int = 3
    query_rerank_weight: float | None = None

    # Level 2-3 (Phase 3)
    cluster_strategy: str | list[dict[str, Any]] | None = None
    cluster_min_size: int = 2
    cluster_threshold: float = 0.7

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundedResultProcessingConfig:
        """Build from a config dict, ignoring unknown keys."""
        import dataclasses

        known_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_names}
        return cls(**filtered)


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
        topic_index: Optional topic index configuration for this source.
            When present, the source uses heading-tree or cluster-based
            retrieval instead of (or in addition to) text_queries.
            Keys: ``type`` (``"heading_tree"``, ``"cluster"``), plus
            type-specific parameters.
    """

    source_type: str = "vector_kb"
    name: str = "knowledge_base"
    weight: int = 1
    options: dict[str, Any] = field(default_factory=dict)
    topic_index: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundedSourceConfig:
        """Build from a config dict."""
        topic_index = data.get("topic_index")
        return cls(
            source_type=data.get("type", "vector_kb"),
            name=data.get("name", "knowledge_base"),
            weight=int(data.get("weight", 1)),
            options={
                k: v for k, v in data.items()
                if k not in ("type", "name", "weight", "topic_index")
            },
            topic_index=topic_index,
        )


@dataclass
class GroundedReasoningConfig:
    """Top-level configuration for :class:`GroundedReasoning`.

    Attributes:
        intent: Intent resolution configuration (replaces ``query_generation``).
        retrieval: Retrieval phase configuration.
        synthesis: Synthesis phase configuration.
        result_processing: Optional result processing pipeline configuration.
            When present, enables post-retrieval processing (normalization,
            filtering, re-ranking) between merge and format.
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
    result_processing: GroundedResultProcessingConfig | None = None
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

        # Result processing config
        rp_data = data.get("result_processing")
        rp_config = (
            GroundedResultProcessingConfig.from_dict(rp_data)
            if rp_data
            else None
        )

        return cls(
            intent=GroundedIntentConfig(**intent_data),
            retrieval=GroundedRetrievalConfig(
                **data.get("retrieval", {}),
            ),
            synthesis=GroundedSynthesisConfig(
                **data.get("synthesis", {}),
            ),
            result_processing=rp_config,
            sources=source_configs,
            store_provenance=data.get("store_provenance", True),
            greeting_template=data.get("greeting_template"),
        )
