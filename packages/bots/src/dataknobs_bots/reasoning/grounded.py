"""Grounded reasoning strategy — deterministic retrieval pipeline.

This strategy guarantees that every substantive turn retrieves from
configured data sources, eliminating the unreliability of LLM-decided
retrieval (the ReAct pattern).  Pipeline:

1. **Intent resolution** — build a :class:`RetrievalIntent` via LLM
   extraction, static config, or Jinja2 template.
2. **Deterministic retrieval** — intent is executed against all
   configured :class:`GroundedSource` instances.
3. **Synthesis** — LLM synthesizes a response grounded in retrieved
   results, or a Jinja2 template formats them deterministically.

Both the intent and synthesis phases can bypass the LLM entirely
via configuration, enabling fully deterministic pipelines when the
config author knows the query shape and output format.

Provenance (intent, results by source, timing) is recorded per turn
in ``manager.metadata["retrieval_provenance"]``.

Sources are pluggable via the :class:`GroundedSource` abstraction from
``dataknobs-data``.  The strategy works with vector knowledge bases,
SQL databases, Elasticsearch, or any custom source.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

import jinja2

from dataknobs_data.sources.base import (
    GroundedSource,
    RetrievalIntent,
    SourceResult,
)
from dataknobs_data.sources.processing import ResultPipeline, build_pipeline

from dataknobs_llm.extraction.grounding import ground_extraction

from dataknobs_bots.knowledge.query.expander import ContextualExpander, is_ambiguous_query
from dataknobs_bots.knowledge.query.transformer import QueryTransformer, TransformerConfig
from dataknobs_bots.knowledge.retrieval.formatter import (
    ContextFormatter,
    FormatterConfig,
)
from dataknobs_bots.knowledge.retrieval.merger import ChunkMerger, MergerConfig
from dataknobs_bots.reasoning.base import ReasoningStrategy
from dataknobs_bots.reasoning.grounded_config import GroundedReasoningConfig

logger = logging.getLogger(__name__)

# JSON Schema for structured intent extraction via SchemaExtractor.
# Used when extraction_config is present — replaces raw text parsing
# with validated JSON output, confidence scoring, and extraction
# resilience from the SchemaExtractor framework.
INTENT_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Concise search queries (2-6 words each) targeting "
                "the user's information need.  Focus on underlying "
                "intent, key concepts, and related topics — not the "
                "literal text."
            ),
        },
        "scope": {
            "type": "string",
            "enum": ["focused", "broad", "exact"],
            "description": (
                "Retrieval breadth: 'focused' for specific topics, "
                "'broad' for exploratory or multi-faceted questions, "
                "'exact' for precise lookups."
            ),
        },
        "output_style": {
            "type": "string",
            "enum": ["conversational", "structured", "hybrid"],
            "description": (
                "Default to 'conversational'.  Only use 'structured' "
                "when the user explicitly asks to see raw sources, a "
                "formatted listing, or provenance details (e.g., "
                "'show me the sources', 'list the relevant sections', "
                "'what does the spec actually say').  Only use 'hybrid' "
                "when the user asks for both explanation and sources."
            ),
        },
    },
    "required": ["text_queries"],
}


# Built-in provenance template used for structured output and hybrid
# appendix when no custom template is configured.
DEFAULT_PROVENANCE_TEMPLATE = """\
{% if results %}
{% for source_name, source_results in results_by_source.items() %}
### {{ source_name }}

{% for r in source_results %}
- **{{ r.metadata.get('headings', [''])|join(' > ') or r.source_id }}** \
(relevance: {{ "%.0f"|format(r.relevance * 100) }}%)
  {{ r.text_preview }}
{% endfor %}
{% endfor %}
---
*{{ results|length }} result{{ 's' if results|length != 1 }} \
from {{ results_by_source|length }} source{{ 's' if results_by_source|length != 1 }}*
{% else %}
No relevant results found.
{% endif %}\
"""

# Valid synthesis styles for runtime resolution.
_VALID_STYLES = frozenset({"conversational", "structured", "hybrid"})


@dataclass
class SynthesisPlan:
    """Resolved synthesis strategy for a single turn.

    Computed once by :meth:`GroundedReasoning.resolve_synthesis` and
    consumed by both the buffered and streaming delivery paths — ensuring
    style resolution, template rendering, and prompt construction happen
    in a single shared code path.

    The :meth:`apply_to_response` method applies the plan to an existing
    ``LLMResponse``, handling all three styles (conversational, structured,
    hybrid).  This is the shared dispatch logic used by both
    ``GroundedReasoning._synthesize`` and ``HybridReasoning``.
    """

    effective_style: str
    """One of ``"conversational"``, ``"structured"``, ``"hybrid"``."""

    template_text: str | None
    """Pre-rendered provenance output (structured full / hybrid appendix)."""

    system_prompt: str | None
    """LLM synthesis prompt (conversational / hybrid)."""

    def apply_to_response(self, response: Any) -> Any:
        """Apply synthesis formatting to an existing LLM response.

        - ``conversational``: return the response unchanged.
        - ``structured``: return the pre-rendered template, discarding
          the LLM response.
        - ``hybrid``: append the template to the LLM response content.

        Args:
            response: An ``LLMResponse`` (or compatible object with
                ``content``, ``model``, ``finish_reason`` attributes).

        Returns:
            The formatted ``LLMResponse``.
        """
        from dataknobs_llm import LLMResponse as _LLMResponse

        if self.effective_style == "structured" and self.template_text is not None:
            return _LLMResponse(
                content=self.template_text,
                model="template",
                finish_reason="stop",
            )

        if self.effective_style == "hybrid" and self.template_text:
            combined = (
                (getattr(response, "content", "") or "")
                + "\n\n"
                + self.template_text
            )
            return _LLMResponse(
                content=combined,
                model=getattr(response, "model", "unknown"),
                finish_reason=getattr(response, "finish_reason", "stop"),
            )

        # conversational or fallback — return as-is
        return response


class GroundedReasoning(ReasoningStrategy):
    """Reasoning strategy with deterministic multi-source retrieval.

    Every turn executes the full retrieval pipeline regardless of whether
    the LLM thinks it knows the answer.  Sources are pluggable — vector
    KBs, SQL databases, Elasticsearch, or any :class:`GroundedSource`.

    Intent resolution supports three modes:

    - ``extract`` — LLM generates search queries (default).
    - ``static`` — intent defined entirely in config (no LLM call).
    - ``template`` — Jinja2 template produces intent (no LLM call).

    Synthesis supports two modes:

    - ``llm`` — LLM synthesizes a natural-language response (default).
    - ``template`` — Jinja2 template formats results deterministically.

    Both bypasses are independent: you can extract intent via LLM but
    template the output, or use static intent with LLM synthesis.

    Example configuration::

        reasoning:
          strategy: grounded
          intent:
            mode: extract
            num_queries: 3
            domain_context: "OAuth 2.0 authorization framework"
          retrieval:
            top_k: 5
            score_threshold: 0.3
          synthesis:
            mode: llm
            require_citations: true
          store_provenance: true
    """

    def __init__(
        self,
        config: GroundedReasoningConfig,
        sources: list[GroundedSource] | None = None,
        query_provider: Any | None = None,
    ) -> None:
        """Initialize the grounded reasoning strategy.

        Args:
            config: Strategy configuration.
            sources: List of grounded sources to query.  When empty,
                use :meth:`set_knowledge_base` or :meth:`add_source`
                to inject sources after construction.
            query_provider: Optional separate LLM provider for query
                generation.  When ``None`` the bot's main LLM (passed
                as ``llm`` to :meth:`generate`) is used.
        """
        super().__init__(greeting_template=config.greeting_template)
        self._config = config
        self._sources: list[GroundedSource] = list(sources) if sources else []
        self._source_weights: dict[str, int] = {
            sc.name: sc.weight for sc in config.sources
        }
        self._query_provider = query_provider
        self._merger = ChunkMerger(MergerConfig()) if config.retrieval.merge_adjacent else None
        self._formatter = ContextFormatter(FormatterConfig(
            include_scores=False,
            include_source=True,
            group_by_source=True,
        ))

        # Query generation components (extract mode only).
        # Two paths: SchemaExtractor (preferred, when extraction_config
        # is present) or QueryTransformer (legacy fallback).
        self._transformer: QueryTransformer | None = None
        self._extractor: Any | None = None
        self._expander: ContextualExpander | None = None
        if config.intent.mode == "extract":
            if config.intent.extraction_config:
                self._extractor = self._create_intent_extractor(
                    config.intent.extraction_config,
                )
            else:
                self._transformer = QueryTransformer(
                    TransformerConfig(
                        enabled=True,
                        num_queries=config.intent.num_queries,
                        domain_context=config.intent.domain_context,
                        suppress_thinking=True,
                    ),
                    provider=query_provider,
                )
            if config.intent.expand_ambiguous_queries:
                self._expander = ContextualExpander(
                    max_context_turns=config.intent.max_context_turns,
                    include_assistant=config.intent.include_assistant_context,
                )

        # Pre-compile Jinja2 environments for template modes (avoids
        # recreating per render call)
        self._jinja_env = jinja2.Environment(undefined=jinja2.Undefined)
        self._intent_template = (
            self._jinja_env.from_string(config.intent.template)
            if config.intent.template
            else None
        )
        self._synthesis_template = (
            self._jinja_env.from_string(config.synthesis.template)
            if config.synthesis.template
            else None
        )
        self._provenance_template = self._jinja_env.from_string(
            config.synthesis.provenance_template
            or DEFAULT_PROVENANCE_TEMPLATE,
        )

        # Result processing pipeline (Phase 2: post-retrieval processing)
        self._result_pipeline: ResultPipeline | None = None
        if config.result_processing is not None:
            rp_dict = asdict(config.result_processing)
            self._result_pipeline = build_pipeline(rp_dict)

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def add_source(self, source: GroundedSource) -> None:
        """Add a source to query during retrieval."""
        self._sources.append(source)

    def set_knowledge_base(self, kb: Any) -> None:
        """Wrap a KnowledgeBase in VectorKnowledgeSource and add it.

        Backward-compatible method called by ``DynaBot.from_config()``
        when the bot has a KB and the strategy is grounded.

        If source configs declare a ``topic_index``, the appropriate
        topic index is constructed and attached to the source.  Topic
        indices are lightweight in lazy mode — they store query
        functions but don't pre-build any structures.

        .. note::

            When called from ``from_config()``, ``auto_context`` is
            auto-disabled.  When called post-construction (programmatic
            KB injection), callers should also set
            ``bot._kb_auto_context = False`` to prevent the bot from
            double-retrieving KB content via auto-injection.
        """
        from dataknobs_bots.knowledge.sources.vector import VectorKnowledgeSource

        # Replace any existing vector KB source
        self._sources = [
            s for s in self._sources
            if s.source_type != "vector_kb"
        ]

        # Check if any source config declares a topic_index
        topic_index_config = self._find_topic_index_config()

        from dataknobs_bots.knowledge.sources.factory import build_topic_index

        # Determine source name from config (if a vector_kb source is declared)
        source_name = "knowledge_base"
        for source_cfg in self._config.sources:
            if source_cfg.source_type == "vector_kb":
                source_name = source_cfg.name
                break

        topic_index = build_topic_index(
            topic_index_config, kb, source_name=source_name,
        )

        self._sources.insert(
            0, VectorKnowledgeSource(
                kb, name=source_name, topic_index=topic_index,
            ),
        )

    def _find_topic_index_config(self) -> dict[str, Any] | None:
        """Find topic_index config from source declarations."""
        for source_cfg in self._config.sources:
            if source_cfg.source_type == "vector_kb" and source_cfg.topic_index:
                return source_cfg.topic_index
        return None

    # ------------------------------------------------------------------
    # Provider management (for test injection)
    # ------------------------------------------------------------------

    def providers(self) -> dict[str, Any]:
        """Return LLM providers managed by this strategy and its sources."""
        result: dict[str, Any] = {}
        if self._query_provider is not None:
            result["grounded_query"] = self._query_provider
        for source in self._sources:
            if hasattr(source, "providers"):
                result.update(source.providers())
        return result

    @staticmethod
    def _create_intent_extractor(
        extraction_config: dict[str, Any],
    ) -> Any:
        """Create a SchemaExtractor for structured intent extraction.

        Uses the same ``SchemaExtractor.from_env_config()`` pattern as
        wizard extraction — the config dict has ``provider``, ``model``,
        ``temperature``, etc.

        Args:
            extraction_config: Provider/model config dict.

        Returns:
            A :class:`~dataknobs_llm.extraction.SchemaExtractor` instance.
        """
        from dataknobs_llm.extraction import SchemaExtractor

        return SchemaExtractor.from_env_config(extraction_config)

    def set_extractor(self, extractor: Any) -> None:
        """Set or replace the intent extractor (test injection).

        Args:
            extractor: A :class:`~dataknobs_llm.extraction.SchemaExtractor`
                (or compatible duck-typed object with an ``extract()`` method).
        """
        self._extractor = extractor

    def set_provider(self, role: str, provider: Any) -> bool:
        """Replace a managed provider (test injection)."""
        if role == "grounded_query":
            self._query_provider = provider
            if self._transformer is not None:
                self._transformer.set_provider(provider)
            return True
        for source in self._sources:
            if hasattr(source, "set_provider") and source.set_provider(role, provider):
                return True
        return False

    async def close(self) -> None:
        """Release resources held by the query provider and sources."""
        if self._query_provider is not None and hasattr(self._query_provider, "close"):
            await self._query_provider.close()
        if self._extractor is not None and hasattr(self._extractor, "close"):
            await self._extractor.close()
        for source in self._sources:
            await source.close()

    # ------------------------------------------------------------------
    # Core pipeline — generate()
    # ------------------------------------------------------------------

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the grounded retrieval pipeline.

        1. Resolve intent (LLM / static / template).
        2. Retrieve from all sources (deterministic — always runs).
        3. Synthesize response (LLM / template).
        4. Record provenance in ``manager.metadata``.

        Args:
            manager: Conversation manager (ReasoningManagerProtocol).
            llm: Bot's main LLM provider.
            tools: Accepted for ABC compliance; unused in this strategy.
                :class:`HybridReasoning` uses tools in its ReAct phase.
            **kwargs: Forwarded to ``manager.complete()`` (temperature, etc.).
        """
        context, provenance = await self.retrieve_context(manager, llm)

        result = await self._synthesize(context, manager, provenance, **kwargs)

        if self._config.store_provenance:
            self._store_provenance(manager, provenance)

        return result

    # ------------------------------------------------------------------
    # Streaming pipeline — stream_generate()
    # ------------------------------------------------------------------

    async def stream_generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream the grounded retrieval pipeline.

        Intent resolution and retrieval run synchronously (fast).
        Only LLM synthesis streams; structured/hybrid template output
        is yielded as discrete chunks.

        Uses :meth:`resolve_synthesis` for style resolution — the same
        shared code path as the buffered :meth:`_synthesize`.

        Yields:
            LLM stream chunks from the synthesis phase.
        """
        from dataknobs_llm import LLMResponse

        context, provenance = await self.retrieve_context(manager, llm)
        plan = self.resolve_synthesis(context, manager, provenance)

        if plan.effective_style == "structured":
            yield LLMResponse(
                content=plan.template_text,
                model="template",
                finish_reason="stop",
            )
        else:
            # conversational or hybrid — stream LLM synthesis
            async for chunk in manager.stream_complete(
                system_prompt_override=plan.system_prompt,
                **kwargs,
            ):
                yield chunk

            if plan.effective_style == "hybrid":
                yield LLMResponse(
                    content="\n\n" + plan.template_text,
                    model="template",
                    finish_reason="stop",
                )

        if self._config.store_provenance:
            self._store_provenance(manager, provenance)

    # ------------------------------------------------------------------
    # Public retrieval pipeline (used by HybridReasoning)
    # ------------------------------------------------------------------

    async def retrieve_context(
        self,
        manager: Any,
        llm: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Run intent resolution + multi-source retrieval + processing.

        Returns the formatted context string and a provenance dict.
        This is the public entry point that :class:`HybridReasoning`
        calls to obtain grounded context before entering a ReAct loop.

        Args:
            manager: Conversation manager.
            llm: LLM provider for query generation (extract mode only).

        Returns:
            ``(formatted_context, provenance_dict)``
        """
        messages = manager.get_messages()
        user_message = self._extract_user_message(messages)

        # Phase 1: Intent resolution
        t0 = time.monotonic()
        intent = await self._resolve_intent(user_message, messages, llm, manager.metadata)
        intent_ms = (time.monotonic() - t0) * 1000

        # Phase 2: Deterministic retrieval across all sources
        t0 = time.monotonic()
        results_by_source = await self._retrieve_from_sources(
            intent, user_message=user_message, llm=llm,
        )
        retrieval_ms = (time.monotonic() - t0) * 1000

        # Merge results across sources
        all_results = self._merge_source_results(results_by_source)

        # Result processing pipeline (normalization, filtering, re-ranking)
        if self._result_pipeline is not None:
            all_results = await self._result_pipeline.process(
                all_results, intent, user_message,
            )

        # Format for synthesis
        formatted_context = self._format_source_results(all_results)

        # Note: results_by_source reflects pre-processing (raw per-source
        # results), while all_results reflects post-processing (normalized,
        # filtered, clustered).  Provenance templates using results_by_source
        # show original scores; templates using results show processed scores.
        provenance = self._build_provenance(
            intent=intent,
            results_by_source=results_by_source,
            all_results=all_results,
            intent_resolution_time_ms=intent_ms,
            retrieval_time_ms=retrieval_ms,
        )

        return formatted_context, provenance

    # ------------------------------------------------------------------
    # Intent resolution (private)
    # ------------------------------------------------------------------

    async def _resolve_intent(
        self,
        user_message: str,
        messages: list[dict[str, Any]],
        llm: Any,
        metadata: dict[str, Any],
    ) -> RetrievalIntent:
        """Resolve retrieval intent based on configured mode."""
        cfg = self._config.intent
        mode = cfg.mode

        if mode == "static":
            intent = self._build_static_intent(user_message)
        elif mode == "template":
            intent = self._build_template_intent(user_message, metadata)
        elif self._extractor is not None:
            # Extract mode with SchemaExtractor (preferred path)
            intent = await self._extract_intent(user_message, messages)
        else:
            # Extract mode with QueryTransformer (legacy fallback)
            queries = await self._generate_queries(user_message, messages, llm)
            intent = RetrievalIntent(
                text_queries=queries,
                scope="focused",
            )

        # Merge default_filters (config-defined constraints always win)
        if cfg.default_filters:
            merged = dict(intent.filters)
            for source_name, filters in cfg.default_filters.items():
                if source_name in merged and isinstance(merged[source_name], dict):
                    merged[source_name] = {**merged[source_name], **filters}
                else:
                    merged[source_name] = filters
            intent = RetrievalIntent(
                text_queries=intent.text_queries,
                filters=merged,
                scope=intent.scope,
                raw_data=intent.raw_data,
            )

        return intent

    def _build_static_intent(self, user_message: str) -> RetrievalIntent:
        """Build intent from static config values."""
        cfg = self._config.intent
        queries = list(cfg.text_queries)
        if cfg.include_message_as_query and user_message:
            queries.append(user_message)

        return RetrievalIntent(
            text_queries=queries,
            filters=dict(cfg.filters),
            scope=cfg.scope,
        )

    def _build_template_intent(
        self,
        user_message: str,
        metadata: dict[str, Any],
    ) -> RetrievalIntent:
        """Render a Jinja2 template to produce intent.

        Template variables:
            ``message``: The user's message.
            ``metadata``: Conversation metadata (session state, etc.).

        The template must produce valid YAML with ``text_queries``,
        and optionally ``filters`` and ``scope``.
        """
        import yaml

        cfg = self._config.intent
        if not cfg.template:
            logger.warning("Intent mode is 'template' but no template configured; falling back to message")
            return RetrievalIntent(text_queries=[user_message])

        template = self._intent_template
        if template is None:
            logger.warning("Intent template was not compiled; falling back to message")
            return RetrievalIntent(text_queries=[user_message])

        rendered = template.render(
            message=user_message,
            metadata=metadata,
        )

        try:
            data = yaml.safe_load(rendered)
        except Exception:
            logger.warning("Failed to parse intent template output as YAML", exc_info=True)
            return RetrievalIntent(text_queries=[user_message])

        if not isinstance(data, dict):
            logger.warning("Intent template produced %s, expected dict", type(data).__name__)
            return RetrievalIntent(text_queries=[user_message])

        text_queries = data.get("text_queries", [])
        if isinstance(text_queries, str):
            text_queries = [text_queries]

        return RetrievalIntent(
            text_queries=text_queries,
            filters=data.get("filters", {}),
            scope=data.get("scope", "focused"),
        )

    async def _extract_intent(
        self,
        user_message: str,
        messages: list[dict[str, Any]],
    ) -> RetrievalIntent:
        """Extract structured intent via :class:`SchemaExtractor`.

        Uses the ``INTENT_EXTRACTION_SCHEMA`` to produce validated JSON
        with ``text_queries`` and optional ``scope``.  Falls back to
        ``[user_message]`` when extraction returns empty queries or
        raises an exception.
        """
        cfg = self._config.intent

        # Optional: enrich ambiguous queries with context keywords
        enriched = user_message
        if self._expander is not None and is_ambiguous_query(user_message):
            enriched = self._expander.expand(user_message, messages)
            logger.debug(
                "Expanded ambiguous query: %r → %r",
                user_message, enriched,
            )

        # Build extraction input — include conversation context if enabled
        conversation_context = self._build_conversation_context(messages, cfg)
        if conversation_context:
            extraction_input = (
                f"Conversation context:\n{conversation_context}\n\n"
                f"Current user message: {enriched}"
            )
        else:
            extraction_input = enriched

        # Build the schema with dynamic constraints.
        # Deep copy protects the module-level constant from mutation.
        schema = copy.deepcopy(INTENT_EXTRACTION_SCHEMA)
        schema["properties"]["text_queries"]["maxItems"] = cfg.num_queries
        if cfg.output_style_hint:
            schema["properties"]["output_style"]["description"] = (
                cfg.output_style_hint
            )

        context = {}
        if cfg.domain_context:
            context["domain"] = cfg.domain_context

        try:
            result = await self._extractor.extract(
                text=extraction_input,
                schema=schema,
                context=context,
            )

            # Ground-check optional extracted fields against the user
            # message.  Ungrounded optional fields (e.g. output_style
            # inferred without explicit user language) are dropped so
            # the resolution cascade falls through to config/session
            # defaults.
            grounding_results = ground_extraction(
                result.data, user_message, schema,
            )
            required_fields = set(schema.get("required", []))
            for fname, gresult in grounding_results.items():
                if not gresult.grounded and fname not in required_fields:
                    logger.debug(
                        "Dropping ungrounded optional field %r: %s",
                        fname, gresult.reason,
                    )
                    result.data.pop(fname, None)

            queries = result.data.get("text_queries", [])
            scope = result.data.get("scope", "focused")

            if not queries:
                logger.warning(
                    "Intent extraction produced no queries "
                    "(confidence=%.2f), falling back to user message",
                    result.confidence,
                )
                return RetrievalIntent(
                    text_queries=[user_message],
                    scope="focused",
                    raw_data=result.data,
                )

            logger.debug(
                "Extracted %d queries (confidence=%.2f, scope=%s): %s",
                len(queries), result.confidence, scope, queries,
            )
            return RetrievalIntent(
                text_queries=queries[:cfg.num_queries],
                scope=scope,
                raw_data=result.data,
            )
        except Exception:
            logger.warning(
                "Intent extraction failed, falling back to raw user message",
                exc_info=True,
            )
            return RetrievalIntent(text_queries=[user_message], scope="focused")

    async def _generate_queries(
        self,
        user_message: str,
        messages: list[dict[str, Any]],
        llm: Any,
    ) -> list[str]:
        """Generate search queries via :class:`QueryTransformer`.

        Legacy path used when no ``extraction_config`` is present.
        Delegates query generation to the transformer, which builds
        structured prompts emphasizing intent, key concepts, and related
        topics.  When ``expand_ambiguous_queries`` is enabled, ambiguous
        queries are first enriched with conversation-context keywords
        via :class:`ContextualExpander`.

        Falls back to ``[user_message]`` on failure.
        """
        if self._transformer is None:
            return [user_message]

        cfg = self._config.intent
        provider = self._query_provider or llm

        # Ensure the transformer has the current provider
        self._transformer.set_provider(provider)

        # Optional: enrich ambiguous queries with context keywords
        enriched = user_message
        if self._expander is not None and is_ambiguous_query(user_message):
            enriched = self._expander.expand(user_message, messages)
            logger.debug(
                "Expanded ambiguous query: %r → %r",
                user_message, enriched,
            )

        try:
            # Build conversation context for the transformer
            conversation_context = self._build_conversation_context(messages, cfg)
            if conversation_context:
                queries = await self._transformer.transform_with_context(
                    enriched, conversation_context,
                )
            else:
                queries = await self._transformer.transform(enriched)

            logger.debug(
                "Generated %d queries for grounded retrieval: %s",
                len(queries), queries,
            )
            return queries
        except Exception:
            logger.warning(
                "Query generation failed, falling back to raw user message",
                exc_info=True,
            )
            return [user_message]

    @staticmethod
    def _build_conversation_context(
        messages: list[dict[str, Any]],
        cfg: Any,
    ) -> str:
        """Build a conversation context string for the transformer.

        Uses ``cfg.max_context_turns`` to determine the message window
        (each turn is roughly 2 messages — user + assistant).

        Args:
            messages: Full conversation message list.
            cfg: Intent config with ``use_conversation_context`` and
                ``max_context_turns`` fields.

        Returns:
            Formatted context string, or empty string if disabled.
        """
        if not cfg.use_conversation_context or len(messages) <= 1:
            return ""

        # max_context_turns counts conversation turns; each turn is
        # roughly 2 messages (user + assistant), so double it for the
        # message window.  Exclude the current (last) message.
        window = cfg.max_context_turns * 2
        recent = messages[-window - 1:-1] if len(messages) > window + 1 else messages[:-1]
        parts: list[str] = []
        for m in recent:
            # Prefer raw_content (unaugmented by KB/memory injection)
            # so query generation sees the user's actual words, not
            # middleware-prepended context chunks.
            content = (
                m.get("metadata", {}).get("raw_content")
                or m.get("content", "")
            )
            if content:
                parts.append(f"{m.get('role', 'user')}: {content}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Retrieval (private)
    # ------------------------------------------------------------------

    async def _retrieve_from_sources(
        self,
        intent: RetrievalIntent,
        user_message: str = "",
        llm: Any | None = None,
    ) -> dict[str, list[SourceResult]]:
        """Execute intent against all configured sources.

        Sources with a ``topic_index`` attribute use topic-index
        retrieval (heading-tree or cluster-based).  Sources without
        it use standard text_queries from the intent.

        Args:
            intent: Structured retrieval intent.
            user_message: Raw user message for topic-index resolution.
            llm: LLM provider for topic-index strategies that need
                classification (e.g. heading selection).
        """
        if not self._sources:
            logger.warning("Grounded strategy has no sources configured — returning empty results")
            return {}

        cfg = self._config.retrieval
        results_by_source: dict[str, list[SourceResult]] = {}

        for source in self._sources:
            try:
                topic_index = getattr(source, "topic_index", None)
                if topic_index is not None:
                    results = await topic_index.resolve(
                        user_message,
                        llm=llm,
                        top_k=cfg.top_k,
                        intent=intent,
                    )
                    # Fall back to standard text queries if topic index
                    # found no results (e.g. vocabulary gap, no seed matches)
                    if not results:
                        logger.info(
                            "Topic index returned empty for source '%s', "
                            "falling back to standard text query retrieval",
                            source.name,
                        )
                        results = await source.query(
                            intent,
                            top_k=cfg.top_k,
                            score_threshold=cfg.score_threshold,
                        )
                else:
                    results = await source.query(
                        intent,
                        top_k=cfg.top_k,
                        score_threshold=cfg.score_threshold,
                    )
                results_by_source[source.name] = results
            except Exception:
                logger.warning(
                    "Source '%s' query failed, skipping",
                    source.name, exc_info=True,
                )

        return results_by_source

    def _merge_source_results(
        self,
        results_by_source: dict[str, list[SourceResult]],
    ) -> list[SourceResult]:
        """Merge results from multiple sources via weighted round-robin.

        Each source contributes ``weight`` results per round-robin cycle
        (default 1).  A source with weight 3 gets 3 results per round
        vs 1 for weight 1, giving it proportionally more representation.
        Within each source, results are already sorted by relevance.
        """
        if not results_by_source:
            return []

        if len(results_by_source) == 1:
            return next(iter(results_by_source.values()))

        # Weighted round-robin interleave
        merged: list[SourceResult] = []
        iterators = {name: iter(results) for name, results in results_by_source.items()}
        exhausted: set[str] = set()

        while len(exhausted) < len(iterators):
            for name, it in iterators.items():
                if name in exhausted:
                    continue
                weight = self._source_weights.get(name, 1)
                for _ in range(weight):
                    try:
                        merged.append(next(it))
                    except StopIteration:
                        exhausted.add(name)
                        break

        if not self._config.retrieval.deduplicate:
            return merged

        # Deduplicate by (source_name, source_id)
        seen: set[tuple[str, str]] = set()
        unique: list[SourceResult] = []
        for r in merged:
            key = (r.source_name, r.source_id)
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    def _format_source_results(
        self,
        results: list[SourceResult],
    ) -> str:
        """Format SourceResult instances for the synthesis prompt.

        When results carry ``cluster_id`` metadata (from a clustering
        processor), they are grouped into ``<cluster>`` XML tags with
        label and query relevance attributes.  Otherwise, the standard
        flat formatting is used.
        """
        if not results:
            return ""

        # Check if results have cluster annotations
        has_clusters = any(
            r.metadata.get("cluster_id", -1) >= 0 for r in results
        )
        if has_clusters:
            return self._format_clustered_results(results)

        return self._format_flat_results(results)

    def _format_flat_results(self, results: list[SourceResult]) -> str:
        """Format results as a flat list (standard path)."""
        result_dicts = [r.to_dict() for r in results]

        if self._merger is not None:
            # Only merge vector_kb results (chunks with heading paths)
            vector_results = [d for d in result_dicts if d.get("source_type") == "vector_kb"]
            other_results = [d for d in result_dicts if d.get("source_type") != "vector_kb"]

            parts: list[str] = []
            if vector_results:
                merged = self._merger.merge(vector_results)
                parts.append(self._formatter.format_merged(merged))
            if other_results:
                parts.append(self._formatter.format(other_results))
            return "\n\n---\n\n".join(parts)

        return self._formatter.format(result_dicts)

    def _format_clustered_results(self, results: list[SourceResult]) -> str:
        """Format results grouped by cluster with XML tags."""
        from collections import defaultdict

        # Group by cluster_id, preserving order
        clusters: dict[int, list[SourceResult]] = defaultdict(list)
        unclustered: list[SourceResult] = []
        for r in results:
            cid = r.metadata.get("cluster_id", -1)
            if cid >= 0:
                clusters[cid].append(r)
            else:
                unclustered.append(r)

        # Sort clusters by query_score (highest first).  QueryClusterScorer
        # sets cluster_query_score uniformly for all members of a cluster,
        # so reading members[0] is correct.  When no scorer has run, all
        # clusters default to 0.0 and retain their original order.
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda item: item[1][0].metadata.get("cluster_query_score", 0.0),
            reverse=True,
        )

        parts: list[str] = []
        for cid, members in sorted_clusters:
            label = members[0].metadata.get("cluster_label", f"cluster_{cid}")
            query_score = members[0].metadata.get("cluster_query_score", 0.0)
            result_dicts = [r.to_dict() for r in members]
            formatted = self._formatter.format(result_dicts)
            parts.append(
                f'<cluster id="{cid}" label="{label}" '
                f'query_relevance="{query_score:.2f}">\n'
                f"{formatted}\n"
                f"</cluster>"
            )

        if unclustered:
            result_dicts = [r.to_dict() for r in unclustered]
            parts.append(self._formatter.format(result_dicts))

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        context: str,
        manager: Any,
        provenance: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Synthesize a response using the resolved synthesis style.

        Delegates style resolution and artifact preparation to
        :meth:`resolve_synthesis`, then handles buffered delivery only.
        For structured style, no LLM call is needed — the template
        output is the response.  For conversational and hybrid, the
        LLM generates a response which is then formatted via
        :meth:`SynthesisPlan.apply_to_response`.
        """
        plan = self.resolve_synthesis(context, manager, provenance)

        if plan.effective_style == "structured":
            return plan.apply_to_response(None)

        # conversational or hybrid — LLM synthesis
        response = await manager.complete(
            system_prompt_override=plan.system_prompt,
            **kwargs,
        )

        return plan.apply_to_response(response)

    # ------------------------------------------------------------------
    # Shared synthesis resolution (used by both buffered and streaming)
    # ------------------------------------------------------------------

    def resolve_synthesis(
        self,
        context: str,
        manager: Any,
        provenance: dict[str, Any],
    ) -> SynthesisPlan:
        """Resolve the effective synthesis style and pre-compute artifacts.

        This is the single shared code path consumed by both
        :meth:`_synthesize` (buffered) and :meth:`stream_generate`
        (streaming).  Only the delivery mechanism differs.
        """
        style = self._resolve_effective_style(manager, provenance)

        template_text: str | None = None
        system_prompt: str | None = None

        if style in ("structured", "hybrid"):
            messages = manager.get_messages()
            user_message = self._extract_user_message(messages)
            try:
                template_text = self._render_provenance_output(
                    context, provenance, user_message, manager.metadata,
                )
            except Exception:
                logger.warning(
                    "Provenance template render failed, falling back "
                    "to conversational style",
                    exc_info=True,
                )
                style = "conversational"
                template_text = None

        if style in ("conversational", "hybrid"):
            system_prompt = self.build_synthesis_system_prompt(
                context, manager.system_prompt,
            )

        return SynthesisPlan(
            effective_style=style,
            template_text=template_text,
            system_prompt=system_prompt,
        )

    def _resolve_effective_style(
        self,
        manager: Any,
        provenance: dict[str, Any],
    ) -> str:
        """Resolve the effective synthesis style for this turn.

        Resolution cascade (highest to lowest priority):

        1. Per-turn ``output_style`` from intent extraction ``raw_data``.
        2. Session-level ``manager.metadata["synthesis_style"]``.
        3. Config-level ``synthesis.style``.
        4. Legacy ``mode`` mapping (``llm`` → conversational,
           ``template`` → structured).
        5. Default: ``"conversational"``.
        """
        # 1. Per-turn from intent extraction
        intent_data = provenance.get("intent", {})
        raw_data = intent_data.get("raw_data")
        if isinstance(raw_data, dict):
            per_turn = raw_data.get("output_style")
            if per_turn in _VALID_STYLES:
                logger.debug("Using per-turn synthesis style: %s", per_turn)
                return per_turn

        # 2. Session-level
        session_style = manager.metadata.get("synthesis_style")
        if session_style in _VALID_STYLES:
            logger.debug(
                "Using session-level synthesis style: %s", session_style,
            )
            return session_style

        # 3. Config-level style
        config_style = self._config.synthesis.style
        if config_style in _VALID_STYLES:
            return config_style

        # 4. Legacy mode mapping
        if self._config.synthesis.mode == "template":
            return "structured"

        # 5. Default
        return "conversational"

    def _render_provenance_output(
        self,
        context: str,
        provenance: dict[str, Any],
        user_message: str,
        metadata: dict[str, Any],
    ) -> str:
        """Render provenance-based output for structured or hybrid styles.

        Template resolution order:

        1. Custom ``synthesis.template`` from config (backward compat with
           legacy ``mode: template``).
        2. ``self._provenance_template`` — pre-compiled from
           ``synthesis.provenance_template`` config or the built-in
           ``DEFAULT_PROVENANCE_TEMPLATE``.

        Template variables:
            ``results``: List of result dicts from provenance.
            ``results_by_source``: Results grouped by source name.
            ``context``: The formatted context string.
            ``message``: The user's message.
            ``metadata``: Conversation metadata.
            ``intent``: The resolved intent dict.
            ``tool_executions``: List of tool execution records (hybrid
                strategy only; empty list for pure grounded).
        """
        # Custom synthesis template takes precedence (backward compat)
        template = self._synthesis_template or self._provenance_template

        return template.render(
            results=provenance.get("results", []),
            results_by_source=provenance.get("results_by_source", {}),
            context=context,
            message=user_message,
            metadata=metadata,
            intent=provenance.get("intent", {}),
            tool_executions=provenance.get("tool_executions", []),
        )

    def build_synthesis_system_prompt(
        self,
        kb_context: str,
        original_system_prompt: str,
    ) -> str:
        """Build the system prompt for the synthesis LLM call."""
        cfg = self._config.synthesis
        parts = [original_system_prompt]

        if kb_context:
            parts.append(
                "\n\n<knowledge_base>\n"
                f"{kb_context}\n"
                "</knowledge_base>"
            )

        grounding_lines = [
            "\nBase your response on the knowledge base content provided above.",
        ]
        if cfg.require_citations:
            cite_what = "section heading" if cfg.citation_format == "section" else "source file"
            grounding_lines.append(
                f"Cite the relevant {cite_what} for each claim.",
            )
        if cfg.allow_parametric == "bridge":
            grounding_lines.append(
                "You may connect and synthesize concepts across the "
                "retrieved content. Identify relationships between "
                "different sections and sources when they help answer "
                "the question. Do not introduce facts from outside "
                "the knowledge base -- only synthesize across what "
                "is provided.",
            )
        elif not cfg.allow_parametric:
            grounding_lines.append(
                "If the knowledge base content does not contain sufficient "
                "information to fully answer the question, explicitly state "
                "what is missing. Do not fill gaps with information from "
                "outside the knowledge base.",
            )
        else:
            grounding_lines.append(
                "You may supplement with general knowledge when the "
                "knowledge base is insufficient, but clearly distinguish "
                "KB-grounded claims from general knowledge.",
            )
        parts.append(" ".join(grounding_lines))

        if cfg.instruction:
            parts.append(cfg.instruction)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_user_message(messages: list[dict[str, Any]]) -> str:
        """Get the last user message from conversation history."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                metadata = msg.get("metadata", {})
                return str(metadata.get("raw_content", msg.get("content", "")))
        return ""

    @staticmethod
    def _build_provenance(
        *,
        intent: RetrievalIntent,
        results_by_source: dict[str, list[SourceResult]],
        all_results: list[SourceResult],
        intent_resolution_time_ms: float,
        retrieval_time_ms: float,
    ) -> dict[str, Any]:
        """Build the provenance record for this turn."""
        prov_by_source: dict[str, list[dict[str, Any]]] = {}
        for source_name, results in results_by_source.items():
            prov_by_source[source_name] = [
                {
                    "source_id": r.source_id,
                    "source_type": r.source_type,
                    "relevance": r.relevance,
                    "text": r.content,
                    "text_preview": r.content[:120],
                    "metadata": r.metadata,
                }
                for r in results
            ]

        total_raw = sum(len(v) for v in results_by_source.values())

        return {
            "intent": {
                "mode": "resolved",
                "text_queries": intent.text_queries,
                "filters": intent.filters,
                "scope": intent.scope,
                "raw_data": intent.raw_data,
            },
            "results_by_source": prov_by_source,
            "results": [
                {
                    "source_id": r.source_id,
                    "source_name": r.source_name,
                    "source_type": r.source_type,
                    "relevance": r.relevance,
                    "text": r.content,
                    "text_preview": r.content[:120],
                }
                for r in all_results
            ],
            "total_results": total_raw,
            "deduplicated_to": len(all_results),
            "retrieval_time_ms": round(retrieval_time_ms, 1),
            "intent_resolution_time_ms": round(intent_resolution_time_ms, 1),
        }

    @staticmethod
    def _store_provenance(
        manager: Any,
        provenance: dict[str, Any],
    ) -> None:
        """Store provenance on the manager with turn-level history.

        Sets two keys on ``manager.metadata``:

        - ``retrieval_provenance``: The current turn's provenance (latest).
        - ``retrieval_provenance_history``: Append-only list of all turns'
          provenance records.
        """
        manager.metadata["retrieval_provenance"] = provenance
        history = manager.metadata.setdefault("retrieval_provenance_history", [])
        history.append(provenance)
