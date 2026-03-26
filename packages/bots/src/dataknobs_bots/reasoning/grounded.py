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

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import jinja2

from dataknobs_data.sources.base import (
    GroundedSource,
    RetrievalIntent,
    SourceResult,
)

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
        self._query_provider = query_provider
        self._merger = ChunkMerger(MergerConfig()) if config.retrieval.merge_adjacent else None
        self._formatter = ContextFormatter(FormatterConfig(
            include_scores=False,
            include_source=True,
            group_by_source=True,
        ))

        # Query generation components (extract mode only)
        self._transformer: QueryTransformer | None = None
        self._expander: ContextualExpander | None = None
        if config.intent.mode == "extract":
            self._transformer = QueryTransformer(
                TransformerConfig(
                    enabled=True,
                    num_queries=config.intent.num_queries,
                    domain_context=config.intent.domain_context,
                ),
                provider=query_provider,
            )
            if config.intent.expand_ambiguous_queries:
                self._expander = ContextualExpander(
                    max_context_turns=config.intent.max_context_turns,
                    include_assistant=config.intent.include_assistant_context,
                )

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
        """
        from dataknobs_bots.knowledge.sources.vector import VectorKnowledgeSource

        # Replace any existing vector KB source
        self._sources = [
            s for s in self._sources
            if s.source_type != "vector_kb"
        ]
        self._sources.insert(0, VectorKnowledgeSource(kb))

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
                Phase 5 hybrid strategy will use tools.
            **kwargs: Forwarded to ``manager.complete()`` (temperature, etc.).
        """
        context, provenance = await self.retrieve_context(manager, llm)

        response = self._synthesize(context, manager, provenance, **kwargs)
        if isinstance(response, AsyncIterator):
            # Should not happen in non-streaming, but guard
            raise TypeError("_synthesize returned async iterator in non-streaming mode")

        result = await response

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
        Only LLM synthesis streams; template synthesis yields one chunk.

        Yields:
            LLM stream chunks from the synthesis phase.
        """
        context, provenance = await self.retrieve_context(manager, llm)

        if self._config.synthesis.mode == "template":
            # Template synthesis — yield single response
            from dataknobs_llm import LLMResponse

            messages = manager.get_messages()
            user_message = self._extract_user_message(messages)
            text = self._render_synthesis_template(
                context, provenance, user_message, manager.metadata,
            )
            yield LLMResponse(content=text, model="template", finish_reason="stop")
        else:
            synthesis_prompt = self._build_synthesis_system_prompt(
                context, manager.system_prompt,
            )
            async for chunk in manager.stream_complete(
                system_prompt_override=synthesis_prompt,
                **kwargs,
            ):
                yield chunk

        if self._config.store_provenance:
            self._store_provenance(manager, provenance)

    # ------------------------------------------------------------------
    # Public retrieval pipeline (composable for Phase 5 hybrid)
    # ------------------------------------------------------------------

    async def retrieve_context(
        self,
        manager: Any,
        llm: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Run intent resolution + multi-source retrieval + processing.

        Returns the formatted context string and a provenance dict.
        This is the public entry point that a future HybridReasoning
        can call to obtain grounded context before entering a ReAct loop.

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
        results_by_source = await self._retrieve_from_sources(intent)
        retrieval_ms = (time.monotonic() - t0) * 1000

        # Merge results across sources
        all_results = self._merge_source_results(results_by_source)

        # Format for synthesis
        formatted_context = self._format_source_results(all_results)

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
        else:
            # Default: extract mode (LLM query generation)
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

        env = jinja2.Environment(undefined=jinja2.Undefined)
        rendered = env.from_string(cfg.template).render(
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

    async def _generate_queries(
        self,
        user_message: str,
        messages: list[dict[str, Any]],
        llm: Any,
    ) -> list[str]:
        """Generate search queries via :class:`QueryTransformer`.

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

        Args:
            messages: Full conversation message list.
            cfg: Intent config with ``use_conversation_context`` flag.

        Returns:
            Formatted context string, or empty string if disabled.
        """
        if not cfg.use_conversation_context or len(messages) <= 1:
            return ""

        recent = messages[-5:-1] if len(messages) > 5 else messages[:-1]
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in recent
            if m.get("content")
        )

    # ------------------------------------------------------------------
    # Retrieval (private)
    # ------------------------------------------------------------------

    async def _retrieve_from_sources(
        self,
        intent: RetrievalIntent,
    ) -> dict[str, list[SourceResult]]:
        """Execute intent against all configured sources."""
        if not self._sources:
            logger.warning("Grounded strategy has no sources configured — returning empty results")
            return {}

        cfg = self._config.retrieval
        results_by_source: dict[str, list[SourceResult]] = {}

        for source in self._sources:
            try:
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
        """Merge results from multiple sources via round-robin interleaving.

        Round-robin prevents one prolific source from dominating.
        Within each source, results are already sorted by relevance.
        """
        if not results_by_source:
            return []

        if len(results_by_source) == 1:
            return next(iter(results_by_source.values()))

        # Round-robin interleave
        merged: list[SourceResult] = []
        iterators = {name: iter(results) for name, results in results_by_source.items()}
        exhausted: set[str] = set()

        while len(exhausted) < len(iterators):
            for name, it in iterators.items():
                if name in exhausted:
                    continue
                try:
                    merged.append(next(it))
                except StopIteration:
                    exhausted.add(name)

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
        """Format SourceResult instances for the synthesis prompt."""
        if not results:
            return ""

        # Convert to dict format for existing formatter infrastructure
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
        """Synthesize a response — LLM or template mode."""
        if self._config.synthesis.mode == "template":
            from dataknobs_llm import LLMResponse

            messages = manager.get_messages()
            user_message = self._extract_user_message(messages)
            text = self._render_synthesis_template(
                context, provenance, user_message, manager.metadata,
            )
            return LLMResponse(content=text, model="template", finish_reason="stop")

        # LLM synthesis (default)
        synthesis_prompt = self._build_synthesis_system_prompt(
            context, manager.system_prompt,
        )
        return await manager.complete(
            system_prompt_override=synthesis_prompt,
            **kwargs,
        )

    def _render_synthesis_template(
        self,
        context: str,
        provenance: dict[str, Any],
        user_message: str,
        metadata: dict[str, Any],
    ) -> str:
        """Render the synthesis Jinja2 template.

        Template variables:
            ``results``: List of result dicts from provenance.
            ``results_by_source``: Results grouped by source name.
            ``context``: The formatted context string.
            ``message``: The user's message.
            ``metadata``: Conversation metadata.
            ``intent``: The resolved intent dict.
        """
        template_str = self._config.synthesis.template
        if not template_str:
            return context  # Fallback: return raw formatted context

        env = jinja2.Environment(undefined=jinja2.Undefined)
        return env.from_string(template_str).render(
            results=provenance.get("results", []),
            results_by_source=provenance.get("results_by_source", {}),
            context=context,
            message=user_message,
            metadata=metadata,
            intent=provenance.get("intent", {}),
        )

    def _build_synthesis_system_prompt(
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
        if not cfg.allow_parametric:
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
