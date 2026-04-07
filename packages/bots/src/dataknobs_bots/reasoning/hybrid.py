"""Hybrid reasoning strategy — grounded retrieval + ReAct tool use.

Composes :class:`GroundedReasoning` (mandatory KB retrieval) with
:class:`ReActReasoning` (optional tool execution) into a single
pipeline:

1. **Grounded phase** — deterministic multi-source retrieval via
   :meth:`GroundedReasoning.retrieve_context`.  Always executes.
2. **Context injection** — retrieved KB context is injected into the
   system prompt so the LLM sees it alongside tool definitions.
3. **ReAct phase** — LLM decides whether/which tools to call.  If no
   tools are available or the LLM makes no tool calls, this degrades
   gracefully to a simple KB-augmented completion.
4. **Post-ReAct synthesis** — optional template-based formatting of
   the combined provenance (KB results + tool executions) using the
   grounded strategy's synthesis styles (conversational / structured /
   hybrid).
5. **Provenance merge** — grounded retrieval provenance is extended
   with tool execution records and stored in conversation metadata.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from dataknobs_bots.reasoning.base import ReasoningStrategy, StrategyCapabilities
from dataknobs_bots.reasoning.grounded import GroundedReasoning
from dataknobs_bots.reasoning.grounded_config import GroundedSynthesisConfig
from dataknobs_bots.reasoning.hybrid_config import HybridReasoningConfig
from dataknobs_bots.reasoning.react import ReActReasoning

logger = logging.getLogger(__name__)


class HybridReasoning(ReasoningStrategy):
    """Reasoning strategy combining grounded retrieval with ReAct tool use.

    Holds a :class:`GroundedReasoning` instance for the mandatory
    retrieval phase and a :class:`ReActReasoning` instance for the
    optional tool-use phase.  Neither child is visible to ``DynaBot``
    directly — ``HybridReasoning`` is the sole registered strategy.

    When no tools are provided, the ReAct phase degrades to a simple
    ``manager.complete()`` call — effectively becoming "grounded
    retrieval + KB-augmented LLM completion."

    Args:
        config: Hybrid reasoning configuration.

    Example:
        .. code-block:: python

            config = HybridReasoningConfig.from_dict({
                "grounded": {
                    "intent": {"mode": "extract", "num_queries": 3},
                    "retrieval": {"top_k": 5},
                },
                "react": {"max_iterations": 5},
            })
            strategy = HybridReasoning(config=config)
    """

    @classmethod
    def capabilities(cls) -> StrategyCapabilities:
        """Hybrid manages sources (grounded) and tools (react)."""
        return StrategyCapabilities(manages_sources=True, manages_tools=True)

    @classmethod
    def get_source_configs(cls, config: dict[str, Any]) -> list[dict[str, Any]]:
        """Sources live under the ``"grounded"`` sub-key for hybrid.

        Warns if a top-level ``"sources"`` key is present, since it is
        likely misplaced.
        """
        if config.get("sources"):
            logger.warning(
                "Hybrid strategy: top-level 'sources' key ignored; "
                "sources must be under 'grounded.sources'.",
            )
        return config.get("grounded", {}).get("sources", [])

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> HybridReasoning:
        """Create HybridReasoning from a configuration dict.

        Args:
            config: Configuration dict (passed to
                ``HybridReasoningConfig.from_dict``).
            **kwargs: Optional ``knowledge_base`` — auto-wrapped via
                the grounded child's ``set_knowledge_base``.

        Returns:
            Configured HybridReasoning instance.
        """
        hybrid_config = HybridReasoningConfig.from_dict(config)
        strategy = cls(config=hybrid_config)
        # Auto-wrap knowledge_base — same guard as GroundedReasoning.
        # Check the grounded child's sources to avoid double-wrapping
        # when the config already declares a vector_kb source.
        knowledge_base = kwargs.get("knowledge_base")
        has_vector_kb_source = any(
            s.source_type == "vector_kb"
            for s in hybrid_config.grounded.sources
        )
        if knowledge_base is not None and not has_vector_kb_source:
            strategy.set_knowledge_base(knowledge_base)
        return strategy

    def __init__(self, *, config: HybridReasoningConfig) -> None:
        super().__init__(greeting_template=config.greeting_template)
        self._config = config

        self._grounded = GroundedReasoning(config=config.grounded)
        self._react = ReActReasoning(
            max_iterations=config.react_max_iterations,
            verbose=config.react_verbose,
            store_trace=config.react_store_trace,
        )

    # ------------------------------------------------------------------
    # Source / knowledge base management (delegated to grounded)
    # ------------------------------------------------------------------

    def set_knowledge_base(self, knowledge_base: Any) -> None:
        """Set the default knowledge base on the grounded child."""
        self._grounded.set_knowledge_base(knowledge_base)

    def add_source(self, source: Any) -> None:
        """Add a grounded source to the retrieval phase."""
        self._grounded.add_source(source)

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def providers(self) -> dict[str, Any]:
        """Return providers from both child strategies."""
        result = self._grounded.providers()
        result.update(self._react.providers())
        return result

    def set_provider(self, role: str, provider: Any) -> bool:
        """Inject provider into all accepting children.

        Both children are given the opportunity to accept the provider
        for the given role — no early return after the first acceptance.
        """
        grounded_accepted = self._grounded.set_provider(role, provider)
        react_accepted = self._react.set_provider(role, provider)
        return grounded_accepted or react_accepted

    async def close(self) -> None:
        """Release resources held by both child strategies."""
        await self._grounded.close()
        await self._react.close()

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
        """Execute the hybrid pipeline: grounded retrieval → ReAct tool loop.

        1. Run deterministic KB retrieval (always executes).
        2. Inject retrieved context into the system prompt.
        3. Run ReAct tool loop (may be zero iterations if no tools/calls).
        4. Collect tool executions from ReAct.
        5. Merge provenance (grounded + tool executions) — must happen
           before synthesis so templates can render tool execution records.
        6. Apply post-ReAct synthesis formatting if configured.

        Args:
            manager: Conversation manager (ReasoningManagerProtocol).
            llm: Bot's main LLM provider.
            tools: Optional tools for the ReAct phase.
            **kwargs: Forwarded to ``manager.complete()``.
        """
        # Defensive clear — same pattern as ReActReasoning.generate().
        # Prevents accumulation if caller skips get_and_clear_tool_executions().
        self._tool_executions.clear()

        # Phase 1: Grounded retrieval (always runs)
        context, provenance = await self._grounded.retrieve_context(manager, llm)

        # Phase 2: Context injection via system_prompt_override
        augmented_prompt = self._build_augmented_prompt(manager, context)
        kwargs["system_prompt_override"] = augmented_prompt

        # Phase 3: ReAct tool loop
        response = await self._react.generate(
            manager, llm, tools=tools, **kwargs,
        )

        # Phase 4: Collect tool executions from ReAct
        react_executions = self._react.get_and_clear_tool_executions()
        self._tool_executions.extend(react_executions)

        # Phase 5: Merge provenance
        self._merge_provenance(manager, provenance, react_executions)

        # Phase 6: Post-ReAct synthesis formatting
        response = self._apply_post_react_synthesis(
            response, context, manager, provenance,
            system_prompt=augmented_prompt,
        )

        return response

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
        """Stream the hybrid pipeline.

        The grounded retrieval phase is non-streaming (fast).  For the
        response phase:

        - **No tools provided:** delegate to ``manager.stream_complete()``
          for true token streaming with KB-augmented context.
        - **Tools provided:** run the full ReAct ``generate()`` loop
          (buffered — can't stream mid-loop) and yield the final
          response.

        Yields:
            LLM response chunks.
        """
        self._tool_executions.clear()

        # Phase 1: Grounded retrieval
        context, provenance = await self._grounded.retrieve_context(manager, llm)

        # Phase 2: Context injection
        augmented_prompt = self._build_augmented_prompt(manager, context)
        kwargs["system_prompt_override"] = augmented_prompt

        if tools:
            # Tools available — must run full ReAct loop (buffered)
            response = await self._react.generate(
                manager, llm, tools=tools, **kwargs,
            )
            react_executions = self._react.get_and_clear_tool_executions()
            self._tool_executions.extend(react_executions)
            self._merge_provenance(manager, provenance, react_executions)

            response = self._apply_post_react_synthesis(
                response, context, manager, provenance,
                system_prompt=augmented_prompt,
            )
            yield response
        else:
            # No tools — resolve synthesis style to decide streaming approach
            self._merge_provenance(manager, provenance, [])
            plan = self._grounded.resolve_synthesis(
                context, manager, provenance,
                system_prompt=augmented_prompt,
            )

            if plan.effective_style == "structured":
                # Structured: template IS the response — yield as single chunk
                yield plan.apply_to_response(None)
            elif plan.effective_style == "hybrid":
                # Hybrid: stream LLM narrative, then yield template appendix
                async for chunk in manager.stream_complete(**kwargs):
                    yield chunk
                from dataknobs_llm import LLMResponse as _LLMResponse
                if plan.template_text:
                    yield _LLMResponse(
                        content="\n\n" + plan.template_text,
                        model="template",
                        finish_reason="stop",
                    )
            else:
                # Conversational: true token streaming
                async for chunk in manager.stream_complete(**kwargs):
                    yield chunk

    # ------------------------------------------------------------------
    # Greeting
    # ------------------------------------------------------------------

    async def greet(
        self,
        manager: Any,
        llm: Any,
        *,
        initial_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any | None:
        """Generate an initial greeting.

        Delegates to the grounded strategy's greet if it has a greeting
        template.  Falls back to the base class greeting_template.
        """
        grounded_greeting = await self._grounded.greet(
            manager, llm, initial_context=initial_context, **kwargs,
        )
        if grounded_greeting is not None:
            return grounded_greeting
        return await super().greet(
            manager, llm, initial_context=initial_context, **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_augmented_prompt(self, manager: Any, context: str) -> str:
        """Build system prompt with KB context via grounded's prompt builder.

        Delegates to :meth:`GroundedReasoning.build_synthesis_system_prompt`
        so that grounding instructions (citation format, parametric knowledge
        policy, custom instructions) from the grounded config are honoured.
        """
        base_prompt = manager.system_prompt or ""
        if not context.strip():
            return base_prompt
        return self._grounded.build_synthesis_system_prompt(context, base_prompt)

    def _merge_provenance(
        self,
        manager: Any,
        provenance: dict[str, Any],
        react_executions: list[Any],
    ) -> None:
        """Extend provenance with tool executions and store in metadata.

        Enriches the provenance dict with tool execution records, then
        delegates storage to :meth:`GroundedReasoning.store_provenance`
        to keep the metadata key convention in a single place.
        """
        if not self._config.store_provenance:
            return

        provenance["tool_executions"] = [
            {
                "tool_name": te.tool_name,
                "parameters": te.parameters,
                "result": str(te.result) if te.result is not None else None,
                "error": te.error,
                "duration_ms": te.duration_ms,
            }
            for te in react_executions
        ]
        GroundedReasoning.store_provenance(manager, provenance)

    def _apply_post_react_synthesis(
        self,
        response: Any,
        context: str,
        manager: Any,
        provenance: dict[str, Any],
        *,
        system_prompt: str | None = None,
        synthesis_config: GroundedSynthesisConfig | None = None,
    ) -> Any:
        """Apply post-ReAct synthesis formatting if configured.

        Delegates to :meth:`SynthesisPlan.apply_to_response` — the
        shared dispatch logic that handles all three synthesis styles.
        The provenance dict at this point includes ``tool_executions``,
        so templates can format both KB results and tool outputs.

        Args:
            response: LLM response to format.
            context: KB context string.
            manager: Conversation manager.
            provenance: Retrieval provenance dict.
            system_prompt: Pre-built synthesis system prompt.  When
                provided, avoids rebuilding the prompt inside
                :meth:`GroundedReasoning.resolve_synthesis`.
            synthesis_config: Optional override for synthesis settings,
                forwarded to :meth:`GroundedReasoning.resolve_synthesis`.
        """
        plan = self._grounded.resolve_synthesis(
            context, manager, provenance,
            system_prompt=system_prompt,
            synthesis_config=synthesis_config,
        )
        return plan.apply_to_response(response)
