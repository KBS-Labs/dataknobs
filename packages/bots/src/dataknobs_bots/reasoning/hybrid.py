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

from dataknobs_llm import LLMResponse

from dataknobs_bots.reasoning.base import ReasoningStrategy
from dataknobs_bots.reasoning.grounded import GroundedReasoning
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
        """Delegate provider injection to children."""
        if self._grounded.set_provider(role, provider):
            return True
        return self._react.set_provider(role, provider)

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
            )
            yield response
        else:
            # No tools — true token streaming
            async for chunk in manager.stream_complete(**kwargs):
                yield chunk

            if self._config.store_provenance:
                self._merge_provenance(manager, provenance, [])

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
        """Extend provenance with tool executions and store in metadata."""
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
        manager.metadata.setdefault(
            "retrieval_provenance", [],
        ).append(provenance)

    def _apply_post_react_synthesis(
        self,
        response: Any,
        context: str,
        manager: Any,
        provenance: dict[str, Any],
    ) -> Any:
        """Apply post-ReAct synthesis formatting if configured.

        Uses the grounded strategy's synthesis resolution to determine
        the effective style:

        - ``conversational`` (default): return ReAct response as-is.
        - ``structured``: render provenance template, discard ReAct
          response.
        - ``hybrid``: append provenance template to ReAct response.

        The provenance dict at this point includes ``tool_executions``,
        so templates can format both KB results and tool outputs.
        """
        plan = self._grounded.resolve_synthesis(context, manager, provenance)

        if plan.effective_style == "conversational":
            return response

        if plan.effective_style == "structured":
            return LLMResponse(
                content=plan.template_text or "",
                model="template",
                finish_reason="stop",
            )

        # hybrid style — LLM narrative + template appendix
        if plan.template_text:
            combined = (response.content or "") + "\n\n" + plan.template_text
            return LLMResponse(
                content=combined,
                model=getattr(response, "model", "unknown"),
                finish_reason=getattr(response, "finish_reason", "stop"),
            )

        return response
