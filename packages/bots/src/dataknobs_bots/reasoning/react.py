"""ReAct (Reasoning + Acting) reasoning strategy."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from dataknobs_common.structured_config import StructuredConfigConsumer
from dataknobs_llm import LLMStreamResponse
from dataknobs_llm.exceptions import ToolsNotSupportedError
from dataknobs_llm.llm.base import LLMResponse
from dataknobs_llm.llm.message_sequence import (
    pair_orphan_tool_calls,
    tool_call_signature,
)
from dataknobs_llm.tools import ToolExecutionContext

from dataknobs_bots.bot.turn import ToolExecution

from .base import ProcessResult, ReasoningStrategy, StrategyCapabilities, TurnHandle
from .react_config import ReActReasoningConfig

logger = logging.getLogger(__name__)


async def _pair_orphan_tool_calls(manager: Any) -> None:
    """Append synthetic ``tool_result``s for any dangling ``tool_use``.

    Thin ``ConversationManager`` adapter over the pure
    :func:`pair_orphan_tool_calls` core.  Invoked on the synthesis branch of
    every ReAct finalize path (i.e. when the loop ended abnormally —
    duplicate break, max iterations, or a DynaBot-level tool-loop timeout —
    rather than returning a stored final answer): reads history via the public
    manager API, runs the pure core, and appends whatever tool results it
    yields so the subsequent ``complete()``/``stream_complete()`` request is
    structurally valid.

    Args:
        manager: Conversation manager whose history is about to be re-sent to
            a synthesis completion call.
    """
    history = await manager.get_history()
    for result in pair_orphan_tool_calls(history):
        await manager.add_message(
            role="tool",
            content=result.content,
            name=result.name,
            tool_call_id=result.tool_call_id,
        )


def _is_truncated_tool_call(response: Any) -> bool:
    """Whether ``response`` is a tool-call turn the provider truncated.

    The provider cut generation off at the token budget *mid-tool-call*
    (Anthropic ``stop_reason == "max_tokens"``, OpenAI
    ``finish_reason == "length"``), so the ``tool_use`` is incomplete — its
    arguments may be missing or malformed even though the call looks
    well-formed.  Executing it would surface downstream as a masked
    "argument required" error, and the model would retry the identical
    oversized call until the duplicate-breaker fires.

    Such a turn is abandoned (not executed) and routed to final synthesis —
    the same terminal handling as a duplicate-tool-call break.  A truncated
    *text* turn (no tool calls) is already terminal (returned to the caller
    as-is) and is deliberately not matched here.  The provider layer has
    already logged the truncation warning; this is the react-layer behavioral
    reaction to :attr:`~dataknobs_llm.LLMResponse.truncated`.

    ``truncated`` is response-level, so a turn carrying multiple ``tool_use``
    blocks is abandoned in full — including any complete blocks — since a
    truncated turn's block completeness (and array parse) cannot be trusted.
    """
    return bool(
        getattr(response, "truncated", False)
        and getattr(response, "tool_calls", None)
    )


@dataclass
class ReActTurnHandle(TurnHandle):
    """ReAct-specific turn handle carrying iteration state.

    Extends :class:`TurnHandle` with fields needed to track the ReAct
    loop across ``process_input`` calls.  Each call to ``process_input``
    corresponds to one iteration of the ReAct loop.

    Attributes:
        iteration: Current iteration index (0-based).
        max_iterations: Maximum number of ReAct iterations.
        prev_tool_calls: Previous iteration's tool calls for duplicate
            detection.  ``None`` on the first iteration.
        trace: Reasoning trace accumulator (``None`` when tracing is
            disabled).
        final_response: Set by ``process_input`` when the LLM returns
            no tool calls (final answer).  Left ``None`` on duplicate
            detection and max iterations — ``finalize_turn`` then
            performs a synthesis LLM call instead of returning directly.
        store_trace: Whether to persist the trace to conversation
            metadata after the loop completes.
        verbose: Whether to use debug-level logging.
    """

    iteration: int = 0
    max_iterations: int = 5
    prev_tool_calls: list[tuple[str, str]] | None = None
    trace: list[dict[str, Any]] | None = None
    final_response: Any | None = None
    store_trace: bool = False
    verbose: bool = False


class ReActReasoning(
    StructuredConfigConsumer[ReActReasoningConfig], ReasoningStrategy
):
    """ReAct (Reasoning + Acting) strategy.

    This strategy implements the ReAct pattern where the LLM:
    1. Reasons about what to do (Thought)
    2. Takes an action (using tools if needed)
    3. Observes the result
    4. Repeats until task is complete

    This is useful for:
    - Multi-step problem solving
    - Tasks requiring tool use
    - Complex reasoning chains

    Attributes:
        max_iterations: Maximum number of reasoning loops
        verbose: Whether to enable debug-level logging
        store_trace: Whether to store reasoning trace in conversation metadata

    Example:
        ```python
        strategy = ReActReasoning(
            max_iterations=5,
            verbose=True,
            store_trace=True
        )
        response = await strategy.generate(
            manager=conversation_manager,
            llm=llm_provider,
            tools=[search_tool, calculator_tool]
        )
        ```
    """

    #: Typed config consumed via the ``StructuredConfigConsumer`` mixin.
    #: Config scalars (``max_iterations``/``verbose``/``store_trace``/
    #: ``greeting_template``) flow through ``CONFIG_CLS``; the injected
    #: runtime collaborators (artifact registry, review executor, context
    #: builder, extra context, prompt refresher) are NOT config — they
    #: travel through the mixin's ``components`` channel
    #: (``cls.from_config({...}, prompt_refresher=fn)``) and are bound in
    #: :meth:`_setup`.
    CONFIG_CLS: ClassVar[type[ReActReasoningConfig]] = ReActReasoningConfig

    @classmethod
    def capabilities(cls) -> StrategyCapabilities:
        """ReAct manages its own tool execution loop."""
        return StrategyCapabilities(manages_tools=True)

    def _setup(self) -> None:
        """Bind scalar config and injected collaborators.

        Scalars come from the typed config; the optional runtime
        collaborators (artifact registry, review executor, context
        builder, extra context, prompt refresher) come from the mixin's
        ``components`` channel and default to ``None`` when not injected.
        """
        config = self.config
        self._greeting_template = config.greeting_template
        self.max_iterations = config.max_iterations
        self.verbose = config.verbose
        self.store_trace = config.store_trace
        self._artifact_registry = self.components.get("artifact_registry")
        self._review_executor = self.components.get("review_executor")
        self._context_builder = self.components.get("context_builder")
        self._extra_context: dict[str, Any] | None = self.components.get(
            "extra_context"
        )
        self._prompt_refresher: Callable[[], str] | None = self.components.get(
            "prompt_refresher"
        )

    @property
    def artifact_registry(self) -> Any | None:
        """Get the artifact registry if configured."""
        return self._artifact_registry

    @property
    def review_executor(self) -> Any | None:
        """Get the review executor if configured."""
        return self._review_executor

    @property
    def context_builder(self) -> Any | None:
        """Get the context builder if configured."""
        return self._context_builder

    # ------------------------------------------------------------------
    # PhasedReasoningProtocol implementation
    # ------------------------------------------------------------------

    async def begin_turn(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> ReActTurnHandle:
        """Phase A: Setup ReAct iteration state.

        Clears stale tool executions, builds extra context for tool
        execution, and returns a :class:`ReActTurnHandle`.  If no tools
        are available, performs a direct LLM call and stores the result
        as ``handle.early_response``.

        Args:
            manager: Conversation manager for this turn.
            llm: LLM provider instance.
            tools: Optional list of available tools.
            **kwargs: Additional generation parameters.

        Returns:
            ReAct turn handle with iteration state initialized.
        """
        handle = ReActTurnHandle(
            manager=manager,
            llm=llm,
            tools=tools,
            kwargs=kwargs,
            max_iterations=self.max_iterations,
            trace=[] if self.store_trace else None,
            store_trace=self.store_trace,
            verbose=self.verbose,
        )

        # Clear stale executions from previous calls.
        self._tool_executions.clear()

        # No-tools fast path — check before building extra_context to
        # avoid unnecessary I/O (context_builder.build may do async work).
        if not tools:
            logger.info(
                "ReAct: No tools available, falling back to simple generation",
                extra={"conversation_id": getattr(manager, "conversation_id", None)},
            )
            handle.early_response = await manager.complete(**kwargs)
            return handle

        # Build static extra_context for tool execution.  These don't
        # change across iterations and are set once on the handle.
        # context_builder is refreshed per-iteration in process_input
        # so tools see updated state after mutations.
        extra: dict[str, Any] = {}
        if self._artifact_registry is not None:
            extra["artifact_registry"] = self._artifact_registry
        if self._review_executor is not None:
            extra["review_executor"] = self._review_executor
        if self._extra_context:
            extra.update(self._extra_context)
        handle.tool_extra_context = extra

        log_level = logging.DEBUG if self.verbose else logging.INFO
        logger.log(
            log_level,
            "ReAct: Starting phased reasoning loop",
            extra={
                "conversation_id": getattr(manager, "conversation_id", None),
                "max_iterations": self.max_iterations,
                "tools_available": len(tools),
            },
        )

        return handle

    async def process_input(
        self,
        handle: TurnHandle,
    ) -> ProcessResult:
        """Phase B: Execute one ReAct iteration.

        Makes a single LLM call with tools.  If the LLM returns tool
        calls, signals DynaBot to execute them and loop back
        (``iterate=True``).  If the LLM returns a final answer,
        stores it on ``handle.final_response``.  On duplicate detection,
        leaves ``final_response`` as ``None`` so ``finalize_turn``
        performs a synthesis call.

        Args:
            handle: ReAct turn handle from ``begin_turn``.

        Returns:
            Process result indicating the iteration outcome.
        """
        if not isinstance(handle, ReActTurnHandle):
            raise TypeError(
                f"Expected ReActTurnHandle, got {type(handle).__name__}"
            )

        log_level = logging.DEBUG if handle.verbose else logging.INFO

        # Max iterations check
        if handle.iteration >= handle.max_iterations:
            logger.log(
                log_level,
                "ReAct: Max iterations reached, generating final response",
                extra={
                    "conversation_id": getattr(
                        handle.manager, "conversation_id", None
                    ),
                    "iterations_used": handle.max_iterations,
                },
            )
            if handle.trace is not None:
                handle.trace.append({"status": "max_iterations_reached"})
                await self._store_trace(handle.manager, handle.trace)
            return ProcessResult(action="max_iterations")

        # Prompt refresh for iterations > 0
        if handle.iteration > 0 and self._prompt_refresher is not None:
            handle.kwargs["system_prompt_override"] = self._prompt_refresher()

        # Refresh conversation_context each iteration so tools see
        # updated state after mutations (e.g. load_from_catalog changing
        # artifact state mid-loop).  Matches generate() behavior.
        if self._context_builder is not None:
            try:
                ctx = await self._context_builder.build(handle.manager)
                handle.tool_extra_context["conversation_context"] = ctx
            except Exception as e:
                logger.warning("Failed to build conversation context: %s", e)
                # Remove stale context from a previous iteration so tools
                # don't silently operate on outdated state.
                handle.tool_extra_context.pop("conversation_context", None)

        iteration_trace: dict[str, Any] = {
            "iteration": handle.iteration + 1,
            "tool_calls": [],
        }

        logger.log(
            log_level,
            "ReAct: Starting iteration",
            extra={
                "conversation_id": getattr(
                    handle.manager, "conversation_id", None
                ),
                "iteration": handle.iteration + 1,
                "max_iterations": handle.max_iterations,
            },
        )

        # LLM call with tools
        try:
            response = await handle.manager.complete(
                tools=handle.tools, **handle.kwargs
            )
        except ToolsNotSupportedError as e:
            logger.error(
                "ReAct: Model '%s' does not support tools — "
                "returning graceful response to user",
                e.model,
                extra={
                    "conversation_id": getattr(
                        handle.manager, "conversation_id", None
                    ),
                },
            )
            return ProcessResult(
                early_response=LLMResponse(
                    content=(
                        "I'm configured to use tools for this task, but my "
                        "current language model doesn't support tool calling. "
                        "Please contact the administrator to update the model "
                        "configuration."
                    ),
                    model=e.model,
                    finish_reason="error",
                ),
                action="tools_not_supported",
            )

        # No tool_calls → final answer
        if not getattr(response, "tool_calls", None):
            logger.log(
                log_level,
                "ReAct: No tool calls in response, finishing",
                extra={
                    "conversation_id": getattr(
                        handle.manager, "conversation_id", None
                    ),
                    "iteration": handle.iteration + 1,
                },
            )
            handle.final_response = response
            if handle.trace is not None:
                iteration_trace["status"] = "completed"
                handle.trace.append(iteration_trace)
                await self._store_trace(handle.manager, handle.trace)
            return ProcessResult(action="final_answer")

        # Truncated mid-tool-call → terminal, not executed.  The tool_use is
        # incomplete; abandon it exactly like a duplicate break (leave
        # final_response=None so finalize_turn pairs the orphan and synthesizes
        # a final answer without tools).  The provider already logged the
        # truncation warning.
        if _is_truncated_tool_call(response):
            logger.warning(
                "ReAct: Response truncated mid-tool-call (token budget) — "
                "abandoning the incomplete tool call and synthesizing a "
                "final answer",
                extra={
                    "conversation_id": getattr(
                        handle.manager, "conversation_id", None
                    ),
                    "iteration": handle.iteration + 1,
                    "tools": [tc.name for tc in response.tool_calls],
                },
            )
            handle.final_response = None  # finalize_turn does the synthesis
            if handle.trace is not None:
                iteration_trace["status"] = "truncated_tool_call"
                handle.trace.append(iteration_trace)
                await self._store_trace(handle.manager, handle.trace)
            return ProcessResult(action="truncated")

        num_tool_calls = len(response.tool_calls)
        logger.log(
            log_level,
            "ReAct: Tool calls requested",
            extra={
                "conversation_id": getattr(
                    handle.manager, "conversation_id", None
                ),
                "iteration": handle.iteration + 1,
                "num_tools": num_tool_calls,
                "tools": [tc.name for tc in response.tool_calls],
            },
        )

        # Duplicate detection — keyed on the shared tool_call_signature so the
        # loop's duplicate-break guard and the orphan-pairing repair agree.
        current_calls = [
            tool_call_signature(tc) for tc in response.tool_calls
        ]

        if (
            handle.prev_tool_calls is not None
            and current_calls == handle.prev_tool_calls
        ):
            logger.warning(
                "ReAct: Duplicate tool calls detected, breaking loop",
                extra={
                    "conversation_id": getattr(
                        handle.manager, "conversation_id", None
                    ),
                    "iteration": handle.iteration + 1,
                    "duplicate_calls": [tc.name for tc in response.tool_calls],
                },
            )
            # No mid-conversation notice is appended here: finalize_turn's
            # _pair_orphan_tool_calls guarantees the abandoned tool_use is
            # paired with a tool_result that carries the "use existing
            # results" guidance inline, at the correct position.  A
            # role="system" append would be hoisted out of the message array
            # by adapters that lift system messages to a top-level param
            # (e.g. Anthropic), leaving the tool_use dangling.
            handle.final_response = None  # finalize_turn does synthesis
            if handle.trace is not None:
                iteration_trace["status"] = "duplicate_tool_calls_detected"
                handle.trace.append(iteration_trace)
                await self._store_trace(handle.manager, handle.trace)
            return ProcessResult(action="duplicate_break")

        handle.prev_tool_calls = current_calls
        handle.iteration += 1

        if handle.trace is not None:
            iteration_trace["status"] = "continued"
            iteration_trace["tool_calls"] = [
                {"name": tc.name, "parameters": tc.parameters}
                for tc in response.tool_calls
            ]
            handle.trace.append(iteration_trace)

        # Signal DynaBot to execute tools, then call process_input again
        return ProcessResult(
            needs_tool_execution=True,
            iterate=True,
            pending_tool_calls=list(response.tool_calls),
            action="tool_calls",
        )

    async def finalize_turn(
        self,
        handle: TurnHandle,
        tool_results: list[ToolExecution] | None = None,
    ) -> Any:
        """Phase C: Return final response or perform synthesis call.

        If ``process_input`` stored a final response on the handle
        (LLM returned no tool calls), returns it directly.  Otherwise
        (max iterations or duplicate break), performs a final LLM call
        without tools to synthesize a response.

        Args:
            handle: ReAct turn handle from ``begin_turn``.
            tool_results: Tool execution records from DynaBot's tool
                loop (unused by ReAct — tool observations are already
                in conversation history).

        Returns:
            LLM response object.
        """
        if not isinstance(handle, ReActTurnHandle):
            raise TypeError(
                f"Expected ReActTurnHandle, got {type(handle).__name__}"
            )

        # If process_input stored a final response, return it
        if handle.final_response is not None:
            return handle.final_response

        # Otherwise: final synthesis (max iterations, duplicate break, or a
        # DynaBot-level tool-loop timeout).  Guarantee no dangling tool_use
        # is left in history before re-sending it to the provider.
        await _pair_orphan_tool_calls(handle.manager)

        if self._prompt_refresher is not None:
            handle.kwargs["system_prompt_override"] = self._prompt_refresher()

        return await handle.manager.complete(**handle.kwargs)

    def stream_finalize_turn(
        self,
        handle: TurnHandle,
        tool_results: list[ToolExecution] | None = None,
    ) -> AsyncIterator[LLMStreamResponse]:
        """Stream Phase C: Return stored response or stream synthesis.

        Streaming counterpart of :meth:`finalize_turn`.  If
        ``process_input`` stored a final response, yields it as a
        single chunk.  Otherwise streams the synthesis call
        token-by-token via ``manager.stream_complete()``.

        Args:
            handle: ReAct turn handle from ``begin_turn``.
            tool_results: Tool execution records from DynaBot's tool
                loop (unused by ReAct).

        Yields:
            :class:`LLMStreamResponse` chunks.
        """
        if not isinstance(handle, ReActTurnHandle):
            raise TypeError(
                f"Expected ReActTurnHandle, got {type(handle).__name__}"
            )
        return self._stream_finalize(handle)

    async def _stream_finalize(
        self,
        handle: ReActTurnHandle,
    ) -> AsyncIterator[LLMStreamResponse]:
        """Inner async generator for stream_finalize_turn."""
        # If process_input stored a final response (always LLMResponse
        # from manager.complete), yield as single chunk.
        if handle.final_response is not None:
            yield LLMStreamResponse(
                delta=handle.final_response.content,
                is_final=True,
                finish_reason="stop",
            )
            return

        # Otherwise: stream synthesis (max iterations, duplicate break, or a
        # DynaBot-level tool-loop timeout).  Guarantee no dangling tool_use
        # is left in history before re-sending it to the provider.
        await _pair_orphan_tool_calls(handle.manager)

        if self._prompt_refresher is not None:
            handle.kwargs["system_prompt_override"] = self._prompt_refresher()

        async for chunk in handle.manager.stream_complete(**handle.kwargs):
            yield chunk

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate response using ReAct loop.

        The ReAct loop:
        1. Generate response (may include tool calls)
        2. If tool calls present, execute them
        3. Add observations to conversation
        4. Repeat until no more tool calls or max iterations

        Args:
            manager: ConversationManager instance
            llm: LLM provider instance
            tools: Optional list of available tools
            **kwargs: Generation parameters

        Returns:
            Final LLM response
        """
        # Clear any stale tool executions from a previous call.
        # Each generate() call should start with a fresh list so
        # concurrent async calls on the same strategy instance don't
        # accumulate records from earlier calls.
        self._tool_executions.clear()

        if not tools:
            # No tools available, fall back to simple generation
            logger.info(
                "ReAct: No tools available, falling back to simple generation",
                extra={"conversation_id": manager.conversation_id},
            )
            return await manager.complete(**kwargs)

        # Initialize trace if enabled
        trace = [] if self.store_trace else None

        # Get log level based on verbose setting
        log_level = logging.DEBUG if self.verbose else logging.INFO

        logger.log(
            log_level,
            "ReAct: Starting reasoning loop",
            extra={
                "conversation_id": manager.conversation_id,
                "max_iterations": self.max_iterations,
                "tools_available": len(tools),
            },
        )

        # Track previous iteration's tool calls for duplicate detection
        prev_tool_calls: list[tuple[str, str]] | None = None

        # ReAct loop
        for iteration in range(self.max_iterations):
            iteration_trace = {
                "iteration": iteration + 1,
                "tool_calls": [],
            }

            logger.log(
                log_level,
                "ReAct: Starting iteration",
                extra={
                    "conversation_id": manager.conversation_id,
                    "iteration": iteration + 1,
                    "max_iterations": self.max_iterations,
                },
            )

            # Generate response with tools
            try:
                response = await manager.complete(tools=tools, **kwargs)
            except ToolsNotSupportedError as e:
                logger.error(
                    "ReAct: Model '%s' does not support tools — "
                    "returning graceful response to user",
                    e.model,
                    extra={"conversation_id": manager.conversation_id},
                )
                return LLMResponse(
                    content=(
                        "I'm configured to use tools for this task, but my "
                        "current language model doesn't support tool calling. "
                        "Please contact the administrator to update the model "
                        "configuration."
                    ),
                    model=e.model,
                    finish_reason="error",
                )

            # Check if we have tool calls
            if not hasattr(response, "tool_calls") or not response.tool_calls:
                # No tool calls, we're done
                logger.log(
                    log_level,
                    "ReAct: No tool calls in response, finishing",
                    extra={
                        "conversation_id": manager.conversation_id,
                        "iteration": iteration + 1,
                    },
                )

                if trace is not None:
                    iteration_trace["status"] = "completed"
                    trace.append(iteration_trace)
                    await self._store_trace(manager, trace)

                return response

            # Truncated mid-tool-call → terminal, not executed.  Abandon the
            # incomplete tool call the same way a duplicate break does: break
            # to the shared orphan-pairing + synthesis after the loop.  The
            # provider already logged the truncation warning.
            if _is_truncated_tool_call(response):
                logger.warning(
                    "ReAct: Response truncated mid-tool-call (token budget) — "
                    "abandoning the incomplete tool call and synthesizing a "
                    "final answer",
                    extra={
                        "conversation_id": manager.conversation_id,
                        "iteration": iteration + 1,
                        "tools": [tc.name for tc in response.tool_calls],
                    },
                )
                if trace is not None:
                    iteration_trace["status"] = "truncated_tool_call"
                    trace.append(iteration_trace)
                    await self._store_trace(manager, trace)
                break

            num_tool_calls = len(response.tool_calls)
            logger.log(
                log_level,
                "ReAct: Executing tool calls",
                extra={
                    "conversation_id": manager.conversation_id,
                    "iteration": iteration + 1,
                    "num_tools": num_tool_calls,
                    "tools": [tc.name for tc in response.tool_calls],
                },
            )

            # Duplicate detection: compare the shared tool_call_signature
            # with the previous iteration to avoid infinite loops
            current_calls = [
                tool_call_signature(tc) for tc in response.tool_calls
            ]

            if prev_tool_calls is not None and current_calls == prev_tool_calls:
                logger.warning(
                    "ReAct: Duplicate tool calls detected, breaking loop",
                    extra={
                        "conversation_id": manager.conversation_id,
                        "iteration": iteration + 1,
                        "duplicate_calls": [tc.name for tc in response.tool_calls],
                    },
                )

                # No mid-conversation notice is appended here: the final
                # synthesis calls _pair_orphan_tool_calls, which pairs the
                # abandoned tool_use with a tool_result carrying the "use
                # existing results" guidance inline.  A role="system" append
                # would be hoisted out of the message array by adapters that
                # lift system messages to a top-level param (e.g. Anthropic),
                # leaving the tool_use dangling.
                if trace is not None:
                    iteration_trace["status"] = "duplicate_tool_calls_detected"
                    trace.append(iteration_trace)
                    await self._store_trace(manager, trace)

                break

            prev_tool_calls = current_calls

            # Build execution context for tools that need it
            tool_context = ToolExecutionContext.from_manager(manager)

            # Extend context with artifact/review infrastructure if available
            extra_context: dict[str, Any] = {}
            if self._artifact_registry is not None:
                extra_context["artifact_registry"] = self._artifact_registry
            if self._review_executor is not None:
                extra_context["review_executor"] = self._review_executor
            if self._context_builder is not None:
                try:
                    conversation_context = await self._context_builder.build(manager)
                    extra_context["conversation_context"] = conversation_context
                except Exception as e:
                    logger.warning("Failed to build conversation context: %s", e)
            if self._extra_context:
                extra_context.update(self._extra_context)
            if extra_context:
                tool_context = tool_context.with_extra(**extra_context)

            # Execute all tool calls
            for tool_call in response.tool_calls:
                tool_trace = {
                    "name": tool_call.name,
                    "parameters": tool_call.parameters,
                }

                try:
                    # Find the tool
                    tool = self._find_tool(tool_call.name, tools)
                    if not tool:
                        observation = f"Error: Tool '{tool_call.name}' not found"
                        tool_trace["status"] = "error"
                        tool_trace["error"] = "Tool not found"

                        logger.warning(
                            "ReAct: Tool not found",
                            extra={
                                "conversation_id": manager.conversation_id,
                                "iteration": iteration + 1,
                                "tool_name": tool_call.name,
                            },
                        )
                    else:
                        # Execute the tool with context injection
                        # Context-aware tools will extract _context and use it
                        # Regular tools will ignore _context via **kwargs
                        t0 = time.monotonic()
                        result = await tool.execute(
                            **tool_call.parameters, _context=tool_context
                        )
                        duration_ms = (time.monotonic() - t0) * 1000
                        try:
                            observation = f"Tool result: {json.dumps(result, default=str)}"
                        except (TypeError, ValueError):
                            observation = f"Tool result: {result}"
                        tool_trace["status"] = "success"
                        tool_trace["result"] = str(result)

                        # Record for DynaBot on_tool_executed middleware hook
                        self._tool_executions.append(ToolExecution(
                            tool_name=tool_call.name,
                            parameters=tool_call.parameters,
                            result=result,
                            duration_ms=duration_ms,
                        ))

                        logger.log(
                            log_level,
                            "ReAct: Tool executed successfully",
                            extra={
                                "conversation_id": manager.conversation_id,
                                "iteration": iteration + 1,
                                "tool_name": tool_call.name,
                                "result_length": len(str(result)),
                            },
                        )

                    # Add observation using role="tool" so providers can
                    # pair it with the assistant's tool_calls in history.
                    await manager.add_message(
                        content=f"Observation from {tool_call.name}: {observation}",
                        role="tool",
                        name=tool_call.name,
                        tool_call_id=tool_call.id,
                    )

                except Exception as e:
                    # Handle tool execution errors — use role="tool" so the
                    # error is paired with the tool call in conversation.
                    error_msg = f"Error executing tool {tool_call.name}: {e!s}"
                    tool_trace["status"] = "error"
                    tool_trace["error"] = str(e)

                    # Record failed execution for middleware hook
                    self._tool_executions.append(ToolExecution(
                        tool_name=tool_call.name,
                        parameters=tool_call.parameters,
                        error=str(e),
                    ))

                    logger.error(
                        "ReAct: Tool execution failed",
                        extra={
                            "conversation_id": manager.conversation_id,
                            "iteration": iteration + 1,
                            "tool_name": tool_call.name,
                            "error": str(e),
                        },
                        exc_info=True,
                    )

                    await manager.add_message(
                        content=error_msg,
                        role="tool",
                        name=tool_call.name,
                        tool_call_id=tool_call.id,
                    )

                if trace is not None:
                    iteration_trace["tool_calls"].append(tool_trace)

            if trace is not None:
                iteration_trace["status"] = "continued"
                trace.append(iteration_trace)

            # Refresh system prompt so the next iteration sees current
            # artifact/bank state (e.g. after load_from_catalog).
            if self._prompt_refresher is not None:
                kwargs["system_prompt_override"] = self._prompt_refresher()

        else:
            # for-else: only reached when the loop exhausts all iterations
            # without a break (i.e. not triggered by duplicate detection)
            logger.log(
                log_level,
                "ReAct: Max iterations reached, generating final response",
                extra={
                    "conversation_id": manager.conversation_id,
                    "iterations_used": self.max_iterations,
                },
            )

            if trace is not None:
                trace.append({"status": "max_iterations_reached"})
                await self._store_trace(manager, trace)

        # Guarantee no dangling tool_use is left in history (e.g. a
        # duplicate-break abandoned the current call) before the final
        # synthesis re-sends history to the provider.
        await _pair_orphan_tool_calls(manager)

        # Refresh prompt for the final complete() call as well.
        if self._prompt_refresher is not None:
            kwargs["system_prompt_override"] = self._prompt_refresher()

        return await manager.complete(**kwargs)

    async def _store_trace(self, manager: Any, trace: list[dict[str, Any]]) -> None:
        """Store reasoning trace in conversation metadata.

        Args:
            manager: ConversationManager instance
            trace: Reasoning trace data
        """
        try:
            # Update in-memory metadata on the manager
            manager.update_metadata({"reasoning_trace": trace})

            # Persist to storage
            await manager.storage.update_metadata(
                conversation_id=manager.conversation_id,
                metadata=manager.metadata,
            )

            logger.debug(
                "ReAct: Stored reasoning trace in conversation metadata",
                extra={
                    "conversation_id": manager.conversation_id,
                    "trace_items": len(trace),
                },
            )
        except Exception as e:
            logger.warning(
                "ReAct: Failed to store reasoning trace",
                extra={
                    "conversation_id": manager.conversation_id,
                    "error": str(e),
                },
            )

    def _find_tool(self, tool_name: str, tools: list[Any]) -> Any | None:
        """Find a tool by name.

        Args:
            tool_name: Name of the tool to find
            tools: List of available tools

        Returns:
            Tool instance or None if not found
        """
        for tool in tools:
            if tool.name == tool_name:
                return tool
        return None
