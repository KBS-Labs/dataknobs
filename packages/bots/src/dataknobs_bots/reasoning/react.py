"""ReAct (Reasoning + Acting) reasoning strategy."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from dataknobs_common import Capability, CapabilityMixin, close_if_owned
from dataknobs_common.callbacks import CallbackRegistry
from dataknobs_common.structured_config import StructuredConfigConsumer
from dataknobs_llm import LLMStreamResponse, TokenCounter, create_llm_provider
from dataknobs_llm.exceptions import (
    ContextLengthExceededError,
    ToolsNotSupportedError,
)
from dataknobs_llm.llm.base import LLMResponse
from dataknobs_llm.llm.message_sequence import (
    pair_orphan_tool_calls,
    tool_call_signature,
)
from dataknobs_llm.tools import ToolExecutionContext

from dataknobs_bots.bot.turn import ToolExecution

from .base import ProcessResult, ReasoningStrategy, StrategyCapabilities, TurnHandle
from .compaction import CompactionStrategy, build_compaction_strategy
from .react_config import HistoryCompactionConfig, ReActReasoningConfig

logger = logging.getLogger(__name__)


class ReActTerminationReason(str, Enum):
    """Why a ReAct tool loop ended.

    ``.value`` is byte-identical to the reasoning-trace ``status`` strings, so
    the always-on ``reasoning_termination`` conversation metadata (written
    regardless of ``store_trace``) and the opt-in ``reasoning_trace`` last
    entry share a single source and can never drift.

    Members:
        COMPLETED: The LLM returned a final answer (no tool calls).
        MAX_ITERATIONS: The iteration cap was hit; ``finalize`` synthesizes.
        TRUNCATED_TOOL_CALL: The response was truncated mid-tool-call and the
            incomplete call was abandoned.
        DUPLICATE_TOOL_CALLS: The duplicate-tool-call break guard fired.
        TOOLS_NOT_SUPPORTED: The model cannot call tools; a graceful message
            was returned.
        TRUNCATION_RETRY_EXHAUSTED: The adaptive-budget retry of a truncated
            tool call did not recover a complete call — the retry was still
            truncated, or the retry ``complete()`` itself errored — so the
            truncated turn was abandoned. Distinguished from
            ``TRUNCATED_TOOL_CALL`` only by the retry having been enabled; the
            log line at the retry site records which sub-case occurred.
    """

    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations_reached"
    TRUNCATED_TOOL_CALL = "truncated_tool_call"
    DUPLICATE_TOOL_CALLS = "duplicate_tool_calls_detected"
    TOOLS_NOT_SUPPORTED = "tools_not_supported"
    TRUNCATION_RETRY_EXHAUSTED = "truncation_retry_exhausted"


#: Callback topic fired once per terminated ReAct turn (``<subsystem>:<operation>:<phase>``
#: convention). Consumers register on ``ReActReasoning.termination_callbacks``
#: and optionally compose ``also_publish_to(bus, topic_prefix="react:")`` for
#: cross-replica EventBus fan-out.
REACT_TERMINATION_TOPIC = "react:turn:end"


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
    StructuredConfigConsumer[ReActReasoningConfig],
    CapabilityMixin,
    ReasoningStrategy,
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

    #: Cross-cutting capability advertisement (``dataknobs_common.Capability``).
    #: ReAct exposes a lazy ``termination_callbacks`` registry, so it advertises
    #: ``CALLBACK_REGISTRY`` — machine-queryable via ``strategy.supports(...)``.
    #: This is orthogonal to the strategy-level :meth:`capabilities` classmethod
    #: (which returns a :class:`StrategyCapabilities`, a distinct surface).
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset(
        {Capability.CALLBACK_REGISTRY}
    )

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
        self._truncation_retry_max_tokens = config.truncation_retry_max_tokens
        #: Opt-in in-loop history compaction (default disabled → no-op). The
        #: strategy + optional dedicated summary provider are built lazily on
        #: first compaction (they need the runtime provider); a consumer may
        #: inject a bespoke ``CompactionStrategy`` via the components channel.
        self._history_compaction: HistoryCompactionConfig | None = (
            config.history_compaction
        )
        self._compaction_strategy: CompactionStrategy | None = (
            self.components.get("compaction_strategy")
        )
        self._summary_provider: Any = None
        self._owns_summary_provider = False
        #: Serializes the lazy strategy build so concurrent first-compactions
        #: (see ``generate``'s concurrent-call contract) cannot both construct
        #: and leak a dedicated summary provider. Loop-free once built.
        self._compaction_lock: asyncio.Lock = asyncio.Lock()
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
            # No tool loop ran, but the model still returned a final answer —
            # record the always-on termination reason so the "every ReAct turn
            # records why it ended" contract holds on the no-tools fast path
            # too (symmetric with generate()'s no-tools branch below).
            # iterations_used=0: the loop body never executed.  When store_trace
            # is on, write a fresh status-only trace so ``reasoning_trace`` can't
            # retain a stale trace from an earlier tool-using turn (mirrors the
            # MAX_ITERATIONS status-only append).
            if handle.trace is not None:
                handle.trace.append(
                    {"status": ReActTerminationReason.COMPLETED.value}
                )
            await self._record_termination(
                manager,
                ReActTerminationReason.COMPLETED,
                iterations_used=0,
                trace=handle.trace,
            )
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

    async def _maybe_retry_truncated_tool_call(
        self,
        response: Any,
        manager: Any,
        tools: list[Any] | None,
        kwargs: dict[str, Any],
        *,
        iteration: int,
    ) -> Any:
        """Opt-in single retry of a truncated tool-call turn at a larger budget.

        Called immediately after a ``complete()`` whose result is a truncated
        tool call (:func:`_is_truncated_tool_call`).  Returns:

        - the original ``response`` unchanged when the retry is disabled
          (``truncation_retry_max_tokens is None`` — the default), so the
          caller's terminal branch abandons it exactly as before; or
        - the retry response (non-truncated → the caller runs its normal
          branching: final answer / execute; still-truncated → the caller's
          terminal branch abandons *it* instead, which is off the same active
          path).

        The truncated node is dropped off the active conversation path via
        :meth:`ConversationManager.branch_from` before the retry, so the retry
        becomes its sibling and no orphan ``tool_use`` lingers in the history
        that the retry (or any later iteration) re-sends.  When the model
        advertises an output ceiling (e.g. the Claude family) the provider
        clamps the requested ``max_tokens`` to it; providers without a known
        ceiling pass it through.  Loop-safety does not depend on the clamp —
        this helper issues exactly one retry ``complete()``, so a still-truncated
        retry simply falls back to terminal synthesis.

        The retry is strictly additive to the abandon-and-synthesize
        degradation contract: if the retry ``complete()`` itself raises (a
        transient provider/network error), the truncated node is restored as
        current (undoing the pre-retry branch) and the *original* truncated
        ``response`` is returned, so the caller's terminal branch pairs that
        node's orphan ``tool_use`` and synthesizes exactly as the disabled
        default would — enabling the feature never converts a graceful abandon
        into a hard turn failure.

        Args:
            response: The just-returned truncated tool-call response.
            manager: Conversation manager for this turn.
            tools: Tools forwarded to the retry ``complete()``.
            kwargs: Additional generation params forwarded to the retry.
            iteration: 1-based iteration index, for log parity only.

        Returns:
            The response the caller should proceed with (see above).
        """
        budget = self._truncation_retry_max_tokens
        if budget is None:
            return response  # opt-out default → terminal, unchanged

        conv_id = getattr(manager, "conversation_id", None)
        logger.warning(
            "ReAct: tool call truncated at the token budget — retrying once "
            "at max_tokens=%d (clamped to the model ceiling where the provider "
            "advertises one)",
            budget,
            extra={"conversation_id": conv_id, "iteration": iteration},
        )
        # Drop the truncated node off the active path; the retry becomes its
        # sibling so history (root→current) excludes the incomplete tool_use.
        # Capture the id first so the error path can restore it (below).
        truncated_node_id = manager.current_node_id
        await manager.branch_from(truncated_node_id)
        # Merge the larger budget into any caller-supplied overrides rather
        # than passing a second ``llm_config_overrides`` (kwargs may already
        # carry one); the retry's max_tokens wins.
        retry_overrides = {
            **(kwargs.get("llm_config_overrides") or {}),
            "max_tokens": budget,
        }
        retry_kwargs = {**kwargs, "llm_config_overrides": retry_overrides}
        try:
            retry = await manager.complete(tools=tools, **retry_kwargs)
        except Exception as e:
            # Degrade, don't escalate: a failed retry falls back to the same
            # abandon-and-synthesize path the disabled default takes.  Restore
            # the truncated node as current (branch_from moved us to its parent,
            # and the raising retry appended nothing), so the caller's terminal
            # branch pairs that node's orphan tool_use and synthesizes with the
            # cut-off attempt in history — byte-identical to the disabled-default
            # abandon.  ``Exception`` (not ``BaseException``) so cancellation
            # propagates.
            logger.warning(
                "ReAct: truncation retry failed (%s) — abandoning "
                "(terminal synthesis)",
                e,
                extra={"conversation_id": conv_id, "iteration": iteration},
            )
            if truncated_node_id is not None:
                await manager.switch_to_node(truncated_node_id)
            return response
        if _is_truncated_tool_call(retry):
            logger.warning(
                "ReAct: retry still truncated at max_tokens=%d — abandoning "
                "(terminal synthesis)",
                budget,
                extra={"conversation_id": conv_id, "iteration": iteration},
            )
            # The still-truncated retry flows back to the caller's truncated
            # terminal branch, which records TRUNCATION_RETRY_EXHAUSTED (it
            # sees retry was enabled) — a single terminal recorder per branch,
            # no double-write. This helper stays purely about retrying.
        return retry

    def _truncation_reason(self) -> ReActTerminationReason:
        """The terminal reason for abandoning a truncated tool call.

        Retry enabled (``_truncation_retry_max_tokens is not None``) means the
        adaptive-budget retry already ran and the response reaching the
        truncated terminal branch is *still* truncated (or the retry errored)
        → the more specific
        :attr:`ReActTerminationReason.TRUNCATION_RETRY_EXHAUSTED`; otherwise a
        plain :attr:`ReActTerminationReason.TRUNCATED_TOOL_CALL`. Shared by the
        phased and monolithic truncated branches so the two cannot drift.
        """
        return (
            ReActTerminationReason.TRUNCATION_RETRY_EXHAUSTED
            if self._truncation_retry_max_tokens is not None
            else ReActTerminationReason.TRUNCATED_TOOL_CALL
        )

    # ------------------------------------------------------------------
    # In-loop history compaction (opt-in; shared by both loop sites, D5)
    # ------------------------------------------------------------------

    def _compaction_enabled(self) -> bool:
        cfg = self._history_compaction
        return cfg is not None and cfg.enabled

    def _resolve_history_budget(self, llm: Any) -> int | None:
        """Resolve the proactive token budget, or ``None`` (proactive off).

        Prefers the provider's resolved input ceiling
        (``ModelConstraints.max_input_tokens``) times ``budget_fraction`` — the
        common path for a provider that publishes a context window (the Claude
        family). A configured absolute ``history_token_budget`` then **caps**
        that resolved budget (the published ceiling is the model's *maximum
        attainable* context, which can exceed a consumer's *effective*
        per-request window — e.g. a beta-gated larger window they have not
        enabled; the cap keeps proactive compaction firing at their real limit
        rather than never). When no ceiling resolves (non-Anthropic providers /
        unknown model) ``history_token_budget`` is the sole threshold, and
        ``None`` is returned when neither is available (proactive disabled; the
        reactive backstop still applies).

        The ceiling is resolved from the provider's default-config constraints;
        a per-call model override to a different family is not reflected here.
        That imprecision is dominated by the coarse char-ratio history estimate
        and covered by the reactive backstop, so it is intentionally not
        plumbed through.
        """
        cfg = self._history_compaction
        if cfg is None:
            return None
        ceiling: int | None = None
        try:
            constraints = llm.get_constraints()
            ceiling = getattr(constraints, "max_input_tokens", None)
        except AttributeError:  # provider predates get_constraints() entirely
            ceiling = None
        if ceiling is not None:
            budget = int(ceiling * cfg.budget_fraction)
            if cfg.history_token_budget is not None:
                budget = min(budget, cfg.history_token_budget)
            return budget
        return cfg.history_token_budget

    async def _get_compaction_strategy(self, llm: Any) -> CompactionStrategy:
        """Lazily build (and cache) the compaction strategy.

        A consumer-injected ``compaction_strategy`` component is used as-is.
        Otherwise the strategy is built from config: ``"window"`` (LLM-free) or
        ``"summarize"`` — the latter reusing the runtime provider by default, or
        a dedicated one built (and owned) from ``summary_llm``.
        """
        if self._compaction_strategy is not None:
            return self._compaction_strategy
        # Serialize the build: the summarize path creates *and initializes*
        # (opens a network client for) a dedicated provider, so two concurrent
        # first-compactions must not both build one and leak the loser. Double-
        # checked under the lock — the fast path above stays lock-free once set.
        async with self._compaction_lock:
            if self._compaction_strategy is not None:
                return self._compaction_strategy
            cfg = self._history_compaction
            assert cfg is not None  # guarded by _compaction_enabled at callers
            provider = llm
            if cfg.strategy == "summarize" and cfg.summary_llm:
                self._summary_provider = create_llm_provider(cfg.summary_llm)
                initialize = getattr(self._summary_provider, "initialize", None)
                if initialize is not None:
                    await initialize()
                self._owns_summary_provider = True
                provider = self._summary_provider
            self._compaction_strategy = build_compaction_strategy(
                cfg.strategy, summary_provider=provider
            )
            return self._compaction_strategy

    async def _compact_now(self, manager: Any, llm: Any) -> int:
        """Compact unconditionally (used by the reactive backstop)."""
        cfg = self._history_compaction
        assert cfg is not None
        strategy = await self._get_compaction_strategy(llm)
        return await strategy.compact(
            manager, keep_recent_iterations=cfg.keep_recent_iterations
        )

    async def _maybe_compact_history(self, manager: Any, llm: Any) -> None:
        """Proactively compact the history when it exceeds the token budget.

        No-op when compaction is disabled (default) or no budget resolves. Uses
        the char-ratio ``TokenCounter`` estimate — imprecise by design; the
        reactive ``ContextLengthExceededError`` backstop covers under-estimates.
        """
        if not self._compaction_enabled():
            return
        budget = self._resolve_history_budget(llm)
        if budget is None:
            return
        try:
            history = await manager.get_history()
        except Exception:  # pragma: no cover - defensive; never fail the turn
            return
        if TokenCounter.estimate_messages_tokens(history) > budget:
            compacted = await self._compact_now(manager, llm)
            if compacted:
                logger.debug(
                    "ReAct: proactively compacted %d tool iterations "
                    "(history over budget)",
                    compacted,
                    extra={"conversation_id": getattr(
                        manager, "conversation_id", None
                    )},
                )

    async def _complete_with_reactive_compaction(
        self, manager: Any, llm: Any, complete: Callable[[], Any]
    ) -> Any:
        """Await ``complete()``; on context overflow, compact once and retry.

        The reactive backstop (D2): a ``ContextLengthExceededError`` from the
        in-loop completion triggers one compaction + one retry instead of
        failing the turn. When compaction is disabled the error propagates
        unchanged (byte-identical to today). ``complete`` is a zero-arg callable
        returning the completion coroutine so the retry re-issues it cleanly.
        """
        try:
            return await complete()
        except ContextLengthExceededError:
            if not self._compaction_enabled():
                raise
            compacted = await self._compact_now(manager, llm)
            conv_id = getattr(manager, "conversation_id", None)
            if compacted:
                logger.info(
                    "ReAct: context overflow — compacted %d tool iterations "
                    "and retrying once",
                    compacted,
                    extra={"conversation_id": conv_id},
                )
            else:
                # Nothing left to compact — the overflow is in the retained
                # head (system + current user + kept tail), not the loop body.
                # Retry anyway: the head may have changed, and one retry is
                # cheaper than reasoning about the exact cause here.
                logger.info(
                    "ReAct: context overflow — nothing compactable "
                    "(head over budget); retrying once",
                    extra={"conversation_id": conv_id},
                )
            return await complete()

    async def close(self) -> None:
        """Release a dedicated summary provider this strategy built + owns."""
        await close_if_owned(
            self._summary_provider, self._owns_summary_provider
        )
        await super().close()

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
                handle.trace.append(
                    {"status": ReActTerminationReason.MAX_ITERATIONS.value}
                )
            await self._record_termination(
                handle.manager,
                ReActTerminationReason.MAX_ITERATIONS,
                iterations_used=handle.max_iterations,
                trace=handle.trace,
            )
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

        # Proactively bound the in-loop history before the completion (no-op
        # unless compaction is enabled and over budget). Symmetric with the
        # monolithic ``generate`` site (top-of-iteration).
        await self._maybe_compact_history(handle.manager, handle.llm)

        # LLM call with tools (reactive compaction backstop wraps the
        # completion: a context overflow compacts once and retries).
        try:
            response = await self._complete_with_reactive_compaction(
                handle.manager,
                handle.llm,
                lambda: handle.manager.complete(
                    tools=handle.tools, **handle.kwargs
                ),
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
            if handle.trace is not None:
                handle.trace.append(
                    {"status": ReActTerminationReason.TOOLS_NOT_SUPPORTED.value}
                )
            await self._record_termination(
                handle.manager,
                ReActTerminationReason.TOOLS_NOT_SUPPORTED,
                iterations_used=handle.iteration + 1,
                trace=handle.trace,
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

        # Opt-in single adaptive-budget retry before abandoning a truncated
        # tool call.  Returns the original response unchanged when disabled
        # (default) or when the retry is still truncated, so every downstream
        # branch below is unaffected on the terminal path.
        if _is_truncated_tool_call(response):
            response = await self._maybe_retry_truncated_tool_call(
                response,
                handle.manager,
                handle.tools,
                handle.kwargs,
                iteration=handle.iteration + 1,
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
                iteration_trace["status"] = ReActTerminationReason.COMPLETED.value
                handle.trace.append(iteration_trace)
            await self._record_termination(
                handle.manager,
                ReActTerminationReason.COMPLETED,
                iterations_used=handle.iteration + 1,
                trace=handle.trace,
            )
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
            # Reaching this branch with the adaptive-budget retry enabled means
            # the retry ran and did not recover → the more specific
            # TRUNCATION_RETRY_EXHAUSTED reason (closes the FU5-B1 seam);
            # otherwise a plain truncation (shared _truncation_reason so the
            # two loop paths cannot drift). A single terminal recorder here —
            # the retry helper does not record — so metadata + trace never
            # double-write and always agree.
            reason = self._truncation_reason()
            handle.final_response = None  # finalize_turn does the synthesis
            if handle.trace is not None:
                iteration_trace["status"] = reason.value
                handle.trace.append(iteration_trace)
            await self._record_termination(
                handle.manager,
                reason,
                iterations_used=handle.iteration + 1,
                trace=handle.trace,
            )
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
                iteration_trace["status"] = (
                    ReActTerminationReason.DUPLICATE_TOOL_CALLS.value
                )
                handle.trace.append(iteration_trace)
            await self._record_termination(
                handle.manager,
                ReActTerminationReason.DUPLICATE_TOOL_CALLS,
                iterations_used=handle.iteration + 1,
                trace=handle.trace,
            )
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

        # Initialize trace if enabled (before the no-tools fast path so it, too,
        # can write a fresh trace).
        trace = [] if self.store_trace else None

        if not tools:
            # No tools available, fall back to simple generation
            logger.info(
                "ReAct: No tools available, falling back to simple generation",
                extra={"conversation_id": manager.conversation_id},
            )
            final = await manager.complete(**kwargs)
            # Symmetric with begin_turn's no-tools fast path: record the
            # always-on termination reason even though no tool loop ran, so a
            # consumer reading reasoning_termination unconditionally never hits
            # a missing key on a no-tools turn. iterations_used=0.  When
            # store_trace is on, write a fresh status-only trace so
            # ``reasoning_trace`` can't retain a stale earlier-turn trace.
            if trace is not None:
                trace.append(
                    {"status": ReActTerminationReason.COMPLETED.value}
                )
            await self._record_termination(
                manager,
                ReActTerminationReason.COMPLETED,
                iterations_used=0,
                trace=trace,
            )
            return final

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

            # Proactively bound the in-loop history before the completion
            # (no-op unless compaction is enabled and over budget).
            await self._maybe_compact_history(manager, llm)

            # Generate response with tools (reactive compaction backstop wraps
            # the completion: a context overflow compacts once and retries).
            try:
                response = await self._complete_with_reactive_compaction(
                    manager,
                    llm,
                    lambda: manager.complete(tools=tools, **kwargs),
                )
            except ToolsNotSupportedError as e:
                logger.error(
                    "ReAct: Model '%s' does not support tools — "
                    "returning graceful response to user",
                    e.model,
                    extra={"conversation_id": manager.conversation_id},
                )
                if trace is not None:
                    trace.append(
                        {
                            "status": (
                                ReActTerminationReason.TOOLS_NOT_SUPPORTED.value
                            )
                        }
                    )
                await self._record_termination(
                    manager,
                    ReActTerminationReason.TOOLS_NOT_SUPPORTED,
                    iterations_used=iteration + 1,
                    trace=trace,
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

            # Opt-in single adaptive-budget retry before abandoning a
            # truncated tool call (shared with the phased ``process_input``
            # path).  Returns the original response unchanged when disabled
            # (default) or when the retry is still truncated.
            if _is_truncated_tool_call(response):
                response = await self._maybe_retry_truncated_tool_call(
                    response, manager, tools, kwargs, iteration=iteration + 1,
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
                    iteration_trace["status"] = (
                        ReActTerminationReason.COMPLETED.value
                    )
                    trace.append(iteration_trace)
                await self._record_termination(
                    manager,
                    ReActTerminationReason.COMPLETED,
                    iterations_used=iteration + 1,
                    trace=trace,
                )

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
                # Reaching this branch with the adaptive-budget retry enabled
                # means the retry ran and did not recover → the more specific
                # TRUNCATION_RETRY_EXHAUSTED reason (closes the FU5-B1 seam);
                # otherwise a plain truncation (shared _truncation_reason so the
                # two loop paths cannot drift). Single terminal recorder — the
                # retry helper does not record — so metadata + trace agree.
                reason = self._truncation_reason()
                if trace is not None:
                    iteration_trace["status"] = reason.value
                    trace.append(iteration_trace)
                await self._record_termination(
                    manager,
                    reason,
                    iterations_used=iteration + 1,
                    trace=trace,
                )
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
                    iteration_trace["status"] = (
                        ReActTerminationReason.DUPLICATE_TOOL_CALLS.value
                    )
                    trace.append(iteration_trace)
                await self._record_termination(
                    manager,
                    ReActTerminationReason.DUPLICATE_TOOL_CALLS,
                    iterations_used=iteration + 1,
                    trace=trace,
                )

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
                trace.append(
                    {"status": ReActTerminationReason.MAX_ITERATIONS.value}
                )
            await self._record_termination(
                manager,
                ReActTerminationReason.MAX_ITERATIONS,
                iterations_used=self.max_iterations,
                trace=trace,
            )

        # Guarantee no dangling tool_use is left in history (e.g. a
        # duplicate-break abandoned the current call) before the final
        # synthesis re-sends history to the provider.
        await _pair_orphan_tool_calls(manager)

        # Refresh prompt for the final complete() call as well.
        if self._prompt_refresher is not None:
            kwargs["system_prompt_override"] = self._prompt_refresher()

        return await manager.complete(**kwargs)

    async def _persist_metadata(self, manager: Any) -> None:
        """Persist ``manager.metadata`` to storage.

        The in-memory metadata is assumed already mutated (via
        ``manager.update_metadata``); this flushes it to the backing store.
        The single flush point for :meth:`_record_termination`, so a terminal
        branch round-trips storage exactly once no matter how many metadata
        keys (``reasoning_termination`` and, when ``store_trace`` is on,
        ``reasoning_trace``) it updated.

        Args:
            manager: ConversationManager instance.
        """
        await manager.storage.update_metadata(
            conversation_id=manager.conversation_id,
            metadata=manager.metadata,
        )

    @property
    def termination_callbacks(self) -> CallbackRegistry:
        """Lazily-created registry fired once per terminated ReAct turn.

        Mirrors :attr:`ToolRegistry.ExecutionTracker.execution_callbacks`:
        zero cost until a consumer registers a callback on
        :data:`REACT_TERMINATION_TOPIC` or composes
        ``also_publish_to(bus, topic_prefix="react:")`` for cross-replica
        EventBus fan-out. Advertised via ``Capability.CALLBACK_REGISTRY``.
        """
        reg = getattr(self, "_termination_callbacks", None)
        if reg is None:
            reg = CallbackRegistry()
            self._termination_callbacks = reg
        return reg

    async def _record_termination(
        self,
        manager: Any,
        reason: ReActTerminationReason,
        *,
        iterations_used: int,
        trace: list[dict[str, Any]] | None = None,
    ) -> None:
        """Surface why the loop ended: always-on metadata + opt-in fan-out.

        Invoked at EVERY terminal branch across BOTH the phased
        ``process_input`` path and the monolithic ``generate`` path, so the
        reason logic lives once (not copy-pasted per branch). Writes the
        always-on ``reasoning_termination`` conversation metadata (independent
        of ``store_trace``) and — when ``trace`` is supplied (``store_trace``
        on) — the ``reasoning_trace`` metadata in the **same** in-memory update,
        so a terminal branch persists to storage **once** rather than
        round-tripping twice. When a consumer has registered a callback, fires
        :data:`REACT_TERMINATION_TOPIC`.

        Recording is non-load-bearing observability: a storage failure is
        logged and swallowed so it can never abort the turn.

        Args:
            manager: ConversationManager instance.
            reason: The terminal reason.
            iterations_used: Human-facing count of iterations used.
            trace: The reasoning trace to persist alongside the reason when
                ``store_trace`` is enabled (the caller appends the terminal
                ``status`` entry first), or ``None`` (default) to write only
                the always-on termination metadata.
        """
        payload = {
            "strategy": "react",
            "reason": reason.value,
            "iterations_used": iterations_used,
        }
        try:
            manager.update_metadata({"reasoning_termination": payload})
            if trace is not None:
                manager.update_metadata({"reasoning_trace": trace})
            # A single persist covers both keys — the update-then-persist
            # round-trip happens once per terminal branch, not once per key.
            await self._persist_metadata(manager)
        except Exception as e:
            logger.warning(
                "ReAct: Failed to record termination reason",
                extra={
                    "conversation_id": getattr(
                        manager, "conversation_id", None
                    ),
                    "error": str(e),
                },
            )
        # Fire the observability topic only when there is something to observe
        # it — a local callback OR a composed EventBus fan-out target. A
        # registry that was never touched isn't even instantiated, and one with
        # neither costs nothing. fire_async so async callbacks and
        # also_publish_to bus delivery are awaited correctly.
        reg = getattr(self, "_termination_callbacks", None)
        if reg is not None and (
            reg.callback_count(REACT_TERMINATION_TOPIC)
            or reg.supports_event_bus_emission()
        ):
            await reg.fire_async(REACT_TERMINATION_TOPIC, payload)

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
