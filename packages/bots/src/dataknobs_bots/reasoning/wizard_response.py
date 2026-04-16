"""Response generation, context building, and auto-advance for wizard flows.

Extracted from ``wizard.py`` in item 77d.  This module owns all
response-generation code paths — template rendering, LLM-based responses,
clarification prompts, validation errors, restart offers, auto-advance
logic, condition evaluation, and strategy delegation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from dataknobs_common.expressions import safe_eval
from dataknobs_llm import LLMStreamResponse

from .base import ReasoningStrategy, StreamStageContext
from .observability import create_transition_record
from .wizard_derivations import DerivationRule
from .wizard_types import StageSchema, TurnContext, WizardState, field_is_present

if TYPE_CHECKING:
    from ..prompts.resolver import PromptResolver
    from .wizard_fsm import WizardFSM
    from .wizard_renderer import WizardRenderer
    from .wizard_subflows import SubflowManager

logger = logging.getLogger(__name__)


@dataclass
class _TemplateResponse:
    """Minimal response object from template-rendered text.

    Duck-type compatible with LLMResponse, carrying the attributes
    that downstream code accesses: ``content``, ``metadata``, and
    ``model``.
    """

    content: str
    model: str = "template"
    finish_reason: str | None = "stop"
    usage: dict[str, int] | None = None
    tool_calls: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StageResponseResult:
    """Result from :meth:`WizardResponder.generate_stage_response`.

    Carries the LLM response alongside lifecycle signals that
    ``generate()`` checks after response generation.  Replaces the
    mutable ``_tool_completion_requested`` / ``_tool_restart_requested``
    bridge on ``WizardReasoning``.
    """

    response: Any
    tool_completion_requested: bool = False
    tool_completion_summary: str = ""
    tool_restart_requested: bool = False


class WizardResponder:
    """Response generation, context building, and auto-advance logic.

    Constructed by :class:`WizardReasoning` and wired via shared
    references and callable callbacks.  Extracted in item 77d.
    """

    def __init__(
        self,
        *,
        # --- Shared references (stable, not replaced during lifecycle) ---
        renderer: WizardRenderer,
        fsm: WizardFSM,
        subflows: SubflowManager,
        # --- Config ---
        context_template: str | None,
        auto_advance_filled_stages: bool,
        default_tool_reasoning: str,
        default_max_iterations: int,
        default_store_trace: bool,
        default_verbose: bool,
        strict_validation: bool,
        field_derivations: list[DerivationRule],
        clarification_groups: list[dict[str, Any]],
        clarification_exclude_derivable: bool,
        clarification_template: str | None,
        prompt_resolver: PromptResolver | None = None,
        # --- Callbacks (orchestrator-owned, may change during lifecycle) ---
        build_wizard_metadata: Callable[[WizardState], dict[str, Any]],
        execute_fsm_step: Callable[..., Awaitable[tuple[str, Any]]],
        make_bank_accessor: Callable[[], Callable[..., Any]],
        get_artifact: Callable[[], Any | None],
        get_catalog: Callable[[], Any | None],
        get_artifact_registry: Callable[[], Any | None],
        get_review_executor: Callable[[], Any | None],
        get_context_builder: Callable[[], Any | None],
        get_banks: Callable[[], dict[str, Any]],
    ) -> None:
        # Shared references
        self._renderer = renderer
        self._fsm = fsm
        self._subflows = subflows

        # Config
        self._context_template = context_template
        self._auto_advance_filled_stages = auto_advance_filled_stages
        self._default_tool_reasoning = default_tool_reasoning
        self._default_max_iterations = default_max_iterations
        self._default_store_trace = default_store_trace
        self._default_verbose = default_verbose
        self._strict_validation = strict_validation
        self._field_derivations = field_derivations
        self._clarification_groups = clarification_groups
        self._clarification_exclude_derivable = clarification_exclude_derivable
        self._clarification_template = clarification_template
        self._prompt_resolver = prompt_resolver

        # Callbacks
        self._build_wizard_metadata = build_wizard_metadata
        self._execute_fsm_step = execute_fsm_step
        self._make_bank_accessor = make_bank_accessor
        self._get_artifact = get_artifact
        self._get_catalog = get_catalog
        self._get_artifact_registry = get_artifact_registry
        self._get_review_executor = get_review_executor
        self._get_context_builder = get_context_builder
        self._get_banks = get_banks

    # =====================================================================
    # Public API — called by wizard.py orchestrator
    # =====================================================================

    async def generate_stage_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        state: WizardState,
        tools: list[Any] | None,
        track_render: bool = True,
    ) -> StageResponseResult:
        """Generate response appropriate for current stage.

        Supports three response paths:

        1. **Template mode** (structured stages with ``response_template``):
           Renders the template with Jinja2 using wizard state data,
           bypassing the LLM entirely.  Template is rendered on every
           turn — the template IS the response (e.g. review summaries).
           If the stage also has ``llm_assist: true`` and the user's
           last message is a question, the LLM is invoked with a
           scoped assist prompt.

        2. **Conversation greeting** (``mode: conversation`` with
           ``response_template``): The template is rendered only on
           the first turn (render count == 0) as a greeting.
           Subsequent turns fall through to LLM mode so the bot
           can actually converse.

        3. **LLM mode** (default, or conversation stages after first
           render): Calls the LLM with stage context injected into
           the system prompt.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            state: Current wizard state
            tools: Available tools
            track_render: If True (default), increment the stage's
                render count after generating the response.  Pass
                False when (a) the caller has already incremented
                the render count explicitly (confirmation path in
                ``process_input``), or (b) for subflow push paths
                where the template is displayed as a question and
                confirmation should fire on the user's next response.

        Returns:
            :class:`StageResponseResult` with response and lifecycle signals.
        """
        stage_name = stage.get("name", "unknown")
        response_template = stage.get("response_template")
        wizard_snapshot = {"wizard": self._build_wizard_metadata(state)}

        # ── Template mode ────────────────────────────────────────
        is_conversation_mode = stage.get("mode") == "conversation"
        is_first_render = state.get_render_count(stage_name) == 0
        use_template = response_template and (
            not is_conversation_mode or is_first_render
        )

        if use_template:
            content, llm_response = await self._resolve_template_content(
                manager, llm, stage, state, response_template, wizard_snapshot,
            )
            response = llm_response if llm_response is not None else (
                self.create_template_response(content)
            )
            self.add_wizard_metadata(response, state, stage)
            if track_render and response_template:
                state.increment_render_count(stage_name)
            return StageResponseResult(response=response)

        # ── LLM mode ────────────────────────────────────────────
        enhanced_prompt, stage_tools, strategy = self._prepare_llm_mode(
            manager, stage, state, tools,
        )

        logger.debug(
            "Generating response for stage '%s' (tools=%s, strategy=%s)",
            stage_name,
            [getattr(t, "name", str(t)) for t in stage_tools] if stage_tools else None,
            type(strategy).__name__ if strategy else "single",
        )

        tool_completion_requested = False
        tool_completion_summary = ""
        tool_restart_requested = False

        if strategy:
            response, tool_completion_requested, tool_completion_summary, tool_restart_requested = (
                await self._strategy_stage_response(
                    strategy, manager, enhanced_prompt, stage, state,
                    stage_tools, metadata=wizard_snapshot,
                )
            )
        else:
            # Single LLM call (default behavior)
            response = await manager.complete(
                system_prompt_override=enhanced_prompt,
                tools=stage_tools,
                metadata=wizard_snapshot,
            )

        # Log response details
        response_content = getattr(response, "content", str(response))
        response_len = len(response_content) if response_content else 0
        logger.debug(
            "Stage '%s' response: %d chars, has_tool_calls=%s",
            stage_name,
            response_len,
            bool(getattr(response, "tool_calls", None)),
        )
        if response_len == 0:
            logger.warning(
                "Empty response generated for stage '%s'",
                stage_name,
            )

        # Add wizard metadata to response
        self.add_wizard_metadata(response, state, stage)

        # Only fires for conversation-mode stages where response_template
        # is truthy but use_template was False (past first render).  For
        # pure LLM stages response_template is falsy so this is a no-op.
        if track_render and response_template:
            state.increment_render_count(stage_name)

        return StageResponseResult(
            response=response,
            tool_completion_requested=tool_completion_requested,
            tool_completion_summary=tool_completion_summary,
            tool_restart_requested=tool_restart_requested,
        )

    async def generate_clarification_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        issues: list[str],
        tools: list[Any] | None = None,
        wizard_state: WizardState | None = None,
    ) -> Any:
        """Generate response asking for clarification.

        When clarification groups are configured, replaces the generic
        issue list with structured grouped questions.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            issues: Extraction issues
            tools: Available tools (so LLM can handle out-of-stage requests)
            wizard_state: Current wizard state for metadata snapshot

        Returns:
            LLM response requesting clarification
        """
        issue_list = (
            "\n".join(f"- {e}" for e in issues)
            if issues
            else "- Response was ambiguous"
        )

        # Build field groups for structured clarification when configured.
        # Requires at least one group to be defined — exclude_derivable
        # alone only affects group content, it doesn't replace the
        # generic issue list.
        if self._clarification_groups and wizard_state is not None:
            missing = StageSchema.from_stage(stage).missing_required(
                wizard_state.data,
            )
            if missing:
                groups = self._build_clarification_groups(
                    missing, stage, wizard_state,
                )
                if groups:
                    if self._clarification_template:
                        default_list = "\n".join(
                            f"- {g['question']}" for g in groups
                        )
                        issue_list = self._renderer.render_simple(
                            self._clarification_template,
                            {"field_groups": groups},
                            fallback=default_list,
                        )
                    else:
                        issue_list = "\n".join(
                            f"- {g['question']}" for g in groups
                        )
        suggestions = stage.get("suggestions", [])
        suggestions_text = (
            f"\n**Suggestions**: {', '.join(suggestions)}" if suggestions else ""
        )

        stage_prompt = stage.get("prompt", "Please provide more specific information.")
        clarification_context = None
        if self._prompt_resolver is not None:
            clarification_context = self._prompt_resolver.resolve(
                "wizard.clarification",
                issue_list=issue_list,
                stage_prompt=stage_prompt,
                suggestions_text=suggestions_text,
            )
        if clarification_context is None:
            clarification_context = (
                f"## Clarification Needed\n\n"
                f"I wasn't able to clearly understand the user's response "
                f"for this stage.\n\n"
                f"**Potential Issues**:\n{issue_list}\n\n"
                f"**What I'm Looking For**: {stage_prompt}{suggestions_text}\n\n"
                f"Please ask a clarifying question to help gather the needed "
                f"information.\n"
                f"Be conversational and helpful - don't make the user feel "
                f"like they did something wrong."
            )
        return await self._complete_with_wizard_context(
            manager, clarification_context, stage, wizard_state, tools,
        )

    async def generate_validation_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        errors: list[str],
        tools: list[Any] | None = None,
        wizard_state: WizardState | None = None,
    ) -> Any:
        """Generate response asking for corrections.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            errors: Validation error messages
            tools: Available tools (so LLM can handle out-of-stage requests)
            wizard_state: Current wizard state for metadata snapshot

        Returns:
            LLM response requesting corrections
        """
        error_list = "\n".join(f"- {e}" for e in errors)
        stage_prompt = stage.get("prompt", "Please provide the required information.")
        error_context = None
        if self._prompt_resolver is not None:
            error_context = self._prompt_resolver.resolve(
                "wizard.validation",
                error_list=error_list,
                stage_prompt=stage_prompt,
            )
        if error_context is None:
            error_context = (
                f"## Validation Required\n\n"
                f"The user's input for this stage needs clarification:\n\n"
                f"**Issues**:\n{error_list}\n\n"
                f"**What's Needed**: {stage_prompt}\n\n"
                f"Please kindly ask the user to provide the missing or "
                f"corrected information.\n"
                f"Be specific about what's needed but remain friendly "
                f"and helpful."
            )
        return await self._complete_with_wizard_context(
            manager, error_context, stage, wizard_state, tools,
        )

    async def generate_transform_error_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        error: str,
        tools: list[Any] | None = None,
        wizard_state: WizardState | None = None,
    ) -> Any:
        """Generate response when a transition transform fails.

        This surfaces transform errors to the user instead of silently
        re-rendering the current stage, which would appear as the wizard
        being "stuck".

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            error: Error message from the failed transform
            tools: Available tools (so LLM can handle out-of-stage requests)
            wizard_state: Current wizard state for metadata snapshot

        Returns:
            LLM response explaining the error and offering retry
        """
        stage_name = stage.get("name", "unknown")
        error_context = None
        if self._prompt_resolver is not None:
            error_context = self._prompt_resolver.resolve(
                "wizard.transform_error",
                stage_name=stage_name,
                error=error,
            )
        if error_context is None:
            error_context = (
                f"## Processing Error\n\n"
                f'An error occurred while processing the transition from '
                f'the "{stage_name}" stage:\n\n'
                f"**Error**: {error}\n\n"
                f"Please apologize for the issue and let the user know they "
                f"can try again.\n"
                f"If the error suggests a configuration or system issue, "
                f"suggest they contact support.\n"
                f"Be concise and helpful."
            )
        return await self._complete_with_wizard_context(
            manager, error_context, stage, wizard_state, tools,
            include_stage_context=False,
        )

    async def generate_restart_offer(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        issues: list[str],
        tools: list[Any] | None = None,
        wizard_state: WizardState | None = None,
    ) -> Any:
        """Generate response offering to restart after multiple failures.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            issues: Extraction issues
            tools: Available tools (so LLM can handle out-of-stage requests)
            wizard_state: Current wizard state for metadata snapshot

        Returns:
            LLM response offering restart option
        """
        stage_name = stage.get("name", "unknown")
        stage_prompt = stage.get("prompt", "Provide information")
        restart_context = None
        if self._prompt_resolver is not None:
            restart_context = self._prompt_resolver.resolve(
                "wizard.restart_offer",
                stage_name=stage_name,
                stage_prompt=stage_prompt,
            )
        if restart_context is None:
            restart_context = (
                f"## Multiple Clarification Attempts\n\n"
                f"We've had difficulty understanding the responses for "
                f"this stage.\n\n"
                f"**Current Stage**: {stage_name}\n"
                f"**Goal**: {stage_prompt}\n\n"
                f"Please offer the user two options:\n"
                f"1. Try one more time with clearer instructions\n"
                f'2. Start the wizard over from the beginning '
                f'(type "restart")\n\n'
                f"Be empathetic and helpful - acknowledge that the questions "
                f"might not be clear."
            )
        return await self._complete_with_wizard_context(
            manager, restart_context, stage, wizard_state, tools,
        )

    def can_auto_advance(
        self,
        wizard_state: WizardState,
        stage: dict[str, Any],
        *,
        after_re_extraction: bool = False,
    ) -> bool:
        """Check if a stage can be auto-advanced.

        A stage can be auto-advanced if:
        1. Auto-advance is enabled for this stage (see precedence below)
        2. The stage has a schema with required fields (or all properties
           if no required list) — *skipped when* ``after_re_extraction``
        3. All required fields have non-empty values in wizard_state.data
           — *skipped when* ``after_re_extraction``
        4. The stage is not an end stage
        5. At least one transition condition is satisfied

        Auto-advance precedence (stage-level wins over global):
        - ``auto_advance: false`` — disabled regardless of global setting
        - ``auto_advance: true``  — enabled regardless of global setting
        - absent (``None``)       — defers to global
          ``auto_advance_filled_stages``

        When ``after_re_extraction`` is ``True``, both the
        ``auto_advance`` gate (Gate 1) and the required-fields gate
        (Gate 2) are relaxed.  Only the transition-condition check
        (Gate 3) is enforced — this is the correct safety boundary
        because it encodes the domain logic about when advancement
        is appropriate.

        Args:
            wizard_state: Current wizard state
            stage: Stage configuration dict
            after_re_extraction: When ``True``, relax the
                ``auto_advance`` gate and the required-fields gate
                because re-extraction just captured data at this stage.

        Returns:
            True if stage can be auto-advanced
        """
        # Check if auto-advance is enabled for this stage.
        # Stage-level setting takes precedence over global when explicitly set.
        stage_auto_advance = stage.get("auto_advance")
        if stage_auto_advance is False and not after_re_extraction:
            # Explicitly disabled at stage level — respect unless we just
            # re-extracted data (re-extraction implies user intent to advance)
            return False
        if (
            not stage_auto_advance
            and not self._auto_advance_filled_stages
            and not after_re_extraction
        ):
            # Not explicitly enabled at stage level, and global is off —
            # respect unless we just re-extracted data
            return False

        # Don't auto-advance end stages
        if stage.get("is_end", False):
            return False

        # Gate 2: required fields — skip when after_re_extraction.
        # After re-extraction, the transition condition (Gate 3) is the
        # correct arbiter.  The required-fields check blocks on unfilled
        # optional fields (e.g., llm_model='') that are irrelevant to
        # whether the user's intent should be honored.
        if not after_re_extraction:
            # Get schema to check required fields
            ss = StageSchema.from_stage(stage)
            properties = ss.properties
            required_fields = ss.required_fields

            # If no required fields specified, treat all properties as required
            if not required_fields:
                required_fields = list(properties.keys())

            # If no fields at all:
            # - Per-stage auto_advance: true → can advance (message/display stage)
            #   Still requires a satisfied transition condition (checked below).
            # - Global auto_advance_filled_stages only → cannot advance
            #   (that setting means "skip stages whose fields are already filled",
            #   not "skip stages that have no fields to fill").
            if not required_fields and not stage_auto_advance:
                return False

            # Check if all required fields have non-empty values
            for field_name in required_fields:
                if field_name not in wizard_state.data:
                    return False
                value = wizard_state.data[field_name]
                if value is None:
                    return False
                # Empty strings don't count as filled
                if isinstance(value, str) and not value.strip():
                    return False

        # Gate 3: transition condition — always checked
        transitions = stage.get("transitions", [])
        for transition in transitions:
            condition = transition.get("condition")
            if condition:
                # Evaluate condition with current data
                if self.evaluate_condition(condition, wizard_state.data):
                    return True
            else:
                # Unconditional transition - can advance
                return True

        return False

    async def run_auto_advance_loop(
        self,
        wizard_state: WizardState,
        active_fsm: WizardFSM,
        initial_stage: dict[str, Any],
        *,
        skip_first_render: bool = False,
        llm: Any | None = None,
        after_re_extraction: bool = False,
    ) -> list[str]:
        """Run the auto-advance loop, collecting rendered templates.

        Advances through consecutive stages that satisfy
        :meth:`can_auto_advance`, rendering each stage's
        ``response_template`` before moving past it.

        A per-call closure is installed as the ``transform_context_factory``
        before each ``step_async`` call, mirroring the pattern in
        ``_execute_fsm_step``, so that auto-advance transforms have
        access to the LLM and a ``TurnContext``.

        Args:
            wizard_state: Current wizard state (mutated in place)
            active_fsm: The currently active FSM instance
            initial_stage: Stage metadata to start advancing from
            skip_first_render: If True, skip the template render on
                the first iteration (used by greet when the start
                stage response is already captured)
            llm: LLM provider for auto-advance transforms (``None``
                when no LLM is available).
            after_re_extraction: When ``True``, relax the
                ``auto_advance`` gate on the **first iteration only**.
                Subsequent stages in the chain follow their own
                ``auto_advance`` setting.

        Returns:
            List of rendered template strings from auto-advanced stages
        """
        from ..artifacts.transforms import TransformContext

        messages: list[str] = []
        count = 0
        max_advances = 10
        stage = initial_stage

        while (
            count < max_advances
            and not wizard_state.completed
            and self.can_auto_advance(
                wizard_state, stage,
                after_re_extraction=after_re_extraction and count == 0,
            )
        ):
            # Render template of the stage being advanced past
            if not (skip_first_render and count == 0):
                rendered = self._render_auto_advance_template(
                    stage, wizard_state
                )
                if rendered:
                    messages.append(rendered)

            count += 1
            old_stage_name = wizard_state.current_stage
            duration_ms = (
                (time.time() - wizard_state.stage_entry_time) * 1000
            )

            # Install per-step closure so auto-advance transforms see
            # the LLM and a TurnContext, matching _execute_fsm_step.
            turn = TurnContext(
                message=None,
                bank_fn=self._make_bank_accessor(),
                intent=wizard_state.data.get("_intent"),
            )

            def _scoped_factory(
                func_context: Any,
                *,
                _turn: TurnContext = turn,
                _llm: Any | None = llm,
            ) -> Any:
                return TransformContext(
                    fsm_context=func_context,
                    turn=_turn,
                    artifact_registry=self._get_artifact_registry(),
                    rubric_executor=self._get_review_executor(),
                    config={"llm": _llm} if _llm else {},
                    banks=self._get_banks(),
                )

            original_factory = active_fsm.get_transform_context_factory()
            active_fsm.set_transform_context_factory(_scoped_factory)
            try:
                auto_step_result = await active_fsm.step_async(
                    wizard_state.data
                )
            finally:
                # Restore the previous factory (the orchestrator's
                # fallback, installed by WizardReasoning at construction).
                if original_factory is not None:
                    active_fsm.set_transform_context_factory(
                        original_factory
                    )
            new_stage_name = active_fsm.current_stage

            if new_stage_name == old_stage_name:
                break

            condition_expr = active_fsm.get_transition_condition(
                old_stage_name, new_stage_name
            )
            transition = create_transition_record(
                from_stage=old_stage_name,
                to_stage=new_stage_name,
                trigger="auto_advance",
                duration_in_stage_ms=duration_ms,
                data_snapshot=wizard_state.data.copy(),
                condition_evaluated=condition_expr,
                condition_result=True if condition_expr else None,
                subflow_depth=wizard_state.subflow_depth,
            )
            wizard_state.transitions.append(transition)

            wizard_state.current_stage = new_stage_name
            if new_stage_name not in wizard_state.history:
                wizard_state.history.append(new_stage_name)
            wizard_state.completed = auto_step_result.is_complete
            wizard_state.stage_entry_time = time.time()

            logger.info(
                "Auto-advanced from %s to %s",
                old_stage_name,
                new_stage_name,
            )

            # Handle subflow pop if needed (no-op when no subflow is active)
            if self._subflows.should_pop(wizard_state):
                self._subflows.handle_pop(wizard_state)
                active_fsm = self._subflows.get_active_fsm()
                wizard_state.completed = False

            stage = active_fsm.current_metadata

        # If we advanced through any stages, mark the landing stage so the
        # next generate() call skips extraction — the user hasn't had a
        # chance to respond to this stage's prompt yet.
        if count > 0:
            wizard_state.skip_extraction = True

        return messages

    def evaluate_condition(self, condition: str, data: dict[str, Any]) -> bool:
        """Safely evaluate a transition condition.

        Uses the shared safe expression engine from
        :mod:`dataknobs_common.expressions` to evaluate condition
        expressions like ``data.get('subject')``, ``has('count')``,
        or ``data.get('count', 0) > 5``.

        Available globals in condition expressions:

        - ``data`` — the wizard state data dict
        - ``has(key)`` — shorthand for ``data.get(key) is not None``;
          preferred for boolean/numeric/list fields where falsy values
          are legitimate.  Note: empty strings are considered present;
          for text fields where non-empty content is required, use
          ``data.get('key')`` (truthiness rejects empty strings)
        - ``bank`` — memory bank accessor
        - ``artifact`` — current artifact
        - ``true``/``false``/``null``/``none`` — YAML/JSON literal aliases

        Args:
            condition: Condition expression string
            data: Current wizard data

        Returns:
            True if condition is satisfied, False otherwise
        """
        # Shallow copy so expression cannot add/remove/replace
        # top-level keys in live wizard state.
        data_snapshot = dict(data)
        result = safe_eval(
            condition,
            scope={
                "data": data_snapshot,
                "has": lambda key: field_is_present(
                    data_snapshot.get(key)
                ),
                "bank": self._make_bank_accessor(),
                "artifact": self._get_artifact(),
            },
            coerce_bool=True,
            default=False,
        )
        if not result.success:
            logger.debug(
                "Condition evaluation failed for '%s': %s",
                condition,
                result.error,
            )
        return result.value

    def calculate_progress(self, state: WizardState) -> float:
        """Calculate wizard completion progress (0.0 to 1.0).

        Args:
            state: Current wizard state

        Returns:
            Progress as float between 0 and 1
        """
        total_stages = self._fsm.stage_count
        if total_stages == 0:
            return 0.0

        visited = len(set(state.history))
        # Subtract 1 for end state in progress calculation
        return min(1.0, visited / max(1, total_stages - 1))

    def add_wizard_metadata(
        self,
        response: Any,
        state: WizardState,
        stage: dict[str, Any],
    ) -> None:
        """Add wizard metadata to response object.

        Note: DynaBot.chat() only returns response content (a string),
        so this metadata is only visible to code that accesses the raw
        response object (e.g. middleware, tests).  For downstream
        consumers like EduBot, wizard metadata flows through
        ``_save_wizard_state`` → ``get_wizard_state()``.

        Args:
            response: LLM response object to modify
            state: Current wizard state
            stage: Current stage metadata (unused, kept for API compat)
        """
        if not hasattr(response, "metadata") or response.metadata is None:
            response.metadata = {}
        response.metadata["wizard"] = self._build_wizard_metadata(state)

    def build_stages_roadmap(
        self,
        state: WizardState,
    ) -> list[dict[str, str]]:
        """Build ordered stages roadmap with labels and status.

        Produces a list of stage entries for UI rendering (breadcrumb,
        checklist, etc.). Each entry contains the stage name, a
        human-readable label, and a status indicating whether the stage
        has been completed, is the current stage, or is still pending.

        When the wizard is inside a subflow, the parent stage is marked
        ``"current"`` (since its subflow is still active) and all stages
        visited *before* the subflow push are marked ``"completed"``.
        The subflow's own stages do NOT appear in this roadmap — they
        are exposed via the ``subflow_stage`` key in wizard metadata.

        Args:
            state: Current wizard state

        Returns:
            List of dicts with ``name``, ``label``, and ``status`` keys.
            Status is one of ``"completed"``, ``"current"``, or
            ``"pending"``.
        """
        # During a subflow, state.history only contains subflow stages
        # (it was reset at push time).  Use the parent's saved history
        # to know which main-flow stages were visited.
        if state.subflow_stack:
            parent_ctx = state.subflow_stack[-1]
            visited = set(parent_ctx.parent_history)
            parent_stage = parent_ctx.parent_stage
        else:
            visited = set(state.history)
            parent_stage = None

        current = state.current_stage
        stages: list[dict[str, str]] = []

        for name, meta in self._fsm.stages.items():
            if parent_stage is not None and name == parent_stage:
                # The parent stage is still active (subflow in progress)
                status = "current"
            elif parent_stage is None and name == current:
                # Normal (non-subflow) flow: current stage
                status = "current"
            elif name in visited and name != parent_stage:
                status = "completed"
            else:
                status = "pending"
            stages.append({
                "name": name,
                "label": meta.get("label", name),
                "status": status,
            })

        return stages

    def build_stage_context(
        self, stage: dict[str, Any], state: WizardState
    ) -> str:
        """Build context prompt for current stage.

        Uses custom template if configured, otherwise falls back to
        default hardcoded format.

        Args:
            stage: Current stage metadata
            state: Current wizard state

        Returns:
            Context string to append to system prompt
        """
        if self._context_template:
            return self._render_custom_context(stage, state)
        return self._build_default_context(stage, state)

    def filter_tools_for_stage(
        self, stage: dict[str, Any], tools: list[Any] | None
    ) -> list[Any] | None:
        """Filter tools to those available for the current stage.

        Args:
            stage: Stage configuration dict
            tools: List of available tools, or None

        Returns:
            Filtered tools for this stage, or None if no tools should be available.

        Tool availability rules:
        - No tools passed in: return None (no tools available)
        - Stage has no 'tools' key: return None (safe default - no tools)
        - Stage has empty 'tools' list: return None (explicitly no tools)
        - Stage has 'tools' list: return only matching tools
        """
        if not tools:
            return None

        stage_tool_names = stage.get("tools")

        # Key change: no 'tools' key means no tools (safe default)
        if stage_tool_names is None:
            return None

        # Explicit empty list means no tools
        if not stage_tool_names:
            return None

        # Filter to stage-specific tools
        filtered = []
        for tool in tools:
            tool_name = getattr(tool, "name", None) or getattr(
                tool, "__name__", None
            )
            if tool_name in stage_tool_names:
                filtered.append(tool)

        return filtered if filtered else None

    def render_suggestions(
        self,
        suggestions: list[str],
        state: WizardState,
    ) -> list[str]:
        """Render suggestion strings through Jinja2 with wizard state data.

        Delegates to :meth:`WizardRenderer.render_list`.

        Args:
            suggestions: List of suggestion template strings
            state: Current wizard state

        Returns:
            List of rendered suggestion strings
        """
        active_fsm = self._subflows.get_active_fsm()
        return self._renderer.render_list(
            suggestions, active_fsm.current_metadata, state,
        )

    @staticmethod
    def prepend_messages_to_response(
        response: Any, messages: list[str]
    ) -> None:
        """Prepend collected auto-advance messages to a response.

        Modifies the response object in place, inserting message stage
        content before the landing stage's response. Messages are joined
        with double newlines.

        Args:
            response: Response object with a ``content`` attribute
            messages: List of rendered template strings to prepend
        """
        if not messages:
            return
        prefix = "\n\n".join(messages) + "\n\n"
        response.content = prefix + response.content

    @staticmethod
    def is_help_request(message: str) -> bool:
        """Check if a user message is a question/help request.

        Used to decide whether to invoke the LLM in ``llm_assist``
        mode for template-driven stages.

        Args:
            message: User message text

        Returns:
            True if the message appears to be a question
        """
        msg = message.strip().lower()
        if msg.endswith("?"):
            return True
        question_starters = (
            "what ", "which ", "how ", "why ", "should ",
            "can ", "could ", "would ", "is ", "are ",
            "do ", "does ", "help", "explain", "tell me",
        )
        return msg.startswith(question_starters)

    @staticmethod
    def create_template_response(content: str) -> Any:
        """Create a minimal response object from template-rendered text.

        The returned object is duck-type compatible with LLMResponse,
        carrying the attributes that downstream code accesses:
        ``content``, ``metadata``, and ``model``.

        Args:
            content: Rendered template text

        Returns:
            Response object compatible with the wizard pipeline
        """
        return _TemplateResponse(content=content)

    def build_confirmation_content(
        self,
        stage: dict[str, Any],
        state: WizardState,
        new_data_keys: set[str],
    ) -> str:
        """Build confirmation content for extracted data.

        When the stage defines ``confirmation_template``, renders it as
        a Jinja2 template with full wizard state context (same as
        ``response_template``).  Otherwise, auto-generates a confirmation
        listing each newly extracted field with its schema description
        and current value.

        Args:
            stage: Current stage metadata.
            state: Current wizard state (data already merged).
            new_data_keys: Set of field names extracted this turn.

        Returns:
            Rendered confirmation string.
        """
        confirmation_template = stage.get("confirmation_template")
        if confirmation_template:
            return self._render_response_template(
                confirmation_template, stage, state,
                extra_context={"new_data_keys": new_data_keys},
            )

        # Auto-generate from schema + extracted data
        schema = stage.get("schema") or {}
        properties = schema.get("properties") or {}
        lines: list[str] = []
        for key in sorted(new_data_keys):
            value = state.data.get(key)
            if value is not None:
                label = properties.get(key, {}).get("description", key)
                formatted = (
                    ", ".join(str(item) for item in value)
                    if isinstance(value, list)
                    else str(value)
                )
                lines.append(f"- **{label}:** {formatted}")
        if not lines:
            # All extracted values were None — auto-generation has nothing
            # to show.  Fall back to response_template re-render.
            stage_name = stage.get("name", "unknown")
            logger.warning(
                "Stage '%s': confirmation auto-generation produced no "
                "lines (all new_data_keys mapped to None). Falling back "
                "to response_template re-render.",
                stage_name,
            )
            response_template = stage.get("response_template", "")
            return self._render_response_template(
                response_template, stage, state,
            )
        return "Here's what I got:\n" + "\n".join(lines) + "\n\nIs that correct?"

    # =====================================================================
    # Private implementation methods
    # =====================================================================

    async def _complete_with_wizard_context(
        self,
        manager: Any,
        additional_context: str,
        stage: dict[str, Any],
        wizard_state: WizardState | None,
        tools: list[Any] | None = None,
        *,
        include_stage_context: bool = True,
    ) -> Any:
        """Complete an LLM call with standard wizard context.

        Shared helper for clarification, validation, restart, and error
        response methods.  Builds the wizard metadata snapshot, optionally
        prepends stage context, and appends the caller-specific context
        block to the system prompt.

        Args:
            manager: ConversationManager instance
            additional_context: Caller-specific context (e.g. clarification
                instructions, validation errors)
            stage: Current stage metadata
            wizard_state: Current wizard state (None skips snapshot/context)
            tools: Available tools passed through to ``manager.complete``
            include_stage_context: Whether to prepend stage context
                (default True; False for transform errors which lack it)

        Returns:
            LLM response
        """
        wizard_snapshot = (
            {"wizard": self._build_wizard_metadata(wizard_state)}
            if wizard_state
            else None
        )
        base = manager.system_prompt
        if include_stage_context and wizard_state:
            stage_context = self.build_stage_context(stage, wizard_state)
            if stage_context:
                base = f"{base}\n\n{stage_context}"
        return await manager.complete(
            system_prompt_override=base + additional_context,
            tools=tools,
            metadata=wizard_snapshot,
        )

    async def _resolve_template_content(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        state: WizardState,
        response_template: str,
        wizard_snapshot: dict[str, Any],
    ) -> tuple[str, Any | None]:
        """Resolve template mode content for a stage.

        Shared by :meth:`generate_stage_response` and
        :meth:`stream_generate_stage_response`.  Handles context
        variable generation, template rendering, llm_assist, and
        conversation store persistence.

        Args:
            manager: ConversationManager instance.
            llm: LLM provider.
            stage: Current stage metadata.
            state: Current wizard state.
            response_template: The template string to render.
            wizard_snapshot: Wizard metadata dict for persistence.

        Returns:
            Tuple of ``(content, llm_response)``.  When llm_assist
            fires, ``llm_response`` is the full response object from
            ``manager.complete()`` (preserving usage, model, and other
            metadata).  For plain template renders, ``llm_response`` is
            ``None`` and the caller should wrap ``content`` via
            :meth:`create_template_response`.
        """
        stage_name = stage.get("name", "unknown")

        extra_context = await self._generate_context_variables(
            stage, state, llm
        )
        rendered = self._render_response_template(
            response_template, stage, state, extra_context=extra_context
        )

        user_message = self._get_last_user_message(manager)
        if stage.get("llm_assist") and user_message and self.is_help_request(user_message):
            assist_prompt = stage.get("llm_assist_prompt") or stage.get("prompt", "")
            scoped_prompt = (
                f"{manager.system_prompt}\n\n"
                f"The user is asking a question during the wizard. "
                f"Context: {assist_prompt}\n"
                f"Answer their question helpfully and concisely. "
                f"Do NOT change the topic or claim any actions."
            )
            logger.debug(
                "LLM assist for stage '%s' (user question detected)",
                stage_name,
            )
            response = await manager.complete(
                system_prompt_override=scoped_prompt,
                metadata=wizard_snapshot,
            )
            content = getattr(response, "content", str(response))
            return content, response

        logger.debug(
            "Template response for stage '%s' (%d chars)",
            stage_name,
            len(rendered),
        )
        await manager.add_message(
            role="assistant",
            content=rendered,
            metadata=wizard_snapshot,
        )
        return rendered, None

    def _prepare_llm_mode(
        self,
        manager: Any,
        stage: dict[str, Any],
        state: WizardState,
        tools: list[Any] | None,
    ) -> tuple[str, list[Any], Any]:
        """Prepare LLM mode parameters for a stage.

        Shared by :meth:`generate_stage_response` and
        :meth:`stream_generate_stage_response`.  Builds the
        enhanced system prompt, filters tools, and resolves the
        reasoning strategy.

        Args:
            manager: ConversationManager instance.
            stage: Current stage metadata.
            state: Current wizard state.
            tools: Available tools from the caller.

        Returns:
            Tuple of ``(enhanced_prompt, stage_tools, strategy)``.
        """
        stage_context = self.build_stage_context(stage, state)
        enhanced_prompt = f"{manager.system_prompt}\n\n{stage_context}"
        stage_tools = self.filter_tools_for_stage(stage, tools)
        strategy = self._resolve_stage_strategy(stage)
        return enhanced_prompt, stage_tools, strategy

    def _render_response_template(
        self,
        template_str: str,
        stage: dict[str, Any],
        state: WizardState,
        extra_context: dict[str, Any] | None = None,
    ) -> str:
        """Render a stage response template with wizard state data.

        Delegates to :attr:`_renderer` with bank/artifact injected via
        *extra_context*.

        Args:
            template_str: Jinja2 template string
            stage: Current stage metadata
            state: Current wizard state
            extra_context: Additional variables to inject into the
                template context (e.g. LLM-generated values). These
                are merged after collected data, so they can override
                data fields if names collide.

        Returns:
            Rendered response string
        """
        merged_extra: dict[str, Any] = {
            "bank": self._make_bank_accessor(),
            "artifact": self._get_artifact(),
        }
        if extra_context:
            merged_extra.update(extra_context)

        return self._renderer.render(
            template_str, stage, state, extra_context=merged_extra,
        )

    async def _generate_context_variables(
        self,
        stage: dict[str, Any],
        state: WizardState,
        llm: Any,
    ) -> dict[str, str]:
        """Generate LLM-produced context variables for template rendering.

        If the stage has a ``context_generation`` block, renders the prompt
        template with current state data and calls the LLM to produce a
        value for the named variable.

        Args:
            stage: Current stage metadata
            state: Current wizard state
            llm: LLM provider instance

        Returns:
            Dict of variable_name -> generated_value (empty if no
            context_generation defined or if generation fails with
            no fallback).
        """
        context_gen = stage.get("context_generation")
        if not context_gen:
            return {}

        prompt_template = context_gen.get("prompt")
        variable_name = context_gen.get("variable")
        fallback = context_gen.get("fallback", "")

        if not prompt_template or not variable_name:
            logger.warning(
                "Stage '%s' has context_generation but missing prompt or variable",
                stage.get("name", "unknown"),
            )
            return {}

        # Render the prompt template with current state data
        try:
            rendered_prompt = self._renderer.render(
                prompt_template, stage, state,
            )
        except Exception as e:
            logger.warning(
                "Failed to render context_generation prompt for stage '%s': %s",
                stage.get("name", "unknown"),
                e,
            )
            return {variable_name: fallback} if fallback else {}

        # Call the LLM with the rendered prompt
        try:
            from dataknobs_llm.llm import LLMMessage

            messages = [LLMMessage(role="user", content=rendered_prompt)]
            response = await llm.complete(messages)
            generated = response.content.strip() if response.content else ""

            if not generated:
                logger.debug(
                    "Empty LLM response for context_generation in stage '%s', using fallback",
                    stage.get("name", "unknown"),
                )
                return {variable_name: fallback}

            logger.debug(
                "Generated context variable '%s' for stage '%s' (%d chars)",
                variable_name,
                stage.get("name", "unknown"),
                len(generated),
            )
            return {variable_name: generated}

        except Exception as e:
            logger.warning(
                "Context generation failed for stage '%s': %s, using fallback",
                stage.get("name", "unknown"),
                e,
            )
            return {variable_name: fallback} if fallback else {}

    def _render_custom_context(
        self, stage: dict[str, Any], state: WizardState
    ) -> str:
        """Render context using custom Jinja2 template.

        Args:
            stage: Current stage metadata
            state: Current wizard state

        Returns:
            Rendered context string
        """
        extra_context = {
            "can_skip": self._fsm.can_skip() if self._fsm else False,
            "can_go_back": (
                self._fsm.can_go_back() if self._fsm else True
            ) and len(state.history) > 1,
        }

        return self._renderer.render(
            self._context_template,
            stage,
            state,
            extra_context=extra_context,
            mixed_mode=True,
        )

    def _build_default_context(
        self, stage: dict[str, Any], state: WizardState
    ) -> str:
        """Build context using default hardcoded format.

        This is the original _build_stage_context() logic, preserved for
        backward compatibility when no custom template is configured.

        Args:
            stage: Current stage metadata
            state: Current wizard state

        Returns:
            Context string to append to system prompt
        """
        banks = self._get_banks()
        lines = ["## Current Wizard Stage"]
        lines.append(f"Stage: {stage.get('name', 'unknown')}")

        if stage.get("prompt"):
            lines.append(f"Goal: {stage['prompt']}")

        if stage.get("help_text"):
            lines.append(f"Additional context: {stage['help_text']}")

        if stage.get("suggestions"):
            lines.append(f"Suggested responses: {', '.join(stage['suggestions'])}")

        # Add collected data context PROMINENTLY before instructions
        if state.data:
            filtered_data = {
                k: v for k, v in state.data.items() if not k.startswith("_")
            }
            if filtered_data:
                lines.append("\n## ALREADY COLLECTED - DO NOT ASK AGAIN")
                lines.append(
                    "The following information has already been provided by the user. "
                    "Do NOT ask for this information again:"
                )
                for key, value in filtered_data.items():
                    lines.append(f"- {key}: {value}")
                lines.append("")

        # CD-2: Inject collection progress during collection iterations
        # Sibling branching prunes prior inputs from the ancestor path,
        # so the LLM needs an explicit summary of what's been collected.
        if stage.get("collection_mode") == "collection":
            col_config = stage.get("collection_config", {})
            bank_name = col_config.get("bank_name", "")
            bank = banks.get(bank_name)
            if bank and bank.count() > 0:
                max_display = 20
                records = bank.all()
                lines.append(f"\n## Collection Progress ({bank_name})")
                lines.append(
                    f"{bank.count()} items collected so far:"
                )
                for record in records[:max_display]:
                    summary = ", ".join(
                        f"{v}" for k, v in record.data.items()
                        if not k.startswith("_")
                    )
                    lines.append(f"- {summary}")
                if len(records) > max_display:
                    lines.append(
                        f"- ... and {len(records) - max_display} more"
                    )
                lines.append("")

        # CD-3: Inject compiled artifact snapshot at guided-to-dynamic boundary
        stage_manages_tools = self._stage_manages_tools(stage)
        artifact = self._get_artifact()
        if stage_manages_tools and artifact:
            has_data = (
                artifact.fields
                or any(
                    bank.count() > 0
                    for bank in artifact.sections.values()
                )
            )
            if has_data:
                max_section_display = 20
                lines.append("\n## Collection Summary")
                compiled = artifact.compile()
                for key, value in compiled.items():
                    if key.startswith("_"):
                        continue
                    if isinstance(value, list):
                        continue  # Sections handled below
                    if value is not None:
                        lines.append(f"- {key}: {value}")
                for section_name, bank in artifact.sections.items():
                    if bank.count() > 0:
                        lines.append(
                            f"\n### {section_name} "
                            f"({bank.count()} records)"
                        )
                        for record in bank.all()[:max_section_display]:
                            summary = ", ".join(
                                f"{v}" for k, v in record.data.items()
                                if not k.startswith("_")
                            )
                            lines.append(f"- {summary}")
                        if bank.count() > max_section_display:
                            lines.append(
                                f"- ... and "
                                f"{bank.count() - max_section_display} more"
                            )
                lines.append("")

        if state.completed:
            lines.append("\nThe wizard is complete. Summarize what was collected.")
        else:
            lines.append(
                "\nYou MUST focus on the goal above. Do NOT ask about topics "
                "from other stages. Only gather information that has NOT "
                "already been collected. Be conversational and helpful."
            )

        return "\n".join(lines)

    def _resolve_stage_strategy(
        self, stage: dict[str, Any],
    ) -> ReasoningStrategy | None:
        """Resolve the reasoning strategy for a wizard stage.

        Looks up the strategy by name in the plugin registry, using the
        per-stage ``reasoning`` key with a fallback to the wizard-level
        ``default_tool_reasoning``.

        Returns ``None`` for the ``"single"`` path (direct
        ``manager.complete()``).

        Args:
            stage: Stage metadata dict

        Returns:
            A freshly-created strategy instance, or ``None`` for the
            single-call fast path.
        """
        from .registry import get_registry

        name = stage.get("reasoning") or self._default_tool_reasoning
        name = name.lower()

        if name == "single":
            return None

        if name == "wizard":
            from dataknobs_common.exceptions import ConfigurationError

            raise ConfigurationError(
                "Cannot use 'reasoning: wizard' inside a wizard stage. "
                "Nested wizards share the conversation manager, causing "
                "metadata collisions, and the inner wizard's multi-turn "
                "FSM cannot function within a single outer-stage turn. "
                "Use wizard subflows instead — see WIZARD_SUBFLOWS.md.",
                context={"stage": stage.get("name", "?")},
            )

        # Build config for strategy creation
        reasoning_config = dict(stage.get("reasoning_config") or {})
        reasoning_config["strategy"] = name

        # Inject wizard-level defaults that strategies may need
        if "max_iterations" not in reasoning_config:
            reasoning_config["max_iterations"] = self._get_max_iterations(stage)
        store_trace = stage.get("store_trace", self._default_store_trace)
        if "store_trace" not in reasoning_config:
            reasoning_config["store_trace"] = store_trace
        verbose = stage.get("verbose", self._default_verbose)
        if "verbose" not in reasoning_config:
            reasoning_config["verbose"] = verbose

        try:
            return get_registry().create(config=reasoning_config)
        except Exception as e:
            from dataknobs_common.exceptions import ConfigurationError

            stage_name = stage.get("name", "?")
            raise ConfigurationError(
                f"Failed to create strategy '{name}' for wizard "
                f"stage '{stage_name}': {e}",
                context={"stage": stage_name, "strategy": name},
            ) from e

    def _stage_manages_tools(self, stage: dict[str, Any]) -> bool:
        """Check if the stage's strategy declares manages_tools capability.

        Looks up the strategy class in the registry and calls its
        class-level ``capabilities()`` — no instance creation needed.

        Returns ``False`` for ``"single"`` or unregistered strategies.
        """
        from .registry import get_registry

        name = stage.get("reasoning") or self._default_tool_reasoning
        name = name.lower()
        if name == "single":
            return False

        registry = get_registry()
        factory = registry.get_factory(name)
        if factory is None:
            return False

        caps = factory.capabilities() if hasattr(factory, "capabilities") else None
        return bool(caps and caps.manages_tools)

    def _get_max_iterations(self, stage: dict[str, Any]) -> int:
        """Get maximum ReAct iterations for a stage.

        Args:
            stage: Stage metadata dict

        Returns:
            Max iterations (from stage config or default)
        """
        return stage.get("max_iterations") or self._default_max_iterations

    def _build_extra_context(self) -> dict[str, Any]:
        """Build the extra context dict for strategy delegation.

        Collects wizard-owned state (banks, artifacts, catalog) into a
        dict that strategies receive via ``extra_context`` kwargs.

        Returns:
            Dict of wizard context entries (may be empty).
        """
        extra_context: dict[str, Any] = {}
        banks = self._get_banks()
        artifact = self._get_artifact()
        catalog = self._get_catalog()
        if banks:
            extra_context["banks"] = banks
        if artifact:
            extra_context["artifact"] = artifact
        if catalog:
            extra_context["catalog"] = catalog
        return extra_context

    def _prepare_strategy_stage(
        self,
        strategy: ReasoningStrategy,
        manager: Any,
        stage: dict[str, Any],
        state: WizardState,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Shared setup for strategy-based stage response generation.

        Builds the extra context with lifecycle signal dicts, creates
        the prompt refresher closure, and injects wizard-owned runtime
        objects into the strategy instance.

        Returns:
            Tuple of (completion_signal, restart_signal) — mutable dicts
            that lifecycle tools populate during generation.
        """
        extra_context = self._build_extra_context()

        # Mutable signal dicts for lifecycle tools to communicate back
        completion_signal: dict[str, Any] = {"requested": False}
        extra_context["_completion_signal"] = completion_signal
        restart_signal: dict[str, Any] = {"requested": False}
        extra_context["_restart_signal"] = restart_signal

        # Build a prompt refresher so loop-based strategies can re-render
        # the system prompt between iterations.  Strategies that don't
        # support prompt refreshing simply ignore this kwarg.
        def prompt_refresher() -> str:
            fresh_context = self.build_stage_context(stage, state)
            return f"{manager.system_prompt}\n\n{fresh_context}"

        # Inject wizard-owned runtime objects into strategies that store
        # them as private instance attributes.  ReActReasoning accepts
        # these as constructor args, but strategies created via the
        # registry's from_config() path receive only serializable config.
        # We inject post-construction by targeting the private attributes
        # directly — strategies that don't have these attrs are skipped.
        artifact_registry = self._get_artifact_registry()
        review_executor = self._get_review_executor()
        context_builder = self._get_context_builder()
        _injections: dict[str, Any] = {
            "_artifact_registry": artifact_registry,
            "_review_executor": review_executor,
            "_context_builder": context_builder,
            "_extra_context": extra_context,
            "_prompt_refresher": prompt_refresher,
        }
        for attr, value in _injections.items():
            if value is not None and hasattr(strategy, attr):
                setattr(strategy, attr, value)

        return completion_signal, restart_signal

    @staticmethod
    def _read_lifecycle_signals(
        completion_signal: dict[str, Any],
        restart_signal: dict[str, Any],
    ) -> tuple[bool, str, bool]:
        """Read lifecycle tool signals after strategy execution.

        Returns:
            Tuple of (completion_requested, completion_summary,
            restart_requested).
        """
        return (
            bool(completion_signal.get("requested")),
            completion_signal.get("summary", ""),
            bool(restart_signal.get("requested")),
        )

    async def _strategy_stage_response(
        self,
        strategy: ReasoningStrategy,
        manager: Any,
        enhanced_prompt: str,
        stage: dict[str, Any],
        state: WizardState,
        tools: list[Any],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, bool, str, bool]:
        """Generate response by delegating to a registered strategy.

        Injects wizard context as kwargs that strategies may use or
        ignore.  Lifecycle signals (completion / restart) are communicated
        via mutable dicts inside ``extra_context``.

        Args:
            strategy: The resolved reasoning strategy instance.
            manager: ConversationManager instance.
            enhanced_prompt: Stage-aware system prompt.
            stage: Stage metadata dict.
            state: Current wizard state.
            tools: Available tools for this stage.
            metadata: Optional metadata to persist on conversation nodes.

        Returns:
            Tuple of (response, completion_requested, completion_summary,
            restart_requested).
        """
        completion_signal, restart_signal = self._prepare_strategy_stage(
            strategy, manager, stage, state,
        )

        response = await strategy.generate(
            manager=manager,
            llm=None,
            tools=tools,
            system_prompt_override=enhanced_prompt,
            metadata=metadata,
        )

        comp_req, comp_summary, restart_req = self._read_lifecycle_signals(
            completion_signal, restart_signal,
        )
        return response, comp_req, comp_summary, restart_req

    # =================================================================
    # Streaming response generation
    # =================================================================

    async def stream_generate_stage_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        state: WizardState,
        tools: list[Any] | None,
        stream_ctx: StreamStageContext,
        track_render: bool = True,
    ) -> AsyncIterator[LLMStreamResponse]:
        """Stream response appropriate for current stage.

        Streaming counterpart of :meth:`generate_stage_response`.  Three
        response paths:

        1. **Template mode** — renders template, yields single chunk.
        2. **Strategy mode** — delegates to
           :meth:`_stream_strategy_stage_response`.
        3. **Single LLM call** — streams via
           ``manager.stream_complete()``.

        Early returns (template, ``llm_assist``) emit as single final
        chunks because they are short, scoped responses that don't
        benefit from token-level streaming.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            state: Current wizard state
            tools: Available tools
            stream_ctx: Mutable context populated with lifecycle signals
                after the stream completes.
            track_render: If True (default), increment the stage's
                render count after the stream completes.  Pass False
                when (a) the caller has already incremented the render
                count explicitly (confirmation path), or (b) for
                subflow push paths where confirmation should fire on
                the user's next response.  The increment is placed
                after the final yield so that abandoned streams (via
                ``aclose()`` / ``GeneratorExit``) skip it — matching
                the pre-consolidation caller-side behavior.

        Yields:
            :class:`LLMStreamResponse` chunks.
        """
        stage_name = stage.get("name", "unknown")
        response_template = stage.get("response_template")
        wizard_snapshot = {"wizard": self._build_wizard_metadata(state)}

        # ── Template mode ────────────────────────────────────────
        is_conversation_mode = stage.get("mode") == "conversation"
        is_first_render = state.get_render_count(stage_name) == 0
        use_template = response_template and (
            not is_conversation_mode or is_first_render
        )

        if use_template:
            content, llm_response = await self._resolve_template_content(
                manager, llm, stage, state, response_template, wizard_snapshot,
            )
            # Propagate LLM metadata (usage, model) when llm_assist fired
            chunk_kwargs: dict[str, Any] = {
                "delta": content,
                "is_final": True,
                "finish_reason": "stop",
                "metadata": wizard_snapshot,
            }
            if llm_response is not None:
                usage = getattr(llm_response, "usage", None)
                model = getattr(llm_response, "model", None)
                if usage:
                    chunk_kwargs["usage"] = usage
                if model:
                    chunk_kwargs["model"] = model
            yield LLMStreamResponse(**chunk_kwargs)
            if track_render and response_template:
                state.increment_render_count(stage_name)
            return

        # ── LLM mode ────────────────────────────────────────────
        enhanced_prompt, stage_tools, strategy = self._prepare_llm_mode(
            manager, stage, state, tools,
        )

        logger.debug(
            "Streaming response for stage '%s' (tools=%s, strategy=%s)",
            stage_name,
            [getattr(t, "name", str(t)) for t in stage_tools] if stage_tools else None,
            type(strategy).__name__ if strategy else "single",
        )

        if strategy:
            async for chunk in self._stream_strategy_stage_response(
                strategy, manager, enhanced_prompt, stage, state,
                stage_tools, stream_ctx, metadata=wizard_snapshot,
            ):
                yield chunk
        else:
            # Single LLM call — stream via manager
            async for chunk in manager.stream_complete(
                system_prompt_override=enhanced_prompt,
                tools=stage_tools,
                metadata=wizard_snapshot,
            ):
                yield chunk

        # Increment after final yield — only reached when stream is
        # fully consumed.  If the caller abandons via aclose(),
        # GeneratorExit is thrown at the yield point and this code
        # is skipped.
        # Only fires for conversation-mode stages where response_template
        # is truthy but use_template was False (past first render).  For
        # pure LLM stages response_template is falsy so this is a no-op.
        if track_render and response_template:
            state.increment_render_count(stage_name)

    async def _stream_strategy_stage_response(
        self,
        strategy: ReasoningStrategy,
        manager: Any,
        enhanced_prompt: str,
        stage: dict[str, Any],
        state: WizardState,
        tools: list[Any],
        stream_ctx: StreamStageContext,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMStreamResponse]:
        """Stream response by delegating to a registered strategy.

        Streaming counterpart of :meth:`_strategy_stage_response`.
        Uses the same :meth:`_prepare_strategy_stage` setup, then
        iterates ``strategy.stream_generate()``.  After iteration
        completes, reads lifecycle signals into ``stream_ctx``.

        Args:
            strategy: The resolved reasoning strategy instance.
            manager: ConversationManager instance.
            enhanced_prompt: Stage-aware system prompt.
            stage: Stage metadata dict.
            state: Current wizard state.
            tools: Available tools for this stage.
            stream_ctx: Mutable context populated with lifecycle signals.
            metadata: Optional metadata to persist on conversation nodes.

        Yields:
            :class:`LLMStreamResponse` chunks.
        """
        completion_signal, restart_signal = self._prepare_strategy_stage(
            strategy, manager, stage, state,
        )

        async for chunk in strategy.stream_generate(
            manager=manager,
            llm=None,
            tools=tools,
            system_prompt_override=enhanced_prompt,
            metadata=metadata,
        ):
            if isinstance(chunk, LLMStreamResponse):
                yield chunk
            else:
                # Strategy yielded a complete LLMResponse — wrap as chunk
                content = getattr(chunk, "content", str(chunk))
                yield LLMStreamResponse(
                    delta=content,
                    is_final=True,
                    finish_reason="stop",
                    metadata=getattr(chunk, "metadata", {}),
                )

        # Read lifecycle signals into the mutable stream context
        (
            stream_ctx.tool_completion_requested,
            stream_ctx.tool_completion_summary,
            stream_ctx.tool_restart_requested,
        ) = self._read_lifecycle_signals(completion_signal, restart_signal)

    def _render_auto_advance_template(
        self, stage: dict[str, Any], state: WizardState
    ) -> str | None:
        """Render a stage's response_template for auto-advance collection.

        Used during auto-advance to capture message stage content before
        the stage is advanced past. Only renders if the stage has a
        response_template.

        Args:
            stage: Stage metadata dict
            state: Current wizard state

        Returns:
            Rendered template string, or None if the stage has no template
        """
        template = stage.get("response_template")
        if not template:
            return None

        rendered = self._render_response_template(template, stage, state)
        stage_name = stage.get("name", "unknown")
        logger.debug(
            "Rendered message stage '%s' template during auto-advance "
            "(%d chars)",
            stage_name,
            len(rendered),
        )
        # Track render count so re-visiting won't re-confirm
        state.increment_render_count(stage_name)
        return rendered

    def _build_clarification_groups(
        self,
        missing_fields: set[str],
        stage: dict[str, Any],
        wizard_state: WizardState | None = None,
    ) -> list[dict[str, Any]]:
        """Build grouped clarification questions for missing fields.

        Maps missing fields to configured groups.  Ungrouped fields get
        individual questions derived from their schema description.

        Args:
            missing_fields: Set of required field names still missing.
            stage: Current stage metadata (for schema access).
            wizard_state: Current wizard state (for derivation source
                checking).

        Returns:
            List of dicts with ``fields`` and ``question`` keys.
            Empty list if no missing fields remain after filtering.
        """
        if not missing_fields:
            return []

        properties = StageSchema.from_stage(stage).properties

        # Filter out derivable fields if configured
        effective_missing = set(missing_fields)
        if (
            self._clarification_exclude_derivable
            and self._field_derivations
            and wizard_state is not None
        ):
            available_fields = set(wizard_state.data) | effective_missing
            derivable = {
                rule.target
                for rule in self._field_derivations
                if rule.source in available_fields
            }
            effective_missing -= derivable

        if not effective_missing:
            return []

        # Match to configured groups
        grouped_fields: set[str] = set()
        result: list[dict[str, Any]] = []

        for group in self._clarification_groups:
            group_fields = set(group.get("fields", []))
            overlap = group_fields & effective_missing
            if overlap:
                result.append({
                    "fields": sorted(overlap),
                    "question": group["question"],
                })
                grouped_fields |= overlap

        # Ungrouped fields get individual entries from schema description
        for fld in sorted(effective_missing - grouped_fields):
            prop = properties.get(fld, {})
            description = prop.get("description", fld.replace("_", " "))
            result.append({
                "fields": [fld],
                "question": f"What is the {description}?",
            })

        return result

    @staticmethod
    def _get_last_user_message(manager: Any) -> str:
        """Extract the last user message from conversation.

        Prefers ``raw_content`` from message/node metadata (set by DynaBot
        when knowledge-base or memory context is prepended) so that
        downstream consumers (including the extraction pipeline) see the
        user's original message without context noise.

        Args:
            manager: ConversationManager instance

        Returns:
            Last user message text
        """
        messages = manager.get_messages()
        for msg in reversed(messages):
            if msg.get("role") == "user":
                # Check for raw_content in metadata (set by DynaBot when
                # knowledge/memory context was prepended to the message).
                metadata = msg.get("metadata") or {}
                raw = metadata.get("raw_content")
                if raw is not None:
                    return raw

                content = msg.get("content", "")
                # Handle plain string content
                if isinstance(content, str):
                    return content
                # Handle structured content (list of content parts)
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")
        return ""
