"""Navigation commands, amendments, and conversation tree branching.

Extracted from :mod:`wizard` (item 77b).  Contains the
:class:`WizardNavigator` class which handles back/skip/restart commands,
post-completion amendment detection, and conversation tree branching for
revisited stages.

The navigator receives its dependencies through constructor injection —
direct references to collaborators (FSM, subflows, hooks, etc.) and
callable callbacks for orchestrator-owned operations that have not yet
been extracted (response generation, FSM stepping).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .observability import create_transition_record
from .wizard_types import NavigationCommandConfig, NavigationConfig, WizardState

if TYPE_CHECKING:
    from .wizard_fsm import WizardFSM
    from .wizard_hooks import WizardHooks
    from .wizard_subflows import SubflowManager

logger = logging.getLogger(__name__)


class WizardNavigator:
    """Handles navigation commands, amendments, and conversation tree branching.

    Two-level architecture:

    - **FSM-level methods** (``navigate_back``, ``navigate_skip``,
      ``navigate_restart``) perform the FSM operation only.  Reusable by
      both ``generate()`` (conversational) and ``advance()`` (non-conversational).
    - **Conversational methods** (``handle_navigation``, ``handle_amendment``,
      ``execute_restart``) add response generation and tree branching on
      top.  Used by ``generate()`` only.

    Args:
        fsm: Main wizard FSM instance.
        subflows: Subflow manager (owns active subflow FSM).
        hooks: Lifecycle hooks (enter, exit, complete, restart).
        navigation_config: Wizard-level navigation keyword config.
        consistent_lifecycle: When True, back fires the enter hook on
            the destination stage, and skip runs the full post-transition
            lifecycle (enter hook, auto-advance, subflow pop, complete
            hook) matching forward transitions.  Back does NOT run
            auto-advance or subflow pop — it targets an explicit history
            entry.
        allow_amendments: Whether post-completion edits are enabled.
        section_to_stage_mapping: Custom section→stage mappings for amendments.
        extractor: Schema extractor for amendment detection (may be None).
        banks: Memory bank instances (cleared on restart).
        artifact: ArtifactCorpus instance (cleared on restart, may be None).
        catalog: Artifact catalog for auto-save before restart (may be None).
        execute_fsm_step: Callback to execute an FSM step with runtime
            context injection.  Signature:
            ``(state, *, user_message, trigger, llm) -> (from_stage, step_result)``
        run_post_transition_lifecycle: Callback to run post-transition
            lifecycle (subflow pop, auto-advance, hooks).  Signature:
            ``(state, *, llm) -> list[str]``
        generate_stage_response: Callback to generate a stage response.
            Signature: ``(manager, llm, stage, state, tools) -> response``
        prepend_messages_to_response: Callback to prepend auto-advance
            messages to a response.  Signature:
            ``(response, messages) -> None``
    """

    def __init__(
        self,
        *,
        fsm: WizardFSM,
        subflows: SubflowManager,
        hooks: WizardHooks | None,
        navigation_config: NavigationConfig,
        consistent_lifecycle: bool,
        allow_amendments: bool,
        section_to_stage_mapping: dict[str, str],
        extractor: Any | None,
        banks: dict[str, Any],
        artifact: Any | None,
        catalog: Any | None,
        execute_fsm_step: Callable[..., Awaitable[tuple[str, Any]]],
        run_post_transition_lifecycle: Callable[..., Awaitable[list[str]]],
        generate_stage_response: Callable[..., Awaitable[Any]],
        prepend_messages_to_response: Callable[[Any, list[str]], None],
    ) -> None:
        self._fsm = fsm
        self._subflows = subflows
        self._hooks = hooks
        self._navigation_config = navigation_config
        self._consistent_lifecycle = consistent_lifecycle
        self._allow_amendments = allow_amendments
        self._section_to_stage_mapping = section_to_stage_mapping
        self._extractor = extractor
        self._banks = banks
        self._artifact = artifact
        self._catalog = catalog
        self._execute_fsm_step = execute_fsm_step
        self._run_post_transition_lifecycle = run_post_transition_lifecycle
        self._generate_stage_response = generate_stage_response
        self._prepend_messages_to_response = prepend_messages_to_response

    # ------------------------------------------------------------------
    # Public API — conversational (used by generate())
    # ------------------------------------------------------------------

    async def handle_navigation(
        self,
        message: str,
        state: WizardState,
        manager: Any,
        llm: Any,
    ) -> Any | None:
        """Handle navigation commands (back, skip, restart).

        Resolves the effective navigation config for the current stage,
        matches the user message against configured keywords, and
        dispatches to the appropriate action method.

        Args:
            message: User message text
            state: Current wizard state
            manager: ConversationManager instance
            llm: LLM provider

        Returns:
            Response if navigation handled, None otherwise
        """
        lower = message.lower().strip()
        nav = self._resolve_navigation_config(state.current_stage)

        if nav.back.enabled and lower in nav.back.keywords:
            return await self._execute_back(message, state, manager, llm)

        if nav.skip.enabled and lower in nav.skip.keywords:
            return await self._execute_skip(message, state, manager, llm)

        if nav.restart.enabled and lower in nav.restart.keywords:
            return await self.execute_restart(message, state, manager, llm)

        return None  # Not a navigation command

    async def handle_amendment(
        self,
        message: str,
        state: WizardState,
        manager: Any,
        llm: Any,
        tools: list[Any] | None,
    ) -> Any | None:
        """Handle post-completion amendment detection and re-opening.

        Checks if the user's message requests an edit to a completed
        wizard.  If an amendment is detected, re-opens the wizard to
        the target stage, branches the conversation tree, and generates
        the stage response.

        Args:
            message: User message text
            state: Current wizard state
            manager: ConversationManager instance
            llm: LLM provider
            tools: Available tools list

        Returns:
            Response if amendment detected and handled, None otherwise.
        """
        amendment = await self.detect_amendment(message, state, llm)
        if not amendment:
            return None

        target_stage = amendment["target_stage"]
        from_stage = state.current_stage
        duration_ms = (time.time() - state.stage_entry_time) * 1000

        # Re-open wizard to target stage
        state.completed = False
        state.current_stage = target_stage
        if target_stage not in state.history:
            state.history.append(target_stage)

        # Restore FSM to target stage
        active_fsm = self._subflows.get_active_fsm()
        active_fsm.restore({
            "current_stage": target_stage,
            "data": state.data,
        })

        # Record the amendment transition
        transition = create_transition_record(
            from_stage=from_stage,
            to_stage=target_stage,
            trigger="amendment",
            duration_in_stage_ms=duration_ms,
            data_snapshot=state.data.copy(),
            user_input=message,
            subflow_depth=state.subflow_depth,
        )
        state.transitions.append(transition)
        state.stage_entry_time = time.time()

        logger.info("Amendment: re-opening wizard at %s", target_stage)

        # Generate response for the re-opened stage
        stage = active_fsm.current_metadata
        await self.branch_for_revisited_stage(manager, target_stage)
        response = await self._generate_stage_response(
            manager, llm, stage, state, tools,
        )
        return response

    # ------------------------------------------------------------------
    # Public API — FSM-level (used by advance())
    # ------------------------------------------------------------------

    async def navigate_back(
        self,
        state: WizardState,
        *,
        user_message: str | None = None,
    ) -> bool:
        """Execute back navigation on the FSM.

        Performs the FSM-level back operation without generating a
        response.  Used by both ``handle_navigation`` (conversational)
        and ``advance()`` (non-conversational).

        Hook coverage (when ``consistent_lifecycle=True``):
        - Enter hook: fires on the destination stage
        - Exit hook: intentionally NOT fired — back returns to a
          previous stage rather than completing the current one
        - Auto-advance / subflow pop: NOT run — back targets an
          explicit history entry

        Args:
            state: Wizard state (mutated in place).
            user_message: Optional user message for the transition record.

        Returns:
            True if navigation succeeded, False if at beginning.
        """
        active_fsm = self._subflows.get_active_fsm()
        if not active_fsm.can_go_back() or len(state.history) <= 1:
            return False

        from_stage = state.current_stage
        duration_ms = (time.time() - state.stage_entry_time) * 1000

        state.history.pop()
        state.current_stage = state.history[-1]
        active_fsm.restore(
            {"current_stage": state.current_stage, "data": state.data}
        )
        state.clarification_attempts = 0
        state.skip_extraction = False

        transition = create_transition_record(
            from_stage=from_stage,
            to_stage=state.current_stage,
            trigger="navigation_back",
            duration_in_stage_ms=duration_ms,
            user_input=user_message,
        )
        state.transitions.append(transition)
        state.stage_entry_time = time.time()

        if self._consistent_lifecycle and self._hooks:
            await self._hooks.trigger_enter(state.current_stage, state.data)

        return True

    async def navigate_skip(
        self,
        state: WizardState,
        *,
        user_message: str | None = None,
    ) -> tuple[bool, list[str]]:
        """Execute skip navigation on the FSM.

        Performs the skip operation (mark skipped, apply defaults, step
        FSM, run post-transition lifecycle) without generating a response.

        Hook coverage (when ``consistent_lifecycle=True``):
        - Full post-transition lifecycle: enter hook, auto-advance,
          subflow pop, complete hook — matching the forward path
          because skip moves forward through the wizard
        - Exit hook: intentionally NOT fired — the stage is being
          skipped, not completed

        Args:
            state: Wizard state (mutated in place).
            user_message: Optional user message for condition evaluation
                and transition record.

        Returns:
            Tuple of (success, auto_advance_messages).  ``success`` is
            False when the stage cannot be skipped; ``auto_advance_messages``
            contains rendered templates from any stages auto-advanced
            through during the post-transition lifecycle.
        """
        active_fsm = self._subflows.get_active_fsm()
        if not active_fsm.can_skip():
            return False, []

        state.data[f"_skipped_{state.current_stage}"] = True
        skip_default = active_fsm.current_metadata.get("skip_default")
        if skip_default and isinstance(skip_default, dict):
            state.data.update(skip_default)
        state.clarification_attempts = 0
        # Clear skip_extraction — if the user skips a stage they were
        # auto-advanced to, the stale flag must not carry over to suppress
        # extraction at the next stage.
        state.skip_extraction = False

        await self._execute_fsm_step(
            state, user_message=user_message, trigger="navigation_skip",
        )
        auto_msgs: list[str] = []
        if self._consistent_lifecycle:
            auto_msgs = await self._run_post_transition_lifecycle(state)
        return True, auto_msgs

    async def navigate_restart(
        self, state: WizardState, user_message: str = "",
    ) -> None:
        """Execute restart navigation on the FSM.

        Hook coverage: only the restart hook fires (via
        ``restart_cleanup``).  Enter/exit hooks are intentionally NOT
        fired — restart is a full state reset, not a stage transition.

        Args:
            state: Wizard state (mutated in place).
            user_message: User message that triggered the restart.
        """
        await self.restart_cleanup(state, user_message, trigger="api_restart")

    # ------------------------------------------------------------------
    # Public API — cleanup and branching (used by generate() directly)
    # ------------------------------------------------------------------

    async def restart_cleanup(
        self,
        state: WizardState,
        message: str,
        trigger: str = "restart",
    ) -> None:
        """Reset wizard state, banks, and artifact for a fresh start.

        If a catalog is configured and the artifact passes validation,
        auto-saves to the catalog before clearing — preventing data loss
        when the LLM calls restart without saving first.

        Performs the cleanup portion of a restart without generating a
        response.  Callers are responsible for branching and response
        generation after cleanup completes.

        Args:
            state: Current wizard state (mutated in place).
            message: User message that triggered the restart.
            trigger: Transition trigger label for the audit trail.
        """
        # Auto-save artifact to catalog before clearing.  This catches
        # the common case where the LLM calls restart_wizard (or the
        # auto-restart guard fires) without calling save_to_catalog first.
        if self._catalog and self._artifact:
            try:
                errors = self._artifact.validate()
                if not errors:
                    self._catalog.save(self._artifact)
                    logger.info(
                        "Auto-saved artifact '%s' to catalog before restart",
                        self._artifact.name,
                    )
                else:
                    logger.debug(
                        "Skipping auto-save before restart "
                        "(validation errors: %s)",
                        errors,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to auto-save artifact before restart: %s", e
                )

        from_stage = state.current_stage
        duration_ms = (time.time() - state.stage_entry_time) * 1000

        # Trigger restart hook if configured
        if self._hooks:
            await self._hooks.trigger_restart()

        self._fsm.restart()
        to_stage = self._fsm.current_stage

        # Record the restart transition (preserving transition history)
        transition = create_transition_record(
            from_stage=from_stage,
            to_stage=to_stage,
            trigger=trigger,
            duration_in_stage_ms=duration_ms,
            data_snapshot=state.data.copy(),
            user_input=message,
        )
        # Preserve transition history but clear other state
        previous_transitions = [*state.transitions, transition]

        state.current_stage = to_stage
        state.data = {}
        state.history = [state.current_stage]
        state.completed = False
        state.clarification_attempts = 0
        state.skip_extraction = False
        state.transitions = previous_transitions
        state.stage_entry_time = time.time()

        # Clear all memory banks on restart (clean slate)
        for bank in self._banks.values():
            bank.clear()
        if self._artifact:
            self._artifact.clear_fields()
            if self._artifact.is_finalized:
                self._artifact.unfinalize()

    async def detect_amendment(
        self,
        message: str,
        state: WizardState,
        llm: Any,
    ) -> dict[str, Any] | None:
        """Detect if a post-completion message requests an edit.

        Uses the extractor to determine if the user wants to modify
        something and which section/stage they want to change.

        Args:
            message: User's message
            state: Current wizard state
            llm: LLM for extraction (unused, extractor has its own)

        Returns:
            Dict with target_stage if amendment detected, None otherwise
        """
        if not self._extractor:
            # Without extractor, can't detect amendments
            return None

        # Simple schema to detect edit intent
        amendment_schema = {
            "type": "object",
            "properties": {
                "wants_edit": {
                    "type": "boolean",
                    "description": (
                        "Does the user want to change, update, or modify "
                        "something that was already configured?"
                    ),
                },
                "target_section": {
                    "type": "string",
                    "description": (
                        "What section or aspect do they want to change? "
                        "Options: llm, model, identity, name, knowledge, kb, "
                        "tools, behavior, template, config"
                    ),
                },
            },
        }

        try:
            result = await self._extractor.extract(
                text=message,
                schema=amendment_schema,
                context={"state": "completed", "prompt": "Detect edit requests"},
            )

            if result.data.get("wants_edit"):
                target = result.data.get("target_section", "")
                target_stage = self.map_section_to_stage(target)
                if target_stage:
                    return {"target_stage": target_stage}
        except Exception as e:
            logger.debug("Amendment detection failed: %s", e)

        return None

    async def branch_for_revisited_stage(
        self, manager: Any, stage_name: str
    ) -> None:
        """Branch the conversation tree when revisiting a wizard stage.

        If the tree already contains an assistant response for
        ``stage_name``, positions the tree so the next message becomes a
        sibling of that node (a new branch from the same parent).

        Does nothing on first visit (no previous node to branch from).

        Args:
            manager: ConversationManager instance.
            stage_name: Wizard stage about to be (re-)entered.
        """
        prev_node_id = self._find_stage_node_id(manager, stage_name)
        if prev_node_id is not None:
            try:
                await manager.branch_from(prev_node_id)
                logger.debug(
                    "Branched conversation tree for revisited stage '%s' "
                    "(sibling of node '%s')",
                    stage_name,
                    prev_node_id,
                )
            except (ValueError, AttributeError):
                # Manager may not support branch_from (e.g. test doubles).
                # Gracefully degrade — tree will just chain deeper.
                logger.debug(
                    "branch_from not available; skipping tree branching "
                    "for stage '%s'",
                    stage_name,
                )

    # ------------------------------------------------------------------
    # Private — conversational execute methods
    # ------------------------------------------------------------------

    async def _execute_back(
        self,
        message: str,
        state: WizardState,
        manager: Any,
        llm: Any,
    ) -> Any:
        """Execute back navigation.

        Args:
            message: Original user message
            state: Current wizard state
            manager: ConversationManager instance
            llm: LLM provider

        Returns:
            Response for the previous stage, or an explanation if
            back navigation is not possible.
        """
        if await self.navigate_back(state, user_message=message):
            stage = self._fsm.current_metadata
            await self.branch_for_revisited_stage(
                manager, state.current_stage
            )
            response = await self._generate_stage_response(
                manager, llm, stage, state, None
            )
            return response
        # Can't go back - inform user
        return await manager.complete(
            system_prompt_override=(
                manager.system_prompt
                + "\n\nThe user asked to go back but we're at the beginning. "
                "Kindly explain we can't go back further and continue with "
                "the current step."
            ),
        )

    async def _execute_skip(
        self,
        message: str,
        state: WizardState,
        manager: Any,
        llm: Any,
    ) -> Any:
        """Execute skip navigation.

        Delegates to ``navigate_skip()`` for FSM stepping and lifecycle,
        then generates the next stage's response.  Bypasses the extraction
        pipeline entirely, consistent with ``_execute_back()`` and
        ``execute_restart()``.

        Args:
            message: Original user message
            state: Current wizard state
            manager: ConversationManager instance
            llm: LLM provider

        Returns:
            Response for the next stage, or an explanation if skip is
            not allowed.
        """
        if not self._fsm.can_skip():
            return await manager.complete(
                system_prompt_override=(
                    manager.system_prompt
                    + "\n\nThe user asked to skip this step but it's required. "
                    "Kindly explain the step cannot be skipped and ask for the "
                    "information needed."
                ),
            )

        _, auto_advance_messages = await self.navigate_skip(
            state, user_message=message,
        )

        stage = self._subflows.get_active_fsm().current_metadata
        response = await self._generate_stage_response(
            manager, llm, stage, state, None
        )
        if auto_advance_messages:
            self._prepend_messages_to_response(response, auto_advance_messages)
        return response

    # ------------------------------------------------------------------
    # Public — full conversational restart (cleanup + response)
    # ------------------------------------------------------------------

    async def execute_restart(
        self,
        message: str,
        state: WizardState,
        manager: Any,
        llm: Any,
    ) -> Any:
        """Execute a full conversational restart (cleanup + branch + response).

        Used by ``generate()`` for both keyword-triggered and tool-initiated
        restarts where a full response cycle is needed (not just the
        FSM-level cleanup from ``navigate_restart``).

        Args:
            message: Original user message
            state: Current wizard state
            manager: ConversationManager instance
            llm: LLM provider

        Returns:
            Response for the restarted first stage.
        """
        await self.restart_cleanup(state, message)

        stage = self._fsm.current_metadata
        await self.branch_for_revisited_stage(
            manager, state.current_stage
        )
        response = await self._generate_stage_response(
            manager, llm, stage, state, None
        )
        return response

    # ------------------------------------------------------------------
    # Private — config resolution and mapping
    # ------------------------------------------------------------------

    def _resolve_navigation_config(self, stage_name: str) -> NavigationConfig:
        """Resolve the effective navigation config for a stage.

        Per-stage overrides use **replace** semantics: if a stage specifies
        keywords for a command, those fully replace the wizard-level keywords
        for that command.  Commands not mentioned in the stage override
        inherit from the wizard-level config.

        Args:
            stage_name: Current stage name.

        Returns:
            Resolved ``NavigationConfig`` for the given stage.
        """
        stage_meta = self._fsm._stage_metadata.get(stage_name, {})
        stage_nav = stage_meta.get("navigation")
        if not stage_nav:
            return self._navigation_config

        # Merge: per-command, stage overrides wizard-level
        def _merge_command(
            base: NavigationCommandConfig,
            override_raw: dict[str, Any] | None,
        ) -> NavigationCommandConfig:
            if override_raw is None:
                return base
            keywords_raw = override_raw.get("keywords")
            keywords = (
                tuple(k.lower() for k in keywords_raw)
                if keywords_raw is not None
                else base.keywords
            )
            enabled = override_raw.get("enabled", base.enabled)
            return NavigationCommandConfig(keywords=keywords, enabled=enabled)

        return NavigationConfig(
            back=_merge_command(
                self._navigation_config.back, stage_nav.get("back")
            ),
            skip=_merge_command(
                self._navigation_config.skip, stage_nav.get("skip")
            ),
            restart=_merge_command(
                self._navigation_config.restart, stage_nav.get("restart")
            ),
        )

    def map_section_to_stage(self, section: str) -> str | None:
        """Map a section name to a wizard stage name.

        First checks custom mapping from settings, then falls back to
        built-in defaults.

        Args:
            section: Section identifier from extraction

        Returns:
            Stage name, or None if no mapping found
        """
        if not section:
            return None

        section_lower = section.lower().strip()

        # Check custom mapping first
        if (
            self._section_to_stage_mapping
            and section_lower in self._section_to_stage_mapping
        ):
            return self._section_to_stage_mapping[section_lower]

        # Default mappings for common wizard patterns
        default_mapping = {
            "llm": "configure_llm",
            "model": "configure_llm",
            "ai": "configure_llm",
            "identity": "configure_identity",
            "name": "configure_identity",
            "knowledge": "configure_knowledge",
            "kb": "configure_knowledge",
            "rag": "configure_knowledge",
            "tools": "configure_tools",
            "behavior": "configure_behavior",
            "template": "select_template",
            "config": "review",
        }

        mapped_stage = default_mapping.get(section_lower)
        if mapped_stage and mapped_stage in self._fsm._stage_metadata:
            return mapped_stage

        return None

    # ------------------------------------------------------------------
    # Private — conversation tree helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_ancestor_of(ancestor_id: str, descendant_id: str) -> bool:
        """Check if *ancestor_id* is an ancestor of *descendant_id*.

        Both IDs use dot-delimited segment notation (e.g. ``"0.1.2"``).
        The root node (empty string) is considered an ancestor of every
        node.  A node is NOT considered its own ancestor.

        Args:
            ancestor_id: Candidate ancestor node ID.
            descendant_id: Node ID to check against.

        Returns:
            ``True`` if *ancestor_id* is a strict ancestor of
            *descendant_id*.
        """
        if not ancestor_id:
            # Root is ancestor of everything
            return True
        return descendant_id.startswith(ancestor_id + ".")

    @staticmethod
    def _find_stage_node_id(manager: Any, stage_name: str) -> str | None:
        """Find the entry-point assistant node for the given wizard stage.

        Searches the conversation tree for assistant response nodes whose
        metadata records ``wizard.current_stage == stage_name``.  Only
        nodes that are **ancestors of the current position** are
        considered — nodes on other branches (e.g. from a previous
        wizard run) are ignored.  This prevents post-restart stage
        transitions from grafting new nodes onto old branches.

        Among ancestor matches, the first in DFS order (the shallowest
        stage entry node) is returned.  This is critical for ReAct
        stages, where the ReAct loop creates many assistant nodes all
        tagged with the same ``current_stage``.  Branching from the
        *last* (deepest) node would nest the new branch inside the old
        ReAct subtree, leaking prior tool-call context into the new
        visit.  Branching from the *first* (entry) node makes the new
        visit a sibling of the entire previous subtree — proper
        isolation.

        Args:
            manager: ConversationManager instance (must have ``state``).
            stage_name: Wizard stage name to search for.

        Returns:
            ``node_id`` string of the stage entry node, or ``None``.
        """
        from dataknobs_llm.conversations.storage import ConversationNode

        state = getattr(manager, "state", None)
        if state is None:
            return None

        current_node_id: str = state.current_node_id

        matches = state.message_tree.find_nodes(
            lambda n: (
                isinstance(n.data, ConversationNode)
                and n.data.message.role == "assistant"
                and n.data.metadata.get("wizard", {}).get("current_stage")
                == stage_name
            ),
        )
        if not matches:
            return None

        # Return the first DFS match that is an ancestor of the current
        # position.  Matches on other branches are stale references from
        # prior wizard runs and must be ignored.
        for match in matches:
            if not isinstance(match.data, ConversationNode):
                continue
            node_id = match.data.node_id
            if WizardNavigator._is_ancestor_of(node_id, current_node_id):
                return node_id

        return None
