"""Wizard subflow lifecycle management.

Manages nested wizard subflows — push/pop lifecycle, data mapping
between parent and child flows, and subflow completion detection.
Extracted from ``wizard.py``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .observability import create_transition_record
from .wizard_types import SubflowContext, WizardState

if TYPE_CHECKING:
    from .wizard_fsm import WizardFSM

logger = logging.getLogger(__name__)

#: Renders the template a stage shows as the turn leaves it, or ``None``
#: when it has nothing left to say.  Satisfied by
#: :meth:`~dataknobs_bots.reasoning.wizard_response.WizardResponder.render_departing_stage`,
#: which the auto-advance loop already uses for the stages it steps past.
RenderDepartingStage = Callable[[dict[str, Any], WizardState], "str | None"]


def _renders_nothing(_stage: dict[str, Any], _state: WizardState) -> str | None:
    """The renderer a manager has before one is injected.

    A manager built without a responder has nothing to render *with*, so
    silence is the honest answer rather than a swallowed one.  The single
    production constructor injects the real renderer one line after it
    injects the condition evaluator.
    """
    return None


class SubflowManager:
    """Manages the subflow stack and active FSM switching.

    Owns the ``_active_subflow_fsm`` reference and provides the
    ``get_active_fsm()`` method that returns either the current
    subflow FSM or the main FSM.

    Args:
        fsm: The main (top-level) :class:`WizardFSM`.
        evaluate_condition: Callback to evaluate transition conditions
            (signature: ``(condition: str, data: dict) -> bool``).
    """

    def __init__(
        self,
        fsm: WizardFSM,
        evaluate_condition: Callable[[str, dict[str, Any]], bool],
    ) -> None:
        self._fsm = fsm
        self._evaluate_condition = evaluate_condition
        self._render_departing_stage: RenderDepartingStage = _renders_nothing
        self._active_subflow_fsm: WizardFSM | None = None

    def set_render_departing_stage(self, render: RenderDepartingStage) -> None:
        """Supply the renderer :meth:`pop_if_ended` gives the end stage.

        Injected rather than passed per call for the same reason
        :meth:`set_evaluate_condition` is: this class is constructed
        before :class:`WizardResponder`, which owns the renderer.  Passing
        it per call would also let the two pop sites drift apart on which
        renderer they use, which is the divergence :meth:`pop_if_ended`
        exists to remove.
        """
        self._render_departing_stage = render

    def set_evaluate_condition(
        self,
        evaluate_condition: Callable[[str, dict[str, Any]], bool],
    ) -> None:
        """Replace the condition evaluator.

        Used to resolve the circular dependency between SubflowManager
        and WizardResponder: SubflowManager is created first with a
        placeholder, then the real evaluator is injected once
        WizardResponder is constructed.
        """
        self._evaluate_condition = evaluate_condition

    # -- Active FSM access ---------------------------------------------------

    def get_active_fsm(self) -> WizardFSM:
        """Get the currently active FSM (subflow or main), per the turn.

        Reads ``_active_subflow_fsm``, which a turn maintains. Callers
        holding a :class:`WizardState` should prefer
        :meth:`fsm_for_state`, which derives the same answer from the
        stack and is therefore also correct outside a turn.

        Returns:
            The active WizardFSM instance.
        """
        return self._active_subflow_fsm if self._active_subflow_fsm else self._fsm

    def fsm_for_state(self, state: WizardState) -> WizardFSM:
        """The FSM *state* says is active, asked of the state rather than the turn.

        The canonical answer to "which FSM owns the stage in play", and
        the reason it lives here rather than on either caller: this class
        owns the stack, and both the strategy and the navigator were
        answering the question for themselves.

        :meth:`get_active_fsm` answers the same question from
        ``_active_subflow_fsm``, an attribute a *turn* maintains. That is
        correct during a turn and stale outside one -- after an undo it
        still names the FSM of the turn being undone, and a snapshot is
        taken outside a turn by definition. The rule here is the one
        every writer of that attribute already applies (a restore, a push
        and a pop all set it from ``subflow_stack``), so this agrees with
        it whenever the attribute is fresh and beats it when it is not.

        Prefer this wherever a ``WizardState`` is in hand.
        :meth:`get_active_fsm` remains for the two places inside this
        class's own push/pop that have already updated the attribute and
        not yet the stack, or vice versa.

        Args:
            state: Wizard state naming the subflow stack, if any.

        Returns:
            The subflow's FSM when a subflow is on the stack and
            resolvable, otherwise the main FSM.
        """
        if state.subflow_stack:
            subflow = self._fsm.get_subflow(state.subflow_stack[-1].subflow_network)
            if subflow is not None:
                return subflow
        return self._fsm

    @property
    def active_subflow_fsm(self) -> WizardFSM | None:
        """The currently active subflow FSM, or ``None`` if in main flow."""
        return self._active_subflow_fsm

    @active_subflow_fsm.setter
    def active_subflow_fsm(self, value: WizardFSM | None) -> None:
        self._active_subflow_fsm = value

    # -- Push / pop ----------------------------------------------------------

    def should_push(
        self,
        wizard_state: WizardState,
        user_message: str,
    ) -> dict[str, Any] | None:
        """Check if the current transition should push a subflow.

        Examines the transitions from the current stage to see if any
        matching transition is a subflow transition.

        Args:
            wizard_state: Current wizard state.
            user_message: User message for context.

        Returns:
            Subflow config dict if should push, ``None`` otherwise.
        """
        # Guard: Don't push subflow if already in one
        # This prevents duplicate pushes after state restoration
        if wizard_state.is_in_subflow:
            return None

        active_fsm = self.get_active_fsm()
        stage_meta = active_fsm.current_metadata
        stage_name = stage_meta.get("name", "?")

        # Check each transition for subflow marker
        declined: list[str] = []
        for transition in stage_meta.get("transitions", []):
            if not transition.get("is_subflow_transition"):
                continue

            # Evaluate condition if present
            condition = transition.get("condition")
            if condition and not self._evaluate_condition(
                condition,
                wizard_state.data,
            ):
                declined.append(condition)
                continue

            # This transition matches and is a subflow transition
            subflow_config: dict[str, Any] = transition.get("subflow_config", {})
            logger.debug(
                "Subflow guard on stage '%s' satisfied by %s: pushing '%s'",
                stage_name,
                f"condition {condition!r}" if condition else "an unconditional transition",
                subflow_config.get("network", "?"),
            )
            return subflow_config

        # A decline is otherwise indistinguishable from there being no
        # subflow transition at all, from a misspelled condition, and from
        # one that raised -- every one of them shows up as nothing
        # happening. Name the conditions that were asked and said no.
        if declined:
            logger.debug(
                "Subflow guard on stage '%s' declined: %d condition(s) not satisfied: %s",
                stage_name,
                len(declined),
                ", ".join(repr(c) for c in declined),
            )

        return None

    def handle_push(
        self,
        wizard_state: WizardState,
        subflow_config: dict[str, Any],
        user_message: str,
    ) -> bool:
        """Push a subflow onto the stack.

        Saves parent state and switches to the subflow FSM.

        Args:
            wizard_state: Current wizard state.
            subflow_config: Subflow configuration dict.
            user_message: User message for context.

        Returns:
            True if subflow was pushed successfully.
        """
        network_name = subflow_config.get("network")
        if not network_name:
            logger.warning("Subflow config missing 'network' field")
            return False

        # Get the subflow FSM
        subflow_fsm = self._fsm.get_subflow(network_name)
        if not subflow_fsm:
            logger.warning("Subflow '%s' not found in registry", network_name)
            return False

        # Create subflow context to save parent state
        from_stage = wizard_state.current_stage
        subflow_context = SubflowContext(
            parent_stage=from_stage,
            parent_data=dict(wizard_state.data),
            parent_history=list(wizard_state.history),
            return_stage=subflow_config.get("return_stage", from_stage),
            result_mapping=subflow_config.get("result_mapping", {}),
            subflow_network=network_name,
        )

        # Apply data mapping (parent -> subflow)
        data_mapping = subflow_config.get("data_mapping", {})
        subflow_data = _apply_data_mapping(wizard_state.data, data_mapping)

        # Push subflow context
        wizard_state.subflow_stack.append(subflow_context)

        # Reset subflow FSM and set initial data
        subflow_fsm.restart()
        subflow_fsm.restore(
            {
                "current_stage": subflow_fsm.current_stage,
                "data": subflow_data,
            }
        )

        # Switch to subflow
        self._active_subflow_fsm = subflow_fsm

        # Update wizard state for subflow
        to_stage = subflow_fsm.current_stage
        duration_ms = (time.time() - wizard_state.stage_entry_time) * 1000

        # Record the push transition
        transition = create_transition_record(
            from_stage=from_stage,
            to_stage=to_stage,
            trigger="subflow_push",
            duration_in_stage_ms=duration_ms,
            data_snapshot=wizard_state.data.copy(),
            user_input=user_message,
            subflow_push=network_name,
            subflow_depth=wizard_state.subflow_depth,
        )
        wizard_state.transitions.append(transition)

        # Update wizard state
        wizard_state.current_stage = to_stage
        wizard_state.replace_data(subflow_data)
        wizard_state.history = [to_stage]
        wizard_state.stage_entry_time = time.time()

        logger.info(
            "Pushed subflow '%s': %s -> %s (depth=%d)",
            network_name,
            from_stage,
            to_stage,
            wizard_state.subflow_depth,
        )

        return True

    def handle_pop(self, wizard_state: WizardState) -> bool:
        """Pop the current subflow and return to parent.

        Applies result mapping and restores parent state.

        Args:
            wizard_state: Current wizard state.

        Returns:
            True if subflow was popped successfully.
        """
        if not wizard_state.subflow_stack:
            return False

        # Pop the subflow context
        subflow_context = wizard_state.subflow_stack.pop()
        network_name = subflow_context.subflow_network
        from_stage = wizard_state.current_stage
        duration_ms = (time.time() - wizard_state.stage_entry_time) * 1000

        # Apply result mapping (subflow -> parent)
        parent_data = dict(subflow_context.parent_data)
        result_data = _apply_result_mapping(
            wizard_state.data,
            subflow_context.result_mapping,
        )
        parent_data.update(result_data)

        # Restore parent state
        return_stage = subflow_context.return_stage

        # Record the pop transition
        transition = create_transition_record(
            from_stage=from_stage,
            to_stage=return_stage,
            trigger="subflow_pop",
            duration_in_stage_ms=duration_ms,
            data_snapshot=wizard_state.data.copy(),
            subflow_pop=network_name,
            subflow_depth=wizard_state.subflow_depth,
        )
        wizard_state.transitions.append(transition)

        # Update wizard state
        wizard_state.current_stage = return_stage
        wizard_state.replace_data(parent_data)
        wizard_state.history = subflow_context.parent_history
        if return_stage not in wizard_state.history:
            wizard_state.history.append(return_stage)
        wizard_state.stage_entry_time = time.time()

        # Switch back to parent FSM (or next subflow if nested)
        if wizard_state.subflow_stack:
            parent_subflow = wizard_state.subflow_stack[-1].subflow_network
            self._active_subflow_fsm = self._fsm.get_subflow(parent_subflow)
        else:
            self._active_subflow_fsm = None

        # Restore parent FSM state
        active_fsm = self.get_active_fsm()
        active_fsm.restore(
            {
                "current_stage": return_stage,
                "data": parent_data,
            }
        )

        logger.info(
            "Popped subflow '%s': %s -> %s (depth=%d)",
            network_name,
            from_stage,
            return_stage,
            wizard_state.subflow_depth,
        )

        return True

    def unwind_all(self, state: WizardState, *, user_message: str | None = None) -> list[str]:
        """Tear the whole subflow stack down, recording a pop for each frame.

        Restart and an amendment that jumps out of a subflow both have to
        leave the stack empty, and both used to do it by clearing the
        list and nulling the attribute at the call site -- two partial
        spellings of a teardown this class owns, neither of which wrote
        anything to the audit trail. A consumer pairing ``subflow_push``
        with ``subflow_pop`` records, or reconstructing depth from them,
        then saw a push nothing ever closed.

        Unlike :meth:`handle_pop` this applies no result mapping and
        restores no parent data: the callers replace the data wholesale
        (restart empties it, an amendment keeps the completed flow's).
        What it guarantees is the part that wedged the wizard -- an empty
        stack and no active subflow FSM -- plus the record of it.

        Args:
            state: Wizard state (mutated in place).
            user_message: User message for the transition records.

        Returns:
            The networks unwound, outermost last. Empty in the main flow,
            where this is a no-op.
        """
        if not state.subflow_stack:
            return []

        unwound: list[str] = []
        from_stage = state.current_stage
        duration_ms = (time.time() - state.stage_entry_time) * 1000
        while state.subflow_stack:
            context = state.subflow_stack.pop()
            unwound.append(context.subflow_network)
            state.transitions.append(
                create_transition_record(
                    from_stage=from_stage,
                    to_stage=context.return_stage,
                    trigger="subflow_unwind",
                    duration_in_stage_ms=duration_ms,
                    data_snapshot=state.data.copy(),
                    user_input=user_message,
                    subflow_pop=context.subflow_network,
                    subflow_depth=state.subflow_depth,
                )
            )
            from_stage = context.return_stage
            duration_ms = 0.0

        self._active_subflow_fsm = None
        logger.info("Unwound subflow stack: %s", ", ".join(unwound))
        return unwound

    def should_pop(self, wizard_state: WizardState) -> bool:
        """Check if the current stage is a subflow end state.

        Args:
            wizard_state: Current wizard state.

        Returns:
            True if current stage is an end stage and we're in a subflow.
        """
        if not wizard_state.is_in_subflow:
            return False

        active_fsm = self.get_active_fsm()
        return active_fsm.is_end_stage(wizard_state.current_stage)

    def pop_if_ended(self, wizard_state: WizardState) -> str | None:
        """Pop a finished subflow, returning what its end stage had to say.

        An ``is_end`` subflow stage is entered and left inside one turn --
        :meth:`should_pop` asks only for a non-empty stack and an end
        stage -- so this is the only moment it can speak.  Its template
        used to render nowhere, and a subflow whose failing exit exists to
        say *nothing was saved, and here is why* said it on the one stage
        that was never on screen.

        **The order is the whole point.** :meth:`handle_pop` swaps the
        active FSM and replaces ``wizard_state.data`` with the parent's,
        so a render placed after it names the parent's stage and
        interpolates the parent's data.  Rendering first is what makes the
        message the subflow's own.

        This exists as one method rather than two lines at each call site
        because there are two pop sites -- the post-transition sequence
        and the auto-advance loop -- and a stage left by either must say
        the same thing.  Clearing ``completed`` belongs here for the same
        reason: the subflow ended, the wizard did not, and both sites were
        already spelling that out identically.

        Args:
            wizard_state: Current wizard state, mutated in place when a
                pop happens.

        Returns:
            The end stage's rendered template, or ``None`` when no pop
            happened *or* the stage had no template to offer.  Callers
            collect the message and re-read the active FSM; none of them
            needs to tell those two cases apart.
        """
        if not self.should_pop(wizard_state):
            return None

        # Asked of the state's stage, because that is the stage
        # ``should_pop`` just approved -- ``current_metadata`` would ask
        # the FSM's, and the two need not agree outside a turn.
        departing = self.get_active_fsm().stage_metadata_for(wizard_state.current_stage)
        message = self._render_departing_stage(departing, wizard_state)
        # ``handle_pop``'s bool is unread on purpose: its only ``False`` is
        # an empty stack, which ``should_pop`` has already ruled out.
        self.handle_pop(wizard_state)
        wizard_state.completed = False
        return message


# ---------------------------------------------------------------------------
# Data mapping helpers (pure functions, no state)
# ---------------------------------------------------------------------------


def _apply_data_mapping(
    source_data: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    """Apply data mapping from parent to subflow.

    Args:
        source_data: Source data dict (parent wizard data).
        mapping: Dict mapping parent field names to subflow field names.

    Returns:
        Mapped data dict for subflow.
    """
    if not mapping:
        return {}

    result: dict[str, Any] = {}
    for parent_field, subflow_field in mapping.items():
        if parent_field in source_data:
            result[subflow_field] = source_data[parent_field]

    return result


def _apply_result_mapping(
    source_data: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    """Apply result mapping from subflow to parent.

    Args:
        source_data: Source data dict (subflow wizard data).
        mapping: Dict mapping subflow field names to parent field names.

    Returns:
        Mapped data dict for parent.
    """
    if not mapping:
        return {}

    result: dict[str, Any] = {}
    for subflow_field, parent_field in mapping.items():
        if subflow_field in source_data:
            result[parent_field] = source_data[subflow_field]

    return result
