"""Wizard subflow lifecycle management.

Manages nested wizard subflows — push/pop lifecycle, data mapping
between parent and child flows, and subflow completion detection.
Extracted from ``wizard.py`` (item 77a).
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
        self._active_subflow_fsm: WizardFSM | None = None

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
        """Get the currently active FSM (subflow or main).

        Returns:
            The active WizardFSM instance.
        """
        return self._active_subflow_fsm if self._active_subflow_fsm else self._fsm

    @property
    def active_subflow_fsm(self) -> WizardFSM | None:
        """The currently active subflow FSM, or ``None`` if in main flow."""
        return self._active_subflow_fsm

    @active_subflow_fsm.setter
    def active_subflow_fsm(self, value: WizardFSM | None) -> None:
        self._active_subflow_fsm = value

    # -- Push / pop ----------------------------------------------------------

    def should_push(
        self, wizard_state: WizardState, user_message: str,
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

        # Check each transition for subflow marker
        for transition in stage_meta.get("transitions", []):
            if not transition.get("is_subflow_transition"):
                continue

            # Evaluate condition if present
            condition = transition.get("condition")
            if condition and not self._evaluate_condition(
                condition, wizard_state.data,
            ):
                continue

            # This transition matches and is a subflow transition
            return transition.get("subflow_config", {})

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
        subflow_fsm.restore({
            "current_stage": subflow_fsm.current_stage,
            "data": subflow_data,
        })

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
        wizard_state.data = subflow_data
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
            wizard_state.data, subflow_context.result_mapping,
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
        wizard_state.data = parent_data
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
        active_fsm.restore({
            "current_stage": return_stage,
            "data": parent_data,
        })

        logger.info(
            "Popped subflow '%s': %s -> %s (depth=%d)",
            network_name,
            from_stage,
            return_stage,
            wizard_state.subflow_depth,
        )

        return True

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
