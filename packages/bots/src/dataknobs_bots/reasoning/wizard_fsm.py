"""WizardFSM wrapper for wizard-specific FSM operations.

This module provides a thin wrapper around AdvancedFSM that adds
wizard-specific conveniences like stage metadata access, navigation
helpers, and serialization for persistence.
"""

import logging
from typing import Any, Callable

from dataknobs_fsm.api.advanced import AdvancedFSM, StepResult
from dataknobs_fsm.execution.context import ExecutionContext

logger = logging.getLogger(__name__)


class WizardFSM:
    """Wrapper around AdvancedFSM with wizard-specific conveniences.

    Provides a simplified interface for wizard operations including:
    - Stage metadata access (prompts, schemas, suggestions)
    - Navigation helpers (back, skip, restart)
    - State serialization for persistence
    - Stage-specific tool and configuration access

    Attributes:
        _fsm: Underlying AdvancedFSM instance
        _stage_metadata: Dict mapping stage names to metadata
        _settings: Wizard-level settings (auto_advance_filled_stages, etc.)
        _context: Current execution context
    """

    def __init__(
        self,
        fsm: AdvancedFSM,
        stage_metadata: dict[str, dict[str, Any]],
        settings: dict[str, Any] | None = None,
    ):
        """Initialize WizardFSM.

        Args:
            fsm: AdvancedFSM instance to wrap
            stage_metadata: Dict mapping stage names to their metadata
            settings: Wizard-level settings dict (optional)
        """
        self._fsm = fsm
        self._stage_metadata = stage_metadata
        self._settings = settings or {}
        self._context: ExecutionContext | None = None

    @property
    def settings(self) -> dict[str, Any]:
        """Get wizard-level settings.

        Returns:
            Dict containing wizard settings like auto_advance_filled_stages
        """
        return self._settings

    @property
    def current_stage(self) -> str:
        """Get current stage name.

        Returns:
            Name of the current stage
        """
        if self._context and self._context.current_state:
            return self._context.current_state
        return self._find_start_stage()

    @property
    def current_metadata(self) -> dict[str, Any]:
        """Get metadata for current stage.

        Returns:
            Dict containing current stage's metadata
        """
        return self._stage_metadata.get(self.current_stage, {})

    @property
    def stages(self) -> dict[str, dict[str, Any]]:
        """Get all stage metadata.

        Returns a copy to prevent external modification.

        Returns:
            Dict mapping stage name to stage configuration dict.
        """
        return dict(self._stage_metadata)

    @property
    def stage_names(self) -> list[str]:
        """Get ordered list of stage names.

        Returns:
            List of stage names in definition order.
        """
        return list(self._stage_metadata.keys())

    @property
    def stage_count(self) -> int:
        """Get total number of stages.

        Returns:
            Number of stages in the wizard.
        """
        return len(self._stage_metadata)

    def get_stage_prompt(self, stage: str | None = None) -> str:
        """Get prompt for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            Stage prompt string
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("prompt", "")

    def get_stage_schema(self, stage: str | None = None) -> dict[str, Any] | None:
        """Get validation schema for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            JSON Schema dict or None
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("schema")

    def get_stage_tools(self, stage: str | None = None) -> list[str]:
        """Get available tool names for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            List of tool names available in the stage
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("tools", [])

    def get_stage_suggestions(self, stage: str | None = None) -> list[str]:
        """Get quick-reply suggestions for a stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            List of suggestion strings
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("suggestions", [])

    def get_transition_condition(
        self, from_stage: str, to_stage: str
    ) -> str | None:
        """Get the condition expression for a transition.

        Args:
            from_stage: Source stage name
            to_stage: Target stage name

        Returns:
            Condition expression string, or None if no condition
        """
        stage_meta = self._stage_metadata.get(from_stage, {})
        transitions = stage_meta.get("transitions", [])

        for transition in transitions:
            if transition.get("target") == to_stage:
                return transition.get("condition")

        return None

    def can_skip(self, stage: str | None = None) -> bool:
        """Check if stage can be skipped.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if stage can be skipped
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("can_skip", False)

    def can_go_back(self, stage: str | None = None) -> bool:
        """Check if back navigation is allowed.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if back navigation is allowed
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("can_go_back", True)

    def is_start_stage(self, stage: str | None = None) -> bool:
        """Check if stage is a start stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if stage is marked as start
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("is_start", False)

    def is_end_stage(self, stage: str | None = None) -> bool:
        """Check if stage is an end stage.

        Args:
            stage: Stage name (defaults to current stage)

        Returns:
            True if stage is marked as end
        """
        stage = stage or self.current_stage
        return self._stage_metadata.get(stage, {}).get("is_end", False)

    def step(self, data: dict[str, Any]) -> StepResult:
        """Execute one FSM step with given data.

        Creates or updates the execution context and executes
        a single FSM transition.

        Args:
            data: Data dict for transition evaluation

        Returns:
            StepResult with transition details
        """
        if not self._context:
            self._context = self._fsm.create_context(data)
        else:
            # Update context data
            if isinstance(self._context.data, dict):
                self._context.data.update(data)
            else:
                self._context.data = data

        return self._fsm.execute_step_sync(self._context)

    def go_back(self, history: list[str]) -> bool:
        """Navigate to previous stage.

        Args:
            history: List of visited stage names

        Returns:
            True if back navigation succeeded
        """
        if len(history) <= 1:
            return False

        if not self.can_go_back():
            return False

        # Get previous stage from history
        previous_stage = history[-2] if len(history) >= 2 else None
        if not previous_stage:
            return False

        # Restore context to previous stage
        if self._context:
            self._context.set_state(previous_stage)
            return True

        return False

    def restart(self) -> None:
        """Reset wizard to start stage.

        Clears the execution context to start fresh.
        """
        self._context = None

    def serialize(self) -> dict[str, Any]:
        """Serialize wizard state for persistence.

        Returns:
            Dict containing serializable wizard state
        """
        return {
            "current_stage": self.current_stage,
            "history": self._get_history(),
            "data": self._get_data(),
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Restore wizard from serialized state.

        Args:
            state: Previously serialized state dict
        """
        current_stage = state.get("current_stage")
        data = state.get("data", {})

        if current_stage:
            # Create new context with restored state
            self._context = self._fsm.create_context(data)
            self._context.set_state(current_stage)

    def _find_start_stage(self) -> str:
        """Find the start stage.

        Returns:
            Name of the start stage
        """
        for name, meta in self._stage_metadata.items():
            if meta.get("is_start"):
                return name
        # Fallback to first stage
        return (
            next(iter(self._stage_metadata.keys()))
            if self._stage_metadata
            else "start"
        )

    def _get_history(self) -> list[str]:
        """Get stage history from context.

        Returns:
            List of visited stage names
        """
        if self._context:
            # Try to get from context metadata
            history = self._context.metadata.get("state_history", [])
            if history:
                return list(history)
        return [self.current_stage]

    def _get_data(self) -> dict[str, Any]:
        """Get current data from context.

        Returns:
            Current data dict
        """
        if self._context:
            if isinstance(self._context.data, dict):
                return self._context.data.copy()
        return {}


def create_wizard_fsm(
    fsm_config: dict[str, Any],
    stage_metadata: dict[str, dict[str, Any]],
    custom_functions: dict[str, Callable[..., Any]] | None = None,
    settings: dict[str, Any] | None = None,
) -> WizardFSM:
    """Factory function to create a WizardFSM instance.

    Args:
        fsm_config: FSM configuration dict
        stage_metadata: Stage metadata dict
        custom_functions: Optional custom functions to register
        settings: Wizard-level settings dict (optional)

    Returns:
        Configured WizardFSM instance
    """
    from dataknobs_fsm.api.advanced import create_advanced_fsm

    advanced_fsm = create_advanced_fsm(fsm_config, custom_functions=custom_functions)
    return WizardFSM(advanced_fsm, stage_metadata, settings=settings)
