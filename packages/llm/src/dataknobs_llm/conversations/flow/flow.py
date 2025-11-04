"""Conversation flow definitions.

This module provides high-level abstractions for defining conversation flows
that are executed using the FSM engine.
"""

from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class TransitionCondition(ABC):
    """Base class for transition conditions.

    Transition conditions determine which state to transition to based on
    the LLM response and conversation context.
    """

    @abstractmethod
    async def evaluate(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate if this transition should be taken.

        Args:
            response: The LLM response text
            context: Current conversation context

        Returns:
            True if this transition should be taken
        """
        pass

    @abstractmethod
    def to_fsm_function(self) -> str:
        """Convert to FSM function name for registration.

        Returns:
            Function name that will be registered in FSM function registry
        """
        pass


@dataclass
class FlowState:
    """Single state in a conversation flow.

    Each state represents a point in the conversation where the system
    generates a response using a prompt from the prompt library.

    Attributes:
        prompt_name: Name of the prompt in the prompt library
        transitions: Map of condition names to next state names
        transition_conditions: Map of condition names to TransitionCondition objects
        max_loops: Maximum times this state can be revisited (None = unlimited)
        prompt_params: Static parameters to pass to the prompt
        on_enter: Hook called when entering this state
        on_exit: Hook called when exiting this state
        metadata: Additional metadata for this state
    """

    prompt_name: str
    transitions: Dict[str, str] = field(default_factory=dict)
    transition_conditions: Dict[str, TransitionCondition] = field(default_factory=dict)

    # Loop detection
    max_loops: int | None = None

    # Prompt parameters
    prompt_params: Dict[str, Any] = field(default_factory=dict)

    # Hooks
    on_enter: Callable | None = None
    on_exit: Callable | None = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate state configuration."""
        if not self.prompt_name:
            raise ValueError("prompt_name is required")

        # Ensure all transitions have corresponding conditions
        for cond_name in self.transitions.keys():
            if cond_name not in self.transition_conditions:
                raise ValueError(
                    f"Transition '{cond_name}' has no corresponding condition"
                )


@dataclass
class ConversationFlow:
    """Complete conversation flow definition.

    A conversation flow defines the states and transitions that make up
    a conversation, managed by the FSM engine.

    Attributes:
        name: Unique name for this flow
        initial_state: Name of the starting state
        states: Map of state names to FlowState objects
        max_total_loops: Maximum total transitions (prevents infinite loops)
        timeout_seconds: Maximum execution time (None = no timeout)
        initial_context: Initial context variables
        description: Human-readable description
        version: Semantic version string
        metadata: Additional metadata
    """

    name: str
    initial_state: str
    states: Dict[str, FlowState]

    # Global settings
    max_total_loops: int = 10
    timeout_seconds: float | None = None

    # Context
    initial_context: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: str | None = None
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate flow configuration."""
        if not self.name:
            raise ValueError("Flow name is required")

        if not self.initial_state:
            raise ValueError("initial_state is required")

        if not self.states:
            raise ValueError("Flow must have at least one state")

        if self.initial_state not in self.states:
            raise ValueError(
                f"initial_state '{self.initial_state}' not found in states"
            )

        # Validate all transition targets exist
        for state_name, state in self.states.items():
            for target_state in state.transitions.values():
                if target_state not in self.states and target_state != "end":
                    raise ValueError(
                        f"State '{state_name}' transitions to unknown state '{target_state}'"
                    )

    def get_state(self, state_name: str) -> FlowState:
        """Get a state by name.

        Args:
            state_name: Name of the state

        Returns:
            FlowState object

        Raises:
            KeyError: If state not found
        """
        if state_name not in self.states:
            raise KeyError(f"State '{state_name}' not found")
        return self.states[state_name]

    def get_reachable_states(self, from_state: str) -> List[str]:
        """Get all states reachable from a given state.

        Args:
            from_state: Starting state name

        Returns:
            List of reachable state names
        """
        state = self.get_state(from_state)
        return list(state.transitions.values())

    def validate_flow(self) -> List[str]:
        """Validate the flow and return any warnings.

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        # Check for unreachable states
        reachable = {self.initial_state}
        to_visit = [self.initial_state]

        while to_visit:
            current = to_visit.pop()
            for next_state in self.get_reachable_states(current):
                if next_state != "end" and next_state not in reachable:
                    reachable.add(next_state)
                    to_visit.append(next_state)

        for state_name in self.states.keys():
            if state_name not in reachable:
                warnings.append(f"State '{state_name}' is unreachable")

        # Check for states with no exit transitions
        for state_name, state in self.states.items():
            if not state.transitions:
                warnings.append(
                    f"State '{state_name}' has no exit transitions (potential dead end)"
                )

        return warnings
