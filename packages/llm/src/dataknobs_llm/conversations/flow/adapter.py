"""Conversation flow adapter for FSM execution.

This module provides the adapter that converts high-level ConversationFlow
definitions into FSM configurations and manages execution.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field

from dataknobs_fsm.api.simple import SimpleFSM
from dataknobs_fsm.core.data_modes import DataHandlingMode

from .flow import ConversationFlow, FlowState, TransitionCondition

logger = logging.getLogger(__name__)


@dataclass
class FlowExecutionState:
    """Tracks execution state during flow execution.

    Attributes:
        loop_counts: Count of visits to each state
        total_transitions: Total number of transitions made
        current_state: Current state name
        context: Current context dictionary
        history: List of (state_name, response) tuples
    """

    loop_counts: Dict[str, int] = field(default_factory=dict)
    total_transitions: int = 0
    current_state: str | None = None
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[tuple] = field(default_factory=list)

    def increment_loop_count(self, state_name: str) -> int:
        """Increment and return loop count for a state."""
        count = self.loop_counts.get(state_name, 0) + 1
        self.loop_counts[state_name] = count
        return count

    def add_to_history(self, state_name: str, response: str):
        """Add a state transition to history."""
        self.history.append((state_name, response))


class ConversationFlowAdapter:
    """Adapts ConversationFlow to FSM execution.

    This class converts high-level conversation flow definitions into
    FSM configurations and manages the execution lifecycle.
    """

    def __init__(
        self,
        flow: ConversationFlow,
        prompt_builder: Any,  # AsyncPromptBuilder
        llm: Any | None = None  # AsyncLLMProvider
    ):
        """Initialize the adapter.

        Args:
            flow: ConversationFlow definition
            prompt_builder: Prompt builder for rendering prompts
            llm: Optional LLM provider (can be in context)
        """
        self.flow = flow
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.execution_state = FlowExecutionState()
        self._function_registry: Dict[str, Any] = {}

    def to_fsm_config(self) -> Dict[str, Any]:
        """Convert ConversationFlow to FSM configuration.

        Returns:
            FSM configuration dictionary
        """
        states = []
        arcs = []

        # Create FSM states from flow states
        for state_name, flow_state in self.flow.states.items():
            # Determine state type
            is_start = (state_name == self.flow.initial_state)
            is_end = (len(flow_state.transitions) == 0)

            fsm_state = {
                "name": state_name,
                "is_start": is_start,
                "is_end": is_end,
                "transform": self._create_state_transform_function(state_name, flow_state)
            }

            states.append(fsm_state)

        # Create FSM arcs from flow transitions
        for state_name, flow_state in self.flow.states.items():
            for condition_name, target_state in flow_state.transitions.items():
                condition = flow_state.transition_conditions[condition_name]

                arc = {
                    "from": state_name,
                    "to": target_state,
                    "name": f"{state_name}_to_{target_state}_{condition_name}",
                    "pre_test": self._register_condition_function(
                        condition_name,
                        condition,
                        state_name
                    )
                }

                arcs.append(arc)

        # Build complete FSM config
        config = {
            "name": self.flow.name,
            "version": self.flow.version,
            "description": self.flow.description or f"Conversation flow: {self.flow.name}",
            "states": states,
            "arcs": arcs,
            "functions": self._function_registry
        }

        return config

    def _create_state_transform_function(
        self,
        state_name: str,
        flow_state: FlowState
    ) -> str:
        """Create and register transform function for a state.

        Args:
            state_name: Name of the state
            flow_state: FlowState configuration

        Returns:
            Function name for FSM registration
        """
        function_name = f"transform_{state_name}"

        async def transform_func(data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            """Transform function for state execution."""
            # Check loop limits
            loop_count = self.execution_state.increment_loop_count(state_name)

            if flow_state.max_loops and loop_count > flow_state.max_loops:
                logger.warning(
                    f"State '{state_name}' exceeded max loops ({flow_state.max_loops})"
                )
                return {
                    **data,
                    "_error": f"Max loops exceeded for state {state_name}",
                    "_force_end": True
                }

            # Check total transition limit
            self.execution_state.total_transitions += 1
            if self.execution_state.total_transitions > self.flow.max_total_loops:
                logger.warning(
                    f"Flow exceeded max total transitions ({self.flow.max_total_loops})"
                )
                return {
                    **data,
                    "_error": "Max total transitions exceeded",
                    "_force_end": True
                }

            # Call on_enter hook if defined
            if flow_state.on_enter:
                try:
                    await flow_state.on_enter(state_name, data, context)
                except Exception as e:
                    logger.error(f"on_enter hook failed for state '{state_name}': {e}")

            # Merge prompt params with data
            prompt_params = {
                **data,
                **flow_state.prompt_params,
                **context,
                "state": state_name,
                "loop_count": loop_count
            }

            # Render and build prompt
            try:
                result = await self.prompt_builder.build_prompt(
                    prompt_name=flow_state.prompt_name,
                    params=prompt_params
                )
            except Exception as e:
                logger.error(f"Failed to build prompt for state '{state_name}': {e}")
                return {
                    **data,
                    "_error": f"Prompt building failed: {e!s}",
                    "response": f"[Error in state {state_name}]"
                }

            # Store response in data
            response = result.content if hasattr(result, 'content') else str(result)

            # Add to history
            self.execution_state.add_to_history(state_name, response)

            # Call on_exit hook if defined
            if flow_state.on_exit:
                try:
                    await flow_state.on_exit(state_name, data, context)
                except Exception as e:
                    logger.error(f"on_exit hook failed for state '{state_name}': {e}")

            # Update data with response
            return {
                **data,
                "response": response,
                "state": state_name,
                "loop_count": loop_count,
                "history": list(self.execution_state.history)
            }

        # Register function
        self._function_registry[function_name] = transform_func

        return function_name

    def _register_condition_function(
        self,
        condition_name: str,
        condition: TransitionCondition,
        state_name: str
    ) -> str:
        """Register a condition function for arc pre_test.

        Args:
            condition_name: Name of the condition
            condition: TransitionCondition instance
            state_name: Name of the source state

        Returns:
            Function name for FSM registration
        """
        function_name = f"condition_{state_name}_{condition_name}"

        async def condition_func(data: Dict[str, Any], context: Dict[str, Any]) -> bool:
            """Condition function for arc evaluation."""
            # Check if forced to end
            if data.get("_force_end"):
                return False

            response = data.get("response", "")

            # Evaluate condition
            try:
                result = await condition.evaluate(response, {**context, **data})
                logger.debug(
                    f"Condition '{condition_name}' for state '{state_name}': {result}"
                )
                return result
            except Exception as e:
                logger.error(
                    f"Condition '{condition_name}' evaluation failed: {e}"
                )
                return False

        # Register function
        self._function_registry[function_name] = condition_func

        return function_name

    async def execute(
        self,
        initial_data: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        """Execute the conversation flow.

        Args:
            initial_data: Initial data for the flow

        Returns:
            Final data after flow execution
        """
        # Initialize execution state
        self.execution_state = FlowExecutionState(
            current_state=self.flow.initial_state,
            context={**self.flow.initial_context}
        )

        # Prepare initial data
        data = initial_data or {}
        data = {**data, **self.flow.initial_context}

        # Convert to FSM config
        fsm_config = self.to_fsm_config()

        # Create and execute FSM
        fsm = SimpleFSM(fsm_config, data_mode=DataHandlingMode.COPY)

        try:
            result = await fsm.process_async(data)
            return result.get("data", result)
        except Exception as e:
            logger.error(f"Flow execution failed: {e}")
            return {
                **data,
                "_error": str(e),
                "_execution_failed": True
            }

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of flow execution.

        Returns:
            Dictionary with execution statistics
        """
        return {
            "total_transitions": self.execution_state.total_transitions,
            "loop_counts": dict(self.execution_state.loop_counts),
            "current_state": self.execution_state.current_state,
            "history_length": len(self.execution_state.history),
            "states_visited": list(self.execution_state.loop_counts.keys())
        }
