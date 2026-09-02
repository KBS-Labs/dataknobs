"""Conversation flow adapter for FSM execution.

This module provides the adapter that converts high-level ConversationFlow
definitions into FSM configurations and manages execution.
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field

from dataknobs_common.exceptions import OperationError
from dataknobs_fsm.api.async_simple import AsyncSimpleFSM
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
        stop_reason: Why the flow stopped early, if a loop guard tripped.
            The engine reports such a run as "no valid transitions", which
            names the state but not the cause; this carries the cause.
    """

    loop_counts: Dict[str, int] = field(default_factory=dict)
    total_transitions: int = 0
    current_state: str | None = None
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[tuple] = field(default_factory=list)
    stop_reason: str | None = None

    def increment_loop_count(self, state_name: str) -> int:
        """Increment and return loop count for a state."""
        count = self.loop_counts.get(state_name, 0) + 1
        self.loop_counts[state_name] = count
        return count

    def add_to_history(self, state_name: str, response: str) -> None:
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
        llm: Any | None = None,  # AsyncLLMProvider
    ):
        """Initialize the adapter.

        Args:
            flow: ConversationFlow definition
            prompt_builder: Prompt builder for rendering prompts
            llm: Optional LLM provider. Seeded into the flow context as
                ``_llm_provider``, where LLMClassifierCondition reads it, so a
                condition needs no llm_config of its own.
        """
        self.flow = flow
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.execution_state = FlowExecutionState()
        self._function_registry: Dict[str, Any] = {}

    def to_fsm_config(self) -> Dict[str, Any]:
        """Convert ConversationFlow to FSM configuration.

        Building the states and arcs registers their transform and condition
        callables in ``_function_registry`` as a side effect, so this must be
        called *before* that registry is read — see :meth:`execute`, which
        passes it to the FSM as ``custom_functions``. The registry is not part
        of the returned config: ``FSMConfig`` forbids unknown keys, and
        callables have never been configurable by value.

        Returns:
            FSM configuration dictionary
        """
        states = []
        arcs = []

        # Create FSM states from flow states
        for state_name, flow_state in self.flow.states.items():
            # Determine state type
            is_start = state_name == self.flow.initial_state
            is_end = len(flow_state.transitions) == 0

            fsm_state = {
                "name": state_name,
                "is_start": is_start,
                "is_end": is_end,
                "transform": self._create_state_transform_function(state_name, flow_state),
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
                        condition_name, condition, state_name
                    ),
                }

                arcs.append(arc)

        # Build complete FSM config
        config = {
            "name": self.flow.name,
            "version": self.flow.version,
            "description": self.flow.description or f"Conversation flow: {self.flow.name}",
            "states": states,
            "arcs": arcs,
        }

        return config

    def _create_state_transform_function(self, state_name: str, flow_state: FlowState) -> str:
        """Create and register transform function for a state.

        Args:
            state_name: Name of the state
            flow_state: FlowState configuration

        Returns:
            Function name for FSM registration
        """
        function_name = f"transform_{state_name}"

        async def transform_func(data: Dict[str, Any], function_context: Any) -> Dict[str, Any]:
            """Render the state's prompt and record the response.

            Both parameters are required. A registered transform is resolved
            through the builder into an ``InterfaceWrapper`` that dispatches
            ``(data, function_context)``, and the engine's non-interface path
            probes arity — it calls ``func(state_obj)`` first and only falls
            back to ``(data, context)`` on ``TypeError``. A defaulted second
            parameter would satisfy that probe and silently bind the state
            definition to ``data``.

            ``function_context`` is the engine's ``FunctionContext`` (injected
            resources, shared variables, current network) — a dataclass, not a
            mapping. The *conversation* context is the adapter's own, and is
            read from the execution state below.
            """
            loop_count = self.execution_state.increment_loop_count(state_name)
            self.execution_state.current_state = state_name

            # Check loop limits
            if flow_state.max_loops and loop_count > flow_state.max_loops:
                logger.warning(
                    "State '%s' exceeded max loops (%s)", state_name, flow_state.max_loops
                )
                reason = f"Max loops exceeded for state {state_name}"
                self.execution_state.stop_reason = reason
                return {**data, "_error": reason, "_force_end": True}

            # Check total transition limit
            self.execution_state.total_transitions += 1
            if self.execution_state.total_transitions > self.flow.max_total_loops:
                logger.warning(
                    "Flow exceeded max total transitions (%s)", self.flow.max_total_loops
                )
                reason = "Max total transitions exceeded"
                self.execution_state.stop_reason = reason
                return {**data, "_error": reason, "_force_end": True}

            context = self.execution_state.context

            # Call on_enter hook if defined
            if flow_state.on_enter:
                try:
                    await flow_state.on_enter(state_name, data, context)
                except Exception:
                    logger.exception("on_enter hook failed for state '%s'", state_name)

            # Merge prompt params with data
            prompt_params = {
                **data,
                **flow_state.prompt_params,
                **context,
                "state": state_name,
                "loop_count": loop_count,
            }

            # Render the state's prompt. A render failure is not an assistant
            # message reporting an error: it propagates, the engine records the
            # state as failed, and the run is reported as failed.
            result = await self.prompt_builder.render_user_prompt(
                flow_state.prompt_name, params=prompt_params
            )
            response = result.content if hasattr(result, "content") else str(result)

            # Add to history
            self.execution_state.add_to_history(state_name, response)

            # Call on_exit hook if defined
            if flow_state.on_exit:
                try:
                    await flow_state.on_exit(state_name, data, context)
                except Exception:
                    logger.exception("on_exit hook failed for state '%s'", state_name)

            # Update data with response
            return {
                **data,
                "response": response,
                "state": state_name,
                "loop_count": loop_count,
                "history": list(self.execution_state.history),
            }

        # Register function
        self._function_registry[function_name] = transform_func

        return function_name

    def _register_condition_function(
        self, condition_name: str, condition: TransitionCondition, state_name: str
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

        async def condition_func(data: Dict[str, Any], function_context: Any) -> bool:
            """Condition function for arc evaluation.

            A condition that *fails* is not a condition that answers "no": the
            engine surfaces a raised exception as a record error rather than
            de-selecting the arc, so that an outage in whatever the condition
            consults cannot be read as a data-quality outcome. Nothing is
            caught here for that reason.
            """
            # Check if forced to end
            if data.get("_force_end"):
                return False

            response = data.get("response", "")

            result = await condition.evaluate(response, {**self.execution_state.context, **data})
            logger.debug("Condition '%s' for state '%s': %s", condition_name, state_name, result)
            return result

        # Register function
        self._function_registry[function_name] = condition_func

        return function_name

    async def execute(self, initial_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Execute the conversation flow.

        Args:
            initial_data: Initial data for the flow

        Returns:
            Final data after flow execution

        Raises:
            OperationError: If the FSM reports the run as failed — a state
                transform raised, or an arc condition could not be evaluated.
        """
        # Initialize execution state
        self.execution_state = FlowExecutionState(
            current_state=self.flow.initial_state, context={**self.flow.initial_context}
        )
        if self.llm is not None:
            self.execution_state.context["_llm_provider"] = self.llm

        # Prepare initial data
        data = initial_data or {}
        data = {**data, **self.flow.initial_context}

        # Convert to FSM config. This populates the function registry, so read
        # it only after (see to_fsm_config).
        fsm_config = self.to_fsm_config()
        custom_functions = dict(self._function_registry)

        # Create and execute FSM. The adapter constructs it, so the adapter
        # closes it.
        async with AsyncSimpleFSM(
            fsm_config,
            data_mode=DataHandlingMode.COPY,
            custom_functions=custom_functions,
        ) as fsm:
            result = await fsm.process(data)

        if not result.get("success"):
            # A tripped loop guard knows why the flow stopped; the engine only
            # knows that the state it stopped in had no arc left to take.
            reason = self.execution_state.stop_reason or result.get("error")
            raise OperationError(
                f"Conversation flow '{self.flow.name}' failed: {reason}",
                context={"flow": self.flow.name, "state": self.execution_state.current_state},
            )

        final_data: Dict[str, Any] = result.get("data", data)
        return final_data

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
            "states_visited": list(self.execution_state.loop_counts.keys()),
            "stop_reason": self.execution_state.stop_reason,
        }
