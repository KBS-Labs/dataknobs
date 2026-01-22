"""Wizard reasoning strategy for guided conversational flows.

This module implements FSM-backed reasoning for DynaBot, enabling
guided conversational wizard flows with validation, data collection,
and branching logic.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .base import ReasoningStrategy
from .observability import TransitionRecord, create_transition_record
from .wizard_hooks import WizardHooks

if TYPE_CHECKING:
    from .wizard_fsm import WizardFSM

logger = logging.getLogger(__name__)


@dataclass
class WizardStageContext:
    """Context for current wizard stage.

    Contains all information needed to interact with the user
    during a specific wizard stage.

    Attributes:
        name: Stage identifier
        prompt: User-facing prompt for this stage
        schema: Optional JSON Schema for data validation
        suggestions: Quick-reply suggestions for the user
        help_text: Additional help text for users who are stuck
        can_skip: Whether this stage can be skipped
        can_go_back: Whether back navigation is allowed
        tools: List of tool names available in this stage
    """

    name: str
    prompt: str
    schema: dict[str, Any] | None = None
    suggestions: list[str] = field(default_factory=list)
    help_text: str | None = None
    can_skip: bool = False
    can_go_back: bool = True
    tools: list[str] = field(default_factory=list)


@dataclass
class WizardState:
    """Persistent wizard state across conversation turns.

    Tracks the wizard's current position, collected data,
    navigation history, and transition audit trail.

    Attributes:
        current_stage: Name of the current stage
        data: Collected data from all stages
        history: List of visited stage names
        completed: Whether the wizard has finished
        clarification_attempts: Track consecutive clarification attempts
        transitions: Audit trail of all state transitions
        stage_entry_time: Timestamp when current stage was entered
    """

    current_stage: str
    data: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    completed: bool = False
    clarification_attempts: int = 0
    transitions: list[TransitionRecord] = field(default_factory=list)
    stage_entry_time: float = field(default_factory=time.time)


class WizardReasoning(ReasoningStrategy):
    """FSM-backed reasoning strategy for guided conversational flows.

    Unlike SimpleReasoning (single LLM call) or ReActReasoning (iterative
    tool use), WizardReasoning guides conversations through defined stages
    with validation, data collection, and branching logic.

    The wizard configuration (wizard.yaml) defines:
    - Stages (states in FSM terms)
    - Transitions between stages
    - Validation schemas per stage
    - Stage-specific prompts and tools

    Configuration example::

        reasoning:
          strategy: wizard
          config:
            wizard_config: path/to/wizard.yaml
            extraction_config:
              provider: ollama
              model: qwen3-coder
            strict_validation: true

    Attributes:
        _fsm: WizardFSM instance managing state transitions
        _extractor: Optional SchemaExtractor for data extraction
        _strict_validation: Whether to enforce schema validation
        _hooks: Optional WizardHooks for lifecycle events
    """

    def __init__(
        self,
        wizard_fsm: "WizardFSM",
        extractor: Any | None = None,
        strict_validation: bool = True,
        hooks: WizardHooks | None = None,
    ):
        """Initialize WizardReasoning.

        Args:
            wizard_fsm: WizardFSM instance for state management
            extractor: Optional SchemaExtractor for data extraction
            strict_validation: Enforce schema validation (default: True)
            hooks: Optional WizardHooks for lifecycle callbacks
        """
        self._fsm = wizard_fsm
        self._extractor = extractor
        self._strict_validation = strict_validation
        self._hooks = hooks

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WizardReasoning":
        """Create WizardReasoning from configuration dict.

        Args:
            config: Configuration dict with:
                - wizard_config: Path to wizard YAML config
                - extraction_config: Optional extraction configuration
                - strict_validation: Whether to enforce validation
                - hooks: Optional hooks configuration dict

        Returns:
            Configured WizardReasoning instance

        Raises:
            ValueError: If wizard_config is not provided

        Example:
            ```yaml
            reasoning:
              strategy: wizard
              wizard_config: wizards/onboarding.yaml
              strict_validation: true
              hooks:
                on_enter:
                  - function: "myapp.hooks:log_entry"
                on_complete:
                  - "myapp.hooks:save_results"
            ```
        """
        from .wizard_loader import WizardConfigLoader

        wizard_config_path = config.get("wizard_config")
        if not wizard_config_path:
            raise ValueError("wizard_config path is required")

        # Load wizard FSM
        loader = WizardConfigLoader()
        wizard_fsm = loader.load(
            wizard_config_path, config.get("custom_functions", {})
        )

        # Create extractor if extraction_config specified
        extractor = None
        extraction_config = config.get("extraction_config")
        if extraction_config:
            try:
                from dataknobs_llm.extraction import SchemaExtractor

                extractor = SchemaExtractor.from_env_config(extraction_config)
            except ImportError:
                logger.warning(
                    "dataknobs_llm.extraction not available, "
                    "extraction will be disabled"
                )

        # Create hooks if hooks config specified
        hooks = None
        hooks_config = config.get("hooks")
        if hooks_config:
            hooks = WizardHooks.from_config(hooks_config)

        return cls(
            wizard_fsm=wizard_fsm,
            extractor=extractor,
            strict_validation=config.get("strict_validation", True),
            hooks=hooks,
        )

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate response guided by wizard FSM.

        This method:
        1. Retrieves or initializes wizard state
        2. Checks for navigation commands (back, skip, restart)
        3. Extracts structured data from user input
        4. Validates extracted data against stage schema
        5. Executes FSM transition on valid input
        6. Generates appropriate response for current/new stage

        Args:
            manager: ConversationManager instance
            llm: LLM provider instance
            tools: Optional list of available tools
            **kwargs: Additional generation parameters

        Returns:
            LLM response object with wizard metadata
        """
        # Get or restore wizard state
        wizard_state = self._get_wizard_state(manager)

        # Get user message
        user_message = self._get_last_user_message(manager)

        # Handle navigation commands
        nav_result = await self._handle_navigation(
            user_message, wizard_state, manager, llm
        )
        if nav_result:
            self._save_wizard_state(manager, wizard_state)
            return nav_result

        # Get current stage context
        stage = self._fsm.current_metadata

        # Extract structured data from user input
        extraction = await self._extract_data(user_message, stage, llm)

        # Handle low confidence extraction
        if not extraction.is_confident:
            wizard_state.clarification_attempts += 1
            # Save state (including incremented clarification_attempts)
            self._save_wizard_state(manager, wizard_state)

            # After 3 failed attempts, offer restart option
            if wizard_state.clarification_attempts >= 3:
                response = await self._generate_restart_offer(
                    manager, llm, stage, extraction.errors
                )
            else:
                response = await self._generate_clarification_response(
                    manager, llm, stage, extraction.errors
                )

            # Add wizard metadata to clarification response
            self._add_wizard_metadata(response, wizard_state, stage)
            return response

        # Reset clarification attempts on successful extraction
        wizard_state.clarification_attempts = 0

        # Merge extracted data with wizard state
        wizard_state.data.update(extraction.data)

        # Validate against stage schema
        if stage.get("schema") and self._strict_validation:
            validation_errors = self._validate_data(
                wizard_state.data, stage["schema"]
            )
            if validation_errors:
                # Save state before returning validation error
                self._save_wizard_state(manager, wizard_state)
                response = await self._generate_validation_response(
                    manager, llm, stage, validation_errors
                )
                self._add_wizard_metadata(response, wizard_state, stage)
                return response

        # Trigger stage exit hook if configured
        if self._hooks:
            await self._hooks.trigger_exit(
                wizard_state.current_stage, wizard_state.data
            )

        # Capture state before transition
        from_stage = wizard_state.current_stage
        duration_ms = (time.time() - wizard_state.stage_entry_time) * 1000

        # Execute FSM transition
        step_result = self._fsm.step(wizard_state.data)
        to_stage = self._fsm.current_stage

        # Record the transition if stage changed
        if to_stage != from_stage:
            # Look up the condition that was evaluated for this transition
            condition_expr = self._fsm.get_transition_condition(from_stage, to_stage)

            transition = create_transition_record(
                from_stage=from_stage,
                to_stage=to_stage,
                trigger="user_input",
                duration_in_stage_ms=duration_ms,
                data_snapshot=wizard_state.data.copy(),
                user_input=user_message,
                condition_evaluated=condition_expr,
                condition_result=True if condition_expr else None,
            )
            wizard_state.transitions.append(transition)
            wizard_state.stage_entry_time = time.time()

        wizard_state.current_stage = to_stage
        if wizard_state.current_stage not in wizard_state.history:
            wizard_state.history.append(wizard_state.current_stage)
        wizard_state.completed = step_result.is_complete

        # Trigger stage entry hook if configured
        if self._hooks:
            await self._hooks.trigger_enter(
                wizard_state.current_stage, wizard_state.data
            )

        # Trigger completion hook if wizard is complete
        if wizard_state.completed and self._hooks:
            await self._hooks.trigger_complete(wizard_state.data)

        # Generate stage-aware response
        new_stage = self._fsm.current_metadata
        response = await self._generate_stage_response(
            manager, llm, new_stage, wizard_state, tools
        )

        # Save wizard state
        self._save_wizard_state(manager, wizard_state)

        return response

    def _get_wizard_state(self, manager: Any) -> WizardState:
        """Get or create wizard state from conversation manager.

        Args:
            manager: ConversationManager instance

        Returns:
            WizardState instance
        """
        wizard_data = manager.metadata.get("wizard", {})
        if wizard_data.get("fsm_state"):
            fsm_state = wizard_data["fsm_state"]
            # Restore transitions from serialized data
            transitions = [
                TransitionRecord.from_dict(t)
                for t in fsm_state.get("transitions", [])
            ]
            state = WizardState(
                current_stage=fsm_state.get(
                    "current_stage", self._fsm.current_stage
                ),
                data=fsm_state.get("data", {}),
                history=fsm_state.get("history", []),
                completed=fsm_state.get("completed", False),
                clarification_attempts=fsm_state.get("clarification_attempts", 0),
                transitions=transitions,
                stage_entry_time=fsm_state.get("stage_entry_time", time.time()),
            )
            # Restore FSM state
            self._fsm.restore(fsm_state)
            return state

        # Initialize new wizard state
        start_stage = self._fsm.current_stage
        return WizardState(
            current_stage=start_stage,
            history=[start_stage],
            stage_entry_time=time.time(),
        )

    def _save_wizard_state(self, manager: Any, state: WizardState) -> None:
        """Save wizard state to conversation manager.

        Args:
            manager: ConversationManager instance
            state: WizardState to save
        """
        manager.metadata["wizard"] = {
            "fsm_state": {
                "current_stage": state.current_stage,
                "history": state.history,
                "data": state.data,
                "completed": state.completed,
                "clarification_attempts": state.clarification_attempts,
                "transitions": [t.to_dict() for t in state.transitions],
                "stage_entry_time": state.stage_entry_time,
            },
            "progress": self._calculate_progress(state),
        }

    def _get_last_user_message(self, manager: Any) -> str:
        """Extract the last user message from conversation.

        Args:
            manager: ConversationManager instance

        Returns:
            Last user message text
        """
        messages = manager.get_messages()
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Handle structured content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")
        return ""

    async def _handle_navigation(
        self,
        message: str,
        state: WizardState,
        manager: Any,
        llm: Any,
    ) -> Any | None:
        """Handle navigation commands (back, skip, restart).

        Args:
            message: User message text
            state: Current wizard state
            manager: ConversationManager instance
            llm: LLM provider

        Returns:
            Response if navigation handled, None otherwise
        """
        lower = message.lower().strip()

        # Back navigation
        if lower in ("back", "go back", "previous"):
            if self._fsm.can_go_back() and len(state.history) > 1:
                from_stage = state.current_stage
                duration_ms = (time.time() - state.stage_entry_time) * 1000

                state.history.pop()
                state.current_stage = state.history[-1]
                self._fsm.restore(
                    {"current_stage": state.current_stage, "data": state.data}
                )
                state.clarification_attempts = 0

                # Record the back navigation transition
                transition = create_transition_record(
                    from_stage=from_stage,
                    to_stage=state.current_stage,
                    trigger="navigation_back",
                    duration_in_stage_ms=duration_ms,
                    user_input=message,
                )
                state.transitions.append(transition)
                state.stage_entry_time = time.time()

                stage = self._fsm.current_metadata
                return await self._generate_stage_response(
                    manager, llm, stage, state, None
                )
            # Can't go back - inform user
            return await manager.complete(
                system_prompt_override=(
                    manager.system_prompt
                    + "\n\nThe user asked to go back but we're at the beginning. "
                    "Kindly explain we can't go back further and continue with "
                    "the current step."
                ),
            )

        # Skip
        if lower in ("skip", "skip this"):
            if self._fsm.can_skip():
                state.data[f"_skipped_{state.current_stage}"] = True
                state.clarification_attempts = 0
                return None  # Continue to normal flow, triggering transition
            return await manager.complete(
                system_prompt_override=(
                    manager.system_prompt
                    + "\n\nThe user asked to skip this step but it's required. "
                    "Kindly explain the step cannot be skipped and ask for the "
                    "information needed."
                ),
            )

        # Restart
        if lower in ("restart", "start over"):
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
                trigger="restart",
                duration_in_stage_ms=duration_ms,
                data_snapshot=state.data.copy(),
                user_input=message,
            )
            # Preserve transition history but clear other state
            previous_transitions = state.transitions + [transition]

            state.current_stage = to_stage
            state.data = {}
            state.history = [state.current_stage]
            state.completed = False
            state.clarification_attempts = 0
            state.transitions = previous_transitions
            state.stage_entry_time = time.time()

            stage = self._fsm.current_metadata
            return await self._generate_stage_response(
                manager, llm, stage, state, None
            )

        return None  # Not a navigation command

    async def _extract_data(
        self, message: str, stage: dict[str, Any], llm: Any
    ) -> Any:
        """Extract structured data from user message.

        Args:
            message: User message text
            stage: Current stage metadata
            llm: LLM provider (fallback if no extractor)

        Returns:
            ExtractionResult with data and confidence
        """
        # Create a simple result class for when extractor is not available
        @dataclass
        class SimpleExtractionResult:
            data: dict[str, Any] = field(default_factory=dict)
            confidence: float = 0.0
            errors: list[str] = field(default_factory=list)

            @property
            def is_confident(self) -> bool:
                return self.confidence >= 0.8 and not self.errors

        schema = stage.get("schema")
        if not schema:
            # No schema defined - pass through any data
            return SimpleExtractionResult(
                data={"_raw_input": message}, confidence=1.0
            )

        if self._extractor:
            # Use schema extractor
            extraction_model = stage.get("extraction_model")
            context = {"stage": stage.get("name"), "prompt": stage.get("prompt")}
            return await self._extractor.extract(
                text=message,
                schema=schema,
                context=context,
                model=extraction_model,
            )

        # Fallback: simple heuristic extraction
        # This is very basic - the extractor should be used for real scenarios
        return SimpleExtractionResult(
            data={"_raw_input": message}, confidence=0.5
        )

    def _validate_data(
        self, data: dict[str, Any], schema: dict[str, Any]
    ) -> list[str]:
        """Validate extracted data against JSON schema.

        Args:
            data: Extracted data to validate
            schema: JSON Schema to validate against

        Returns:
            List of validation error messages
        """
        errors = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields
        for field_name in required:
            if field_name not in data or data[field_name] is None:
                errors.append(f"Missing required field: {field_name}")

        # Check enum constraints
        for name, value in data.items():
            if name.startswith("_"):
                continue  # Skip internal fields
            if name in properties:
                prop = properties[name]
                if "enum" in prop and value not in prop["enum"]:
                    errors.append(
                        f"Invalid value for {name}: must be one of {prop['enum']}"
                    )

        return errors

    async def _generate_stage_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        state: WizardState,
        tools: list[Any] | None,
    ) -> Any:
        """Generate response appropriate for current stage.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            state: Current wizard state
            tools: Available tools

        Returns:
            LLM response with wizard metadata
        """
        # Build stage-aware system prompt
        stage_context = self._build_stage_context(stage, state)
        enhanced_prompt = f"{manager.system_prompt}\n\n{stage_context}"

        # Filter tools to stage-specific ones
        stage_tools = self._filter_tools_for_stage(stage, tools)

        # Generate response
        response = await manager.complete(
            system_prompt_override=enhanced_prompt,
            tools=stage_tools,
        )

        # Add wizard metadata to response
        self._add_wizard_metadata(response, state, stage)

        return response

    def _add_wizard_metadata(
        self,
        response: Any,
        state: WizardState,
        stage: dict[str, Any],
    ) -> None:
        """Add wizard metadata to response object.

        Args:
            response: LLM response object to modify
            state: Current wizard state
            stage: Current stage metadata
        """
        if not hasattr(response, "metadata") or response.metadata is None:
            response.metadata = {}
        response.metadata["wizard"] = {
            "stage": state.current_stage,
            "progress": self._calculate_progress(state),
            "completed": state.completed,
            "can_skip": self._fsm.can_skip(),
            "can_go_back": self._fsm.can_go_back() and len(state.history) > 1,
            "stage_prompt": stage.get("prompt", ""),
            "suggestions": stage.get("suggestions", []),
        }

    def _build_stage_context(
        self, stage: dict[str, Any], state: WizardState
    ) -> str:
        """Build context prompt for current stage.

        Args:
            stage: Current stage metadata
            state: Current wizard state

        Returns:
            Context string to append to system prompt
        """
        lines = ["## Current Wizard Stage"]
        lines.append(f"Stage: {stage.get('name', 'unknown')}")

        if stage.get("prompt"):
            lines.append(f"Goal: {stage['prompt']}")

        if stage.get("help_text"):
            lines.append(f"Additional context: {stage['help_text']}")

        if stage.get("suggestions"):
            lines.append(f"Suggested responses: {', '.join(stage['suggestions'])}")

        if state.completed:
            lines.append("\nThe wizard is complete. Summarize what was collected.")
        else:
            lines.append(
                "\nGuide the user through this stage. Be conversational and helpful."
            )

        # Add collected data context
        if state.data:
            filtered_data = {
                k: v for k, v in state.data.items() if not k.startswith("_")
            }
            if filtered_data:
                lines.append(f"\nCollected so far: {filtered_data}")

        return "\n".join(lines)

    def _filter_tools_for_stage(
        self, stage: dict[str, Any], tools: list[Any] | None
    ) -> list[Any] | None:
        """Filter tools to those available in current stage.

        Args:
            stage: Current stage metadata
            tools: All available tools

        Returns:
            Filtered list of tools or None
        """
        if not tools:
            return None

        stage_tool_names = stage.get("tools", [])
        if not stage_tool_names:
            return tools  # No filter, allow all tools

        # Filter to stage-specific tools
        filtered = []
        for tool in tools:
            tool_name = getattr(tool, "name", None) or getattr(
                tool, "__name__", None
            )
            if tool_name in stage_tool_names:
                filtered.append(tool)

        return filtered if filtered else None

    def _calculate_progress(self, state: WizardState) -> float:
        """Calculate wizard completion progress (0.0 to 1.0).

        Args:
            state: Current wizard state

        Returns:
            Progress as float between 0 and 1
        """
        total_stages = len(self._fsm._stage_metadata)
        if total_stages == 0:
            return 0.0

        visited = len(set(state.history))
        # Subtract 1 for end state in progress calculation
        return min(1.0, visited / max(1, total_stages - 1))

    async def _generate_validation_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        errors: list[str],
    ) -> Any:
        """Generate response asking for corrections.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            errors: Validation error messages

        Returns:
            LLM response requesting corrections
        """
        error_list = "\n".join(f"- {e}" for e in errors)
        error_context = f"""
## Validation Required

The user's input for this stage needs clarification:

**Issues**:
{error_list}

**What's Needed**: {stage.get('prompt', 'Please provide the required information.')}

Please kindly ask the user to provide the missing or corrected information.
Be specific about what's needed but remain friendly and helpful.
"""
        return await manager.complete(
            system_prompt_override=manager.system_prompt + error_context,
        )

    async def _generate_clarification_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        issues: list[str],
    ) -> Any:
        """Generate response asking for clarification.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            issues: Extraction issues

        Returns:
            LLM response requesting clarification
        """
        issue_list = (
            "\n".join(f"- {e}" for e in issues)
            if issues
            else "- Response was ambiguous"
        )
        suggestions = stage.get("suggestions", [])
        suggestions_text = (
            f"\n**Suggestions**: {', '.join(suggestions)}" if suggestions else ""
        )

        clarification_context = f"""
## Clarification Needed

I wasn't able to clearly understand the user's response for this stage.

**Potential Issues**:
{issue_list}

**What I'm Looking For**: {stage.get('prompt', 'Please provide more specific information.')}{suggestions_text}

Please ask a clarifying question to help gather the needed information.
Be conversational and helpful - don't make the user feel like they did something wrong.
"""
        return await manager.complete(
            system_prompt_override=manager.system_prompt + clarification_context,
        )

    async def _generate_restart_offer(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        issues: list[str],
    ) -> Any:
        """Generate response offering to restart after multiple failures.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            issues: Extraction issues

        Returns:
            LLM response offering restart option
        """
        restart_context = f"""
## Multiple Clarification Attempts

We've had difficulty understanding the responses for this stage.

**Current Stage**: {stage.get('name', 'unknown')}
**Goal**: {stage.get('prompt', 'Provide information')}

Please offer the user two options:
1. Try one more time with clearer instructions
2. Start the wizard over from the beginning (type "restart")

Be empathetic and helpful - acknowledge that the questions might not be clear.
"""
        return await manager.complete(
            system_prompt_override=manager.system_prompt + restart_context,
        )
