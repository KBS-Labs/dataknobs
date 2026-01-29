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
from .observability import (
    TransitionRecord,
    WizardStateSnapshot,
    WizardTaskList,
    create_transition_record,
)
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
    navigation history, transition audit trail, and task completion.

    Attributes:
        current_stage: Name of the current stage
        data: Collected data from all stages
        history: List of visited stage names
        completed: Whether the wizard has finished
        clarification_attempts: Track consecutive clarification attempts
        transitions: Audit trail of all state transitions
        stage_entry_time: Timestamp when current stage was entered
        tasks: List of trackable tasks with completion status
    """

    current_stage: str
    data: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    completed: bool = False
    clarification_attempts: int = 0
    transitions: list[TransitionRecord] = field(default_factory=list)
    stage_entry_time: float = field(default_factory=time.time)
    tasks: WizardTaskList = field(default_factory=WizardTaskList)


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
        auto_advance_filled_stages: bool = False,
        context_template: str | None = None,
        allow_post_completion_edits: bool = False,
        section_to_stage_mapping: dict[str, str] | None = None,
        default_tool_reasoning: str = "single",
        default_max_iterations: int = 3,
        artifact_registry: Any | None = None,
        review_executor: Any | None = None,
        context_builder: Any | None = None,
    ):
        """Initialize WizardReasoning.

        Args:
            wizard_fsm: WizardFSM instance for state management
            extractor: Optional SchemaExtractor for data extraction
            strict_validation: Enforce schema validation (default: True)
            hooks: Optional WizardHooks for lifecycle callbacks
            auto_advance_filled_stages: Automatically skip stages where all
                required fields are already filled (default: False)
            context_template: Custom Jinja2 template for stage context.
                When set, replaces the default context formatting.
            allow_post_completion_edits: Allow re-opening wizard after completion
                when user requests changes (default: False)
            section_to_stage_mapping: Custom mapping of section names to stage
                names for amendment detection (optional)
            default_tool_reasoning: Default reasoning mode for stages with tools.
                "single" for single LLM call, "react" for ReAct-style loop.
            default_max_iterations: Default max iterations for ReAct-style reasoning.
            artifact_registry: Optional ArtifactRegistry for artifact management.
            review_executor: Optional ReviewExecutor for running reviews.
            context_builder: Optional ContextBuilder for building conversation context.
        """
        self._fsm = wizard_fsm
        self._extractor = extractor
        self._strict_validation = strict_validation
        self._hooks = hooks
        self._auto_advance_filled_stages = auto_advance_filled_stages
        self._context_template = context_template
        self._allow_amendments = allow_post_completion_edits
        self._section_to_stage_mapping = section_to_stage_mapping or {}
        self._default_tool_reasoning = default_tool_reasoning
        self._default_max_iterations = default_max_iterations
        self._artifact_registry = artifact_registry
        self._review_executor = review_executor
        self._context_builder = context_builder

    async def close(self) -> None:
        """Close the reasoning strategy and release resources.

        Closes the SchemaExtractor's LLM provider if present, releasing
        HTTP connections. Should be called when the reasoning strategy
        is no longer needed (typically via DynaBot.close()).
        """
        if self._extractor is not None and hasattr(self._extractor, "close"):
            await self._extractor.close()
            logger.debug("Closed WizardReasoning extractor")

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

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WizardReasoning":
        """Create WizardReasoning from configuration dict.

        Args:
            config: Configuration dict with:
                - wizard_config: Path to wizard YAML config
                - extraction_config: Optional extraction configuration
                - strict_validation: Whether to enforce validation
                - hooks: Optional hooks configuration dict
                - artifacts: Optional artifact configuration with definitions
                - review_protocols: Optional review protocol definitions

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
              artifacts:
                definitions:
                  assessment_questions:
                    type: content
                    reviews: [adversarial, skeptical]
              review_protocols:
                adversarial:
                  persona: adversarial
                  score_threshold: 0.7
                skeptical:
                  persona: skeptical
                  score_threshold: 0.8
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

        # Get settings from wizard FSM
        auto_advance = wizard_fsm.settings.get("auto_advance_filled_stages", False)
        context_template = wizard_fsm.settings.get("context_template")
        allow_amendments = wizard_fsm.settings.get("allow_post_completion_edits", False)
        section_mapping = wizard_fsm.settings.get("section_to_stage_mapping", {})
        tool_reasoning = wizard_fsm.settings.get("tool_reasoning", "single")
        max_iterations = wizard_fsm.settings.get("max_tool_iterations", 3)

        # Create artifact registry if artifact definitions configured
        artifact_registry = None
        artifacts_config = config.get("artifacts", {})
        if artifacts_config:
            try:
                from ..artifacts import ArtifactDefinition, ArtifactRegistry

                artifact_registry = ArtifactRegistry()
                for def_id, def_config in artifacts_config.get("definitions", {}).items():
                    definition = ArtifactDefinition(
                        id=def_id,
                        type=def_config.get("type", "content"),
                        name=def_config.get("name"),
                        schema=def_config.get("schema"),
                        reviews=def_config.get("reviews", []),
                        required_status_for_stage_completion=def_config.get(
                            "required_status_for_stage_completion"
                        ),
                        auto_submit_for_review=def_config.get(
                            "auto_submit_for_review", False
                        ),
                    )
                    artifact_registry.register_definition(definition)
                logger.info(
                    "Created artifact registry with %d definitions",
                    len(artifacts_config.get("definitions", {})),
                )
            except ImportError:
                logger.warning(
                    "Artifact modules not available, artifact tracking disabled"
                )

        # Create review executor if review protocols configured
        review_executor = None
        review_config = config.get("review_protocols", {})
        if review_config:
            try:
                from ..review import ReviewExecutor, ReviewProtocolDefinition

                protocols = {}
                for proto_id, proto_config in review_config.items():
                    protocols[proto_id] = ReviewProtocolDefinition.from_config(
                        proto_id, proto_config
                    )
                review_executor = ReviewExecutor(protocols=protocols)
                logger.info(
                    "Created review executor with %d protocols", len(protocols)
                )
            except ImportError:
                logger.warning("Review modules not available, reviews disabled")

        # Create context builder if registry or executor available
        context_builder = None
        if artifact_registry is not None:
            try:
                from ..context import ContextBuilder

                context_builder = ContextBuilder(artifact_registry=artifact_registry)
            except ImportError:
                logger.warning("Context modules not available")

        return cls(
            wizard_fsm=wizard_fsm,
            extractor=extractor,
            strict_validation=config.get("strict_validation", True),
            hooks=hooks,
            auto_advance_filled_stages=auto_advance,
            context_template=context_template,
            allow_post_completion_edits=allow_amendments,
            section_to_stage_mapping=section_mapping,
            default_tool_reasoning=tool_reasoning,
            default_max_iterations=max_iterations,
            artifact_registry=artifact_registry,
            review_executor=review_executor,
            context_builder=context_builder,
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
        2. Handles post-completion amendments (if enabled)
        3. Checks for navigation commands (back, skip, restart)
        4. Extracts structured data from user input
        5. Validates extracted data against stage schema
        6. Executes FSM transition on valid input
        7. Generates appropriate response for current/new stage

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

        # Handle post-completion amendments
        if wizard_state.completed and self._allow_amendments:
            amendment = await self._detect_amendment(user_message, wizard_state, llm)

            if amendment:
                target_stage = amendment["target_stage"]
                from_stage = wizard_state.current_stage
                duration_ms = (time.time() - wizard_state.stage_entry_time) * 1000

                # Re-open wizard to target stage
                wizard_state.completed = False
                wizard_state.current_stage = target_stage
                if target_stage not in wizard_state.history:
                    wizard_state.history.append(target_stage)

                # Restore FSM to target stage
                self._fsm.restore({
                    "current_stage": target_stage,
                    "data": wizard_state.data,
                })

                # Record the amendment transition
                transition = create_transition_record(
                    from_stage=from_stage,
                    to_stage=target_stage,
                    trigger="amendment",
                    duration_in_stage_ms=duration_ms,
                    data_snapshot=wizard_state.data.copy(),
                    user_input=user_message,
                )
                wizard_state.transitions.append(transition)
                wizard_state.stage_entry_time = time.time()

                logger.info("Amendment: re-opening wizard at %s", target_stage)

                # Generate response for the re-opened stage
                stage = self._fsm.current_metadata
                response = await self._generate_stage_response(
                    manager, llm, stage, wizard_state, tools
                )
                self._save_wizard_state(manager, wizard_state)
                return response

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

        # Update field-extraction tasks
        self._update_field_tasks(wizard_state, extraction.data)

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

        # Update stage-exit tasks before leaving
        self._update_stage_exit_tasks(wizard_state, wizard_state.current_stage)

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

        # Auto-advance through stages where all required fields are filled
        new_stage = self._fsm.current_metadata
        auto_advance_count = 0
        max_auto_advances = 10  # Safety limit to prevent infinite loops

        while (
            auto_advance_count < max_auto_advances
            and not wizard_state.completed
            and self._can_auto_advance(wizard_state, new_stage)
        ):
            auto_advance_count += 1
            old_stage_name = wizard_state.current_stage
            duration_ms = (time.time() - wizard_state.stage_entry_time) * 1000

            # Execute FSM transition for auto-advance
            auto_step_result = self._fsm.step(wizard_state.data)
            new_stage_name = self._fsm.current_stage

            if new_stage_name == old_stage_name:
                # No transition occurred, stop auto-advancing
                break

            # Record the auto-advance transition
            condition_expr = self._fsm.get_transition_condition(
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
            )
            wizard_state.transitions.append(transition)

            # Update wizard state
            wizard_state.current_stage = new_stage_name
            if new_stage_name not in wizard_state.history:
                wizard_state.history.append(new_stage_name)
            wizard_state.completed = auto_step_result.is_complete
            wizard_state.stage_entry_time = time.time()

            logger.info(
                "Auto-advanced from %s to %s (all required fields present)",
                old_stage_name,
                new_stage_name,
            )

            # Get new stage metadata for next iteration
            new_stage = self._fsm.current_metadata

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
            # Restore tasks from serialized data
            tasks_data = fsm_state.get("tasks", {})
            tasks = (
                WizardTaskList.from_dict(tasks_data)
                if tasks_data
                else WizardTaskList()
            )
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
                tasks=tasks,
            )
            # Restore FSM state
            self._fsm.restore(fsm_state)
            return state

        # Initialize new wizard state with tasks from config
        start_stage = self._fsm.current_stage
        initial_tasks = self._build_initial_tasks()
        return WizardState(
            current_stage=start_stage,
            history=[start_stage],
            stage_entry_time=time.time(),
            tasks=initial_tasks,
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
                "tasks": state.tasks.to_dict(),
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
        if lower in ("skip", "skip this", "use default", "use defaults"):
            if self._fsm.can_skip():
                state.data[f"_skipped_{state.current_stage}"] = True
                # Apply skip_default values if configured
                skip_default = self._fsm.current_metadata.get("skip_default")
                if skip_default and isinstance(skip_default, dict):
                    state.data.update(skip_default)
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

        Schema 'default' values are stripped before extraction to prevent
        the LLM from auto-filling them. This ensures extraction only captures
        what the user actually said.

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

        # Strip defaults to prevent extraction LLM from auto-filling them
        extraction_schema = self._strip_schema_defaults(schema)

        if self._extractor:
            # Use schema extractor
            extraction_model = stage.get("extraction_model")
            context = {"stage": stage.get("name"), "prompt": stage.get("prompt")}
            return await self._extractor.extract(
                text=message,
                schema=extraction_schema,
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

        # Check if stage should use ReAct-style reasoning
        if stage_tools and self._use_react_for_stage(stage):
            response = await self._react_stage_response(
                manager, enhanced_prompt, stage, state, stage_tools
            )
        else:
            # Single LLM call (default behavior)
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

    def _use_react_for_stage(self, stage: dict[str, Any]) -> bool:
        """Check if a stage should use ReAct-style reasoning.

        A stage uses ReAct if:
        - Stage has `reasoning: react` explicitly set, OR
        - No explicit reasoning set and default_tool_reasoning is "react"

        Args:
            stage: Stage metadata dict

        Returns:
            True if ReAct should be used for this stage
        """
        stage_reasoning = stage.get("reasoning")
        if stage_reasoning:
            return stage_reasoning.lower() == "react"
        return self._default_tool_reasoning.lower() == "react"

    def _get_max_iterations(self, stage: dict[str, Any]) -> int:
        """Get maximum ReAct iterations for a stage.

        Args:
            stage: Stage metadata dict

        Returns:
            Max iterations (from stage config or default)
        """
        return stage.get("max_iterations") or self._default_max_iterations

    async def _react_stage_response(
        self,
        manager: Any,
        enhanced_prompt: str,
        stage: dict[str, Any],
        state: WizardState,
        tools: list[Any],
    ) -> Any:
        """Generate response using ReAct loop for tool-using stage.

        This allows the LLM to make multiple tool calls within a single
        wizard turn, reasoning about results before responding.

        Args:
            manager: ConversationManager instance
            enhanced_prompt: Stage-aware system prompt
            stage: Stage metadata dict
            state: Current wizard state
            tools: Available tools for this stage

        Returns:
            Final LLM response after ReAct loop completes
        """
        from dataknobs_llm.tools import ToolExecutionContext

        max_iterations = self._get_max_iterations(stage)
        stage_name = stage.get("name", "unknown")

        logger.debug(
            "Starting ReAct loop for stage '%s' (max_iterations=%d)",
            stage_name,
            max_iterations,
        )

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
                conversation_context = self._context_builder.build(manager)
                extra_context["conversation_context"] = conversation_context
            except Exception as e:
                logger.warning("Failed to build conversation context: %s", e)
        if extra_context:
            tool_context = tool_context.with_extra(**extra_context)

        for iteration in range(max_iterations):
            # Make LLM call
            response = await manager.complete(
                system_prompt_override=enhanced_prompt,
                tools=tools,
            )

            # Check if response has tool calls
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                # No tool calls - this is the final response
                logger.debug(
                    "ReAct iteration %d/%d: No tool calls, returning response",
                    iteration + 1,
                    max_iterations,
                )
                return response

            logger.debug(
                "ReAct iteration %d/%d: Executing %d tool call(s): %s",
                iteration + 1,
                max_iterations,
                len(tool_calls),
                [tc.name for tc in tool_calls],
            )

            # Execute tool calls and add observations
            for tool_call in tool_calls:
                result = await self._execute_react_tool_call(
                    tool_call, tools, state, tool_context
                )

                # Add observation to conversation for next iteration
                observation = f"Tool result from {tool_call.name}: {result}"
                await manager.add_message(content=observation, role="system")

        # Max iterations reached - get final response without tools
        logger.warning(
            "ReAct max iterations (%d) reached for stage '%s'",
            max_iterations,
            stage_name,
        )
        return await manager.complete(
            system_prompt_override=enhanced_prompt,
            tools=None,  # Force text response
        )

    async def _execute_react_tool_call(
        self,
        tool_call: Any,
        tools: list[Any],
        state: WizardState,
        tool_context: Any,
    ) -> Any:
        """Execute a single tool call within a ReAct loop.

        Args:
            tool_call: Tool call object with name and parameters
            tools: Available tools
            state: Wizard state (for tools that need it)
            tool_context: ToolExecutionContext for context-aware tools

        Returns:
            Tool execution result or error dict
        """
        tool_name = tool_call.name
        tool_args = getattr(tool_call, "parameters", {}) or {}

        # Find the tool
        tool = self._find_tool(tool_name, tools)
        if tool is None:
            error_msg = f"Tool '{tool_name}' not found"
            logger.warning("ReAct: %s", error_msg)
            return {"error": error_msg}

        try:
            # Execute tool with context injection
            # Context-aware tools will extract _context and use it
            # Regular tools will ignore _context via **kwargs
            result = await tool.execute(**tool_args, _context=tool_context)

            logger.debug("ReAct: Tool '%s' executed successfully", tool_name)
            return result

        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.error("ReAct: Tool '%s' failed: %s", tool_name, e)
            return {"error": error_msg}

    def _find_tool(self, tool_name: str, tools: list[Any]) -> Any | None:
        """Find a tool by name.

        Args:
            tool_name: Name of the tool to find
            tools: List of available tools

        Returns:
            Tool instance or None if not found
        """
        for tool in tools:
            if getattr(tool, "name", None) == tool_name:
                return tool
        return None

    def _build_stage_context(
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

    def _render_custom_context(
        self, stage: dict[str, Any], state: WizardState
    ) -> str:
        """Render context using custom Jinja2 template.

        Template variables available:
        - stage_name: Current stage name
        - stage_prompt: Stage's goal/prompt text
        - help_text: Additional help text (may be empty string)
        - suggestions: List of quick-reply suggestions
        - collected_data: Data collected so far (no _ prefixed keys)
        - raw_data: All wizard data including internal keys
        - completed: Whether wizard is complete
        - history: List of visited stage names
        - can_skip: Whether current stage can be skipped
        - can_go_back: Whether back navigation is allowed

        Args:
            stage: Current stage metadata
            state: Current wizard state

        Returns:
            Rendered context string
        """
        from dataknobs_llm.prompts import render_template

        # Filter out internal keys for display
        collected_data = {
            k: v for k, v in state.data.items() if not k.startswith("_")
        }

        params = {
            "stage_name": stage.get("name", "unknown"),
            "stage_prompt": stage.get("prompt", ""),
            "help_text": stage.get("help_text") or "",
            "suggestions": stage.get("suggestions", []),
            "collected_data": collected_data,
            "raw_data": state.data,
            "completed": state.completed,
            "history": state.history,
            "can_skip": self._fsm.can_skip() if self._fsm else False,
            "can_go_back": (
                self._fsm.can_go_back() if self._fsm else True
            ) and len(state.history) > 1,
        }

        return render_template(self._context_template, params)

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

        if state.completed:
            lines.append("\nThe wizard is complete. Summarize what was collected.")
        else:
            lines.append(
                "\nGuide the user through this stage. Be conversational and helpful. "
                "Focus only on gathering information that has NOT already been collected."
            )

        return "\n".join(lines)

    def _filter_tools_for_stage(
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

    def _strip_schema_defaults(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Deep-copy schema with 'default' removed from all properties.

        Schema defaults serve a different purpose (documenting valid defaults for
        consumers) than extraction (parsing what the user actually said). The
        extraction prompt already instructs: "If information is missing, omit
        the field."

        Args:
            schema: JSON Schema dict with potential 'default' values

        Returns:
            Copy of schema with all 'default' keys removed from properties
        """
        import copy

        clean = copy.deepcopy(schema)

        # Handle properties at top level and nested
        self._strip_defaults_from_properties(clean)

        return clean

    def _strip_defaults_from_properties(self, schema_part: dict[str, Any]) -> None:
        """Recursively strip 'default' from properties in schema.

        Handles nested schemas (objects with nested properties, items in arrays).

        Args:
            schema_part: Schema or sub-schema dict to process in place
        """
        # Strip from direct properties
        for prop in schema_part.get("properties", {}).values():
            prop.pop("default", None)
            # Recurse into nested object properties
            if prop.get("type") == "object":
                self._strip_defaults_from_properties(prop)
            # Handle array items
            if prop.get("type") == "array" and isinstance(prop.get("items"), dict):
                self._strip_defaults_from_properties(prop["items"])

        # Handle allOf, anyOf, oneOf
        for key in ("allOf", "anyOf", "oneOf"):
            if key in schema_part:
                for sub_schema in schema_part[key]:
                    if isinstance(sub_schema, dict):
                        self._strip_defaults_from_properties(sub_schema)

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

    def _can_auto_advance(
        self, wizard_state: WizardState, stage: dict[str, Any]
    ) -> bool:
        """Check if a stage can be auto-advanced.

        A stage can be auto-advanced if:
        1. Global auto_advance_filled_stages is enabled, OR the stage has
           auto_advance: true in its config
        2. The stage has a schema with required fields (or all properties
           if no required list)
        3. All required fields have non-empty values in wizard_state.data
        4. The stage is not an end stage
        5. At least one transition condition is satisfied

        Args:
            wizard_state: Current wizard state
            stage: Stage configuration dict

        Returns:
            True if stage can be auto-advanced
        """
        # Check if auto-advance is enabled for this stage
        stage_auto_advance = stage.get("auto_advance", False)
        if not (stage_auto_advance or self._auto_advance_filled_stages):
            return False

        # Don't auto-advance end stages
        if stage.get("is_end", False):
            return False

        # Get schema to check required fields
        schema = stage.get("schema") or {}
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        # If no required fields specified, treat all properties as required
        if not required_fields:
            required_fields = list(properties.keys())

        # If no fields at all, can't auto-advance based on data
        if not required_fields:
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

        # Check if any transition condition is satisfied
        transitions = stage.get("transitions", [])
        for transition in transitions:
            condition = transition.get("condition")
            if condition:
                # Evaluate condition with current data
                if self._evaluate_condition(condition, wizard_state.data):
                    return True
            else:
                # Unconditional transition - can advance
                return True

        return False

    def _evaluate_condition(self, condition: str, data: dict[str, Any]) -> bool:
        """Safely evaluate a transition condition.

        Uses a restricted execution environment to evaluate condition
        expressions like "data.get('subject')" or "data.get('count', 0) > 5".

        Args:
            condition: Condition expression string
            data: Current wizard data

        Returns:
            True if condition is satisfied, False otherwise
        """
        try:
            # Wrap in return statement if not already
            code = condition.strip()
            if not code.startswith("return"):
                code = f"return {code}"

            # Create a function to evaluate the condition
            # Note: 'data' must be in globals for the function to access it
            global_vars: dict[str, Any] = {"data": data}
            local_vars: dict[str, Any] = {}
            exec_code = f"def _test():\n    {code}\n_result = _test()"
            exec(exec_code, global_vars, local_vars)  # nosec B102
            return bool(local_vars.get("_result", False))
        except Exception as e:
            logger.debug("Condition evaluation failed for '%s': %s", condition, e)
            return False

    # =========================================================================
    # Post-Completion Amendment Methods
    # =========================================================================

    async def _detect_amendment(
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
                target_stage = self._map_section_to_stage(target)
                if target_stage:
                    return {"target_stage": target_stage}
        except Exception as e:
            logger.debug("Amendment detection failed: %s", e)

        return None

    def _map_section_to_stage(self, section: str) -> str | None:
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
        if self._section_to_stage_mapping:
            if section_lower in self._section_to_stage_mapping:
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
        if mapped_stage:
            # Verify the stage exists in the FSM
            if mapped_stage in self._fsm._stage_metadata:
                return mapped_stage

        return None

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

    # =========================================================================
    # Task Tracking Methods
    # =========================================================================

    def _build_initial_tasks(self) -> WizardTaskList:
        """Build initial task list from wizard configuration.

        Extracts task definitions from stage metadata and creates
        a WizardTaskList with all tasks in pending status.

        Returns:
            WizardTaskList with initial tasks
        """
        from .observability import WizardTask

        tasks: list[WizardTask] = []
        global_tasks_added = False

        # Extract tasks from each stage's metadata
        for stage_name, stage_meta in self._fsm._stage_metadata.items():
            # Per-stage tasks
            stage_tasks = stage_meta.get("tasks", [])
            for task_def in stage_tasks:
                if task_def.get("id"):  # Only add if id is defined
                    tasks.append(WizardTask(
                        id=task_def.get("id"),
                        description=task_def.get("description", task_def.get("id", "")),
                        status="pending",
                        stage=stage_name,
                        required=task_def.get("required", True),
                        depends_on=task_def.get("depends_on", []),
                        completed_by=task_def.get("completed_by"),
                        field_name=task_def.get("field_name"),
                        tool_name=task_def.get("tool_name"),
                    ))

            # Global tasks (only need to add once)
            if not global_tasks_added:
                global_tasks = stage_meta.get("_global_tasks", [])
                for task_def in global_tasks:
                    if task_def.get("id"):  # Only add if id is defined
                        tasks.append(WizardTask(
                            id=task_def.get("id"),
                            description=task_def.get(
                                "description", task_def.get("id", "")
                            ),
                            status="pending",
                            stage=None,  # Global task
                            required=task_def.get("required", True),
                            depends_on=task_def.get("depends_on", []),
                            completed_by=task_def.get("completed_by"),
                            field_name=task_def.get("field_name"),
                            tool_name=task_def.get("tool_name"),
                        ))
                if global_tasks:
                    global_tasks_added = True

        return WizardTaskList(tasks=tasks)

    def _update_field_tasks(
        self, state: WizardState, extracted_data: dict[str, Any]
    ) -> None:
        """Mark field-extraction tasks as complete when fields are collected.

        Args:
            state: Current wizard state
            extracted_data: Data that was just extracted
        """
        for field_name, value in extracted_data.items():
            if value is not None and not field_name.startswith("_"):
                for task in state.tasks.tasks:
                    if (
                        task.completed_by == "field_extraction"
                        and task.field_name == field_name
                        and task.status == "pending"
                    ):
                        state.tasks.complete_task(task.id)
                        logger.debug("Task %s completed via field extraction", task.id)

    def _update_tool_tasks(
        self, state: WizardState, tool_name: str, success: bool
    ) -> None:
        """Mark tool-result tasks as complete when tools succeed.

        Args:
            state: Current wizard state
            tool_name: Name of the tool that was executed
            success: Whether the tool execution succeeded
        """
        if success:
            for task in state.tasks.tasks:
                if (
                    task.completed_by == "tool_result"
                    and task.tool_name == tool_name
                    and task.status == "pending"
                ):
                    state.tasks.complete_task(task.id)
                    logger.debug("Task %s completed via tool result", task.id)

    def _update_stage_exit_tasks(self, state: WizardState, stage: str) -> None:
        """Mark stage-exit tasks as complete when leaving a stage.

        Args:
            state: Current wizard state
            stage: The stage being exited
        """
        for task in state.tasks.tasks:
            if (
                task.completed_by == "stage_exit"
                and task.stage == stage
                and task.status == "pending"
            ):
                state.tasks.complete_task(task.id)
                logger.debug("Task %s completed via stage exit", task.id)

    # =========================================================================
    # Public State Query Methods
    # =========================================================================

    def get_state_snapshot(self, manager: Any) -> WizardStateSnapshot:
        """Get current wizard state as a read-only snapshot.

        This method provides access to wizard state without processing a message.
        Useful for UI components that need to display current state, progress,
        and available actions.

        Args:
            manager: ConversationManager instance

        Returns:
            WizardStateSnapshot with complete state information
        """
        wizard_state = self._get_wizard_state(manager)
        stage = self._fsm.current_metadata

        # Calculate stage index
        stage_names = list(self._fsm._stage_metadata.keys())
        try:
            stage_index = stage_names.index(wizard_state.current_stage)
        except ValueError:
            stage_index = 0

        # Get task info
        task_list = wizard_state.tasks
        available_tasks = task_list.get_available_tasks()

        return WizardStateSnapshot(
            current_stage=wizard_state.current_stage,
            data=dict(wizard_state.data),
            history=list(wizard_state.history),
            transitions=list(wizard_state.transitions),
            completed=wizard_state.completed,
            snapshot_timestamp=time.time(),
            clarification_attempts=wizard_state.clarification_attempts,
            # Task info
            tasks=[t.to_dict() for t in task_list.tasks],
            pending_tasks=len(task_list.get_pending_tasks()),
            completed_tasks=len(task_list.get_completed_tasks()),
            total_tasks=len(task_list),
            available_task_ids=[t.id for t in available_tasks],
            task_progress_percent=task_list.calculate_progress(),
            # Stage info
            stage_index=stage_index,
            total_stages=len(self._fsm._stage_metadata),
            can_skip=self._fsm.can_skip(),
            can_go_back=self._fsm.can_go_back() and len(wizard_state.history) > 1,
            suggestions=stage.get("suggestions", []),
        )

    @staticmethod
    def snapshot_from_metadata(
        metadata: dict[str, Any],
        stage_definitions: dict[str, Any] | None = None,
    ) -> WizardStateSnapshot | None:
        """Create snapshot from conversation manager metadata.

        This static method is useful when you have access to conversation
        metadata but not the WizardReasoning instance itself.

        Args:
            metadata: Conversation manager metadata dict
            stage_definitions: Optional stage definitions for index calculation

        Returns:
            WizardStateSnapshot if wizard metadata exists, None otherwise

        Example:
            ```python
            # From conversation metadata
            snapshot = WizardReasoning.snapshot_from_metadata(
                manager.metadata,
                stage_definitions=wizard_config.get("stages"),
            )
            if snapshot:
                print(f"Current stage: {snapshot.current_stage}")
                print(f"Progress: {snapshot.task_progress_percent}%")
            ```
        """
        wizard_meta = metadata.get("wizard")
        if not wizard_meta:
            return None

        fsm_state = wizard_meta.get("fsm_state", {})

        # Parse transitions
        transitions = [
            TransitionRecord.from_dict(t)
            for t in fsm_state.get("transitions", [])
        ]

        # Parse tasks
        tasks_data = fsm_state.get("tasks", {})
        task_list = (
            WizardTaskList.from_dict(tasks_data)
            if tasks_data
            else WizardTaskList()
        )
        available_tasks = task_list.get_available_tasks()

        # Calculate stage index if definitions provided
        stage_index = 0
        total_stages = 0
        current_stage = fsm_state.get("current_stage", "unknown")

        if stage_definitions:
            if isinstance(stage_definitions, dict):
                stage_names = list(stage_definitions.keys())
            elif isinstance(stage_definitions, list):
                stage_names = [s.get("name", "") for s in stage_definitions]
            else:
                stage_names = []

            total_stages = len(stage_names)
            try:
                stage_index = stage_names.index(current_stage)
            except ValueError:
                stage_index = 0

        return WizardStateSnapshot(
            current_stage=current_stage,
            data=fsm_state.get("data", {}),
            history=fsm_state.get("history", []),
            transitions=transitions,
            completed=fsm_state.get("completed", False),
            snapshot_timestamp=time.time(),
            clarification_attempts=fsm_state.get("clarification_attempts", 0),
            tasks=[t.to_dict() for t in task_list.tasks],
            pending_tasks=len(task_list.get_pending_tasks()),
            completed_tasks=len(task_list.get_completed_tasks()),
            total_tasks=len(task_list),
            available_task_ids=[t.id for t in available_tasks],
            task_progress_percent=task_list.calculate_progress(),
            stage_index=stage_index,
            total_stages=total_stages,
        )
