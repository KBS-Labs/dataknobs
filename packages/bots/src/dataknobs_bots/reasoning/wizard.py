"""Wizard reasoning strategy for guided conversational flows.

This module implements FSM-backed reasoning for DynaBot, enabling
guided conversational wizard flows with validation, data collection,
and branching logic.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dataknobs_common.serialization import sanitize_for_json

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

# Framework-level keys that are always transient (non-persistent).
# These are either non-serializable runtime objects or ephemeral per-step
# data that should never reach persistent storage.
DEFAULT_EPHEMERAL_KEYS: frozenset[str] = frozenset({
    "_corpus",              # Live ArtifactCorpus (non-serializable)
    "_message",             # Per-step raw user message (already popped)
    "_intent",              # Per-step intent detection result
    "_transform_error",     # Per-step error (may be Exception)
})


def _is_json_safe(value: Any) -> bool:
    """Check whether a value is JSON-serializable using isinstance checks.

    This is a lightweight alternative to ``json.dumps`` — it recurses into
    dicts, lists, and dataclasses but does not attempt actual serialization.
    Dataclasses are accepted because ``sanitize_for_json`` converts them to
    dicts via ``dataclasses.asdict``.

    Args:
        value: Value to check.

    Returns:
        True if the value is composed entirely of JSON-safe types
        (including dataclasses with JSON-safe fields).
    """
    import dataclasses

    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_json_safe(v) for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return all(
            _is_json_safe(getattr(value, f.name))
            for f in dataclasses.fields(value)
        )
    return False


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
class SubflowContext:
    """Context for a pushed subflow.

    Stores the parent wizard state so it can be restored when the
    subflow completes.

    Attributes:
        parent_stage: Stage name in the parent flow before push
        parent_data: Copy of wizard data at push time
        parent_history: Copy of stage history at push time
        return_stage: Stage to transition to when subflow completes
        result_mapping: Mapping of subflow field names to parent field names
        subflow_network: Name of the subflow network being executed
        push_timestamp: When the subflow was pushed
    """

    parent_stage: str
    parent_data: dict[str, Any]
    parent_history: list[str]
    return_stage: str
    result_mapping: dict[str, str]
    subflow_network: str
    push_timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the subflow context
        """
        return {
            "parent_stage": self.parent_stage,
            "parent_data": sanitize_for_json(self.parent_data),
            "parent_history": self.parent_history,
            "return_stage": self.return_stage,
            "result_mapping": self.result_mapping,
            "subflow_network": self.subflow_network,
            "push_timestamp": self.push_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubflowContext:
        """Create from dictionary.

        Args:
            data: Dictionary containing subflow context fields

        Returns:
            SubflowContext instance
        """
        return cls(
            parent_stage=data["parent_stage"],
            parent_data=data["parent_data"],
            parent_history=data["parent_history"],
            return_stage=data["return_stage"],
            result_mapping=data.get("result_mapping", {}),
            subflow_network=data["subflow_network"],
            push_timestamp=data.get("push_timestamp", time.time()),
        )


@dataclass
class WizardState:
    """Persistent wizard state across conversation turns.

    Tracks the wizard's current position, collected data,
    navigation history, transition audit trail, task completion,
    and subflow stack for nested wizard flows.

    Attributes:
        current_stage: Name of the current stage
        data: Persistent collected data from all stages
        transient: Ephemeral per-step data that is NOT persisted (e.g.
            live objects like ArtifactCorpus, per-step error messages).
            Merged into the working dict before FSM execution so
            templates and conditions see all keys, but stripped before
            saving to storage.
        history: List of visited stage names
        completed: Whether the wizard has finished
        clarification_attempts: Track consecutive clarification attempts
        transitions: Audit trail of all state transitions
        stage_entry_time: Timestamp when current stage was entered
        tasks: List of trackable tasks with completion status
        subflow_stack: Stack of subflow contexts for nested flows
    """

    current_stage: str
    data: dict[str, Any] = field(default_factory=dict)
    transient: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    completed: bool = False
    clarification_attempts: int = 0
    transitions: list[TransitionRecord] = field(default_factory=list)
    stage_entry_time: float = field(default_factory=time.time)
    tasks: WizardTaskList = field(default_factory=WizardTaskList)
    subflow_stack: list[SubflowContext] = field(default_factory=list)

    @property
    def is_in_subflow(self) -> bool:
        """Check if currently executing within a subflow.

        Returns:
            True if subflow_stack is not empty
        """
        return len(self.subflow_stack) > 0

    @property
    def subflow_depth(self) -> int:
        """Get current subflow nesting depth.

        Returns:
            Number of subflows on the stack (0 = main flow)
        """
        return len(self.subflow_stack)

    @property
    def current_subflow(self) -> SubflowContext | None:
        """Get the current subflow context.

        Returns:
            Current SubflowContext or None if in main flow
        """
        return self.subflow_stack[-1] if self.subflow_stack else None

    # -- Render count helpers -------------------------------------------------
    # Centralise stage render-count management so every code path (greet,
    # generate, restart, back) uses the same logic — eliminating the class
    # of bug where a caller forgets to increment after rendering.

    def increment_render_count(self, stage_name: str) -> int:
        """Increment and return the render count for a stage."""
        counts: dict[str, int] = self.data.setdefault(
            "_stage_render_counts", {}
        )
        counts[stage_name] = counts.get(stage_name, 0) + 1
        return counts[stage_name]

    def get_render_count(self, stage_name: str) -> int:
        """Get the current render count for a stage."""
        return self.data.get("_stage_render_counts", {}).get(stage_name, 0)

    def save_stage_snapshot(self, stage_name: str, schema_props: set[str]) -> None:
        """Save current schema property values for confirm_on_new_data comparison."""
        snapshots: dict[str, dict[str, Any]] = self.data.setdefault(
            "_stage_rendered_snapshot", {}
        )
        snapshots[stage_name] = {
            k: self.data[k]
            for k in schema_props
            if k in self.data and self.data[k] is not None
        }

    def get_stage_snapshot(self, stage_name: str) -> dict[str, Any]:
        """Get saved schema snapshot for a stage."""
        return self.data.get("_stage_rendered_snapshot", {}).get(stage_name, {})


# ---------------------------------------------------------------------------
# Navigation keyword configuration
# ---------------------------------------------------------------------------

DEFAULT_BACK_KEYWORDS: tuple[str, ...] = ("back", "go back", "previous")
DEFAULT_SKIP_KEYWORDS: tuple[str, ...] = ("skip", "skip this", "use default", "use defaults")
DEFAULT_RESTART_KEYWORDS: tuple[str, ...] = ("restart", "start over")


@dataclass(frozen=True)
class NavigationCommandConfig:
    """Configuration for a single navigation command.

    Attributes:
        keywords: Tuple of keyword strings that trigger this command.
            All keywords are stored in lowercase.
        enabled: Whether this command is active. When ``False``, the
            command is disabled regardless of keywords.
    """

    keywords: tuple[str, ...]
    enabled: bool = True


@dataclass(frozen=True)
class NavigationConfig:
    """Configuration for wizard navigation commands (back, skip, restart).

    Wizard authors can customize navigation keywords at the wizard level
    (via ``settings.navigation``) and override them per-stage. When no
    configuration is provided, the hardcoded defaults are used, preserving
    backward compatibility.

    Attributes:
        back: Configuration for the back/previous navigation command.
        skip: Configuration for the skip/use-default command.
        restart: Configuration for the restart/start-over command.
    """

    back: NavigationCommandConfig
    skip: NavigationCommandConfig
    restart: NavigationCommandConfig

    @classmethod
    def defaults(cls) -> NavigationConfig:
        """Create a ``NavigationConfig`` with the default keywords."""
        return cls(
            back=NavigationCommandConfig(keywords=DEFAULT_BACK_KEYWORDS),
            skip=NavigationCommandConfig(keywords=DEFAULT_SKIP_KEYWORDS),
            restart=NavigationCommandConfig(keywords=DEFAULT_RESTART_KEYWORDS),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NavigationConfig:
        """Build a ``NavigationConfig`` from a settings dict.

        Missing commands fall back to defaults. Keywords are normalised
        to lowercase.

        Args:
            data: Dict with optional ``back``, ``skip``, ``restart`` keys.
                Each value is a dict with optional ``keywords`` (list of
                strings) and ``enabled`` (bool) keys.

        Returns:
            A new ``NavigationConfig`` instance.
        """
        if not data:
            return cls.defaults()

        def _build_command(
            raw: dict[str, Any] | None,
            default_keywords: tuple[str, ...],
        ) -> NavigationCommandConfig:
            if raw is None:
                return NavigationCommandConfig(keywords=default_keywords)
            keywords = raw.get("keywords")
            if keywords is not None:
                keywords = tuple(k.lower() for k in keywords)
            else:
                keywords = default_keywords
            enabled = raw.get("enabled", True)
            return NavigationCommandConfig(keywords=keywords, enabled=enabled)

        return cls(
            back=_build_command(data.get("back"), DEFAULT_BACK_KEYWORDS),
            skip=_build_command(data.get("skip"), DEFAULT_SKIP_KEYWORDS),
            restart=_build_command(data.get("restart"), DEFAULT_RESTART_KEYWORDS),
        )


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
        wizard_fsm: WizardFSM,
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
        extraction_scope: str = "wizard_session",
        conflict_strategy: str = "latest_wins",
        log_conflicts: bool = True,
        initial_data: dict[str, Any] | None = None,
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
            extraction_scope: Scope for data extraction. "wizard_session" extracts
                from all user messages in the wizard session (default), while
                "current_message" only extracts from the current message.
            conflict_strategy: Strategy for handling conflicting values when
                the same field is extracted from multiple messages. "latest_wins"
                (default) uses the most recent value.
            log_conflicts: Whether to log when field values conflict (default: True).
            initial_data: Optional dict of data to inject into the wizard state
                when a new conversation starts. Useful for passing configuration
                values (e.g., quiz_bank_ids) from the bot config into the wizard
                data dict where transforms can access them.
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
        self._extraction_scope = extraction_scope
        self._conflict_strategy = conflict_strategy
        self._log_conflicts = log_conflicts
        self._initial_data: dict[str, Any] = initial_data or {}
        # Active subflow FSM (None when in main flow)
        self._active_subflow_fsm: WizardFSM | None = None
        # LLM provider set by generate() for transform context access
        self._current_llm: Any = None

        # Merge framework-level ephemeral keys with config-declared ones
        config_ephemeral = wizard_fsm.settings.get("ephemeral_keys", [])
        self._ephemeral_keys: frozenset[str] = (
            DEFAULT_EPHEMERAL_KEYS | frozenset(config_ephemeral)
        )

        # Build navigation keyword config from wizard-level settings
        nav_settings = wizard_fsm.settings.get("navigation", {})
        self._navigation_config: NavigationConfig = NavigationConfig.from_dict(
            nav_settings or {}
        )

        # Bridge FSM custom functions so they receive a TransformContext
        # instead of a raw FunctionContext.  This lets transforms that
        # need artifact_registry, rubric_executor, etc. work seamlessly.
        self._wrap_custom_functions(wizard_fsm)

    def _wrap_custom_functions(self, wizard_fsm: WizardFSM) -> None:
        """Wrap FSM custom functions to inject TransformContext.

        The FSM calls transform functions with ``(data, FunctionContext)``,
        but artifact-aware transforms expect ``(data, TransformContext)``.
        This method wraps each custom function so that the FSM's
        ``FunctionContext`` is translated into a ``TransformContext``
        carrying the artifact registry, review executor, and other
        services stored on this WizardReasoning instance.

        Args:
            wizard_fsm: The WizardFSM whose custom functions to wrap.
        """
        # Functions are registered on the core FSM's function_registry
        # (via FSMBuilder.register_function in WizardConfigLoader), not
        # on AdvancedFSM._custom_functions.  The AdvancedFSM merges both
        # registries in _execute_arc_transform_async, with _custom_functions
        # taking precedence.  We wrap the core registry functions and put
        # them into _custom_functions so the wrapped versions are used.
        advanced_fsm = wizard_fsm._fsm
        core_fsm = advanced_fsm.fsm
        registry = getattr(core_fsm, "function_registry", {})

        # Extract the raw function dict from the registry
        if hasattr(registry, "functions"):
            raw_fns: dict[str, Any] = dict(registry.functions)
        elif isinstance(registry, dict):
            raw_fns = dict(registry)
        else:
            raw_fns = {}

        if not raw_fns:
            return

        try:
            from ..artifacts import transforms as _transforms_mod
        except ImportError:
            # Artifact modules not available — wrapping not needed
            return
        del _transforms_mod

        wrapped: dict[str, Any] = {}
        for name, func in raw_fns.items():
            wrapped[name] = self._make_context_wrapper(func)

        # Store wrapped functions in AdvancedFSM._custom_functions which
        # overrides the core registry in _execute_arc_transform_async
        advanced_fsm._custom_functions = wrapped

        logger.debug(
            "Wrapped %d custom functions with TransformContext bridge",
            len(wrapped),
        )

    def _make_context_wrapper(
        self, func: Any
    ) -> Any:
        """Create a wrapper that injects TransformContext for a transform.

        Wizard transforms follow an in-place mutation convention: they
        modify ``data`` and return ``None``.  The FSM pipeline, however,
        chains transforms by passing each return value as the next
        transform's ``data``.  The wrapper preserves the original ``data``
        dict when the transform returns ``None``.

        Args:
            func: The original transform callable.

        Returns:
            Wrapped callable with same signature as FSM expects.
        """
        import asyncio
        import inspect

        from ..artifacts.transforms import TransformContext

        async def _async_wrapper(
            data: dict[str, Any], _func_context: Any = None, **kwargs: Any
        ) -> Any:
            transform_ctx = TransformContext(
                artifact_registry=self._artifact_registry,
                rubric_executor=self._review_executor,
                config={"llm": self._current_llm} if self._current_llm else {},
            )
            result = func(data, transform_ctx, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            # Transforms that mutate data in-place return None;
            # preserve the data dict for the next transform in the chain.
            return result if result is not None else data

        def _sync_wrapper(
            data: dict[str, Any], _func_context: Any = None, **kwargs: Any
        ) -> Any:
            transform_ctx = TransformContext(
                artifact_registry=self._artifact_registry,
                rubric_executor=self._review_executor,
                config={"llm": self._current_llm} if self._current_llm else {},
            )
            result = func(data, transform_ctx, **kwargs)
            return result if result is not None else data

        # Preserve async nature of the original function
        if asyncio.iscoroutinefunction(func):
            return _async_wrapper
        return _sync_wrapper

    async def close(self) -> None:
        """Close the reasoning strategy and release resources.

        Closes the SchemaExtractor's LLM provider if present, releasing
        HTTP connections. Should be called when the reasoning strategy
        is no longer needed (typically via DynaBot.close()).
        """
        if self._extractor is not None and hasattr(self._extractor, "close"):
            await self._extractor.close()
            logger.debug("Closed WizardReasoning extractor")

    def _partition_data(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Partition a flat working dict into persistent and transient parts.

        Keys in ``self._ephemeral_keys`` are routed to transient.  Keys not
        in the ephemeral set but failing the ``_is_json_safe`` check are
        also routed to transient as a safety net (and logged as warnings).

        Args:
            data: Flat working dict containing all wizard state keys.

        Returns:
            ``(persistent, transient)`` tuple of dicts.
        """
        persistent: dict[str, Any] = {}
        transient: dict[str, Any] = {}
        for key, value in data.items():
            if key in self._ephemeral_keys:
                transient[key] = value
            elif not _is_json_safe(value):
                logger.warning(
                    "Non-serializable key '%s' (type=%s) moved to transient",
                    key,
                    type(value).__name__,
                )
                transient[key] = value
            else:
                persistent[key] = value
        return persistent, transient

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
    def from_config(cls, config: dict[str, Any]) -> WizardReasoning:
        """Create WizardReasoning from configuration dict.

        Args:
            config: Configuration dict with:
                - wizard_config: Path to wizard YAML config file, or an
                  inline dict (compatible with
                  ``WizardConfigLoader.load_from_dict()``)
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

        wizard_config_value = config.get("wizard_config")
        if not wizard_config_value:
            raise ValueError("wizard_config is required")

        # Load wizard FSM — supports both file paths and inline dicts
        loader = WizardConfigLoader()
        custom_fns = config.get("custom_functions", {})

        if isinstance(wizard_config_value, dict):
            wizard_fsm = loader.load_from_dict(wizard_config_value, custom_fns)
        else:
            wizard_fsm = loader.load(wizard_config_value, custom_fns)

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
        extraction_scope = wizard_fsm.settings.get("extraction_scope", "wizard_session")
        conflict_strategy = wizard_fsm.settings.get("conflict_strategy", "latest_wins")
        log_conflicts = wizard_fsm.settings.get("log_conflicts", True)

        # Create artifact registry if artifact definitions configured
        artifact_registry = None
        artifacts_config = config.get("artifacts", {})
        if artifacts_config:
            try:
                from ..artifacts import ArtifactRegistry, ArtifactTypeDefinition
                from dataknobs_data.backends.memory import AsyncMemoryDatabase

                # Build type definitions from config
                type_definitions: dict[str, ArtifactTypeDefinition] = {}
                for def_id, def_config in artifacts_config.get(
                    "definitions", {}
                ).items():
                    type_definitions[def_id] = ArtifactTypeDefinition.from_config(
                        def_id, def_config
                    )

                # Create database backend (default: in-memory for conversation scope)
                db_backend = artifacts_config.get("backend", "memory")
                if db_backend == "memory":
                    artifact_db = AsyncMemoryDatabase()
                else:
                    from dataknobs_data import async_database_factory

                    artifact_db = async_database_factory.create(backend=db_backend)

                artifact_registry = ArtifactRegistry(
                    db=artifact_db,
                    type_definitions=type_definitions,
                )
                logger.info(
                    "Created artifact registry with %d type definitions",
                    len(type_definitions),
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
            extraction_scope=extraction_scope,
            conflict_strategy=conflict_strategy,
            log_conflicts=log_conflicts,
            initial_data=config.get("initial_data"),
        )

    async def greet(
        self,
        manager: Any,
        llm: Any,
        **kwargs: Any,
    ) -> Any | None:
        """Generate a bot-initiated greeting from the wizard's start stage.

        Initializes wizard state (restarting the FSM to the start stage) and
        generates a response using the start stage's ``response_template`` or
        LLM prompt — exactly as ``generate()`` would, but without a user
        message.

        This enables wizard scenarios to begin with the bot greeting the user
        (e.g. "Welcome! What is your name?") so the user's first turn answers
        the wizard's question rather than sending a throwaway message.

        Args:
            manager: ConversationManager or compatible manager instance
            llm: LLM provider instance
            **kwargs: Additional generation parameters

        Returns:
            LLM response object with wizard metadata
        """
        # Store LLM reference so transform context wrappers can access it
        self._current_llm = llm

        # Initialize fresh wizard state (restarts FSM to start stage)
        wizard_state = self._get_wizard_state(manager)

        logger.info(
            "Wizard greet: stage='%s', history=%s",
            wizard_state.current_stage,
            wizard_state.history,
        )

        # Get start stage metadata and generate the response
        stage = self._get_active_fsm().current_metadata
        response = await self._generate_stage_response(
            manager, llm, stage, wizard_state, tools=[],
        )

        # Record that the start stage template has been rendered so that
        # generate() does not re-render it as a "first confirmation" when
        # the user's first message arrives with extracted data.
        if stage.get("response_template"):
            wizard_state.increment_render_count(stage.get("name", "unknown"))

        # Persist wizard state
        self._save_wizard_state(manager, wizard_state)

        return response

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
        # Store LLM reference so transform context wrappers can access it
        self._current_llm = llm

        # Get or restore wizard state
        wizard_state = self._get_wizard_state(manager)

        # Get user message
        user_message = self._get_last_user_message(manager)

        logger.debug(
            "Wizard generate: stage='%s', completed=%s, "
            "data_keys=%s, history=%s, subflow_depth=%d",
            wizard_state.current_stage,
            wizard_state.completed,
            list(wizard_state.data.keys()),
            wizard_state.history,
            wizard_state.subflow_depth,
        )

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
                active_fsm = self._get_active_fsm()
                active_fsm.restore({
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
                    subflow_depth=wizard_state.subflow_depth,
                )
                wizard_state.transitions.append(transition)
                wizard_state.stage_entry_time = time.time()

                logger.info("Amendment: re-opening wizard at %s", target_stage)

                # Generate response for the re-opened stage
                stage = active_fsm.current_metadata
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

        # Get current stage context from active FSM (subflow or main)
        active_fsm = self._get_active_fsm()
        stage = active_fsm.current_metadata
        is_conversation = stage.get("mode") == "conversation"

        if is_conversation:
            # Conversation mode: skip extraction, run intent detection
            stage_name = stage.get("name", "unknown")
            logger.debug(
                "Conversation mode for stage '%s': skipping extraction",
                stage_name,
            )
            await self._detect_intent(user_message, stage, wizard_state, llm)
        else:
            # Structured mode: extract data and validate

            # Extract structured data from user input
            extraction = await self._extract_data(
                user_message, stage, llm, manager, wizard_state
            )

            # Log extraction results for debugging
            stage_name = stage.get("name", "unknown")
            logger.debug(
                "Extraction for stage '%s': confidence=%.2f, data_keys=%s",
                stage_name,
                extraction.confidence,
                list(extraction.data.keys()) if extraction.data else [],
            )
            if extraction.data:
                # Log actual extracted values (truncate long values)
                for key, value in extraction.data.items():
                    if not key.startswith("_"):
                        value_str = str(value)[:100]
                        logger.debug("  Extracted %s = %r", key, value_str)

            # Run intent detection on structured stages IF configured.
            # This runs before the confidence check so that detour intents
            # (e.g. "help", "confused") can trigger transitions even when
            # extraction fails.
            if stage.get("intent_detection"):
                await self._detect_intent(
                    user_message, stage, wizard_state, llm
                )

            # If an intent was detected, skip clarification/validation and
            # proceed to transition evaluation so the detour can fire.
            if "_intent" not in wizard_state.data:
                # Handle low confidence: check if required fields can be
                # satisfied from existing state + this extraction.
                if not extraction.is_confident:
                    schema = stage.get("schema", {})
                    required_fields = schema.get("required", [])
                    can_satisfy = all(
                        wizard_state.data.get(f) is not None
                        or extraction.data.get(f) is not None
                        for f in required_fields
                    )

                    if not can_satisfy:
                        wizard_state.clarification_attempts += 1
                        self._save_wizard_state(manager, wizard_state)

                        if wizard_state.clarification_attempts >= 3:
                            response = await self._generate_restart_offer(
                                manager, llm, stage, extraction.errors
                            )
                        else:
                            response = (
                                await self._generate_clarification_response(
                                    manager, llm, stage, extraction.errors
                                )
                            )

                        self._add_wizard_metadata(
                            response, wizard_state, stage
                        )
                        return response

                # Reset clarification attempts on viable extraction
                wizard_state.clarification_attempts = 0

                # Normalize extracted data against schema before merging
                schema = stage.get("schema")
                if schema and extraction.data:
                    extraction.data = self._normalize_extracted_data(
                        extraction.data, schema
                    )

                # Merge extracted data, tracking which keys are new or
                # changed so we can decide whether to show a confirmation
                # template before allowing a transition.
                new_data_keys: set[str] = set()
                for k, v in extraction.data.items():
                    if v is not None:
                        if (
                            k not in wizard_state.data
                            or wizard_state.data[k] != v
                        ):
                            new_data_keys.add(k)
                            wizard_state.data[k] = v
                    elif k not in wizard_state.data:
                        wizard_state.data[k] = v

                # Update field-extraction tasks
                self._update_field_tasks(wizard_state, extraction.data)

                # Apply schema defaults for properties the user didn't
                # mention.  This ensures template conditions (e.g.
                # ``{% if difficulty %}``) evaluate True when defaults
                # exist in the schema definition.
                default_keys = self._apply_schema_defaults(
                    wizard_state, stage
                )
                if default_keys:
                    new_data_keys |= default_keys

                # When meaningful new data was extracted at a stage with a
                # response_template, decide whether to render a
                # confirmation before evaluating transitions.
                #
                # Two modes:
                # 1. First-render (render_counts == 0): always confirm
                #    when new data exists.
                # 2. confirm_on_new_data: re-confirm whenever schema
                #    property values changed since the last render
                #    (catches "change difficulty to hard" after the
                #    initial summary was shown).
                #
                # Stages whose template has already been shown AND
                # that lack confirm_on_new_data skip this — the
                # user's response is an action (e.g. "review") and
                # should trigger a transition.
                should_confirm = False
                if new_data_keys and stage.get("response_template"):
                    if wizard_state.get_render_count(stage_name) == 0:
                        should_confirm = True  # First render — always
                    elif stage.get("confirm_on_new_data"):
                        # Re-confirm when schema property values changed
                        schema_props = set(
                            stage.get("schema", {})
                            .get("properties", {})
                            .keys()
                        )
                        current_snapshot = {
                            k: wizard_state.data[k]
                            for k in schema_props
                            if k in wizard_state.data
                            and wizard_state.data[k] is not None
                        }
                        prior_snapshot = wizard_state.get_stage_snapshot(
                            stage_name
                        )
                        if current_snapshot != prior_snapshot:
                            should_confirm = True

                if should_confirm:
                    render_count = wizard_state.increment_render_count(
                        stage_name
                    )
                    # Save snapshot for confirm_on_new_data comparison
                    if stage.get("confirm_on_new_data"):
                        schema_props = set(
                            stage.get("schema", {})
                            .get("properties", {})
                            .keys()
                        )
                        wizard_state.save_stage_snapshot(
                            stage_name, schema_props
                        )
                    logger.debug(
                        "New data extracted (%s) at stage '%s' — "
                        "rendering confirmation (render #%d)",
                        new_data_keys,
                        stage_name,
                        render_count,
                    )
                    response = await self._generate_stage_response(
                        manager, llm, stage, wizard_state, tools
                    )
                    self._save_wizard_state(manager, wizard_state)
                    return response

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

        # Check for subflow push BEFORE regular FSM transition
        subflow_config = self._should_push_subflow(wizard_state, user_message)
        if subflow_config:
            # Push to subflow
            if self._handle_subflow_push(wizard_state, subflow_config, user_message):
                # Generate response for subflow's first stage
                active_fsm = self._get_active_fsm()
                new_stage = active_fsm.current_metadata
                response = await self._generate_stage_response(
                    manager, llm, new_stage, wizard_state, tools
                )
                self._save_wizard_state(manager, wizard_state)
                return response

        # Capture state before transition
        from_stage = wizard_state.current_stage
        duration_ms = (time.time() - wizard_state.stage_entry_time) * 1000

        # Apply data derivations from transition configs before evaluating
        # conditions.  This lets transitions fill in values that enable their
        # own conditions and subsequent auto-advance checks.
        self._apply_transition_derivations(stage, wizard_state)

        # Log pre-transition state
        logger.debug(
            "FSM transition attempt: from_stage='%s', data_keys=%s",
            from_stage,
            list(wizard_state.data.keys()),
        )

        # Inject raw message for condition evaluation (prefixed with _ per convention)
        wizard_state.data["_message"] = user_message
        # Execute FSM transition using active FSM (async to support async transforms)
        step_result = await active_fsm.step_async(wizard_state.data)
        wizard_state.data.pop("_message", None)
        to_stage = active_fsm.current_stage

        # Log transition result
        if to_stage != from_stage:
            logger.info(
                "FSM transition: '%s' -> '%s' (is_complete=%s, depth=%d)",
                from_stage,
                to_stage,
                step_result.is_complete,
                wizard_state.subflow_depth,
            )
        elif not step_result.success:
            # A transition condition matched but the transform failed.
            # Log the error prominently so it's visible in server logs.
            # Don't return early — fall through to normal stage response
            # generation (template rendering) so the user sees their
            # collected data.  Store the error in wizard state so the UI
            # can surface it via metadata.
            logger.warning(
                "FSM transition transform failed at '%s': %s",
                from_stage,
                step_result.error,
            )
            wizard_state.data["_transform_error"] = step_result.error
        else:
            logger.debug(
                "FSM no transition: stayed at '%s' (is_complete=%s)",
                from_stage,
                step_result.is_complete,
            )

        # Record the transition if stage changed
        if to_stage != from_stage:
            # Look up the condition that was evaluated for this transition
            condition_expr = active_fsm.get_transition_condition(from_stage, to_stage)

            transition = create_transition_record(
                from_stage=from_stage,
                to_stage=to_stage,
                trigger="user_input",
                duration_in_stage_ms=duration_ms,
                data_snapshot=wizard_state.data.copy(),
                user_input=user_message,
                condition_evaluated=condition_expr,
                condition_result=True if condition_expr else None,
                subflow_depth=wizard_state.subflow_depth,
            )
            wizard_state.transitions.append(transition)
            wizard_state.stage_entry_time = time.time()

        wizard_state.current_stage = to_stage
        if wizard_state.current_stage not in wizard_state.history:
            wizard_state.history.append(wizard_state.current_stage)
        wizard_state.completed = step_result.is_complete

        # Check for subflow pop (reached end state in subflow)
        if self._should_pop_subflow(wizard_state):
            self._handle_subflow_pop(wizard_state)
            # Update active FSM reference after pop
            active_fsm = self._get_active_fsm()
            # Not completed since we returned to parent flow
            wizard_state.completed = False

        # Auto-advance through stages where all required fields are filled
        new_stage = active_fsm.current_metadata
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

            # Execute FSM transition for auto-advance (async to support async transforms)
            auto_step_result = await active_fsm.step_async(wizard_state.data)
            new_stage_name = active_fsm.current_stage

            if new_stage_name == old_stage_name:
                # No transition occurred, stop auto-advancing
                break

            # Record the auto-advance transition
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

            # Check for subflow pop during auto-advance
            if self._should_pop_subflow(wizard_state):
                self._handle_subflow_pop(wizard_state)
                active_fsm = self._get_active_fsm()
                wizard_state.completed = False

            # Get new stage metadata for next iteration
            new_stage = active_fsm.current_metadata

        # Trigger stage entry hook if configured
        if self._hooks:
            await self._hooks.trigger_enter(
                wizard_state.current_stage, wizard_state.data
            )

        # Trigger completion hook if wizard is complete
        if wizard_state.completed and self._hooks:
            await self._hooks.trigger_complete(wizard_state.data)

        # Generate stage-aware response
        new_stage = active_fsm.current_metadata
        response = await self._generate_stage_response(
            manager, llm, new_stage, wizard_state, tools
        )

        # Mark this stage's template as rendered so subsequent messages
        # at this stage don't trigger the first-render confirmation logic.
        stage_rendered_name = new_stage.get("name", "")
        if stage_rendered_name and new_stage.get("response_template"):
            wizard_state.increment_render_count(stage_rendered_name)

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
            # Restore subflow stack from serialized data
            subflow_stack = [
                SubflowContext.from_dict(s)
                for s in fsm_state.get("subflow_stack", [])
            ]
            state = WizardState(
                current_stage=fsm_state.get(
                    "current_stage", self._fsm.current_stage
                ),
                data=copy.deepcopy(fsm_state.get("data", {})),
                history=list(fsm_state.get("history", [])),
                completed=fsm_state.get("completed", False),
                clarification_attempts=fsm_state.get("clarification_attempts", 0),
                transitions=transitions,
                stage_entry_time=fsm_state.get("stage_entry_time", time.time()),
                tasks=tasks,
                subflow_stack=subflow_stack,
            )
            # Restore FSM state
            self._fsm.restore(fsm_state)
            # Restore active subflow FSM if in subflow
            if subflow_stack:
                subflow_name = subflow_stack[-1].subflow_network
                self._active_subflow_fsm = self._fsm.get_subflow(subflow_name)
                if self._active_subflow_fsm:
                    self._active_subflow_fsm.restore(fsm_state)
            else:
                self._active_subflow_fsm = None
            return state

        # Initialize new wizard state with tasks from config
        # Reset FSM context to ensure we start at the beginning, not at a
        # stale stage from a previous conversation (if bot instance is cached)
        self._fsm.restart()
        start_stage = self._fsm.current_stage
        initial_tasks = self._build_initial_tasks()
        self._active_subflow_fsm = None  # Ensure we start in main flow

        # Inject initial data from reasoning config (e.g., quiz_bank_ids).
        # These are set at bot creation time and available to all transforms.
        initial_data: dict[str, Any] = dict(self._initial_data)

        # Inject wizard settings into initial data so transforms can access them.
        # output_paths provides configurable file output locations.
        output_paths = self._fsm.settings.get("output_paths")
        if output_paths:
            initial_data["_output_paths"] = dict(output_paths)

        return WizardState(
            current_stage=start_stage,
            data=initial_data,
            history=[start_stage],
            stage_entry_time=time.time(),
            tasks=initial_tasks,
        )

    def _build_wizard_metadata(self, state: WizardState) -> dict[str, Any]:
        """Build canonical wizard metadata from current state.

        Single source of truth for wizard metadata.  Used by both
        ``_save_wizard_state`` (persistence) and ``_add_wizard_metadata``
        (response decoration) to ensure consistency.

        When inside a subflow, ``stage_index``, ``total_stages``, and
        ``progress_percent`` always report **main-flow** progress so
        the UI can render an accurate overall progress indicator.  The
        active subflow's stage is exposed separately via
        ``subflow_stage``.

        Args:
            state: Current wizard state

        Returns:
            Canonical wizard metadata dict.
        """
        active_fsm = self._get_active_fsm()

        # Always report main-flow progress for the roadmap / breadcrumb
        main_stage_names = self._fsm.stage_names

        if state.subflow_stack:
            # During a subflow, the "effective" main-flow stage is the
            # parent stage that pushed the subflow.
            effective_main_stage = state.subflow_stack[-1].parent_stage
        else:
            effective_main_stage = state.current_stage

        try:
            stage_index = main_stage_names.index(effective_main_stage)
        except ValueError:
            stage_index = 0

        total_stages = len(main_stage_names)
        progress = stage_index / max(total_stages - 1, 1)

        metadata: dict[str, Any] = {
            "current_stage": state.current_stage,
            "stage_index": stage_index,
            "total_stages": total_stages,
            "progress": progress,
            "progress_percent": progress * 100,
            "completed": state.completed,
            "data": sanitize_for_json({**state.data, **state.transient}),
            "history": state.history,
            "can_skip": active_fsm.can_skip(),
            "can_go_back": active_fsm.can_go_back() and len(state.history) > 1,
            "stage_prompt": active_fsm.get_stage_prompt(),
            "suggestions": self._render_suggestions(
                active_fsm.get_stage_suggestions(), state
            ),
            "stages": self._build_stages_roadmap(state),
            "stage_mode": active_fsm.current_metadata.get("mode") or "structured",
        }

        # Expose subflow context when active
        if state.subflow_stack:
            subflow_stage_meta = active_fsm.current_metadata
            metadata["subflow_stage"] = {
                "name": state.current_stage,
                "label": subflow_stage_meta.get(
                    "label", state.current_stage
                ),
            }

        return metadata

    def _save_wizard_state(self, manager: Any, state: WizardState) -> None:
        """Save wizard state to conversation manager.

        Partitions ``state.data`` into persistent and transient parts before
        persisting.  Only persistent data is written to ``fsm_state``.
        ``_build_wizard_metadata`` is called *before* partition so the response
        metadata (sent to the UI) still contains all keys.

        Args:
            manager: ConversationManager instance
            state: WizardState to save
        """
        # Build metadata BEFORE partition so UI sees all keys (incl. transient)
        wizard_meta = self._build_wizard_metadata(state)

        # Partition: separate ephemeral/non-serializable keys from persistent
        state.data, state.transient = self._partition_data(state.data)

        wizard_meta["fsm_state"] = {
            "current_stage": state.current_stage,
            "history": state.history,
            "data": sanitize_for_json(state.data),
            "completed": state.completed,
            "clarification_attempts": state.clarification_attempts,
            "transitions": [
                sanitize_for_json(t.to_dict()) for t in state.transitions
            ],
            "stage_entry_time": state.stage_entry_time,
            "tasks": state.tasks.to_dict(),
            "subflow_stack": [s.to_dict() for s in state.subflow_stack],
        }
        manager.metadata["wizard"] = wizard_meta

    # =========================================================================
    # Subflow Management Methods
    # =========================================================================

    def _get_active_fsm(self) -> WizardFSM:
        """Get the currently active FSM (subflow or main).

        Returns:
            The active WizardFSM instance
        """
        return self._active_subflow_fsm if self._active_subflow_fsm else self._fsm

    def _should_push_subflow(
        self, wizard_state: WizardState, user_message: str
    ) -> dict[str, Any] | None:
        """Check if the current transition should push a subflow.

        Examines the transitions from the current stage to see if any
        matching transition is a subflow transition.

        Args:
            wizard_state: Current wizard state
            user_message: User message for context

        Returns:
            Subflow config dict if should push, None otherwise
        """
        # Guard: Don't push subflow if already in one
        # This prevents duplicate pushes after state restoration
        if wizard_state.is_in_subflow:
            return None

        active_fsm = self._get_active_fsm()
        stage_meta = active_fsm.current_metadata

        # Check each transition for subflow marker
        for transition in stage_meta.get("transitions", []):
            if not transition.get("is_subflow_transition"):
                continue

            # Evaluate condition if present
            condition = transition.get("condition")
            if condition:
                if not self._evaluate_condition(condition, wizard_state.data):
                    continue

            # This transition matches and is a subflow transition
            return transition.get("subflow_config", {})

        return None

    def _handle_subflow_push(
        self,
        wizard_state: WizardState,
        subflow_config: dict[str, Any],
        user_message: str,
    ) -> bool:
        """Push a subflow onto the stack.

        Saves parent state and switches to the subflow FSM.

        Args:
            wizard_state: Current wizard state
            subflow_config: Subflow configuration dict
            user_message: User message for context

        Returns:
            True if subflow was pushed successfully
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
        subflow_data = self._apply_data_mapping(wizard_state.data, data_mapping)

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

    def _handle_subflow_pop(
        self,
        wizard_state: WizardState,
    ) -> bool:
        """Pop the current subflow and return to parent.

        Applies result mapping and restores parent state.

        Args:
            wizard_state: Current wizard state

        Returns:
            True if subflow was popped successfully
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
        result_data = self._apply_result_mapping(
            wizard_state.data, subflow_context.result_mapping
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
        active_fsm = self._get_active_fsm()
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

    def _apply_data_mapping(
        self,
        source_data: dict[str, Any],
        mapping: dict[str, str],
    ) -> dict[str, Any]:
        """Apply data mapping from parent to subflow.

        Args:
            source_data: Source data dict (parent wizard data)
            mapping: Dict mapping parent field names to subflow field names

        Returns:
            Mapped data dict for subflow
        """
        if not mapping:
            return {}

        result: dict[str, Any] = {}
        for parent_field, subflow_field in mapping.items():
            if parent_field in source_data:
                result[subflow_field] = source_data[parent_field]

        return result

    def _apply_result_mapping(
        self,
        source_data: dict[str, Any],
        mapping: dict[str, str],
    ) -> dict[str, Any]:
        """Apply result mapping from subflow to parent.

        Args:
            source_data: Source data dict (subflow wizard data)
            mapping: Dict mapping subflow field names to parent field names

        Returns:
            Mapped data dict for parent
        """
        if not mapping:
            return {}

        result: dict[str, Any] = {}
        for subflow_field, parent_field in mapping.items():
            if subflow_field in source_data:
                result[parent_field] = source_data[subflow_field]

        return result

    def _should_pop_subflow(self, wizard_state: WizardState) -> bool:
        """Check if the current stage is a subflow end state.

        Args:
            wizard_state: Current wizard state

        Returns:
            True if current stage is an end stage and we're in a subflow
        """
        if not wizard_state.is_in_subflow:
            return False

        active_fsm = self._get_active_fsm()
        return active_fsm.is_end_stage(wizard_state.current_stage)

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

    def _get_last_bot_response(self, manager: Any) -> str:
        """Extract the last assistant message from conversation.

        Used to provide the bot's previous response as context for
        extraction, so the extraction model can resolve references
        like "the first suggestion" or "yes to that".

        Args:
            manager: ConversationManager instance

        Returns:
            Last assistant message text, or empty string if none found.
        """
        messages = manager.get_messages()
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Handle structured content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")
        return ""

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
            keywords = override_raw.get("keywords")
            if keywords is not None:
                keywords = tuple(k.lower() for k in keywords)
            else:
                keywords = base.keywords
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

    async def _handle_navigation(
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
            return await self._execute_skip(state, manager)

        if nav.restart.enabled and lower in nav.restart.keywords:
            return await self._execute_restart(message, state, manager, llm)

        return None  # Not a navigation command

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
            response = await self._generate_stage_response(
                manager, llm, stage, state, None
            )
            # Record render so next input doesn't trigger first-render
            # confirmation.
            if stage.get("response_template"):
                state.increment_render_count(
                    stage.get("name", "unknown")
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
        state: WizardState,
        manager: Any,
    ) -> Any | None:
        """Execute skip navigation.

        Args:
            state: Current wizard state
            manager: ConversationManager instance

        Returns:
            ``None`` on success (falls through to transition evaluation),
            or an explanation response if skip is not allowed.
        """
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

    async def _execute_restart(
        self,
        message: str,
        state: WizardState,
        manager: Any,
        llm: Any,
    ) -> Any:
        """Execute restart navigation.

        Args:
            message: Original user message
            state: Current wizard state
            manager: ConversationManager instance
            llm: LLM provider

        Returns:
            Response for the restarted first stage.
        """
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
        response = await self._generate_stage_response(
            manager, llm, stage, state, None
        )
        # Record that the start stage template has been rendered so
        # the next user message doesn't trigger first-render confirmation.
        if stage.get("response_template"):
            state.increment_render_count(stage.get("name", "unknown"))
        return response

    async def _detect_intent(
        self,
        message: str,
        stage: dict[str, Any],
        state: WizardState,
        llm: Any,
    ) -> None:
        """Detect user intent and store in wizard state data.

        Examines the stage's ``intent_detection`` configuration and
        classifies the user message into one of the configured intents.
        The result is stored in ``state.data["_intent"]`` for use in
        transition conditions.

        Supports two detection methods:

        - **keyword**: Fast substring matching against configured keywords.
          First matching intent wins.
        - **llm**: Lightweight LLM classification.  Builds a prompt listing
          intents and their descriptions, asks the LLM to pick one.

        Args:
            message: Raw user message text
            stage: Current stage metadata (must contain ``intent_detection``)
            state: Current wizard state (``_intent`` is set here)
            llm: LLM provider instance (used only for ``method: llm``)
        """
        state.data.pop("_intent", None)

        intent_config = stage.get("intent_detection")
        if not intent_config:
            return

        method = intent_config.get("method", "keyword")
        intents = intent_config.get("intents", [])

        if method == "keyword":
            lower_msg = message.lower()
            for intent in intents:
                if any(kw in lower_msg for kw in intent.get("keywords", [])):
                    state.data["_intent"] = intent["id"]
                    logger.debug("Keyword intent detected: %s", intent["id"])
                    return

        elif method == "llm":
            intent_list = "\n".join(
                f"- {i['id']}: {i.get('description', '')}" for i in intents
            )
            prompt = (
                f"Classify the user's intent from this message:\n"
                f'"{message}"\n\n'
                f"Possible intents:\n{intent_list}\n\n"
                f"Return ONLY the intent ID, or 'none' if no intent matches."
            )
            try:
                from dataknobs_llm import LLMMessage

                response = await llm.complete(
                    messages=[LLMMessage(role="user", content=prompt)],
                )
                if response and response.content:
                    intent_id = response.content.strip().lower()
                    valid_ids = {i["id"] for i in intents}
                    if intent_id in valid_ids:
                        state.data["_intent"] = intent_id
                        logger.debug("LLM intent detected: %s", intent_id)
            except Exception:
                logger.warning("LLM intent detection failed", exc_info=True)

    async def _extract_data(
        self,
        message: str,
        stage: dict[str, Any],
        llm: Any,
        manager: Any | None = None,
        wizard_state: WizardState | None = None,
    ) -> Any:
        """Extract structured data from user message or wizard session.

        When extraction_scope is "wizard_session", builds context from all
        user messages in the wizard session for extraction. This allows the
        wizard to remember information provided in earlier messages.

        Schema 'default' values are stripped before extraction to prevent
        the LLM from auto-filling them. This ensures extraction only captures
        what the user actually said.

        Args:
            message: Current user message text
            stage: Current stage metadata
            llm: LLM provider (fallback if no extractor)
            manager: ConversationManager for accessing message history
            wizard_state: Current wizard state for conflict detection

        Returns:
            ExtractionResult with data and confidence
        """
        # Create a simple result class for when extractor is not available
        @dataclass
        class SimpleExtractionResult:
            data: dict[str, Any] = field(default_factory=dict)
            confidence: float = 0.0
            errors: list[str] = field(default_factory=list)
            metadata: dict[str, Any] = field(default_factory=dict)

            @property
            def is_confident(self) -> bool:
                return self.confidence >= 0.8 and not self.errors

        schema = stage.get("schema")
        stage_name = stage.get("name", "unknown")

        logger.debug(
            "Extraction start: stage='%s', has_schema=%s, "
            "has_extractor=%s, input_len=%d",
            stage_name,
            schema is not None,
            self._extractor is not None,
            len(message),
        )

        if not schema:
            # No schema defined - pass through any data
            logger.debug(
                "Extraction skip: stage='%s' has no schema, returning raw input",
                stage_name,
            )
            return SimpleExtractionResult(
                data={"_raw_input": message}, confidence=1.0
            )

        # Build extraction input based on scope (stage override or wizard default)
        extraction_scope = self._get_extraction_scope(stage)
        if (
            extraction_scope == "wizard_session"
            and manager is not None
            and wizard_state is not None
        ):
            # Build context from wizard session conversation
            wizard_context = self._build_wizard_context(manager, wizard_state)
            if wizard_context:
                extraction_input = (
                    f"{wizard_context}\n\nCurrent message: {message}"
                )
                logger.debug(
                    "Wizard session extraction: %d chars of context + current message",
                    len(wizard_context),
                )
            else:
                extraction_input = message
        else:
            # Current message only (original behavior)
            extraction_input = message

        # Include the bot's last response so the extraction model can
        # resolve references like "the first suggestion" or "yes to that".
        if manager is not None:
            bot_response = self._get_last_bot_response(manager)
            if bot_response:
                # Truncate very long responses to avoid overwhelming extraction
                if len(bot_response) > 1500:
                    bot_response = bot_response[:1500] + "..."
                extraction_input = (
                    f"Bot's previous message:\n{bot_response}\n\n"
                    f"User's response:\n{extraction_input}"
                )
                logger.debug(
                    "Included bot response (%d chars) in extraction context",
                    len(bot_response),
                )

        # Strip defaults to prevent extraction LLM from auto-filling them
        extraction_schema = self._strip_schema_defaults(schema)

        if self._extractor:
            # Use schema extractor
            extraction_model = stage.get("extraction_model")
            context = {"stage": stage.get("name"), "prompt": stage.get("prompt")}
            result = await self._extractor.extract(
                text=extraction_input,
                schema=extraction_schema,
                context=context,
                model=extraction_model,
            )

            logger.debug(
                "Extraction result: stage='%s', keys=%s, confidence=%.2f, "
                "errors=%s",
                stage_name,
                list(result.data.keys()) if result.data else [],
                getattr(result, "confidence", -1.0),
                getattr(result, "errors", []),
            )

            # Detect conflicts with existing data
            if wizard_state is not None and result.data:
                conflicts = self._detect_conflicts(wizard_state.data, result.data)
                if conflicts:
                    if self._log_conflicts:
                        for conflict in conflicts:
                            logger.info(
                                "Data conflict detected for field '%s': "
                                "'%s' -> '%s' (using %s)",
                                conflict["field"],
                                conflict["previous"],
                                conflict["new"],
                                self._conflict_strategy,
                            )
                    # Add conflicts to result metadata for downstream use
                    if not hasattr(result, "metadata") or result.metadata is None:
                        result.metadata = {}
                    result.metadata["conflicts"] = conflicts

            return result

        # Fallback: simple heuristic extraction
        # This is very basic - the extractor should be used for real scenarios
        return SimpleExtractionResult(
            data={"_raw_input": message}, confidence=0.5
        )

    def _build_wizard_context(
        self, manager: Any, wizard_state: WizardState
    ) -> str:
        """Build extraction context from wizard session history.

        Collects all user messages from the conversation to provide
        full context for extraction. This allows the wizard to
        "remember" information provided in earlier messages.

        Args:
            manager: ConversationManager instance
            wizard_state: Current wizard state

        Returns:
            Formatted context string from previous user messages,
            or empty string if no previous messages.
        """
        messages = manager.get_messages()

        # Collect user messages (excluding the most recent which is current)
        user_messages: list[str] = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_messages.append(content)
                elif isinstance(content, list):
                    # Handle structured content
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            user_messages.append(part.get("text", ""))

        # Exclude the last message (it's the current one we're processing)
        previous_messages = user_messages[:-1] if len(user_messages) > 1 else []

        if not previous_messages:
            return ""

        # Format as context
        formatted = ["Previous conversation:"]
        for i, msg in enumerate(previous_messages, 1):
            # Truncate very long messages
            truncated = msg[:500] + "..." if len(msg) > 500 else msg
            formatted.append(f"  Message {i}: {truncated}")

        return "\n".join(formatted)

    def _detect_conflicts(
        self,
        existing_data: dict[str, Any],
        new_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Detect conflicts between existing and newly extracted data.

        A conflict occurs when a field exists in both dicts with
        different non-None values.

        Args:
            existing_data: Data already in wizard state
            new_data: Newly extracted data

        Returns:
            List of conflict dicts with field, previous, and new values.
        """
        conflicts: list[dict[str, Any]] = []

        for field_name, new_value in new_data.items():
            # Skip internal fields
            if field_name.startswith("_"):
                continue

            # Skip if new value is None
            if new_value is None:
                continue

            # Check if field exists with a different value
            if field_name in existing_data:
                existing_value = existing_data[field_name]
                # Only count as conflict if existing is non-None and different
                if existing_value is not None and existing_value != new_value:
                    conflicts.append({
                        "field": field_name,
                        "previous": existing_value,
                        "new": new_value,
                    })

        return conflicts

    # -- Boolean truthy/falsy strings for normalization --
    _BOOL_TRUE = frozenset({"yes", "true", "1", "y", "on", "enable", "enabled"})
    _BOOL_FALSE = frozenset({"no", "false", "0", "n", "off", "disable", "disabled"})
    _ALL_KEYWORDS = frozenset({"all", "everything", "all of them", "every one"})
    _NONE_KEYWORDS = frozenset({"none", "nothing", "no tools", "empty"})

    def _normalize_extracted_data(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize extracted data to match schema types.

        Applies deterministic, schema-driven corrections to LLM-extracted
        data *before* it enters wizard state.  Only acts when the extracted
        type doesn't match the declared schema type, so existing well-typed
        extractions are passed through untouched.

        Normalizations performed:

        * **Boolean coercion** - string ``"yes"``/``"true"`` → ``True``, etc.
        * **Array wrapping** - bare string for an ``array`` field → ``[value]``
        * **Array shortcut expansion** - ``["all"]`` for ``array`` + ``items.enum``
          → all enum values; ``["none"]`` → ``[]``
        * **Number coercion** - string digits for ``integer``/``number`` → cast

        Args:
            data:   Extracted data dict (will be shallow-copied).
            schema: JSON Schema for the current stage.

        Returns:
            New dict with normalized values.
        """
        properties = schema.get("properties", {})
        if not properties:
            return data

        normalized = dict(data)

        for field_name, value in data.items():
            if field_name.startswith("_") or field_name not in properties:
                continue

            prop = properties[field_name]
            declared_type = prop.get("type")

            # --- Boolean coercion ---
            if declared_type == "boolean" and isinstance(value, str):
                lower = value.strip().lower()
                if lower in self._BOOL_TRUE:
                    normalized[field_name] = True
                    logger.debug("Normalized %s: %r → True", field_name, value)
                elif lower in self._BOOL_FALSE:
                    normalized[field_name] = False
                    logger.debug("Normalized %s: %r → False", field_name, value)

            # --- Integer coercion ---
            elif declared_type == "integer" and isinstance(value, str):
                stripped = value.strip()
                if stripped.lstrip("-").isdigit():
                    normalized[field_name] = int(stripped)
                    logger.debug(
                        "Normalized %s: %r → %d", field_name, value, int(stripped)
                    )

            # --- Number (float) coercion ---
            elif declared_type == "number" and isinstance(value, str):
                stripped = value.strip()
                try:
                    normalized[field_name] = float(stripped)
                    logger.debug(
                        "Normalized %s: %r → %f",
                        field_name,
                        value,
                        float(stripped),
                    )
                except ValueError:
                    pass  # Leave as-is; validation will catch it

            # --- Array handling ---
            elif declared_type == "array":
                items_schema = prop.get("items", {})
                enum_values = items_schema.get("enum", [])

                # Wrap bare string → list
                if isinstance(value, str):
                    value = [value]
                    normalized[field_name] = value
                    logger.debug(
                        "Normalized %s: wrapped string → list", field_name
                    )

                # Expand "all"/"none" shortcuts when enum is defined
                if isinstance(value, list) and enum_values:
                    lower_items = {
                        v.strip().lower() for v in value if isinstance(v, str)
                    }
                    if lower_items & self._ALL_KEYWORDS:
                        normalized[field_name] = list(enum_values)
                        logger.debug(
                            "Normalized %s: 'all' → %s",
                            field_name,
                            enum_values,
                        )
                    elif lower_items & self._NONE_KEYWORDS:
                        normalized[field_name] = []
                        logger.debug(
                            "Normalized %s: 'none' → []", field_name
                        )

        return normalized

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

        Supports two modes:

        1. **Template mode** (when stage has ``response_template``):
           Renders the template with Jinja2 using wizard state data,
           bypassing the LLM entirely. If the stage also has
           ``llm_assist: true`` and the user's last message is a
           question, the LLM is invoked with a scoped assist prompt.

        2. **LLM mode** (default): Calls the LLM with stage context
           injected into the system prompt, as before.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            state: Current wizard state
            tools: Available tools

        Returns:
            LLM response with wizard metadata
        """
        stage_name = stage.get("name", "unknown")
        response_template = stage.get("response_template")

        # Build wizard metadata snapshot once — passed to whichever call
        # creates the conversation node so every path persists it.
        wizard_snapshot = {"wizard": self._build_wizard_metadata(state)}

        # ── Template mode ────────────────────────────────────────
        if response_template:
            # Generate LLM context variables if configured
            extra_context = await self._generate_context_variables(
                stage, state, llm
            )

            rendered = self._render_response_template(
                response_template, stage, state, extra_context=extra_context
            )

            # Check if user is asking a question and llm_assist is enabled
            user_message = self._get_last_user_message(manager)
            if stage.get("llm_assist") and user_message and self._is_help_request(user_message):
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
            else:
                logger.debug(
                    "Template response for stage '%s' (%d chars)",
                    stage_name,
                    len(rendered),
                )
                response = self._create_template_response(rendered)
                # Persist template response to conversation store
                # (manager.complete() does this automatically, but template
                # mode bypasses the LLM so we must persist explicitly)
                await manager.add_message(
                    role="assistant",
                    content=rendered,
                    metadata=wizard_snapshot,
                )

            self._add_wizard_metadata(response, state, stage)
            return response

        # ── LLM mode (original behavior) ─────────────────────────
        # Build stage-aware system prompt
        stage_context = self._build_stage_context(stage, state)
        enhanced_prompt = f"{manager.system_prompt}\n\n{stage_context}"

        # Filter tools to stage-specific ones
        stage_tools = self._filter_tools_for_stage(stage, tools)

        # Check if stage should use ReAct-style reasoning
        logger.debug(
            "Generating response for stage '%s' (tools=%s, react=%s)",
            stage_name,
            [getattr(t, "name", str(t)) for t in stage_tools] if stage_tools else None,
            stage_tools and self._use_react_for_stage(stage),
        )

        if stage_tools and self._use_react_for_stage(stage):
            response = await self._react_stage_response(
                manager, enhanced_prompt, stage, state, stage_tools,
                metadata=wizard_snapshot,
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
        self._add_wizard_metadata(response, state, stage)

        return response

    def _render_response_template(
        self,
        template_str: str,
        stage: dict[str, Any],
        state: WizardState,
        extra_context: dict[str, Any] | None = None,
    ) -> str:
        """Render a stage response template with wizard state data.

        Uses Jinja2 to render the template with collected wizard data,
        stage metadata, and optional extra context variables (e.g. from
        LLM context generation).

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
        import jinja2

        env = jinja2.Environment(undefined=jinja2.Undefined)
        template = env.from_string(template_str)

        # Non-internal keys for backward-compatible "collected_data" dict
        collected_data = {
            k: v for k, v in state.data.items() if not k.startswith("_")
        }

        # Build template context — ALL state data is available as
        # top-level variables so templates can reference both user-facing
        # fields (topic, difficulty) and transform outputs (_questions,
        # _bank_questions).  Both persistent (state.data) and transient
        # (state.transient) keys are included so templates see everything
        # even after partition.
        all_data = {**state.data, **state.transient}
        context: dict[str, Any] = {
            # Stage metadata
            "stage_name": stage.get("name", "unknown"),
            "stage_label": stage.get("label", stage.get("name", "")),
            # All data as top-level variables (including _-prefixed)
            **all_data,
            # Filtered dict (backward compatibility)
            "collected_data": collected_data,
            # Full data dict reference
            "all_data": all_data,
            # Wizard progress
            "history": state.history,
            "completed": state.completed,
        }

        # Merge extra context (e.g. LLM-generated variables)
        if extra_context:
            context.update(extra_context)

        logger.debug(
            "Template render: stage='%s', template_len=%d, "
            "context_keys=%s",
            stage.get("name", "unknown"),
            len(template_str),
            list(context.keys()),
        )

        return template.render(**context)

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

        The ``context_generation`` block supports:
        - ``prompt``: Jinja2 template string rendered with wizard state data,
          then sent to the LLM as the user message.
        - ``variable``: Name of the variable to inject into the template context.
        - ``model``: LLM model or ``$resource:`` reference (optional; defaults
          to the wizard's extraction model setting).
        - ``fallback``: Value to use if the LLM call fails or times out.

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
        import jinja2

        try:
            env = jinja2.Environment(undefined=jinja2.Undefined)
            rendered_prompt = env.from_string(prompt_template).render(
                **{k: v for k, v in state.data.items() if not k.startswith("_")}
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

    def _render_suggestions(
        self,
        suggestions: list[str],
        state: WizardState,
    ) -> list[str]:
        """Render suggestion strings through Jinja2 with wizard state data.

        Allows suggestions to reference collected data, e.g.
        ``"Call it '{{ subject }} Ace'"`` becomes ``"Call it 'Chemistry Ace'"``.

        Plain suggestions (no ``{{ }}``) pass through unchanged for
        efficiency.

        Args:
            suggestions: List of suggestion template strings
            state: Current wizard state

        Returns:
            List of rendered suggestion strings
        """
        if not suggestions:
            return suggestions

        # Quick check: if no templates, return as-is
        if not any("{%" in s or "{{" in s for s in suggestions):
            return suggestions

        import jinja2

        env = jinja2.Environment(undefined=jinja2.Undefined)
        collected_data = {
            k: v for k, v in state.data.items() if not k.startswith("_")
        }

        rendered = []
        for suggestion in suggestions:
            if "{%" not in suggestion and "{{" not in suggestion:
                rendered.append(suggestion)
                continue
            try:
                rendered.append(env.from_string(suggestion).render(**collected_data))
            except Exception:
                rendered.append(suggestion)
        return rendered

    @staticmethod
    def _create_template_response(content: str) -> Any:
        """Create a minimal response object from template-rendered text.

        The returned object is duck-type compatible with LLMResponse,
        carrying the attributes that downstream code accesses:
        ``content``, ``metadata``, and ``model``.

        Args:
            content: Rendered template text

        Returns:
            Response object compatible with the wizard pipeline
        """
        from dataclasses import dataclass as _dataclass
        from dataclasses import field as _field
        from datetime import datetime

        @_dataclass
        class _TemplateResponse:
            content: str
            model: str = "template"
            finish_reason: str | None = "stop"
            usage: dict[str, int] | None = None
            tool_calls: list[Any] | None = None
            metadata: dict[str, Any] = _field(default_factory=dict)
            created_at: datetime = _field(default_factory=datetime.now)

        return _TemplateResponse(content=content)

    @staticmethod
    def _is_help_request(message: str) -> bool:
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

    def _add_wizard_metadata(
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

    def _build_stages_roadmap(
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

        for name, meta in self._fsm._stage_metadata.items():
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

    def _get_extraction_scope(self, stage: dict[str, Any]) -> str:
        """Get extraction scope for a stage.

        Allows per-stage override of the global extraction_scope setting.

        Args:
            stage: Stage metadata dict

        Returns:
            Extraction scope from stage config or wizard default
        """
        return stage.get("extraction_scope") or self._extraction_scope

    async def _react_stage_response(
        self,
        manager: Any,
        enhanced_prompt: str,
        stage: dict[str, Any],
        state: WizardState,
        tools: list[Any],
        metadata: dict[str, Any] | None = None,
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
            metadata: Optional metadata to persist on conversation nodes

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
                conversation_context = await self._context_builder.build(manager)
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
                metadata=metadata,
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
            metadata=metadata,
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
                "\nYou MUST focus on the goal above. Do NOT ask about topics "
                "from other stages. Only gather information that has NOT "
                "already been collected. Be conversational and helpful."
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

    def _apply_schema_defaults(
        self, wizard_state: WizardState, stage: dict[str, Any]
    ) -> set[str]:
        """Apply schema defaults to wizard data for unset properties.

        After extraction, defaults defined in the stage schema (e.g.
        ``"default": "medium"``) are applied to any property that was
        not explicitly set by the user.  This ensures template conditions
        like ``{% if difficulty %}`` evaluate True even when the user
        didn't mention a value.

        Only top-level properties are considered — nested object/array
        defaults are not auto-applied (they would require recursive
        merging that is unlikely to match user intent).

        Args:
            wizard_state: Current wizard state whose ``data`` may be
                updated in place.
            stage: Stage metadata dict containing ``schema``.

        Returns:
            Set of property names whose defaults were applied.
        """
        schema = stage.get("schema")
        if not schema:
            return set()

        applied: set[str] = set()
        for prop_name, prop_def in schema.get("properties", {}).items():
            if "default" not in prop_def:
                continue
            current = wizard_state.data.get(prop_name)
            if current is None:
                wizard_state.data[prop_name] = prop_def["default"]
                applied.add(prop_name)
                logger.debug(
                    "Applied schema default for '%s': %r",
                    prop_name,
                    prop_def["default"],
                )
        return applied

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

    def _apply_transition_derivations(
        self,
        stage: dict[str, Any],
        state: WizardState,
    ) -> None:
        """Apply data derivation rules from a stage's transition configs.

        Processes each transition's ``derive`` block before transition
        conditions are evaluated.  This lets transitions fill in values
        that enable their own conditions and downstream auto-advance.

        Derivation rules are key-value pairs where:
        - String values containing ``{{ }}`` are rendered as Jinja2
          templates with current wizard state data.
        - Other values (bool, int, etc.) are used as-is.

        Derived values are only set for keys not already present in
        ``state.data``, to avoid overwriting user-provided data.

        Args:
            stage: Current stage metadata (contains ``transitions`` list)
            state: Current wizard state (data is modified in-place)
        """
        transitions = stage.get("transitions", [])
        if not transitions:
            return

        import jinja2

        env = jinja2.Environment(undefined=jinja2.Undefined)
        collected_data = {
            k: v for k, v in state.data.items() if not k.startswith("_")
        }

        for transition in transitions:
            derive = transition.get("derive")
            if not derive or not isinstance(derive, dict):
                continue

            for key, value in derive.items():
                # Don't overwrite existing user-provided data
                if key in state.data:
                    continue

                if isinstance(value, str) and "{{" in value:
                    try:
                        resolved = env.from_string(value).render(**collected_data)
                        # Skip empty renders (template variable was undefined)
                        if resolved.strip():
                            state.data[key] = resolved.strip()
                            logger.debug(
                                "Derived %s = %r from transition to '%s'",
                                key,
                                resolved.strip(),
                                transition.get("target", "?"),
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to derive '%s' from template: %s", key, e
                        )
                else:
                    state.data[key] = value
                    logger.debug(
                        "Derived %s = %r (literal) from transition to '%s'",
                        key,
                        value,
                        transition.get("target", "?"),
                    )

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

    async def _generate_transform_error_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        error: str,
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

        Returns:
            LLM response explaining the error and offering retry
        """
        stage_name = stage.get("name", "unknown")
        error_context = f"""
## Processing Error

An error occurred while processing the transition from the "{stage_name}" stage:

**Error**: {error}

Please apologize for the issue and let the user know they can try again.
If the error suggests a configuration or system issue, suggest they contact support.
Be concise and helpful.
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
            stages=self._build_stages_roadmap(wizard_state),
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

        # Build stages roadmap from definitions if available
        stages: list[dict[str, str]] = []
        if stage_definitions:
            history_set = set(fsm_state.get("history", []))
            if isinstance(stage_definitions, dict):
                for name, meta in stage_definitions.items():
                    label = (
                        meta.get("label", name)
                        if isinstance(meta, dict)
                        else name
                    )
                    if name == current_stage:
                        status = "current"
                    elif name in history_set:
                        status = "completed"
                    else:
                        status = "pending"
                    stages.append(
                        {"name": name, "label": label, "status": status}
                    )
            elif isinstance(stage_definitions, list):
                for s in stage_definitions:
                    name = s.get("name", "")
                    label = s.get("label", name)
                    if name == current_stage:
                        status = "current"
                    elif name in history_set:
                        status = "completed"
                    else:
                        status = "pending"
                    stages.append(
                        {"name": name, "label": label, "status": status}
                    )

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
            stages=stages,
        )
