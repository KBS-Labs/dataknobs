"""Wizard reasoning strategy for guided conversational flows.

This module implements FSM-backed reasoning for DynaBot, enabling
guided conversational wizard flows with validation, data collection,
and branching logic.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataknobs_common.serialization import sanitize_for_json
from dataknobs_llm.conversations.storage import ConversationNode, get_node_by_id

from .base import ReasoningStrategy
from .observability import (
    TransitionRecord,
    WizardStateSnapshot,
    WizardTaskList,
    create_transition_record,
)
from .wizard_derivations import (
    DerivationRule,
    apply_field_derivations,
    parse_derivation_rules,
)
from .wizard_grounding import MergeFilter, SchemaGroundingFilter
from .wizard_hooks import WizardHooks

if TYPE_CHECKING:
    from dataknobs_data import SyncDatabase

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
    "_bank_fn",             # Per-step bank accessor (non-serializable callable)
})

# Extraction scope breadth ordering — used by scope escalation to
# determine whether the current scope is narrower than the target.
SCOPE_BREADTH: dict[str, int] = {
    "current_message": 0,
    "recent_messages": 1,
    "wizard_session": 2,
}


@dataclass
class TurnContext:
    """Per-turn values that travel with the FSM step but are never persisted.

    Delivered to transforms via :class:`TransformContext.turn` through the
    ``transform_context_factory`` mechanism.  Transforms can access per-turn
    values through this context rather than reading ``state.data`` directly.
    """

    message: str | None = None
    bank_fn: Any | None = None
    intent: str | None = None
    transform_error: str | None = None
    corpus: Any | None = None


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


def _load_merge_filter(dotted_path: str) -> MergeFilter:
    """Load a custom :class:`MergeFilter` from a dotted import path.

    Args:
        dotted_path: Fully qualified class name, e.g.
            ``"mypackage.filters.ConfigBotMergeFilter"``.

    Returns:
        Instantiated :class:`MergeFilter`.

    Raises:
        ConfigurationError: If the path is invalid, the class cannot
            be found, or the instance doesn't satisfy the protocol.
    """
    from dataknobs_common.exceptions import ConfigurationError

    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ConfigurationError(
            f"Invalid merge_filter path: {dotted_path!r} "
            "(expected 'module.ClassName')",
            context={"merge_filter": dotted_path},
        )
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConfigurationError(
            f"Cannot import merge filter module {module_path!r}: {exc}",
            context={"merge_filter": dotted_path},
        ) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ConfigurationError(
            f"Merge filter class {class_name!r} not found in {module_path!r}",
            context={"merge_filter": dotted_path},
        )
    instance = cls()
    # Runtime-checkable protocol validates method name presence
    # only, not full signature.  A class with a non-callable
    # ``should_merge`` attribute would pass this check.
    if not isinstance(instance, MergeFilter):
        raise ConfigurationError(
            f"Merge filter {dotted_path!r} does not implement "
            "the MergeFilter protocol",
            context={"merge_filter": dotted_path},
        )
    return instance


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
        skip_extraction: One-shot flag set by auto-advance; suppresses
            extraction on the landing stage's first generate() call.
            Persisted (not transient) because it must survive state
            serialization between the auto-advance turn and the user's
            next reply.
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
    skip_extraction: bool = False

    def __post_init__(self) -> None:
        """Seed history with current_stage when constructed with empty history.

        Back navigation requires the initial stage in history to function
        correctly.  This ensures ``WizardState(current_stage="x")`` always
        produces a usable state without requiring callers to duplicate the
        stage name into ``history``.
        """
        if not self.history:
            self.history = [self.current_stage]

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary suitable for ``json.dumps()``.
        """
        return {
            "current_stage": self.current_stage,
            "data": self.data,
            "history": self.history,
            "completed": self.completed,
            "clarification_attempts": self.clarification_attempts,
            "transitions": [t.to_dict() for t in self.transitions],
            "stage_entry_time": self.stage_entry_time,
            "tasks": self.tasks.to_dict(),
            "subflow_stack": [s.to_dict() for s in self.subflow_stack],
            "skip_extraction": self.skip_extraction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WizardState:
        """Restore from a dictionary produced by :meth:`to_dict`.

        Args:
            data: Dictionary containing wizard state fields.

        Returns:
            Reconstructed ``WizardState`` with proper nested types.
        """
        tasks_data = data.get("tasks")
        return cls(
            current_stage=data["current_stage"],
            data=data.get("data", {}),
            history=data.get("history", []),
            completed=data.get("completed", False),
            clarification_attempts=data.get("clarification_attempts", 0),
            transitions=[
                TransitionRecord.from_dict(t)
                for t in data.get("transitions", [])
            ],
            stage_entry_time=data.get("stage_entry_time", time.time()),
            tasks=(
                WizardTaskList.from_dict(tasks_data)
                if tasks_data
                else WizardTaskList()
            ),
            subflow_stack=[
                SubflowContext.from_dict(s)
                for s in data.get("subflow_stack", [])
            ],
            skip_extraction=data.get("skip_extraction", False),
        )


@dataclass
class WizardAdvanceResult:
    """Result from a non-conversational wizard advance.

    Returned by :meth:`WizardReasoning.advance` to give callers all
    the information they need to render their own UI and persist state.

    Attributes:
        state: Updated wizard state (caller should persist this).
        stage_name: Name of the current stage after advance.
        stage_prompt: Prompt text for the current stage.
        stage_schema: JSON Schema for the current stage (if any).
        suggestions: Quick-reply suggestions for the current stage.
        can_skip: Whether the current stage can be skipped.
        can_go_back: Whether back navigation is allowed.
        completed: Whether the wizard has reached its end state.
        transitioned: Whether a stage transition occurred.
        from_stage: Stage before the advance (None if no transition).
        auto_advance_messages: Rendered template strings from stages
            auto-advanced through during post-transition lifecycle.
            Empty when no auto-advance occurred.
        metadata: Full wizard metadata dict for UI rendering.
    """

    state: WizardState
    stage_name: str
    stage_prompt: str
    stage_schema: dict[str, Any] | None
    suggestions: list[str]
    can_skip: bool
    can_go_back: bool
    completed: bool
    transitioned: bool
    from_stage: str | None
    auto_advance_messages: list[str]
    metadata: dict[str, Any]


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
        default_store_trace: bool = False,
        default_verbose: bool = False,
        artifact_registry: Any | None = None,
        review_executor: Any | None = None,
        context_builder: Any | None = None,
        extraction_scope: str = "wizard_session",
        conflict_strategy: str = "latest_wins",
        log_conflicts: bool = True,
        extraction_grounding: bool = True,
        merge_filter: MergeFilter | None = None,
        grounding_overlap_threshold: float = 0.5,
        scope_escalation_enabled: bool = False,
        scope_escalation_scope: str = "wizard_session",
        recent_messages_count: int = 3,
        field_derivations: list[DerivationRule] | None = None,
        initial_data: dict[str, Any] | None = None,
        consistent_navigation_lifecycle: bool = True,
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
            default_store_trace: Default store_trace for ReAct stages. Stores
                reasoning trace in conversation metadata (default: False).
            default_verbose: Default verbose for ReAct stages. Enables
                debug-level logging for reasoning steps (default: False).
            artifact_registry: Optional ArtifactRegistry for artifact management.
            review_executor: Optional ReviewExecutor for running reviews.
            context_builder: Optional ContextBuilder for building conversation context.
            extraction_scope: Scope for data extraction. ``"wizard_session"``
                extracts from all user messages in the wizard session
                (default), ``"recent_messages"`` extracts from the last
                N user messages (controlled by ``recent_messages_count``),
                and ``"current_message"`` only extracts from the current
                message.
            conflict_strategy: Strategy for handling conflicting values when
                the same field is extracted from multiple messages. "latest_wins"
                (default) uses the most recent value.
            log_conflicts: Whether to log when field values conflict (default: True).
            extraction_grounding: Enable schema-driven grounding checks that
                verify extracted values against the user's message before
                allowing them to overwrite existing data.  When True (default),
                ungrounded extraction values cannot overwrite previously
                accumulated data --- only grounded values or first-time
                extractions (no existing value) are merged.
            merge_filter: Custom :class:`MergeFilter` implementation that
                replaces the built-in grounding check entirely.  When
                provided, ``extraction_grounding`` is ignored.
            grounding_overlap_threshold: Minimum word-overlap ratio for
                string grounding (0.0--1.0).  Defaults to 0.5.  Only used
                when the built-in grounding filter is active.
            scope_escalation_enabled: When True, automatically retry extraction
                with a broader scope when required fields are missing after
                the initial extraction.  Defaults to False (backward-compat).
            scope_escalation_scope: The scope to escalate to when
                ``scope_escalation_enabled`` is True.  One of
                ``"recent_messages"`` or ``"wizard_session"`` (default).
            recent_messages_count: Number of prior user messages to include
                when using the ``"recent_messages"`` extraction scope.
                Defaults to 3.  Configured under ``scope_escalation``
                in the wizard settings YAML.
            field_derivations: List of
                :class:`~dataknobs_bots.reasoning.wizard_derivations.DerivationRule`
                objects defining deterministic field-to-field transforms.
                When a source field is present but a target field is
                missing, the framework derives the target without an
                LLM call.  Configured under ``derivations`` in the
                wizard settings YAML.
            initial_data: Optional dict of data to inject into the wizard state
                when a new conversation starts. Useful for passing configuration
                values (e.g., quiz_bank_ids) from the bot config into the wizard
                data dict where transforms can access them.
            consistent_navigation_lifecycle: When True (default), back and skip
                navigation fire the same lifecycle hooks (enter, complete) and
                run auto-advance/subflow-pop as forward transitions do.  Set to
                False to restore the original behavior where back/skip only
                performed the FSM operation without lifecycle hooks.
        """
        super().__init__()
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
        self._default_store_trace = default_store_trace
        self._default_verbose = default_verbose
        self._artifact_registry = artifact_registry
        self._review_executor = review_executor
        self._context_builder = context_builder
        self._extraction_scope = extraction_scope
        self._conflict_strategy = conflict_strategy
        self._log_conflicts = log_conflicts
        self._extraction_grounding = extraction_grounding
        self._grounding_overlap_threshold = grounding_overlap_threshold
        # Build merge filter: custom > built-in grounding > none
        if merge_filter is not None:
            self._merge_filter: MergeFilter | None = merge_filter
        elif extraction_grounding:
            self._merge_filter = SchemaGroundingFilter(
                overlap_threshold=grounding_overlap_threshold,
            )
        else:
            self._merge_filter = None
        self._scope_escalation_enabled = scope_escalation_enabled
        self._scope_escalation_scope = scope_escalation_scope
        self._recent_messages_count = recent_messages_count
        self._field_derivations = field_derivations or []
        self._initial_data: dict[str, Any] = initial_data or {}
        self._consistent_navigation_lifecycle = consistent_navigation_lifecycle
        # Active subflow FSM (None when in main flow)
        self._active_subflow_fsm: WizardFSM | None = None
        # LLM provider set by generate() for transform context access
        self._current_llm: Any = None
        # Completion signal bridge: set by CompleteWizardTool in
        # _react_stage_response, checked by generate() after response.
        self._tool_completion_requested: bool = False
        self._tool_completion_summary: str = ""
        # Restart signal bridge: set by RestartWizardTool, checked
        # by generate() after response to delegate to _execute_restart.
        self._tool_restart_requested: bool = False

        # Per-turn context delivered to transforms via the context factory.
        # Set at the start of each FSM step, cleared after.
        self._current_turn: TurnContext | None = None

        # Per-turn keys: cleared at the start of each turn, suppressed
        # in conflict detection, and treated as ephemeral (non-persistent).
        self._per_turn_keys: frozenset[str] = frozenset(
            wizard_fsm.settings.get("per_turn_keys", [])
        )

        # Merge framework-level ephemeral keys with config-declared ones
        # and per-turn keys (which are implicitly ephemeral).
        config_ephemeral = wizard_fsm.settings.get("ephemeral_keys", [])
        self._ephemeral_keys: frozenset[str] = (
            DEFAULT_EPHEMERAL_KEYS
            | frozenset(config_ephemeral)
            | self._per_turn_keys
        )

        # Track last wizard state for cleanup in close()
        self._last_wizard_state: WizardState | None = None

        # Initialise MemoryBank instances from wizard-level ``banks`` config
        self._bank_configs: dict[str, dict[str, Any]] = dict(
            wizard_fsm.settings.get("banks", {})
        )
        self._banks: dict[str, Any] = {}
        self._init_banks()

        # Initialise ArtifactBank if ``artifact`` config is present.
        # When artifact is configured, its sections ARE the banks —
        # ``self._banks`` references the same MemoryBank instances.
        self._artifact: Any = None
        self._catalog: Any = None
        artifact_config = wizard_fsm.settings.get("artifact")
        if artifact_config:
            self._init_artifact(artifact_config)

        # Build navigation keyword config from wizard-level settings
        nav_settings = wizard_fsm.settings.get("navigation", {})
        self._navigation_config: NavigationConfig = NavigationConfig.from_dict(
            nav_settings or {}
        )

        # Store the factory — it will be set on the ExecutionContext
        # before FSM steps execute (see WizardFSM._ensure_context_factory).
        self._wizard_fsm = wizard_fsm
        self._wizard_fsm.set_transform_context_factory(
            self._build_transform_context
        )

    def _build_transform_context(self, func_context: Any) -> Any:
        """Build a :class:`TransformContext` from an FSM ``FunctionContext``.

        Registered as the ``transform_context_factory`` on the
        :class:`ExecutionContext`.  The FSM calls this instead of passing
        the raw ``FunctionContext`` to transforms, giving them access to
        the artifact registry, rubric executor, LLM, and memory banks.

        Args:
            func_context: The :class:`FunctionContext` built by the FSM.

        Returns:
            ``TransformContext`` with wizard services and FSM context.
        """
        from ..artifacts.transforms import TransformContext

        return TransformContext(
            fsm_context=func_context,
            turn=self._current_turn,
            artifact_registry=self._artifact_registry,
            rubric_executor=self._review_executor,
            config={"llm": self._current_llm} if self._current_llm else {},
            banks=self._banks,
        )

    # -----------------------------------------------------------------
    # MemoryBank management
    # -----------------------------------------------------------------

    def _create_bank_db(
        self, bank_name: str, cfg: dict[str, Any]
    ) -> tuple[SyncDatabase, str]:
        """Create database backend and storage mode for a memory bank.

        Args:
            bank_name: Bank identifier (used as default table name).
            cfg: Per-bank configuration dict from wizard settings.

        Returns:
            Tuple of ``(database, storage_mode)``.
        """
        backend = cfg.get("backend", "memory")
        if backend == "memory":
            from dataknobs_data.backends.memory import SyncMemoryDatabase

            return SyncMemoryDatabase(), "inline"
        from dataknobs_data import database_factory

        backend_config = dict(cfg.get("backend_config", {}))
        backend_config["backend"] = backend
        backend_config.setdefault("table", bank_name)
        db = database_factory.create(**backend_config)
        db.connect()
        return db, "external"

    def _init_banks(self) -> None:
        """Create ``MemoryBank`` instances from wizard ``banks`` config."""
        if not self._bank_configs:
            return
        from ..memory.bank import MemoryBank

        for name, cfg in self._bank_configs.items():
            # Support both flat keys (duplicate_strategy, match_fields)
            # and nested duplicate_detection.{strategy, match_fields}.
            dup_cfg = cfg.get("duplicate_detection", {})
            dup_strategy = (
                cfg.get("duplicate_strategy")
                or dup_cfg.get("strategy", "allow")
            )
            match_fields = (
                cfg.get("match_fields")
                or dup_cfg.get("match_fields")
            )
            db, storage_mode = self._create_bank_db(name, cfg)
            self._banks[name] = MemoryBank(
                name=name,
                schema=cfg.get("schema", {}),
                db=db,
                max_records=cfg.get("max_records"),
                duplicate_strategy=dup_strategy,
                match_fields=match_fields,
                storage_mode=storage_mode,
            )
        logger.debug("Initialised %d memory banks: %s",
                      len(self._banks), list(self._banks))

    def _restore_banks(self, banks_data: dict[str, Any]) -> None:
        """Restore ``MemoryBank`` instances from persisted data.

        Banks that exist in the persisted data are deserialized.  Banks
        declared in config but not yet persisted are freshly initialised.

        For persistent backends (non-memory), the database is reconnected
        via ``_create_bank_db`` so records already in the backend are
        accessible without re-insertion.
        """
        from ..memory.bank import MemoryBank

        for name, bank_dict in banks_data.items():
            cfg = self._bank_configs.get(name, {})
            backend = cfg.get("backend", "memory")
            if backend != "memory":
                db, _mode = self._create_bank_db(name, cfg)
                self._banks[name] = MemoryBank.from_dict(bank_dict, db=db)
            else:
                self._banks[name] = MemoryBank.from_dict(bank_dict)
        # Ensure any newly-configured banks that weren't persisted yet
        # are also initialised.
        for name in self._bank_configs:
            if name not in self._banks:
                self._init_banks()
                break
        if banks_data:
            logger.debug(
                "Restored %d memory banks from persisted data",
                len(banks_data),
            )

    def _make_bank_accessor(self) -> Any:
        """Return a callable ``bank(name) -> MemoryBank | EmptyBankProxy``."""
        from ..memory.bank import EmptyBankProxy

        banks = self._banks

        def _bank(name: str) -> Any:
            return banks.get(name, EmptyBankProxy(name))

        return _bank

    def _init_artifact(self, artifact_config: dict[str, Any]) -> None:
        """Create an ``ArtifactBank`` from wizard ``artifact`` config.

        When an artifact is configured, its sections replace the banks —
        ``self._banks`` and ``self._bank_configs`` are populated from
        the artifact's sections.

        If the config contains a ``catalog`` key, an
        ``ArtifactBankCatalog`` is also created and stored on
        ``self._catalog``.

        Args:
            artifact_config: Artifact configuration dict with ``name``,
                ``fields``, and ``sections`` keys.  Optional ``catalog``
                sub-dict for catalog backend configuration.

        Raises:
            ConfigurationError: If both ``banks`` and ``artifact`` are
                configured.
        """
        from dataknobs_common.exceptions import ConfigurationError

        from ..memory.artifact_bank import ArtifactBank

        if self._bank_configs:
            raise ConfigurationError(
                "Cannot configure both 'banks' and 'artifact'. "
                "Use artifact.sections instead of banks.",
                context={"setting": "artifact"},
            )
        self._artifact = ArtifactBank.from_config(
            artifact_config, db_factory=self._create_bank_db
        )
        self._banks = dict(self._artifact.sections)
        self._bank_configs = dict(artifact_config.get("sections", {}))

        # Optionally create a catalog for storing/loading artifacts.
        catalog_config = artifact_config.get("catalog")
        if catalog_config:
            from ..memory.catalog import ArtifactBankCatalog

            self._catalog = ArtifactBankCatalog.from_config({
                **catalog_config,
                "artifact_config": artifact_config,
            })

        # Optionally seed the artifact from a file.
        seed_config = artifact_config.get("seed")
        if seed_config:
            self._seed_artifact(seed_config)

    def _seed_artifact(self, seed_config: dict[str, Any]) -> None:
        """Populate the artifact from a seed file.

        Loads a JSON or JSONL file and populates the existing artifact
        using ``populate_from_compiled()``.  Format is auto-detected
        from the file extension unless explicitly specified.

        For JSONL books, ``select`` matches against ``_artifact_name``
        first, then against any top-level string field value (e.g. a
        recipe name).  Without ``select``, the first entry is used.

        Failures are logged as warnings and do not prevent the wizard
        from starting with an empty artifact.

        Args:
            seed_config: Seed configuration with ``source`` (file path),
                optional ``format`` (``"json"`` or ``"jsonl"``), and
                optional ``select`` (name to match in JSONL book).
        """
        import json
        from pathlib import Path

        source = seed_config.get("source")
        if not source:
            logger.warning("Seed config missing 'source', skipping seed")
            return

        source_path = Path(source)
        if not source_path.exists():
            logger.warning(
                "Seed file not found: %s, proceeding with empty artifact",
                source_path,
            )
            return

        # Determine format: explicit config or auto-detect from extension
        fmt = seed_config.get("format")
        if not fmt:
            fmt = "jsonl" if source_path.suffix.lower() == ".jsonl" else "json"

        try:
            if fmt == "jsonl":
                data = self._load_jsonl_entry(
                    source_path, seed_config.get("select"),
                )
            else:
                text = source_path.read_text(encoding="utf-8")
                data = json.loads(text)
                if not isinstance(data, dict):
                    logger.warning(
                        "Seed file %s does not contain a JSON object",
                        source_path,
                    )
                    return

            # Full-state format (from to_dict()) has nested structure;
            # convert to compiled format for populate_from_compiled().
            if "name" in data and "sections" in data and "_artifact_name" not in data:
                from ..memory.artifact_bank import ArtifactBank

                temp = ArtifactBank.from_dict(data)
                data = temp.compile()

            self._artifact.populate_from_compiled(
                data, source_stage="seed",
            )
            logger.info(
                "Seeded artifact '%s' from %s",
                self._artifact.name,
                source_path,
            )
        except Exception:
            logger.warning(
                "Failed to seed artifact from %s, proceeding with empty artifact",
                source_path,
                exc_info=True,
            )

    @staticmethod
    def _load_jsonl_entry(
        path: Path,
        select: str | None,
    ) -> dict[str, Any]:
        """Read a single entry from a JSONL book file.

        Args:
            path: JSONL file path.
            select: Optional name to match.  Checked against
                ``_artifact_name`` first, then any top-level string
                field value.  If ``None``, the first entry is returned.

        Returns:
            Parsed JSON dict for the matched entry.

        Raises:
            ValueError: If no entries or no match found.
        """
        import json

        entries: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    obj = json.loads(stripped)
                    if isinstance(obj, dict):
                        entries.append(obj)

        if not entries:
            raise ValueError(f"No entries in JSONL book: {path}")

        if select is None:
            return entries[0]

        # Match on _artifact_name first
        for entry in entries:
            if entry.get("_artifact_name") == select:
                return entry

        # Match on any top-level string field value
        for entry in entries:
            for key, value in entry.items():
                if not key.startswith("_") and isinstance(value, str) and value == select:
                    return entry

        available = [
            entry.get("_artifact_name", "<unnamed>") for entry in entries
        ]
        raise ValueError(
            f"No entry matching '{select}' in {path}. "
            f"Available artifact names: {available}"
        )

    def _sync_artifact_fields(self, state: WizardState) -> None:
        """Sync wizard state data into artifact fields.

        Called before state serialization so that artifact fields
        auto-populate from matching wizard ``state.data`` keys.

        Args:
            state: Current wizard state.
        """
        if not self._artifact or self._artifact.is_finalized:
            return
        for field_name in self._artifact.field_defs:
            value = state.data.get(field_name)
            if value is not None and value != self._artifact.field(field_name):
                self._artifact.set_field(field_name, value)

    def _reverse_sync_artifact_to_state(self, state: WizardState) -> None:
        """Sync artifact field values back into wizard state data.

        Called before the forward sync (``_sync_artifact_fields``) so that
        tool-driven changes to the artifact (e.g. ``LoadFromCatalogTool``
        replacing all fields) are reflected in ``state.data``.

        Args:
            state: Current wizard state.
        """
        if not self._artifact:
            return
        for field_name in self._artifact.field_defs:
            artifact_value = self._artifact.field(field_name)
            if artifact_value is not None:
                state.data[field_name] = artifact_value

    # -----------------------------------------------------------------
    # Collection mode helpers
    # -----------------------------------------------------------------

    async def _handle_collection_mode(
        self,
        user_message: str,
        extracted_data: dict[str, Any],
        stage: dict[str, Any],
        state: WizardState,
        manager: Any,
        llm: Any,
        tools: Any,
    ) -> Any | None:
        """Handle a collection-mode stage after extraction.

        If the user signalled "done", sets ``_collection_done`` in state
        data and returns ``None`` to let normal transition evaluation
        proceed.  Otherwise, adds extracted data to the target bank,
        clears schema fields for the next record, renders the stage
        response, saves state, and returns the response.

        Returns:
            A response object if the stage should loop, or ``None``
            if the collection is done and transition evaluation should
            continue.
        """
        col_config = stage.get("collection_config", {})
        bank_name = col_config.get("bank_name", "")
        done_keywords = col_config.get("done_keywords", [])

        # Check for done signal
        if self._is_done_signal(user_message, done_keywords):
            state.data["_collection_done"] = True
            logger.debug(
                "Collection done signal at stage '%s'",
                stage.get("name"),
            )
            return None  # Fall through to transition evaluation

        # Add extracted data to the bank
        bank_accessor = self._make_bank_accessor()
        bank = bank_accessor(bank_name)

        # Filter to only schema-defined fields
        schema_props = set(
            stage.get("schema", {}).get("properties", {}).keys()
        )
        record_data = {
            k: v for k, v in extracted_data.items()
            if k in schema_props and v is not None
        }

        if record_data:
            try:
                bank.add(record_data, source_stage=stage.get("name", ""))
                logger.debug(
                    "Added record to bank '%s' (count=%d)",
                    bank_name,
                    bank.count(),
                )
            except ValueError as e:
                logger.warning(
                    "Failed to add record to bank '%s': %s",
                    bank_name,
                    e,
                )

        # Clear schema fields from state.data so the next extraction
        # starts fresh (don't keep the previous record's values).
        for field_name in schema_props:
            state.data.pop(field_name, None)

        # Branch the conversation tree so this iteration becomes a sibling
        # of the previous collection response, not a child.  Without this,
        # each loop iteration deepens the tree (0 → 0.0 → 0.0.0 …) instead
        # of branching (0, 0.1, 0.2 …).
        await self._branch_for_revisited_stage(
            manager, stage.get("name", ""),
        )

        # Render the stage response
        response = await self._generate_stage_response(
            manager, llm, stage, state, tools,
        )
        await self._save_wizard_state(manager, state)
        return response

    @staticmethod
    def _is_done_signal(message: str, done_keywords: list[str]) -> bool:
        """Check whether a user message matches a collection done keyword."""
        if not done_keywords:
            return False
        normalised = message.strip().lower()
        return any(
            normalised == kw.strip().lower() for kw in done_keywords
        )

    @staticmethod
    def _classify_collection_intent(
        message: str, stage: dict[str, Any],
    ) -> str:
        """Classify user intent during a collection-mode stage.

        Runs rule-based checks to distinguish help requests from data
        input **before** extraction.  Navigation and done signals are
        handled upstream (``_handle_navigation`` and
        ``_is_done_signal``), so this method only discriminates between:

        - ``"help"`` — the user is asking a question about what to
          provide, not providing data.
        - ``"data_input"`` — default; proceed to extraction.

        Custom help keywords can be supplied per-stage via
        ``collection_config.help_keywords``.

        Args:
            message: Raw user message text.
            stage: Current stage metadata dict.

        Returns:
            Intent string: ``"help"`` or ``"data_input"``.
        """
        msg = message.strip().lower()

        # Stage-configurable help keywords (exact match)
        col_config = stage.get("collection_config") or {}
        help_keywords = col_config.get("help_keywords", [])
        if help_keywords and any(msg == kw.strip().lower() for kw in help_keywords):
            return "help"

        # Built-in heuristic: question marks or common help phrasing
        if msg.endswith("?"):
            return "help"

        help_starters = (
            "what should i",
            "what do i",
            "what do you need",
            "what goes here",
            "help",
            "explain",
            "i don't understand",
            "i don't know what",
            "what kind of",
            "what format",
        )
        if any(msg.startswith(s) for s in help_starters):
            return "help"

        return "data_input"

    def providers(self) -> dict[str, Any]:
        """Return the extraction provider if an extractor is configured.

        Duck-typed extractors (e.g. ``ConfigurableExtractor``) that lack
        a ``.provider`` attribute are skipped — they don't wrap an LLM
        provider that needs lifecycle management.
        """
        from dataknobs_bots.bot.base import PROVIDER_ROLE_EXTRACTION

        if self._extractor is not None and hasattr(self._extractor, "provider"):
            return {PROVIDER_ROLE_EXTRACTION: self._extractor.provider}
        return {}

    @property
    def extractor(self) -> Any | None:
        """The current extractor instance, or None if not configured."""
        return self._extractor

    def set_extractor(self, extractor: Any) -> None:
        """Replace the extractor instance.

        Use this to inject a ``ConfigurableExtractor`` or any other
        duck-typed extractor that implements ``async extract(text,
        schema, context, model)``.  Unlike ``set_provider()`` (which
        swaps the LLM provider *inside* an existing ``SchemaExtractor``),
        this replaces the extractor object entirely.

        Args:
            extractor: New extractor instance.
        """
        self._extractor = extractor

    def set_provider(self, role: str, provider: Any) -> bool:
        """Replace the extraction provider if the role matches."""
        from dataknobs_bots.bot.base import PROVIDER_ROLE_EXTRACTION

        if role == PROVIDER_ROLE_EXTRACTION and self._extractor is not None:
            self._extractor.provider = provider
            self._extractor._owns_provider = False
            return True
        return False

    async def close(self) -> None:
        """Close the reasoning strategy and release resources.

        Cancels any in-flight asyncio tasks stored in ephemeral keys,
        closes the SchemaExtractor's LLM provider if present (releasing
        HTTP connections), and closes all memory bank database connections.
        Should be called when the reasoning strategy is no longer needed
        (typically via DynaBot.close()).

        Note:
            Wizard state is per-conversation (stored in manager.metadata).
            This method cancels tasks accessible via the last-used manager's
            state. If multiple conversations are active simultaneously,
            only the tasks from the most recently accessed state are
            cancelled here.
        """
        # Cancel asyncio tasks stored in ephemeral wizard state keys
        if hasattr(self, "_last_wizard_state") and self._last_wizard_state:
            cancelled = 0
            for key in self._ephemeral_keys:
                val = self._last_wizard_state.data.get(key)
                if isinstance(val, asyncio.Task) and not val.done():
                    val.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await val
                    cancelled += 1
            if cancelled:
                logger.debug("Cancelled %d ephemeral task(s)", cancelled)

        # Close extractor's LLM provider
        if self._extractor is not None and hasattr(self._extractor, "close"):
            await self._extractor.close()
            logger.debug("Closed WizardReasoning extractor")

        # Close memory bank database connections
        for _name, bank in self._banks.items():
            bank.close()
        if self._banks:
            logger.debug(
                "Closed %d memory bank database(s)", len(self._banks)
            )

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
                - config_base_path: Optional base directory for resolving
                  relative ``wizard_config`` paths. When set, relative
                  paths are resolved against this directory instead of CWD.
                - extraction_config: Optional extraction configuration
                - strict_validation: Whether to enforce validation
                - hooks: Optional hooks configuration dict
                - artifacts: Optional artifact configuration with definitions
                - review_protocols: Optional review protocol definitions
                - consistent_navigation_lifecycle: Whether back/skip fire
                  the same lifecycle hooks as forward transitions
                  (default: True)

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

        # Resolve relative wizard_config paths against config_base_path
        config_base_path_str = config.get("config_base_path")
        config_base_path = Path(config_base_path_str) if config_base_path_str else None

        # Load wizard FSM — supports both file paths and inline dicts
        loader = WizardConfigLoader()
        custom_fns = config.get("custom_functions", {})

        if isinstance(wizard_config_value, dict):
            wizard_fsm = loader.load_from_dict(
                wizard_config_value,
                custom_fns,
                config_base_path=config_base_path,
            )
        else:
            config_path = Path(wizard_config_value)
            if not config_path.is_absolute() and config_base_path:
                config_path = config_base_path / config_path
            wizard_fsm = loader.load(str(config_path), custom_fns)

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
        extraction_grounding = wizard_fsm.settings.get("extraction_grounding", True)
        grounding_overlap_threshold = wizard_fsm.settings.get(
            "grounding_overlap_threshold", 0.5,
        )

        # Load custom merge filter if specified
        merge_filter: MergeFilter | None = None
        merge_filter_path = wizard_fsm.settings.get("merge_filter")
        if merge_filter_path:
            merge_filter = _load_merge_filter(merge_filter_path)

        # Load scope escalation settings
        scope_escalation_config = wizard_fsm.settings.get("scope_escalation", {})
        scope_escalation_enabled = scope_escalation_config.get("enabled", False)
        scope_escalation_scope = scope_escalation_config.get(
            "escalation_scope", "wizard_session",
        )
        if scope_escalation_scope not in SCOPE_BREADTH:
            logger.warning(
                "Unknown scope_escalation_scope %r — must be one of %s. "
                "Defaulting to 'wizard_session'.",
                scope_escalation_scope,
                list(SCOPE_BREADTH),
            )
            scope_escalation_scope = "wizard_session"
        elif SCOPE_BREADTH[scope_escalation_scope] == 0:
            logger.warning(
                "scope_escalation_scope %r is the narrowest scope and "
                "cannot be an escalation target. "
                "Defaulting to 'wizard_session'.",
                scope_escalation_scope,
            )
            scope_escalation_scope = "wizard_session"
        recent_messages_count = scope_escalation_config.get(
            "recent_messages_count", 3,
        )

        # Load field derivation rules
        derivation_config = wizard_fsm.settings.get("derivations", [])
        field_derivations: list[DerivationRule] = []
        if derivation_config and isinstance(derivation_config, list):
            field_derivations = parse_derivation_rules(derivation_config)
            if field_derivations:
                logger.info(
                    "Loaded %d field derivation rules",
                    len(field_derivations),
                )

        store_trace = wizard_fsm.settings.get("store_trace", False)
        verbose = wizard_fsm.settings.get("verbose", False)

        # Create artifact registry if artifact definitions configured
        artifact_registry = None
        artifacts_config = config.get("artifacts", {})
        if artifacts_config:
            try:
                from dataknobs_data.backends.memory import AsyncMemoryDatabase

                from ..artifacts import ArtifactRegistry, ArtifactTypeDefinition

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
            extraction_grounding=extraction_grounding,
            merge_filter=merge_filter,
            grounding_overlap_threshold=grounding_overlap_threshold,
            scope_escalation_enabled=scope_escalation_enabled,
            scope_escalation_scope=scope_escalation_scope,
            recent_messages_count=recent_messages_count,
            field_derivations=field_derivations,
            default_store_trace=store_trace,
            default_verbose=verbose,
            initial_data=config.get("initial_data"),
            consistent_navigation_lifecycle=config.get(
                "consistent_navigation_lifecycle", True
            ),
        )

    async def greet(
        self,
        manager: Any,
        llm: Any,
        *,
        initial_context: dict[str, Any] | None = None,
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
            initial_context: Optional dict of initial data to seed into
                ``wizard_state.data`` before generating the greeting.
                These values are available to the start stage's prompt
                template and transforms.
            **kwargs: Additional generation parameters

        Returns:
            LLM response object with wizard metadata
        """
        # Store LLM reference so transform context wrappers can access it
        self._current_llm = llm

        # Initialize fresh wizard state (restarts FSM to start stage)
        wizard_state = self._get_wizard_state(manager)

        # Merge initial context into wizard state data
        if initial_context:
            wizard_state.data.update(initial_context)

        logger.info(
            "Wizard greet: stage='%s', history=%s",
            wizard_state.current_stage,
            wizard_state.history,
        )

        # Get start stage metadata and generate the response
        active_fsm = self._get_active_fsm()
        stage = active_fsm.current_metadata
        await self._branch_for_revisited_stage(manager, stage.get("name", ""))
        response = await self._generate_stage_response(
            manager, llm, stage, wizard_state, tools=[],
        )

        # Record that the start stage template has been rendered so that
        # generate() does not re-render it as a "first confirmation" when
        # the user's first message arrives with extracted data.
        if stage.get("response_template"):
            wizard_state.increment_render_count(stage.get("name", "unknown"))

        # Auto-advance through message stages if the start stage has
        # auto_advance: true. The start stage response is already captured,
        # so skip_first_render=True avoids re-rendering it.
        if self._can_auto_advance(wizard_state, stage):
            auto_advance_messages = [response.content]
            loop_messages = await self._run_auto_advance_loop(
                wizard_state, active_fsm, stage, skip_first_render=True,
            )
            auto_advance_messages.extend(loop_messages)

            # Re-fetch active_fsm in case a subflow pop occurred
            active_fsm = self._get_active_fsm()

            # Generate the landing stage's response and prepend
            # the collected messages from auto-advanced stages.
            landing_stage = active_fsm.current_metadata
            if landing_stage.get("name") != stage.get("name"):
                await self._branch_for_revisited_stage(
                    manager, landing_stage.get("name", "")
                )
                response = await self._generate_stage_response(
                    manager, llm, landing_stage, wizard_state, tools=[],
                )
                if landing_stage.get("response_template"):
                    wizard_state.increment_render_count(
                        landing_stage.get("name", "unknown")
                    )
                self._prepend_messages_to_response(
                    response, auto_advance_messages
                )

            # Clear skip_extraction — the user's first message after greet()
            # IS directed at the landing stage (unlike generate() auto-advance
            # where the user's message was directed at the previous stage).
            wizard_state.skip_extraction = False

        # Persist wizard state
        await self._save_wizard_state(manager, wizard_state)

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

        # Clear per-turn keys from previous turn to prevent stale values
        for key in self._per_turn_keys:
            wizard_state.data.pop(key, None)
            wizard_state.transient.pop(key, None)

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
                await self._branch_for_revisited_stage(
                    manager, target_stage
                )
                response = await self._generate_stage_response(
                    manager, llm, stage, wizard_state, tools
                )
                await self._save_wizard_state(manager, wizard_state)
                return response

        # Auto-restart when the wizard can't meaningfully continue:
        #
        # 1. Wizard completed (via complete_wizard tool) with amendments
        #    disabled — the workflow is done, new message = fresh start.
        # 2. Artifact finalized (via finalize_artifact tool) but wizard
        #    NOT completed — the LLM forgot to call complete_wizard or
        #    restart_wizard.  The artifact is locked so no further edits
        #    are possible; the wizard is stuck.
        #
        # In both cases, clear state/banks/artifact and fall through so
        # the user's message is processed by the fresh first stage
        # (extraction, transitions, etc.) rather than the stale ReAct
        # loop on the old stage.
        _should_auto_restart = (
            (wizard_state.completed and not self._allow_amendments)
            or (
                not wizard_state.completed
                and self._artifact is not None
                and self._artifact.is_finalized
            )
        )

        if _should_auto_restart:
            logger.info(
                "Auto-restarting wizard (completed=%s, "
                "artifact_finalized=%s, amendments=%s)",
                wizard_state.completed,
                self._artifact.is_finalized if self._artifact else False,
                self._allow_amendments,
            )
            await self._restart_cleanup(
                wizard_state, user_message, trigger="auto_restart"
            )
            # Branch the conversation tree so the new recipe's context
            # starts fresh — the LLM won't see the old recipe's detailed
            # tool calls and record-level operations.
            await self._branch_for_revisited_stage(
                manager, wizard_state.current_stage
            )

        # Handle navigation commands
        nav_result = await self._handle_navigation(
            user_message, wizard_state, manager, llm
        )
        if nav_result:
            await self._save_wizard_state(manager, wizard_state)
            return nav_result

        # Get current stage context from active FSM (subflow or main)
        active_fsm = self._get_active_fsm()
        stage = active_fsm.current_metadata
        is_conversation = stage.get("mode") == "conversation"

        # ── Skip extraction on auto-advance landing stage ──
        # When auto-advance lands on a new stage, the user hasn't responded
        # to it yet.  Their message was directed at the previous stage.
        # Consume the flag (one-shot) so extraction runs normally next turn.
        _skip_extraction = wizard_state.skip_extraction
        if _skip_extraction:
            wizard_state.skip_extraction = False
            logger.debug(
                "Skipping extraction at stage '%s' — landed via auto-advance, "
                "waiting for user's first response to this stage",
                stage.get("name"),
            )

        # ── Collection-mode done signal (before extraction) ──
        # When a collection-mode stage receives a done keyword, skip
        # extraction entirely.  The keyword (e.g. "done") is not a data
        # record and would fail the extraction confidence check, causing
        # a clarification loop that never reaches _handle_collection_mode.
        _collection_done_signal = False
        if (
            not _skip_extraction
            and stage.get("collection_mode") == "collection"
            and not is_conversation
        ):
            col_config = stage.get("collection_config", {})
            done_keywords = col_config.get("done_keywords", [])
            if self._is_done_signal(user_message, done_keywords):
                wizard_state.data["_collection_done"] = True
                _collection_done_signal = True
                logger.debug(
                    "Collection done signal (pre-extraction) at stage '%s'",
                    stage.get("name"),
                )

        # ── Collection-mode help intent (before extraction) ──
        # When a collection-mode stage receives a help request, skip
        # extraction and generate a contextual help response.
        _collection_help = False
        if (
            not _skip_extraction
            and stage.get("collection_mode") == "collection"
            and not is_conversation
            and not _collection_done_signal
        ):
            collection_intent = self._classify_collection_intent(
                user_message, stage,
            )
            if collection_intent == "help":
                _collection_help = True
                logger.debug(
                    "Collection help intent at stage '%s'",
                    stage.get("name"),
                )

        if _skip_extraction:
            # Auto-advance landing stage — skip extraction and fall through
            # to transition evaluation / response generation below.
            # Field derivations are also skipped here — they already ran
            # on the prior turn when extraction executed.
            pass
        elif is_conversation:
            # Conversation mode: skip extraction, run intent detection
            stage_name = stage.get("name", "unknown")
            logger.debug(
                "Conversation mode for stage '%s': skipping extraction",
                stage_name,
            )
            await self._detect_intent(user_message, stage, wizard_state, llm)
        elif _collection_done_signal:
            # Done keyword detected — skip extraction and fall through
            # to transition evaluation below.
            pass
        elif _collection_help:
            # Help request during collection — generate a contextual
            # response without extraction, then return immediately so
            # the stage loops for the next data input.
            stage_context = self._build_stage_context(stage, wizard_state)
            enhanced = f"{manager.system_prompt}\n\n{stage_context}"
            help_context = (
                "\n\n## User Needs Help\n"
                "The user is asking for guidance about what to provide. "
                "Answer their question helpfully and concisely, then "
                "invite them to provide the next item."
            )
            wizard_snapshot = {"wizard": self._build_wizard_metadata(wizard_state)}
            response = await manager.complete(
                system_prompt_override=enhanced + help_context,
                tools=self._filter_tools_for_stage(stage, tools),
                metadata=wizard_snapshot,
            )
            self._add_wizard_metadata(response, wizard_state, stage)
            await self._save_wizard_state(manager, wizard_state)
            return response
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
                # ── Always normalize and merge extracted data ──
                # Partial data must accumulate across turns even when
                # extraction confidence is low.  Without this, multi-turn
                # gather stages enter a dead loop where wizard_state.data
                # stays {} and can_satisfy never becomes True.
                schema = stage.get("schema")
                if schema and extraction.data:
                    extraction.data = self._normalize_extracted_data(
                        extraction.data, schema
                    )

                # Merge extracted data into wizard_state.data (in-place),
                # tracking which keys are new or changed so we can
                # decide whether to show a confirmation template before
                # allowing a transition.
                # Note: user_message is always the *last* user message.
                # With wizard_session extraction scope, the extractor
                # sees the full history, but the grounding check
                # intentionally grounds against only the current
                # message.  First-write (existing_value is None) is
                # always allowed through, which compensates for
                # session-scope re-extraction of prior turns.
                new_data_keys = self._merge_extraction_result(
                    extraction.data, wizard_state, stage, user_message,
                )

                # ── Scope escalation ──
                # When required fields are still missing after the
                # initial merge and the current scope is narrower than
                # the escalation target, retry extraction with a
                # broader scope so that information from earlier turns
                # can fill the gaps.  Only escalate when there IS
                # prior conversation history — on the first turn,
                # the broader scope would see the same content and
                # waste an LLM call.
                if self._scope_escalation_enabled:
                    effective_scope = self._get_extraction_scope(stage)
                    target_breadth = SCOPE_BREADTH.get(
                        self._scope_escalation_scope,
                        SCOPE_BREADTH["wizard_session"],
                    )
                    current_breadth = SCOPE_BREADTH.get(
                        effective_scope, 0,
                    )
                    if current_breadth < target_breadth:
                        missing = self._check_required_fields_missing(
                            wizard_state, stage,
                        )
                        # Check for prior history using the same
                        # scope window the escalated extraction will
                        # use.  Without earlier turns visible in that
                        # window, escalation would re-extract the
                        # same content at higher cost for no benefit.
                        guard_max = (
                            self._recent_messages_count
                            if self._scope_escalation_scope
                            == "recent_messages"
                            else None
                        )
                        has_prior = bool(
                            self._build_wizard_context(
                                manager, wizard_state,
                                max_messages=guard_max,
                            )
                        ) if missing and manager is not None else False
                        if missing and has_prior:
                            logger.debug(
                                "Scope escalation: %d required fields "
                                "missing after '%s' extraction: %s "
                                "— retrying with '%s' scope",
                                len(missing),
                                effective_scope,
                                sorted(missing),
                                self._scope_escalation_scope,
                            )
                            escalated_stage = {
                                **stage,
                                "extraction_scope": (
                                    self._scope_escalation_scope
                                ),
                            }
                            escalated = await self._extract_data(
                                user_message,
                                escalated_stage,
                                llm,
                                manager,
                                wizard_state,
                            )
                            if escalated.data:
                                if schema:
                                    escalated.data = (
                                        self._normalize_extracted_data(
                                            escalated.data, schema,
                                        )
                                    )
                                # Merge escalated data into
                                # wizard_state.data (in-place).
                                escalated_keys = (
                                    self._merge_extraction_result(
                                        escalated.data,
                                        wizard_state,
                                        stage,
                                        user_message,
                                    )
                                )
                                new_data_keys |= escalated_keys
                                # Use the escalated result for
                                # the downstream confidence gate:
                                # broader context with data is a
                                # more informed assessment.  When
                                # escalation returns empty data,
                                # keep the original extraction so
                                # its confidence drives the gate.
                                extraction = escalated

                # Apply schema defaults for properties the user didn't
                # mention.  This ensures template conditions (e.g.
                # ``{% if difficulty %}``) evaluate True when defaults
                # exist in the schema definition.
                default_keys = self._apply_schema_defaults(
                    wizard_state, stage
                )
                if default_keys:
                    new_data_keys |= default_keys

                # ── Field derivations ──
                # Derive missing fields from present ones using
                # configured rules (e.g. domain_id → domain_name
                # via title_case).  Runs before the confidence gate
                # so derived fields count as "present" for the
                # required-field check.
                if self._field_derivations:
                    derived_keys = self._apply_field_derivations(
                        wizard_state, stage,
                    )
                    if derived_keys:
                        new_data_keys |= derived_keys

                # ── Confidence gate ──
                # After merge, check whether we have enough data to
                # proceed.  If confidence is low and required fields are
                # still missing, ask for clarification — but the partial
                # data is preserved in wizard_state.data for the next turn.
                if not extraction.is_confident:
                    required_fields = (
                        schema.get("required", []) if schema else []
                    )
                    can_satisfy = all(
                        self._field_is_present(wizard_state.data.get(f))
                        for f in required_fields
                    )

                    if not can_satisfy:
                        wizard_state.clarification_attempts += 1
                        await self._save_wizard_state(manager, wizard_state)

                        if wizard_state.clarification_attempts >= 3:
                            response = await self._generate_restart_offer(
                                manager, llm, stage, extraction.errors,
                                tools=tools, wizard_state=wizard_state,
                            )
                        else:
                            response = (
                                await self._generate_clarification_response(
                                    manager, llm, stage, extraction.errors,
                                    tools=tools, wizard_state=wizard_state,
                                )
                            )

                        self._add_wizard_metadata(
                            response, wizard_state, stage
                        )
                        return response

                # Reset clarification attempts on viable extraction
                wizard_state.clarification_attempts = 0

                # Update field-extraction tasks from the full accumulated
                # state.  This runs after the confidence gate so tasks are
                # not marked complete on clarification turns (where the
                # early return fires before reaching this line).  We pass
                # wizard_state.data (not extraction.data) because fields
                # merged during prior clarification turns also need their
                # tasks marked complete once the wizard proceeds.
                self._update_field_tasks(wizard_state, wizard_state.data)

                # ── Collection mode handling ──
                # When a stage is in "collection" mode, extracted data
                # is added to a MemoryBank rather than triggering a
                # transition.  The stage loops until a done signal.
                if stage.get("collection_mode") == "collection":
                    col_response = await self._handle_collection_mode(
                        user_message,
                        extraction.data,
                        stage,
                        wizard_state,
                        manager,
                        llm,
                        tools,
                    )
                    if col_response is not None:
                        return col_response

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
                    await self._save_wizard_state(manager, wizard_state)
                    return response

                # Validate against stage schema
                if stage.get("schema") and self._strict_validation:
                    validation_errors = self._validate_data(
                        wizard_state.data, stage["schema"]
                    )
                    if validation_errors:
                        # Save state before returning validation error
                        await self._save_wizard_state(manager, wizard_state)
                        response = await self._generate_validation_response(
                            manager, llm, stage, validation_errors,
                            tools=tools, wizard_state=wizard_state,
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
        if subflow_config and self._handle_subflow_push(
            wizard_state, subflow_config, user_message
        ):
                # Generate response for subflow's first stage
                active_fsm = self._get_active_fsm()
                new_stage = active_fsm.current_metadata
                await self._branch_for_revisited_stage(
                    manager, new_stage.get("name", "")
                )
                response = await self._generate_stage_response(
                    manager, llm, new_stage, wizard_state, tools
                )
                await self._save_wizard_state(manager, wizard_state)
                return response

        # Apply data derivations from transition configs before evaluating
        # conditions.  This lets transitions fill in values that enable their
        # own conditions and subsequent auto-advance checks.
        self._apply_transition_derivations(stage, wizard_state)

        # Log pre-transition state
        logger.debug(
            "FSM transition attempt: from_stage='%s', data_keys=%s",
            wizard_state.current_stage,
            list(wizard_state.data.keys()),
        )

        # Execute FSM step (shared method: inject keys, step, record, update).
        # When _skip_extraction is active, the user_message was directed at
        # the previous stage — don't inject it as _message for FSM condition
        # evaluation on the landing stage.
        from_stage, _step_result = await self._execute_fsm_step(
            wizard_state,
            user_message=None if _skip_extraction else user_message,
        )

        # Auto-save artifact to catalog on stage transition.  Collection
        # stages add records via bank.add() directly (not through tools),
        # so tool-level auto-save won't capture them.  Saving here ensures
        # the catalog stays current as the wizard progresses.
        if wizard_state.current_stage != from_stage and self._catalog and self._artifact:
            try:
                errors = self._artifact.validate()
                if not errors:
                    self._catalog.save(self._artifact)
                    logger.debug(
                        "Auto-saved artifact on stage transition %s -> %s",
                        from_stage,
                        wizard_state.current_stage,
                    )
            except Exception:
                logger.warning(
                    "Auto-save on stage transition failed",
                    exc_info=True,
                )

        # Post-transition lifecycle (shared method: subflow pop, auto-advance, hooks)
        auto_advance_messages = await self._run_post_transition_lifecycle(
            wizard_state,
        )

        # Re-fetch active FSM for response generation
        active_fsm = self._get_active_fsm()

        # Generate stage-aware response
        new_stage = active_fsm.current_metadata
        if new_stage.get("name") != from_stage:
            await self._branch_for_revisited_stage(
                manager, new_stage.get("name", "")
            )
        completed_before = wizard_state.completed
        response = await self._generate_stage_response(
            manager, llm, new_stage, wizard_state, tools
        )

        # Prepend any messages collected from intermediate auto-advance stages
        if auto_advance_messages:
            self._prepend_messages_to_response(response, auto_advance_messages)

        # Check for tool-initiated restart (RestartWizardTool signal).
        # Restart takes priority over completion — if both are set, restart
        # clears the wizard state so completion would be meaningless.
        if self._tool_restart_requested:
            self._tool_restart_requested = False
            self._tool_completion_requested = False
            logger.info("Wizard restart signaled by restart_wizard tool")
            response = await self._execute_restart(
                user_message, wizard_state, manager, llm
            )

        # Check for tool-initiated completion (CompleteWizardTool signal)
        elif not completed_before and self._tool_completion_requested:
            wizard_state.completed = True
            self._tool_completion_requested = False
            logger.info("Wizard completion signaled by complete_wizard tool")
            if self._hooks:
                await self._hooks.trigger_complete(wizard_state.data)

        # Mark this stage's template as rendered so subsequent messages
        # at this stage don't trigger the first-render confirmation logic.
        stage_rendered_name = new_stage.get("name", "")
        if stage_rendered_name and new_stage.get("response_template"):
            wizard_state.increment_render_count(stage_rendered_name)

        # Save wizard state
        await self._save_wizard_state(manager, wizard_state)

        return response

    # =========================================================================
    # Non-Conversational (Advance) API
    # =========================================================================

    @property
    def initial_stage(self) -> str:
        """Name of the wizard's initial (start) stage."""
        return self._fsm.start_stage

    def get_wizard_metadata(self, state: WizardState) -> dict[str, Any]:
        """Build wizard metadata from state without advancing.

        Useful for initial page renders or status checks.  Restores the
        FSM to match the given state before building metadata so that
        stage-level queries (prompt, suggestions, etc.) are accurate.

        Args:
            state: Current wizard state.

        Returns:
            Wizard metadata dict with progress, stage info, etc.
        """
        # Restore FSM (and subflow FSM if applicable) to match state
        self._restore_fsm_state(state)
        return self._build_wizard_metadata(state)

    async def advance(
        self,
        user_input: dict[str, Any],
        state: WizardState,
        *,
        navigation: str | None = None,
    ) -> WizardAdvanceResult:
        """Advance the wizard one step without DynaBot infrastructure.

        This is the non-conversational counterpart to ``generate()``.  It
        performs the same FSM operations and hook lifecycle but without LLM
        calls, conversation storage, or message formatting.

        The caller manages state persistence — pass in the current state,
        receive the updated state in the result, and persist it however
        you choose.

        Args:
            user_input: Structured data for the current stage.  Merged into
                ``state.data`` before evaluating transitions.
            state: Current wizard state.  Mutated in place and also returned
                in the result for convenience.
            navigation: Optional navigation command.  One of:

                - ``"back"`` — go to previous stage
                - ``"skip"`` — skip current stage (must be skippable)
                - ``"restart"`` — reset to initial state

        Returns:
            :class:`WizardAdvanceResult` with updated state, current stage
            info, and completion status.

        Example::

            # Initial state
            state = WizardState(current_stage=reasoning.initial_stage)

            # Advance through wizard stages
            result = await reasoning.advance(
                user_input={"name": "Alice"},
                state=state,
            )
            print(result.stage_prompt)  # Next stage's prompt

            # Navigate back
            result = await reasoning.advance(
                user_input={},
                state=result.state,
                navigation="back",
            )
        """
        # Clear per-turn keys from previous turn
        for key in self._per_turn_keys:
            state.data.pop(key, None)
            state.transient.pop(key, None)

        # Restore FSM (and subflow FSM if applicable) to match state
        self._restore_fsm_state(state)

        from_stage = state.current_stage

        # Handle navigation
        auto_advance_messages: list[str] = []
        if navigation == "back":
            transitioned = await self._navigate_back(state)
        elif navigation == "skip":
            transitioned, auto_advance_messages = await self._navigate_skip(
                state,
            )
        elif navigation == "restart":
            await self._navigate_restart(state)
            transitioned = state.current_stage != from_stage
        else:
            # Normal advance: merge user input and step FSM
            state.data.update(user_input)

            # Exit hook
            if self._hooks:
                await self._hooks.trigger_exit(
                    state.current_stage, state.data
                )
            self._update_stage_exit_tasks(state, state.current_stage)

            # Apply transition derivations
            active_fsm = self._get_active_fsm()
            stage = active_fsm.current_metadata
            self._apply_transition_derivations(stage, state)

            # Execute FSM step (shared method)
            await self._execute_fsm_step(state)

            transitioned = state.current_stage != from_stage

            # Artifact auto-save on transition
            if transitioned and self._catalog and self._artifact:
                try:
                    errors = self._artifact.validate()
                    if not errors:
                        self._catalog.save(self._artifact)
                except Exception:
                    logger.warning(
                        "Auto-save on transition failed", exc_info=True
                    )

            # Post-transition lifecycle (shared method) — only when a
            # transition actually occurred.  Without this guard, hooks like
            # ``trigger_enter`` fire spuriously for the current stage when the
            # FSM stays put (no matching transition condition).
            if transitioned:
                auto_advance_messages = (
                    await self._run_post_transition_lifecycle(state)
                )

        # Build result
        active_fsm = self._get_active_fsm()
        stage_meta = active_fsm.current_metadata
        metadata = self._build_wizard_metadata(state)

        return WizardAdvanceResult(
            state=state,
            stage_name=state.current_stage,
            stage_prompt=active_fsm.get_stage_prompt(),
            stage_schema=stage_meta.get("schema"),
            suggestions=active_fsm.get_stage_suggestions(),
            can_skip=active_fsm.can_skip(),
            can_go_back=active_fsm.can_go_back() and len(state.history) > 1,
            completed=state.completed,
            transitioned=transitioned,
            from_stage=from_stage if transitioned else None,
            auto_advance_messages=auto_advance_messages,
            metadata=metadata,
        )

    def _get_wizard_state(self, manager: Any) -> WizardState:
        """Get or create wizard state from conversation manager.

        Stores a reference to the returned state as ``_last_wizard_state``
        so that ``close()`` can cancel any dangling asyncio tasks stored
        in ephemeral keys.

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
                skip_extraction=fsm_state.get("skip_extraction", False),
            )
            # Restore FSM state (and subflow FSM if applicable)
            self._restore_fsm_state(state)

            # Restore MemoryBank instances from persisted data
            self._restore_banks(wizard_data.get("banks", {}))

            # Restore ArtifactBank from persisted data
            artifact_data = wizard_data.get("artifact")
            if artifact_data and self._artifact is not None:
                from ..memory.artifact_bank import ArtifactBank

                self._artifact = ArtifactBank.from_dict(
                    artifact_data, db_factory=self._create_bank_db
                )
                self._banks = dict(self._artifact.sections)

            self._last_wizard_state = state
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

        state = WizardState(
            current_stage=start_stage,
            data=initial_data,
            history=[start_stage],
            stage_entry_time=time.time(),
            tasks=initial_tasks,
        )
        self._last_wizard_state = state
        return state

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

    async def _save_wizard_state(self, manager: Any, state: WizardState) -> None:
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

        # Reverse sync: artifact → state.data (picks up tool-driven changes
        # like LoadFromCatalogTool replacing fields).
        self._reverse_sync_artifact_to_state(state)
        # Forward sync: state.data → artifact fields before serialization
        self._sync_artifact_fields(state)

        # Partition: separate ephemeral/non-serializable keys from persistent
        state.data, state.transient = self._partition_data(state.data)

        wizard_meta["fsm_state"] = {
            "current_stage": state.current_stage,
            "history": state.history,
            "data": sanitize_for_json(state.data, on_drop="warn"),
            "completed": state.completed,
            "clarification_attempts": state.clarification_attempts,
            "transitions": [
                sanitize_for_json(t, on_drop="warn") for t in state.transitions
            ],
            "stage_entry_time": state.stage_entry_time,
            "tasks": state.tasks.to_dict(),
            "subflow_stack": [s.to_dict() for s in state.subflow_stack],
        }
        # Persist MemoryBank data alongside FSM state
        if self._banks:
            wizard_meta["banks"] = {
                name: bank.to_dict() for name, bank in self._banks.items()
            }
        # Persist ArtifactBank data alongside banks
        if self._artifact:
            wizard_meta["artifact"] = self._artifact.to_dict()
        manager.metadata["wizard"] = wizard_meta

        # Also store fsm_state on the current tree node so any node can
        # serve as an undo restoration point.  The FSM state is small
        # (stage, history, data, transitions) and fits naturally as
        # per-node metadata.
        if hasattr(manager, "state") and manager.state is not None:
            current_node = get_node_by_id(
                manager.state.message_tree, manager.state.current_node_id
            )
            if current_node is not None and hasattr(current_node, "data"):
                node_data = current_node.data
                if isinstance(node_data, ConversationNode):
                    node_data.metadata["wizard_fsm_state"] = wizard_meta[
                        "fsm_state"
                    ]

        # Persist so conversation-level metadata is up to date in storage.
        # Without this, the metadata written by manager.complete() /
        # manager.add_message() inside _generate_stage_response would
        # snapshot the *previous* turn's wizard metadata because those
        # calls run _save_state() before we reach this point.
        if hasattr(manager, "_save_state"):
            await manager._save_state()

    # =========================================================================
    # Subflow Management Methods
    # =========================================================================

    def _get_active_fsm(self) -> WizardFSM:
        """Get the currently active FSM (subflow or main).

        Returns:
            The active WizardFSM instance
        """
        return self._active_subflow_fsm if self._active_subflow_fsm else self._fsm

    def _restore_fsm_state(self, state: WizardState) -> None:
        """Restore the FSM (and subflow FSM if applicable) from state.

        Sets ``_active_subflow_fsm`` when ``state.subflow_stack`` is
        non-empty so that subsequent calls to ``_get_active_fsm()``
        return the correct FSM.

        Args:
            state: Wizard state to restore from.
        """
        fsm_state = {
            "current_stage": state.current_stage,
            "data": state.data,
        }
        self._fsm.restore(fsm_state)

        if state.subflow_stack:
            subflow_name = state.subflow_stack[-1].subflow_network
            self._active_subflow_fsm = self._fsm.get_subflow(subflow_name)
            if self._active_subflow_fsm:
                self._active_subflow_fsm.restore(fsm_state)
        else:
            self._active_subflow_fsm = None

    # =========================================================================
    # Shared Wizard Lifecycle Methods
    # =========================================================================
    # These methods consolidate the core wizard lifecycle operations used by
    # both generate() (conversational path) and advance() (non-conversational
    # path).  Each was extracted from inline code that was duplicated across
    # generate(), _execute_skip(), and _execute_back().

    async def _execute_fsm_step(
        self,
        state: WizardState,
        *,
        user_message: str | None = None,
        trigger: str = "user_input",
    ) -> tuple[str, Any]:
        """Execute an FSM step with standard runtime key injection and state update.

        Injects runtime keys (``_bank_fn``, ``_message``), executes the FSM
        step, cleans up runtime keys, and updates wizard state
        (``current_stage``, ``history``, ``completed``).  Records the
        transition if the stage changed.

        This is the shared core used by ``generate()``, ``advance()``,
        and ``_navigate_skip()``.

        Args:
            state: Wizard state (mutated in place).
            user_message: Optional message for condition evaluation.
            trigger: Transition trigger label for audit trail.

        Returns:
            Tuple of (from_stage, step_result).
        """
        active_fsm = self._get_active_fsm()
        from_stage = state.current_stage
        duration_ms = (time.time() - state.stage_entry_time) * 1000

        # Inject runtime keys for condition/transform evaluation
        if user_message is not None:
            state.data["_message"] = user_message
        state.data["_bank_fn"] = self._make_bank_accessor()
        self._current_turn = TurnContext(
            message=user_message,
            bank_fn=self._make_bank_accessor(),
            intent=state.data.get("_intent"),
        )

        # Execute FSM step
        step_result = await active_fsm.step_async(state.data)

        # Clean up runtime keys
        self._current_turn = None
        state.data.pop("_message", None)
        state.data.pop("_bank_fn", None)

        to_stage = active_fsm.current_stage

        # Log transition result
        if to_stage != from_stage:
            logger.info(
                "FSM transition: '%s' -> '%s' (is_complete=%s, depth=%d)",
                from_stage,
                to_stage,
                step_result.is_complete,
                state.subflow_depth,
            )
        elif not step_result.success:
            logger.warning(
                "FSM transition transform failed at '%s': %s",
                from_stage,
                step_result.error,
            )
            state.data["_transform_error"] = step_result.error
        else:
            logger.debug(
                "FSM no transition: stayed at '%s' (is_complete=%s)",
                from_stage,
                step_result.is_complete,
            )

        # Record transition if stage changed
        if to_stage != from_stage:
            condition_expr = active_fsm.get_transition_condition(
                from_stage, to_stage
            )
            transition = create_transition_record(
                from_stage=from_stage,
                to_stage=to_stage,
                trigger=trigger,
                duration_in_stage_ms=duration_ms,
                data_snapshot=state.data.copy(),
                user_input=user_message,
                condition_evaluated=condition_expr,
                condition_result=True if condition_expr else None,
                subflow_depth=state.subflow_depth,
            )
            state.transitions.append(transition)
            state.stage_entry_time = time.time()

        # Update wizard state from FSM result
        state.current_stage = to_stage
        if to_stage not in state.history:
            state.history.append(to_stage)
        state.completed = step_result.is_complete

        return from_stage, step_result

    async def _run_post_transition_lifecycle(
        self,
        state: WizardState,
    ) -> list[str]:
        """Run post-transition lifecycle: subflow pop, auto-advance, hooks.

        Called after ``_execute_fsm_step()`` to handle the standard
        post-transition sequence.  Shared by ``generate()`` and
        ``advance()``.

        Args:
            state: Wizard state (mutated in place).

        Returns:
            List of rendered template strings from auto-advanced stages.
        """
        # Subflow pop check
        if self._should_pop_subflow(state):
            self._handle_subflow_pop(state)
            state.completed = False

        # Auto-advance through stages with all required fields filled
        active_fsm = self._get_active_fsm()
        stage = active_fsm.current_metadata
        auto_advance_messages = await self._run_auto_advance_loop(
            state, active_fsm, stage,
        )

        # Re-fetch in case subflow pop occurred during auto-advance
        active_fsm = self._get_active_fsm()

        # Stage entry hook
        if self._hooks:
            await self._hooks.trigger_enter(
                state.current_stage, state.data
            )

        # Completion hook
        if state.completed and self._hooks:
            await self._hooks.trigger_complete(state.data)

        return auto_advance_messages

    async def _navigate_back(
        self,
        state: WizardState,
        *,
        user_message: str | None = None,
    ) -> bool:
        """Execute back navigation on the FSM.

        Performs the FSM-level back operation without generating a
        response.  Used by both ``_execute_back()`` (conversational)
        and ``advance()`` (non-conversational).

        Hook coverage (when ``consistent_navigation_lifecycle=True``):
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
        active_fsm = self._get_active_fsm()
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

        if self._consistent_navigation_lifecycle and self._hooks:
            await self._hooks.trigger_enter(state.current_stage, state.data)

        return True

    async def _navigate_skip(
        self,
        state: WizardState,
        *,
        user_message: str | None = None,
    ) -> tuple[bool, list[str]]:
        """Execute skip navigation on the FSM.

        Performs the skip operation (mark skipped, apply defaults, step
        FSM, run post-transition lifecycle) without generating a response.

        Hook coverage (when ``consistent_navigation_lifecycle=True``):
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
        active_fsm = self._get_active_fsm()
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
        if self._consistent_navigation_lifecycle:
            auto_msgs = await self._run_post_transition_lifecycle(state)
        return True, auto_msgs

    async def _navigate_restart(
        self, state: WizardState, user_message: str = "",
    ) -> None:
        """Execute restart navigation on the FSM.

        Hook coverage: only the restart hook fires (via
        ``_restart_cleanup``).  Enter/exit hooks are intentionally NOT
        fired — restart is a full state reset, not a stage transition.

        Args:
            state: Wizard state (mutated in place).
            user_message: User message that triggered the restart.
        """
        await self._restart_cleanup(state, user_message, trigger="api_restart")

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
            if condition and not self._evaluate_condition(condition, wizard_state.data):
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

        Prefers ``raw_content`` from node metadata (set by DynaBot when
        knowledge-base or memory context is prepended) so that schema
        extraction sees the user's original message without context noise.

        Args:
            manager: ConversationManager instance

        Returns:
            Last user message text
        """
        # Try node-level access first (ConversationManager with state).
        # Nodes carry per-message metadata including raw_content.
        if hasattr(manager, "state") and manager.state is not None:
            nodes = manager.state.get_current_nodes()
            for node in reversed(nodes):
                if node.message.role == "user":
                    raw = node.metadata.get("raw_content")
                    if raw is not None:
                        return raw
                    content = node.message.content
                    if isinstance(content, str):
                        return content
                    # Handle structured content (list of content parts)
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                return part.get("text", "")
            return ""

        # Fallback: dict-based access (test managers without state)
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
            return await self._execute_skip(message, state, manager, llm)

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
        if await self._navigate_back(state, user_message=message):
            stage = self._fsm.current_metadata
            await self._branch_for_revisited_stage(
                manager, state.current_stage
            )
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
        message: str,
        state: WizardState,
        manager: Any,
        llm: Any,
    ) -> Any:
        """Execute skip navigation.

        Delegates to ``_navigate_skip()`` for FSM stepping and lifecycle,
        then generates the next stage's response.  Bypasses the extraction
        pipeline entirely, consistent with ``_execute_back()`` and
        ``_execute_restart()``.

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

        _, auto_advance_messages = await self._navigate_skip(
            state, user_message=message,
        )

        stage = self._get_active_fsm().current_metadata
        response = await self._generate_stage_response(
            manager, llm, stage, state, None
        )
        if stage.get("response_template"):
            state.increment_render_count(
                stage.get("name", "unknown")
            )
        if auto_advance_messages:
            self._prepend_messages_to_response(response, auto_advance_messages)
        return response

    async def _restart_cleanup(
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
        await self._restart_cleanup(state, message)

        stage = self._fsm.current_metadata
        await self._branch_for_revisited_stage(
            manager, state.current_stage
        )
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

        When extraction_scope is ``"wizard_session"`` or
        ``"recent_messages"``, builds context from user messages in the
        wizard session for extraction.  For ``"recent_messages"``, only
        the last ``recent_messages_count`` messages are included.  This
        allows the wizard to remember information provided in earlier
        messages.

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

        # Verbatim capture: skip LLM extraction for trivial schemas
        # (single required string field, no constraints) or when
        # capture_mode is explicitly set to "verbatim".
        #
        # However, when a bot response is available in the conversation,
        # verbatim capture is unsafe: the user may be using deictic
        # references like "the first one" that require the bot's prior
        # response as context to resolve.  In that case, fall through to
        # LLM extraction so the bot-response prepending code can provide
        # the necessary context.
        has_bot_response = False
        if manager is not None:
            has_bot_response = bool(self._get_last_bot_response(manager))

        if not self._needs_llm_extraction(schema, stage) and not has_bot_response:
            field_name = next(iter(schema.get("properties", {})))
            logger.debug(
                "Verbatim capture: stage='%s', field='%s'",
                stage_name,
                field_name,
            )
            return SimpleExtractionResult(
                data={field_name: message},
                confidence=1.0,
                metadata={"capture_mode": "verbatim"},
            )

        # Build extraction input based on scope (stage override or wizard default)
        extraction_scope = self._get_extraction_scope(stage)
        if (
            extraction_scope in ("wizard_session", "recent_messages")
            and manager is not None
            and wizard_state is not None
        ):
            # Build context from wizard session conversation.
            # For recent_messages scope, limit to last N user messages.
            max_msgs = (
                self._recent_messages_count
                if extraction_scope == "recent_messages"
                else None
            )
            wizard_context = self._build_wizard_context(
                manager, wizard_state, max_messages=max_msgs,
            )
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
        self,
        manager: Any,
        wizard_state: WizardState,
        *,
        max_messages: int | None = None,
    ) -> str:
        """Build extraction context from wizard session history.

        Collects user messages from the conversation to provide context
        for extraction. This allows the wizard to "remember" information
        provided in earlier messages.

        Prefers ``raw_content`` from node metadata when available, so
        that session-wide extraction context is not polluted by KB/memory
        augmentation from prior turns.

        Args:
            manager: ConversationManager instance
            wizard_state: Current wizard state
            max_messages: When set, include only the most recent *N*
                prior user messages (for ``recent_messages`` scope).
                ``None`` means include all prior messages (full session).

        Returns:
            Formatted context string from previous user messages,
            or empty string if no previous messages.
        """
        user_messages: list[str] = []

        # Try node-level access first (prefers raw_content from metadata)
        if hasattr(manager, "state") and manager.state is not None:
            nodes = manager.state.get_current_nodes()
            for node in nodes:
                if node.message.role == "user":
                    raw = node.metadata.get("raw_content")
                    if raw is not None:
                        user_messages.append(raw)
                    elif isinstance(node.message.content, str):
                        user_messages.append(node.message.content)
                    elif isinstance(node.message.content, list):
                        # Handle structured content (list of content parts)
                        for part in node.message.content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                user_messages.append(part.get("text", ""))
                                break
        else:
            # Fallback: dict-based access (test managers without state)
            for msg in manager.get_messages():
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

        # Limit to most recent N messages for recent_messages scope
        if max_messages is not None and len(previous_messages) > max_messages:
            previous_messages = previous_messages[-max_messages:]

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
            # Skip internal fields and per-turn keys (expected to change each turn)
            if field_name.startswith("_") or field_name in self._per_turn_keys:
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
            if WizardReasoning._is_ancestor_of(node_id, current_node_id):
                return node_id

        return None

    async def _branch_for_revisited_stage(
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
            # MemoryBank accessor for template expressions
            "bank": self._make_bank_accessor(),
            # ArtifactBank (None if not configured)
            "artifact": self._artifact,
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

    def _needs_llm_extraction(
        self, schema: dict[str, Any], stage: dict[str, Any],
    ) -> bool:
        """Determine whether LLM extraction is needed for a schema.

        Returns ``False`` when the schema describes a single required string
        field with no enum or format constraints — the user's raw input can
        be used directly (verbatim capture).

        The decision can be overridden via ``collection_config.capture_mode``:

        - ``"auto"`` (default): use schema-based detection described above.
        - ``"verbatim"``: always skip LLM extraction.
        - ``"extract"``: always use LLM extraction.

        Args:
            schema: JSON Schema dict for the current stage.
            stage: Current stage metadata dict.

        Returns:
            ``True`` if LLM extraction should be used, ``False`` for
            verbatim capture.
        """
        # capture_mode can be set as a top-level stage field or nested
        # under collection_config.  Top-level takes precedence.
        capture_mode = stage.get("capture_mode")
        if capture_mode is None:
            col_config = stage.get("collection_config") or {}
            capture_mode = col_config.get("capture_mode", "auto")

        if capture_mode == "verbatim":
            return False
        if capture_mode == "extract":
            return True

        # Auto-detect: single required string field with no constraints
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if len(required) == 1 and len(properties) == 1:
            field_name = required[0]
            field_def = properties.get(field_name, {})
            if (
                field_def.get("type") == "string"
                and "enum" not in field_def
                and "pattern" not in field_def
                and "format" not in field_def
            ):
                return False

        return True

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

        Delegates to ``ReActReasoning`` which provides duplicate tool call
        detection, ToolsNotSupportedError handling, and optional trace
        storage (when ``store_trace`` is enabled via stage or wizard settings).

        Args:
            manager: ConversationManager instance
            enhanced_prompt: Stage-aware system prompt
            stage: Stage metadata dict
            state: Current wizard state (kept for call-site stability)
            tools: Available tools for this stage
            metadata: Optional metadata to persist on conversation nodes

        Returns:
            Final LLM response after ReAct loop completes
        """
        from .react import ReActReasoning

        max_iterations = self._get_max_iterations(stage)

        extra_context: dict[str, Any] = {}
        if self._banks:
            extra_context["banks"] = self._banks
        if self._artifact:
            extra_context["artifact"] = self._artifact
        if self._catalog:
            extra_context["catalog"] = self._catalog

        # Mutable signal dicts for lifecycle tools to communicate back
        completion_signal: dict[str, Any] = {"requested": False}
        extra_context["_completion_signal"] = completion_signal
        restart_signal: dict[str, Any] = {"requested": False}
        extra_context["_restart_signal"] = restart_signal

        # Build a prompt refresher so the ReAct loop can re-render the
        # system prompt between iterations.  This ensures that mutating
        # tools (load_from_catalog, add_bank_record, etc.) produce an
        # up-to-date collection summary for the next LLM call.
        def prompt_refresher() -> str:
            fresh_context = self._build_stage_context(stage, state)
            return f"{manager.system_prompt}\n\n{fresh_context}"

        # Inherit store_trace and verbose from stage or wizard-level default
        store_trace = stage.get("store_trace", self._default_store_trace)
        verbose = stage.get("verbose", self._default_verbose)

        react = ReActReasoning(
            max_iterations=max_iterations,
            verbose=verbose,
            store_trace=store_trace,
            artifact_registry=self._artifact_registry,
            review_executor=self._review_executor,
            context_builder=self._context_builder,
            extra_context=extra_context or None,
            prompt_refresher=prompt_refresher,
        )

        response = await react.generate(
            manager=manager,
            llm=None,
            tools=tools,
            system_prompt_override=enhanced_prompt,
            metadata=metadata,
        )

        # Check if lifecycle tools signaled completion or restart
        if completion_signal.get("requested"):
            self._tool_completion_requested = True
            self._tool_completion_summary = completion_signal.get(
                "summary", ""
            )
        if restart_signal.get("requested"):
            self._tool_restart_requested = True

        return response

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

        # CD-2: Inject collection progress during collection iterations
        # Sibling branching prunes prior inputs from the ancestor path,
        # so the LLM needs an explicit summary of what's been collected.
        if stage.get("collection_mode") == "collection":
            col_config = stage.get("collection_config", {})
            bank_name = col_config.get("bank_name", "")
            bank = self._banks.get(bank_name)
            if bank and bank.count() > 0:
                max_display = 20
                records = bank.all()
                lines.append(f"\n## Collection Progress ({bank_name})")
                lines.append(
                    f"{bank.count()} items collected so far:"
                )
                for record in records[:max_display]:
                    summary = ", ".join(
                        f"{v}" for k, v in record.data.items()
                        if not k.startswith("_")
                    )
                    lines.append(f"- {summary}")
                if len(records) > max_display:
                    lines.append(
                        f"- ... and {len(records) - max_display} more"
                    )
                lines.append("")

        # CD-3: Inject compiled artifact snapshot at guided-to-dynamic boundary
        # When transitioning to a ReAct review stage, provide the full artifact
        # overview so the LLM doesn't need tool calls to see collected data.
        if self._use_react_for_stage(stage) and self._artifact:
            has_data = (
                self._artifact.fields
                or any(
                    bank.count() > 0
                    for bank in self._artifact.sections.values()
                )
            )
            if has_data:
                max_section_display = 20
                lines.append("\n## Collection Summary")
                compiled = self._artifact.compile()
                for key, value in compiled.items():
                    if key.startswith("_"):
                        continue
                    if isinstance(value, list):
                        continue  # Sections handled below
                    if value is not None:
                        lines.append(f"- {key}: {value}")
                for section_name, bank in self._artifact.sections.items():
                    if bank.count() > 0:
                        lines.append(
                            f"\n### {section_name} "
                            f"({bank.count()} records)"
                        )
                        for record in bank.all()[:max_section_display]:
                            summary = ", ".join(
                                f"{v}" for k, v in record.data.items()
                                if not k.startswith("_")
                            )
                            lines.append(f"- {summary}")
                        if bank.count() > max_section_display:
                            lines.append(
                                f"- ... and "
                                f"{bank.count() - max_section_display} more"
                            )
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

    def _apply_field_derivations(
        self,
        wizard_state: WizardState,
        stage: dict[str, Any],
    ) -> set[str]:
        """Apply field derivation rules to fill missing fields.

        Runs after merge/escalation/defaults and before the confidence
        gate.  Derived values never overwrite user-provided or extracted
        data unless the rule specifies ``when: always``.

        Per-stage override: set ``derivation_enabled: false`` on a
        stage to suppress derivation for that stage.

        Args:
            wizard_state: Current wizard state (data modified in-place).
            stage: Current stage metadata.

        Returns:
            Set of keys that were derived (newly added to data).
        """
        if not self._field_derivations:
            return set()

        # Per-stage override
        stage_enabled = stage.get("derivation_enabled")
        if stage_enabled is False:
            return set()

        return apply_field_derivations(
            self._field_derivations,
            wizard_state.data,
            field_is_present=self._field_is_present,
        )

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
        1. Auto-advance is enabled for this stage (see precedence below)
        2. The stage has a schema with required fields (or all properties
           if no required list)
        3. All required fields have non-empty values in wizard_state.data
        4. The stage is not an end stage
        5. At least one transition condition is satisfied

        Auto-advance precedence (stage-level wins over global):
        - ``auto_advance: false`` — disabled regardless of global setting
        - ``auto_advance: true``  — enabled regardless of global setting
        - absent (``None``)       — defers to global
          ``auto_advance_filled_stages``

        Args:
            wizard_state: Current wizard state
            stage: Stage configuration dict

        Returns:
            True if stage can be auto-advanced
        """
        # Check if auto-advance is enabled for this stage.
        # Stage-level setting takes precedence over global when explicitly set.
        stage_auto_advance = stage.get("auto_advance")
        if stage_auto_advance is False:
            # Explicitly disabled at stage level — respect regardless of global
            return False
        if not stage_auto_advance and not self._auto_advance_filled_stages:
            # Not explicitly enabled at stage level, and global is off
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

        # If no fields at all:
        # - Per-stage auto_advance: true → can advance (message/display stage)
        #   Still requires a satisfied transition condition (checked below).
        # - Global auto_advance_filled_stages only → cannot advance
        #   (that setting means "skip stages whose fields are already filled",
        #   not "skip stages that have no fields to fill").
        if not required_fields and not stage_auto_advance:
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

    async def _run_auto_advance_loop(
        self,
        wizard_state: WizardState,
        active_fsm: WizardFSM,
        initial_stage: dict[str, Any],
        *,
        skip_first_render: bool = False,
    ) -> list[str]:
        """Run the auto-advance loop, collecting rendered templates.

        Advances through consecutive stages that satisfy
        ``_can_auto_advance``, rendering each stage's
        ``response_template`` before moving past it.

        Args:
            wizard_state: Current wizard state (mutated in place)
            active_fsm: The currently active FSM instance
            initial_stage: Stage metadata to start advancing from
            skip_first_render: If True, skip the template render on
                the first iteration (used by greet when the start
                stage response is already captured)

        Returns:
            List of rendered template strings from auto-advanced stages
        """
        messages: list[str] = []
        count = 0
        max_advances = 10
        stage = initial_stage

        while (
            count < max_advances
            and not wizard_state.completed
            and self._can_auto_advance(wizard_state, stage)
        ):
            # Render template of the stage being advanced past
            if not (skip_first_render and count == 0):
                rendered = self._render_auto_advance_template(
                    stage, wizard_state
                )
                if rendered:
                    messages.append(rendered)

            count += 1
            old_stage_name = wizard_state.current_stage
            duration_ms = (
                (time.time() - wizard_state.stage_entry_time) * 1000
            )

            auto_step_result = await active_fsm.step_async(
                wizard_state.data
            )
            new_stage_name = active_fsm.current_stage

            if new_stage_name == old_stage_name:
                break

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

            wizard_state.current_stage = new_stage_name
            if new_stage_name not in wizard_state.history:
                wizard_state.history.append(new_stage_name)
            wizard_state.completed = auto_step_result.is_complete
            wizard_state.stage_entry_time = time.time()

            logger.info(
                "Auto-advanced from %s to %s",
                old_stage_name,
                new_stage_name,
            )

            # Handle subflow pop if needed (no-op when no subflow is active)
            if self._should_pop_subflow(wizard_state):
                self._handle_subflow_pop(wizard_state)
                active_fsm = self._get_active_fsm()
                wizard_state.completed = False

            stage = active_fsm.current_metadata

        # If we advanced through any stages, mark the landing stage so the
        # next generate() call skips extraction — the user hasn't had a
        # chance to respond to this stage's prompt yet.
        if count > 0:
            wizard_state.skip_extraction = True

        return messages

    def _render_auto_advance_template(
        self, stage: dict[str, Any], state: WizardState
    ) -> str | None:
        """Render a stage's response_template for auto-advance collection.

        Used during auto-advance to capture message stage content before
        the stage is advanced past. Only renders if the stage has a
        response_template.

        Args:
            stage: Stage metadata dict
            state: Current wizard state

        Returns:
            Rendered template string, or None if the stage has no template
        """
        template = stage.get("response_template")
        if not template:
            return None

        rendered = self._render_response_template(template, stage, state)
        stage_name = stage.get("name", "unknown")
        logger.debug(
            "Rendered message stage '%s' template during auto-advance "
            "(%d chars)",
            stage_name,
            len(rendered),
        )
        # Track render count so re-visiting won't re-confirm
        state.increment_render_count(stage_name)
        return rendered

    @staticmethod
    def _prepend_messages_to_response(
        response: Any, messages: list[str]
    ) -> None:
        """Prepend collected auto-advance messages to a response.

        Modifies the response object in place, inserting message stage
        content before the landing stage's response. Messages are joined
        with double newlines.

        Args:
            response: Response object with a ``content`` attribute
            messages: List of rendered template strings to prepend
        """
        if not messages:
            return
        prefix = "\n\n".join(messages) + "\n\n"
        response.content = prefix + response.content

    @staticmethod
    def _field_is_present(value: Any) -> bool:
        """A field has been provided if its value is not None.

        Centralises the "field presence" semantic used by the
        ``has()`` condition helper and the ``can_satisfy`` confidence
        gate.

        Note: ``_can_auto_advance`` uses stricter logic — it
        additionally rejects empty strings because auto-advance
        requires fields to be *filled*, not merely *present*.
        """
        return value is not None

    def _check_required_fields_missing(
        self,
        wizard_state: WizardState,
        stage: dict[str, Any],
    ) -> set[str]:
        """Return required field names not yet present in wizard_state.data.

        Uses ``_field_is_present`` semantics (value is not None) to match
        the ``can_satisfy`` confidence gate check.

        Args:
            wizard_state: Current wizard state with accumulated data
            stage: Stage configuration dict containing the schema

        Returns:
            Set of required field names whose values are absent or None
        """
        schema = stage.get("schema")
        if not schema:
            return set()
        required_fields = schema.get("required", [])
        if not required_fields:
            return set()
        return {
            f for f in required_fields
            if not self._field_is_present(wizard_state.data.get(f))
        }

    def _merge_extraction_result(
        self,
        extraction_data: dict[str, Any],
        wizard_state: WizardState,
        stage: dict[str, Any],
        user_message: str,
    ) -> set[str]:
        """Merge extracted data into wizard state, returning new/changed keys.

        Applies the grounding filter (per-stage override or wizard-level)
        to protect existing data from ungrounded overwrites.  Skips None
        values.

        Args:
            extraction_data: Dict of field→value from extraction
            wizard_state: Wizard state whose ``.data`` is updated in-place
            stage: Stage configuration dict (for schema and grounding config)
            user_message: Current user message (for grounding checks)

        Returns:
            Set of keys that were newly added or changed
        """
        schema = stage.get("schema")
        schema_props = schema.get("properties", {}) if schema else {}

        # Resolve merge filter: per-stage grounding override, then
        # fall back to wizard-level filter.
        stage_grounding = stage.get("extraction_grounding")
        if stage_grounding is not None:
            if stage_grounding:
                active_filter: MergeFilter | None = (
                    self._merge_filter
                    or SchemaGroundingFilter(
                        overlap_threshold=self._grounding_overlap_threshold,
                    )
                )
            else:
                active_filter = None
        else:
            active_filter = self._merge_filter

        new_data_keys: set[str] = set()
        for k, v in extraction_data.items():
            if v is None:
                continue
            if active_filter is not None:
                existing = wizard_state.data.get(k)
                prop_def = schema_props.get(k, {})
                if not active_filter.should_merge(
                    k, v, existing, user_message, prop_def,
                ):
                    continue
            if k not in wizard_state.data or wizard_state.data[k] != v:
                new_data_keys.add(k)
                wizard_state.data[k] = v
        return new_data_keys

    def _evaluate_condition(self, condition: str, data: dict[str, Any]) -> bool:
        """Safely evaluate a transition condition.

        Uses a restricted execution environment to evaluate condition
        expressions like ``data.get('subject')``, ``has('count')``,
        or ``data.get('count', 0) > 5``.

        Available globals in condition expressions:

        - ``data`` — the wizard state data dict
        - ``has(key)`` — shorthand for ``data.get(key) is not None``;
          preferred for boolean/numeric/list fields where falsy values
          are legitimate.  Note: empty strings are considered present;
          for text fields where non-empty content is required, use
          ``data.get('key')`` (truthiness rejects empty strings)
        - ``bank`` — memory bank accessor
        - ``artifact`` — current artifact
        - ``true``/``false``/``null``/``none`` — YAML/JSON literal aliases

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

            # Create a function to evaluate the condition.
            # Shallow copy so exec cannot add/remove/replace top-level
            # keys in live wizard state.  Nested mutable values are
            # still shared.
            data_snapshot = dict(data)
            global_vars: dict[str, Any] = {
                "data": data_snapshot,
                "has": lambda key: self._field_is_present(data_snapshot.get(key)),
                "bank": self._make_bank_accessor(),
                "artifact": self._artifact,
                # Common aliases for YAML/JSON boolean literals
                "true": True,
                "false": False,
                "null": None,
                "none": None,
            }
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
        if self._section_to_stage_mapping and section_lower in self._section_to_stage_mapping:
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

    async def _generate_validation_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        errors: list[str],
        tools: list[Any] | None = None,
        wizard_state: WizardState | None = None,
    ) -> Any:
        """Generate response asking for corrections.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            errors: Validation error messages
            tools: Available tools (so LLM can handle out-of-stage requests)
            wizard_state: Current wizard state for metadata snapshot

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
        # CD-8: Include full stage context (collection progress, already
        # collected data) so non-happy-path responses have the same
        # context richness as happy-path responses.
        stage_context = (
            self._build_stage_context(stage, wizard_state)
            if wizard_state
            else ""
        )
        wizard_snapshot = (
            {"wizard": self._build_wizard_metadata(wizard_state)}
            if wizard_state
            else None
        )
        base = manager.system_prompt
        if stage_context:
            base = f"{base}\n\n{stage_context}"
        return await manager.complete(
            system_prompt_override=base + error_context,
            tools=tools,
            metadata=wizard_snapshot,
        )

    async def _generate_transform_error_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        error: str,
        tools: list[Any] | None = None,
        wizard_state: WizardState | None = None,
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
            tools: Available tools (so LLM can handle out-of-stage requests)
            wizard_state: Current wizard state for metadata snapshot

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
        wizard_snapshot = (
            {"wizard": self._build_wizard_metadata(wizard_state)}
            if wizard_state
            else None
        )
        return await manager.complete(
            system_prompt_override=manager.system_prompt + error_context,
            tools=tools,
            metadata=wizard_snapshot,
        )

    async def _generate_clarification_response(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        issues: list[str],
        tools: list[Any] | None = None,
        wizard_state: WizardState | None = None,
    ) -> Any:
        """Generate response asking for clarification.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            issues: Extraction issues
            tools: Available tools (so LLM can handle out-of-stage requests)
            wizard_state: Current wizard state for metadata snapshot

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

**What I'm Looking For**: \
{stage.get('prompt', 'Please provide more specific information.')}\
{suggestions_text}

Please ask a clarifying question to help gather the needed information.
Be conversational and helpful - don't make the user feel like they did something wrong.
"""
        # CD-8: Include full stage context (collection progress, already
        # collected data) so non-happy-path responses have the same
        # context richness as happy-path responses.
        stage_context = (
            self._build_stage_context(stage, wizard_state)
            if wizard_state
            else ""
        )
        wizard_snapshot = (
            {"wizard": self._build_wizard_metadata(wizard_state)}
            if wizard_state
            else None
        )
        base = manager.system_prompt
        if stage_context:
            base = f"{base}\n\n{stage_context}"
        return await manager.complete(
            system_prompt_override=base + clarification_context,
            tools=tools,
            metadata=wizard_snapshot,
        )

    async def _generate_restart_offer(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        issues: list[str],
        tools: list[Any] | None = None,
        wizard_state: WizardState | None = None,
    ) -> Any:
        """Generate response offering to restart after multiple failures.

        Args:
            manager: ConversationManager instance
            llm: LLM provider
            stage: Current stage metadata
            issues: Extraction issues
            tools: Available tools (so LLM can handle out-of-stage requests)
            wizard_state: Current wizard state for metadata snapshot

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
        # CD-8: Include full stage context (collection progress, already
        # collected data) so non-happy-path responses have the same
        # context richness as happy-path responses.
        stage_context = (
            self._build_stage_context(stage, wizard_state)
            if wizard_state
            else ""
        )
        wizard_snapshot = (
            {"wizard": self._build_wizard_metadata(wizard_state)}
            if wizard_state
            else None
        )
        base = manager.system_prompt
        if stage_context:
            base = f"{base}\n\n{stage_context}"
        return await manager.complete(
            system_prompt_override=base + restart_context,
            tools=tools,
            metadata=wizard_snapshot,
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
