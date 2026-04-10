"""Wizard data types, value objects, and module-level constants.

This module contains all data types, constants, and standalone helper
functions used across the wizard reasoning subsystem.  Extracted from
``wizard.py`` (item 77a) to reduce monolith size and provide clean
import paths.

Types defined here:
- :class:`TurnContext` — per-turn transient values
- :class:`WizardStageContext` — stage-level context snapshot
- :class:`SubflowContext` — nested subflow parent state
- :class:`WizardState` — central persistent state object
- :class:`WizardAdvanceResult` — return type for ``advance()``
- :class:`ExtractionPipelineResult` — return type for extraction pipeline
- :class:`StageSchema` — JSON Schema wrapper for stage fields
- :class:`NavigationCommandConfig` — single navigation command config
- :class:`NavigationConfig` — full navigation configuration
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dataknobs_common.serialization import sanitize_for_json

from .observability import TransitionRecord, WizardTaskList
from .wizard_grounding import MergeFilter
from .wizard_utils import word_in_text

if TYPE_CHECKING:
    from .wizard_fsm import WizardFSM

logger = logging.getLogger(__name__)


# =========================================================================
# Constants
# =========================================================================

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

# Recovery strategy identifiers
RECOVERY_DERIVATION = "derivation"
RECOVERY_SCOPE_ESCALATION = "scope_escalation"
RECOVERY_FOCUSED_RETRY = "focused_retry"
RECOVERY_BOOLEAN = "boolean_recovery"
RECOVERY_CLARIFICATION = "clarification"

# Default pipeline order (when no explicit pipeline is configured).
# Each strategy only fires when required fields are still missing.
# Note: focused_retry and boolean_recovery are not in the default —
# consumers opt in by adding them to their pipeline.  focused_retry
# additionally requires focused_retry.enabled: true.
DEFAULT_RECOVERY_PIPELINE: list[str] = [
    RECOVERY_DERIVATION,
    RECOVERY_SCOPE_ESCALATION,
]

# Valid strategy names for pipeline configuration.
# Note: "clarification" is a no-op placeholder — clarification is
# handled by the confidence gate, not the pipeline engine.  Including
# it in a pipeline documents intent but doesn't change behavior.
VALID_RECOVERY_STRATEGIES: frozenset[str] = frozenset({
    RECOVERY_DERIVATION,
    RECOVERY_SCOPE_ESCALATION,
    RECOVERY_FOCUSED_RETRY,
    RECOVERY_BOOLEAN,
    RECOVERY_CLARIFICATION,
})

# Default signal words for boolean extraction recovery.
# Overridable per-field via x-extraction.affirmative_signals
# and x-extraction.negative_signals.
_DEFAULT_AFFIRMATIVE_SIGNALS: frozenset[str] = frozenset({
    "yes", "confirm", "save", "approve", "correct", "sure",
    "ok", "okay", "agreed", "accept", "absolutely", "definitely",
    "yep", "yeah",
})

_DEFAULT_AFFIRMATIVE_PHRASES: tuple[str, ...] = (
    "looks good", "go ahead", "that's right", "sounds good",
    "i confirm", "let's do it", "let's go",
)

_DEFAULT_NEGATIVE_SIGNALS: frozenset[str] = frozenset({
    "no", "wait", "stop", "cancel", "wrong", "redo", "nope",
    "nah", "incorrect",
})

_DEFAULT_NEGATIVE_PHRASES: tuple[str, ...] = (
    "not yet", "hold on", "start over", "go back", "not right",
    "don't save", "i disagree",
)

# Navigation keyword defaults
DEFAULT_BACK_KEYWORDS: tuple[str, ...] = ("back", "go back", "previous")
DEFAULT_SKIP_KEYWORDS: tuple[str, ...] = ("skip", "skip this", "use default", "use defaults")
DEFAULT_RESTART_KEYWORDS: tuple[str, ...] = ("restart", "start over")


# =========================================================================
# Dataclasses and value objects
# =========================================================================


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
        extraction: Extraction result when ``advance()`` ran with raw
            text input.  ``None`` when ``user_input`` was a dict.
        missing_fields: Required fields still missing after extraction.
            ``None`` when no extraction was performed.
        changed_fields: Fields newly set or changed during extraction.
            ``None`` when no extraction was performed.
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
    extraction: Any | None = None
    """Extraction result when ``advance()`` ran with raw text input.

    ``None`` when ``user_input`` was a dict (no extraction performed).
    When populated, contains the extraction result with ``.data``,
    ``.confidence``, ``.errors``, and ``.is_confident`` attributes.
    """
    missing_fields: set[str] | None = None
    """Required fields still missing after extraction.

    ``None`` when no extraction was performed.  Empty set when all
    required fields are satisfied.
    """
    changed_fields: set[str] | None = None
    """Fields that were newly set or changed during extraction.

    ``None`` when no extraction was performed.  Useful for UI clients
    that need to highlight updated fields or show a diff.
    """


@dataclass
class ExtractionPipelineResult:
    """Result from the shared extraction pipeline.

    Encapsulates the data-processing outcome (extract, normalize, merge,
    defaults, derivations, recovery) without any presentation concerns.
    Used by both ``generate()`` and ``advance()`` to share the extraction
    logic.
    """

    extraction: Any
    """The raw extraction result from ``_extract_data()``."""
    new_data_keys: set[str]
    """Keys that were newly set or changed in ``state.data``."""
    missing_fields: set[str]
    """Required fields still missing after the full pipeline."""
    is_confident: bool
    """Whether the extraction met the confidence threshold."""


# ---------------------------------------------------------------------------
# StageSchema — normalized view of a wizard stage's JSON Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSchema:
    """Normalized view of a wizard stage's JSON Schema.

    Centralises schema access so that every site in the wizard subsystem
    uses the same None-handling semantics.  All properties return safe
    defaults (empty list, empty dict) when the schema is absent or
    incomplete — callers never need to guard against ``None``.

    The canonical way to obtain a ``StageSchema`` is
    ``StageSchema.from_stage(stage)``; no other wizard code should call
    ``stage.get("schema")`` directly.

    .. note::

       ``frozen=True`` prevents re-assigning ``_raw`` but does **not**
       prevent mutation of the underlying dict.  The ``raw`` property
       returns the same dict reference, so callers that need a mutable
       copy (e.g. ``_strip_schema_defaults``) must ``copy.deepcopy``
       it themselves — which they already do.
    """

    _raw: dict[str, Any]

    @classmethod
    def from_stage(cls, stage: dict[str, Any]) -> StageSchema:
        """Create from a stage metadata dict.

        Args:
            stage: Stage configuration dict, optionally containing a
                ``"schema"`` key.

        Returns:
            ``StageSchema`` wrapping the raw schema (or empty dict).
        """
        return cls(_raw=stage.get("schema") or {})

    @classmethod
    def from_dict(cls, schema: dict[str, Any]) -> StageSchema:
        """Create from a raw JSON Schema dict.

        Use this when you have the schema dict directly (e.g. a
        constructed focused or amendment schema) rather than a stage
        metadata dict.

        Args:
            schema: JSON Schema dict.

        Returns:
            ``StageSchema`` wrapping the provided dict.
        """
        return cls(_raw=schema)

    @property
    def exists(self) -> bool:
        """Whether the stage has a non-empty schema.

        Returns ``False`` for both missing schemas (``None``) and
        empty schemas (``{}``), since neither defines extractable
        fields.
        """
        return bool(self._raw)

    @property
    def required_fields(self) -> list[str]:
        """Required field names (empty list if none or no schema)."""
        return self._raw.get("required", [])

    @property
    def has_required_fields(self) -> bool:
        """Whether there are any required fields."""
        return bool(self.required_fields)

    @property
    def properties(self) -> dict[str, Any]:
        """Schema properties dict (empty if none or no schema)."""
        return self._raw.get("properties", {})

    @property
    def property_names(self) -> set[str]:
        """Set of all property names."""
        return set(self.properties.keys())

    def get_property(self, name: str) -> dict[str, Any]:
        """Get a single property definition (empty dict if not found)."""
        return self.properties.get(name, {})

    def field_type(self, name: str) -> str | None:
        """Get the declared type of a field (None if not found)."""
        return self.get_property(name).get("type")

    @property
    def raw(self) -> dict[str, Any]:
        """Raw schema dict for passing to extractor/validator.

        Returns the underlying dict (empty dict if no schema was
        defined).  Use ``exists`` to distinguish "no schema" from
        "empty schema" when the distinction matters.
        """
        return self._raw

    def can_satisfy_required(self, data: dict[str, Any]) -> bool:
        """Check if all required fields are present in data.

        Returns ``True`` (vacuous satisfaction) for:
        - No schema (no required fields)
        - Schema with ``required: []``
        - All required fields present in *data* (value is not ``None``)
        """
        return all(data.get(f) is not None for f in self.required_fields)

    def missing_required(self, data: dict[str, Any]) -> set[str]:
        """Required field names not yet present in data."""
        return {f for f in self.required_fields if data.get(f) is None}


# ---------------------------------------------------------------------------
# Navigation keyword configuration
# ---------------------------------------------------------------------------


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


# =========================================================================
# Helper functions
# =========================================================================


def validate_strategy_names(
    default_reasoning: str, wizard_fsm: WizardFSM,
) -> None:
    """Warn at config-load time if strategy names are unregistered.

    This catches typos in built-in strategy names early rather than
    failing at runtime when the wizard first enters the misconfigured
    stage.  3rd-party strategies registered after wizard construction
    will produce false-positive warnings — the actual resolution at
    stage entry time will find them.
    """
    from .registry import get_registry

    registry = get_registry()
    names_to_check: list[tuple[str, str]] = []

    if default_reasoning.lower() != "single":
        names_to_check.append((default_reasoning, "wizard settings.tool_reasoning"))

    for state_name, meta in wizard_fsm.stages.items():
        stage_reasoning = meta.get("reasoning") if meta else None
        if stage_reasoning and stage_reasoning.lower() != "single":
            names_to_check.append((stage_reasoning, f"stage '{state_name}'.reasoning"))

    for name, location in names_to_check:
        if not registry.is_registered(name.lower()):
            logger.warning(
                "Unknown reasoning strategy '%s' in %s; "
                "registered strategies: %s",
                name, location, registry.list_keys(),
            )


def is_json_safe(value: Any) -> bool:
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
            isinstance(k, str) and is_json_safe(v) for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(is_json_safe(item) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return all(
            is_json_safe(getattr(value, f.name))
            for f in dataclasses.fields(value)
        )
    return False


def load_merge_filter(dotted_path: str) -> MergeFilter:
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
    # ``filter`` attribute would pass this check.
    if not isinstance(instance, MergeFilter):
        raise ConfigurationError(
            f"Merge filter {dotted_path!r} does not implement "
            "the MergeFilter protocol (must have a 'filter' method)",
            context={"merge_filter": dotted_path},
        )
    return instance


def normalize_enum_value(
    value: str,
    enum_values: list[str],
    *,
    threshold: float = 0.7,
) -> str | None:
    """Normalize an extracted value to the closest enum match.

    Returns the canonical enum value if a match is found above the
    *threshold*, or ``None`` if no match is close enough.

    Matching tiers (first match wins):

    1. Exact match (case-sensitive)
    2. Case-insensitive match
    3. Substring match (enum value ⊆ extracted or extracted ⊆ enum,
       after underscore/hyphen normalisation)
    4. Token overlap ≥ *threshold*

    Args:
        value: Raw extracted string.
        enum_values: Canonical enum entries from the JSON Schema.
        threshold: Minimum token-overlap score for tier-4 matching.

    Returns:
        Canonical enum entry, or ``None`` when no match qualifies.
    """
    if not value or not enum_values:
        return None

    # Filter to string entries only — JSON Schema enums may contain
    # integers or other types that would crash .lower() calls.
    str_enums = [ev for ev in enum_values if isinstance(ev, str)]
    if not str_enums:
        return None

    # Tier 1: exact match
    if value in str_enums:
        return value

    # Tier 2: case-insensitive
    value_lower = value.lower()
    for ev in str_enums:
        if ev.lower() == value_lower:
            return ev

    # Tier 3: whole-word substring match (bidirectional)
    # "tutor bot" contains "tutor"; "study_companion" contains "study"
    # Uses word-boundary anchors to prevent false positives like
    # "no" matching "nobody" or "base" matching "database".
    # When multiple enums match, prefer the longest match to avoid
    # schema-order-dependent ambiguity.
    val_norm = value_lower.replace("_", " ").replace("-", " ")
    if len(val_norm) >= 2:
        tier3_match: str | None = None
        tier3_len = 0
        for ev in str_enums:
            ev_norm = ev.lower().replace("_", " ").replace("-", " ")
            if len(ev_norm) < 2:
                continue
            if word_in_text(ev_norm, val_norm) or word_in_text(val_norm, ev_norm):
                if len(ev_norm) > tier3_len:
                    tier3_match = ev
                    tier3_len = len(ev_norm)
        if tier3_match is not None:
            return tier3_match

    # Tier 4: token overlap
    val_tokens = set(val_norm.split())
    best_match: str | None = None
    best_score = 0.0
    for ev in str_enums:
        ev_tokens = set(
            ev.lower().replace("_", " ").replace("-", " ").split()
        )
        if not ev_tokens:
            continue
        overlap = val_tokens & ev_tokens
        score = len(overlap) / max(len(val_tokens), len(ev_tokens))
        if score > best_score:
            best_score = score
            best_match = ev

    if best_score >= threshold and best_match is not None:
        return best_match

    return None


# Backward-compatible aliases for underscore-prefixed names.
# wizard.py re-exports these under the old names.
_is_json_safe = is_json_safe
_load_merge_filter = load_merge_filter
_normalize_enum_value = normalize_enum_value
_validate_strategy_names = validate_strategy_names
