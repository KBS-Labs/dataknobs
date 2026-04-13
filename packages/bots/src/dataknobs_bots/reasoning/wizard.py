"""Wizard reasoning strategy for guided conversational flows.

This module implements FSM-backed reasoning for DynaBot, enabling
guided conversational wizard flows with validation, data collection,
and branching logic.

Data types, constants, and standalone helpers live in
:mod:`wizard_types` (extracted in item 77a).  This module re-exports
them for backward compatibility.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataknobs_common.serialization import sanitize_for_json
from dataknobs_llm.conversations.storage import ConversationNode, get_node_by_id

from dataknobs_llm import LLMStreamResponse

from .base import ProcessResult, ReasoningStrategy, StreamStageContext, ToolCallSpec, TurnHandle
from .observability import (
    TransitionRecord,
    WizardStateSnapshot,
    WizardTaskList,
    create_transition_record,
)
from .wizard_derivations import (
    DerivationRule,
    parse_derivation_rules,
)
from .wizard_extraction import WizardExtractor
from .wizard_grounding import (
    CompositeMergeFilter,
    MergeFilter,
    SchemaGroundingFilter,
)
from .wizard_hooks import WizardHooks
from .wizard_navigation import WizardNavigator
from .wizard_response import WizardResponder
from .wizard_types import (  # noqa: F401 — re-exports for backward compat
    DEFAULT_BACK_KEYWORDS,
    DEFAULT_EPHEMERAL_KEYS,
    DEFAULT_RECOVERY_PIPELINE,
    DEFAULT_RESTART_KEYWORDS,
    DEFAULT_SKIP_KEYWORDS,
    ExtractionPipelineResult,
    NavigationCommandConfig,
    NavigationConfig,
    RECOVERY_BOOLEAN,
    RECOVERY_CLARIFICATION,
    RECOVERY_DERIVATION,
    RECOVERY_FOCUSED_RETRY,
    RECOVERY_SCOPE_ESCALATION,
    SCOPE_BREADTH,
    StageSchema,
    SubflowContext,
    ToolResultMappingEntry,
    TurnContext,
    VALID_RECOVERY_STRATEGIES,
    WizardAdvanceResult,
    WizardStageContext,
    WizardState,
    WizardTurnHandle,
    _DEFAULT_AFFIRMATIVE_PHRASES,
    _DEFAULT_AFFIRMATIVE_SIGNALS,
    _DEFAULT_NEGATIVE_PHRASES,
    _DEFAULT_NEGATIVE_SIGNALS,
    _is_json_safe,
    _load_merge_filter,
    _normalize_enum_value,
    _validate_strategy_names,
    is_json_safe,
    load_merge_filter,
    normalize_enum_value,
    validate_strategy_names,
)
from .wizard_types import FinalizePreambleResult
from .wizard_subflows import SubflowManager
from .wizard_tasks import (
    build_initial_tasks,
    update_field_tasks,
    update_stage_exit_tasks,
    update_tool_tasks,
)

if TYPE_CHECKING:
    from dataknobs_bots.bot.turn import ToolExecution
    from dataknobs_data import SyncDatabase

    from .wizard_fsm import WizardFSM

logger = logging.getLogger(__name__)


# Re-exports for backward compatibility are handled by the import block
# above (from .wizard_types import ...).  All public names that were
# previously defined here are still importable from this module.


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
        _navigator: WizardNavigator handling navigation commands and amendments
        _extraction: WizardExtractor handling the extraction pipeline
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
        skip_builtin_grounding: bool = False,
        grounding_overlap_threshold: float = 0.5,
        scope_escalation_enabled: bool = False,
        scope_escalation_scope: str = "wizard_session",
        recent_messages_count: int = 3,
        field_derivations: list[DerivationRule] | None = None,
        enum_normalize: bool = True,
        normalize_threshold: float = 0.7,
        reject_unmatched: bool = True,
        boolean_recovery: bool = True,
        recovery_pipeline: list[str] | None = None,
        focused_retry_enabled: bool = False,
        focused_retry_max_retries: int = 1,
        clarification_groups: list[dict[str, Any]] | None = None,
        clarification_exclude_derivable: bool = False,
        clarification_template: str | None = None,
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
            merge_filter: Custom :class:`MergeFilter` implementation for
                domain-specific validation.  When provided alongside
                ``extraction_grounding=True``, both compose via
                :class:`CompositeMergeFilter` (grounding runs first,
                then the custom filter).  Set
                ``skip_builtin_grounding=True`` to bypass grounding and
                run only the custom filter.
            skip_builtin_grounding: When True and a ``merge_filter`` is
                provided, skip the built-in grounding check entirely.
                The custom filter is responsible for all validation.
                Has no effect when no ``merge_filter`` is set.
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
            enum_normalize: When True (default), normalize extracted enum
                values to canonical enum entries via case-insensitive and
                fuzzy matching.  Per-field ``x-extraction.normalize``
                overrides this setting.  Configured under
                ``extraction_hints.enum_normalize`` in wizard settings.
            normalize_threshold: Minimum token-overlap score (0.0--1.0)
                for fuzzy enum matching.  Defaults to 0.7.  Only used
                when enum normalization is active.  Per-field
                ``x-extraction.normalize_threshold`` overrides.
            reject_unmatched: When True (default), enum values that are
                not valid entries after normalization are rejected — the
                field is set to None so it is not merged into wizard
                data.  Works independently of normalization: when
                normalization is disabled, acts as a strict enum
                membership check.  Per-field
                ``x-extraction.reject_unmatched`` overrides.  Configured
                under ``extraction_hints.reject_unmatched`` in wizard
                settings YAML.
            boolean_recovery: When True (default), enable deterministic
                signal-word recovery for boolean fields that extraction
                failed to fill.  Scans the user's message for
                affirmative/negative signal words and sets the field
                value accordingly.  Only runs when ``"boolean_recovery"``
                is included in the recovery pipeline.  Per-field
                ``x-extraction.boolean_recovery`` overrides.  Configured
                under ``extraction_hints.boolean_recovery`` in wizard
                settings YAML.
            recovery_pipeline: Ordered list of recovery strategy names
                to execute when required fields are missing after
                initial extraction.  Valid names: ``"derivation"``,
                ``"scope_escalation"``, ``"focused_retry"``,
                ``"boolean_recovery"``, ``"clarification"``.  Defaults
                to ``["derivation", "scope_escalation"]``.  Add
                ``"boolean_recovery"`` or ``"focused_retry"`` explicitly
                to opt in.
                The pipeline short-circuits when all required fields
                are satisfied.
            focused_retry_enabled: When True, the ``"focused_retry"``
                pipeline strategy builds a minimal schema with only the
                missing required fields and re-extracts.  Defaults to
                False — consumers must opt in.
            focused_retry_max_retries: Maximum number of focused retry
                attempts per turn.  Defaults to 1.
            clarification_groups: List of field group dicts for structured
                clarification questions.  Each dict has ``fields``
                (list of field names) and ``question`` (human-readable
                question text).  When configured, the clarification
                response groups related missing fields into natural
                questions instead of a generic list.
            clarification_exclude_derivable: When True, exclude fields
                that have derivation rules from clarification questions
                (they'll be derived once a source field is provided).
                This applies even when the source is also missing —
                the clarification prompt will ask for the source, and
                derivation fills the target from the user's answer.
            clarification_template: Optional Jinja2 template string for
                rendering clarification questions.  Receives a
                ``field_groups`` variable (list of dicts with ``fields``
                and ``question`` keys).
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
        self._allow_amendments = allow_post_completion_edits
        self._section_to_stage_mapping = section_to_stage_mapping or {}

        # Validate strategy names at construction time
        _validate_strategy_names(default_tool_reasoning, wizard_fsm)
        self._artifact_registry = artifact_registry
        self._review_executor = review_executor
        self._context_builder = context_builder
        # Field derivations needed by _apply_transition_derivations (stays
        # in wizard.py) and WizardResponder — kept as instance attribute.
        self._field_derivations = field_derivations or []
        self._initial_data: dict[str, Any] = initial_data or {}
        self._consistent_navigation_lifecycle = consistent_navigation_lifecycle

        # Build merge filter chain: grounding (if enabled) → custom.
        # The composite filter is passed directly to WizardExtractor;
        # no instance attribute needed.
        filters: list[MergeFilter] = []
        if extraction_grounding and not skip_builtin_grounding:
            filters.append(SchemaGroundingFilter(
                overlap_threshold=grounding_overlap_threshold,
            ))
        if merge_filter is not None:
            filters.append(merge_filter)
        if len(filters) > 1:
            _merge_filter: MergeFilter | None = (
                CompositeMergeFilter(filters)
            )
        elif len(filters) == 1:
            _merge_filter = filters[0]
        else:
            _merge_filter = None

        # Validate recovery pipeline (local — passed to WizardExtractor).
        if recovery_pipeline is not None:
            unknown = set(recovery_pipeline) - VALID_RECOVERY_STRATEGIES
            if unknown:
                logger.warning(
                    "Unknown recovery strategies %s — removing. "
                    "Valid: %s",
                    sorted(unknown),
                    sorted(VALID_RECOVERY_STRATEGIES),
                )
            _validated_pipeline = [
                s for s in recovery_pipeline
                if s in VALID_RECOVERY_STRATEGIES
            ]
        else:
            _validated_pipeline = list(DEFAULT_RECOVERY_PIPELINE)

        # Consolidated rendering layer — routes all template rendering
        # through a single class with consistent context, sandboxing,
        # and error handling.
        from dataknobs_bots.reasoning.wizard_renderer import WizardRenderer

        self._renderer = WizardRenderer()

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

        # Extraction pipeline — handles extract, normalize, merge,
        # defaults, derivations, recovery (extracted in item 77c).
        # Extraction-config params are passed directly (no instance attrs).
        self._extraction = WizardExtractor(
            extractor=self._extractor,
            merge_filter=_merge_filter,
            grounding_overlap_threshold=grounding_overlap_threshold,
            enum_normalize=enum_normalize,
            normalize_threshold=normalize_threshold,
            reject_unmatched=reject_unmatched,
            extraction_scope=extraction_scope,
            recent_messages_count=recent_messages_count,
            conflict_strategy=conflict_strategy,
            log_conflicts=log_conflicts,
            per_turn_keys=self._per_turn_keys,
            recovery_pipeline=_validated_pipeline,
            boolean_recovery=boolean_recovery,
            scope_escalation_enabled=scope_escalation_enabled,
            scope_escalation_scope=scope_escalation_scope,
            focused_retry_enabled=focused_retry_enabled,
            focused_retry_max_retries=max(1, focused_retry_max_retries),
            field_derivations=self._field_derivations,
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

        # Subflow manager — owns active subflow FSM and push/pop lifecycle.
        # Created with a placeholder evaluate_condition; patched below
        # after WizardResponder is constructed (circular dependency:
        # SubflowManager needs evaluate_condition from WizardResponder,
        # WizardResponder needs SubflowManager as a shared reference).
        self._subflows = SubflowManager(
            fsm=wizard_fsm,
            evaluate_condition=lambda _c, _d: False,
        )

        # Response generation module — handles all response paths,
        # auto-advance, condition evaluation, strategy delegation
        # (extracted in item 77d).
        self._response = WizardResponder(
            renderer=self._renderer,
            fsm=self._fsm,
            subflows=self._subflows,
            context_template=context_template,
            auto_advance_filled_stages=auto_advance_filled_stages,
            default_tool_reasoning=default_tool_reasoning,
            default_max_iterations=default_max_iterations,
            default_store_trace=default_store_trace,
            default_verbose=default_verbose,
            strict_validation=self._strict_validation,
            field_derivations=self._field_derivations,
            clarification_groups=clarification_groups or [],
            clarification_exclude_derivable=clarification_exclude_derivable,
            clarification_template=clarification_template,
            build_wizard_metadata=self._build_wizard_metadata,
            execute_fsm_step=self._execute_fsm_step,
            make_bank_accessor=self._make_bank_accessor,
            get_artifact=lambda: self._artifact,
            get_catalog=lambda: self._catalog,
            get_artifact_registry=lambda: self._artifact_registry,
            get_review_executor=lambda: self._review_executor,
            get_context_builder=lambda: self._context_builder,
            get_banks=lambda: self._banks,
        )

        # Inject the real condition evaluator now that the responder
        # is constructed (resolves the circular dependency).
        self._subflows.set_evaluate_condition(self._response.evaluate_condition)

        # Build navigation keyword config from wizard-level settings
        nav_settings = wizard_fsm.settings.get("navigation", {})
        self._navigation_config: NavigationConfig = NavigationConfig.from_dict(
            nav_settings or {}
        )

        # Navigation module — handles back/skip/restart, amendments,
        # and conversation tree branching (extracted in item 77b).
        self._navigator = WizardNavigator(
            fsm=self._fsm,
            subflows=self._subflows,
            hooks=self._hooks,
            navigation_config=self._navigation_config,
            consistent_lifecycle=self._consistent_navigation_lifecycle,
            allow_amendments=self._allow_amendments,
            section_to_stage_mapping=self._section_to_stage_mapping,
            extractor=self._extractor,
            banks=self._banks,
            artifact=self._artifact,
            catalog=self._catalog,
            execute_fsm_step=self._execute_fsm_step,
            run_post_transition_lifecycle=self._run_post_transition_lifecycle,
            generate_stage_response=self._generate_stage_response_for_nav,
            prepend_messages_to_response=WizardResponder.prepend_messages_to_response,
        )

        # Register the fallback factory via set_transform_context_factory.
        # The primary factory is a per-call closure installed by
        # _execute_fsm_step; this fallback handles initial-state entry.
        self._wizard_fsm = wizard_fsm
        self._wizard_fsm.set_transform_context_factory(
            self._build_transform_context
        )

    def _build_transform_context(self, func_context: Any) -> Any:
        """Fallback transform context factory.

        The primary factory is a per-call closure installed by
        ``_execute_fsm_step``.  This fallback provides a safe default
        if ``step_async`` is ever called outside that method (e.g.
        during initial-state entry before any turn begins).

        Args:
            func_context: The :class:`FunctionContext` built by the FSM.

        Returns:
            ``TransformContext`` with wizard services and FSM context.
        """
        from ..artifacts.transforms import TransformContext

        return TransformContext(
            fsm_context=func_context,
            turn=None,
            artifact_registry=self._artifact_registry,
            rubric_executor=self._review_executor,
            config={},
            banks=self._banks,
        )

    async def _generate_stage_response_for_nav(
        self,
        manager: Any,
        llm: Any,
        stage: dict[str, Any],
        state: WizardState,
        tools: list[Any] | None,
    ) -> Any:
        """Wrapper for navigator callback — unwraps StageResponseResult.

        Uses the default ``track_render=True`` because all navigation
        paths (back, skip, restart, amendment) should record renders.

        Note: prior to the render-count consolidation (dk-42),
        ``handle_amendment`` did not increment render_count while
        back/skip/restart did.  This was an inconsistency — amendment
        re-opens a stage and renders its response, so it should track
        the render like the other navigation paths.
        """
        result = await self._response.generate_stage_response(
            manager, llm, stage, state, tools,
        )
        return result.response

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
        if WizardExtractor.is_done_signal(user_message, done_keywords):
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
        schema_props = StageSchema.from_stage(stage).property_names
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
        await self._navigator.branch_for_revisited_stage(
            manager, stage.get("name", ""),
        )

        # Render the stage response
        stage_result = await self._response.generate_stage_response(
            manager, llm, stage, state, tools,
        )
        await self._save_wizard_state(manager, state)
        return stage_result.response

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

        Also updates the navigator's and extraction pipeline's extractor
        references so amendment detection and data extraction see the new
        extractor without requiring a separate sync step.

        Args:
            extractor: New extractor instance.
        """
        self._extractor = extractor
        self._navigator._extractor = extractor
        self._extraction._extractor = extractor

    def set_hooks(self, hooks: WizardHooks | None) -> None:
        """Replace the hooks instance.

        Also updates the navigator's hooks reference so lifecycle
        events fire correctly through the new hooks.

        Args:
            hooks: New hooks instance, or None to disable hooks.
        """
        self._hooks = hooks
        self._navigator._hooks = hooks

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
    def from_config(cls, config: dict[str, Any], **_kwargs: Any) -> WizardReasoning:  # type: ignore[override]
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
        skip_builtin_grounding = wizard_fsm.settings.get(
            "skip_builtin_grounding", False,
        )

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

        # Load extraction hints (class-level normalization settings)
        extraction_hints = wizard_fsm.settings.get("extraction_hints") or {}
        enum_normalize = extraction_hints.get("enum_normalize", True)
        normalize_threshold = extraction_hints.get("normalize_threshold", 0.7)
        reject_unmatched = extraction_hints.get("reject_unmatched", True)
        boolean_recovery = extraction_hints.get("boolean_recovery", True)

        # Load recovery pipeline settings
        recovery_config = wizard_fsm.settings.get("recovery", {})
        recovery_pipeline: list[str] | None = recovery_config.get("pipeline")
        focused_retry_config = recovery_config.get("focused_retry", {})
        focused_retry_enabled = focused_retry_config.get("enabled", False)
        focused_retry_max_retries = focused_retry_config.get("max_retries", 1)

        # Load clarification grouping settings
        clarification_config = recovery_config.get("clarification", {})
        clarification_groups: list[dict[str, Any]] = (
            clarification_config.get("groups", [])
        )
        clarification_exclude_derivable: bool = (
            clarification_config.get("exclude_derivable", False)
        )
        clarification_template: str | None = (
            clarification_config.get("template")
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
            skip_builtin_grounding=skip_builtin_grounding,
            grounding_overlap_threshold=grounding_overlap_threshold,
            scope_escalation_enabled=scope_escalation_enabled,
            scope_escalation_scope=scope_escalation_scope,
            recent_messages_count=recent_messages_count,
            field_derivations=field_derivations,
            enum_normalize=enum_normalize,
            normalize_threshold=normalize_threshold,
            reject_unmatched=reject_unmatched,
            boolean_recovery=boolean_recovery,
            recovery_pipeline=recovery_pipeline,
            focused_retry_enabled=focused_retry_enabled,
            focused_retry_max_retries=focused_retry_max_retries,
            clarification_groups=clarification_groups,
            clarification_exclude_derivable=clarification_exclude_derivable,
            clarification_template=clarification_template,
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
        active_fsm = self._subflows.get_active_fsm()
        stage = active_fsm.current_metadata
        await self._navigator.branch_for_revisited_stage(manager, stage.get("name", ""))
        stage_result = await self._response.generate_stage_response(
            manager, llm, stage, wizard_state, tools=[],
        )
        response = stage_result.response

        # Auto-advance through message stages if the start stage has
        # auto_advance: true. The start stage response is already captured,
        # so skip_first_render=True avoids re-rendering it.
        if self._response.can_auto_advance(wizard_state, stage):
            auto_advance_messages = [response.content]
            loop_messages = await self._response.run_auto_advance_loop(
                wizard_state, active_fsm, stage,
                skip_first_render=True, llm=llm,
            )
            auto_advance_messages.extend(loop_messages)

            # Re-fetch active_fsm in case a subflow pop occurred
            active_fsm = self._subflows.get_active_fsm()

            # Generate the landing stage's response and prepend
            # the collected messages from auto-advanced stages.
            landing_stage = active_fsm.current_metadata
            if landing_stage.get("name") != stage.get("name"):
                await self._navigator.branch_for_revisited_stage(
                    manager, landing_stage.get("name", "")
                )
                stage_result = await self._response.generate_stage_response(
                    manager, llm, landing_stage, wizard_state, tools=[],
                )
                response = stage_result.response
                WizardResponder.prepend_messages_to_response(
                    response, auto_advance_messages
                )

            # Clear skip_extraction — the user's first message after greet()
            # IS directed at the landing stage (unlike generate() auto-advance
            # where the user's message was directed at the previous stage).
            wizard_state.skip_extraction = False

        # Persist wizard state
        await self._save_wizard_state(manager, wizard_state)

        return response

    # =========================================================================
    # Phased Turn Protocol (PhasedReasoningProtocol)
    # =========================================================================
    #
    # The phased protocol splits generate() into three phases:
    #   1. begin_turn — restore state, handle navigation/amendments
    #   2. process_input — extract data, validate, handle collection modes
    #   3. finalize_turn — FSM transition, response generation, save state
    #
    # DynaBot can interleave tool execution between process_input and
    # finalize_turn, enabling tool results to update wizard state
    # (e.g. update_tool_tasks) before state is saved.
    #
    # generate() is retained as a backward-compatible wrapper.

    async def begin_turn(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> WizardTurnHandle:
        """Phase A: Restore state, handle navigation and amendments.

        Restores wizard state from conversation metadata, clears per-turn
        transient keys, and handles navigation commands and post-completion
        amendments.  If the turn resolves to a navigation or amendment
        response, ``handle.early_response`` is set.

        Args:
            manager: ConversationManager instance.
            llm: LLM provider instance.
            tools: Optional list of available tools.
            **kwargs: Additional generation parameters.

        Returns:
            :class:`WizardTurnHandle` with wizard state and user message.
        """
        handle = WizardTurnHandle(
            manager=manager, llm=llm, tools=tools, kwargs=kwargs,
        )

        # Get or restore wizard state
        wizard_state = self._get_wizard_state(manager)
        handle.wizard_state = wizard_state

        # Clear per-turn keys from previous turn to prevent stale values
        for key in self._per_turn_keys:
            wizard_state.data.pop(key, None)
            wizard_state.transient.pop(key, None)

        # Get user message
        handle.user_message = self._get_last_user_message(manager)

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
            amendment_response = await self._navigator.handle_amendment(
                handle.user_message, wizard_state, manager, llm, tools,
            )
            if amendment_response:
                await self._save_wizard_state(manager, wizard_state)
                handle.early_response = amendment_response
                return handle

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
            await self._navigator.restart_cleanup(
                wizard_state, handle.user_message, trigger="auto_restart"
            )
            # Branch the conversation tree so the new recipe's context
            # starts fresh — the LLM won't see the old recipe's detailed
            # tool calls and record-level operations.
            await self._navigator.branch_for_revisited_stage(
                manager, wizard_state.current_stage
            )

        # Handle navigation commands
        nav_result = await self._navigator.handle_navigation(
            handle.user_message, wizard_state, manager, llm
        )
        if nav_result:
            await self._save_wizard_state(manager, wizard_state)
            handle.early_response = nav_result
            return handle

        # Capture skip_extraction flag for process_input
        handle.skip_extraction = wizard_state.skip_extraction

        return handle

    async def process_input(
        self,
        handle: TurnHandle,
    ) -> ProcessResult:
        """Phase B: Extract data, validate, prepare for transition.

        Processes the user's input through the extraction pipeline,
        validates against stage schema, and handles collection modes.
        If the turn resolves to an early response (clarification,
        validation error, etc.), ``result.early_response`` is set.

        Args:
            handle: Turn handle from :meth:`begin_turn`.

        Returns:
            :class:`ProcessResult` indicating outcome and whether
            DynaBot should execute tools before :meth:`finalize_turn`.
        """
        if not isinstance(handle, WizardTurnHandle):
            raise TypeError(
                f"WizardReasoning.process_input requires WizardTurnHandle, "
                f"got {type(handle).__name__}"
            )
        result = ProcessResult()
        wizard_state = handle.wizard_state
        if wizard_state is None:
            raise ValueError("WizardTurnHandle.wizard_state is None — begin_turn must be called first")
        manager = handle.manager
        llm = handle.llm
        tools = handle.tools
        user_message = handle.user_message

        # Get current stage context from active FSM (subflow or main)
        active_fsm = self._subflows.get_active_fsm()
        stage = active_fsm.current_metadata
        is_conversation = stage.get("mode") == "conversation"

        # ── Skip extraction on auto-advance landing stage ──
        # When auto-advance lands on a new stage, the user hasn't responded
        # to it yet.  Their message was directed at the previous stage.
        # Consume the flag (one-shot) so extraction runs normally next turn.
        _skip_extraction = handle.skip_extraction
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
            if WizardExtractor.is_done_signal(user_message, done_keywords):
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
            collection_intent = WizardExtractor.classify_collection_intent(
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
            result.action = "skip"
        elif is_conversation:
            # Conversation mode: skip extraction, run intent detection.
            # Field derivations are also skipped — no new data extracted,
            # so there is nothing to derive from.
            stage_name = stage.get("name", "unknown")
            logger.debug(
                "Conversation mode for stage '%s': skipping extraction",
                stage_name,
            )
            await self._extraction.detect_intent(user_message, stage, wizard_state, llm)
            result.action = "intent_only"
        elif _collection_done_signal:
            # Done keyword detected — skip extraction (and derivation)
            # and fall through to transition evaluation below.
            result.action = "collection_done"
        elif _collection_help:
            # Help request during collection — generate a contextual
            # response without extraction, then return immediately so
            # the stage loops for the next data input.
            stage_context = self._response.build_stage_context(stage, wizard_state)
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
                tools=self._response.filter_tools_for_stage(stage, tools),
                metadata=wizard_snapshot,
            )
            self._response.add_wizard_metadata(response, wizard_state, stage)
            await self._save_wizard_state(manager, wizard_state)
            result.early_response = response
            result.action = "collection_help"
            return result
        else:
            # Structured mode: extract data and validate

            # Run intent detection on structured stages IF configured.
            # This runs before extraction so that detour intents
            # (e.g. "help", "confused") can skip extraction entirely
            # and trigger transitions via transition conditions.
            if stage.get("intent_detection"):
                await self._extraction.detect_intent(
                    user_message, stage, wizard_state, llm
                )

            # If an intent was detected, skip extraction/validation and
            # proceed to transition evaluation so the detour can fire.
            if "_intent" not in wizard_state.data:
                # Run the shared extraction pipeline (extract, normalize,
                # merge, defaults, derivations, recovery).
                pipeline_result = await self._extraction.run_extraction_pipeline(
                    user_message, stage, wizard_state, llm,
                    manager=manager,
                )
                extraction = pipeline_result.extraction
                new_data_keys = pipeline_result.new_data_keys
                stage_name = stage.get("name", "unknown")

                # ── Confidence gate ──
                # After merge, check whether we have enough data to
                # proceed.  If confidence is low and required fields are
                # still missing, ask for clarification — but the partial
                # data is preserved in wizard_state.data for the next turn.
                if not pipeline_result.is_confident:
                    wizard_state.clarification_attempts += 1
                    await self._save_wizard_state(manager, wizard_state)

                    if wizard_state.clarification_attempts >= 3:
                        response = await self._response.generate_restart_offer(
                            manager, llm, stage, extraction.errors,
                            tools=tools, wizard_state=wizard_state,
                        )
                    else:
                        response = (
                            await self._response.generate_clarification_response(
                                manager, llm, stage, extraction.errors,
                                tools=tools, wizard_state=wizard_state,
                            )
                        )

                    self._response.add_wizard_metadata(
                        response, wizard_state, stage
                    )
                    result.early_response = response
                    result.action = "clarification"
                    return result

                # Reset clarification attempts on viable extraction
                wizard_state.clarification_attempts = 0

                # Update field-extraction tasks from the full accumulated
                # state.  This runs after the confidence gate so tasks are
                # not marked complete on clarification turns (where the
                # early return fires before reaching this line).  We pass
                # wizard_state.data (not extraction.data) because fields
                # merged during prior clarification turns also need their
                # tasks marked complete once the wizard proceeds.
                update_field_tasks(wizard_state, wizard_state.data)

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
                        result.early_response = col_response
                        result.action = "collection_loop"
                        return result

                # When meaningful new data was extracted at a stage with a
                # response_template, decide whether to render a
                # confirmation before evaluating transitions.
                #
                # Two modes:
                # 1. First-render (render_counts == 0): confirm when new
                #    data exists, unless confirm_first_render is false.
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
                        # First render — confirm unless stage opts out
                        if stage.get("confirm_first_render", True) is not False:
                            should_confirm = True
                    elif stage.get("confirm_on_new_data"):
                        # Re-confirm when schema property values changed
                        ss_props = StageSchema.from_stage(stage).property_names
                        current_snapshot = {
                            k: wizard_state.data[k]
                            for k in ss_props
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
                        wizard_state.save_stage_snapshot(
                            stage_name,
                            StageSchema.from_stage(stage).property_names,
                        )
                    logger.debug(
                        "New data extracted (%s) at stage '%s' — "
                        "rendering confirmation (render #%d)",
                        new_data_keys,
                        stage_name,
                        render_count,
                    )
                    confirmation_content = (
                        self._response.build_confirmation_content(
                            stage,
                            wizard_state,
                            new_data_keys,
                        )
                    )
                    response = self._response.create_template_response(
                        confirmation_content
                    )
                    self._response.add_wizard_metadata(
                        response, wizard_state, stage
                    )
                    # Persist confirmation to conversation history so the
                    # LLM sees it on subsequent turns (matches the old
                    # generate_stage_response path via
                    # _resolve_template_content).
                    wizard_snapshot = {
                        "wizard": self._response._build_wizard_metadata(
                            wizard_state
                        )
                    }
                    await manager.add_message(
                        role="assistant",
                        content=confirmation_content,
                        metadata=wizard_snapshot,
                    )
                    await self._save_wizard_state(manager, wizard_state)
                    result.early_response = response
                    result.action = "confirmation"
                    return result

                # Validate against stage schema
                ss_validate = StageSchema.from_stage(stage)
                if ss_validate.exists and self._strict_validation:
                    validation_errors = self._extraction.validate_data(
                        wizard_state.data, ss_validate
                    )
                    if validation_errors:
                        # Save state before returning validation error
                        await self._save_wizard_state(manager, wizard_state)
                        response = await self._response.generate_validation_response(
                            manager, llm, stage, validation_errors,
                            tools=tools, wizard_state=wizard_state,
                        )
                        self._response.add_wizard_metadata(response, wizard_state, stage)
                        result.early_response = response
                        result.action = "validation_error"
                        return result

            result.action = "extracted"

            # Check for tool_result_mapping on current stage.
            # When present, build ToolCallSpec list from extracted state
            # and signal DynaBot to execute tools before finalize_turn.
            # Guard: only fire when extraction actually ran (not intent-only
            # turns where _intent short-circuited extraction).
            raw_trm = stage.get("tool_result_mapping", [])
            if raw_trm and "_intent" not in wizard_state.data:
                parsed_entries = [
                    ToolResultMappingEntry(
                        tool_name=entry["tool"],
                        params=entry.get("params", {}),
                        mapping=entry.get("mapping", {}),
                        on_error=entry.get("on_error", "skip"),
                    )
                    for entry in raw_trm
                ]
                specs = []
                for trm_entry in parsed_entries:
                    params: dict[str, Any] = {}
                    for param_name, state_key in trm_entry.params.items():
                        value = wizard_state.data.get(state_key)
                        if value is not None:
                            params[param_name] = value
                    specs.append(ToolCallSpec(
                        name=trm_entry.tool_name,
                        parameters=params,
                    ))
                result.pending_tool_calls = specs
                result.needs_tool_execution = True
                handle.tool_result_mapping = parsed_entries

        return result

    async def _finalize_preamble(
        self,
        handle: WizardTurnHandle,
        tool_results: list[ToolExecution] | None,
    ) -> FinalizePreambleResult:
        """Shared pre-response setup for finalize_turn / stream_finalize_turn.

        Runs exit hooks, applies tool results and tool_result_mapping,
        checks for subflow pushes, runs FSM transition, and executes
        post-transition lifecycle.

        Returns a :class:`FinalizePreambleResult` whose ``subflow_pushed``
        flag tells the caller whether to generate a subflow response
        (early path) or a normal stage response (main path).

        Args:
            handle: Validated wizard turn handle.
            tool_results: Tool execution records from DynaBot's tool
                loop (``None`` when no tools were executed).

        Returns:
            Preamble result with either subflow push info or full
            transition output.
        """
        wizard_state = handle.wizard_state
        if wizard_state is None:
            raise ValueError(
                "WizardTurnHandle.wizard_state is None — begin_turn must be called first"
            )
        manager = handle.manager
        llm = handle.llm
        tools = handle.tools
        user_message = handle.user_message

        # Trigger stage exit hook if configured
        if self._hooks:
            await self._hooks.trigger_exit(
                wizard_state.current_stage, wizard_state.data
            )

        # Update stage-exit tasks before leaving
        update_stage_exit_tasks(wizard_state, wizard_state.current_stage)

        # Update tool tasks from DynaBot-level tool execution results.
        if tool_results:
            for execution in tool_results:
                update_tool_tasks(
                    wizard_state,
                    execution.tool_name,
                    success=(execution.error is None),
                )

        # Apply tool_result_mapping: write tool results into wizard state.
        # Runs AFTER update_tool_tasks so task tracking sees the execution,
        # and BEFORE FSM transition so conditions can check tool-populated
        # state keys.  Index-based matching: trm_entries[i] corresponds to
        # tool_results[i] (process_input builds one ToolCallSpec per
        # trm_entry in order, and _execute_tools preserves order).
        # This handles duplicate tool names correctly — each entry maps
        # to its own execution result by position, not by name lookup.
        trm_entries = handle.tool_result_mapping
        if trm_entries and tool_results:
            for idx, trm_entry in enumerate(trm_entries):
                if idx >= len(tool_results):
                    break
                execution = tool_results[idx]
                if execution.error:
                    if trm_entry.on_error == "fail":
                        wizard_state.data[f"_tool_error_{trm_entry.tool_name}"] = execution.error
                    continue
                result_data = execution.result
                if isinstance(result_data, dict):
                    for result_key, state_key in trm_entry.mapping.items():
                        if result_key in result_data:
                            wizard_state.data[state_key] = result_data[result_key]
                elif trm_entry.mapping:
                    # Non-dict result: map to the first target key
                    first_state_key = next(iter(trm_entry.mapping.values()))
                    wizard_state.data[first_state_key] = result_data

        # Check for subflow push BEFORE regular FSM transition
        subflow_config = self._subflows.should_push(wizard_state, user_message)
        if subflow_config and self._subflows.handle_push(
            wizard_state, subflow_config, user_message
        ):
            active_fsm = self._subflows.get_active_fsm()
            new_stage = active_fsm.current_metadata
            await self._navigator.branch_for_revisited_stage(
                manager, new_stage.get("name", "")
            )
            return FinalizePreambleResult(
                wizard_state=wizard_state,
                manager=manager,
                llm=llm,
                tools=tools,
                user_message=user_message,
                subflow_pushed=True,
                subflow_new_stage=new_stage,
            )

        # Get current stage for transition derivations and routing
        active_fsm = self._subflows.get_active_fsm()
        stage = active_fsm.current_metadata

        # Apply data derivations from transition configs before evaluating
        # conditions.  This lets transitions fill in values that enable their
        # own conditions and subsequent auto-advance checks.
        self._apply_transition_derivations(stage, wizard_state)

        # Execute routing transforms (before condition evaluation)
        await self._execute_routing_transforms(stage, wizard_state)

        # Log pre-transition state
        logger.debug(
            "FSM transition attempt: from_stage='%s', data_keys=%s",
            wizard_state.current_stage,
            list(wizard_state.data.keys()),
        )

        # Execute FSM step (shared method: inject keys, step, record, update).
        # When skip_extraction is active, the user_message was directed at
        # the previous stage — don't inject it as _message for FSM condition
        # evaluation on the landing stage.
        from_stage, _step_result = await self._execute_fsm_step(
            wizard_state,
            user_message=None if handle.skip_extraction else user_message,
            llm=handle.llm,
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
            wizard_state, llm=handle.llm,
        )

        # Re-fetch active FSM for response generation
        active_fsm = self._subflows.get_active_fsm()
        new_stage = active_fsm.current_metadata
        if new_stage.get("name") != from_stage:
            await self._navigator.branch_for_revisited_stage(
                manager, new_stage.get("name", "")
            )

        return FinalizePreambleResult(
            wizard_state=wizard_state,
            manager=manager,
            llm=llm,
            tools=tools,
            user_message=user_message,
            from_stage=from_stage,
            new_stage=new_stage,
            auto_advance_messages=auto_advance_messages,
            completed_before=wizard_state.completed,
        )

    async def finalize_turn(
        self,
        handle: TurnHandle,
        tool_results: list[ToolExecution] | None = None,
    ) -> Any:
        """Phase C: Transition FSM, generate response, save state.

        Executes the FSM transition, generates the stage response, and
        saves wizard state.  Receives tool execution results from the
        DynaBot tool loop (if any tools ran between ``process_input``
        and ``finalize_turn``).

        Args:
            handle: Turn handle from :meth:`begin_turn`.
            tool_results: Tool execution records from DynaBot's tool
                loop (``None`` when no tools were executed).

        Returns:
            LLM response object with wizard metadata.
        """
        if not isinstance(handle, WizardTurnHandle):
            raise TypeError(
                f"WizardReasoning.finalize_turn requires WizardTurnHandle, "
                f"got {type(handle).__name__}"
            )

        pre = await self._finalize_preamble(handle, tool_results)

        # ── Subflow push: generate response and return ───────────
        if pre.subflow_pushed:
            stage_result = await self._response.generate_stage_response(
                pre.manager, pre.llm, pre.subflow_new_stage,
                pre.wizard_state, pre.tools, track_render=False,
            )
            await self._save_wizard_state(pre.manager, pre.wizard_state)
            return stage_result.response

        # ── Normal path: generate response with lifecycle handling ─
        stage_result = await self._response.generate_stage_response(
            pre.manager, pre.llm, pre.new_stage, pre.wizard_state, pre.tools,
        )
        response = stage_result.response

        # Prepend any messages collected from intermediate auto-advance stages
        if pre.auto_advance_messages:
            WizardResponder.prepend_messages_to_response(
                response, pre.auto_advance_messages,
            )

        # Check for tool-initiated restart (RestartWizardTool signal).
        # Restart takes priority over completion — if both are set, restart
        # clears the wizard state so completion would be meaningless.
        if stage_result.tool_restart_requested:
            logger.info("Wizard restart signaled by restart_wizard tool")
            response = await self._navigator.execute_restart(
                pre.user_message, pre.wizard_state, pre.manager, pre.llm,
            )

        # Check for tool-initiated completion (CompleteWizardTool signal)
        elif not pre.completed_before and stage_result.tool_completion_requested:
            pre.wizard_state.completed = True
            completion_summary = stage_result.tool_completion_summary
            if completion_summary:
                logger.info(
                    "Wizard completion signaled by complete_wizard tool: %s",
                    completion_summary,
                )
                pre.wizard_state.data["_completion_summary"] = completion_summary
            else:
                logger.info("Wizard completion signaled by complete_wizard tool")
            if self._hooks:
                await self._hooks.trigger_complete(pre.wizard_state.data)

        # Save wizard state
        await self._save_wizard_state(pre.manager, pre.wizard_state)

        return response

    async def stream_finalize_turn(
        self,
        handle: TurnHandle,
        tool_results: list[ToolExecution] | None = None,
    ) -> AsyncIterator[LLMStreamResponse]:
        """Stream Phase C: Transition FSM, stream response, save state.

        Streaming counterpart of :meth:`finalize_turn`.  Uses the same
        :meth:`_finalize_preamble` for pre-response setup; only the
        response generation step differs (streaming vs. buffered).

        State is saved only after the stream is fully consumed.  If the
        caller abandons the stream via ``aclose()``, state is NOT saved —
        consistent with DynaBot's ``stream_fully_consumed`` guard.

        Args:
            handle: Turn handle from :meth:`begin_turn`.
            tool_results: Tool execution records from DynaBot's tool
                loop (``None`` when no tools were executed).

        Yields:
            :class:`LLMStreamResponse` chunks.
        """
        if not isinstance(handle, WizardTurnHandle):
            raise TypeError(
                f"WizardReasoning.stream_finalize_turn requires WizardTurnHandle, "
                f"got {type(handle).__name__}"
            )

        pre = await self._finalize_preamble(handle, tool_results)

        # ── Subflow push: stream response and return ─────────────
        if pre.subflow_pushed:
            stream_ctx = StreamStageContext()
            async for chunk in self._response.stream_generate_stage_response(
                pre.manager, pre.llm, pre.subflow_new_stage,
                pre.wizard_state, pre.tools, stream_ctx,
                track_render=False,
            ):
                yield chunk
            await self._save_wizard_state(pre.manager, pre.wizard_state)
            return

        # ── Normal path: stream response with lifecycle handling ──

        # Yield auto-advance messages as initial chunk
        if pre.auto_advance_messages:
            prefix = "\n\n".join(pre.auto_advance_messages) + "\n\n"
            yield LLMStreamResponse(delta=prefix, is_final=False)

        # Stream stage response.
        # If the caller abandons the stream via aclose(), GeneratorExit
        # is thrown at the yield point and the code below never runs —
        # state is NOT saved, consistent with DynaBot's stream_fully_consumed
        # guard.
        stream_ctx = StreamStageContext()
        async for chunk in self._response.stream_generate_stage_response(
            pre.manager, pre.llm, pre.new_stage, pre.wizard_state,
            pre.tools, stream_ctx,
        ):
            yield chunk

        # ── Post-stream work (only reached when fully consumed) ──

        # Check for tool-initiated restart.
        # In the non-streaming path, execute_restart replaces the response
        # entirely.  In streaming, the stage response has already been
        # yielded — we can't un-yield it.  Instead, perform the restart
        # cleanup (reset state, FSM back to start) without emitting a
        # replacement response.  The consumer's next turn will see the
        # restarted wizard and generate the first-stage greeting naturally.
        if stream_ctx.tool_restart_requested:
            logger.info("Wizard restart signaled by restart_wizard tool (streaming)")
            await self._navigator.restart_cleanup(
                pre.wizard_state, pre.user_message,
            )

        # Check for tool-initiated completion
        elif not pre.completed_before and stream_ctx.tool_completion_requested:
            pre.wizard_state.completed = True
            completion_summary = stream_ctx.tool_completion_summary
            if completion_summary:
                logger.info(
                    "Wizard completion signaled by complete_wizard tool: %s",
                    completion_summary,
                )
                pre.wizard_state.data["_completion_summary"] = completion_summary
            else:
                logger.info("Wizard completion signaled by complete_wizard tool")
            if self._hooks:
                await self._hooks.trigger_complete(pre.wizard_state.data)

        # Save wizard state (only reached when stream fully consumed)
        await self._save_wizard_state(pre.manager, pre.wizard_state)

    async def generate(
        self,
        manager: Any,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate response guided by wizard FSM.

        Backward-compatible single-call entry point that delegates to
        the three-phase protocol (:meth:`begin_turn`,
        :meth:`process_input`, :meth:`finalize_turn`).

        Args:
            manager: ConversationManager instance
            llm: LLM provider instance
            tools: Optional list of available tools
            **kwargs: Additional generation parameters

        Returns:
            LLM response object with wizard metadata
        """
        handle = await self.begin_turn(manager, llm, tools, **kwargs)
        if handle.early_response:
            return handle.early_response

        result = await self.process_input(handle)
        if result.early_response:
            return result.early_response

        # tool_results is None here — this single-call path has no tool
        # interleaving.  DynaBot's phased routing passes actual results
        # via _generate_phased_response → finalize_turn(handle, tool_results).
        return await self.finalize_turn(handle)

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
        user_input: dict[str, Any] | str,
        state: WizardState,
        *,
        navigation: str | None = None,
        llm: Any | None = None,
    ) -> WizardAdvanceResult:
        """Advance the wizard one step without DynaBot infrastructure.

        This is the non-conversational counterpart to ``generate()``.  It
        performs the same FSM operations and hook lifecycle but without
        conversation storage or message formatting.

        The caller manages state persistence — pass in the current state,
        receive the updated state in the result, and persist it however
        you choose.

        Args:
            user_input: Either a ``dict`` of pre-extracted structured data
                (merged directly into ``state.data``), or a ``str`` of raw
                user text that triggers the extraction pipeline (extract,
                normalize, merge, defaults, derivations, recovery).
            state: Current wizard state.  Mutated in place and also returned
                in the result for convenience.
            navigation: Optional navigation command.  One of:

                - ``"back"`` — go to previous stage
                - ``"skip"`` — skip current stage (must be skippable)
                - ``"restart"`` — reset to initial state

            llm: LLM provider for extraction.  Required when
                ``user_input`` is a ``str``.  Ignored when ``user_input``
                is a ``dict``.

        Returns:
            :class:`WizardAdvanceResult` with updated state, current stage
            info, and completion status.  When ``user_input`` was a ``str``,
            the result also includes ``extraction`` and ``missing_fields``.

        Raises:
            ValueError: If ``user_input`` is a ``str``, ``llm`` is
                ``None``, and no ``navigation`` command is provided.

        Example::

            # Pre-extracted input (existing behavior)
            result = await reasoning.advance(
                user_input={"name": "Alice"},
                state=state,
            )

            # Raw text input with extraction
            result = await reasoning.advance(
                user_input="My name is Alice",
                state=state,
                llm=provider,
            )
            if result.missing_fields:
                print(f"Still need: {result.missing_fields}")
        """
        extract_mode = isinstance(user_input, str)
        if extract_mode and llm is None and navigation is None:
            raise ValueError(
                "llm parameter is required when user_input is a string "
                "and no navigation command is provided"
            )

        # Clear per-turn keys from previous turn
        for key in self._per_turn_keys:
            state.data.pop(key, None)
            state.transient.pop(key, None)

        # Restore FSM (and subflow FSM if applicable) to match state
        self._restore_fsm_state(state)

        from_stage = state.current_stage
        pipeline_result: ExtractionPipelineResult | None = None

        # Handle navigation
        auto_advance_messages: list[str] = []
        if navigation == "back":
            transitioned = await self._navigator.navigate_back(state)
        elif navigation == "skip":
            transitioned, auto_advance_messages = await self._navigator.navigate_skip(
                state,
            )
        elif navigation == "restart":
            await self._navigator.navigate_restart(state)
            transitioned = state.current_stage != from_stage
        else:
            active_fsm = self._subflows.get_active_fsm()
            stage = active_fsm.current_metadata

            if extract_mode:
                # Extraction mode: run the full pipeline
                pipeline_result = await self._extraction.run_extraction_pipeline(
                    user_input, stage, state, llm,
                )
            else:
                # Dict mode: direct merge (existing behavior)
                state.data.update(user_input)

            # Exit hook
            if self._hooks:
                await self._hooks.trigger_exit(
                    state.current_stage, state.data
                )
            update_stage_exit_tasks(state, state.current_stage)

            # Apply transition derivations
            self._apply_transition_derivations(stage, state)

            # Execute routing transforms (before condition evaluation)
            await self._execute_routing_transforms(stage, state)

            # Execute FSM step (shared method)
            await self._execute_fsm_step(state, llm=llm)

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
                    await self._run_post_transition_lifecycle(
                        state, llm=llm,
                    )
                )

        # Build result
        active_fsm = self._subflows.get_active_fsm()
        stage_meta = active_fsm.current_metadata
        metadata = self._build_wizard_metadata(state)

        raw_prompt = active_fsm.get_stage_prompt()
        raw_suggestions = active_fsm.get_stage_suggestions()
        nav_context = {
            "can_skip": active_fsm.can_skip(),
            "can_go_back": (
                active_fsm.can_go_back() and len(state.history) > 1
            ),
        }

        return WizardAdvanceResult(
            state=state,
            stage_name=state.current_stage,
            stage_prompt=self._renderer.render(
                raw_prompt, stage_meta, state,
                extra_context=nav_context, fallback=raw_prompt,
            ),
            stage_schema=StageSchema.from_stage(stage_meta).raw or None,
            suggestions=self._renderer.render_list(
                raw_suggestions, stage_meta, state,
            ),
            can_skip=active_fsm.can_skip(),
            can_go_back=active_fsm.can_go_back() and len(state.history) > 1,
            completed=state.completed,
            transitioned=transitioned,
            from_stage=from_stage if transitioned else None,
            auto_advance_messages=auto_advance_messages,
            metadata=metadata,
            extraction=(
                pipeline_result.extraction if pipeline_result else None
            ),
            missing_fields=(
                pipeline_result.missing_fields if pipeline_result else None
            ),
            changed_fields=(
                pipeline_result.new_data_keys if pipeline_result else None
            ),
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
        initial_tasks = build_initial_tasks(self._fsm.stages)
        self._subflows.active_subflow_fsm = None  # Ensure we start in main flow

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
        active_fsm = self._subflows.get_active_fsm()

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
            "stage_prompt": self._renderer.render(
                active_fsm.get_stage_prompt(),
                active_fsm.current_metadata,
                state,
                extra_context={
                    "can_skip": active_fsm.can_skip(),
                    "can_go_back": (
                        active_fsm.can_go_back()
                        and len(state.history) > 1
                    ),
                },
                fallback=active_fsm.get_stage_prompt(),
            ),
            "suggestions": self._response.render_suggestions(
                active_fsm.get_stage_suggestions(), state
            ),
            "stages": self._response.build_stages_roadmap(state),
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

    def _restore_fsm_state(self, state: WizardState) -> None:
        """Restore the FSM (and subflow FSM if applicable) from state.

        Sets ``_subflows.active_subflow_fsm`` when ``state.subflow_stack``
        is non-empty so that subsequent calls to
        ``self._subflows.get_active_fsm()`` return the correct FSM.

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
            self._subflows.active_subflow_fsm = self._fsm.get_subflow(subflow_name)
            if self._subflows.active_subflow_fsm:
                self._subflows.active_subflow_fsm.restore(fsm_state)
        else:
            self._subflows.active_subflow_fsm = None

    # =========================================================================
    # Shared Wizard Lifecycle Methods
    # =========================================================================
    # These methods consolidate the core wizard lifecycle operations used by
    # both generate() (conversational path) and advance() (non-conversational
    # path).  Each was extracted from inline code that was duplicated across
    # generate(), WizardNavigator._execute_skip(), and WizardNavigator._execute_back().


    async def _execute_fsm_step(
        self,
        state: WizardState,
        *,
        user_message: str | None = None,
        trigger: str = "user_input",
        llm: Any | None = None,
    ) -> tuple[str, Any]:
        """Execute an FSM step with standard runtime key injection and state update.

        Injects runtime keys (``_bank_fn``, ``_message``), executes the FSM
        step, cleans up runtime keys, and updates wizard state
        (``current_stage``, ``history``, ``completed``).  Records the
        transition if the stage changed.

        A per-call closure is installed as the ``transform_context_factory``
        so that each FSM step captures its own ``llm`` and ``TurnContext``
        on the stack — eliminating the concurrency hazard of instance-level
        ``_current_llm`` / ``_current_turn`` attributes.

        This is the shared core used by ``generate()``, ``advance()``,
        and ``WizardNavigator.navigate_skip()``.

        Args:
            state: Wizard state (mutated in place).
            user_message: Optional message for condition evaluation.
            trigger: Transition trigger label for audit trail.
            llm: LLM provider for this step's transforms (``None`` when
                no LLM is available, e.g. navigation skips).

        Returns:
            Tuple of (from_stage, step_result).
        """
        active_fsm = self._subflows.get_active_fsm()
        from_stage = state.current_stage
        duration_ms = (time.time() - state.stage_entry_time) * 1000

        # Inject runtime keys for condition/transform evaluation
        if user_message is not None:
            state.data["_message"] = user_message
        state.data["_bank_fn"] = self._make_bank_accessor()

        # Build per-call TurnContext — lives on the stack, not on self
        turn = TurnContext(
            message=user_message,
            bank_fn=self._make_bank_accessor(),
            intent=state.data.get("_intent"),
        )

        # Per-call closure captures local llm and turn values so that
        # concurrent FSM steps each see their own context.
        # Note: _artifact_registry, _review_executor, and _banks are
        # accessed via self (not captured by value) — they are set once
        # at construction and are stable during FSM execution.
        from ..artifacts.transforms import TransformContext

        def _scoped_factory(func_context: Any) -> Any:
            return TransformContext(
                fsm_context=func_context,
                turn=turn,
                artifact_registry=self._artifact_registry,
                rubric_executor=self._review_executor,
                config={"llm": llm} if llm else {},
                banks=self._banks,
            )

        active_fsm.set_transform_context_factory(_scoped_factory)
        try:
            # Execute FSM step
            step_result = await active_fsm.step_async(state.data)
        finally:
            # Restore the fallback factory on the *current* active FSM.
            # step_async may have triggered a subflow push, changing which
            # FSM is active — restore on the post-step FSM, not the
            # pre-step local reference.
            self._subflows.get_active_fsm().set_transform_context_factory(
                self._build_transform_context
            )

        # Clean up runtime keys
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
        *,
        llm: Any | None = None,
    ) -> list[str]:
        """Run post-transition lifecycle: subflow pop, auto-advance, hooks.

        Called after ``_execute_fsm_step()`` to handle the standard
        post-transition sequence.  Shared by ``generate()`` and
        ``advance()``.

        Args:
            state: Wizard state (mutated in place).
            llm: LLM provider for auto-advance transforms (``None``
                when no LLM is available).

        Returns:
            List of rendered template strings from auto-advanced stages.
        """
        # Subflow pop check
        if self._subflows.should_pop(state):
            self._subflows.handle_pop(state)
            state.completed = False

        # Auto-advance through stages with all required fields filled
        active_fsm = self._subflows.get_active_fsm()
        stage = active_fsm.current_metadata
        auto_advance_messages = await self._response.run_auto_advance_loop(
            state, active_fsm, stage, llm=llm,
        )

        # Re-fetch in case subflow pop occurred during auto-advance
        active_fsm = self._subflows.get_active_fsm()

        # Stage entry hook
        if self._hooks:
            await self._hooks.trigger_enter(
                state.current_stage, state.data
            )

        # Completion hook
        if state.completed and self._hooks:
            await self._hooks.trigger_complete(state.data)

        return auto_advance_messages

    # ------------------------------------------------------------------
    # Navigation delegation — thin forwards to WizardNavigator.
    # These exist so that unit tests and advance() can call navigation
    # methods via the same interface as before the 77b extraction.
    # ------------------------------------------------------------------

    async def _handle_navigation(
        self, message: str, state: WizardState, manager: Any, llm: Any,
    ) -> Any | None:
        """Delegate to navigator.  See :meth:`WizardNavigator.handle_navigation`."""
        return await self._navigator.handle_navigation(message, state, manager, llm)

    async def _navigate_back(
        self, state: WizardState, *, user_message: str | None = None,
    ) -> bool:
        """Delegate to navigator.  See :meth:`WizardNavigator.navigate_back`."""
        return await self._navigator.navigate_back(state, user_message=user_message)

    async def _navigate_skip(
        self, state: WizardState, *, user_message: str | None = None,
    ) -> tuple[bool, list[str]]:
        """Delegate to navigator.  See :meth:`WizardNavigator.navigate_skip`."""
        return await self._navigator.navigate_skip(state, user_message=user_message)

    async def _navigate_restart(
        self, state: WizardState, user_message: str = "",
    ) -> None:
        """Delegate to navigator.  See :meth:`WizardNavigator.navigate_restart`."""
        await self._navigator.navigate_restart(state, user_message=user_message)

    async def _restart_cleanup(
        self, state: WizardState, message: str, trigger: str = "restart",
    ) -> None:
        """Delegate to navigator.  See :meth:`WizardNavigator.restart_cleanup`."""
        await self._navigator.restart_cleanup(state, message, trigger=trigger)

    async def _execute_restart(
        self, message: str, state: WizardState, manager: Any, llm: Any,
    ) -> Any:
        """Delegate to navigator.  See :meth:`WizardNavigator.execute_restart`."""
        return await self._navigator.execute_restart(message, state, manager, llm)

    async def _branch_for_revisited_stage(
        self, manager: Any, stage_name: str,
    ) -> None:
        """Delegate to navigator.  See :meth:`WizardNavigator.branch_for_revisited_stage`."""
        await self._navigator.branch_for_revisited_stage(manager, stage_name)

    async def _detect_amendment(
        self, message: str, state: WizardState, llm: Any,
    ) -> dict[str, Any] | None:
        """Delegate to navigator.  See :meth:`WizardNavigator.detect_amendment`."""
        return await self._navigator.detect_amendment(message, state, llm)

    def _map_section_to_stage(self, section: str) -> str | None:
        """Delegate to navigator.  See :meth:`WizardNavigator.map_section_to_stage`."""
        return self._navigator.map_section_to_stage(section)

    @staticmethod
    def _get_last_user_message(manager: Any) -> str:
        """Delegate to responder.  See :meth:`WizardResponder._get_last_user_message`."""
        return WizardResponder._get_last_user_message(manager)

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

        collected_data = self._renderer.get_collected_data(state)

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
                        resolved = self._renderer.render_simple(
                            value, collected_data,
                        )
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

    async def _execute_routing_transforms(
        self,
        stage: dict[str, Any],
        state: WizardState,
    ) -> None:
        """Execute routing transforms declared on a stage.

        Routing transforms run after extraction/merge/derivation but
        *before* transition condition evaluation.  They set routing
        signals in ``state.data`` (e.g. ``classified_need``) that
        transition conditions depend on.

        Declared in wizard YAML as::

            stages:
              - name: assess_needs
                routing_transforms:
                  - classify_user_need

        Each name is resolved via the FSM's function registry.
        Functions receive the wizard state data dict as the sole
        argument.  They may return an updated dict or mutate the
        dict in place and return ``None``.

        Args:
            stage: Current stage metadata.
            state: Wizard state — ``data`` may be mutated by transforms.
        """
        transform_names = stage.get("routing_transforms", [])
        if not transform_names:
            return

        active_fsm = self._subflows.get_active_fsm()
        for name in transform_names:
            func = active_fsm.resolve_function(name)
            if func is None:
                logger.warning(
                    "Routing transform '%s' not found in function registry "
                    "for stage '%s'",
                    name,
                    stage.get("name", "?"),
                )
                continue

            try:
                result = func(state.data)
                if inspect.isawaitable(result):
                    result = await result

                # Transforms may return updated data or mutate in-place
                if isinstance(result, dict):
                    state.data.update(result)

                logger.debug(
                    "Routing transform '%s' executed for stage '%s'",
                    name,
                    stage.get("name", "?"),
                )
            except Exception as e:
                logger.warning(
                    "Routing transform '%s' failed for stage '%s': %s",
                    name,
                    stage.get("name", "?"),
                    e,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Response generation delegation — thin forwards to WizardResponder.
    # These exist so that unit tests can call response methods via the
    # same interface as before the 77d extraction.
    # ------------------------------------------------------------------

    async def _generate_stage_response(
        self, manager: Any, llm: Any, stage: dict[str, Any],
        state: WizardState, tools: list[Any] | None,
    ) -> Any:
        """Delegate to responder.  See :meth:`WizardResponder.generate_stage_response`."""
        result = await self._response.generate_stage_response(
            manager, llm, stage, state, tools,
        )
        return result.response

    def _can_auto_advance(
        self, wizard_state: WizardState, stage: dict[str, Any],
    ) -> bool:
        """Delegate to responder.  See :meth:`WizardResponder.can_auto_advance`."""
        return self._response.can_auto_advance(wizard_state, stage)

    def _evaluate_condition(self, condition: str, data: dict[str, Any]) -> bool:
        """Delegate to responder.  See :meth:`WizardResponder.evaluate_condition`."""
        return self._response.evaluate_condition(condition, data)

    @staticmethod
    def _prepend_messages_to_response(
        response: Any, messages: list[str],
    ) -> None:
        """Delegate to responder.  See :meth:`WizardResponder.prepend_messages_to_response`."""
        WizardResponder.prepend_messages_to_response(response, messages)

    @staticmethod
    def _is_help_request(message: str) -> bool:
        """Delegate to responder.  See :meth:`WizardResponder.is_help_request`."""
        return WizardResponder.is_help_request(message)

    @staticmethod
    def _create_template_response(content: str) -> Any:
        """Delegate to responder.  See :meth:`WizardResponder.create_template_response`."""
        return WizardResponder.create_template_response(content)

    def _build_stage_context(
        self, stage: dict[str, Any], state: WizardState,
    ) -> str:
        """Delegate to responder.  See :meth:`WizardResponder.build_stage_context`."""
        return self._response.build_stage_context(stage, state)

    def _filter_tools_for_stage(
        self, stage: dict[str, Any], tools: list[Any] | None,
    ) -> list[Any] | None:
        """Delegate to responder.  See :meth:`WizardResponder.filter_tools_for_stage`."""
        return self._response.filter_tools_for_stage(stage, tools)

    def _add_wizard_metadata(
        self, response: Any, state: WizardState, stage: dict[str, Any],
    ) -> None:
        """Delegate to responder.  See :meth:`WizardResponder.add_wizard_metadata`."""
        self._response.add_wizard_metadata(response, state, stage)

    def _build_clarification_groups(
        self,
        missing_fields: set[str],
        stage: dict[str, Any],
        wizard_state: WizardState | None = None,
    ) -> list[dict[str, Any]]:
        """Delegate to responder.  See :meth:`WizardResponder._build_clarification_groups`."""
        return self._response._build_clarification_groups(
            missing_fields, stage, wizard_state,
        )

    async def _run_auto_advance_loop(
        self, wizard_state: WizardState, active_fsm: Any,
        initial_stage: dict[str, Any], **kwargs: Any,
    ) -> list[str]:
        """Delegate to responder.  See :meth:`WizardResponder.run_auto_advance_loop`."""
        return await self._response.run_auto_advance_loop(
            wizard_state, active_fsm, initial_stage, **kwargs,
        )

    def _render_response_template(
        self, template_str: str, stage: dict[str, Any],
        state: WizardState, extra_context: dict[str, Any] | None = None,
    ) -> str:
        """Delegate to responder.  See :meth:`WizardResponder._render_response_template`."""
        return self._response._render_response_template(
            template_str, stage, state, extra_context=extra_context,
        )

    async def _generate_context_variables(
        self, stage: dict[str, Any], state: WizardState, llm: Any,
    ) -> dict[str, str]:
        """Delegate to responder.  See :meth:`WizardResponder._generate_context_variables`."""
        return await self._response._generate_context_variables(stage, state, llm)

    def _render_suggestions(
        self, suggestions: list[str], state: WizardState,
    ) -> list[str]:
        """Delegate to responder.  See :meth:`WizardResponder.render_suggestions`."""
        return self._response.render_suggestions(suggestions, state)

    def _build_default_context(
        self, stage: dict[str, Any], state: WizardState,
    ) -> str:
        """Delegate to responder.  See :meth:`WizardResponder._build_default_context`."""
        return self._response._build_default_context(stage, state)

    def _render_custom_context(
        self, stage: dict[str, Any], state: WizardState,
    ) -> str:
        """Delegate to responder.  See :meth:`WizardResponder._render_custom_context`."""
        return self._response._render_custom_context(stage, state)

    def _resolve_stage_strategy(
        self, stage: dict[str, Any],
    ) -> ReasoningStrategy | None:
        """Delegate to responder.  See :meth:`WizardResponder._resolve_stage_strategy`."""
        return self._response._resolve_stage_strategy(stage)

    async def _strategy_stage_response(
        self, strategy: ReasoningStrategy, manager: Any,
        enhanced_prompt: str, stage: dict[str, Any],
        state: WizardState, tools: list[Any],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, bool, str, bool]:
        """Delegate to responder.  See :meth:`WizardResponder._strategy_stage_response`."""
        return await self._response._strategy_stage_response(
            strategy, manager, enhanced_prompt, stage, state,
            tools, metadata=metadata,
        )

    def _get_max_iterations(self, stage: dict[str, Any]) -> int:
        """Delegate to responder.  See :meth:`WizardResponder._get_max_iterations`."""
        return self._response._get_max_iterations(stage)

    def _calculate_progress(self, state: WizardState) -> float:
        """Delegate to responder.  See :meth:`WizardResponder.calculate_progress`."""
        return self._response.calculate_progress(state)

    def _build_stages_roadmap(self, state: WizardState) -> list[dict[str, str]]:
        """Delegate to responder.  See :meth:`WizardResponder.build_stages_roadmap`."""
        return self._response.build_stages_roadmap(state)

    def _build_extra_context(self) -> dict[str, Any]:
        """Delegate to responder.  See :meth:`WizardResponder._build_extra_context`."""
        return self._response._build_extra_context()

    # =========================================================================
    # Task Tracking Methods
    # =========================================================================

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
        stage_names = self._fsm.stage_names
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
            total_stages=self._fsm.stage_count,
            can_skip=self._fsm.can_skip(),
            can_go_back=self._fsm.can_go_back() and len(wizard_state.history) > 1,
            suggestions=stage.get("suggestions", []),
            stages=self._response.build_stages_roadmap(wizard_state),
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
