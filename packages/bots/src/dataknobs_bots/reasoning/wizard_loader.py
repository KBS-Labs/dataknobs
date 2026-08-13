"""WizardConfigLoader for translating wizard YAML to FSM configuration.

This module provides the translation layer between user-friendly wizard
YAML configuration and the underlying FSM configuration format.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from dataknobs_common.expressions import safe_eval
from dataknobs_common.paths import PathAnchor, PathEscapeError
from dataknobs_fsm.api.advanced import AdvancedFSM
from dataknobs_fsm.config.builder import FSMBuilder

from .function_resolver import resolve_functions
from .stage_synthesizers import (
    StageSynthesizer,
    iter_stage_synthesizers,
    register_stage_synthesizer,
    unregister_stage_synthesizer,
    validate_no_conflicting_fields,
)

# Importing wizard_intent_confirm auto-registers
# IntentConfirmSynthesizer at module load (side-effect import). Done
# here at module level so `iter_stage_synthesizers()` reflects the
# in-tree default before any loader instance is built.
from . import wizard_intent_confirm as _wizard_intent_confirm  # noqa: F401
from .wizard_fsm import WizardFSM

__all__ = [
    "StageSynthesizer",
    "WizardConfigLoader",
    "load_wizard_config",
    "register_stage_synthesizer",
    "unregister_stage_synthesizer",
    "validate_no_conflicting_fields",
]

logger = logging.getLogger(__name__)

# Sentinel target for subflow transitions
SUBFLOW_TARGET = "_subflow"


def _default_transform_context_factory(fn_ctx: Any) -> Any:
    """Default factory wrapping FunctionContext in a minimal TransformContext.

    Ensures all transforms receive a :class:`TransformContext` (with empty
    defaults for registries and banks) so they can access ``.banks``,
    ``.config``, etc. without ``getattr`` guards.

    Consumers can override with a richer factory that populates registries.
    """
    from ..artifacts.transforms import TransformContext

    return TransformContext(fsm_context=fn_ctx)


# ── Stage field registry ─────────────────────────────────────────────
#
# Single source of truth for all recognized stage-level config fields.
# KNOWN_STAGE_FIELDS and _extract_metadata() are both derived from this
# registry, so adding a new field requires only one entry here.
#
# Fields with special extraction logic (transitions, tasks) are not in
# the registry — they are handled inline in _extract_metadata().
#
# HOW TO ADD A NEW STAGE FIELD:
#
#   1. Add a _StageField("field_name") entry below (with default if
#      not None).  This automatically updates KNOWN_STAGE_FIELDS and
#      _extract_metadata().
#
#   2. Add the field to StageConfig in config/wizard_builder.py.
#      StageConfig.to_dict() and from_dict() pick it up automatically
#      via dataclass field introspection.  If the field is a tuple or
#      nested dataclass type, also add it to the classification
#      constants at the top of that file (_TUPLE_PRIMITIVE_FIELDS,
#      _TUPLE_DICT_FIELDS, or _NESTED_FIELDS).
#
#   3. Run tests — TestStageFieldRegistrySync will fail if StageConfig
#      is missing the new field.


class _StageField:
    """Descriptor for a single wizard stage config field."""

    __slots__ = ("default", "name")

    def __init__(self, name: str, default: Any = None) -> None:
        self.name = name
        self.default = default

    def extract(self, stage: dict[str, Any]) -> Any:
        """Read this field from a raw stage config dict.

        Returns a fresh copy for mutable defaults (lists) so that
        callers cannot mutate the shared default object.
        """
        if self.default is None:
            return stage.get(self.name)
        value = stage.get(self.name, self.default)
        # Return a copy of mutable defaults to prevent shared-state bugs
        if value is self.default and isinstance(value, list):
            return list(value)
        return value


# Registry of all stage fields with their extraction defaults.
# Fields default to None unless an explicit default is given.
_STAGE_FIELDS: tuple[_StageField, ...] = (
    # Identity
    _StageField("name"),
    _StageField("label"),  # label default uses stage["name"]; handled in _extract_metadata
    _StageField("is_start", default=False),
    _StageField("is_end", default=False),
    # Prompts and templates
    _StageField("prompt", default=""),
    # Declarative derived template variables: name -> Jinja expression,
    # evaluated against the render context and merged into the template
    # scope (later-wins) so response templates can reference computed
    # values without a consumer subclassing the renderer.
    _StageField("inputs"),
    _StageField("response_template"),
    _StageField("clarification_template"),
    _StageField("confirmation_template"),
    _StageField("llm_assist", default=False),
    _StageField("llm_assist_prompt"),
    # Schema and suggestions
    _StageField("schema"),
    _StageField("suggestions", default=[]),
    _StageField("help_text"),
    # Navigation
    _StageField("can_skip", default=False),
    _StageField("skip_default"),
    _StageField("can_go_back", default=True),
    _StageField("auto_advance"),
    # Confirmation
    _StageField("confirm_on_new_data", default=False),
    _StageField("confirm_first_render", default=True),
    # Tools and reasoning
    _StageField("tools", default=[]),
    _StageField("reasoning"),
    _StageField("reasoning_config"),
    _StageField("max_iterations"),
    _StageField("extraction_model"),
    _StageField("store_trace"),
    _StageField("verbose"),
    # Context and mode
    _StageField("context_generation"),
    _StageField("mode"),
    _StageField("intent_detection"),
    _StageField("intent_confirm"),
    _StageField("navigation"),
    # Collection
    _StageField("collection_mode"),
    _StageField("collection_config"),
    # Extraction control
    _StageField("capture_mode"),
    _StageField("extraction_scope"),
    _StageField("extraction_grounding"),
    _StageField("derivation_enabled"),
    _StageField("recovery_enabled"),
    _StageField("re_extract_on_entry"),
    # Routing and post-extraction
    _StageField("routing_transforms", default=[]),
    _StageField("tool_result_mapping", default=[]),
)

# Derived: set of all recognized stage field names (includes fields
# handled specially by _extract_metadata: transitions, tasks).
KNOWN_STAGE_FIELDS: frozenset[str] = frozenset(f.name for f in _STAGE_FIELDS) | {
    "transitions",
    "tasks",
}

# Patterns that suggest a condition is natural language rather than Python
_ENGLISH_CONDITION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bis\s+provided\b", re.IGNORECASE),
    re.compile(r"\bis\s+not\s+empty\b", re.IGNORECASE),
    re.compile(r"\bhas\s+been\s+(set|given|entered)\b", re.IGNORECASE),
    re.compile(r"\bwas\s+(set|given|entered)\b", re.IGNORECASE),
)

# Pattern for Python str.format()-style placeholders (e.g. {name})
# that should be Jinja2 ({{ name }})
_PYTHON_FORMAT_PATTERN: re.Pattern[str] = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")


def _null_bank(name: str) -> Any:
    """Fallback bank accessor when no banks are configured.

    Returns an ``EmptyBankProxy`` for any bank name so that condition
    expressions like ``bank('x').count() > 0`` evaluate safely.
    """
    from ..memory.bank import EmptyBankProxy

    return EmptyBankProxy(name)


class WizardConfigLoader:
    """Translates wizard YAML configuration to FSM configuration.

    The wizard config format is user-friendly, focusing on:
    - Stages (what users interact with)
    - Prompts and suggestions (conversational UX)
    - Schemas (data validation)
    - Simple transitions

    This is translated to the more powerful FSM format which supports:
    - Networks, states, arcs (full state machine)
    - Complex conditions and transforms
    - Resource management

    Example wizard config::

        name: onboarding-wizard
        version: "1.0"

        stages:
          - name: welcome
            is_start: true
            prompt: "What kind of bot would you like to create?"
            schema:
              type: object
              properties:
                intent:
                  type: string
                  enum: [tutor, quiz, companion]
            transitions:
              - target: select_template
                condition: "data.get('intent')"

          - name: select_template
            prompt: "Would you like to start from a template?"
            transitions:
              - target: configure
                condition: "data.get('use_template')"
              - target: complete

          - name: complete
            is_end: true
            prompt: "You're all set!"
    """

    def load(
        self,
        config_path: str | Path,
        custom_functions: dict[str, Callable[..., Any] | str] | None = None,
        transform_context_factory: Callable[..., Any] | None = None,
        config_root: str | Path | None = None,
    ) -> WizardFSM:
        """Load wizard config and create WizardFSM.

        Args:
            config_path: Path to wizard YAML config file
            custom_functions: Optional custom functions for transitions.
                Values can be either callables or "module:function" strings.
            transform_context_factory: Optional callable that receives a
                :class:`FunctionContext` and returns the application-specific
                context for transforms. If ``None``, a default factory is
                used.
            config_root: Directory that subflow names may address within, at
                any depth. Defaults to ``config_path``'s own directory. A
                nested subflow resolves its own subflows relative to *itself*
                but stays bounded to this root, so a shared subflow directory
                beside the wizard is reachable while the tree is still a
                boundary.

        Returns:
            Configured WizardFSM instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config is invalid YAML
            ValueError: If config structure is invalid
            PathEscapeError: A subflow name addresses a file outside
                ``config_root``, or ``config_root`` does not contain
                ``config_path``.
        """
        config_path = Path(config_path)

        with open(config_path) as f:
            wizard_config = yaml.safe_load(f)

        return self.load_from_dict(
            wizard_config,
            custom_functions,
            config_base_path=config_path.parent,
            transform_context_factory=transform_context_factory,
            config_root=config_root,
        )

    def load_from_dict(
        self,
        wizard_config: dict[str, Any],
        custom_functions: dict[str, Callable[..., Any] | str] | None = None,
        config_base_path: Path | None = None,
        transform_context_factory: Callable[..., Any] | None = None,
        config_root: str | Path | None = None,
    ) -> WizardFSM:
        """Load wizard config from dict and create WizardFSM.

        Args:
            wizard_config: Wizard configuration dict
            custom_functions: Optional custom functions for transitions.
                Values can be either:
                - Callable objects (used directly)
                - String references in "module.path:function_name" format
            config_base_path: Base path for resolving relative subflow paths
            transform_context_factory: Optional callable that receives a
                :class:`FunctionContext` and returns the application-specific
                context for transforms (e.g. :class:`TransformContext`).
                If ``None``, a default factory is used that wraps the
                :class:`FunctionContext` in a minimal
                :class:`TransformContext`.
            config_root: Directory that subflow names may address within, at
                any depth. Defaults to ``config_base_path``.

        Returns:
            Configured WizardFSM instance

        Raises:
            ValueError: If config structure is invalid
            PathEscapeError: A subflow name addresses a file outside
                ``config_root``.

        Example:
            ```python
            # Custom functions can be callables or string references
            loader.load_from_dict(
                wizard_config,
                custom_functions={
                    "validate": my_validate_func,  # Callable
                    "transform": "myapp.transforms:apply_template",  # String
                }
            )
            ```
        """
        # Validate required fields
        if "stages" not in wizard_config:
            raise ValueError("Wizard config must have 'stages' field")

        if not wizard_config["stages"]:
            raise ValueError("Wizard config must have at least one stage")

        # Expand stage primitives via the synthesizer registry.
        # Runs BEFORE _validate_config so the broader validator and the
        # FSM build pipeline see the post-expansion shape (a pre-synthesis
        # `intent_confirm:` stage has no schema/response_template and
        # would otherwise trip the "pure LLM-driven" warning).
        self._synthesize_stages(wizard_config)

        # Warn about common config issues
        self._validate_config(wizard_config)

        # Translate wizard config to FSM config
        fsm_config = self._translate_to_fsm(wizard_config)

        # Extract stage metadata (includes subflow transition info)
        stage_metadata = self._extract_metadata(wizard_config)

        # Extract wizard-level settings
        settings = wizard_config.get("settings", {})

        # Build FSM
        builder = FSMBuilder()
        if custom_functions:
            # Resolve string references to callables
            resolved_functions = resolve_functions(custom_functions)
            for name, func in resolved_functions.items():
                builder.register_function(name, func)

        # Register inline condition functions
        self._register_inline_conditions(builder, wizard_config)

        fsm = builder.build(fsm_config)
        advanced_fsm = AdvancedFSM(fsm)

        # Load subflow networks, bounded to the wizard's config tree
        anchor = (
            PathAnchor.rooted_at(config_base_path, config_root)
            if config_base_path is not None
            else None
        )
        subflow_registry = self._load_subflow_networks(wizard_config, custom_functions, anchor)

        # Use provided factory or default that wraps FunctionContext
        # in a minimal TransformContext
        if transform_context_factory is None:
            transform_context_factory = _default_transform_context_factory

        return WizardFSM(
            advanced_fsm,
            stage_metadata,
            settings=settings,
            subflow_registry=subflow_registry,
            transform_context_factory=transform_context_factory,
        )

    def _synthesize_stages(self, wizard_config: dict[str, Any]) -> None:
        """Apply registered stage synthesizers.

        Iterates each registered
        :class:`~dataknobs_bots.reasoning.stage_synthesizers.StageSynthesizer`;
        for each stage that declares the synthesizer's field, validates
        (when ``validate`` is defined) then expands the primitive in
        place. Runs BEFORE :meth:`_validate_config` and
        :meth:`_translate_to_fsm` so the broader validator and the FSM
        build pipeline see the post-expansion shape.

        Consumer-extensible: any consumer can call
        :func:`register_stage_synthesizer` to ship their own primitive.
        The in-tree :class:`IntentConfirmSynthesizer` is registered at
        module import (see the ``wizard_intent_confirm`` import above).
        """
        synthesizers = iter_stage_synthesizers()
        for stage in wizard_config.get("stages", []):
            for field, synthesizer in synthesizers.items():
                if field in stage and stage[field] is not None:
                    validate = getattr(synthesizer, "validate", None)
                    if callable(validate):
                        validate(stage)
                    synthesizer.synthesize(stage)

    def _validate_config(self, wizard_config: dict[str, Any]) -> None:
        """Validate wizard config and warn about common issues.

        Checks for:
        1. Unrecognized stage fields (e.g. ``extracts``)
        2. Non-end stages with no ``schema`` and no ``response_template``
           (pure LLM-driven — unreliable for data collection)
        3. Conditions that look like English rather than Python
        4. Template syntax using Python str.format() instead of Jinja2

        All issues are logged as warnings, not errors — the config will
        still load.

        Args:
            wizard_config: Wizard configuration dict
        """
        for stage in wizard_config.get("stages", []):
            stage_name = stage.get("name", "<unnamed>")

            # 1. Unrecognized fields
            for key in stage:
                if key not in KNOWN_STAGE_FIELDS:
                    logger.warning(
                        "Stage '%s': unrecognized field '%s' (will be ignored). Known fields: %s",
                        stage_name,
                        key,
                        ", ".join(sorted(KNOWN_STAGE_FIELDS)),
                    )

            # 2. No schema + no response_template on non-end stages.
            #    Suppressed for conversation-mode and tool-driven stages
            #    (ReAct stages use tools for interaction, not extraction).
            if (
                not stage.get("is_end")
                and not stage.get("schema")
                and not stage.get("response_template")
                and stage.get("mode") != "conversation"
                and not stage.get("tools")
            ):
                logger.warning(
                    "Stage '%s': no 'schema' and no 'response_template'. "
                    "Pure LLM-driven stages are unreliable for data "
                    "collection. Consider adding a schema for extraction "
                    "and/or a response_template for deterministic output.",
                    stage_name,
                )

            # 3. English-language conditions
            for transition in stage.get("transitions", []):
                condition = transition.get("condition", "")
                if condition:
                    for pattern in _ENGLISH_CONDITION_PATTERNS:
                        if pattern.search(condition):
                            logger.warning(
                                "Stage '%s': condition '%s' appears to be "
                                "natural language, not Python. Conditions "
                                "are evaluated as Python code. "
                                "Try: data.get('%s')",
                                stage_name,
                                condition,
                                condition.split()[0],
                            )
                            break

            # 4. Invalid re_extract_on_entry values
            re_extract = stage.get("re_extract_on_entry")
            if (
                re_extract is not None
                and re_extract is not True
                and (re_extract is not False and re_extract != "capture_only")
            ):
                logger.warning(
                    "Stage '%s': re_extract_on_entry=%r is not a "
                    "recognized value. Use True, False, or "
                    "'capture_only'.",
                    stage_name,
                    re_extract,
                )

            # 5. Python str.format() syntax in templates and prompts
            for field_name in ("response_template", "prompt"):
                text = stage.get(field_name, "")
                if text and _PYTHON_FORMAT_PATTERN.search(text):
                    matches = _PYTHON_FORMAT_PATTERN.findall(text)
                    logger.warning(
                        "Stage '%s': %s uses Python format syntax "
                        "{%s} — did you mean Jinja2 {{ %s }}?",
                        stage_name,
                        field_name,
                        matches[0],
                        matches[0],
                    )

    def _translate_to_fsm(self, wizard_config: dict[str, Any]) -> Any:
        """Translate wizard config to FSM format.

        Args:
            wizard_config: Wizard configuration dict

        Returns:
            FSMConfig object
        """
        from dataknobs_fsm.config.schema import (
            DataModeConfig,
            FSMConfig,
            NetworkConfig,
        )

        # Create network config with states
        states = []
        for stage in wizard_config.get("stages", []):
            state_config = self._translate_stage(stage)
            states.append(state_config)

        network = NetworkConfig(
            name="main",
            states=states,
            metadata={"description": wizard_config.get("description", "")},
        )

        # Create FSM config
        fsm_config = FSMConfig(
            name=wizard_config.get("name", "wizard"),
            version=wizard_config.get("version", "1.0.0"),
            description=wizard_config.get("description", ""),
            networks=[network],
            main_network="main",
            resources=[],
            data_mode=DataModeConfig(),
        )

        return fsm_config

    def _translate_stage(self, stage: dict[str, Any]) -> Any:
        """Translate wizard stage to FSM state.

        Args:
            stage: Stage configuration dict

        Returns:
            StateConfig object
        """
        from dataknobs_fsm.config.schema import StateConfig

        # Build arcs from transitions
        arcs = []
        for idx, transition in enumerate(stage.get("transitions", [])):
            arc = self._translate_transition(stage["name"], transition, idx)
            arcs.append(arc)

        # If no transitions and not end state, warn.
        # Suppressed for stages with tools — lifecycle tools like
        # complete_wizard and restart_wizard handle transitions
        # programmatically rather than via config arcs.
        if not arcs and not stage.get("is_end") and not stage.get("tools"):
            logger.warning(
                "Stage '%s' has no transitions and is not an end state",
                stage["name"],
            )

        state_config = StateConfig(
            name=stage["name"],
            is_start=stage.get("is_start", False),
            is_end=stage.get("is_end", False),
            arcs=arcs,
            metadata={
                "prompt": stage.get("prompt", ""),
                "response_template": stage.get("response_template"),
                "clarification_template": stage.get(
                    "clarification_template",
                ),
                "suggestions": stage.get("suggestions", []),
                "help_text": stage.get("help_text"),
                "can_skip": stage.get("can_skip", False),
                "skip_default": stage.get("skip_default"),
                "can_go_back": stage.get("can_go_back", True),
                "tools": stage.get("tools", []),
                "mode": stage.get("mode"),
                "intent_detection": stage.get("intent_detection"),
                "navigation": stage.get("navigation"),
            },
            data_schema=stage.get("schema"),
        )

        return state_config

    def _translate_transition(self, source_stage: str, transition: dict[str, Any], idx: int) -> Any:
        """Translate wizard transition to FSM arc.

        Handles both regular transitions and subflow transitions.
        For subflow transitions (target: "_subflow"), the actual target
        becomes a self-loop and subflow metadata is stored for handling
        at the wizard reasoning level.

        Args:
            source_stage: Source stage name
            transition: Transition configuration dict
            idx: Transition index for naming

        Returns:
            ArcConfig object
        """
        from dataknobs_fsm.config.schema import ArcConfig, FunctionReference

        target = transition.get("target")
        if not target:
            raise ValueError(f"Transition in stage '{source_stage}' missing 'target'")

        # Handle subflow transitions specially
        # For subflow transitions, the FSM stays at the current stage
        # The actual subflow handling happens in WizardReasoning
        is_subflow_transition = target == SUBFLOW_TARGET
        actual_target = source_stage if is_subflow_transition else target

        # Build condition function reference if specified.
        # The function was already pre-registered by
        # _register_inline_conditions, so reference it by name
        # (type="registered") rather than passing inline code.
        # Using type="inline" would cause the FSM builder to create
        # a second function from the code text, without ``bank`` or
        # other wizard-specific names in scope, resulting in silent
        # NameError when conditions reference bank().
        condition = None
        if "condition" in transition:
            condition = FunctionReference(
                type="registered",
                name=f"condition_{source_stage}_{actual_target}_{idx}",
            )

        # Build transform function reference(s) if specified
        # Supports strings, dicts with config, and lists of either:
        #   transform: apply_template                         # single string
        #   transform: [apply_template, save]                 # list of strings
        #   transform: {name: create_corpus, config: {...}}   # single with config
        #   transform:                                        # list with config
        #     - apply_template
        #     - name: create_corpus
        #       config: {corpus_type: quiz_bank}
        transform: FunctionReference | list[FunctionReference] | None = None
        if "transform" in transition:
            raw_transform = transition["transform"]
            if isinstance(raw_transform, list):
                transform = []
                for item in raw_transform:
                    if isinstance(item, dict):
                        params = {}
                        item_config = item.get("config")
                        if item_config:
                            params["config"] = item_config
                        transform.append(
                            FunctionReference(
                                type="registered",
                                name=item["name"],
                                params=params,
                            )
                        )
                    else:
                        transform.append(FunctionReference(type="registered", name=item))
            elif isinstance(raw_transform, dict):
                params = {}
                raw_config = raw_transform.get("config")
                if raw_config:
                    params["config"] = raw_config
                transform = FunctionReference(
                    type="registered",
                    name=raw_transform["name"],
                    params=params,
                )
            else:
                transform = FunctionReference(
                    type="registered",
                    name=raw_transform,
                )

        # Build arc metadata, including subflow config if present
        arc_metadata = dict(transition.get("metadata", {}))
        if is_subflow_transition:
            subflow_config = transition.get("subflow", {})
            arc_metadata["is_subflow_transition"] = True
            arc_metadata["subflow_config"] = {
                "network": subflow_config.get("network"),
                "return_stage": subflow_config.get("return_stage"),
                "data_mapping": subflow_config.get("data_mapping", {}),
                "result_mapping": subflow_config.get("result_mapping", {}),
            }

        arc = ArcConfig(
            target=actual_target,
            condition=condition,
            transform=transform,
            priority=transition.get("priority", idx),
            metadata=arc_metadata,
        )

        return arc

    def _extract_metadata(self, wizard_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Extract stage metadata from wizard config.

        Args:
            wizard_config: Wizard configuration dict

        Returns:
            Dict mapping stage names to their metadata
        """
        metadata = {}

        # Extract global tasks (defined at wizard level, not stage level)
        global_tasks = self._extract_global_tasks(wizard_config)

        for stage in wizard_config.get("stages", []):
            # Extract transition conditions for observability
            transitions = []
            for transition in stage.get("transitions", []):
                trans_meta: dict[str, Any] = {
                    "target": transition.get("target"),
                    "condition": transition.get("condition"),
                    "priority": transition.get("priority"),
                    "derive": transition.get("derive"),
                }

                # Normalize transform references to a list of names
                raw_tf = transition.get("transform")
                if raw_tf is None:
                    tf_names: list[str] = []
                elif isinstance(raw_tf, list):
                    tf_names = [item["name"] if isinstance(item, dict) else item for item in raw_tf]
                elif isinstance(raw_tf, dict):
                    tf_names = [raw_tf["name"]]
                elif isinstance(raw_tf, str):
                    tf_names = [raw_tf]
                else:
                    raise ValueError(
                        f"Unsupported transform value type "
                        f"{type(raw_tf).__name__!r} in stage "
                        f"{stage['name']!r} — expected str, list, or dict"
                    )
                trans_meta["transforms"] = tf_names

                # Include subflow config if this is a subflow transition
                if transition.get("target") == SUBFLOW_TARGET:
                    trans_meta["is_subflow_transition"] = True
                    trans_meta["subflow_config"] = transition.get("subflow", {})
                transitions.append(trans_meta)

            # Extract per-stage tasks
            stage_tasks = self._extract_stage_tasks(stage)

            # Build metadata from field registry
            stage_meta = {f.name: f.extract(stage) for f in _STAGE_FIELDS}
            # Override label default: falls back to stage name, not None
            if stage_meta["label"] is None:
                stage_meta["label"] = stage["name"]
            # Special fields with pre-processed extraction
            stage_meta["transitions"] = transitions
            stage_meta["tasks"] = stage_tasks
            metadata[stage["name"]] = stage_meta

        # Add global tasks to the first stage's metadata
        # The WizardReasoning can then collect them during initialization
        if global_tasks and metadata:
            first_stage = next(iter(metadata))
            metadata[first_stage]["_global_tasks"] = global_tasks

        return metadata

    def _extract_stage_tasks(self, stage: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract task definitions from a stage.

        Args:
            stage: Stage configuration dict

        Returns:
            List of task definition dicts
        """
        tasks = []
        for task_def in stage.get("tasks", []):
            tasks.append(
                {
                    "id": task_def.get("id"),
                    "description": task_def.get("description", task_def.get("id", "")),
                    "required": task_def.get("required", True),
                    "depends_on": task_def.get("depends_on", []),
                    "completed_by": task_def.get("completed_by"),
                    "field_name": task_def.get("field_name"),
                    "tool_name": task_def.get("tool_name"),
                }
            )
        return tasks

    def _extract_global_tasks(self, wizard_config: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract global task definitions from wizard config.

        Global tasks are defined at the wizard level (not per-stage) and
        typically represent cross-cutting concerns like preview, validate,
        and save.

        Args:
            wizard_config: Wizard configuration dict

        Returns:
            List of global task definition dicts
        """
        tasks = []
        for task_def in wizard_config.get("global_tasks", []):
            tasks.append(
                {
                    "id": task_def.get("id"),
                    "description": task_def.get("description", task_def.get("id", "")),
                    "required": task_def.get("required", True),
                    "depends_on": task_def.get("depends_on", []),
                    "completed_by": task_def.get("completed_by"),
                    "field_name": task_def.get("field_name"),
                    "tool_name": task_def.get("tool_name"),
                    "stage": None,  # Mark as global
                }
            )
        return tasks

    def _register_inline_conditions(
        self, builder: FSMBuilder, wizard_config: dict[str, Any]
    ) -> None:
        """Pre-register inline condition functions with the builder.

        This ensures consistent function naming between translation
        and execution.

        Args:
            builder: FSMBuilder to register functions with
            wizard_config: Wizard configuration dict
        """
        for stage in wizard_config.get("stages", []):
            for idx, transition in enumerate(stage.get("transitions", [])):
                if "condition" not in transition:
                    continue

                condition_code = transition["condition"]
                target = transition.get("target", "unknown")
                # For subflow transitions the FSM arc target is the
                # source stage (self-loop), not the raw "_subflow"
                # sentinel.  Mirror _translate_transition's logic so
                # the registered name matches the FunctionReference.
                if target == SUBFLOW_TARGET:
                    actual_target = stage["name"]
                else:
                    actual_target = target
                func_name = f"condition_{stage['name']}_{actual_target}_{idx}"

                # Wrap in return statement if not already
                if not condition_code.strip().startswith("return"):
                    condition_code = f"return {condition_code}"

                # Create the function
                try:
                    # Create a function that evaluates the condition
                    # using the shared safe expression engine.
                    def make_condition(code: str, name: str) -> Callable[[Any, Any], bool]:
                        def condition_func(data: dict[str, Any], context: Any = None) -> bool:
                            result = safe_eval(
                                code,
                                scope={
                                    "data": data,
                                    "has": lambda key: data.get(key) is not None,
                                    "bank": data.get("_bank_fn", _null_bank),
                                },
                                coerce_bool=True,
                                default=False,
                            )
                            if not result.success:
                                logger.warning(
                                    "Condition '%s' evaluation failed: %s (code=%r, data_keys=%s)",
                                    name,
                                    result.error,
                                    code,
                                    list(data.keys()),
                                )
                            else:
                                logger.debug(
                                    "Condition '%s': code=%r, result=%s, data_keys=%s",
                                    name,
                                    code,
                                    result.value,
                                    list(data.keys()),
                                )
                            return result.value

                        return condition_func

                    builder.register_function(func_name, make_condition(condition_code, func_name))
                except Exception as e:
                    logger.warning("Failed to register condition '%s': %s", func_name, e)

    def _load_subflow_networks(
        self,
        wizard_config: dict[str, Any],
        custom_functions: dict[str, Callable[..., Any] | str] | None,
        anchor: PathAnchor | None,
    ) -> dict[str, WizardFSM]:
        """Load subflow networks referenced in transitions.

        Scans all transitions for subflow references and loads the
        corresponding wizard configurations.

        Args:
            wizard_config: Main wizard configuration dict
            custom_functions: Custom functions to pass to subflows
            anchor: The wizard's config tree and the position within it that
                names resolve from, or ``None`` when no base path was given
                and file probes are therefore skipped

        Returns:
            Dict mapping subflow names to WizardFSM instances
        """
        subflow_registry: dict[str, WizardFSM] = {}

        # Collect all referenced subflow networks
        subflow_refs: set[str] = set()
        for stage in wizard_config.get("stages", []):
            for transition in stage.get("transitions", []):
                if transition.get("target") == SUBFLOW_TARGET:
                    subflow_config = transition.get("subflow", {})
                    network_name = subflow_config.get("network")
                    if network_name:
                        subflow_refs.add(network_name)

        # Also check for explicitly defined subflows in config
        explicit_subflows = wizard_config.get("subflows", {})
        for name in explicit_subflows:
            subflow_refs.add(name)

        if not subflow_refs:
            return subflow_registry

        # Load each referenced subflow
        for subflow_name in subflow_refs:
            try:
                subflow_fsm = self._load_single_subflow(
                    subflow_name,
                    wizard_config,
                    custom_functions,
                    anchor,
                )
                if subflow_fsm:
                    subflow_registry[subflow_name] = subflow_fsm
                    logger.debug("Loaded subflow: %s", subflow_name)
            except PathEscapeError:
                # Refusing a name that addresses outside the config
                # directory is not "this subflow failed to load" — it is
                # the config asking for something it may not have.
                # Rewriting it into a bare ValueError here would undo the
                # narrowing the guard exists to provide, one frame above
                # the guard.
                logger.error(
                    "Refused subflow '%s': addresses outside the config tree", subflow_name
                )
                raise
            except Exception as e:
                logger.error("Failed to load subflow '%s': %s", subflow_name, e)
                raise ValueError(f"Failed to load subflow '{subflow_name}': {e}") from e

        return subflow_registry

    def _load_single_subflow(
        self,
        subflow_name: str,
        wizard_config: dict[str, Any],
        custom_functions: dict[str, Callable[..., Any] | str] | None,
        anchor: PathAnchor | None,
    ) -> WizardFSM | None:
        """Load a single subflow network.

        Attempts to load the subflow from:
        1. Explicit subflow definition in wizard_config["subflows"]
        2. File path relative to the loading wizard's own directory
        3. File path in that directory's subflows/ subdirectory

        The name comes out of config *content* — a ``subflows:`` key or a
        transition's ``subflow.network`` value — so both file probes are
        bounded. A name that leaves the tree, via ``..`` or by being
        absolute, raises :class:`~dataknobs_common.paths.PathEscapeError`
        rather than loading a state machine from outside it. A name in a
        subdirectory is legal; the ``subflows/`` layout the second probe
        serves is exactly that.

        **Resolution is per-hop; the boundary is not.** A subflow loads its
        own subflows relative to *itself*, because that is where a nested
        wizard's names have always been read from. What it may reach is the
        anchor's root, fixed when the outermost wizard was loaded — so
        ``cfg/subflows/a.yaml`` naming ``../shared`` reaches
        ``cfg/shared.yaml``, which is inside the wizard's tree, while
        nothing reaches outside it at any depth.

        Args:
            subflow_name: Name of the subflow to load
            wizard_config: Main wizard configuration dict
            custom_functions: Custom functions to pass to subflow
            anchor: The config tree and the position within it that this
                wizard's names resolve from, or ``None`` if no base path
                was given

        Returns:
            WizardFSM for the subflow, or None if not found

        Raises:
            PathEscapeError: If ``subflow_name`` addresses a file outside the
                tree under *either* probe. Both are candidate readings of one
                name, so a name that escapes under either is refused rather
                than silently reinterpreted as the other — the same rule
                :func:`~dataknobs_common.config_loading.find_config_file`
                applies across its extensions.
        """
        # Check for inline subflow definition
        explicit_subflows = wizard_config.get("subflows", {})
        if subflow_name in explicit_subflows:
            subflow_config = explicit_subflows[subflow_name]
            return self.load_from_dict(
                subflow_config,
                custom_functions,
                config_base_path=anchor.base if anchor else None,
                config_root=anchor.root if anchor else None,
            )

        # Try to load from file
        if anchor is None:
            logger.warning(
                "Cannot load subflow '%s' from file: no config_base_path provided",
                subflow_name,
            )
            return None

        # Both probes are guarded, and each opens the path the guard
        # returned rather than recomposing from the raw name: a symlinked
        # subdirectory plus a ``..`` resolves through the link's target.
        for parts in ((f"{subflow_name}.yaml",), ("subflows", f"{subflow_name}.yaml")):
            subflow_path = anchor.resolve(
                *parts,
                what="subflow name",
                outside="the wizard's config tree",
                supplied=subflow_name,
            )
            if subflow_path.exists():
                # The root travels with the recursion; the position moves to
                # the subflow's own directory when `load` re-anchors there.
                return self.load(str(subflow_path), custom_functions, config_root=anchor.root)

        logger.warning(
            "Subflow '%s' not found in config or as file at %s",
            subflow_name,
            anchor.base,
        )
        return None


def load_wizard_config(
    config_path: str | Path,
    custom_functions: dict[str, Callable[..., Any] | str] | None = None,
    transform_context_factory: Callable[..., Any] | None = None,
) -> WizardFSM:
    """Convenience function to load wizard config.

    Args:
        config_path: Path to wizard YAML config file
        custom_functions: Optional custom functions for transitions.
            Values can be either callables or "module:function" strings.
        transform_context_factory: Optional callable that receives a
            :class:`FunctionContext` and returns the application-specific
            context for transforms. If ``None``, a default factory is used.

    Returns:
        Configured WizardFSM instance
    """
    loader = WizardConfigLoader()
    return loader.load(config_path, custom_functions, transform_context_factory)
