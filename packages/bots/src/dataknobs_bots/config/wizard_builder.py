"""Fluent builder for wizard configurations.

Provides a composable builder for creating wizard configurations
programmatically. Produces output compatible with
``WizardConfigLoader.load_from_dict()``.

Example:
    ```python
    from dataknobs_bots.config import WizardConfigBuilder

    wizard = (
        WizardConfigBuilder("my-wizard")
        .set_version("1.0.0")
        .set_settings(tool_reasoning="react", max_tool_iterations=3)
        .add_conversation_stage(
            name="chat",
            prompt="Have a conversation about {subject}.",
            tools=["knowledge_search"],
            is_start=True,
        )
        .add_structured_stage(
            name="quiz",
            prompt="Let's start a quiz.",
            schema={"type": "object", "properties": {"num_q": {"type": "integer"}}},
        )
        .add_transition("chat", "quiz", condition="data.get('_intent') == 'quiz'")
        .build()
    )

    yaml_str = wizard.to_yaml()
    wizard.to_file("configs/wizards/my-wizard.yaml")
    ```
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from typing_extensions import Self

from .validation import ValidationResult

if TYPE_CHECKING:
    from .tool_catalog import ToolCatalog

logger = logging.getLogger(__name__)

# Valid values for enum-like fields
_VALID_REASONING = {"single", "react"}
_VALID_MODE = {"conversation"}


@dataclass(frozen=True)
class TransitionConfig:
    """Configuration for a single stage transition."""

    target: str
    condition: str | None = None
    transform: str | list[str] | None = None
    priority: int | None = None
    derive: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    subflow: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict compatible with WizardConfigLoader."""
        d: dict[str, Any] = {"target": self.target}
        if self.condition is not None:
            d["condition"] = self.condition
        if self.transform is not None:
            d["transform"] = self.transform
        if self.priority is not None:
            d["priority"] = self.priority
        if self.derive is not None:
            d["derive"] = self.derive
        if self.metadata is not None:
            d["metadata"] = self.metadata
        if self.subflow is not None:
            d["subflow"] = self.subflow
        return d


@dataclass(frozen=True)
class IntentDetectionConfig:
    """Configuration for intent detection on a stage."""

    method: str
    intents: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict compatible with WizardConfigLoader."""
        d: dict[str, Any] = {"method": self.method}
        if self.intents:
            d["intents"] = [dict(i) for i in self.intents]
        return d


@dataclass(frozen=True)
class ContextGenerationConfig:
    """Configuration for LLM-generated context variables."""

    variables: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict compatible with WizardConfigLoader."""
        return {"variables": dict(self.variables)}


@dataclass(frozen=True)
class StageConfig:
    """Configuration for a single wizard stage."""

    name: str
    prompt: str
    # Role flags
    is_start: bool = False
    is_end: bool = False
    # Navigation
    can_skip: bool = False
    skip_default: Any = None
    can_go_back: bool = True
    auto_advance: bool = False
    confirm_on_new_data: bool = False
    # Display
    label: str | None = None
    suggestions: tuple[str, ...] = ()
    help_text: str | None = None
    # Schema
    schema: dict[str, Any] | None = None
    # Transitions
    transitions: tuple[TransitionConfig, ...] = ()
    # Tools and reasoning
    tools: tuple[str, ...] = ()
    reasoning: str | None = None
    max_iterations: int | None = None
    extraction_model: str | None = None
    # Response generation
    response_template: str | None = None
    llm_assist: bool = False
    llm_assist_prompt: str | None = None
    context_generation: ContextGenerationConfig | None = None
    # Conversation mode
    mode: str | None = None
    intent_detection: IntentDetectionConfig | None = None
    # Tasks
    tasks: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict compatible with WizardConfigLoader."""
        d: dict[str, Any] = {"name": self.name, "prompt": self.prompt}
        if self.is_start:
            d["is_start"] = True
        if self.is_end:
            d["is_end"] = True
        if self.can_skip:
            d["can_skip"] = True
        if self.skip_default is not None:
            d["skip_default"] = self.skip_default
        if not self.can_go_back:
            d["can_go_back"] = False
        if self.auto_advance:
            d["auto_advance"] = True
        if self.confirm_on_new_data:
            d["confirm_on_new_data"] = True
        if self.label is not None:
            d["label"] = self.label
        if self.suggestions:
            d["suggestions"] = list(self.suggestions)
        if self.help_text is not None:
            d["help_text"] = self.help_text
        if self.schema is not None:
            d["schema"] = self.schema
        if self.transitions:
            d["transitions"] = [t.to_dict() for t in self.transitions]
        if self.tools:
            d["tools"] = list(self.tools)
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        if self.max_iterations is not None:
            d["max_iterations"] = self.max_iterations
        if self.extraction_model is not None:
            d["extraction_model"] = self.extraction_model
        if self.response_template is not None:
            d["response_template"] = self.response_template
        if self.llm_assist:
            d["llm_assist"] = True
        if self.llm_assist_prompt is not None:
            d["llm_assist_prompt"] = self.llm_assist_prompt
        if self.context_generation is not None:
            d["context_generation"] = self.context_generation.to_dict()
        if self.mode is not None:
            d["mode"] = self.mode
        if self.intent_detection is not None:
            d["intent_detection"] = self.intent_detection.to_dict()
        if self.tasks:
            d["tasks"] = [dict(t) for t in self.tasks]
        return d


@dataclass(frozen=True)
class WizardConfig:
    """Immutable wizard configuration produced by WizardConfigBuilder.

    This is the validated output of the builder. Use ``to_dict()`` to
    get a dict compatible with ``WizardConfigLoader.load_from_dict()``,
    ``to_yaml()`` for YAML serialization, or ``to_file()`` to write
    directly to disk.
    """

    name: str
    version: str
    description: str
    settings: dict[str, Any]
    stages: tuple[StageConfig, ...]
    global_tasks: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict compatible with WizardConfigLoader.load_from_dict()."""
        d: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
        }
        if self.description:
            d["description"] = self.description
        if self.settings:
            d["settings"] = dict(self.settings)
        d["stages"] = [s.to_dict() for s in self.stages]
        if self.global_tasks:
            d["global_tasks"] = [dict(t) for t in self.global_tasks]
        return d

    def to_yaml(self) -> str:
        """Serialize as YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def to_file(self, path: str | Path) -> None:
        """Write YAML to a file.

        Args:
            path: File path to write to. Parent directories are created
                if they do not exist.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.to_yaml(), encoding="utf-8")


class WizardConfigBuilder:
    """Fluent builder for wizard configurations.

    Provides setter methods for constructing wizard configs
    programmatically. Produces output compatible with
    ``WizardConfigLoader.load_from_dict()``.

    Example:
        ```python
        wizard = (
            WizardConfigBuilder("onboarding")
            .add_structured_stage("welcome", "What is your name?", is_start=True)
            .add_end_stage("done", "All set!")
            .add_transition("welcome", "done")
            .build()
        )
        ```
    """

    def __init__(self, name: str) -> None:
        """Initialize the builder.

        Args:
            name: Wizard identifier.
        """
        self._name = name
        self._version = "1.0"
        self._description = ""
        self._settings: dict[str, Any] = {}
        self._stages: list[StageConfig] = []
        self._pending_transitions: list[tuple[str, TransitionConfig]] = []
        self._pending_intents: list[tuple[str, IntentDetectionConfig]] = []
        self._global_tasks: list[dict[str, Any]] = []
        self._tool_catalog: ToolCatalog | None = None

    # -- Metadata --

    def set_version(self, version: str) -> Self:
        """Set the wizard version.

        Args:
            version: Semantic version string.

        Returns:
            self for method chaining.
        """
        self._version = version
        return self

    def set_description(self, description: str) -> Self:
        """Set the wizard description.

        Args:
            description: Human-readable description.

        Returns:
            self for method chaining.
        """
        self._description = description
        return self

    # -- Settings --

    def set_settings(self, **kwargs: Any) -> Self:
        """Set wizard-level settings.

        Args:
            **kwargs: Settings such as tool_reasoning, max_tool_iterations,
                auto_advance_filled_stages, extraction_scope,
                conflict_strategy, timeout_seconds, etc.

        Returns:
            self for method chaining.
        """
        self._settings.update(kwargs)
        return self

    def set_tool_catalog(self, catalog: ToolCatalog) -> Self:
        """Set a tool catalog for validation of stage tool references.

        When set, ``build()`` validates that all tool names referenced in
        stages exist in the catalog.

        Args:
            catalog: Tool catalog for name validation.

        Returns:
            self for method chaining.
        """
        self._tool_catalog = catalog
        return self

    # -- Stage addition --

    def add_conversation_stage(
        self,
        name: str,
        prompt: str,
        tools: list[str] | None = None,
        is_start: bool = False,
        suggestions: list[str] | None = None,
        intent_detection: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Add a conversation-mode stage.

        Automatically sets ``mode: conversation``. Use this for
        free-form chat stages.

        Args:
            name: Stage identifier.
            prompt: User-facing prompt.
            tools: Available tool names.
            is_start: Whether this is the start stage.
            suggestions: Quick-reply suggestions.
            intent_detection: Intent detection config dict with
                ``method`` and ``intents`` keys.
            **kwargs: Additional StageConfig fields.

        Returns:
            self for method chaining.
        """
        intent_cfg = None
        if intent_detection is not None:
            intent_cfg = IntentDetectionConfig(
                method=intent_detection.get("method", "keyword"),
                intents=tuple(intent_detection.get("intents", [])),
            )
        stage = StageConfig(
            name=name,
            prompt=prompt,
            mode="conversation",
            tools=tuple(tools) if tools else (),
            is_start=is_start,
            suggestions=tuple(suggestions) if suggestions else (),
            intent_detection=intent_cfg,
            **kwargs,
        )
        self._stages.append(stage)
        return self

    def add_structured_stage(
        self,
        name: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        tools: list[str] | None = None,
        is_start: bool = False,
        is_end: bool = False,
        can_skip: bool = False,
        skip_default: Any = None,
        suggestions: list[str] | None = None,
        response_template: str | None = None,
        help_text: str | None = None,
        reasoning: str | None = None,
        max_iterations: int | None = None,
        context_generation: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Add a structured (data-collection) stage.

        Args:
            name: Stage identifier.
            prompt: User-facing prompt.
            schema: JSON Schema for input validation.
            tools: Available tool names.
            is_start: Whether this is the start stage.
            is_end: Whether this is an end stage.
            can_skip: Whether the user can skip this stage.
            skip_default: Default value if skipped.
            suggestions: Quick-reply suggestions.
            response_template: Template-driven response (bypasses LLM).
            help_text: Help message for the user.
            reasoning: Reasoning mode ('single' or 'react').
            max_iterations: Max iterations for ReAct reasoning.
            context_generation: LLM context generation config with
                ``variables`` key.
            **kwargs: Additional StageConfig fields.

        Returns:
            self for method chaining.
        """
        ctx_cfg = None
        if context_generation is not None:
            ctx_cfg = ContextGenerationConfig(
                variables=context_generation.get("variables", {}),
            )
        stage = StageConfig(
            name=name,
            prompt=prompt,
            schema=schema,
            tools=tuple(tools) if tools else (),
            is_start=is_start,
            is_end=is_end,
            can_skip=can_skip,
            skip_default=skip_default,
            suggestions=tuple(suggestions) if suggestions else (),
            response_template=response_template,
            help_text=help_text,
            reasoning=reasoning,
            max_iterations=max_iterations,
            context_generation=ctx_cfg,
            **kwargs,
        )
        self._stages.append(stage)
        return self

    def add_end_stage(
        self,
        name: str,
        prompt: str,
        **kwargs: Any,
    ) -> Self:
        """Add an end stage.

        Automatically sets ``is_end=True``.

        Args:
            name: Stage identifier.
            prompt: User-facing prompt.
            **kwargs: Additional StageConfig fields.

        Returns:
            self for method chaining.
        """
        stage = StageConfig(
            name=name,
            prompt=prompt,
            is_end=True,
            **kwargs,
        )
        self._stages.append(stage)
        return self

    def add_stage(self, stage: StageConfig) -> Self:
        """Add a pre-built StageConfig directly.

        Args:
            stage: A StageConfig instance.

        Returns:
            self for method chaining.
        """
        self._stages.append(stage)
        return self

    # -- Transitions --

    def add_transition(
        self,
        from_stage: str,
        to_stage: str,
        condition: str | None = None,
        transform: str | list[str] | None = None,
        priority: int | None = None,
        derive: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        """Add a transition between stages.

        Transitions are attached to their source stage when ``build()``
        is called. This allows stages and transitions to be added in
        any order.

        Args:
            from_stage: Source stage name.
            to_stage: Target stage name.
            condition: Python expression evaluated with ``data`` in scope.
            transform: Transform function name(s).
            priority: Evaluation priority.
            derive: Data derivation rules.
            metadata: Custom metadata.

        Returns:
            self for method chaining.
        """
        transition = TransitionConfig(
            target=to_stage,
            condition=condition,
            transform=transform,
            priority=priority,
            derive=derive,
            metadata=metadata,
        )
        self._pending_transitions.append((from_stage, transition))
        return self

    # -- Intent detection --

    def add_intent_detection(
        self,
        stage: str,
        method: str = "keyword",
        intents: list[dict[str, Any]] | None = None,
    ) -> Self:
        """Add intent detection to a stage.

        Intent detection is attached to its stage when ``build()``
        is called.

        Args:
            stage: Stage name to add intent detection to.
            method: Detection method (e.g., 'keyword', 'llm').
            intents: List of intent definitions, each with at least
                an ``id`` key and method-specific fields.

        Returns:
            self for method chaining.
        """
        intent_cfg = IntentDetectionConfig(
            method=method,
            intents=tuple(intents) if intents else (),
        )
        self._pending_intents.append((stage, intent_cfg))
        return self

    # -- Global tasks --

    def add_global_task(
        self,
        task_id: str,
        description: str,
        required: bool = True,
        depends_on: list[str] | None = None,
        completed_by: str | None = None,
        field_name: str | None = None,
        tool_name: str | None = None,
    ) -> Self:
        """Add a wizard-level (global) task.

        Args:
            task_id: Task identifier.
            description: Human-readable description.
            required: Whether the task must be completed.
            depends_on: Task IDs this depends on.
            completed_by: Completion mechanism.
            field_name: Field name for field-driven completion.
            tool_name: Tool name for tool-driven completion.

        Returns:
            self for method chaining.
        """
        task: dict[str, Any] = {
            "id": task_id,
            "description": description,
            "required": required,
        }
        if depends_on:
            task["depends_on"] = depends_on
        if completed_by is not None:
            task["completed_by"] = completed_by
        if field_name is not None:
            task["field_name"] = field_name
        if tool_name is not None:
            task["tool_name"] = tool_name
        self._global_tasks.append(task)
        return self

    # -- Output --

    def validate(self) -> ValidationResult:
        """Validate the current configuration without building.

        Returns:
            ValidationResult with any errors and warnings.
        """
        return self._validate(self._assemble_stages())

    def build(self) -> WizardConfig:
        """Build the immutable WizardConfig.

        Validates and raises ValueError if there are errors.

        Returns:
            Validated, immutable WizardConfig.

        Raises:
            ValueError: If the configuration has validation errors.
        """
        assembled = self._assemble_stages()
        result = self._validate(assembled)
        if not result.valid:
            raise ValueError(
                "Wizard configuration validation failed:\n"
                + "\n".join(f"  - {e}" for e in result.errors)
            )
        for warning in result.warnings:
            logger.warning("Wizard config warning: %s", warning)

        return WizardConfig(
            name=self._name,
            version=self._version,
            description=self._description,
            settings=dict(self._settings),
            stages=tuple(assembled),
            global_tasks=tuple(self._global_tasks),
        )

    # -- Roundtrip --

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> WizardConfigBuilder:
        """Create a builder pre-populated from a wizard config dict.

        Args:
            config: Wizard configuration dictionary (same format as
                YAML wizard configs).

        Returns:
            A new builder instance with the config loaded.
        """
        builder = cls(config.get("name", "wizard"))
        builder._version = config.get("version", "1.0")
        builder._description = config.get("description", "")
        builder._settings = dict(config.get("settings", {}))
        builder._global_tasks = [
            dict(t) for t in config.get("global_tasks", [])
        ]

        for stage_dict in config.get("stages", []):
            stage = _stage_from_dict(stage_dict)
            builder._stages.append(stage)

        return builder

    @classmethod
    def from_file(cls, path: str | Path) -> WizardConfigBuilder:
        """Create a builder from a wizard YAML file.

        Args:
            path: Path to a wizard YAML configuration file.

        Returns:
            A new builder instance with the config loaded.
        """
        file_path = Path(path)
        with file_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    # -- Convenience factory --

    @classmethod
    def conversation_start(
        cls,
        name: str,
        prompt: str,
        tools: list[str] | None = None,
        tool_reasoning: str = "react",
        max_tool_iterations: int = 3,
    ) -> WizardConfigBuilder:
        """Create a builder pre-configured for the conversation-start pattern.

        This is the minimal unified wizard config: a single
        ``mode: conversation`` stage with tool reasoning enabled.

        Args:
            name: Wizard identifier.
            prompt: Conversation prompt.
            tools: Available tool names.
            tool_reasoning: Tool reasoning mode ('single' or 'react').
            max_tool_iterations: Max tool reasoning iterations.

        Returns:
            A pre-configured builder instance.
        """
        return (
            cls(name)
            .set_version("1.0.0")
            .set_settings(
                tool_reasoning=tool_reasoning,
                max_tool_iterations=max_tool_iterations,
            )
            .add_conversation_stage(
                name="conversation",
                prompt=prompt,
                tools=tools,
                is_start=True,
            )
        )

    # -- Private --

    def _assemble_stages(self) -> list[StageConfig]:
        """Assemble stages with their pending transitions and intents."""
        stage_map: dict[str, StageConfig] = {s.name: s for s in self._stages}

        # Attach pending transitions to their source stages
        transitions_by_stage: dict[str, list[TransitionConfig]] = {}
        for from_stage, transition in self._pending_transitions:
            transitions_by_stage.setdefault(from_stage, []).append(transition)

        # Attach pending intent detection to their stages
        intents_by_stage: dict[str, IntentDetectionConfig] = {}
        for stage_name, intent_cfg in self._pending_intents:
            intents_by_stage[stage_name] = intent_cfg

        assembled: list[StageConfig] = []
        for stage in self._stages:
            extra_transitions = transitions_by_stage.get(stage.name, [])
            intent_override = intents_by_stage.get(stage.name)

            if extra_transitions or intent_override:
                # Rebuild stage with merged transitions and/or intent
                new_transitions = stage.transitions + tuple(extra_transitions)
                new_intent = intent_override or stage.intent_detection
                stage = StageConfig(
                    name=stage.name,
                    prompt=stage.prompt,
                    is_start=stage.is_start,
                    is_end=stage.is_end,
                    can_skip=stage.can_skip,
                    skip_default=stage.skip_default,
                    can_go_back=stage.can_go_back,
                    auto_advance=stage.auto_advance,
                    label=stage.label,
                    suggestions=stage.suggestions,
                    help_text=stage.help_text,
                    schema=stage.schema,
                    transitions=new_transitions,
                    tools=stage.tools,
                    reasoning=stage.reasoning,
                    max_iterations=stage.max_iterations,
                    extraction_model=stage.extraction_model,
                    response_template=stage.response_template,
                    llm_assist=stage.llm_assist,
                    llm_assist_prompt=stage.llm_assist_prompt,
                    context_generation=stage.context_generation,
                    mode=stage.mode,
                    intent_detection=new_intent,
                    tasks=stage.tasks,
                )
            assembled.append(stage)

        # Check for transitions referencing non-existent source stages
        for from_stage, _ in self._pending_transitions:
            if from_stage not in stage_map:
                # Will be caught by validation
                pass

        return assembled

    def _validate(self, stages: list[StageConfig]) -> ValidationResult:
        """Run wizard-specific validations."""
        result = ValidationResult.ok()

        # Must have at least one stage
        if not stages:
            return ValidationResult.error(
                "Wizard must have at least one stage"
            )

        stage_names = {s.name for s in stages}

        # Check for duplicate stage names
        seen: set[str] = set()
        for stage in stages:
            if stage.name in seen:
                result = result.merge(
                    ValidationResult.error(
                        f"Duplicate stage name: '{stage.name}'"
                    )
                )
            seen.add(stage.name)

        # Exactly one start stage
        start_stages = [s for s in stages if s.is_start]
        if len(start_stages) == 0:
            result = result.merge(
                ValidationResult.error("Wizard must have exactly one start stage")
            )
        elif len(start_stages) > 1:
            names = [s.name for s in start_stages]
            result = result.merge(
                ValidationResult.error(
                    f"Wizard has multiple start stages: {names}"
                )
            )

        # All transition targets reference existing stages
        for stage in stages:
            for transition in stage.transitions:
                if (
                    transition.target not in stage_names
                    and transition.target != "_subflow"
                ):
                    result = result.merge(
                        ValidationResult.error(
                            f"Stage '{stage.name}' has transition to "
                            f"unknown stage '{transition.target}'"
                        )
                    )

        # Check for transitions from non-existent source stages
        for from_stage, transition in self._pending_transitions:
            if from_stage not in stage_names:
                result = result.merge(
                    ValidationResult.error(
                        f"Transition from unknown stage '{from_stage}' "
                        f"to '{transition.target}'"
                    )
                )

        # Check for intent detection on non-existent stages
        for stage_name, _ in self._pending_intents:
            if stage_name not in stage_names:
                result = result.merge(
                    ValidationResult.error(
                        f"Intent detection references unknown stage "
                        f"'{stage_name}'"
                    )
                )

        # Validate reasoning values
        for stage in stages:
            if stage.reasoning is not None and stage.reasoning not in _VALID_REASONING:
                result = result.merge(
                    ValidationResult.error(
                        f"Stage '{stage.name}' has invalid reasoning "
                        f"value '{stage.reasoning}'. "
                        f"Valid values: {sorted(_VALID_REASONING)}"
                    )
                )

        # Validate mode values
        for stage in stages:
            if stage.mode is not None and stage.mode not in _VALID_MODE:
                result = result.merge(
                    ValidationResult.error(
                        f"Stage '{stage.name}' has invalid mode "
                        f"value '{stage.mode}'. "
                        f"Valid values: {sorted(_VALID_MODE)}"
                    )
                )

        # Warnings: max_iterations without reasoning
        for stage in stages:
            if stage.max_iterations is not None and stage.reasoning is None:
                result = result.merge(
                    ValidationResult.warning(
                        f"Stage '{stage.name}' sets max_iterations "
                        f"but has no reasoning mode"
                    )
                )

        # Warnings: end stages with transitions
        for stage in stages:
            if stage.is_end and stage.transitions:
                result = result.merge(
                    ValidationResult.warning(
                        f"End stage '{stage.name}' has transitions "
                        f"that will never be followed"
                    )
                )

        # Warnings: orphan stages (unreachable from start)
        if start_stages and len(stages) > 1:
            reachable = _find_reachable(stages, start_stages[0].name)
            for stage in stages:
                if stage.name not in reachable and not stage.is_start:
                    result = result.merge(
                        ValidationResult.warning(
                            f"Stage '{stage.name}' is not reachable "
                            f"from the start stage"
                        )
                    )

        # Warnings: pure LLM-driven data-collection stages
        for stage in stages:
            if (
                not stage.is_end
                and stage.mode != "conversation"
                and not stage.schema
                and not stage.response_template
            ):
                result = result.merge(
                    ValidationResult.warning(
                        f"Stage '{stage.name}' has no schema and no "
                        f"response_template — pure LLM-driven stages "
                        f"are unreliable for data collection"
                    )
                )

        # Warnings: Python str.format() in templates (should be Jinja2)
        _fmt_re = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")
        for stage in stages:
            if stage.response_template:
                match = _fmt_re.search(stage.response_template)
                if match:
                    result = result.merge(
                        ValidationResult.warning(
                            f"Stage '{stage.name}' response_template uses "
                            f"Python format syntax {{{match.group(1)}}} — "
                            f"did you mean Jinja2 {{{{ {match.group(1)} }}}}?"
                        )
                    )

        # Catalog validation: check stage tool names exist in catalog
        if self._tool_catalog is not None:
            for stage in stages:
                for tool_name in stage.tools:
                    if not self._tool_catalog.has(tool_name):
                        result = result.merge(
                            ValidationResult.error(
                                f"Stage '{stage.name}' references unknown "
                                f"tool: '{tool_name}'"
                            )
                        )

        return result


def _find_reachable(stages: list[StageConfig], start: str) -> set[str]:
    """Find all stages reachable from a start stage via BFS."""
    adjacency: dict[str, list[str]] = {}
    for stage in stages:
        targets = [
            t.target for t in stage.transitions
            if t.target != "_subflow"
        ]
        adjacency[stage.name] = targets

    visited: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for target in adjacency.get(current, []):
            if target not in visited:
                queue.append(target)
    return visited


def _stage_from_dict(d: dict[str, Any]) -> StageConfig:
    """Construct a StageConfig from a wizard config stage dict."""
    intent_detection = None
    if d.get("intent_detection"):
        raw_intent = d["intent_detection"]
        intent_detection = IntentDetectionConfig(
            method=raw_intent.get("method", "keyword"),
            intents=tuple(raw_intent.get("intents", [])),
        )

    context_generation = None
    if d.get("context_generation"):
        raw_ctx = d["context_generation"]
        context_generation = ContextGenerationConfig(
            variables=raw_ctx.get("variables", {}),
        )

    transitions = tuple(
        TransitionConfig(
            target=t["target"],
            condition=t.get("condition"),
            transform=t.get("transform"),
            priority=t.get("priority"),
            derive=t.get("derive"),
            metadata=t.get("metadata"),
            subflow=t.get("subflow"),
        )
        for t in d.get("transitions", [])
    )

    return StageConfig(
        name=d["name"],
        prompt=d.get("prompt", ""),
        is_start=d.get("is_start", False),
        is_end=d.get("is_end", False),
        can_skip=d.get("can_skip", False),
        skip_default=d.get("skip_default"),
        can_go_back=d.get("can_go_back", True),
        auto_advance=d.get("auto_advance", False),
        label=d.get("label"),
        suggestions=tuple(d.get("suggestions", [])),
        help_text=d.get("help_text"),
        schema=d.get("schema"),
        transitions=transitions,
        tools=tuple(d.get("tools", [])),
        reasoning=d.get("reasoning"),
        max_iterations=d.get("max_iterations"),
        extraction_model=d.get("extraction_model"),
        response_template=d.get("response_template"),
        llm_assist=d.get("llm_assist", False),
        llm_assist_prompt=d.get("llm_assist_prompt"),
        context_generation=context_generation,
        mode=d.get("mode"),
        intent_detection=intent_detection,
        tasks=tuple(d.get("tasks", [])),
    )
