"""Testing utilities for dataknobs-bots.

This module provides testing infrastructure for bot interactions:

High-level helpers (preferred for new tests):
- ``BotTestHarness``: Single-object test setup for DynaBot. Creates a
  DynaBot via ``from_config()``, wires EchoProvider and ConfigurableExtractor,
  and provides ``chat()``/``greet()`` with automatic wizard state capture.
  Supports tools (for ReAct testing) and middleware injection.
- ``WizardConfigBuilder``: Fluent builder for wizard config dicts, replacing
  verbose inline dict construction with a readable chained API.
- ``GroundedConfigBuilder``: Fluent builder for grounded-strategy bot config
  dicts.  Produces a complete ``bot_config`` dict for ``BotTestHarness``.
- ``TurnResult``: Dataclass capturing response + wizard state snapshot per turn.

Low-level primitives:
- ``inject_providers``: Injects LLM providers (and optionally a
  ``ConfigurableExtractor``) into a DynaBot instance.
- ``CaptureReplay``: Loads a capture JSON file and creates pre-loaded
  EchoProviders for deterministic replay of recorded conversations.
- ``ErrorRaisingStrategy``: Test strategy that raises on every hook.

Example — BotTestHarness (preferred):
    ```python
    from dataknobs_bots.testing import BotTestHarness, WizardConfigBuilder

    config = (WizardConfigBuilder("test")
        .stage("gather", is_start=True, prompt="Tell me your name.")
            .field("name", field_type="string", required=True)
            .transition("done", "data.get('name')")
        .stage("done", is_end=True, prompt="All done!")
        .build())

    async with await BotTestHarness.create(
        wizard_config=config,
        main_responses=["Got it!"],
        extraction_results=[[{"name": "Alice"}]],
    ) as harness:
        result = await harness.chat("My name is Alice")
        assert harness.wizard_data["name"] == "Alice"
    ```

Example — CaptureReplay (low-level):
    ```python
    from dataknobs_bots.testing import CaptureReplay

    replay = CaptureReplay.from_file("captures/configbot_basic.json")
    bot = DynaBot(config=config, llm=real_provider, ...)
    replay.inject_into_bot(bot)
    ```
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from dataknobs_llm import EchoProvider
from dataknobs_llm.llm.base import AsyncLLMProvider, LLMResponse
from dataknobs_llm.testing import (
    ConfigurableExtractor,
    SimpleExtractionResult,
    llm_response_from_dict,
)

from .reasoning.base import ReasoningManagerProtocol, ReasoningStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# WizardConfigBuilder — fluent wizard config construction
# =============================================================================


class WizardConfigBuilder:
    """Fluent builder for wizard configuration dicts.

    Replaces verbose inline dict construction (40+ lines) with a readable
    chained API. Performs build-time validation to catch common mistakes.

    Example:
        ```python
        config = (WizardConfigBuilder("my-wizard")
            .stage("gather", is_start=True, prompt="Tell me your name.")
                .field("name", field_type="string", required=True)
                .field("domain", field_type="string")
                .transition("done", "data.get('name') and data.get('domain')")
            .stage("done", is_end=True, prompt="All done!")
            .settings(extraction_scope="current_message")
            .build())
        ```
    """

    def __init__(self, name: str, version: str = "1.0") -> None:
        self._name = name
        self._version = version
        self._stages: list[dict[str, Any]] = []
        self._settings: dict[str, Any] = {}
        self._subflows: dict[str, dict[str, Any]] = {}
        self._current_stage: dict[str, Any] | None = None

    def stage(
        self,
        name: str,
        *,
        is_start: bool = False,
        is_end: bool = False,
        prompt: str = "",
        response_template: str | None = None,
        confirmation_template: str | None = None,
        mode: str | None = None,
        extraction_scope: str | None = None,
        auto_advance: bool | None = None,
        skip_extraction: bool | None = None,
        derivation_enabled: bool | None = None,
        recovery_enabled: bool | None = None,
        confirm_first_render: bool | None = None,
        confirm_on_new_data: bool | None = None,
        can_skip: bool | None = None,
        skip_default: bool | None = None,
        can_go_back: bool | None = None,
        reasoning: str | None = None,
        max_iterations: int | None = None,
        capture_mode: str | None = None,
        routing_transforms: list[str] | None = None,
        tool_result_mapping: list[dict[str, Any]] | None = None,
        **extra_fields: Any,
    ) -> WizardConfigBuilder:
        """Add a stage to the wizard config.

        After calling ``stage()``, subsequent ``field()`` and
        ``transition()`` calls apply to this stage.

        Args:
            name: Stage name (unique identifier).
            is_start: Whether this is the start stage.
            is_end: Whether this is an end stage.
            prompt: Stage prompt text.
            response_template: Jinja2 template rendered after extraction
                to confirm captured data.
            confirmation_template: Optional Jinja2 template rendered
                during confirmation instead of the auto-generated
                summary.  Requires ``response_template`` to also be
                set (the confirmation flow only triggers on stages
                that have a ``response_template``).  When omitted,
                confirmation auto-generates a summary from the stage
                schema and extracted data.
            mode: Stage mode (e.g. ``"conversation"``).
            extraction_scope: Per-stage extraction scope override.
            auto_advance: Per-stage auto-advance override.
            skip_extraction: Whether to skip extraction on this stage.
            derivation_enabled: Per-stage field derivation override.
                Set to ``False`` to suppress derivation on this stage.
            recovery_enabled: Per-stage recovery pipeline override.
                Set to ``False`` to suppress all recovery on this stage.
            confirm_first_render: Whether to pause for confirmation on
                first render when new data is extracted. Default ``True``.
                Set to ``False`` to skip confirmation and evaluate
                transitions immediately.
            confirm_on_new_data: Whether to re-confirm when schema
                property values change on subsequent renders.
            can_skip: Whether the user can skip this stage.
            skip_default: Default value to use when the stage is skipped.
            can_go_back: Whether the user can navigate back from this
                stage.
            reasoning: Reasoning strategy for this stage
                (e.g. ``"react"``).
            max_iterations: Maximum ReAct iterations for this stage.
            capture_mode: Extraction capture mode — ``"auto"``
                (default), ``"verbatim"`` (raw input), or ``"extract"``
                (force LLM extraction).
            routing_transforms: List of transform function names to
                execute before transition condition evaluation.
            tool_result_mapping: Post-extraction tool calls with
                result-to-state mapping.  Each entry is a dict with
                ``tool``, ``params``, ``mapping``, and optional
                ``on_error`` keys.
            **extra_fields: Additional stage config fields passed through
                to the stage dict verbatim. Use for less common fields
                (e.g. ``llm_assist=True``, ``navigation={...}``) without
                needing explicit builder parameters.

        Returns:
            Self for method chaining.
        """
        # Finalize previous stage
        self._finalize_current_stage()

        stage: dict[str, Any] = {"name": name, "prompt": prompt}
        if is_start:
            stage["is_start"] = True
        if is_end:
            stage["is_end"] = True
        if response_template is not None:
            stage["response_template"] = response_template
        if confirmation_template is not None:
            stage["confirmation_template"] = confirmation_template
        if mode is not None:
            stage["mode"] = mode
        if extraction_scope is not None:
            stage["extraction_scope"] = extraction_scope
        if auto_advance is not None:
            stage["auto_advance"] = auto_advance
        if skip_extraction is not None:
            stage["skip_extraction"] = skip_extraction
        if derivation_enabled is not None:
            stage["derivation_enabled"] = derivation_enabled
        if recovery_enabled is not None:
            stage["recovery_enabled"] = recovery_enabled
        if confirm_first_render is not None:
            stage["confirm_first_render"] = confirm_first_render
        if confirm_on_new_data is not None:
            stage["confirm_on_new_data"] = confirm_on_new_data
        if can_skip is not None:
            stage["can_skip"] = can_skip
        if skip_default is not None:
            stage["skip_default"] = skip_default
        if can_go_back is not None:
            stage["can_go_back"] = can_go_back
        if reasoning is not None:
            stage["reasoning"] = reasoning
        if max_iterations is not None:
            stage["max_iterations"] = max_iterations
        if capture_mode is not None:
            stage["capture_mode"] = capture_mode
        if routing_transforms is not None:
            stage["routing_transforms"] = routing_transforms
        if tool_result_mapping is not None:
            stage["tool_result_mapping"] = tool_result_mapping
        if extra_fields:
            # Prevent accidental override of structural keys set by
            # positional/explicit parameters above.
            reserved = {"name", "prompt", "is_start", "is_end"}
            safe_fields = {
                k: v for k, v in extra_fields.items()
                if k not in reserved
            }
            stage.update(safe_fields)

        self._current_stage = stage
        return self

    def field(
        self,
        name: str,
        *,
        field_type: str = "string",
        required: bool = False,
        description: str | None = None,
        enum: list[str] | None = None,
        default: Any = None,
        x_extraction: dict[str, Any] | None = None,
    ) -> WizardConfigBuilder:
        """Add a field to the current stage's schema.

        Must be called after ``stage()``.

        Args:
            name: Field name.
            field_type: JSON Schema type (``"string"``, ``"integer"``, etc.).
            required: Whether this field is required.
            description: Field description.
            enum: Allowed values.
            default: Default value.
            x_extraction: Extraction hints (``x-extraction`` schema extension).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If no current stage is set.
        """
        if self._current_stage is None:
            raise ValueError("field() must be called after stage()")

        schema = self._current_stage.setdefault("schema", {
            "type": "object",
            "properties": {},
            "required": [],
        })
        props = schema.setdefault("properties", {})

        field_def: dict[str, Any] = {"type": field_type}
        if description is not None:
            field_def["description"] = description
        if enum is not None:
            field_def["enum"] = enum
        if default is not None:
            field_def["default"] = default
        if x_extraction is not None:
            field_def["x-extraction"] = x_extraction

        props[name] = field_def

        if required:
            req_list = schema.setdefault("required", [])
            if name not in req_list:
                req_list.append(name)

        return self

    def transition(
        self,
        target: str,
        condition: str | None = None,
        priority: int | None = None,
        *,
        subflow_network: str | None = None,
        return_stage: str | None = None,
        data_mapping: dict[str, str] | None = None,
        result_mapping: dict[str, str] | None = None,
    ) -> WizardConfigBuilder:
        """Add a transition from the current stage.

        Must be called after ``stage()``.

        For subflow transitions, pass ``subflow_network`` to push a
        subflow instead of transitioning directly.  The ``target`` is
        used as the ``return_stage`` default (where to go after the
        subflow completes).

        Args:
            target: Target stage name (or return stage for subflows).
            condition: Python expression evaluated against wizard state.
            priority: Transition evaluation priority (lower = first).
            subflow_network: Name of the subflow network to push.
                When set, this becomes a subflow transition with
                ``target="_subflow"``.
            return_stage: Stage to return to after subflow completes.
                Defaults to ``target`` if not specified.
            data_mapping: Parent → subflow data key mapping.
            result_mapping: Subflow → parent data key mapping.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If no current stage is set.
        """
        if self._current_stage is None:
            raise ValueError("transition() must be called after stage()")

        transitions = self._current_stage.setdefault("transitions", [])

        if subflow_network is not None:
            t: dict[str, Any] = {"target": "_subflow"}
            subflow_config: dict[str, Any] = {
                "network": subflow_network,
                "return_stage": return_stage or target,
            }
            if data_mapping:
                subflow_config["data_mapping"] = data_mapping
            if result_mapping:
                subflow_config["result_mapping"] = result_mapping
            t["subflow"] = subflow_config
        else:
            t = {"target": target}

        if condition is not None:
            t["condition"] = condition
        if priority is not None:
            t["priority"] = priority
        transitions.append(t)
        return self

    def settings(self, **kwargs: Any) -> WizardConfigBuilder:
        """Set wizard-level settings.

        Args:
            **kwargs: Settings key-value pairs (e.g.
                ``extraction_scope="current_message"``,
                ``scope_escalation={"enabled": True}``).

        Returns:
            Self for method chaining.
        """
        self._settings.update(kwargs)
        return self

    def subflow(
        self, name: str, config: dict[str, Any]
    ) -> WizardConfigBuilder:
        """Register an inline subflow network.

        The config should be a wizard config dict (with ``stages``,
        ``name``, etc.) — typically built with another
        ``WizardConfigBuilder``.

        Args:
            name: Subflow network name (referenced by
                ``subflow_network`` in transitions).
            config: Complete wizard config dict for the subflow.

        Returns:
            Self for method chaining.
        """
        self._subflows[name] = config
        return self

    def build(self) -> dict[str, Any]:
        """Build and validate the wizard config dict.

        Returns:
            Complete wizard configuration dict compatible with
            ``WizardConfigLoader.load_from_dict()``.

        Raises:
            ValueError: If validation fails (no start stage, no end stage,
                transition to nonexistent stage).
        """
        self._finalize_current_stage()

        config: dict[str, Any] = {
            "name": self._name,
            "version": self._version,
            "stages": list(self._stages),
        }
        if self._settings:
            config["settings"] = dict(self._settings)
        if self._subflows:
            config["subflows"] = dict(self._subflows)

        self._validate(config)
        return config

    def _finalize_current_stage(self) -> None:
        """Add the current stage to the stages list."""
        if self._current_stage is not None:
            self._stages.append(self._current_stage)
            self._current_stage = None

    def _validate(self, config: dict[str, Any]) -> None:
        """Validate the built config."""
        stages = config.get("stages", [])
        stage_names = {s["name"] for s in stages}

        has_start = any(s.get("is_start") for s in stages)
        has_end = any(s.get("is_end") for s in stages)

        if not has_start:
            raise ValueError("Wizard config must have at least one start stage")
        if not has_end:
            raise ValueError("Wizard config must have at least one end stage")

        for stage in stages:
            for t in stage.get("transitions", []):
                target = t.get("target")
                # "_subflow" is a sentinel for subflow transitions —
                # not a real stage name.
                if target and target != "_subflow" and target not in stage_names:
                    raise ValueError(
                        f"Stage {stage['name']!r} has transition to "
                        f"nonexistent stage {target!r}"
                    )


# =============================================================================
# GroundedConfigBuilder — fluent grounded-strategy bot config construction
# =============================================================================


class GroundedConfigBuilder:
    """Fluent builder for grounded-strategy bot configuration dicts.

    Produces a complete ``bot_config`` dict suitable for
    :meth:`BotTestHarness.create(bot_config=...)`.  Eliminates verbose
    inline dict construction in grounded reasoning tests.

    Example:
        ```python
        config = (GroundedConfigBuilder()
            .intent(mode="extract", num_queries=3, domain_context="OAuth 2.0")
            .retrieval(top_k=5)
            .synthesis(style="hybrid", require_citations=True)
            .build())

        async with await BotTestHarness.create(
            bot_config=config,
            main_responses=[text_response("LLM synthesis")],
        ) as harness:
            harness.bot.reasoning_strategy.set_knowledge_base(kb)
            result = await harness.chat("How does OAuth work?")
        ```
    """

    def __init__(self) -> None:
        self._intent: dict[str, Any] = {
            "mode": "extract",
            "num_queries": 2,
        }
        self._retrieval: dict[str, Any] = {
            "top_k": 5,
            "score_threshold": 0.0,
        }
        self._synthesis: dict[str, Any] = {
            "mode": "llm",
            "require_citations": True,
        }
        self._sources: list[dict[str, Any]] = []
        self._store_provenance: bool = True
        self._greeting_template: str | None = None
        self._result_processing: dict[str, Any] = {}
        self._extra_reasoning: dict[str, Any] = {}
        self._llm: dict[str, Any] = {
            "provider": "echo",
            "model": "echo-test",
        }
        self._conversation_storage: dict[str, Any] = {"backend": "memory"}

    def intent(self, **kwargs: Any) -> GroundedConfigBuilder:
        """Configure intent resolution.

        Args:
            **kwargs: Keys merged into the intent config dict. Common:
                ``mode`` (``"extract"``/``"static"``/``"template"``),
                ``num_queries``, ``domain_context``,
                ``use_conversation_context``, ``extraction_config``,
                ``text_queries``, ``template``,
                ``expand_ambiguous_queries``.

        Returns:
            Self for method chaining.
        """
        self._intent.update(kwargs)
        return self

    def retrieval(self, **kwargs: Any) -> GroundedConfigBuilder:
        """Configure the retrieval phase.

        Args:
            **kwargs: Keys merged into the retrieval config dict.
                Common: ``top_k``, ``score_threshold``,
                ``merge_adjacent``, ``deduplicate``.

        Returns:
            Self for method chaining.
        """
        self._retrieval.update(kwargs)
        return self

    def synthesis(self, **kwargs: Any) -> GroundedConfigBuilder:
        """Configure the synthesis phase.

        Args:
            **kwargs: Keys merged into the synthesis config dict.
                Common: ``mode``, ``style``, ``require_citations``,
                ``allow_parametric``, ``citation_format``, ``template``,
                ``provenance_template``, ``instruction``.

        Returns:
            Self for method chaining.
        """
        self._synthesis.update(kwargs)
        return self

    def result_processing(self, **kwargs: Any) -> GroundedConfigBuilder:
        """Configure the result processing pipeline.

        Args:
            **kwargs: Keys set on the result_processing config dict.
                Common: ``normalize_strategy``, ``relative_threshold``,
                ``min_results``, ``query_rerank_weight``,
                ``cluster_strategy``.

        Returns:
            Self for method chaining.
        """
        self._result_processing.update(kwargs)
        return self

    def source(self, **kwargs: Any) -> GroundedConfigBuilder:
        """Add a source configuration.

        Args:
            **kwargs: Source config dict. Keys: ``type``, ``name``,
                ``weight``, plus source-specific options.

        Returns:
            Self for method chaining.
        """
        self._sources.append(kwargs)
        return self

    def provenance(self, enabled: bool = True) -> GroundedConfigBuilder:
        """Enable or disable provenance recording.

        Returns:
            Self for method chaining.
        """
        self._store_provenance = enabled
        return self

    def llm(self, **kwargs: Any) -> GroundedConfigBuilder:
        """Override default LLM config (default: echo provider).

        Returns:
            Self for method chaining.
        """
        self._llm.update(kwargs)
        return self

    def build(self) -> dict[str, Any]:
        """Build the complete bot_config dict.

        Returns:
            A dict suitable for ``BotTestHarness.create(bot_config=...)``.
        """
        reasoning: dict[str, Any] = {
            "strategy": "grounded",
            "intent": dict(self._intent),
            "retrieval": dict(self._retrieval),
            "synthesis": dict(self._synthesis),
            "store_provenance": self._store_provenance,
        }
        if self._result_processing:
            reasoning["result_processing"] = dict(self._result_processing)
        if self._sources:
            reasoning["sources"] = list(self._sources)
        if self._greeting_template is not None:
            reasoning["greeting_template"] = self._greeting_template
        reasoning.update(self._extra_reasoning)

        return {
            "llm": dict(self._llm),
            "conversation_storage": dict(self._conversation_storage),
            "reasoning": reasoning,
        }


# =============================================================================
# TurnResult — per-turn result with wizard state snapshot
# =============================================================================


@dataclass
class TurnResult:
    """Result of a single ``bot.chat()``, ``bot.greet()``, or
    ``bot.stream_chat()`` turn.

    Captures the bot response along with a snapshot of wizard state
    at the end of the turn.
    """

    response: str
    """Bot response text."""

    wizard_stage: str | None = None
    """Current wizard stage after this turn, or None if no wizard."""

    wizard_data: dict[str, Any] = field(default_factory=dict)
    """Wizard state data dict after this turn."""

    wizard_state: dict[str, Any] | None = None
    """Full normalized wizard state after this turn, or None."""

    turn_index: int = 0
    """One-based turn index (1 = first turn)."""

    chunks: list[str] = field(default_factory=list)
    """Stream chunks from ``stream_chat()``.  Empty for non-streaming turns."""


# =============================================================================
# BotTestHarness — single-object test setup for wizard bots
# =============================================================================


class BotTestHarness:
    """High-level test helper for ALL DynaBot behavioral tests.

    Wraps the full setup ceremony (bot creation, provider injection, tool
    registration, middleware wiring, context management) into one object.
    Use ``create()`` to build, ``chat()``/``greet()`` to run turns.

    For **non-wizard tests**, use ``bot_config=`` with any DynaBot config:

    Example:
        ```python
        async with await BotTestHarness.create(
            bot_config={
                "llm": {"provider": "echo", "model": "test"},
                "conversation_storage": {"backend": "memory"},
                "reasoning": {"strategy": "simple"},
            },
            main_responses=[
                tool_call_response("my_tool", {"q": "test"}),
                text_response("Here are the results"),
            ],
            tools=[my_tool],
            middleware=[my_middleware],
        ) as harness:
            result = await harness.chat("search")
            assert result.response == "Here are the results"
            # Streaming: harness.bot.stream_chat("msg", harness.context)
        ```

    For **wizard tests**, use ``wizard_config=`` with ``WizardConfigBuilder``:

    Example:
        ```python
        async with await BotTestHarness.create(
            wizard_config=config,
            main_responses=["Got it!", "All set!"],
            extraction_results=[
                [{"name": "Alice"}],
                [{"domain_id": "chess"}, {"name": "Alice", "domain_id": "chess"}],
            ],
        ) as harness:
            result = await harness.chat("My name is Alice")
            assert harness.wizard_data["name"] == "Alice"
            assert harness.wizard_stage == "gather"
        ```
    """

    def __init__(
        self,
        bot: Any,
        provider: EchoProvider,
        extractor: ConfigurableExtractor | None,
        context: Any,
    ) -> None:
        self._bot = bot
        self._provider = provider
        self._extractor = extractor
        self._context = context
        self._turn_count = 0
        self._last_result: TurnResult | None = None

    @classmethod
    async def create(
        cls,
        *,
        wizard_config: dict[str, Any] | None = None,
        bot_config: dict[str, Any] | None = None,
        main_responses: list[Any] | None = None,
        extraction_results: list[list[dict[str, Any]]] | None = None,
        system_prompt: str = "You are a helpful assistant.",
        conversation_id: str = "test-conv",
        client_id: str = "test",
        extraction_scope: str = "current_message",
        tools: list[Any] | None = None,
        middleware: list[Any] | None = None,
        strategy: ReasoningStrategy | None = None,
        strict_tools: bool = True,
        strict: bool = False,
    ) -> BotTestHarness:
        """Create a harness with a fully wired DynaBot.

        Provide either ``wizard_config`` (auto-wires bot config) or
        ``bot_config`` (full control).

        Args:
            wizard_config: Wizard config dict (e.g. from
                ``WizardConfigBuilder.build()``). Auto-wires EchoProvider,
                ConfigurableExtractor, and memory storage.
            bot_config: Complete bot config dict for ``DynaBot.from_config()``.
                When provided, ``wizard_config`` is ignored.
            main_responses: Responses to queue on the main EchoProvider.
                Accepts strings or ``LLMResponse`` objects (e.g. from
                ``text_response()`` / ``tool_call_response()``).
            extraction_results: Per-turn extraction results. Each inner list
                contains dicts for one turn's extraction calls. Flattened
                into a ``ConfigurableExtractor`` sequence internally.
            system_prompt: System prompt text.
            conversation_id: Conversation ID for the test context.
            client_id: Client ID for the test context.
            extraction_scope: Default extraction scope for the wizard.
                Only applies when ``wizard_config`` is used; ignored
                when ``bot_config`` is provided directly.
            tools: Optional list of ``Tool`` instances to register on the
                bot. Useful for ReAct strategy tests that need tool
                execution.
            middleware: Optional list of ``Middleware`` instances to append
                to the bot. Useful for testing middleware hooks like
                ``after_turn`` and ``on_tool_executed``.
            strategy: Optional reasoning strategy instance to replace the
                one created by ``from_config()``.  Useful for testing
                custom strategy implementations (e.g. strategies that
                implement ``StreamingPhasedProtocol`` with ``iterate=True``)
                through the full DynaBot orchestration.
            strict_tools: If True (default), the EchoProvider raises
                ValueError when a scripted response contains tool_calls
                but no tools were provided to complete(). Set to False
                for tests that intentionally exercise unexpected
                tool_calls with no registered tools.
            strict: If True, the EchoProvider raises
                ``ResponseQueueExhaustedError`` when all scripted
                responses have been consumed, instead of falling back
                to echo behavior.  Catches under-scripted tests.

        Returns:
            Configured ``BotTestHarness`` instance.

        Raises:
            ValueError: If neither ``wizard_config`` nor ``bot_config``
                is provided.
        """
        from .bot.base import DynaBot
        from .bot.context import BotContext

        if bot_config is None and wizard_config is None:
            raise ValueError(
                "Either wizard_config or bot_config must be provided"
            )

        # Build extraction results
        extractor: ConfigurableExtractor | None = None
        if extraction_results is not None:
            flat_results = [
                SimpleExtractionResult(data=data, confidence=0.9)
                for turn_results in extraction_results
                for data in turn_results
            ]
            extractor = ConfigurableExtractor(results=flat_results)

        # Build bot config if not provided
        if bot_config is None:
            assert wizard_config is not None
            wizard_cfg = copy.deepcopy(wizard_config)

            existing_settings = wizard_cfg.get("settings", {})
            if "extraction_scope" not in existing_settings:
                wizard_cfg["settings"] = {
                    "extraction_scope": extraction_scope,
                    **existing_settings,
                }

            # When scripted extraction results are provided, force LLM
            # extraction on stages that would otherwise use verbatim
            # capture (single required string field).  Without this,
            # the ConfigurableExtractor is silently bypassed and tests
            # get the raw user message instead of scripted results.
            #
            # This applies to ALL schema stages uniformly.  In multi-stage
            # wizards where a specific stage should still use verbatim
            # capture, set ``capture_mode="verbatim"`` explicitly on that
            # stage — the guard below respects explicit overrides at both
            # the top-level and collection_config levels.
            if extraction_results is not None:
                for stage_def in wizard_cfg.get("stages", []):
                    if (
                        stage_def.get("schema")
                        and stage_def.get("capture_mode") in (None, "auto")
                    ):
                        col = stage_def.get("collection_config") or {}
                        if col.get("capture_mode") in (None, "auto"):
                            stage_def["capture_mode"] = "extract"

            bot_config = {
                "llm": {"provider": "echo", "model": "echo-test"},
                "conversation_storage": {"backend": "memory"},
                "prompts": {
                    "assistant": system_prompt,
                },
                "system_prompt": "assistant",
                "reasoning": {
                    "strategy": "wizard",
                    "wizard_config": wizard_cfg,
                    "extraction_config": {
                        "provider": "echo",
                        "model": "echo-extraction",
                    },
                },
            }

        # Create bot
        bot = await DynaBot.from_config(bot_config)

        # Close the original provider created by from_config() — we replace
        # it with a fresh EchoProvider that has a clean response queue.
        original_provider = bot.llm
        if hasattr(original_provider, "close"):
            await original_provider.close()

        # Create a fresh provider with known state
        provider = EchoProvider(
            {"provider": "echo", "model": "echo-test"},
            strict_tools=strict_tools,
            strict=strict,
        )
        if main_responses:
            provider.set_responses(main_responses)

        # Replace reasoning strategy before inject_providers so that
        # provider/extractor injection lands on the actual strategy.
        if strategy is not None:
            bot.reasoning_strategy = strategy

        # Inject fresh provider and extractor
        inject_providers(bot, main_provider=provider, extractor=extractor)

        # Register tools if provided
        if tools:
            for tool in tools:
                bot.tool_registry.register_tool(tool)

        # Append middleware if provided
        if middleware:
            for mw in middleware:
                bot.middleware.append(mw)

        context = BotContext(
            conversation_id=conversation_id,
            client_id=client_id,
        )

        return cls(
            bot=bot,
            provider=provider,
            extractor=extractor,
            context=context,
        )

    async def chat(self, message: str, **kwargs: Any) -> TurnResult:
        """Run a chat turn and capture wizard state.

        Args:
            message: User message.
            **kwargs: Additional kwargs passed to ``bot.chat()``.

        Returns:
            ``TurnResult`` with response and wizard state snapshot.
        """
        response = await self._bot.chat(message, self._context, **kwargs)
        self._turn_count += 1

        state = await self._bot.get_wizard_state(
            self._context.conversation_id,
        )
        result = TurnResult(
            response=response or "",
            wizard_stage=state["current_stage"] if state else None,
            wizard_data=state.get("data", {}) if state else {},
            wizard_state=state,
            turn_index=self._turn_count,
        )
        self._last_result = result
        return result

    async def greet(self, **kwargs: Any) -> TurnResult:
        """Run a greet turn and capture wizard state.

        Args:
            **kwargs: Additional kwargs passed to ``bot.greet()``.

        Returns:
            ``TurnResult`` with response and wizard state snapshot.
        """
        response = await self._bot.greet(self._context, **kwargs)
        self._turn_count += 1

        state = await self._bot.get_wizard_state(
            self._context.conversation_id,
        )
        result = TurnResult(
            response=response or "",
            wizard_stage=state["current_stage"] if state else None,
            wizard_data=state.get("data", {}) if state else {},
            wizard_state=state,
            turn_index=self._turn_count,
        )
        self._last_result = result
        return result

    async def stream_chat(self, message: str, **kwargs: Any) -> TurnResult:
        """Run a streaming chat turn and capture wizard state.

        Consumes the full stream, joins chunks into a response string,
        and snapshots wizard state — same contract as :meth:`chat` but
        exercises the ``stream_chat()`` code path.

        Args:
            message: User message.
            **kwargs: Additional kwargs passed to ``bot.stream_chat()``.

        Returns:
            ``TurnResult`` with response, chunks, and wizard state snapshot.
        """
        chunks: list[str] = []
        async for chunk in self._bot.stream_chat(
            message, self._context, **kwargs
        ):
            chunks.append(chunk.delta)
        self._turn_count += 1

        state = await self._bot.get_wizard_state(
            self._context.conversation_id,
        )
        result = TurnResult(
            response="".join(chunks),
            wizard_stage=state["current_stage"] if state else None,
            wizard_data=state.get("data", {}) if state else {},
            wizard_state=state,
            turn_index=self._turn_count,
            chunks=chunks,
        )
        self._last_result = result
        return result

    @property
    def wizard_stage(self) -> str | None:
        """Current wizard stage from the last turn."""
        return self._last_result.wizard_stage if self._last_result else None

    @property
    def wizard_data(self) -> dict[str, Any]:
        """Wizard state data dict from the last turn."""
        return self._last_result.wizard_data if self._last_result else {}

    @property
    def wizard_state(self) -> dict[str, Any] | None:
        """Full wizard state from the last turn."""
        return self._last_result.wizard_state if self._last_result else None

    @property
    def last_response(self) -> str:
        """Response text from the last turn."""
        return self._last_result.response if self._last_result else ""

    @property
    def turn_count(self) -> int:
        """Number of turns executed."""
        return self._turn_count

    @property
    def bot(self) -> Any:
        """The underlying DynaBot instance."""
        return self._bot

    @property
    def context(self) -> Any:
        """The BotContext used for this harness's turns."""
        return self._context

    @property
    def provider(self) -> EchoProvider:
        """The main EchoProvider (for call history assertions)."""
        return self._provider

    @property
    def extractor(self) -> ConfigurableExtractor | None:
        """The ConfigurableExtractor (for call verification)."""
        return self._extractor

    async def close(self) -> None:
        """Close the bot and release resources."""
        await self._bot.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


def inject_providers(
    bot: Any,
    main_provider: AsyncLLMProvider | None = None,
    extraction_provider: AsyncLLMProvider | None = None,
    *,
    extractor: Any | None = None,
    **role_providers: AsyncLLMProvider,
) -> None:
    """Inject LLM providers into a DynaBot instance for testing.

    For ``main_provider``, directly replaces ``bot.llm`` (the ``"main"``
    role is always served from this attribute, not the registry catalog).

    For ``extraction_provider`` and ``**role_providers``, updates both the
    registry catalog and the actual subsystem wiring via ``set_provider()``.

    For ``extractor``, calls ``strategy.set_extractor()`` to replace
    the reasoning strategy's extractor entirely.  Use this to inject a
    ``ConfigurableExtractor`` (which is not an ``AsyncLLMProvider`` and
    cannot be wired through ``set_provider()``).

    **Lifecycle note:** ``bot.close()`` will close ``self.llm`` (the main
    provider) unconditionally — the caller should be aware that an
    injected ``main_provider`` will be closed when the bot is closed.
    For subsystem providers (memory embedding, extraction), ownership
    flags control whether ``close()`` acts on them.

    If ``bot`` does not implement ``register_provider``, catalog
    registration is skipped; only subsystem wiring via ``set_provider()``
    is performed.

    Args:
        bot: A DynaBot instance (or any object with ``llm`` and
            ``reasoning_strategy`` attributes).
        main_provider: Provider to use for main LLM calls. If None,
            the existing provider is kept.
        extraction_provider: Provider to use for schema extraction.
            If None, the existing provider is kept.
        extractor: A ``ConfigurableExtractor`` (or compatible object)
            to replace the wizard's ``SchemaExtractor`` directly.
            Mutually exclusive with ``extraction_provider``.
        **role_providers: Additional providers keyed by role name
            (e.g. ``memory_embedding=echo_provider``).  Each provider
            is registered in the catalog AND wired into the owning
            subsystem via ``set_provider()``.

    Example:
        ```python
        from dataknobs_llm import EchoProvider
        from dataknobs_bots.testing import inject_providers

        main = EchoProvider()
        extraction = EchoProvider()
        inject_providers(bot, main, extraction)
        ```
    """
    if extractor is not None and extraction_provider is not None:
        raise ValueError(
            "extractor and extraction_provider are mutually exclusive"
        )

    if main_provider is not None:
        bot.llm = main_provider

    if extractor is not None:
        strategy = getattr(bot, "reasoning_strategy", None)
        if strategy is not None and hasattr(strategy, "set_extractor"):
            strategy.set_extractor(extractor)
        else:
            logger.warning(
                "Bot has no reasoning_strategy.set_extractor — "
                "skipping extractor injection"
            )

    if extraction_provider is not None:
        from dataknobs_bots.bot.base import PROVIDER_ROLE_EXTRACTION

        # Update the registry entry
        if hasattr(bot, "register_provider"):
            bot.register_provider(PROVIDER_ROLE_EXTRACTION, extraction_provider)

        # Also update the actual extractor so subsystem calls use it
        strategy = getattr(bot, "reasoning_strategy", None)
        if strategy is None:
            logger.warning(
                "Bot has no reasoning_strategy — skipping extraction provider injection"
            )
        elif hasattr(strategy, "set_provider"):
            strategy.set_provider(PROVIDER_ROLE_EXTRACTION, extraction_provider)
        else:
            # Fallback for strategies without set_provider (e.g. test stubs)
            extractor = getattr(strategy, "_extractor", None)
            if extractor is None:
                logger.warning(
                    "Reasoning strategy has no _extractor — "
                    "skipping extraction provider injection"
                )
            else:
                extractor.provider = extraction_provider
                if hasattr(extractor, "_owns_provider"):
                    extractor._owns_provider = False

    # Wire role-based providers into catalog AND subsystems
    for role, provider in role_providers.items():
        if hasattr(bot, "register_provider"):
            bot.register_provider(role, provider)

        # Wire into the actual subsystem that owns this role
        _wire_role_provider(bot, role, provider)


def _wire_role_provider(bot: Any, role: str, provider: AsyncLLMProvider) -> None:
    """Wire a role provider into the subsystem that owns it.

    Iterates over the bot's subsystems (memory, knowledge_base,
    reasoning_strategy) and calls ``set_provider(role, provider)``
    on the first one that claims the role (first-wins).

    This is safe for ``CompositeMemory`` because its ``set_provider()``
    delegates to all sub-strategies internally and returns ``True``,
    so the early return after ``CompositeMemory`` accepts is correct.

    Args:
        bot: DynaBot instance (or compatible stub).
        role: Provider role name.
        provider: Replacement provider instance.
    """
    subsystems = [
        getattr(bot, "memory", None),
        getattr(bot, "knowledge_base", None),
        getattr(bot, "reasoning_strategy", None),
    ]
    for subsystem in subsystems:
        if (
            subsystem is not None
            and hasattr(subsystem, "set_provider")
            and subsystem.set_provider(role, provider)
        ):
            return
    logger.debug(
        "Role %r registered in catalog but no subsystem claimed it", role
    )


class CaptureReplay:
    """Loads a capture JSON file and creates pre-loaded EchoProviders.

    Capture files contain serialized LLM request/response pairs from real
    provider runs, organized by turn. CaptureReplay deserializes these and
    creates EchoProviders queued with the correct responses, enabling
    deterministic replay of captured conversations.

    Attributes:
        metadata: Capture session metadata (description, model info, timestamps)
        turns: List of turn dicts with wizard state, user messages, bot responses
        format_version: Capture file format version

    Example:
        ```python
        replay = CaptureReplay.from_file("captures/quiz_basic.json")

        # Get providers for replay
        main = replay.main_provider()
        extraction = replay.extraction_provider()

        # Or inject directly into a bot
        replay.inject_into_bot(bot)
        ```
    """

    def __init__(
        self,
        data: dict[str, Any],
    ) -> None:
        self.format_version: str = data.get("format_version", "1.0")
        self.metadata: dict[str, Any] = data.get("metadata", {})
        self.turns: list[dict[str, Any]] = data.get("turns", [])
        self._data = data

        # Pre-separate LLM calls by role for provider creation
        self._main_responses: list[LLMResponse] = []
        self._extraction_responses: list[LLMResponse] = []
        self._parse_calls()

    def _parse_calls(self) -> None:
        """Parse all LLM calls from turns and separate by role."""
        for turn in self.turns:
            for call in turn.get("llm_calls", []):
                response = llm_response_from_dict(call["response"])
                role = call.get("role", "main")
                if role == "extraction":
                    self._extraction_responses.append(response)
                else:
                    self._main_responses.append(response)

    @classmethod
    def from_file(cls, path: str | Path) -> CaptureReplay:
        """Load a capture replay from a JSON file.

        Args:
            path: Path to the capture JSON file

        Returns:
            CaptureReplay instance

        Raises:
            FileNotFoundError: If the file does not exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        with open(path) as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptureReplay:
        """Create a CaptureReplay from a dict (e.g., already-parsed JSON).

        Args:
            data: Capture data dict

        Returns:
            CaptureReplay instance
        """
        return cls(data)

    def main_provider(self) -> EchoProvider:
        """Create an EchoProvider queued with main-role responses.

        Returns:
            EchoProvider with responses in capture order
        """
        provider = EchoProvider({"provider": "echo", "model": "capture-replay"})
        if self._main_responses:
            provider.set_responses(self._main_responses)
        return provider

    def extraction_provider(self) -> EchoProvider:
        """Create an EchoProvider queued with extraction-role responses.

        Returns:
            EchoProvider with responses in capture order
        """
        provider = EchoProvider({"provider": "echo", "model": "capture-replay"})
        if self._extraction_responses:
            provider.set_responses(self._extraction_responses)
        return provider

    def inject_into_bot(self, bot: Any) -> None:
        """Replace providers on a DynaBot with capture-replay EchoProviders.

        Creates main and extraction EchoProviders from the captured data
        and injects them into the bot using ``inject_providers``.

        Args:
            bot: A DynaBot instance
        """
        inject_providers(
            bot,
            main_provider=self.main_provider(),
            extraction_provider=self.extraction_provider() if self._extraction_responses else None,
        )


class ErrorRaisingStrategy(ReasoningStrategy):
    """Test strategy that raises a configurable exception on every hook.

    Useful for testing error handling paths in DynaBot without needing
    a real reasoning strategy or LLM provider.

    Args:
        error: The exception to raise. Defaults to ``ValueError("test error")``.

    Example:
        ```python
        from dataknobs_bots.testing import ErrorRaisingStrategy

        strategy = ErrorRaisingStrategy(RuntimeError("boom"))
        bot = DynaBot(llm=provider, ..., reasoning_strategy=strategy)
        # bot.chat() / bot.greet() will raise RuntimeError("boom")
        ```
    """

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self._error = error or ValueError("test error")

    async def generate(
        self,
        manager: ReasoningManagerProtocol,
        llm: Any,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        raise self._error

    async def greet(
        self,
        manager: ReasoningManagerProtocol,
        llm: Any,
        *,
        initial_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any | None:
        raise self._error


# =============================================================================
# StubManager — lightweight ReasoningManagerProtocol for unit tests
# =============================================================================


class StubManager:
    """Minimal implementation of :class:`ReasoningManagerProtocol` for tests.

    Use this when testing strategy methods (``greet()``, ``generate()``)
    directly — i.e., without the full ``DynaBot`` / ``BotTestHarness``
    lifecycle.  Tracks messages and metadata in memory.

    For integration tests that exercise the DynaBot pipeline, use
    :class:`BotTestHarness` instead.

    Example:
        .. code-block:: python

            from dataknobs_bots.testing import StubManager

            manager = StubManager(
                system_prompt="You are helpful.",
                messages=[{"role": "user", "content": "Hello"}],
            )
            result = await strategy.greet(
                manager, None, initial_context={"user": "Alice"},
            )
            assert result is not None
    """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        metadata: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        conversation_id: str = "stub-conversation",
    ) -> None:
        self._system_prompt = system_prompt
        self._metadata: dict[str, Any] = metadata if metadata is not None else {}
        self._messages: list[dict[str, Any]] = list(messages) if messages else []
        self.conversation_id = conversation_id

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def get_messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    async def add_message(
        self,
        role: str = "user",
        content: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        msg: dict[str, Any] = {"role": role, "content": content}
        if metadata:
            msg["metadata"] = metadata
        self._messages.append(msg)

    async def complete(
        self,
        *,
        system_prompt_override: str | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="", model="stub", finish_reason="stop")

    def stream_complete(
        self,
        *,
        system_prompt_override: str | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async def _empty() -> AsyncIterator[Any]:
            return
            yield  # pragma: no cover

        return _empty()
