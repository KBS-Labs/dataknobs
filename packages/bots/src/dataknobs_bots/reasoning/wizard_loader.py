"""WizardConfigLoader for translating wizard YAML to FSM configuration.

This module provides the translation layer between user-friendly wizard
YAML configuration and the underlying FSM configuration format.
"""

import logging
from pathlib import Path
from typing import Any, Callable

import yaml

from dataknobs_fsm.api.advanced import AdvancedFSM
from dataknobs_fsm.config.builder import FSMBuilder

from .function_resolver import resolve_functions
from .wizard_fsm import WizardFSM

logger = logging.getLogger(__name__)


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
    ) -> WizardFSM:
        """Load wizard config and create WizardFSM.

        Args:
            config_path: Path to wizard YAML config file
            custom_functions: Optional custom functions for transitions.
                Values can be either callables or "module:function" strings.

        Returns:
            Configured WizardFSM instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config is invalid YAML
            ValueError: If config structure is invalid
        """
        config_path = Path(config_path)

        with open(config_path) as f:
            wizard_config = yaml.safe_load(f)

        return self.load_from_dict(wizard_config, custom_functions)

    def load_from_dict(
        self,
        wizard_config: dict[str, Any],
        custom_functions: dict[str, Callable[..., Any] | str] | None = None,
    ) -> WizardFSM:
        """Load wizard config from dict and create WizardFSM.

        Args:
            wizard_config: Wizard configuration dict
            custom_functions: Optional custom functions for transitions.
                Values can be either:
                - Callable objects (used directly)
                - String references in "module.path:function_name" format

        Returns:
            Configured WizardFSM instance

        Raises:
            ValueError: If config structure is invalid

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

        # Translate wizard config to FSM config
        fsm_config = self._translate_to_fsm(wizard_config)

        # Extract stage metadata
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

        return WizardFSM(advanced_fsm, stage_metadata, settings=settings)

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
            TransactionConfig,
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
            transaction=TransactionConfig(),
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

        # If no transitions and not end state, add default arc
        if not arcs and not stage.get("is_end"):
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
                "suggestions": stage.get("suggestions", []),
                "help_text": stage.get("help_text"),
                "can_skip": stage.get("can_skip", False),
                "skip_default": stage.get("skip_default"),
                "can_go_back": stage.get("can_go_back", True),
                "tools": stage.get("tools", []),
            },
            data_schema=stage.get("schema"),
        )

        return state_config

    def _translate_transition(
        self, source_stage: str, transition: dict[str, Any], idx: int
    ) -> Any:
        """Translate wizard transition to FSM arc.

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
            raise ValueError(
                f"Transition in stage '{source_stage}' missing 'target'"
            )

        # Build condition function reference if specified
        condition = None
        if "condition" in transition:
            condition_code = transition["condition"]
            # Wrap in return statement if not already
            if not condition_code.strip().startswith("return"):
                condition_code = f"return {condition_code}"

            condition = FunctionReference(
                type="inline",
                name=f"condition_{source_stage}_{target}_{idx}",
                code=condition_code,
            )

        # Build transform function reference if specified
        transform = None
        if "transform" in transition:
            transform_name = transition["transform"]
            transform = FunctionReference(
                type="registered",
                name=transform_name,
            )

        arc = ArcConfig(
            target=target,
            condition=condition,
            transform=transform,
            priority=transition.get("priority", idx),
            metadata=transition.get("metadata", {}),
        )

        return arc

    def _extract_metadata(
        self, wizard_config: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
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
                transitions.append({
                    "target": transition.get("target"),
                    "condition": transition.get("condition"),
                    "priority": transition.get("priority"),
                })

            # Extract per-stage tasks
            stage_tasks = self._extract_stage_tasks(stage)

            metadata[stage["name"]] = {
                "name": stage["name"],  # Include name in metadata for template access
                "prompt": stage.get("prompt", ""),
                "schema": stage.get("schema"),
                "suggestions": stage.get("suggestions", []),
                "help_text": stage.get("help_text"),
                "can_skip": stage.get("can_skip", False),
                "skip_default": stage.get("skip_default"),
                "can_go_back": stage.get("can_go_back", True),
                "auto_advance": stage.get("auto_advance", False),
                "tools": stage.get("tools", []),
                "extraction_model": stage.get("extraction_model"),
                "is_start": stage.get("is_start", False),
                "is_end": stage.get("is_end", False),
                "transitions": transitions,  # Include transitions for observability
                "tasks": stage_tasks,  # Per-stage tasks
                # ReAct-style tool reasoning settings
                "reasoning": stage.get("reasoning"),  # "single" or "react"
                "max_iterations": stage.get("max_iterations"),
            }

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
            tasks.append({
                "id": task_def.get("id"),
                "description": task_def.get("description", task_def.get("id", "")),
                "required": task_def.get("required", True),
                "depends_on": task_def.get("depends_on", []),
                "completed_by": task_def.get("completed_by"),
                "field_name": task_def.get("field_name"),
                "tool_name": task_def.get("tool_name"),
            })
        return tasks

    def _extract_global_tasks(
        self, wizard_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
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
            tasks.append({
                "id": task_def.get("id"),
                "description": task_def.get("description", task_def.get("id", "")),
                "required": task_def.get("required", True),
                "depends_on": task_def.get("depends_on", []),
                "completed_by": task_def.get("completed_by"),
                "field_name": task_def.get("field_name"),
                "tool_name": task_def.get("tool_name"),
                "stage": None,  # Mark as global
            })
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
                func_name = f"condition_{stage['name']}_{target}_{idx}"

                # Wrap in return statement if not already
                if not condition_code.strip().startswith("return"):
                    condition_code = f"return {condition_code}"

                # Create the function
                try:
                    # Create a function that evaluates the condition
                    def make_condition(code: str) -> Callable[[Any, Any], bool]:
                        def condition_func(
                            data: dict[str, Any], context: Any = None
                        ) -> bool:
                            try:
                                # Simple evaluation - data is available directly
                                local_vars: dict[str, Any] = {"data": data}
                                exec_code = f"def _test():\n    {code}\n_result = _test()"
                                exec(exec_code, {}, local_vars)  # nosec B102
                                return bool(local_vars.get("_result", False))
                            except Exception as e:
                                logger.warning(
                                    "Condition evaluation failed: %s", e
                                )
                                return False

                        return condition_func

                    builder.register_function(func_name, make_condition(condition_code))
                except Exception as e:
                    logger.warning(
                        "Failed to register condition '%s': %s", func_name, e
                    )


def load_wizard_config(
    config_path: str | Path,
    custom_functions: dict[str, Callable[..., Any] | str] | None = None,
) -> WizardFSM:
    """Convenience function to load wizard config.

    Args:
        config_path: Path to wizard YAML config file
        custom_functions: Optional custom functions for transitions.
            Values can be either callables or "module:function" strings.

    Returns:
        Configured WizardFSM instance
    """
    loader = WizardConfigLoader()
    return loader.load(config_path, custom_functions)
