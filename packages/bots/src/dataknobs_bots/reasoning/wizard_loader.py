"""WizardConfigLoader for translating wizard YAML to FSM configuration.

This module provides the translation layer between user-friendly wizard
YAML configuration and the underlying FSM configuration format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import yaml

from dataknobs_fsm.api.advanced import AdvancedFSM
from dataknobs_fsm.config.builder import FSMBuilder

from .function_resolver import resolve_functions
from .wizard_fsm import WizardFSM

logger = logging.getLogger(__name__)

# Sentinel target for subflow transitions
SUBFLOW_TARGET = "_subflow"


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

        return self.load_from_dict(
            wizard_config, custom_functions, config_base_path=config_path.parent
        )

    def load_from_dict(
        self,
        wizard_config: dict[str, Any],
        custom_functions: dict[str, Callable[..., Any] | str] | None = None,
        config_base_path: Path | None = None,
    ) -> WizardFSM:
        """Load wizard config from dict and create WizardFSM.

        Args:
            wizard_config: Wizard configuration dict
            custom_functions: Optional custom functions for transitions.
                Values can be either:
                - Callable objects (used directly)
                - String references in "module.path:function_name" format
            config_base_path: Base path for resolving relative subflow paths

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

        # Load subflow networks
        subflow_registry = self._load_subflow_networks(
            wizard_config, custom_functions, config_base_path
        )

        return WizardFSM(
            advanced_fsm, stage_metadata, settings=settings, subflow_registry=subflow_registry
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
            raise ValueError(
                f"Transition in stage '{source_stage}' missing 'target'"
            )

        # Handle subflow transitions specially
        # For subflow transitions, the FSM stays at the current stage
        # The actual subflow handling happens in WizardReasoning
        is_subflow_transition = target == SUBFLOW_TARGET
        actual_target = source_stage if is_subflow_transition else target

        # Build condition function reference if specified
        condition = None
        if "condition" in transition:
            condition_code = transition["condition"]
            # Wrap in return statement if not already
            if not condition_code.strip().startswith("return"):
                condition_code = f"return {condition_code}"

            condition = FunctionReference(
                type="inline",
                name=f"condition_{source_stage}_{actual_target}_{idx}",
                code=condition_code,
            )

        # Build transform function reference(s) if specified
        # Supports both single string and list of strings:
        #   transform: apply_template          # single
        #   transform: [apply_template, save]  # list
        transform: FunctionReference | list[FunctionReference] | None = None
        if "transform" in transition:
            raw_transform = transition["transform"]
            if isinstance(raw_transform, list):
                transform = [
                    FunctionReference(type="registered", name=name)
                    for name in raw_transform
                ]
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
                trans_meta: dict[str, Any] = {
                    "target": transition.get("target"),
                    "condition": transition.get("condition"),
                    "priority": transition.get("priority"),
                }
                # Include subflow config if this is a subflow transition
                if transition.get("target") == SUBFLOW_TARGET:
                    trans_meta["is_subflow_transition"] = True
                    trans_meta["subflow_config"] = transition.get("subflow", {})
                transitions.append(trans_meta)

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
                                # Pass data in globals so _test() can access it
                                exec_globals: dict[str, Any] = {"data": data}
                                exec_code = f"def _test():\n    {code}\n_result = _test()"
                                exec(exec_code, exec_globals)  # nosec B102
                                return bool(exec_globals.get("_result", False))
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

    def _load_subflow_networks(
        self,
        wizard_config: dict[str, Any],
        custom_functions: dict[str, Callable[..., Any] | str] | None,
        config_base_path: Path | None,
    ) -> dict[str, WizardFSM]:
        """Load subflow networks referenced in transitions.

        Scans all transitions for subflow references and loads the
        corresponding wizard configurations.

        Args:
            wizard_config: Main wizard configuration dict
            custom_functions: Custom functions to pass to subflows
            config_base_path: Base path for resolving relative paths

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
                    config_base_path,
                )
                if subflow_fsm:
                    subflow_registry[subflow_name] = subflow_fsm
                    logger.debug("Loaded subflow: %s", subflow_name)
            except Exception as e:
                logger.error("Failed to load subflow '%s': %s", subflow_name, e)
                raise ValueError(f"Failed to load subflow '{subflow_name}': {e}") from e

        return subflow_registry

    def _load_single_subflow(
        self,
        subflow_name: str,
        wizard_config: dict[str, Any],
        custom_functions: dict[str, Callable[..., Any] | str] | None,
        config_base_path: Path | None,
    ) -> WizardFSM | None:
        """Load a single subflow network.

        Attempts to load the subflow from:
        1. Explicit subflow definition in wizard_config["subflows"]
        2. File path relative to config_base_path
        3. File path in subflows/ subdirectory

        Args:
            subflow_name: Name of the subflow to load
            wizard_config: Main wizard configuration dict
            custom_functions: Custom functions to pass to subflow
            config_base_path: Base path for resolving relative paths

        Returns:
            WizardFSM for the subflow, or None if not found
        """
        # Check for inline subflow definition
        explicit_subflows = wizard_config.get("subflows", {})
        if subflow_name in explicit_subflows:
            subflow_config = explicit_subflows[subflow_name]
            return self.load_from_dict(
                subflow_config, custom_functions, config_base_path
            )

        # Try to load from file
        if config_base_path is None:
            logger.warning(
                "Cannot load subflow '%s' from file: no config_base_path provided",
                subflow_name,
            )
            return None

        # Try direct path (subflow_name.yaml)
        subflow_path = config_base_path / f"{subflow_name}.yaml"
        if subflow_path.exists():
            return self.load(str(subflow_path), custom_functions)

        # Try subflows/ subdirectory
        subflow_path = config_base_path / "subflows" / f"{subflow_name}.yaml"
        if subflow_path.exists():
            return self.load(str(subflow_path), custom_functions)

        logger.warning(
            "Subflow '%s' not found in config or as file at %s",
            subflow_name,
            config_base_path,
        )
        return None


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
