"""Configuration loader for FSM configurations.

This module provides functionality to load FSM configurations from various sources:
- Files (JSON, YAML)
- Dictionaries
- Templates
- Environment variables
"""

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, List, Set, Union

from dataknobs_common.config_loading import (
    ConfigLoadError,
    ConfigParseError,
    ConfigUnsupportedFormatError,
    ConfigYAMLNotInstalledError,
    load_yaml_or_json,
)
from dataknobs_config import Config as DataknobsConfig
from dataknobs_config import substitute_env_vars

from dataknobs_fsm.config.schema import (
    FSMConfig,
    TemplateConfig,
    UseCaseTemplate,
    apply_template,
    validate_config,
)


@contextlib.contextmanager
def _alias_prefixed_env_vars(prefix: str) -> Iterator[None]:
    """Temporarily expose prefixed env vars as their unprefixed names.

    For each ``os.environ[{prefix}{NAME}]`` where ``os.environ[NAME]`` is
    unset, set ``os.environ[NAME] = os.environ[{prefix}{NAME}]`` for the
    duration of the context, then restore. Preserves the documented FSM
    behavior where a YAML reference to ``${PORT}`` resolves to the value
    of ``FSM_PORT`` when ``PORT`` is not set in the environment.

    On exit, only aliases whose value is still what this call wrote are
    popped. If something else (a nested ``ConfigLoader`` invocation or
    user code) changed the bare-name value mid-context, that change is
    preserved rather than blindly clobbered.

    The empty-prefix branch is defensive: ``ConfigLoader._env_prefix``
    is hardcoded to ``"FSM_"`` today, but the guard avoids walking
    ``os.environ`` (and aliasing every var to itself) if a future
    consumer sets ``_env_prefix = ""``.

    Caveat: mutates ``os.environ`` for the duration of the context.
    Concurrent calls in different threads may observe each other's
    aliases. FSM config loading is a startup-time operation in every
    in-tree consumer, so this is acceptable. Out-of-tree multi-thread
    consumers should serialize loader calls.
    """
    aliased: list[tuple[str, str]] = []
    try:
        if prefix:
            for key, value in list(os.environ.items()):
                if key.startswith(prefix):
                    bare = key[len(prefix):]
                    if bare and bare not in os.environ:
                        os.environ[bare] = value
                        aliased.append((bare, value))
        yield
    finally:
        for name, injected_value in aliased:
            if os.environ.get(name) == injected_value:
                os.environ.pop(name, None)


class ConfigLoader:
    """Load and process FSM configurations from various sources."""

    def __init__(self, use_dataknobs_config: bool = False):
        """Initialize the ConfigLoader.
        
        Args:
            use_dataknobs_config: Whether to use dataknobs_config for advanced features.
        """
        self.use_dataknobs_config = use_dataknobs_config
        self._env_prefix = "FSM_"
        self._included_configs: Dict[str, Dict[str, Any]] = {}
        self._registered_functions: Set[str] = set()

    def add_registered_function(self, name: str) -> None:
        """Add a function name to the set of registered functions.

        Args:
            name: Function name that has been registered.
        """
        self._registered_functions.add(name)

    def _convert_to_function_reference(self, value: Any) -> Dict[str, Any]:
        """Convert a value to a function reference dictionary.

        Args:
            value: The value to convert (string, dict, etc.)

        Returns:
            Function reference dictionary with 'type' and appropriate fields.
        """
        if isinstance(value, dict):
            # Already a function reference
            return value
        elif isinstance(value, str):
            # Check if it's a registered function
            if value in self._registered_functions:
                return {
                    'type': 'registered',
                    'name': value
                }
            else:
                # Treat as inline code
                return {
                    'type': 'inline',
                    'code': value
                }
        else:
            # Convert to string and treat as inline code
            return {
                'type': 'inline',
                'code': str(value)
            }

    def load_from_file(
        self,
        file_path: Union[str, Path],
        resolve_env: bool = True,
        resolve_references: bool = True,
    ) -> FSMConfig:
        """Load configuration from a file.
        
        Args:
            file_path: Path to configuration file (JSON or YAML).
            resolve_env: Whether to resolve environment variables.
            resolve_references: Whether to resolve file references.
            
        Returns:
            Validated FSMConfig instance.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is not supported.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Load raw configuration
        raw_config = self._load_file(file_path)
        
        # Process with dataknobs_config if enabled
        if self.use_dataknobs_config:
            config_obj = DataknobsConfig(raw_config)
            processed_config = config_obj.to_dict()
        else:
            processed_config = raw_config
        
        # Resolve environment variables
        if resolve_env:
            processed_config = self._resolve_environment_vars(processed_config)
        
        # Resolve file references (includes/imports)
        if resolve_references:
            processed_config = self._resolve_references(processed_config, file_path.parent)
        
        # Apply common transformations and validate
        return self._finalize_config(processed_config)

    def load_from_dict(
        self,
        config_dict: Dict[str, Any],
        resolve_env: bool = True,
    ) -> FSMConfig:
        """Load configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary.
            resolve_env: Whether to resolve environment variables.
            
        Returns:
            Validated FSMConfig instance.
        """
        processed_config = config_dict.copy()
        
        # Process with dataknobs_config if enabled
        if self.use_dataknobs_config:
            config_obj = DataknobsConfig(processed_config)
            processed_config = config_obj.to_dict()
        
        # Resolve environment variables
        if resolve_env:
            processed_config = self._resolve_environment_vars(processed_config)
        
        # Apply common transformations and validate
        return self._finalize_config(processed_config)

    def load_from_template(
        self,
        template: Union[UseCaseTemplate, str],
        params: Dict[str, Any] | None = None,
        overrides: Dict[str, Any] | None = None,
    ) -> FSMConfig:
        """Load configuration from a template.
        
        Args:
            template: Template name or enum value.
            params: Template parameters.
            overrides: Configuration overrides.
            
        Returns:
            Validated FSMConfig instance.
        """
        if isinstance(template, str):
            template = UseCaseTemplate(template)
        
        # Apply template
        config_dict = apply_template(template, params, overrides)
        
        # Load from dictionary
        return self.load_from_dict(config_dict)

    def load_template_config(self, template_config: TemplateConfig) -> FSMConfig:
        """Load configuration from a template configuration object.
        
        Args:
            template_config: Template configuration.
            
        Returns:
            Validated FSMConfig instance.
        """
        return self.load_from_template(
            template_config.template,
            template_config.params,
            template_config.overrides,
        )

    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load raw configuration from a file.

        Args:
            file_path: Path to configuration file.

        Returns:
            Raw configuration dictionary.

        Raises:
            ValueError: If the file format is unsupported, parsing
                fails, or PyYAML is required but not installed.
        """
        # FSM tolerates non-dict roots (legacy behavior); higher-level
        # validation happens in `_finalize_config`. Helper-raised
        # errors are re-wrapped as ``ValueError`` to preserve the
        # historical surface (parse errors used to propagate as
        # ``yaml.YAMLError`` / ``json.JSONDecodeError``; this
        # consolidates them under ``ValueError`` so callers see a
        # single, stable type).
        try:
            return load_yaml_or_json(file_path, require_dict=False)
        except ConfigUnsupportedFormatError as e:
            raise ValueError(str(e)) from e
        except ConfigYAMLNotInstalledError as e:
            raise ValueError(str(e)) from e
        except ConfigParseError as e:
            raise ValueError(str(e)) from e
        except ConfigLoadError as e:
            raise ValueError(str(e)) from e

    def _resolve_environment_vars(self, config: Any) -> Any:
        """Resolve environment variables in configuration.

        Delegates to ``dataknobs_config.substitute_env_vars`` for the actual
        substitution. Adds an FSM-specific prefix-fallback: when a ``${VAR}``
        reference is unset but ``${FSM_VAR}`` is set, the prefixed value is
        used (the prefix comes from ``self._env_prefix``).

        Supported syntax (canonical):
        - ``${VAR}`` — required; raises ``ValueError`` if unset
        - ``${VAR:default}`` — DataKnobs legacy default form
        - ``${VAR:-default}`` — bash-style default
        - ``${VAR:?error_msg}`` — bash-style required with custom error

        ``substitute_keys=False`` matches the legacy inline parser,
        which only walked dict values. Out-of-tree configs may have
        literal ``${...}`` strings as dict keys; expanding them is a
        behavior change beyond the migration's documented scope.

        ``expand_user_paths=False`` matches the legacy inline parser,
        which returned ``os.environ[name]`` verbatim. The canonical helper
        defaults to ``True``, but the other in-tree config-loading callers
        (``Config._load_dict``, the legacy ``VariableSubstitution`` shim)
        also pass ``False`` — letting downstream code decide whether to
        expand ``~``-prefixed values.

        Args:
            config: Configuration to process.

        Returns:
            Configuration with resolved environment variables.
        """
        with _alias_prefixed_env_vars(self._env_prefix):
            return substitute_env_vars(
                config,
                substitute_keys=False,
                expand_user_paths=False,
            )

    def _finalize_config(self, config: Dict[str, Any]) -> FSMConfig:
        """Apply final transformations and validate configuration.
        
        This method applies all common transformations that should happen
        regardless of the source of the configuration.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            Validated FSMConfig instance.
        """
        # Transform simple format to network format if needed
        config = self._transform_simple_to_network(config)
        
        # Transform network-level arcs to state-level arcs if present
        config = self._transform_network_arcs(config)
        
        # Transform state functions field to transforms list
        config = self._transform_state_functions(config)
        
        # Validate and return
        return validate_config(config)
    
    def _transform_simple_to_network(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform simple format to network format if needed.
        
        Detects if the config is in simple format (has 'states' and 'arcs' at
        top level without 'networks') and transforms it to network format.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            Transformed configuration.
        """
        # Check if this is a simple format config
        if 'states' in config and 'networks' not in config:
            # Convert states from dict to list format if needed
            states = config['states']
            arcs = list(config.get('arcs', []))  # Start with existing arcs if any
            
            if isinstance(states, dict):
                # Convert dict-style states to list-style
                states_list = []
                initial_state = config.get('initial_state')
                
                for name, state_config in states.items():
                    state = state_config.copy() if isinstance(state_config, dict) else {}
                    state['name'] = name
                    
                    # Mark initial state if specified
                    if initial_state and name == initial_state:
                        state['is_start'] = True
                    
                    # Extract inline transitions and convert to arcs
                    for transition_type in ['on_complete', 'on_error', 'on_timeout']:
                        if transition_type in state:
                            transition = state.pop(transition_type)
                            if isinstance(transition, dict) and 'target' in transition:
                                arc = {
                                    'from': name,
                                    'to': transition['target'],
                                    'type': transition_type.replace('on_', '')  # on_complete -> complete
                                }
                                # Add any conditions or transforms
                                if 'condition' in transition:
                                    arc['condition'] = transition['condition']
                                if 'transform' in transition:
                                    arc['transform'] = transition['transform']
                                arcs.append(arc)
                    
                    # Mark final states
                    if state.get('final'):
                        state['is_end'] = True
                        state.pop('final')  # Remove the 'final' field as we use 'is_end'
                    
                    states_list.append(state)
                states = states_list
                
                # If no initial state was specified, mark the first state as start
                if not initial_state and states_list:
                    states_list[0]['is_start'] = True
            
            # Transform to network format
            network_config = {
                'name': config.get('name', 'default_fsm'),
                'networks': [{
                    'name': 'main',
                    'states': states,
                    'arcs': self._add_type_to_transforms(arcs)
                }],
                'main_network': 'main'
            }
            
            # Handle data_mode transformation
            if 'data_mode' in config:
                mode = config['data_mode']
                if isinstance(mode, str):
                    # Convert string to proper data_mode config
                    network_config['data_mode'] = {
                        'default': mode.lower() if mode.lower() in ['copy', 'reference', 'direct'] else 'copy'
                    }
                else:
                    network_config['data_mode'] = mode
            else:
                network_config['data_mode'] = {'default': 'copy'}
            
            # Handle initial_state if present
            if 'initial_state' in config:
                network_config['networks'][0]['initial_state'] = config['initial_state']
            
            # Copy over other top-level fields
            for key in ['resources', 'templates', 'functions', 'execution', 'metadata']:
                if key in config:
                    network_config[key] = config[key]
            
            return network_config
        
        return config
    
    def _add_type_to_transforms(self, arcs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add type field to arc transforms if missing.
        
        Args:
            arcs: List of arc configurations.
            
        Returns:
            Updated arc configurations.
        """
        updated_arcs = []
        for arc in arcs:
            arc_copy = arc.copy()
            if 'transform' in arc_copy and isinstance(arc_copy['transform'], dict):
                if 'type' not in arc_copy['transform']:
                    # Infer type from the content
                    if 'code' in arc_copy['transform']:
                        arc_copy['transform']['type'] = 'inline'
                    elif 'lambda' in arc_copy['transform']:
                        arc_copy['transform']['type'] = 'lambda'
                    elif 'module' in arc_copy['transform']:
                        arc_copy['transform']['type'] = 'module'
                    else:
                        arc_copy['transform']['type'] = 'inline'
            updated_arcs.append(arc_copy)
        return updated_arcs
    
    def _transform_network_arcs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform network-level arcs format to state-level arcs format.
        
        This allows for a more intuitive configuration format where arcs
        are defined at the network level with 'from' and 'to' fields,
        rather than attached to the source state.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            Transformed configuration.
        """
        config = config.copy()
        
        # Process each network
        if 'networks' in config:
            for network in config['networks']:
                if 'arcs' in network and isinstance(network['arcs'], list):
                    # Build a map of state name to state config
                    state_map = {}
                    for state in network.get('states', []):
                        state_map[state['name']] = state
                        # Ensure each state has an arcs list
                        if 'arcs' not in state:
                            state['arcs'] = []
                    
                    # Transform network-level arcs to state-level arcs
                    for arc in network['arcs']:
                        if 'from' in arc and 'to' in arc:
                            from_state = arc['from']
                            to_state = arc['to']
                            
                            # Create state-level arc config
                            state_arc = {
                                'target': to_state
                            }
                            
                            # Handle legacy pre_test format
                            if 'pre_test' in arc and 'condition' not in arc:
                                pre_test = arc['pre_test']
                                if isinstance(pre_test, dict) and 'test' in pre_test:
                                    # Convert pre_test.test to condition
                                    state_arc['condition'] = self._convert_to_function_reference(pre_test['test'])
                                elif isinstance(pre_test, str):
                                    # Direct function reference or inline code
                                    state_arc['condition'] = self._convert_to_function_reference(pre_test)
                            
                            # Copy optional fields
                            for field in ['name', 'condition', 'transform', 'priority', 'metadata']:
                                if field in arc:
                                    if field == 'name':
                                        # Store arc name in metadata
                                        if 'metadata' not in state_arc:
                                            state_arc['metadata'] = {}
                                        state_arc['metadata']['name'] = arc[field]
                                    elif field == 'condition':
                                        # Handle condition field
                                        condition = arc[field]
                                        if isinstance(condition, dict):
                                            # Check for simple condition types
                                            if condition.get('type') == 'success':
                                                # Transform to check validation success in data
                                                state_arc['condition'] = {
                                                    'type': 'inline',
                                                    'code': 'data.get("valid", True)'
                                                }
                                            elif condition.get('type') == 'failure':
                                                # Transform to check validation failure in data
                                                state_arc['condition'] = {
                                                    'type': 'inline',
                                                    'code': 'not data.get("valid", True)'
                                                }
                                            else:
                                                # Keep as is
                                                state_arc[field] = condition
                                        elif isinstance(condition, str):
                                            # Simple string condition
                                            if condition == 'success':
                                                state_arc['condition'] = {
                                                    'type': 'inline',
                                                    'code': 'data.get("valid", True)'
                                                }
                                            elif condition == 'failure':
                                                state_arc['condition'] = {
                                                    'type': 'inline',
                                                    'code': 'not data.get("valid", True)'
                                                }
                                            else:
                                                # Check if registered function or inline code
                                                state_arc['condition'] = self._convert_to_function_reference(condition)
                                        else:
                                            state_arc[field] = condition
                                    else:
                                        state_arc[field] = arc[field]
                            
                            # Add arc to the source state
                            if from_state in state_map:
                                state_map[from_state]['arcs'].append(state_arc)
                    
                    # Remove network-level arcs since they've been transformed
                    del network['arcs']
                    
        
        return config
    
    def _transform_state_functions(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform state 'functions' field to proper schema format.
        
        This converts the legacy 'functions' format to the proper schema:
        - functions.validate -> validators (state validation)
        - functions.transform -> transforms (state transformation when entering state)
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            Transformed configuration.
        """
        config = config.copy()
        
        # Process each network
        if 'networks' in config:
            for network in config['networks']:
                if 'states' in network:
                    for state in network['states']:
                        # Check if state has 'functions' field
                        if 'functions' in state and isinstance(state['functions'], dict):
                            functions = state['functions']
                            
                            # Convert validate function to validators list
                            if 'validate' in functions:
                                validate_func = functions['validate']
                                state['validators'] = [self._convert_to_function_reference(validate_func)]
                            
                            # Convert transform function to transforms list (StateTransform)
                            if 'transform' in functions:
                                transform_func = functions['transform']
                                state['transforms'] = [self._convert_to_function_reference(transform_func)]
                            
                            # Remove the functions field as it's not in the schema
                            del state['functions']

                        # Also handle direct 'transform' field (singular) for convenience
                        if 'transform' in state and 'transforms' not in state:
                            transform = state['transform']
                            if isinstance(transform, list):
                                state['transforms'] = transform
                            else:
                                state['transforms'] = [self._convert_to_function_reference(transform)]
                            del state['transform']

                        # Similarly handle direct 'validator' field (singular)
                        if 'validator' in state and 'validators' not in state:
                            validator = state['validator']
                            if isinstance(validator, list):
                                state['validators'] = validator
                            else:
                                state['validators'] = [self._convert_to_function_reference(validator)]
                            del state['validator']

        return config
    
    def _resolve_references(self, config: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
        """Resolve file references (includes/imports) in configuration.
        
        Supports:
        - $include: path/to/file.yaml
        - $import: { file: path/to/file.yaml, path: some.nested.path }
        
        Args:
            config: Configuration dictionary.
            base_path: Base path for resolving relative paths.
            
        Returns:
            Configuration with resolved references.
        """
        processed = {}
        
        for key, value in config.items():
            if key == "$include" and isinstance(value, str):
                # Load and merge included file
                include_path = base_path / value
                if include_path.as_posix() not in self._included_configs:
                    included = self._load_file(include_path)
                    self._included_configs[include_path.as_posix()] = included
                else:
                    included = self._included_configs[include_path.as_posix()]
                
                # Recursively resolve references in included content
                included = self._resolve_references(included, include_path.parent)
                
                # Merge with current config
                for inc_key, inc_value in included.items():
                    if inc_key not in processed:
                        processed[inc_key] = inc_value
            
            elif key == "$import" and isinstance(value, dict):
                # Import specific path from file
                file_path = base_path / value["file"]
                path_expr = value.get("path", "")
                
                if file_path.as_posix() not in self._included_configs:
                    imported = self._load_file(file_path)
                    self._included_configs[file_path.as_posix()] = imported
                else:
                    imported = self._included_configs[file_path.as_posix()]
                
                # Navigate to specified path
                if path_expr:
                    for part in path_expr.split("."):
                        imported = imported.get(part, {})
                
                # Recursively resolve references
                if isinstance(imported, dict):
                    imported = self._resolve_references(imported, file_path.parent)
                
                return imported
            
            elif isinstance(value, dict):
                processed[key] = self._resolve_references(value, base_path)
            
            elif isinstance(value, list):
                processed[key] = [
                    self._resolve_references(item, base_path) if isinstance(item, dict) else item
                    for item in value
                ]
            
            else:
                processed[key] = value
        
        return processed

    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """Validate a configuration file without fully loading it.
        
        Args:
            file_path: Path to configuration file.
            
        Returns:
            True if valid, False otherwise.
        """
        try:
            self.load_from_file(file_path)
            return True
        except Exception:
            return False

    def merge_configs(self, *configs: FSMConfig) -> FSMConfig:
        """Merge multiple FSM configurations.
        
        Later configurations override earlier ones.
        
        Args:
            *configs: FSMConfig instances to merge.
            
        Returns:
            Merged FSMConfig instance.
        """
        merged_dict = {}
        
        for config in configs:
            config_dict = config.model_dump()
            self._deep_merge(merged_dict, config_dict)
        
        return validate_config(merged_dict)

    def _deep_merge(self, base: Dict, updates: Dict) -> Dict:
        """Deep merge two dictionaries.
        
        Args:
            base: Base dictionary (modified in place).
            updates: Updates to apply.
            
        Returns:
            Merged dictionary.
        """
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            elif key in base and isinstance(base[key], list) and isinstance(value, list):
                # For lists, we extend rather than replace
                base[key].extend(value)
            else:
                base[key] = value
        
        return base
