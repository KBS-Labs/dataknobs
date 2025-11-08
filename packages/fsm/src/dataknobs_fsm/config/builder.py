"""FSM builder for constructing FSM instances from configuration.

This module provides the FSMBuilder class that constructs executable FSM
instances from configuration objects, including:
- Resource registration and initialization
- Function resolution and registration
- Network and state construction
- Validation of completeness
"""

import importlib
from typing import Any, Callable, Dict, List, Type


from dataknobs_fsm.config.schema import (
    ArcConfig,
    FSMConfig,
    FunctionReference,
    NetworkConfig,
    PushArcConfig,
    ResourceConfig,
    ResourceType,
    StateConfig,
)
from dataknobs_fsm.core.arc import ArcDefinition, PushArc
from dataknobs_fsm.core.data_modes import DataHandler, DataHandlingMode, get_data_handler
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateDefinition, StateType
from dataknobs_fsm.core.transactions import (
    TransactionManager,
    TransactionStrategy,
    SingleTransactionManager,
    BatchTransactionManager,
    ManualTransactionManager,
)
from dataknobs_fsm.core.fsm import FSM as CoreFSMClass  # noqa: N811
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.functions.base import (
    IResource,
    IStateTestFunction,
    ITransformFunction,
    IValidationFunction,
)
from dataknobs_fsm.resources.manager import ResourceManager
from dataknobs_fsm.functions.manager import (
    FunctionManager,
    FunctionSource
)


class FSMBuilder:
    """Build executable FSM instances from configuration."""

    def __init__(self):
        """Initialize the FSMBuilder."""
        self._resource_manager = ResourceManager()
        self._function_manager = FunctionManager()
        self._networks: Dict[str, StateNetwork] = {}
        self._data_handlers: Dict[DataHandlingMode, DataHandler] = {}
        self._transaction_manager: TransactionManager | None = None

        # Register built-in functions on initialization
        self._register_builtin_functions()

    def build(self, config: FSMConfig) -> CoreFSMClass:
        """Build an FSM instance from configuration.
        
        Args:
            config: FSM configuration.
            
        Returns:
            Executable FSM instance.
            
        Raises:
            ValueError: If configuration is invalid or incomplete.
        """
        # Clear previous build state
        self._networks.clear()
        
        # 1. Register resources
        self._register_resources(config.resources)
        
        # 2. Initialize data handlers
        self._init_data_handlers(config.data_mode)
        
        # 3. Initialize transaction manager
        self._init_transaction_manager(config.transaction)
        
        # 4. Build networks
        for network_config in config.networks:
            network = self._build_network(network_config, config)
            self._networks[network.name] = network
        
        # 5. Validate completeness
        self._validate_completeness(config)
        
        # 6. Create core FSM instance
        from dataknobs_fsm.core.modes import ProcessingMode as CoreDataMode
        from dataknobs_fsm.core.modes import TransactionMode as CoreTransactionMode
        
        # Map config modes to core modes
        data_mode = CoreDataMode.SINGLE  # Default to SINGLE for now
        transaction_mode = CoreTransactionMode.NONE  # Default to NONE
        
        fsm = CoreFSMClass(
            name=config.name,
            data_mode=data_mode,
            transaction_mode=transaction_mode,
            description=config.description,
            resource_manager=self._resource_manager,
            transaction_manager=self._transaction_manager,
        )
        
        # Store config in FSM for reference
        fsm.config = config
        
        # Register all functions from builder into core FSM's function registry
        for func_name in self._function_manager.list_functions():
            wrapper = self._function_manager.get_function(func_name)
            if wrapper:
                # The FSM's function registry expects callable functions
                # If it's a FunctionWrapper, get the actual function
                if hasattr(wrapper, 'func'):
                    fsm.function_registry.register(func_name, wrapper.func)
                else:
                    fsm.function_registry.register(func_name, wrapper)
        
        # Add networks to FSM
        for network_name, network in self._networks.items():
            fsm.add_network(network, is_main=(network_name == config.main_network))
        
        # Return the core FSM directly
        return fsm

    def register_function(self, name: str, func: Callable) -> None:
        """Register a custom function.
        
        Args:
            name: Function name for reference in configuration.
            func: Function implementation.
        """
        self._function_manager.register_function(name, func, FunctionSource.REGISTERED)

    def _register_builtin_functions(self) -> None:
        """Register built-in functions from the library."""
        # Import built-in function modules
        try:
            from dataknobs_fsm.functions.library import validators, transformers
            
            # Register validators
            for name in dir(validators):
                if not name.startswith("_"):
                    obj = getattr(validators, name)
                    if callable(obj):
                        self._function_manager.register_function(
                            f"validators.{name}", obj, FunctionSource.BUILTIN
                        )
            
            # Register transformers
            for name in dir(transformers):
                if not name.startswith("_"):
                    obj = getattr(transformers, name)
                    if callable(obj):
                        self._function_manager.register_function(
                            f"transformers.{name}", obj, FunctionSource.BUILTIN
                        )
        
        except ImportError:
            # Built-in functions not yet implemented
            pass

    def _register_resources(self, resources: List[ResourceConfig]) -> None:
        """Register resources with the resource manager.
        
        Args:
            resources: Resource configurations.
        """
        for resource_config in resources:
            resource = self._create_resource(resource_config)
            self._resource_manager.register_provider(resource_config.name, resource)

    def _create_resource(self, config: ResourceConfig) -> IResource:
        """Create a resource instance from configuration.
        
        Args:
            config: Resource configuration.
            
        Returns:
            Resource instance.
            
        Raises:
            ValueError: If resource type is not supported.
        """
        # Map resource types to classes
        resource_classes = {
            "database": "dataknobs_fsm.resources.database.DatabaseResourceAdapter",
            "filesystem": "dataknobs_fsm.resources.filesystem.FileSystemResource",
            "http": "dataknobs_fsm.resources.http.HTTPServiceResource",
            "llm": "dataknobs_fsm.resources.llm.LLMResource",
            "vector_store": "dataknobs_fsm.resources.vector_store.VectorStoreResource",
        }
        
        if config.type == "custom":
            # Custom resource must be in configuration
            if "class" not in config.config:
                raise ValueError("Custom resource requires 'class' in configuration")
            class_path = config.config["class"]
        else:
            class_path = resource_classes.get(config.type)
            if not class_path:
                raise ValueError(f"Unsupported resource type: {config.type}")
        
        # Import and instantiate resource class
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        resource_class = getattr(module, class_name)
        
        # Create resource with configuration, adding name if needed
        kwargs = config.config.copy()
        if hasattr(resource_class, "__init__") and "name" in resource_class.__init__.__code__.co_varnames:
            kwargs["name"] = config.name
        return resource_class(**kwargs)

    def _init_data_handlers(self, config: Any) -> None:
        """Initialize data handlers for each mode.
        
        Args:
            config: Data mode configuration.
        """
        for mode in DataHandlingMode:
            self._data_handlers[mode] = get_data_handler(mode)

    def _init_transaction_manager(self, config: Any) -> None:
        """Initialize transaction manager.
        
        Args:
            config: Transaction configuration.
        """
        if config.strategy == TransactionStrategy.SINGLE:
            self._transaction_manager = SingleTransactionManager()
        elif config.strategy == TransactionStrategy.BATCH:
            self._transaction_manager = BatchTransactionManager(
                batch_size=config.batch_size
            )
        elif config.strategy == TransactionStrategy.MANUAL:
            self._transaction_manager = ManualTransactionManager()
        else:
            # Default to single transaction
            self._transaction_manager = SingleTransactionManager()

    def _build_network(self, network_config: NetworkConfig, fsm_config: FSMConfig) -> StateNetwork:
        """Build a state network from configuration.
        
        Args:
            network_config: Network configuration.
            fsm_config: Parent FSM configuration.
            
        Returns:
            StateNetwork instance.
        """
        network = StateNetwork(
            name=network_config.name,
            description=network_config.metadata.get("description", "") if network_config.metadata else None,
        )
        
        # Create states
        state_defs = {}
        for state_config in network_config.states:
            state_def = self._build_state(state_config, fsm_config)
            state_defs[state_def.name] = state_def
            # Pass initial and final flags based on state type
            from dataknobs_fsm.core.state import StateType
            network.add_state(
                state_def,
                initial=(state_def.type in [StateType.START, StateType.START_END]),
                final=(state_def.type in [StateType.END, StateType.START_END])
            )
        
        # Create arcs with definition order tracking
        arc_definition_order = 0
        for state_config in network_config.states:
            state_def = state_defs[state_config.name]
            for arc_config in state_config.arcs:
                arc = self._build_arc(arc_config, state_def, network, fsm_config, arc_definition_order)
                arc_definition_order += 1
                # Add arc to both the state definition and the network
                state_def.outgoing_arcs.append(arc)
                # Also register the arc with the network for execution
                # Extract function names for network registration
                pre_test_name = None
                transform_name = None
                if arc.pre_test:
                    pre_test_name = getattr(arc.pre_test, '__name__', str(arc.pre_test))
                if arc.transform:
                    transform_name = getattr(arc.transform, '__name__', str(arc.transform))
                
                network.add_arc(
                    source_state=state_config.name,
                    target_state=arc_config.target,
                    pre_test=pre_test_name,
                    transform=transform_name,
                    metadata=arc_config.metadata
                )
        
        return network

    def _build_state(self, state_config: StateConfig, fsm_config: FSMConfig) -> StateDefinition:
        """Build a state definition from configuration.
        
        Args:
            state_config: State configuration.
            fsm_config: Parent FSM configuration.
            
        Returns:
            StateDefinition instance.
        """
        # Build schema if provided
        schema = None
        if state_config.data_schema:
            schema = self._build_schema(state_config.data_schema)
        
        # Resolve pre-validators
        pre_validators = []
        for func_ref in state_config.pre_validators:
            pre_validator = self._resolve_function(func_ref, IValidationFunction)
            pre_validators.append(pre_validator)

        # Resolve validators
        validators = []
        for func_ref in state_config.validators:
            validator = self._resolve_function(func_ref, IValidationFunction)
            validators.append(validator)
        
        # Resolve transforms
        transforms = []
        for func_ref in state_config.transforms:
            transform = self._resolve_function(func_ref, ITransformFunction)
            transforms.append(transform)
            # Don't re-register wrapped functions - they're already in the manager
            # The transform is already an InterfaceWrapper with proper async handling
        
        # Determine data mode
        data_mode = state_config.data_mode or fsm_config.data_mode.default
        
        # Create state definition with correct field names
        state_def = StateDefinition(name=state_config.name)
        state_def.schema = schema
        state_def.pre_validation_functions = pre_validators
        state_def.validation_functions = validators
        state_def.transform_functions = transforms
        # Look up actual resource configs from the FSM config
        resource_map = {res.name: res for res in fsm_config.resources}
        state_def.resource_requirements = [
            resource_map[r] if r in resource_map else ResourceConfig(name=r, type=ResourceType.CUSTOM)
            for r in state_config.resources
        ]
        state_def.data_mode = data_mode
        # Handle states that are both start and end
        if state_config.is_start and state_config.is_end:
            state_def.type = StateType.START_END
        elif state_config.is_start:
            state_def.type = StateType.START
        elif state_config.is_end:
            state_def.type = StateType.END
        else:
            state_def.type = StateType.NORMAL
        state_def.metadata = state_config.metadata
        
        return state_def

    def _get_function_name(self, func: Any) -> str | None:
        """Extract the name from a function or wrapped function.

        Args:
            func: Function or wrapped function

        Returns:
            Function name or None
        """
        if not func:
            return None

        # Check for various name attributes
        if hasattr(func, 'name'):
            return func.name
        elif hasattr(func, '__name__'):
            # Skip generic names that would cause collisions
            name = func.__name__
            if name not in ['<lambda>', 'inline_func']:
                return name
        elif hasattr(func, 'wrapper') and hasattr(func.wrapper, 'name'):
            # InterfaceWrapper case
            return func.wrapper.name
        else:
            # Search for the function in the manager
            for fname in self._function_manager.list_functions():
                wrapper = self._function_manager.get_function(fname)
                if wrapper and wrapper.func == func:
                    return fname
        return None

    def _build_arc(
        self,
        arc_config: ArcConfig,
        source_state: StateDefinition,
        network: StateNetwork,
        fsm_config: FSMConfig,
        definition_order: int = 0,
    ) -> ArcDefinition:
        """Build an arc definition from configuration.

        Args:
            arc_config: Arc configuration.
            source_state: Source state definition.
            network: Parent network.
            fsm_config: Parent FSM configuration.
            definition_order: Order in which this arc was defined.

        Returns:
            ArcDefinition instance.
        """
        # Resolve condition function
        condition = None
        condition_name = None
        if arc_config.condition:
            condition = self._resolve_function(arc_config.condition, IStateTestFunction)
            # Get or generate a unique name for the condition
            condition_name = self._get_function_name(condition)
            # Lambda functions get the unhelpful name "<lambda>" so we need to generate a unique name
            if not condition_name or condition_name == "<lambda>":
                # Generate a unique name based on arc endpoints and code/function id
                if arc_config.condition.type == "inline" and arc_config.condition.code:
                    # Use hash of code for uniqueness
                    condition_name = f"condition_{source_state.name}_{arc_config.target}_{abs(hash(arc_config.condition.code))}"
                else:
                    condition_name = f"condition_{source_state.name}_{arc_config.target}_{id(condition)}"
            # Register the function with the function manager so it gets transferred to FSM later
            if condition_name and not self._function_manager.has_function(condition_name):
                # Register the resolved function - if it's an InterfaceWrapper, register it as-is
                # since InterfaceWrapper is callable and handles the interface correctly
                if hasattr(condition, 'test'):
                    # It's an IStateTestFunction interface, register the test method
                    self._function_manager.register_function(condition_name, condition.test, FunctionSource.INLINE)
                else:
                    # Register as-is
                    self._function_manager.register_function(condition_name, condition, FunctionSource.INLINE)

        # Resolve transform function
        transform = None
        transform_name = None
        if arc_config.transform:
            transform = self._resolve_function(arc_config.transform, ITransformFunction)
            # Get or generate a unique name for the transform
            transform_name = self._get_function_name(transform)
            # Lambda functions get the unhelpful name "<lambda>" so we need to generate a unique name
            if not transform_name or transform_name == "<lambda>":
                # Generate a unique name based on arc endpoints and code/function id
                if arc_config.transform.type == "inline" and arc_config.transform.code:
                    # Use hash of code for uniqueness
                    transform_name = f"transform_{source_state.name}_{arc_config.target}_{abs(hash(arc_config.transform.code))}"
                else:
                    transform_name = f"transform_{source_state.name}_{arc_config.target}_{id(transform)}"
            # Register the function with the function manager so it gets transferred to FSM later
            if transform_name and not self._function_manager.has_function(transform_name):
                # Register the resolved function - we need the actual callable
                self._function_manager.register_function(transform_name, transform, FunctionSource.INLINE)
        
        # Create appropriate arc type
        if isinstance(arc_config, PushArcConfig):
            # Use the names we determined above
            pre_test_name = condition_name
            arc_transform_name = transform_name
            
            # Push arc to another network
            arc = PushArc(
                target_state=arc_config.target,  # Even push arcs have a target state
                target_network=arc_config.target_network,
                return_state=arc_config.return_state,
                pre_test=pre_test_name,
                transform=arc_transform_name,
                priority=arc_config.priority,
                definition_order=definition_order,
                metadata=arc_config.metadata,
            )
            # Store resources separately if needed
            arc.required_resources = {r: r for r in arc_config.resources}
            return arc
        else:
            # Regular arc within network
            # Use the names we determined above
            pre_test_name = condition_name
            arc_transform_name = transform_name
            
            arc = ArcDefinition(
                target_state=arc_config.target,
                pre_test=pre_test_name,
                transform=arc_transform_name,
                priority=arc_config.priority,
                definition_order=definition_order,
                metadata=arc_config.metadata,
            )
            # Store resources separately if needed
            arc.required_resources = {r: r for r in arc_config.resources}
            return arc

    def _build_schema(self, schema_config: Dict[str, Any]) -> Any:
        """Build a schema from configuration.
        
        Args:
            schema_config: Schema configuration (JSON Schema format).
            
        Returns:
            Schema object for validation.
        """
        # Handle JSON Schema format - create a simple validation schema
        # that can be used by the FSM's validation system
        
        # For JSON Schema format, create a simple validator wrapper
        class JSONSchemaValidator:
            """Simple JSON Schema validator for FSM data validation.

            Validates data against JSON Schema definitions, checking required fields
            and type constraints. Supports object schemas with properties and required fields.
            """

            def __init__(self, schema_def):
                self.schema_def = schema_def
            
            def validate(self, data):
                """Validate data against JSON schema."""
                from dataknobs_data import Record
                
                # Convert Record to dict if needed
                if isinstance(data, Record):
                    data_dict = data.to_dict()
                elif isinstance(data, dict):
                    data_dict = data
                else:
                    return type('Result', (), {
                        'valid': False,
                        'errors': [f'Expected object or Record, got {type(data).__name__}']
                    })()
                
                # Simple validation for basic JSON schema
                if self.schema_def.get('type') == 'object':
                    errors = []
                    properties = self.schema_def.get('properties', {})
                    required = self.schema_def.get('required', [])
                    
                    # Check required fields
                    for field in required:
                        if field not in data_dict:
                            errors.append(f"Required field '{field}' is missing")
                    
                    # Check field types
                    for field, value in data_dict.items():
                        if field in properties:
                            field_schema = properties[field]
                            field_type = field_schema.get('type')
                            if field_type and not self._validate_type(value, field_type):
                                errors.append(f"Field '{field}' has wrong type")
                    
                    return type('Result', (), {
                        'valid': len(errors) == 0,
                        'errors': errors
                    })()
                else:
                    # Simple pass-through for non-object schemas
                    return type('Result', (), {'valid': True, 'errors': []})()
            
            def _validate_type(self, value, expected_type):
                """Validate value type."""
                type_map = {
                    'string': str,
                    'integer': int,
                    'number': (int, float),
                    'boolean': bool,
                    'array': list,
                    'object': dict
                }
                expected_python_type = type_map.get(expected_type)
                if expected_python_type:
                    return isinstance(value, expected_python_type)
                return True
        
        return JSONSchemaValidator(schema_config)

    def _resolve_function(
        self,
        func_ref: FunctionReference,
        expected_type: Type | None = None,
    ) -> Callable:
        """Resolve a function reference to a callable.
        
        Args:
            func_ref: Function reference.
            expected_type: Expected function interface type.
            
        Returns:
            Resolved function callable.
            
        Raises:
            ValueError: If function cannot be resolved.
        """
        if func_ref.type == "builtin":
            # Look up built-in function
            wrapper = self._function_manager.get_function(func_ref.name)
            if not wrapper:
                raise ValueError(f"Built-in function not found: {func_ref.name}")
            func = wrapper
        
        elif func_ref.type == "registered":
            # Look up registered function
            wrapper = self._function_manager.get_function(func_ref.name)
            if not wrapper:
                raise ValueError(f"Registered function not found: {func_ref.name}")
            func = wrapper
        
        elif func_ref.type == "custom":
            # Check manager first
            wrapper = self._function_manager.get_function(func_ref.name)
            if wrapper:
                func = wrapper
            else:
                # Import custom function
                module = importlib.import_module(func_ref.module)
                func = getattr(module, func_ref.name)
                # Register it for future use
                self._function_manager.register_function(func_ref.name, func, FunctionSource.REGISTERED)
        
        elif func_ref.type == "inline":
            # Use function manager's inline handling
            wrapper = self._function_manager.resolve_function(func_ref.code, expected_type)
            if not wrapper:
                raise ValueError(f"Failed to create inline function from: {func_ref.code}")
            func = wrapper
            # Mark as already wrapped to avoid double wrapping
            func._is_wrapped = True
        
        else:
            raise ValueError(f"Unknown function type: {func_ref.type}")
        
        # Apply parameters if provided
        if func_ref.params:
            # Create a partial function with parameters
            import functools
            func = functools.partial(func, **func_ref.params)
        
        # Validate type if specified
        # Skip wrapping if already wrapped by function manager
        if expected_type and not isinstance(func, expected_type) and not getattr(func, '_is_wrapped', False):
            # Wrap function to match expected interface
            wrapped = self._wrap_function(func, expected_type)
            # Preserve the original function name if it exists
            if hasattr(func, 'name'):
                wrapped.name = func.name
            elif func_ref.type == "registered" and func_ref.name:
                wrapped.name = func_ref.name
            func = wrapped
        
        # For inline functions, ensure they have a name
        if func_ref.type == "inline" and hasattr(func, 'name'):
            # Name already set by function manager
            pass
        
        return func

    def _wrap_function(self, func: Callable, interface: Type) -> Any:
        """Wrap a function to match an expected interface using unified manager.

        Args:
            func: Function to wrap.
            interface: Expected interface type.

        Returns:
            Wrapped function implementing the interface.
        """
        # Resolve the function through the unified manager
        wrapper = self._function_manager.resolve_function(func, interface)
        return wrapper

    def _validate_completeness(self, config: FSMConfig) -> None:
        """Validate that the FSM configuration is complete and consistent.
        
        Args:
            config: FSM configuration.
            
        Raises:
            ValueError: If configuration is incomplete or inconsistent.
        """
        # Check main network exists
        if config.main_network not in self._networks:
            raise ValueError(f"Main network '{config.main_network}' not found")
        
        # Check all arc targets exist
        for network in self._networks.values():
            state_names = {state.name for state in network.states.values()}
            for state in network.states.values():
                for arc in network.get_arcs_from_state(state.name):
                    if isinstance(arc, PushArc):
                        # Check target network exists
                        if arc.target_network not in self._networks:
                            raise ValueError(f"Target network '{arc.target_network}' not found")
                        # Check return state exists if specified
                        if arc.return_state and arc.return_state not in state_names:
                            raise ValueError(f"Return state '{arc.return_state}' not found")
                    else:
                        # Check target state exists
                        if arc.target_state not in state_names:
                            raise ValueError(f"Arc target '{arc.target_state}' not found in network")
        
        # Check resource references
        resource_names = {res.name for res in config.resources}
        for network in self._networks.values():
            for state in network.states.values():
                for resource_req in state.resource_requirements:
                    if resource_req.name not in resource_names:
                        raise ValueError(f"Resource '{resource_req.name}' not found")

    def _create_execution_context(self, config: FSMConfig) -> ExecutionContext:
        """Create execution context from configuration.
        
        Args:
            config: FSM configuration.
            
        Returns:
            ExecutionContext instance.
        """
        # Map our DataHandlingMode to the execution context's ProcessingMode
        from dataknobs_fsm.core.modes import ProcessingMode as CoreDataMode
        
        # Simple mapping - we'll use SINGLE mode for now
        # This could be enhanced to support batch and stream modes
        return ExecutionContext(
            data_mode=CoreDataMode.SINGLE,
            resources={}  # Resources will be populated during execution
        )


# FSM wrapper class removed - functionality moved to core FSM
# The FSMBuilder now returns the core FSM directly with all
# execution capabilities integrated
