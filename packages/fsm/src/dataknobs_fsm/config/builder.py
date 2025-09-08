"""FSM builder for constructing FSM instances from configuration.

This module provides the FSMBuilder class that constructs executable FSM
instances from configuration objects, including:
- Resource registration and initialization
- Function resolution and registration
- Network and state construction
- Validation of completeness
"""

import importlib
from typing import Any, Callable, Dict, List, Optional, Type

from dataknobs_data import Field

from dataknobs_fsm.config.schema import (
    ArcConfig,
    FSMConfig,
    FunctionReference,
    NetworkConfig,
    PushArcConfig,
    ResourceConfig,
    StateConfig,
)
from dataknobs_fsm.core.arc import ArcDefinition, PushArc
from dataknobs_fsm.core.data_modes import DataHandler, DataMode, get_data_handler
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.core.state import StateDefinition, StateType
from dataknobs_fsm.core.transactions import (
    TransactionManager,
    TransactionStrategy,
    SingleTransactionManager,
    BatchTransactionManager,
    ManualTransactionManager,
)
from dataknobs_fsm.core.fsm import FSM as CoreFSM
from dataknobs_fsm.execution.context import ExecutionContext
from dataknobs_fsm.execution.engine import ExecutionEngine
from dataknobs_fsm.functions.base import (
    IResource,
    IStateTestFunction,
    ITransformFunction,
    IValidationFunction,
)
from dataknobs_fsm.resources.manager import ResourceManager


class FSMBuilder:
    """Build executable FSM instances from configuration."""

    def __init__(self):
        """Initialize the FSMBuilder."""
        self._resource_manager = ResourceManager()
        self._function_registry: Dict[str, Callable] = {}
        self._builtin_functions: Dict[str, Callable] = {}
        self._networks: Dict[str, StateNetwork] = {}
        self._data_handlers: Dict[DataMode, DataHandler] = {}
        self._transaction_manager: Optional[TransactionManager] = None
        
        # Register built-in functions on initialization
        self._register_builtin_functions()

    def build(self, config: FSMConfig) -> "FSM":
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
        from dataknobs_fsm.core.modes import DataMode as CoreDataMode
        from dataknobs_fsm.core.modes import TransactionMode as CoreTransactionMode
        
        # Map config modes to core modes
        data_mode = CoreDataMode.SINGLE  # Default to SINGLE for now
        transaction_mode = CoreTransactionMode.NONE  # Default to NONE
        
        fsm = CoreFSM(
            name=config.name,
            data_mode=data_mode,
            transaction_mode=transaction_mode,
            description=config.description,
        )
        
        # Register all functions from builder into core FSM's function registry
        for func_name, func in self._function_registry.items():
            fsm.function_registry.register(func_name, func)
        
        # Add networks to FSM
        for network_name, network in self._networks.items():
            fsm.add_network(network, is_main=(network_name == config.main_network))
        
        # 7. Create wrapper FSM with execution capabilities
        return FSM(
            core_fsm=fsm,
            config=config,
            resource_manager=self._resource_manager,
            transaction_manager=self._transaction_manager,
            function_registry=self._function_registry,
        )

    def register_function(self, name: str, func: Callable) -> None:
        """Register a custom function.
        
        Args:
            name: Function name for reference in configuration.
            func: Function implementation.
        """
        self._function_registry[name] = func

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
                        self._builtin_functions[f"validators.{name}"] = obj
            
            # Register transformers
            for name in dir(transformers):
                if not name.startswith("_"):
                    obj = getattr(transformers, name)
                    if callable(obj):
                        self._builtin_functions[f"transformers.{name}"] = obj
        
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
        for mode in DataMode:
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
            network.add_state(state_def)
        
        # Create arcs
        for state_config in network_config.states:
            state_def = state_defs[state_config.name]
            for arc_config in state_config.arcs:
                arc = self._build_arc(arc_config, state_def, network, fsm_config)
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
        if state_config.schema:
            schema = self._build_schema(state_config.schema)
        
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
            # Register transform in function registry for potential arc use
            func_name = f"{state_config.name}_transform"
            self._function_registry[func_name] = transform
        
        # Determine data mode
        data_mode = state_config.data_mode or fsm_config.data_mode.default
        
        # Create state definition with correct field names
        state_def = StateDefinition(name=state_config.name)
        state_def.schema = schema
        state_def.validation_functions = validators
        state_def.transform_functions = transforms
        state_def.resource_requirements = [
            ResourceConfig(name=r) for r in state_config.resources
        ]
        state_def.data_mode = data_mode
        state_def.type = StateType.START if state_config.is_start else (
            StateType.END if state_config.is_end else StateType.NORMAL
        )
        state_def.metadata = state_config.metadata
        
        return state_def

    def _build_arc(
        self,
        arc_config: ArcConfig,
        source_state: StateDefinition,
        network: StateNetwork,
        fsm_config: FSMConfig,
    ) -> ArcDefinition:
        """Build an arc definition from configuration.
        
        Args:
            arc_config: Arc configuration.
            source_state: Source state definition.
            network: Parent network.
            fsm_config: Parent FSM configuration.
            
        Returns:
            ArcDefinition instance.
        """
        # Resolve condition function
        condition = None
        if arc_config.condition:
            condition = self._resolve_function(arc_config.condition, IStateTestFunction)
        
        # Resolve transform function
        transform = None
        if arc_config.transform:
            transform = self._resolve_function(arc_config.transform, ITransformFunction)
        
        # Create appropriate arc type
        if isinstance(arc_config, PushArcConfig):
            # Push arc to another network
            arc = PushArc(
                target_state=arc_config.target,  # Even push arcs have a target state
                target_network=arc_config.target_network,
                return_state=arc_config.return_state,
                pre_test=condition.name if condition and hasattr(condition, 'name') else None,
                transform=transform.name if transform and hasattr(transform, 'name') else None,
                priority=arc_config.priority,
                metadata=arc_config.metadata,
            )
            # Store resources separately if needed
            arc.required_resources = {r: r for r in arc_config.resources}
            return arc
        else:
            # Regular arc within network
            arc = ArcDefinition(
                target_state=arc_config.target,
                pre_test=condition.name if condition and hasattr(condition, 'name') else None,
                transform=transform.name if transform and hasattr(transform, 'name') else None,
                priority=arc_config.priority,
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
            def __init__(self, schema_def):
                self.schema_def = schema_def
            
            def validate(self, data):
                """Validate data against JSON schema."""
                # Simple validation for basic JSON schema
                if self.schema_def.get('type') == 'object':
                    if not isinstance(data, dict):
                        return type('Result', (), {
                            'valid': False,
                            'errors': [f'Expected object, got {type(data).__name__}']
                        })()
                    
                    errors = []
                    properties = self.schema_def.get('properties', {})
                    required = self.schema_def.get('required', [])
                    
                    # Check required fields
                    for field in required:
                        if field not in data:
                            errors.append(f"Required field '{field}' is missing")
                    
                    # Check field types
                    for field, value in data.items():
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
        expected_type: Optional[Type] = None,
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
            if func_ref.name not in self._builtin_functions:
                raise ValueError(f"Built-in function not found: {func_ref.name}")
            func = self._builtin_functions[func_ref.name]
        
        elif func_ref.type == "custom":
            # Check registry first
            if func_ref.name in self._function_registry:
                func = self._function_registry[func_ref.name]
            else:
                # Import custom function
                module = importlib.import_module(func_ref.module)
                func = getattr(module, func_ref.name)
        
        elif func_ref.type == "inline":
            # Compile inline code
            namespace = {}
            
            # Check if the code is a lambda expression
            code = func_ref.code.strip()
            if code.startswith('lambda'):
                # Wrap lambda in assignment to make it accessible
                exec(f"func = {code}", namespace)
                func = namespace['func']
            else:
                # Execute code and find function
                exec(code, namespace)
                # Find the function in namespace
                funcs = [v for v in namespace.values() if callable(v)]
                if not funcs:
                    raise ValueError("No function found in inline code")
                func = funcs[0]
        
        else:
            raise ValueError(f"Unknown function type: {func_ref.type}")
        
        # Apply parameters if provided
        if func_ref.params:
            # Create a partial function with parameters
            import functools
            func = functools.partial(func, **func_ref.params)
        
        # Validate type if specified
        if expected_type and not isinstance(func, expected_type):
            # Wrap function to match expected interface
            func = self._wrap_function(func, expected_type)
        
        return func

    def _wrap_function(self, func: Callable, interface: Type) -> Any:
        """Wrap a function to match an expected interface.
        
        Args:
            func: Function to wrap.
            interface: Expected interface type.
            
        Returns:
            Wrapped function implementing the interface.
        """
        # Create a wrapper class that implements the interface
        class FunctionWrapper:
            def __init__(self, func):
                self.func = func
            
            def __call__(self, *args, **kwargs):
                return self.func(*args, **kwargs)
        
        # Add interface methods
        if interface == IValidationFunction:
            FunctionWrapper.validate = lambda self, data: self.func(data)
        elif interface == ITransformFunction:
            # Create a state-like object for the lambda that has a 'data' attribute
            def transform_wrapper(self, data, context=None):
                # Create a simple state object with a data attribute
                class State:
                    def __init__(self, data):
                        self.data = data
                state = State(data)
                return self.func(state)
            FunctionWrapper.transform = transform_wrapper
            # Also make it callable directly
            def call_wrapper(self, data, context=None):
                class State:
                    def __init__(self, data):
                        self.data = data
                state = State(data)
                return self.func(state)
            FunctionWrapper.__call__ = call_wrapper
        elif interface == IStateTestFunction:
            FunctionWrapper.test = lambda self, state: self.func(state)
        
        return FunctionWrapper(func)

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
        # Map our DataMode to the execution context's DataMode
        from dataknobs_fsm.core.modes import DataMode as CoreDataMode
        
        # Simple mapping - we'll use SINGLE mode for now
        # This could be enhanced to support batch and stream modes
        return ExecutionContext(
            data_mode=CoreDataMode.SINGLE,
            resources={}  # Resources will be populated during execution
        )


class FSM:
    """Executable FSM instance wrapper for easy use."""

    def __init__(
        self,
        core_fsm: CoreFSM,
        config: FSMConfig,
        resource_manager: ResourceManager,
        transaction_manager: Optional[TransactionManager] = None,
        function_registry: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize FSM wrapper instance.
        
        Args:
            core_fsm: Core FSM instance.
            config: FSM configuration.
            resource_manager: Resource manager.
            transaction_manager: Optional transaction manager.
            function_registry: Optional function registry.
        """
        self.core_fsm = core_fsm
        self.config = config
        self.resource_manager = resource_manager
        self.function_registry = function_registry or {}
        self.transaction_manager = transaction_manager
        
        # Convenience properties
        self.networks = core_fsm.networks
        self.main_network = self.networks.get(core_fsm.main_network) if core_fsm.main_network else None
        self.name = core_fsm.main_network  # ExecutionEngine expects this
        
        # Engine will be created on demand
        self._engine: Optional[ExecutionEngine] = None

    def get_engine(self) -> ExecutionEngine:
        """Get or create the execution engine.
        
        Returns:
            ExecutionEngine instance.
        """
        if self._engine is None:
            # Create engine with the core FSM
            from dataknobs_fsm.execution.engine import TraversalStrategy
            
            # Map config execution strategy to engine strategy
            strategy_map = {
                "depth_first": TraversalStrategy.DEPTH_FIRST,
                "breadth_first": TraversalStrategy.BREADTH_FIRST,
                "resource_optimized": TraversalStrategy.RESOURCE_OPTIMIZED,
                "stream_optimized": TraversalStrategy.STREAM_OPTIMIZED,
            }
            
            strategy = strategy_map.get(
                self.config.execution_strategy.value,
                TraversalStrategy.DEPTH_FIRST
            )
            
            self._engine = ExecutionEngine(
                fsm=self.core_fsm,
                strategy=strategy,
            )
        
        return self._engine

    async def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the FSM with initial data.
        
        This is a simplified API for running the FSM.
        
        Args:
            initial_data: Initial data for execution.
            
        Returns:
            Execution result.
        """
        engine = self.get_engine()
        
        # Create execution context with initial data
        context = ExecutionContext(
            data_mode=self.core_fsm.data_mode,
            transaction_mode=self.core_fsm.transaction_mode,
        )
        
        # The engine would need to be updated to handle this properly
        # For now, this is a placeholder that shows the intended API
        return {"status": "completed", "data": initial_data}

    def validate(self) -> bool:
        """Validate the FSM configuration and structure.
        
        Returns:
            True if valid, False otherwise.
        """
        try:
            # Validate network consistency
            for network in self.networks.values():
                network.validate()
            return True
        except Exception:
            return False
    
    def get_start_state(self, network_name: Optional[str] = None):
        """Get the start state from a network.
        
        Args:
            network_name: Network to get start state from. Uses main network if None.
            
        Returns:
            Start state definition.
        """
        if network_name is None:
            network = self.main_network
        else:
            network = self.networks.get(network_name)
        
        if not network:
            raise ValueError(f"Network '{network_name or 'main'}' not found")
        
        # Find start state in the network
        for state in network.states.values():
            if state.is_start_state():
                return state
        
        raise ValueError(f"No start state found in network '{network.name}'")
    
    def get_state(self, state_name: str, network_name: Optional[str] = None):
        """Get a state definition by name.
        
        Args:
            state_name: Name of the state.
            network_name: Network to search in. Searches all if None.
            
        Returns:
            State definition.
        """
        if network_name:
            network = self.networks.get(network_name)
            if network and state_name in network.states:
                return network.states[state_name]
        else:
            # Search all networks
            for network in self.networks.values():
                if state_name in network.states:
                    return network.states[state_name]
        
        raise ValueError(f"State '{state_name}' not found")