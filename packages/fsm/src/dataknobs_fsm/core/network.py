"""State network implementation for FSM."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from dataknobs_fsm.core.state import State


@dataclass
class Arc:
    """Represents an arc (transition) between states.
    
    Attributes:
        source_state: Name of the source state.
        target_state: Name of the target state.
        pre_test: Optional pre-test function name.
        transform: Optional transform function name.
        metadata: Additional arc metadata.
    """
    source_state: str
    target_state: str
    pre_test: str | None = None
    transform: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        """Make Arc hashable for use in sets."""
        return hash((self.source_state, self.target_state, self.pre_test, self.transform))
    
    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Arc):
            return False
        return (
            self.source_state == other.source_state and
            self.target_state == other.target_state and
            self.pre_test == other.pre_test and
            self.transform == other.transform
        )
    
    @property
    def name(self) -> str:
        """Generate a name for the arc."""
        # Use metadata name if available, otherwise generate from states
        if 'name' in self.metadata:
            return self.metadata['name']
        return f"{self.source_state}->{self.target_state}"


@dataclass
class NetworkResourceRequirements:
    """Aggregated resource requirements for a network.
    
    Attributes:
        databases: Set of required database resources.
        filesystems: Set of required filesystem resources.
        http_services: Set of required HTTP service resources.
        llms: Set of required LLM resources.
        custom: Dictionary of custom resource requirements.
        streaming_enabled: Whether any state requires streaming.
        estimated_memory_mb: Estimated memory requirement in MB.
    """
    databases: Set[str] = field(default_factory=set)
    filesystems: Set[str] = field(default_factory=set)
    http_services: Set[str] = field(default_factory=set)
    llms: Set[str] = field(default_factory=set)
    custom: Dict[str, Set[str]] = field(default_factory=dict)
    streaming_enabled: bool = False
    estimated_memory_mb: int = 0
    
    def merge(self, other: "NetworkResourceRequirements") -> None:
        """Merge another set of requirements into this one.
        
        Args:
            other: Requirements to merge.
        """
        self.databases.update(other.databases)
        self.filesystems.update(other.filesystems)
        self.http_services.update(other.http_services)
        self.llms.update(other.llms)
        
        for key, values in other.custom.items():
            if key not in self.custom:
                self.custom[key] = set()
            self.custom[key].update(values)
        
        self.streaming_enabled = self.streaming_enabled or other.streaming_enabled
        self.estimated_memory_mb = max(self.estimated_memory_mb, other.estimated_memory_mb)
    
    def is_empty(self) -> bool:
        """Check if there are no resource requirements.
        
        Returns:
            True if no resources are required.
        """
        return (
            not self.databases and
            not self.filesystems and
            not self.http_services and
            not self.llms and
            not self.custom
        )


class StateNetwork:
    """Represents a network of states and their transitions.
    
    A state network is a directed graph where nodes are states
    and edges are arcs (transitions) between states.
    """
    
    def __init__(self, name: str, description: str | None = None):
        """Initialize state network.
        
        Args:
            name: Network name/identifier.
            description: Optional network description.
        """
        self.name = name
        self.description = description
        
        # State management
        self._states: Dict[str, State] = {}
        self._initial_state: str | None = None
        self._final_states: Set[str] = set()
        
        # Arc management
        self._arcs: List[Arc] = []
        self._arc_index: Dict[str, List[Arc]] = {}  # source_state -> [arcs]
        
        # Resource tracking
        self._resource_requirements = NetworkResourceRequirements()
        self._streaming_enabled = False
        
        # Validation cache
        self._validation_cache: Dict[str, Any] | None = None
    
    @property
    def states(self) -> Dict[str, State]:
        """Get all states in the network."""
        return self._states
    
    @property
    def arcs(self) -> Dict[str, Any]:
        """Get all arcs in the network."""
        # Import here to avoid circular dependency
        from dataknobs_fsm.core.arc import ArcDefinition
        
        # Return arcs as a dict indexed by "source:target" 
        # Convert Arc to ArcDefinition for compatibility
        arc_dict = {}
        for arc in self._arcs:
            key = f"{arc.source_state}:{arc.target_state}"
            # Create ArcDefinition from Arc
            arc_def = ArcDefinition(
                target_state=arc.target_state,
                pre_test=arc.pre_test,
                transform=arc.transform
            )
            # Copy metadata if it exists
            if hasattr(arc, 'metadata') and arc.metadata:
                arc_def.metadata = arc.metadata.copy()
            arc_dict[key] = arc_def
        return arc_dict
    
    @property
    def initial_states(self) -> Set[str]:
        """Get initial states (returns set for compatibility)."""
        if self._initial_state:
            return {self._initial_state}
        return set()
    
    @property
    def final_states(self) -> Set[str]:
        """Get final states."""
        return self._final_states.copy()
    
    def is_initial_state(self, state_name: str) -> bool:
        """Check if a state is an initial state.
        
        Args:
            state_name: Name of the state to check
            
        Returns:
            True if the state is an initial state
        """
        return self._initial_state == state_name
    
    def is_final_state(self, state_name: str) -> bool:
        """Check if a state is a final state.
        
        Args:
            state_name: Name of the state to check
            
        Returns:
            True if the state is a final state
        """
        return state_name in self._final_states
    
    @property
    def resource_requirements(self) -> Dict[str, Any]:
        """Get resource requirements."""
        return {
            'databases': self._resource_requirements.databases,
            'filesystems': self._resource_requirements.filesystems,
            'http_services': self._resource_requirements.http_services,
            'llms': self._resource_requirements.llms,
            'custom': self._resource_requirements.custom
        }
    
    @property
    def supports_streaming(self) -> bool:
        """Check if network supports streaming."""
        return self._streaming_enabled
    
    def add_state(
        self,
        state: State,
        initial: bool = False,
        final: bool = False
    ) -> None:
        """Add a state to the network.
        
        Args:
            state: State to add.
            initial: Mark as initial state.
            final: Mark as final state.
            
        Raises:
            ValueError: If state with same name already exists.
        """
        if state.name in self._states:
            raise ValueError(f"State '{state.name}' already exists in network")
        
        self._states[state.name] = state
        
        if initial:
            if self._initial_state:
                raise ValueError(
                    f"Initial state already set to '{self._initial_state}'"
                )
            self._initial_state = state.name
        
        if final:
            self._final_states.add(state.name)
        
        # Update resource requirements
        self._update_resource_requirements(state)
        
        # Invalidate validation cache
        self._validation_cache = None
    
    def remove_state(self, state_name: str) -> None:
        """Remove a state from the network.
        
        Args:
            state_name: Name of state to remove.
            
        Raises:
            KeyError: If state doesn't exist.
        """
        if state_name not in self._states:
            raise KeyError(f"State '{state_name}' not found in network")
        
        # Remove state
        del self._states[state_name]
        
        # Remove from initial/final if needed
        if self._initial_state == state_name:
            self._initial_state = None
        self._final_states.discard(state_name)
        
        # Remove arcs involving this state
        self._arcs = [
            arc for arc in self._arcs
            if arc.source_state != state_name and arc.target_state != state_name
        ]
        
        # Rebuild arc index
        self._rebuild_arc_index()
        
        # Recalculate resource requirements
        self._recalculate_resource_requirements()
        
        # Invalidate validation cache
        self._validation_cache = None
    
    def add_arc(
        self,
        source_state: str,
        target_state: str,
        pre_test: str | None = None,
        transform: str | None = None,
        metadata: Dict[str, Any] | None = None
    ) -> Arc:
        """Add an arc between two states.
        
        Args:
            source_state: Source state name.
            target_state: Target state name.
            pre_test: Optional pre-test function name.
            transform: Optional transform function name.
            metadata: Optional arc metadata.
            
        Returns:
            Created arc.
            
        Raises:
            ValueError: If states don't exist.
        """
        if source_state not in self._states:
            raise ValueError(f"Source state '{source_state}' not found")
        if target_state not in self._states:
            raise ValueError(f"Target state '{target_state}' not found")
        
        arc = Arc(
            source_state=source_state,
            target_state=target_state,
            pre_test=pre_test,
            transform=transform,
            metadata=metadata or {}
        )
        
        self._arcs.append(arc)
        
        # Update arc index
        if source_state not in self._arc_index:
            self._arc_index[source_state] = []
        self._arc_index[source_state].append(arc)
        
        # Invalidate validation cache
        self._validation_cache = None
        
        return arc
    
    def remove_arc(self, arc: Arc) -> None:
        """Remove an arc from the network.
        
        Args:
            arc: Arc to remove.
            
        Raises:
            ValueError: If arc doesn't exist.
        """
        if arc not in self._arcs:
            raise ValueError("Arc not found in network")
        
        self._arcs.remove(arc)
        
        # Update arc index
        if arc.source_state in self._arc_index:
            self._arc_index[arc.source_state].remove(arc)
            if not self._arc_index[arc.source_state]:
                del self._arc_index[arc.source_state]
        
        # Invalidate validation cache
        self._validation_cache = None
    
    def get_state(self, name: str) -> State | None:
        """Get a state by name.
        
        Args:
            name: State name.
            
        Returns:
            State if found, None otherwise.
        """
        return self._states.get(name)
    
    def get_arcs_from_state(self, state_name: str) -> List[Arc]:
        """Get all arcs originating from a state.
        
        Args:
            state_name: Source state name.
            
        Returns:
            List of arcs from the state.
        """
        return self._arc_index.get(state_name, [])
    
    def get_arcs_to_state(self, state_name: str) -> List[Arc]:
        """Get all arcs targeting a state.
        
        Args:
            state_name: Target state name.
            
        Returns:
            List of arcs to the state.
        """
        return [arc for arc in self._arcs if arc.target_state == state_name]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate network consistency.
        
        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        # Return cached result if available
        if self._validation_cache is not None:
            return self._validation_cache['is_valid'], self._validation_cache['errors']
        
        errors = []
        
        # Check for initial state
        if not self._initial_state:
            errors.append("No initial state defined")
        elif self._initial_state not in self._states:
            errors.append(f"Initial state '{self._initial_state}' not found")
        
        # Check for final states
        if not self._final_states:
            errors.append("No final states defined")
        else:
            for final_state in self._final_states:
                if final_state not in self._states:
                    errors.append(f"Final state '{final_state}' not found")
        
        # Check for unreachable states
        if self._initial_state:
            reachable = self._find_reachable_states(self._initial_state)
            unreachable = set(self._states.keys()) - reachable
            for state in unreachable:
                errors.append(f"State '{state}' is unreachable from initial state")
        
        # Check for states with no outgoing arcs (except final states)
        for state_name in self._states:
            if state_name not in self._final_states:
                if state_name not in self._arc_index or not self._arc_index[state_name]:
                    errors.append(f"Non-final state '{state_name}' has no outgoing arcs")
        
        # Check for cycles that don't include final states
        cycles = self._find_cycles()
        for cycle in cycles:
            if not any(state in self._final_states for state in cycle):
                errors.append(
                    f"Cycle detected without final states: {' -> '.join(cycle)}"
                )
        
        # Cache validation result
        is_valid = len(errors) == 0
        self._validation_cache = {
            'is_valid': is_valid,
            'errors': errors
        }
        
        return is_valid, errors
    
    def get_resource_requirements(self) -> NetworkResourceRequirements:
        """Get aggregated resource requirements for the network.
        
        Returns:
            Resource requirements.
        """
        return self._resource_requirements
    
    def is_streaming_enabled(self) -> bool:
        """Check if any state in the network requires streaming.
        
        Returns:
            True if streaming is required.
        """
        return self._streaming_enabled
    
    def analyze_dependencies(self) -> Dict[str, Set[str]]:
        """Analyze resource dependencies between states.
        
        Returns:
            Dictionary mapping resources to dependent states.
        """
        dependencies = {}
        
        for state_name, state in self._states.items():
            if hasattr(state, 'resource_requirements'):
                for resource in state.resource_requirements:
                    if resource not in dependencies:
                        dependencies[resource] = set()
                    dependencies[resource].add(state_name)
        
        return dependencies
    
    def _update_resource_requirements(self, state: State) -> None:
        """Update resource requirements based on a state.
        
        Args:
            state: State to analyze.
        """
        if hasattr(state, 'resource_requirements'):
            reqs = state.resource_requirements
            
            # Update resource sets based on type
            if hasattr(reqs, 'databases'):
                self._resource_requirements.databases.update(reqs.databases)
            if hasattr(reqs, 'filesystems'):
                self._resource_requirements.filesystems.update(reqs.filesystems)
            if hasattr(reqs, 'http_services'):
                self._resource_requirements.http_services.update(reqs.http_services)
            if hasattr(reqs, 'llms'):
                self._resource_requirements.llms.update(reqs.llms)
            
            # Update streaming flag
            if hasattr(reqs, 'streaming_enabled'):
                self._streaming_enabled = self._streaming_enabled or reqs.streaming_enabled
    
    def _recalculate_resource_requirements(self) -> None:
        """Recalculate all resource requirements from scratch."""
        self._resource_requirements = NetworkResourceRequirements()
        self._streaming_enabled = False
        
        for state in self._states.values():
            self._update_resource_requirements(state)
    
    def _rebuild_arc_index(self) -> None:
        """Rebuild the arc index from scratch."""
        self._arc_index = {}
        for arc in self._arcs:
            if arc.source_state not in self._arc_index:
                self._arc_index[arc.source_state] = []
            self._arc_index[arc.source_state].append(arc)
    
    def _find_reachable_states(self, start_state: str) -> Set[str]:
        """Find all states reachable from a given state.
        
        Args:
            start_state: Starting state name.
            
        Returns:
            Set of reachable state names.
        """
        reachable = set()
        to_visit = [start_state]
        
        while to_visit:
            current = to_visit.pop()
            if current in reachable:
                continue
            
            reachable.add(current)
            
            # Add target states of outgoing arcs
            for arc in self.get_arcs_from_state(current):
                if arc.target_state not in reachable:
                    to_visit.append(arc.target_state)
        
        return reachable
    
    def _find_cycles(self) -> List[List[str]]:
        """Find all cycles in the network.
        
        Returns:
            List of cycles (each cycle is a list of state names).
        """
        cycles = []
        visited = set()
        rec_stack = []
        
        def dfs(state: str) -> None:
            visited.add(state)
            rec_stack.append(state)
            
            for arc in self.get_arcs_from_state(state):
                if arc.target_state not in visited:
                    dfs(arc.target_state)
                elif arc.target_state in rec_stack:
                    # Found a cycle
                    cycle_start = rec_stack.index(arc.target_state)
                    cycle = rec_stack[cycle_start:] + [arc.target_state]
                    cycles.append(cycle)
            
            rec_stack.pop()
        
        # Start DFS from all unvisited states
        for state_name in self._states:
            if state_name not in visited:
                dfs(state_name)
        
        return cycles
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert network to dictionary representation.
        
        Returns:
            Dictionary representation.
        """
        return {
            'name': self.name,
            'description': self.description,
            'initial_state': self._initial_state,
            'final_states': list(self._final_states),
            'states': {
                name: state.to_dict() if hasattr(state, 'to_dict') else str(state)
                for name, state in self._states.items()
            },
            'arcs': [
                {
                    'source': arc.source_state,
                    'target': arc.target_state,
                    'pre_test': arc.pre_test,
                    'transform': arc.transform,
                    'metadata': arc.metadata
                }
                for arc in self._arcs
            ],
            'resource_requirements': {
                'databases': list(self._resource_requirements.databases),
                'filesystems': list(self._resource_requirements.filesystems),
                'http_services': list(self._resource_requirements.http_services),
                'llms': list(self._resource_requirements.llms),
                'custom': {
                    k: list(v) for k, v in self._resource_requirements.custom.items()
                },
                'streaming_enabled': self._resource_requirements.streaming_enabled,
                'estimated_memory_mb': self._resource_requirements.estimated_memory_mb
            },
            'streaming_enabled': self._streaming_enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateNetwork":
        """Create network from dictionary representation.
        
        Args:
            data: Dictionary representation.
            
        Returns:
            StateNetwork instance.
        """
        network = cls(
            name=data['name'],
            description=data.get('description')
        )
        
        # Add states
        for state_name in data.get('states', {}):
            # Create basic state (can be enhanced with proper State deserialization)
            state = State(name=state_name)
            is_initial = state_name == data.get('initial_state')
            is_final = state_name in data.get('final_states', [])
            network.add_state(state, initial=is_initial, final=is_final)
        
        # Add arcs
        for arc_data in data.get('arcs', []):
            network.add_arc(
                source_state=arc_data['source'],
                target_state=arc_data['target'],
                pre_test=arc_data.get('pre_test'),
                transform=arc_data.get('transform'),
                metadata=arc_data.get('metadata', {})
            )
        
        return network
