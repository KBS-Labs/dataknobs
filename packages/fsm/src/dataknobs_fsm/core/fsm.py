"""Core FSM class for managing state machines."""

from typing import Any, Dict, List, Optional, Set, Tuple

from dataknobs_fsm.core.modes import ProcessingMode, TransactionMode
from dataknobs_fsm.core.network import StateNetwork
from dataknobs_fsm.functions.base import FunctionRegistry


class FSM:
    """Finite State Machine core class.
    
    This class manages:
    - Multiple state networks
    - Function registry
    - Data and transaction modes
    - Resource requirements
    - Configuration
    """
    
    def __init__(
        self,
        name: str,
        data_mode: ProcessingMode = ProcessingMode.SINGLE,
        transaction_mode: TransactionMode = TransactionMode.NONE,
        description: Optional[str] = None
    ):
        """Initialize FSM.
        
        Args:
            name: Name of the FSM.
            data_mode: Data processing mode.
            transaction_mode: Transaction handling mode.
            description: Optional FSM description.
        """
        self.name = name
        self.data_mode = data_mode
        self.transaction_mode = transaction_mode
        self.description = description
        
        # Networks
        self.networks: Dict[str, StateNetwork] = {}
        self.main_network: Optional[str] = None
        
        # Function registry
        self.function_registry = FunctionRegistry()
        
        # Resource requirements
        self.resource_requirements: Dict[str, Any] = {}
        
        # Configuration
        self.config: Dict[str, Any] = {}
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
        self.version: str = "1.0.0"
        self.created_at: Optional[float] = None
        self.updated_at: Optional[float] = None
    
    def add_network(
        self,
        network: StateNetwork,
        is_main: bool = False
    ) -> None:
        """Add a network to the FSM.
        
        Args:
            network: Network to add.
            is_main: Whether this is the main network.
        """
        self.networks[network.name] = network
        
        if is_main or self.main_network is None:
            self.main_network = network.name
        
        # Aggregate resource requirements
        for resource_type, requirements in network.resource_requirements.items():
            if resource_type not in self.resource_requirements:
                self.resource_requirements[resource_type] = set()
            self.resource_requirements[resource_type].update(requirements)
    
    def remove_network(self, network_name: str) -> bool:
        """Remove a network from the FSM.
        
        Args:
            network_name: Name of network to remove.
            
        Returns:
            True if removed successfully.
        """
        if network_name in self.networks:
            del self.networks[network_name]
            
            # Update main network if needed
            if self.main_network == network_name:
                if self.networks:
                    self.main_network = next(iter(self.networks.keys()))
                else:
                    self.main_network = None
            
            return True
        return False
    
    def get_network(self, network_name: Optional[str] = None) -> Optional[StateNetwork]:
        """Get a network by name.
        
        Args:
            network_name: Name of network (None for main network).
            
        Returns:
            Network or None if not found.
        """
        if network_name is None:
            network_name = self.main_network
        
        if network_name:
            return self.networks.get(network_name)
        return None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the FSM.
        
        Returns:
            Tuple of (valid, list of errors).
        """
        errors = []
        
        # Check for at least one network
        if not self.networks:
            errors.append("FSM has no networks")
        
        # Check main network exists
        if self.main_network and self.main_network not in self.networks:
            errors.append(f"Main network '{self.main_network}' not found")
        
        # Validate each network
        for network_name, network in self.networks.items():
            valid, network_errors = network.validate()
            if not valid:
                for error in network_errors:
                    errors.append(f"Network '{network_name}': {error}")
        
        # Check function references
        all_functions = self._get_all_function_references()
        for func_name in all_functions:
            if not self.function_registry.get_function(func_name):
                errors.append(f"Function '{func_name}' not registered")
        
        return len(errors) == 0, errors
    
    def _get_all_function_references(self) -> Set[str]:
        """Get all function references from all networks.
        
        Returns:
            Set of function names referenced.
        """
        functions = set()
        
        for network in self.networks.values():
            for arc in network.arcs.values():
                if hasattr(arc, 'pre_test') and arc.pre_test:
                    functions.add(arc.pre_test)
                if hasattr(arc, 'transform') and arc.transform:
                    functions.add(arc.transform)
        
        return functions
    
    def get_all_states(self) -> Dict[str, List[str]]:
        """Get all states from all networks.
        
        Returns:
            Dictionary of network_name -> list of state names.
        """
        all_states = {}
        
        for network_name, network in self.networks.items():
            all_states[network_name] = list(network.states.keys())
        
        return all_states
    
    def get_all_arcs(self) -> Dict[str, List[str]]:
        """Get all arcs from all networks.
        
        Returns:
            Dictionary of network_name -> list of arc IDs.
        """
        all_arcs = {}
        
        for network_name, network in self.networks.items():
            all_arcs[network_name] = list(network.arcs.keys())
        
        return all_arcs
    
    def supports_streaming(self) -> bool:
        """Check if FSM supports streaming.
        
        Returns:
            True if any network supports streaming.
        """
        return any(network.supports_streaming for network in self.networks.values())
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource requirements summary.
        
        Returns:
            Resource requirements summary.
        """
        summary = {
            'total_networks': len(self.networks),
            'total_states': sum(len(n.states) for n in self.networks.values()),
            'total_arcs': sum(len(n.arcs) for n in self.networks.values()),
            'resource_types': list(self.resource_requirements.keys()),
            'supports_streaming': self.supports_streaming(),
            'data_mode': self.data_mode.value,
            'transaction_mode': self.transaction_mode.value
        }
        
        # Add resource counts
        for resource_type, requirements in self.resource_requirements.items():
            summary[f'{resource_type}_count'] = len(requirements)
        
        return summary
    
    def clone(self) -> 'FSM':
        """Create a clone of this FSM.
        
        Returns:
            Cloned FSM.
        """
        clone = FSM(
            name=f"{self.name}_clone",
            data_mode=self.data_mode,
            transaction_mode=self.transaction_mode,
            description=self.description
        )
        
        # Clone networks
        for network_name, network in self.networks.items():
            # Note: This is a shallow copy - for deep clone would need to implement network.clone()
            clone.networks[network_name] = network
        
        clone.main_network = self.main_network
        clone.function_registry = self.function_registry
        clone.resource_requirements = self.resource_requirements.copy()
        clone.config = self.config.copy()
        clone.metadata = self.metadata.copy()
        clone.version = self.version
        
        return clone
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert FSM to dictionary representation.
        
        Returns:
            Dictionary representation.
        """
        return {
            'name': self.name,
            'description': self.description,
            'data_mode': self.data_mode.value,
            'transaction_mode': self.transaction_mode.value,
            'main_network': self.main_network,
            'networks': list(self.networks.keys()),
            'resource_requirements': {
                k: list(v) if isinstance(v, set) else v
                for k, v in self.resource_requirements.items()
            },
            'config': self.config,
            'metadata': self.metadata,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FSM':
        """Create FSM from dictionary representation.
        
        Args:
            data: Dictionary with FSM data.
            
        Returns:
            New FSM instance.
        """
        fsm = cls(
            name=data['name'],
            data_mode=ProcessingMode(data.get('data_mode', 'single')),
            transaction_mode=TransactionMode(data.get('transaction_mode', 'none')),
            description=data.get('description')
        )
        
        fsm.main_network = data.get('main_network')
        fsm.config = data.get('config', {})
        fsm.metadata = data.get('metadata', {})
        fsm.version = data.get('version', '1.0.0')
        
        # Resource requirements
        for resource_type, requirements in data.get('resource_requirements', {}).items():
            fsm.resource_requirements[resource_type] = set(requirements)
        
        return fsm