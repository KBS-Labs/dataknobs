"""Execution context for FSM state machines."""

from typing import Any, Dict, List, Optional, Tuple


class ExecutionContext:
    """Execution context for FSM processing.
    
    This is a stub implementation that will be fully developed in Phase 4.
    """
    
    def __init__(self):
        """Initialize execution context."""
        self.current_state: Optional[str] = None
        self.network_stack: List[Tuple[str, Optional[str]]] = []
        self.data: Any = None
        self.metadata: Dict[str, Any] = {}
    
    def push_network(self, network_name: str, return_state: Optional[str] = None) -> None:
        """Push a network onto the execution stack.
        
        Args:
            network_name: Name of network to push.
            return_state: State to return to after network completes.
        """
        self.network_stack.append((network_name, return_state))
    
    def pop_network(self) -> Tuple[str, Optional[str]]:
        """Pop a network from the execution stack.
        
        Returns:
            Tuple of (network_name, return_state).
        """
        if self.network_stack:
            return self.network_stack.pop()
        return ("", None)