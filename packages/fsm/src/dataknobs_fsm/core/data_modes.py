"""Data mode system for FSM state management.

This module defines the data handling strategies for state instances:
- COPY: Deep copy on state entry, isolated modifications
- REFERENCE: Lazy loading with optimistic locking
- DIRECT: In-place modifications (single-threaded only)
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from threading import Lock
from typing import Any, Dict, TypeVar

T = TypeVar("T")


class DataHandlingMode(Enum):
    """Data handling modes for state instances - defines how data is managed within states."""
    
    COPY = "copy"  # Deep copy on entry, isolated changes
    REFERENCE = "reference"  # Lazy loading with locking
    DIRECT = "direct"  # In-place modifications


class DataHandler(ABC):
    """Abstract base class for data mode handlers."""
    
    def __init__(self, mode: DataHandlingMode):
        """Initialize the data handler.
        
        Args:
            mode: The data mode this handler implements.
        """
        self.mode = mode
    
    @abstractmethod
    def on_entry(self, data: Any) -> Any:
        """Handle data when entering a state.
        
        Args:
            data: The input data to the state.
            
        Returns:
            The data to be used within the state.
        """
        pass
    
    @abstractmethod
    def on_modification(self, data: Any) -> Any:
        """Handle data modification within a state.
        
        Args:
            data: The data being modified.
            
        Returns:
            The modified data.
        """
        pass
    
    @abstractmethod
    def on_exit(self, data: Any, commit: bool = True) -> Any:
        """Handle data when exiting a state.
        
        Args:
            data: The state's data.
            commit: Whether to commit changes.
            
        Returns:
            The final data after exit processing.
        """
        pass
    
    @abstractmethod
    def supports_concurrent_access(self) -> bool:
        """Check if this handler supports concurrent access.
        
        Returns:
            True if concurrent access is supported.
        """
        pass


class CopyModeHandler(DataHandler):
    """Handler for COPY mode - deep copy with isolated modifications."""
    
    def __init__(self):
        """Initialize the COPY mode handler."""
        super().__init__(DataHandlingMode.COPY)
        self._originals: Dict[int, Any] = {}
        self._lock = Lock()
    
    def on_entry(self, data: Any) -> Any:
        """Create a deep copy of the data on state entry.
        
        Args:
            data: The input data to copy.
            
        Returns:
            A deep copy of the data.
        """
        copied = deepcopy(data)
        with self._lock:
            self._originals[id(copied)] = data
        return copied
    
    def on_modification(self, data: Any) -> Any:
        """Allow modifications to the copied data.
        
        Args:
            data: The copied data being modified.
            
        Returns:
            The same data (modifications are isolated).
        """
        return data
    
    def on_exit(self, data: Any, commit: bool = True) -> Any:
        """Handle exit, optionally committing changes.
        
        Args:
            data: The modified copy.
            commit: Whether to return modified or original data.
            
        Returns:
            Modified data if commit=True, original otherwise.
        """
        with self._lock:
            original = self._originals.pop(id(data), None)
        
        if commit:
            return data
        else:
            return original if original is not None else data
    
    def supports_concurrent_access(self) -> bool:
        """COPY mode supports concurrent access.
        
        Returns:
            True, as each state gets its own copy.
        """
        return True


class ReferenceModeHandler(DataHandler):
    """Handler for REFERENCE mode - lazy loading with optimistic locking."""
    
    def __init__(self):
        """Initialize the REFERENCE mode handler."""
        super().__init__(DataHandlingMode.REFERENCE)
        # Don't use WeakValueDictionary as it can't hold dict objects
        self._references: Dict[int, Any] = {}
        self._versions: Dict[int, int] = {}
        self._locks: Dict[int, Lock] = {}
        self._global_lock = Lock()
    
    def on_entry(self, data: Any) -> Any:
        """Store a reference to the data with versioning.
        
        Args:
            data: The input data to reference.
            
        Returns:
            The same data object (by reference).
        """
        data_id = id(data)
        with self._global_lock:
            if data_id not in self._references:
                self._references[data_id] = data
                self._versions[data_id] = 0
                self._locks[data_id] = Lock()
        return data
    
    def on_modification(self, data: Any) -> Any:
        """Track modifications with optimistic locking.
        
        Args:
            data: The data being modified.
            
        Returns:
            The same data object.
        """
        data_id = id(data)
        with self._global_lock:
            if data_id in self._locks:
                lock = self._locks[data_id]
            else:
                lock = Lock()
                self._locks[data_id] = lock
        
        with lock:
            if data_id in self._versions:
                self._versions[data_id] += 1
        
        return data
    
    def on_exit(self, data: Any, commit: bool = True) -> Any:
        """Clean up references on exit.
        
        Args:
            data: The referenced data.
            commit: Whether changes should be kept.
            
        Returns:
            The data object.
        """
        data_id = id(data)
        
        if not commit:
            # In REFERENCE mode, we can't truly rollback in-place changes
            # This would require snapshot/restore functionality
            pass
        
        # Clean up tracking if no longer needed
        with self._global_lock:
            # Clean up if version is 0 (no modifications)
            if data_id in self._versions and self._versions[data_id] == 0:
                self._references.pop(data_id, None)
                self._versions.pop(data_id, None)
                self._locks.pop(data_id, None)
        
        return data
    
    def supports_concurrent_access(self) -> bool:
        """REFERENCE mode supports concurrent access with locking.
        
        Returns:
            True, with optimistic locking for safety.
        """
        return True


class DirectModeHandler(DataHandler):
    """Handler for DIRECT mode - in-place modifications without copying."""
    
    def __init__(self):
        """Initialize the DIRECT mode handler."""
        super().__init__(DataHandlingMode.DIRECT)
        self._active_data: Any | None = None
        self._lock = Lock()
    
    def on_entry(self, data: Any) -> Any:
        """Use data directly without copying.
        
        Args:
            data: The input data to use directly.
            
        Returns:
            The same data object.
        """
        with self._lock:
            if self._active_data is not None:
                raise RuntimeError(
                    "DIRECT mode does not support concurrent access. "
                    "Another state is currently using direct mode."
                )
            self._active_data = data
        return data
    
    def on_modification(self, data: Any) -> Any:
        """Allow direct in-place modifications.
        
        Args:
            data: The data being modified.
            
        Returns:
            The same data object.
        """
        return data
    
    def on_exit(self, data: Any, commit: bool = True) -> Any:
        """Clear the active data reference.
        
        Args:
            data: The data object.
            commit: Whether to keep changes (always True for DIRECT).
            
        Returns:
            The data object.
        """
        with self._lock:
            self._active_data = None
        
        # In DIRECT mode, changes are always committed (in-place)
        return data
    
    def supports_concurrent_access(self) -> bool:
        """DIRECT mode does not support concurrent access.
        
        Returns:
            False, as only one state can use DIRECT mode at a time.
        """
        return False


class DataModeManager:
    """Manager for data mode handlers."""
    
    def __init__(self, default_mode: DataHandlingMode = DataHandlingMode.COPY):
        """Initialize the data mode manager.
        
        Args:
            default_mode: The default data mode to use.
        """
        self.default_mode = default_mode
        self._handlers: Dict[DataHandlingMode, DataHandler] = {
            DataHandlingMode.COPY: CopyModeHandler(),
            DataHandlingMode.REFERENCE: ReferenceModeHandler(),
            DataHandlingMode.DIRECT: DirectModeHandler(),
        }
    
    def get_handler(self, mode: DataHandlingMode | None = None) -> DataHandler:
        """Get a handler for the specified mode.
        
        Args:
            mode: The data mode, or None for default.
            
        Returns:
            The appropriate data handler.
        """
        if mode is None:
            mode = self.default_mode
        return self._handlers[mode]
    
    def set_default_mode(self, mode: DataHandlingMode) -> None:
        """Set the default data mode.
        
        Args:
            mode: The new default data mode.
        """
        self.default_mode = mode


# Global registry of data handlers
_GLOBAL_HANDLERS = {
    DataHandlingMode.COPY: CopyModeHandler(),
    DataHandlingMode.REFERENCE: ReferenceModeHandler(),
    DataHandlingMode.DIRECT: DirectModeHandler(),
}


def get_data_handler(mode: DataHandlingMode) -> DataHandler:
    """Get a data handler for the specified mode.
    
    Args:
        mode: The data mode.
        
    Returns:
        The appropriate data handler.
    """
    return _GLOBAL_HANDLERS[mode]
