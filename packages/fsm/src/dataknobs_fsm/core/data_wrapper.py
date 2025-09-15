"""Data wrapper for FSM that provides a consistent interface for data access.

This module implements a hybrid solution for data handling in the FSM:
- User functions receive raw dict data by default (simple, predictable)
- Optionally, users can work with FSMData wrapper for enhanced functionality
- Internal FSM operations use the wrapper for consistency
"""

from typing import Any, Dict, Union, Iterator, KeysView, ValuesView, ItemsView
from collections.abc import MutableMapping
import copy


class FSMData(MutableMapping):
    """A data wrapper that supports both dict-style and attribute access.

    This class provides:
    1. Dict-style access: data['key']
    2. Attribute access: data.key
    3. Compatibility with existing functions expecting either pattern
    4. Transparent conversion to/from dict

    The FSM internally uses this wrapper but always passes raw dict data
    to user functions unless they explicitly request the wrapper.
    """

    # Explicitly mark as unhashable (mutable mapping)
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, data: Dict[str, Any] | None = None):
        """Initialize FSMData wrapper.

        Args:
            data: Initial data dictionary. Defaults to empty dict.
        """
        # Store data in __dict__ to avoid recursion with __getattr__
        object.__setattr__(self, '_data', data if data is not None else {})

    # Dict-style access methods
    def __getitem__(self, key: str) -> Any:
        """Get item using dict-style access."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item using dict-style access."""
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete item using dict-style access."""
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in data."""
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Get number of items."""
        return len(self._data)

    # Attribute-style access methods
    def __getattr__(self, name: str) -> Any:
        """Get attribute using dot notation."""
        if name.startswith('_'):
            # Don't intercept private attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute using dot notation."""
        if name.startswith('_'):
            # Store private attributes normally
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete attribute using dot notation."""
        if name.startswith('_'):
            object.__delattr__(self, name)
        else:
            try:
                del self._data[name]
            except KeyError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None

    # Dict-like methods
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return self._data.get(key, default)

    def keys(self) -> KeysView[str]:
        """Get keys view."""
        return self._data.keys()

    def values(self) -> ValuesView[Any]:
        """Get values view."""
        return self._data.values()

    def items(self) -> ItemsView[str, Any]:
        """Get items view."""
        return self._data.items()

    def update(self, other: Union[Dict[str, Any], 'FSMData'] = None, **kwargs) -> None:
        """Update data from dict or another FSMData."""
        if other is not None:
            if isinstance(other, FSMData):
                self._data.update(other._data)
            else:
                self._data.update(other)
        self._data.update(kwargs)

    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()

    def copy(self) -> 'FSMData':
        """Create a shallow copy."""
        return FSMData(self._data.copy())

    def deepcopy(self) -> 'FSMData':
        """Create a deep copy."""
        return FSMData(copy.deepcopy(self._data))

    def pop(self, key: str, default: Any = None) -> Any:
        """Remove and return value."""
        return self._data.pop(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set default value if key doesn't exist."""
        return self._data.setdefault(key, default)

    # Conversion methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary.

        Returns:
            The underlying data dictionary.
        """
        return self._data

    def __json__(self) -> Dict[str, Any]:
        """Support JSON serialization via json.dumps with default handler.

        Returns:
            The underlying data dictionary for JSON serialization.
        """
        return self._data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FSMData':
        """Create from dictionary.

        Args:
            data: Dictionary to wrap.

        Returns:
            New FSMData instance.
        """
        return cls(data)

    # Special methods for compatibility
    def __repr__(self) -> str:
        """String representation."""
        return f"FSMData({self._data!r})"

    def __str__(self) -> str:
        """String conversion."""
        return str(self._data)

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if isinstance(other, FSMData):
            return self._data == other._data
        elif isinstance(other, dict):
            return self._data == other
        return False

    def __bool__(self) -> bool:
        """Boolean conversion."""
        return bool(self._data)


class StateDataWrapper:
    """Wrapper for state data that provides backward compatibility.

    This wrapper is used for inline lambda functions that expect
    `state.data` access pattern. It wraps the FSMData to provide
    the expected interface.
    """

    data: Dict[str, Any]  # Always stores the raw dict
    _fsm_data: FSMData  # The FSMData wrapper

    def __init__(self, data: Union[Dict[str, Any], FSMData, Any] = None):
        """Initialize state wrapper.

        Args:
            data: Data to wrap (dict or FSMData).
        """
        # Always expose the underlying dict for lambdas
        if isinstance(data, FSMData):
            self.data = data._data  # Expose raw dict
            self._fsm_data = data
        elif isinstance(data, dict):
            self.data = data  # Expose raw dict
            self._fsm_data = FSMData(data)
        else:
            # Convert to dict
            data_dict = dict(data) if data else {}
            self.data = data_dict
            self._fsm_data = FSMData(data_dict)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to data."""
        return getattr(self.data, name)

    def __getitem__(self, key: str) -> Any:
        """Forward dict-style access to data."""
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Forward dict-style setting to data."""
        self.data[key] = value


def ensure_dict(data: Union[Dict[str, Any], FSMData, StateDataWrapper, Any]) -> Dict[str, Any]:
    """Ensure data is a plain dictionary.

    This utility function converts various data types to a plain dict,
    which is what user functions expect to receive.

    Args:
        data: Data in any supported format.

    Returns:
        Plain dictionary.
    """
    if isinstance(data, dict):
        return data
    elif isinstance(data, FSMData):
        return data.to_dict()
    elif isinstance(data, StateDataWrapper):
        return data.data.to_dict()
    elif hasattr(data, '_data'):
        # Handle other wrapper types
        return data._data
    elif hasattr(data, 'data'):
        # Handle objects with data attribute
        inner = data.data
        if isinstance(inner, dict):
            return inner
        elif isinstance(inner, FSMData):
            return inner.to_dict()
        elif hasattr(inner, '_data'):
            return inner._data
    # Last resort - try to convert
    return dict(data) if data else {}


def wrap_for_lambda(data: Union[Dict[str, Any], FSMData]) -> StateDataWrapper:
    """Wrap data for inline lambda functions.

    This creates a wrapper that provides the `state.data` access pattern
    expected by inline lambda functions in the FSM configuration.

    Args:
        data: Data to wrap.

    Returns:
        StateDataWrapper instance.
    """
    return StateDataWrapper(data)
