# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

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

    #: The record itself. Declared rather than only assigned: it is set
    #: through ``object.__setattr__`` to dodge ``__setattr__``, and with
    #: ``__getattr__`` in the class an undeclared attribute reads as ``Any``,
    #: which made every accessor below return ``Any`` from a typed signature.
    _data: Dict[str, Any]

    # Explicitly mark as unhashable (mutable mapping)
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, data: Dict[str, Any] | None = None):
        """Initialize FSMData wrapper.

        Args:
            data: Initial data dictionary. Defaults to empty dict.
        """
        # Store data in __dict__ to avoid recursion with __getattr__
        object.__setattr__(self, "_data", data if data is not None else {})

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

    def __contains__(self, key: object) -> bool:
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
        if name.startswith("_"):
            # Don't intercept private attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute using dot notation."""
        if name.startswith("_"):
            # Store private attributes normally
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __delattr__(self, name: str) -> None:
        """Delete attribute using dot notation."""
        if name.startswith("_"):
            object.__delattr__(self, name)
        else:
            try:
                del self._data[name]
            except KeyError:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                ) from None

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

    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()

    def copy(self) -> "FSMData":
        """Create a shallow copy."""
        return FSMData(self._data.copy())

    def deepcopy(self) -> "FSMData":
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
    def from_dict(cls, data: Dict[str, Any]) -> "FSMData":
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


class StateDataWrapper(MutableMapping):
    """The record, as an inline one-argument function receives it.

    ``wrap_for_lambda`` builds one of these for every one-argument *plain*
    callable an FSM reaches --- a registered function, a registered gate, an
    inline ``lambda state: ...``. The documented way to read it is
    ``state.data[...]``, and :attr:`data` is the raw record itself rather than
    a copy, so a transform that mutates it in place mutates the record.

    **It is also a mapping in its own right**, because it is handed to
    functions as the thing they were given and they read it as one.
    ``__getitem__`` and ``__setitem__`` alone were not enough for that:
    Python answers ``in`` and iteration from the *type*, so with no
    ``__contains__`` the ``in`` operator fell back to the old integer-indexing
    protocol and ``'field' in state`` raised ``KeyError: 0`` --- naming a key
    the caller never wrote. Declaring ``MutableMapping`` and its five methods
    answers ``in``, ``len``, iteration, ``bool``, ``==`` and ``dict(state)``
    together, and matches :class:`FSMData`, which has been a ``MutableMapping``
    since the commit that made this class' ``.data`` a raw dict.

    Attribute access still reaches the *record object's* attributes, not its
    keys: ``state.get(...)`` and ``state.keys()`` are the dict's, and
    ``state.field`` is not a way to read a field. ``state.data[...]`` is.
    """

    data: Dict[str, Any]  # Always the raw record, never a wrapper

    def __init__(self, data: Union[Dict[str, Any], FSMData, Any] = None):
        """Initialize state wrapper.

        Args:
            data: Data to wrap (dict or FSMData).
        """
        # Always expose the underlying dict for lambdas
        if isinstance(data, FSMData):
            self.data = data.to_dict()  # Expose raw dict
        elif isinstance(data, dict):
            self.data = data  # Expose raw dict
        else:
            # Convert to dict
            self.data = dict(data) if data else {}

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the record object.

        ``data`` and private names are refused here rather than forwarded.
        ``__getattr__`` runs only for a name ordinary lookup did not find, so
        forwarding ``data`` asked ``self.data`` for ``data`` on an instance
        whose ``data`` is not set yet --- which is every half-built instance
        the copy and pickle protocols probe, and the reason ``copy.copy`` and
        ``copy.deepcopy`` both died of ``RecursionError``.
        """
        if name == "data" or name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return getattr(self.data, name)
        except AttributeError:
            # Say which object was asked. The forwarded error named ``dict``,
            # which points at neither this wrapper nor the record's keys.
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'; "
                f"read a field as state.data[{name!r}] or state[{name!r}]"
            ) from None

    # -- the record, as a mapping ------------------------------------------ #

    def __getitem__(self, key: str) -> Any:
        """Forward dict-style access to data."""
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Forward dict-style setting to data."""
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        """Forward dict-style deletion to data."""
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the record's keys."""
        return iter(self.data)

    def __len__(self) -> int:
        """Number of fields in the record. Also what makes ``bool`` correct."""
        return len(self.data)

    # The mixin would derive these from the five above; delegating instead
    # keeps the dict's own view objects and its one-call lookups, and matches
    # how FSMData answers the same questions.
    def __contains__(self, key: object) -> bool:
        """Whether the record has this field."""
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return self.data.get(key, default)

    def keys(self) -> KeysView[str]:
        """Get keys view."""
        return self.data.keys()

    def values(self) -> ValuesView[Any]:
        """Get values view."""
        return self.data.values()

    def items(self) -> ItemsView[str, Any]:
        """Get items view."""
        return self.data.items()

    def copy(self) -> Dict[str, Any]:
        """A shallow copy of the record, as a plain dict.

        ``MutableMapping`` supplies no ``copy``; this is the one dict method
        that attribute forwarding used to answer and the mixin does not.
        """
        return self.data.copy()

    def to_dict(self) -> Dict[str, Any]:
        """The underlying record.

        Named to match :meth:`FSMData.to_dict`. The two wrappers being asked
        this and answering differently --- one with the record, one with
        ``AttributeError`` --- is what ``ensure_dict`` had to be fixed for.
        """
        return self.data

    def __repr__(self) -> str:
        """String representation showing the record, not an address."""
        return f"{type(self).__name__}({self.data!r})"


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
    # Both wrappers in this module answer `to_dict()` with the raw record.
    # They did not always: `StateDataWrapper.data` became a raw dict in the
    # same commit that made `FSMData` a MutableMapping, leaving this function
    # calling `.to_dict()` on a plain dict — a latent AttributeError that had
    # to be special-cased here. The wrapper answers it now, so the two shapes
    # are one branch again.
    if isinstance(data, (FSMData, StateDataWrapper)):
        return data.to_dict()
    # Other wrapper types, reached by duck typing. Each `_data` is checked
    # rather than returned on sight: this function's whole contract is that
    # what comes back is a dict, and a wrapper holding something else used to
    # make it return that something else --- which the engines then stored as
    # `context.data`.
    inner_data = getattr(data, "_data", None)
    if isinstance(inner_data, dict):
        return inner_data
    if hasattr(data, "data"):
        # Handle objects with data attribute
        inner = data.data
        if isinstance(inner, dict):
            return inner
        elif isinstance(inner, FSMData):
            return inner.to_dict()
        nested = getattr(inner, "_data", None)
        if isinstance(nested, dict):
            return nested
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
