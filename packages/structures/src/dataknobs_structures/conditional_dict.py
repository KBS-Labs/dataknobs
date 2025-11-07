"""Conditional dictionary with validation using the strategy pattern.

This module provides cdict, a dictionary subclass that validates key-value pairs
before accepting them. Rejected items are tracked separately, making it useful for
data filtering, validation, and conditional processing.

The strategy pattern allows flexible validation logic to be injected at creation time,
enabling use cases like:
- Type validation (only accept specific value types)
- Range checking (only accept values within bounds)
- Duplicate prevention (only accept unique keys)
- Business rule enforcement (custom validation logic)

Typical usage example:

    ```python
    from dataknobs_structures import cdict

    # Only accept positive numbers
    positive_dict = cdict(lambda d, k, v: isinstance(v, (int, float)) and v > 0)
    positive_dict['a'] = 10   # Accepted
    positive_dict['b'] = -5   # Rejected

    print(positive_dict)          # {'a': 10}
    print(positive_dict.rejected)  # {'b': -5}
    ```
"""

from collections.abc import Callable
from typing import Any, Dict


class cdict(dict):  # noqa: N801
    """Dictionary that validates key-value pairs before acceptance.

    A dictionary subclass that applies a validation function to each item before
    allowing it to be set. Items that fail validation are stored in a separate
    "rejected" dictionary rather than being added to the main dictionary.

    This uses the strategy pattern where the validation logic is provided as a
    function at initialization time, allowing flexible and reusable validation rules.

    Attributes:
        rejected: Dictionary containing key-value pairs that failed validation.
        accept_fn: The validation function used to check items.

    Example:
        ```python
        # Only accept string keys
        str_keys = cdict(lambda d, k, v: isinstance(k, str))
        str_keys['valid'] = 1      # Accepted
        str_keys[123] = 2          # Rejected

        # Only accept even numbers
        evens = cdict(lambda d, k, v: isinstance(v, int) and v % 2 == 0)
        evens['a'] = 2    # Accepted
        evens['b'] = 3    # Rejected
        evens['c'] = 4    # Accepted

        print(evens)           # {'a': 2, 'c': 4}
        print(evens.rejected)  # {'b': 3}

        # Prevent duplicates using dict state
        no_dups = cdict(lambda d, k, v: k not in d)
        no_dups['x'] = 1
        no_dups['x'] = 2  # Rejected (key already exists)
        ```
    """

    def __init__(
        self, accept_fn: Callable[[Dict, Any, Any], bool], *args: Any, **kwargs: Any
    ) -> None:
        """Initialize conditional dictionary with validation function.

        Args:
            accept_fn: Validation function with signature (dict, key, value) -> bool.
                Returns True to accept the key-value pair, False to reject it.
                The dict parameter is the current state of this dictionary.
            *args: Initial items as a mapping or iterable of key-value pairs.
            **kwargs: Additional initial items as keyword arguments.

        Example:
            ```python
            # Validation function receives dict, key, and value
            def validate(d, key, val):
                return isinstance(val, int) and val > 0

            # Initialize empty
            d1 = cdict(validate)

            # Initialize with items
            d2 = cdict(validate, {'a': 1, 'b': 2})

            # Initialize with kwargs
            d3 = cdict(validate, x=5, y=10)
            ```
        """
        super().__init__()
        self._rejected: Dict[Any, Any] = {}
        self.accept_fn = accept_fn
        # super().__init__(*args, **kwargs)
        self.update(*args, **kwargs)

    @property
    def rejected(self) -> Dict:
        """Dictionary of rejected key-value pairs.

        Returns:
            Dictionary containing items that failed validation.

        Example:
            ```python
            d = cdict(lambda _, k, v: v > 0)
            d['a'] = 5   # Accepted
            d['b'] = -1  # Rejected

            print(d.rejected)  # {'b': -1}
            ```
        """
        return self._rejected

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set an item if it passes validation, otherwise reject it.

        Args:
            key: The dictionary key.
            value: The value to set.

        Example:
            ```python
            d = cdict(lambda _, k, v: isinstance(v, str))
            d['name'] = 'Alice'  # Accepted
            d['age'] = 30        # Rejected (not a string)
            ```
        """
        if self.accept_fn(self, key, value):
            super().__setitem__(key, value)
        else:
            self._rejected[key] = value

    def setdefault(self, key: Any, default: Any = None) -> Any:
        """Set default value if key doesn't exist and passes validation.

        If the key exists, returns its value. If not, validates the default value
        and sets it if accepted (or rejects it if not).

        Args:
            key: The dictionary key.
            default: Default value to set if key doesn't exist. Defaults to None.

        Returns:
            The existing value if key exists, the default value if set, or None
            if the default was rejected.

        Example:
            ```python
            d = cdict(lambda _, k, v: v > 0)
            d['a'] = 5

            print(d.setdefault('a', 10))  # 5 (existing value)
            print(d.setdefault('b', 3))   # 3 (accepted and set)
            print(d.setdefault('c', -1))  # None (rejected)
            print(d)  # {'a': 5, 'b': 3}
            ```
        """
        rv = None
        if key not in self:
            if self.accept_fn(self, key, default):
                super().__setitem__(key, default)
                rv = default
            else:
                self._rejected[key] = default
        else:
            rv = self[key]
        return rv

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update dictionary with key-value pairs, validating each.

        Accepts either a mapping object, an iterable of key-value pairs, or
        keyword arguments. Each item is validated and added or rejected individually.

        Args:
            *args: Mapping or iterable of (key, value) pairs.
            **kwargs: Key-value pairs as keyword arguments.

        Example:
            ```python
            d = cdict(lambda _, k, v: isinstance(v, int))

            # Update from dict
            d.update({'a': 1, 'b': 'text', 'c': 3})
            print(d)           # {'a': 1, 'c': 3}
            print(d.rejected)  # {'b': 'text'}

            # Update from kwargs
            d.update(x=5, y='invalid')
            print(d)           # {'a': 1, 'c': 3, 'x': 5}
            print(d.rejected)  # {'b': 'text', 'y': 'invalid'}
            ```
        """
        # Handle positional argument if present
        if args:
            other = args[0]
            if hasattr(other, "keys"):
                # It's a mapping-like object
                for key in other.keys():
                    self.__setitem__(key, other[key])
            else:
                # It's an iterable of key-value pairs
                for key, value in other:
                    self.__setitem__(key, value)
        # Handle keyword arguments
        for key, value in kwargs.items():
            self.__setitem__(key, value)
