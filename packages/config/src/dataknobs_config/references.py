"""String reference system for cross-referencing configurations."""

import re
from typing import TYPE_CHECKING, Any, Dict, Tuple, Union

if TYPE_CHECKING:
    from .config import Config

from .exceptions import InvalidReferenceError


class ReferenceResolver:
    """Handles parsing and resolution of string references.

    Reference format: xref:<type>[<name_or_index>]

    Examples:
        - xref:foo[bar] - Named reference
        - xref:foo[0] - Index reference
        - xref:foo[-1] - Last item reference
        - xref:foo - First/only item reference
    """

    # Regex pattern for parsing references
    REFERENCE_PATTERN = re.compile(r"^xref:([a-zA-Z_][a-zA-Z0-9_]*)?(?:\[([^\]]+)\])?$")

    def __init__(self, config_instance: "Config") -> None:
        """Initialize the reference resolver.

        Args:
            config_instance: The Config instance to resolve references against
        """
        self._config = config_instance
        self._resolving: set[str] = set()  # Track references being resolved to detect cycles

    def resolve(self, ref: str) -> dict:
        """Resolve a string reference to a configuration.

        Args:
            ref: String reference (e.g., "xref:foo[bar]")

        Returns:
            Referenced configuration dictionary

        Raises:
            InvalidReferenceError: If reference format is invalid
            ConfigNotFoundError: If referenced configuration doesn't exist
        """
        # Check for circular references
        if ref in self._resolving:
            raise InvalidReferenceError(f"Circular reference detected: {ref}")

        # Parse the reference
        type_name, name_or_index = self.parse_reference(ref)

        # Track that we're resolving this reference
        self._resolving.add(ref)

        try:
            # Get the configuration
            config = self._config.get(type_name, name_or_index)

            # Resolve any nested references
            config = self._resolve_nested_references(config)

            return config
        finally:
            # Remove from resolving set
            self._resolving.discard(ref)

    def parse_reference(self, ref: str) -> Tuple[str, Union[str, int]]:
        """Parse a string reference into its components.

        Args:
            ref: String reference

        Returns:
            Tuple of (type_name, name_or_index)

        Raises:
            InvalidReferenceError: If reference format is invalid
        """
        if not ref.startswith("xref:"):
            raise InvalidReferenceError(f"Reference must start with 'xref:': {ref}")

        match = self.REFERENCE_PATTERN.match(ref)
        if not match:
            raise InvalidReferenceError(f"Invalid reference format: {ref}")

        type_name = match.group(1)
        selector = match.group(2)

        if not type_name:
            raise InvalidReferenceError(f"Missing type in reference: {ref}")

        # Parse selector
        name_or_index: Union[str, int]
        if selector is None:
            # No selector means first item (index 0)
            name_or_index = 0
        elif selector.isdigit() or (selector.startswith("-") and selector[1:].isdigit()):
            # Numeric selector (index)
            name_or_index = int(selector)
        else:
            # String selector (name)
            name_or_index = selector

        return type_name, name_or_index

    def build(self, type_name: str, name_or_index: Union[str, int]) -> str:
        """Build a string reference for a configuration.

        Args:
            type_name: Type name
            name_or_index: Configuration name or index

        Returns:
            String reference
        """
        if isinstance(name_or_index, int):
            if name_or_index == 0:
                # First item can be referenced without selector
                return f"xref:{type_name}"
            else:
                return f"xref:{type_name}[{name_or_index}]"
        else:
            return f"xref:{type_name}[{name_or_index}]"

    def is_reference(self, value: Any) -> bool:
        """Check if a value is a string reference.

        Args:
            value: Value to check

        Returns:
            True if value is a string reference
        """
        if not isinstance(value, str):
            return False
        return value.startswith("xref:")

    def _resolve_nested_references(self, config: dict) -> dict:
        """Recursively resolve any string references within a configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with resolved references
        """
        resolved: Dict[str, Any] = {}

        for key, value in config.items():
            if self.is_reference(value):
                # Resolve the reference
                resolved[key] = self.resolve(value)
            elif isinstance(value, dict):
                # Recursively resolve nested dictionaries
                resolved[key] = self._resolve_nested_references(value)
            elif isinstance(value, list):
                # Resolve references in lists
                resolved_list: list[Any] = [
                    self.resolve(item) if self.is_reference(item) else item for item in value
                ]
                resolved[key] = resolved_list
            else:
                # Keep value as-is
                resolved[key] = value

        return resolved
