"""Environment variable override system."""

import os
import re
from typing import Any, Dict, Tuple, Union

from .exceptions import InvalidReferenceError


class EnvironmentOverrides:
    """Handles environment variable overrides for configurations.

    Environment variable format:
    DATAKNOBS_<TYPE>__<NAME_OR_INDEX>__<ATTRIBUTE>

    Examples:
        - DATAKNOBS_FOO__BAR__PARAM -> xref:foo[bar].param
        - DATAKNOBS_FOO__0__PARAM -> xref:foo[0].param
    """

    ENV_PREFIX = "DATAKNOBS_"
    ENV_SEPARATOR = "__"

    def __init__(self, prefix: str | None = None) -> None:
        """Initialize the environment override handler.

        Args:
            prefix: Custom environment variable prefix (default: DATAKNOBS_)
        """
        self.prefix = prefix or self.ENV_PREFIX

    def get_overrides(self) -> Dict[str, Any]:
        """Get all environment variable overrides.

        Returns:
            Dictionary mapping references to override values
        """
        overrides = {}

        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                try:
                    # Parse the environment variable
                    ref = self._env_var_to_reference(key)

                    # Convert value to appropriate type
                    typed_value = self._parse_value(value)

                    overrides[ref] = typed_value
                except Exception:
                    # Skip invalid environment variables
                    continue

        return overrides

    def _env_var_to_reference(self, env_var: str) -> str:
        """Convert an environment variable name to a reference string.

        Args:
            env_var: Environment variable name

        Returns:
            Reference string

        Raises:
            InvalidReferenceError: If environment variable format is invalid
        """
        # Remove prefix
        if not env_var.startswith(self.prefix):
            raise InvalidReferenceError(f"Environment variable must start with {self.prefix}")

        parts = env_var[len(self.prefix) :].split(self.ENV_SEPARATOR)

        if len(parts) < 3:
            raise InvalidReferenceError(f"Invalid environment variable format: {env_var}")

        type_name = parts[0].lower()
        name_or_index = parts[1]
        attribute = self.ENV_SEPARATOR.join(parts[2:]).lower()

        # Convert index strings to integers
        name_or_index_typed: Union[str, int]
        if name_or_index.isdigit() or (
            name_or_index.startswith("-") and name_or_index[1:].isdigit()
        ):
            name_or_index_typed = int(name_or_index)
        else:
            # Keep string names lowercase for consistency
            name_or_index_typed = name_or_index.lower()

        # Build reference string with attribute
        if isinstance(name_or_index_typed, int) and name_or_index_typed == 0:
            ref = f"xref:{type_name}.{attribute}"
        else:
            ref = f"xref:{type_name}[{name_or_index_typed}].{attribute}"

        return ref

    def reference_to_env_var(self, ref: str, attribute: str | None = None) -> str:
        """Convert a reference string to an environment variable name.

        Args:
            ref: Reference string (e.g., "xref:foo[bar]")
            attribute: Attribute name (if not included in ref)

        Returns:
            Environment variable name
        """
        # Parse reference components
        type_name, name_or_index, attr = self.parse_env_reference(ref)

        if attribute:
            attr = attribute

        if not attr:
            raise InvalidReferenceError(f"Attribute required for environment variable: {ref}")

        # Build environment variable name
        type_part = type_name.upper()
        selector_part = str(name_or_index).upper()
        attr_part = attr.upper().replace(".", self.ENV_SEPARATOR)

        return f"{self.prefix}{type_part}{self.ENV_SEPARATOR}{selector_part}{self.ENV_SEPARATOR}{attr_part}"

    def parse_env_reference(self, ref: str) -> Tuple[str, Union[str, int], str | None]:
        """Parse a reference string with attribute.

        Args:
            ref: Reference string with attribute (e.g., "xref:foo[bar].param")

        Returns:
            Tuple of (type_name, name_or_index, attribute)
        """
        # Check if reference includes attribute
        if "." in ref and not ref.endswith("]"):
            # Split reference and attribute
            base_ref, attribute = ref.rsplit(".", 1)
        else:
            base_ref = ref
            attribute = None

        # Parse base reference
        pattern = re.compile(r"^xref:([a-zA-Z_][a-zA-Z0-9_]*)?(?:\[([^\]]+)\])?$")
        match = pattern.match(base_ref)

        if not match:
            # Try parsing the full reference if it includes attribute
            if "." in ref:
                parts = ref.split(".")
                if parts[0].startswith("xref:"):
                    # Handle xref:type.attribute format
                    type_part = parts[0][5:]  # Remove 'xref:' prefix

                    # Check for selector in type part
                    if "[" in type_part:
                        type_name, selector = type_part.split("[", 1)
                        selector = selector.rstrip("]")
                    else:
                        type_name = type_part
                        selector = None

                    attribute = ".".join(parts[1:])
                else:
                    raise InvalidReferenceError(f"Invalid reference format: {ref}")
            else:
                raise InvalidReferenceError(f"Invalid reference format: {ref}")
        else:
            type_name = match.group(1)
            selector = match.group(2)

        if not type_name:
            raise InvalidReferenceError(f"Missing type in reference: {ref}")

        # Parse selector
        name_or_index: Union[str, int]
        if selector is None:
            name_or_index = 0
        elif selector.isdigit() or (selector.startswith("-") and selector[1:].isdigit()):
            name_or_index = int(selector)
        else:
            name_or_index = selector

        return type_name, name_or_index, attribute

    def _parse_value(self, value: str) -> Any:
        """Parse an environment variable value to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Parsed value (string, int, float, bool, or original string)
        """
        # Try to parse as boolean
        if value.lower() in ["true", "yes", "1"]:
            return True
        elif value.lower() in ["false", "no", "0"]:
            return False

        # Try to parse as integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try to parse as float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value
