"""Environment variable substitution for configuration values."""

import os
import re
from typing import Any, Dict, List, Union


class VariableSubstitution:
    """Handles environment variable substitution in configuration values.
    
    Supports patterns:
    - ${VAR} - Replace with environment variable VAR, error if not found
    - ${VAR:default} - Replace with VAR or use default if not found
    - ${VAR:-default} - Same as above (bash-style)
    """

    # Pattern to match ${VAR} or ${VAR:default} or ${VAR:-default}
    VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::(-)?([^}]*))?\}')

    def substitute(self, value: Any) -> Any:
        """Recursively substitute environment variables in a value.
        
        Args:
            value: Value to process (can be string, dict, list, or other)
            
        Returns:
            Value with environment variables substituted
            
        Raises:
            ValueError: If a required environment variable is not found
        """
        if isinstance(value, str):
            return self._substitute_string(value)
        elif isinstance(value, dict):
            return self._substitute_dict(value)
        elif isinstance(value, list):
            return self._substitute_list(value)
        else:
            # Return other types unchanged
            return value

    def _substitute_string(self, text: str) -> Union[str, int, float, bool]:
        """Substitute environment variables in a string.
        
        Args:
            text: String potentially containing ${VAR} patterns
            
        Returns:
            String with substitutions, or converted type if entire string is a variable
            
        Raises:
            ValueError: If a required environment variable is not found
        """
        # Check if the entire string is a single variable reference
        if text.startswith('${') and text.endswith('}') and text.count('${') == 1:
            # Single variable - can return non-string types
            match = self.VAR_PATTERN.match(text)
            if match:
                var_name = match.group(1)
                has_default = match.group(2) is not None or match.group(3) is not None
                default_value = match.group(3) if match.group(3) is not None else ""

                if var_name in os.environ:
                    value = os.environ[var_name]
                    # Try to convert to appropriate type
                    return self._convert_type(value)
                elif has_default:
                    # Use default value and convert type
                    return self._convert_type(default_value)
                else:
                    raise ValueError(f"Environment variable '{var_name}' not found")

        # Multiple variables or mixed content - always return string
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            has_default = match.group(2) is not None or match.group(3) is not None
            default_value = match.group(3) if match.group(3) is not None else ""

            if var_name in os.environ:
                return os.environ[var_name]
            elif has_default:
                return default_value
            else:
                raise ValueError(f"Environment variable '{var_name}' not found")

        return self.VAR_PATTERN.sub(replacer, text)

    def _substitute_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute variables in a dictionary.
        
        Args:
            data: Dictionary to process
            
        Returns:
            Dictionary with substituted values
        """
        result = {}
        for key, value in data.items():
            # Keys are not substituted, only values
            result[key] = self.substitute(value)
        return result

    def _substitute_list(self, data: List[Any]) -> List[Any]:
        """Recursively substitute variables in a list.
        
        Args:
            data: List to process
            
        Returns:
            List with substituted values
        """
        return [self.substitute(item) for item in data]

    def _convert_type(self, value: str) -> Union[str, int, float, bool]:
        """Convert a string value to appropriate type.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value (int, float, bool, or original string)
        """
        # Try to convert to boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False

        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass

        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def has_variables(self, value: Any) -> bool:
        """Check if a value contains environment variable references.
        
        Args:
            value: Value to check
            
        Returns:
            True if value contains ${...} patterns
        """
        if isinstance(value, str):
            return bool(self.VAR_PATTERN.search(value))
        elif isinstance(value, dict):
            return any(self.has_variables(v) for v in value.values())
        elif isinstance(value, list):
            return any(self.has_variables(item) for item in value)
        else:
            return False
