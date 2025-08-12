"""Global settings and defaults management."""

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Union


class SettingsManager:
    """Manages global settings, defaults, and path resolution.

    Settings attributes:
        - config_root: Root directory for configuration files
        - global_root: Global root directory for all types
        - <type>.global_root: Type-specific root directory
        - path_resolution_attributes: List of attributes to resolve as paths
        - <type>.<attribute>: Type-specific defaults
        - <attribute>: Global defaults
    """

    def __init__(self) -> None:
        """Initialize the settings manager."""
        self._settings: Dict[str, Any] = {}

    def load_settings(self, settings: dict) -> None:
        """Load settings from a dictionary.

        Args:
            settings: Settings dictionary
        """
        # Merge with existing settings (first seen takes precedence)
        for key, value in settings.items():
            if key not in self._settings:
                self._settings[key] = value

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a setting value.

        Args:
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value or default
        """
        return self._settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value.

        Args:
            key: Setting key
            value: Setting value
        """
        self._settings[key] = value

    def apply_defaults(self, config: dict, type_name: str) -> dict:
        """Apply default values to a configuration.

        Args:
            config: Atomic configuration dictionary
            type_name: Type of the configuration

        Returns:
            Configuration with defaults applied
        """
        result = copy.deepcopy(config)

        # Apply global defaults
        for key, value in self._settings.items():
            if "." not in key and key not in result:
                # Global default for any attribute
                if key not in ["config_root", "global_root", "path_resolution_attributes"]:
                    result[key] = copy.deepcopy(value)

        # Apply type-specific defaults
        type_prefix = f"{type_name}."
        for key, value in self._settings.items():
            if key.startswith(type_prefix):
                attr_name = key[len(type_prefix) :]
                if attr_name not in result:
                    result[attr_name] = copy.deepcopy(value)

        return result

    def resolve_paths(self, config: dict, type_name: str) -> dict:
        """Resolve relative paths in configuration.

        Args:
            config: Atomic configuration dictionary
            type_name: Type of the configuration

        Returns:
            Configuration with resolved paths
        """
        result = copy.deepcopy(config)

        # Get path resolution attributes
        path_attrs = self.get_setting("path_resolution_attributes", [])
        if not path_attrs:
            return result

        # Determine base path for resolution
        base_path = self._get_base_path(type_name)

        # Resolve paths for matching attributes
        for attr_pattern in path_attrs:
            # Check if it's a regex pattern (starts with / and ends with /)
            if (
                isinstance(attr_pattern, str)
                and attr_pattern.startswith("/")
                and attr_pattern.endswith("/")
            ):
                # Regex pattern
                pattern_str = attr_pattern[1:-1]  # Remove the / delimiters
                try:
                    pattern = re.compile(pattern_str)
                    # Apply to all matching attributes
                    for key in list(result.keys()):
                        if pattern.match(key):
                            result[key] = self._resolve_path(result[key], base_path)
                except re.error:
                    # Invalid regex, skip
                    continue
            elif "." in attr_pattern:
                # Type-specific attribute
                type_prefix, attr_name = attr_pattern.split(".", 1)
                if type_prefix == type_name:
                    # Check if attr_name is a regex
                    if attr_name.startswith("/") and attr_name.endswith("/"):
                        pattern_str = attr_name[1:-1]
                        try:
                            pattern = re.compile(pattern_str)
                            for key in list(result.keys()):
                                if pattern.match(key):
                                    result[key] = self._resolve_path(result[key], base_path)
                        except re.error:
                            continue
                    elif attr_name in result:
                        result[attr_name] = self._resolve_path(result[attr_name], base_path)
            # Global attribute (exact match)
            elif attr_pattern in result:
                result[attr_pattern] = self._resolve_path(result[attr_pattern], base_path)

        return result

    def _get_base_path(self, type_name: str) -> str | None:
        """Get the base path for resolving relative paths.

        Args:
            type_name: Type name

        Returns:
            Base path or None
        """
        # Check for type-specific root
        type_root = self.get_setting(f"{type_name}.global_root")
        if type_root:
            return str(type_root)

        # Check for global root
        global_root = self.get_setting("global_root")
        if global_root:
            return str(global_root)

        # No base path available - will be handled by _resolve_path
        return None

    def _resolve_path(self, path: Union[str, Any], base_path: str | None) -> Union[str, Any]:
        """Resolve a single path value.

        Args:
            path: Path value (may not be a string)
            base_path: Base path for resolution

        Returns:
            Resolved path or original value

        Raises:
            ConfigError: If relative path needs resolution but no base path is available
        """
        # Only resolve string paths
        if not isinstance(path, str):
            return path

        # Check if already absolute
        if os.path.isabs(path):
            return path

        # Relative path needs base path for resolution
        if not base_path:
            from .exceptions import ConfigError

            raise ConfigError(
                f"Cannot resolve relative path '{path}': no global_root or type-specific "
                f"global_root is set. Set 'global_root' or '<type>.global_root' in settings."
            )

        # Resolve relative path
        resolved = Path(base_path) / path
        return str(resolved.resolve())

    def to_dict(self) -> dict:
        """Export settings as a dictionary.

        Returns:
            Settings dictionary
        """
        return copy.deepcopy(self._settings)
