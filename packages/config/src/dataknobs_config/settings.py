"""Global settings and defaults management."""

import copy
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, Union


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

    #: Settings that configure the loader rather than the objects it loads.
    #: Every other dotless setting becomes a default attribute on every
    #: atomic config, so a loader setting not named here would silently
    #: appear as a key on everything the config produces.
    _LOADER_SETTINGS = frozenset(
        {
            "config_root",
            "global_root",
            "path_resolution_attributes",
        }
    )

    #: Settings a caller may pass but config *content* may not carry, because
    #: they govern a boundary that content is on the far side of. A config
    #: file naming one of these would be an input relaxing the check that
    #: bounds it, which is not a check. They are refused rather than dropped:
    #: failing closed on a silent drop leaves an operator who wrote one in
    #: YAML watching their references raise with nothing pointing at why.
    _CALLER_ONLY_SETTINGS: ClassVar[dict[str, str]] = {
        "allow_reference_outside_config_root": (
            "pass allow_reference_outside_config_root=True to Config(...) instead"
        ),
    }

    def __init__(self) -> None:
        """Initialize the settings manager."""
        self._settings: Dict[str, Any] = {}

    def load_settings(self, settings: dict) -> None:
        """Load settings that came from a configuration source.

        This is the *content* plane — a ``settings:`` block in a loaded file,
        or the same block in a dict handed to :class:`~dataknobs_config.Config`.
        :meth:`set_setting` is the caller plane. The split is what keeps a
        boundary out of reach of the content it bounds, so a setting named in
        :attr:`_CALLER_ONLY_SETTINGS` is refused here and settable only there.

        Args:
            settings: Settings dictionary

        Raises:
            ConfigError: ``settings`` names a caller-only setting.
        """
        for key in settings:
            if key in self._CALLER_ONLY_SETTINGS:
                from .exceptions import ConfigError

                raise ConfigError(
                    f"{key!r} may not be set from configuration content: "
                    f"{self._CALLER_ONLY_SETTINGS[key]}"
                )

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
                if key not in self._LOADER_SETTINGS:
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
        """Resolve a single path value against ``global_root``.

        ``global_root`` is a **resolution base, not a boundary**, and this is
        the one place where the difference is worth stating — the composition
        below looks like the ``@``-reference one in
        :meth:`~dataknobs_config.Config._load_referenced_file`, which *is*
        bounded, and the two are answering different questions.

        Two properties of this feature say it is not a boundary, and both are
        the published contract rather than an oversight: an absolute value is
        returned untouched (there is nothing to resolve), and a relative one
        may climb out — ``config_path: ../configs/db.conf`` resolving to a
        sibling of ``global_root`` is asserted by
        ``test_resolve_relative_paths``. Bounding only the relative spelling
        while the absolute one passes through would be the "reject ``..`` and
        stop there" half-guard that :mod:`dataknobs_common.paths` exists to
        name, and bounding both would change what the feature is.

        Nothing is opened here either way: this rewrites a string that a
        *consumer* later opens, so the boundary that matters is the
        consumer's. Turning ``global_root`` into one is a deliberate change to
        ``path_resolution_attributes``, not a fix to make here.

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
        if Path(path).is_absolute():
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
