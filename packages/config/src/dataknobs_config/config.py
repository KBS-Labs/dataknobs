"""Core Config class implementation."""

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml  # type: ignore[import-untyped]

from .builders import ObjectBuilder
from .environment import EnvironmentOverrides
from .exceptions import (
    ConfigError,
    ConfigNotFoundError,
    FileNotFoundError as ConfigFileNotFoundError,
    ValidationError,
)
from .references import ReferenceResolver
from .settings import SettingsManager


class Config:
    """A modular configuration system for composable settings.

    Internally stores configurations as a dictionary of lists of atomic
    configuration dictionaries, organized by type.
    """

    def __init__(self, *sources: Union[str, Path, dict], **kwargs: Any) -> None:
        """Initialize a Config object from one or more sources.

        Args:
            *sources: Variable number of sources (file paths or dictionaries)
            **kwargs: Additional keyword arguments
        """
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._settings_manager = SettingsManager()
        self._reference_resolver = ReferenceResolver(self)
        self._environment_overrides = EnvironmentOverrides()
        self._object_builder = ObjectBuilder(self)

        # Load sources
        for source in sources:
            self.load(source)

        # Apply environment overrides if enabled
        if kwargs.get("use_env", True):
            self._apply_environment_overrides()

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """Create a Config object from a file.

        Args:
            path: Path to configuration file (YAML or JSON)

        Returns:
            Config object
        """
        return cls(path)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create a Config object from a dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Config object
        """
        return cls(data)

    def load(self, source: Union[str, Path, dict]) -> None:
        """Load configuration from a source.

        Args:
            source: File path or dictionary
        """
        if isinstance(source, dict):
            self._load_dict(source)
        elif isinstance(source, (str, Path)):
            self._load_file(source)
        else:
            raise ValidationError(f"Invalid source type: {type(source)}")

    def _load_file(self, path: Union[str, Path]) -> None:
        """Load configuration from a file.

        Args:
            path: Path to configuration file
        """
        path = Path(path).resolve()

        if not path.exists():
            raise ConfigFileNotFoundError(f"Configuration file not found: {path}")

        # Update config_root if not set
        if not self._settings_manager.get_setting("config_root"):
            self._settings_manager.set_setting("config_root", str(path.parent))

        # Load file based on extension
        suffix = path.suffix.lower()
        with open(path) as f:
            if suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif suffix == ".json":
                data = json.load(f)
            else:
                raise ValidationError(f"Unsupported file format: {suffix}")

        if data:
            self._load_dict(data)

    def _load_dict(self, data: dict) -> None:
        """Load configuration from a dictionary.

        Args:
            data: Configuration dictionary
        """
        # Handle settings first
        if "settings" in data:
            self._settings_manager.load_settings(data["settings"])

        # Process other types
        for type_name, configs in data.items():
            if type_name == "settings":
                continue

            # Ensure configs is a list
            if not isinstance(configs, list):
                configs = [configs]

            # Initialize the type list if it doesn't exist
            if type_name not in self._data:
                self._data[type_name] = []

            # Remember the starting count for this batch
            start_count = len(self._data[type_name])

            # Process each atomic configuration
            for idx, config in enumerate(configs):
                # Handle file references
                if isinstance(config, str) and config.startswith("@"):
                    config = self._load_referenced_file(config[1:])

                # Validate and normalize atomic config
                # Pass the absolute position (start_count + idx)
                config = self._normalize_atomic_config(config, type_name, start_count + idx)

                # Apply path resolution
                config = self._resolve_paths(config, type_name)

                # Store configuration
                self._data[type_name].append(config)

    def _load_referenced_file(self, path: str) -> dict:
        """Load a referenced configuration file.

        Args:
            path: Path to the referenced file (relative or absolute)

        Returns:
            Loaded configuration dictionary
        """
        # Resolve relative paths using config_root
        if not os.path.isabs(path):
            config_root = self._settings_manager.get_setting("config_root")
            if not config_root:
                raise ConfigError("config_root not set for relative file reference")
            path = os.path.join(config_root, path)

        path_obj = Path(path).resolve()

        if not path_obj.exists():
            raise ConfigFileNotFoundError(f"Referenced configuration file not found: {path_obj}")

        # Load file based on extension
        suffix = path_obj.suffix.lower()
        with open(path_obj) as f:
            if suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f) or {}
            elif suffix == ".json":
                data: Dict[Any, Any] = json.load(f)
                return data
            else:
                raise ValidationError(f"Unsupported file format: {suffix}")

    def _normalize_atomic_config(self, config: dict, type_name: str, idx: int) -> dict:
        """Normalize an atomic configuration.

        Args:
            config: Atomic configuration dictionary
            type_name: Type of the configuration
            idx: Index position for this config (absolute position in the type's list)

        Returns:
            Normalized configuration
        """
        config = copy.deepcopy(config)

        # Ensure type is set
        if "type" not in config:
            config["type"] = type_name
        elif config["type"] != type_name:
            raise ValidationError(f"Type mismatch: expected {type_name}, got {config['type']}")

        # Ensure name is set
        if "name" not in config:
            # The idx passed should already be the absolute position
            config["name"] = str(idx)

        # Apply defaults
        config = self._settings_manager.apply_defaults(config, type_name)

        return config

    def _resolve_paths(self, config: dict, type_name: str) -> dict:
        """Resolve relative paths in configuration.

        Args:
            config: Atomic configuration dictionary
            type_name: Type of the configuration

        Returns:
            Configuration with resolved paths
        """
        return self._settings_manager.resolve_paths(config, type_name)

    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configurations."""
        overrides = self._environment_overrides.get_overrides()

        for ref, value in overrides.items():
            try:
                # Parse reference and apply override
                type_name, name_or_index, attr = self._environment_overrides.parse_env_reference(
                    ref
                )

                # Get the configuration
                config = self.get(type_name, name_or_index)

                # Apply the override
                config[attr] = value

                # Set it back
                self.set(type_name, name_or_index, config)
            except Exception as e:
                # Log warning but don't fail
                print(f"Warning: Failed to apply environment override {ref}: {e}")

    def get_types(self) -> List[str]:
        """Get all configuration types.

        Returns:
            List of type names
        """
        return list(self._data.keys())

    def get_count(self, type_name: str) -> int:
        """Get the count of configurations for a type.

        Args:
            type_name: Type name

        Returns:
            Number of configurations
        """
        return len(self._data.get(type_name, []))

    def get_names(self, type_name: str) -> List[str]:
        """Get all configuration names for a type.

        Args:
            type_name: Type name

        Returns:
            List of configuration names
        """
        configs = self._data.get(type_name, [])
        return [config.get("name", str(i)) for i, config in enumerate(configs)]

    def get(self, type_name: str, name_or_index: Union[str, int] = 0) -> dict:
        """Get a configuration by type and name/index.

        Args:
            type_name: Type name
            name_or_index: Configuration name or index

        Returns:
            Configuration dictionary
        """
        if type_name not in self._data:
            raise ConfigNotFoundError(f"Type not found: {type_name}")

        configs = self._data[type_name]

        if isinstance(name_or_index, int):
            # Handle index access
            try:
                return copy.deepcopy(configs[name_or_index])
            except IndexError:
                raise ConfigNotFoundError(f"Index out of range: {name_or_index}")
        else:
            # Handle name access
            for config in configs:
                if config.get("name") == name_or_index:
                    return copy.deepcopy(config)
            raise ConfigNotFoundError(f"Configuration not found: {type_name}[{name_or_index}]")

    def set(self, type_name: str, name_or_index: Union[str, int], config: dict) -> None:
        """Set a configuration by type and name/index.

        Args:
            type_name: Type name
            name_or_index: Configuration name or index
            config: Configuration dictionary
        """
        if type_name not in self._data:
            self._data[type_name] = []

        configs = self._data[type_name]
        config = copy.deepcopy(config)

        # Normalize the configuration
        if isinstance(name_or_index, int):
            config = self._normalize_atomic_config(config, type_name, name_or_index)
            if name_or_index < len(configs):
                configs[name_or_index] = config
            elif name_or_index == len(configs):
                configs.append(config)
            else:
                raise ValidationError(f"Index out of range: {name_or_index}")
        else:
            # Find and replace by name
            config["name"] = name_or_index

            # Find the existing config to replace
            replaced = False
            for i, existing in enumerate(configs):
                if existing.get("name") == name_or_index:
                    # Normalize with the correct index
                    config = self._normalize_atomic_config(config, type_name, i)
                    configs[i] = config
                    replaced = True
                    break

            if not replaced:
                # New config, add to end
                config = self._normalize_atomic_config(config, type_name, len(configs))
                configs.append(config)

    def resolve_reference(self, ref: str) -> dict:
        """Resolve a string reference to a configuration.

        Args:
            ref: String reference (e.g., "xref:foo[bar]")

        Returns:
            Referenced configuration dictionary
        """
        return self._reference_resolver.resolve(ref)

    def build_reference(self, type_name: str, name_or_index: Union[str, int]) -> str:
        """Build a string reference for a configuration.

        Args:
            type_name: Type name
            name_or_index: Configuration name or index

        Returns:
            String reference
        """
        return self._reference_resolver.build(type_name, name_or_index)

    def merge(self, other: "Config", precedence: str = "first") -> None:
        """Merge another configuration into this one.

        Args:
            other: Configuration to merge
            precedence: Precedence rule ("first" or "last")
        """
        if precedence not in ["first", "last"]:
            raise ValidationError(f"Invalid precedence: {precedence}")

        # Merge settings first
        if precedence == "first":
            # Current settings take precedence
            other_settings = other._settings_manager._settings.copy()
            other_settings.update(self._settings_manager._settings)
            self._settings_manager._settings = other_settings
        else:
            # Other settings take precedence
            self._settings_manager._settings.update(other._settings_manager._settings)

        # Merge configurations
        for type_name in other.get_types():
            if type_name not in self._data:
                self._data[type_name] = []

            # Add all configurations from other
            for config in other._data[type_name]:
                # Check for duplicate names
                existing_names = self.get_names(type_name)
                if config.get("name") in existing_names:
                    if precedence == "first":
                        # Skip if we already have this name
                        continue
                    else:
                        # Replace existing with same name
                        name = config.get("name")
                        if name is not None:
                            self.set(type_name, name, config)
                else:
                    # Add new configuration
                    self._data[type_name].append(copy.deepcopy(config))

    def to_dict(self) -> dict:
        """Export configuration as a dictionary.

        Returns:
            Configuration dictionary
        """
        result = copy.deepcopy(self._data)

        # Add settings if any
        settings = self._settings_manager.to_dict()
        if settings:
            result["settings"] = settings  # type: ignore[assignment]

        return result

    def to_file(self, path: Union[str, Path], format: str | None = None) -> None:
        """Save configuration to a file.

        Args:
            path: Output file path
            format: Output format ("yaml" or "json"), auto-detected if not specified
        """
        path = Path(path)

        # Auto-detect format from extension if not specified
        if format is None:
            suffix = path.suffix.lower()
            if suffix in [".yaml", ".yml"]:
                format = "yaml"
            elif suffix == ".json":
                format = "json"
            else:
                raise ValidationError(f"Cannot determine format from extension: {suffix}")

        data = self.to_dict()

        with open(path, "w") as f:
            if format == "yaml":
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            elif format == "json":
                json.dump(data, f, indent=2)
            else:
                raise ValidationError(f"Unsupported format: {format}")

    def build_object(self, ref: str, cache: bool = True, **kwargs: Any) -> Any:
        """Build an object from a configuration reference.

        Args:
            ref: String reference to configuration
            cache: Whether to cache the built object
            **kwargs: Additional keyword arguments for construction

        Returns:
            Built object instance
        """
        return self._object_builder.build(ref, cache=cache, **kwargs)

    def clear_object_cache(self, ref: str | None = None) -> None:
        """Clear cached objects.

        Args:
            ref: Specific reference to clear, or None to clear all
        """
        self._object_builder.clear_cache(ref)

    def get_factory(self, type_name: str, name_or_index: Union[str, int] = 0) -> Any:
        """Get a factory instance for a configuration.

        This method lazily instantiates and caches factory instances defined
        in configurations with a 'factory' attribute.

        Args:
            type_name: Type name
            name_or_index: Configuration name or index

        Returns:
            Factory instance

        Raises:
            ConfigError: If no factory is defined for the configuration
        """
        # Get the configuration
        config = self.get(type_name, name_or_index)

        if "factory" not in config:
            raise ConfigError(
                f"No factory defined for {type_name}[{name_or_index}]. "
                f"Add 'factory' attribute to the configuration."
            )

        # Build a reference for caching
        ref = self.build_reference(type_name, name_or_index)
        factory_ref = f"{ref}.factory"

        # Check if factory is already cached
        cached = self._object_builder.get_cached(factory_ref)
        if cached is not None:
            return cached

        # Load and instantiate the factory
        factory_path = config["factory"]
        factory_cls = self._object_builder._load_class(factory_path)

        # Create factory instance
        try:
            factory = factory_cls()
        except TypeError:
            # Factory might be a module-level function or callable class
            factory = factory_cls

        # Cache the factory instance
        self._object_builder._cache[factory_ref] = factory

        return factory

    def get_instance(
        self, type_name: str, name_or_index: Union[str, int] = 0, **kwargs: Any
    ) -> Any:
        """Get an instance from a configuration.

        This is a convenience method that combines get() and build_object().
        If the configuration has a 'class' or 'factory' attribute, it will
        build and return an instance. Otherwise, it returns the config dict.

        Args:
            type_name: Type name
            name_or_index: Configuration name or index
            **kwargs: Additional keyword arguments for construction

        Returns:
            Built instance or configuration dictionary
        """
        config = self.get(type_name, name_or_index)

        # If config has class or factory, build an object
        if "class" in config or "factory" in config:
            ref = self.build_reference(type_name, name_or_index)
            return self.build_object(ref, **kwargs)

        # Otherwise return the config itself
        return config
