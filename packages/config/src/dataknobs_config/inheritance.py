"""Configuration inheritance utilities.

This module provides simple configuration inheritance via an `extends` field,
complementing the existing Config system with single-file config loading.

The InheritableConfigLoader supports:
- Loading YAML/JSON configuration files
- Configuration inheritance via 'extends' field
- Deep merge with child overriding parent
- Environment variable substitution
- Caching for performance

Example:
    ```yaml
    # base.yaml
    llm:
      provider: openai
      model: gpt-4
      temperature: 0.7

    knowledge_base:
      chunk_size: 500
      overlap: 50

    # domain.yaml
    extends: base

    llm:
      model: gpt-4-turbo  # Override just this field

    domain_specific:
      feature_enabled: true
    ```

    ```python
    loader = InheritableConfigLoader("./configs")
    config = loader.load("domain")  # Merges base.yaml + domain.yaml
    ```
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class InheritanceError(Exception):
    """Error during configuration inheritance resolution."""
    pass


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Recursively merges override into base, with override values taking precedence.
    Nested dictionaries are merged recursively; all other types are replaced.

    Args:
        base: Base dictionary (values used when not overridden)
        override: Override dictionary (takes precedence)

    Returns:
        New merged dictionary

    Example:
        >>> base = {"a": 1, "nested": {"x": 10, "y": 20}}
        >>> override = {"a": 2, "nested": {"y": 25, "z": 30}}
        >>> deep_merge(base, override)
        {'a': 2, 'nested': {'x': 10, 'y': 25, 'z': 30}}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override takes precedence
            result[key] = value

    return result


def substitute_env_vars(data: Any) -> Any:
    """Recursively substitute environment variables in configuration.

    Supports formats:
    - ${VAR_NAME}: Required variable, raises error if not set
    - ${VAR_NAME:default_value}: Optional with default

    Also expands ~ in paths after substitution.

    Args:
        data: Configuration data (dict, list, string, or primitive)

    Returns:
        Data with environment variables substituted

    Raises:
        ValueError: If required environment variable not set

    Example:
        >>> os.environ["MY_VAR"] = "hello"
        >>> substitute_env_vars({"key": "${MY_VAR}", "default": "${MISSING:world}"})
        {'key': 'hello', 'default': 'world'}
    """
    if isinstance(data, dict):
        return {k: substitute_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [substitute_env_vars(item) for item in data]
    elif isinstance(data, str):
        return _substitute_string(data)
    else:
        return data


def _substitute_string(value: str) -> str:
    """Substitute environment variables in a string.

    Args:
        value: String potentially containing ${VAR} references

    Returns:
        String with variables substituted

    Raises:
        ValueError: If required variable not set
    """
    # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
    pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_value = match.group(2)

        env_value = os.environ.get(var_name)

        if env_value is not None:
            return env_value
        elif default_value is not None:
            return default_value
        else:
            raise ValueError(f"Required environment variable not set: {var_name}")

    result = re.sub(pattern, replacer, value)
    if not result:
        return result
    return str(Path(result).expanduser())


class InheritableConfigLoader:
    """Configuration loader with inheritance support.

    Loads YAML/JSON configuration files with support for configuration
    inheritance via an `extends` field. Child configurations override
    parent values through deep merge.

    Attributes:
        config_dir: Directory containing configuration files
        cache: Configuration cache for performance
    """

    def __init__(self, config_dir: str | Path | None = None):
        """Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files.
                       If None, uses ./configs
        """
        self.config_dir = Path(config_dir) if config_dir else Path("./configs")
        self._cache: dict[str, dict[str, Any]] = {}
        self._loading: set[str] = set()  # Track configs being loaded to detect cycles

    def load(
        self,
        name: str,
        use_cache: bool = True,
        substitute_vars: bool = True,
    ) -> dict[str, Any]:
        """Load and resolve configuration with inheritance.

        Args:
            name: Configuration name (without extension)
            use_cache: Whether to use cached configuration if available
            substitute_vars: Whether to substitute environment variables

        Returns:
            Resolved configuration dictionary

        Raises:
            InheritanceError: If config not found, cycle detected, or other error

        Example:
            ```python
            loader = InheritableConfigLoader("./configs")
            config = loader.load("my-domain")
            ```
        """
        # Check cache
        if use_cache and name in self._cache:
            logger.debug(f"Using cached config: {name}")
            return self._cache[name]

        # Detect circular inheritance
        if name in self._loading:
            raise InheritanceError(f"Circular inheritance detected: {name}")

        self._loading.add(name)

        try:
            # Load raw configuration
            raw_config = self._load_file(name)

            # Handle inheritance
            if raw_config.get("extends"):
                parent_name = raw_config["extends"]
                logger.debug(f"Config '{name}' extends '{parent_name}'")

                # Load parent configuration (recursively handles inheritance)
                parent_config = self.load(parent_name, use_cache=use_cache, substitute_vars=False)

                # Deep merge: child overrides parent
                raw_config = deep_merge(parent_config, raw_config)

                # Remove extends field from final config
                raw_config.pop("extends", None)

            # Substitute environment variables
            if substitute_vars:
                raw_config = substitute_env_vars(raw_config)

            # Cache the result
            self._cache[name] = raw_config
            logger.info(f"Loaded configuration: {name}")

            return raw_config

        finally:
            self._loading.discard(name)

    def load_from_file(
        self,
        filepath: str | Path,
        substitute_vars: bool = True,
    ) -> dict[str, Any]:
        """Load configuration from a specific file path.

        This method bypasses the config_dir and loads directly from the path.
        Inheritance is resolved relative to the file's directory.

        Args:
            filepath: Path to configuration file
            substitute_vars: Whether to substitute environment variables

        Returns:
            Resolved configuration dictionary

        Raises:
            InheritanceError: If file not found or other error
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise InheritanceError(f"Configuration file not found: {filepath}")

        # Temporarily change config_dir to file's directory for inheritance
        old_config_dir = self.config_dir
        self.config_dir = filepath.parent

        try:
            return self.load(filepath.stem, use_cache=False, substitute_vars=substitute_vars)
        finally:
            self.config_dir = old_config_dir

    def _load_file(self, name: str) -> dict[str, Any]:
        """Load raw configuration file.

        Args:
            name: Configuration name (without extension)

        Returns:
            Parsed configuration dictionary

        Raises:
            InheritanceError: If file not found or parse error
        """
        # Try YAML first, then JSON
        yaml_path = self.config_dir / f"{name}.yaml"
        yml_path = self.config_dir / f"{name}.yml"
        json_path = self.config_dir / f"{name}.json"

        if yaml_path.exists():
            filepath = yaml_path
        elif yml_path.exists():
            filepath = yml_path
        elif json_path.exists():
            filepath = json_path
        else:
            raise InheritanceError(
                f"Configuration file not found: {name}.yaml or {name}.json "
                f"in {self.config_dir}"
            )

        try:
            with open(filepath, encoding="utf-8") as f:
                if filepath.suffix in [".yaml", ".yml"]:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            if not isinstance(data, dict):
                raise InheritanceError(
                    f"Configuration file must contain a dictionary: {filepath}"
                )

            return data

        except yaml.YAMLError as e:
            raise InheritanceError(f"Failed to parse YAML file {filepath}: {e}") from e
        except json.JSONDecodeError as e:
            raise InheritanceError(f"Failed to parse JSON file {filepath}: {e}") from e
        except OSError as e:
            raise InheritanceError(f"Failed to read configuration file {filepath}: {e}") from e

    def clear_cache(self, name: str | None = None) -> None:
        """Clear configuration cache.

        Args:
            name: Specific config to clear, or None to clear all
        """
        if name:
            self._cache.pop(name, None)
            logger.debug(f"Cleared cache for: {name}")
        else:
            self._cache.clear()
            logger.debug("Cleared all cached configurations")

    def list_available(self) -> list[str]:
        """List all available configuration files.

        Returns:
            List of configuration names (without extensions)
        """
        if not self.config_dir.exists():
            return []

        configs = set()
        for pattern in ["*.yaml", "*.yml", "*.json"]:
            for file in self.config_dir.glob(pattern):
                if file.is_file():
                    configs.add(file.stem)

        return sorted(configs)

    def validate(self, name: str) -> tuple[bool, str | None]:
        """Validate a configuration file.

        Args:
            name: Configuration name

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self.load(name, use_cache=False)
            return True, None
        except InheritanceError as e:
            return False, str(e)
        except ValueError as e:
            return False, str(e)


def load_config_with_inheritance(
    filepath: str | Path,
    substitute_vars: bool = True,
) -> dict[str, Any]:
    """Convenience function to load a config file with inheritance.

    Args:
        filepath: Path to configuration file
        substitute_vars: Whether to substitute environment variables

    Returns:
        Resolved configuration dictionary

    Example:
        ```python
        config = load_config_with_inheritance("configs/my-domain.yaml")
        ```
    """
    loader = InheritableConfigLoader()
    return loader.load_from_file(filepath, substitute_vars=substitute_vars)
