"""Environment-aware configuration with late-binding resource resolution.

This module provides the EnvironmentAwareConfig class that supports:
- Logical resource references resolved per-environment
- Late-binding of environment variables (at instantiation time, not load time)
- Separation of portable app config from infrastructure bindings

Example:
    ```python
    # Load with auto-detected environment
    config = EnvironmentAwareConfig.load_app(
        "my-bot",
        app_dir="config/apps",
        env_dir="config/environments"
    )

    # Get resolved config for object building (late binding happens here)
    resolved = config.resolve_for_build()

    # Get portable config for storage (no env vars resolved)
    portable = config.get_portable_config()
    ```

App config format (config/apps/my-bot.yaml):
    ```yaml
    name: my-bot
    version: "1.0.0"

    bot:
      llm:
        $resource: default
        type: llm_providers
        temperature: 0.7

      conversation_storage:
        $resource: conversations
        type: databases
    ```
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from .environment_config import EnvironmentConfig
from .inheritance import substitute_env_vars

logger = logging.getLogger(__name__)


class EnvironmentAwareConfigError(Exception):
    """Error related to environment-aware configuration."""

    pass


class EnvironmentAwareConfig:
    """Configuration with environment-aware resource resolution.

    Manages application configuration with support for:
    - Logical resource references that resolve per-environment
    - Late-binding of environment variables
    - Portable config storage (unresolved)

    Attributes:
        environment: The EnvironmentConfig for resource resolution
        app_name: Name of the loaded application (if any)
    """

    def __init__(
        self,
        config: dict[str, Any],
        environment: EnvironmentConfig | None = None,
        app_name: str | None = None,
    ):
        """Initialize environment-aware configuration.

        Args:
            config: Application configuration dictionary
            environment: Environment configuration for resource resolution.
                        If None, auto-detects and loads environment.
            app_name: Optional name for this application config
        """
        self._config = config
        self._environment = environment or EnvironmentConfig.load()
        self._app_name = app_name or config.get("name")

    @property
    def environment(self) -> EnvironmentConfig:
        """Get the current environment configuration."""
        return self._environment

    @property
    def environment_name(self) -> str:
        """Get the current environment name."""
        return self._environment.name

    @property
    def app_name(self) -> str | None:
        """Get the application name."""
        return self._app_name

    @classmethod
    def load_app(
        cls,
        app_name: str,
        app_dir: str | Path = "config/apps",
        env_dir: str | Path = "config/environments",
        environment: str | None = None,
    ) -> EnvironmentAwareConfig:
        """Load an application configuration with environment bindings.

        This is the primary entry point for loading configs in an
        environment-aware manner. Config files are loaded WITHOUT
        environment variable substitution (late binding).

        Args:
            app_name: Application/bot name (without .yaml extension)
            app_dir: Directory containing app configs
            env_dir: Directory containing environment configs
            environment: Environment name, or None to auto-detect

        Returns:
            EnvironmentAwareConfig with both app and environment loaded

        Raises:
            EnvironmentAwareConfigError: If app config not found or invalid
        """
        app_dir = Path(app_dir)
        env_config = EnvironmentConfig.load(environment, env_dir)

        # Find and load app config file
        config_path = cls._find_config_file(app_dir, app_name)
        if config_path is None:
            raise EnvironmentAwareConfigError(
                f"Application config not found: {app_name}.yaml in {app_dir}"
            )

        try:
            config = cls._load_file(config_path)
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise EnvironmentAwareConfigError(
                f"Failed to parse app config {config_path}: {e}"
            ) from e
        except OSError as e:
            raise EnvironmentAwareConfigError(
                f"Failed to read app config {config_path}: {e}"
            ) from e

        logger.info(
            f"Loaded app config '{app_name}' for environment '{env_config.name}'"
        )

        return cls(
            config=config,
            environment=env_config,
            app_name=app_name,
        )

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        environment: str | None = None,
        env_dir: str | Path = "config/environments",
    ) -> EnvironmentAwareConfig:
        """Create from a configuration dictionary.

        Args:
            config: Application configuration dictionary
            environment: Environment name, or None to auto-detect
            env_dir: Directory containing environment configs

        Returns:
            EnvironmentAwareConfig instance
        """
        env_config = EnvironmentConfig.load(environment, env_dir)
        return cls(config=config, environment=env_config)

    @classmethod
    def _find_config_file(cls, config_dir: Path, name: str) -> Path | None:
        """Find a config file by name.

        Args:
            config_dir: Directory to search
            name: Config name (without extension)

        Returns:
            Path to config file, or None if not found
        """
        for ext in [".yaml", ".yml", ".json"]:
            path = config_dir / f"{name}{ext}"
            if path.exists():
                return path
        return None

    @classmethod
    def _load_file(cls, path: Path) -> dict[str, Any]:
        """Load and parse a config file WITHOUT env var substitution.

        Args:
            path: Path to config file

        Returns:
            Parsed configuration dictionary (with env var placeholders intact)
        """
        with open(path, encoding="utf-8") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        if not isinstance(data, dict):
            raise EnvironmentAwareConfigError(
                f"Config file must contain a dictionary: {path}"
            )

        return data

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the config.

        Args:
            key: Configuration key (supports dot notation for nested access)
            default: Default value if not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return copy.deepcopy(value)

    def resolve_for_build(
        self,
        config_key: str | None = None,
        resolve_resources: bool = True,
        resolve_env_vars: bool = True,
    ) -> dict[str, Any]:
        """Resolve configuration for object building.

        This is the late-binding resolution point where:
        1. Logical resource names are resolved to concrete configs
        2. Environment variables are substituted
        3. Final merged configuration is returned

        Call this method immediately before instantiating objects.

        Args:
            config_key: Specific config key to resolve, or None for root
            resolve_resources: Whether to resolve logical resource refs
            resolve_env_vars: Whether to substitute environment variables

        Returns:
            Fully resolved configuration dictionary
        """
        # Get the base configuration
        if config_key:
            config = self.get(config_key)
            if config is None:
                raise EnvironmentAwareConfigError(
                    f"Config key not found: {config_key}"
                )
        else:
            config = copy.deepcopy(self._config)

        # Resolve logical resource references
        if resolve_resources:
            config = self._resolve_resource_refs(config)

        # Resolve environment variables (late binding)
        if resolve_env_vars:
            config = substitute_env_vars(config)

        return config

    def _resolve_resource_refs(self, config: Any) -> Any:
        """Resolve logical resource references in configuration.

        Finds resource references in the config and replaces them
        with concrete configurations from the environment.

        Resource references are dicts with `$resource` key:
        ```yaml
        database:
          $resource: conversations
          type: databases
          extra_param: value  # merged into resolved config
        ```

        Args:
            config: Configuration to process

        Returns:
            Configuration with resource references resolved
        """
        if isinstance(config, dict):
            if "$resource" in config:
                # This is a resource reference
                resource_name = config["$resource"]
                resource_type = config.get("type", "default")

                # Get defaults from the reference (exclude markers)
                defaults = {
                    k: v
                    for k, v in config.items()
                    if k not in ("$resource", "type")
                }

                try:
                    resolved = self._environment.get_resource(
                        resource_type, resource_name, defaults
                    )
                    # Recursively resolve any nested references in the resolved config
                    return self._resolve_resource_refs(resolved)
                except KeyError:
                    # Resource not found - return config with defaults only
                    # This allows graceful degradation
                    logger.warning(
                        f"Resource '{resource_name}' of type '{resource_type}' "
                        f"not found in environment '{self._environment.name}', "
                        f"using defaults"
                    )
                    return defaults if defaults else config
            else:
                # Regular dict - recurse into values
                return {
                    key: self._resolve_resource_refs(value)
                    for key, value in config.items()
                }
        elif isinstance(config, list):
            # Recurse into list items
            return [self._resolve_resource_refs(item) for item in config]
        else:
            # Return other types unchanged
            return config

    def get_portable_config(self) -> dict[str, Any]:
        """Get the portable (unresolved) configuration.

        Returns the configuration with:
        - Logical resource references intact
        - Environment variables as placeholders

        This is the config that should be stored in databases
        for cross-environment portability.

        Returns:
            Unresolved configuration dictionary
        """
        return copy.deepcopy(self._config)

    def to_dict(self) -> dict[str, Any]:
        """Get the raw configuration dictionary.

        Alias for get_portable_config().

        Returns:
            Configuration dictionary
        """
        return self.get_portable_config()

    def with_environment(
        self,
        environment: str | EnvironmentConfig,
        env_dir: str | Path = "config/environments",
    ) -> EnvironmentAwareConfig:
        """Create a new instance with a different environment.

        Useful for testing or multi-environment scenarios.

        Args:
            environment: Environment name or EnvironmentConfig instance
            env_dir: Directory containing environment configs (if name provided)

        Returns:
            New EnvironmentAwareConfig with the specified environment
        """
        if isinstance(environment, str):
            env_config = EnvironmentConfig.load(environment, env_dir)
        else:
            env_config = environment

        return EnvironmentAwareConfig(
            config=copy.deepcopy(self._config),
            environment=env_config,
            app_name=self._app_name,
        )

    def get_resource(
        self,
        resource_type: str,
        logical_name: str,
        defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get a resolved resource configuration.

        Convenience method to directly access environment resources.

        Args:
            resource_type: Type of resource
            logical_name: Logical name of resource
            defaults: Default values if resource not found

        Returns:
            Resolved resource configuration
        """
        return self._environment.get_resource(resource_type, logical_name, defaults)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get an environment setting.

        Args:
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value
        """
        return self._environment.get_setting(key, default)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EnvironmentAwareConfig(app={self._app_name!r}, "
            f"environment={self._environment.name!r})"
        )
