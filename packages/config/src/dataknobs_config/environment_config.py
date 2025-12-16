"""Environment-specific configuration and resource bindings.

This module provides environment-aware configuration management for deploying
the same application across different environments (development, staging,
production) where infrastructure differs.

Key features:
- Environment detection (via DATAKNOBS_ENVIRONMENT or cloud indicators)
- Resource bindings (logical names -> concrete implementations)
- Environment-wide settings management

Example:
    ```python
    # Auto-detect environment
    env = EnvironmentConfig.load()

    # Or specify explicitly
    env = EnvironmentConfig.load("production", config_dir="config/environments")

    # Get concrete config for a logical resource
    db_config = env.get_resource("databases", "conversations")
    # Returns: {"backend": "postgres", "connection_string": "..."}
    ```

Environment file format (config/environments/production.yaml):
    ```yaml
    name: production
    description: AWS production environment

    settings:
      log_level: INFO
      enable_metrics: true

    resources:
      databases:
        default:
          backend: postgres
          connection_string: ${DATABASE_URL}
        conversations:
          backend: postgres
          connection_string: ${DATABASE_URL}
          table: conversations

      vector_stores:
        default:
          backend: pgvector
          connection_string: ${DATABASE_URL}
    ```
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class EnvironmentConfigError(Exception):
    """Error related to environment configuration."""

    pass


class ResourceNotFoundError(EnvironmentConfigError, KeyError):
    """Resource not found in environment configuration."""

    pass


@dataclass
class ResourceBinding:
    """A binding from logical name to concrete implementation.

    Attributes:
        name: Logical name of the resource
        resource_type: Type of resource (e.g., "databases", "vector_stores")
        config: Concrete configuration for the resource
    """

    name: str
    resource_type: str
    config: dict[str, Any]


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration and resource bindings.

    Manages the mapping from logical resource names to concrete
    implementations for a specific deployment environment.

    Attributes:
        name: Environment name (e.g., "development", "staging", "production")
        resources: Nested dict of {resource_type: {logical_name: config}}
        settings: Environment-wide settings (log levels, feature flags, etc.)
        description: Optional description of the environment
    """

    name: str
    resources: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def detect_environment(cls) -> str:
        """Detect current environment from env vars or indicators.

        Checks in order:
        1. DATAKNOBS_ENVIRONMENT env var
        2. Common cloud indicators (AWS_EXECUTION_ENV, etc.)
        3. Default to "development"

        Returns:
            Detected environment name (lowercase)
        """
        # Explicit setting takes precedence
        if env := os.environ.get("DATAKNOBS_ENVIRONMENT"):
            return env.lower()

        # AWS Lambda or ECS
        if os.environ.get("AWS_EXECUTION_ENV"):
            # Could be Lambda, ECS, etc.
            env_tier = os.environ.get("ENVIRONMENT", "production")
            return env_tier.lower()

        # AWS ECS Fargate
        if os.environ.get("ECS_CONTAINER_METADATA_URI"):
            env_tier = os.environ.get("ENVIRONMENT", "production")
            return env_tier.lower()

        # Kubernetes
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            env_tier = os.environ.get("ENVIRONMENT", "production")
            return env_tier.lower()

        # Google Cloud Run
        if os.environ.get("K_SERVICE"):
            env_tier = os.environ.get("ENVIRONMENT", "production")
            return env_tier.lower()

        # Azure Functions
        if os.environ.get("FUNCTIONS_WORKER_RUNTIME"):
            env_tier = os.environ.get("ENVIRONMENT", "production")
            return env_tier.lower()

        # Default to development
        return "development"

    @classmethod
    def load(
        cls,
        environment: str | None = None,
        config_dir: str | Path = "config/environments",
    ) -> EnvironmentConfig:
        """Load environment configuration from file.

        Args:
            environment: Environment name, or None to auto-detect
            config_dir: Directory containing environment config files

        Returns:
            Loaded EnvironmentConfig instance

        Raises:
            EnvironmentConfigError: If config file is invalid
        """
        if environment is None:
            environment = cls.detect_environment()

        config_dir = Path(config_dir)
        config_path = cls._find_config_file(config_dir, environment)

        if config_path is None:
            # Return empty config for environments without config files
            logger.debug(
                f"No environment config found for '{environment}' in {config_dir}, "
                "using empty configuration"
            )
            return cls(name=environment)

        try:
            data = cls._load_file(config_path)
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise EnvironmentConfigError(
                f"Failed to parse environment config {config_path}: {e}"
            ) from e
        except OSError as e:
            raise EnvironmentConfigError(
                f"Failed to read environment config {config_path}: {e}"
            ) from e

        return cls(
            name=data.get("name", environment),
            resources=data.get("resources", {}),
            settings=data.get("settings", {}),
            description=data.get("description", ""),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentConfig:
        """Create EnvironmentConfig from a dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            EnvironmentConfig instance
        """
        return cls(
            name=data.get("name", "unknown"),
            resources=data.get("resources", {}),
            settings=data.get("settings", {}),
            description=data.get("description", ""),
        )

    @classmethod
    def _find_config_file(cls, config_dir: Path, environment: str) -> Path | None:
        """Find the config file for an environment.

        Args:
            config_dir: Directory to search
            environment: Environment name

        Returns:
            Path to config file, or None if not found
        """
        for ext in [".yaml", ".yml", ".json"]:
            path = config_dir / f"{environment}{ext}"
            if path.exists():
                return path
        return None

    @classmethod
    def _load_file(cls, path: Path) -> dict[str, Any]:
        """Load and parse a config file.

        Args:
            path: Path to config file

        Returns:
            Parsed configuration dictionary
        """
        with open(path, encoding="utf-8") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        if not isinstance(data, dict):
            raise EnvironmentConfigError(
                f"Environment config must be a dictionary: {path}"
            )

        return data

    def get_resource(
        self,
        resource_type: str,
        logical_name: str,
        defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get concrete config for a logical resource.

        Args:
            resource_type: Type of resource ("databases", "vector_stores", etc.)
            logical_name: Logical name referenced in app config
            defaults: Default config values if resource not found

        Returns:
            Concrete configuration for the resource

        Raises:
            ResourceNotFoundError: If resource not found and no defaults provided
        """
        type_resources = self.resources.get(resource_type, {})

        if logical_name in type_resources:
            # Copy to avoid mutation
            config = type_resources[logical_name].copy()

            # Apply defaults for missing keys
            if defaults:
                for key, value in defaults.items():
                    config.setdefault(key, value)

            return config

        if defaults is not None:
            return defaults.copy()

        raise ResourceNotFoundError(
            f"Resource '{logical_name}' of type '{resource_type}' "
            f"not found in environment '{self.name}'"
        )

    def has_resource(self, resource_type: str, logical_name: str) -> bool:
        """Check if a resource exists.

        Args:
            resource_type: Type of resource
            logical_name: Logical name of resource

        Returns:
            True if resource exists
        """
        return logical_name in self.resources.get(resource_type, {})

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get an environment-wide setting.

        Args:
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value
        """
        return self.settings.get(key, default)

    def get_resource_types(self) -> list[str]:
        """Get all resource types in this environment.

        Returns:
            List of resource type names
        """
        return list(self.resources.keys())

    def get_resource_names(self, resource_type: str) -> list[str]:
        """Get all resource names for a type.

        Args:
            resource_type: Type of resource

        Returns:
            List of logical resource names
        """
        return list(self.resources.get(resource_type, {}).keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of environment config
        """
        result: dict[str, Any] = {"name": self.name}

        if self.description:
            result["description"] = self.description

        if self.settings:
            result["settings"] = self.settings.copy()

        if self.resources:
            result["resources"] = {
                rtype: {name: config.copy() for name, config in resources.items()}
                for rtype, resources in self.resources.items()
            }

        return result

    def merge(self, other: EnvironmentConfig) -> EnvironmentConfig:
        """Merge another environment config into this one.

        The other config's values take precedence.

        Args:
            other: Environment config to merge

        Returns:
            New merged EnvironmentConfig
        """
        # Deep merge resources
        merged_resources: dict[str, dict[str, dict[str, Any]]] = {}

        # Start with self's resources
        for rtype, resources in self.resources.items():
            merged_resources[rtype] = {
                name: config.copy() for name, config in resources.items()
            }

        # Merge in other's resources
        for rtype, resources in other.resources.items():
            if rtype not in merged_resources:
                merged_resources[rtype] = {}
            for name, config in resources.items():
                if name in merged_resources[rtype]:
                    # Merge configs
                    merged_resources[rtype][name].update(config)
                else:
                    merged_resources[rtype][name] = config.copy()

        # Merge settings
        merged_settings = self.settings.copy()
        merged_settings.update(other.settings)

        return EnvironmentConfig(
            name=other.name,
            resources=merged_resources,
            settings=merged_settings,
            description=other.description or self.description,
        )
