"""Environment-specific configuration and resource bindings.

This module provides environment-aware configuration management for deploying
the same application across different environments (development, staging,
production) where infrastructure differs.

Key features:
- Environment detection (via DATAKNOBS_ENVIRONMENT or cloud indicators)
- Resource bindings (logical names -> concrete implementations)
- Environment-wide settings management
- ``${VAR}`` / ``${VAR:default}`` substitution applied by default in
  :meth:`EnvironmentConfig.load` and :meth:`EnvironmentConfig.from_dict`,
  matching the behaviour of :meth:`InheritableConfigLoader.load` for
  domain configs (pass ``substitute_vars=False`` to opt out).

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

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dataknobs_common.config_loading import (
    ConfigLoadError,
    find_config_file,
    load_yaml_or_json,
)

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
        substituted: Whether ``${VAR}`` substitution has already been applied
            to the values held here. See below.
    """

    name: str
    resources: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    substituted: bool = field(default=False, compare=False)
    """Whether ``${VAR}`` substitution has already been applied to the values
    in this config.

    :meth:`load` and :meth:`from_dict` set this when they substitute. It stays
    ``False`` for direct dataclass construction, which is what keeps the
    downstream substitution passes in :class:`EnvironmentAwareConfig` and
    :class:`ConfigBindingResolver` load-bearing for that path.

    Downstream layers read this so they substitute each source **exactly
    once**. Substituting a second time re-expands the *output* of the first,
    which reinterprets the content of a value as a template — so a secret
    whose own text contains ``${...}`` is replaced by the value of whatever
    unrelated variable that text happens to name.

    Excluded from equality: two configs holding the same values are the same
    environment regardless of which layer expanded them.
    """

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
        *,
        substitute_vars: bool = True,
    ) -> EnvironmentConfig:
        """Load environment configuration from file.

        Args:
            environment: Environment name, or None to auto-detect
            config_dir: Directory containing environment config files
            substitute_vars: When True (default), apply ``${VAR}`` /
                ``${VAR:default}`` substitution to every value in the
                loaded YAML before constructing the model — matching the
                behaviour of :meth:`InheritableConfigLoader.load` for
                domain configs. Pass ``False`` only if you specifically
                need to preserve raw refs (e.g., to inspect or transform
                them).

        Returns:
            Loaded EnvironmentConfig instance

        Raises:
            EnvironmentConfigError: If config file is invalid
            ValueError: If ``substitute_vars=True`` and a required
                ``${VAR}`` ref has no default and no value in the
                environment.
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

        data = cls._load_file(config_path)

        if substitute_vars:
            # Local import to keep the dependency on inheritance.py
            # explicit at the call site and defensive against future
            # refactors that could re-introduce a top-level cycle.
            from .inheritance import substitute_env_vars

            data = substitute_env_vars(data)

        return cls(
            name=data.get("name", environment),
            resources=data.get("resources", {}),
            settings=data.get("settings", {}),
            description=data.get("description", ""),
            substituted=substitute_vars,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        substitute_vars: bool = True,
    ) -> EnvironmentConfig:
        """Create EnvironmentConfig from a dictionary.

        Args:
            data: Configuration dictionary
            substitute_vars: When True (default), apply ``${VAR}`` /
                ``${VAR:default}`` substitution to every value in
                ``data`` before constructing the model — same semantics
                as :meth:`load`.

        Returns:
            EnvironmentConfig instance

        Raises:
            ValueError: If ``substitute_vars=True`` and a required
                ``${VAR}`` ref has no default and no value in the
                environment.
        """
        if substitute_vars:
            from .inheritance import substitute_env_vars

            data = substitute_env_vars(data)
        return cls(
            name=data.get("name", "unknown"),
            resources=data.get("resources", {}),
            settings=data.get("settings", {}),
            description=data.get("description", ""),
            substituted=substitute_vars,
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
        return find_config_file(config_dir, environment)

    @classmethod
    def _load_file(cls, path: Path) -> dict[str, Any]:
        """Load and parse a config file.

        Args:
            path: Path to config file

        Returns:
            Parsed configuration dictionary
        """
        try:
            return load_yaml_or_json(path)
        except ConfigLoadError as e:
            raise EnvironmentConfigError(
                f"Failed to load environment config {path}: {e}"
            ) from e
        except OSError as e:
            raise EnvironmentConfigError(
                f"Failed to read environment config {path}: {e}"
            ) from e

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

    def substituted_view(self) -> EnvironmentConfig:
        """Return an equivalent config with ``${VAR}`` substitution applied.

        Returns ``self`` when substitution has already been applied, so
        calling this is always safe and never expands a value twice.

        Never mutates: a caller holding an unsubstituted config for direct
        :meth:`get_resource` reads keeps the config it asked for, even when
        a resolution layer reads through it.

        Returns:
            ``self`` if already substituted, otherwise a substituted copy.

        Raises:
            ValueError: If a required ``${VAR}`` ref has no default and no
                value in the environment.
        """
        if self.substituted:
            return self

        from .inheritance import substitute_env_vars

        return replace(
            self,
            resources=substitute_env_vars(self.resources),
            settings=substitute_env_vars(self.settings),
            substituted=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Values are emitted as held — already substituted for a config built
        by :meth:`load` or :meth:`from_dict` with the default
        ``substitute_vars=True``. Provenance is deliberately **not** emitted,
        so feeding the result back through :meth:`from_dict` substitutes a
        second time. Round-trip with
        ``EnvironmentConfig.from_dict(cfg.to_dict(), substitute_vars=False)``
        to preserve values whose own text contains ``${...}``.

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

        When the two sides disagree on :attr:`substituted`, the unsubstituted
        side is substituted **during** the merge and the result is marked
        substituted. ``substituted`` is a single flag describing the whole
        config, so it is only sound if provenance is uniform within one
        instance; degrading the result to ``False`` instead would leave the
        already-substituted half exposed to a second pass downstream, which
        is the exact defect the flag exists to prevent.

        Args:
            other: Environment config to merge

        Returns:
            New merged EnvironmentConfig
        """
        # Normalize mixed provenance before merging so the result's single
        # ``substituted`` flag is true of every value in it.
        if self.substituted != other.substituted:
            return self.substituted_view().merge(other.substituted_view())

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
            substituted=self.substituted,
        )
