"""Environment-aware configuration with late-binding resource resolution.

This module provides the EnvironmentAwareConfig class that supports:
- Logical resource references resolved per-environment
- Late-binding of environment variables (at instantiation time, not load time)
- Separation of portable app config from infrastructure bindings

Substitution rule: **substitute each source exactly once, at the latest point
that source is still separable.** For the environment that is its load; for
the app config it is entry to :meth:`EnvironmentAwareConfig.resolve_for_build`,
which is still before resource refs are spliced in. Once spliced, the two are
merged beyond distinguishing, and a pass over the result would expand
environment values a second time — reinterpreting the *content* of a value as
a template. :attr:`EnvironmentConfig.substituted` is what makes the
environment's provenance knowable rather than guessed.

Two things are separable for longer than that, and so are expanded later: an
environment resource, which is expanded as it is spliced rather than with the
environment as a whole, and a reference's inline defaults, which the splice
discards wherever the environment supplies the key. Both follow the same rule
— expanding either earlier reads values the build then throws away, and an
unset required ``${VAR}`` among them aborts a build that never used it.

That makes a splice, not the entry point, the place expansion happens, and a
splice merges two sources whose provenances differ. So each source is
expanded and resolved on its own terms *before* the merge
(:meth:`EnvironmentAwareConfig._resolve_source`) rather than the merged
result being walked once afterwards: one flag cannot be true of both halves,
and walking the result under it would either expand an environment's values a
second time or leave a default's nested refs raw.

A ``$resource`` block nested inside either source is held to one expansion
too, though not in the same place. Carried by an inline default — or by a
resource an *unsubstituted* environment supplies — it reaches its own splice
raw, because the pass over that source defers it, and is expanded there.
Carried by a resource an already substituted environment supplies, it was
expanded at that environment's load, and ``substituted`` is what stops the
splice expanding it a second time.

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
import logging
from pathlib import Path
from typing import Any

from dataknobs_common.config_loading import (
    ConfigLoadError,
    find_config_file,
    load_yaml_or_json,
)

from .environment_config import EnvironmentConfig
from .inheritance import substitute_env_vars

logger = logging.getLogger(__name__)

#: Keys of a ``$resource`` block that select and constrain the resource
#: rather than supplying a default for it. Everything else in the block is an
#: inline default, kept raw by the entry pass and expanded at the splice.
RESOURCE_MARKER_KEYS = frozenset({"$resource", "type", "$requires"})


def _substitute_deferring_defaults(
    config: Any, *, defer_defaults: bool = True
) -> Any:
    """Expand ``${VAR}`` refs, holding ``$resource`` inline defaults back.

    Run over each source as it enters resolution — the app config on the way
    in, and every value a splice carries. Markers are always expanded, so a
    reference can select its resource by variable.

    Inline defaults are held back for two reasons that point the same way.
    The splice discards every one the environment supplies, so expanding one
    here reads a value the build then throws away — and an unset required
    ``${VAR}`` among them aborts a build that never used it. And the splice
    expands each default that survives it, so one expanded here would be
    expanded a second time there. Substitution is not idempotent: the second
    pass re-reads the first pass's output, so a value whose own text contains
    ``${...}`` is treated as a template.

    Holding a default back therefore keeps every value at exactly one
    expansion, performed at the splice that proves it survived.

    ``defer_defaults=False`` expands everything, for the caller that will not
    splice at all. With no splice nothing discards a default and nothing else
    would expand it, so deferring there would strand a raw ``${VAR}``.
    """
    if isinstance(config, dict):
        deferring = defer_defaults and "$resource" in config
        substituted: dict[Any, Any] = {}
        for key, value in config.items():
            # :func:`substitute_env_vars` expands keys as well as values, and
            # this is a wrapper around it. Re-walking the structure here and
            # calling it only at the leaves would keep everything it does to a
            # value and silently drop everything it does at a container --
            # which is how expanding keys got lost. Handing the key back to it
            # is also what passes a non-string key through untouched.
            key = substitute_env_vars(key)
            if deferring:
                substituted[key] = (
                    substitute_env_vars(value)
                    if key in RESOURCE_MARKER_KEYS
                    else value
                )
            else:
                substituted[key] = _substitute_deferring_defaults(
                    value, defer_defaults=defer_defaults
                )
        return substituted
    if isinstance(config, list):
        return [
            _substitute_deferring_defaults(
                item, defer_defaults=defer_defaults
            )
            for item in config
        ]
    return substitute_env_vars(config)


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

        config = cls._load_file(config_path)

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
        return find_config_file(config_dir, name)

    @classmethod
    def _load_file(cls, path: Path) -> dict[str, Any]:
        """Load and parse a config file WITHOUT env var substitution.

        Args:
            path: Path to config file

        Returns:
            Parsed configuration dictionary (with env var placeholders intact)
        """
        try:
            return load_yaml_or_json(path)
        except ConfigLoadError as e:
            raise EnvironmentAwareConfigError(
                f"Failed to load app config {path}: {e}"
            ) from e
        except OSError as e:
            raise EnvironmentAwareConfigError(
                f"Failed to read app config {path}: {e}"
            ) from e

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

        # Late-bind app-authored ${VAR} refs BEFORE splicing in environment
        # values. Environment values were already substituted when the
        # environment was loaded (or are substituted exactly once, below,
        # when it was not) -- substituting after the splice would expand
        # them a second time, reinterpreting the *content* of a value as a
        # template. See the module docstring.
        #
        # Inline defaults are held back: the splice discards every one the
        # environment supplies, so expanding them here would read values the
        # build is about to throw away -- the same argument that keeps this
        # from expanding the whole environment, applied to the other source.
        #
        # Only when there is a splice to hold them for, though. Without one
        # nothing discards a default and nothing else expands it, so
        # deferring would hand a caller literal ${VAR} text.
        if resolve_env_vars:
            config = _substitute_deferring_defaults(
                config, defer_defaults=resolve_resources
            )

        # Resolve logical resource references. Each resource is substituted
        # as it is spliced, not the environment as a whole: a resource is
        # still separable at the splice point, which is the latest point it
        # can be expanded, and expanding the whole environment would read
        # values no reference names -- so an unset required ${VAR} in an
        # unrelated resource would abort a build that never looked at it.
        if resolve_resources:
            config = self._resolve_resource_refs(
                config, self._environment, substitute=resolve_env_vars
            )

        return config

    def _resolve_resource_refs(
        self,
        config: Any,
        environment: EnvironmentConfig | None = None,
        substitute: bool = False,
    ) -> Any:
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
            environment: Environment to resolve against. Defaults to this
                config's own environment.
            substitute: Whether the config being walked still needs
                expanding. Each source a splice merges is finished by
                :meth:`_resolve_source` on its own terms before the merge, so
                an environment that arrived expanded is not expanded again
                while the inline defaults merged alongside it — which always
                arrive raw — are, once each, and only where the environment
                did not supply the key.

        Returns:
            Configuration with resource references resolved
        """
        if environment is None:
            environment = self._environment

        if isinstance(config, dict):
            if "$resource" in config:
                # This is a resource reference
                resource_name = config["$resource"]
                resource_type = config.get("type", "default")

                # Get defaults from the reference (exclude markers and metadata)
                defaults = {
                    k: v
                    for k, v in config.items()
                    if k not in RESOURCE_MARKER_KEYS
                }
                requires = config.get("$requires", [])

                if not environment.has_resource(resource_type, resource_name):
                    # Resource not found - degrade to the reference's inline
                    # defaults. Membership is tested explicitly rather than
                    # caught: get_resource returns the supplied defaults
                    # instead of raising whenever a defaults dict is passed,
                    # and the dict comprehension above always produces one
                    # (possibly empty). Relying on ResourceNotFoundError here
                    # made this branch unreachable, so a mistyped $resource
                    # name degraded in total silence to those inline defaults
                    # -- an empty config only when none are declared.
                    # This line is the only signal an operator gets, so it
                    # distinguishes the two degradations: falling back to
                    # declared defaults is a config that still builds, while
                    # falling back to nothing is a factory about to be called
                    # with no arguments at all.
                    logger.warning(
                        "Resource '%s' of type '%s' not found in environment "
                        "'%s'; %s",
                        resource_name,
                        resource_type,
                        environment.name,
                        (
                            "falling back to its inline defaults"
                            if defaults
                            else "it declares no inline defaults, so this "
                            "resolves to an empty config"
                        ),
                    )
                    # Degrade to the inline defaults only -- matching what
                    # get_resource returned on this path all along. The
                    # unreachable branch this replaced fell back to the
                    # reference dict itself when there were no defaults,
                    # which would put the `$resource` / `type` marker keys
                    # into the resolved config and hand them to a factory as
                    # keyword arguments. Making the branch reachable must not
                    # also make its never-exercised return value live.
                    #
                    # Nothing overrides them here, so every default survives
                    # and every one is expanded. A degraded config is still
                    # config, so it gets the same $requires check and the
                    # same recursive walk as a found one.
                    resolved = self._resolve_source(
                        defaults, environment, substitute=substitute
                    )
                else:
                    # The two sources are walked separately and merged after,
                    # because they do not share a provenance. An environment
                    # loaded with substitution arrives expanded; inline
                    # defaults always arrive raw. Walking the merged result
                    # under one flag would either re-expand the environment's
                    # values or leave the defaults' nested refs raw.
                    env_needs_pass = substitute and not environment.substituted
                    resolved = self._resolve_source(
                        environment.get_resource(resource_type, resource_name),
                        environment,
                        substitute=env_needs_pass,
                    )
                    # Inline defaults fill gaps *after* the source above, and
                    # each is expanded only once it is known to survive -- so
                    # no value is handed to a second expansion, and none is
                    # expanded that the environment overrode.
                    for key, value in defaults.items():
                        if key not in resolved:
                            resolved[key] = self._resolve_source(
                                value, environment, substitute=substitute
                            )

                # Validate $requires against capabilities metadata
                if requires and isinstance(resolved, dict):
                    declared = resolved.get("capabilities")
                    if declared is not None:
                        missing = set(requires) - set(declared)
                        if missing:
                            from dataknobs_config.exceptions import (
                                ConfigError,
                            )

                            raise ConfigError(
                                f"Resource '{resource_name}' missing "
                                f"required capabilities: {sorted(missing)}. "
                                f"Declared: {declared}"
                            )

                # Each source was already walked as it was spliced, so the
                # merged result is fully resolved -- walking it again here
                # would be the second expansion this splits sources to avoid.
                return resolved
            else:
                # Regular dict - recurse into values
                return {
                    key: self._resolve_resource_refs(
                        value, environment, substitute=substitute
                    )
                    for key, value in config.items()
                }
        elif isinstance(config, list):
            # Recurse into list items
            return [
                self._resolve_resource_refs(
                    item, environment, substitute=substitute
                )
                for item in config
            ]
        else:
            # Return other types unchanged
            return config

    def _resolve_source(
        self,
        value: Any,
        environment: EnvironmentConfig,
        *,
        substitute: bool,
    ) -> Any:
        """Expand one source's ``${VAR}`` refs, then resolve its references.

        A splice merges two sources with different provenances, and the
        single ``substitute`` flag can only be true of one of them. So each
        is finished here, on its own terms, before the merge — rather than
        merged first and walked once under a flag that is wrong for half of
        the result.

        ``substitute`` says whether *this* source still needs expanding: an
        environment loaded with substitution does not, an inline default
        always does. The pass defers nested ``$resource`` defaults so that
        the walk below expands them at their own splice, exactly once.
        """
        if substitute:
            value = _substitute_deferring_defaults(value)
        return self._resolve_resource_refs(
            value, environment, substitute=substitute
        )

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
