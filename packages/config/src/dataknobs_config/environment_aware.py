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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataknobs_common.config_loading import (
    ConfigLoadError,
    ConfigPathEscapeError,
    find_config_file,
    load_yaml_or_json,
)

from .environment_config import EnvironmentConfig, ResourceNotFoundError
from .exceptions import ConfigError
from .inheritance import substitute_env_vars

logger = logging.getLogger(__name__)

#: Keys of a ``$resource`` block that select and constrain the resource
#: rather than supplying a default for it. Everything else in the block is an
#: inline default, kept raw by the entry pass and expanded at the splice.
#:
#: The set is **closed and enforced**: a ``$``-prefixed key outside it is a
#: malformed reference, not an inline default. It has to be, because the
#: comprehension that builds the defaults takes everything this set does not,
#: so a misspelled marker is otherwise promoted to a default and handed to a
#: factory as a keyword argument -- which is how ``$requred: true`` would
#: silently mean *not required*.
RESOURCE_MARKER_KEYS = frozenset({"$resource", "type", "$requires", "$required"})

#: The environment-level spelling of the policy, read through
#: :meth:`EnvironmentConfig.get_setting`. It is the only level of the chain a
#: deployment can reach when its references are generated at runtime: there is
#: no authored reference to annotate, and every other level lives in code the
#: operator does not deploy.
STRICT_RESOURCES_SETTING = "strict_resources"


def _validate_reference_markers(reference: Mapping[str, Any]) -> None:
    """Reject a ``$``-prefixed key in a reference block that is not a marker.

    Runs on both the found and the missing path. A malformed reference is
    malformed in every environment, and catching it only where the resource
    happens to be absent would surface it first in whichever deployment is
    least equipped to read the message.
    """
    unknown = sorted(
        key
        for key in reference
        if isinstance(key, str) and key.startswith("$") and key not in RESOURCE_MARKER_KEYS
    )
    if not unknown:
        return

    markers = ", ".join(sorted(key for key in RESOURCE_MARKER_KEYS if key.startswith("$")))
    raise ConfigError(
        f"Unknown marker key(s) {unknown} in the $resource reference for "
        f"'{reference.get('$resource')}'. A $-prefixed key must be one of: "
        f"{markers}. Everything else in the block is an inline default, so an "
        f"unrecognised marker would otherwise be passed to a factory as a "
        f"keyword argument rather than rejected."
    )


def _parse_required_marker(value: Any, *, where: str) -> bool:
    """Read a strictness flag as a bool, or as exactly ``true`` / ``false``.

    Strings are accepted because both spellings of the flag can arrive through
    ``${VAR}`` expansion, which does not coerce types -- ``$required:
    ${STRICT_BINDINGS}`` and ``strict_resources: ${STRICT_BINDINGS}`` both
    reach here as text.

    Anything else raises rather than falling back to ``False``. A value that
    silently reads as "off" is the same silent-degrade defect this vocabulary
    exists to close, so the match is explicit and never truthiness: ``1``,
    ``"yes"`` and ``"on"`` are errors, not strict-mode-off.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ConfigError(
        f"{where} must be a boolean or the string 'true'/'false', got {value!r}. "
        f"It is parsed explicitly rather than by truthiness, because a value "
        f"that silently read as 'false' would turn the policy off without saying so."
    )


def _reference_is_required(
    *,
    declared: bool | None,
    requires: Sequence[Any],
    resolver_default: bool | None,
    environment: EnvironmentConfig,
) -> tuple[bool, str]:
    """Decide whether a missing resource fails, and say which lever decided.

    The chain, most specific first. Each level is ``None``-means-defer, so
    "explicitly false" and "unspecified" stay distinguishable, and each level
    is the only one its owner can reach:

    1. the reference's ``$required``            -- the reference author
    2. its non-empty ``$requires``              -- the reference author
    3. ``strict_resources=`` on the resolution  -- the calling code, then the
       embedding application (collapsed by the caller, since both are code)
    4. the environment's ``strict_resources``   -- the operator
    5. ``False``                                -- unchanged default

    ``$requires`` sits *above* the code levels rather than below them because
    it is a claim about this reference in particular: a resolver-wide
    ``strict_resources=False`` is a statement about references that did not
    say anything, and this one did. Only the same author's ``$required:
    false`` overrides it -- "if it is there it must do X; it may be absent" is
    coherent, so it is expressible.

    Returns:
        ``(required, why)``. ``why`` is a phrase naming the level that made it
        strict, for the failure message; it is empty when ``required`` is
        False. An operator reading the failure needs to know which lever
        produced it in order to choose the right response.
    """
    if declared is not None:
        return (declared, "it declares `$required: true`" if declared else "")

    if requires:
        return (
            True,
            f"it declares `$requires: {list(requires)}`, which a resource that "
            f"is absent cannot satisfy",
        )

    if resolver_default is not None:
        if resolver_default:
            return (True, "this resolution ran with `strict_resources=True`")
        return (False, "")

    setting = environment.get_setting(STRICT_RESOURCES_SETTING)
    if setting is not None:
        strict = _parse_required_marker(
            setting,
            where=(f"Setting '{STRICT_RESOURCES_SETTING}' in environment '{environment.name}'"),
        )
        if strict:
            return (
                True,
                f"environment '{environment.name}' sets `{STRICT_RESOURCES_SETTING}: true`",
            )
        return (False, "")

    return (False, "")


@dataclass(frozen=True)
class UnresolvedResourceRef:
    """A ``$resource`` reference whose resource the environment does not define.

    Reported by :meth:`EnvironmentAwareConfig.find_unresolved_resources`, which
    surveys rather than resolves: an operator auditing a config tree wants
    every unresolvable reference in one pass, not one per run.

    Attributes:
        path: Dotted path to the reference within the config, list items
            spelled ``[0]`` -- e.g. ``bot.knowledge_base.vector_store``.
        resource_type: The reference's ``type`` (``"default"`` when absent).
        resource_name: The reference's ``$resource``, **after** ``${VAR}``
            expansion, so a reference that selects its resource by variable is
            reported under the name it would actually look up.
        required: The effective policy for this reference, per the precedence
            chain. ``True`` means a build would raise here rather than degrade.
        has_inline_defaults: Whether the reference declares any inline
            defaults. This distinguishes the two degradations: falling back to
            declared defaults is a config that still builds, while falling
            back to nothing is a factory about to be called with no arguments.
    """

    path: str
    resource_type: str
    resource_name: str
    required: bool
    has_inline_defaults: bool


def _substitute_deferring_defaults(config: Any, *, defer_defaults: bool = True) -> Any:
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

    A default's *key* is expanded here regardless, because what proves the
    default survived is the key: the splice asks ``if key not in resolved``.
    Deferring it would only move the same expansion to the splice, which
    expands every default's key there to ask the question. So an unset
    required ``${VAR}`` in key position does abort the build even when the
    environment supplies that key — a value's survival is decided by
    something else, and a key's is decided by itself.

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
                    substitute_env_vars(value) if key in RESOURCE_MARKER_KEYS else value
                )
            else:
                substituted[key] = _substitute_deferring_defaults(
                    value, defer_defaults=defer_defaults
                )
        return substituted
    if isinstance(config, list):
        return [
            _substitute_deferring_defaults(item, defer_defaults=defer_defaults) for item in config
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
        *,
        strict_resources: bool | None = None,
    ):
        """Initialize environment-aware configuration.

        Args:
            config: Application configuration dictionary
            environment: Environment configuration for resource resolution.
                        If None, auto-detects and loads environment.
            app_name: Optional name for this application config
            strict_resources: Whether a ``$resource`` reference naming a
                resource this environment does not define should raise instead
                of degrading to the reference's inline defaults. ``None``
                (default) defers to the environment's ``strict_resources``
                setting, then to ``False`` -- so the default behaviour is
                unchanged. This is the level for an application that hands its
                config to a library which calls
                :meth:`resolve_for_build` itself; a reference's own
                ``$required`` still overrides it.
        """
        self._config = config
        self._environment = environment or EnvironmentConfig.load()
        self._app_name = app_name or config.get("name")
        self._strict_resources = strict_resources

    @property
    def environment(self) -> EnvironmentConfig:
        """Get the current environment configuration."""
        return self._environment

    @property
    def strict_resources(self) -> bool | None:
        """The instance-level missing-resource policy, or None to defer."""
        return self._strict_resources

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
        *,
        allow_outside: bool = False,
        strict_resources: bool | None = None,
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
            allow_outside: Opt out of the containment bound, for a deployment
                whose layout genuinely spans sibling trees. Off by default.
                It applies to **both** lookups this method performs -- the
                app name against ``app_dir`` and the environment name against
                ``env_dir`` -- because a layout that spans trees on one side
                generally does on the other, and a flag that silently covered
                only one would be worse than none. Note what that widens: with
                ``environment=None`` the environment name comes from
                ``DATAKNOBS_ENVIRONMENT`` or ``ENVIRONMENT``. An escaping name
                is logged at WARNING when it escapes.
            strict_resources: Missing-resource policy for this config, per
                :meth:`__init__`. ``None`` leaves today's behaviour unchanged.

        Returns:
            EnvironmentAwareConfig with both app and environment loaded

        Raises:
            EnvironmentAwareConfigError: If app config not found or invalid
        """
        app_dir = Path(app_dir)
        env_config = EnvironmentConfig.load(environment, env_dir, allow_outside=allow_outside)

        # Find and load app config file
        config_path = cls._find_config_file(app_dir, app_name, allow_outside=allow_outside)
        if config_path is None:
            raise EnvironmentAwareConfigError(
                f"Application config not found: {app_name}.yaml in {app_dir}"
            )

        config = cls._load_file(config_path)

        logger.info(f"Loaded app config '{app_name}' for environment '{env_config.name}'")

        return cls(
            config=config,
            environment=env_config,
            app_name=app_name,
            strict_resources=strict_resources,
        )

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        environment: str | None = None,
        env_dir: str | Path = "config/environments",
        *,
        strict_resources: bool | None = None,
    ) -> EnvironmentAwareConfig:
        """Create from a configuration dictionary.

        Args:
            config: Application configuration dictionary
            environment: Environment name, or None to auto-detect
            env_dir: Directory containing environment configs
            strict_resources: Missing-resource policy, per :meth:`__init__`

        Returns:
            EnvironmentAwareConfig instance
        """
        env_config = EnvironmentConfig.load(environment, env_dir)
        return cls(config=config, environment=env_config, strict_resources=strict_resources)

    @classmethod
    def _find_config_file(
        cls, config_dir: Path, name: str, *, allow_outside: bool = False
    ) -> Path | None:
        """Find a config file by name.

        The name is bounded by ``config_dir``: it may address a subdirectory,
        but one that *lands* outside -- whether spelled with ``..`` or as an
        absolute path -- raises rather than reading the file it points at.

        Args:
            config_dir: Directory to search
            name: Config name (without extension)

        Returns:
            Path to config file, or None if not found

        Raises:
            EnvironmentAwareConfigError: If ``name`` addresses a file outside
                ``config_dir`` and ``allow_outside`` is False
        """
        try:
            return find_config_file(config_dir, name, allow_outside=allow_outside)
        except ConfigPathEscapeError as e:
            raise EnvironmentAwareConfigError(str(e)) from e

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
            raise EnvironmentAwareConfigError(f"Failed to load app config {path}: {e}") from e
        except OSError as e:
            raise EnvironmentAwareConfigError(f"Failed to read app config {path}: {e}") from e

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
        *,
        strict_resources: bool | None = None,
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
            strict_resources: Whether a reference naming a resource this
                environment does not define raises rather than degrading to
                the reference's inline defaults. ``None`` (default) defers to
                the instance default, then to the environment's
                ``strict_resources`` setting, then to ``False``. A reference's
                own ``$required`` overrides every level.

                ``resolve_for_build(strict_resources=True)`` **is** the
                startup preflight: it resolves without constructing anything,
                so it is safe to run at boot purely to prove every binding
                this config names exists in this environment. Use
                :meth:`find_unresolved_resources` instead to get every failure
                in one pass rather than the first one.

        Returns:
            Fully resolved configuration dictionary

        Raises:
            ResourceNotFoundError: If a reference names a resource this
                environment does not define and the effective policy is strict
            ConfigError: If a reference is malformed -- an unknown
                ``$``-prefixed marker key, or an unparseable ``$required``
        """
        # Get the base configuration. Annotated because every step below —
        # ``get``, the substitution pass, the resource splice — walks an
        # arbitrary config tree and so is declared ``Any``; this method is
        # where the shape is actually promised.
        config: dict[str, Any]
        if config_key:
            config = self.get(config_key)
            if config is None:
                raise EnvironmentAwareConfigError(f"Config key not found: {config_key}")
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
            config = _substitute_deferring_defaults(config, defer_defaults=resolve_resources)

        # Resolve logical resource references. Each resource is substituted
        # as it is spliced, not the environment as a whole: a resource is
        # still separable at the splice point, which is the latest point it
        # can be expanded, and expanding the whole environment would read
        # values no reference names -- so an unset required ${VAR} in an
        # unrelated resource would abort a build that never looked at it.
        if resolve_resources:
            config = self._resolve_resource_refs(
                config,
                self._environment,
                substitute=resolve_env_vars,
                strict_resources=self._effective_strict(strict_resources),
            )

        return config

    def _effective_strict(self, strict_resources: bool | None) -> bool | None:
        """Collapse the two *code* levels of the precedence chain into one.

        Both are code -- one owned by the caller of a single resolution, the
        other by the application embedding this config -- so a failure message
        that distinguished them would name a distinction its reader cannot
        act on differently. They collapse here rather than deeper so the
        recursion threads a single value; the levels that remain distinguishable
        (the reference's own marker, and the operator's environment setting)
        are read where they live.
        """
        return strict_resources if strict_resources is not None else self._strict_resources

    def _resolve_resource_refs(
        self,
        config: Any,
        environment: EnvironmentConfig | None = None,
        substitute: bool = False,
        strict_resources: bool | None = None,
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
            strict_resources: The collapsed code-level policy for a missing
                resource, per :meth:`_effective_strict`. Forwarded through
                every recursion below: a missed forward is silent, reverting a
                nested reference to leniency inside an otherwise strict
                resolution.

        Returns:
            Configuration with resource references resolved
        """
        if environment is None:
            environment = self._environment

        if isinstance(config, dict):
            if "$resource" in config:
                # This is a resource reference
                _validate_reference_markers(config)
                resource_name = config["$resource"]
                resource_type = config.get("type", "default")

                # Get defaults from the reference (exclude markers and metadata)
                defaults = {k: v for k, v in config.items() if k not in RESOURCE_MARKER_KEYS}
                requires = config.get("$requires", [])
                # Parsed here rather than on the missing path, for the reason
                # the marker guard runs here: a malformed value is malformed
                # in every environment, and deferring the parse would surface
                # it first in whichever deployment lacks the resource.
                declared_required = (
                    _parse_required_marker(
                        config["$required"],
                        where=f"`$required` on the reference to '{resource_name}'",
                    )
                    if "$required" in config
                    else None
                )

                if not environment.has_resource(resource_type, resource_name):
                    required, why = _reference_is_required(
                        declared=declared_required,
                        requires=requires,
                        resolver_default=strict_resources,
                        environment=environment,
                    )
                    if required:
                        raise ResourceNotFoundError(
                            f"Resource '{resource_name}' of type "
                            f"'{resource_type}' not found in environment "
                            f"'{environment.name}', and {why}"
                        )
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
                        "Resource '%s' of type '%s' not found in environment '%s'; %s",
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
                        defaults,
                        environment,
                        substitute=substitute,
                        strict_resources=strict_resources,
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
                        strict_resources=strict_resources,
                    )
                    # Inline defaults fill gaps *after* the source above, and
                    # each is expanded only once it is known to survive -- so
                    # no value is handed to a second expansion, and none is
                    # expanded that the environment overrode.
                    for key, value in defaults.items():
                        if key not in resolved:
                            resolved[key] = self._resolve_source(
                                value,
                                environment,
                                substitute=substitute,
                                strict_resources=strict_resources,
                            )

                # Validate $requires against capabilities metadata. On the
                # degraded path this is now reached only when the reference
                # declared `$required: false` -- absence otherwise fails
                # above, since a resource that is not there satisfies no
                # capability at all. Where the author did opt out, the check
                # still runs against whatever capabilities the inline defaults
                # declare: "it may be absent" is not "and anything will do".
                if requires and isinstance(resolved, dict):
                    declared_capabilities = resolved.get("capabilities")
                    if declared_capabilities is not None:
                        missing = set(requires) - set(declared_capabilities)
                        if missing:
                            raise ConfigError(
                                f"Resource '{resource_name}' missing "
                                f"required capabilities: {sorted(missing)}. "
                                f"Declared: {declared_capabilities}"
                            )

                # Each source was already walked as it was spliced, so the
                # merged result is fully resolved -- walking it again here
                # would be the second expansion this splits sources to avoid.
                return resolved
            else:
                # Regular dict - recurse into values
                return {
                    key: self._resolve_resource_refs(
                        value,
                        environment,
                        substitute=substitute,
                        strict_resources=strict_resources,
                    )
                    for key, value in config.items()
                }
        elif isinstance(config, list):
            # Recurse into list items
            return [
                self._resolve_resource_refs(
                    item,
                    environment,
                    substitute=substitute,
                    strict_resources=strict_resources,
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
        strict_resources: bool | None = None,
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

        ``strict_resources`` is carried through unchanged: a reference nested
        inside a resource, or inside an inline default, is as much a binding
        as the one that spliced it in, and dropping the policy here would make
        strictness stop one level down without saying so.
        """
        if substitute:
            value = _substitute_deferring_defaults(value)
        return self._resolve_resource_refs(
            value, environment, substitute=substitute, strict_resources=strict_resources
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
            # Carried, not re-defaulted. This method is on the common path for
            # a caller that supplies a config and an environment separately,
            # so dropping the policy here would silently revert strict mode to
            # lenient at precisely the point a second environment enters --
            # which is the point the policy is most likely to matter.
            strict_resources=self._strict_resources,
        )

    def find_unresolved_resources(
        self,
        config_key: str | None = None,
        strict_resources: bool | None = None,
    ) -> list[UnresolvedResourceRef]:
        """Every ``$resource`` reference whose resource this environment lacks.

        Raise-on-first is right for a build and wrong for a preflight: an
        operator auditing a config tree wants every unresolvable reference in
        one pass, not one per run. This constructs nothing and raises nothing
        for a missing resource.

        References nested inside a resolved resource are surveyed too, since
        those are bindings a build would reach; a resource that refers to
        itself is visited once rather than followed round.

        **An empty list means no reference names an absent resource -- not
        that the build will succeed.** A resource that is present but does not
        declare a capability its reference ``$requires`` still fails at build
        time, and is not reported here: this surveys *presence*, which is the
        one question answerable without resolving anything. Use
        ``resolve_for_build(strict_resources=True)`` for the complete check.

        Args:
            config_key: Specific config key to survey, or None for the whole
                config
            strict_resources: Policy used to populate
                :attr:`UnresolvedResourceRef.required`, per
                :meth:`resolve_for_build`. It does not affect *which*
                references are reported -- every unresolvable one is, whatever
                the policy -- only whether each is reported as fatal.

        Returns:
            One entry per unresolvable reference, depth-first in config order

        Raises:
            ConfigError: If a reference is malformed. A survey that skipped
                malformed references would report a tree as sound while the
                build that follows raises on it.
        """
        config: Any
        if config_key:
            config = self.get(config_key)
            if config is None:
                raise EnvironmentAwareConfigError(f"Config key not found: {config_key}")
        else:
            config = copy.deepcopy(self._config)

        # Expand first, so a reference selecting its resource by variable is
        # reported under the name it would actually look up. The raw
        # `${LLM_BINDING}` text would be a finding nobody can act on.
        config = _substitute_deferring_defaults(config)

        found: list[UnresolvedResourceRef] = []
        self._survey_resource_refs(
            config,
            path="",
            resolver_default=self._effective_strict(strict_resources),
            seen=set(),
            found=found,
        )
        return found

    def _survey_resource_refs(
        self,
        config: Any,
        *,
        path: str,
        resolver_default: bool | None,
        seen: set[tuple[str, str]],
        found: list[UnresolvedResourceRef],
    ) -> None:
        """Walk the config collecting unresolvable references. Never builds."""
        environment = self._environment

        if isinstance(config, dict):
            if "$resource" in config:
                _validate_reference_markers(config)
                resource_name = config["$resource"]
                resource_type = config.get("type", "default")
                defaults = {k: v for k, v in config.items() if k not in RESOURCE_MARKER_KEYS}
                declared_required = (
                    _parse_required_marker(
                        config["$required"],
                        where=f"`$required` on the reference to '{resource_name}'",
                    )
                    if "$required" in config
                    else None
                )

                if not environment.has_resource(resource_type, resource_name):
                    required, _why = _reference_is_required(
                        declared=declared_required,
                        requires=config.get("$requires", []),
                        resolver_default=resolver_default,
                        environment=environment,
                    )
                    found.append(
                        UnresolvedResourceRef(
                            path=path,
                            resource_type=resource_type,
                            resource_name=resource_name,
                            required=required,
                            has_inline_defaults=bool(defaults),
                        )
                    )
                    # The defaults are what a lenient build would resolve to,
                    # so a reference nested among them is still reachable.
                    self._survey_children(
                        _substitute_deferring_defaults(defaults),
                        path=path,
                        resolver_default=resolver_default,
                        seen=seen,
                        found=found,
                    )
                    return

                marker = (resource_type, resource_name)
                if marker in seen:
                    return
                seen.add(marker)
                # Each source is expanded on its own terms before it is
                # surveyed, mirroring :meth:`_resolve_source`. Inline defaults
                # always arrive raw, because the entry pass holds them back
                # for the splice; an environment that recorded itself as
                # substituted arrives expanded already. Skipping either would
                # report a variable-selected reference under raw `${VAR}` text
                # while the build looks up the name it expands to -- a survey
                # naming something other than what fails is worse than none.
                resource = environment.get_resource(resource_type, resource_name)
                if not environment.substituted:
                    resource = _substitute_deferring_defaults(resource)
                self._survey_children(
                    resource,
                    path=path,
                    resolver_default=resolver_default,
                    seen=seen,
                    found=found,
                )
                self._survey_children(
                    _substitute_deferring_defaults(defaults),
                    path=path,
                    resolver_default=resolver_default,
                    seen=seen,
                    found=found,
                )
                return

            self._survey_children(
                config, path=path, resolver_default=resolver_default, seen=seen, found=found
            )
        elif isinstance(config, list):
            for index, item in enumerate(config):
                self._survey_resource_refs(
                    item,
                    path=f"{path}[{index}]",
                    resolver_default=resolver_default,
                    seen=seen,
                    found=found,
                )

    def _survey_children(
        self,
        config: Any,
        *,
        path: str,
        resolver_default: bool | None,
        seen: set[tuple[str, str]],
        found: list[UnresolvedResourceRef],
    ) -> None:
        """Survey each value of a mapping, extending the dotted path."""
        if not isinstance(config, dict):
            return
        for key, value in config.items():
            self._survey_resource_refs(
                value,
                path=f"{path}.{key}" if path else str(key),
                resolver_default=resolver_default,
                seen=seen,
                found=found,
            )

    def get_resource(
        self,
        resource_type: str,
        logical_name: str,
        defaults: dict[str, Any] | None = None,
        *,
        required: bool | None = None,
    ) -> dict[str, Any]:
        """Get a resolved resource configuration.

        Convenience method to directly access environment resources.

        Args:
            resource_type: Type of resource
            logical_name: Logical name of resource
            defaults: Default values if resource not found
            required: Whether an absent resource raises, independently of
                whether ``defaults`` were supplied. See
                :meth:`EnvironmentConfig.get_resource`. Note that this is the
                *direct-access* policy: a ``$resource`` reference resolved
                through :meth:`resolve_for_build` decides its own, and is not
                affected by this parameter.

        Returns:
            Resolved resource configuration
        """
        return self._environment.get_resource(
            resource_type, logical_name, defaults, required=required
        )

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
