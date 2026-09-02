"""DataKnobs Config Package

A modular, reusable configuration system for composable settings.

Environment variable substitution is provided by
:func:`substitute_env_vars` (canonical helper). It supports the bash
superset ``${VAR}`` / ``${VAR:default}`` / ``${VAR:-default}`` /
``${VAR:?error_msg}`` and three keyword-only options
(``type_coerce``, ``expand_user_paths``, ``substitute_keys``). The
:class:`VariableSubstitution` class is a deprecated thin shim over
``substitute_env_vars(data, type_coerce=True, expand_user_paths=False,
substitute_keys=False)`` and emits ``DeprecationWarning`` on
construction; new code should use ``substitute_env_vars`` directly.

:class:`ConfigurableBase` is soft-deprecated in favor of
:class:`dataknobs_common.structured_config.StructuredConfigConsumer`.
New code should adopt the typed-dispatch successor; existing
``ConfigurableBase`` consumers continue to work, and no runtime
warning is raised so the migration can proceed quietly across
multiple release cycles.
"""

from .binding_resolver import (
    AsyncCallableFactory,
    BindingResolverError,
    CallableFactory,
    ConfigBindingResolver,
    FactoryNotFoundError,
    SimpleFactory,
)
from .builders import ConfigurableBase, FactoryBase
from .config import Config
from .environment_aware import (
    RESOURCE_MARKER_KEYS,
    EnvironmentAwareConfig,
    EnvironmentAwareConfigError,
    MarkerViolation,
    UnresolvedResourceRef,
    collect_marker_violations,
    resolve_resource_references,
)
from .environment_config import (
    STRICT_RESOURCES_SETTING,
    EnvironmentConfig,
    EnvironmentConfigError,
    ResourceBinding,
    ResourceNotFoundError,
)
from .exceptions import (
    ConfigError,
    ConfigNotFoundError,
    InvalidReferenceError,
    ValidationError,
)
from .inheritance import (
    InheritableConfigLoader,
    InheritanceError,
    RequiredEnvVarError,
    deep_merge,
    load_config_with_inheritance,
    substitute_env_vars,
)
from .substitution import VariableSubstitution
from .template_vars import substitute_template_vars

__version__ = "0.7.1"
__all__ = [
    "Config",
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigurableBase",
    "FactoryBase",
    "InvalidReferenceError",
    "ValidationError",
    "VariableSubstitution",
    # Inheritance utilities
    "InheritableConfigLoader",
    "InheritanceError",
    "RequiredEnvVarError",
    "deep_merge",
    "load_config_with_inheritance",
    "substitute_env_vars",
    # Template variable substitution
    "substitute_template_vars",
    # Environment-aware configuration
    "EnvironmentConfig",
    "EnvironmentConfigError",
    "ResourceBinding",
    "ResourceNotFoundError",
    "EnvironmentAwareConfig",
    "EnvironmentAwareConfigError",
    # Resource reference vocabulary. `resolve_resource_references` is the
    # first of these to reach for: a consumer holding a config tree and an
    # environment should resolve with it rather than write a third reader of
    # the format, which is how one arrived with its markers unvalidated, its
    # inline defaults dropped and a fallback branch that could not be reached.
    #
    # A reader that genuinely cannot resolve -- a validator, an editor, a
    # config-authoring tool with no environment in hand -- reaches for
    # `collect_marker_violations`, which applies the same marker rule to an
    # authored tree and reports rather than raises. The marker set itself is
    # the last resort, for a reader that needs the literals: offering only the
    # set is what left the one validator that asked to transcribe the rule
    # around it, and a transcription of a rule drifts even when the set cannot.
    "resolve_resource_references",
    "collect_marker_violations",
    "MarkerViolation",
    "RESOURCE_MARKER_KEYS",
    "STRICT_RESOURCES_SETTING",
    "UnresolvedResourceRef",
    # Binding resolver
    "ConfigBindingResolver",
    "BindingResolverError",
    "FactoryNotFoundError",
    "SimpleFactory",
    "CallableFactory",
    "AsyncCallableFactory",
]
