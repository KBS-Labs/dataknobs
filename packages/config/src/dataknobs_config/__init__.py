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
    EnvironmentAwareConfig,
    EnvironmentAwareConfigError,
)
from .environment_config import (
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

__version__ = "0.3.11"
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
    # Binding resolver
    "ConfigBindingResolver",
    "BindingResolverError",
    "FactoryNotFoundError",
    "SimpleFactory",
    "CallableFactory",
    "AsyncCallableFactory",
]
