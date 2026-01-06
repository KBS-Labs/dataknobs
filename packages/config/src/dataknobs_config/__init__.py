"""DataKnobs Config Package

A modular, reusable configuration system for composable settings.
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
    deep_merge,
    load_config_with_inheritance,
    substitute_env_vars,
)
from .substitution import VariableSubstitution

__version__ = "0.3.3"
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
    "deep_merge",
    "load_config_with_inheritance",
    "substitute_env_vars",
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
