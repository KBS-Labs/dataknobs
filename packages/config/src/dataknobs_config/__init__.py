"""DataKnobs Config Package

A modular, reusable configuration system for composable settings.
"""

from .builders import ConfigurableBase, FactoryBase
from .config import Config
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

__version__ = "0.3.1"
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
]
