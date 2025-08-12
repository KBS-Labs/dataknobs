"""Optional object construction and caching functionality."""

import copy
import importlib
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from .config import Config

from .exceptions import ConfigError, ValidationError


class ObjectBuilder:
    """Handles object construction from configurations.

    Supports:
        - Direct class instantiation via 'class' attribute
        - Factory pattern via 'factory' attribute
        - Object caching
    """

    def __init__(self, config_instance: "Config") -> None:
        """Initialize the object builder.

        Args:
            config_instance: The Config instance to build objects from
        """
        self._config = config_instance
        self._cache: Dict[str, Any] = {}

    def build(self, ref: str, cache: bool = True, **kwargs: Any) -> Any:
        """Build an object from a configuration reference.

        Args:
            ref: String reference to configuration
            cache: Whether to cache the built object
            **kwargs: Additional keyword arguments for construction

        Returns:
            Built object instance

        Raises:
            ConfigError: If object cannot be built
        """
        # Check cache first
        if cache and ref in self._cache:
            return self._cache[ref]

        # Resolve the configuration
        config = self._config.resolve_reference(ref)

        # Build the object
        obj = self._build_from_config(config, **kwargs)

        # Cache if requested
        if cache:
            self._cache[ref] = obj

        return obj

    def _build_from_config(self, config: dict, **kwargs: Any) -> Any:
        """Build an object from a configuration dictionary.

        Args:
            config: Configuration dictionary
            **kwargs: Additional keyword arguments

        Returns:
            Built object instance
        """
        # Make a copy to avoid modifying original
        config = copy.deepcopy(config)

        # Merge additional kwargs
        config.update(kwargs)

        # Check for factory attribute
        if "factory" in config:
            return self._build_with_factory(config)

        # Check for class attribute
        if "class" in config:
            return self._build_with_class(config)

        # No construction method specified
        raise ConfigError(
            "Configuration must specify either 'class' or 'factory' for object construction"
        )

    def _build_with_class(self, config: dict) -> Any:
        """Build an object using direct class instantiation.

        Args:
            config: Configuration dictionary with 'class' attribute

        Returns:
            Object instance
        """
        class_path = config.pop("class")

        # Load the class
        cls = self._load_class(class_path)

        # Remove metadata attributes
        config.pop("type", None)
        config.pop("name", None)

        # Check if class has from_config method
        if hasattr(cls, "from_config"):
            return cls.from_config(config)

        # Otherwise use direct instantiation
        try:
            return cls(**config)
        except TypeError as e:
            raise ConfigError(f"Failed to instantiate {class_path}: {e}")

    def _build_with_factory(self, config: dict) -> Any:
        """Build an object using a factory class.

        Args:
            config: Configuration dictionary with 'factory' attribute

        Returns:
            Object instance
        """
        factory_path = config.pop("factory")

        # Load the factory class
        factory_cls = self._load_class(factory_path)

        # Remove metadata attributes
        config.pop("type", None)
        config.pop("name", None)

        # Create factory instance
        try:
            factory = factory_cls()
        except TypeError:
            # Factory might be a module-level function
            factory = factory_cls

        # Check for standard factory methods
        if hasattr(factory, "create"):
            return factory.create(**config)
        elif hasattr(factory, "build"):
            return factory.build(**config)
        elif callable(factory):
            return factory(**config)
        else:
            raise ConfigError(
                f"Factory {factory_path} must have 'create', 'build' method or be callable"
            )

    def _load_class(self, class_path: str) -> Type[Any]:
        """Load a class from a module path.

        Args:
            class_path: Full path to class (e.g., "mymodule.MyClass")

        Returns:
            Class object

        Raises:
            ConfigError: If class cannot be loaded
        """
        try:
            # Split module and class name
            if "." in class_path:
                module_path, class_name = class_path.rsplit(".", 1)
            else:
                raise ValidationError(f"Invalid class path: {class_path}")

            # Import module
            module = importlib.import_module(module_path)

            # Get class from module
            if not hasattr(module, class_name):
                raise ConfigError(f"Class {class_name} not found in {module_path}")

            cls: Type[Any] = getattr(module, class_name)
            return cls

        except ImportError as e:
            raise ConfigError(f"Failed to import {class_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load class {class_path}: {e}")

    def clear_cache(self, ref: str | None = None) -> None:
        """Clear cached objects.

        Args:
            ref: Specific reference to clear, or None to clear all
        """
        if ref:
            self._cache.pop(ref, None)
        else:
            self._cache.clear()

    def get_cached(self, ref: str) -> Any | None:
        """Get a cached object without building.

        Args:
            ref: String reference

        Returns:
            Cached object or None
        """
        return self._cache.get(ref)


class ConfigurableBase:
    """Base class for objects that can be configured.

    Classes that inherit from this can implement custom
    configuration loading logic.
    """

    @classmethod
    def from_config(cls, config: dict) -> "ConfigurableBase":
        """Create an instance from a configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Instance of the class
        """
        return cls(**config)


class FactoryBase:
    """Base class for factory objects.

    Factories that inherit from this should implement
    the create method.
    """

    def create(self, **config: Any) -> Any:
        """Create an object from configuration.

        Args:
            **config: Configuration parameters

        Returns:
            Created object
        """
        raise NotImplementedError("Subclasses must implement create method")
