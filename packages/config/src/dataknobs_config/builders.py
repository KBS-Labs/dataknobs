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
            raise ConfigError(f"Failed to instantiate {class_path}: {e}") from e

    def _load_factory(self, factory_path: str) -> Any:
        """Resolve a factory object from a registered name or module path.

        Shared by the sync and async factory-build paths so they cannot
        drift in how they locate a factory.

        Args:
            factory_path: Registered factory name or dotted module path.

        Returns:
            The factory object (instance, class, or callable).
        """
        # Check registered factories first (if they exist)
        if hasattr(self._config, '_registered_factories') and self._config._registered_factories.has(factory_path):
            return self._config._registered_factories.get(factory_path)
        # Fall back to loading as a module path
        factory_cls = self._load_class(factory_path)
        # Create factory instance
        try:
            return factory_cls()
        except TypeError:
            # Factory might be a module-level function
            return factory_cls

    def _build_with_factory(self, config: dict) -> Any:
        """Build an object using a factory class.

        Args:
            config: Configuration dictionary with 'factory' attribute

        Returns:
            Object instance
        """
        factory_path = config.pop("factory")

        # Remove metadata attributes
        config.pop("type", None)
        config.pop("name", None)

        factory = self._load_factory(factory_path)

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

    async def build_async(
        self, ref: str, cache: bool = True, **kwargs: Any
    ) -> Any:
        """Build an object asynchronously from a configuration reference.

        Mirror of :meth:`build` for targets whose construction is async.
        A target class that defines ``from_config_async`` (the
        ``StructuredConfigConsumer`` async entry point) or a factory that
        defines ``create_async`` is awaited; anything else falls back to
        the synchronous construction path, so this is safe to call for
        any reference.

        Args:
            ref: String reference to configuration
            cache: Whether to cache the built object
            **kwargs: Additional keyword arguments for construction

        Returns:
            Built object instance
        """
        if cache and ref in self._cache:
            return self._cache[ref]

        config = self._config.resolve_reference(ref)
        obj = await self._build_from_config_async(config, **kwargs)

        if cache:
            self._cache[ref] = obj

        return obj

    async def _build_from_config_async(
        self, config: dict, **kwargs: Any
    ) -> Any:
        """Async counterpart of :meth:`_build_from_config`."""
        config = copy.deepcopy(config)
        config.update(kwargs)

        if "factory" in config:
            return await self._build_with_factory_async(config)
        if "class" in config:
            return await self._build_with_class_async(config)

        raise ConfigError(
            "Configuration must specify either 'class' or 'factory' for object construction"
        )

    async def _build_with_class_async(self, config: dict) -> Any:
        """Async counterpart of :meth:`_build_with_class`.

        Prefers ``cls.from_config_async`` (awaited) when the target
        defines it, then falls back to the synchronous ``from_config`` /
        direct-instantiation path.
        """
        class_path = config.pop("class")
        cls = self._load_class(class_path)

        config.pop("type", None)
        config.pop("name", None)

        if hasattr(cls, "from_config_async"):
            return await cls.from_config_async(config)
        if hasattr(cls, "from_config"):
            return cls.from_config(config)
        try:
            return cls(**config)
        except TypeError as e:
            raise ConfigError(f"Failed to instantiate {class_path}: {e}") from e

    async def _build_with_factory_async(self, config: dict) -> Any:
        """Async counterpart of :meth:`_build_with_factory`.

        Prefers ``factory.create_async`` (awaited) when present, then
        falls back to the synchronous factory methods.
        """
        factory_path = config.pop("factory")

        config.pop("type", None)
        config.pop("name", None)

        factory = self._load_factory(factory_path)

        if hasattr(factory, "create_async"):
            return await factory.create_async(**config)
        if hasattr(factory, "create"):
            return factory.create(**config)
        if hasattr(factory, "build"):
            return factory.build(**config)
        if callable(factory):
            return factory(**config)
        raise ConfigError(
            f"Factory {factory_path} must have 'create', 'build', "
            "'create_async' method or be callable"
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
            raise ConfigError(f"Failed to import {class_path}: {e}") from e
        except Exception as e:
            raise ConfigError(f"Failed to load class {class_path}: {e}") from e

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

    .. deprecated::
        Prefer
        :class:`~dataknobs_common.structured_config.StructuredConfigConsumer`
        for new code. ``ConfigurableBase`` performs kwarg-splat
        construction (``cls(**config)``);
        ``StructuredConfigConsumer[ConfigT]`` provides typed-dispatch
        construction with auto-derived ``from_dict``, a
        ``_normalize_dict`` override hook, and a unified parity guard.

        Existing consumers continue to work; no runtime warning is
        raised so the transition stays quiet across the multi-cycle
        migration. Removal is scheduled for a future release once the
        in-tree migration is complete.

    Classes that inherit from this can implement custom configuration
    loading logic.
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
