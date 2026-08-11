"""Example service classes for configuration."""

from typing import Any, ClassVar, Dict, Self

from dataknobs_config import ConfigurableBase


class ServiceManager(ConfigurableBase):
    """Example service manager.

    Demonstrates a more complex configurable class that manages
    multiple services.
    """

    def __init__(self, name: str, **kwargs: Any):
        """Initialize service manager.

        Args:
            name: Manager name
            **kwargs: Additional configuration
        """
        self.name = name
        self.services: Dict[str, Any] = {}
        self.config = kwargs
        self.auto_start = kwargs.get("auto_start", False)
        self.max_retries = kwargs.get("max_retries", 3)

    def register_service(self, name: str, service: Any) -> None:
        """Register a service."""
        self.services[name] = service

    def get_service(self, name: str) -> Any | None:
        """Get a registered service."""
        return self.services.get(name)

    def start_all(self) -> Dict[str, str]:
        """Start all services."""
        results = {}
        for name in self.services:
            results[name] = f"Started {name}"
        return results


class ServiceRegistry:
    """Example service registry.

    Demonstrates a singleton-like pattern for service registration.
    """

    #: One instance per class, not one for the hierarchy. A single shared slot
    #: hands a subclass the base class's instance — which is why annotating
    #: ``__new__`` at all made both checkers object, mypy to the return type and
    #: ruff to the pattern. This is example code, so it demonstrates the form
    #: that survives being subclassed.
    _instances: ClassVar[dict[type, Any]] = {}

    def __new__(cls, **kwargs: Any) -> Self:
        """Ensure single instance."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
        instance: Self = cls._instances[cls]
        return instance

    def __init__(self, **kwargs: Any):
        """Initialize registry."""
        if not hasattr(self, "initialized"):
            self.services: Dict[str, Any] = {}
            self.config = kwargs
            self.initialized = True

    def register(self, name: str, service: Any) -> None:
        """Register a service."""
        self.services[name] = service

    def get(self, name: str) -> Any | None:
        """Get a service."""
        return self.services.get(name)

    def list_services(self) -> list:
        """List all registered services."""
        return list(self.services.keys())
