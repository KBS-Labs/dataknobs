"""Example service classes for configuration."""

from typing import Any, Dict, Optional

from dataknobs_config import ConfigurableBase


class ServiceManager(ConfigurableBase):
    """Example service manager.

    Demonstrates a more complex configurable class that manages
    multiple services.
    """

    def __init__(self, name: str, **kwargs):
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

    _instance: Optional["ServiceRegistry"] = None

    def __new__(cls, **kwargs):
        """Ensure single instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
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
