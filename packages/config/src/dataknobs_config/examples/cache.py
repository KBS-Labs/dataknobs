"""Example cache classes for configuration."""

from typing import Any, Dict


class Cache:
    """Example cache implementation.

    This demonstrates a class that doesn't inherit from ConfigurableBase
    but can still be instantiated via configuration.
    """

    def __init__(self, host: str, port: int, ttl: int = 3600, **kwargs):
        """Initialize cache.

        Args:
            host: Cache server host
            port: Cache server port
            ttl: Time to live in seconds
            **kwargs: Additional options
        """
        self.host = host
        self.port = port
        self.ttl = ttl
        self.prefix = kwargs.get("prefix", "")
        self.extra = {k: v for k, v in kwargs.items() if k != "prefix"}
        self._storage: Dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        full_key = f"{self.prefix}{key}" if self.prefix else key
        return self._storage.get(full_key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        full_key = f"{self.prefix}{key}" if self.prefix else key
        self._storage[full_key] = value

    def clear(self) -> None:
        """Clear all cache entries."""
        self._storage.clear()


class CacheFactory:
    """Example cache factory using callable pattern.

    This demonstrates a factory that uses __call__ instead of
    inheriting from FactoryBase.
    """

    def __init__(self):
        """Initialize factory."""
        self.created_count = 0
        self.default_ttl = 3600

    def __call__(self, **config):
        """Create a cache instance.

        Args:
            **config: Cache configuration

        Returns:
            Cache instance
        """
        # Add factory defaults
        config.setdefault("ttl", self.default_ttl)

        # Track creation
        self.created_count += 1

        # Create and return instance
        return Cache(**config)


def create_cache(**config):
    """Function factory for creating cache instances.

    This demonstrates using a simple function as a factory.

    Args:
        **config: Cache configuration

    Returns:
        Cache instance
    """
    # Add function-specific defaults
    config.setdefault("ttl", 7200)
    config.setdefault("prefix", "app:")

    return Cache(**config)
