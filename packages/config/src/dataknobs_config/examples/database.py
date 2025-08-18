"""Example database classes for configuration."""

from dataknobs_config import ConfigurableBase, FactoryBase


class Database(ConfigurableBase):
    """Example database connection class.

    This demonstrates using ConfigurableBase for automatic configuration
    integration.
    """

    def __init__(self, host: str, port: int, **kwargs):
        """Initialize database connection.

        Args:
            host: Database host
            port: Database port
            **kwargs: Additional configuration options
        """
        self.host = host
        self.port = port
        self.database = kwargs.get("database", "default")
        self.pool_size = kwargs.get("pool_size", 10)
        self.timeout = kwargs.get("timeout", 30)
        self.extra = {
            k: v for k, v in kwargs.items() if k not in ["database", "pool_size", "timeout"]
        }
        self._connected = False

    def connect(self):
        """Simulate database connection."""
        self._connected = True
        return f"Connected to {self.host}:{self.port}/{self.database}"

    def disconnect(self):
        """Simulate database disconnection."""
        self._connected = False
        return "Disconnected"

    @property
    def is_connected(self):
        """Check if database is connected."""
        return self._connected


class DatabaseFactory(FactoryBase):
    """Example database factory.

    This demonstrates using FactoryBase for creating configured instances.
    """

    def __init__(self):
        """Initialize factory."""
        self.created_count = 0
        self.default_pool_size = 10

    def create(self, **config):
        """Create a database instance.

        Args:
            **config: Database configuration

        Returns:
            Database instance
        """
        # Add factory-specific defaults
        config.setdefault("pool_size", self.default_pool_size)
        config.setdefault("timeout", 30)

        # Track creation
        self.created_count += 1

        # Create and return instance
        return Database(**config)
