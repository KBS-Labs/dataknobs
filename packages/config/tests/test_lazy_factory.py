"""Tests for lazy factory instance access."""

import pytest

from dataknobs_config import Config, ConfigError, FactoryBase


# Mock factory classes for testing
class MockDatabaseFactory(FactoryBase):
    """Mock database factory for testing."""

    def __init__(self):
        self.created_count = 0

    def create(self, **config):
        self.created_count += 1
        return {"id": self.created_count, "config": config}


class MockCacheFactory:
    """Mock cache factory using callable pattern."""

    def __init__(self):
        self.created_count = 0

    def __call__(self, **config):
        self.created_count += 1
        return {"id": self.created_count, "type": "cache", "config": config}


def mock_function_factory(**config):
    """Mock function factory (module-level)."""
    return {"type": "function", "config": config}


class TestLazyFactoryAccess:
    """Test lazy factory instance access."""

    def test_get_factory_instance(self):
        """Test getting a factory instance lazily."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "factory": "test_lazy_factory.MockDatabaseFactory",
                        "host": "localhost",
                        "port": 5432,
                    }
                ]
            }
        )

        # Get the factory instance
        factory = config.get_factory("database", "primary")

        assert isinstance(factory, MockDatabaseFactory)
        assert factory.created_count == 0  # Factory created but not used yet

        # Use the factory
        db1 = factory.create(host="db1.example.com", port=5432)
        assert db1["id"] == 1
        assert factory.created_count == 1

        db2 = factory.create(host="db2.example.com", port=5433)
        assert db2["id"] == 2
        assert factory.created_count == 2

    def test_factory_instance_cached(self):
        """Test that factory instances are cached."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "factory": "test_lazy_factory.MockDatabaseFactory",
                    }
                ]
            }
        )

        # Get factory twice
        factory1 = config.get_factory("database", "primary")
        factory2 = config.get_factory("database", "primary")

        # Should be the same instance
        assert factory1 is factory2

        # Verify by using the factory
        factory1.create()
        assert factory1.created_count == 1
        assert factory2.created_count == 1  # Same instance

    def test_get_factory_no_factory_defined(self):
        """Test error when getting factory for config without factory attribute."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "class": "some.Database",  # Has class, not factory
                        "host": "localhost",
                    }
                ]
            }
        )

        with pytest.raises(ConfigError, match="No factory defined"):
            config.get_factory("database", "primary")

    def test_get_factory_by_index(self):
        """Test getting factory by index."""
        config = Config(
            {
                "cache": [
                    {"name": "redis1", "factory": "test_lazy_factory.MockCacheFactory"},
                    {"name": "redis2", "factory": "test_lazy_factory.MockDatabaseFactory"},
                ]
            }
        )

        # Get factories by index
        factory0 = config.get_factory("cache", 0)
        factory1 = config.get_factory("cache", 1)

        assert isinstance(factory0, MockCacheFactory)
        assert isinstance(factory1, MockDatabaseFactory)

    def test_get_instance_with_factory(self):
        """Test get_instance with factory configuration."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "factory": "test_lazy_factory.MockDatabaseFactory",
                        "host": "localhost",
                        "port": 5432,
                    }
                ]
            }
        )

        # get_instance should build an object using the factory
        instance = config.get_instance("database", "primary")

        assert instance["id"] == 1
        assert instance["config"]["host"] == "localhost"
        assert instance["config"]["port"] == 5432

    def test_get_instance_with_class(self):
        """Test get_instance with class configuration."""

        class MockDatabase:
            def __init__(self, host, port, **kwargs):
                self.host = host
                self.port = port
                self.extra = kwargs

        # Store the class in the module for import
        import sys

        sys.modules[__name__].MockDatabase = MockDatabase

        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "class": "test_lazy_factory.MockDatabase",
                        "host": "localhost",
                        "port": 5432,
                    }
                ]
            }
        )

        # get_instance should build an object using the class
        instance = config.get_instance("database", "primary")

        assert isinstance(instance, MockDatabase)
        assert instance.host == "localhost"
        assert instance.port == 5432

    def test_get_instance_without_class_or_factory(self):
        """Test get_instance returns config dict when no class/factory."""
        config = Config({"database": [{"name": "primary", "host": "localhost", "port": 5432}]})

        # get_instance should return the config dict itself
        instance = config.get_instance("database", "primary")

        assert isinstance(instance, dict)
        assert instance["host"] == "localhost"
        assert instance["port"] == 5432
        assert instance["name"] == "primary"

    def test_get_instance_with_kwargs(self):
        """Test get_instance with additional kwargs."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "factory": "test_lazy_factory.MockDatabaseFactory",
                        "host": "localhost",
                    }
                ]
            }
        )

        # Pass additional kwargs
        instance = config.get_instance("database", "primary", port=5433, ssl=True)

        assert instance["config"]["host"] == "localhost"
        assert instance["config"]["port"] == 5433
        assert instance["config"]["ssl"] is True

    def test_multiple_factory_types(self):
        """Test different types of factories."""
        config = Config(
            {
                "services": [
                    {"name": "db", "factory": "test_lazy_factory.MockDatabaseFactory"},
                    {"name": "cache", "factory": "test_lazy_factory.MockCacheFactory"},
                ]
            }
        )

        # Get different factory types
        db_factory = config.get_factory("services", "db")
        cache_factory = config.get_factory("services", "cache")

        assert isinstance(db_factory, MockDatabaseFactory)
        assert isinstance(cache_factory, MockCacheFactory)

        # Use them
        db = db_factory.create(host="dbhost")
        cache = cache_factory(host="cachehost")

        assert db["id"] == 1
        assert cache["type"] == "cache"

    def test_factory_cache_clearing(self):
        """Test that factory cache can be cleared."""
        config = Config(
            {"database": [{"name": "primary", "factory": "test_lazy_factory.MockDatabaseFactory"}]}
        )

        # Get factory and use it
        factory1 = config.get_factory("database", "primary")
        factory1.create()
        assert factory1.created_count == 1

        # Clear cache
        config.clear_object_cache()

        # Get factory again - should be new instance
        factory2 = config.get_factory("database", "primary")
        assert factory2 is not factory1
        assert factory2.created_count == 0  # New instance
