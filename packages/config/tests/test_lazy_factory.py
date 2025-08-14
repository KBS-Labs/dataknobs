"""Tests for lazy factory instance access."""

import pytest

from dataknobs_config import Config, ConfigError
from dataknobs_config.examples import (
    CacheFactory,
    Database,
    DatabaseFactory,
)


class TestLazyFactoryAccess:
    """Test lazy factory instance access."""

    def test_get_factory_instance(self):
        """Test getting a factory instance lazily."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "factory": "dataknobs_config.examples.DatabaseFactory",
                        "host": "localhost",
                        "port": 5432,
                    }
                ]
            }
        )

        # Get the factory instance
        factory = config.get_factory("database", "primary")

        assert isinstance(factory, DatabaseFactory)
        assert factory.created_count == 0  # Factory created but not used yet

        # Use the factory
        db1 = factory.create(host="db1.example.com", port=5432)
        assert isinstance(db1, Database)
        assert factory.created_count == 1

        db2 = factory.create(host="db2.example.com", port=5433)
        assert isinstance(db2, Database)
        assert factory.created_count == 2

    def test_factory_instance_cached(self):
        """Test that factory instances are cached."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "factory": "dataknobs_config.examples.DatabaseFactory",
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
        factory1.create(host="test", port=5432)
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
                    {"name": "redis1", "factory": "dataknobs_config.examples.CacheFactory"},
                    {"name": "redis2", "factory": "dataknobs_config.examples.DatabaseFactory"},
                ]
            }
        )

        # Get factories by index
        factory0 = config.get_factory("cache", 0)
        factory1 = config.get_factory("cache", 1)

        assert isinstance(factory0, CacheFactory)
        assert isinstance(factory1, DatabaseFactory)

    def test_get_instance_with_factory(self):
        """Test get_instance with factory configuration."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "factory": "dataknobs_config.examples.DatabaseFactory",
                        "host": "localhost",
                        "port": 5432,
                    }
                ]
            }
        )

        # get_instance should build an object using the factory
        instance = config.get_instance("database", "primary")

        assert isinstance(instance, Database)
        assert instance.host == "localhost"
        assert instance.port == 5432

    def test_get_instance_with_class(self):
        """Test get_instance with class configuration."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "class": "dataknobs_config.examples.Database",
                        "host": "localhost",
                        "port": 5432,
                    }
                ]
            }
        )

        # get_instance should build an object using the class
        instance = config.get_instance("database", "primary")

        assert isinstance(instance, Database)
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
                        "factory": "dataknobs_config.examples.DatabaseFactory",
                        "host": "localhost",
                    }
                ]
            }
        )

        # Pass additional kwargs
        instance = config.get_instance("database", "primary", port=5433, ssl=True)

        assert instance.host == "localhost"
        assert instance.port == 5433
        assert instance.extra.get("ssl") is True

    def test_multiple_factory_types(self):
        """Test different types of factories."""
        config = Config(
            {
                "services": [
                    {"name": "db", "factory": "dataknobs_config.examples.DatabaseFactory"},
                    {"name": "cache", "factory": "dataknobs_config.examples.CacheFactory"},
                ]
            }
        )

        # Get different factory types
        db_factory = config.get_factory("services", "db")
        cache_factory = config.get_factory("services", "cache")

        assert isinstance(db_factory, DatabaseFactory)
        assert isinstance(cache_factory, CacheFactory)

        # Use them
        db = db_factory.create(host="dbhost", port=5432)
        cache = cache_factory(host="cachehost", port=6379)

        assert isinstance(db, Database)
        assert hasattr(cache, "host")

    def test_factory_cache_clearing(self):
        """Test that factory cache can be cleared."""
        config = Config(
            {
                "database": [
                    {"name": "primary", "factory": "dataknobs_config.examples.DatabaseFactory"}
                ]
            }
        )

        # Get factory and use it
        factory1 = config.get_factory("database", "primary")
        factory1.create(host="test", port=5432)
        assert factory1.created_count == 1

        # Clear cache
        config.clear_object_cache()

        # Get factory again - should be new instance
        factory2 = config.get_factory("database", "primary")
        assert factory2 is not factory1
        assert factory2.created_count == 0  # New instance
