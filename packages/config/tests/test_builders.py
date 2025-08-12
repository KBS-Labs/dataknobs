"""Tests for object construction and builders."""

import pytest

from dataknobs_config import Config, ConfigError, ConfigurableBase, FactoryBase


# Test classes for object construction
class MockDatabase(ConfigurableBase):
    """Mock database class for testing."""

    def __init__(self, host, port, **kwargs):
        self.host = host
        self.port = port
        self.extra = kwargs


class MockCache:
    """Mock cache class without ConfigurableBase."""

    def __init__(self, host, port, ttl=3600):
        self.host = host
        self.port = port
        self.ttl = ttl


class MockDatabaseFactory(FactoryBase):
    """Mock factory for databases."""

    def create(self, **config):
        # Add some factory logic
        config.setdefault("pool_size", 10)
        return MockDatabase(**config)


class MockCallableFactory:
    """Mock callable factory."""

    def __call__(self, **config):
        return MockCache(**config)


class TestObjectConstruction:
    """Test object construction from configurations."""

    def test_build_with_class(self):
        """Test building object with class attribute."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "class": "test_builders.MockDatabase",
                        "host": "localhost",
                        "port": 5432,
                    }
                ]
            }
        )

        obj = config.build_object("xref:database[primary]")

        assert isinstance(obj, MockDatabase)
        assert obj.host == "localhost"
        assert obj.port == 5432

    def test_build_with_configurable_base(self):
        """Test building object that inherits from ConfigurableBase."""
        config = Config(
            {
                "database": [
                    {
                        "name": "test",
                        "class": "test_builders.MockDatabase",
                        "host": "testhost",
                        "port": 3306,
                        "extra_param": "value",
                    }
                ]
            }
        )

        obj = config.build_object("xref:database[test]")

        assert isinstance(obj, MockDatabase)
        assert obj.host == "testhost"
        assert obj.extra["extra_param"] == "value"

    def test_build_with_factory(self):
        """Test building object with factory attribute."""
        config = Config(
            {
                "database": [
                    {
                        "name": "primary",
                        "factory": "test_builders.MockDatabaseFactory",
                        "host": "localhost",
                        "port": 5432,
                    }
                ]
            }
        )

        obj = config.build_object("xref:database[primary]")

        assert isinstance(obj, MockDatabase)
        assert obj.host == "localhost"
        assert obj.port == 5432
        assert obj.extra.get("pool_size") == 10  # Added by factory

    def test_build_with_callable_factory(self):
        """Test building with callable factory."""
        config = Config(
            {
                "cache": [
                    {
                        "name": "redis",
                        "factory": "test_builders.MockCallableFactory",
                        "host": "localhost",
                        "port": 6379,
                        "ttl": 7200,
                    }
                ]
            }
        )

        obj = config.build_object("xref:cache[redis]")

        assert isinstance(obj, MockCache)
        assert obj.host == "localhost"
        assert obj.ttl == 7200

    def test_build_without_class_or_factory(self):
        """Test that building without class or factory raises error."""
        config = Config({"database": [{"name": "test", "host": "localhost"}]})

        with pytest.raises(ConfigError):
            config.build_object("xref:database[test]")

    def test_invalid_class_path(self):
        """Test that invalid class path raises error."""
        config = Config({"database": [{"name": "test", "class": "nonexistent.module.Class"}]})

        with pytest.raises(ConfigError):
            config.build_object("xref:database[test]")

    def test_build_with_kwargs(self):
        """Test building with additional kwargs."""
        config = Config(
            {
                "database": [
                    {"name": "test", "class": "test_builders.MockDatabase", "host": "localhost"}
                ]
            }
        )

        obj = config.build_object("xref:database[test]", port=3306, extra_param="added")

        assert obj.host == "localhost"
        assert obj.port == 3306
        assert obj.extra["extra_param"] == "added"


class TestObjectCaching:
    """Test object caching functionality."""

    def test_cache_enabled(self):
        """Test that objects are cached by default."""
        config = Config(
            {
                "cache": [
                    {
                        "name": "redis",
                        "class": "test_builders.MockCache",
                        "host": "localhost",
                        "port": 6379,
                    }
                ]
            }
        )

        obj1 = config.build_object("xref:cache[redis]")
        obj2 = config.build_object("xref:cache[redis]")

        assert obj1 is obj2  # Same object instance

    def test_cache_disabled(self):
        """Test building without caching."""
        config = Config(
            {
                "cache": [
                    {
                        "name": "redis",
                        "class": "test_builders.MockCache",
                        "host": "localhost",
                        "port": 6379,
                    }
                ]
            }
        )

        obj1 = config.build_object("xref:cache[redis]", cache=False)
        obj2 = config.build_object("xref:cache[redis]", cache=False)

        assert obj1 is not obj2  # Different instances
        assert obj1.host == obj2.host  # But same config

    def test_clear_cache(self):
        """Test clearing object cache."""
        config = Config(
            {
                "cache": [
                    {
                        "name": "redis",
                        "class": "test_builders.MockCache",
                        "host": "localhost",
                        "port": 6379,
                    }
                ]
            }
        )

        obj1 = config.build_object("xref:cache[redis]")
        config.clear_object_cache("xref:cache[redis]")
        obj2 = config.build_object("xref:cache[redis]")

        assert obj1 is not obj2  # Different instances after cache clear

    def test_clear_all_cache(self):
        """Test clearing all cached objects."""
        config = Config(
            {
                "cache": [
                    {
                        "name": "redis1",
                        "class": "test_builders.MockCache",
                        "host": "host1",
                        "port": 6379,
                    },
                    {
                        "name": "redis2",
                        "class": "test_builders.MockCache",
                        "host": "host2",
                        "port": 6380,
                    },
                ]
            }
        )

        obj1a = config.build_object("xref:cache[redis1]")
        obj2a = config.build_object("xref:cache[redis2]")

        config.clear_object_cache()  # Clear all

        obj1b = config.build_object("xref:cache[redis1]")
        obj2b = config.build_object("xref:cache[redis2]")

        assert obj1a is not obj1b
        assert obj2a is not obj2b
