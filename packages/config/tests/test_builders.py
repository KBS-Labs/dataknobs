"""Tests for object construction and builders."""

import pytest

from dataknobs_config import Config, ConfigError
from dataknobs_config.examples import (
    AsyncWidget,
    Cache,
    Database,
    PlainWidget,
    SyncWidget,
)


class TestObjectConstruction:
    """Test object construction from configurations."""

    def test_build_with_class(self):
        """Test building object with class attribute."""
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

        obj = config.build_object("xref:database[primary]")

        assert isinstance(obj, Database)
        assert obj.host == "localhost"
        assert obj.port == 5432

    def test_build_with_configurable_base(self):
        """Test building object that inherits from ConfigurableBase."""
        config = Config(
            {
                "database": [
                    {
                        "name": "test",
                        "class": "dataknobs_config.examples.Database",
                        "host": "testhost",
                        "port": 3306,
                        "extra_param": "value",
                    }
                ]
            }
        )

        obj = config.build_object("xref:database[test]")

        assert isinstance(obj, Database)
        assert obj.host == "testhost"
        assert obj.extra["extra_param"] == "value"

    def test_build_with_factory(self):
        """Test building object with factory attribute."""
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

        obj = config.build_object("xref:database[primary]")

        assert isinstance(obj, Database)
        assert obj.host == "localhost"
        assert obj.port == 5432
        assert obj.pool_size == 10  # Added by factory

    def test_build_with_callable_factory(self):
        """Test building with callable factory."""
        config = Config(
            {
                "cache": [
                    {
                        "name": "redis",
                        "factory": "dataknobs_config.examples.CacheFactory",
                        "host": "localhost",
                        "port": 6379,
                        "ttl": 7200,
                    }
                ]
            }
        )

        obj = config.build_object("xref:cache[redis]")

        assert isinstance(obj, Cache)
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
                    {
                        "name": "test",
                        "class": "dataknobs_config.examples.Database",
                        "host": "localhost",
                    }
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
                        "class": "dataknobs_config.examples.Cache",
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
                        "class": "dataknobs_config.examples.Cache",
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
                        "class": "dataknobs_config.examples.Cache",
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
                        "class": "dataknobs_config.examples.Cache",
                        "host": "host1",
                        "port": 6379,
                    },
                    {
                        "name": "redis2",
                        "class": "dataknobs_config.examples.Cache",
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


class TestAsyncObjectConstruction:
    """``build_object_async`` prefers async entry points."""

    @pytest.mark.asyncio
    async def test_prefers_from_config_async(self) -> None:
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "class": "dataknobs_config.examples.AsyncWidget",
                    }
                ]
            }
        )
        obj = await config.build_object_async("xref:widget[w]")
        assert isinstance(obj, AsyncWidget)
        # ``_ainit`` ran — only the async path sets this.
        assert obj.warmed is True

    @pytest.mark.asyncio
    async def test_falls_back_to_sync_from_config(self) -> None:
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "class": "dataknobs_config.examples.SyncWidget",
                        "extra_field": "ignored-by-from_config",
                    }
                ]
            }
        )
        obj = await config.build_object_async("xref:widget[w]")
        assert isinstance(obj, SyncWidget)
        assert obj.name == "sync"

    @pytest.mark.asyncio
    async def test_falls_back_to_direct_instantiation(self) -> None:
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "class": "dataknobs_config.examples.PlainWidget",
                    }
                ]
            }
        )
        obj = await config.build_object_async("xref:widget[w]")
        assert isinstance(obj, PlainWidget)
        assert obj.name == "plain"

    @pytest.mark.asyncio
    async def test_async_build_caches(self) -> None:
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "class": "dataknobs_config.examples.AsyncWidget",
                    }
                ]
            }
        )
        a = await config.build_object_async("xref:widget[w]")
        b = await config.build_object_async("xref:widget[w]")
        assert a is b

    def test_sync_build_object_does_not_run_ainit(self) -> None:
        """The sync path leaves the async-only flag unset."""
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "class": "dataknobs_config.examples.AsyncWidget",
                    }
                ]
            }
        )
        obj = config.build_object("xref:widget[w]")
        assert isinstance(obj, AsyncWidget)
        assert obj.warmed is False

    @pytest.mark.asyncio
    async def test_factory_path_runs_consumer_async_init(self) -> None:
        """A consumer referenced via ``factory:`` runs ``_ainit`` (async).

        A genuine factory's ``create``/``build``/``__call__`` is preferred;
        the consumer protocol (``from_config_async``) is the last-resort
        path, so a consumer class used as a factory builds correctly
        instead of erroring.
        """
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "factory": "dataknobs_config.examples.AsyncWidget",
                    }
                ]
            }
        )
        obj = await config.build_object_async("xref:widget[w]")
        assert isinstance(obj, AsyncWidget)
        assert obj.warmed is True

    def test_factory_path_builds_consumer_sync(self) -> None:
        """Sync ``factory:`` path builds a consumer via ``from_config``."""
        config = Config(
            {
                "widget": [
                    {
                        "name": "w",
                        "factory": "dataknobs_config.examples.AsyncWidget",
                    }
                ]
            }
        )
        obj = config.build_object("xref:widget[w]")
        assert isinstance(obj, AsyncWidget)
        assert obj.warmed is False
