"""Tests for PluginRegistry class."""

import asyncio
from typing import Any, Dict

import pytest

from dataknobs_common import NotFoundError, OperationError, PluginRegistry


# Test classes for the registry
class BaseHandler:
    """Base handler class for testing."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config


class CustomHandler(BaseHandler):
    """Custom handler implementation."""

    pass


class AnotherHandler(BaseHandler):
    """Another handler implementation."""

    pass


class InvalidHandler:
    """Handler that doesn't inherit from BaseHandler."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name


def handler_factory(name: str, config: Dict[str, Any]) -> BaseHandler:
    """Factory function that creates handlers."""
    return BaseHandler(name, config)


async def async_handler_factory(name: str, config: Dict[str, Any]) -> BaseHandler:
    """Async factory function."""
    await asyncio.sleep(0.001)  # Simulate async work
    return BaseHandler(name, config)


class TestPluginRegistryInit:
    """Test PluginRegistry initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        registry = PluginRegistry[BaseHandler]("handlers")
        assert registry.name == "handlers"
        assert registry.list_keys() == []

    def test_init_with_default(self):
        """Test initialization with default factory."""
        registry = PluginRegistry[BaseHandler]("handlers", default_factory=BaseHandler)
        assert registry.name == "handlers"

    def test_init_with_validation(self):
        """Test initialization with type validation."""
        registry = PluginRegistry[BaseHandler](
            "handlers",
            validate_type=BaseHandler,
        )
        assert registry.name == "handlers"


class TestPluginRegistration:
    """Test plugin registration functionality."""

    def test_register_class(self):
        """Test registering a class."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("custom", CustomHandler)

        assert registry.is_registered("custom")

    def test_register_factory_function(self):
        """Test registering a factory function."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("factory", handler_factory)

        assert registry.is_registered("factory")

    def test_register_lambda(self):
        """Test registering a lambda factory."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("lambda", lambda name, config: BaseHandler(name, config))

        assert registry.is_registered("lambda")

    def test_register_duplicate_error(self):
        """Test error when registering duplicate without override."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)

        with pytest.raises(OperationError, match="already registered"):
            registry.register("handler", AnotherHandler)

    def test_register_duplicate_with_override(self):
        """Test overriding existing registration."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)
        registry.register("handler", AnotherHandler, override=True)

        instance = registry.get("handler", config={})
        assert isinstance(instance, AnotherHandler)

    def test_register_clears_cache_on_override(self):
        """Test that override clears cached instance."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)

        # Create and cache instance
        instance1 = registry.get("handler", config={"v": 1})
        assert isinstance(instance1, CustomHandler)

        # Override registration
        registry.register("handler", AnotherHandler, override=True)

        # Get should create new instance
        instance2 = registry.get("handler", config={"v": 2})
        assert isinstance(instance2, AnotherHandler)

    def test_register_non_callable_error(self):
        """Test error when registering non-callable."""
        registry = PluginRegistry[BaseHandler]("handlers")

        with pytest.raises(TypeError, match="must be a class or callable"):
            registry.register("invalid", "not callable")  # type: ignore

    def test_register_type_validation(self):
        """Test type validation on registration."""
        registry = PluginRegistry[BaseHandler](
            "handlers",
            validate_type=BaseHandler,
        )

        # Valid subclass
        registry.register("valid", CustomHandler)

        # Invalid class (not a subclass)
        with pytest.raises(TypeError, match="must be a subclass"):
            registry.register("invalid", InvalidHandler)


class TestPluginUnregistration:
    """Test plugin unregistration."""

    def test_unregister(self):
        """Test unregistering a plugin."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)

        registry.unregister("handler")
        assert not registry.is_registered("handler")

    def test_unregister_clears_cache(self):
        """Test that unregister clears cached instance."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)

        # Create and cache
        registry.get("handler", config={})

        # Unregister
        registry.unregister("handler")

        # Re-register and get should create new instance
        registry.register("handler", AnotherHandler)
        instance = registry.get("handler", config={})
        assert isinstance(instance, AnotherHandler)

    def test_unregister_not_found(self):
        """Test error when unregistering non-existent plugin."""
        registry = PluginRegistry[BaseHandler]("handlers")

        with pytest.raises(NotFoundError, match="not found"):
            registry.unregister("nonexistent")


class TestPluginGet:
    """Test plugin instance retrieval."""

    def test_get_instance(self):
        """Test getting a plugin instance."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("custom", CustomHandler)

        instance = registry.get("custom", config={"timeout": 30})

        assert isinstance(instance, CustomHandler)
        assert instance.name == "custom"
        assert instance.config == {"timeout": 30}

    def test_get_cached_instance(self):
        """Test that instances are cached."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)

        instance1 = registry.get("handler", config={"v": 1})
        instance2 = registry.get("handler", config={"v": 2})

        # Should be same cached instance
        assert instance1 is instance2
        assert instance1.config == {"v": 1}

    def test_get_without_cache(self):
        """Test getting without caching."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)

        instance1 = registry.get("handler", config={"v": 1}, use_cache=False)
        instance2 = registry.get("handler", config={"v": 2}, use_cache=False)

        # Should be different instances
        assert instance1 is not instance2

    def test_get_with_default(self):
        """Test using default factory."""
        registry = PluginRegistry[BaseHandler]("handlers", default_factory=BaseHandler)

        # Get unregistered key uses default
        instance = registry.get("unknown", config={})

        assert isinstance(instance, BaseHandler)
        assert instance.name == "unknown"

    def test_get_without_default_error(self):
        """Test error when no default available."""
        registry = PluginRegistry[BaseHandler]("handlers")

        with pytest.raises(NotFoundError, match="not registered"):
            registry.get("unknown", config={})

    def test_get_disable_default(self):
        """Test disabling default fallback."""
        registry = PluginRegistry[BaseHandler]("handlers", default_factory=BaseHandler)

        with pytest.raises(NotFoundError):
            registry.get("unknown", config={}, use_default=False)

    def test_get_with_factory_function(self):
        """Test getting instance from factory function."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("factory", handler_factory)

        instance = registry.get("factory", config={"key": "value"})

        assert isinstance(instance, BaseHandler)
        assert instance.config == {"key": "value"}

    def test_get_instance_type_validation(self):
        """Test type validation of created instances."""
        registry = PluginRegistry[BaseHandler](
            "handlers",
            validate_type=BaseHandler,
        )

        # Register factory that returns wrong type
        def bad_factory(name, config):
            return "not a handler"

        registry.register("bad", bad_factory)

        with pytest.raises(OperationError, match="Failed to create"):
            registry.get("bad", config={})

    def test_get_factory_error_handling(self):
        """Test error handling when factory fails."""
        registry = PluginRegistry[BaseHandler]("handlers")

        def failing_factory(name, config):
            raise ValueError("Factory error")

        registry.register("failing", failing_factory)

        with pytest.raises(OperationError, match="Failed to create"):
            registry.get("failing", config={})


class TestPluginGetAsync:
    """Test async plugin instance retrieval."""

    @pytest.mark.asyncio
    async def test_get_async_sync_factory(self):
        """Test get_async with sync factory."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)

        instance = await registry.get_async("handler", config={"async": True})

        assert isinstance(instance, CustomHandler)
        assert instance.config == {"async": True}

    @pytest.mark.asyncio
    async def test_get_async_with_async_factory(self):
        """Test get_async with async factory."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("async", async_handler_factory)

        instance = await registry.get_async("async", config={"async": True})

        assert isinstance(instance, BaseHandler)
        assert instance.name == "async"

    @pytest.mark.asyncio
    async def test_get_async_cached(self):
        """Test that async get caches instances."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", async_handler_factory)

        instance1 = await registry.get_async("handler", config={})
        instance2 = await registry.get_async("handler", config={})

        assert instance1 is instance2

    @pytest.mark.asyncio
    async def test_get_async_with_default(self):
        """Test get_async with default factory."""
        registry = PluginRegistry[BaseHandler]("handlers", default_factory=BaseHandler)

        instance = await registry.get_async("unknown", config={})

        assert isinstance(instance, BaseHandler)
        assert instance.name == "unknown"

    @pytest.mark.asyncio
    async def test_get_async_not_found(self):
        """Test get_async error when not found."""
        registry = PluginRegistry[BaseHandler]("handlers")

        with pytest.raises(NotFoundError):
            await registry.get_async("unknown", config={})


class TestPluginRegistryUtilities:
    """Test utility methods."""

    def test_list_keys(self):
        """Test listing registered keys."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("a", CustomHandler)
        registry.register("b", AnotherHandler)

        keys = registry.list_keys()
        assert set(keys) == {"a", "b"}

    def test_list_keys_empty(self):
        """Test listing keys on empty registry."""
        registry = PluginRegistry[BaseHandler]("handlers")
        assert registry.list_keys() == []

    def test_is_registered(self):
        """Test is_registered check."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)

        assert registry.is_registered("handler")
        assert not registry.is_registered("other")

    def test_clear_cache_single(self):
        """Test clearing cache for single key."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("a", CustomHandler)
        registry.register("b", AnotherHandler)

        # Create cached instances
        registry.get("a", config={})
        registry.get("b", config={})

        # Clear only a
        registry.clear_cache("a")

        # a should create new instance, b should return cached
        instance_a = registry.get("a", config={"new": True})
        assert instance_a.config == {"new": True}

    def test_clear_cache_all(self):
        """Test clearing all cached instances."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("a", CustomHandler)
        registry.register("b", AnotherHandler)

        # Create cached instances
        registry.get("a", config={})
        registry.get("b", config={})

        # Clear all
        registry.clear_cache()

        # Both should create new instances
        instance_a = registry.get("a", config={"new": True})
        instance_b = registry.get("b", config={"new": True})

        assert instance_a.config == {"new": True}
        assert instance_b.config == {"new": True}

    def test_get_factory(self):
        """Test getting the registered factory."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)

        factory = registry.get_factory("handler")
        assert factory is CustomHandler

    def test_get_factory_not_found(self):
        """Test get_factory returns None for unknown key."""
        registry = PluginRegistry[BaseHandler]("handlers")

        factory = registry.get_factory("unknown")
        assert factory is None

    def test_set_default_factory(self):
        """Test setting default factory after init."""
        registry = PluginRegistry[BaseHandler]("handlers")

        # Initially no default
        with pytest.raises(NotFoundError):
            registry.get("unknown", config={})

        # Set default
        registry.set_default_factory(BaseHandler)

        # Now should use default
        instance = registry.get("unknown", config={})
        assert isinstance(instance, BaseHandler)

    def test_bulk_register(self):
        """Test bulk registration."""
        registry = PluginRegistry[BaseHandler]("handlers")

        factories = {
            "custom": CustomHandler,
            "another": AnotherHandler,
        }
        registry.bulk_register(factories)

        assert registry.is_registered("custom")
        assert registry.is_registered("another")

    def test_bulk_register_with_override(self):
        """Test bulk registration with override."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("custom", CustomHandler)

        # Bulk register with override
        registry.bulk_register({"custom": AnotherHandler}, override=True)

        instance = registry.get("custom", config={})
        assert isinstance(instance, AnotherHandler)


class TestPluginRegistryThreadSafety:
    """Test thread safety of registry operations."""

    def test_concurrent_registration(self):
        """Test concurrent registration is safe."""
        import threading

        registry = PluginRegistry[BaseHandler]("handlers")
        errors = []

        def register_handler(i):
            try:
                registry.register(f"handler_{i}", CustomHandler)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_handler, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.list_keys()) == 10

    def test_concurrent_get(self):
        """Test concurrent get is safe."""
        import threading

        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)
        instances = []
        lock = threading.Lock()

        def get_handler():
            instance = registry.get("handler", config={})
            with lock:
                instances.append(instance)

        threads = [threading.Thread(target=get_handler) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should get the same cached instance
        assert all(inst is instances[0] for inst in instances)


class TestPluginRegistryEdgeCases:
    """Test edge cases."""

    def test_none_config(self):
        """Test that None config is handled."""
        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("handler", CustomHandler)

        instance = registry.get("handler", config=None)
        assert instance.config == {}

    def test_empty_registry_name(self):
        """Test registry with empty name."""
        registry = PluginRegistry[BaseHandler]("")
        assert registry.name == ""

    def test_factory_receives_correct_args(self):
        """Test factory receives correct arguments."""
        received = {}

        def tracking_factory(name, config):
            received["name"] = name
            received["config"] = config
            return BaseHandler(name, config)

        registry = PluginRegistry[BaseHandler]("handlers")
        registry.register("tracked", tracking_factory)

        registry.get("tracked", config={"key": "value"})

        assert received["name"] == "tracked"
        assert received["config"] == {"key": "value"}

    def test_default_factory_receives_correct_args(self):
        """Test default factory receives the key and config."""
        received = {}

        def tracking_default(name, config):
            received["name"] = name
            received["config"] = config
            return BaseHandler(name, config)

        registry = PluginRegistry[BaseHandler]("handlers", default_factory=tracking_default)

        registry.get("unknown_key", config={"default": True})

        assert received["name"] == "unknown_key"
        assert received["config"] == {"default": True}
