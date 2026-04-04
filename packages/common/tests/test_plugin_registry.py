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


# --- Fixtures for create() tests ---


class FromConfigHandler(BaseHandler):
    """Handler with a from_config classmethod."""

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], **kwargs: Any
    ) -> "FromConfigHandler":
        instance = cls("from_config", config)
        instance.extra = kwargs  # type: ignore[attr-defined]
        return instance


class PlainClassHandler:
    """Handler without from_config, accepts (config, **kwargs)."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        self.config = config
        self.kwargs = kwargs


def config_factory(config: Dict[str, Any], **kwargs: Any) -> BaseHandler:
    """Factory with (config, **kwargs) signature."""
    handler = BaseHandler("config_factory", config)
    handler.kwargs = kwargs  # type: ignore[attr-defined]
    return handler


class TestPluginRegistryCreate:
    """Test the create() method for fresh-instance factory mode."""

    def test_create_with_callable_factory(self) -> None:
        """create() calls callable as factory(config, **kwargs)."""
        registry = PluginRegistry[BaseHandler]("test")
        registry.register("h", config_factory)

        result = registry.create("h", {"x": 1})
        assert result.config == {"x": 1}
        assert result.name == "config_factory"

    def test_create_with_class_from_config(self) -> None:
        """create() detects and calls from_config on class factories."""
        registry = PluginRegistry[BaseHandler]("test")
        registry.register("fc", FromConfigHandler)

        result = registry.create("fc", {"y": 2}, extra="val")
        assert result.config == {"y": 2}
        assert result.extra == {"extra": "val"}  # type: ignore[attr-defined]

    def test_create_with_class_no_from_config(self) -> None:
        """create() calls class(config, **kwargs) when no from_config."""
        registry = PluginRegistry[PlainClassHandler]("test")
        registry.register("plain", PlainClassHandler)

        result = registry.create("plain", {"z": 3}, flag=True)
        assert result.config == {"z": 3}
        assert result.kwargs == {"flag": True}

    def test_create_never_caches(self) -> None:
        """Two create() calls return different instances."""
        registry = PluginRegistry[PlainClassHandler]("test")
        registry.register("h", PlainClassHandler)

        a = registry.create("h", {"v": 1})
        b = registry.create("h", {"v": 1})
        assert a is not b

    def test_create_does_not_pollute_get_cache(self) -> None:
        """create() does not write to the instance cache."""
        registry = PluginRegistry[PlainClassHandler]("test")
        registry.register("h", PlainClassHandler)

        # create() should not populate the cache
        registry.create("h", {"a": 1})
        assert "h" not in registry.cached_instances

        # Multiple creates don't accumulate cache entries
        registry.create("h", {"b": 2})
        assert len(registry.cached_instances) == 0

    def test_create_not_found(self) -> None:
        """create() raises NotFoundError for unregistered key."""
        registry = PluginRegistry[BaseHandler]("test")

        with pytest.raises(NotFoundError):
            registry.create("missing", {})

    def test_create_no_default_fallback(self) -> None:
        """create() does not use default_factory."""
        registry = PluginRegistry[BaseHandler](
            "test", default_factory=BaseHandler
        )

        with pytest.raises(NotFoundError):
            registry.create("missing", {})

    def test_create_validates_type(self) -> None:
        """create() validates against validate_type."""
        registry = PluginRegistry[BaseHandler](
            "test", validate_type=BaseHandler
        )
        registry.register("bad", lambda cfg, **kw: "not a handler")  # type: ignore[arg-type]

        with pytest.raises(OperationError, match="BaseHandler"):
            registry.create("bad", {})

    def test_create_factory_error_wrapped(self) -> None:
        """Factory exceptions are wrapped in OperationError."""
        def bad_factory(config: Dict[str, Any], **kwargs: Any) -> BaseHandler:
            raise RuntimeError("boom")

        registry = PluginRegistry[BaseHandler]("test")
        registry.register("bad", bad_factory)

        with pytest.raises(OperationError, match="boom"):
            registry.create("bad", {})

    def test_create_with_kwargs(self) -> None:
        """**kwargs are forwarded to the factory."""
        registry = PluginRegistry[BaseHandler]("test")
        registry.register("h", config_factory)

        result = registry.create("h", {"a": 1}, kb="test_kb")
        assert result.kwargs == {"kb": "test_kb"}  # type: ignore[attr-defined]

    def test_create_config_defaults_to_empty_dict(self) -> None:
        """config=None becomes {} when passed to factory."""
        received: Dict[str, Any] = {}

        def tracking(config: Dict[str, Any], **kwargs: Any) -> BaseHandler:
            received["config"] = config
            return BaseHandler("t", config)

        registry = PluginRegistry[BaseHandler]("test")
        registry.register("h", tracking)

        registry.create("h")
        assert received["config"] == {}


class TestPluginRegistryConfigKey:
    """Test config_key, config_key_default, and strip_config_key."""

    def test_create_extracts_key_from_config(self) -> None:
        """config_key field used as lookup key."""
        registry = PluginRegistry[PlainClassHandler](
            "test", config_key="strategy"
        )
        registry.register("simple", PlainClassHandler)

        result = registry.create(config={"strategy": "simple", "val": 1})
        assert result.config == {"strategy": "simple", "val": 1}

    def test_create_uses_config_key_default(self) -> None:
        """Fallback when field absent from config."""
        registry = PluginRegistry[PlainClassHandler](
            "test", config_key="strategy", config_key_default="simple"
        )
        registry.register("simple", PlainClassHandler)

        result = registry.create(config={"val": 1})
        assert result.config == {"val": 1}

    def test_create_explicit_key_overrides_config_key(self) -> None:
        """Explicit key takes precedence over config_key extraction."""
        registry = PluginRegistry[PlainClassHandler](
            "test", config_key="strategy"
        )
        registry.register("explicit", PlainClassHandler)

        result = registry.create(
            "explicit", config={"strategy": "other", "val": 1}
        )
        assert result.config == {"strategy": "other", "val": 1}

    def test_create_no_config_key_requires_explicit_key(self) -> None:
        """ValueError when key is None and config_key not configured."""
        registry = PluginRegistry[BaseHandler]("test")

        with pytest.raises(ValueError, match="config_key is not configured"):
            registry.create(config={"x": 1})

    def test_create_no_key_no_default_no_field(self) -> None:
        """ValueError when config lacks field and no default."""
        registry = PluginRegistry[BaseHandler](
            "test", config_key="strategy"
        )

        with pytest.raises(ValueError, match="config must contain"):
            registry.create(config={"val": 1})

    def test_config_key_with_canonicalize(self) -> None:
        """Extracted key is canonicalized."""
        registry = PluginRegistry[PlainClassHandler](
            "test", config_key="strategy", canonicalize_keys=True
        )
        registry.register("simple", PlainClassHandler)

        result = registry.create(config={"strategy": "SIMPLE", "v": 1})
        assert result.config == {"strategy": "SIMPLE", "v": 1}

    def test_strip_config_key_removes_routing_field(self) -> None:
        """Factory receives config without the key field."""
        registry = PluginRegistry[PlainClassHandler](
            "test", config_key="backend", strip_config_key=True
        )
        registry.register("mem", PlainClassHandler)

        result = registry.create(config={"backend": "mem", "size": 100})
        assert result.config == {"size": 100}
        assert "backend" not in result.config

    def test_strip_config_key_preserves_other_fields(self) -> None:
        """Non-key fields are passed through unchanged."""
        registry = PluginRegistry[PlainClassHandler](
            "test", config_key="backend", strip_config_key=True
        )
        registry.register("pg", PlainClassHandler)

        result = registry.create(
            config={"backend": "pg", "host": "localhost", "port": 5432}
        )
        assert result.config == {"host": "localhost", "port": 5432}

    def test_strip_config_key_no_effect_with_explicit_key(self) -> None:
        """Stripping only applies when key extracted from config."""
        registry = PluginRegistry[PlainClassHandler](
            "test", config_key="backend", strip_config_key=True
        )
        registry.register("mem", PlainClassHandler)

        # Explicit key — config passed through unchanged
        result = registry.create(
            "mem", config={"backend": "mem", "size": 100}
        )
        assert result.config == {"backend": "mem", "size": 100}


class TestPluginRegistryCanonicalizeKeys:
    """Test canonicalize_keys parameter."""

    def test_register_and_get_case_insensitive(self) -> None:
        """'Foo' and 'foo' resolve to same registration."""
        registry = PluginRegistry[BaseHandler](
            "test", canonicalize_keys=True
        )
        registry.register("Foo", CustomHandler)

        h = registry.get("foo", config={})
        assert isinstance(h, CustomHandler)

        h2 = registry.get("FOO", config={})
        assert h is h2  # Same cached instance

    def test_is_registered_case_insensitive(self) -> None:
        """is_registered ignores case."""
        registry = PluginRegistry[BaseHandler](
            "test", canonicalize_keys=True
        )
        registry.register("MyPlugin", CustomHandler)

        assert registry.is_registered("myplugin")
        assert registry.is_registered("MYPLUGIN")
        assert registry.is_registered("MyPlugin")

    def test_get_factory_case_insensitive(self) -> None:
        """get_factory ignores case."""
        registry = PluginRegistry[BaseHandler](
            "test", canonicalize_keys=True
        )
        registry.register("Handler", CustomHandler)

        assert registry.get_factory("handler") is CustomHandler
        assert registry.get_factory("HANDLER") is CustomHandler

    def test_unregister_case_insensitive(self) -> None:
        """unregister ignores case."""
        registry = PluginRegistry[BaseHandler](
            "test", canonicalize_keys=True
        )
        registry.register("Plugin", CustomHandler)

        registry.unregister("PLUGIN")
        assert not registry.is_registered("plugin")

    def test_list_keys_returns_canonical(self) -> None:
        """All keys stored in lowercase."""
        registry = PluginRegistry[BaseHandler](
            "test", canonicalize_keys=True
        )
        registry.register("Alpha", CustomHandler)
        registry.register("BETA", AnotherHandler)

        keys = registry.list_keys()
        assert set(keys) == {"alpha", "beta"}

    def test_create_case_insensitive(self) -> None:
        """create() ignores case."""
        registry = PluginRegistry[PlainClassHandler](
            "test", canonicalize_keys=True
        )
        registry.register("handler", PlainClassHandler)

        result = registry.create("HANDLER", {"v": 1})
        assert result.config == {"v": 1}

    def test_canonicalize_off_by_default(self) -> None:
        """Default preserves case — 'Foo' and 'foo' are different."""
        registry = PluginRegistry[BaseHandler]("test")
        registry.register("Foo", CustomHandler)

        assert registry.is_registered("Foo")
        assert not registry.is_registered("foo")


class TestPluginRegistryLazyInit:
    """Test on_first_access callback."""

    def test_on_first_access_called_once(self) -> None:
        """Callback runs on first access only."""
        calls: list[int] = []

        def init_cb(reg: PluginRegistry[BaseHandler]) -> None:
            calls.append(1)
            reg.register("auto", CustomHandler)

        registry = PluginRegistry[BaseHandler](
            "test", on_first_access=init_cb
        )

        assert len(calls) == 0
        assert registry.is_registered("auto")
        assert len(calls) == 1
        # Second access — no re-invocation
        registry.list_keys()
        assert len(calls) == 1

    def test_on_first_access_not_called_without_config(self) -> None:
        """No overhead when on_first_access is None."""
        registry = PluginRegistry[BaseHandler]("test")
        registry.register("h", CustomHandler)

        # Should work without any initialization callback
        assert registry.is_registered("h")

    def test_callback_can_register(self) -> None:
        """Callback calls register() without deadlock (re-entrant)."""
        def init_cb(reg: PluginRegistry[BaseHandler]) -> None:
            reg.register("a", CustomHandler)
            reg.register("b", AnotherHandler)

        registry = PluginRegistry[BaseHandler](
            "test", on_first_access=init_cb
        )

        keys = registry.list_keys()
        assert "a" in keys
        assert "b" in keys

    def test_callback_failure_allows_retry(self) -> None:
        """_initialized rolled back on exception; next call retries."""
        attempt = [0]

        def flaky_init(reg: PluginRegistry[BaseHandler]) -> None:
            attempt[0] += 1
            if attempt[0] == 1:
                raise RuntimeError("first attempt fails")
            reg.register("h", CustomHandler)

        registry = PluginRegistry[BaseHandler](
            "test", on_first_access=flaky_init
        )

        # First access fails
        with pytest.raises(RuntimeError, match="first attempt fails"):
            registry.list_keys()

        # Second access retries and succeeds
        keys = registry.list_keys()
        assert "h" in keys
        assert attempt[0] == 2

    def test_callback_receives_registry(self) -> None:
        """Registry instance is passed to callback."""
        received: list[Any] = []

        def init_cb(reg: PluginRegistry[BaseHandler]) -> None:
            received.append(reg)

        registry = PluginRegistry[BaseHandler](
            "test", on_first_access=init_cb
        )

        registry.list_keys()
        assert received[0] is registry

    def test_ensure_initialized_idempotent(self) -> None:
        """Multiple accesses produce single invocation."""
        calls: list[int] = []

        def init_cb(reg: PluginRegistry[BaseHandler]) -> None:
            calls.append(1)

        registry = PluginRegistry[BaseHandler](
            "test", on_first_access=init_cb
        )

        registry.list_keys()
        registry.is_registered("x")
        registry.get_factory("x")
        _ = len(registry)
        assert len(calls) == 1


class TestPluginRegistryMetadata:
    """Test metadata support on register() and get_metadata()."""

    def test_register_with_metadata(self) -> None:
        """Metadata stored and retrievable."""
        registry = PluginRegistry[BaseHandler]("test")
        registry.register("h", CustomHandler, metadata={
            "description": "Custom handler",
            "version": "1.0",
        })

        meta = registry.get_metadata("h")
        assert meta == {"description": "Custom handler", "version": "1.0"}

    def test_get_metadata_returns_copy(self) -> None:
        """Returned dict is a copy, not a reference."""
        registry = PluginRegistry[BaseHandler]("test")
        registry.register("h", CustomHandler, metadata={"key": "val"})

        meta1 = registry.get_metadata("h")
        meta1["key"] = "modified"

        meta2 = registry.get_metadata("h")
        assert meta2["key"] == "val"

    def test_get_metadata_missing_key(self) -> None:
        """Returns empty dict for unregistered key."""
        registry = PluginRegistry[BaseHandler]("test")

        assert registry.get_metadata("missing") == {}

    def test_get_metadata_no_metadata(self) -> None:
        """Returns empty dict when registered without metadata."""
        registry = PluginRegistry[BaseHandler]("test")
        registry.register("h", CustomHandler)

        assert registry.get_metadata("h") == {}

    def test_unregister_clears_metadata(self) -> None:
        """Metadata removed on unregister."""
        registry = PluginRegistry[BaseHandler]("test")
        registry.register("h", CustomHandler, metadata={"key": "val"})

        registry.unregister("h")
        assert registry.get_metadata("h") == {}
