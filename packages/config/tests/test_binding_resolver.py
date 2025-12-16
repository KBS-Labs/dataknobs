"""Tests for ConfigBindingResolver class."""

import pytest

from dataknobs_config.binding_resolver import (
    AsyncCallableFactory,
    BindingResolverError,
    CallableFactory,
    ConfigBindingResolver,
    FactoryNotFoundError,
    SimpleFactory,
)
from dataknobs_config.environment_config import EnvironmentConfig


class MockDatabase:
    """Mock database class for testing."""

    def __init__(self, backend: str, host: str = "localhost", **kwargs):
        self.backend = backend
        self.host = host
        self.extra = kwargs


class MockAsyncDatabase:
    """Mock async database class for testing."""

    def __init__(self, backend: str, connected: bool = False):
        self.backend = backend
        self.connected = connected


class MockFactory:
    """Mock factory for testing."""

    def create(self, backend: str, **kwargs):
        return MockDatabase(backend=backend, **kwargs)


class MockAsyncFactory:
    """Mock async factory for testing."""

    async def create_async(self, backend: str, **kwargs):
        db = MockAsyncDatabase(backend=backend, connected=True)
        return db


class TestConfigBindingResolverBasics:
    """Test basic ConfigBindingResolver functionality."""

    @pytest.fixture
    def env_config(self):
        """Sample environment configuration."""
        return EnvironmentConfig(
            name="test",
            resources={
                "databases": {
                    "primary": {
                        "backend": "postgres",
                        "host": "db.example.com",
                    },
                    "cache": {
                        "backend": "redis",
                        "host": "cache.example.com",
                    },
                },
            },
        )

    @pytest.fixture
    def resolver(self, env_config):
        """Create resolver with registered factory."""
        resolver = ConfigBindingResolver(env_config)
        resolver.register_factory("databases", MockFactory())
        return resolver

    def test_register_factory(self, env_config):
        """Test registering a factory."""
        resolver = ConfigBindingResolver(env_config)
        factory = MockFactory()
        resolver.register_factory("databases", factory)

        assert resolver.has_factory("databases")
        assert "databases" in resolver.get_registered_types()

    def test_unregister_factory(self, resolver):
        """Test unregistering a factory."""
        resolver.unregister_factory("databases")
        assert not resolver.has_factory("databases")

    def test_unregister_missing_factory(self, resolver):
        """Test error when unregistering missing factory."""
        with pytest.raises(KeyError, match="No factory registered"):
            resolver.unregister_factory("nonexistent")

    def test_resolve(self, resolver):
        """Test resolving a resource."""
        db = resolver.resolve("databases", "primary")

        assert isinstance(db, MockDatabase)
        assert db.backend == "postgres"
        assert db.host == "db.example.com"

    def test_resolve_with_overrides(self, resolver):
        """Test resolving with config overrides."""
        db = resolver.resolve("databases", "primary", port=5432)

        assert db.backend == "postgres"
        assert db.extra["port"] == 5432

    def test_resolve_caching(self, resolver):
        """Test that resolved instances are cached."""
        db1 = resolver.resolve("databases", "primary")
        db2 = resolver.resolve("databases", "primary")

        assert db1 is db2

    def test_resolve_no_cache(self, resolver):
        """Test resolving without caching."""
        db1 = resolver.resolve("databases", "primary", use_cache=False)
        db2 = resolver.resolve("databases", "primary", use_cache=False)

        assert db1 is not db2

    def test_resolve_missing_factory(self, env_config):
        """Test error when factory not registered."""
        resolver = ConfigBindingResolver(env_config)

        with pytest.raises(FactoryNotFoundError, match="No factory registered"):
            resolver.resolve("databases", "primary")

    def test_resolve_missing_resource(self, resolver):
        """Test error when resource not in environment."""
        from dataknobs_config.environment_config import ResourceNotFoundError
        with pytest.raises(ResourceNotFoundError, match="not found"):
            resolver.resolve("databases", "nonexistent")


class TestCacheManagement:
    """Test cache management."""

    @pytest.fixture
    def env_config(self):
        """Environment configuration."""
        return EnvironmentConfig(
            name="test",
            resources={
                "databases": {
                    "primary": {"backend": "postgres"},
                    "secondary": {"backend": "mysql"},
                },
                "caches": {
                    "redis": {"backend": "redis"},
                },
            },
        )

    @pytest.fixture
    def resolver(self, env_config):
        """Create resolver with factories."""
        resolver = ConfigBindingResolver(env_config)
        resolver.register_factory("databases", MockFactory())
        resolver.register_factory("caches", MockFactory())
        return resolver

    def test_clear_cache_all(self, resolver):
        """Test clearing all cache."""
        resolver.resolve("databases", "primary")
        resolver.resolve("databases", "secondary")
        resolver.resolve("caches", "redis")

        resolver.clear_cache()

        assert not resolver.is_cached("databases", "primary")
        assert not resolver.is_cached("databases", "secondary")
        assert not resolver.is_cached("caches", "redis")

    def test_clear_cache_by_type(self, resolver):
        """Test clearing cache by type."""
        resolver.resolve("databases", "primary")
        resolver.resolve("databases", "secondary")
        resolver.resolve("caches", "redis")

        resolver.clear_cache("databases")

        assert not resolver.is_cached("databases", "primary")
        assert not resolver.is_cached("databases", "secondary")
        assert resolver.is_cached("caches", "redis")

    def test_get_cached(self, resolver):
        """Test getting cached instance."""
        assert resolver.get_cached("databases", "primary") is None

        db = resolver.resolve("databases", "primary")
        cached = resolver.get_cached("databases", "primary")

        assert cached is db

    def test_is_cached(self, resolver):
        """Test checking if cached."""
        assert not resolver.is_cached("databases", "primary")

        resolver.resolve("databases", "primary")
        assert resolver.is_cached("databases", "primary")

    def test_cache_instance_manually(self, resolver):
        """Test manually caching an instance."""
        db = MockDatabase(backend="manual")
        resolver.cache_instance("databases", "manual", db)

        cached = resolver.get_cached("databases", "manual")
        assert cached is db


class TestAsyncResolution:
    """Test async resolution."""

    @pytest.fixture
    def env_config(self):
        """Environment configuration."""
        return EnvironmentConfig(
            name="test",
            resources={
                "databases": {
                    "primary": {"backend": "postgres"},
                },
            },
        )

    @pytest.mark.asyncio
    async def test_resolve_async_with_async_factory(self, env_config):
        """Test async resolution with async factory."""
        resolver = ConfigBindingResolver(env_config)
        resolver.register_factory("databases", MockAsyncFactory())

        db = await resolver.resolve_async("databases", "primary")

        assert isinstance(db, MockAsyncDatabase)
        assert db.backend == "postgres"
        assert db.connected is True

    @pytest.mark.asyncio
    async def test_resolve_async_with_sync_factory(self, env_config):
        """Test async resolution falls back to sync factory."""
        resolver = ConfigBindingResolver(env_config)
        resolver.register_factory("databases", MockFactory())

        db = await resolver.resolve_async("databases", "primary")

        assert isinstance(db, MockDatabase)
        assert db.backend == "postgres"

    @pytest.mark.asyncio
    async def test_resolve_async_caching(self, env_config):
        """Test that async resolution uses cache."""
        resolver = ConfigBindingResolver(env_config)
        resolver.register_factory("databases", MockAsyncFactory())

        db1 = await resolver.resolve_async("databases", "primary")
        db2 = await resolver.resolve_async("databases", "primary")

        assert db1 is db2


class TestEnvVarResolution:
    """Test environment variable resolution in configs."""

    @pytest.fixture
    def env_config_with_vars(self):
        """Environment config with env var placeholders."""
        return EnvironmentConfig(
            name="test",
            resources={
                "databases": {
                    "primary": {
                        "backend": "postgres",
                        "host": "${DB_HOST:localhost}",
                    },
                },
            },
        )

    def test_resolve_with_env_vars(self, env_config_with_vars, monkeypatch):
        """Test that env vars are resolved."""
        monkeypatch.setenv("DB_HOST", "db.example.com")

        resolver = ConfigBindingResolver(env_config_with_vars, resolve_env_vars=True)
        resolver.register_factory("databases", MockFactory())

        db = resolver.resolve("databases", "primary")
        assert db.host == "db.example.com"

    def test_resolve_without_env_vars(self, env_config_with_vars):
        """Test that env vars are not resolved when disabled."""
        resolver = ConfigBindingResolver(env_config_with_vars, resolve_env_vars=False)
        resolver.register_factory("databases", MockFactory())

        db = resolver.resolve("databases", "primary")
        assert db.host == "${DB_HOST:localhost}"


class TestSimpleFactory:
    """Test SimpleFactory utility class."""

    def test_simple_factory_create(self):
        """Test creating with SimpleFactory."""
        factory = SimpleFactory(MockDatabase)
        db = factory.create(backend="postgres", host="db.local")

        assert isinstance(db, MockDatabase)
        assert db.backend == "postgres"
        assert db.host == "db.local"

    def test_simple_factory_with_defaults(self):
        """Test SimpleFactory with default kwargs."""
        factory = SimpleFactory(MockDatabase, host="default.local")
        db = factory.create(backend="postgres")

        assert db.backend == "postgres"
        assert db.host == "default.local"

    def test_simple_factory_override_defaults(self):
        """Test overriding SimpleFactory defaults."""
        factory = SimpleFactory(MockDatabase, host="default.local")
        db = factory.create(backend="postgres", host="override.local")

        assert db.host == "override.local"


class TestCallableFactory:
    """Test CallableFactory utility class."""

    def test_callable_factory_create(self):
        """Test creating with CallableFactory."""

        def create_db(backend, host="localhost", **kwargs):
            return MockDatabase(backend=backend, host=host, **kwargs)

        factory = CallableFactory(create_db)
        db = factory.create(backend="postgres", host="db.local")

        assert isinstance(db, MockDatabase)
        assert db.backend == "postgres"
        assert db.host == "db.local"

    def test_callable_factory_with_defaults(self):
        """Test CallableFactory with defaults."""

        def create_db(backend, host, **kwargs):
            return MockDatabase(backend=backend, host=host, **kwargs)

        factory = CallableFactory(create_db, host="default.local")
        db = factory.create(backend="postgres")

        assert db.host == "default.local"


class TestAsyncCallableFactory:
    """Test AsyncCallableFactory utility class."""

    @pytest.mark.asyncio
    async def test_async_callable_factory_create(self):
        """Test creating with AsyncCallableFactory."""

        async def create_db(backend, **kwargs):
            return MockAsyncDatabase(backend=backend, connected=True)

        factory = AsyncCallableFactory(create_db)
        db = await factory.create_async(backend="postgres")

        assert isinstance(db, MockAsyncDatabase)
        assert db.backend == "postgres"
        assert db.connected is True

    def test_async_factory_sync_create_raises(self):
        """Test that sync create raises for async factory."""

        async def create_db(backend, **kwargs):
            return MockAsyncDatabase(backend=backend)

        factory = AsyncCallableFactory(create_db)

        with pytest.raises(RuntimeError, match="requires async context"):
            factory.create(backend="postgres")


class TestFactoryProtocol:
    """Test different factory implementations work."""

    @pytest.fixture
    def env_config(self):
        """Environment configuration."""
        return EnvironmentConfig(
            name="test",
            resources={
                "databases": {
                    "primary": {"backend": "postgres"},
                },
            },
        )

    def test_factory_with_build_method(self, env_config):
        """Test factory with build() instead of create()."""

        class BuildFactory:
            def build(self, backend, **kwargs):
                return MockDatabase(backend=backend, **kwargs)

        resolver = ConfigBindingResolver(env_config)
        resolver.register_factory("databases", BuildFactory())

        db = resolver.resolve("databases", "primary")
        assert db.backend == "postgres"

    def test_callable_as_factory(self, env_config):
        """Test using a callable directly as factory."""

        def create_database(backend, **kwargs):
            return MockDatabase(backend=backend, **kwargs)

        resolver = ConfigBindingResolver(env_config)
        resolver.register_factory("databases", create_database)

        db = resolver.resolve("databases", "primary")
        assert db.backend == "postgres"

    def test_class_as_factory(self, env_config):
        """Test using a class directly as callable factory."""
        resolver = ConfigBindingResolver(env_config)
        resolver.register_factory("databases", MockDatabase)

        db = resolver.resolve("databases", "primary")
        assert isinstance(db, MockDatabase)
        assert db.backend == "postgres"
