"""Tests for ConfigBindingResolver class."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)
from dataknobs_config.binding_resolver import (
    AsyncCallableFactory,
    BindingResolverError,
    CallableFactory,
    ConfigBindingResolver,
    FactoryNotFoundError,
    SimpleFactory,
)
from dataknobs_config.environment_config import EnvironmentConfig


@dataclass(frozen=True)
class _WidgetConfig(StructuredConfig):
    backend: str = "memory"
    host: str = "localhost"


class _AsyncWidget(StructuredConfigConsumer[_WidgetConfig]):
    """StructuredConfig consumer used as a resolver target class."""

    CONFIG_CLS: ClassVar[type[_WidgetConfig]] = _WidgetConfig

    def _setup(self) -> None:
        self.warmed = False

    async def _ainit(self) -> None:
        self.warmed = True


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

    @pytest.mark.asyncio
    async def test_resolve_async_prefers_from_config_async(self, env_config) -> None:
        """A StructuredConfigConsumer target builds via from_config_async.

        The consumer's ``_ainit`` (async-only) runs, and the typed config
        is projected from the resource dict — proving the resolver prefers
        ``from_config_async`` over the kwarg-splat factory paths.
        """
        resolver = ConfigBindingResolver(env_config)
        resolver.register_factory("databases", _AsyncWidget)

        widget = await resolver.resolve_async("databases", "primary")

        assert isinstance(widget, _AsyncWidget)
        assert widget.warmed is True
        assert widget.config.backend == "postgres"


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


class TestSubstitutionProvenance:
    """The resolver substitutes only environments that have not been.

    This is the most severe of the double-expansion sites: the corrupted
    value goes straight into a live resource via a factory, leaving no
    resolved-config artifact for anyone to diff. Assertions here are made on
    the config the factory actually receives, not on an intermediate.
    """

    SECRET = "p${PROV_INNER}ss"

    @pytest.fixture(autouse=True)
    def _prov_env(self, monkeypatch):
        monkeypatch.setenv("PROV_PW", self.SECRET)
        monkeypatch.setenv("PROV_INNER", "INJECTED")

    @staticmethod
    def _resolve_through_factory(env, **kwargs):
        """Return the config a factory is handed for ``databases[main]``."""
        built: dict = {}

        def factory(**config):
            built.update(config)
            return object()

        resolver = ConfigBindingResolver(env)
        resolver.register_factory("databases", factory)
        resolver.resolve("databases", "main", **kwargs)
        return built

    @staticmethod
    def _payload():
        return {
            "name": "test",
            "resources": {"databases": {"main": {"backend": "postgres", "password": "${PROV_PW}"}}},
        }

    def test_substituted_environment_is_not_substituted_again(self):
        env = EnvironmentConfig.from_dict(self._payload())

        built = self._resolve_through_factory(env)

        assert built["password"] == self.SECRET

    def test_directly_constructed_environment_is_still_substituted(self):
        """The path the removed comment correctly identified as load-bearing.

        A dataclass-constructed environment has never had substitution
        applied, so the resolver must apply it — exactly once.
        """
        env = EnvironmentConfig(
            name="test",
            resources={"databases": {"main": {"backend": "postgres", "password": "${PROV_PW}"}}},
        )

        built = self._resolve_through_factory(env)

        assert built["password"] == self.SECRET

    def test_opt_out_environment_is_still_substituted(self):
        env = EnvironmentConfig.from_dict(self._payload(), substitute_vars=False)

        built = self._resolve_through_factory(env)

        assert built["password"] == self.SECRET

    def test_resolve_env_vars_false_leaves_refs_intact(self):
        env = EnvironmentConfig.from_dict(self._payload(), substitute_vars=False)
        built: dict = {}

        def factory(**config):
            built.update(config)
            return object()

        resolver = ConfigBindingResolver(env, resolve_env_vars=False)
        resolver.register_factory("databases", factory)
        resolver.resolve("databases", "main")

        assert built["password"] == "${PROV_PW}"

    @pytest.mark.parametrize("substitute_vars", [True, False])
    def test_overrides_get_their_own_single_pass(self, substitute_vars):
        """Overrides are a separate source with separate provenance.

        They are caller-supplied and have never been substituted, so they are
        expanded here regardless of whether the *environment* was expanded at
        load. Gating them on the environment's provenance would silently stop
        substituting them for every normally-loaded environment — a
        behaviour change to a source the flag says nothing about.

        Whether overrides should be substituted at all is a separate
        question (199-FU2); this pins the behaviour they have always had.
        """
        env = EnvironmentConfig.from_dict(self._payload(), substitute_vars=substitute_vars)

        built = self._resolve_through_factory(env, password="${PROV_PW}")

        assert built["password"] == self.SECRET, (
            f"override expanded the wrong number of times with substitute_vars={substitute_vars}"
        )

    def test_overrides_are_not_substituted_when_disabled(self):
        env = EnvironmentConfig.from_dict(self._payload())
        built: dict = {}

        def factory(**config):
            built.update(config)
            return object()

        resolver = ConfigBindingResolver(env, resolve_env_vars=False)
        resolver.register_factory("databases", factory)
        resolver.resolve("databases", "main", password="${PROV_PW}")

        assert built["password"] == "${PROV_PW}"

    def test_resolving_does_not_mutate_the_environment(self):
        """The environment outlives the resolution and must come out intact.

        This replaces a test that asserted a caller's ``**overrides`` mapping
        was unmodified. ``resolve(**overrides)`` materializes a fresh dict per
        call, so nothing downstream could ever have written through to the
        caller's own mapping and the assertion could not fail. The property
        actually at risk is one layer over: the resolved config is derived
        from the environment's stored values, and the environment is reused.
        """
        env = EnvironmentConfig.from_dict(self._payload())

        first = self._resolve_through_factory(env, password="override")
        second = self._resolve_through_factory(env)

        assert first["password"] == "override"
        assert second["password"] == self.SECRET

    def test_a_factory_adjusting_a_nested_section_cannot_reach_the_env(self):
        """``get_resource`` copies deeply, so a factory owns what it is given.

        A shallow copy leaves nested containers pointing at the environment's
        own objects. It went unnoticed while a substitution pass always ran
        afterwards -- rebuilding the structure through comprehensions isolated
        the result incidentally -- and surfaced once that pass was correctly
        skipped for an already-substituted environment.
        """
        payload = self._payload()
        payload["resources"]["databases"]["main"]["pool"] = {"max": 10}
        env = EnvironmentConfig.from_dict(payload)

        first = self._resolve_through_factory(env)
        first["pool"]["max"] = 999

        assert self._resolve_through_factory(env)["pool"]["max"] == 10
