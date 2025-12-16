"""Tests for resource resolution utilities."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCreateBotResolver:
    """Tests for create_bot_resolver function."""

    @pytest.fixture
    def mock_env_config(self):
        """Create mock environment config."""
        mock = MagicMock()
        mock.name = "test"
        mock.resources = {
            "llm_providers": {
                "default": {"provider": "echo", "model": "test"},
            },
            "databases": {
                "conversations": {"backend": "memory"},
            },
            "vector_stores": {
                "knowledge": {"backend": "memory", "dimensions": 768},
            },
            "embedding_providers": {
                "default": {"provider": "echo", "model": "test-embed"},
            },
        }
        mock.get_resource = MagicMock(
            side_effect=lambda type_, name: mock.resources[type_][name]
        )
        return mock

    def test_create_bot_resolver_registers_all_factories(self, mock_env_config):
        """Test that create_bot_resolver registers all default factories."""
        # Patch at the source modules where imports come from
        with patch("dataknobs_config.ConfigBindingResolver") as mock_resolver_class, \
             patch("dataknobs_llm.llm.LLMProviderFactory"), \
             patch("dataknobs_data.factory.AsyncDatabaseFactory"), \
             patch("dataknobs_data.vector.stores.VectorStoreFactory"):
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver

            from dataknobs_bots.config import create_bot_resolver

            resolver = create_bot_resolver(mock_env_config)

            # Should register 4 factories
            assert mock_resolver.register_factory.call_count == 4

            # Check which types were registered
            registered_types = [
                call.args[0] for call in mock_resolver.register_factory.call_args_list
            ]
            assert "llm_providers" in registered_types
            assert "databases" in registered_types
            assert "vector_stores" in registered_types
            assert "embedding_providers" in registered_types

    def test_create_bot_resolver_no_defaults(self, mock_env_config):
        """Test creating resolver without registering defaults."""
        with patch("dataknobs_config.ConfigBindingResolver") as mock_resolver_class:
            mock_resolver = MagicMock()
            mock_resolver_class.return_value = mock_resolver

            from dataknobs_bots.config import create_bot_resolver

            resolver = create_bot_resolver(mock_env_config, register_defaults=False)

            # Should not register any factories
            assert mock_resolver.register_factory.call_count == 0


class TestIndividualFactoryRegistration:
    """Tests for individual factory registration functions."""

    @pytest.fixture
    def mock_resolver(self):
        """Create mock resolver."""
        return MagicMock()

    def test_register_llm_factory(self, mock_resolver):
        """Test registering LLM factory."""
        with patch("dataknobs_llm.llm.LLMProviderFactory") as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory_class.return_value = mock_factory

            from dataknobs_bots.config.resolution import register_llm_factory

            register_llm_factory(mock_resolver)

            mock_factory_class.assert_called_once_with(is_async=True)
            mock_resolver.register_factory.assert_called_once_with(
                "llm_providers", mock_factory
            )

    def test_register_database_factory(self, mock_resolver):
        """Test registering database factory."""
        with patch("dataknobs_data.factory.AsyncDatabaseFactory") as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory_class.return_value = mock_factory

            from dataknobs_bots.config.resolution import register_database_factory

            register_database_factory(mock_resolver)

            mock_factory_class.assert_called_once()
            mock_resolver.register_factory.assert_called_once_with(
                "databases", mock_factory
            )

    def test_register_vector_store_factory(self, mock_resolver):
        """Test registering vector store factory."""
        with patch("dataknobs_data.vector.stores.VectorStoreFactory") as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory_class.return_value = mock_factory

            from dataknobs_bots.config.resolution import register_vector_store_factory

            register_vector_store_factory(mock_resolver)

            mock_factory_class.assert_called_once()
            mock_resolver.register_factory.assert_called_once_with(
                "vector_stores", mock_factory
            )

    def test_register_embedding_factory(self, mock_resolver):
        """Test registering embedding factory."""
        with patch("dataknobs_llm.llm.LLMProviderFactory") as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory_class.return_value = mock_factory

            from dataknobs_bots.config.resolution import register_embedding_factory

            register_embedding_factory(mock_resolver)

            mock_factory_class.assert_called_once_with(is_async=True)
            mock_resolver.register_factory.assert_called_once_with(
                "embedding_providers", mock_factory
            )


class TestBotResourceResolver:
    """Tests for BotResourceResolver class."""

    @pytest.fixture
    def mock_env_config(self):
        """Create mock environment config."""
        mock = MagicMock()
        mock.name = "test"
        return mock

    @pytest.fixture
    def mock_underlying_resolver(self):
        """Create mock underlying resolver."""
        mock = MagicMock()
        mock.get_registered_types.return_value = [
            "llm_providers",
            "databases",
            "vector_stores",
            "embedding_providers",
        ]
        return mock

    def test_init_creates_resolver(self, mock_env_config):
        """Test that init creates underlying resolver."""
        with patch(
            "dataknobs_bots.config.resolution.create_bot_resolver"
        ) as mock_create:
            mock_resolver = MagicMock()
            mock_create.return_value = mock_resolver

            from dataknobs_bots.config import BotResourceResolver

            bot_resolver = BotResourceResolver(mock_env_config)

            mock_create.assert_called_once_with(
                mock_env_config, resolve_env_vars=True
            )
            assert bot_resolver.environment is mock_env_config
            assert bot_resolver.resolver is mock_resolver

    @pytest.mark.asyncio
    async def test_get_llm(self, mock_env_config):
        """Test getting initialized LLM provider."""
        with patch(
            "dataknobs_bots.config.resolution.create_bot_resolver"
        ) as mock_create:
            mock_resolver = MagicMock()
            mock_llm = AsyncMock()
            mock_resolver.resolve.return_value = mock_llm
            mock_create.return_value = mock_resolver

            from dataknobs_bots.config import BotResourceResolver

            bot_resolver = BotResourceResolver(mock_env_config)
            llm = await bot_resolver.get_llm("default")

            mock_resolver.resolve.assert_called_once_with(
                "llm_providers", "default", use_cache=True
            )
            mock_llm.initialize.assert_awaited_once()
            assert llm is mock_llm

    @pytest.mark.asyncio
    async def test_get_llm_with_overrides(self, mock_env_config):
        """Test getting LLM with overrides."""
        with patch(
            "dataknobs_bots.config.resolution.create_bot_resolver"
        ) as mock_create:
            mock_resolver = MagicMock()
            mock_llm = AsyncMock()
            mock_resolver.resolve.return_value = mock_llm
            mock_create.return_value = mock_resolver

            from dataknobs_bots.config import BotResourceResolver

            bot_resolver = BotResourceResolver(mock_env_config)
            llm = await bot_resolver.get_llm(
                "default", use_cache=False, temperature=0.5
            )

            mock_resolver.resolve.assert_called_once_with(
                "llm_providers", "default", use_cache=False, temperature=0.5
            )

    @pytest.mark.asyncio
    async def test_get_database(self, mock_env_config):
        """Test getting initialized database."""
        with patch(
            "dataknobs_bots.config.resolution.create_bot_resolver"
        ) as mock_create:
            mock_resolver = MagicMock()
            mock_db = AsyncMock()
            mock_resolver.resolve.return_value = mock_db
            mock_create.return_value = mock_resolver

            from dataknobs_bots.config import BotResourceResolver

            bot_resolver = BotResourceResolver(mock_env_config)
            db = await bot_resolver.get_database("conversations")

            mock_resolver.resolve.assert_called_once_with(
                "databases", "conversations", use_cache=True
            )
            mock_db.connect.assert_awaited_once()
            assert db is mock_db

    @pytest.mark.asyncio
    async def test_get_database_no_connect(self, mock_env_config):
        """Test getting database without connect method."""
        with patch(
            "dataknobs_bots.config.resolution.create_bot_resolver"
        ) as mock_create:
            mock_resolver = MagicMock()
            mock_db = MagicMock(spec=[])  # No connect method
            mock_resolver.resolve.return_value = mock_db
            mock_create.return_value = mock_resolver

            from dataknobs_bots.config import BotResourceResolver

            bot_resolver = BotResourceResolver(mock_env_config)
            db = await bot_resolver.get_database("conversations")

            assert db is mock_db

    @pytest.mark.asyncio
    async def test_get_vector_store(self, mock_env_config):
        """Test getting initialized vector store."""
        with patch(
            "dataknobs_bots.config.resolution.create_bot_resolver"
        ) as mock_create:
            mock_resolver = MagicMock()
            mock_vs = AsyncMock()
            mock_resolver.resolve.return_value = mock_vs
            mock_create.return_value = mock_resolver

            from dataknobs_bots.config import BotResourceResolver

            bot_resolver = BotResourceResolver(mock_env_config)
            vs = await bot_resolver.get_vector_store("knowledge")

            mock_resolver.resolve.assert_called_once_with(
                "vector_stores", "knowledge", use_cache=True
            )
            mock_vs.initialize.assert_awaited_once()
            assert vs is mock_vs

    @pytest.mark.asyncio
    async def test_get_embedding_provider(self, mock_env_config):
        """Test getting initialized embedding provider."""
        with patch(
            "dataknobs_bots.config.resolution.create_bot_resolver"
        ) as mock_create:
            mock_resolver = MagicMock()
            mock_embedder = AsyncMock()
            mock_resolver.resolve.return_value = mock_embedder
            mock_create.return_value = mock_resolver

            from dataknobs_bots.config import BotResourceResolver

            bot_resolver = BotResourceResolver(mock_env_config)
            embedder = await bot_resolver.get_embedding_provider("default")

            mock_resolver.resolve.assert_called_once_with(
                "embedding_providers", "default", use_cache=True
            )
            mock_embedder.initialize.assert_awaited_once()
            assert embedder is mock_embedder

    def test_clear_cache(self, mock_env_config):
        """Test clearing cache."""
        with patch(
            "dataknobs_bots.config.resolution.create_bot_resolver"
        ) as mock_create:
            mock_resolver = MagicMock()
            mock_create.return_value = mock_resolver

            from dataknobs_bots.config import BotResourceResolver

            bot_resolver = BotResourceResolver(mock_env_config)

            # Clear all cache
            bot_resolver.clear_cache()
            mock_resolver.clear_cache.assert_called_with(None)

            # Clear specific type
            bot_resolver.clear_cache("llm_providers")
            mock_resolver.clear_cache.assert_called_with("llm_providers")

    def test_repr(self, mock_env_config):
        """Test string representation."""
        with patch(
            "dataknobs_bots.config.resolution.create_bot_resolver"
        ) as mock_create:
            mock_resolver = MagicMock()
            mock_resolver.get_registered_types.return_value = ["llm_providers", "databases"]
            mock_create.return_value = mock_resolver

            from dataknobs_bots.config import BotResourceResolver

            bot_resolver = BotResourceResolver(mock_env_config)
            repr_str = repr(bot_resolver)

            assert "BotResourceResolver" in repr_str
            assert "test" in repr_str
            assert "llm_providers" in repr_str


class TestModuleExports:
    """Test module exports."""

    def test_config_module_exports(self):
        """Test that config module exports expected items."""
        from dataknobs_bots.config import (
            BotResourceResolver,
            create_bot_resolver,
            register_database_factory,
            register_embedding_factory,
            register_llm_factory,
            register_vector_store_factory,
        )

        # Just verify they're importable
        assert callable(create_bot_resolver)
        assert callable(register_llm_factory)
        assert callable(register_database_factory)
        assert callable(register_vector_store_factory)
        assert callable(register_embedding_factory)
        assert isinstance(BotResourceResolver, type)


class TestIntegrationWithRealResolver:
    """Integration tests using the real ConfigBindingResolver."""

    @pytest.fixture
    def test_env_config(self):
        """Create a real environment config for testing."""
        from dataknobs_config import EnvironmentConfig

        return EnvironmentConfig(
            name="test",
            resources={
                "llm_providers": {
                    "default": {"provider": "echo", "model": "test"},
                },
                "databases": {
                    "conversations": {"backend": "memory"},
                },
                "vector_stores": {
                    "knowledge": {"backend": "memory", "dimensions": 768},
                },
                "embedding_providers": {
                    "default": {"provider": "echo", "model": "test-embed"},
                },
            },
        )

    def test_create_resolver_with_real_env(self, test_env_config):
        """Test creating resolver with real environment config."""
        from dataknobs_bots.config import create_bot_resolver

        resolver = create_bot_resolver(test_env_config)

        # Verify all factories are registered
        assert resolver.has_factory("llm_providers")
        assert resolver.has_factory("databases")
        assert resolver.has_factory("vector_stores")
        assert resolver.has_factory("embedding_providers")

        # Verify registered types
        types = resolver.get_registered_types()
        assert len(types) == 4

    def test_resolver_env_vars_flag(self, test_env_config):
        """Test resolve_env_vars parameter."""
        from dataknobs_bots.config import create_bot_resolver

        # With env var resolution (default)
        resolver = create_bot_resolver(test_env_config, resolve_env_vars=True)
        assert resolver._resolve_env_vars is True

        # Without env var resolution
        resolver = create_bot_resolver(test_env_config, resolve_env_vars=False)
        assert resolver._resolve_env_vars is False

    def test_bot_resource_resolver_with_real_env(self, test_env_config):
        """Test BotResourceResolver with real environment config."""
        from dataknobs_bots.config import BotResourceResolver

        bot_resolver = BotResourceResolver(test_env_config)

        assert bot_resolver.environment is test_env_config
        assert bot_resolver.resolver.has_factory("llm_providers")

    def test_resolver_cache_clear(self, test_env_config):
        """Test cache clearing on resolver."""
        from dataknobs_bots.config import BotResourceResolver

        bot_resolver = BotResourceResolver(test_env_config)

        # Clear cache should not error
        bot_resolver.clear_cache()
        bot_resolver.clear_cache("llm_providers")
