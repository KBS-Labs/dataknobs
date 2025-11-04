"""Tests for LLM providers infrastructure."""

import pytest
from dataknobs_llm.llm.base import LLMConfig, LLMMessage, AsyncLLMProvider
from dataknobs_llm.llm.providers import (
    LLMProviderFactory,
    create_llm_provider,
    EchoProvider,
    SyncProviderAdapter,
)


class TestLLMProviderFactory:
    """Test LLMProviderFactory."""

    def test_factory_creation_async(self):
        """Test creating async factory."""
        factory = LLMProviderFactory(is_async=True)
        assert factory.is_async is True

    def test_factory_creation_sync(self):
        """Test creating sync factory."""
        factory = LLMProviderFactory(is_async=False)
        assert factory.is_async is False

    def test_factory_creates_echo_provider_async(self):
        """Test factory creates EchoProvider in async mode."""
        factory = LLMProviderFactory(is_async=True)
        config = LLMConfig(provider="echo", model="echo-model")

        provider = factory.create(config)

        assert isinstance(provider, EchoProvider)
        assert isinstance(provider, AsyncLLMProvider)
        assert provider.config.provider == "echo"
        assert provider.config.model == "echo-model"

    def test_factory_creates_echo_provider_sync(self):
        """Test factory creates wrapped EchoProvider in sync mode."""
        factory = LLMProviderFactory(is_async=False)
        config = LLMConfig(provider="echo", model="echo-model")

        provider = factory.create(config)

        assert isinstance(provider, SyncProviderAdapter)
        assert isinstance(provider.async_provider, EchoProvider)

    def test_factory_create_from_dict(self):
        """Test factory creates provider from dictionary config."""
        factory = LLMProviderFactory(is_async=True)
        config_dict = {
            "provider": "echo",
            "model": "echo-model",
            "options": {"echo_prefix": "Test: "}
        }

        provider = factory.create(config_dict)

        assert isinstance(provider, EchoProvider)
        assert provider.echo_prefix == "Test: "

    def test_factory_create_from_llm_config(self):
        """Test factory creates provider from LLMConfig."""
        factory = LLMProviderFactory(is_async=True)
        config = LLMConfig(
            provider="echo",
            model="echo-model",
            temperature=0.5,
            max_tokens=100
        )

        provider = factory.create(config)

        assert isinstance(provider, EchoProvider)
        assert provider.config.temperature == 0.5
        assert provider.config.max_tokens == 100

    def test_factory_unknown_provider_raises_error(self):
        """Test factory raises error for unknown provider."""
        factory = LLMProviderFactory(is_async=True)
        config = LLMConfig(provider="unknown", model="test")

        with pytest.raises(ValueError) as exc_info:
            factory.create(config)

        assert "Unknown provider: unknown" in str(exc_info.value)
        assert "Available providers:" in str(exc_info.value)

    def test_factory_callable_interface(self):
        """Test factory can be called directly."""
        factory = LLMProviderFactory(is_async=True)
        config = LLMConfig(provider="echo", model="echo-model")

        # Test __call__ method
        provider = factory(config)

        assert isinstance(provider, EchoProvider)

    def test_factory_register_custom_provider(self):
        """Test registering a custom provider class."""
        # Create a simple custom provider for testing
        class CustomTestProvider(AsyncLLMProvider):
            async def complete(self, messages, **kwargs):
                return None

            async def stream_complete(self, messages, **kwargs):
                yield None

            async def embed(self, texts, **kwargs):
                return []

            async def function_call(self, messages, functions, **kwargs):
                return None

            async def validate_model(self):
                return True

            def get_capabilities(self):
                return []

        # Register the custom provider
        LLMProviderFactory.register_provider('custom', CustomTestProvider)

        # Create provider using factory
        factory = LLMProviderFactory(is_async=True)
        config = LLMConfig(provider="custom", model="test")
        provider = factory.create(config)

        assert isinstance(provider, CustomTestProvider)

    def test_factory_case_insensitive_provider_name(self):
        """Test factory handles case-insensitive provider names."""
        factory = LLMProviderFactory(is_async=True)

        # Test various casings
        for provider_name in ["echo", "Echo", "ECHO", "EcHo"]:
            config = LLMConfig(provider=provider_name, model="echo-model")
            provider = factory.create(config)
            assert isinstance(provider, EchoProvider)

    def test_factory_preserves_config_options(self):
        """Test factory preserves all config options."""
        factory = LLMProviderFactory(is_async=True)
        config = LLMConfig(
            provider="echo",
            model="echo-model",
            temperature=0.8,
            max_tokens=500,
            top_p=0.9,
            options={
                "echo_prefix": ">>> ",
                "embedding_dim": 384
            }
        )

        provider = factory.create(config)

        assert provider.config.temperature == 0.8
        assert provider.config.max_tokens == 500
        assert provider.config.top_p == 0.9
        assert provider.echo_prefix == ">>> "
        assert provider.embedding_dim == 384


class TestCreateLLMProvider:
    """Test create_llm_provider convenience function."""

    def test_create_llm_provider_async(self):
        """Test create_llm_provider creates async provider."""
        config = LLMConfig(provider="echo", model="echo-model")
        provider = create_llm_provider(config, is_async=True)

        assert isinstance(provider, EchoProvider)
        assert isinstance(provider, AsyncLLMProvider)

    def test_create_llm_provider_sync(self):
        """Test create_llm_provider creates sync provider."""
        config = LLMConfig(provider="echo", model="echo-model")
        provider = create_llm_provider(config, is_async=False)

        assert isinstance(provider, SyncProviderAdapter)
        assert isinstance(provider.async_provider, EchoProvider)

    def test_create_llm_provider_from_dict(self):
        """Test create_llm_provider from dictionary."""
        config_dict = {
            "provider": "echo",
            "model": "echo-model"
        }
        provider = create_llm_provider(config_dict)

        assert isinstance(provider, EchoProvider)

    def test_create_llm_provider_defaults_to_async(self):
        """Test create_llm_provider defaults to async mode."""
        config = LLMConfig(provider="echo", model="echo-model")
        provider = create_llm_provider(config)

        # Should be async by default
        assert isinstance(provider, EchoProvider)
        assert not isinstance(provider, SyncProviderAdapter)


class TestSyncProviderAdapter:
    """Test SyncProviderAdapter."""

    @pytest.fixture
    def async_echo_provider(self):
        """Create an async EchoProvider for testing."""
        config = LLMConfig(provider="echo", model="echo-model")
        return EchoProvider(config)

    @pytest.fixture
    def sync_adapter(self, async_echo_provider):
        """Create a SyncProviderAdapter wrapping EchoProvider."""
        return SyncProviderAdapter(async_echo_provider)

    def test_adapter_wraps_async_provider(self, async_echo_provider):
        """Test adapter wraps async provider."""
        adapter = SyncProviderAdapter(async_echo_provider)
        assert adapter.async_provider is async_echo_provider

    def test_adapter_initialize(self, sync_adapter):
        """Test adapter initialize method."""
        # Should not raise an error
        sync_adapter.initialize()
        assert sync_adapter.async_provider.is_initialized

    def test_adapter_close(self, sync_adapter):
        """Test adapter close method."""
        sync_adapter.initialize()
        sync_adapter.close()
        assert not sync_adapter.async_provider.is_initialized

    def test_adapter_complete(self, sync_adapter):
        """Test adapter complete method."""
        sync_adapter.initialize()

        response = sync_adapter.complete("Hello, world!")

        assert response.content == "Echo: Hello, world!"
        assert response.model == "echo-model"

        sync_adapter.close()

    def test_adapter_complete_with_messages(self, sync_adapter):
        """Test adapter complete with message list."""
        sync_adapter.initialize()

        messages = [
            LLMMessage(role="user", content="Test message")
        ]
        response = sync_adapter.complete(messages)

        assert response.content == "Echo: Test message"

        sync_adapter.close()

    def test_adapter_embed(self, sync_adapter):
        """Test adapter embed method."""
        sync_adapter.initialize()

        embedding = sync_adapter.embed("Test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 768  # Default embedding_dim
        assert all(isinstance(x, float) for x in embedding)

        sync_adapter.close()

    def test_adapter_embed_multiple(self, sync_adapter):
        """Test adapter embed with multiple texts."""
        sync_adapter.initialize()

        texts = ["First", "Second", "Third"]
        embeddings = sync_adapter.embed(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)

        sync_adapter.close()

    def test_adapter_function_call(self, sync_adapter):
        """Test adapter function_call method."""
        sync_adapter.initialize()

        messages = [LLMMessage(role="user", content="Test")]
        functions = [
            {
                "name": "test_func",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param": {"type": "string"}
                    }
                }
            }
        ]

        response = sync_adapter.function_call(messages, functions)

        assert response.finish_reason == "function_call"
        assert response.function_call is not None
        assert response.function_call["name"] == "test_func"

        sync_adapter.close()

    def test_adapter_stream(self, sync_adapter):
        """Test adapter stream method."""
        sync_adapter.initialize()

        chunks = list(sync_adapter.stream("Hi"))

        assert len(chunks) > 0
        # Reconstruct message
        content = "".join(chunk.delta for chunk in chunks)
        assert content == "Echo: Hi"
        assert chunks[-1].is_final is True

        sync_adapter.close()

    def test_adapter_is_initialized_property(self, async_echo_provider):
        """Test adapter is_initialized property."""
        adapter = SyncProviderAdapter(async_echo_provider)

        assert not adapter.is_initialized

        adapter.initialize()
        assert adapter.is_initialized

        adapter.close()
        assert not adapter.is_initialized

    def test_adapter_get_capabilities(self, sync_adapter):
        """Test adapter get_capabilities method."""
        capabilities = sync_adapter.get_capabilities()

        # Should delegate to async provider
        assert len(capabilities) > 0

    def test_adapter_validate_model(self, sync_adapter):
        """Test adapter validate_model method."""
        is_valid = sync_adapter.validate_model()

        # EchoProvider always returns True
        assert is_valid is True


class TestProviderIntegration:
    """Integration tests using multiple components."""

    def test_factory_to_adapter_workflow(self):
        """Test complete workflow from factory to sync adapter."""
        # Create factory for sync mode
        factory = LLMProviderFactory(is_async=False)

        # Create provider from config
        config = LLMConfig(
            provider="echo",
            model="echo-model",
            options={"echo_prefix": "Test> "}
        )
        provider = factory.create(config)

        # Verify it's wrapped
        assert isinstance(provider, SyncProviderAdapter)
        assert isinstance(provider.async_provider, EchoProvider)

        # Use it
        provider.initialize()
        try:
            response = provider.complete("Hello")
            assert response.content == "Test> Hello"
        finally:
            provider.close()

    def test_create_llm_provider_end_to_end(self):
        """Test create_llm_provider end-to-end."""
        # Create from dict in sync mode
        provider = create_llm_provider({
            "provider": "echo",
            "model": "test",
            "options": {
                "echo_prefix": "[ECHO] ",
                "embedding_dim": 256
            }
        }, is_async=False)

        # Use the provider
        provider.initialize()
        try:
            # Test completion
            response = provider.complete("Test message")
            assert "[ECHO] Test message" in response.content

            # Test embedding
            embedding = provider.embed("Test")
            assert len(embedding) == 256

            # Test streaming
            chunks = list(provider.stream("Hi"))
            assert len(chunks) > 0
        finally:
            provider.close()

    def test_multiple_providers_from_same_factory(self):
        """Test creating multiple providers from same factory."""
        factory = LLMProviderFactory(is_async=True)

        # Create multiple echo providers with different configs
        provider1 = factory.create({
            "provider": "echo",
            "model": "echo-1",
            "options": {"echo_prefix": "A> "}
        })

        provider2 = factory.create({
            "provider": "echo",
            "model": "echo-2",
            "options": {"echo_prefix": "B> "}
        })

        # They should be independent
        assert provider1.echo_prefix == "A> "
        assert provider2.echo_prefix == "B> "
        assert provider1.config.model == "echo-1"
        assert provider2.config.model == "echo-2"
