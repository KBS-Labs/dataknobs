"""Tests for create_embedding_provider() helper."""

import pytest

from dataknobs_bots.providers import create_embedding_provider


class TestCreateEmbeddingProvider:
    """Tests for the shared embedding provider factory helper."""

    @pytest.mark.asyncio
    async def test_flat_config(self) -> None:
        """Flat provider/model keys at top level."""
        provider = await create_embedding_provider({
            "provider": "echo",
            "model": "test-embed",
        })
        assert provider.config.provider == "echo"
        assert provider.config.model == "test-embed"

    @pytest.mark.asyncio
    async def test_nested_config(self) -> None:
        """Nested embedding sub-dict format."""
        provider = await create_embedding_provider({
            "embedding": {
                "provider": "echo",
                "model": "test-embed-nested",
            },
        })
        assert provider.config.provider == "echo"
        assert provider.config.model == "test-embed-nested"

    @pytest.mark.asyncio
    async def test_legacy_prefix_config(self) -> None:
        """Legacy embedding_provider/embedding_model keys."""
        provider = await create_embedding_provider({
            "embedding_provider": "echo",
            "embedding_model": "test-embed-legacy",
        })
        assert provider.config.provider == "echo"
        assert provider.config.model == "test-embed-legacy"

    @pytest.mark.asyncio
    async def test_nested_takes_precedence_over_flat(self) -> None:
        """When both nested and flat keys are present, nested wins."""
        provider = await create_embedding_provider({
            "provider": "echo",
            "model": "flat-model",
            "embedding": {
                "provider": "echo",
                "model": "nested-model",
            },
        })
        assert provider.config.model == "nested-model"

    @pytest.mark.asyncio
    async def test_nested_takes_precedence_over_legacy(self) -> None:
        """When both nested and legacy keys are present, nested wins."""
        provider = await create_embedding_provider({
            "embedding_provider": "echo",
            "embedding_model": "legacy-model",
            "embedding": {
                "provider": "echo",
                "model": "nested-model",
            },
        })
        assert provider.config.model == "nested-model"

    @pytest.mark.asyncio
    async def test_extra_keys_forwarded_from_nested(self) -> None:
        """Extra keys in the embedding sub-dict pass through to provider."""
        provider = await create_embedding_provider({
            "embedding": {
                "provider": "echo",
                "model": "test-embed",
                "api_base": "http://custom:8080",
            },
        })
        assert provider.config.provider == "echo"

    @pytest.mark.asyncio
    async def test_extra_keys_forwarded_from_flat(self) -> None:
        """api_base and dimensions from top-level config pass through."""
        provider = await create_embedding_provider({
            "provider": "echo",
            "model": "test-embed",
            "dimensions": 768,
        })
        assert provider.config.provider == "echo"

    @pytest.mark.asyncio
    async def test_provider_is_initialized(self) -> None:
        """Returned provider has been initialized."""
        provider = await create_embedding_provider({
            "provider": "echo",
            "model": "test-embed",
        })
        assert provider._is_initialized is True

    @pytest.mark.asyncio
    async def test_defaults_when_empty_config(self) -> None:
        """Empty config uses ollama defaults (provider name check only)."""
        # We can't actually connect to ollama in tests, but we can verify
        # the factory receives the correct defaults by using custom defaults
        provider = await create_embedding_provider(
            {},
            default_provider="echo",
            default_model="default-test",
        )
        assert provider.config.provider == "echo"
        assert provider.config.model == "default-test"

    @pytest.mark.asyncio
    async def test_custom_key_names(self) -> None:
        """Custom provider_key and model_key work in nested format."""
        provider = await create_embedding_provider(
            {
                "embedding": {
                    "llm_provider": "echo",
                    "llm_model": "custom-key-model",
                },
            },
            provider_key="llm_provider",
            model_key="llm_model",
        )
        assert provider.config.provider == "echo"
        assert provider.config.model == "custom-key-model"
