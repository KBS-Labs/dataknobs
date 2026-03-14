"""Tests for create_embedding_provider() helper."""

import pytest

from dataknobs_bots.providers import create_embedding_provider


class TestCreateEmbeddingProvider:
    """Tests for the shared embedding provider factory helper."""

    @pytest.mark.asyncio
    async def test_nested_config(self) -> None:
        """Nested embedding sub-dict format."""
        provider = await create_embedding_provider({
            "embedding": {
                "provider": "echo",
                "model": "test-embed-nested",
            },
        })
        try:
            assert provider.config.provider == "echo"
            assert provider.config.model == "test-embed-nested"
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_legacy_prefix_config(self) -> None:
        """Legacy embedding_provider/embedding_model keys."""
        provider = await create_embedding_provider({
            "embedding_provider": "echo",
            "embedding_model": "test-embed-legacy",
        })
        try:
            assert provider.config.provider == "echo"
            assert provider.config.model == "test-embed-legacy"
        finally:
            await provider.close()

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
        try:
            assert provider.config.model == "nested-model"
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_extra_keys_forwarded_from_nested(self) -> None:
        """Extra keys in the embedding sub-dict pass through to provider."""
        provider = await create_embedding_provider({
            "embedding": {
                "provider": "echo",
                "model": "test-embed",
                "api_base": "http://custom:8080",
                "dimensions": 768,
            },
        })
        try:
            assert provider.config.provider == "echo"
            assert provider.config.api_base == "http://custom:8080"
            assert provider.config.dimensions == 768
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_extra_keys_forwarded_from_legacy(self) -> None:
        """api_base, api_key, and dimensions from top-level config pass through."""
        provider = await create_embedding_provider({
            "embedding_provider": "echo",
            "embedding_model": "test-embed",
            "dimensions": 768,
            "api_base": "http://custom:8080",
            "api_key": "test-key-123",
        })
        try:
            assert provider.config.provider == "echo"
            assert provider.config.api_base == "http://custom:8080"
            assert provider.config.dimensions == 768
            assert provider.config.api_key == "test-key-123"
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_unrelated_top_level_keys_ignored(self) -> None:
        """Keys like 'backend' or 'type' don't bleed into provider config."""
        provider = await create_embedding_provider({
            "embedding_provider": "echo",
            "embedding_model": "test-embed",
            "backend": "memory",
            "type": "vector",
            "max_results": 5,
        })
        try:
            assert provider.config.provider == "echo"
            assert provider.config.model == "test-embed"
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_provider_is_initialized(self) -> None:
        """Returned provider has been initialized."""
        provider = await create_embedding_provider({
            "embedding_provider": "echo",
            "embedding_model": "test-embed",
        })
        try:
            assert provider.is_initialized is True
        finally:
            await provider.close()

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
        try:
            assert provider.config.provider == "echo"
            assert provider.config.model == "default-test"
        finally:
            await provider.close()
