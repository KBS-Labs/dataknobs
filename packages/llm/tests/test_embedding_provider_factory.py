"""Tests for create_embedding_provider().

This is the canonical test suite — the function lives in
``dataknobs_llm`` and is re-exported from ``dataknobs_bots.providers``
for backward compatibility.
"""

import pytest

from dataknobs_llm import CompletionMode, LLMConfig, create_embedding_provider


class TestCreateEmbeddingProvider:
    """Tests for the embedding provider factory."""

    def test_importable_from_top_level(self) -> None:
        """create_embedding_provider is importable from dataknobs_llm."""
        assert callable(create_embedding_provider)

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
        """Empty config uses custom defaults when provided."""
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

    @pytest.mark.asyncio
    async def test_completion_mode_is_embedding_nested(self) -> None:
        """Nested format sets CompletionMode.EMBEDDING."""
        provider = await create_embedding_provider({
            "embedding": {
                "provider": "echo",
                "model": "test-embed",
            },
        })
        try:
            assert provider.config.mode == CompletionMode.EMBEDDING
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_completion_mode_is_embedding_legacy(self) -> None:
        """Legacy format also sets CompletionMode.EMBEDDING."""
        provider = await create_embedding_provider({
            "embedding_provider": "echo",
            "embedding_model": "test-embed",
        })
        try:
            assert provider.config.mode == CompletionMode.EMBEDDING
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_mode_cannot_be_overridden_by_caller(self) -> None:
        """mode: embedding is forced even if caller passes a different mode."""
        provider = await create_embedding_provider({
            "embedding": {
                "provider": "echo",
                "model": "test-embed",
                "mode": "chat",
            },
        })
        try:
            assert provider.config.mode == CompletionMode.EMBEDDING
        finally:
            await provider.close()

    # --- typed LLMConfig affordance ---

    @pytest.mark.asyncio
    async def test_accepts_typed_llmconfig(self) -> None:
        """A typed LLMConfig is accepted directly (not just a dict)."""
        provider = await create_embedding_provider(
            LLMConfig(provider="echo", model="typed-embed")
        )
        try:
            assert provider.config.provider == "echo"
            assert provider.config.model == "typed-embed"
            assert provider.is_initialized is True
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_typed_llmconfig_forces_embedding_mode(self) -> None:
        """A typed LLMConfig with a non-embedding mode is forced to EMBEDDING."""
        provider = await create_embedding_provider(
            LLMConfig(provider="echo", model="typed-embed", mode=CompletionMode.CHAT)
        )
        try:
            assert provider.config.mode == CompletionMode.EMBEDDING
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_typed_llmconfig_already_embedding_passthrough(self) -> None:
        """A typed LLMConfig already in EMBEDDING mode is used as-is."""
        config = LLMConfig(
            provider="echo", model="typed-embed", mode=CompletionMode.EMBEDDING
        )
        provider = await create_embedding_provider(config)
        try:
            # Already in EMBEDDING mode → no clone: the caller's exact object
            # is used as-is (the optimization that skips a needless copy).
            assert provider.config is config
            assert provider.config.mode == CompletionMode.EMBEDDING
            assert provider.config.model == "typed-embed"
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_typed_llmconfig_does_not_mutate_caller_config(self) -> None:
        """Forcing embedding mode clones — the caller's frozen config is intact."""
        config = LLMConfig(
            provider="echo", model="typed-embed", mode=CompletionMode.CHAT
        )
        provider = await create_embedding_provider(config)
        try:
            # The original is frozen and untouched; only the provider's copy
            # carries the forced embedding mode.
            assert config.mode == CompletionMode.CHAT
            assert provider.config.mode == CompletionMode.EMBEDDING
        finally:
            await provider.close()
