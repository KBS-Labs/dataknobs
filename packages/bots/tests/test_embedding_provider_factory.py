"""Smoke test for create_embedding_provider() re-export from dataknobs-bots.

The comprehensive test suite lives in dataknobs-llm
(packages/llm/tests/test_embedding_provider_factory.py).  This file
validates only that the backward-compatible re-export works.
"""

import pytest

from dataknobs_bots.providers import build_embedding_config, create_embedding_provider
from dataknobs_llm import CompletionMode


class TestCreateEmbeddingProviderReExport:
    """Validates the backward-compatible re-export from dataknobs_bots."""

    @pytest.mark.asyncio
    async def test_reexport_creates_working_provider(self) -> None:
        """The bots re-export creates an initialized embedding provider."""
        provider = await create_embedding_provider({
            "embedding": {
                "provider": "echo",
                "model": "test-embed",
            },
        })
        try:
            assert provider.config.provider == "echo"
            assert provider.config.mode == CompletionMode.EMBEDDING
            assert provider.is_initialized is True
        finally:
            await provider.close()


class TestBuildEmbeddingConfig:
    """build_embedding_config projects typed fields onto the helper dict."""

    def test_only_set_keys_included(self) -> None:
        """None-valued fields are omitted (sparse-dict parity)."""
        assert build_embedding_config(embedding_provider="echo") == {
            "embedding_provider": "echo",
        }

    def test_legacy_flat_passthrough_projected(self) -> None:
        """api_base/api_key/dimensions are forwarded for the legacy flat form.

        These are the legacy-flat passthrough keys create_embedding_provider
        reads from the top level; build_embedding_config must surface them so
        the typed-config path matches the pre-adoption whole-dict behavior.
        """
        result = build_embedding_config(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            api_base="https://proxy.example/v1",
            api_key="sk-test-key",
            dimensions=1536,
        )
        assert result == {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "api_base": "https://proxy.example/v1",
            "api_key": "sk-test-key",
            "dimensions": 1536,
        }
