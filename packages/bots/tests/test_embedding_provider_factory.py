"""Smoke test for create_embedding_provider() re-export from dataknobs-bots.

The comprehensive test suite lives in dataknobs-llm
(packages/llm/tests/test_embedding_provider_factory.py).  This file
validates only that the backward-compatible re-export works.
"""

import pytest

from dataknobs_bots.providers import create_embedding_provider
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
