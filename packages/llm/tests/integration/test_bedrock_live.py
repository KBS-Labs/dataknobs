"""Live Amazon Bedrock behavioural test.

Skipped unless ``DK_TEST_BEDROCK`` is truthy AND AWS credentials resolve
(the :func:`requires_bedrock` marker). This is the only test that makes a
real, paid ``bedrock-runtime`` call, so it never runs in CI by default.

Configure the model/region via env when running:

    DK_TEST_BEDROCK=true \
    AWS_DEFAULT_REGION=us-west-2 \
    DK_BEDROCK_CHAT_MODEL=anthropic.claude-3-haiku-20240307-v1:0 \
    DK_BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0 \
    uv run pytest packages/llm/tests/integration/test_bedrock_live.py
"""

from __future__ import annotations

import os

import pytest

from dataknobs_common.testing import requires_bedrock
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers.bedrock import BedrockProvider

pytestmark = [pytest.mark.integration, requires_bedrock, pytest.mark.bedrock]


def _region() -> str:
    return os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


@pytest.mark.asyncio
async def test_bedrock_converse_live() -> None:
    """A real Converse call returns content and usage."""
    model = os.environ.get(
        "DK_BEDROCK_CHAT_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"
    )
    config = LLMConfig(
        provider="bedrock",
        model=model,
        max_tokens=64,
        temperature=0.0,
        options={"region_name": _region()},
    )
    async with BedrockProvider(config) as provider:
        response = await provider.complete(
            "Reply with exactly the word: pong"
        )

    assert response.content.strip()
    assert response.usage is not None
    assert response.usage["total_tokens"] > 0


@pytest.mark.asyncio
async def test_bedrock_embed_live() -> None:
    """A real embedding call returns a vector of the expected dimension."""
    model = os.environ.get(
        "DK_BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0"
    )
    config = LLMConfig(
        provider="bedrock",
        model=model,
        dimensions=256,
        options={"region_name": _region()},
    )
    async with BedrockProvider(config) as provider:
        vector = await provider.embed("hello world")

    assert isinstance(vector, list)
    assert len(vector) == 256
    assert all(isinstance(x, float) for x in vector)
