"""An over-long text sent to a real Ollama embedding model is refused by name.

Ollama's ``/api/embeddings`` endpoint answers an input longer than the model's
context window with **HTTP 500** and the body
``{"error":"the input length exceeds the context length"}`` (measured on Ollama
0.33.2 against ``mxbai-embed-large``, ``nomic-embed-text`` and
``nomic-embed-text-v2-moe``). Before the fix that surfaced as a generic
``OperationError (HTTP 500)``: the status gate admitted only a 400, and the
wording matched none of the shared overflow markers.

These run against the real server, because the earlier unit test for this case
pinned a status and a wording Ollama never sends, and passed while the real
path was broken. ``mxbai-embed-large`` has the smallest window of the three
(512 tokens), so a modest text overflows it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiohttp
import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.testing import requires_ollama, requires_ollama_model
from dataknobs_llm.exceptions import ContextLengthExceededError
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.providers.ollama import OllamaProvider

_MODEL = "mxbai-embed-large"

pytestmark = [requires_ollama, requires_ollama_model(_MODEL)]

# About 3,000 words: several times the model's 512-token window.
_OVER_LONG = " ".join(f"word{i} lorem ipsum" for i in range(1000))


@pytest.fixture
async def provider() -> AsyncIterator[OllamaProvider]:
    llm = OllamaProvider(LLMConfig(provider="ollama", model=_MODEL))
    await llm.initialize()
    try:
        yield llm
    finally:
        await llm.close()


async def test_an_over_long_text_raises_context_length_exceeded(
    provider: OllamaProvider,
) -> None:
    with pytest.raises(ContextLengthExceededError) as excinfo:
        await provider.embed(_OVER_LONG)
    # Still the caller's-input family, so an existing ``except ValidationError``
    # keeps matching.
    assert isinstance(excinfo.value, ValidationError)


async def test_the_message_is_ours_and_the_vendor_text_stays_on_the_cause(
    provider: OllamaProvider,
) -> None:
    with pytest.raises(ContextLengthExceededError) as excinfo:
        await provider.embed(_OVER_LONG)
    assert "input length" not in str(excinfo.value)
    cause = excinfo.value.__cause__
    assert isinstance(cause, aiohttp.ClientResponseError)
    assert "exceeds the context length" in str(cause)


async def test_an_over_long_member_of_a_batch_raises_the_same_type(
    provider: OllamaProvider,
) -> None:
    with pytest.raises(ContextLengthExceededError):
        await provider.embed(["a short text", _OVER_LONG])


async def test_a_text_that_fits_is_embedded(provider: OllamaProvider) -> None:
    """The positive control: the refusal is about length, not the model."""
    vector = await provider.embed("a short text")
    assert len(vector) == 1024
