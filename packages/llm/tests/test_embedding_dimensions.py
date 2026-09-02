"""A stated embedding width is honoured or refused --- never ignored.

``LLMConfig.dimensions`` is documented as the embedding dimensionality and
``AsyncLLMProvider.embed`` documents a per-call ``dimensions`` keyword as
"target dimensions (if supported)". Between them they promise a consumer can
say how wide the vectors should be. One provider read the config field and
none read the keyword, so five of six answered a width request by returning
whatever the model happened to produce.

That is worse than not offering the knob. The vectors are valid, merely not
the width that was asked for, and nothing raises at any layer --- the first
thing that notices is a vector store rejecting the write, which names the
store rather than the misconfiguration. For the OpenAI 3-series it is a
priced mistake as well: asking ``text-embedding-3-large`` for 512 and
receiving 3072 means paying for the larger vector.

**One rule, whatever the provider.** A width stated in config or passed to
the call is resolved once, in the base. A provider whose model can select a
width forwards it; a provider whose model cannot *checks* the answer and
raises when it contradicts what was asked. Neither ignores it. The
difference is visible before any call is made, through
``ModelCapability.EMBEDDING_DIMENSIONS``, which is what lets a consumer
create a fixed-width vector column before it has embedded anything.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from dataknobs_llm.llm.base import LLMConfig, ModelCapability
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.llm.providers.huggingface import HuggingFaceProvider
from dataknobs_llm.llm.providers.ollama import OllamaProvider
from dataknobs_llm.llm.providers.openai import OpenAIProvider

from _aiohttp_error_stub import FakeResponse, FakeSession


# ---------------------------------------------------------------------------
# Boundary stubs. Each sits at the vendor edge --- the OpenAI SDK's
# ``embeddings.create``, aiohttp's ``session.post`` --- so the provider's own
# request shaping runs for real and only the network is absent.
# ---------------------------------------------------------------------------


class _RecordingEmbeddings:
    """``client.embeddings`` that records its kwargs and answers a fixed width."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        n = len(kwargs.get("input") or [])
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=[0.0] * self.width) for _ in range(n)]
        )


def _openai(
    width: int, *, model: str = "text-embedding-3-large", **config: Any
) -> tuple[OpenAIProvider, _RecordingEmbeddings]:
    provider = OpenAIProvider(LLMConfig(provider="openai", model=model, **config))
    embeddings = _RecordingEmbeddings(width)
    provider._client = types.SimpleNamespace(embeddings=embeddings)
    provider._is_initialized = True
    return provider, embeddings


def _ollama(width: int, **config: Any) -> OllamaProvider:
    provider = OllamaProvider(LLMConfig(provider="ollama", model="nomic-embed-text", **config))
    provider._session = FakeSession(
        [
            FakeSession.responding(FakeResponse(200, json_data={"embedding": [0.0] * width}))
            for _ in range(8)
        ]
    )
    provider._is_initialized = True
    return provider


def _huggingface(width: int, **config: Any) -> HuggingFaceProvider:
    provider = HuggingFaceProvider(
        LLMConfig(provider="huggingface", model="sentence-transformers/all-MiniLM-L6-v2", **config)
    )
    provider._session = FakeSession(
        [FakeSession.responding(FakeResponse(200, json_data=[[0.0] * width]))]
    )
    provider._is_initialized = True
    return provider


# ---------------------------------------------------------------------------
# A model that CAN select a width is asked for one
# ---------------------------------------------------------------------------


async def test_openai_forwards_a_configured_width() -> None:
    """``config.dimensions`` reaches the embeddings API.

    The field two docstrings describe, on the one in-tree provider whose
    vendor API accepts it. Before this it was read by nobody on this path:
    ``embeddings.create`` was called with ``input`` and ``model`` alone.
    """
    provider, api = _openai(512, dimensions=512)

    await provider.embed(["anything"])

    assert api.calls[0].get("dimensions") == 512


async def test_openai_forwards_a_per_call_width() -> None:
    """The keyword ``AsyncLLMProvider.embed`` documents, honoured for the first time.

    ``**kwargs`` was accepted and discarded, so a caller reading the base's
    own docstring got no error and no effect.
    """
    provider, api = _openai(256)

    await provider.embed(["anything"], dimensions=256)

    assert api.calls[0].get("dimensions") == 256


async def test_the_call_beats_the_config() -> None:
    """One precedence rule, resolved in the base so no provider invents its own."""
    provider, api = _openai(256, dimensions=1024)

    await provider.embed(["anything"], dimensions=256)

    assert api.calls[0].get("dimensions") == 256


async def test_a_width_nobody_stated_is_not_invented() -> None:
    """No ``dimensions`` in config or call means none on the wire.

    ``text-embedding-ada-002`` rejects the parameter outright, so sending a
    default would break the model this provider falls back to.
    """
    provider, api = _openai(1536, model="text-embedding-ada-002")

    await provider.embed(["anything"])

    assert "dimensions" not in api.calls[0]


# ---------------------------------------------------------------------------
# A model that CANNOT select a width checks the answer instead
# ---------------------------------------------------------------------------


async def test_ollama_refuses_a_width_it_cannot_deliver() -> None:
    """The silent case, made loud.

    Ollama's ``/api/embeddings`` takes a model and a prompt; the width is the
    model's. A config asking for 512 from a 768-wide model used to return 768
    vectors and say nothing, which is how a width promised by config and a
    width written to a store come apart.
    """
    provider = _ollama(768, dimensions=512)

    with pytest.raises(ValueError, match="512"):
        await provider.embed(["anything"])


async def test_ollama_accepts_a_width_that_matches() -> None:
    """No false refusal.

    Declaring the width a model actually produces is the shape the embedder
    seam's own error message recommends, and it must keep working. The rule
    is 'never ignored', not 'never stated'.
    """
    provider = _ollama(768, dimensions=768)

    vectors = await provider.embed(["anything"])

    assert len(vectors[0]) == 768


async def test_ollama_refuses_a_per_call_width_it_cannot_deliver() -> None:
    """The keyword is checked on the providers that cannot forward it, too."""
    provider = _ollama(768)

    with pytest.raises(ValueError, match="512"):
        await provider.embed(["anything"], dimensions=512)


async def test_huggingface_refuses_a_width_it_cannot_deliver() -> None:
    """The same rule, on the other provider that posts a fixed-width request."""
    provider = _huggingface(384, dimensions=512)

    with pytest.raises(ValueError, match="512"):
        await provider.embed(["anything"])


async def test_the_refusal_names_what_was_asked_and_what_arrived() -> None:
    """A message that identifies the misconfiguration rather than the symptom.

    The failure this replaces surfaced as a vector store rejecting a write.
    That message names the store, so it sends a reader to the wrong file.
    """
    provider = _ollama(768, dimensions=512)

    with pytest.raises(ValueError) as excinfo:
        await provider.embed(["anything"])

    message = str(excinfo.value)
    assert "512" in message and "768" in message
    assert "nomic-embed-text" in message


# ---------------------------------------------------------------------------
# Echo: a testing construct whose width is now the one its config states
# ---------------------------------------------------------------------------


async def test_echo_honours_the_documented_field() -> None:
    """The behaviour change, and the reason it is worth making.

    ``EchoProvider`` sized its vectors from ``config.options["embedding_dim"]``
    --- a key predating ``LLMConfig.dimensions`` --- and ignored the field
    every other surface documents. A test asking for 16-wide vectors from the
    project's own testing provider got 768, which made the seam's width check
    a demonstration of the defect rather than a guard against it.
    """
    provider = EchoProvider(LLMConfig(provider="echo", model="embed-test", dimensions=16))

    vectors = await provider.embed(["anything"])

    assert len(vectors[0]) == 16


async def test_echo_still_answers_to_the_legacy_option() -> None:
    """The old key keeps working for configs that use it."""
    provider = EchoProvider(
        LLMConfig(
            provider="echo",
            model="embed-test",
            options={"embedding_dim": 32},
        )
    )

    vectors = await provider.embed(["anything"])

    assert len(vectors[0]) == 32


async def test_echo_prefers_the_documented_field_to_the_legacy_option() -> None:
    """A config stating both gets the one the rest of the stack reads."""
    provider = EchoProvider(
        LLMConfig(
            provider="echo",
            model="embed-test",
            dimensions=16,
            options={"embedding_dim": 32},
        )
    )

    vectors = await provider.embed(["anything"])

    assert len(vectors[0]) == 16


async def test_echo_honours_a_per_call_width() -> None:
    provider = EchoProvider(LLMConfig(provider="echo", model="embed-test", dimensions=16))

    vectors = await provider.embed(["anything"], dimensions=8)

    assert len(vectors[0]) == 8


# ---------------------------------------------------------------------------
# The question is answerable before anything is embedded
# ---------------------------------------------------------------------------


def test_a_selectable_model_advertises_it() -> None:
    """What makes a fixed-width vector column declarable.

    A vector column is created at a width before the first embedding exists,
    so the consumer that has to choose that width needs the answer without
    making a call.
    """
    provider, _ = _openai(512, model="text-embedding-3-large")

    assert ModelCapability.EMBEDDING_DIMENSIONS in provider.get_capabilities()


def test_a_fixed_width_model_does_not_advertise_it() -> None:
    """``text-embedding-ada-002`` produces 1536 and takes no parameter for it."""
    provider, _ = _openai(1536, model="text-embedding-ada-002")

    assert ModelCapability.EMBEDDING_DIMENSIONS not in provider.get_capabilities()


def test_a_provider_whose_api_has_no_width_parameter_does_not_advertise_it() -> None:
    provider = _ollama(768)

    assert ModelCapability.EMBEDDING_DIMENSIONS not in provider.get_capabilities()


def test_the_advertisement_is_config_overridable() -> None:
    """A model the bundled table does not know about is the consumer's to declare.

    The table goes stale between releases; the override path is why that is
    not a release-blocking problem.
    """
    provider, _ = _openai(
        512,
        model="some-new-embedding-model",
        model_profile_overrides={
            "some-new-embedding-model": {"capabilities": ["embeddings", "embedding_dimensions"]}
        },
    )

    assert ModelCapability.EMBEDDING_DIMENSIONS in provider.get_capabilities()
