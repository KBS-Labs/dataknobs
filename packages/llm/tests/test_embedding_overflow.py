"""An over-long embedding input is refused by default, and truncated only on request.

A model embeds at most its context window. Past that, a provider can refuse
the text or cut it, and the two are not interchangeable: a cut text's vector
describes its opening words only, and a retrieval over it finds nothing the
rest of the text said --- with nothing on the vector to show it.

So refusing is the default, and ``LLMConfig.embedding_overflow`` is where a
consumer who wants the cut says so. Three things follow, and each is pinned
here:

- **Every provider honours the setting or refuses it by name.** A provider
  whose API cannot truncate must say so when asked, rather than refusing the
  over-long text later as though nothing had been asked for.
- **A truncated vector is not the same vector.** It reaches both caches and
  the staleness key written beside a stored vector, so switching the setting
  off does not go on serving the vectors it produced.
- **Each cut is reported**, by position, and never with the text itself.

``EchoProvider`` carries a synthetic window (``options["embedding_window"]``,
counted in words) so all of this runs without a server. The Ollama cases at
the end run against the real one.
"""

from __future__ import annotations

import logging
import math
from collections.abc import AsyncIterator

import pytest

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.testing import requires_ollama, requires_ollama_model
from dataknobs_data.vector import CachedEmbedder
from dataknobs_llm.exceptions import ContextLengthExceededError
from dataknobs_llm.llm.base import LLMConfig, ModelCapability
from dataknobs_llm.llm.embedding import LLMProviderEmbedder
from dataknobs_llm.llm.providers import (
    LLMProviderFactory,
    create_embedding_provider,
    create_llm_provider,
)
from dataknobs_llm.llm.providers.caching import (
    CachingEmbedProvider,
    MemoryEmbeddingCache,
    _cache_identity,
)
from dataknobs_llm.llm.providers.echo import EchoProvider
from dataknobs_llm.llm.providers.ollama import OllamaProvider
from dataknobs_llm.testing import CapturingProvider

from _aiohttp_error_stub import FakeResponse, FakeSession

#: Four words, against Echo's three-word window below. Distinctive, so a
#: report quoting any of it is caught.
_OVER_LONG = "alpha bravo charlie delta"
_WINDOW = 3


def _echo(policy: str = "refuse", **options: object) -> EchoProvider:
    return EchoProvider(
        LLMConfig(
            provider="echo",
            model="e",
            embedding_overflow=policy,  # type: ignore[arg-type]
            options={"embedding_window": _WINDOW, **options},
        )
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------


def test_refusing_is_the_default() -> None:
    assert LLMConfig(provider="echo", model="e").embedding_overflow == "refuse"


@pytest.mark.parametrize("build", ["direct", "from_dict", "clone"])
def test_an_unknown_policy_is_refused_naming_the_ones_there_are(build: str) -> None:
    """Checked when the config is built, on every path that builds one."""
    with pytest.raises(ValidationError) as excinfo:
        if build == "direct":
            LLMConfig(provider="echo", model="e", embedding_overflow="clip")  # type: ignore[arg-type]
        elif build == "from_dict":
            LLMConfig.from_dict({"provider": "echo", "model": "e", "embedding_overflow": "clip"})
        else:
            LLMConfig(provider="echo", model="e").clone(embedding_overflow="clip")
    assert "'refuse'" in str(excinfo.value)
    assert "'truncate'" in str(excinfo.value)
    assert "clip" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Every provider honours it or refuses it by name
# ---------------------------------------------------------------------------

#: The providers that honour ``"truncate"``, each with its own proof in this
#: file. A provider that starts declaring the policy has to join this set,
#: which is the prompt to write that proof.
_HONOURING = {"echo", "ollama"}


@pytest.mark.parametrize("name", LLMProviderFactory.list_providers())
async def test_every_provider_honours_truncation_or_refuses_it_by_name(name: str) -> None:
    """The registry is the population, so a provider added later is covered.

    Ignoring the setting is the outcome this exists to rule out: a config
    asking for truncation from a provider that cannot do it would otherwise
    discover the answer one over-long text at a time, as an overflow error it
    had configured itself out of.
    """
    provider = create_llm_provider(
        LLMConfig(provider=name, model="m", api_key="k", embedding_overflow="truncate")
    )
    if "truncate" in provider._embedding_overflow_policies:
        assert name in _HONOURING, f"{name} declares truncation and nothing here proves it"
        return
    try:
        await provider.embed("text")
    except NotImplementedError:
        return  # a provider with no embedding models has nothing to truncate
    except ValidationError as exc:
        assert type(provider).__name__ in str(exc)
        assert "truncate" in str(exc)
        return
    pytest.fail(f"{name} was asked to truncate and neither did nor refused")


# ---------------------------------------------------------------------------
# The two policies, offline
# ---------------------------------------------------------------------------


async def test_refuse_raises_the_overflow_error() -> None:
    with pytest.raises(ContextLengthExceededError) as excinfo:
        await _echo("refuse").embed(_OVER_LONG)
    assert "delta" not in str(excinfo.value)


async def test_a_text_inside_the_window_is_embedded_under_either_policy() -> None:
    """The positive control: the refusal is about length."""
    for policy in ("refuse", "truncate"):
        assert len(await _echo(policy).embed("alpha bravo")) == 768


async def test_truncate_embeds_the_opening_window(caplog: pytest.LogCaptureFixture) -> None:
    provider = _echo("truncate")
    with caplog.at_level(logging.WARNING):
        vectors = await provider.embed(["short", _OVER_LONG])

    assert vectors[1] == await provider.embed("alpha bravo charlie")
    assert vectors[0] == await provider.embed("short")

    [report] = _warnings(caplog)
    assert "[1]" in report
    assert str(_WINDOW) in report
    for word in _OVER_LONG.split():
        assert word not in report, "the report quoted the caller's text"


async def test_nothing_is_reported_when_nothing_was_cut(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        await _echo("truncate").embed(["short", "alpha bravo charlie"])
    assert _warnings(caplog) == []


# ---------------------------------------------------------------------------
# A truncated vector has its own identity, in every place one is kept
# ---------------------------------------------------------------------------


async def test_the_llm_cache_does_not_serve_a_truncated_vector_to_a_refusing_provider() -> None:
    """The cache would otherwise undo the refusal, silently and for good.

    A vector cached while truncation was on is exactly the vector the
    refusing configuration exists not to produce.
    """
    cache = MemoryEmbeddingCache()
    cutting = CachingEmbedProvider(_echo("truncate"), cache)
    await cutting.initialize()
    await cutting.embed(_OVER_LONG)
    assert await cache.count() == 1

    refusing = CachingEmbedProvider(_echo("refuse"), cache)
    await refusing.initialize()
    with pytest.raises(ContextLengthExceededError):
        await refusing.embed(_OVER_LONG)


async def test_the_data_cache_does_not_either() -> None:
    """``CachedEmbedder`` keys on ``model_id``, so the key has to carry it."""
    cache = MemoryEmbeddingCache()
    await CachedEmbedder(LLMProviderEmbedder(_echo("truncate")), cache).embed([_OVER_LONG])

    with pytest.raises(ContextLengthExceededError):
        await CachedEmbedder(LLMProviderEmbedder(_echo("refuse")), cache).embed([_OVER_LONG])


def test_a_wrapper_reports_the_variant_of_what_it_wraps() -> None:
    """Whether a cache sits in the path must not change the stored key."""
    inner = _echo("truncate")
    bare = LLMProviderEmbedder(inner).model_id

    assert LLMProviderEmbedder(CachingEmbedProvider(inner, MemoryEmbeddingCache())).model_id == bare
    assert LLMProviderEmbedder(CapturingProvider(inner)).model_id == bare


def test_the_key_names_the_variant_and_only_when_there_is_one() -> None:
    assert LLMProviderEmbedder(_echo("refuse")).model_id == "echo:e"
    assert LLMProviderEmbedder(_echo("truncate")).model_id == "echo:e#truncate"

    ollama = OllamaProvider(LLMConfig(provider="ollama", model="nomic-embed-text"))
    assert LLMProviderEmbedder(ollama).model_id == "ollama:nomic-embed-text#api-embed"

    cutting = OllamaProvider(
        LLMConfig(provider="ollama", model="nomic-embed-text", embedding_overflow="truncate")
    )
    assert LLMProviderEmbedder(cutting).model_id == "ollama:nomic-embed-text#api-embed+truncate"


def test_the_cache_identity_carries_each_part_only_when_set() -> None:
    assert _cache_identity("m", None, None) == "m"
    assert _cache_identity("m", 256, None) == "m@256"
    assert _cache_identity("m", None, "api-embed") == "m#api-embed"
    assert _cache_identity("m", 256, "api-embed") == "m#api-embed@256"


# ---------------------------------------------------------------------------
# Every config door carries the setting
# ---------------------------------------------------------------------------


async def test_the_flat_config_form_forwards_the_setting() -> None:
    """The legacy flat form forwards a fixed set of keys, and this is one."""
    provider = await create_embedding_provider(
        {"embedding_provider": "echo", "embedding_model": "e", "embedding_overflow": "truncate"}
    )
    assert provider.config.embedding_overflow == "truncate"


async def test_the_nested_config_form_forwards_the_setting() -> None:
    provider = await create_embedding_provider(
        {"embedding": {"provider": "echo", "model": "e", "embedding_overflow": "truncate"}}
    )
    assert provider.config.embedding_overflow == "truncate"


# ---------------------------------------------------------------------------
# Ollama, at its HTTP boundary
# ---------------------------------------------------------------------------


def _ollama(*responses: FakeResponse, **config: object) -> tuple[OllamaProvider, FakeSession]:
    options = {"model_metadata_live": False, **config.pop("options", {})}  # type: ignore[dict-item]
    provider = OllamaProvider(
        LLMConfig(provider="ollama", model="mxbai-embed-large", options=options, **config)  # type: ignore[arg-type]
    )
    session = FakeSession([FakeSession.responding(r) for r in responses])
    provider._session = session
    provider._is_initialized = True
    return provider, session


def _embedded(prompt_eval_count: int) -> FakeResponse:
    return FakeResponse(
        200, json_data={"embeddings": [[0.0] * 4], "prompt_eval_count": prompt_eval_count}
    )


#: The window reaches the provider the way the server's does, as the
#: profile's context window --- here from a config override, offline.
_WINDOW_512 = {"model_profile_overrides": {"context_window": 512}}


async def test_ollama_always_says_whether_it_may_truncate() -> None:
    """``/api/embed`` truncates unless told not to.

    So a request that omits the flag has turned the refusal into truncation
    without anyone asking. Sent under both policies, not only the one that
    differs from the endpoint's default.
    """
    refusing, session = _ollama(_embedded(7))
    await refusing.embed("text")
    assert session.calls == [f"{refusing.base_url}/api/embed"]
    assert session.payloads == [{"model": "mxbai-embed-large", "input": "text", "truncate": False}]

    cutting, session = _ollama(_embedded(7), embedding_overflow="truncate")
    await cutting.embed("text")
    assert session.payloads[0]["truncate"] is True


async def test_ollama_reports_a_text_that_reached_the_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single text whose token count reaches the window was cut.

    "Reaches", not "exceeds": the server reports the window exactly for a text
    it cut, and a text that fills it precisely looks the same.
    """
    provider, _ = _ollama(
        _embedded(9), _embedded(512), embedding_overflow="truncate", **_WINDOW_512
    )
    with caplog.at_level(logging.WARNING):
        await provider.embed(["short", "long"])

    [report] = _warnings(caplog)
    assert "[1]" in report
    assert "512" in report
    assert "mxbai-embed-large" in report


async def test_ollama_says_once_when_it_cannot_tell(caplog: pytest.LogCaptureFixture) -> None:
    """With no window known, a cut is undetectable; saying so every call is noise."""
    provider, _ = _ollama(_embedded(512), _embedded(512), embedding_overflow="truncate")
    with caplog.at_level(logging.WARNING):
        await provider.embed("one")
        await provider.embed("two")

    [notice] = _warnings(caplog)
    assert "cannot" in notice


async def test_ollama_refreshes_its_model_metadata_before_embedding() -> None:
    """Otherwise an embedding-only provider never learns its model's window.

    ``complete`` refreshes at the request boundary; ``embed`` did not, so a
    provider used only to embed ran on the name heuristic for good.
    """
    provider, _ = _ollama(_embedded(7), options={"model_metadata_live": True})
    assert provider._live_source.is_stale()
    await provider.embed("text")
    assert not provider._live_source.is_stale()


# ---------------------------------------------------------------------------
# Ollama, against the real server
# ---------------------------------------------------------------------------

_MODEL = "mxbai-embed-large"
_MXBAI_WINDOW = 512
# About 3,000 words: several times the model's 512-token window.
_SERVER_OVER_LONG = " ".join(f"word{i} lorem ipsum" for i in range(1000))


@pytest.fixture
async def live() -> AsyncIterator[dict[str, OllamaProvider]]:
    providers = {
        policy: OllamaProvider(
            LLMConfig(provider="ollama", model=_MODEL, embedding_overflow=policy)  # type: ignore[arg-type]
        )
        for policy in ("refuse", "truncate")
    }
    for provider in providers.values():
        await provider.initialize()
    try:
        yield providers
    finally:
        for provider in providers.values():
            await provider.close()


@requires_ollama
@requires_ollama_model(_MODEL)
class TestAgainstTheServer:
    async def test_the_default_still_refuses(self, live: dict[str, OllamaProvider]) -> None:
        with pytest.raises(ContextLengthExceededError):
            await live["refuse"].embed(_SERVER_OVER_LONG)

    async def test_truncate_embeds_and_reports_the_cut(
        self, live: dict[str, OllamaProvider], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            vector = await live["truncate"].embed(_SERVER_OVER_LONG)
        assert len(vector) == 1024

        [report] = _warnings(caplog)
        assert "[0]" in report
        assert str(_MXBAI_WINDOW) in report
        assert "lorem" not in report

    async def test_embedding_learns_the_models_window(
        self, live: dict[str, OllamaProvider]
    ) -> None:
        provider = live["refuse"]
        await provider.embed("a short text")
        assert provider.get_constraints().max_input_tokens == _MXBAI_WINDOW


@requires_ollama
@requires_ollama_model("nomic-embed-text")
async def test_a_nomic_vector_arrives_at_unit_length() -> None:
    """``/api/embeddings`` returned this model's vectors at a norm of about 22.

    ``/api/embed`` normalizes them. A store under a non-cosine metric that
    holds the old ones ranks wrongly against the new until it is rebuilt,
    which is why the identity changed with the endpoint.
    """
    provider = OllamaProvider(LLMConfig(provider="ollama", model="nomic-embed-text"))
    await provider.initialize()
    try:
        vector = await provider.embed("a short text")
    finally:
        await provider.close()
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-3)


@requires_ollama
@requires_ollama_model("gemma3:1b")
async def test_a_completion_model_does_not_claim_embeddings() -> None:
    """``/api/embed`` answers a completion model with a 501 (Ollama 0.33.2)."""
    provider = OllamaProvider(LLMConfig(provider="ollama", model="gemma3:1b"))
    await provider.initialize()
    try:
        await provider.refresh_model_metadata()
        assert ModelCapability.EMBEDDINGS not in provider.get_capabilities()
    finally:
        await provider.close()
