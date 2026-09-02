"""The ``llm`` half of the embedder seam, and the cross-package claims.

``dataknobs-data`` declares ``TextEmbedder`` and ``dataknobs-llm`` implements
it, so the assertions that the two halves actually meet can only live here ---
``data`` cannot import ``llm``, and a test in ``packages/data/tests`` asserting
against a provider would invert the dependency the seam exists to respect.

Two claims are load-bearing beyond this file:

* ``LLMProviderEmbedder`` satisfies ``TextEmbedder`` structurally, with no
  conversion in the adapter. That absence is the seam's whole justification.
* ``dataknobs-llm``'s shipped embedding caches satisfy ``data``'s narrow
  ``VectorCache`` port without moving. That is what let the proposed hoist of
  ``EmbeddingCache`` into ``data`` be withdrawn rather than executed.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector import CachedEmbedder, TextEmbedder, VectorCache
from dataknobs_llm.llm.base import LLMConfig
from dataknobs_llm.llm.embedding import LLMProviderEmbedder
from dataknobs_llm.llm.providers.caching import MemoryEmbeddingCache
from dataknobs_llm.llm.providers.echo import EchoProvider


def echo(**config: object) -> EchoProvider:
    """An ``EchoProvider`` whose ``embed`` is real, deterministic and offline."""
    return EchoProvider(LLMConfig(provider="echo", model="embed-test", **config))


# --------------------------------------------------------------------------
# The adapter satisfies the protocol declared one package down
# --------------------------------------------------------------------------


def test_adapter_satisfies_the_data_side_protocol() -> None:
    assert isinstance(LLMProviderEmbedder(echo()), TextEmbedder)


async def test_embed_returns_one_vector_per_text_with_no_conversion() -> None:
    """``AsyncLLMProvider.embed`` already returns the protocol's shape.

    If this ever needs a conversion, the seam has stopped doing its job and
    the return type of one side or the other has drifted.
    """
    embedder = LLMProviderEmbedder(echo())

    vectors = await embedder.embed(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    assert all(isinstance(v, list) for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])
    assert len({len(v) for v in vectors}) == 1


async def test_empty_batch_never_reaches_the_provider() -> None:
    """Providers disagree about an empty embed request; the protocol does not."""
    provider = echo()
    embedder = LLMProviderEmbedder(provider)

    assert await embedder.embed([]) == []
    assert provider.embed_call_count == 0


async def test_model_id_names_provider_and_model() -> None:
    embedder = LLMProviderEmbedder(echo())

    assert embedder.model_id == "echo:embed-test"


async def test_model_override_renames_without_redirecting() -> None:
    """The override changes the reported identity only --- the provider is unchanged."""
    embedder = LLMProviderEmbedder(echo(), model="actually-nomic")

    assert embedder.model_id == "echo:actually-nomic"


# --------------------------------------------------------------------------
# dimensions — declared, learned, or refused, but never guessed
# --------------------------------------------------------------------------


async def test_dimensions_are_learned_when_nothing_declares_them() -> None:
    embedder = LLMProviderEmbedder(echo())

    vectors = await embedder.embed(["something"])

    assert embedder.dimensions == len(vectors[0])


def test_dimensions_refuse_to_guess_before_anything_is_known() -> None:
    """Better a raise here than a wrong width enforced later by a vector store."""
    with pytest.raises(ValueError, match="dimensions are unknown"):
        _ = LLMProviderEmbedder(echo()).dimensions


def test_explicit_dimensions_are_answered_without_a_call() -> None:
    assert LLMProviderEmbedder(echo(), dimensions=32).dimensions == 32


async def test_a_provider_whose_width_contradicts_this_embedder_raises() -> None:
    """The check is not a formality, and it is the last line rather than the first.

    A width declared *here* is one the provider never sees:
    ``LLMProviderEmbedder(..., dimensions=N)`` names the length this embedder
    promises its callers, and nothing downstream of the constructor can
    reconcile it with what the provider actually returns. So the seam checks,
    and raising beats writing vectors of one width under a key promising
    another.

    This cell used to reach the same check through ``config.dimensions`` and
    ``EchoProvider``, which sized its vectors from
    ``options["embedding_dim"]`` and ignored the documented field --- a config
    asking for 16 got 768 with nothing raised. That is fixed: every provider
    now honours a stated width or refuses it, so a *config* contradiction is
    caught one layer down, by the provider that knows whether its model can
    deliver the width. What is left here is the contradiction no provider can
    see, which is why the guard stays.
    """
    embedder = LLMProviderEmbedder(echo(), dimensions=16)

    with pytest.raises(ValueError, match="declares 16"):
        await embedder.embed(["anything"])


async def test_a_config_contradiction_is_caught_by_the_provider_first() -> None:
    """Where the check moved to, and why that is the better place.

    ``EchoProvider`` can produce any width, so a config asking for 16 now
    gets 16 and the seam has nothing to object to. A provider that *cannot*
    deliver a stated width raises from ``embed`` itself, naming the model ---
    a message that points at the misconfiguration rather than at whatever
    downstream component noticed the wrong-sized vectors.
    """
    embedder = LLMProviderEmbedder(echo(dimensions=16))

    vectors = await embedder.embed(["anything"])

    assert len(vectors[0]) == 16
    assert embedder.dimensions == 16


async def test_a_matching_declaration_passes_through() -> None:
    observed = len((await LLMProviderEmbedder(echo()).embed(["probe"]))[0])

    embedder = LLMProviderEmbedder(echo(dimensions=observed))

    assert len((await embedder.embed(["probe"]))[0]) == observed
    assert embedder.dimensions == observed


# --------------------------------------------------------------------------
# The withdrawn hoist — llm's caches satisfy data's port where they already are
# --------------------------------------------------------------------------


def test_shipped_llm_cache_satisfies_the_data_side_port() -> None:
    """Nothing moved, and this is why it did not have to.

    An earlier design proposed hoisting ``EmbeddingCache`` and its backends
    into ``data`` on the grounds that they sat on the wrong side of the
    dependency edge. They do not: register item-14 deliberately declined
    ``AsyncDatabase`` for the backend, so these classes carry no
    ``dataknobs-data`` coupling. A narrow structural port in ``data`` reaches
    them where they are.
    """
    assert isinstance(MemoryEmbeddingCache(), VectorCache)


async def test_cached_embedder_works_against_the_real_llm_cache() -> None:
    """The port is not merely satisfied on paper --- the two run together."""
    cache = MemoryEmbeddingCache()
    embedder = CachedEmbedder(LLMProviderEmbedder(echo()), cache)

    first = await embedder.embed(["one", "two"])
    second = await embedder.embed(["one", "two"])

    assert first == second
    assert await cache.count() == 2


async def test_a_model_swap_misses_against_the_real_llm_cache() -> None:
    """Acceptance criterion 4, against the shipped cache rather than a stand-in."""
    cache = MemoryEmbeddingCache()

    await CachedEmbedder(DeterministicEmbedder(dimensions=8, model_id="v1"), cache).embed(
        ["shared"]
    )
    assert await cache.count() == 1

    await CachedEmbedder(DeterministicEmbedder(dimensions=8, model_id="v2"), cache).embed(
        ["shared"]
    )

    assert await cache.count() == 2, (
        "the second model reused the first model's cache entry — a swap must "
        "miss, not serve vectors from a different vector space"
    )


# --------------------------------------------------------------------------
# What counts as "the provider answered a batch with one flat vector"
# --------------------------------------------------------------------------


class _NdarrayEmbedProvider(EchoProvider):
    """A real provider whose ``embed`` answers with a 2-D ``np.ndarray``.

    Not a hypothetical shape. ``AsyncLLMProvider.embed`` is documented to
    return ``list[list[float]]``, and every provider shipped here does --- but
    the *type* is unenforced, an ndarray is the natural output of a locally
    hosted model, and the union this repo already names ``BatchVectors``
    admits it one package over.
    """

    async def embed(self, texts: str | list[str], **kwargs: object) -> object:
        vectors = await super().embed(texts, **kwargs)
        return np.asarray(vectors, dtype=np.float32)


class _FlatEmbedProvider(EchoProvider):
    """A provider that answers a *list* input with a single flat vector.

    The real error the guard exists for: arity-polymorphic ``embed`` taking
    the string branch for a list input. Left undetected, a batch of one
    N-dimensional vector is mistaken for N vectors of one dimension.
    """

    async def embed(self, texts: str | list[str], **kwargs: object) -> object:
        vectors = await super().embed(["one"], **kwargs)
        return list(vectors[0])


async def test_a_batch_of_ndarray_rows_is_not_mistaken_for_a_flat_vector() -> None:
    """The guard must classify by *shape*, not by ``list``-ness.

    ``isinstance(raw[0], (list, tuple))`` is ``False`` for a row of a 2-D
    ndarray, so a perfectly valid batch raised "returned a flat vector for a
    list of N texts" --- a message that is not merely wrong but actively
    misdirecting, since it accuses the provider of the one thing it did not
    do.
    """
    embedder = LLMProviderEmbedder(_NdarrayEmbedProvider(LLMConfig(provider="echo", model="m")))

    vectors = await embedder.embed(["alpha", "beta"])

    assert len(vectors) == 2
    assert all(isinstance(v, list) for v in vectors), "rows must be converted to lists"
    assert all(isinstance(x, float) for x in vectors[0]), "and their elements to floats"


async def test_a_genuinely_flat_answer_still_raises() -> None:
    """The other direction, which is why the guard is there at all."""
    embedder = LLMProviderEmbedder(_FlatEmbedProvider(LLMConfig(provider="echo", model="m")))

    with pytest.raises(TypeError, match="flat vector"):
        await embedder.embed(["alpha", "beta"])
