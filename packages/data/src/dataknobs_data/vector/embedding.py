"""The shape of a thing that turns text into vectors.

Eight mutually incompatible spellings of "embed this text" were in use across
this package --- varying on arity (one text or many), on synchrony (``def``,
``async def``, or *either* with a runtime branch to find out), on return type
(``list[float]``, ``np.ndarray``), and on whether they were typed at all. None
of them matched :meth:`AsyncLLMProvider.embed`, which is where the vectors
actually come from, so every consumer wrote its own adapter between the two.

:class:`TextEmbedder` is the one shape. It is deliberately narrower than any of
the eight:

**Batch-only.** ``AsyncLLMProvider.embed`` is arity-polymorphic --- ``str`` in,
``list[float]`` out; ``list[str]`` in, ``list[list[float]]`` out. Convenient at
a call site and awful as a contract, because every consumer has to narrow the
union, and much of the fragmentation above *is* that narrowing done six
different ways. One text is ``(await embedder.embed([t]))[0]``, which costs
nothing and says what it means.

**Async-only.** Every real embedding source is network or GPU I/O, and
``.claude/rules/async-transport.md`` forbids running that on the event loop
regardless. A synchronous caller reaches an embedder through
``dataknobs_common.sync_bridge``, not through a second protocol.

**Returns ``list[list[float]]``, not ``list[np.ndarray]``.** It is the shape
``sources/processing.py`` and ``sources/cluster_index.py`` each arrived at
independently --- the only one anyone in this tree converged on twice --- and it
is what ``AsyncLLMProvider.embed`` already returns, so the adapter that spans
the two packages needs no conversion. That is the seam's whole job. ``numpy`` is
also undeclared in ``dataknobs-llm``; putting ``np.ndarray`` in a public ``llm``
signature would force a new dependency declaration to buy a conversion at the
other end. ``np.asarray(...)`` at a ``data``-side consumer is one call, on the
side where numpy is declared.

**Carries its own identity.** ``bulk_embed_and_store`` takes ``embedding_fn``,
``model_name`` and ``model_version`` as three independent parameters and trusts
the caller to keep the name in step with the function. :attr:`TextEmbedder.model_id`
removes that class of error: a stored vector's staleness key comes from the
thing that produced it.

Nothing here deletes the callable path. An untyped ``embedding_fn`` still has to
be classified before it is called, and
:func:`~dataknobs_data.vector.embedding_fn.call_embedding_fn` is the one correct
place that happens. An *adopted* site has nothing left to classify --- a
``TextEmbedder`` is async by declaration --- which is how the branch stops being
needed rather than being removed while callers still need it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = [
    "CachedEmbedder",
    "TextEmbedder",
    "VectorCache",
    "embedding_cache_key",
]


@runtime_checkable
class TextEmbedder(Protocol):
    """Turns text into dense vectors.

    Batch-first and async-only, for the reasons in the module docstring.

    ``isinstance(x, TextEmbedder)`` works and checks that the three members are
    present; it does not check their signatures, and ``issubclass`` is not
    available at all (a protocol carrying non-method members cannot support
    it). Treat the check as a smoke test, not a contract.
    """

    @property
    def dimensions(self) -> int:
        """Length of every vector :meth:`embed` returns."""
        ...

    @property
    def model_id(self) -> str:
        """Stable identity of the model producing these vectors.

        This is the staleness key. When it changes, vectors already stored
        were produced by a different model, sit in a different vector space,
        and every similarity computed against them is meaningless --- so they
        are invalid and must be regenerated rather than compared.

        "Stable" means across processes and across runs, not merely within
        one: it is written into stored metadata and read back later by
        something that never saw the embedder.
        """
        ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text, returning one vector per input, in order.

        Args:
            texts: The texts to embed. May be empty, in which case the result
                is empty --- an empty batch is not an error.

        Returns:
            One ``dimensions``-length vector per input text, positionally
            aligned with *texts*.
        """
        ...


@runtime_checkable
class VectorCache(Protocol):
    """The part of an embedding cache that :class:`CachedEmbedder` needs.

    Deliberately two methods rather than eight. ``dataknobs-llm`` already ships
    an ``EmbeddingCache`` ABC with a memory and a SQLite backend, and register
    item-14 put them there *on purpose* --- it declined ``AsyncDatabase`` for
    the backend, so those classes carry no ``dataknobs-data`` coupling and
    there is no wrong side of a dependency edge for them to be on. An earlier
    draft of this seam proposed hoisting them here; that was withdrawn.

    So this is a port, not a home. Both shipped ``llm`` caches satisfy it
    structurally, as does anything else with the same two methods, and nothing
    had to move for that to be true. The batch pair is the whole surface
    because :meth:`TextEmbedder.embed` is batch-only, so the single-text
    accessors would never be called.
    """

    async def get_batch(self, model: str, texts: list[str]) -> list[list[float] | None]:
        """Look up each text, returning a list parallel to *texts*.

        Each element is the cached vector, or ``None`` on a miss.
        """
        ...

    async def put_batch(self, model: str, texts: list[str], vectors: list[list[float]]) -> None:
        """Store *vectors* against *texts*, under *model*."""
        ...


def embedding_cache_key(model: str, text: str) -> str:
    """A collision-resistant key for one ``(model, text)`` pair.

    The null-byte separator is what stops ``("model", "text")`` and
    ``("mode", "ltext")`` sharing a key. This mirrors the derivation
    ``dataknobs_llm.llm.providers.caching`` uses for the same pair, so a cache
    written through one and read through the other agrees with itself; it is
    restated here rather than imported because ``data`` cannot import ``llm``.

    Only needed by a cache implementing :class:`VectorCache` for itself ---
    :class:`CachedEmbedder` passes ``model`` and ``text`` through and lets the
    cache key them.
    """
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class CachedEmbedder:
    """Wraps a :class:`TextEmbedder`, serving repeat texts from a cache.

    The key is ``(inner.model_id, text)`` and the ``model_id`` half is not
    optional. A cache keyed on text alone silently serves vectors from the
    previous model after a model swap: nothing raises, every lookup succeeds,
    and every similarity is computed across two vector spaces. Including the
    model identity means a swap misses rather than lies.

    Only the texts that miss reach the inner embedder, and they reach it in one
    batch, preserving the batching the protocol exists to keep.

    Example:
        ```python
        embedder = CachedEmbedder(LLMProviderEmbedder(provider), cache)
        vectors = await embedder.embed(["a", "b", "a"])
        ```
    """

    def __init__(self, inner: TextEmbedder, cache: VectorCache) -> None:
        """Args:
        inner: The embedder that produces vectors on a miss.
        cache: Where hits come from. Any :class:`VectorCache`, including
            ``dataknobs-llm``'s shipped ``MemoryEmbeddingCache`` and
            ``SqliteEmbeddingCache``, which satisfy it structurally.
        """
        self._inner = inner
        self._cache = cache

    @property
    def dimensions(self) -> int:
        """The inner embedder's, unchanged --- caching does not reshape."""
        return self._inner.dimensions

    @property
    def model_id(self) -> str:
        """The inner embedder's, unchanged.

        Forwarded rather than decorated so that a vector stored through a
        cached embedder and one stored through the bare embedder carry the
        same staleness key. A caller must not be able to tell from the stored
        metadata whether a cache was in the path.
        """
        return self._inner.model_id

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Serve what the cache holds; embed and store only what it does not."""
        wanted = list(texts)
        if not wanted:
            return []

        model = self._inner.model_id
        cached = await self._cache.get_batch(model, wanted)

        # Positions rather than texts, because `texts` may repeat: two
        # occurrences of one string are two output slots, and a set of misses
        # would embed it twice or --- worse --- fill only the first slot.
        missing = [i for i, hit in enumerate(cached) if hit is None]
        if not missing:
            return [hit for hit in cached if hit is not None]

        # Deduplicated, so a batch of one text repeated N times costs one
        # embedding. `dict.fromkeys` keeps first-seen order, so the request is
        # stable rather than set-ordered.
        to_embed = list(dict.fromkeys(wanted[i] for i in missing))
        fresh = await self._inner.embed(to_embed)
        by_text = dict(zip(to_embed, fresh, strict=True))

        await self._cache.put_batch(model, to_embed, fresh)

        return [by_text[wanted[i]] if hit is None else hit for i, hit in enumerate(cached)]
