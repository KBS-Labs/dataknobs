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

The key is ``model_name``, and it is the *reader* that makes that true.
``VectorTextSynchronizer`` compares it under ``SyncConfig.track_model_name``,
alongside the older ``model_version`` comparison --- because defaulting a key
nothing consults writes a value that describes the vector and protects nobody.
A stored ``None`` is not a mismatch: a vector predating the seam records no
name, and re-embedding every such corpus on upgrade answers a question nothing
asked.

Nothing here deletes the callable path. An untyped ``embedding_fn`` still has to
be classified before it is called, and ``vector/embedding_fn.py`` is the one
correct place that happens --- :func:`~dataknobs_data.vector.embedding_fn.call_embedding_fn`
for one text and :func:`~dataknobs_data.vector.embedding_fn.call_embedding_fn_batch`
for a corpus, over a single shared resolver, because a rule written twice is a
rule that will hold in one place and not the other. An *adopted* site has
nothing left to classify --- a ``TextEmbedder`` is async by declaration --- which
is how the branch stops being needed rather than being removed while callers
still need it.

:func:`embed_texts` and :func:`embed_text` are where a site chooses between the
two. They exist so that the choice is written once: twenty-five sites each
writing their own ``if embedder is not None`` is how the eight shapes above
arose, and a twenty-sixth site is the failure mode this module is against.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, Protocol, Self, cast, runtime_checkable

from .embedding_fn import call_embedding_fn, call_embedding_fn_batch

__all__ = [
    "CachedEmbedder",
    "SyncTextEmbedder",
    "TextEmbedder",
    "VectorCache",
    "default_model_name",
    "embed_text",
    "embed_texts",
    "embedding_cache_key",
    "require_embedding_source",
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
        one: it is written into stored metadata --- as the ``VectorField``'s
        ``model_name`` --- and read back later by something that never saw the
        embedder. That reader is
        :meth:`~dataknobs_data.vector.sync.VectorTextSynchronizer._has_current_vector`,
        which compares it under ``SyncConfig.track_model_name``. Both halves
        are required: a key written and never compared is a description, not a
        guard.
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
    an ``EmbeddingCache`` ABC with a memory and a SQLite backend, and those live
    there *on purpose*: the backend was deliberately not built on
    ``AsyncDatabase``, so the classes carry no ``dataknobs-data`` coupling and
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
        """Serve what the cache holds; embed and store only what it does not.

        Raises:
            ValueError: The cache answered with a different number of entries
                than it was asked about. See below on why that is checked
                here rather than left to fail downstream.
        """
        wanted = list(texts)
        if not wanted:
            return []

        model = self._inner.model_id
        cached = await self._cache.get_batch(model, wanted)
        # `VectorCache.get_batch` promises a list *parallel to texts*, and
        # nothing below re-establishes that. A short answer produced a short
        # result --- misaligned from the dropped position onward, so every
        # caller pairing the return against its own input list stored the
        # wrong vector for the wrong text, silently. A long one indexed past
        # the end of `wanted` and raised `IndexError` from a comprehension
        # naming neither the cache nor the mismatch.
        #
        # `zip(..., strict=True)` already holds the inner embedder to this;
        # the cache is the other supplier of the same guarantee and had no
        # equivalent.
        if len(cached) != len(wanted):
            raise ValueError(
                f"{type(self._cache).__name__}.get_batch returned {len(cached)} "
                f"entries for {len(wanted)} texts; a cache answer must be "
                f"parallel to the texts it was asked about"
            )

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


def require_embedding_source(
    embedder: TextEmbedder | None,
    embedding_fn: Callable[..., Any] | None,
    *,
    allow_neither: bool = False,
) -> None:
    """Check that exactly one embedding source was supplied.

    :func:`embed_texts` and :func:`embed_text` apply this themselves, so a
    site that always reaches one of them does not need it. A site that may
    *not* reach one --- a bulk loop over a possibly-empty batch --- calls it up
    front, so that "you gave me no embedder" is still raised for an empty
    input rather than turning into a silent empty result.

    Args:
        embedder: The typed source, if the caller gave one.
        embedding_fn: The callable source, if the caller gave one.
        allow_neither: Permit *no* source. For a constructor whose object is
            useful without embedding at all --- a migration that only adds a
            schema field, a dedup checker with semantic matching switched off
            --- where the demand for a source belongs at the embedding call
            rather than at construction. The *conflict* is still refused,
            because two sources are a mistake wherever they are supplied.

    Raises:
        ValueError: Both were supplied; or neither was and *allow_neither* is
            false. Both is refused rather than resolved by precedence:
            silently preferring one leaves a caller believing vectors came
            from a source that never ran, which is the class of error
            :attr:`TextEmbedder.model_id` exists to close.
    """
    if embedder is not None and embedding_fn is not None:
        raise ValueError(
            "pass either `embedder` or `embedding_fn`, not both — with both "
            "supplied, one of them silently does not run"
        )
    if embedder is None and embedding_fn is None and not allow_neither:
        raise ValueError("an embedder is required: pass `embedder=` or `embedding_fn=`")


def default_model_name(model_name: str | None, embedder: TextEmbedder | None) -> str | None:
    """The staleness key to store: what the caller said, else what the embedder is.

    Every site taking both an ``embedder`` and a ``model_name`` owes the same
    two lines, and they are here rather than at each site for the reason the
    parameter exists at all. ``model_name`` and the embedding source were
    independent, so nothing stopped a caller naming one model while embedding
    with another --- and the name is what a later reader compares. Defaulting
    it from the thing that produced the vector is what closes that, and a rule
    written once cannot be applied at four sites and forgotten at a fifth.

    An explicit *model_name* wins. A caller who said what they meant is not
    overridden, including one deliberately recording a name that differs from
    the embedder's own.

    Args:
        model_name: What the caller passed, if anything.
        embedder: The embedder in use, if the caller passed one.

    Returns:
        *model_name* when it is not ``None``; otherwise ``embedder.model_id``,
        or ``None`` when there is no embedder to ask.
    """
    if model_name is not None:
        return model_name
    return embedder.model_id if embedder is not None else None


async def embed_texts(
    texts: Sequence[str],
    *,
    embedder: TextEmbedder | None = None,
    embedding_fn: Callable[..., Any] | None = None,
) -> Any:
    """One batch of vectors, from whichever source the caller supplied.

    Every batch site in this package takes both an ``embedder`` and a legacy
    ``embedding_fn``, and the choice between them is the same three lines
    everywhere. Written at each site it would be the duplication that produced
    the eight shapes in the first place, so it is written here.

    Args:
        texts: The batch to embed. Empty is not an error.
        embedder: The typed path. Awaited directly --- a
            :class:`TextEmbedder` is async by declaration, so there is nothing
            to classify.
        embedding_fn: The callable path, kept for callers that predate the
            protocol. Routed through
            :func:`~dataknobs_data.vector.embedding_fn.call_embedding_fn_batch`,
            so a synchronous one is offloaded rather than run on the loop and
            an untyped callable's synchrony --- which is not knowable from its
            annotation --- is classified in the one place that does that.

    Returns:
        ``list[list[float]]`` on the ``embedder`` path. On the ``embedding_fn``
        path, **whatever that callable returned, unconverted** --- usually an
        ``np.ndarray``. The shapes are deliberately not reconciled: normalizing
        the callable's answer would change what existing callers hand
        downstream, and this seam is additive. A site wanting one shape from
        both converts what it gets.

    Raises:
        ValueError: Neither was supplied, or both were. Both is refused rather
            than resolved by precedence: silently preferring one leaves a
            caller believing vectors came from a source that never ran, which
            is the class of error :attr:`TextEmbedder.model_id` exists to close.
    """
    require_embedding_source(embedder, embedding_fn)
    if embedder is not None:
        return await embedder.embed(list(texts))
    # `cast` rather than a second `is None` raise: the guard above already
    # decided, and restating the check here would be a second place for the
    # rule to live.
    return await call_embedding_fn_batch(cast("Callable[..., Any]", embedding_fn), list(texts))


async def embed_text(
    text: str,
    *,
    embedder: TextEmbedder | None = None,
    embedding_fn: Callable[..., Any] | None = None,
    timeout: float | None = None,
) -> Any:
    """One vector, from whichever source the caller supplied.

    The per-text counterpart of :func:`embed_texts`, for the sites that embed a
    single query string rather than a corpus.

    Args:
        text: The text to embed.
        embedder: The typed path. Called as a batch of one, since
            :class:`TextEmbedder` is batch-only.
        embedding_fn: The callable path, routed through
            :func:`~dataknobs_data.vector.embedding_fn.call_embedding_fn` so a
            synchronous callable is offloaded rather than run on the loop.
        timeout: Seconds to allow an async ``embedding_fn``. Ignored on the
            ``embedder`` path, where a caller wanting a bound uses
            :func:`asyncio.timeout` around the call --- adding a second
            timeout mechanism to the typed path is what the seam is for
            avoiding.

    Returns:
        ``list[float]`` on the ``embedder`` path; the callable's own return
        otherwise. See :func:`embed_texts` on why those are not reconciled.

    Raises:
        ValueError: Neither was supplied, or both were.
    """
    require_embedding_source(embedder, embedding_fn)
    if embedder is not None:
        return (await embedder.embed([text]))[0]
    return await call_embedding_fn(cast("Callable[..., Any]", embedding_fn), text, timeout=timeout)


class SyncTextEmbedder:
    """A :class:`TextEmbedder` reached from synchronous code.

    Five embedding sites in this package are plain ``def`` --- ``Query.near_text``,
    ``VectorField.from_text``, and the three sync ``bulk_embed_and_store``
    lanes. None of them can await, so none can take a :class:`TextEmbedder`
    directly, and giving the protocol a synchronous twin would put the seam
    back where it started: two shapes for one concept.

    This is the other way round. It holds one
    :class:`~dataknobs_common.sync_bridge.SyncLoopBridge` --- a private event
    loop on a daemon thread, so it is callable from plain sync code *and* from
    inside a running loop without the ``run_until_complete`` deadlock --- and
    exposes the protocol's two arities as ordinary methods. Those methods are
    the shape the sync sites already declare, so **no sync signature changes**:

    "Callable from inside a running loop" means it does not *deadlock*. It
    still *blocks*: the calling thread waits on the bridge's result for the
    whole embedding, so every other task on the caller's loop is stalled for
    a network round trip. From async code, ``await embedder.embed(...)``
    directly --- this class is for the five ``def`` sites that cannot.

    ```python
    sync = SyncTextEmbedder(await create_text_embedder(config))
    try:
        query.near_text("some text", sync.embed_one)
        store.bulk_embed_and_store(records, "body", embedding_fn=sync.embed)
    finally:
        sync.close()
    ```

    :meth:`embed_one` and :meth:`embed` are separate methods rather than one
    arity-polymorphic call, because that polymorphism is what every consumer
    of :meth:`AsyncLLMProvider.embed` had to narrow for itself and is half of
    what this module exists to end.

    The bridge costs one daemon thread for the object's lifetime, so build one
    and keep it rather than one per call. It is a daemon, so it can never block
    process exit; :meth:`close` is for deterministic teardown, and the class is
    a context manager for the same reason.
    """

    def __init__(self, embedder: TextEmbedder, *, timeout: float | None = None) -> None:
        """Args:
        embedder: The async embedder to reach.
        timeout: Seconds to allow each call, giving a synchronous caller an
            upper bound on a blocking wait it cannot otherwise cancel.
        """
        from dataknobs_common.sync_bridge import SyncLoopBridge

        self._embedder = embedder
        self._timeout = timeout
        self._bridge = SyncLoopBridge(thread_name="dk-sync-embedder")

    @property
    def dimensions(self) -> int:
        """The wrapped embedder's --- a bridge does not reshape."""
        return self._embedder.dimensions

    @property
    def model_id(self) -> str:
        """The wrapped embedder's staleness key, unchanged.

        A vector stored through the bridge must be indistinguishable from one
        stored without it.
        """
        return self._embedder.model_id

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Blocking :meth:`TextEmbedder.embed`.

        Satisfies the batch sync sites' ``embedding_fn`` parameter, which
        declares ``Callable[[list[str]], np.ndarray | list[list[float]]]``.
        That union is not an accommodation for this class: those sites hand
        the result to ``pair_records_with_vectors``, which requires only
        "something indexable and sized", so the list arm was always accepted
        at runtime and the annotation simply understated it.
        """
        return self._bridge.run(self._embedder.embed(list(texts)), timeout=self._timeout)

    def embed_one(self, text: str) -> list[float]:
        """Blocking single-text embed, for the per-text sync sites.

        Satisfies ``Query.near_text``'s ``Callable[[str], np.ndarray |
        list[float]]`` and ``VectorField.from_text``'s
        ``Callable[[str], Any]``. ``near_text`` hands the result straight to
        :meth:`Query.similar_to`, which has always declared the same union.
        """
        return self.embed([text])[0]

    def close(self) -> None:
        """Stop the bridge's loop and join its thread.

        Idempotent. Closes only the bridge --- the wrapped embedder is not
        this object's to close, since it was handed in already built.
        """
        self._bridge.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
