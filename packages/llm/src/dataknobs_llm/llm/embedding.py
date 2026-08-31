"""Adapting an LLM provider to the ``TextEmbedder`` seam.

``dataknobs-data`` declares the protocol and ``dataknobs-llm`` supplies the
implementation, because the dependency runs that way and only that way: ``data``
cannot import ``llm``. Every vector path in ``data`` can therefore *name* an
embedder without any of them knowing that a provider exists.

The adapter is thin on purpose. :meth:`AsyncLLMProvider.embed` already returns
``list[list[float]]`` for a list input --- which is exactly what
:meth:`TextEmbedder.embed` returns --- so there is no conversion here, and that
absence is the point of the seam rather than an oversight. What the adapter
*does* add is the two things a bare provider cannot answer for itself in the
shape a stored vector needs: a settled ``dimensions``, and a stable
``model_id`` to write beside the vector so a later reader can tell whether it
is still valid.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from dataknobs_llm.llm.providers import create_embedding_provider

if TYPE_CHECKING:
    from dataknobs_llm.llm.base import AsyncLLMProvider, LLMConfig

__all__ = ["LLMProviderEmbedder", "create_text_embedder"]


class LLMProviderEmbedder:
    """Presents an :class:`AsyncLLMProvider` as a ``TextEmbedder``.

    Satisfies ``dataknobs_data.vector.TextEmbedder`` structurally; the protocol
    is not imported, because doing so would make ``dataknobs-llm`` depend on
    ``dataknobs-data`` for a type it only needs to *match*.

    Example:
        ```python
        provider = await create_embedding_provider(config)
        embedder = LLMProviderEmbedder(provider)
        vectors = await embedder.embed(["one", "two"])
        ```
    """

    def __init__(
        self,
        provider: AsyncLLMProvider,
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        """Args:
        provider: The initialized embedding provider to adapt.
        model: Model name to report in :attr:`model_id`, overriding the
            provider's configured one. Does **not** change which model the
            provider calls --- it renames the vectors, so passing it when it
            does not match is how a staleness key comes to lie. Use it only
            where the provider's own config understates the model actually
            in use.
        dimensions: Vector length, when the provider's config does not carry
            one. Required in that case: see :attr:`dimensions`.
        """
        self._provider = provider
        self._model = model or getattr(provider.config, "model", None)
        self._dimensions = dimensions

    @property
    def provider(self) -> AsyncLLMProvider:
        """The adapted provider, for a caller that needs to close it."""
        return self._provider

    @property
    def dimensions(self) -> int:
        """Length of every vector this embedder returns.

        Answered from what was *declared* --- the constructor argument, else
        the provider's configured ``dimensions`` --- and otherwise from what
        was *observed* on the first :meth:`embed`. Never by probing: a probe
        is a network round trip, and this is a property callers read freely.

        A declared value is checked against the first batch rather than
        trusted, because the two can disagree and nothing else in the stack
        would notice. ``EchoProvider``, for one, sizes its vectors from
        ``config.options["embedding_dim"]`` and ignores ``config.dimensions``
        entirely, so a config asking for 16 yields 768 with nothing raised.
        See :meth:`embed`.

        Raises:
            ValueError: Nothing declared one and nothing has been embedded
                yet, so the answer is genuinely unknown. Raising beats
                guessing a length that a vector store will then enforce: a
                dimension mismatch discovered at write time names the store,
                not the embedder that was actually misconfigured.
        """
        if self._dimensions is not None:
            return self._dimensions
        configured = getattr(self._provider.config, "dimensions", None)
        if configured is None:
            raise ValueError(
                "embedder dimensions are unknown: the provider config carries no "
                "`dimensions`, none was passed, and nothing has been embedded yet. "
                "Set `dimensions` on the LLMConfig, or pass "
                "LLMProviderEmbedder(..., dimensions=N)."
            )
        self._dimensions = int(configured)
        return self._dimensions

    @property
    def model_id(self) -> str:
        """``provider:model`` --- the staleness key written beside a vector.

        Built from the provider's own name rather than a caller-supplied
        label so that two embedders reaching the same model agree, and two
        reaching different models do not. ``provider_name`` is the provider's
        resolved name, so an override set at construction is reflected here.
        """
        name = getattr(self._provider, "provider_name", None) or type(self._provider).__name__
        return f"{name}:{self._model or 'unknown'}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text, in order.

        An empty batch returns without calling the provider at all. That is a
        contract requirement of the protocol rather than an optimization ---
        providers disagree about what an empty embed request means, and some
        error on one.

        The width of the first batch settles :attr:`dimensions` when nothing
        declared it, and is checked against it when something did. That check
        is the seam earning its keep: a provider whose vectors are not the
        width its config advertises is a real and currently-shipping
        condition, and until something declared a width beside ``embed`` there
        was nowhere for it to be caught. Downstream it is caught by a vector
        store rejecting a write, which names the store.

        Raises:
            TypeError: The provider answered a list input with a single flat
                vector. ``AsyncLLMProvider.embed`` is arity-polymorphic and is
                documented to return ``list[list[float]]`` here; a flat answer
                would otherwise be silently mistaken for a batch of
                one-dimensional vectors.
            ValueError: The vectors are not the width this embedder declares.
        """
        wanted = list(texts)
        if not wanted:
            return []

        raw: Any = await self._provider.embed(wanted)

        if raw and not isinstance(raw[0], (list, tuple)):
            raise TypeError(
                f"{type(self._provider).__name__}.embed returned a flat vector for a "
                f"list of {len(wanted)} texts; expected one vector per text"
            )
        vectors = [[float(x) for x in vector] for vector in raw]

        if vectors:
            observed = len(vectors[0])
            declared = self._dimensions
            if declared is None:
                declared = getattr(self._provider.config, "dimensions", None)
            if declared is None:
                self._dimensions = observed
            elif int(declared) != observed:
                raise ValueError(
                    f"{type(self._provider).__name__} returned {observed}-dimensional "
                    f"vectors but this embedder declares {int(declared)}. Storing them "
                    f"would put vectors of one width under a key promising another. "
                    f"Reconcile the provider's configuration with the declared "
                    f"`dimensions`."
                )

        return vectors


async def create_text_embedder(
    config: LLMConfig | dict[str, Any],
    *,
    dimensions: int | None = None,
) -> LLMProviderEmbedder:
    """Build an embedder from configuration.

    A wrapper over :func:`create_embedding_provider`, which already accepts a
    typed ``LLMConfig`` or any of the dict forms and already forces
    ``mode=embedding``. There is deliberately **no new config type**: an
    embedder config *is* an ``LLMConfig``, the shape embedding providers were
    already configured by, so this adds a runtime surface and not a
    configuration one.

    Args:
        config: Anything :func:`create_embedding_provider` accepts.
        dimensions: Passed through, for a config that carries none.

    Example:
        ```python
        embedder = await create_text_embedder(
            {"embedding": {"provider": "ollama", "model": "nomic-embed-text"}}
        )
        ```
    """
    provider = await create_embedding_provider(config)
    return LLMProviderEmbedder(provider, dimensions=dimensions)
