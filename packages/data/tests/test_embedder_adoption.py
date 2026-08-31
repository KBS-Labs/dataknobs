"""How a site chooses between an ``embedder`` and an ``embedding_fn``.

Twenty-five parameter declarations across this package name a thing that
turns text into vectors, and each is gaining an ``embedder`` alongside its
existing callable. If each also wrote its own three-line choice between the
two, that would be the duplication that produced the eight incompatible
spellings in the first place --- so the choice is :func:`embed_texts` and
:func:`embed_text`, and this is where it is pinned.

The two rules worth stating outright, because both are refusals rather than
behaviours and neither is visible from a call site:

* **Neither source is an error**, and it must be raised *before* the loop.
  A bulk method that only discovers it inside the loop returns a
  successful-looking ``[]`` for an empty input.
* **Both sources is an error too.** Resolving it by precedence means one of
  the two silently does not run, and the caller cannot tell which --- the
  same class of error ``model_id`` exists to close.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np
import pytest

from dataknobs_data.testing import DeterministicEmbedder
from dataknobs_data.vector import (
    SyncTextEmbedder,
    TextEmbedder,
    embed_text,
    embed_texts,
    require_embedding_source,
)


def _batch_sync(texts: list[str]) -> np.ndarray:
    """The commonest legacy shape: batch in, ``np.ndarray`` out, synchronous."""
    return np.array([[float(len(t)), 1.0] for t in texts])


async def _batch_async(texts: list[str]) -> np.ndarray:
    """The same, async --- the shape only some sites' annotations admit."""
    return _batch_sync(texts)


class _AsyncCallableObject:
    """A callable *object* whose ``__call__`` is ``async def``.

    The shape an embedder holding a model handle naturally takes, and the one
    :func:`inspect.iscoroutinefunction` gets wrong --- it reports this as
    synchronous, after which it is handed to a worker thread that returns the
    coroutine instead of a vector, with nothing raised.
    """

    async def __call__(self, texts: list[str]) -> np.ndarray:
        return _batch_sync(texts)


class _SyncFunctionReturningACoroutine:
    """A plain ``def`` that hands back an awaitable.

    Classifying the *callable* says sync and classifying the *result* says
    async. Only the second is right here, which is why both are asked.
    """

    def __call__(self, texts: list[str]) -> Any:
        return _batch_async(texts)


# --------------------------------------------------------------------------
# embed_texts — the batch choice
# --------------------------------------------------------------------------


async def test_an_embedder_is_awaited_directly() -> None:
    embedder = DeterministicEmbedder(dimensions=8)

    vectors = await embed_texts(["a", "b"], embedder=embedder)

    assert vectors == await embedder.embed(["a", "b"])


async def test_a_sync_callable_is_called() -> None:
    result = await embed_texts(["alpha"], embedding_fn=_batch_sync)

    assert result[0][0] == pytest.approx(len("alpha"))


async def test_an_async_callable_is_awaited() -> None:
    result = await embed_texts(["alpha"], embedding_fn=_batch_async)

    assert result[0][0] == pytest.approx(len("alpha"))


async def test_an_async_callable_object_is_awaited() -> None:
    """The case seven hand-rolled copies of this branch got wrong."""
    result = await embed_texts(["alpha"], embedding_fn=_AsyncCallableObject())

    assert result[0][0] == pytest.approx(len("alpha"))


async def test_a_sync_callable_returning_an_awaitable_is_awaited() -> None:
    """Neither predicate alone covers this; asking both does.

    One of the two dispatch copies this replaces classified the callable and
    the other classified the result, so each handled a case the other missed.
    """
    result = await embed_texts(["alpha"], embedding_fn=_SyncFunctionReturningACoroutine())

    assert result[0][0] == pytest.approx(len("alpha"))


async def test_the_callables_return_value_is_not_converted() -> None:
    """The callable path must reach a store exactly as it did before.

    Normalizing it to ``list[list[float]]`` for symmetry with the embedder
    path would change the dtype an existing caller's array arrives with, and
    this seam is additive.
    """
    result = await embed_texts(["alpha"], embedding_fn=_batch_sync)

    assert isinstance(result, np.ndarray)


async def test_neither_source_raises() -> None:
    with pytest.raises(ValueError, match="embedder is required"):
        await embed_texts(["alpha"])


async def test_both_sources_raise() -> None:
    with pytest.raises(ValueError, match="not both"):
        await embed_texts(["alpha"], embedder=DeterministicEmbedder(), embedding_fn=_batch_sync)


def test_the_guard_is_callable_on_its_own() -> None:
    """Bulk sites apply it before their loop, so an empty input still reports."""
    require_embedding_source(DeterministicEmbedder(), None)
    require_embedding_source(None, _batch_sync)

    with pytest.raises(ValueError, match="embedder is required"):
        require_embedding_source(None, None)
    with pytest.raises(ValueError, match="not both"):
        require_embedding_source(DeterministicEmbedder(), _batch_sync)


# --------------------------------------------------------------------------
# embed_text — the per-text choice
# --------------------------------------------------------------------------


async def test_embed_text_takes_the_first_of_a_batch_of_one() -> None:
    embedder = DeterministicEmbedder(dimensions=8)

    vector = await embed_text("alpha", embedder=embedder)

    assert vector == (await embedder.embed(["alpha"]))[0]


async def test_embed_text_offloads_a_sync_callable() -> None:
    """Routed through ``call_embedding_fn``, which offloads rather than inlines.

    A synchronous embed run on the event loop stalls every other task on it,
    which is invisible in a single-request test and catastrophic under
    concurrency.
    """
    loop_thread = threading.current_thread()
    seen: list[threading.Thread] = []

    def one(text: str) -> list[float]:
        seen.append(threading.current_thread())
        return [float(len(text))]

    vector = await embed_text("alpha", embedding_fn=one)

    assert vector == [5.0]
    assert seen == [seen[0]] and seen[0] is not loop_thread, (
        "a synchronous embedding function ran on the event loop"
    )


async def test_embed_text_awaits_an_async_callable() -> None:
    async def one(text: str) -> list[float]:
        return [float(len(text))]

    assert await embed_text("alpha", embedding_fn=one) == [5.0]


async def test_embed_text_refuses_neither_and_both() -> None:
    async def one(text: str) -> list[float]:
        return [1.0]

    with pytest.raises(ValueError, match="embedder is required"):
        await embed_text("alpha")
    with pytest.raises(ValueError, match="not both"):
        await embed_text("alpha", embedder=DeterministicEmbedder(), embedding_fn=one)


# --------------------------------------------------------------------------
# SyncTextEmbedder — how a `def` site reaches an async embedder
# --------------------------------------------------------------------------


def test_sync_embedder_runs_from_plain_sync_code() -> None:
    with SyncTextEmbedder(DeterministicEmbedder(dimensions=8)) as sync:
        vectors = sync.embed(["a", "b"])

    assert len(vectors) == 2
    assert all(len(v) == 8 for v in vectors)


def test_sync_embedder_answers_what_the_async_one_would() -> None:
    embedder = DeterministicEmbedder(dimensions=8)
    expected = asyncio.run(embedder.embed(["alpha", "bravo"]))

    with SyncTextEmbedder(embedder) as sync:
        assert sync.embed(["alpha", "bravo"]) == expected
        assert sync.embed_one("alpha") == expected[0]


async def test_sync_embedder_runs_from_inside_a_running_loop() -> None:
    """The case that makes the bridge necessary rather than convenient.

    ``asyncio.run`` and ``run_until_complete`` both fail when a loop is
    already running on the calling thread, which is exactly what happens when
    a sync wrapper is reached from async code. The bridge owns its own loop on
    its own thread, so it cannot deadlock against the caller's.
    """
    sync = SyncTextEmbedder(DeterministicEmbedder(dimensions=4))
    try:
        vectors = await asyncio.to_thread(sync.embed, ["alpha"])
    finally:
        sync.close()

    assert len(vectors[0]) == 4


def test_sync_embedder_forwards_identity_unchanged() -> None:
    """A vector stored through the bridge must look like one stored without it."""
    with SyncTextEmbedder(DeterministicEmbedder(dimensions=16, model_id="m")) as sync:
        assert sync.model_id == "m"
        assert sync.dimensions == 16


def test_sync_embedder_close_is_idempotent() -> None:
    sync = SyncTextEmbedder(DeterministicEmbedder())
    sync.close()
    sync.close()


def test_sync_embedders_methods_fit_the_legacy_parameters() -> None:
    """The reason no sync signature had to change to reach the seam.

    ``embed`` satisfies ``Callable[[list[str]], list[list[float]]]`` and
    ``embed_one`` satisfies ``Callable[[str], list[float]]`` --- between them,
    the shapes the five synchronous sites already declare.
    """
    with SyncTextEmbedder(DeterministicEmbedder(dimensions=4)) as sync:
        batch_param: Any = sync.embed
        single_param: Any = sync.embed_one

        assert len(batch_param(["a", "b"])) == 2
        assert len(single_param("a")) == 4


def test_sync_embedder_is_not_a_text_embedder_but_isinstance_says_it_is() -> None:
    """The protocol's runtime check cannot see the difference, and this proves it.

    ``TextEmbedder.embed`` is ``async def`` and this one is a plain ``def``, so
    a ``SyncTextEmbedder`` is not an embedder in any usable sense. But
    ``isinstance`` against a runtime-checkable protocol checks that the three
    members are *present* and nothing about their signatures, so it answers
    ``True`` --- which is the limitation the protocol's docstring names, met
    here by the one class in this package shaped to trip it.

    Pinned rather than fixed: renaming ``embed`` would break the property the
    class exists for --- that its methods already fit the parameters the
    synchronous sites declare. The annotation is what stops the mistake; the
    ``isinstance`` is a smoke test and this is the smoke.
    """
    with SyncTextEmbedder(DeterministicEmbedder()) as sync:
        assert not asyncio.iscoroutinefunction(sync.embed)
        assert isinstance(sync, TextEmbedder), (
            "if this ever answers False the protocol grew signature checking, "
            "and the docs saying otherwise are now wrong"
        )


async def test_a_sync_embedder_used_as_an_async_one_fails_loudly() -> None:
    """The consolation for the check above: the mistake cannot be silent.

    A non-vector must never reach a store. It does not: the plain ``def``
    returns a list where the caller awaits, which raises immediately rather
    than producing something vector-shaped enough to persist.
    """
    with SyncTextEmbedder(DeterministicEmbedder(dimensions=4)) as sync:
        with pytest.raises(TypeError, match="can't be used in 'await' expression"):
            await embed_texts(["alpha"], embedder=sync)  # type: ignore[arg-type]
