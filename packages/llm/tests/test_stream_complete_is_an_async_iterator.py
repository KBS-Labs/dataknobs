"""``stream_complete`` hands back the iterator, and every provider agrees.

The abstract declaration was ``async def stream_complete(...) -> AsyncIterator``
with a ``pass`` body, which types the call as a *coroutine* wrapping an
iterator. Every implementation is an async generator, so the call really
returns the iterator and ``async for`` over it --- what every call site in the
tree does --- is correct.

Nothing broke, because nobody believed the declaration. What it cost was
advice: mypy reported each correct call site as "not async iterable ... Maybe
you forgot to use 'await'?", and taking that advice raises ``TypeError`` at
runtime. It also reported six of the seven providers, and the capturing
provider in ``testing``, as incompatible overrides of their own base. Thirteen
findings across twelve files, every one of them the declaration being wrong
rather than the code.

``SyncProviderAdapter``, the sync sibling in the same class family, has always
spelled its own ``stream_complete`` as ``def ... -> Iterator``; the async
declaration now matches it.
"""

from __future__ import annotations

import inspect

import pytest

from dataknobs_llm.llm.base import AsyncLLMProvider, LLMConfig
from dataknobs_llm.llm.providers import create_llm_provider


def _implementations() -> list[type[AsyncLLMProvider]]:
    """Every concrete provider, found rather than listed.

    Naming them would pass while an eighth provider silently declared
    ``stream_complete`` as a coroutine --- which is the shape this file exists
    to rule out.
    """
    seen: list[type[AsyncLLMProvider]] = []
    stack = [AsyncLLMProvider]
    while stack:
        for sub in stack.pop().__subclasses__():
            stack.append(sub)
            if "stream_complete" in vars(sub):
                seen.append(sub)
    return seen


def test_the_declaration_is_not_a_coroutine() -> None:
    """The base returns the iterator, so a caller needs no ``await``."""
    assert not inspect.iscoroutinefunction(AsyncLLMProvider.stream_complete)


@pytest.mark.parametrize("provider_cls", _implementations(), ids=lambda c: c.__name__)
def test_every_provider_implements_it_as_an_async_generator(
    provider_cls: type[AsyncLLMProvider],
) -> None:
    """An async generator function, which is what makes the base declaration true."""
    assert inspect.isasyncgenfunction(provider_cls.stream_complete), (
        f"{provider_cls.__name__}.stream_complete is not an async generator, so "
        "`async for` over its result does not work and the base declaration is a lie"
    )


async def test_a_provider_streams_without_being_awaited() -> None:
    """The property end to end, through a real provider rather than by inspection."""
    provider = create_llm_provider(LLMConfig(provider="echo", model="test"), is_async=True)
    await provider.initialize()
    try:
        chunks = [chunk async for chunk in provider.stream_complete("hello")]
    finally:
        await provider.close()

    assert "".join(chunk.delta for chunk in chunks)
