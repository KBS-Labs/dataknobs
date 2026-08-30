"""How an ``embedding_fn`` gets called.

Every class in this package that embeds text asks the same question first --- is
this callable async, or does it have to be offloaded to a thread? --- and each
one answered it separately:

===================================  =====
Site                                 Copies
===================================  =====
``migration.py``                     5
``sync.py``                          1
``bulk_embed_mixin.py``              1
===================================  =====

Seven copies of a three-line branch is not a problem while every copy is
right. All seven had the same wrong version of it. They asked
:func:`asyncio.iscoroutinefunction`, which answers for *functions* and reports
a callable **object** with an ``async def __call__`` as sync --- and that shape
is the natural way to write an embedder, because an embedder holds a model
handle. Misclassified, it was handed to :func:`asyncio.to_thread`, which called
it in a worker thread and returned the **coroutine** rather than an embedding.

Nothing raised. ``migration.py`` then wrote that coroutine object into the
record as if it were a vector; the record persisted; the vector was garbage.

So the branch lives here, once, over
:func:`~dataknobs_common.callbacks.is_async_callable`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from dataknobs_common.callbacks import is_async_callable

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["call_embedding_fn"]


async def call_embedding_fn(
    embedding_fn: Callable[..., Any],
    text: str,
    *,
    timeout: float | None = None,
) -> Any:
    """Produce one embedding, awaiting or offloading as the callable requires.

    An async callable is awaited on the loop. A synchronous one is offloaded
    with :func:`asyncio.to_thread`, because embedding is CPU- or network-bound
    work and running it inline stalls every other task on the loop.

    Args:
        embedding_fn: The caller's embedding function. Any callable shape:
            an ``async def``, a plain function, or an object whose
            ``__call__`` is either.
        text: The assembled text to embed.
        timeout: Seconds to allow an **async** callable. A synchronous one is
            not bounded: it is already on a worker thread, and
            :func:`asyncio.wait_for` cannot cancel a thread --- it would return
            control while the work carried on, which is a worse answer than
            waiting.

    Returns:
        Whatever ``embedding_fn`` returned. Validating the shape of that is
        the caller's job, and each caller wants something different from it.

    Raises:
        TimeoutError: An async callable exceeded ``timeout``.
    """
    if is_async_callable(embedding_fn):
        result = embedding_fn(text)
        if timeout is not None:
            return await asyncio.wait_for(result, timeout=timeout)
        return await result
    return await asyncio.to_thread(embedding_fn, text)
