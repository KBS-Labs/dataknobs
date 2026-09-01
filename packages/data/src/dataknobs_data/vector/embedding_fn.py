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

**Classifying the callable is necessary and not sufficient**, which the first
version of this module got wrong in the same way as the seven it replaced ---
just for a different shape. A plain ``def`` that *returns* a coroutine is
genuinely synchronous, so ``is_async_callable`` is right to say so and the
offload is right to run it on a thread. What comes back is still a coroutine,
and returning it stores the same garbage value by a second route. The result is
therefore re-examined after the call as well.

That defect survived because it was reachable through one entry point and not
the other: the batch dispatch re-examined its result and the per-text one did
not, so a single callable produced a vector from ``embed_texts`` and a coroutine
from ``embed_text``. Both arities now share :func:`_await_or_offload`, so
neither can drift from the other again --- and both offload a synchronous
callable, which the batch lane did not do at all.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

from dataknobs_common.callbacks import is_async_callable, run_callback_off_loop

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["call_embedding_fn", "call_embedding_fn_batch"]


async def _await_or_offload(
    embedding_fn: Callable[..., Any],
    argument: Any,
    timeout: float | None,
) -> Any:
    """Call *embedding_fn* with one argument, resolving whatever comes back.

    Both arities ask this same question, so it is asked once. Three shapes
    reach here and each needs a different answer:

    * **An async callable** --- an ``async def``, or an object whose
      ``__call__`` is one. Awaited on the loop. It is already cooperative, and
      pushing it to a thread would return the coroutine instead of running it.
    * **A synchronous callable returning a value.** Offloaded with
      :func:`asyncio.to_thread`, because embedding is CPU- or network-bound
      work and running it inline stalls every other task on the loop.
    * **A synchronous callable returning an awaitable** --- a wrapper someone
      forgot to mark ``async``, or a lambda over an async embedder. The
      classification is *correct* about the callable and still not enough: the
      call yields a coroutine, so the result has to be re-examined after the
      fact. Building the coroutine on the worker thread is harmless; awaiting
      it here is what turns it into an embedding.

    That third case is why this cannot be a single ``if``. It was handled by
    the batch dispatch and not by the per-text one, so the same callable
    produced a vector through one entry point and a coroutine object through
    the other --- silently, since neither raises.

    **The three shapes are no longer decided here.** They are exactly what
    :func:`~dataknobs_common.callbacks.run_callback_off_loop` decides, and a
    second copy of one judgement is how the two entry points came to disagree
    in the first place. This copy had already drifted from the shared one: it
    asked ``hasattr(result, "__await__")``, which answers ``False`` for a
    generator-based coroutine that :func:`inspect.isawaitable` accepts.

    What stays here is the ``timeout``, whose scope is deliberately narrower
    than the whole call --- see :func:`call_embedding_fn` for why a worker
    thread is left unbounded. Delegating the sync arm wholesale would widen
    it silently, which is why that one arm is still spelled out.
    """
    if timeout is None:
        return await run_callback_off_loop(embedding_fn, argument)

    if is_async_callable(embedding_fn):
        # Nothing is offloaded on this arm, so bounding the whole dispatch
        # bounds exactly the awaitable the contract says it bounds.
        return await asyncio.wait_for(run_callback_off_loop(embedding_fn, argument), timeout)

    # A synchronous callable. The thread stays unbounded on purpose; only a
    # coroutine it hands back is bounded.
    result = await asyncio.to_thread(embedding_fn, argument)
    if inspect.isawaitable(result):
        return await asyncio.wait_for(result, timeout=timeout)
    return result


async def call_embedding_fn(
    embedding_fn: Callable[..., Any],
    text: str,
    *,
    timeout: float | None = None,
) -> Any:
    """Produce one embedding, awaiting or offloading as the callable requires.

    Args:
        embedding_fn: The caller's embedding function. Any callable shape:
            an ``async def``, a plain function, or an object whose
            ``__call__`` is either --- including a plain function that
            *returns* a coroutine.
        text: The assembled text to embed.
        timeout: Seconds to allow an awaitable result. A synchronous callable
            returning a value is not bounded: it is already on a worker
            thread, and :func:`asyncio.wait_for` cannot cancel a thread --- it
            would return control while the work carried on, which is a worse
            answer than waiting.

    Returns:
        Whatever ``embedding_fn`` returned, resolved if it was awaitable.
        Validating the shape of that is the caller's job, and each caller
        wants something different from it.

    Raises:
        TimeoutError: An awaitable result exceeded ``timeout``.
    """
    return await _await_or_offload(embedding_fn, text, timeout)


async def call_embedding_fn_batch(
    embedding_fn: Callable[..., Any],
    texts: list[str],
) -> Any:
    """Produce a batch of embeddings, on the same rules as :func:`call_embedding_fn`.

    The batch sibling exists because the per-text one is not simply looped:
    a batch callable takes the whole list in one call, and that call is the
    *longer* stall of the two. It ran inline on the loop until this shared
    dispatch existed, so the rule the per-text path had always applied was
    stated in one half of this module and broken in the other.

    Args:
        embedding_fn: The caller's batch embedding function, in any of the
            shapes :func:`call_embedding_fn` accepts.
        texts: The batch to embed, passed through as a single argument.

    Returns:
        Whatever ``embedding_fn`` returned, resolved if it was awaitable ---
        usually an ``np.ndarray`` or a ``list[list[float]]``.

    Note:
        No ``timeout``. A bound on a partly-finished batch cancels the work
        and returns nothing usable, so it would cost a corpus to save a
        deadline. A caller wanting one puts :func:`asyncio.timeout` around a
        call it is prepared to lose.
    """
    return await _await_or_offload(embedding_fn, texts, None)
