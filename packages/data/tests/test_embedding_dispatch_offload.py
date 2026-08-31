"""Reproduce-first async-correctness tests for the embedding dispatch.

``call_embedding_fn`` offloads a synchronous ``embedding_fn`` with
``asyncio.to_thread``, and says why in its own docstring: *"embedding is CPU-
or network-bound work and running it inline stalls every other task on the
loop."* Its batch sibling ``embed_texts`` called the same kind of callable
**inline on the loop**, so the rule was stated in one half of one module and
broken in the other — with the broken half being the one that blocks longer,
since it embeds a whole corpus rather than one text.

That asymmetry is not a regression. None of the three batch dispatches
``embed_texts`` replaced offloaded either; it inherited the defect while
consolidating it, which is what makes it fixable in one place.

Ruff's ``ASYNC2xx`` family cannot see this. Those checks detect *known*
blocking calls — ``open``, ``time.sleep``, ``subprocess`` — and an arbitrary
caller-supplied callable is none of them. So the guard has to be a test, and
these are it.

Two proofs, deliberately, because each covers the other's blind spot:

* **Blocking detection** — ``assert_no_blocking`` catches a real blocking
  syscall made on the loop. Semantic, but blind to a callable that burns CPU
  without one.
* **Thread identity** — the callable records the thread it ran on. Structural,
  and indifferent to what the callable actually does.

Both FAIL against the inline call and PASS once offloaded.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_data.vector.embedding import embed_text, embed_texts

pytestmark = pytest.mark.asyncio


def _vectors(texts: list[str]) -> np.ndarray:
    return np.asarray([[float(len(t))] * 4 for t in texts], dtype=np.float32)


class TestABatchCallableIsNotRunOnTheLoop:
    """``embed_texts``, whose synchronous path is the one that blocks longest."""

    @requires_blockbuster
    async def test_a_blocking_batch_callable_does_not_block_the_loop(self) -> None:
        """The semantic half.

        A synchronous ``embedding_fn`` that makes a blocking call is the
        realistic shape --- a provider SDK with no async client, a local model
        doing a blocking read. Run inline it stalls every co-tenant of the loop
        for the length of the whole corpus.
        """

        def blocking_fn(texts: list[str]) -> np.ndarray:
            time.sleep(0.01)
            return _vectors(texts)

        with assert_no_blocking():
            vectors = await embed_texts(["one", "two"], embedding_fn=blocking_fn)

        assert len(vectors) == 2

    async def test_a_sync_batch_callable_runs_off_the_event_loop_thread(self) -> None:
        """The structural half, which needs no blocking syscall to be true.

        A callable that burns CPU rather than waiting on a syscall stalls the
        loop exactly as badly and is invisible to ``blockbuster``. Thread
        identity is the proof that survives that: offloaded work runs somewhere
        other than the thread the loop is on.
        """
        loop_thread = threading.get_ident()
        ran_on: list[int] = []

        def recording_fn(texts: list[str]) -> np.ndarray:
            ran_on.append(threading.get_ident())
            return _vectors(texts)

        await embed_texts(["one", "two"], embedding_fn=recording_fn)

        assert len(ran_on) == 1, "the callable must be called exactly once per batch"
        assert ran_on[0] != loop_thread, (
            "a synchronous embedding_fn must run on a worker thread, not the "
            "thread the event loop is running on"
        )

    async def test_an_async_batch_callable_is_still_awaited_on_the_loop(self) -> None:
        """The other half of the branch, which must not change.

        An ``async def`` callable is already cooperative. Pushing it to a
        thread would be wrong twice over --- it would return a coroutine from
        the worker rather than a vector, which is the exact defect
        ``embedding_fn.py`` documents having found in seven copies.
        """
        loop_thread = threading.get_ident()
        ran_on: list[int] = []

        async def async_fn(texts: list[str]) -> np.ndarray:
            ran_on.append(threading.get_ident())
            return _vectors(texts)

        vectors = await embed_texts(["one", "two"], embedding_fn=async_fn)

        assert ran_on == [loop_thread]
        assert len(vectors) == 2

    async def test_an_async_callable_object_is_awaited_not_offloaded(self) -> None:
        """The shape that motivated the shared dispatch in the first place.

        An embedder holding a model handle is naturally written as an object
        with an ``async def __call__``. ``asyncio.iscoroutinefunction`` reports
        it as sync, which is how seven copies came to hand it to a thread and
        store the resulting coroutine as a vector.
        """
        loop_thread = threading.get_ident()
        ran_on: list[int] = []

        class AsyncCallable:
            async def __call__(self, texts: list[str]) -> np.ndarray:
                ran_on.append(threading.get_ident())
                return _vectors(texts)

        vectors = await embed_texts(["one", "two"], embedding_fn=AsyncCallable())

        assert ran_on == [loop_thread]
        assert isinstance(vectors, np.ndarray)


class TestTheTwoAritiesAgree:
    """The per-text path already offloaded; this pins that they now match."""

    async def test_both_arities_put_a_sync_callable_on_a_worker_thread(self) -> None:
        """One rule, stated once, obeyed by both.

        ``embed_text`` routes to ``call_embedding_fn`` and always offloaded.
        Asserting the two together is what stops the batch path drifting back:
        a future change that reverts one lane fails against the other.
        """
        loop_thread = threading.get_ident()
        batch_thread: list[int] = []
        single_thread: list[int] = []

        def batch_fn(texts: list[str]) -> np.ndarray:
            batch_thread.append(threading.get_ident())
            return _vectors(texts)

        def single_fn(text: str) -> np.ndarray:
            single_thread.append(threading.get_ident())
            return _vectors([text])[0]

        await embed_texts(["one"], embedding_fn=batch_fn)
        await embed_text("one", embedding_fn=single_fn)

        assert batch_thread[0] != loop_thread
        assert single_thread[0] != loop_thread

    async def test_the_loop_keeps_running_while_a_batch_embeds(self) -> None:
        """What the offload actually buys, stated as a co-tenant.

        The harm this rule names is to *other* tasks on the loop, so the test
        that names it directly is a second task making progress while a
        synchronous batch callable is mid-flight.
        """
        ticks = 0

        async def co_tenant() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.001)

        def slow_fn(texts: list[str]) -> np.ndarray:
            time.sleep(0.05)
            return _vectors(texts)

        task = asyncio.create_task(co_tenant())
        try:
            await asyncio.sleep(0)
            before = ticks
            await embed_texts(["one", "two"], embedding_fn=slow_fn)
            assert ticks > before, (
                "a co-tenant task made no progress while a synchronous batch "
                "callable ran, which is the stall this offload exists to prevent"
            )
        finally:
            task.cancel()
