# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Base adapter for synchronous LLM provider access."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any, List, TypeVar, Union

from dataknobs_common import SyncBridgeAdapter, SyncLoopBridge

from ..base import (
    AsyncLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMStreamResponse,
    ModelCapability,
)

_T = TypeVar("_T")


async def _anext(iterator: AsyncIterator[_T]) -> _T:
    """One step of ``iterator``, as a coroutine.

    ``SyncLoopBridge.run`` hands its argument to
    ``asyncio.run_coroutine_threadsafe``, which requires an actual coroutine.
    An async generator's ``__anext__`` returns one, but ``AsyncIterator`` ---
    the type ``stream_complete`` is declared to return --- promises only an
    ``Awaitable``. Awaiting it here is what makes the declared interface
    drivable rather than only the implementations that happen to exceed it.
    """
    return await iterator.__anext__()


class SyncProviderAdapter(SyncBridgeAdapter):
    """Sync adapter for async LLM providers.

    Every method that *awaits* the wrapped provider runs its coroutine on a
    private :class:`~dataknobs_common.sync_bridge.SyncLoopBridge` loop, so the
    adapter is callable from plain synchronous code and from inside a running
    event loop alike. The alternative --- ``loop.run_until_complete`` on
    whatever loop the caller is on --- raises ``RuntimeError: This event loop
    is already running`` in the second case, which is the case a synchronous
    wrapper exists to serve. :meth:`get_capabilities` and the ``config`` /
    ``provider_name`` / ``is_initialized`` properties also reach the provider,
    and await nothing, so they never touch the bridge.

    "Callable from inside a running loop" means it does not *deadlock*. It
    still **blocks**: the calling thread waits on the bridge's result for the
    whole completion, so every other task on the caller's loop is stalled for
    as long as the provider takes. From async code, ``await provider.complete(
    ...)`` on the async provider directly --- this adapter is for the ``def``
    sites that cannot await. Pass ``timeout`` for an upper bound on a wait a
    synchronous caller has no other way to cancel.

    The bridge costs one daemon thread, created when the adapter first reaches
    its provider and held until :meth:`close`. Building an adapter to read
    ``provider_name`` or :meth:`get_capabilities` therefore costs nothing.
    Hand in a ``bridge`` to share one across adapters --- a resource holding
    one adapter per model otherwise holds one thread per model --- in which
    case it belongs to the caller and :meth:`close` leaves it running.

    The class is a context manager, which is the reliable form::

        with create_llm_provider(config, is_async=False) as provider:
            provider.initialize()
            print(provider.complete("hello").content)

    ``async with`` is the same guarantee for an async holder --- code that
    builds one of these to hand to a ``def`` site in a worker thread, and has
    a loop of its own to keep unblocked while the provider's HTTP session is
    torn down. Entry initializes nothing, unlike
    :meth:`AsyncLLMProvider.__aenter__`: every method here blocks the calling
    thread, ``initialize`` included, so an async holder runs those in a worker
    too and takes the async form only for the teardown::

        async with create_llm_provider(config, is_async=False) as provider:
            await asyncio.to_thread(provider.initialize)
            response = await asyncio.to_thread(provider.complete, "hello")
    """

    #: Loop-thread name for this adapter's bridge. Registered by the bridge
    #: itself, so the leaked-thread guard sees it rather than being escaped by
    #: it; it is a diagnostic label, not an opt-out.
    BRIDGE_THREAD_NAME = "dk-sync-llm-provider"

    def __init__(
        self,
        async_provider: AsyncLLMProvider,
        *,
        bridge: SyncLoopBridge | None = None,
        timeout: float | None = None,
    ):
        """Initialize with async provider.

        Args:
            async_provider: The async provider to wrap.
            bridge: A bridge to run this adapter's coroutines on. The default
                builds a private one on first use and ends it in
                :meth:`close`. A bridge handed in here belongs to the caller:
                it is shared with whatever else uses it and :meth:`close`
                leaves it running.
            timeout: Seconds to allow each call, giving a synchronous caller
                an upper bound on a blocking wait it cannot otherwise cancel.
                ``None`` (the default) waits as long as the provider takes.
        """
        super().__init__(bridge=bridge, timeout=timeout)
        self.async_provider = async_provider

    @property
    def config(self) -> Any:
        """The wrapped provider's configuration.

        Forwarded because this adapter is the object a sync consumer holds, and
        ``provider.config.provider`` is the documented way to recover the
        verbatim configured spelling.
        """
        return self.async_provider.config

    @property
    def provider_name(self) -> str:
        """Canonical family key of the provider this adapter wraps.

        This adapter is not an ``LLMProvider``, so it inherits nothing from
        that base — and it is the only sync provider object the factory
        actually returns (there are no ``SyncLLMProvider`` subclasses in
        tree). Without this forward, every sync consumer degrades to the
        adapter's *class* name, which is the exact defect the family/impl
        split exists to prevent, surviving on the sync half.
        """
        return self.async_provider.provider_name

    @property
    def impl_name(self) -> str:
        """This adapter's own class — it is what served the call.

        The same split every wrapper reports: billed as the family it wraps,
        diagnosed as the class that actually ran.
        """
        return type(self).__name__

    def initialize(self) -> None:
        """Initialize the provider synchronously."""
        return self._run(self.async_provider.initialize())

    def _close_inner(self) -> None:
        """Close the provider --- which this adapter *does* own.

        Unlike the base's other two adopters, the provider was not handed in
        by a consumer who keeps a reference: the factory builds it and hands
        back only this adapter, so nothing else can close it.
        """
        self._run_teardown(self.async_provider.close())

    async def _aclose_inner(self) -> None:
        """Close the provider on the bridge, off the holder's loop.

        Not ``await self.async_provider.close()``. :meth:`initialize` goes
        through :meth:`~SyncBridgeAdapter._run`, so a real provider's
        ``aiohttp.ClientSession`` --- and every transport under it, and every
        task :class:`AsyncLLMProvider` tracks in ``_in_flight`` --- is created
        on the **bridge's** loop and belongs to it. Awaiting the teardown on
        the holder's loop instead closes loop-bound objects from a loop that
        owns none of them: ``AsyncLLMProvider.close``'s ``asyncio.gather``
        over those tasks raises ``got Future ... attached to a different
        loop``, and what does not raise is doing its cleanup across a thread
        boundary it was never meant to cross.

        What :meth:`~SyncBridgeAdapter.aclose` actually promises is that the
        *holder's* loop is not blocked, and :func:`asyncio.to_thread` keeps
        that promise without moving the teardown. The provider is closed on
        the bridge either way; only the thread that waits for it differs.
        """
        await asyncio.to_thread(self._close_inner)

    def complete(self, messages: Union[str, List[LLMMessage]], **kwargs: Any) -> LLMResponse:
        """Generate completion synchronously."""
        return self._run(self.async_provider.complete(messages, **kwargs))

    def stream(
        self, messages: Union[str, List[LLMMessage]], **kwargs: Any
    ) -> Iterator[LLMStreamResponse]:
        """Stream completion synchronously.

        The provider's own generator is driven directly rather than through a
        wrapper. Closing a wrapper does **not** close what it was iterating:
        ``async for`` has no implicit ``aclose``, so the provider's generator
        would merely lose its last reference and be finalized later, by the
        loop's async-generator hook, at a moment nobody chose --- and its
        ``finally`` is where a real provider releases the HTTP response the
        stream was reading. Driving it directly makes abandoning a stream
        close it, synchronously, on the loop it is bound to.
        """
        # The generator is bound to whichever loop first iterates it, so every
        # ``__anext__`` and the final ``aclose`` have to go through the same
        # bridge --- including when the consumer abandons the stream partway
        # and the ``finally`` below is what runs.
        async_gen = self.async_provider.stream_complete(messages, **kwargs)
        try:
            while True:
                try:
                    yield self._run(_anext(async_gen))
                except StopAsyncIteration:
                    break
        finally:
            self._close_stream(async_gen)

    def _close_stream(self, async_gen: AsyncIterator[LLMStreamResponse]) -> None:
        """Close an abandoned provider generator, on the loop it is bound to.

        The question is whether there is still a loop to close it *on*, which
        is not the same as whether this adapter is open. Reading ``_closed``
        answered it correctly for an adapter that owns its bridge --- teardown
        runs ``shutdown_asyncgens``, which closes the generator --- and wrongly
        for one that borrowed it: ``close()`` deliberately leaves a borrowed
        bridge running, so nothing drains it, the provider's ``finally`` never
        runs, and the HTTP response it was reading is held until the shared
        bridge closes. Asking the bridge covers both, and covers the adapter
        that was closed before it ever built one.

        A ``run`` issued here still races a concurrent close of a *borrowed*
        bridge, which the bridge documents as undefined; quiesce a shared
        bridge's holders before closing it, as its own docstring says.

        ``stream_complete`` is declared ``-> AsyncIterator`` while every
        implementation is an async *generator* (the base says so, in a comment
        on the abstract method). ``aclose`` belongs to the narrower type, so
        the check is what lets a conforming iterator that is not a generator
        through rather than crashing on it.
        """
        if not isinstance(async_gen, AsyncGenerator):
            return
        bridge = self._bridge
        if bridge is None or bridge.is_closed:
            return
        # Not `_run`: this also has to work after `close()`, which is exactly
        # the borrowed-bridge case above.
        self._run_teardown(async_gen.aclose())

    def embed(
        self, texts: Union[str, List[str]], **kwargs: Any
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings synchronously."""
        return self._run(self.async_provider.embed(texts, **kwargs))

    def validate_model(self) -> bool:
        """Validate model synchronously."""
        return self._run(self.async_provider.validate_model())

    def get_capabilities(self) -> List[ModelCapability]:
        """Get capabilities synchronously."""
        return self.async_provider.get_capabilities()

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self.async_provider.is_initialized
