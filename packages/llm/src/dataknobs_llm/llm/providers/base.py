# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Base adapter for synchronous LLM provider access."""

import threading
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine, Iterator
from typing import Any, List, Self, TypeVar, Union

from dataknobs_common import SyncLoopBridge

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


class SyncProviderAdapter:
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
        self.async_provider = async_provider
        self._timeout = timeout
        self._owns_bridge = bridge is None
        self._bridge = bridge
        # Guards lazy construction only. The bridge itself is thread-safe, so
        # nothing past the first `_ensure_bridge` needs serializing.
        self._bridge_lock = threading.Lock()
        self._closed = False

    def _ensure_bridge(self) -> SyncLoopBridge:
        """This adapter's bridge, built on first use.

        Deferred because construction is also how a consumer reaches
        ``provider_name`` and :meth:`get_capabilities`, neither of which
        awaits anything --- and a daemon thread per discovery-only
        construction is a cost the factory did not previously have.
        """
        bridge = self._bridge
        if bridge is not None:
            return bridge
        with self._bridge_lock:
            if self._bridge is None:
                self._bridge = SyncLoopBridge(thread_name=self.BRIDGE_THREAD_NAME)
            return self._bridge

    def _run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run ``coro`` on this adapter's bridge and return its result.

        The one place the bridge is reached, so the six async-reaching methods
        cannot disagree about which loop they run on or whether ``timeout``
        applies --- one copy of that decision per method is what made this a
        six-method defect rather than a one-method one.
        """
        if self._closed:
            # Without this, an adapter closed before it was ever used would
            # build a *second* bridge here and quietly work --- because the
            # first was never built, so there is no closed bridge left to
            # refuse. The thread that one allocates is then leaked by
            # construction: `close()` has already run.
            coro.close()
            raise RuntimeError(
                "SyncProviderAdapter is closed; build another rather than reusing this one"
            )
        try:
            bridge = self._ensure_bridge()
        except BaseException:
            # Symmetric with the bridge's own refusal paths: a coroutine that
            # is never handed to a loop must be closed, or it warns as
            # never-awaited from wherever collection happens to run.
            coro.close()
            raise
        return bridge.run(coro, timeout=self._timeout)

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

    def close(self) -> None:
        """Close the provider synchronously, then this adapter's bridge.

        Idempotent, as it was before the bridge owned the loop. The provider
        is closed once --- ``_closed`` records that, and only that. The bridge
        is asked on **every** call, because the bridge is already idempotent,
        concurrency-safe, and the thing that actually holds the thread: an
        early return on a flag beside it is how a teardown that raises part
        way leaves the flag true, the loop thread running, and every later
        ``close()`` returning without ever reaching it. Reachable through the
        re-entrancy :meth:`SyncLoopBridge.run` documents by name --- both
        calls below refuse from the bridge's own thread --- and unrecoverable
        once it happens, since nothing else holds a reference to the thread.
        """
        try:
            if not self._closed:
                self._closed = True
                # Not through `_run`, which refuses once `_closed` is set ---
                # and this method has just set it. Teardown is the one call
                # allowed after the adapter is marked closed.
                self._ensure_bridge().run(self.async_provider.close(), timeout=self._timeout)
        finally:
            self._end_bridge()

    async def aclose(self) -> None:
        """Close the provider from async code, then this adapter's bridge.

        The async twin of :meth:`close`, and the reason it exists is that
        :meth:`close` reaches the provider *through* the bridge --- so an
        async holder calling it blocks its own event loop for the provider's
        entire teardown, which for a real provider is an HTTP round trip.
        Here the provider is awaited directly, and the only synchronous part
        left is the bridge's own shutdown: a ``stop`` on an idle loop and a
        thread join, no I/O.

        Shares :meth:`_end_bridge` with :meth:`close` rather than repeating
        it, so the two cannot come to disagree about whose bridge it is.
        """
        try:
            if not self._closed:
                self._closed = True
                await self.async_provider.close()
        finally:
            self._end_bridge()

    def _end_bridge(self) -> None:
        """End the bridge iff this adapter built it.

        A bridge handed to the constructor is shared --- ending it here would
        close it under whatever else is using it.
        """
        if self._owns_bridge and self._bridge is not None:
            self._bridge.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

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
            # Skipped once the adapter is closed: a ``run`` after ``close``
            # raises, and this ``finally`` runs from a finalizer whenever the
            # consumer merely dropped the stream. Nothing is left undone ---
            # the bridge's own teardown runs ``shutdown_asyncgens``, which is
            # what closed this generator.
            #
            # ``stream_complete`` is declared ``-> AsyncIterator`` while every
            # implementation is an async *generator* (the base says so, in a
            # comment on the abstract method). ``aclose`` belongs to the
            # narrower type, so the check is what lets a conforming iterator
            # that is not a generator through rather than crashing on it.
            if not self._closed and isinstance(async_gen, AsyncGenerator):
                self._run(async_gen.aclose())

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
