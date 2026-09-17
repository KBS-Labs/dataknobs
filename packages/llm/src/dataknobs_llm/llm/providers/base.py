# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Base adapter for synchronous LLM provider access."""

from collections.abc import AsyncGenerator, Iterator
from typing import List, Union, Any

from dataknobs_common import SyncLoopBridge

from ..base import (
    AsyncLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMStreamResponse,
    ModelCapability,
)


class SyncProviderAdapter:
    """Sync adapter for async LLM providers.

    Every method that reaches the wrapped provider runs its coroutine on a
    private :class:`~dataknobs_common.sync_bridge.SyncLoopBridge` loop, so the
    adapter is callable from plain synchronous code and from inside a running
    event loop alike. The alternative --- ``loop.run_until_complete`` on
    whatever loop the caller is on --- raises ``RuntimeError: This event loop
    is already running`` in the second case, which is the case a synchronous
    wrapper exists to serve.

    The bridge costs one daemon thread for the adapter's lifetime.
    :meth:`close` ends it.
    """

    #: Loop-thread name for this adapter's bridge. Registered by the bridge
    #: itself, so the leaked-thread guard sees it rather than being escaped by
    #: it; it is a diagnostic label, not an opt-out.
    BRIDGE_THREAD_NAME = "dk-sync-llm-provider"

    def __init__(self, async_provider: AsyncLLMProvider):
        """Initialize with async provider.

        Args:
            async_provider: The async provider to wrap.
        """
        self.async_provider = async_provider
        self._bridge = SyncLoopBridge(thread_name=self.BRIDGE_THREAD_NAME)
        self._closed = False

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
        return self._bridge.run(self.async_provider.initialize())

    def close(self) -> None:
        """Close the provider synchronously, then this adapter's bridge.

        Idempotent, as it was before the bridge owned the loop: a second call
        has no provider left to close and no loop left to close it on, so it
        returns. The bridge goes down in a ``finally`` so a provider that
        fails to close does not strand the thread.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._bridge.run(self.async_provider.close())
        finally:
            self._bridge.close()

    def complete(self, messages: Union[str, List[LLMMessage]], **kwargs: Any) -> LLMResponse:
        """Generate completion synchronously."""
        return self._bridge.run(self.async_provider.complete(messages, **kwargs))

    def stream(
        self, messages: Union[str, List[LLMMessage]], **kwargs: Any
    ) -> Iterator[LLMStreamResponse]:
        """Stream completion synchronously."""

        async def _stream() -> AsyncGenerator[LLMStreamResponse, None]:
            async for chunk in self.async_provider.stream_complete(messages, **kwargs):
                yield chunk

        # Drive the async generator one step at a time on the bridge loop. The
        # generator is bound to whichever loop first iterates it, so every
        # ``__anext__`` and the final ``aclose`` have to go through the same
        # bridge --- including when the consumer abandons the stream partway
        # and the ``finally`` below is what runs.
        async_gen = _stream()
        try:
            while True:
                try:
                    yield self._bridge.run(async_gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            self._bridge.run(async_gen.aclose())

    def embed(
        self, texts: Union[str, List[str]], **kwargs: Any
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings synchronously."""
        return self._bridge.run(self.async_provider.embed(texts, **kwargs))

    def validate_model(self) -> bool:
        """Validate model synchronously."""
        return self._bridge.run(self.async_provider.validate_model())

    def get_capabilities(self) -> List[ModelCapability]:
        """Get capabilities synchronously."""
        return self.async_provider.get_capabilities()

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self.async_provider.is_initialized
