"""Base adapter for synchronous LLM provider access."""

from typing import List, Union, Dict, Any

from ..base import LLMMessage, LLMResponse, AsyncLLMProvider, ModelCapability


class SyncProviderAdapter:
    """Sync adapter for async LLM providers."""

    def __init__(self, async_provider: AsyncLLMProvider):
        """Initialize with async provider.

        Args:
            async_provider: The async provider to wrap.
        """
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
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.async_provider.initialize())

    def close(self) -> None:
        """Close the provider synchronously."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.async_provider.close())

    def complete(self, messages: Union[str, List[LLMMessage]], **kwargs) -> LLMResponse:
        """Generate completion synchronously."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.async_provider.complete(messages, **kwargs))

    def stream(self, messages: Union[str, List[LLMMessage]], **kwargs):
        """Stream completion synchronously."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _stream():
            async for chunk in self.async_provider.stream_complete(messages, **kwargs):
                yield chunk

        # Convert async generator to sync generator
        async_gen = _stream()
        try:
            while True:
                try:
                    yield loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.run_until_complete(async_gen.aclose())

    def embed(
        self, texts: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings synchronously."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.async_provider.embed(texts, **kwargs))

    def function_call(
        self, messages: List[LLMMessage], functions: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        """Make function call synchronously."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.async_provider.function_call(messages, functions, **kwargs)
        )

    def validate_model(self) -> bool:
        """Validate model synchronously."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.async_provider.validate_model())  # type: ignore

    def get_capabilities(self) -> List[ModelCapability]:
        """Get capabilities synchronously."""
        return self.async_provider.get_capabilities()

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self.async_provider.is_initialized
