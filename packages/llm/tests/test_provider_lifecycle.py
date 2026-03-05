"""Tests for LLM provider lifecycle management.

Verifies that:
- _check_ready() guards against not-initialized and closing states
- close() cancels in-flight requests
- close() is idempotent (double-close is safe)
- close() calls _close_client() on each provider
- async with context manager works correctly
- sync with on AsyncLLMProvider raises TypeError
- Requests after close() raise ResourceError
"""

from __future__ import annotations

import asyncio

import pytest
from dataknobs_common.exceptions import ResourceError

from dataknobs_llm import EchoProvider, LLMMessage
from dataknobs_llm.llm.base import AsyncLLMProvider
from dataknobs_llm.testing import text_response


@pytest.fixture()
def provider() -> EchoProvider:
    return EchoProvider({"provider": "echo", "model": "test"})


def _msg(content: str = "hi") -> list[LLMMessage]:
    return [LLMMessage(role="user", content=content)]


class TestCheckReady:
    """Tests for LLMProvider._check_ready()."""

    def test_raises_when_not_initialized(self, provider: EchoProvider) -> None:
        """Uninitialized provider raises ResourceError."""
        provider._is_initialized = False
        with pytest.raises(ResourceError, match="not initialized"):
            provider._check_ready()

    def test_raises_when_closing(self, provider: EchoProvider) -> None:
        """Provider that is closing raises ResourceError."""
        provider._is_initialized = True
        provider._is_closing = True
        with pytest.raises(ResourceError, match="closing"):
            provider._check_ready()

    def test_passes_when_ready(self, provider: EchoProvider) -> None:
        """Initialized, non-closing provider passes the check."""
        provider._is_initialized = True
        provider._is_closing = False
        provider._check_ready()  # Should not raise


class TestClose:
    """Tests for AsyncLLMProvider.close()."""

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self, provider: EchoProvider) -> None:
        """Calling close() twice does not error."""
        await provider.initialize()
        await provider.close()
        await provider.close()  # Should not raise
        assert not provider.is_initialized

    @pytest.mark.asyncio
    async def test_close_sets_not_initialized(self, provider: EchoProvider) -> None:
        """After close(), provider is no longer initialized."""
        await provider.initialize()
        assert provider.is_initialized
        await provider.close()
        assert not provider.is_initialized

    @pytest.mark.asyncio
    async def test_close_cancels_in_flight(self) -> None:
        """close() cancels tracked in-flight requests."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        await provider.initialize()

        async def slow_work() -> str:
            await asyncio.sleep(10)
            return "done"

        # Simulate an in-flight task
        task = asyncio.create_task(slow_work())
        provider._in_flight.add(task)

        # close() should cancel it
        await provider.close()
        assert task.cancelled()
        assert len(provider._in_flight) == 0

    @pytest.mark.asyncio
    async def test_close_resets_is_closing(self, provider: EchoProvider) -> None:
        """After close() completes, _is_closing is reset to False."""
        await provider.initialize()
        await provider.close()
        assert not provider._is_closing


class TestCloseClient:
    """Tests for _close_client() hook on providers."""

    @pytest.mark.asyncio
    async def test_echo_close_client_is_noop(self) -> None:
        """EchoProvider._close_client() is a no-op."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        await provider.initialize()
        # Should complete without error
        await provider._close_client()

    @pytest.mark.asyncio
    async def test_close_calls_close_client(self) -> None:
        """close() delegates to _close_client()."""
        close_client_called = False

        class _TrackingProvider(EchoProvider):
            async def _close_client(self) -> None:
                nonlocal close_client_called
                close_client_called = True

        provider = _TrackingProvider({"provider": "echo", "model": "test"})
        await provider.initialize()
        await provider.close()
        assert close_client_called


class TestAsyncContextManager:
    """Tests for async with usage."""

    @pytest.mark.asyncio
    async def test_async_with_initializes_and_closes(self) -> None:
        """async with initializes on entry and closes on exit."""
        provider = EchoProvider({"provider": "echo", "model": "test"})
        provider.set_responses([text_response("ok")])

        async with provider:
            assert provider.is_initialized
            result = await provider.complete(_msg())
            assert result.content == "ok"

        assert not provider.is_initialized


class TestSyncContextManagerBlocked:
    """Tests that sync 'with' raises TypeError on AsyncLLMProvider."""

    def test_sync_with_raises_type_error(self, provider: EchoProvider) -> None:
        """Using 'with' on an async provider raises TypeError."""
        with pytest.raises(TypeError, match="async with"):
            with provider:
                pass  # pragma: no cover


class TestAnalyzeResponse:
    """Tests for _analyze_response() hook."""

    @pytest.mark.asyncio
    async def test_analyze_response_passthrough(
        self, provider: EchoProvider
    ) -> None:
        """Default _analyze_response returns the response unchanged."""
        provider.set_responses([text_response("hello")])
        await provider.initialize()
        response = await provider.complete(_msg())
        analyzed = provider._analyze_response(response)
        assert analyzed is response
        assert analyzed.content == "hello"
