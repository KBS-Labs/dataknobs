"""Tests for EchoProvider error simulation via ErrorResponse."""

import pytest

from dataknobs_llm import EchoProvider, LLMMessage
from dataknobs_llm.testing import ErrorResponse, text_response


@pytest.fixture
def provider() -> EchoProvider:
    """Create a fresh EchoProvider for testing."""
    return EchoProvider({"provider": "echo", "model": "test"})


@pytest.mark.asyncio
async def test_error_response_raises(provider: EchoProvider) -> None:
    """ErrorResponse in queue raises the contained exception."""
    provider.set_responses([ErrorResponse(RuntimeError("provider unavailable"))])

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await provider.complete([LLMMessage(role="user", content="hello")])


@pytest.mark.asyncio
async def test_mixed_success_and_error_queue(provider: EchoProvider) -> None:
    """Interleaved success and error responses work correctly."""
    provider.set_responses([
        text_response("ok first"),
        ErrorResponse(RuntimeError("fail")),
        text_response("ok third"),
    ])

    # First call succeeds
    response1 = await provider.complete("first")
    assert response1.content == "ok first"

    # Second call raises
    with pytest.raises(RuntimeError, match="fail"):
        await provider.complete("second")

    # Third call succeeds
    response3 = await provider.complete("third")
    assert response3.content == "ok third"


@pytest.mark.asyncio
async def test_error_response_with_call_history(provider: EchoProvider) -> None:
    """Failed calls are tracked in call history."""
    provider.set_responses([
        text_response("ok"),
        ErrorResponse(ValueError("bad input")),
    ])

    await provider.complete("first")
    with pytest.raises(ValueError, match="bad input"):
        await provider.complete("second")

    # Both calls recorded
    assert provider.call_count == 2
    assert provider.calls[0]["response"] is not None
    assert provider.calls[1]["response"] is None
    assert provider.calls[1]["error"] is True


@pytest.mark.asyncio
async def test_error_response_in_pattern_matching(provider: EchoProvider) -> None:
    """ErrorResponse in pattern matching raises the exception."""
    provider.add_pattern_response(r"fail", ErrorResponse(ConnectionError("timeout")))
    provider.add_pattern_response(r"ok", text_response("all good"))

    # Matching pattern triggers error
    with pytest.raises(ConnectionError, match="timeout"):
        await provider.complete("please fail now")

    # Non-matching pattern returns normal response
    response = await provider.complete("everything is ok")
    assert response.content == "all good"


@pytest.mark.asyncio
async def test_error_response_via_response_function(provider: EchoProvider) -> None:
    """ErrorResponse returned by response function raises the exception."""
    call_count = 0

    def dynamic_response(messages: list[LLMMessage]):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return ErrorResponse(IOError("disk full"))
        return text_response(f"response {call_count}")

    provider.set_response_function(dynamic_response)

    r1 = await provider.complete("first")
    assert r1.content == "response 1"

    with pytest.raises(IOError, match="disk full"):
        await provider.complete("second")

    r3 = await provider.complete("third")
    assert r3.content == "response 3"
