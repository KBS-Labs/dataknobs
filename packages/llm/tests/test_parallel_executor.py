"""Tests for ParallelLLMExecutor."""

import asyncio

import pytest

from dataknobs_common.retry import RetryConfig
from dataknobs_llm import (
    EchoProvider,
    LLMMessage,
    LLMResponse,
    LLMTask,
    DeterministicTask,
    ParallelLLMExecutor,
    TaskResult,
)
from dataknobs_llm.testing import ErrorResponse, text_response


@pytest.fixture
def provider() -> EchoProvider:
    """Create a fresh EchoProvider for testing."""
    return EchoProvider({"provider": "echo", "model": "test"})


def _msg(content: str) -> list[LLMMessage]:
    """Shorthand for creating a single-message list."""
    return [LLMMessage(role="user", content=content)]


@pytest.mark.asyncio
async def test_parallel_execution(provider: EchoProvider) -> None:
    """Three LLM tasks run concurrently and all succeed."""
    provider.set_responses([
        text_response("r1"),
        text_response("r2"),
        text_response("r3"),
    ])
    executor = ParallelLLMExecutor(provider, max_concurrency=5)

    results = await executor.execute({
        "a": LLMTask(messages=_msg("q1")),
        "b": LLMTask(messages=_msg("q2")),
        "c": LLMTask(messages=_msg("q3")),
    })

    assert len(results) == 3
    for tag in ("a", "b", "c"):
        assert results[tag].success is True
        assert results[tag].tag == tag
        assert isinstance(results[tag].value, LLMResponse)


@pytest.mark.asyncio
async def test_mixed_execution(provider: EchoProvider) -> None:
    """LLM + sync + async deterministic tasks run together."""
    provider.set_responses([text_response("llm result")])
    executor = ParallelLLMExecutor(provider, max_concurrency=5)

    def sync_fn(x: int, y: int) -> int:
        return x + y

    async def async_fn(text: str) -> str:
        return text.upper()

    results = await executor.execute_mixed({
        "llm": LLMTask(messages=_msg("hello")),
        "sync": DeterministicTask(fn=sync_fn, args=(3, 4)),
        "async": DeterministicTask(fn=async_fn, args=("hello",)),
    })

    assert len(results) == 3
    assert results["llm"].success is True
    assert isinstance(results["llm"].value, LLMResponse)
    assert results["sync"].success is True
    assert results["sync"].value == 7
    assert results["async"].success is True
    assert results["async"].value == "HELLO"


@pytest.mark.asyncio
async def test_error_isolation(provider: EchoProvider) -> None:
    """One failing task does not affect others."""
    provider.set_responses([
        text_response("ok1"),
        ErrorResponse(RuntimeError("simulated")),
        text_response("ok3"),
    ])
    executor = ParallelLLMExecutor(provider, max_concurrency=1)

    results = await executor.execute({
        "a": LLMTask(messages=_msg("q1")),
        "b": LLMTask(messages=_msg("q2")),
        "c": LLMTask(messages=_msg("q3")),
    })

    assert results["a"].success is True
    assert results["b"].success is False
    assert isinstance(results["b"].error, RuntimeError)
    assert results["c"].success is True


@pytest.mark.asyncio
async def test_concurrency_limit(provider: EchoProvider) -> None:
    """With max_concurrency=1, tasks run sequentially."""
    delay_per_task = 0.05  # 50ms

    async def slow_complete(
        messages, config_overrides=None, **kwargs
    ) -> LLMResponse:
        await asyncio.sleep(delay_per_task)
        return text_response("ok")

    # Replace complete with slow version
    provider.complete = slow_complete  # type: ignore[assignment]
    executor = ParallelLLMExecutor(provider, max_concurrency=1)

    start = asyncio.get_event_loop().time()
    results = await executor.execute({
        "a": LLMTask(messages=_msg("q1")),
        "b": LLMTask(messages=_msg("q2")),
        "c": LLMTask(messages=_msg("q3")),
    })
    elapsed = asyncio.get_event_loop().time() - start

    assert all(r.success for r in results.values())
    # Sequential execution: total time >= 3 * delay
    assert elapsed >= 3 * delay_per_task * 0.9  # 10% tolerance


@pytest.mark.asyncio
async def test_retry_on_failure(provider: EchoProvider) -> None:
    """Task with retry succeeds after transient failure."""
    provider.set_responses([
        ErrorResponse(RuntimeError("transient")),
        text_response("recovered"),
    ])
    executor = ParallelLLMExecutor(provider, max_concurrency=5)

    retry = RetryConfig(max_attempts=3, initial_delay=0.01)
    results = await executor.execute({
        "a": LLMTask(messages=_msg("hello"), retry=retry),
    })

    assert results["a"].success is True
    assert results["a"].value.content == "recovered"
    # Provider was called more than once (first failed, second succeeded)
    assert provider.call_count >= 2


@pytest.mark.asyncio
async def test_sequential_with_context_passing(provider: EchoProvider) -> None:
    """Sequential tasks pass previous result as assistant message."""
    provider.set_responses([
        text_response("step1 output"),
        text_response("step2 output"),
    ])
    executor = ParallelLLMExecutor(provider, max_concurrency=5)

    results = await executor.execute_sequential(
        [
            LLMTask(messages=_msg("step 1")),
            LLMTask(messages=_msg("step 2")),
        ],
        pass_result=True,
    )

    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is True

    # Verify second call received the assistant message
    second_call_messages = provider.calls[1]["messages"]
    assistant_msgs = [m for m in second_call_messages if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].content == "step1 output"


@pytest.mark.asyncio
async def test_empty_tasks(provider: EchoProvider) -> None:
    """Executing empty task dict returns empty result."""
    executor = ParallelLLMExecutor(provider, max_concurrency=5)
    results = await executor.execute({})
    assert results == {}


@pytest.mark.asyncio
async def test_duration_tracking(provider: EchoProvider) -> None:
    """TaskResult tracks execution duration."""
    provider.set_responses([text_response("ok")])
    executor = ParallelLLMExecutor(provider, max_concurrency=5)

    results = await executor.execute({
        "a": LLMTask(messages=_msg("hello")),
    })

    assert results["a"].duration_ms > 0
