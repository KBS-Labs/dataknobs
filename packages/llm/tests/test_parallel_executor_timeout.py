"""Tests for ParallelLLMExecutor per-task timeout."""

import asyncio
import time

import pytest

from dataknobs_common.retry import RetryConfig
from dataknobs_llm import (
    DeterministicTask,
    EchoProvider,
    LLMMessage,
    LLMResponse,
    LLMTask,
    ParallelLLMExecutor,
)
from dataknobs_llm.testing import text_response


@pytest.fixture
def provider() -> EchoProvider:
    """Create a fresh EchoProvider for testing."""
    return EchoProvider({"provider": "echo", "model": "test"})


def _msg(content: str) -> list[LLMMessage]:
    """Shorthand for creating a single-message list."""
    return [LLMMessage(role="user", content=content)]


def _patch_slow_complete(
    provider: EchoProvider, delay: float, counter: list[int] | None = None
) -> None:
    """Monkey-patch ``provider.complete`` to sleep ``delay`` then succeed."""

    async def slow_complete(
        messages: list[LLMMessage],
        config_overrides: dict | None = None,
        **kwargs,
    ) -> LLMResponse:
        if counter is not None:
            counter.append(1)
        await asyncio.sleep(delay)
        return text_response("ok")

    provider.complete = slow_complete  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_task_timeout_returns_timeout_error(
    provider: EchoProvider,
) -> None:
    """A per-task timeout fires when the provider exceeds it."""
    _patch_slow_complete(provider, delay=0.5)
    executor = ParallelLLMExecutor(provider, max_concurrency=5)

    start = time.monotonic()
    results = await executor.execute({
        "a": LLMTask(messages=_msg("hello"), timeout=0.05),
    })
    elapsed = time.monotonic() - start

    assert results["a"].success is False
    assert isinstance(results["a"].error, asyncio.TimeoutError)
    # Resolved well before the 0.5s the provider would otherwise take.
    assert elapsed < 0.3


@pytest.mark.asyncio
async def test_default_timeout_applies_when_per_task_unset(
    provider: EchoProvider,
) -> None:
    """Executor-level default_per_task_timeout applies when task.timeout is None."""
    _patch_slow_complete(provider, delay=0.5)
    executor = ParallelLLMExecutor(
        provider, max_concurrency=5, default_per_task_timeout=0.05
    )

    results = await executor.execute({
        "a": LLMTask(messages=_msg("hello")),
    })

    assert results["a"].success is False
    assert isinstance(results["a"].error, asyncio.TimeoutError)


@pytest.mark.asyncio
async def test_per_task_timeout_overrides_default(provider: EchoProvider) -> None:
    """Per-task timeout takes precedence over the executor default."""
    _patch_slow_complete(provider, delay=0.2)
    executor = ParallelLLMExecutor(
        provider, max_concurrency=5, default_per_task_timeout=2.0
    )

    start = time.monotonic()
    results = await executor.execute({
        "a": LLMTask(messages=_msg("hello"), timeout=0.05),
    })
    elapsed = time.monotonic() - start

    assert results["a"].success is False
    assert isinstance(results["a"].error, asyncio.TimeoutError)
    # The per-task 0.05s timeout should fire well before the 2s default.
    assert elapsed < 0.15


@pytest.mark.asyncio
async def test_no_timeout_default_off(provider: EchoProvider) -> None:
    """Without a timeout configured, slow tasks complete normally (regression guard)."""
    _patch_slow_complete(provider, delay=0.05)
    executor = ParallelLLMExecutor(provider, max_concurrency=5)

    results = await executor.execute({
        "a": LLMTask(messages=_msg("hello")),
    })

    assert results["a"].success is True
    assert isinstance(results["a"].value, LLMResponse)


@pytest.mark.asyncio
async def test_timeout_bounds_each_retry_attempt_not_total(
    provider: EchoProvider,
) -> None:
    """Per-task timeout bounds each retry attempt individually.

    With ``max_attempts=3`` and a per-task timeout of 50ms over a 200ms
    provider call, every attempt times out — three attempts fire (not just
    one) and the final result is ``TaskResult(success=False,
    error=TimeoutError)``.
    """
    counter: list[int] = []
    _patch_slow_complete(provider, delay=0.2, counter=counter)
    retry = RetryConfig(max_attempts=3, initial_delay=0.0)
    executor = ParallelLLMExecutor(provider, max_concurrency=5)

    results = await executor.execute({
        "a": LLMTask(messages=_msg("hello"), retry=retry, timeout=0.05),
    })

    assert results["a"].success is False
    assert isinstance(results["a"].error, asyncio.TimeoutError)
    # Each attempt timed out; provider was invoked 3 times.
    assert len(counter) == 3


@pytest.mark.asyncio
async def test_fail_fast_with_timeouts_cancels_on_first_timeout(
    provider: EchoProvider,
) -> None:
    """A timeout produces a failure that triggers fail_fast cancellation."""

    async def dispatching_complete(
        messages: list[LLMMessage],
        config_overrides: dict | None = None,
        **kwargs,
    ) -> LLMResponse:
        content = messages[0].content if messages else ""
        if content == "fast_fail":
            await asyncio.sleep(0.5)  # times out at 0.05s
            return text_response("never")
        # All other tasks: very slow
        await asyncio.sleep(2.0)
        return text_response("never")

    provider.complete = dispatching_complete  # type: ignore[assignment]
    executor = ParallelLLMExecutor(provider, max_concurrency=5, fail_fast=True)

    start = time.monotonic()
    results = await executor.execute({
        "a": LLMTask(messages=_msg("fast_fail"), timeout=0.05),
        "b": LLMTask(messages=_msg("slow1")),
        "c": LLMTask(messages=_msg("slow2")),
        "d": LLMTask(messages=_msg("slow3")),
    })
    elapsed = time.monotonic() - start

    # Resolves shortly after the 50ms timeout, far below the 2s slow-task time.
    assert elapsed < 0.5, f"elapsed={elapsed}"
    assert results["a"].success is False
    assert isinstance(results["a"].error, asyncio.TimeoutError)
    for tag in ("b", "c", "d"):
        assert results[tag].success is False
        assert isinstance(results[tag].error, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_deterministic_task_timeout(provider: EchoProvider) -> None:
    """A DeterministicTask with timeout reports TimeoutError on overrun."""

    async def slow_async(delay: float) -> str:
        await asyncio.sleep(delay)
        return "done"

    executor = ParallelLLMExecutor(provider, max_concurrency=5)
    results = await executor.execute_mixed({
        "a": DeterministicTask(fn=slow_async, args=(0.5,), timeout=0.05),
    })

    assert results["a"].success is False
    assert isinstance(results["a"].error, asyncio.TimeoutError)
