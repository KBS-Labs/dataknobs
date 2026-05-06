"""Tests for ParallelLLMExecutor.fail_fast cancellation."""

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


def _patch_dispatching_complete(
    provider: EchoProvider,
    handlers: dict[str, "asyncio.Future[LLMResponse] | None"],
    counter: dict[str, int] | None = None,
) -> None:
    """Monkey-patch ``provider.complete`` to dispatch by user-message content.

    Each handler is an async callable taking ``messages`` and returning an
    ``LLMResponse`` (or raising). Behavior is selected by the first user
    message's content; unknown content raises a clear test error.

    Why monkey-patch instead of ``set_response_function``: ``EchoProvider``'s
    response function is sync, so it cannot ``await asyncio.sleep`` to model
    long-running provider calls. Existing tests in this package use the same
    pattern (see ``test_concurrency_limit``).
    """

    async def dispatching_complete(
        messages: list[LLMMessage],
        config_overrides: dict | None = None,
        **kwargs,
    ) -> LLMResponse:
        content = messages[0].content if messages else ""
        if counter is not None:
            counter[content] = counter.get(content, 0) + 1
        handler = handlers.get(content)
        if handler is None:
            raise AssertionError(
                f"No handler registered for content {content!r}"
            )
        return await handler(messages)

    provider.complete = dispatching_complete  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers building per-content async handlers
# ---------------------------------------------------------------------------


def _slow_response(delay: float, content: str):
    async def _h(messages: list[LLMMessage]) -> LLMResponse:
        await asyncio.sleep(delay)
        return text_response(f"ok-{content}")

    return _h


def _immediate_error(exc: Exception):
    async def _h(messages: list[LLMMessage]) -> LLMResponse:
        raise exc

    return _h


def _delayed_error(delay: float, exc: Exception):
    async def _h(messages: list[LLMMessage]) -> LLMResponse:
        await asyncio.sleep(delay)
        raise exc

    return _h


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_fail_fast_cancels_remaining_on_first_error(
    provider: EchoProvider,
) -> None:
    """First task fails immediately; remaining slow tasks must be cancelled."""
    handlers = {
        "fail": _immediate_error(RuntimeError("simulated")),
        "slow1": _slow_response(1.0, "slow1"),
        "slow2": _slow_response(1.0, "slow2"),
        "slow3": _slow_response(1.0, "slow3"),
        "slow4": _slow_response(1.0, "slow4"),
    }
    _patch_dispatching_complete(provider, handlers)
    executor = ParallelLLMExecutor(provider, max_concurrency=5, fail_fast=True)

    start = time.monotonic()
    results = await executor.execute({
        "a": LLMTask(messages=_msg("fail")),
        "b": LLMTask(messages=_msg("slow1")),
        "c": LLMTask(messages=_msg("slow2")),
        "d": LLMTask(messages=_msg("slow3")),
        "e": LLMTask(messages=_msg("slow4")),
    })
    elapsed = time.monotonic() - start

    # Far below the 1.0s slow-task delay; CI-stable headroom.
    assert elapsed < 0.5, f"fail_fast should short-circuit, elapsed={elapsed}"
    assert results["a"].success is False
    assert isinstance(results["a"].error, RuntimeError)
    for tag in ("b", "c", "d", "e"):
        assert results[tag].success is False
        assert isinstance(results[tag].error, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_execute_mixed_fail_fast_cancels_remaining(
    provider: EchoProvider,
) -> None:
    """fail_fast cancels both LLM and async deterministic siblings."""
    handlers = {
        "fail": _immediate_error(RuntimeError("simulated")),
        "slow_llm": _slow_response(1.0, "slow_llm"),
    }
    _patch_dispatching_complete(provider, handlers)

    async def slow_async_fn(delay: float) -> str:
        await asyncio.sleep(delay)
        return "completed"

    executor = ParallelLLMExecutor(provider, max_concurrency=5, fail_fast=True)

    start = time.monotonic()
    results = await executor.execute_mixed({
        "fail_llm": LLMTask(messages=_msg("fail")),
        "slow_llm": LLMTask(messages=_msg("slow_llm")),
        "slow_det": DeterministicTask(fn=slow_async_fn, args=(1.0,)),
    })
    elapsed = time.monotonic() - start

    # Far below the 1.0s slow-task delay; CI-stable headroom.
    assert elapsed < 0.5, f"fail_fast should short-circuit, elapsed={elapsed}"
    assert results["fail_llm"].success is False
    assert isinstance(results["fail_llm"].error, RuntimeError)
    for tag in ("slow_llm", "slow_det"):
        assert results[tag].success is False
        assert isinstance(results[tag].error, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_execute_mixed_fail_fast_preserves_completed_results(
    provider: EchoProvider,
) -> None:
    """Tasks completing before the failure keep their original results."""
    handlers = {
        "a": _slow_response(0.005, "a"),
        "b": _slow_response(0.005, "b"),
        "c": _slow_response(0.005, "c"),
        "d": _delayed_error(0.05, RuntimeError("d failed")),
        "e": _slow_response(1.0, "e"),
    }
    _patch_dispatching_complete(provider, handlers)
    executor = ParallelLLMExecutor(provider, max_concurrency=5, fail_fast=True)

    start = time.monotonic()
    results = await executor.execute({
        "a": LLMTask(messages=_msg("a")),
        "b": LLMTask(messages=_msg("b")),
        "c": LLMTask(messages=_msg("c")),
        "d": LLMTask(messages=_msg("d")),
        "e": LLMTask(messages=_msg("e")),
    })
    elapsed = time.monotonic() - start

    # Should resolve well before the 1s slow task would finish.
    assert elapsed < 0.5, f"fail_fast should short-circuit, elapsed={elapsed}"
    for tag in ("a", "b", "c"):
        assert results[tag].success is True, (
            f"task {tag!r} should have completed before failure: "
            f"error={results[tag].error}"
        )
        assert isinstance(results[tag].value, LLMResponse)
    assert results["d"].success is False
    assert isinstance(results["d"].error, RuntimeError)
    assert results["e"].success is False
    assert isinstance(results["e"].error, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_execute_fail_fast_default_off(provider: EchoProvider) -> None:
    """Without fail_fast, a failure does NOT cancel siblings (regression guard)."""
    handlers = {
        "fail": _immediate_error(RuntimeError("simulated")),
        "ok1": _slow_response(0.05, "ok1"),
        "ok2": _slow_response(0.05, "ok2"),
    }
    _patch_dispatching_complete(provider, handlers)
    executor = ParallelLLMExecutor(provider, max_concurrency=5)
    # No fail_fast kwarg, executor default False.

    results = await executor.execute({
        "a": LLMTask(messages=_msg("fail")),
        "b": LLMTask(messages=_msg("ok1")),
        "c": LLMTask(messages=_msg("ok2")),
    })

    assert results["a"].success is False
    assert isinstance(results["a"].error, RuntimeError)
    assert results["b"].success is True
    assert results["c"].success is True
    # No cancellation occurred
    for tag in ("b", "c"):
        assert not isinstance(results[tag].error, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_per_call_fail_fast_overrides_executor_default(
    provider: EchoProvider,
) -> None:
    """Per-call fail_fast overrides the executor's __init__ value (both directions)."""
    # (a) executor default False, per-call True → cancellation behavior.
    handlers = {
        "fail": _immediate_error(RuntimeError("err")),
        "slow": _slow_response(0.5, "slow"),
    }
    _patch_dispatching_complete(provider, handlers)
    executor_off = ParallelLLMExecutor(provider, max_concurrency=5, fail_fast=False)
    results = await executor_off.execute(
        {
            "a": LLMTask(messages=_msg("fail")),
            "b": LLMTask(messages=_msg("slow")),
        },
        fail_fast=True,
    )
    assert results["a"].success is False
    assert isinstance(results["b"].error, asyncio.CancelledError)

    # (b) executor default True, per-call False → all complete.
    provider2 = EchoProvider({"provider": "echo", "model": "test"})
    _patch_dispatching_complete(provider2, handlers)
    executor_on = ParallelLLMExecutor(provider2, max_concurrency=5, fail_fast=True)
    results = await executor_on.execute(
        {
            "a": LLMTask(messages=_msg("fail")),
            "b": LLMTask(messages=_msg("slow")),
        },
        fail_fast=False,
    )
    assert results["a"].success is False
    assert results["b"].success is True
    assert not isinstance(results["b"].error, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_fail_fast_with_retry_config(provider: EchoProvider) -> None:
    """RetryConfig completes its retries before fail_fast cancels siblings.

    Documents the contract: cancellation triggers on TaskResult.success=False,
    which only occurs after the retry loop has exhausted ``max_attempts``.
    Sibling tasks are not cancelled while a retry is in progress.
    """
    counter: dict[str, int] = {}
    handlers = {
        "task0": _immediate_error(RuntimeError("transient")),
        "task1": _slow_response(0.5, "task1"),
        "task2": _slow_response(0.5, "task2"),
        "task3": _slow_response(0.5, "task3"),
    }
    _patch_dispatching_complete(provider, handlers, counter=counter)
    retry = RetryConfig(max_attempts=3, initial_delay=0.0)
    executor = ParallelLLMExecutor(provider, max_concurrency=5, fail_fast=True)

    results = await executor.execute({
        "task0": LLMTask(messages=_msg("task0"), retry=retry),
        "task1": LLMTask(messages=_msg("task1")),
        "task2": LLMTask(messages=_msg("task2")),
        "task3": LLMTask(messages=_msg("task3")),
    })

    # task0 retried fully (3 attempts) before signalling failure.
    assert counter["task0"] == 3
    assert results["task0"].success is False
    assert isinstance(results["task0"].error, RuntimeError)
    # No retry attempt fires for siblings after their cancel.
    for tag in ("task1", "task2", "task3"):
        assert results[tag].success is False
        assert isinstance(results[tag].error, asyncio.CancelledError)
        # Each sibling was either started (counter == 1) or queued
        # behind the semaphore (counter missing); never retried.
        assert counter.get(tag, 0) <= 1


@pytest.mark.asyncio
async def test_fail_fast_no_failures_completes_normally(
    provider: EchoProvider,
) -> None:
    """Under fail_fast=True, all-success runs return all results successfully."""
    provider.set_responses([
        text_response("r1"),
        text_response("r2"),
        text_response("r3"),
        text_response("r4"),
        text_response("r5"),
    ])
    executor = ParallelLLMExecutor(provider, max_concurrency=5, fail_fast=True)

    results = await executor.execute({
        "a": LLMTask(messages=_msg("q1")),
        "b": LLMTask(messages=_msg("q2")),
        "c": LLMTask(messages=_msg("q3")),
        "d": LLMTask(messages=_msg("q4")),
        "e": LLMTask(messages=_msg("q5")),
    })

    assert len(results) == 5
    for tag in ("a", "b", "c", "d", "e"):
        assert results[tag].success is True
        assert isinstance(results[tag].value, LLMResponse)
        # No cancellation surfaces.
        assert results[tag].error is None
