"""Tests for dataknobs_common.retry module."""

import asyncio
import dataclasses
import inspect
import random
from unittest.mock import AsyncMock

import pytest

from dataknobs_common.retry import (
    BackoffStrategy,
    RetryConfig,
    RetryExecutor,
    compute_backoff_delay,
)
from dataknobs_common.structured_config import StructuredConfig
from dataknobs_common.testing import assert_structured_config_roundtrip


class _AsyncCallableObject:
    """A callable whose ``__call__`` is ``async`` — ``iscoroutinefunction``
    returns False for the *instance*, so calling it produces a coroutine even
    though function-level async detection misses it. Used to exercise the
    executor's result-level (not function-level) async detection.
    """

    def __init__(self, result: object = "ok") -> None:
        self.calls = 0
        self._result = result

    async def __call__(self) -> object:
        self.calls += 1
        return self._result


class _FakeHTTPError(Exception):
    """A real exception carrying a status code — the kind of attribute-encoded
    retryability (``code >= 500``) that a value predicate discriminates but a
    single exception type cannot. A concrete class, not a mock.
    """

    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


# ---------------------------------------------------------------------------
# BackoffStrategy delay calculation
# ---------------------------------------------------------------------------


class TestBackoffDelayCalculation:
    """Test _calculate_delay for each backoff strategy."""

    def test_fixed_delay(self):
        config = RetryConfig(
            initial_delay=2.0,
            backoff_strategy=BackoffStrategy.FIXED,
        )
        executor = RetryExecutor(config)
        assert executor._calculate_delay(1) == 2.0
        assert executor._calculate_delay(2) == 2.0
        assert executor._calculate_delay(5) == 2.0

    def test_linear_delay(self):
        config = RetryConfig(
            initial_delay=1.0,
            backoff_strategy=BackoffStrategy.LINEAR,
        )
        executor = RetryExecutor(config)
        assert executor._calculate_delay(1) == 1.0
        assert executor._calculate_delay(2) == 2.0
        assert executor._calculate_delay(3) == 3.0

    def test_exponential_delay(self):
        config = RetryConfig(
            initial_delay=1.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
        )
        executor = RetryExecutor(config)
        assert executor._calculate_delay(1) == 1.0   # 1.0 * 2^0
        assert executor._calculate_delay(2) == 2.0   # 1.0 * 2^1
        assert executor._calculate_delay(3) == 4.0   # 1.0 * 2^2

    def test_jitter_delay_stays_within_bounds(self):
        config = RetryConfig(
            initial_delay=1.0,
            backoff_strategy=BackoffStrategy.JITTER,
            backoff_multiplier=2.0,
            jitter_range=0.1,
        )
        executor = RetryExecutor(config)
        # With jitter_range=0.1, attempt 1 base delay is 1.0
        # Result should be in [0.9, 1.1]
        for _ in range(50):
            delay = executor._calculate_delay(1)
            assert 0.9 <= delay <= 1.1

    def test_decorrelated_delay_first_attempt(self):
        config = RetryConfig(
            initial_delay=1.0,
            backoff_strategy=BackoffStrategy.DECORRELATED,
        )
        executor = RetryExecutor(config)
        # First attempt (no previous_delay) returns initial_delay
        assert executor._calculate_delay(1, previous_delay=None) == 1.0

    def test_decorrelated_delay_subsequent_attempts(self):
        config = RetryConfig(
            initial_delay=1.0,
            backoff_strategy=BackoffStrategy.DECORRELATED,
        )
        executor = RetryExecutor(config)
        for _ in range(50):
            delay = executor._calculate_delay(2, previous_delay=2.0)
            # Should be in [initial_delay, previous_delay * 3] = [1.0, 6.0]
            assert 1.0 <= delay <= 6.0

    def test_max_delay_caps_result(self):
        config = RetryConfig(
            initial_delay=10.0,
            max_delay=5.0,
            backoff_strategy=BackoffStrategy.FIXED,
        )
        executor = RetryExecutor(config)
        assert executor._calculate_delay(1) == 5.0

    def test_exponential_capped_by_max_delay(self):
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=10.0,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
        )
        executor = RetryExecutor(config)
        # attempt 5: 1.0 * 2^4 = 16.0, capped to 10.0
        assert executor._calculate_delay(5) == 10.0


class TestComputeBackoffDelay:
    """Direct unit tests for the public pure compute_backoff_delay().

    RetryExecutor._calculate_delay delegates to this; the
    TestBackoffDelayCalculation class above is the delegation regression
    guard. These tests pin the pure function's own contract.
    """

    def test_fixed(self):
        for attempt in (1, 2, 5):
            assert (
                compute_backoff_delay(
                    BackoffStrategy.FIXED,
                    attempt=attempt,
                    initial_delay=2.0,
                    max_delay=60.0,
                )
                == 2.0
            )

    def test_linear(self):
        assert (
            compute_backoff_delay(
                BackoffStrategy.LINEAR, attempt=1, initial_delay=1.0, max_delay=60.0
            )
            == 1.0
        )
        assert (
            compute_backoff_delay(
                BackoffStrategy.LINEAR, attempt=3, initial_delay=1.0, max_delay=60.0
            )
            == 3.0
        )

    def test_exponential(self):
        assert (
            compute_backoff_delay(
                BackoffStrategy.EXPONENTIAL,
                attempt=1,
                initial_delay=1.0,
                max_delay=60.0,
                backoff_multiplier=2.0,
            )
            == 1.0
        )
        assert (
            compute_backoff_delay(
                BackoffStrategy.EXPONENTIAL,
                attempt=4,
                initial_delay=1.0,
                max_delay=60.0,
                backoff_multiplier=2.0,
            )
            == 8.0  # 1.0 * 2^3
        )

    def test_jitter_within_bounds_seeded(self):
        random.seed(1234)
        for _ in range(100):
            delay = compute_backoff_delay(
                BackoffStrategy.JITTER,
                attempt=1,
                initial_delay=1.0,
                max_delay=60.0,
                backoff_multiplier=2.0,
                jitter_range=0.1,
            )
            assert 0.9 <= delay <= 1.1

    def test_decorrelated_first_attempt(self):
        assert (
            compute_backoff_delay(
                BackoffStrategy.DECORRELATED,
                attempt=1,
                initial_delay=1.0,
                max_delay=60.0,
                previous_delay=None,
            )
            == 1.0
        )

    def test_decorrelated_subsequent_seeded(self):
        random.seed(99)
        for _ in range(100):
            delay = compute_backoff_delay(
                BackoffStrategy.DECORRELATED,
                attempt=2,
                initial_delay=1.0,
                max_delay=60.0,
                previous_delay=2.0,
            )
            assert 1.0 <= delay <= 6.0  # [initial_delay, previous_delay * 3]

    def test_max_delay_caps_result(self):
        assert (
            compute_backoff_delay(
                BackoffStrategy.FIXED,
                attempt=1,
                initial_delay=10.0,
                max_delay=5.0,
            )
            == 5.0
        )

    def test_exponential_capped_by_max_delay(self):
        # attempt 5: 1.0 * 2^4 = 16.0, capped to 10.0
        assert (
            compute_backoff_delay(
                BackoffStrategy.EXPONENTIAL,
                attempt=5,
                initial_delay=1.0,
                max_delay=10.0,
                backoff_multiplier=2.0,
            )
            == 10.0
        )


# ---------------------------------------------------------------------------
# Async execution with retries
# ---------------------------------------------------------------------------


class TestRetryExecutorAsyncExecution:
    """Test RetryExecutor.execute with async callables."""

    async def test_success_on_first_attempt(self):
        config = RetryConfig(max_attempts=3, initial_delay=0.0)
        executor = RetryExecutor(config)
        call_count = 0

        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await executor.execute(succeed)
        assert result == "ok"
        assert call_count == 1

    async def test_retries_on_exception_then_succeeds(self):
        config = RetryConfig(max_attempts=3, initial_delay=0.0)
        executor = RetryExecutor(config)
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        result = await executor.execute(fail_then_succeed)
        assert result == "ok"
        assert call_count == 3

    async def test_raises_after_max_attempts_exhausted(self):
        config = RetryConfig(max_attempts=2, initial_delay=0.0)
        executor = RetryExecutor(config)

        async def always_fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await executor.execute(always_fail)

    async def test_no_retry_on_non_matching_exception(self):
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.0,
            retry_on_exceptions=[ValueError],
        )
        executor = RetryExecutor(config)
        call_count = 0

        async def raise_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            await executor.execute(raise_type_error)
        # Should not retry — TypeError is not in retry_on_exceptions
        assert call_count == 1

    async def test_retries_only_matching_exceptions(self):
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.0,
            retry_on_exceptions=[ValueError],
        )
        executor = RetryExecutor(config)
        call_count = 0

        async def fail_with_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("retry me")

        with pytest.raises(ValueError, match="retry me"):
            await executor.execute(fail_with_value_error)
        assert call_count == 3

    async def test_retries_when_exception_predicate_true(self):
        """A value predicate retries on the attribute-encoded condition it
        matches (HTTP 5xx), letting a flaky call recover.
        """
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.0,
            backoff_strategy=BackoffStrategy.FIXED,
            retry_on_exception=lambda e: getattr(e, "code", 0) >= 500,
        )
        executor = RetryExecutor(config)
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _FakeHTTPError(503)
            return "ok"

        assert await executor.execute(fail_then_succeed) == "ok"
        assert call_count == 3

    async def test_raises_immediately_when_exception_predicate_false(self):
        """The value analogue of a non-matching type: a predicate returning
        False re-raises on the first attempt, no retry.
        """
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.0,
            retry_on_exception=lambda e: getattr(e, "code", 0) >= 500,
        )
        executor = RetryExecutor(config)
        call_count = 0

        async def raise_404():
            nonlocal call_count
            call_count += 1
            raise _FakeHTTPError(404)

        with pytest.raises(_FakeHTTPError, match="HTTP 404"):
            await executor.execute(raise_404)
        assert call_count == 1

    async def test_exhaustion_with_predicate_fires_on_failure(self):
        """A predicate-retryable exception flows through the full exhaustion
        path: re-raised after the bound, on_failure fired once.
        """
        failures: list[Exception] = []
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.0,
            backoff_strategy=BackoffStrategy.FIXED,
            retry_on_exception=lambda e: True,
            on_failure=failures.append,
        )
        executor = RetryExecutor(config)
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise _FakeHTTPError(503)

        with pytest.raises(_FakeHTTPError, match="HTTP 503"):
            await executor.execute(always_fail)
        assert call_count == 3
        assert len(failures) == 1

    async def test_raising_predicate_propagates(self):
        """A predicate that itself raises is a caller bug — its exception
        propagates rather than being wrapped or swallowed.
        """

        def boom(_e: Exception) -> bool:
            raise KeyError("predicate blew up")

        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.0,
            retry_on_exception=boom,
        )
        executor = RetryExecutor(config)

        async def raise_value_error():
            raise ValueError("original")

        with pytest.raises(KeyError, match="predicate blew up"):
            await executor.execute(raise_value_error)

    async def test_awaits_async_callable_object_result(self):
        """execute() detects async-ness at the result level, not just via
        iscoroutinefunction() on the callable.

        An object with ``async def __call__`` is not a coroutine *function*,
        so the old function-level check returned its (un-awaited) coroutine as
        the result. execute() must await the awaitable and return the value.
        """
        config = RetryConfig(max_attempts=2, initial_delay=0.0)
        executor = RetryExecutor(config)
        obj = _AsyncCallableObject("done")

        result = await executor.execute(obj)

        assert result == "done"
        assert not inspect.iscoroutine(result)
        assert obj.calls == 1

    async def test_retries_async_callable_object_on_exception(self):
        """A retryable async callable object is awaited each attempt, so the
        exception raised inside its coroutine body is caught and retried.
        """
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.0,
            backoff_strategy=BackoffStrategy.FIXED,
        )
        executor = RetryExecutor(config)

        class _FlakyAsyncObject:
            def __init__(self) -> None:
                self.calls = 0

            async def __call__(self) -> str:
                self.calls += 1
                if self.calls < 3:
                    raise ValueError("not yet")
                return "ok"

        obj = _FlakyAsyncObject()
        assert await executor.execute(obj) == "ok"
        assert obj.calls == 3


# ---------------------------------------------------------------------------
# Sync callable execution
# ---------------------------------------------------------------------------


class TestRetryExecutorSyncExecution:
    """Test RetryExecutor.execute with sync callables."""

    async def test_sync_callable_succeeds(self):
        config = RetryConfig(max_attempts=3, initial_delay=0.0)
        executor = RetryExecutor(config)

        def sync_func():
            return 42

        result = await executor.execute(sync_func)
        assert result == 42

    async def test_sync_callable_retries_on_failure(self):
        config = RetryConfig(max_attempts=3, initial_delay=0.0)
        executor = RetryExecutor(config)
        call_count = 0

        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("not yet")
            return "done"

        result = await executor.execute(fail_then_succeed)
        assert result == "done"
        assert call_count == 2


# ---------------------------------------------------------------------------
# Result-based retry
# ---------------------------------------------------------------------------


class TestResultBasedRetry:
    """Test retry_on_result configuration."""

    async def test_retries_when_result_predicate_returns_true(self):
        config = RetryConfig(
            max_attempts=4,
            initial_delay=0.0,
            retry_on_result=lambda r: r is None,
        )
        executor = RetryExecutor(config)
        call_count = 0

        async def return_none_then_value():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None
            return "got it"

        result = await executor.execute(return_none_then_value)
        assert result == "got it"
        assert call_count == 3

    async def test_returns_last_result_if_max_attempts_exhausted(self):
        config = RetryConfig(
            max_attempts=2,
            initial_delay=0.0,
            retry_on_result=lambda r: r == "bad",
        )
        executor = RetryExecutor(config)

        async def always_bad():
            return "bad"

        # When max attempts exhausted on result-based retry, returns the result
        result = await executor.execute(always_bad)
        assert result == "bad"


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class TestRetryHooks:
    """Test on_retry and on_failure hooks."""

    async def test_on_retry_hook_called(self):
        retry_calls = []

        def on_retry(attempt, exc):
            retry_calls.append((attempt, str(exc)))

        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.0,
            on_retry=on_retry,
        )
        executor = RetryExecutor(config)
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"fail {call_count}")
            return "ok"

        result = await executor.execute(fail_then_succeed)
        assert result == "ok"
        assert len(retry_calls) == 2
        assert retry_calls[0] == (1, "fail 1")
        assert retry_calls[1] == (2, "fail 2")

    async def test_on_failure_hook_called(self):
        failure_calls = []

        def on_failure(exc):
            failure_calls.append(str(exc))

        config = RetryConfig(
            max_attempts=2,
            initial_delay=0.0,
            on_failure=on_failure,
        )
        executor = RetryExecutor(config)

        async def always_fail():
            raise RuntimeError("final boom")

        with pytest.raises(RuntimeError, match="final boom"):
            await executor.execute(always_fail)

        assert len(failure_calls) == 1
        assert failure_calls[0] == "final boom"

    async def test_on_failure_not_called_on_success(self):
        failure_calls = []

        def on_failure(exc):
            failure_calls.append(str(exc))

        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.0,
            on_failure=on_failure,
        )
        executor = RetryExecutor(config)

        async def succeed():
            return "ok"

        await executor.execute(succeed)
        assert len(failure_calls) == 0


# ---------------------------------------------------------------------------
# Synchronous execution entry point (execute_sync)
# ---------------------------------------------------------------------------


class TestExecuteSync:
    """RetryExecutor.execute_sync — the synchronous twin of execute().

    Behavioral parity with execute() (bounded retry, exception filtering,
    result-based retry, hooks) plus the coroutine-function guard. All tests are
    synchronous — execute_sync has no event loop.
    """

    def test_success_on_first_attempt(self):
        config = RetryConfig(max_attempts=3, initial_delay=0.0)
        executor = RetryExecutor(config)
        call_count = 0

        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert executor.execute_sync(succeed) == "ok"
        assert call_count == 1

    def test_retries_on_exception_then_succeeds(self):
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.0,
            backoff_strategy=BackoffStrategy.FIXED,
        )
        executor = RetryExecutor(config)
        call_count = 0

        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        assert executor.execute_sync(fail_then_succeed) == "ok"
        assert call_count == 3

    def test_raises_after_max_attempts_exhausted(self):
        config = RetryConfig(max_attempts=2, initial_delay=0.0)
        executor = RetryExecutor(config)

        def always_fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            executor.execute_sync(always_fail)

    def test_no_retry_on_non_matching_exception(self):
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.0,
            retry_on_exceptions=[ValueError],
        )
        executor = RetryExecutor(config)
        call_count = 0

        def raise_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            executor.execute_sync(raise_type_error)
        assert call_count == 1

    def test_retries_only_matching_exceptions(self):
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.0,
            retry_on_exceptions=[ValueError],
        )
        executor = RetryExecutor(config)
        call_count = 0

        def fail_with_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("retry me")

        with pytest.raises(ValueError, match="retry me"):
            executor.execute_sync(fail_with_value_error)
        assert call_count == 3

    def test_retries_when_exception_predicate_true(self):
        """The value predicate reaches execute_sync through the same gate as
        execute — a matching (5xx) error retries then succeeds.
        """
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.0,
            backoff_strategy=BackoffStrategy.FIXED,
            retry_on_exception=lambda e: getattr(e, "code", 0) >= 500,
        )
        executor = RetryExecutor(config)
        call_count = 0

        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _FakeHTTPError(503)
            return "ok"

        assert executor.execute_sync(fail_then_succeed) == "ok"
        assert call_count == 3

    def test_raises_immediately_when_exception_predicate_false(self):
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.0,
            retry_on_exception=lambda e: getattr(e, "code", 0) >= 500,
        )
        executor = RetryExecutor(config)
        call_count = 0

        def raise_404():
            nonlocal call_count
            call_count += 1
            raise _FakeHTTPError(404)

        with pytest.raises(_FakeHTTPError, match="HTTP 404"):
            executor.execute_sync(raise_404)
        assert call_count == 1

    def test_exhaustion_with_predicate_fires_on_failure(self):
        failures: list[Exception] = []
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.0,
            backoff_strategy=BackoffStrategy.FIXED,
            retry_on_exception=lambda e: True,
            on_failure=failures.append,
        )
        executor = RetryExecutor(config)
        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise _FakeHTTPError(503)

        with pytest.raises(_FakeHTTPError, match="HTTP 503"):
            executor.execute_sync(always_fail)
        assert call_count == 3
        assert len(failures) == 1

    def test_returns_last_result_if_max_attempts_exhausted(self):
        config = RetryConfig(
            max_attempts=2,
            initial_delay=0.0,
            retry_on_result=lambda r: r == "bad",
        )
        executor = RetryExecutor(config)

        def always_bad():
            return "bad"

        # Result-based retry at the bound returns the (unsatisfactory) result.
        assert executor.execute_sync(always_bad) == "bad"

    def test_retries_when_result_predicate_returns_true(self):
        config = RetryConfig(
            max_attempts=4,
            initial_delay=0.0,
            retry_on_result=lambda r: r is None,
        )
        executor = RetryExecutor(config)
        call_count = 0

        def return_none_then_value():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None
            return "got it"

        assert executor.execute_sync(return_none_then_value) == "got it"
        assert call_count == 3

    def test_on_retry_and_on_failure_hooks_fire(self):
        retry_calls = []
        failure_calls = []
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.0,
            on_retry=lambda attempt, exc: retry_calls.append((attempt, str(exc))),
            on_failure=lambda exc: failure_calls.append(str(exc)),
        )
        executor = RetryExecutor(config)

        def always_fail():
            raise RuntimeError("final boom")

        with pytest.raises(RuntimeError, match="final boom"):
            executor.execute_sync(always_fail)

        # on_retry fires for attempts 1 and 2; on_failure once at exhaustion.
        assert retry_calls == [(1, "final boom"), (2, "final boom")]
        assert failure_calls == ["final boom"]

    def test_coroutine_function_raises_type_error(self):
        config = RetryConfig(max_attempts=3, initial_delay=0.0)
        executor = RetryExecutor(config)
        call_count = 0

        async def async_func():
            nonlocal call_count
            call_count += 1
            return "never"

        with pytest.raises(TypeError, match="synchronous callable"):
            executor.execute_sync(async_func)
        # The guard fires before invoking func — no un-awaited coroutine created.
        assert call_count == 0

    def test_async_callable_object_raises_type_error(self):
        """A callable object with ``async def __call__`` is not a coroutine
        *function*, so it slips past the up-front iscoroutinefunction() reject.
        The result-level guard must still reject it (rather than hand back an
        un-awaited coroutine) once the awaitable result is observed.
        """
        config = RetryConfig(max_attempts=3, initial_delay=0.0)
        executor = RetryExecutor(config)
        obj = _AsyncCallableObject()

        with pytest.raises(TypeError, match="awaitable|synchronous callable"):
            executor.execute_sync(obj)

    def test_sync_callable_returning_coroutine_raises_type_error(self):
        """A plain sync callable whose *return value* is a coroutine is also
        rejected at the result level — the returned coroutine is never handed
        back un-awaited.
        """
        config = RetryConfig(max_attempts=3, initial_delay=0.0)
        executor = RetryExecutor(config)

        async def _inner() -> str:
            return "never"

        def returns_coroutine() -> object:
            return _inner()  # a coroutine object, not awaited

        with pytest.raises(TypeError, match="awaitable|synchronous callable"):
            executor.execute_sync(returns_coroutine)

    def test_sleeps_between_attempts(self, monkeypatch):
        """A positive initial_delay sleeps once per retry, via time.sleep."""
        sleeps: list[float] = []
        monkeypatch.setattr(
            "dataknobs_common.retry.time.sleep", lambda d: sleeps.append(d)
        )
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            backoff_strategy=BackoffStrategy.FIXED,
        )
        executor = RetryExecutor(config)

        def always_fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            executor.execute_sync(always_fail)
        # Two retries between three attempts, each sleeping the FIXED delay.
        assert sleeps == [0.5, 0.5]


# ---------------------------------------------------------------------------
# RetryConfig validation
# ---------------------------------------------------------------------------


class TestRetryConfigValidation:
    """RetryConfig self-validates max_attempts at construction.

    A non-positive bound would run zero attempts and fall through the retry
    loop to an obscure RuntimeError. The config rejects it up front so every
    consumer — direct construction, ``from_dict`` loading, and downstream
    helpers like ``allocate`` — fails loud on the misconfiguration.
    """

    def test_zero_max_attempts_rejected(self):
        with pytest.raises(ValueError, match="max_attempts"):
            RetryConfig(max_attempts=0)

    def test_negative_max_attempts_rejected(self):
        with pytest.raises(ValueError, match="max_attempts"):
            RetryConfig(max_attempts=-3)

    def test_from_dict_rejects_nonpositive_max_attempts(self):
        with pytest.raises(ValueError, match="max_attempts"):
            RetryConfig.from_dict({"max_attempts": 0})

    def test_one_attempt_is_allowed(self):
        """The boundary value (a single attempt, no retries) is valid."""
        config = RetryConfig(max_attempts=1)
        assert config.max_attempts == 1

    def test_executor_cannot_be_built_with_nonpositive_bound(self):
        """With the config guarding max_attempts, a RetryExecutor can never be
        built around a zero bound and fall through its loop to the
        type-checker-only RuntimeError — the config is rejected first.
        """
        with pytest.raises(ValueError, match="max_attempts"):
            RetryExecutor(RetryConfig(max_attempts=0))

    def test_predicate_and_type_filter_mutually_exclusive(self):
        """The value predicate is the general form of the type filter, so
        setting both has no unambiguous meaning and is rejected at construction.
        """
        with pytest.raises(ValueError, match="mutually exclusive"):
            RetryConfig(
                retry_on_exception=lambda e: True,
                retry_on_exceptions=[ValueError],
            )

    def test_from_dict_rejects_both_predicate_and_type_filter(self):
        """The mutual-exclusion invariant holds on the load path too, like the
        max_attempts guard.
        """
        with pytest.raises(ValueError, match="mutually exclusive"):
            RetryConfig.from_dict(
                {
                    "retry_on_exception": lambda e: True,
                    "retry_on_exceptions": [ValueError],
                }
            )

    def test_predicate_alone_is_allowed(self):
        config = RetryConfig(retry_on_exception=lambda e: True)
        assert config.retry_on_exception is not None
        assert config.retry_on_exceptions is None

    def test_type_filter_alone_is_allowed(self):
        config = RetryConfig(retry_on_exceptions=[ValueError])
        assert config.retry_on_exceptions == [ValueError]
        assert config.retry_on_exception is None


# ---------------------------------------------------------------------------
# RetryConfig defaults
# ---------------------------------------------------------------------------


class TestRetryConfigDefaults:
    """Test RetryConfig default values."""

    def test_defaults(self):
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert config.backoff_multiplier == 2.0
        assert config.jitter_range == 0.1
        assert config.retry_on_exceptions is None
        assert config.retry_on_exception is None
        assert config.retry_on_result is None
        assert config.on_retry is None
        assert config.on_failure is None


# ---------------------------------------------------------------------------
# BackoffStrategy enum values
# ---------------------------------------------------------------------------


class TestBackoffStrategyEnum:
    """Test BackoffStrategy enum members and values."""

    def test_all_strategies_exist(self):
        assert BackoffStrategy.FIXED.value == "fixed"
        assert BackoffStrategy.LINEAR.value == "linear"
        assert BackoffStrategy.EXPONENTIAL.value == "exponential"
        assert BackoffStrategy.JITTER.value == "jitter"
        assert BackoffStrategy.DECORRELATED.value == "decorrelated"

    def test_strategy_count(self):
        assert len(BackoffStrategy) == 5


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------


class TestPackageImport:
    """Test that retry classes are importable from the package root."""

    def test_import_from_dataknobs_common(self):
        from dataknobs_common import BackoffStrategy as BS
        from dataknobs_common import RetryConfig as RC
        from dataknobs_common import RetryExecutor as RE

        assert BS is BackoffStrategy
        assert RC is RetryConfig
        assert RE is RetryExecutor


# ---------------------------------------------------------------------------
# StructuredConfig migration
# ---------------------------------------------------------------------------


class TestRetryConfigStructured:
    """RetryConfig is a frozen StructuredConfig (dict-loadable, immutable)."""

    def test_is_structured_config(self):
        assert issubclass(RetryConfig, StructuredConfig)

    def test_construction_parity(self):
        """from_dict produces a value equal to direct construction."""
        from_dict = RetryConfig.from_dict({"max_attempts": 5, "initial_delay": 0.5})
        direct = RetryConfig(max_attempts=5, initial_delay=0.5)
        assert from_dict == direct

    def test_roundtrip_default(self):
        assert_structured_config_roundtrip(RetryConfig())

    def test_roundtrip_with_exception_type(self):
        """Callable / type fields round-trip by identity (deepcopy-atomic)."""
        assert_structured_config_roundtrip(
            RetryConfig(retry_on_exceptions=[ValueError])
        )

    def test_roundtrip_with_exception_predicate(self):
        """The value predicate round-trips by identity like the other callable
        fields (retry_on_result, hooks) — an in-process dict, not JSON.
        """
        predicate = lambda e: getattr(e, "code", 0) >= 500  # noqa: E731
        config = RetryConfig(retry_on_exception=predicate)
        assert_structured_config_roundtrip(config)
        assert config.from_dict(config.to_dict()).retry_on_exception is predicate

    def test_backoff_strategy_enum_roundtrips(self):
        config = RetryConfig(backoff_strategy=BackoffStrategy.EXPONENTIAL)
        recovered = RetryConfig.from_dict(config.to_dict())
        assert recovered.backoff_strategy is BackoffStrategy.EXPONENTIAL

    def test_frozen(self):
        config = RetryConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.max_attempts = 99  # type: ignore[misc]
