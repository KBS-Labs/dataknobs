"""Tests for dataknobs_common.retry module."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from dataknobs_common.retry import BackoffStrategy, RetryConfig, RetryExecutor


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
