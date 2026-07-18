"""Retry utilities for resilient operation execution.

This module provides general-purpose retry primitives with configurable
backoff strategies. These are standalone utilities with no FSM dependency,
suitable for any code that needs retry logic.

Example:
    ```python
    from dataknobs_common.retry import RetryExecutor, RetryConfig, BackoffStrategy

    config = RetryConfig(
        max_attempts=5,
        initial_delay=0.5,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
    )
    executor = RetryExecutor(config)
    result = await executor.execute(my_flaky_function, arg1, arg2)
    ```
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from dataknobs_common.structured_config import StructuredConfig

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retries."""

    FIXED = "fixed"
    """Fixed delay between retries."""

    LINEAR = "linear"
    """Delay increases linearly with each attempt."""

    EXPONENTIAL = "exponential"
    """Delay doubles (or multiplies by backoff_multiplier) with each attempt."""

    JITTER = "jitter"
    """Exponential backoff with random jitter applied."""

    DECORRELATED = "decorrelated"
    """Decorrelated jitter: each delay is random between initial_delay and 3x previous delay."""


@dataclass(frozen=True)
class RetryConfig(StructuredConfig):
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of execution attempts (including the first).
        initial_delay: Base delay in seconds before the first retry.
        max_delay: Upper bound on delay in seconds.
        backoff_strategy: Algorithm for computing delay between retries.
        backoff_multiplier: Multiplier for exponential and jitter strategies.
        jitter_range: Fractional jitter range for the JITTER strategy (e.g. 0.1 = +/-10%).
        retry_on_exceptions: If set, only retry when the exception is an instance of one of
            these types. Other exceptions propagate immediately.
        retry_on_result: If set, called with the result value. Return True to trigger a retry
            (e.g. to retry on empty or sentinel results).
        on_retry: Hook called before each retry sleep with (attempt_number, exception).
        on_failure: Hook called when all attempts are exhausted with the final exception.
    """

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1

    retry_on_exceptions: list[type] | None = None
    retry_on_result: Callable[[Any], bool] | None = None

    on_retry: Callable[[int, Exception], None] | None = None
    on_failure: Callable[[Exception], None] | None = None


def compute_backoff_delay(
    strategy: BackoffStrategy,
    *,
    attempt: int,
    initial_delay: float,
    max_delay: float,
    backoff_multiplier: float = 2.0,
    jitter_range: float = 0.1,
    previous_delay: float | None = None,
) -> float:
    """Compute a back-off delay for a given strategy and attempt.

    Pure function shared by :class:`RetryExecutor` (bounded "give up after
    N" retries) and the internal event-bus supervised-loop helper
    (unbounded "never give up" listeners). The delay math lives here once
    so neither caller re-implements it.

    Args:
        strategy: The backoff algorithm to apply.
        attempt: The attempt number (1-based) that just failed.
        initial_delay: Base delay in seconds.
        max_delay: Upper bound on the returned delay in seconds.
        backoff_multiplier: Multiplier for EXPONENTIAL and JITTER strategies.
        jitter_range: Fractional jitter range for JITTER (e.g. 0.1 = +/-10%).
        previous_delay: The delay used before the previous attempt
            (only consulted by DECORRELATED).

    Returns:
        Delay in seconds, capped at ``max_delay``.
    """
    if strategy == BackoffStrategy.FIXED:
        delay = initial_delay

    elif strategy == BackoffStrategy.LINEAR:
        delay = initial_delay * attempt

    elif strategy == BackoffStrategy.EXPONENTIAL:
        delay = initial_delay * (backoff_multiplier ** (attempt - 1))

    elif strategy == BackoffStrategy.JITTER:
        base_delay = initial_delay * (backoff_multiplier ** (attempt - 1))
        jitter = random.uniform(-jitter_range, jitter_range)
        delay = base_delay * (1 + jitter)

    elif strategy == BackoffStrategy.DECORRELATED:
        if previous_delay is None:
            delay = initial_delay
        else:
            delay = random.uniform(initial_delay, previous_delay * 3)

    else:
        delay = initial_delay

    return min(delay, max_delay)


class RetryExecutor:
    """Executes a callable with retry logic and configurable backoff.

    :meth:`execute` is the async entry point (awaits async callables, invokes
    sync callables directly) and yields cooperatively between attempts.
    :meth:`execute_sync` is its synchronous twin for callers with no event
    loop; it blocks the calling thread between attempts and rejects coroutine
    callables. Both share one retry policy core, so backoff,
    ``retry_on_exceptions``, ``retry_on_result``, and the hooks behave
    identically across the two entry points.

    Example:
        ```python
        config = RetryConfig(max_attempts=3, backoff_strategy=BackoffStrategy.FIXED)
        executor = RetryExecutor(config)

        # Async usage
        result = await executor.execute(fetch_data, url)

        # Sync callable also works (called from async context)
        result = await executor.execute(parse_json, raw_text)

        # Synchronous entry point (no event loop)
        result = executor.execute_sync(parse_json, raw_text)
        ```
    """

    def __init__(self, config: RetryConfig) -> None:
        self.config = config

    def _calculate_delay(
        self, attempt: int, previous_delay: float | None = None
    ) -> float:
        """Calculate delay for the next retry attempt.

        Args:
            attempt: The attempt number (1-based) that just failed.
            previous_delay: The delay used before the previous attempt (for DECORRELATED).

        Returns:
            Delay in seconds, capped at config.max_delay.
        """
        cfg = self.config
        return compute_backoff_delay(
            cfg.backoff_strategy,
            attempt=attempt,
            initial_delay=cfg.initial_delay,
            max_delay=cfg.max_delay,
            backoff_multiplier=cfg.backoff_multiplier,
            jitter_range=cfg.jitter_range,
            previous_delay=previous_delay,
        )

    def _should_retry_on_result(self, attempt: int, result: Any) -> bool:
        """Whether a result value should trigger a retry.

        True only when a result predicate is configured, it matches the
        result, and the attempt bound has not been reached. At the bound this
        returns False so the caller returns the (unsatisfactory) result,
        preserving the "returns the last result if attempts are exhausted"
        contract.
        """
        return bool(
            self.config.retry_on_result
            and self.config.retry_on_result(result)
            and attempt < self.config.max_attempts
        )

    def _delay_before_retry_on_result(
        self, attempt: int, previous_delay: float | None
    ) -> float:
        """Compute and log the delay before a result-triggered retry.

        Result-based retries fire no ``on_retry`` hook (that hook is
        exception-scoped), matching the established behavior.
        """
        delay = self._calculate_delay(attempt, previous_delay)
        logger.debug(
            "Retry on result (attempt %d/%d), delay=%.2fs",
            attempt, self.config.max_attempts, delay,
        )
        return delay

    def _delay_before_retry_on_exception(
        self, attempt: int, e: Exception, previous_delay: float | None
    ) -> float:
        """Decide the fate of a failed attempt.

        Re-raises ``e`` immediately when it is not a retryable type, or when
        the attempt bound is reached (firing ``on_failure`` first). Otherwise
        fires ``on_retry`` and returns the delay before the next attempt.
        """
        if self.config.retry_on_exceptions and not any(
            isinstance(e, exc_type) for exc_type in self.config.retry_on_exceptions
        ):
            raise e
        if attempt >= self.config.max_attempts:
            if self.config.on_failure:
                self.config.on_failure(e)
            raise e
        delay = self._calculate_delay(attempt, previous_delay)
        if self.config.on_retry:
            self.config.on_retry(attempt, e)
        logger.debug(
            "Retry after exception (attempt %d/%d), delay=%.2fs: %s",
            attempt, self.config.max_attempts, delay, e,
        )
        return delay

    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a callable with retry logic.

        Args:
            func: The callable to execute. May be sync or async.
            *args: Positional arguments forwarded to func.
            **kwargs: Keyword arguments forwarded to func.

        Returns:
            The return value of func on a successful attempt.

        Raises:
            Exception: The exception from the final failed attempt, or any
                non-retryable exception immediately.
        """
        previous_delay: float | None = None
        is_coro = asyncio.iscoroutinefunction(func)

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await func(*args, **kwargs) if is_coro else func(*args, **kwargs)
                if not self._should_retry_on_result(attempt, result):
                    return result
                previous_delay = self._delay_before_retry_on_result(
                    attempt, previous_delay
                )
            except Exception as e:
                previous_delay = self._delay_before_retry_on_exception(
                    attempt, e, previous_delay
                )
            await asyncio.sleep(previous_delay)

        # The final attempt always returns (a good result, or a bad result at
        # the bound) or raises (an exhausted exception via the helper), so the
        # loop never falls through. This satisfies the type checker only.
        raise RuntimeError("retry loop exited without return or raise")

    def execute_sync(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a synchronous callable with retry logic, blocking the thread.

        The synchronous twin of :meth:`execute`: same bounded-retry, backoff,
        ``retry_on_exceptions``, ``retry_on_result``, and hook policy, but it
        sleeps the calling thread between attempts instead of awaiting. Use it
        from code that has no event loop.

        Args:
            func: A synchronous callable to execute.
            *args: Positional arguments forwarded to func.
            **kwargs: Keyword arguments forwarded to func.

        Returns:
            The return value of func on a successful attempt.

        Raises:
            TypeError: If ``func`` is a coroutine function — it would create an
                un-awaited coroutine that never runs. Use :meth:`execute`.
            Exception: The exception from the final failed attempt, or any
                non-retryable exception immediately.
        """
        if asyncio.iscoroutinefunction(func):
            raise TypeError(
                "execute_sync requires a synchronous callable; "
                "use execute() for coroutine functions"
            )
        previous_delay: float | None = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                if not self._should_retry_on_result(attempt, result):
                    return result
                previous_delay = self._delay_before_retry_on_result(
                    attempt, previous_delay
                )
            except Exception as e:
                previous_delay = self._delay_before_retry_on_exception(
                    attempt, e, previous_delay
                )
            time.sleep(previous_delay)

        # Unreachable — see execute(); satisfies the type checker only.
        raise RuntimeError("retry loop exited without return or raise")
