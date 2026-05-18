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
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

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


@dataclass
class RetryConfig:
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

    Supports both sync and async callables. Sync callables are invoked directly;
    async callables are awaited.

    Example:
        ```python
        config = RetryConfig(max_attempts=3, backoff_strategy=BackoffStrategy.FIXED)
        executor = RetryExecutor(config)

        # Async usage
        result = await executor.execute(fetch_data, url)

        # Sync callable also works (called from async context)
        result = await executor.execute(parse_json, raw_text)
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
        last_exception: Exception | None = None
        previous_delay: float | None = None
        is_coro = asyncio.iscoroutinefunction(func)

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await func(*args, **kwargs) if is_coro else func(*args, **kwargs)

                # Check if we should retry based on the result value
                if self.config.retry_on_result and self.config.retry_on_result(result):
                    if attempt < self.config.max_attempts:
                        delay = self._calculate_delay(attempt, previous_delay)
                        previous_delay = delay
                        logger.debug(
                            "Retry on result (attempt %d/%d), delay=%.2fs",
                            attempt, self.config.max_attempts, delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                return result

            except Exception as e:
                last_exception = e

                # If exception filtering is configured, only retry matching types
                if self.config.retry_on_exceptions:
                    if not any(
                        isinstance(e, exc_type)
                        for exc_type in self.config.retry_on_exceptions
                    ):
                        raise

                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt, previous_delay)
                    previous_delay = delay

                    if self.config.on_retry:
                        self.config.on_retry(attempt, e)

                    logger.debug(
                        "Retry after exception (attempt %d/%d), delay=%.2fs: %s",
                        attempt, self.config.max_attempts, delay, e,
                    )
                    await asyncio.sleep(delay)
                else:
                    if self.config.on_failure:
                        self.config.on_failure(e)
                    raise

        # Should not be reached, but satisfies type checker
        raise last_exception  # type: ignore[misc]
