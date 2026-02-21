"""In-memory rate limiter implementation."""

from __future__ import annotations

import asyncio
import logging
import time

from dataknobs_common.exceptions import TimeoutError

from .types import RateLimit, RateLimiterConfig, RateLimitStatus

logger = logging.getLogger(__name__)

# Polling interval for blocking acquire (seconds)
_POLL_INTERVAL = 0.05


class InMemoryRateLimiter:
    """Sliding-window rate limiter backed by in-memory data structures.

    This implementation is suitable for:
    - Single-process applications
    - Development and testing
    - Scenarios where rate limiting does not need to cross process boundaries

    Features:
    - Sliding window log algorithm for accurate rate tracking
    - Per-category rate configuration with fallback to default rates
    - Weighted acquire support
    - Non-blocking (``try_acquire``) and blocking (``acquire``) modes
    - Thread-safe using ``asyncio.Lock``

    Limitations:
    - State is not persisted; restarting the process resets all counters
    - Does not work across multiple processes or machines
    - For distributed rate limiting, use ``PyrateRateLimiter`` with a
      Redis or PostgreSQL bucket

    Example:
        ```python
        from dataknobs_common.ratelimit import InMemoryRateLimiter, RateLimiterConfig, RateLimit

        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=10, interval=60)],
            categories={
                "api_write": [RateLimit(limit=5, interval=60)],
            },
        )
        limiter = InMemoryRateLimiter(config)

        # Non-blocking check
        if await limiter.try_acquire("api_write"):
            await make_api_call()

        # Blocking — waits until capacity is available
        await limiter.acquire("api_read")
        await make_api_call()
        ```
    """

    def __init__(self, config: RateLimiterConfig) -> None:
        """Initialize the in-memory rate limiter.

        Args:
            config: Rate limiter configuration with default rates and
                optional per-category overrides.
        """
        self._config = config
        # Per-bucket sliding window: name -> list of (monotonic_time, weight)
        self._buckets: dict[str, list[tuple[float, int]]] = {}
        self._lock = asyncio.Lock()

    def _get_rates(self, name: str) -> list[RateLimit]:
        """Look up the applicable rates for a category.

        Args:
            name: Category name.

        Returns:
            Category-specific rates if configured, otherwise default rates.
        """
        return self._config.categories.get(name, self._config.default_rates)

    def _prune(self, name: str, now: float, rates: list[RateLimit]) -> None:
        """Remove expired entries from a bucket.

        An entry is expired if it falls outside every rate's interval.

        Args:
            name: Bucket name.
            now: Current monotonic time.
            rates: Applicable rate limits.
        """
        if name not in self._buckets:
            return
        max_interval = max(r.interval for r in rates)
        cutoff = now - max_interval
        bucket = self._buckets[name]
        # Find first non-expired entry via linear scan (list is ordered by time)
        i = 0
        while i < len(bucket) and bucket[i][0] <= cutoff:
            i += 1
        if i > 0:
            self._buckets[name] = bucket[i:]

    def _is_allowed(
        self, name: str, weight: int, now: float, rates: list[RateLimit]
    ) -> bool:
        """Check whether an acquire is allowed under all applicable rates.

        Args:
            name: Bucket name.
            weight: Weight of the operation.
            now: Current monotonic time.
            rates: Applicable rate limits.

        Returns:
            True if every rate's window has sufficient remaining capacity.
        """
        bucket = self._buckets.get(name, [])
        for rate in rates:
            window_start = now - rate.interval
            window_weight = sum(
                w for ts, w in bucket if ts > window_start
            )
            if window_weight + weight > rate.limit:
                return False
        return True

    def _record(self, name: str, weight: int, now: float) -> None:
        """Record an acquire in the bucket.

        Args:
            name: Bucket name.
            weight: Weight of the operation.
            now: Current monotonic time.
        """
        if name not in self._buckets:
            self._buckets[name] = []
        self._buckets[name].append((now, weight))

    async def try_acquire(self, name: str = "default", weight: int = 1) -> bool:
        """Attempt to acquire capacity without blocking.

        Args:
            name: Category name. Uses category-specific rates if configured,
                otherwise falls back to default rates.
            weight: Weight of the operation (default 1).

        Returns:
            True if the acquire succeeded, False if the rate limit would
            be exceeded.
        """
        rates = self._get_rates(name)
        async with self._lock:
            now = time.monotonic()
            self._prune(name, now, rates)
            if self._is_allowed(name, weight, now, rates):
                self._record(name, weight, now)
                logger.debug(
                    "Rate limit acquired for %s (weight=%d)", name, weight
                )
                return True
            logger.debug(
                "Rate limit denied for %s (weight=%d)", name, weight
            )
            return False

    async def acquire(
        self,
        name: str = "default",
        weight: int = 1,
        timeout: float | None = None,
    ) -> None:
        """Acquire capacity, blocking until available.

        Polls at 50 ms intervals until capacity is available or the
        timeout is exceeded.

        Args:
            name: Category name.
            weight: Weight of the operation (default 1).
            timeout: Maximum seconds to wait. ``None`` means wait
                indefinitely.

        Raises:
            TimeoutError: If the timeout is exceeded before capacity
                becomes available.
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        while True:
            if await self.try_acquire(name, weight):
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Rate limit acquire timed out for '{name}'",
                    context={"name": name, "weight": weight, "timeout": timeout},
                )
            await asyncio.sleep(_POLL_INTERVAL)

    async def get_status(self, name: str = "default") -> RateLimitStatus:
        """Get the current status of a rate limiter bucket.

        Reports against the tightest (smallest limit) applicable rate.

        Args:
            name: Category name.

        Returns:
            Current status including count, limit, remaining capacity,
            and time until the oldest entry expires.
        """
        rates = self._get_rates(name)
        async with self._lock:
            now = time.monotonic()
            self._prune(name, now, rates)
            bucket = self._buckets.get(name, [])

            # Find the tightest rate (most restrictive remaining capacity)
            tightest_rate = rates[0]
            tightest_remaining = tightest_rate.limit
            for rate in rates:
                window_start = now - rate.interval
                window_weight = sum(w for ts, w in bucket if ts > window_start)
                remaining = rate.limit - window_weight
                if remaining < tightest_remaining:
                    tightest_remaining = remaining
                    tightest_rate = rate

            # Calculate current count and reset_after for the tightest rate
            window_start = now - tightest_rate.interval
            current_count = sum(w for ts, w in bucket if ts > window_start)
            remaining = max(0, tightest_rate.limit - current_count)

            # reset_after: time until the oldest entry in the window expires
            reset_after = 0.0
            window_entries = [(ts, w) for ts, w in bucket if ts > window_start]
            if window_entries:
                oldest_ts = window_entries[0][0]
                reset_after = max(0.0, (oldest_ts + tightest_rate.interval) - now)

            return RateLimitStatus(
                name=name,
                current_count=current_count,
                limit=tightest_rate.limit,
                remaining=remaining,
                reset_after=reset_after,
            )

    async def reset(self, name: str | None = None) -> None:
        """Reset rate limiter state.

        Args:
            name: Category to reset. If ``None``, resets all categories.
        """
        async with self._lock:
            if name is None:
                self._buckets.clear()
                logger.debug("All rate limiter buckets reset")
            else:
                self._buckets.pop(name, None)
                logger.debug("Rate limiter bucket %s reset", name)

    async def close(self) -> None:
        """Release resources.

        For the in-memory implementation this is a no-op, but it
        satisfies the ``RateLimiter`` protocol.
        """
        pass
