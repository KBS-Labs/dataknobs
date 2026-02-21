"""Rate limiter data types and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RateLimit:
    """A single rate limit rule.

    Defines the maximum number of operations (or total weight) allowed
    within a time interval.

    Attributes:
        limit: Maximum operations (or total weight) per interval.
        interval: Window duration in seconds.

    Example:
        ```python
        # 100 requests per minute
        rate = RateLimit(limit=100, interval=60)

        # 10 requests per second
        rate = RateLimit(limit=10, interval=1)
        ```
    """

    limit: int
    interval: float


@dataclass
class RateLimiterConfig:
    """Configuration for a rate limiter.

    Supports per-category rate overrides. When ``acquire()`` is called
    with a category name, rates are looked up in ``categories`` first;
    if the category is not found, ``default_rates`` are used.

    Attributes:
        default_rates: Fallback rates applied when a category has no
            specific configuration.
        categories: Per-category rate overrides mapping category names
            to their rate lists.

    Example:
        ```python
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=100, interval=60)],
            categories={
                "api_read":  [RateLimit(limit=100, interval=6)],
                "api_write": [RateLimit(limit=10, interval=6)],
            },
        )
        ```
    """

    default_rates: list[RateLimit]
    categories: dict[str, list[RateLimit]] = field(default_factory=dict)


@dataclass
class RateLimitStatus:
    """Current status of a rate limiter bucket.

    Attributes:
        name: Category or bucket name.
        current_count: Number of operations (total weight) in the
            current window.
        limit: Maximum allowed from the tightest applicable rate.
        remaining: Operations remaining before the limit is reached.
        reset_after: Seconds until the oldest entry expires and
            capacity is freed.

    Example:
        ```python
        status = await limiter.get_status("api_write")
        if status.remaining < 5:
            logger.warning("Approaching rate limit for %s", status.name)
        ```
    """

    name: str
    current_count: int
    limit: int
    remaining: int
    reset_after: float
