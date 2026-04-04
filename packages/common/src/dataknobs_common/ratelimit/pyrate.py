"""Rate limiter backed by pyrate-limiter.

This module wraps the ``pyrate-limiter`` library (MIT, v4.x) to provide
distributed rate limiting with pluggable bucket backends.

Requires the ``pyrate-limiter`` package::

    pip install pyrate-limiter

For Redis or PostgreSQL bucket backends, install the corresponding extras::

    pip install pyrate-limiter[redis]
    pip install pyrate-limiter[postgres]

Example:
    ```python
    from dataknobs_common.ratelimit import create_rate_limiter

    limiter = create_rate_limiter({
        "backend": "pyrate",
        "bucket": "redis",
        "default_rates": [{"limit": 100, "interval": 60}],
        "redis": {"url": "redis://localhost:6379"},
    })

    if await limiter.try_acquire("api_read"):
        await make_api_call()
    ```
"""

from __future__ import annotations

import asyncio
import logging
import time
from inspect import isawaitable
from typing import Any

from dataknobs_common.exceptions import TimeoutError

from .types import RateLimit, RateLimiterConfig, RateLimitStatus

logger = logging.getLogger(__name__)

try:
    from pyrate_limiter import (
        AbstractBucket,
        BucketFactory,
        Duration,
        InMemoryBucket,
        Limiter,
        Rate,
        RateItem,
    )

    _HAS_PYRATE = True
except ImportError:
    _HAS_PYRATE = False
    # Sentinel so _CategoryBucketFactory can be defined at module level
    # without crashing when pyrate-limiter is not installed. Actual usage
    # is gated by the _HAS_PYRATE check in PyrateRateLimiter.__init__().
    BucketFactory = object  # type: ignore[assignment,misc]


def _to_pyrate_rates(rates: list[RateLimit]) -> list[Any]:
    """Convert RateLimit objects to pyrate-limiter Rate objects.

    Args:
        rates: List of RateLimit objects.

    Returns:
        List of pyrate-limiter Rate objects, sorted by interval.
    """
    pyrate_rates = []
    for r in sorted(rates, key=lambda x: x.interval):
        # Duration expects milliseconds
        interval_ms = int(r.interval * Duration.SECOND)
        pyrate_rates.append(Rate(r.limit, interval_ms))
    return pyrate_rates


def _create_bucket(rates: list[Any], config: dict[str, Any]) -> Any:
    """Create a pyrate-limiter bucket from config.

    Args:
        rates: pyrate-limiter Rate objects for the bucket.
        config: Full configuration dict with bucket backend settings.

    Returns:
        An AbstractBucket instance.

    Raises:
        ValueError: If the bucket backend is not recognized.
        ImportError: If the required package for the bucket is not installed.
    """
    bucket_type = config.get("bucket", "memory")

    if bucket_type == "memory":
        return InMemoryBucket(rates)

    elif bucket_type == "sqlite":
        try:
            from pyrate_limiter import SQLiteBucket
        except ImportError as e:
            raise ImportError(
                "SQLiteBucket requires pyrate-limiter with sqlite support"
            ) from e
        sqlite_config = config.get("sqlite", {})
        db_path = sqlite_config.get("db_path", "rate_limits.db")
        table = sqlite_config.get("table", "rate_limits")
        return SQLiteBucket.init_from_file(
            rates, db_path=db_path, table=table
        )

    elif bucket_type == "redis":
        try:
            from pyrate_limiter import RedisBucket
        except ImportError as e:
            raise ImportError(
                "RedisBucket requires: pip install pyrate-limiter[redis]"
            ) from e
        redis_config = config.get("redis", {})
        try:
            from redis import Redis
        except ImportError as e:
            raise ImportError(
                "Redis bucket requires the 'redis' package: pip install redis"
            ) from e
        # Using sync Redis client so RedisBucket.init() returns synchronously.
        # SSL is inferred from the URL scheme: use rediss:// for TLS.
        conn = Redis.from_url(
            redis_config.get("url", "redis://localhost:6379"),
        )
        return RedisBucket.init(rates, conn, "rate_limit")

    elif bucket_type == "postgres":
        try:
            from pyrate_limiter import PostgresBucket
        except ImportError as e:
            raise ImportError(
                "PostgresBucket requires: pip install pyrate-limiter[postgres]"
            ) from e
        pg_config = config.get("postgres", {})
        pool = pg_config.get("pool")
        if pool is None:
            raise ValueError(
                "PostgreSQL bucket requires 'postgres.pool' (a psycopg pool instance)"
            )
        table = pg_config.get("table", "rate_limits")
        return PostgresBucket(pool, table, rates)

    else:
        raise ValueError(
            f"Unknown pyrate bucket backend: {bucket_type}. "
            f"Available: memory, sqlite, redis, postgres"
        )


class _CategoryBucketFactory(BucketFactory):  # type: ignore[misc]
    """BucketFactory that creates per-category buckets with different rates.

    Each category gets its own bucket with the rates configured for that
    category, falling back to default rates for unknown categories.
    """

    def __init__(
        self,
        parsed_config: RateLimiterConfig,
        raw_config: dict[str, Any],
    ) -> None:
        self._parsed = parsed_config
        self._raw = raw_config
        self._buckets: dict[str, AbstractBucket] = {}  # type: ignore[no-any-unimported]

    def wrap_item(
        self, name: str, weight: int = 1
    ) -> RateItem:  # type: ignore[no-any-unimported]
        """Wrap an item name and weight into a RateItem.

        pyrate-limiter uses millisecond timestamps internally
        (``Duration.SECOND = 1000``). The timestamp must be in the
        same unit as the rate intervals passed to ``Rate()``.

        Args:
            name: Category name.
            weight: Operation weight.

        Returns:
            A RateItem for the limiter.
        """
        now_ms = int(time.monotonic() * 1_000)
        return RateItem(name, now_ms, weight=weight)

    def get(
        self, item: RateItem  # type: ignore[no-any-unimported]
    ) -> AbstractBucket:  # type: ignore[no-any-unimported]
        """Get or create the bucket for a given item's category.

        Args:
            item: The rate item whose name determines the category.

        Returns:
            The bucket for the item's category.
        """
        name = item.name
        if name not in self._buckets:
            rates_config = self._parsed.categories.get(
                name, self._parsed.default_rates
            )
            pyrate_rates = _to_pyrate_rates(rates_config)
            self._buckets[name] = _create_bucket(pyrate_rates, self._raw)
            logger.debug(
                "Created pyrate bucket for category %s with %d rate(s)",
                name,
                len(pyrate_rates),
            )
        return self._buckets[name]


class PyrateRateLimiter:
    """Rate limiter backed by pyrate-limiter.

    Wraps the ``pyrate-limiter`` library to provide distributed rate
    limiting with pluggable bucket backends (in-memory, SQLite, Redis,
    PostgreSQL).

    This implementation requires the ``pyrate-limiter`` package.
    Install it with::

        pip install pyrate-limiter

    For distributed backends::

        pip install pyrate-limiter[redis]
        pip install pyrate-limiter[postgres]

    Limitations:
    - ``get_status()`` returns approximate data because pyrate-limiter
      does not expose detailed bucket state.

    Example:
        ```python
        from dataknobs_common.ratelimit.pyrate import PyrateRateLimiter
        from dataknobs_common.ratelimit import RateLimiterConfig, RateLimit

        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=100, interval=60)],
        )
        limiter = PyrateRateLimiter(config, {"bucket": "memory"})

        if await limiter.try_acquire("api_read"):
            await process_request()

        await limiter.close()
        ```
    """

    def __init__(
        self,
        config: RateLimiterConfig,
        raw_config: dict[str, Any],
    ) -> None:
        """Initialize the pyrate-limiter backed rate limiter.

        Args:
            config: Parsed rate limiter configuration.
            raw_config: Raw configuration dict for bucket backend settings.

        Raises:
            ImportError: If pyrate-limiter is not installed.
        """
        if not _HAS_PYRATE:
            raise ImportError(
                "PyrateRateLimiter requires the 'pyrate-limiter' package. "
                "Install it with: pip install pyrate-limiter"
            )

        self._config = config
        self._raw_config = raw_config
        self._factory = _CategoryBucketFactory(config, raw_config)
        self._limiter = Limiter(self._factory)
        logger.debug(
            "PyrateRateLimiter initialized with bucket backend: %s",
            raw_config.get("bucket", "memory"),
        )

    async def try_acquire(self, name: str = "default", weight: int = 1) -> bool:
        """Attempt to acquire capacity without blocking.

        Args:
            name: Category name.
            weight: Weight of the operation (default 1).

        Returns:
            True if the acquire succeeded, False otherwise.
        """
        result = self._limiter.try_acquire(name, weight=weight, blocking=False)
        if isawaitable(result):
            result = await result
        return bool(result)

    async def acquire(
        self,
        name: str = "default",
        weight: int = 1,
        timeout: float | None = None,
    ) -> None:
        """Acquire capacity, waiting asynchronously until available.

        Polls with ``asyncio.sleep`` so the event loop stays free.
        Does NOT use pyrate-limiter's built-in ``blocking=True`` mode
        because that path calls ``time.sleep()`` which blocks the
        event loop thread.

        Args:
            name: Category name.
            weight: Weight of the operation (default 1).
            timeout: Maximum seconds to wait. ``None`` means wait
                indefinitely.

        Raises:
            TimeoutError: If the timeout is exceeded.
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
            await asyncio.sleep(0.05)

    async def get_status(self, name: str = "default") -> RateLimitStatus:
        """Get approximate status of a rate limiter bucket.

        Note: pyrate-limiter does not expose detailed bucket state,
        so this returns approximate information based on the configured
        rates.

        Args:
            name: Category name.

        Returns:
            Approximate status of the bucket.
        """
        rates = self._config.categories.get(name, self._config.default_rates)
        tightest = min(rates, key=lambda r: r.limit)
        return RateLimitStatus(
            name=name,
            current_count=0,
            limit=tightest.limit,
            remaining=tightest.limit,
            reset_after=0.0,
        )

    async def reset(self, name: str | None = None) -> None:
        """Reset rate limiter state.

        Note: For the pyrate backend, this disposes of the relevant
        bucket(s). New buckets will be created on the next acquire.

        Args:
            name: Category to reset. If ``None``, resets all categories.
        """
        if name is not None:
            self._factory._buckets.pop(name, None)
        else:
            self._factory._buckets.clear()
        logger.debug(
            "PyrateRateLimiter reset: %s",
            name if name else "all",
        )

    async def close(self) -> None:
        """Release resources held by the rate limiter."""
        self._factory._buckets.clear()
        logger.debug("PyrateRateLimiter closed")
