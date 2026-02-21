"""RateLimiter protocol and factory function."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import RateLimit, RateLimiterConfig, RateLimitStatus


@runtime_checkable
class RateLimiter(Protocol):
    """Abstract rate limiter protocol supporting multiple backends.

    The RateLimiter provides both blocking and non-blocking rate limiting
    with per-category rate configuration. Different implementations
    support various deployment scenarios:

    - InMemoryRateLimiter: Single process, no external dependencies
    - PyrateRateLimiter: Wraps pyrate-limiter for distributed backends
      (Redis, PostgreSQL, SQLite)

    All implementations follow this protocol, allowing configuration-driven
    backend selection without code changes.

    Example:
        ```python
        from dataknobs_common.ratelimit import create_rate_limiter

        limiter = create_rate_limiter({
            "default_rates": [{"limit": 100, "interval": 60}],
            "categories": {
                "api_write": {"rates": [{"limit": 10, "interval": 60}]},
            },
        })

        # Non-blocking
        if await limiter.try_acquire("api_write"):
            await make_api_call()

        # Blocking with timeout
        await limiter.acquire("api_read", timeout=5.0)
        ```
    """

    async def acquire(
        self,
        name: str = "default",
        weight: int = 1,
        timeout: float | None = None,
    ) -> None:
        """Acquire capacity, blocking until available.

        Args:
            name: Category name for rate lookup.
            weight: Weight of the operation (default 1).
            timeout: Maximum seconds to wait. ``None`` means wait
                indefinitely.

        Raises:
            TimeoutError: If the timeout is exceeded.
        """
        ...

    async def try_acquire(self, name: str = "default", weight: int = 1) -> bool:
        """Attempt to acquire capacity without blocking.

        Args:
            name: Category name for rate lookup.
            weight: Weight of the operation (default 1).

        Returns:
            True if the acquire succeeded, False otherwise.
        """
        ...

    async def get_status(self, name: str = "default") -> RateLimitStatus:
        """Get the current status of a rate limiter bucket.

        Args:
            name: Category name.

        Returns:
            Current status of the bucket.
        """
        ...

    async def reset(self, name: str | None = None) -> None:
        """Reset rate limiter state.

        Args:
            name: Category to reset. If ``None``, resets all categories.
        """
        ...

    async def close(self) -> None:
        """Release resources held by the rate limiter."""
        ...


def _parse_rates(raw: list[dict[str, Any]]) -> list[RateLimit]:
    """Parse a list of rate dicts into RateLimit objects.

    Args:
        raw: List of dicts with ``limit`` and ``interval`` keys.

    Returns:
        List of RateLimit instances.

    Raises:
        ValueError: If a rate dict is missing required keys.
    """
    rates: list[RateLimit] = []
    for entry in raw:
        if "limit" not in entry or "interval" not in entry:
            raise ValueError(
                f"Each rate must have 'limit' and 'interval' keys, got: {entry}"
            )
        rates.append(RateLimit(limit=entry["limit"], interval=entry["interval"]))
    return rates


def _parse_config(config: dict[str, Any]) -> RateLimiterConfig:
    """Parse a config dict into a RateLimiterConfig.

    Supports ``rates`` as a shorthand alias for ``default_rates``.

    Args:
        config: Configuration dictionary.

    Returns:
        Parsed RateLimiterConfig.

    Raises:
        ValueError: If neither ``rates`` nor ``default_rates`` is provided.
    """
    raw_rates = config.get("default_rates") or config.get("rates")
    if not raw_rates:
        raise ValueError(
            "Rate limiter config must include 'rates' or 'default_rates'"
        )
    default_rates = _parse_rates(raw_rates)

    categories: dict[str, list[RateLimit]] = {}
    raw_categories = config.get("categories", {})
    for cat_name, cat_config in raw_categories.items():
        cat_rates = cat_config.get("rates", [])
        if cat_rates:
            categories[cat_name] = _parse_rates(cat_rates)

    return RateLimiterConfig(default_rates=default_rates, categories=categories)


def create_rate_limiter(config: dict[str, Any]) -> RateLimiter:
    """Create a rate limiter from configuration.

    Factory function that creates the appropriate RateLimiter implementation
    based on the ``backend`` key in the config.

    Args:
        config: Configuration dict. See below for keys.

    Returns:
        RateLimiter instance.

    Raises:
        ValueError: If the backend is not recognized or required config
            is missing.

    Config keys:
        backend: ``"memory"`` (default) or ``"pyrate"``.
        rates: Shorthand for ``default_rates``.
        default_rates: List of ``{"limit": int, "interval": float}`` dicts.
        categories: Dict mapping category names to
            ``{"rates": [{"limit": ..., "interval": ...}]}``.
        bucket: Pyrate bucket backend (``"memory"``, ``"sqlite"``,
            ``"redis"``, ``"postgres"``).
        redis: Redis connection config (for pyrate redis bucket).
        postgres: Postgres config (for pyrate postgres bucket).
        sqlite: SQLite config (for pyrate sqlite bucket).

    Example:
        ```python
        # Simple in-memory limiter
        limiter = create_rate_limiter({
            "rates": [{"limit": 10, "interval": 60}],
        })

        # Per-category with pyrate + Redis
        limiter = create_rate_limiter({
            "backend": "pyrate",
            "bucket": "redis",
            "default_rates": [{"limit": 100, "interval": 60}],
            "categories": {
                "api_write": {"rates": [{"limit": 10, "interval": 60}]},
            },
            "redis": {"url": "redis://localhost:6379"},
        })
        ```
    """
    from .memory import InMemoryRateLimiter

    backend = config.get("backend", "memory")
    parsed = _parse_config(config)

    if backend == "memory":
        return InMemoryRateLimiter(parsed)
    elif backend == "pyrate":
        from .pyrate import PyrateRateLimiter

        return PyrateRateLimiter(parsed, config)
    else:
        raise ValueError(
            f"Unknown rate limiter backend: {backend}. "
            f"Available backends: memory, pyrate"
        )
