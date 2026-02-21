"""Rate limiting abstraction supporting multiple backends.

This module provides a unified rate limiter interface with configurable
backends for controlling operation throughput. Choose the appropriate
backend based on your deployment scenario:

Backends:
- InMemoryRateLimiter: Single process, no external dependencies
- PyrateRateLimiter: Wraps pyrate-limiter for distributed backends
  (Redis, PostgreSQL, SQLite)

Example:
    ```python
    from dataknobs_common.ratelimit import create_rate_limiter

    # Create a rate limiter from configuration
    limiter = create_rate_limiter({
        "default_rates": [{"limit": 100, "interval": 60}],
        "categories": {
            "api_write": {"rates": [{"limit": 10, "interval": 60}]},
        },
    })

    # Non-blocking acquire
    if await limiter.try_acquire("api_write"):
        await make_api_call()

    # Blocking acquire with timeout
    await limiter.acquire("api_read", timeout=5.0)

    # Check status
    status = await limiter.get_status("api_write")
    print(f"Remaining: {status.remaining}/{status.limit}")

    # Cleanup
    await limiter.close()
    ```

Configuration Examples:
    ```python
    # In-memory (development/testing)
    config = {"rates": [{"limit": 10, "interval": 60}]}

    # Per-category rates
    config = {
        "default_rates": [{"limit": 100, "interval": 60}],
        "categories": {
            "api_read":  {"rates": [{"limit": 100, "interval": 6}]},
            "api_write": {"rates": [{"limit": 10, "interval": 6}]},
        },
    }

    # Pyrate-limiter with Redis (production, distributed)
    config = {
        "backend": "pyrate",
        "bucket": "redis",
        "default_rates": [{"limit": 100, "interval": 60}],
        "redis": {"url": "redis://elasticache:6379", "ssl": True},
    }
    ```
"""

from __future__ import annotations

from .limiter import RateLimiter, create_rate_limiter
from .memory import InMemoryRateLimiter
from .types import RateLimit, RateLimiterConfig, RateLimitStatus

__all__ = [
    # Protocol
    "RateLimiter",
    # Factory
    "create_rate_limiter",
    # Types
    "RateLimit",
    "RateLimiterConfig",
    "RateLimitStatus",
    # Implementations
    "InMemoryRateLimiter",
]

# Note: PyrateRateLimiter is not exported by default to avoid requiring
# pyrate-limiter as a dependency. Import it directly:
#
#   from dataknobs_common.ratelimit.pyrate import PyrateRateLimiter
