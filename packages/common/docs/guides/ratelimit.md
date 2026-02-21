# Rate Limiting

The Rate Limiter provides a unified interface for controlling operation throughput in the DataKnobs ecosystem. It supports per-category rate configuration, weighted operations, and pluggable backends for scaling from development to production.

## Overview

The rate limiter abstraction allows you to:

- **Limit operation throughput** with configurable rates per time window
- **Define per-category rates** for different operation types (e.g., reads vs writes)
- **Choose blocking or non-blocking** acquire modes
- **Switch backends** without changing application code
- **Scale from single-process** (in-memory) **to distributed** (Redis/PostgreSQL via pyrate-limiter)

## Installation

The in-memory rate limiter is included in `dataknobs-common`:

```bash
pip install dataknobs-common
```

For distributed backends (Redis, PostgreSQL, SQLite):

```bash
pip install pyrate-limiter
pip install pyrate-limiter[redis]     # Redis bucket
pip install pyrate-limiter[postgres]  # PostgreSQL bucket
```

## Quick Start

```python
import asyncio
from dataknobs_common.ratelimit import create_rate_limiter

async def main():
    # Create a rate limiter: 10 operations per minute
    limiter = create_rate_limiter({
        "rates": [{"limit": 10, "interval": 60}],
    })

    # Non-blocking: check before proceeding
    if await limiter.try_acquire("api"):
        print("Request permitted")
    else:
        print("Rate limited — try again later")

    # Blocking: wait until capacity is available
    await limiter.acquire("api", timeout=5.0)
    print("Capacity acquired")

    # Check current status
    status = await limiter.get_status("api")
    print(f"Remaining: {status.remaining}/{status.limit}")

    # Cleanup
    await limiter.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Concepts

### Rate Limits

A `RateLimit` defines a maximum operation count within a time interval:

```python
from dataknobs_common.ratelimit import RateLimit

# 100 requests per minute
rate = RateLimit(limit=100, interval=60)

# 10 requests per second
rate = RateLimit(limit=10, interval=1)
```

Multiple rates can be combined — the most restrictive rate always applies:

```python
rates = [
    RateLimit(limit=10, interval=1),      # 10 per second
    RateLimit(limit=100, interval=60),     # 100 per minute
    RateLimit(limit=1000, interval=3600),  # 1000 per hour
]
```

### Categories

A single rate limiter supports different rates for named categories. When `acquire()` or `try_acquire()` is called with a category name, the limiter looks up rates specific to that category, falling back to default rates for unknown categories.

```python
limiter = create_rate_limiter({
    "default_rates": [{"limit": 100, "interval": 60}],
    "categories": {
        "api_read":  {"rates": [{"limit": 100, "interval": 6}]},
        "api_write": {"rates": [{"limit": 10, "interval": 6}]},
    },
})

await limiter.try_acquire("api_read")   # Uses api_read rates
await limiter.try_acquire("api_write")  # Uses api_write rates
await limiter.try_acquire("other")      # Uses default_rates
```

### Blocking vs Non-Blocking

| Method | Behavior | Use When |
|--------|----------|----------|
| `try_acquire()` | Returns `True`/`False` immediately | You want to fail fast or handle the denial yourself |
| `acquire()` | Blocks (async sleep) until capacity is available | You want to wait for capacity |
| `acquire(timeout=...)` | Blocks up to `timeout` seconds, then raises `TimeoutError` | You want to wait but with a deadline |

### Weighted Operations

Operations can have different weights. A weight of 3 consumes three units of capacity:

```python
# Heavy operation consumes 5 units
await limiter.acquire("batch_import", weight=5)

# Light operation consumes 1 unit (default)
await limiter.acquire("single_read")
```

### Status

Check the current state of a rate limiter bucket:

```python
status = await limiter.get_status("api_write")
print(f"Name: {status.name}")
print(f"Current count: {status.current_count}")
print(f"Limit: {status.limit}")
print(f"Remaining: {status.remaining}")
print(f"Reset after: {status.reset_after:.1f}s")
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Category or bucket name |
| `current_count` | `int` | Operations (total weight) in the current window |
| `limit` | `int` | Maximum allowed from the tightest rate |
| `remaining` | `int` | Remaining capacity before the limit |
| `reset_after` | `float` | Seconds until the oldest entry expires |

## Backend Selection

Choose your backend based on deployment needs:

### In-Memory (Development/Testing)

Single-process, no external dependencies. Uses a sliding window log algorithm.

```python
limiter = create_rate_limiter({
    "rates": [{"limit": 100, "interval": 60}],
})
```

**Use when:**

- Unit testing
- Local development
- Single-process applications

### Pyrate-Limiter with In-Memory Bucket

Uses pyrate-limiter's in-memory backend. Functionally similar to the built-in in-memory limiter but uses pyrate-limiter's leaky bucket algorithm.

```python
limiter = create_rate_limiter({
    "backend": "pyrate",
    "bucket": "memory",
    "rates": [{"limit": 100, "interval": 60}],
})
```

### Pyrate-Limiter with Redis (Distributed Production)

Distributed rate limiting using Redis. Suitable for multi-process and multi-machine deployments.

```python
limiter = create_rate_limiter({
    "backend": "pyrate",
    "bucket": "redis",
    "default_rates": [{"limit": 100, "interval": 60}],
    "redis": {
        "url": "redis://localhost:6379",
        "ssl": False,
    },
})
```

For AWS ElastiCache:

```python
limiter = create_rate_limiter({
    "backend": "pyrate",
    "bucket": "redis",
    "default_rates": [{"limit": 100, "interval": 60}],
    "redis": {
        "url": "redis://my-cluster.cache.amazonaws.com:6379",
        "ssl": True,
    },
})
```

**Use when:**

- Multiple application instances
- Need atomic, distributed rate limiting
- High-throughput scenarios

### Pyrate-Limiter with PostgreSQL (Persistent Production)

Persistent rate limiting using PostgreSQL.

```python
limiter = create_rate_limiter({
    "backend": "pyrate",
    "bucket": "postgres",
    "default_rates": [{"limit": 1000, "interval": 3600}],
    "postgres": {
        "pool": existing_pool,  # psycopg pool instance
        "table": "rate_limits",
    },
})
```

**Use when:**

- You already have PostgreSQL
- Need persistent rate limit state across restarts

**Note:** Import the pyrate backend directly for type hints:

```python
from dataknobs_common.ratelimit.pyrate import PyrateRateLimiter
```

## Configuration-Driven Usage

Rate limiter configuration integrates with environment-specific YAML config:

```yaml
# environments/development.yaml
rate_limiters:
  api:
    backend: memory
    default_rates:
      - {limit: 1000, interval: 60}

# environments/production.yaml
rate_limiters:
  api:
    backend: pyrate
    bucket: redis
    default_rates:
      - {limit: 1000, interval: 60}
    categories:
      api_read:
        rates: [{limit: 100, interval: 6}]
      api_write:
        rates: [{limit: 10, interval: 6}]
    redis:
      url: "redis://elasticache:6379"
      ssl: true
```

```python
from dataknobs_common.ratelimit import create_rate_limiter

limiter = create_rate_limiter(config["rate_limiters"]["api"])
```

## Configuration Reference

### Common Config Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | str | `"memory"` | `"memory"` or `"pyrate"` |
| `rates` | list | — | Shorthand for `default_rates` |
| `default_rates` | list | — | Default rates (required if `rates` not set) |
| `categories` | dict | `{}` | Per-category rate overrides |

### Pyrate Backend Config Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `bucket` | str | `"memory"` | Bucket backend: `"memory"`, `"sqlite"`, `"redis"`, `"postgres"` |
| `redis` | dict | — | Redis connection config |
| `redis.url` | str | `"redis://localhost:6379"` | Redis connection URL |
| `redis.ssl` | bool | `false` | Enable TLS |
| `postgres` | dict | — | PostgreSQL config |
| `postgres.pool` | object | — | psycopg pool instance (required) |
| `postgres.table` | str | `"rate_limits"` | Table name for rate limit state |
| `sqlite` | dict | — | SQLite config |
| `sqlite.db_path` | str | `"rate_limits.db"` | Path to SQLite database |
| `sqlite.table` | str | `"rate_limits"` | Table name |

## Testing

For testing, use the in-memory rate limiter directly — it is the **testing construct** (like `InMemoryEventBus`):

```python
import pytest
from dataknobs_common.ratelimit import (
    InMemoryRateLimiter,
    RateLimit,
    RateLimiterConfig,
)

@pytest.fixture
def rate_limiter():
    config = RateLimiterConfig(
        default_rates=[RateLimit(limit=100, interval=60)],
        categories={
            "api_write": [RateLimit(limit=5, interval=60)],
        },
    )
    return InMemoryRateLimiter(config)

async def test_rate_limiting(rate_limiter):
    # Permits within limit
    for _ in range(5):
        assert await rate_limiter.try_acquire("api_write") is True

    # Denies beyond limit
    assert await rate_limiter.try_acquire("api_write") is False

    # Reset restores capacity
    await rate_limiter.reset("api_write")
    assert await rate_limiter.try_acquire("api_write") is True
```

## API Reference

### RateLimiter Protocol

```python
@runtime_checkable
class RateLimiter(Protocol):
    async def acquire(
        self, name: str = "default", weight: int = 1,
        timeout: float | None = None,
    ) -> None:
        """Acquire capacity, blocking until available.

        Raises TimeoutError if timeout is exceeded.
        """

    async def try_acquire(
        self, name: str = "default", weight: int = 1,
    ) -> bool:
        """Attempt to acquire without blocking. Returns True/False."""

    async def get_status(self, name: str = "default") -> RateLimitStatus:
        """Get current bucket status."""

    async def reset(self, name: str | None = None) -> None:
        """Reset state for a category or all categories."""

    async def close(self) -> None:
        """Release resources."""
```

### Factory Function

```python
def create_rate_limiter(config: dict) -> RateLimiter:
    """Create a rate limiter from configuration.

    Args:
        config: Configuration dict with rate and backend settings.

    Returns:
        RateLimiter implementation.

    Raises:
        ValueError: If backend is unknown or config is invalid.
    """
```

### RateLimitError

```python
class RateLimitError(OperationError):
    """Raised when a rate limit is exceeded.

    Attributes:
        retry_after: Optional seconds to wait before retrying.
    """
```

## Module Exports

```python
from dataknobs_common.ratelimit import (
    # Protocol
    RateLimiter,
    # Factory
    create_rate_limiter,
    # Types
    RateLimit,
    RateLimiterConfig,
    RateLimitStatus,
    # In-memory implementation
    InMemoryRateLimiter,
)

# Exception (also available from dataknobs_common.exceptions)
from dataknobs_common import RateLimitError

# Pyrate backend (import directly — requires pyrate-limiter)
from dataknobs_common.ratelimit.pyrate import PyrateRateLimiter
```
