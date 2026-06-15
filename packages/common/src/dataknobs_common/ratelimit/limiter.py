"""RateLimiter protocol and registry-driven factory function.

``create_rate_limiter()`` parses the top-level rate / category config and
resolves the ``backend`` discriminator through the
:data:`rate_limiter_backends` :class:`PluginRegistry`. Out-of-tree
consumers can add a custom :class:`RateLimiter` backend (e.g. a direct
Redis or etcd limiter, an HTTP-call quota tracker) without forking
DataKnobs::

    from dataknobs_common.ratelimit import (
        rate_limiter_backends,
        create_rate_limiter,
    )

    def _make_quota_limiter(config, *, parsed):
        from my_pkg.quota_limiter import QuotaRateLimiter
        return QuotaRateLimiter(parsed, url=config["url"])

    rate_limiter_backends.register("quota_http", _make_quota_limiter)
    limiter = create_rate_limiter({
        "backend": "quota_http",
        "rates": [{"limit": 100, "interval": 60}],
        "url": "https://quota.internal/api",
    })

The registry abstraction (config-key resolution, not-found error shape,
sync/async dispatch) is shared with ``event_bus_backends`` and
``lock_backends`` — see :class:`~dataknobs_common.registry.PluginRegistry`
for the underlying contract.

Each built-in wrapper imports its concrete backend *lazily* (inside the
factory call) so importing this module never pulls optional backend
dependencies (``pyrate-limiter``) at module load time.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dataknobs_common.registry import PluginRegistry

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


rate_limiter_backends: PluginRegistry[RateLimiter] = PluginRegistry(
    name="rate_limiter_backends",
    validate_type=RateLimiter,
    config_key="backend",
    config_key_default="memory",
    not_found_kind="rate limiter backend",
    not_found_exception=ValueError,
)
"""Registry of named :class:`RateLimiter` factories.

Each registered factory receives both the raw config dict AND the parsed
:class:`RateLimiterConfig` via the keyword ``parsed=`` — every backend
needs the parsed rates, so the shim parses once at the top and forwards
to the registry. Out-of-tree factories follow the same convention::

    def my_factory(config: dict, *, parsed: RateLimiterConfig) -> RateLimiter:
        ...

Register a custom backend with
``rate_limiter_backends.register("name", factory)`` and select it via
``create_rate_limiter({"backend": "name", ...})``. The registry conforms
to :class:`~dataknobs_common.registry.BackendRegistry` for ``isinstance``
checks.
"""


def _create_memory_limiter(
    config: dict[str, Any], *, parsed: RateLimiterConfig
) -> RateLimiter:
    from .memory import InMemoryRateLimiter

    return InMemoryRateLimiter(parsed)


def _create_pyrate_limiter(
    config: dict[str, Any], *, parsed: RateLimiterConfig
) -> RateLimiter:
    # Lazy import keeps ``limiter.py`` (and ``dataknobs_common.ratelimit``)
    # importable without ``pyrate-limiter`` installed — the optional
    # ``pyrate`` extra is only required when this backend is selected.
    from .pyrate import PyrateRateLimiter

    return PyrateRateLimiter(parsed, config)


rate_limiter_backends.register("memory", _create_memory_limiter)
rate_limiter_backends.register("pyrate", _create_pyrate_limiter)


def create_rate_limiter(config: dict[str, Any]) -> RateLimiter:
    """Create a rate limiter from configuration.

    Parses the top-level rate / category config and dispatches the
    ``backend`` key through the :data:`rate_limiter_backends` registry,
    so out-of-tree consumers can register and select a custom backend
    without forking DataKnobs:

        ```python
        from dataknobs_common.ratelimit import (
            rate_limiter_backends,
            create_rate_limiter,
        )

        rate_limiter_backends.register("quota_http", my_quota_factory)
        limiter = create_rate_limiter({
            "backend": "quota_http",
            "rates": [{"limit": 100, "interval": 60}],
            "url": "...",
        })
        ```

    Args:
        config: Configuration dict. See below for keys.

    Returns:
        RateLimiter instance.

    Raises:
        ValueError: If the backend is not registered, or required rate
            config is missing. The not-found message lists all registered
            backends (including consumer-registered ones).
        OperationError: If the backend factory raises during construction
            (invalid backend-specific config, missing optional dependency,
            etc.). Wraps the originating exception via ``__cause__``.

    Config keys:
        backend: ``"memory"`` (default) or ``"pyrate"``, plus any
            consumer-registered backend name.
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
    parsed = _parse_config(config)
    return rate_limiter_backends.create(config=config, parsed=parsed)


async def create_rate_limiter_async(config: dict[str, Any]) -> RateLimiter:
    """Async-symmetric counterpart to :func:`create_rate_limiter`.

    Performs the same top-level rate-list / category normalization as the
    sync shim, then dispatches through
    :meth:`PluginRegistry.create_async`. Today every built-in backend
    constructs synchronously, so this function returns the same instance
    type as :func:`create_rate_limiter`; the surface is shipped for API
    symmetry and consumer-extensibility (an out-of-tree backend's
    ``from_config_async`` is detected and awaited).

    Args:
        config: Configuration dict. Same shape as :func:`create_rate_limiter`.

    Returns:
        RateLimiter instance.

    Raises:
        ValueError: If the backend is not registered, or required rate
            config is missing. Normalization runs *before* backend
            dispatch, so a missing-``rates`` config raises here rather
            than from the backend factory.
        OperationError: If the backend factory raises during construction.
            Wraps the originating exception via ``__cause__``. Same
            behaviour as the sync :func:`create_rate_limiter`.
    """
    parsed = _parse_config(config)
    return await rate_limiter_backends.create_async(
        config=config, parsed=parsed
    )
