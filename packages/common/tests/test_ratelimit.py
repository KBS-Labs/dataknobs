"""Tests for the rate limiting abstraction."""

import asyncio

import pytest

from dataknobs_common.exceptions import OperationError, RateLimitError, TimeoutError
from dataknobs_common.ratelimit import (
    InMemoryRateLimiter,
    RateLimit,
    RateLimiter,
    RateLimiterConfig,
    RateLimitStatus,
    create_rate_limiter,
)


class TestRateLimitTypes:
    """Tests for rate limit data types."""

    def test_rate_limit_creation(self):
        """Test creating a RateLimit."""
        rate = RateLimit(limit=10, interval=60)
        assert rate.limit == 10
        assert rate.interval == 60

    def test_rate_limit_frozen(self):
        """Test that RateLimit is immutable."""
        rate = RateLimit(limit=10, interval=60)
        with pytest.raises(AttributeError):
            rate.limit = 20  # type: ignore[misc]

    def test_config_creation(self):
        """Test creating a RateLimiterConfig."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=100, interval=60)],
        )
        assert len(config.default_rates) == 1
        assert config.categories == {}

    def test_config_with_categories(self):
        """Test config with per-category overrides."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=100, interval=60)],
            categories={
                "api_read": [RateLimit(limit=50, interval=10)],
                "api_write": [RateLimit(limit=5, interval=10)],
            },
        )
        assert len(config.categories) == 2
        assert config.categories["api_write"][0].limit == 5

    def test_status_creation(self):
        """Test creating a RateLimitStatus."""
        status = RateLimitStatus(
            name="test",
            current_count=3,
            limit=10,
            remaining=7,
            reset_after=5.0,
        )
        assert status.name == "test"
        assert status.remaining == 7


class TestInMemoryTryAcquire:
    """Tests for InMemoryRateLimiter.try_acquire."""

    @pytest.fixture
    def limiter(self):
        """Create a limiter with 3 ops per 0.1s window."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=3, interval=0.1)],
        )
        return InMemoryRateLimiter(config)

    @pytest.mark.asyncio
    async def test_permits_within_limit(self, limiter):
        """Test that requests within limit are permitted."""
        assert await limiter.try_acquire() is True
        assert await limiter.try_acquire() is True
        assert await limiter.try_acquire() is True

    @pytest.mark.asyncio
    async def test_denies_at_limit(self, limiter):
        """Test that requests beyond limit are denied."""
        for _ in range(3):
            assert await limiter.try_acquire() is True
        assert await limiter.try_acquire() is False

    @pytest.mark.asyncio
    async def test_window_expiry_restores_capacity(self, limiter):
        """Test that capacity is restored after the window expires."""
        for _ in range(3):
            assert await limiter.try_acquire() is True
        assert await limiter.try_acquire() is False

        # Wait for the window to expire
        await asyncio.sleep(0.15)

        assert await limiter.try_acquire() is True

    @pytest.mark.asyncio
    async def test_independent_categories(self):
        """Test that different category names have independent state."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=2, interval=0.1)],
        )
        limiter = InMemoryRateLimiter(config)

        assert await limiter.try_acquire("cat_a") is True
        assert await limiter.try_acquire("cat_a") is True
        assert await limiter.try_acquire("cat_a") is False

        # cat_b is independent
        assert await limiter.try_acquire("cat_b") is True
        assert await limiter.try_acquire("cat_b") is True

    @pytest.mark.asyncio
    async def test_weighted_acquire(self):
        """Test that weight is respected."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=10, interval=1.0)],
        )
        limiter = InMemoryRateLimiter(config)

        assert await limiter.try_acquire(weight=7) is True
        assert await limiter.try_acquire(weight=3) is True
        assert await limiter.try_acquire(weight=1) is False

    @pytest.mark.asyncio
    async def test_multiple_rates_most_restrictive_wins(self):
        """Test that the most restrictive rate applies."""
        config = RateLimiterConfig(
            default_rates=[
                RateLimit(limit=2, interval=0.1),   # 2 per 0.1s
                RateLimit(limit=100, interval=60),   # 100 per minute
            ],
        )
        limiter = InMemoryRateLimiter(config)

        assert await limiter.try_acquire() is True
        assert await limiter.try_acquire() is True
        # Short window rate (2/0.1s) is hit
        assert await limiter.try_acquire() is False


class TestInMemoryPerCategory:
    """Tests for per-category rate configuration."""

    @pytest.fixture
    def limiter(self):
        """Create a limiter with per-category overrides."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=10, interval=1.0)],
            categories={
                "strict": [RateLimit(limit=2, interval=1.0)],
                "relaxed": [RateLimit(limit=100, interval=1.0)],
            },
        )
        return InMemoryRateLimiter(config)

    @pytest.mark.asyncio
    async def test_category_specific_rates(self, limiter):
        """Test that category-specific rates are enforced."""
        assert await limiter.try_acquire("strict") is True
        assert await limiter.try_acquire("strict") is True
        assert await limiter.try_acquire("strict") is False

    @pytest.mark.asyncio
    async def test_unknown_category_uses_default(self, limiter):
        """Test that unknown categories fall back to default rates."""
        for _ in range(10):
            assert await limiter.try_acquire("unknown_category") is True
        assert await limiter.try_acquire("unknown_category") is False

    @pytest.mark.asyncio
    async def test_relaxed_category(self, limiter):
        """Test that a relaxed category allows more operations."""
        for _ in range(50):
            assert await limiter.try_acquire("relaxed") is True


class TestInMemoryAcquire:
    """Tests for InMemoryRateLimiter.acquire (blocking)."""

    @pytest.mark.asyncio
    async def test_succeeds_immediately_when_available(self):
        """Test that acquire returns immediately when capacity is available."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=5, interval=1.0)],
        )
        limiter = InMemoryRateLimiter(config)

        # Should not block
        await asyncio.wait_for(limiter.acquire(), timeout=0.5)

    @pytest.mark.asyncio
    async def test_blocks_then_succeeds(self):
        """Test that acquire blocks and succeeds after window expires."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=1, interval=0.1)],
        )
        limiter = InMemoryRateLimiter(config)

        # Exhaust capacity
        await limiter.acquire()

        # Should block briefly then succeed after window expires
        await asyncio.wait_for(limiter.acquire(), timeout=0.5)

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        """Test that acquire raises TimeoutError when timeout is exceeded."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=1, interval=10.0)],
        )
        limiter = InMemoryRateLimiter(config)

        # Exhaust capacity
        await limiter.acquire()

        # Should raise TimeoutError quickly
        with pytest.raises(TimeoutError) as excinfo:
            await limiter.acquire(timeout=0.1)

        assert "timed out" in str(excinfo.value).lower()


class TestInMemoryStatus:
    """Tests for InMemoryRateLimiter.get_status."""

    @pytest.mark.asyncio
    async def test_empty_status(self):
        """Test status when no requests have been made."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=10, interval=1.0)],
        )
        limiter = InMemoryRateLimiter(config)

        status = await limiter.get_status()
        assert status.name == "default"
        assert status.current_count == 0
        assert status.limit == 10
        assert status.remaining == 10
        assert status.reset_after == 0.0

    @pytest.mark.asyncio
    async def test_status_after_requests(self):
        """Test status reflects consumed capacity."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=10, interval=1.0)],
        )
        limiter = InMemoryRateLimiter(config)

        await limiter.try_acquire()
        await limiter.try_acquire()
        await limiter.try_acquire()

        status = await limiter.get_status()
        assert status.current_count == 3
        assert status.remaining == 7
        assert status.reset_after > 0.0

    @pytest.mark.asyncio
    async def test_status_named_category(self):
        """Test status for a specific category."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=10, interval=1.0)],
            categories={"api": [RateLimit(limit=5, interval=1.0)]},
        )
        limiter = InMemoryRateLimiter(config)

        await limiter.try_acquire("api")

        status = await limiter.get_status("api")
        assert status.name == "api"
        assert status.current_count == 1
        assert status.limit == 5
        assert status.remaining == 4


class TestInMemoryReset:
    """Tests for InMemoryRateLimiter.reset."""

    @pytest.mark.asyncio
    async def test_reset_specific_category(self):
        """Test resetting a single category."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=2, interval=1.0)],
        )
        limiter = InMemoryRateLimiter(config)

        await limiter.try_acquire("cat_a")
        await limiter.try_acquire("cat_a")
        assert await limiter.try_acquire("cat_a") is False

        await limiter.reset("cat_a")
        assert await limiter.try_acquire("cat_a") is True

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """Test resetting all categories."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=1, interval=1.0)],
        )
        limiter = InMemoryRateLimiter(config)

        await limiter.try_acquire("cat_a")
        await limiter.try_acquire("cat_b")
        assert await limiter.try_acquire("cat_a") is False
        assert await limiter.try_acquire("cat_b") is False

        await limiter.reset()
        assert await limiter.try_acquire("cat_a") is True
        assert await limiter.try_acquire("cat_b") is True

    @pytest.mark.asyncio
    async def test_reset_nonexistent_is_noop(self):
        """Test resetting a nonexistent category does not raise."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=10, interval=1.0)],
        )
        limiter = InMemoryRateLimiter(config)
        await limiter.reset("nonexistent")  # Should not raise


class TestCreateRateLimiter:
    """Tests for the create_rate_limiter factory."""

    @pytest.mark.asyncio
    async def test_memory_backend(self):
        """Test creating a memory backend limiter."""
        limiter = create_rate_limiter({
            "backend": "memory",
            "rates": [{"limit": 10, "interval": 60}],
        })
        assert isinstance(limiter, InMemoryRateLimiter)

    @pytest.mark.asyncio
    async def test_default_is_memory(self):
        """Test that the default backend is memory."""
        limiter = create_rate_limiter({
            "rates": [{"limit": 10, "interval": 60}],
        })
        assert isinstance(limiter, InMemoryRateLimiter)

    @pytest.mark.asyncio
    async def test_default_rates_alias(self):
        """Test that 'default_rates' works as an alias for 'rates'."""
        limiter = create_rate_limiter({
            "default_rates": [{"limit": 10, "interval": 60}],
        })
        assert isinstance(limiter, InMemoryRateLimiter)

    @pytest.mark.asyncio
    async def test_per_category_config_roundtrip(self):
        """Test that per-category config is parsed and applied."""
        limiter = create_rate_limiter({
            "default_rates": [{"limit": 100, "interval": 60}],
            "categories": {
                "api_write": {"rates": [{"limit": 2, "interval": 0.1}]},
            },
        })

        assert await limiter.try_acquire("api_write") is True
        assert await limiter.try_acquire("api_write") is True
        assert await limiter.try_acquire("api_write") is False

    @pytest.mark.asyncio
    async def test_unknown_backend_raises(self):
        """Test that an unknown backend raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            create_rate_limiter({
                "backend": "unknown",
                "rates": [{"limit": 10, "interval": 60}],
            })
        assert "unknown" in str(excinfo.value).lower()
        assert "memory" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_missing_rates_raises(self):
        """Test that missing rates raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            create_rate_limiter({"backend": "memory"})
        assert "rates" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_rate_dict_raises(self):
        """Test that a rate dict missing keys raises ValueError."""
        with pytest.raises(ValueError):
            create_rate_limiter({
                "rates": [{"limit": 10}],  # missing 'interval'
            })


class TestRateLimiterProtocol:
    """Tests that implementations satisfy the RateLimiter protocol."""

    @pytest.mark.asyncio
    async def test_inmemory_satisfies_protocol(self):
        """Test InMemoryRateLimiter satisfies the RateLimiter protocol."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=10, interval=60)],
        )
        limiter = InMemoryRateLimiter(config)

        assert hasattr(limiter, "acquire")
        assert hasattr(limiter, "try_acquire")
        assert hasattr(limiter, "get_status")
        assert hasattr(limiter, "reset")
        assert hasattr(limiter, "close")
        assert isinstance(limiter, RateLimiter)


class TestRateLimitError:
    """Tests for the RateLimitError exception."""

    def test_creation(self):
        """Test creating a RateLimitError."""
        error = RateLimitError("Too many requests")
        assert str(error) == "Too many requests"

    def test_default_message(self):
        """Test default error message."""
        error = RateLimitError()
        assert str(error) == "Rate limit exceeded"

    def test_retry_after(self):
        """Test retry_after attribute."""
        error = RateLimitError("wait", retry_after=2.5)
        assert error.retry_after == 2.5

    def test_retry_after_none_by_default(self):
        """Test that retry_after is None by default."""
        error = RateLimitError()
        assert error.retry_after is None

    def test_extends_operation_error(self):
        """Test that RateLimitError extends OperationError."""
        error = RateLimitError("test")
        assert isinstance(error, OperationError)

    def test_context_dict(self):
        """Test context dict support."""
        error = RateLimitError(
            "Rate limited",
            retry_after=5.0,
            context={"category": "api_write", "limit": 10},
        )
        assert error.context["category"] == "api_write"
        assert error.context["limit"] == 10
        assert error.retry_after == 5.0


class TestPackageImport:
    """Tests that all exports are importable from dataknobs_common."""

    def test_ratelimit_exports_from_package(self):
        """Test that rate limit types are importable from the main package."""
        from dataknobs_common import (  # noqa: F401
            InMemoryRateLimiter,
            RateLimit,
            RateLimiter,
            RateLimiterConfig,
            RateLimitError,
            RateLimitStatus,
            create_rate_limiter,
        )

    def test_ratelimit_exports_from_subpackage(self):
        """Test that rate limit types are importable from the subpackage."""
        from dataknobs_common.ratelimit import (  # noqa: F401
            InMemoryRateLimiter,
            RateLimit,
            RateLimiter,
            RateLimiterConfig,
            RateLimitStatus,
            create_rate_limiter,
        )


class TestInMemoryClose:
    """Tests for InMemoryRateLimiter.close."""

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        """Test that close completes without error."""
        config = RateLimiterConfig(
            default_rates=[RateLimit(limit=10, interval=1.0)],
        )
        limiter = InMemoryRateLimiter(config)
        await limiter.close()  # Should not raise
