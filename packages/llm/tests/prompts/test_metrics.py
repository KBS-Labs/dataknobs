"""Tests for metrics tracking functionality."""

import pytest
from datetime import datetime, timedelta

from dataknobs_llm.prompts.versioning import (
    MetricsCollector,
    PromptMetrics,
    MetricEvent,
)


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector instance for testing."""
        return MetricsCollector()

    @pytest.mark.asyncio
    async def test_record_event(self, collector):
        """Test recording a usage event."""
        event = await collector.record_event(
            version_id="v1",
            success=True,
            response_time=0.5,
            tokens=100,
            user_rating=4.5
        )

        assert event.version_id == "v1"
        assert event.success is True
        assert event.response_time == 0.5
        assert event.tokens == 100
        assert event.user_rating == 4.5

    @pytest.mark.asyncio
    async def test_record_event_validates_rating(self, collector):
        """Test that invalid rating raises error."""
        with pytest.raises(ValueError, match="rating must be between"):
            await collector.record_event(
                version_id="v1",
                user_rating=6.0  # Invalid: > 5.0
            )

        with pytest.raises(ValueError, match="rating must be between"):
            await collector.record_event(
                version_id="v1",
                user_rating=0.5  # Invalid: < 1.0
            )

    @pytest.mark.asyncio
    async def test_get_metrics_empty(self, collector):
        """Test getting metrics for version with no events."""
        metrics = await collector.get_metrics("nonexistent")

        assert metrics.version_id == "nonexistent"
        assert metrics.total_uses == 0
        assert metrics.success_rate == 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_after_events(self, collector):
        """Test getting aggregated metrics after recording events."""
        # Record some events
        await collector.record_event(
            version_id="v1",
            success=True,
            response_time=0.5,
            tokens=100
        )

        await collector.record_event(
            version_id="v1",
            success=True,
            response_time=0.7,
            tokens=150
        )

        await collector.record_event(
            version_id="v1",
            success=False,  # One failure
            response_time=1.0,
            tokens=200
        )

        metrics = await collector.get_metrics("v1")

        assert metrics.total_uses == 3
        assert metrics.success_count == 2
        assert metrics.error_count == 1
        assert abs(metrics.success_rate - 2/3) < 0.01
        assert abs(metrics.avg_response_time - 0.73) < 0.05
        assert abs(metrics.avg_tokens - 150.0) < 0.1

    @pytest.mark.asyncio
    async def test_metrics_success_rate(self, collector):
        """Test success rate calculation."""
        # All successful
        for _ in range(10):
            await collector.record_event(version_id="v1", success=True)

        metrics = await collector.get_metrics("v1")
        assert metrics.success_rate == 1.0

        # Add some failures
        for _ in range(5):
            await collector.record_event(version_id="v1", success=False)

        metrics = await collector.get_metrics("v1")
        assert abs(metrics.success_rate - 10/15) < 0.01

    @pytest.mark.asyncio
    async def test_metrics_avg_response_time(self, collector):
        """Test average response time calculation."""
        await collector.record_event(version_id="v1", response_time=0.5)
        await collector.record_event(version_id="v1", response_time=1.5)
        await collector.record_event(version_id="v1", response_time=1.0)

        metrics = await collector.get_metrics("v1")
        assert abs(metrics.avg_response_time - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_metrics_avg_tokens(self, collector):
        """Test average tokens calculation."""
        await collector.record_event(version_id="v1", tokens=100)
        await collector.record_event(version_id="v1", tokens=200)
        await collector.record_event(version_id="v1", tokens=150)

        metrics = await collector.get_metrics("v1")
        assert abs(metrics.avg_tokens - 150.0) < 0.1

    @pytest.mark.asyncio
    async def test_metrics_avg_rating(self, collector):
        """Test average rating calculation."""
        await collector.record_event(version_id="v1", user_rating=5.0)
        await collector.record_event(version_id="v1", user_rating=4.0)
        await collector.record_event(version_id="v1", user_rating=3.0)

        metrics = await collector.get_metrics("v1")
        assert abs(metrics.avg_rating - 4.0) < 0.01

    @pytest.mark.asyncio
    async def test_metrics_last_used(self, collector):
        """Test that last_used timestamp is updated."""
        before = datetime.utcnow()

        await collector.record_event(version_id="v1", success=True)

        after = datetime.utcnow()

        metrics = await collector.get_metrics("v1")
        assert metrics.last_used is not None
        assert before <= metrics.last_used <= after

    @pytest.mark.asyncio
    async def test_get_events(self, collector):
        """Test retrieving raw events."""
        # Record some events
        await collector.record_event(version_id="v1", success=True)
        await collector.record_event(version_id="v1", success=False)
        await collector.record_event(version_id="v1", success=True)

        events = await collector.get_events("v1")

        assert len(events) == 3
        # Should be sorted newest first
        assert events[0].timestamp >= events[1].timestamp
        assert events[1].timestamp >= events[2].timestamp

    @pytest.mark.asyncio
    async def test_get_events_with_limit(self, collector):
        """Test limiting number of events returned."""
        # Record 10 events
        for i in range(10):
            await collector.record_event(version_id="v1", success=True)

        events = await collector.get_events("v1", limit=5)

        assert len(events) == 5

    @pytest.mark.asyncio
    async def test_get_events_with_time_filter(self, collector):
        """Test filtering events by time range."""
        now = datetime.utcnow()

        # Record event (this will have current time)
        await collector.record_event(version_id="v1", success=True)

        # Filter to future time range (should get nothing)
        future = now + timedelta(hours=1)
        events = await collector.get_events(
            version_id="v1",
            start_time=future
        )
        assert len(events) == 0

        # Filter to past time range (should get the event)
        past = now - timedelta(hours=1)
        events = await collector.get_events(
            version_id="v1",
            start_time=past
        )
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_compare_variants(self, collector):
        """Test comparing metrics across versions."""
        # Record events for version 1
        for _ in range(5):
            await collector.record_event(version_id="v1", success=True)

        # Record events for version 2
        for _ in range(10):
            await collector.record_event(version_id="v2", success=True)
        for _ in range(5):
            await collector.record_event(version_id="v2", success=False)

        comparison = await collector.compare_variants(["v1", "v2"])

        assert "v1" in comparison
        assert "v2" in comparison

        assert comparison["v1"].total_uses == 5
        assert comparison["v1"].success_rate == 1.0

        assert comparison["v2"].total_uses == 15
        assert abs(comparison["v2"].success_rate - 10/15) < 0.01

    @pytest.mark.asyncio
    async def test_reset_metrics(self, collector):
        """Test resetting metrics for a version."""
        # Record some events
        await collector.record_event(version_id="v1", success=True)
        await collector.record_event(version_id="v1", success=True)

        # Verify metrics exist
        metrics = await collector.get_metrics("v1")
        assert metrics.total_uses == 2

        # Reset
        reset = await collector.reset_metrics("v1")
        assert reset is True

        # Verify metrics are empty
        metrics = await collector.get_metrics("v1")
        assert metrics.total_uses == 0

    @pytest.mark.asyncio
    async def test_reset_nonexistent_metrics(self, collector):
        """Test resetting metrics for nonexistent version."""
        reset = await collector.reset_metrics("nonexistent")
        assert reset is False

    @pytest.mark.asyncio
    async def test_get_summary(self, collector):
        """Test getting summary across multiple versions."""
        # Version 1: 5 successful
        for _ in range(5):
            await collector.record_event(version_id="v1", success=True)

        # Version 2: 10 successful, 5 failed
        for _ in range(10):
            await collector.record_event(version_id="v2", success=True)
        for _ in range(5):
            await collector.record_event(version_id="v2", success=False)

        summary = await collector.get_summary(["v1", "v2"])

        assert summary["total_versions"] == 2
        assert summary["total_uses"] == 20
        assert summary["total_successes"] == 15
        assert summary["total_errors"] == 5
        assert abs(summary["overall_success_rate"] - 15/20) < 0.01

        assert "v1" in summary["versions"]
        assert "v2" in summary["versions"]

    @pytest.mark.asyncio
    async def test_get_top_versions_by_success_rate(self, collector):
        """Test getting top performing versions."""
        # Version 1: 100% success
        for _ in range(10):
            await collector.record_event(version_id="v1", success=True)

        # Version 2: 50% success
        for _ in range(5):
            await collector.record_event(version_id="v2", success=True)
        for _ in range(5):
            await collector.record_event(version_id="v2", success=False)

        # Version 3: 75% success
        for _ in range(15):
            await collector.record_event(version_id="v3", success=True)
        for _ in range(5):
            await collector.record_event(version_id="v3", success=False)

        top = await collector.get_top_versions(
            ["v1", "v2", "v3"],
            metric="success_rate",
            limit=2
        )

        assert len(top) == 2
        assert top[0][0] == "v1"  # Highest success rate
        assert top[1][0] == "v3"  # Second highest

    @pytest.mark.asyncio
    async def test_get_top_versions_by_rating(self, collector):
        """Test ranking versions by user rating."""
        # Version 1: 5.0 average
        for _ in range(5):
            await collector.record_event(version_id="v1", user_rating=5.0)

        # Version 2: 3.0 average
        for _ in range(5):
            await collector.record_event(version_id="v2", user_rating=3.0)

        # Version 3: 4.5 average
        for _ in range(10):
            await collector.record_event(version_id="v3", user_rating=4.5)

        top = await collector.get_top_versions(
            ["v1", "v2", "v3"],
            metric="avg_rating",
            limit=3
        )

        assert len(top) == 3
        assert top[0][0] == "v1"  # Highest rating
        assert top[1][0] == "v3"
        assert top[2][0] == "v2"

    @pytest.mark.asyncio
    async def test_get_top_versions_by_response_time(self, collector):
        """Test ranking versions by response time (lower is better)."""
        # Version 1: 0.5s average
        for _ in range(5):
            await collector.record_event(version_id="v1", response_time=0.5)

        # Version 2: 1.0s average
        for _ in range(5):
            await collector.record_event(version_id="v2", response_time=1.0)

        # Version 3: 0.3s average (fastest)
        for _ in range(5):
            await collector.record_event(version_id="v3", response_time=0.3)

        top = await collector.get_top_versions(
            ["v1", "v2", "v3"],
            metric="avg_response_time",
            limit=3
        )

        # Should be sorted fastest to slowest
        assert top[0][0] == "v3"  # Fastest
        assert top[1][0] == "v1"
        assert top[2][0] == "v2"

    @pytest.mark.asyncio
    async def test_get_top_versions_invalid_metric(self, collector):
        """Test that invalid metric name raises error."""
        with pytest.raises(ValueError, match="Invalid metric"):
            await collector.get_top_versions(
                ["v1"],
                metric="invalid_metric"
            )

    @pytest.mark.asyncio
    async def test_get_top_versions_excludes_empty(self, collector):
        """Test that versions with no data are excluded."""
        # Version 1: has data
        await collector.record_event(version_id="v1", success=True)

        # Version 2: no data

        top = await collector.get_top_versions(
            ["v1", "v2"],
            metric="success_rate"
        )

        assert len(top) == 1
        assert top[0][0] == "v1"

    @pytest.mark.asyncio
    async def test_metrics_to_dict_and_from_dict(self, collector):
        """Test serialization/deserialization of PromptMetrics."""
        # Record some events
        await collector.record_event(
            version_id="v1",
            success=True,
            response_time=0.5,
            tokens=100,
            user_rating=4.5
        )

        original = await collector.get_metrics("v1")

        # Convert to dict
        metrics_dict = original.to_dict()

        # Verify computed properties are included
        assert "success_rate" in metrics_dict
        assert "avg_response_time" in metrics_dict
        assert "avg_tokens" in metrics_dict
        assert "avg_rating" in metrics_dict

        # Convert back
        restored = PromptMetrics.from_dict(metrics_dict)

        assert restored.version_id == original.version_id
        assert restored.total_uses == original.total_uses
        assert restored.success_count == original.success_count
        assert restored.error_count == original.error_count
        # Computed properties should match
        assert restored.success_rate == original.success_rate
        assert restored.avg_response_time == original.avg_response_time

    @pytest.mark.asyncio
    async def test_event_to_dict_and_from_dict(self):
        """Test serialization/deserialization of MetricEvent."""
        original = MetricEvent(
            version_id="v1",
            success=True,
            response_time=0.5,
            tokens=100,
            user_rating=4.5,
            metadata={"key": "value"}
        )

        # Convert to dict
        event_dict = original.to_dict()

        # Convert back
        restored = MetricEvent.from_dict(event_dict)

        assert restored.version_id == original.version_id
        assert restored.success == original.success
        assert restored.response_time == original.response_time
        assert restored.tokens == original.tokens
        assert restored.user_rating == original.user_rating
        assert restored.metadata == original.metadata
