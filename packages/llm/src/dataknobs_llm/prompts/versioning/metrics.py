"""Metrics tracking for prompt versions.

This module provides:
- Event-based metrics collection
- Aggregated metrics computation
- Performance comparison across versions
- Experiment metrics analysis
"""

from typing import Any, Dict, List
from datetime import datetime

from .types import (
    PromptMetrics,
    MetricEvent,
)


class MetricsCollector:
    """Collects and aggregates metrics for prompt versions.

    Tracks usage, performance, and user feedback for each version.
    Supports both real-time event recording and aggregated metrics retrieval.

    Example:
        ```python
        collector = MetricsCollector(storage_backend)

        # Record a usage event
        await collector.record_event(
            version_id="v1",
            success=True,
            response_time=0.5,
            tokens=150,
            user_rating=4.5
        )

        # Get aggregated metrics
        metrics = await collector.get_metrics("v1")
        print(f"Success rate: {metrics.success_rate:.2%}")
        print(f"Avg response time: {metrics.avg_response_time:.2f}s")

        # Compare variants in experiment
        comparison = await collector.compare_variants(
            experiment_id="exp1"
        )
        ```
    """

    def __init__(self, storage: Any | None = None):
        """Initialize metrics collector.

        Args:
            storage: Backend storage (dict for in-memory, database for persistence)
                    If None, uses in-memory dictionary
        """
        self.storage = storage if storage is not None else {}
        self._metrics: Dict[str, PromptMetrics] = {}  # version_id -> PromptMetrics
        self._events: Dict[str, List[MetricEvent]] = {}  # version_id -> [events]

    async def record_event(
        self,
        version_id: str,
        success: bool = True,
        response_time: float | None = None,
        tokens: int | None = None,
        user_rating: float | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> MetricEvent:
        """Record a single usage event.

        Args:
            version_id: Version ID this event belongs to
            success: Whether the use was successful
            response_time: Response time in seconds (None if not applicable)
            tokens: Number of tokens used (None if not applicable)
            user_rating: User rating 1-5 (None if not provided)
            metadata: Additional event metadata

        Returns:
            Created MetricEvent

        Raises:
            ValueError: If user_rating is not in valid range
        """
        if user_rating is not None and not (1.0 <= user_rating <= 5.0):
            raise ValueError(f"User rating must be between 1.0 and 5.0, got {user_rating}")

        # Create event
        event = MetricEvent(
            version_id=version_id,
            timestamp=datetime.utcnow(),
            success=success,
            response_time=response_time,
            tokens=tokens,
            user_rating=user_rating,
            metadata=metadata or {},
        )

        # Store event
        if version_id not in self._events:
            self._events[version_id] = []
        self._events[version_id].append(event)

        # Update aggregated metrics
        await self._update_metrics(version_id, event)

        # Persist event if backend available
        if hasattr(self.storage, "append"):
            await self._persist_event(event)

        return event

    async def get_metrics(
        self,
        version_id: str,
    ) -> PromptMetrics:
        """Get aggregated metrics for a version.

        If no events have been recorded, returns empty metrics.

        Args:
            version_id: Version ID

        Returns:
            PromptMetrics with aggregated statistics
        """
        if version_id not in self._metrics:
            # Return empty metrics
            return PromptMetrics(version_id=version_id)

        return self._metrics[version_id]

    async def get_events(
        self,
        version_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> List[MetricEvent]:
        """Get raw events for a version.

        Args:
            version_id: Version ID
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum number of events to return (most recent first)

        Returns:
            List of MetricEvent objects
        """
        events = self._events.get(version_id, [])

        # Apply time filters
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # Sort by timestamp (most recent first)
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)

        # Apply limit
        if limit:
            events = events[:limit]

        return events

    async def compare_variants(
        self,
        version_ids: List[str],
    ) -> Dict[str, PromptMetrics]:
        """Compare metrics across multiple versions.

        Args:
            version_ids: List of version IDs to compare

        Returns:
            Dictionary mapping version_id to PromptMetrics
        """
        comparison = {}
        for version_id in version_ids:
            comparison[version_id] = await self.get_metrics(version_id)
        return comparison

    async def get_experiment_metrics(
        self,
        experiment_id: str,
        variant_versions: List[str],
    ) -> Dict[str, PromptMetrics]:
        """Get metrics for all variants in an experiment.

        Convenience method for comparing experiment variants.

        Args:
            experiment_id: Experiment ID (for metadata)
            variant_versions: List of version strings in the experiment

        Returns:
            Dictionary mapping version to PromptMetrics
        """
        return await self.compare_variants(variant_versions)

    async def reset_metrics(
        self,
        version_id: str,
    ) -> bool:
        """Reset metrics for a version.

        Warning: This permanently deletes all events and metrics for this version.

        Args:
            version_id: Version ID

        Returns:
            True if reset, False if version not found
        """
        if version_id not in self._metrics and version_id not in self._events:
            return False

        # Clear metrics and events
        if version_id in self._metrics:
            del self._metrics[version_id]
        if version_id in self._events:
            del self._events[version_id]

        # Persist deletion if backend available
        if hasattr(self.storage, "delete"):
            await self.storage.delete(f"metrics:{version_id}")

        return True

    async def get_summary(
        self,
        version_ids: List[str],
    ) -> Dict[str, Any]:
        """Get summary statistics across multiple versions.

        Args:
            version_ids: List of version IDs

        Returns:
            Summary dictionary with aggregated statistics
        """
        all_metrics = await self.compare_variants(version_ids)

        total_uses = sum(m.total_uses for m in all_metrics.values())
        total_successes = sum(m.success_count for m in all_metrics.values())
        total_errors = sum(m.error_count for m in all_metrics.values())

        return {
            "total_versions": len(version_ids),
            "total_uses": total_uses,
            "total_successes": total_successes,
            "total_errors": total_errors,
            "overall_success_rate": total_successes / total_uses if total_uses > 0 else 0.0,
            "versions": {
                vid: {
                    "uses": m.total_uses,
                    "success_rate": m.success_rate,
                    "avg_response_time": m.avg_response_time,
                    "avg_tokens": m.avg_tokens,
                    "avg_rating": m.avg_rating,
                }
                for vid, m in all_metrics.items()
            }
        }

    # ===== Helper Methods =====

    async def _update_metrics(
        self,
        version_id: str,
        event: MetricEvent,
    ):
        """Update aggregated metrics with new event."""
        # Get or create metrics
        if version_id not in self._metrics:
            self._metrics[version_id] = PromptMetrics(version_id=version_id)

        metrics = self._metrics[version_id]

        # Update counters
        metrics.total_uses += 1
        if event.success:
            metrics.success_count += 1
        else:
            metrics.error_count += 1

        # Update response time
        if event.response_time is not None:
            metrics.total_response_time += event.response_time

        # Update tokens
        if event.tokens is not None:
            metrics.total_tokens += event.tokens

        # Update ratings
        if event.user_rating is not None:
            metrics.user_ratings.append(event.user_rating)

        # Update last used timestamp
        metrics.last_used = event.timestamp

        # Persist if backend available
        if hasattr(self.storage, "set"):
            await self._persist_metrics(metrics)

    async def _persist_event(self, event: MetricEvent):
        """Persist event to backend storage."""
        if hasattr(self.storage, "append"):
            key = f"events:{event.version_id}"
            await self.storage.append(key, event.to_dict())

    async def _persist_metrics(self, metrics: PromptMetrics):
        """Persist metrics to backend storage."""
        if hasattr(self.storage, "set"):
            key = f"metrics:{metrics.version_id}"
            await self.storage.set(key, metrics.to_dict())

    async def get_top_versions(
        self,
        version_ids: List[str],
        metric: str = "success_rate",
        limit: int = 5,
    ) -> List[tuple[str, float]]:
        """Get top performing versions by a specific metric.

        Args:
            version_ids: List of version IDs to rank
            metric: Metric to rank by ("success_rate", "avg_rating", "avg_response_time")
            limit: Number of top versions to return

        Returns:
            List of (version_id, metric_value) tuples, sorted by metric

        Raises:
            ValueError: If metric name is invalid
        """
        valid_metrics = ["success_rate", "avg_rating", "avg_response_time", "avg_tokens"]
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric: {metric}. Valid metrics: {', '.join(valid_metrics)}"
            )

        # Get metrics for all versions
        all_metrics = await self.compare_variants(version_ids)

        # Extract metric values
        metric_values = [
            (vid, getattr(metrics, metric))
            for vid, metrics in all_metrics.items()
            if metrics.total_uses > 0  # Only include versions with data
        ]

        # Sort by metric value
        # For response_time, lower is better (reverse=False)
        # For success_rate, rating, higher is better (reverse=True)
        reverse = metric != "avg_response_time"
        sorted_versions = sorted(metric_values, key=lambda x: x[1], reverse=reverse)

        return sorted_versions[:limit]

    async def get_version_performance_over_time(
        self,
        version_id: str,
        bucket_size: str = "hour",
    ) -> List[Dict[str, Any]]:
        """Get performance metrics bucketed by time period.

        Args:
            version_id: Version ID
            bucket_size: Time bucket size ("hour", "day", "week")

        Returns:
            List of time-bucketed metrics

        Note:
            This is a simplified implementation. Production would use
            proper time-series bucketing.
        """
        events = await self.get_events(version_id)

        if not events:
            return []

        # Group events by time bucket
        buckets: Dict[str, List[MetricEvent]] = {}

        for event in events:
            # Create bucket key based on bucket_size
            if bucket_size == "hour":
                bucket_key = event.timestamp.strftime("%Y-%m-%d %H:00")
            elif bucket_size == "day":
                bucket_key = event.timestamp.strftime("%Y-%m-%d")
            elif bucket_size == "week":
                bucket_key = event.timestamp.strftime("%Y-W%W")
            else:
                bucket_key = event.timestamp.strftime("%Y-%m-%d")

            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(event)

        # Compute metrics for each bucket
        result = []
        for bucket_key, bucket_events in sorted(buckets.items()):
            total = len(bucket_events)
            successes = sum(1 for e in bucket_events if e.success)

            result.append({
                "time_bucket": bucket_key,
                "total_uses": total,
                "success_count": successes,
                "success_rate": successes / total if total > 0 else 0.0,
                "avg_response_time": sum(
                    e.response_time for e in bucket_events if e.response_time
                ) / total if total > 0 else 0.0,
            })

        return result
