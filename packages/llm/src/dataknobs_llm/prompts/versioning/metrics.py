# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Metrics tracking for prompt versions.

This module provides:
- Event-based metrics collection
- Aggregated metrics computation
- Performance comparison across versions
- Experiment metrics analysis
"""

from typing import Any, Dict, List
from datetime import UTC, datetime

from .store import InMemoryVersionStore, MetricsStore, require_store
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
        # In memory when nothing is passed; DatabaseVersionStore(db)
        # keeps metrics in any of the seven dataknobs backends.
        collector = MetricsCollector()

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

    def __init__(self, store: MetricsStore | None = None):
        """Initialize metrics collector.

        Args:
            store: Where aggregates and events live. Defaults to
                :class:`~.store.InMemoryVersionStore`, which is what this
                collector used to hold in two instance dictionaries. Pass
                :class:`~.store.DatabaseVersionStore` for any of the seven
                ``dataknobs_data`` backends.

        Raises:
            TypeError: If ``store`` is not a :class:`~.store.MetricsStore`.
                Events used to be persisted by ``append``-ing to a list under
                one key, which no backend offers, so an object written for the
                old parameter is caught here rather than at the first event it
                would have dropped.
        """
        self.store: MetricsStore = require_store(
            store if store is not None else InMemoryVersionStore(),
            MetricsStore,
            holder="MetricsCollector",
        )

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
            timestamp=datetime.now(UTC),
            success=success,
            response_time=response_time,
            tokens=tokens,
            user_rating=user_rating,
            metadata=metadata or {},
        )

        # Appending the event and folding it into the aggregate is one store
        # operation, not two. Done here it would be a read-modify-write with a
        # suspension point in the middle, and two concurrent recordings for one
        # version would read the same aggregate and lose an increment.
        await self.store.record_event(event)

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
        metrics = await self.store.load_metrics(version_id)
        if metrics is None:
            # A version nobody has used yet has metrics; they are all zero.
            return PromptMetrics(version_id=version_id)

        return metrics

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
            limit: Maximum number of events to return (most recent first).
                ``0`` returns none; ``None`` returns every event.

        Returns:
            List of MetricEvent objects, most recent first
        """
        # The bound goes to the store, which is the only place it can stop an
        # unbounded stream being materialized -- but only when nothing is
        # filtered out afterwards, since a page taken before a time filter is
        # not the same page as one taken after it.
        unfiltered = start_time is None and end_time is None
        events = await self.store.load_events(version_id, limit=limit if unfiltered else None)
        if unfiltered:
            return events

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # ``is not None`` rather than truthiness: a limit of 0 asks for no
        # events, where the previous reading of it returned every one of them.
        return events if limit is None else events[:limit]

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
            True if reset, False if the version had neither metrics nor events.
            The store removes both, so an aggregate cannot outlive the events
            it was computed from.
        """
        return await self.store.delete_metrics(version_id)

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
            },
        }

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
            raise ValueError(f"Invalid metric: {metric}. Valid metrics: {', '.join(valid_metrics)}")

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

            result.append(
                {
                    "time_bucket": bucket_key,
                    "total_uses": total,
                    "success_count": successes,
                    "success_rate": successes / total if total > 0 else 0.0,
                    "avg_response_time": sum(
                        e.response_time for e in bucket_events if e.response_time
                    )
                    / total
                    if total > 0
                    else 0.0,
                }
            )

        return result
