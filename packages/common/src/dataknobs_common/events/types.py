"""Event types and data structures for the event bus."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(Enum):
    """Standard event types for registry and resource changes.

    These types represent common lifecycle events for managed resources.

    Example:
        ```python
        from dataknobs_common.events import Event, EventType

        event = Event(
            type=EventType.CREATED,
            topic="registry:bots",
            payload={"bot_id": "my-bot", "config": {...}}
        )
        ```
    """

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class Event:
    """An event message for the event bus.

    Events are immutable messages that represent something that happened.
    They contain a type, topic, payload, and metadata.

    Attributes:
        type: The type of event (created, updated, deleted, etc.)
        topic: The topic/channel this event belongs to (e.g., "registry:bots")
        payload: The event data as a dictionary
        timestamp: When the event was created (defaults to now)
        event_id: Unique identifier for this event (auto-generated)
        source: Optional identifier for the event source
        correlation_id: Optional ID to correlate related events
        metadata: Additional metadata for the event

    Example:
        ```python
        event = Event(
            type=EventType.UPDATED,
            topic="registry:bots",
            payload={"bot_id": "my-bot", "changes": ["config"]}
        )

        # Serialize for transport
        data = event.to_dict()

        # Restore from transport
        restored = Event.from_dict(data)
        ```
    """

    type: EventType
    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization.

        Returns:
            Dictionary representation with ISO timestamp and string enum.
        """
        return {
            "type": self.type.value,
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        """Create event from dictionary.

        Args:
            data: Dictionary with event data

        Returns:
            Event instance
        """
        return cls(
            type=EventType(data["type"]),
            topic=data["topic"],
            payload=data.get("payload", {}),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if isinstance(data.get("timestamp"), str)
                else data.get("timestamp", datetime.now(timezone.utc))
            ),
            event_id=data.get("event_id", str(uuid.uuid4())),
            source=data.get("source"),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )

    def with_correlation(self, correlation_id: str) -> Event:
        """Create a new event with a correlation ID.

        Useful for tracking related events through a workflow.

        Args:
            correlation_id: The correlation ID to set

        Returns:
            New Event with the correlation ID set
        """
        return Event(
            type=self.type,
            topic=self.topic,
            payload=self.payload,
            timestamp=self.timestamp,
            event_id=self.event_id,
            source=self.source,
            correlation_id=correlation_id,
            metadata=self.metadata,
        )


@dataclass
class Subscription:
    """Handle for managing an event subscription.

    Subscriptions are returned when subscribing to events and can be
    used to cancel the subscription later.

    Attributes:
        subscription_id: Unique identifier for this subscription
        topic: The topic pattern this subscription is for
        handler: Reference to the handler function
        pattern: Optional wildcard pattern if using pattern matching
        created_at: When the subscription was created

    Example:
        ```python
        async def my_handler(event: Event) -> None:
            print(f"Got event: {event.type}")

        subscription = await event_bus.subscribe("registry:*", my_handler)

        # Later, to unsubscribe:
        await subscription.cancel()
        ```
    """

    subscription_id: str
    topic: str
    handler: Any  # Callable[[Event], Any] - Any to avoid complex typing
    pattern: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # This will be set by the event bus that creates the subscription
    _cancel_callback: Any = field(default=None, repr=False)

    async def cancel(self) -> None:
        """Cancel this subscription.

        After canceling, the handler will no longer receive events.
        """
        if self._cancel_callback:
            await self._cancel_callback(self.subscription_id)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Subscription(id={self.subscription_id!r}, "
            f"topic={self.topic!r}, pattern={self.pattern!r})"
        )
