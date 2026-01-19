"""EventBus protocol definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from .types import Event, Subscription


@runtime_checkable
class EventBus(Protocol):
    """Abstract event bus protocol supporting multiple backends.

    The EventBus provides a publish-subscribe interface for distributing
    events across components. Different implementations support various
    deployment scenarios:

    - InMemoryEventBus: Single process, no external dependencies
    - PostgresEventBus: Uses LISTEN/NOTIFY, works on local and RDS
    - RedisEventBus: Redis pub/sub, works with ElastiCache

    All implementations follow this protocol, allowing configuration-driven
    backend selection without code changes.

    Example:
        ```python
        from dataknobs_common.events import create_event_bus, Event, EventType

        # Create event bus from config
        bus = create_event_bus({"backend": "memory"})
        await bus.connect()

        # Subscribe to events
        async def handler(event: Event) -> None:
            print(f"Got {event.type} on {event.topic}")

        subscription = await bus.subscribe("registry:bots", handler)

        # Publish an event
        await bus.publish("registry:bots", Event(
            type=EventType.CREATED,
            topic="registry:bots",
            payload={"bot_id": "new-bot"}
        ))

        # Cleanup
        await subscription.cancel()
        await bus.close()
        ```

    Pattern Matching:
        Some backends support wildcard patterns for subscriptions:
        - "registry:*" matches "registry:bots", "registry:users", etc.
        - "*:created" matches "bots:created", "users:created", etc.
        Check backend documentation for supported patterns.
    """

    async def connect(self) -> None:
        """Initialize the event bus connection.

        Called before the bus is used. Should be idempotent.
        """
        ...

    async def close(self) -> None:
        """Close connections and cleanup resources.

        Should cancel all active subscriptions and release resources.
        """
        ...

    async def publish(self, topic: str, event: Event) -> None:
        """Publish an event to a topic.

        The event will be delivered to all subscribers of the topic.
        Delivery semantics depend on the backend:
        - Memory: Synchronous, in-process delivery
        - Postgres: Fire-and-forget via NOTIFY
        - Redis: Fire-and-forget via PUBLISH

        Args:
            topic: The topic/channel to publish to
            event: The event to publish
        """
        ...

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Any],
        pattern: str | None = None,
    ) -> Subscription:
        """Subscribe to events on a topic.

        The handler will be called for each event published to the topic.
        Handlers should be async functions that accept an Event.

        Args:
            topic: The topic to subscribe to
            handler: Async function to call with each event
            pattern: Optional pattern for wildcard matching (backend-specific)

        Returns:
            Subscription handle that can be used to cancel the subscription
        """
        ...


def create_event_bus(config: dict[str, Any]) -> EventBus:
    """Create an event bus from configuration.

    Factory function that creates the appropriate EventBus implementation
    based on the 'backend' key in the config.

    Args:
        config: Configuration dict with 'backend' key and backend-specific options

    Returns:
        EventBus instance

    Raises:
        ValueError: If backend type is not recognized

    Example:
        ```python
        # Memory backend (default)
        bus = create_event_bus({"backend": "memory"})

        # Postgres backend
        bus = create_event_bus({
            "backend": "postgres",
            "connection_string": "postgresql://user:pass@host/db"
        })

        # Redis backend
        bus = create_event_bus({
            "backend": "redis",
            "host": "localhost",
            "port": 6379
        })
        ```
    """
    # Import here to avoid circular imports
    from .memory import InMemoryEventBus

    backend = config.get("backend", "memory")

    if backend == "memory":
        return InMemoryEventBus()
    elif backend == "postgres":
        from .postgres import PostgresEventBus

        return PostgresEventBus(
            connection_string=config.get("connection_string", ""),
            channel_prefix=config.get("channel_prefix", "events"),
        )
    elif backend == "redis":
        from .redis import RedisEventBus

        return RedisEventBus(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            password=config.get("password"),
            ssl=config.get("ssl", False),
            channel_prefix=config.get("channel_prefix", "events"),
        )
    else:
        raise ValueError(
            f"Unknown event bus backend: {backend}. "
            f"Available backends: memory, postgres, redis"
        )
