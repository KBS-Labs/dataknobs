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
    based on the 'backend' key in the config. Backends are resolved through
    the :data:`~dataknobs_common.events.registry.event_bus_backends`
    registry, so out-of-tree consumers can register and select a custom
    ``EventBus`` backend without forking DataKnobs:

        ```python
        from dataknobs_common.events import event_bus_backends, create_event_bus

        event_bus_backends.register("kafka", my_kafka_factory)
        bus = create_event_bus({"backend": "kafka", "brokers": "..."})
        ```

    Args:
        config: Configuration dict with 'backend' key and backend-specific options

    Returns:
        EventBus instance

    Raises:
        ValueError: If the backend is not registered. The message lists all
            registered backends (including consumer-registered ones).
        OperationError: If the backend factory raises during construction
            (invalid config, missing required fields, etc.). Wraps the
            originating exception via ``__cause__``. This includes the
            ``ValueError`` raised by ``SqsEventBusConfig`` when ``queue_url``
            is missing.

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
    # Imported here (not at module load) so registry.py can safely
    # ``from .bus import EventBus`` without a module-load cycle.
    from .registry import event_bus_backends

    return event_bus_backends.create(config=config)


async def create_event_bus_async(config: dict[str, Any]) -> EventBus:
    """Async-symmetric counterpart to :func:`create_event_bus`.

    For backends whose construction is asynchronous (eager-connecting
    pools, LLM-warmed wrappers, …). Today every built-in backend
    constructs synchronously, so this function returns the same instance
    type as :func:`create_event_bus`; the surface is shipped for API
    symmetry and consumer-extensibility (an out-of-tree backend's
    ``from_config_async`` is detected and awaited via
    :meth:`PluginRegistry.create_async`).

    Args:
        config: Configuration dict with a 'backend' key and
            backend-specific options.

    Returns:
        EventBus instance.

    Raises:
        ValueError: If the backend is not registered. The message lists
            all registered backends (including consumer-registered ones).
        OperationError: If the backend factory raises during construction
            (invalid config, missing required fields, etc.). Wraps the
            originating exception via ``__cause__``. Same behaviour as
            the sync :func:`create_event_bus`.
    """
    from .registry import event_bus_backends

    return await event_bus_backends.create_async(config=config)
