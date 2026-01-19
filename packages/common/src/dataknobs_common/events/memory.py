"""In-memory event bus implementation."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import uuid
from typing import TYPE_CHECKING, Any

from .types import Event, Subscription

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class InMemoryEventBus:
    """Simple in-memory pub/sub event bus.

    This implementation is suitable for:
    - Single-process applications
    - Development and testing
    - Scenarios where events don't need to cross process boundaries

    Features:
    - Fast, synchronous delivery within the same process
    - Support for wildcard patterns using fnmatch
    - Thread-safe using asyncio.Lock

    Limitations:
    - Events are not persisted
    - Does not work across multiple processes or machines
    - Events are lost if there are no subscribers

    Example:
        ```python
        from dataknobs_common.events import InMemoryEventBus, Event, EventType

        bus = InMemoryEventBus()
        await bus.connect()

        events_received = []

        async def handler(event: Event) -> None:
            events_received.append(event)

        # Subscribe with pattern matching
        await bus.subscribe("registry:*", handler, pattern="registry:*")

        # Publish
        await bus.publish("registry:bots", Event(
            type=EventType.CREATED,
            topic="registry:bots",
            payload={"bot_id": "test"}
        ))

        assert len(events_received) == 1
        await bus.close()
        ```
    """

    def __init__(self) -> None:
        """Initialize the in-memory event bus."""
        self._subscriptions: dict[str, Subscription] = {}
        self._topic_subscribers: dict[str, set[str]] = {}  # topic -> subscription_ids
        self._pattern_subscribers: list[tuple[str, str]] = []  # (pattern, sub_id)
        self._lock = asyncio.Lock()
        self._connected = False

    async def connect(self) -> None:
        """Initialize the event bus.

        For in-memory bus, this just sets the connected flag.
        """
        async with self._lock:
            self._connected = True
            logger.debug("InMemoryEventBus connected")

    async def close(self) -> None:
        """Close the event bus and cancel all subscriptions."""
        async with self._lock:
            self._subscriptions.clear()
            self._topic_subscribers.clear()
            self._pattern_subscribers.clear()
            self._connected = False
            logger.debug("InMemoryEventBus closed")

    async def publish(self, topic: str, event: Event) -> None:
        """Publish an event to a topic.

        Delivers the event to all subscribers of the topic and any
        pattern subscribers that match.

        Args:
            topic: The topic to publish to
            event: The event to publish
        """
        if not self._connected:
            logger.warning("Publishing to disconnected event bus")

        handlers_to_call: list[tuple[str, Callable[[Event], Any]]] = []

        async with self._lock:
            # Get exact topic subscribers
            if topic in self._topic_subscribers:
                for sub_id in self._topic_subscribers[topic]:
                    if sub_id in self._subscriptions:
                        sub = self._subscriptions[sub_id]
                        handlers_to_call.append((sub_id, sub.handler))

            # Get pattern subscribers
            for pattern, sub_id in self._pattern_subscribers:
                if fnmatch.fnmatch(topic, pattern):
                    if sub_id in self._subscriptions:
                        sub = self._subscriptions[sub_id]
                        handlers_to_call.append((sub_id, sub.handler))

        # Call handlers outside the lock to avoid deadlocks
        for sub_id, handler in handlers_to_call:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Error in event handler for subscription %s",
                    sub_id,
                )

        logger.debug(
            "Published event %s to topic %s, delivered to %d handlers",
            event.event_id[:8],
            topic,
            len(handlers_to_call),
        )

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Any],
        pattern: str | None = None,
    ) -> Subscription:
        """Subscribe to events on a topic.

        Args:
            topic: The topic to subscribe to
            handler: Function to call with each event
            pattern: Optional wildcard pattern (uses fnmatch syntax)

        Returns:
            Subscription handle for managing the subscription
        """
        subscription_id = str(uuid.uuid4())

        subscription = Subscription(
            subscription_id=subscription_id,
            topic=topic,
            handler=handler,
            pattern=pattern,
            _cancel_callback=self._unsubscribe,
        )

        async with self._lock:
            self._subscriptions[subscription_id] = subscription

            if pattern:
                # Pattern-based subscription
                self._pattern_subscribers.append((pattern, subscription_id))
                logger.debug(
                    "Subscribed %s to pattern %s",
                    subscription_id[:8],
                    pattern,
                )
            else:
                # Exact topic subscription
                if topic not in self._topic_subscribers:
                    self._topic_subscribers[topic] = set()
                self._topic_subscribers[topic].add(subscription_id)
                logger.debug(
                    "Subscribed %s to topic %s",
                    subscription_id[:8],
                    topic,
                )

        return subscription

    async def _unsubscribe(self, subscription_id: str) -> None:
        """Cancel a subscription.

        Args:
            subscription_id: The subscription to cancel
        """
        async with self._lock:
            if subscription_id not in self._subscriptions:
                return

            sub = self._subscriptions.pop(subscription_id)

            if sub.pattern:
                # Remove from pattern subscribers
                self._pattern_subscribers = [
                    (p, sid)
                    for p, sid in self._pattern_subscribers
                    if sid != subscription_id
                ]
            else:
                # Remove from topic subscribers
                if sub.topic in self._topic_subscribers:
                    self._topic_subscribers[sub.topic].discard(subscription_id)
                    if not self._topic_subscribers[sub.topic]:
                        del self._topic_subscribers[sub.topic]

            logger.debug("Unsubscribed %s", subscription_id[:8])

    @property
    def subscription_count(self) -> int:
        """Get the number of active subscriptions."""
        return len(self._subscriptions)
