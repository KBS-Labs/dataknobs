"""PostgreSQL event bus implementation using LISTEN/NOTIFY."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from .types import Event, Subscription

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class PostgresEventBus:
    """Event bus using PostgreSQL LISTEN/NOTIFY.

    This implementation uses Postgres LISTEN/NOTIFY for pub/sub messaging,
    which provides:

    Advantages:
    - No additional infrastructure (reuses existing database)
    - Works on local Postgres and AWS RDS
    - Transactional consistency with database operations
    - Multi-instance support (all instances sharing the same DB)

    Limitations:
    - Payload size limited to ~8000 bytes (use for metadata, not bulk data)
    - Fire-and-forget (no message persistence or retry)
    - Requires a dedicated connection for LISTEN
    - No native pattern matching (implemented in Python)

    The bus uses channel prefixes to namespace events:
    - Default prefix: "events"
    - Topic "registry:bots" becomes channel "events_registry_bots"

    Example:
        ```python
        from dataknobs_common.events import PostgresEventBus, Event, EventType

        bus = PostgresEventBus(
            connection_string="postgresql://user:pass@localhost/mydb"
        )
        await bus.connect()

        async def handler(event: Event) -> None:
            print(f"Got event: {event.type}")

        await bus.subscribe("registry:bots", handler)

        await bus.publish("registry:bots", Event(
            type=EventType.CREATED,
            topic="registry:bots",
            payload={"bot_id": "new-bot"}
        ))

        await bus.close()
        ```

    Requires:
        asyncpg: Async PostgreSQL driver (pip install asyncpg)
    """

    def __init__(
        self,
        connection_string: str,
        channel_prefix: str = "events",
    ) -> None:
        """Initialize the Postgres event bus.

        Args:
            connection_string: PostgreSQL connection string
            channel_prefix: Prefix for NOTIFY channels (default: "events")
        """
        self._connection_string = connection_string
        self._channel_prefix = channel_prefix
        self._conn: Any = None  # asyncpg.Connection
        self._listen_conn: Any = None  # Separate connection for LISTEN
        self._subscriptions: dict[str, Subscription] = {}
        self._topic_channels: dict[str, str] = {}  # topic -> channel name
        self._channel_topics: dict[str, str] = {}  # channel -> topic
        self._lock = asyncio.Lock()
        self._listen_task: asyncio.Task[Any] | None = None
        self._connected = False

    def _topic_to_channel(self, topic: str) -> str:
        """Convert a topic name to a Postgres channel name.

        Postgres channel names must be valid identifiers, so we:
        - Replace : and . with _
        - Add the channel prefix

        Args:
            topic: The topic name

        Returns:
            Valid Postgres channel name
        """
        safe_topic = topic.replace(":", "_").replace(".", "_").replace("-", "_")
        return f"{self._channel_prefix}_{safe_topic}"

    async def connect(self) -> None:
        """Initialize database connections.

        Creates two connections:
        - Main connection for NOTIFY (publishing)
        - Listener connection for LISTEN (subscribing)
        """
        if self._connected:
            return

        try:
            import asyncpg
        except ImportError as e:
            raise ImportError(
                "asyncpg is required for PostgresEventBus. "
                "Install it with: pip install asyncpg"
            ) from e

        async with self._lock:
            # Main connection for publishing
            self._conn = await asyncpg.connect(self._connection_string)

            # Listener connection (separate to avoid blocking)
            self._listen_conn = await asyncpg.connect(self._connection_string)

            # Add notification handler
            self._listen_conn.add_listener("*", self._notification_handler)

            self._connected = True
            logger.info("PostgresEventBus connected")

    async def close(self) -> None:
        """Close connections and cleanup resources."""
        async with self._lock:
            if self._listen_task:
                self._listen_task.cancel()
                try:
                    await self._listen_task
                except asyncio.CancelledError:
                    pass

            # Unlisten from all channels
            for channel in self._channel_topics:
                try:
                    if self._listen_conn:
                        await self._listen_conn.execute(f"UNLISTEN {channel}")
                except Exception:
                    pass

            if self._listen_conn:
                await self._listen_conn.close()
                self._listen_conn = None

            if self._conn:
                await self._conn.close()
                self._conn = None

            self._subscriptions.clear()
            self._topic_channels.clear()
            self._channel_topics.clear()
            self._connected = False
            logger.info("PostgresEventBus closed")

    async def publish(self, topic: str, event: Event) -> None:
        """Publish an event using NOTIFY.

        Args:
            topic: The topic to publish to
            event: The event to publish

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected or not self._conn:
            raise RuntimeError("PostgresEventBus not connected")

        channel = self._topic_to_channel(topic)
        payload = json.dumps(event.to_dict())

        # Postgres NOTIFY payload limit is ~8000 bytes
        if len(payload) > 7500:
            logger.warning(
                "Event payload for topic %s is %d bytes, "
                "may exceed Postgres NOTIFY limit",
                topic,
                len(payload),
            )

        await self._conn.execute(f"NOTIFY {channel}, $1", payload)
        logger.debug("Published event %s to channel %s", event.event_id[:8], channel)

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Any],
        pattern: str | None = None,
    ) -> Subscription:
        """Subscribe to events on a topic.

        Note: Pattern matching is implemented in Python, not using Postgres
        features. The pattern uses fnmatch syntax.

        Args:
            topic: The topic to subscribe to
            handler: Async function to call with each event
            pattern: Optional wildcard pattern (handled in Python)

        Returns:
            Subscription handle
        """
        if not self._connected or not self._listen_conn:
            raise RuntimeError("PostgresEventBus not connected")

        subscription_id = str(uuid.uuid4())
        channel = self._topic_to_channel(topic)

        subscription = Subscription(
            subscription_id=subscription_id,
            topic=topic,
            handler=handler,
            pattern=pattern,
            _cancel_callback=self._unsubscribe,
        )

        async with self._lock:
            self._subscriptions[subscription_id] = subscription

            # Start listening on this channel if not already
            if channel not in self._channel_topics:
                await self._listen_conn.execute(f"LISTEN {channel}")
                self._channel_topics[channel] = topic
                self._topic_channels[topic] = channel
                logger.debug("Started listening on channel %s", channel)

        logger.debug(
            "Subscribed %s to topic %s (channel %s)",
            subscription_id[:8],
            topic,
            channel,
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
            topic = sub.topic
            channel = self._topic_to_channel(topic)

            # Check if any other subscriptions are using this channel
            has_other_subs = any(
                s.topic == topic for s in self._subscriptions.values()
            )

            if not has_other_subs and channel in self._channel_topics:
                try:
                    if self._listen_conn:
                        await self._listen_conn.execute(f"UNLISTEN {channel}")
                except Exception:
                    pass
                del self._channel_topics[channel]
                if topic in self._topic_channels:
                    del self._topic_channels[topic]
                logger.debug("Stopped listening on channel %s", channel)

            logger.debug("Unsubscribed %s from topic %s", subscription_id[:8], topic)

    def _notification_handler(
        self,
        connection: Any,
        pid: int,
        channel: str,
        payload: str,
    ) -> None:
        """Handle incoming Postgres notifications.

        This is called by asyncpg when a NOTIFY is received.
        We dispatch to the appropriate handlers.
        """
        try:
            event_data = json.loads(payload)
            event = Event.from_dict(event_data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse notification payload: %s", e)
            return

        # Find subscribers for this channel
        topic = self._channel_topics.get(channel)
        if not topic:
            return

        # Dispatch to handlers (in a new task to not block the notification handler)
        asyncio.create_task(self._dispatch_event(topic, event))

    async def _dispatch_event(self, topic: str, event: Event) -> None:
        """Dispatch an event to all matching subscribers.

        Args:
            topic: The topic the event was published to
            event: The event to dispatch
        """
        import fnmatch

        handlers_to_call: list[tuple[str, Callable[[Event], Any]]] = []

        async with self._lock:
            for sub_id, sub in self._subscriptions.items():
                if sub.pattern:
                    if fnmatch.fnmatch(topic, sub.pattern):
                        handlers_to_call.append((sub_id, sub.handler))
                elif sub.topic == topic:
                    handlers_to_call.append((sub_id, sub.handler))

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
            "Dispatched event %s to %d handlers",
            event.event_id[:8],
            len(handlers_to_call),
        )

    @property
    def subscription_count(self) -> int:
        """Get the number of active subscriptions."""
        return len(self._subscriptions)
