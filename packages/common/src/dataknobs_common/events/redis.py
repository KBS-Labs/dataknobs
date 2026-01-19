"""Redis event bus implementation using pub/sub."""

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


class RedisEventBus:
    """Event bus using Redis pub/sub.

    This implementation uses Redis PUBLISH/SUBSCRIBE for messaging,
    which provides:

    Advantages:
    - Works with local Redis and AWS ElastiCache
    - Native pattern matching support (PSUBSCRIBE)
    - Very fast message delivery
    - Multi-instance support across processes and machines
    - Supports large payloads (up to 512MB per message)

    Limitations:
    - Requires Redis infrastructure
    - Fire-and-forget (no persistence or retry)
    - Messages lost if no subscribers are connected

    The bus uses channel prefixes to namespace events:
    - Default prefix: "events"
    - Topic "registry:bots" becomes channel "events:registry:bots"

    Example:
        ```python
        from dataknobs_common.events import RedisEventBus, Event, EventType

        bus = RedisEventBus(host="localhost", port=6379)
        await bus.connect()

        async def handler(event: Event) -> None:
            print(f"Got event: {event.type}")

        # Subscribe with pattern matching
        await bus.subscribe("registry:*", handler, pattern="registry:*")

        await bus.publish("registry:bots", Event(
            type=EventType.CREATED,
            topic="registry:bots",
            payload={"bot_id": "new-bot"}
        ))

        await bus.close()
        ```

    Requires:
        redis: Async Redis client (pip install redis)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        ssl: bool = False,
        channel_prefix: str = "events",
    ) -> None:
        """Initialize the Redis event bus.

        Args:
            host: Redis host
            port: Redis port (default: 6379)
            password: Redis password (optional)
            ssl: Whether to use SSL/TLS (for ElastiCache)
            channel_prefix: Prefix for channels (default: "events")
        """
        self._host = host
        self._port = port
        self._password = password
        self._ssl = ssl
        self._channel_prefix = channel_prefix
        self._redis: Any = None  # redis.asyncio.Redis
        self._pubsub: Any = None  # redis.asyncio.PubSub
        self._subscriptions: dict[str, Subscription] = {}
        self._channel_subscriptions: dict[str, set[str]] = {}  # channel -> sub_ids
        self._pattern_subscriptions: dict[str, set[str]] = {}  # pattern -> sub_ids
        self._lock = asyncio.Lock()
        self._listener_task: asyncio.Task[Any] | None = None
        self._connected = False
        self._running = False

    def _topic_to_channel(self, topic: str) -> str:
        """Convert a topic name to a Redis channel name.

        Args:
            topic: The topic name

        Returns:
            Redis channel name
        """
        return f"{self._channel_prefix}:{topic}"

    def _channel_to_topic(self, channel: str) -> str:
        """Convert a Redis channel name back to a topic.

        Args:
            channel: The Redis channel name

        Returns:
            Original topic name
        """
        prefix = f"{self._channel_prefix}:"
        if channel.startswith(prefix):
            return channel[len(prefix) :]
        return channel

    async def connect(self) -> None:
        """Initialize Redis connection and pubsub."""
        if self._connected:
            return

        try:
            import redis.asyncio as redis
        except ImportError as e:
            raise ImportError(
                "redis package is required for RedisEventBus. "
                "Install it with: pip install redis"
            ) from e

        async with self._lock:
            ssl_context = True if self._ssl else None

            self._redis = redis.Redis(
                host=self._host,
                port=self._port,
                password=self._password,
                ssl=ssl_context,
                decode_responses=True,
            )

            # Test connection
            await self._redis.ping()

            # Create pubsub connection
            self._pubsub = self._redis.pubsub()

            self._connected = True
            self._running = True

            # Start the message listener
            self._listener_task = asyncio.create_task(self._message_listener())

            logger.info(
                "RedisEventBus connected to %s:%d",
                self._host,
                self._port,
            )

    async def close(self) -> None:
        """Close connections and cleanup resources."""
        async with self._lock:
            self._running = False

            if self._listener_task:
                self._listener_task.cancel()
                try:
                    await self._listener_task
                except asyncio.CancelledError:
                    pass
                self._listener_task = None

            if self._pubsub:
                # Unsubscribe from all channels and patterns
                try:
                    await self._pubsub.unsubscribe()
                    await self._pubsub.punsubscribe()
                    await self._pubsub.close()
                except Exception:
                    pass
                self._pubsub = None

            if self._redis:
                await self._redis.close()
                self._redis = None

            self._subscriptions.clear()
            self._channel_subscriptions.clear()
            self._pattern_subscriptions.clear()
            self._connected = False
            logger.info("RedisEventBus closed")

    async def publish(self, topic: str, event: Event) -> None:
        """Publish an event using Redis PUBLISH.

        Args:
            topic: The topic to publish to
            event: The event to publish

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected or not self._redis:
            raise RuntimeError("RedisEventBus not connected")

        channel = self._topic_to_channel(topic)
        payload = json.dumps(event.to_dict())

        await self._redis.publish(channel, payload)
        logger.debug("Published event %s to channel %s", event.event_id[:8], channel)

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Any],
        pattern: str | None = None,
    ) -> Subscription:
        """Subscribe to events on a topic.

        If a pattern is provided, uses PSUBSCRIBE for pattern matching.
        Redis patterns use glob-style syntax:
        - * matches any sequence of characters
        - ? matches any single character
        - [abc] matches any character in the brackets

        Args:
            topic: The topic to subscribe to
            handler: Async function to call with each event
            pattern: Optional wildcard pattern (uses Redis PSUBSCRIBE)

        Returns:
            Subscription handle
        """
        if not self._connected or not self._pubsub:
            raise RuntimeError("RedisEventBus not connected")

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
                # Pattern subscription using PSUBSCRIBE
                channel_pattern = self._topic_to_channel(pattern)
                if channel_pattern not in self._pattern_subscriptions:
                    await self._pubsub.psubscribe(channel_pattern)
                    self._pattern_subscriptions[channel_pattern] = set()
                    logger.debug("Started pattern subscription on %s", channel_pattern)
                self._pattern_subscriptions[channel_pattern].add(subscription_id)
            else:
                # Exact channel subscription
                channel = self._topic_to_channel(topic)
                if channel not in self._channel_subscriptions:
                    await self._pubsub.subscribe(channel)
                    self._channel_subscriptions[channel] = set()
                    logger.debug("Started subscription on channel %s", channel)
                self._channel_subscriptions[channel].add(subscription_id)

        logger.debug(
            "Subscribed %s to topic %s (pattern=%s)",
            subscription_id[:8],
            topic,
            pattern,
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
                channel_pattern = self._topic_to_channel(sub.pattern)
                if channel_pattern in self._pattern_subscriptions:
                    self._pattern_subscriptions[channel_pattern].discard(subscription_id)
                    if not self._pattern_subscriptions[channel_pattern]:
                        try:
                            if self._pubsub:
                                await self._pubsub.punsubscribe(channel_pattern)
                        except Exception:
                            pass
                        del self._pattern_subscriptions[channel_pattern]
                        logger.debug(
                            "Stopped pattern subscription on %s",
                            channel_pattern,
                        )
            else:
                channel = self._topic_to_channel(sub.topic)
                if channel in self._channel_subscriptions:
                    self._channel_subscriptions[channel].discard(subscription_id)
                    if not self._channel_subscriptions[channel]:
                        try:
                            if self._pubsub:
                                await self._pubsub.unsubscribe(channel)
                        except Exception:
                            pass
                        del self._channel_subscriptions[channel]
                        logger.debug("Stopped subscription on channel %s", channel)

            logger.debug("Unsubscribed %s", subscription_id[:8])

    async def _message_listener(self) -> None:
        """Listen for incoming messages and dispatch to handlers."""
        while self._running and self._pubsub:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if message:
                    await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in Redis message listener")
                await asyncio.sleep(1.0)  # Back off on error

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle an incoming Redis message.

        Args:
            message: The Redis message dict
        """
        message_type = message.get("type")
        if message_type not in ("message", "pmessage"):
            return

        channel = message.get("channel", "")
        pattern = message.get("pattern")  # Only for pmessage
        data = message.get("data", "")

        # Parse the event
        try:
            event_data = json.loads(data)
            event = Event.from_dict(event_data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse Redis message: %s", e)
            return

        topic = self._channel_to_topic(channel)

        # Dispatch to handlers
        await self._dispatch_event(topic, event, pattern)

    async def _dispatch_event(
        self,
        topic: str,
        event: Event,
        pattern: str | None = None,
    ) -> None:
        """Dispatch an event to matching handlers.

        Args:
            topic: The topic the event was published to
            event: The event to dispatch
            pattern: The pattern that matched (for pattern subscriptions)
        """
        handlers_to_call: list[tuple[str, Callable[[Event], Any]]] = []

        async with self._lock:
            # Check exact channel subscriptions
            channel = self._topic_to_channel(topic)
            if channel in self._channel_subscriptions:
                for sub_id in self._channel_subscriptions[channel]:
                    if sub_id in self._subscriptions:
                        sub = self._subscriptions[sub_id]
                        handlers_to_call.append((sub_id, sub.handler))

            # Check pattern subscriptions
            if pattern:
                if pattern in self._pattern_subscriptions:
                    for sub_id in self._pattern_subscriptions[pattern]:
                        if sub_id in self._subscriptions:
                            sub = self._subscriptions[sub_id]
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
