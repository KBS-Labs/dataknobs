"""PostgreSQL event bus implementation using LISTEN/NOTIFY."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any, ClassVar, cast

from dataknobs_common.structured_config import (
    StructuredConfig,
    StructuredConfigConsumer,
)

from ._resilient_loop import run_supervised_loop
from .config import PostgresEventBusConfig
from .types import Event, Subscription

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

_LISTEN_POLL_INTERVAL = 2.0
"""Seconds between dedicated-LISTEN-connection liveness probes."""

_LISTEN_RECONNECT_TIMEOUT = 10.0
"""Bound on a single LISTEN-connection reconnect attempt (seconds)."""


class PostgresEventBus(StructuredConfigConsumer[PostgresEventBusConfig]):
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
        asyncpg: Async PostgreSQL driver
            (``pip install 'dataknobs-common[postgres]'``)
    """

    CONFIG_CLS: ClassVar[type[PostgresEventBusConfig]] = (
        PostgresEventBusConfig
    )

    def __init__(
        self,
        connection_string: str | None = None,
        channel_prefix: str | None = None,
        *,
        config: PostgresEventBusConfig | Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the Postgres event bus.

        Three construction shapes are supported:

        - **Typed config** (recommended): pass a
          :class:`PostgresEventBusConfig` via ``config=``.
        - **Loose dict config**: pass a dict via ``config=`` (normalized
          through :func:`normalize_postgres_connection_config`, so every
          input shape is supported: individual host/port/... keys,
          ``DATABASE_URL``, ``POSTGRES_*`` env-var fallbacks). May be
          combined with the legacy ``connection_string`` /
          ``channel_prefix`` positionals — those take precedence.
        - **Legacy positionals**: pass ``connection_string`` and/or
          ``channel_prefix`` directly.

        Mixing a typed :class:`PostgresEventBusConfig` with legacy
        positionals raises ``TypeError``.

        Args:
            connection_string: PostgreSQL connection string. Retained
                for backward compatibility — new callers should prefer
                ``config`` (dict or :class:`PostgresEventBusConfig`).
            channel_prefix: Prefix for NOTIFY channels
                (default: ``"events"``).
            config: Optional typed :class:`PostgresEventBusConfig` or
                dict accepted by
                :func:`normalize_postgres_connection_config`.

        Raises:
            ConfigurationError: If no postgres connection is resolvable
                from ``config``, ``connection_string``, or env vars.
            TypeError: If a typed :class:`PostgresEventBusConfig` is
                passed alongside the legacy positional kwargs.
        """
        if connection_string is not None or channel_prefix is not None:
            if isinstance(config, PostgresEventBusConfig):
                raise TypeError(
                    "PostgresEventBus: cannot mix typed "
                    "`PostgresEventBusConfig` with legacy positional "
                    "kwargs (`connection_string`, `channel_prefix`)."
                )
            merged: dict[str, Any] = dict(config or {})
            if connection_string is not None:
                merged["connection_string"] = connection_string
            if channel_prefix is not None:
                merged["channel_prefix"] = channel_prefix
            config = merged
        super().__init__(config=config)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | StructuredConfig,
    ) -> PostgresEventBus:
        """Construct from a config dict or typed config.

        Overrides
        :meth:`~dataknobs_common.structured_config.StructuredConfigConsumer.from_config`
        so the typed config is delivered via the keyword-only
        ``config=`` slot rather than the legacy
        ``connection_string`` positional.
        """
        if isinstance(config, PostgresEventBusConfig):
            return cls(config=config)
        # ``config`` is a Mapping here; mypy cannot narrow against the
        # concrete config class, so assert the residual union away.
        return cls(
            config=PostgresEventBusConfig.from_dict(
                cast("Mapping[str, Any]", config)
            )
        )

    def _setup(self) -> None:
        # Sanitize the channel prefix the same way we sanitize topics —
        # it is interpolated into LISTEN/UNLISTEN SQL statements which
        # do not support parameterized queries. Preserve the typed-
        # config invariant by replacing the dataclass with the
        # sanitized value when sanitization changed it.
        safe_prefix = re.sub(
            r"[^a-zA-Z0-9_]", "", self._config.channel_prefix
        )
        if not safe_prefix:
            raise ValueError(
                f"channel_prefix {self._config.channel_prefix!r} is "
                "empty after sanitization"
            )
        if safe_prefix != self._config.channel_prefix:
            self._config = PostgresEventBusConfig(
                connection_string=self._config.connection_string,
                channel_prefix=safe_prefix,
            )
        self._conn: Any = None  # asyncpg.Connection
        self._listen_conn: Any = None  # Separate connection for LISTEN
        self._subscriptions: dict[str, Subscription] = {}
        self._topic_channels: dict[str, str] = {}  # topic -> channel name
        self._channel_topics: dict[str, str] = {}  # channel -> topic
        self._lock = asyncio.Lock()
        self._listen_task: asyncio.Task[Any] | None = None
        self._connected = False

    def _topic_to_channel(self, topic: str) -> str:
        """Convert a topic name to a safe Postgres channel name.

        Postgres channel names must be valid identifiers. We:
        - Replace common separators (: . -) with _
        - Strip any remaining non-alphanumeric/underscore characters
        - Validate the result is non-empty
        - Add the channel prefix

        Args:
            topic: The topic name

        Returns:
            Valid Postgres channel name

        Raises:
            ValueError: If the topic produces an empty channel name
        """
        import re

        safe_topic = topic.replace(":", "_").replace(".", "_").replace("-", "_")
        # Strip any characters that are not valid in a Postgres identifier
        safe_topic = re.sub(r"[^a-zA-Z0-9_]", "", safe_topic)
        if not safe_topic:
            raise ValueError(
                f"Topic {topic!r} produces an empty channel name after sanitization"
            )
        return f"{self._config.channel_prefix}_{safe_topic}"

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
                "Install it with: pip install 'dataknobs-common[postgres]'"
            ) from e

        async with self._lock:
            # Main connection for publishing
            self._conn = await asyncpg.connect(self._config.connection_string)

            # Listener connection (separate to avoid blocking)
            self._listen_conn = await asyncpg.connect(self._config.connection_string)

            self._connected = True

            # Supervise the dedicated LISTEN connection. It is push-based
            # (asyncpg add_listener callbacks), so without this a dropped
            # _listen_conn would silently stop all delivery with no error
            # surfaced to subscribers. close() already cancels+awaits
            # self._listen_task, so this is torn down cleanly.
            self._listen_task = asyncio.create_task(
                run_supervised_loop(
                    self._reconnect_iteration,
                    should_run=lambda: self._connected,
                    name="PostgresEventBus listen-supervisor",
                )
            )
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

            # Unlisten from all channels and remove listeners
            for channel in self._channel_topics:
                try:
                    if self._listen_conn:
                        await self._listen_conn.remove_listener(
                            channel, self._notification_handler
                        )
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

        # Use pg_notify() function instead of NOTIFY statement because
        # NOTIFY is a utility statement that does not support parameterized
        # queries ($1). Using NOTIFY with $1 sends the literal string "$1"
        # as the payload, causing silent event loss.
        await self._conn.execute("SELECT pg_notify($1, $2)", channel, payload)
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
                await self._listen_conn.add_listener(
                    channel, self._notification_handler
                )
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
                        await self._listen_conn.remove_listener(
                            channel, self._notification_handler
                        )
                        await self._listen_conn.execute(f"UNLISTEN {channel}")
                except Exception:
                    pass
                del self._channel_topics[channel]
                if topic in self._topic_channels:
                    del self._topic_channels[topic]
                logger.debug("Stopped listening on channel %s", channel)

            logger.debug("Unsubscribed %s from topic %s", subscription_id[:8], topic)

    async def _listen_conn_is_alive(self) -> bool:
        """Return whether the dedicated LISTEN connection is usable.

        ``is_closed()`` is cheap but does not notice a server-side
        termination until the next query, so a lightweight ``SELECT 1``
        probe is also issued — any raised connection error means dead.

        The caller must hold ``self._lock``: the probe runs ``execute``
        on ``_listen_conn``, and asyncpg forbids concurrent operations
        on one connection, so it must be serialized against
        ``subscribe``/``_unsubscribe``'s ``LISTEN``/``UNLISTEN`` on the
        same connection. The invariant is asserted (cheap) rather than
        left implicit, so a future caller that forgets the lock fails
        loudly instead of racing asyncpg.
        """
        assert self._lock.locked(), "_listen_conn_is_alive requires self._lock"
        conn = self._listen_conn
        if conn is None or conn.is_closed():
            return False
        try:
            await conn.execute("SELECT 1")
        except Exception:
            return False
        return True

    async def _reestablish_listen_conn_locked(self) -> None:
        """Re-open ``_listen_conn`` and re-register every active channel.

        The caller must already hold ``self._lock`` (this method does
        not re-acquire it — asyncio locks are not reentrant). The new
        connection is swapped in only after every ``LISTEN`` +
        ``add_listener`` succeeds, so a partially-registered connection
        is never observed (mirrors the Redis pub/sub re-establish
        discipline). Both statements are awaited — the missing-``await``
        class of bug this guards against. ``asyncpg.connect`` is bounded
        by an explicit timeout so a wedged reconnect cannot stall
        ``close()`` (which acquires the same lock) indefinitely. The
        held-lock invariant is asserted rather than left implicit.
        """
        assert (
            self._lock.locked()
        ), "_reestablish_listen_conn_locked requires self._lock"
        import asyncpg

        new_conn = await asyncpg.connect(
            self._config.connection_string,
            timeout=_LISTEN_RECONNECT_TIMEOUT,
        )
        try:
            for channel in self._channel_topics:
                await new_conn.execute(f"LISTEN {channel}")
                await new_conn.add_listener(
                    channel, self._notification_handler
                )
        except Exception:
            try:
                await new_conn.close()
            except Exception:
                pass
            raise
        old_conn, self._listen_conn = self._listen_conn, new_conn
        channel_count = len(self._channel_topics)

        if old_conn is not None:
            try:
                await old_conn.close()
            except Exception:
                pass
        logger.warning(
            "PostgresEventBus re-established dropped LISTEN connection "
            "(%d channels re-registered)",
            channel_count,
        )

    async def _reconnect_iteration(self) -> None:
        """One supervised iteration of the LISTEN-connection watchdog.

        The liveness probe and the rebuild both touch ``_listen_conn``,
        so both run under ``self._lock`` to serialize against
        ``subscribe``/``_unsubscribe`` (asyncpg has no concurrent-op
        safety on a single connection).

        Healthy: nothing to do. Dead: rebuild and re-register; any
        failure propagates so the shared supervisor logs and backs off
        (exponential + jitter) before retrying — never giving up.

        Either way (healthy probe *or* successful rebuild) the watchdog
        then sleeps the poll interval before the next probe, so the
        cadence is uniform and a successful reconnect does not spin the
        loop back into an immediate re-probe. The sleep is outside the
        lock so it never blocks ``subscribe``/``_unsubscribe``. If the
        rebuild raised, control never reaches the sleep — the supervisor
        owns the back-off for the failure path.
        """
        async with self._lock:
            if not await self._listen_conn_is_alive():
                await self._reestablish_listen_conn_locked()
        await asyncio.sleep(_LISTEN_POLL_INTERVAL)

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

    @property
    def is_listening(self) -> bool:
        """Whether the dedicated-LISTEN-connection watchdog is running.

        ``True`` between a successful ``connect()`` and ``close()`` while
        the supervisory task is alive. Push-based delivery depends on
        that task re-establishing a dropped LISTEN connection, so this is
        the observable health signal for the delivery path (and lets
        callers/tests assert supervision without reaching into
        ``_listen_task``).
        """
        return self._listen_task is not None and not self._listen_task.done()
