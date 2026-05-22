"""AWS SQS event bus implementation.

A single SQS queue carries every topic; the topic name travels in a
message attribute (configurable, default ``"topic"``) and each
subscriber long-polls and filters by exact-match on that attribute.

Delivery is **at-least-once** (native SQS semantics): a handler that
raises does *not* delete its message, so the message reappears after
the queue's visibility timeout and is redelivered. **Handlers must be
idempotent.** ``IngestOrchestrator.ingest_if_changed`` already is, which
is why single-queue at-least-once is sufficient for the trigger path.

``aioboto3`` is an *optional* dependency. The base ``dataknobs-common``
install does not pull it; it is imported lazily inside :meth:`connect`
so importing this module (or merely registering the ``"sqs"`` backend)
never requires it. Install with::

    pip install 'dataknobs-common[sqs]'
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from dataknobs_common.structured_config import StructuredConfigConsumer

from ._resilient_loop import run_supervised_loop
from .config import SqsEventBusConfig
from .types import Event, EventType, Subscription

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class SqsEventBus(StructuredConfigConsumer[SqsEventBusConfig]):
    """Event bus backed by a single AWS SQS queue.

    Topic routing: every :class:`Event` is sent to the one
    ``queue_url`` with the topic carried in a ``String`` message
    attribute named ``topic_attribute``. A subscription long-polls the
    queue and dispatches only messages whose attribute equals its
    topic; non-matching messages are left untouched (not deleted) so a
    subscriber on a different topic can still receive them after the
    visibility timeout. See **Single-topic bridge mode** below for the
    opt-in handling of messages that arrive *without* a topic
    attribute (AWS-native bridges).

    Advantages:
    - AWS-native, fully managed; no broker to operate
    - Durable + at-least-once; survives subscriber restarts
    - Works behind an SNS→SQS fan-out with no code change

    Limitations:
    - At-least-once only — handlers MUST be idempotent
    - Single-queue topic-attribute routing (queue-per-topic is out of
      scope); an unconsumed topic's messages recirculate until their
      retention expires
    - Wildcard ``pattern`` subscriptions are unsupported (raise
      ``NotImplementedError`` — loud rather than silently mis-routing)

    Single-topic bridge mode:
        When ``require_topic_attribute=False`` is set, the bus is
        dedicated to a single topic — ``subscribe`` enforces this by
        raising ``ValueError`` if a second subscription is attempted.
        Use this mode for queues fed by AWS-native event sources
        (EventBridge → SQS targets, S3 → SQS bucket notifications,
        raw SNS → SQS delivery) that cannot set arbitrary SQS message
        attributes. Attribute-less messages are dispatched to the one
        subscription; bodies that are valid JSON but not
        ``Event.to_dict()``-shaped are delivered as synthesised
        ``Event(type=EventType.CUSTOM, topic=<the subscription's
        topic>, payload=<decoded-body>)`` events with
        ``event_id="sqs:<MessageId>"`` and
        ``metadata={"sqs_message_id": ..., "sqs_synthesised": True}``
        — stable across redeliveries so handler idempotency keyed on
        ``event_id`` works. Messages with a matching attribute
        continue to dispatch by attribute; messages with a mismatched
        attribute are still released back to the queue.

    Example:
        ```python
        from dataknobs_common.events import Event, EventType
        from dataknobs_common.events.sqs import SqsEventBus

        bus = SqsEventBus(queue_url="https://sqs.us-east-1.../events")
        await bus.connect()

        async def handler(event: Event) -> None:
            ...  # must be idempotent

        await bus.subscribe("registry:bots", handler)
        await bus.publish("registry:bots", Event(
            type=EventType.CREATED,
            topic="registry:bots",
            payload={"bot_id": "new-bot"},
        ))

        await bus.close()
        ```

    Construction shapes are provided by
    :class:`~dataknobs_common.structured_config.StructuredConfigConsumer`:
    a typed :class:`SqsEventBusConfig`, a dict via ``config=``, or
    loose ``**kwargs`` (``queue_url=...``, ``region=...``, etc.).
    Mixing typed ``config=`` with loose kwargs raises ``TypeError``.

    Requires:
        aioboto3: ``pip install 'dataknobs-common[sqs]'``
    """

    CONFIG_CLS: ClassVar[type[SqsEventBusConfig]] = SqsEventBusConfig

    def _setup(self) -> None:
        # ``__post_init__`` has already validated queue_url; derived
        # attribute follows directly from the typed config.
        self._is_fifo = self._config.queue_url.endswith(".fifo")

        self._session: Any = None  # aioboto3.Session
        self._client: Any = None  # aiobotocore SQS client
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._subscriptions: dict[str, Subscription] = {}
        self._poll_tasks: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()
        self._connected = False
        self._running = False

    @property
    def require_topic_attribute(self) -> bool:
        """Whether attribute-less messages release back to the queue.

        ``True`` (default): attribute-less messages are released back to
        the queue (multi-topic safety mode). ``False`` (single-topic
        bridge mode): they are dispatched to the bus's single
        subscription instead. See :meth:`__init__` for the full
        behaviour matrix.

        Shortcut for ``bus.config.require_topic_attribute``.
        """
        return self._config.require_topic_attribute

    def _session_kwargs(self) -> dict[str, Any]:
        """Build ``aioboto3.Session`` kwargs.

        Only explicitly provided values are set; everything else is left
        to boto's default region/credential chain. This is the minimal
        local passthrough that replaces an (illegal) ``dataknobs-data``
        import — ``dataknobs-common`` is the lowest layer and must not
        depend on a higher one.
        """
        kwargs: dict[str, Any] = {}
        if self._config.region:
            kwargs["region_name"] = self._config.region
        if self._config.aws_access_key_id and self._config.aws_secret_access_key:
            kwargs["aws_access_key_id"] = self._config.aws_access_key_id
            kwargs["aws_secret_access_key"] = self._config.aws_secret_access_key
        return kwargs

    def _client_kwargs(self) -> dict[str, Any]:
        """Build SQS client kwargs (endpoint + explicit timeouts).

        Read timeout exceeds the long-poll wait so a full
        ``WaitTimeSeconds`` receive never trips the socket read timeout
        (security rule 2 — every external call has an explicit timeout).
        """
        from botocore.config import Config

        kwargs: dict[str, Any] = {
            "config": Config(
                connect_timeout=10,
                read_timeout=self._config.wait_time_seconds + 10,
                retries={"max_attempts": 3, "mode": "standard"},
            )
        }
        if self._config.endpoint_url:
            kwargs["endpoint_url"] = self._config.endpoint_url
        return kwargs

    async def connect(self) -> None:
        """Create the SQS client. Idempotent.

        Raises:
            ImportError: If the optional ``aioboto3`` dependency is not
                installed.
        """
        if self._connected:
            return

        try:
            import aioboto3
        except ImportError as e:
            raise ImportError(
                "aioboto3 is required for SqsEventBus. "
                "Install it with: pip install 'dataknobs-common[sqs]'"
            ) from e

        async with self._lock:
            if self._connected:
                return
            self._session = aioboto3.Session(**self._session_kwargs())
            self._exit_stack = contextlib.AsyncExitStack()
            self._client = await self._exit_stack.enter_async_context(
                self._session.client("sqs", **self._client_kwargs())
            )
            self._connected = True
            self._running = True
            logger.info(
                "SqsEventBus connected to queue %s (fifo=%s)",
                self._config.queue_url,
                self._is_fifo,
            )

    async def close(self) -> None:
        """Cancel all poll tasks and close the client. Idempotent."""
        async with self._lock:
            self._running = False

            tasks = list(self._poll_tasks.values())
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self._poll_tasks.clear()

            if self._exit_stack is not None:
                with contextlib.suppress(Exception):
                    await self._exit_stack.aclose()
                self._exit_stack = None
            self._client = None
            self._session = None
            self._subscriptions.clear()
            self._connected = False
            logger.info("SqsEventBus closed for queue %s", self._config.queue_url)

    async def publish(self, topic: str, event: Event) -> None:
        """Publish an event via ``SendMessage``.

        The serialized :class:`Event` is the message body; the topic is
        a ``String`` message attribute. For a FIFO queue the
        ``MessageGroupId`` is the topic (per-topic ordering) and the
        ``MessageDeduplicationId`` is the event id.

        Args:
            topic: The topic to publish to.
            event: The event to publish.

        Raises:
            RuntimeError: If the bus is not connected.
        """
        if not self._connected or self._client is None:
            raise RuntimeError("SqsEventBus not connected")

        params: dict[str, Any] = {
            "QueueUrl": self._config.queue_url,
            "MessageBody": json.dumps(event.to_dict()),
            "MessageAttributes": {
                self._config.topic_attribute: {
                    "StringValue": topic,
                    "DataType": "String",
                }
            },
        }
        if self._is_fifo:
            params["MessageGroupId"] = topic
            params["MessageDeduplicationId"] = event.event_id

        await self._client.send_message(**params)
        logger.debug(
            "Published event %s to topic %s (queue=%s)",
            event.event_id[:8],
            topic,
            self._config.queue_url,
        )

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Event], Any],
        pattern: str | None = None,
    ) -> Subscription:
        """Subscribe to a topic with a dedicated long-poll task.

        Lazily ensures the client (so ``subscribe`` works before an
        explicit ``connect()``), then starts one poll task that receives
        from the shared queue and dispatches only messages whose topic
        attribute equals ``topic``.

        Args:
            topic: The exact topic to receive.
            handler: Sync or async callable invoked per matching event.
            pattern: Unsupported — any non-``None`` value raises
                ``NotImplementedError`` (single-queue attribute routing
                has no wildcard semantics; failing loudly prevents
                silent mis-routing).

        Returns:
            A :class:`Subscription`; its ``cancel()`` stops the task.

        Raises:
            NotImplementedError: If ``pattern`` is not ``None``.
            ValueError: When ``require_topic_attribute=False`` (bridge
                mode) and a subscription already exists on this bus.
                Bridge mode is dedicated-to-a-single-topic by contract;
                a second subscription would make synthesised events'
                ``topic`` field non-deterministic (it carries the
                receiving poll task's topic) and would silently
                cross-deliver attribute-less messages.
        """
        if pattern is not None:
            raise NotImplementedError(
                "SqsEventBus does not support wildcard pattern "
                "subscriptions; subscribe to an exact topic instead"
            )

        if not self._connected:
            await self.connect()

        subscription_id = str(uuid.uuid4())
        subscription = Subscription(
            subscription_id=subscription_id,
            topic=topic,
            handler=handler,
            pattern=None,
            _cancel_callback=self._unsubscribe,
        )

        async with self._lock:
            if (
                not self._config.require_topic_attribute
                and self._subscriptions
            ):
                existing_topics = sorted(
                    {sub.topic for sub in self._subscriptions.values()}
                )
                raise ValueError(
                    "SqsEventBus is in single-topic bridge mode "
                    "(require_topic_attribute=False) and already has a "
                    f"subscription on topic(s) {existing_topics}; a "
                    "second subscription is not allowed because the "
                    "queue is dedicated to one topic. Open a separate "
                    "bus per bridge queue."
                )
            self._subscriptions[subscription_id] = subscription
            self._poll_tasks[subscription_id] = asyncio.create_task(
                self._poll_loop(subscription_id, topic, handler)
            )

        logger.debug(
            "Subscribed %s to topic %s (queue=%s)",
            subscription_id[:8],
            topic,
            self._config.queue_url,
        )
        return subscription

    async def _unsubscribe(self, subscription_id: str) -> None:
        """Cancel a subscription's poll task and forget it."""
        async with self._lock:
            self._subscriptions.pop(subscription_id, None)
            task = self._poll_tasks.pop(subscription_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        logger.debug("Unsubscribed %s", subscription_id[:8])

    async def _poll_loop(
        self,
        subscription_id: str,
        topic: str,
        handler: Callable[[Event], Any],
    ) -> None:
        """Long-poll the queue, dispatch matching events for one topic.

        The supervised-loop helper owns the ``while self._running``
        lifecycle, cancellation, and the exponential-with-jitter
        back-off; this method only supplies one poll-and-dispatch
        iteration. Cancellation / ``_running`` / ``_unsubscribe``
        semantics are unchanged for callers.
        """

        async def _one() -> None:
            response = await self._client.receive_message(
                QueueUrl=self._config.queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=self._config.wait_time_seconds,
                VisibilityTimeout=self._config.visibility_timeout,
                MessageAttributeNames=[self._config.topic_attribute],
            )
            for message in response.get("Messages", []):
                await self._handle_message(
                    message, subscription_id, topic, handler
                )

        await run_supervised_loop(
            _one,
            should_run=lambda: self._running,
            name=f"SqsEventBus poll {subscription_id[:8]}",
        )

    def _extract_topic_attribute(
        self, message: dict[str, Any]
    ) -> str | None:
        """Pull the topic attribute's ``StringValue`` from an SQS message.

        Returns ``None`` when the attribute is absent (the message was
        published without it, or the source is an AWS-native bridge
        that cannot set arbitrary attributes — EventBridge target, S3
        bucket notification, raw SNS → SQS delivery).
        """
        attrs = message.get("MessageAttributes") or {}
        topic_attr = attrs.get(self._config.topic_attribute) or {}
        value = topic_attr.get("StringValue")
        return value if isinstance(value, str) else None

    async def _release_visibility(
        self, receipt_handle: str | None
    ) -> None:
        """Return a message to the queue immediately (visibility 0).

        Best-effort: a benign expired / invalid-receipt-handle race
        with another subscriber's delete is suppressed so a transient
        SQS error never propagates into the supervised poll loop as a
        failure.
        """
        if not receipt_handle:
            return
        with contextlib.suppress(Exception):
            await self._client.change_message_visibility(
                QueueUrl=self._config.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=0,
            )

    async def _handle_message(
        self,
        message: dict[str, Any],
        subscription_id: str,
        topic: str,
        handler: Callable[[Event], Any],
    ) -> None:
        """Filter / dispatch a single SQS message.

        Three structural cases on the configured topic attribute:

        - Attribute present and matches this subscription's topic →
          parse + dispatch + delete on success (at-least-once retains
          the message on handler failure).
        - Attribute present but does NOT match → release the message
          back to the queue with visibility 0 so another subscriber
          picks it up.
        - Attribute absent → when ``require_topic_attribute`` is
          ``True`` (default), behave as "mismatched" and release. When
          ``False`` (single-topic bridge mode), dispatch to every
          active subscription on the bus.
        """
        msg_topic = self._extract_topic_attribute(message)
        receipt_handle = message.get("ReceiptHandle")

        if msg_topic is None:
            if not self._config.require_topic_attribute:
                await self._dispatch_to_all(
                    message, subscription_id, topic, receipt_handle
                )
                return
            # Default mode: absent attribute is "not for this
            # subscription" → release (same outcome as a mismatched
            # attribute under the existing single-queue routing model).
            await self._release_visibility(receipt_handle)
            return

        if msg_topic != topic:
            # Another topic's message; let the matching subscriber
            # grab it on its next poll.
            await self._release_visibility(receipt_handle)
            return

        await self._dispatch_single(
            message, subscription_id, topic, handler, receipt_handle
        )

    async def _dispatch_single(
        self,
        message: dict[str, Any],
        subscription_id: str,
        topic: str,
        handler: Callable[[Event], Any],
        receipt_handle: str | None,
    ) -> None:
        """Parse, invoke handler, delete-on-success for one matching message.

        Behaviour identical to the body of pre-refactor
        ``_handle_message`` after the topic-filter check.
        """
        message_id = message.get("MessageId", "")

        try:
            event = Event.from_dict(json.loads(message["Body"]))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Unparseable poison message: delete it so it does not
            # recirculate forever (it can never be dispatched).
            logger.warning(
                "Discarding unparseable SQS message %s on topic %s: %s",
                message_id,
                topic,
                e,
            )
            if receipt_handle:
                await self._client.delete_message(
                    QueueUrl=self._config.queue_url, ReceiptHandle=receipt_handle
                )
            return

        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            # At-least-once: do NOT delete; redelivered after the
            # visibility timeout. Never log the payload (security 5).
            logger.exception(
                "SqsEventBus handler failed for subscription %s "
                "(topic=%s, message_id=%s); will be redelivered",
                subscription_id[:8],
                topic,
                message_id,
            )
            return

        if receipt_handle:
            await self._client.delete_message(
                QueueUrl=self._config.queue_url, ReceiptHandle=receipt_handle
            )
        logger.debug(
            "Dispatched SQS message %s to subscription %s (topic=%s)",
            message_id,
            subscription_id[:8],
            topic,
        )

    async def _dispatch_to_all(
        self,
        message: dict[str, Any],
        receiving_subscription_id: str,
        receiving_topic: str,
        receipt_handle: str | None,
    ) -> None:
        """Dispatch an attribute-less message to the active subscription.

        Reached only when ``require_topic_attribute=False`` AND the
        message has no topic attribute. By the single-topic bridge
        mode contract (enforced at :meth:`subscribe`), at most one
        subscription is active. The snapshot is iterated defensively
        — a concurrent ``unsubscribe`` between message receipt and
        snapshot may leave it empty, in which case the message is
        deleted (no handler available; debug-logged for visibility).

        - Handler succeeds → ``delete_message``.
        - Handler raises → no delete → message redelivered after the
          visibility timeout (at-least-once preserved; the idempotent
          handler re-receives it).
        - Snapshot is empty (concurrent unsubscribe race) → delete
          + debug log so operators can correlate the disappearance.

        Subscriptions are snapshotted at entry so a concurrent
        ``subscribe`` / ``unsubscribe`` cannot mutate the iteration
        target. The lock is NOT held during handler invocation — a
        handler that initiates a subscribe / unsubscribe would
        otherwise deadlock.
        """
        message_id = message.get("MessageId", "")
        body = message.get("Body", "")

        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as e:
            logger.warning(
                "Discarding unparseable SQS message %s (fanout): %s",
                message_id,
                e,
            )
            if receipt_handle:
                await self._client.delete_message(
                    QueueUrl=self._config.queue_url, ReceiptHandle=receipt_handle
                )
            return

        try:
            event = Event.from_dict(decoded)
        except (KeyError, ValueError, TypeError):
            # Valid JSON but not Event.to_dict()-shaped. Synthesise a
            # CUSTOM event carrying the receiving poll task's topic
            # and the decoded body as payload.
            #
            # event_id is derived from the SQS MessageId (stable across
            # redeliveries) rather than auto-generated, so handlers can
            # use event.event_id for idempotency keys without breaking
            # on at-least-once redelivery of the same logical message.
            payload = (
                decoded
                if isinstance(decoded, dict)
                else {"body": decoded}
            )
            event_kwargs: dict[str, Any] = {
                "type": EventType.CUSTOM,
                "topic": receiving_topic,
                "payload": payload,
                "metadata": {
                    "sqs_message_id": message_id,
                    "sqs_synthesised": True,
                },
            }
            if message_id:
                event_kwargs["event_id"] = f"sqs:{message_id}"
            event = Event(**event_kwargs)
            logger.warning(
                "Synthesised CUSTOM event for non-Event SQS body "
                "(message_id=%s, topic=%s, body_shape=%s)",
                message_id,
                receiving_topic,
                type(decoded).__name__,
            )

        async with self._lock:
            subscriptions = list(self._subscriptions.values())
        if not subscriptions:
            # Concurrent unsubscribe race: receiving poll task popped
            # its own subscription between message receipt and the
            # snapshot. No handler available — delete the message so
            # it doesn't recirculate, and log for operator visibility.
            logger.debug(
                "Fanout snapshot empty for SQS message %s "
                "(concurrent unsubscribe race); deleting",
                message_id,
            )
            if receipt_handle:
                await self._client.delete_message(
                    QueueUrl=self._config.queue_url, ReceiptHandle=receipt_handle
                )
            return
        any_failed = False
        for sub in subscriptions:
            try:
                result = sub.handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                any_failed = True
                logger.exception(
                    "SqsEventBus fanout handler failed for "
                    "subscription %s (target_topic=%s, message_id=%s); "
                    "message will be redelivered to all subscribers",
                    sub.subscription_id[:8],
                    sub.topic,
                    message_id,
                )

        if any_failed:
            # At-least-once: do not delete. Visibility timeout will
            # expire and the message will be redelivered to whichever
            # subscriber polls next. All idempotent handlers re-run.
            return

        if receipt_handle:
            await self._client.delete_message(
                QueueUrl=self._config.queue_url, ReceiptHandle=receipt_handle
            )
        logger.debug(
            "Fanout-dispatched SQS message %s to %d subscriptions "
            "(receiving=%s, receiving_topic=%s)",
            message_id,
            len(subscriptions),
            receiving_subscription_id[:8],
            receiving_topic,
        )

    @property
    def subscription_count(self) -> int:
        """Number of active subscriptions."""
        return len(self._subscriptions)
