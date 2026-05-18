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
from typing import TYPE_CHECKING, Any

from ._resilient_loop import run_supervised_loop
from .types import Event, Subscription

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class SqsEventBus:
    """Event bus backed by a single AWS SQS queue.

    Topic routing: every :class:`Event` is sent to the one
    ``queue_url`` with the topic carried in a ``String`` message
    attribute named ``topic_attribute``. A subscription long-polls the
    queue and dispatches only messages whose attribute equals its
    topic; non-matching messages are left untouched (not deleted) so a
    subscriber on a different topic can still receive them after the
    visibility timeout.

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

    Requires:
        aioboto3: ``pip install 'dataknobs-common[sqs]'``
    """

    def __init__(
        self,
        queue_url: str,
        region: str | None = None,
        endpoint_url: str | None = None,
        wait_time_seconds: int = 20,
        visibility_timeout: int = 60,
        topic_attribute: str = "topic",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        """Initialize the SQS event bus.

        Args:
            queue_url: Full SQS queue URL (required, non-empty). A URL
                ending in ``.fifo`` is treated as a FIFO queue —
                ``MessageGroupId``/``MessageDeduplicationId`` are set on
                publish.
            region: AWS region. When ``None``, boto's default region
                chain is used (``AWS_DEFAULT_REGION``, ``~/.aws/config``,
                instance metadata).
            endpoint_url: Override the SQS endpoint (LocalStack, a VPC
                endpoint). When ``None``, the AWS public endpoint for the
                resolved region is used.
            wait_time_seconds: ``ReceiveMessage`` long-poll wait
                (0-20). Also drives the client read timeout.
            visibility_timeout: Seconds a received message is hidden
                before redelivery if not deleted (the at-least-once
                retry window).
            topic_attribute: Message-attribute name carrying the topic.
            aws_access_key_id: Explicit access key. When ``None``,
                boto's default credential chain is used.
            aws_secret_access_key: Explicit secret key (paired with
                ``aws_access_key_id``).

        Raises:
            ValueError: If ``queue_url`` is empty.
        """
        if not queue_url:
            raise ValueError("SqsEventBus requires a non-empty queue_url")
        self._queue_url = queue_url
        self._region = region
        self._endpoint_url = endpoint_url
        self._wait_time_seconds = wait_time_seconds
        self._visibility_timeout = visibility_timeout
        self._topic_attribute = topic_attribute
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._is_fifo = queue_url.endswith(".fifo")

        self._session: Any = None  # aioboto3.Session
        self._client: Any = None  # aiobotocore SQS client
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._subscriptions: dict[str, Subscription] = {}
        self._poll_tasks: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()
        self._connected = False
        self._running = False

    def _session_kwargs(self) -> dict[str, Any]:
        """Build ``aioboto3.Session`` kwargs.

        Only explicitly provided values are set; everything else is left
        to boto's default region/credential chain. This is the minimal
        local passthrough that replaces an (illegal) ``dataknobs-data``
        import — ``dataknobs-common`` is the lowest layer and must not
        depend on a higher one.
        """
        kwargs: dict[str, Any] = {}
        if self._region:
            kwargs["region_name"] = self._region
        if self._aws_access_key_id and self._aws_secret_access_key:
            kwargs["aws_access_key_id"] = self._aws_access_key_id
            kwargs["aws_secret_access_key"] = self._aws_secret_access_key
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
                read_timeout=self._wait_time_seconds + 10,
                retries={"max_attempts": 3, "mode": "standard"},
            )
        }
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
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
                self._queue_url,
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
            logger.info("SqsEventBus closed for queue %s", self._queue_url)

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
            "QueueUrl": self._queue_url,
            "MessageBody": json.dumps(event.to_dict()),
            "MessageAttributes": {
                self._topic_attribute: {
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
            self._queue_url,
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
            self._subscriptions[subscription_id] = subscription
            self._poll_tasks[subscription_id] = asyncio.create_task(
                self._poll_loop(subscription_id, topic, handler)
            )

        logger.debug(
            "Subscribed %s to topic %s (queue=%s)",
            subscription_id[:8],
            topic,
            self._queue_url,
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
                QueueUrl=self._queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=self._wait_time_seconds,
                VisibilityTimeout=self._visibility_timeout,
                MessageAttributeNames=[self._topic_attribute],
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

    async def _handle_message(
        self,
        message: dict[str, Any],
        subscription_id: str,
        topic: str,
        handler: Callable[[Event], Any],
    ) -> None:
        """Filter by topic attribute, invoke handler, delete on success.

        A message whose topic attribute does not equal this
        subscription's topic is left in the queue (no delete) so a
        subscriber on that other topic can consume it after the
        visibility timeout. A handler that raises also leaves the
        message — it is redelivered (at-least-once).
        """
        attrs = message.get("MessageAttributes") or {}
        topic_attr = attrs.get(self._topic_attribute) or {}
        msg_topic = topic_attr.get("StringValue")
        receipt_handle = message.get("ReceiptHandle")
        if msg_topic != topic:
            # Another topic's message. Return it to the queue
            # *immediately* (visibility 0) rather than letting it sit
            # invisible for the full visibility timeout: on a shared
            # single queue, parking it lets a subscriber on a different
            # topic repeatedly "steal and hold" the message and starve
            # the subscriber that actually handles this topic. Resetting
            # visibility makes it instantly available for the right
            # consumer's next poll. Best-effort: the matching subscriber
            # may have already received+deleted it (benign expired/
            # invalid receipt handle race) — never propagate that into
            # the supervised poll loop as a failure.
            if receipt_handle:
                with contextlib.suppress(Exception):
                    await self._client.change_message_visibility(
                        QueueUrl=self._queue_url,
                        ReceiptHandle=receipt_handle,
                        VisibilityTimeout=0,
                    )
            return

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
                    QueueUrl=self._queue_url, ReceiptHandle=receipt_handle
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
                QueueUrl=self._queue_url, ReceiptHandle=receipt_handle
            )
        logger.debug(
            "Dispatched SQS message %s to subscription %s (topic=%s)",
            message_id,
            subscription_id[:8],
            topic,
        )

    @property
    def subscription_count(self) -> int:
        """Number of active subscriptions."""
        return len(self._subscriptions)
