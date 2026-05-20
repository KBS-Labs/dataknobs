"""Behavioural tests for ``SqsEventBus`` against real LocalStack.

Per the optional-dependency-backend testing mandate these exercise the
real ``aioboto3`` SQS client against a real LocalStack queue — never a
moto-mock or hand-rolled fake (a fake appending to a list has the same
blindness as a ``MagicMock``). LocalStack is the SQS analog of "real
Postgres via ``bin/dk up``". The whole module skips when LocalStack is
unavailable (``@requires_localstack``).

Start the service with ``bin/dk up`` (LocalStack runs ``SERVICES=s3,sqs``).
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
import pytest_asyncio

from dataknobs_common.events import Event, EventType, create_event_bus
from dataknobs_common.events.sqs import SqsEventBus
from dataknobs_common.testing import (
    get_localstack_endpoint,
    requires_localstack,
)

pytestmark = requires_localstack


ENDPOINT = get_localstack_endpoint()
REGION = "us-east-1"


@pytest_asyncio.fixture
async def make_queue():
    """Factory creating LocalStack SQS queues, deleted on teardown."""
    import aioboto3

    session = aioboto3.Session(
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    created: list[str] = []

    async def _make(fifo: bool = False) -> str:
        name = f"dk-test-{uuid.uuid4().hex[:12]}"
        attrs: dict[str, str] = {}
        if fifo:
            name += ".fifo"
            attrs["FifoQueue"] = "true"
            attrs["ContentBasedDeduplication"] = "true"
        async with session.client(
            "sqs", endpoint_url=ENDPOINT
        ) as client:
            resp = await client.create_queue(
                QueueName=name, Attributes=attrs
            )
            url = resp["QueueUrl"]
        created.append(url)
        return url

    yield _make

    async with session.client("sqs", endpoint_url=ENDPOINT) as client:
        for url in created:
            try:
                await client.delete_queue(QueueUrl=url)
            except Exception:
                pass


def _make_bus(queue_url: str, **overrides) -> SqsEventBus:
    kwargs = dict(
        queue_url=queue_url,
        region=REGION,
        endpoint_url=ENDPOINT,
        wait_time_seconds=1,
        visibility_timeout=2,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    kwargs.update(overrides)
    return SqsEventBus(**kwargs)


async def _wait_for(predicate, timeout: float = 15.0) -> None:
    """Poll ``predicate`` until true or ``timeout`` (SQS is eventual)."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.2)
    raise AssertionError("condition not met within timeout")


class TestSqsEventBus:
    """Real-LocalStack publish/subscribe behaviour."""

    @pytest.mark.asyncio
    async def test_publish_subscribe_round_trip(self, make_queue):
        url = await make_queue()
        bus = _make_bus(url)
        await bus.connect()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        try:
            await bus.subscribe("t", handler)
            await bus.publish(
                "t",
                Event(
                    type=EventType.CREATED, topic="t", payload={"k": 1}
                ),
            )
            await _wait_for(lambda: len(received) >= 1)
        finally:
            await bus.close()

        assert received[0].topic == "t"
        assert received[0].payload == {"k": 1}
        assert received[0].type is EventType.CREATED

    @pytest.mark.asyncio
    async def test_topic_isolation(self, make_queue):
        """A subscriber on 'a' must not receive a 'b' message."""
        url = await make_queue()
        bus = _make_bus(url)
        await bus.connect()
        a_events: list[Event] = []
        b_events: list[Event] = []

        async def a_handler(event: Event) -> None:
            a_events.append(event)

        async def b_handler(event: Event) -> None:
            b_events.append(event)

        try:
            await bus.subscribe("a", a_handler)
            await bus.subscribe("b", b_handler)
            await bus.publish(
                "b", Event(type=EventType.UPDATED, topic="b", payload={})
            )
            await _wait_for(lambda: len(b_events) >= 1)
            # Give 'a' ample opportunity to (wrongly) pick it up.
            await asyncio.sleep(2.0)
        finally:
            await bus.close()

        assert len(b_events) == 1
        assert len(a_events) == 0

    @pytest.mark.asyncio
    async def test_at_least_once_redelivery_on_handler_error(
        self, make_queue
    ):
        """A raising handler does not delete; the message redelivers."""
        url = await make_queue()
        bus = _make_bus(url, visibility_timeout=2)
        await bus.connect()
        attempts: list[int] = []

        async def handler(event: Event) -> None:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("transient failure on first delivery")

        try:
            await bus.subscribe("retry", handler)
            await bus.publish(
                "retry",
                Event(type=EventType.CUSTOM, topic="retry", payload={}),
            )
            # First call raises (no delete); after visibility_timeout
            # the message reappears and the second call succeeds.
            await _wait_for(lambda: len(attempts) >= 2, timeout=20.0)
        finally:
            await bus.close()

        assert len(attempts) >= 2

    @pytest.mark.asyncio
    async def test_poll_loop_survives_transient_receive_error(
        self, make_queue
    ):
        """A transient receive_message failure must not kill the loop.

        The supervised loop logs the failure, backs off (non-zero
        elapsed), then resumes delivery on the next iteration.
        """
        url = await make_queue()
        bus = _make_bus(url)
        await bus.connect()
        received: list[Event] = []
        timeline: dict[str, float] = {}

        async def handler(event: Event) -> None:
            timeline["delivered"] = asyncio.get_event_loop().time()
            received.append(event)

        # bus._client is the live aioboto3 SQS client (real LocalStack
        # queue — no mock). Wrapping its receive_message is the single
        # narrowest seam to inject ONE transient failure into the real
        # poll path: it exercises run_supervised_loop's actual back-off
        # + recovery against a real broker rather than a fake. The
        # type: ignore is the expected cost of patching a bound method
        # on a third-party client instance.
        real_receive = bus._client.receive_message
        state = {"failed": False}

        async def flaky_receive(**kwargs):
            if not state["failed"]:
                state["failed"] = True
                timeline["failed"] = asyncio.get_event_loop().time()
                raise RuntimeError("injected transient receive failure")
            return await real_receive(**kwargs)

        bus._client.receive_message = flaky_receive  # type: ignore[method-assign]

        try:
            await bus.subscribe("flaky", handler)
            await bus.publish(
                "flaky",
                Event(type=EventType.CREATED, topic="flaky", payload={"n": 1}),
            )
            await _wait_for(lambda: len(received) >= 1, timeout=20.0)
        finally:
            await bus.close()

        assert state["failed"], "transient failure was never injected"
        assert received[0].payload == {"n": 1}, "delivery did not resume"
        # A back-off elapsed between the injected failure and recovery
        # (base_delay 1.0 with +/-10% jitter -> at least ~0.8s).
        assert timeline["delivered"] - timeline["failed"] >= 0.8

    @pytest.mark.asyncio
    async def test_connect_idempotent_no_double_poll(self, make_queue):
        """Double connect() then one publish → exactly one delivery."""
        url = await make_queue()
        bus = _make_bus(url)
        await bus.connect()
        await bus.connect()  # idempotent — must not start a second client
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        try:
            await bus.subscribe("once", handler)
            await bus.publish(
                "once",
                Event(type=EventType.CREATED, topic="once", payload={}),
            )
            await _wait_for(lambda: len(received) >= 1)
            await asyncio.sleep(2.0)  # ensure no duplicate poll path
        finally:
            await bus.close()

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_close_cancels_poll_tasks(self, make_queue):
        url = await make_queue()
        bus = _make_bus(url)
        await bus.connect()

        async def handler(event: Event) -> None:
            pass

        await bus.subscribe("c", handler)
        tasks = list(bus._poll_tasks.values())
        assert tasks and all(not t.done() for t in tasks)

        await bus.close()

        assert bus._poll_tasks == {}
        assert bus.subscription_count == 0
        assert all(t.done() for t in tasks)
        # Publishing after close fails fast rather than silently.
        with pytest.raises(RuntimeError):
            await bus.publish(
                "c", Event(type=EventType.CREATED, topic="c", payload={})
            )

    @pytest.mark.asyncio
    async def test_fifo_ordering_within_topic(self, make_queue):
        url = await make_queue(fifo=True)
        bus = _make_bus(url)
        await bus.connect()
        assert bus._is_fifo is True
        order: list[int] = []

        async def handler(event: Event) -> None:
            order.append(event.payload["seq"])

        try:
            await bus.subscribe("ordered", handler)
            for seq in range(5):
                await bus.publish(
                    "ordered",
                    Event(
                        type=EventType.CUSTOM,
                        topic="ordered",
                        payload={"seq": seq},
                    ),
                )
            await _wait_for(lambda: len(order) >= 5)
        finally:
            await bus.close()

        assert order == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_factory_path_round_trip(self, make_queue):
        """create_event_bus({"backend": "sqs", ...}) works end to end."""
        url = await make_queue()
        bus = create_event_bus(
            {
                "backend": "sqs",
                "queue_url": url,
                "region": REGION,
                "endpoint_url": ENDPOINT,
                "wait_time_seconds": 1,
                "visibility_timeout": 2,
                "aws_access_key_id": "test",
                "aws_secret_access_key": "test",
            }
        )
        assert isinstance(bus, SqsEventBus)
        await bus.connect()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        try:
            await bus.subscribe("f", handler)
            await bus.publish(
                "f", Event(type=EventType.CREATED, topic="f", payload={"v": 9})
            )
            await _wait_for(lambda: len(received) >= 1)
        finally:
            await bus.close()

        assert received[0].payload == {"v": 9}

    @pytest.mark.asyncio
    async def test_pattern_subscription_raises(self, make_queue):
        """Wildcard patterns are unsupported — fail loudly, not silently."""
        url = await make_queue()
        bus = _make_bus(url)
        await bus.connect()
        try:
            with pytest.raises(NotImplementedError):
                await bus.subscribe("x", lambda e: None, pattern="x:*")
        finally:
            await bus.close()


async def _send_raw(
    queue_url: str,
    body: str,
    attributes: dict[str, dict[str, str]] | None = None,
) -> None:
    """Send a raw SQS message via a one-shot aioboto3 client.

    Bypasses ``SqsEventBus.publish`` so a test can put a message on
    the queue *without* the topic attribute (the AWS-native bridge
    case) or with an arbitrary attribute value.
    """
    import aioboto3

    session = aioboto3.Session(
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    async with session.client("sqs", endpoint_url=ENDPOINT) as client:
        params: dict[str, object] = {
            "QueueUrl": queue_url,
            "MessageBody": body,
        }
        if attributes:
            params["MessageAttributes"] = attributes
        await client.send_message(**params)


async def _receive_one(queue_url: str, wait_seconds: int = 2) -> dict | None:
    """Single ``ReceiveMessage`` call for assertions about released messages."""
    import aioboto3

    session = aioboto3.Session(
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    async with session.client("sqs", endpoint_url=ENDPOINT) as client:
        resp = await client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_seconds,
            MessageAttributeNames=["All"],
        )
        messages = resp.get("Messages", [])
        return messages[0] if messages else None


class TestSqsEventBusSingleTopicMode:
    """Real-LocalStack behaviour for ``require_topic_attribute=False``."""

    @pytest.mark.asyncio
    async def test_default_is_require_topic_attribute_true(self, make_queue):
        """Regression guard: the new param defaults to today's behaviour."""
        url = await make_queue()
        bus = _make_bus(url)
        assert bus._require_topic_attribute is True

    @pytest.mark.asyncio
    async def test_attribute_required_default_drops_attribute_less_message(
        self, make_queue
    ):
        """Default mode: attribute-less message returns to queue."""
        url = await make_queue()
        bus = _make_bus(url)
        await bus.connect()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        try:
            await bus.subscribe("knowledge:trigger", handler)
            await _send_raw(
                url,
                json.dumps(
                    Event(
                        type=EventType.CUSTOM,
                        topic="knowledge:trigger",
                        payload={"k": 1},
                    ).to_dict()
                ),
                attributes=None,
            )
            # The poll loop receives, sees no attribute, and (in default
            # mode) releases visibility. Give it time to make that call.
            await asyncio.sleep(2.0)
        finally:
            await bus.close()

        assert received == [], (
            "default mode must NOT dispatch attribute-less messages"
        )
        # Message should be released back to the queue.
        msg = await _receive_one(url, wait_seconds=3)
        assert msg is not None, (
            "released message must reappear on the queue"
        )

    @pytest.mark.asyncio
    async def test_attribute_optional_dispatches_attribute_less_message(
        self, make_queue
    ):
        """New mode: attribute-less Event-shaped body dispatches once."""
        url = await make_queue()
        bus = _make_bus(url, require_topic_attribute=False)
        await bus.connect()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        try:
            await bus.subscribe("knowledge:trigger", handler)
            event_payload = {
                "tenant_id": "acme",
                "domain_id": "support",
            }
            event = Event(
                type=EventType.CUSTOM,
                topic="knowledge:trigger",
                payload=event_payload,
            )
            await _send_raw(url, json.dumps(event.to_dict()), attributes=None)
            await _wait_for(lambda: len(received) >= 1)
            await asyncio.sleep(1.0)  # guard against double-delivery
        finally:
            await bus.close()

        assert len(received) == 1
        assert received[0].topic == "knowledge:trigger"
        assert received[0].payload == event_payload
        # Message should be deleted — a follow-up receive returns nothing.
        msg = await _receive_one(url, wait_seconds=2)
        assert msg is None, "successful dispatch must delete the message"

    @pytest.mark.asyncio
    async def test_attribute_optional_dispatches_non_event_shaped_body(
        self, make_queue, caplog
    ):
        """New mode: non-Event JSON body → synthesised CUSTOM event."""
        import logging as _logging

        url = await make_queue()
        bus = _make_bus(url, require_topic_attribute=False)
        await bus.connect()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        try:
            await bus.subscribe("knowledge:trigger", handler)
            with caplog.at_level(
                _logging.WARNING,
                logger="dataknobs_common.events.sqs",
            ):
                await _send_raw(
                    url,
                    json.dumps({"some": "payload"}),
                    attributes=None,
                )
                await _wait_for(lambda: len(received) >= 1)
                await asyncio.sleep(1.0)
        finally:
            await bus.close()

        assert len(received) == 1
        assert received[0].type is EventType.CUSTOM
        assert received[0].topic == "knowledge:trigger"
        assert received[0].payload == {"some": "payload"}
        synth_warnings = [
            r
            for r in caplog.records
            if "Synthesised CUSTOM event" in r.getMessage()
        ]
        assert len(synth_warnings) == 1, (
            "expected exactly one synthesis warning"
        )

    @pytest.mark.asyncio
    async def test_attribute_optional_with_matching_attribute_still_filters(
        self, make_queue
    ):
        """New mode: matching attribute still routes only to that sub."""
        url = await make_queue()
        bus = _make_bus(url, require_topic_attribute=False)
        await bus.connect()
        a_events: list[Event] = []
        b_events: list[Event] = []

        async def a_handler(event: Event) -> None:
            a_events.append(event)

        async def b_handler(event: Event) -> None:
            b_events.append(event)

        try:
            await bus.subscribe("topic-a", a_handler)
            await bus.subscribe("topic-b", b_handler)
            await bus.publish(
                "topic-a",
                Event(
                    type=EventType.CUSTOM,
                    topic="topic-a",
                    payload={"n": 1},
                ),
            )
            await _wait_for(lambda: len(a_events) >= 1)
            await asyncio.sleep(2.0)  # give b_handler a chance to wrongly fire
        finally:
            await bus.close()

        assert len(a_events) == 1
        assert a_events[0].payload == {"n": 1}
        assert b_events == [], (
            "matching-attribute messages must not fan out to other subs"
        )

    @pytest.mark.asyncio
    async def test_attribute_optional_with_mismatched_attribute_still_releases(
        self, make_queue
    ):
        """New mode: mismatched attribute → release back to queue."""
        url = await make_queue()
        bus = _make_bus(url, require_topic_attribute=False)
        await bus.connect()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        try:
            await bus.subscribe("topic-a", handler)
            await _send_raw(
                url,
                json.dumps(
                    Event(
                        type=EventType.CUSTOM,
                        topic="other-topic",
                        payload={},
                    ).to_dict()
                ),
                attributes={
                    "topic": {
                        "StringValue": "other-topic",
                        "DataType": "String",
                    }
                },
            )
            await asyncio.sleep(2.0)
        finally:
            await bus.close()

        assert received == [], (
            "mismatched-attribute messages must not dispatch"
        )
        msg = await _receive_one(url, wait_seconds=3)
        assert msg is not None, (
            "mismatched message must be released back to the queue"
        )

    @pytest.mark.asyncio
    async def test_attribute_optional_fanout_to_multiple_subscriptions(
        self, make_queue
    ):
        """New mode: attribute-less message → every active sub receives it."""
        url = await make_queue()
        bus = _make_bus(url, require_topic_attribute=False)
        await bus.connect()
        a_events: list[Event] = []
        b_events: list[Event] = []

        async def a_handler(event: Event) -> None:
            a_events.append(event)

        async def b_handler(event: Event) -> None:
            b_events.append(event)

        try:
            await bus.subscribe("topic-a", a_handler)
            await bus.subscribe("topic-b", b_handler)
            await _send_raw(
                url,
                json.dumps(
                    Event(
                        type=EventType.CUSTOM,
                        topic="knowledge:trigger",
                        payload={"shared": True},
                    ).to_dict()
                ),
                attributes=None,
            )
            await _wait_for(
                lambda: len(a_events) >= 1 and len(b_events) >= 1
            )
            await asyncio.sleep(1.0)
        finally:
            await bus.close()

        assert len(a_events) == 1
        assert len(b_events) == 1
        assert a_events[0].payload == {"shared": True}
        assert b_events[0].payload == {"shared": True}

    @pytest.mark.asyncio
    async def test_attribute_optional_fanout_handler_error_redelivers(
        self, make_queue
    ):
        """New mode: a handler raising → message not deleted → redelivered.

        Two subscribers, one raises once and then succeeds. The
        message must be redelivered to BOTH (at-least-once preserved
        across the fanout set) — idempotent handlers re-receive it.
        """
        url = await make_queue()
        bus = _make_bus(url, require_topic_attribute=False)
        await bus.connect()
        a_calls: list[Event] = []
        b_calls: list[Event] = []
        b_state = {"raised": False}

        async def a_handler(event: Event) -> None:
            a_calls.append(event)

        async def b_handler(event: Event) -> None:
            b_calls.append(event)
            if not b_state["raised"]:
                b_state["raised"] = True
                raise RuntimeError("transient failure on first delivery")

        try:
            await bus.subscribe("topic-a", a_handler)
            await bus.subscribe("topic-b", b_handler)
            await _send_raw(
                url,
                json.dumps(
                    Event(
                        type=EventType.CUSTOM,
                        topic="knowledge:trigger",
                        payload={"retry": True},
                    ).to_dict()
                ),
                attributes=None,
            )
            # Wait for both initial deliveries, then for redelivery
            # past the visibility timeout (2s + jitter).
            await _wait_for(
                lambda: len(a_calls) >= 2 and len(b_calls) >= 2,
                timeout=20.0,
            )
        finally:
            await bus.close()

        # Both handlers must have been called twice: once on initial
        # delivery (where b raised, blocking the delete) and again on
        # redelivery (where b succeeds, allowing the delete).
        assert len(a_calls) >= 2, (
            f"a_handler must be redelivered (calls={len(a_calls)})"
        )
        assert len(b_calls) >= 2, (
            f"b_handler must retry after raising (calls={len(b_calls)})"
        )
        assert b_state["raised"] is True
