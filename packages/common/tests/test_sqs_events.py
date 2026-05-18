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
import os
import uuid

import pytest
import pytest_asyncio

from dataknobs_common.events import Event, EventType, create_event_bus
from dataknobs_common.events.sqs import SqsEventBus
from dataknobs_common.testing import requires_localstack

pytestmark = requires_localstack


def _localstack_endpoint() -> str:
    """Resolve the LocalStack edge endpoint (Docker- and host-aware)."""
    if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
        default_host = "localstack"
    else:
        default_host = "localhost"
    return os.environ.get(
        "LOCALSTACK_ENDPOINT", f"http://{default_host}:4566"
    )


ENDPOINT = _localstack_endpoint()
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
