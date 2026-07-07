"""Reproduce-first async-correctness test for ``SqsEventBus.connect()``.

``SqsEventBus`` used to build its ``aioboto3.Session`` inline on the event
loop and create the SQS client from a cold session, so the first
``session.client("sqs")`` loaded botocore's bundled data files (service
model, endpoints, sdk-default-config) synchronously on the loop — a
blocking-I/O-on-the-event-loop violation (see ``rules/async-transport.md``).

The fix routes ``connect()`` through
:func:`dataknobs_common.aws.create_aioboto3_session`, which builds and
*warms* the session on a worker thread, so the first real client creation
is a botocore cache hit and does not block.

**Testing shape (mirrors ``test_aioboto3_session_offload.py``).** The warm
establishes a running loop on a worker thread and ``blockbuster``'s
detection is process-global, so letting ``connect()`` trigger the cold-cache
warm *inside* ``assert_no_blocking`` would flag the warm's own ``os.stat``
(a false failure). The correct shape pre-warms the shared session cache
OUTSIDE the block (via the identical config ``connect()`` will use), then
wraps only ``connect()`` — which hits the cache and performs just the
on-loop client creation, needing no network.

These tests need ``aioboto3`` (the async transport) but NOT LocalStack:
creating a client loads data files locally; no SQS API call is made.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from dataknobs_common.aws import (
    clear_aioboto3_session_cache,
    create_aioboto3_session,
)
from dataknobs_common.events.sqs import SqsEventBus
from dataknobs_common.testing import (
    assert_no_blocking,
    requires_blockbuster,
    requires_package,
)

pytestmark = [pytest.mark.asyncio, requires_package("aioboto3")]


@pytest.fixture(autouse=True)
def _clear_session_cache() -> Iterator[None]:
    """Reset the process-wide session cache around each test.

    ``create_aioboto3_session`` caches warmed sessions process-wide, so a
    session warmed by another test (or by the pre-warm step here) must not
    leak in and mask the per-test build/warm behavior under assertion.
    """
    clear_aioboto3_session_cache()
    yield
    clear_aioboto3_session_cache()


@pytest.fixture(autouse=True)
def _isolate_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ambient AWS env vars and shadow ~/.aws/* for boto's chain.

    Ambient ``AWS_*`` env vars or ``~/.aws/config`` would perturb the
    botocore data load under test; clearing them makes the cache-hit
    assertion deterministic. ``AWS_DEFAULT_REGION`` is set so client
    creation does not stall on region resolution.
    """
    for key in (
        "AWS_REGION",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_SQS",
        "LOCALSTACK_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest_asyncio.fixture
async def bus() -> AsyncIterator[SqsEventBus]:
    """A disconnected ``SqsEventBus`` with explicit creds and no endpoint.

    Explicit credentials make botocore's credential provider chain
    short-circuit (no ``~/.aws/credentials`` stat on every client
    creation), isolating the concern the warm fixes: the one-time
    botocore data-file load. No ``endpoint_url`` → the client targets the
    public SQS endpoint, but no API call is made, so no network occurs.
    """
    instance = SqsEventBus(
        queue_url="https://sqs.us-east-1.amazonaws.com/123456789012/dk-test",
        region="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    try:
        yield instance
    finally:
        await instance.close()


@requires_blockbuster
async def test_connect_does_not_block_the_event_loop(bus: SqsEventBus) -> None:
    """``connect()`` must not load botocore data files on the event loop.

    FAILS pre-fix (the inline cold-session client creation loaded botocore
    data on the loop) and PASSES once ``connect()`` routes through the
    warmed shared session factory. The cache is pre-warmed OUTSIDE the
    block (see module docstring) using the exact config ``connect()`` uses.
    """
    # Pre-warm the shared cache off the detector, with the identical
    # config/warm_service connect() will request → connect() hits the cache.
    await create_aioboto3_session(bus._aws_session_config(), warm_service="sqs")

    with assert_no_blocking():
        await bus.connect()

    assert bus.subscription_count == 0  # connected, no subscriptions yet


async def test_connect_reuses_the_shared_warmed_session(bus: SqsEventBus) -> None:
    """``connect()`` adopts the shared factory's cached session object.

    Functional proof (no blockbuster) that ``connect()`` routes through
    :func:`create_aioboto3_session` rather than building its own session:
    the session it holds is the same cached object the factory returns for
    the bus's config.
    """
    warmed = await create_aioboto3_session(
        bus._aws_session_config(), warm_service="sqs"
    )
    await bus.connect()
    assert bus._session is warmed
