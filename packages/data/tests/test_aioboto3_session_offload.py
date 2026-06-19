"""Reproduce-first async-correctness test for ``create_aioboto3_session``.

Two things block the loop the first time any aioboto3 consumer
(``AsyncS3Database``, ``SqsEventBus``, ``S3KnowledgeBackend``) uses a
session built by this factory:

1. ``aioboto3.Session(...)`` construction (botocore loader setup, reading
   the service-model registry + ``~/.aws`` config from disk).
2. aiobotocore's lazy, synchronous load of botocore's bundled data files
   (``endpoints``, ``sdk-default-configuration``, the S3 service model)
   the first time a client is created from the session.

The fix offloads the whole factory onto a ``to_thread`` worker that builds
the session AND warms its botocore caches by creating-and-discarding one
throwaway client in a private event loop on that worker thread. After the
warm, the first real client creation by any consumer is a cache hit — no
disk, no loop block.

**Testing gotcha (why the session is built outside the block).** The warm
establishes a running loop on the worker thread, and ``blockbuster``'s
detection is per-running-loop and process-global, so wrapping
``create_aioboto3_session`` itself in ``assert_no_blocking`` would flag the
warm's own ``os.stat`` (the warm then raises, is swallowed, the caches stay
cold, and the real client creation blocks — a false failure). The correct
shape builds the session OUTSIDE the block (the warm runs without the
detector active) and wraps only the subsequent client creation, which needs
no network — it just loads data, and with the warm done it is a cache hit.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from dataknobs_common.testing import assert_no_blocking, requires_blockbuster

from dataknobs_data.pooling.s3 import (
    S3SessionConfig,
    clear_aioboto3_session_cache,
    create_aioboto3_session,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clear_session_cache() -> Iterator[None]:
    """Reset the process-wide session cache around each test.

    ``create_aioboto3_session`` caches warmed sessions process-wide, so
    without this a session warmed by one test would leak into the next
    and mask per-test build/warm behavior.
    """
    clear_aioboto3_session_cache()
    yield
    clear_aioboto3_session_cache()


@pytest.fixture(autouse=True)
def _isolate_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ambient AWS env vars and shadow ~/.aws/* for boto's chain.

    Ambient ``AWS_*`` env vars or ``~/.aws/config`` would perturb the
    botocore data load under test; clearing them makes the cache-hit
    assertion deterministic. ``AWS_DEFAULT_REGION`` is set to a fixed
    value so S3 client creation does not stall on region resolution.
    """
    for key in (
        "AWS_REGION",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "LOCALSTACK_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


# Explicit credentials so botocore's credential provider chain
# short-circuits (no ``~/.aws/credentials`` file stat). botocore never
# caches a "no credentials found" result, so a credential-less session
# re-runs the chain — and re-stats config files — on every client
# creation, which the session warm cannot prevent. Supplying credentials
# isolates the concern the warm DOES fix: the one-time botocore data-file
# load (endpoints / service model). Mirrors a realistic consumer whose
# credentials come from config/env.
_CREDS = S3SessionConfig(
    aws_access_key_id="testing",
    aws_secret_access_key="testing",
    region_name="us-east-1",
)


@requires_blockbuster
async def test_session_client_creation_does_not_block() -> None:
    """First client creation from a factory-built session must not block.

    FAILS pre-fix (the first ``session.client("s3")`` loads botocore data
    files on the loop) and PASSES once the factory warms the caches
    off-loop. The session is built OUTSIDE the block (see module docstring).
    """
    session = await create_aioboto3_session(_CREDS)
    with assert_no_blocking():
        async with session.client("s3"):
            pass


async def test_create_aioboto3_session_returns_usable_session() -> None:
    """The returned session can create a working S3 client.

    Functional guard that the offload + warm did not break ordinary use
    (no blockbuster involved).
    """
    session = await create_aioboto3_session(_CREDS)
    async with session.client("s3") as s3:
        assert s3 is not None


async def test_repeated_calls_reuse_one_warmed_session() -> None:
    """Same config → same cached session object (warm runs once).

    Guards the cross-call caching that keeps a multi-config consumer
    (e.g. several backends sharing one bucket) from re-warming a session
    per instance.
    """
    first = await create_aioboto3_session(_CREDS)
    second = await create_aioboto3_session(_CREDS)
    assert first is second


async def test_distinct_configs_get_distinct_sessions() -> None:
    """Different session kwargs key to different cached sessions."""
    other = S3SessionConfig(
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-west-2",
    )
    session_a = await create_aioboto3_session(_CREDS)
    session_b = await create_aioboto3_session(other)
    assert session_a is not session_b


async def test_clear_cache_forces_rebuild() -> None:
    """After a clear, the same config yields a fresh session object."""
    first = await create_aioboto3_session(_CREDS)
    clear_aioboto3_session_cache()
    second = await create_aioboto3_session(_CREDS)
    assert first is not second
