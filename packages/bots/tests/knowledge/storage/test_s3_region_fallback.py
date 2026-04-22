"""Region-handling tests for S3KnowledgeBackend.

The discovery defect: ``S3KnowledgeBackend`` hardcoded its region to
``us-east-1``, silently overriding ``AWS_REGION`` env / IAM-role
metadata for any consumer deployed elsewhere. The fix routes client
construction through the shared :class:`S3SessionConfig` factory so
the region defaults to ``None`` and boto's resolution chain takes over.

The reproduce-first test
(:func:`test_no_region_defers_to_aws_region_env`) was written against
unfixed code, confirmed to FAIL, then verified to PASS after the
backend was routed through the factory.
"""

from __future__ import annotations

import pytest
from moto import mock_aws

from dataknobs_bots.knowledge.storage.s3 import S3KnowledgeBackend
from dataknobs_data.pooling.s3 import S3SessionConfig


@pytest.fixture(autouse=True)
def _isolate_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ambient AWS env vars and shadow ~/.aws/* for boto's chain.

    Tests opt back in via ``monkeypatch.setenv``. The shared/config
    files are pointed at ``/dev/null`` so the developer's
    ``~/.aws/config`` default-region setting can't affect assertions
    that test the "no AWS config anywhere" terminal-fallback path.

    ``AWS_ENDPOINT_URL`` / ``AWS_ENDPOINT_URL_S3`` / ``LOCALSTACK_ENDPOINT``
    are cleared because botocore 1.34+ honors them as global default
    endpoints — including inside ``mock_aws()``. ``bin/test.sh`` exports
    these for integration tests, and without clearing them, boto3 in
    the unit tests would route to the running LocalStack container
    instead of moto, persisting bucket state across tests.
    """
    for key in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "LOCALSTACK_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/dev/null")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
    # Disable IMDS so boto doesn't pause trying to reach EC2 metadata.
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


async def _initialize_with_bucket(
    backend: S3KnowledgeBackend, bucket: str, region: str
) -> None:
    """Create the bucket, then initialize the backend.

    moto requires the bucket to exist before ``head_bucket`` succeeds
    inside ``initialize()``. We create it in the same region we want
    the test to observe.
    """
    import boto3

    create_kwargs: dict = {"Bucket": bucket}
    if region != "us-east-1":
        create_kwargs["CreateBucketConfiguration"] = {
            "LocationConstraint": region
        }
    boto3.client("s3", region_name=region).create_bucket(**create_kwargs)
    await backend.initialize()


async def test_no_region_defers_to_aws_default_region_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproducer: omit region kwarg → boto chain reads AWS_DEFAULT_REGION.

    Pre-fix: ``S3KnowledgeBackend(bucket=...)`` hardcoded
    ``region="us-east-1"``, so the client's region was ``us-east-1``
    regardless of the env value. Post-fix: the env value wins.

    ``AWS_DEFAULT_REGION`` is the env var that botocore actually maps
    to the region session var (``AWS_REGION`` is *not* in botocore's
    region resolution chain in current versions). The env var is set
    *inside* ``mock_aws()`` to override moto's ``us-east-1`` seed.
    """
    with mock_aws():
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        backend = S3KnowledgeBackend(bucket="reproducer-bucket")
        await _initialize_with_bucket(
            backend, "reproducer-bucket", "us-west-2"
        )
        assert backend._client is not None
        assert backend._client.meta.region_name == "us-west-2"
        await backend.close()


async def test_region_kwarg_accepted() -> None:
    """Explicit region= kwarg pins the region (overriding boto chain)."""
    with mock_aws():
        backend = S3KnowledgeBackend(
            bucket="explicit-bucket", region="eu-west-1"
        )
        await _initialize_with_bucket(
            backend, "explicit-bucket", "eu-west-1"
        )
        assert backend._client is not None
        assert backend._client.meta.region_name == "eu-west-1"
        await backend.close()


async def test_region_name_via_from_config() -> None:
    """``region_name`` (boto-native key) accepted in from_config dict."""
    with mock_aws():
        backend = S3KnowledgeBackend.from_config(
            {"bucket": "from-config-bucket", "region_name": "eu-west-1"}
        )
        await _initialize_with_bucket(
            backend, "from-config-bucket", "eu-west-1"
        )
        assert backend._client is not None
        assert backend._client.meta.region_name == "eu-west-1"
        await backend.close()


async def test_region_via_from_config() -> None:
    """``region`` (legacy key) accepted in from_config dict."""
    with mock_aws():
        backend = S3KnowledgeBackend.from_config(
            {"bucket": "legacy-key-bucket", "region": "eu-west-1"}
        )
        await _initialize_with_bucket(
            backend, "legacy-key-bucket", "eu-west-1"
        )
        assert backend._client is not None
        assert backend._client.meta.region_name == "eu-west-1"
        await backend.close()


async def test_session_config_injection_wins_over_kwargs() -> None:
    """A pre-built session_config beats individual region/credential kwargs."""
    with mock_aws():
        session_config = S3SessionConfig(region_name="eu-west-1")
        backend = S3KnowledgeBackend(
            bucket="injected-bucket",
            region="us-east-2",  # ignored: session_config takes precedence
            session_config=session_config,
        )
        await _initialize_with_bucket(
            backend, "injected-bucket", "eu-west-1"
        )
        assert backend._client is not None
        assert backend._client.meta.region_name == "eu-west-1"
        await backend.close()


async def test_initialize_succeeds_with_no_aws_config() -> None:
    """No env, no kwarg → boto's chain terminates at us-east-1."""
    with mock_aws():
        backend = S3KnowledgeBackend(bucket="default-chain-bucket")
        await _initialize_with_bucket(
            backend, "default-chain-bucket", "us-east-1"
        )
        assert backend._client is not None
        assert backend._client.meta.region_name == "us-east-1"
        await backend.close()
