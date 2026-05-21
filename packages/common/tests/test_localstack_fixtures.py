"""Tests for ensure_localstack_s3_bucket and the LocalStack fixtures.

Deterministic unit tests — no LocalStack service required. The
``aioboto3.Session`` factory is monkeypatched to a fake that records
head/create calls and replays scripted ``ClientError`` responses so
the helper's head-then-create flow, idempotency, and race-swallow
contract are exercised end to end.
"""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError

from dataknobs_common.testing import (
    ensure_localstack_s3_bucket,
    get_localstack_endpoint,
)


# -- Fake aioboto3 client --------------------------------------------------


class _FakeS3Client:
    """Async-context-manager-yielding S3 client with scripted errors.

    Records every head/create call along with the ``Bucket=`` arg.
    ``head_error`` / ``create_error`` (if set) are raised on the
    corresponding call.
    """

    def __init__(
        self,
        head_error: ClientError | None = None,
        create_error: ClientError | None = None,
    ) -> None:
        self.head_error = head_error
        self.create_error = create_error
        self.head_calls: list[str] = []
        self.create_calls: list[str] = []

    async def head_bucket(self, *, Bucket: str) -> dict[str, Any]:  # noqa: N803
        self.head_calls.append(Bucket)
        if self.head_error is not None:
            raise self.head_error
        return {}

    async def create_bucket(self, *, Bucket: str) -> dict[str, Any]:  # noqa: N803
        self.create_calls.append(Bucket)
        if self.create_error is not None:
            raise self.create_error
        return {}


class _FakeSession:
    """Captures the kwargs passed to ``session.client("s3", ...)``.

    Returns a single ``_FakeS3Client`` as an async context manager
    (the ``aioboto3.Session.client`` contract). The captured kwargs
    let tests assert that ``endpoint_url`` / credentials are forwarded
    correctly.
    """

    def __init__(self, client: _FakeS3Client) -> None:
        self._client = client
        self.client_kwargs: dict[str, Any] = {}
        self.client_service: str | None = None

    def client(self, service: str, **kwargs: Any) -> "_FakeSessionCM":
        self.client_service = service
        self.client_kwargs = kwargs
        return _FakeSessionCM(self._client)


class _FakeSessionCM:
    def __init__(self, client: _FakeS3Client) -> None:
        self._client = client

    async def __aenter__(self) -> _FakeS3Client:
        return self._client

    async def __aexit__(self, *_args: Any) -> None:
        return None


def _client_error(code: str, op: str = "HeadBucket") -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": "fake"}},
        operation_name=op,
    )


def _patch_session(
    monkeypatch: Any, fake: _FakeSession
) -> None:
    """Patch ``aioboto3.Session`` (which the helper instantiates) to
    return our fake session."""
    import aioboto3

    monkeypatch.setattr(aioboto3, "Session", lambda: fake)


# -- ensure_localstack_s3_bucket -------------------------------------------


@pytest.mark.asyncio
async def test_existing_bucket_skips_create(monkeypatch: Any) -> None:
    """head_bucket succeeds → create_bucket is NOT called."""
    client = _FakeS3Client()
    fake = _FakeSession(client)
    _patch_session(monkeypatch, fake)

    await ensure_localstack_s3_bucket("test-bucket", endpoint="http://x:1")

    assert client.head_calls == ["test-bucket"]
    assert client.create_calls == []


@pytest.mark.asyncio
async def test_missing_bucket_404_triggers_create(monkeypatch: Any) -> None:
    """``404`` from head_bucket triggers create_bucket."""
    client = _FakeS3Client(head_error=_client_error("404"))
    fake = _FakeSession(client)
    _patch_session(monkeypatch, fake)

    await ensure_localstack_s3_bucket("test-bucket", endpoint="http://x:1")

    assert client.head_calls == ["test-bucket"]
    assert client.create_calls == ["test-bucket"]


@pytest.mark.asyncio
async def test_missing_bucket_nosuchbucket_triggers_create(
    monkeypatch: Any,
) -> None:
    """``NoSuchBucket`` from head_bucket also triggers create_bucket."""
    client = _FakeS3Client(head_error=_client_error("NoSuchBucket"))
    fake = _FakeSession(client)
    _patch_session(monkeypatch, fake)

    await ensure_localstack_s3_bucket("test-bucket", endpoint="http://x:1")

    assert client.create_calls == ["test-bucket"]


@pytest.mark.asyncio
async def test_missing_bucket_notfound_triggers_create(
    monkeypatch: Any,
) -> None:
    """``NotFound`` from head_bucket (some boto variants) also creates."""
    client = _FakeS3Client(head_error=_client_error("NotFound"))
    fake = _FakeSession(client)
    _patch_session(monkeypatch, fake)

    await ensure_localstack_s3_bucket("test-bucket", endpoint="http://x:1")

    assert client.create_calls == ["test-bucket"]


@pytest.mark.asyncio
async def test_bucket_already_owned_by_you_is_swallowed(
    monkeypatch: Any,
) -> None:
    """Race: head says missing, create races with a concurrent setup."""
    client = _FakeS3Client(
        head_error=_client_error("404"),
        create_error=_client_error(
            "BucketAlreadyOwnedByYou", op="CreateBucket"
        ),
    )
    fake = _FakeSession(client)
    _patch_session(monkeypatch, fake)

    # No exception — by contract, after the call returns the bucket exists.
    await ensure_localstack_s3_bucket("test-bucket", endpoint="http://x:1")

    assert client.create_calls == ["test-bucket"]


@pytest.mark.asyncio
async def test_bucket_already_exists_is_swallowed(monkeypatch: Any) -> None:
    """Race: create_bucket reports ``BucketAlreadyExists`` — also swallowed."""
    client = _FakeS3Client(
        head_error=_client_error("404"),
        create_error=_client_error(
            "BucketAlreadyExists", op="CreateBucket"
        ),
    )
    fake = _FakeSession(client)
    _patch_session(monkeypatch, fake)

    await ensure_localstack_s3_bucket("test-bucket", endpoint="http://x:1")

    assert client.create_calls == ["test-bucket"]


@pytest.mark.asyncio
async def test_unexpected_head_client_error_propagates(
    monkeypatch: Any,
) -> None:
    """A non-404 head_bucket error (e.g. AccessDenied) propagates."""
    client = _FakeS3Client(head_error=_client_error("AccessDenied"))
    fake = _FakeSession(client)
    _patch_session(monkeypatch, fake)

    with pytest.raises(ClientError) as exc:
        await ensure_localstack_s3_bucket(
            "test-bucket", endpoint="http://x:1"
        )
    assert exc.value.response["Error"]["Code"] == "AccessDenied"
    assert client.create_calls == []


@pytest.mark.asyncio
async def test_unexpected_create_client_error_propagates(
    monkeypatch: Any,
) -> None:
    """A non-race create_bucket error (e.g. ServiceUnavailable) propagates."""
    client = _FakeS3Client(
        head_error=_client_error("404"),
        create_error=_client_error(
            "ServiceUnavailable", op="CreateBucket"
        ),
    )
    fake = _FakeSession(client)
    _patch_session(monkeypatch, fake)

    with pytest.raises(ClientError) as exc:
        await ensure_localstack_s3_bucket(
            "test-bucket", endpoint="http://x:1"
        )
    assert exc.value.response["Error"]["Code"] == "ServiceUnavailable"


@pytest.mark.asyncio
async def test_explicit_endpoint_is_forwarded(monkeypatch: Any) -> None:
    """The ``endpoint`` arg lands on the boto client kwargs."""
    client = _FakeS3Client()
    fake = _FakeSession(client)
    _patch_session(monkeypatch, fake)

    await ensure_localstack_s3_bucket(
        "test-bucket", endpoint="http://probe-target:9999"
    )

    assert fake.client_service == "s3"
    assert fake.client_kwargs["endpoint_url"] == "http://probe-target:9999"
    assert fake.client_kwargs["region_name"] == "us-east-1"
    assert fake.client_kwargs["aws_access_key_id"] == "test"
    assert fake.client_kwargs["aws_secret_access_key"] == "test"


@pytest.mark.asyncio
async def test_default_endpoint_uses_get_localstack_endpoint(
    monkeypatch: Any,
) -> None:
    """When ``endpoint`` is None, ``get_localstack_endpoint`` is consulted."""
    for name in (
        "DOCKER_CONTAINER",
        "LOCALSTACK_ENDPOINT",
        "LOCALSTACK_HOST",
        "LOCALSTACK_PORT",
        "AWS_ENDPOINT_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    # Force the "not in Docker" arm so the default is localhost.
    monkeypatch.setattr(
        "dataknobs_common.testing._core.os.path.exists",
        lambda p: False if p == "/.dockerenv" else False,
    )

    client = _FakeS3Client()
    fake = _FakeSession(client)
    _patch_session(monkeypatch, fake)

    await ensure_localstack_s3_bucket("test-bucket")

    assert fake.client_kwargs["endpoint_url"] == get_localstack_endpoint()
    assert fake.client_kwargs["endpoint_url"] == "http://localhost:4566"


@pytest.mark.asyncio
async def test_explicit_region_is_forwarded(monkeypatch: Any) -> None:
    """The ``region`` keyword overrides the default ``us-east-1``."""
    client = _FakeS3Client()
    fake = _FakeSession(client)
    _patch_session(monkeypatch, fake)

    await ensure_localstack_s3_bucket(
        "test-bucket", endpoint="http://x:1", region="eu-west-1"
    )

    assert fake.client_kwargs["region_name"] == "eu-west-1"


# -- pytest11 plugin: localstack_endpoint / make_localstack_s3_bucket ------


def test_localstack_endpoint_fixture_resolves(localstack_endpoint: str) -> None:
    """Session fixture returns a well-formed endpoint URL."""
    assert localstack_endpoint.startswith(("http://", "https://"))
    assert localstack_endpoint.endswith((":4566", ":4566/"))  # default port


def test_make_localstack_s3_bucket_yields_config(
    monkeypatch: Any, make_localstack_s3_bucket: Any
) -> None:
    """Factory ensures the bucket and yields a backend-shaped config dict."""
    # Patch the helper to a no-op so we don't actually try to reach LocalStack.
    calls: list[tuple[str, str | None]] = []

    async def _fake_ensure(
        bucket: str, endpoint: str | None = None, **_: Any
    ) -> None:
        calls.append((bucket, endpoint))

    monkeypatch.setattr(
        "dataknobs_common.testing.localstack_fixtures."
        "ensure_localstack_s3_bucket",
        _fake_ensure,
    )

    gen = make_localstack_s3_bucket("test-bucket")
    config = next(gen)
    try:
        assert config["bucket"] == "test-bucket"
        assert config["endpoint_url"].startswith("http")
        assert config["region"] == "us-east-1"
        assert config["access_key_id"] == "test"
        assert config["secret_access_key"] == "test"
        assert len(calls) == 1
        assert calls[0][0] == "test-bucket"
        assert calls[0][1] == config["endpoint_url"]
    finally:
        # Drain the generator so its finally clause (none today, but
        # future-proof) runs.
        with pytest.raises(StopIteration):
            next(gen)
