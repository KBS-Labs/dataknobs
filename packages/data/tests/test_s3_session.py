"""Tests for the S3-specific session helpers in ``pooling.s3``.

Covers :func:`create_boto3_s3_client` (sync client + boto default-region
chain via ``moto``), :class:`S3PoolConfig` normalization + its projection
onto the shared :class:`AwsSessionConfig`, :func:`validate_s3_session`
kwarg shaping, and the deprecated ``S3SessionConfig`` alias.

The pure :class:`AwsSessionConfig` kwarg-shaping / alias-normalization
unit tests live with the config in
``packages/common/tests/test_aws_session.py`` (the class was relocated to
:mod:`dataknobs_common.aws`).
"""

from __future__ import annotations

import pytest
from dataknobs_common.aws import AwsSessionConfig
from moto import mock_aws

from dataknobs_data.pooling.s3 import (
    S3PoolConfig,
    create_boto3_s3_client,
)


# ---------------------------------------------------------------------------
# Test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ambient AWS env vars and shadow ~/.aws/* for boto's chain.

    Without this, the developer's shell or ``~/.aws/config`` may
    pre-populate ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` and confuse
    the default-chain assertions.

    ``AWS_ENDPOINT_URL`` / ``AWS_ENDPOINT_URL_S3`` / ``LOCALSTACK_ENDPOINT``
    are cleared because botocore 1.34+ honors them as global default
    endpoints — including inside ``mock_aws()``. ``bin/test.sh`` exports
    these for integration tests against a running LocalStack container,
    and without clearing them, boto3 in the unit tests would route
    there instead of moto.
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
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


# ---------------------------------------------------------------------------
# create_boto3_s3_client — region resolves through boto's default chain
# ---------------------------------------------------------------------------


@mock_aws
def test_create_boto3_s3_client_no_region_uses_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No region in config + AWS_DEFAULT_REGION env set → client honors env."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    client = create_boto3_s3_client(AwsSessionConfig())
    assert client.meta.region_name == "us-west-2"


@mock_aws
def test_create_boto3_s3_client_explicit_region_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit config region overrides AWS_DEFAULT_REGION env var."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    client = create_boto3_s3_client({"region": "eu-central-1"})
    assert client.meta.region_name == "eu-central-1"


# ---------------------------------------------------------------------------
# S3PoolConfig parity with shared shape
# ---------------------------------------------------------------------------


def test_pool_config_to_session_config_round_trip() -> None:
    pool = S3PoolConfig(
        bucket="b",
        prefix="p/",
        region_name="eu-west-1",
        aws_access_key_id="AK",
        aws_secret_access_key="SK",
        aws_session_token="ST",
        endpoint_url="https://example.com",
    )
    sess = pool.to_session_config()
    assert sess.region_name == "eu-west-1"
    assert sess.endpoint_url == "https://example.com"
    assert sess.aws_access_key_id == "AK"
    assert sess.aws_secret_access_key == "SK"
    assert sess.aws_session_token == "ST"


def test_pool_config_from_dict_accepts_both_region_keys() -> None:
    cfg_legacy = S3PoolConfig.from_dict({"bucket": "b", "region": "eu-west-1"})
    cfg_native = S3PoolConfig.from_dict(
        {"bucket": "b", "region_name": "eu-west-1"}
    )
    assert cfg_legacy.region_name == "eu-west-1"
    assert cfg_native.region_name == "eu-west-1"


def test_pool_config_from_dict_accepts_legacy_credential_keys() -> None:
    """``S3PoolConfig.from_dict`` must honor short-form credential keys.

    Without this, a config dict using ``access_key_id`` / etc. would
    work for ``SyncS3Database`` (which routes through
    ``AwsSessionConfig.from_dict``) but silently drop credentials on
    the async path (``AsyncS3Database``, which routes through
    ``S3PoolConfig.from_dict``) — violating the module-level
    sync/async alignment contract.
    """
    cfg = S3PoolConfig.from_dict(
        {
            "bucket": "b",
            "access_key_id": "AK",
            "secret_access_key": "SK",
            "session_token": "ST",
        }
    )
    assert cfg.aws_access_key_id == "AK"
    assert cfg.aws_secret_access_key == "SK"
    assert cfg.aws_session_token == "ST"


# ---------------------------------------------------------------------------
# validate_s3_session — omit empty endpoint_url
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_s3_session_kwarg_shaping() -> None:
    """``validate_s3_session`` must shape per-client kwargs correctly.

    Asserts three plumbing rules that have nothing to do with real
    ``aioboto3`` behavior:

    - No ``endpoint_url`` configured → kwarg is omitted entirely (we
      don't pass ``endpoint_url=None`` to boto).
    - ``https://`` endpoint → only ``endpoint_url`` is set; ``use_ssl``
      is left to boto's default.
    - ``http://`` endpoint (LocalStack, MinIO) → ``use_ssl=False`` is
      added so botocore doesn't attempt TLS on a plain-HTTP port,
      matching the sync ``create_boto3_s3_client`` behavior.

    A hand-rolled ``_Session`` is acceptable here because we're
    asserting only on the kwargs the validator would have passed —
    not on any real boto behavior. ``moto`` doesn't expose a
    convenient interception point for ``session.client(...)``
    construction kwargs, and the project rule (testing-practices.md)
    permits a fake when the real dependency lacks a clean way to
    observe the value under test. Behavioral coverage of the actual
    boto round-trip lives in the sync ``create_boto3_s3_client`` tests
    above and the ``AwsSessionConfig`` parity tests.
    """
    from contextlib import asynccontextmanager

    from dataknobs_data.pooling.s3 import validate_s3_session

    captured_kwargs: dict[str, object] = {}

    class _RecordingS3:
        async def head_bucket(self, *, Bucket: str) -> None:  # noqa: N803
            assert Bucket == "b"

    @asynccontextmanager
    async def _client_cm(**kwargs: object):
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        yield _RecordingS3()

    class _Session:
        def client(self, _service: str, **kwargs: object):
            return _client_cm(**kwargs)

    # Case 1: no endpoint_url
    await validate_s3_session(_Session(), "b", S3PoolConfig(bucket="b"))
    assert "endpoint_url" not in captured_kwargs
    assert "use_ssl" not in captured_kwargs

    # Case 2: https:// endpoint — use_ssl left to boto default
    await validate_s3_session(
        _Session(),
        "b",
        S3PoolConfig(bucket="b", endpoint_url="https://example.com"),
    )
    assert captured_kwargs.get("endpoint_url") == "https://example.com"
    assert "use_ssl" not in captured_kwargs

    # Case 3: http:// endpoint — use_ssl forced False
    await validate_s3_session(
        _Session(),
        "b",
        S3PoolConfig(bucket="b", endpoint_url="http://localhost:4566"),
    )
    assert captured_kwargs.get("endpoint_url") == "http://localhost:4566"
    assert captured_kwargs.get("use_ssl") is False

    # Case 4: AwsSessionConfig (no bucket on the config itself) — bucket
    # comes from the explicit arg, endpoint shaping from the session.
    await validate_s3_session(
        _Session(),
        "b",
        AwsSessionConfig(endpoint_url="http://localhost:4566"),
    )
    assert captured_kwargs.get("endpoint_url") == "http://localhost:4566"
    assert captured_kwargs.get("use_ssl") is False

    # Case 5: config=None → bucket-only HEAD with no endpoint overrides.
    await validate_s3_session(_Session(), "b")
    assert "endpoint_url" not in captured_kwargs
    assert "use_ssl" not in captured_kwargs


# ---------------------------------------------------------------------------
# Deprecated S3SessionConfig alias
# ---------------------------------------------------------------------------


def test_deprecated_s3_session_config_alias_warns_and_resolves() -> None:
    """``pooling.s3.S3SessionConfig`` warns and resolves to ``AwsSessionConfig``.

    The class was renamed and relocated to ``dataknobs_common.aws``. The
    old module-path access keeps working via a PEP 562 ``__getattr__`` but
    emits a ``DeprecationWarning`` so external stragglers get a migration
    signal while their imports stay green.
    """
    import dataknobs_data.pooling.s3 as s3_mod

    with pytest.warns(DeprecationWarning, match="AwsSessionConfig"):
        alias = s3_mod.S3SessionConfig
    assert alias is AwsSessionConfig


def test_pooling_root_alias_resolves_without_warning(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """``pooling.S3SessionConfig`` (package root) resolves with no warning.

    The package root keeps the alias importable for import-site
    stability; only the ``pooling.s3`` module path emits the deprecation
    warning.
    """
    from dataknobs_data.pooling import S3SessionConfig as RootAlias

    assert RootAlias is AwsSessionConfig
    assert not [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
