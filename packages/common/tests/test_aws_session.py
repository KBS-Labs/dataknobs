"""Unit tests for the shared AWS session config (:class:`AwsSessionConfig`).

Pure kwarg-shaping and alias-normalization checks — no network, no boto
client construction. The client-construction behavior (boto's default
region chain) and the S3-specific helpers are covered by
``packages/data/tests/test_s3_session.py``; the async factory offload /
caching behavior by ``test_aioboto3_session_offload.py``.
"""

from __future__ import annotations

import pytest

from dataknobs_common.aws import AwsSessionConfig
from dataknobs_common.exceptions import ConfigurationError
from dataknobs_common.testing import requires_package

# ---------------------------------------------------------------------------
# AwsSessionConfig.from_dict — region key parity
# ---------------------------------------------------------------------------


def test_session_config_from_dict_accepts_region() -> None:
    cfg = AwsSessionConfig.from_dict({"region": "eu-west-1"})
    assert cfg.region_name == "eu-west-1"


def test_session_config_from_dict_accepts_region_name() -> None:
    cfg = AwsSessionConfig.from_dict({"region_name": "eu-west-1"})
    assert cfg.region_name == "eu-west-1"


def test_region_name_wins_over_region_when_both_present() -> None:
    cfg = AwsSessionConfig.from_dict(
        {"region": "us-east-1", "region_name": "eu-west-1"}
    )
    assert cfg.region_name == "eu-west-1"


def test_missing_region_yields_none() -> None:
    cfg = AwsSessionConfig.from_dict({})
    assert cfg.region_name is None


# ---------------------------------------------------------------------------
# AwsSessionConfig.from_dict — credential / pool / retry aliasing
# ---------------------------------------------------------------------------


def test_legacy_credential_keys_accepted() -> None:
    cfg = AwsSessionConfig.from_dict(
        {
            "access_key_id": "AK",
            "secret_access_key": "SK",
            "session_token": "ST",
        }
    )
    assert cfg.aws_access_key_id == "AK"
    assert cfg.aws_secret_access_key == "SK"
    assert cfg.aws_session_token == "ST"


# ---------------------------------------------------------------------------
# Credential-completeness invariant (fail closed on partial credentials)
# ---------------------------------------------------------------------------


def test_no_credentials_is_valid() -> None:
    """Neither key set → defers to boto's default chain, no error."""
    cfg = AwsSessionConfig(region_name="eu-west-1")
    assert cfg.aws_access_key_id is None
    assert cfg.aws_secret_access_key is None


def test_both_credentials_is_valid() -> None:
    """Complete explicit pair constructs cleanly."""
    cfg = AwsSessionConfig(aws_access_key_id="AK", aws_secret_access_key="SK")
    assert cfg.aws_access_key_id == "AK"
    assert cfg.aws_secret_access_key == "SK"


def test_access_key_without_secret_raises() -> None:
    """A lone access key is a misconfiguration — reject at construction.

    The old SQS session builder silently dropped both credentials in this
    case and fell through to boto's ambient chain (authenticating as a
    different identity with no signal). The shared config now fails closed
    with an actionable error naming the missing field.
    """
    with pytest.raises(ConfigurationError, match="aws_secret_access_key"):
        AwsSessionConfig(aws_access_key_id="AK")


def test_secret_without_access_key_raises() -> None:
    """A lone secret key is the symmetric misconfiguration."""
    with pytest.raises(ConfigurationError, match="aws_access_key_id"):
        AwsSessionConfig(aws_secret_access_key="SK")


def test_session_token_without_pair_raises() -> None:
    """A session token requires both access key and secret."""
    with pytest.raises(ConfigurationError, match="aws_session_token"):
        AwsSessionConfig(aws_session_token="ST")


def test_session_token_with_only_access_key_raises() -> None:
    """Token + one-of-the-pair is still incomplete.

    The key-pair check fires first (missing secret), so the error names
    the missing pair member rather than the token — but the construction
    must still be rejected. Covers the partial-pair-plus-token shape.
    """
    with pytest.raises(ConfigurationError, match="aws_secret_access_key"):
        AwsSessionConfig(aws_access_key_id="AK", aws_session_token="ST")


def test_session_token_with_full_pair_is_valid() -> None:
    """Temporary-credential triple (key + secret + token) is valid."""
    cfg = AwsSessionConfig(
        aws_access_key_id="AK",
        aws_secret_access_key="SK",
        aws_session_token="ST",
    )
    assert cfg.aws_session_token == "ST"


def test_from_dict_partial_credentials_raises() -> None:
    """The invariant holds through the ``from_dict`` construction path too."""
    with pytest.raises(ConfigurationError, match="aws_secret_access_key"):
        AwsSessionConfig.from_dict({"access_key_id": "AK"})


def test_to_boto_config_kwargs_uses_max_workers_alias() -> None:
    cfg = AwsSessionConfig.from_dict({"max_workers": 25})
    assert cfg.max_pool_connections == 25


def test_to_boto_config_kwargs_uses_max_retries_alias() -> None:
    cfg = AwsSessionConfig.from_dict({"max_retries": 7})
    assert cfg.max_attempts == 7


# ---------------------------------------------------------------------------
# to_boto_config_kwargs — timeout passthrough
# ---------------------------------------------------------------------------


def test_to_boto_config_kwargs_omits_timeouts_by_default() -> None:
    """No timeout args → neither key emitted (boto defaults apply)."""
    kwargs = AwsSessionConfig().to_boto_config_kwargs()
    assert "connect_timeout" not in kwargs
    assert "read_timeout" not in kwargs


def test_to_boto_config_kwargs_includes_timeouts_when_given() -> None:
    """Explicit timeouts flow through to the botocore Config kwargs."""
    kwargs = AwsSessionConfig().to_boto_config_kwargs(
        connect_timeout=10, read_timeout=70
    )
    assert kwargs["connect_timeout"] == 10
    assert kwargs["read_timeout"] == 70


# ---------------------------------------------------------------------------
# to_session_client_kwargs — per-client kwargs for a session-based client
# ---------------------------------------------------------------------------


class TestSessionClientKwargs:
    """The shared builder for ``session.client(service, ...)`` kwargs.

    Requires ``botocore`` (the ``Config`` object it builds); guarded so a
    botocore-less environment skips rather than errors — the same contract
    as the aioboto3-guarded offload tests.
    """

    pytestmark = requires_package("botocore")

    def test_builds_config_with_retry_and_pool(self) -> None:
        """Retry/pool tuning rides on the botocore Config, not raw kwargs.

        Reproduces the security-rule-2 / drift gap: a consumer that only
        returned ``{"endpoint_url": ...}`` shipped NO retry, pool, or
        timeout config. This shared builder makes the full set the default.
        """
        cfg = AwsSessionConfig(max_pool_connections=25, max_attempts=7)
        kwargs = cfg.to_session_client_kwargs()
        boto_config = kwargs["config"]
        assert boto_config.max_pool_connections == 25
        assert boto_config.retries == {"max_attempts": 7, "mode": "standard"}

    def test_applies_timeouts(self) -> None:
        """Read timeout wires the consumer's budget onto the client (rule 2)."""
        kwargs = AwsSessionConfig().to_session_client_kwargs(
            connect_timeout=10, read_timeout=60.0
        )
        boto_config = kwargs["config"]
        assert boto_config.connect_timeout == 10
        assert boto_config.read_timeout == 60.0

    def test_omits_credentials_and_region(self) -> None:
        """Creds + region ride on the session, never repeated on the client."""
        cfg = AwsSessionConfig(
            region_name="eu-west-1",
            aws_access_key_id="AK",
            aws_secret_access_key="SK",
        )
        kwargs = cfg.to_session_client_kwargs()
        assert "aws_access_key_id" not in kwargs
        assert "aws_secret_access_key" not in kwargs
        assert "region_name" not in kwargs

    def test_includes_endpoint_and_http_ssl(self) -> None:
        """http:// endpoint → endpoint_url + use_ssl=False (LocalStack/MinIO)."""
        cfg = AwsSessionConfig(endpoint_url="http://localhost:4566")
        kwargs = cfg.to_session_client_kwargs()
        assert kwargs["endpoint_url"] == "http://localhost:4566"
        assert kwargs["use_ssl"] is False

    def test_https_endpoint_leaves_ssl_unset(self) -> None:
        cfg = AwsSessionConfig(endpoint_url="https://bedrock.example.com")
        kwargs = cfg.to_session_client_kwargs()
        assert kwargs["endpoint_url"] == "https://bedrock.example.com"
        assert "use_ssl" not in kwargs

    def test_omits_endpoint_when_unset(self) -> None:
        kwargs = AwsSessionConfig().to_session_client_kwargs()
        assert "endpoint_url" not in kwargs
        assert "use_ssl" not in kwargs

    def test_extra_client_kwargs_applied_last(self) -> None:
        """extra_client_kwargs override the inferred values."""
        cfg = AwsSessionConfig(
            endpoint_url="http://localhost:4566",
            extra_client_kwargs={"use_ssl": True},
        )
        kwargs = cfg.to_session_client_kwargs()
        assert kwargs["use_ssl"] is True


# ---------------------------------------------------------------------------
# to_client_kwargs / to_boto_config_kwargs — shape assertions
# ---------------------------------------------------------------------------


def test_to_client_kwargs_omits_unset_fields() -> None:
    kwargs = AwsSessionConfig().to_client_kwargs()
    assert kwargs == {}


def test_to_client_kwargs_includes_set_fields() -> None:
    cfg = AwsSessionConfig(
        region_name="eu-west-1",
        endpoint_url="https://example.com",
        aws_access_key_id="AK",
        aws_secret_access_key="SK",
        aws_session_token="ST",
    )
    kwargs = cfg.to_client_kwargs()
    assert kwargs == {
        "region_name": "eu-west-1",
        "endpoint_url": "https://example.com",
        "aws_access_key_id": "AK",
        "aws_secret_access_key": "SK",
        "aws_session_token": "ST",
    }


def test_to_boto_config_kwargs_omits_region_when_none() -> None:
    kwargs = AwsSessionConfig().to_boto_config_kwargs()
    assert "region_name" not in kwargs
    assert kwargs["max_pool_connections"] == 10
    assert kwargs["retries"] == {"max_attempts": 3, "mode": "standard"}


def test_to_boto_config_kwargs_never_includes_region_name() -> None:
    """``region_name`` belongs on the client kwargs, not ``BotoConfig``.

    Passing it via both channels is redundant — the direct client
    kwarg wins, so the ``BotoConfig`` copy is dead weight. This test
    locks in that ``to_boto_config_kwargs`` never emits it, even when
    the session config has a region set.
    """
    kwargs = AwsSessionConfig(region_name="eu-west-1").to_boto_config_kwargs()
    assert "region_name" not in kwargs


def test_extra_client_kwargs_passthrough() -> None:
    cfg = AwsSessionConfig.from_dict({"extra_client_kwargs": {"verify": False}})
    assert cfg.to_client_kwargs() == {"verify": False}


# ---------------------------------------------------------------------------
# to_session_kwargs — session-level (credentials + region), no endpoint
# ---------------------------------------------------------------------------


def test_to_session_kwargs_omits_unset_fields() -> None:
    assert AwsSessionConfig().to_session_kwargs() == {}


def test_to_session_kwargs_includes_only_session_level_fields() -> None:
    """``endpoint_url`` is a per-client kwarg, never a session kwarg.

    ``to_session_kwargs`` must carry only the credential + region knobs
    that ``aioboto3.Session(...)`` accepts, and only when set. An
    endpoint set on the config must NOT leak into the session kwargs —
    it is applied later at ``session.client(service, endpoint_url=...)``.
    """
    cfg = AwsSessionConfig(
        region_name="eu-west-1",
        endpoint_url="http://localhost:4566",
        aws_access_key_id="AK",
        aws_secret_access_key="SK",
        aws_session_token="ST",
    )
    assert cfg.to_session_kwargs() == {
        "region_name": "eu-west-1",
        "aws_access_key_id": "AK",
        "aws_secret_access_key": "SK",
        "aws_session_token": "ST",
    }


# ---------------------------------------------------------------------------
# use_ssl inference for http:// endpoints
# ---------------------------------------------------------------------------


def test_http_endpoint_disables_ssl() -> None:
    """``http://`` endpoint (LocalStack, MinIO) → ``use_ssl=False``."""
    cfg = AwsSessionConfig(endpoint_url="http://localhost:4566")
    kwargs = cfg.to_client_kwargs()
    assert kwargs["endpoint_url"] == "http://localhost:4566"
    assert kwargs["use_ssl"] is False


def test_https_endpoint_leaves_use_ssl_unset() -> None:
    """``https://`` endpoint → ``use_ssl`` not set (boto default ``True``)."""
    cfg = AwsSessionConfig(endpoint_url="https://example.com")
    kwargs = cfg.to_client_kwargs()
    assert kwargs["endpoint_url"] == "https://example.com"
    assert "use_ssl" not in kwargs


def test_no_endpoint_leaves_use_ssl_unset() -> None:
    """No endpoint → ``use_ssl`` not set."""
    kwargs = AwsSessionConfig().to_client_kwargs()
    assert "use_ssl" not in kwargs


def test_extra_client_kwargs_can_override_use_ssl() -> None:
    """Caller can force ``use_ssl=True`` via ``extra_client_kwargs``."""
    cfg = AwsSessionConfig(
        endpoint_url="http://localhost:4566",
        extra_client_kwargs={"use_ssl": True},
    )
    kwargs = cfg.to_client_kwargs()
    assert kwargs["use_ssl"] is True
