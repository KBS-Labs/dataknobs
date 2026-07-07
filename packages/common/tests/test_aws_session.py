"""Unit tests for the shared AWS session config (:class:`AwsSessionConfig`).

Pure kwarg-shaping and alias-normalization checks — no network, no boto
client construction. The client-construction behavior (boto's default
region chain) and the S3-specific helpers are covered by
``packages/data/tests/test_s3_session.py``; the async factory offload /
caching behavior by ``test_aioboto3_session_offload.py``.
"""

from __future__ import annotations

from dataknobs_common.aws import AwsSessionConfig

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


def test_to_boto_config_kwargs_uses_max_workers_alias() -> None:
    cfg = AwsSessionConfig.from_dict({"max_workers": 25})
    assert cfg.max_pool_connections == 25


def test_to_boto_config_kwargs_uses_max_retries_alias() -> None:
    cfg = AwsSessionConfig.from_dict({"max_retries": 7})
    assert cfg.max_attempts == 7


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
