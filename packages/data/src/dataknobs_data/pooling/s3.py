"""S3-specific boto3 / aioboto3 helpers for dataknobs.

The AWS-session-generic pieces — :class:`AwsSessionConfig`,
:func:`create_aioboto3_session`, :func:`clear_aioboto3_session_cache` —
live in :mod:`dataknobs_data.pooling.aws` (they are shared by every AWS
consumer: S3, ``bedrock-runtime``, etc.). This module keeps the
genuinely S3-specific surface: the S3 connection-pool config
(:class:`S3PoolConfig`), the sync S3 client factory
(:func:`create_boto3_s3_client`), and S3 session validation
(:func:`validate_s3_session`).

.. deprecated::
   ``S3SessionConfig`` is a deprecated alias for
   :class:`dataknobs_data.pooling.aws.AwsSessionConfig`. Accessing it
   from this module emits a :class:`DeprecationWarning`. Import
   ``AwsSessionConfig`` from :mod:`dataknobs_data.pooling.aws` (or the
   :mod:`dataknobs_data.pooling` package root) instead.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

from .aws import (
    AwsSessionConfig,
    _use_ssl_for_endpoint,
    clear_aioboto3_session_cache,  # noqa: F401  (re-export for import stability)
    create_aioboto3_session,  # noqa: F401  (re-export for import stability)
)
from .base import BasePoolConfig

logger = logging.getLogger(__name__)


def create_boto3_s3_client(
    config: AwsSessionConfig | dict[str, Any] | None = None,
) -> Any:
    """Create a configured sync ``boto3`` S3 client.

    Accepts an :class:`AwsSessionConfig` or a raw config dict (normalized
    internally via :meth:`AwsSessionConfig.from_dict`). All optional
    fields fall back to boto's default chain when absent — especially
    ``region_name``.
    """
    import boto3
    from botocore.config import Config as BotoConfig

    cfg = (
        config
        if isinstance(config, AwsSessionConfig)
        else AwsSessionConfig.from_dict(config)
    )
    boto_config = BotoConfig(**cfg.to_boto_config_kwargs())
    return boto3.client("s3", config=boto_config, **cfg.to_client_kwargs())


@dataclass
class S3PoolConfig(BasePoolConfig):
    """Configuration for S3 connection pools."""
    bucket: str
    prefix: str = ""
    region_name: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    endpoint_url: str | None = None

    def to_connection_string(self) -> str:
        """Convert to connection string (not used for S3, but required by base)."""
        return f"s3://{self.bucket}/{self.prefix}"

    def to_hash_key(self) -> tuple:
        """Create a hashable key for this configuration."""
        return (self.bucket, self.prefix, self.region_name, self.endpoint_url)

    @classmethod
    def from_dict(cls, config: dict) -> S3PoolConfig:
        """Create from configuration dictionary.

        Delegates to :meth:`AwsSessionConfig.from_dict` for all shared
        session fields (credentials, region, endpoint) so legacy and
        canonical aliases stay consistent across sync and async paths.
        Only ``bucket`` and ``prefix`` are pool-specific.
        """
        bucket = config.get("bucket")
        if bucket is None:
            raise ValueError("S3 bucket configuration is required")

        sess = AwsSessionConfig.from_dict(config)
        return cls(
            bucket=bucket,
            prefix=config.get("prefix", ""),
            region_name=sess.region_name,
            aws_access_key_id=sess.aws_access_key_id,
            aws_secret_access_key=sess.aws_secret_access_key,
            aws_session_token=sess.aws_session_token,
            endpoint_url=sess.endpoint_url,
        )

    def to_session_config(self) -> AwsSessionConfig:
        """Project this pool config onto the shared session-config shape.

        Pool-specific fields (``bucket``, ``prefix``) are dropped; the
        rest map directly. Used by :func:`create_aioboto3_session` and
        :func:`validate_s3_session` to share a single normalized
        kwarg-shaping layer with sync callers.
        """
        return AwsSessionConfig(
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )


async def validate_s3_session(
    session: Any,
    bucket: str,
    config: S3PoolConfig | AwsSessionConfig | None = None,
) -> None:
    """Validate an S3 session by checking bucket access.

    ``bucket`` is passed separately since it is not part of the shared
    session layer (``AwsSessionConfig`` has no bucket field). ``config``
    is optional and supplies only endpoint-shaping — credentials and
    region live on the session object itself. Accepts either an
    :class:`S3PoolConfig` or an :class:`AwsSessionConfig`; the former is
    projected onto the session shape internally, matching the pattern
    used by :func:`create_aioboto3_session`.

    Omits ``endpoint_url`` from client kwargs when none is configured
    (so we don't pass an empty-string override to boto), and adds
    ``use_ssl=False`` when the endpoint uses plain ``http://`` so the
    behavior matches the sync ``create_boto3_s3_client`` path.
    """
    sess_cfg: AwsSessionConfig | None
    if isinstance(config, S3PoolConfig):
        sess_cfg = config.to_session_config()
    else:
        sess_cfg = config

    client_kwargs: dict[str, Any] = {}
    if sess_cfg is not None and sess_cfg.endpoint_url:
        client_kwargs["endpoint_url"] = sess_cfg.endpoint_url
        client_kwargs.update(_use_ssl_for_endpoint(sess_cfg.endpoint_url))
    async with session.client("s3", **client_kwargs) as s3:
        await s3.head_bucket(Bucket=bucket)


# ---------------------------------------------------------------------------
# Deprecated alias (PEP 562 module-level ``__getattr__``)
# ---------------------------------------------------------------------------

_DEPRECATED_ALIASES = {"S3SessionConfig": AwsSessionConfig}


def __getattr__(name: str) -> Any:
    """Resolve deprecated names, warning on access.

    ``S3SessionConfig`` was renamed to :class:`AwsSessionConfig` and
    relocated to :mod:`dataknobs_data.pooling.aws`. The old name keeps
    resolving here (returning ``AwsSessionConfig``) but emits a
    :class:`DeprecationWarning`, mirroring the ``VariableSubstitution``
    warn-on-use precedent.
    """
    target = _DEPRECATED_ALIASES.get(name)
    if target is not None:
        warnings.warn(
            f"{name} is deprecated; import AwsSessionConfig from "
            "dataknobs_data.pooling.aws (or dataknobs_data.pooling) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return target
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
