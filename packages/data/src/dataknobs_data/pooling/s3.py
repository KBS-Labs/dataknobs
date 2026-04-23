"""Shared boto3 / aioboto3 S3 session construction for dataknobs.

Single source of truth for how dataknobs initializes S3 clients. Owns
region resolution, ``endpoint_url`` handling, credential passthrough,
and retry/pool defaults. Both sync (``boto3``) and async (``aioboto3``)
paths route through this module so behavior stays aligned across
``S3KnowledgeBackend``, ``SyncS3Database``, ``AsyncS3Database`` and any
future S3 consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import BasePoolConfig


@dataclass(frozen=True)
class S3SessionConfig:
    """Normalized S3 session configuration.

    Accepts both ``region`` (dataknobs-legacy) and ``region_name``
    (boto-native) on input; stores ``region_name`` canonically. All
    optional fields default to ``None`` so client construction can omit
    them — letting boto's default chain resolve region and credentials
    (``AWS_DEFAULT_REGION`` env var, ``~/.aws/config`` ``[default]``
    region, EC2/ECS instance metadata, then ``us-east-1`` as boto's
    terminal fallback). Note: botocore reads ``AWS_DEFAULT_REGION`` for
    region resolution, not ``AWS_REGION``.
    """

    region_name: str | None = None
    endpoint_url: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    max_pool_connections: int = 10
    max_attempts: int = 3
    retry_mode: str = "standard"
    extra_client_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None = None) -> S3SessionConfig:
        """Normalize a config dict into an :class:`S3SessionConfig`.

        Accepted aliases:

        - ``region`` (legacy) or ``region_name`` (boto-native); when
          both are present, ``region_name`` wins.
        - Credentials: canonical ``aws_access_key_id`` /
          ``aws_secret_access_key`` / ``aws_session_token`` or the
          legacy short forms ``access_key_id`` / ``secret_access_key``
          / ``session_token``.
        - ``max_pool_connections`` (or ``max_workers`` alias).
        - ``max_attempts`` (or ``max_retries`` alias).

        Missing keys default to ``None`` so boto's default resolution
        chain takes over at client-construction time.
        """
        cfg = dict(config or {})
        return cls(
            region_name=cfg.get("region_name") or cfg.get("region"),
            endpoint_url=cfg.get("endpoint_url"),
            aws_access_key_id=(
                cfg.get("aws_access_key_id") or cfg.get("access_key_id")
            ),
            aws_secret_access_key=(
                cfg.get("aws_secret_access_key") or cfg.get("secret_access_key")
            ),
            aws_session_token=(
                cfg.get("aws_session_token") or cfg.get("session_token")
            ),
            max_pool_connections=int(
                cfg.get("max_pool_connections", cfg.get("max_workers", 10))
            ),
            max_attempts=int(
                cfg.get("max_attempts", cfg.get("max_retries", 3))
            ),
            retry_mode=cfg.get("retry_mode", "standard"),
            extra_client_kwargs=dict(cfg.get("extra_client_kwargs", {}) or {}),
        )

    def to_boto_config_kwargs(self) -> dict[str, Any]:
        """Return kwargs for ``botocore.config.Config(...)``.

        Only retry / pool-connection settings belong here. ``region_name``
        is carried on the client kwargs (see :meth:`to_client_kwargs`)
        rather than ``BotoConfig`` to avoid passing it through both
        channels — the direct client kwarg wins, so the ``BotoConfig``
        copy was dead weight.
        """
        return {
            "retries": {
                "max_attempts": self.max_attempts,
                "mode": self.retry_mode,
            },
            "max_pool_connections": self.max_pool_connections,
        }

    def to_client_kwargs(self) -> dict[str, Any]:
        """Return kwargs for ``boto3.client('s3', ...)`` / aioboto3 client.

        Optional fields are omitted when unset — including
        ``endpoint_url``, all credentials, and ``region_name`` —
        so boto's defaults apply. ``extra_client_kwargs`` is applied
        last and may override (e.g., to force ``use_ssl=True`` for an
        ``http://`` endpoint that should still attempt TLS).

        When ``endpoint_url`` starts with ``http://`` (LocalStack,
        MinIO, dev S3-compatible servers), ``use_ssl=False`` is added
        automatically so botocore doesn't attempt TLS on a plain-HTTP
        port. ``https://`` endpoints leave ``use_ssl`` unset so boto's
        default (``True``) applies.
        """
        kwargs: dict[str, Any] = {}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
            kwargs.update(_use_ssl_for_endpoint(self.endpoint_url))
        if self.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            kwargs["aws_session_token"] = self.aws_session_token
        if self.region_name:
            kwargs["region_name"] = self.region_name
        if self.extra_client_kwargs:
            kwargs.update(self.extra_client_kwargs)
        return kwargs


def _use_ssl_for_endpoint(endpoint_url: str | None) -> dict[str, Any]:
    """Return ``{'use_ssl': False}`` for plain ``http://`` endpoints.

    Preserves the prior ``SyncS3Database`` behavior of disabling SSL
    when an HTTP-scheme endpoint is configured (LocalStack, MinIO).
    Returns an empty dict for ``https://`` or absent endpoints so
    boto's default (``use_ssl=True``) applies.
    """
    if endpoint_url and endpoint_url.lower().startswith("http://"):
        return {"use_ssl": False}
    return {}


def create_boto3_s3_client(
    config: S3SessionConfig | dict[str, Any] | None = None,
) -> Any:
    """Create a configured sync ``boto3`` S3 client.

    Accepts an :class:`S3SessionConfig` or a raw config dict (normalized
    internally via :meth:`S3SessionConfig.from_dict`). All optional
    fields fall back to boto's default chain when absent — especially
    ``region_name``.
    """
    import boto3
    from botocore.config import Config as BotoConfig

    cfg = (
        config
        if isinstance(config, S3SessionConfig)
        else S3SessionConfig.from_dict(config)
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

        Delegates to :meth:`S3SessionConfig.from_dict` for all shared
        session fields (credentials, region, endpoint) so legacy and
        canonical aliases stay consistent across sync and async paths.
        Only ``bucket`` and ``prefix`` are pool-specific.
        """
        bucket = config.get("bucket")
        if bucket is None:
            raise ValueError("S3 bucket configuration is required")

        sess = S3SessionConfig.from_dict(config)
        return cls(
            bucket=bucket,
            prefix=config.get("prefix", ""),
            region_name=sess.region_name,
            aws_access_key_id=sess.aws_access_key_id,
            aws_secret_access_key=sess.aws_secret_access_key,
            aws_session_token=sess.aws_session_token,
            endpoint_url=sess.endpoint_url,
        )

    def to_session_config(self) -> S3SessionConfig:
        """Project this pool config onto the shared session-config shape.

        Pool-specific fields (``bucket``, ``prefix``) are dropped; the
        rest map directly. Used by :func:`create_aioboto3_session` and
        :func:`validate_s3_session` to share a single normalized
        kwarg-shaping layer with sync callers.
        """
        return S3SessionConfig(
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )


async def create_aioboto3_session(
    config: S3PoolConfig | S3SessionConfig,
) -> Any:
    """Create an aioboto3 session for S3 operations.

    Accepts an :class:`S3PoolConfig` (existing async-pool callers) or
    an :class:`S3SessionConfig` (new shared shape). Internally projects
    onto :class:`S3SessionConfig` so kwarg shaping is identical to
    :func:`create_boto3_s3_client`.
    """
    import aioboto3

    sess_cfg = (
        config.to_session_config()
        if isinstance(config, S3PoolConfig)
        else config
    )
    # aioboto3 sessions accept credentials and region; ``endpoint_url``
    # is a per-client kwarg (applied at ``session.client('s3', ...)``
    # time), not a session-level option.
    session_kwargs: dict[str, Any] = {}
    if sess_cfg.aws_access_key_id:
        session_kwargs["aws_access_key_id"] = sess_cfg.aws_access_key_id
    if sess_cfg.aws_secret_access_key:
        session_kwargs["aws_secret_access_key"] = sess_cfg.aws_secret_access_key
    if sess_cfg.aws_session_token:
        session_kwargs["aws_session_token"] = sess_cfg.aws_session_token
    if sess_cfg.region_name:
        session_kwargs["region_name"] = sess_cfg.region_name
    return aioboto3.Session(**session_kwargs)


async def validate_s3_session(
    session: Any,
    bucket: str,
    config: S3PoolConfig | S3SessionConfig | None = None,
) -> None:
    """Validate an S3 session by checking bucket access.

    ``bucket`` is passed separately since it is not part of the shared
    session layer (``S3SessionConfig`` has no bucket field). ``config``
    is optional and supplies only endpoint-shaping — credentials and
    region live on the session object itself. Accepts either an
    :class:`S3PoolConfig` or an :class:`S3SessionConfig`; the former is
    projected onto the session shape internally, matching the pattern
    used by :func:`create_aioboto3_session`.

    Omits ``endpoint_url`` from client kwargs when none is configured
    (so we don't pass an empty-string override to boto), and adds
    ``use_ssl=False`` when the endpoint uses plain ``http://`` so the
    behavior matches the sync ``create_boto3_s3_client`` path.
    """
    sess_cfg: S3SessionConfig | None
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
