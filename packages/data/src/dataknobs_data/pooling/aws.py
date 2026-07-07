"""Shared boto3 / aioboto3 AWS session construction for dataknobs.

Single source of truth for how dataknobs initializes AWS clients (S3,
``bedrock-runtime``, and any future service). Owns region resolution,
``endpoint_url`` handling, credential passthrough, and retry/pool
defaults. Both sync (``boto3``) and async (``aioboto3``) paths route
through this module so behavior stays aligned across ``S3KnowledgeBackend``,
``SyncS3Database``, ``AsyncS3Database``, the Bedrock LLM provider, and any
future AWS consumer.

The S3-specific pieces (``S3PoolConfig``, ``create_boto3_s3_client``,
``validate_s3_session``) live in :mod:`dataknobs_data.pooling.s3`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .s3 import S3PoolConfig

logger = logging.getLogger(__name__)

# Process-wide cache of warmed ``aioboto3`` sessions, keyed by a digest
# of the normalized session kwargs plus the warmed service name. A warmed
# session holds botocore's loader caches (loop-independent data files) and
# NO open transport — clients are short-lived per-operation context
# managers — so one warmed session is safely reused across calls AND across
# event loops. Without this, every ``create_aioboto3_session`` call
# re-warms from scratch; consumers that build sessions per-instance rather
# than once at startup (e.g. a multi-tenant bot registry loading several
# runtime configs against the same bucket, or one ``AsyncS3Database`` per
# event loop) would otherwise pay the warm cost repeatedly. The warmed
# service name is part of the key because the warm pre-loads that service's
# botocore data files; an ``s3``-warmed session would still block on the
# loop the first time a ``bedrock-runtime`` client is created from it, so
# distinct services key to distinct warmed sessions. The key is a hash of
# the kwargs rather than the kwargs themselves so credentials are never
# held as plaintext dict keys.
_SESSION_CACHE: dict[str, Any] = {}


def _session_cache_key(
    session_kwargs: dict[str, Any], warm_service: str
) -> str:
    """Return a stable digest identifying session kwargs + warmed service.

    Hashing keeps credentials out of the cache keys (the warmed session
    value intrinsically holds them, but the key surface stays clean). The
    warmed service name is folded in so a session warmed for one service
    is not reused for another (whose data files it has not pre-loaded).
    """
    payload = repr(
        (warm_service, sorted(session_kwargs.items()))
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clear_aioboto3_session_cache() -> None:
    """Drop all cached ``aioboto3`` sessions.

    Sessions hold no open transport, so this is a pure cache reset — the
    next :func:`create_aioboto3_session` for a given config re-warms.
    Primarily for test isolation and explicit teardown.
    """
    _SESSION_CACHE.clear()


@dataclass(frozen=True)
class AwsSessionConfig:
    """Normalized AWS session configuration.

    Accepts both ``region`` (dataknobs-legacy) and ``region_name``
    (boto-native) on input; stores ``region_name`` canonically. All
    optional fields default to ``None`` so client construction can omit
    them — letting boto's default chain resolve region and credentials
    (``AWS_DEFAULT_REGION`` env var, ``~/.aws/config`` ``[default]``
    region, EC2/ECS instance metadata, then ``us-east-1`` as boto's
    terminal fallback). Note: botocore reads ``AWS_DEFAULT_REGION`` for
    region resolution, not ``AWS_REGION``.

    The fields here are AWS-session-generic (region, credentials,
    endpoint, retry/pool tuning) — shared by every AWS consumer. Only
    service-specific config (e.g. an S3 bucket/prefix) lives on the
    per-service pool configs.
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
    def from_dict(cls, config: dict[str, Any] | None = None) -> AwsSessionConfig:
        """Normalize a config dict into an :class:`AwsSessionConfig`.

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
        """Return kwargs for ``boto3.client(service, ...)`` / aioboto3 client.

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


async def create_aioboto3_session(
    config: S3PoolConfig | AwsSessionConfig,
    *,
    warm_service: str = "s3",
) -> Any:
    """Create an aioboto3 session for AWS operations.

    Accepts an :class:`AwsSessionConfig` (the shared session shape) or
    any per-service pool config exposing ``to_session_config()`` (e.g.
    :class:`~dataknobs_data.pooling.s3.S3PoolConfig`), which is projected
    onto :class:`AwsSessionConfig` internally so kwarg shaping is
    identical to :func:`create_boto3_s3_client`.

    Both the synchronous ``aioboto3.Session(...)`` construction (botocore
    loader setup + ``~/.aws`` read) and aiobotocore's lazy, synchronous
    first-client load of botocore's bundled data files (``endpoints``,
    ``sdk-default-configuration``, the service model) block the event
    loop. The whole factory is therefore offloaded onto a worker thread
    via :func:`asyncio.to_thread`, where it also warms the session's
    botocore caches by creating-and-discarding one throwaway client, so
    the first real client creation by any consumer is a cache hit and
    does not block.

    ``warm_service`` selects which service's client is warmed (default
    ``"s3"``, which additionally pre-loads the S3 ``list_objects_v2``
    paginator model). Pass ``warm_service="bedrock-runtime"`` for the
    Bedrock provider; the returned session is service-agnostic — the
    consumer opens ``session.client(service, ...)`` per operation.

    The warmed session is cached process-wide keyed by the normalized
    session kwargs plus ``warm_service`` (see :data:`_SESSION_CACHE`), so
    repeated calls for the same config reuse one warmed session instead
    of re-warming.
    """
    sess_cfg = (
        config
        if isinstance(config, AwsSessionConfig)
        else config.to_session_config()
    )
    # aioboto3 sessions accept credentials and region; ``endpoint_url``
    # is a per-client kwarg (applied at ``session.client(service, ...)``
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

    cache_key = _session_cache_key(session_kwargs, warm_service)
    cached = _SESSION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    session = await asyncio.to_thread(
        _build_aioboto3_session, session_kwargs, warm_service
    )
    # Dict assignment is atomic under the GIL. A rare concurrent
    # first-call for the same config double-builds harmlessly — both
    # sessions are valid, the last write wins, the loser is GC'd — so no
    # lock is needed (a module-level ``asyncio.Lock`` would also bind to
    # the first loop that awaits it and break cross-loop reuse).
    _SESSION_CACHE[cache_key] = session
    return session


def _build_aioboto3_session(
    session_kwargs: dict[str, Any], warm_service: str = "s3"
) -> Any:
    """Build an ``aioboto3.Session`` and warm its botocore caches.

    Runs on a ``to_thread`` worker (no event loop of its own). Creating
    and discarding one throwaway ``warm_service`` client triggers — and
    caches on the session — the first-client botocore data loads (service
    model, endpoints, sdk-default-config; no endpoint, no network: client
    creation only loads data files). For ``warm_service="s3"``, client
    creation alone does **not** load the paginator model, so the warm also
    builds a throwaway ``list_objects_v2`` paginator to pre-load
    ``paginators-1.json`` here; non-S3 services skip the paginator warm.
    The consumer's first real client creation (and, for S3, first
    paginator build) then reuse the cache instead of loading on the loop.
    The warm runs in a private event loop on this worker thread; warm-up
    failure is logged and swallowed (best-effort fallback to the original
    first-use load).

    Warm-up failure is logged at WARNING, not DEBUG: because the warmed
    session is cached, a persistently-failing warm (e.g. a corrupt
    botocore data bundle) would otherwise silently defeat the offload —
    every consumer's first real client creation would block on the loop
    with no diagnostic trail.
    """
    import aioboto3

    session = aioboto3.Session(**session_kwargs)

    async def _warm() -> None:
        async with session.client(warm_service) as client:
            if warm_service == "s3":
                # Touch the paginator API so botocore's paginator model
                # (paginators-1.json) loads here, on this worker thread's
                # private loop, rather than lazily on the consumer's event
                # loop the first time a list_objects_v2 paginator is built.
                # get_paginator is a sync call that triggers the data-file
                # load; the paginator is discarded.
                client.get_paginator("list_objects_v2")

    # Install the private loop as this worker thread's current loop so any
    # aiobotocore path that consults ``asyncio.get_event_loop()`` during
    # client construction resolves it, then detach it before close so a
    # reused thread-pool worker never inherits a closed loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_warm())
    except Exception as exc:  # best-effort; falls back to first-use load
        logger.warning("aioboto3 session warm-up skipped: %s", exc)
    finally:
        asyncio.set_event_loop(None)
        loop.close()
    return session
