"""Connection pooling utilities for database backends.

The AWS-session-generic surface (:class:`AwsSessionConfig`,
:func:`create_aioboto3_session`, :func:`clear_aioboto3_session_cache`)
now lives in :mod:`dataknobs_common.aws` so every AWS consumer across the
stack shares one implementation. It is re-exported here for convenience
and import stability. The genuinely S3-specific helpers
(:class:`S3PoolConfig`, :func:`create_boto3_s3_client`,
:func:`validate_s3_session`) live in :mod:`dataknobs_data.pooling.s3`.
"""

from dataknobs_common.aws import (
    AwsSessionConfig,
    clear_aioboto3_session_cache,
    create_aioboto3_session,
)

from .base import BasePoolConfig, ConnectionPoolManager, PoolProtocol
from .s3 import (
    S3PoolConfig,
    create_boto3_s3_client,
    validate_s3_session,
)

# Permanent compatibility alias. ``S3SessionConfig`` was the historical
# name before the config was generalized to :class:`AwsSessionConfig` and
# relocated to :mod:`dataknobs_common.aws`. This package-root alias is a
# *stable* re-export kept for import-site stability — it resolves without
# a warning (unlike the ``dataknobs_data.pooling.s3`` module path, which
# emits a ``DeprecationWarning`` to nudge stragglers). New code should
# import ``AwsSessionConfig``.
S3SessionConfig = AwsSessionConfig

__all__ = [
    "AwsSessionConfig",
    "BasePoolConfig",
    "ConnectionPoolManager",
    "PoolProtocol",
    "S3PoolConfig",
    "S3SessionConfig",
    "clear_aioboto3_session_cache",
    "create_aioboto3_session",
    "create_boto3_s3_client",
    "validate_s3_session",
]
