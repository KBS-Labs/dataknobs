"""Connection pooling utilities for database backends."""

from .aws import (
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

# Deprecated alias, kept importable from the package root for import-site
# stability. Prefer :class:`AwsSessionConfig`. Accessing ``S3SessionConfig``
# via the ``dataknobs_data.pooling.s3`` module path emits a
# ``DeprecationWarning``; this package-root alias resolves without one.
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
