"""Connection pooling utilities for database backends."""

from .base import BasePoolConfig, ConnectionPoolManager, PoolProtocol
from .s3 import (
    S3PoolConfig,
    S3SessionConfig,
    create_aioboto3_session,
    create_boto3_s3_client,
    validate_s3_session,
)

__all__ = [
    "BasePoolConfig",
    "ConnectionPoolManager",
    "PoolProtocol",
    "S3PoolConfig",
    "S3SessionConfig",
    "create_aioboto3_session",
    "create_boto3_s3_client",
    "validate_s3_session",
]
