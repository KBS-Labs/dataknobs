"""S3-specific connection pooling implementation."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BasePoolConfig


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
        """Create from configuration dictionary."""
        bucket = config.get("bucket")
        if bucket is None:
            raise ValueError("S3 bucket configuration is required")

        return cls(
            bucket=bucket,
            prefix=config.get("prefix", ""),
            region_name=config.get("region_name"),
            aws_access_key_id=config.get("aws_access_key_id"),
            aws_secret_access_key=config.get("aws_secret_access_key"),
            aws_session_token=config.get("aws_session_token"),
            endpoint_url=config.get("endpoint_url")
        )


async def create_aioboto3_session(config: S3PoolConfig):
    """Create an aioboto3 session for S3 operations."""
    import aioboto3

    # Create session with credentials if provided
    session_config = {}

    if config.aws_access_key_id:
        session_config["aws_access_key_id"] = config.aws_access_key_id
    if config.aws_secret_access_key:
        session_config["aws_secret_access_key"] = config.aws_secret_access_key
    if config.aws_session_token:
        session_config["aws_session_token"] = config.aws_session_token
    if config.region_name:
        session_config["region_name"] = config.region_name

    # Create and return the session
    return aioboto3.Session(**session_config)


async def validate_s3_session(session, config: S3PoolConfig) -> None:
    """Validate an S3 session by checking bucket access."""
    async with session.client("s3", endpoint_url=config.endpoint_url) as s3:
        # Try to head the bucket to verify access
        await s3.head_bucket(Bucket=config.bucket)
