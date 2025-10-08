"""PostgreSQL-specific connection pooling implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BasePoolConfig


@dataclass
class PostgresPoolConfig(BasePoolConfig):
    """Configuration for PostgreSQL connection pools."""
    host: str = "localhost"
    port: int = 5432
    database: str = "postgres"
    user: str = "postgres"
    password: str = ""
    min_size: int = 2
    max_size: int = 5
    command_timeout: float | None = None
    ssl: Any | None = None

    def to_connection_string(self) -> str:
        """Convert to PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def to_hash_key(self) -> tuple:
        """Create a hashable key for this configuration."""
        return (self.host, self.port, self.database, self.user)

    @classmethod
    def from_dict(cls, config: dict) -> PostgresPoolConfig:
        """Create from configuration dictionary.

        Supports either a connection_string parameter or individual parameters.

        Args:
            config: Configuration dict with either:
                - connection_string: PostgreSQL connection string (postgresql://user:pass@host:port/db)
                - OR individual parameters: host, port, database, user, password

        Returns:
            PostgresPoolConfig instance
        """
        # Check if connection_string is provided
        connection_string = config.get("connection_string")

        if connection_string:
            from urllib.parse import urlparse
            parsed = urlparse(connection_string)

            # Extract connection parameters from connection string
            host = parsed.hostname or "localhost"
            port = parsed.port or 5432
            database = parsed.path[1:] if parsed.path and len(parsed.path) > 1 else "postgres"
            user = parsed.username or "postgres"
            password = parsed.password or ""
        else:
            # Use individual parameters
            host = config.get("host", "localhost")
            port = config.get("port", 5432)
            database = config.get("database", "postgres")
            user = config.get("user", "postgres")
            password = config.get("password", "")

        return cls(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            min_size=config.get("min_pool_size", 2),
            max_size=config.get("max_pool_size", 5),
            command_timeout=config.get("command_timeout"),
            ssl=config.get("ssl")
        )


async def create_asyncpg_pool(config: PostgresPoolConfig):
    """Create an asyncpg connection pool."""
    import asyncpg
    return await asyncpg.create_pool(
        config.to_connection_string(),
        min_size=config.min_size,
        max_size=config.max_size,
        command_timeout=config.command_timeout,
        ssl=config.ssl
    )


async def validate_asyncpg_pool(pool) -> None:
    """Validate an asyncpg pool by running a simple query."""
    async with pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
