"""PostgreSQL-specific connection pooling implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dataknobs_common import normalize_postgres_connection_config

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

        Accepts any input shape supported by
        ``normalize_postgres_connection_config``:

        - ``connection_string`` (parsed into individual keys)
        - Individual ``host``/``port``/``database``/``user``/``password`` keys
        - ``DATABASE_URL`` env var
        - ``POSTGRES_HOST``/``POSTGRES_PORT``/``POSTGRES_DB``/
          ``POSTGRES_USER``/``POSTGRES_PASSWORD`` env var fallbacks
        - ``.env`` / ``.project_vars`` files (when ``python-dotenv`` is
          installed)

        When nothing is configured anywhere, the dataclass defaults
        (``localhost``/``5432``/``postgres``/``postgres``/``""``) are
        used — preserving the historical "useful for local-dev"
        behavior that matches the underlying dataclass defaults.

        Performance: if ``config`` has already been passed through
        ``normalize_postgres_connection_config`` (detected by the
        presence of both ``connection_string`` and ``host`` keys),
        the normalizer is skipped to avoid re-parsing. This keeps
        ``AsyncPostgresDatabase.__init__`` from paying the
        normalization cost twice.

        Args:
            config: Configuration dict (may be empty).

        Returns:
            PostgresPoolConfig instance.
        """
        already_normalized = (
            "connection_string" in config
            and all(config.get(k) is not None for k in (
                "host", "port", "database", "user",
            ))
        )
        if already_normalized:
            source: dict[str, Any] = config
        else:
            normalized = normalize_postgres_connection_config(
                config, require=False,
            )
            source = normalized if normalized is not None else config
        return cls(
            host=source.get("host", "localhost"),
            port=int(source.get("port", 5432)),
            database=source.get("database", "postgres"),
            user=source.get("user", "postgres"),
            password=source.get("password", ""),
            min_size=config.get("min_pool_size", 2),
            max_size=config.get("max_pool_size", 5),
            command_timeout=config.get("command_timeout"),
            ssl=config.get("ssl"),
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
