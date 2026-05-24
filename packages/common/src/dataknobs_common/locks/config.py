"""Structured configuration dataclass for the Postgres advisory lock.

Mirrors :class:`dataknobs_common.events.config.PostgresEventBusConfig`:
the single ``connection_string`` field is the consumed surface, and
:meth:`_normalize_dict` routes every accepted input shape
(``connection_string``, individual host/port/database/user/password
keys, ``DATABASE_URL``, ``POSTGRES_*`` env-var fallbacks) through the
shared :func:`normalize_postgres_connection_config` so the lock and the
event bus resolve a DSN identically.

The auto-derived :meth:`StructuredConfig.from_dict
<dataknobs_common.structured_config.StructuredConfig.from_dict>` is the
recommended construction path; direct
``PostgresLockConfig(connection_string=...)`` is supported but bypasses
normalization (it expects an already-resolved DSN).

The dataclass is ``frozen=True`` so ``lock.config`` is a safe read-only
window onto the resolved DSN — runtime mutation is intentionally
unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True)
class PostgresLockConfig(StructuredConfig):
    """Configuration for :class:`PostgresAdvisoryLock`.

    Attributes:
        connection_string: Resolved PostgreSQL DSN. ``from_dict`` fills
            this from any shape ``normalize_postgres_connection_config``
            accepts; direct construction expects an already-resolved DSN.
    """

    connection_string: str

    @classmethod
    def _normalize_dict(cls, raw: dict[str, Any]) -> dict[str, Any]:
        from dataknobs_common.postgres_config import (
            normalize_postgres_connection_config,
        )

        # ``require=True`` raises ConfigurationError when nothing is
        # resolvable — preserves the lock's historical error contract.
        # The cast narrows the return type for the type checker (mirrors
        # PostgresEventBusConfig exactly).
        normalized = cast(
            "dict[str, Any]",
            normalize_postgres_connection_config(raw, require=True),
        )
        # The normalizer returns a superset of canonical keys; project
        # only the single field this dataclass consumes.
        return {"connection_string": normalized["connection_string"]}


__all__ = ["PostgresLockConfig"]
